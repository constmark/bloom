//! Native Laya decision-model adapter.
//!
//! Laya is a non-autoregressive ModernBERT decision model. Its public
//! contract is a JSON `{state, questions}` request and a JSON response with
//! typed probabilities, so it does not fit Bloom's token-generation Candle
//! path. The small `laya-decision` runtime keeps this path native and CPU/
//! Metal capable without requiring a Python service.

use std::path::Path;

use anyhow::{Result, anyhow};
use bloomai_core::{
    DeviceCapability, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily,
    ModelFormat, ModelManifest,
};
use laya::agent::LoadOptions;
use laya::{Agent, Questions, State};

use crate::{
    engine::{Engine, EngineCapability, SupportLevel, default_engine_supports},
    io::{ModelInput, ModelOutput, OutputChunk},
    model::{LoadedModel, ModelMetadata},
};

pub struct LayaEngine;

/// A Laya checkpoint has its task config at the root and its ModernBERT
/// config below `encoder/`, unlike a normal Hugging Face text-generation
/// package which puts `config.json` at the root.
pub fn is_laya_model_layout(model_path: &Path) -> bool {
    model_path.is_dir()
        && model_path.join("rl_agent_config.json").is_file()
        && model_path.join("encoder/config.json").is_file()
        && model_path.join("tokenizer/tokenizer.json").is_file()
        && model_path.join("model.safetensors").is_file()
}

impl Engine for LayaEngine {
    fn name(&self) -> &'static str {
        "laya"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu]
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![ModelFamily::Custom("laya".to_string())],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::BF16,
            ],
            supported_formats: vec![ModelFormat::Safetensors],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
                DeviceClass::DiscreteGpu,
            ],
            supported_modalities: vec![Modality::Text],
            supports_streaming: false,
            supports_quantized_models: false,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: true,
            max_context_tokens: Some(8192),
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![crate::core::parallelism::ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Beta,
            diagnostic_tips: vec![
                "Laya requests are JSON with `state` and `questions`; the result contains typed probabilities rather than generated tokens.".to_string(),
                "The native adapter uses the laya-decision Candle runtime and supports CPU; enable the crate's Metal feature for Apple GPU execution.".to_string(),
            ],
            construction_guide:
                "Point --model at a Laya checkpoint directory containing rl_agent_config.json, encoder/config.json, tokenizer/tokenizer.json, and model.safetensors.".to_string(),
        }
    }

    fn supports(&self, manifest: &ModelManifest, device_cap: &DeviceCapability) -> SupportLevel {
        default_engine_supports(&self.capability(), manifest, device_cap)
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        if !is_laya_model_layout(model_path) {
            return Err(anyhow!(
                "unsupported Laya model layout in {}. Expected rl_agent_config.json, encoder/config.json, tokenizer/tokenizer.json, and model.safetensors",
                model_path.display()
            ));
        }

        let (device_name, actual_device) = match device {
            DeviceKind::Cpu => ("cpu", DeviceKind::Cpu),
            DeviceKind::Gpu if cfg!(all(target_os = "macos", feature = "metal")) => {
                ("metal", DeviceKind::Gpu)
            }
            DeviceKind::Gpu => {
                tracing::warn!(
                    "Laya's native runtime currently supports CPU and Metal; using CPU for the requested GPU device"
                );
                ("cpu", DeviceKind::Cpu)
            }
            DeviceKind::Npu => return Err(anyhow!("Laya does not support NPU execution")),
        };
        let metadata = ModelMetadata {
            id: model_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("laya")
                .to_string(),
            modality: Modality::Text,
            quantized: false,
            manifest: crate::manifest_adapter::load_manifest(model_path)?,
        };
        let agent = Agent::load(
            &model_path.to_string_lossy(),
            LoadOptions {
                device: Some(device_name.to_string()),
                ..LoadOptions::default()
            },
        )
        .map_err(|error| anyhow!("failed to load Laya checkpoint: {error}"))?;

        Ok(Box::new(LayaModel {
            metadata,
            agent,
            actual_device,
        }))
    }
}

struct LayaModel {
    metadata: ModelMetadata,
    agent: Agent,
    actual_device: DeviceKind,
}

impl LoadedModel for LayaModel {
    fn actual_device(&self) -> Option<DeviceKind> {
        Some(self.actual_device)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let mut text = String::new();
        self.infer_stream(input, params, &mut |chunk| {
            if let OutputChunk::TextDelta(delta) = chunk {
                text.push_str(&delta);
            }
            Ok(())
        })?;
        Ok(ModelOutput {
            text: Some(text),
            logits: None,
            image: None,
            audio: None,
            video: None,
        })
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        _params: &GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        let prompt = match input {
            ModelInput::Text { prompt } => prompt,
            _ => {
                return Err(anyhow!(
                    "Laya expects a JSON text request with state and questions"
                ));
            }
        };
        let request: serde_json::Value = serde_json::from_str(&prompt).map_err(|error| {
            anyhow!("Laya request must be valid JSON with `state` and `questions`: {error}")
        })?;
        let state = request
            .get("state")
            .cloned()
            .ok_or_else(|| anyhow!("Laya request must contain a `state` field"))?;
        let questions_value = request
            .get("questions")
            .cloned()
            .ok_or_else(|| anyhow!("Laya request must contain a `questions` field"))?;
        let questions: Questions = serde_json::from_value(questions_value)
            .map_err(|error| anyhow!("invalid Laya questions object: {error}"))?;
        let result = self
            .agent
            .predict(&state as &State, &questions)
            .map_err(|error| anyhow!("Laya inference failed: {error}"))?;
        sink.on_chunk(OutputChunk::TextDelta(serde_json::to_string(
            &result.to_json(),
        )?))?;
        sink.on_chunk(OutputChunk::End)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn detects_laya_layout() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("encoder")).unwrap();
        std::fs::create_dir_all(dir.path().join("tokenizer")).unwrap();
        for file in [
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "model.safetensors",
        ] {
            std::fs::write(dir.path().join(file), b"{}").unwrap();
        }
        assert!(is_laya_model_layout(dir.path()));
    }

    #[test]
    fn rejects_incomplete_laya_layout() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("rl_agent_config.json"), b"{}").unwrap();
        assert!(!is_laya_model_layout(dir.path()));
    }

    #[test]
    fn capability_is_structured_decision_runtime() {
        let cap = LayaEngine.capability();
        assert_eq!(cap.engine_name, "laya");
        assert!(cap.supports_structured_output);
        assert!(cap.supported_formats.contains(&ModelFormat::Safetensors));
    }
}
