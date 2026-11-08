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

fn execution_device(device: DeviceKind) -> Result<&'static str> {
    match device {
        DeviceKind::Cpu => Ok("cpu"),
        #[cfg(all(target_os = "macos", feature = "metal"))]
        DeviceKind::Gpu => {
            // The dependency silently falls back when Metal cannot initialize.
            // Fail before loading weights instead of reporting CPU work as GPU work.
            candle_core::Device::new_metal(0)
                .map_err(|error| anyhow!("Laya Metal device is unavailable: {error}"))?;
            Ok("metal")
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        DeviceKind::Gpu => Err(anyhow!(
            "Laya GPU execution requires macOS and the `metal` feature; select CPU explicitly"
        )),
        DeviceKind::Npu => Err(anyhow!("Laya does not support NPU execution")),
    }
}

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
        if cfg!(all(target_os = "macos", feature = "metal")) {
            vec![DeviceKind::Cpu, DeviceKind::Gpu]
        } else {
            vec![DeviceKind::Cpu]
        }
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
            supported_devices: if cfg!(all(target_os = "macos", feature = "metal")) {
                vec![
                    DeviceClass::Cpu,
                    DeviceClass::IntegratedGpu,
                    DeviceClass::DiscreteGpu,
                ]
            } else {
                vec![DeviceClass::Cpu]
            },
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
        let device_name = execution_device(device)?;
        if !is_laya_model_layout(model_path) {
            return Err(anyhow!(
                "unsupported Laya model layout in {}. Expected rl_agent_config.json, encoder/config.json, tokenizer/tokenizer.json, and model.safetensors",
                model_path.display()
            ));
        }

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
            actual_device: device,
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

    #[test]
    fn advertised_devices_match_compiled_runtime() {
        let metal = cfg!(all(target_os = "macos", feature = "metal"));
        let devices = LayaEngine.supported_devices();
        let cap = LayaEngine.capability();
        assert!(devices.contains(&DeviceKind::Cpu));
        assert_eq!(devices.contains(&DeviceKind::Gpu), metal);
        assert_eq!(
            cap.supported_devices.contains(&DeviceClass::IntegratedGpu),
            metal
        );
        assert_eq!(
            cap.supported_devices.contains(&DeviceClass::DiscreteGpu),
            metal
        );
        assert!(!devices.contains(&DeviceKind::Npu));
        assert!(!cap.supported_devices.contains(&DeviceClass::Npu));
    }

    #[test]
    fn npu_is_rejected_before_reading_weights() {
        let error = LayaEngine
            .load(Path::new("missing-checkpoint"), DeviceKind::Npu)
            .err()
            .expect("NPU must be rejected");
        assert!(error.to_string().contains("NPU"));
    }

    #[test]
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn gpu_is_rejected_without_metal_instead_of_falling_back() {
        let error = LayaEngine
            .load(Path::new("missing-checkpoint"), DeviceKind::Gpu)
            .err()
            .expect("GPU must be rejected");
        assert!(error.to_string().contains("metal"));
    }

    #[cfg(feature = "candle-engine")]
    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/laya_fixture.rs"
        ));
    }
    #[cfg(feature = "candle-engine")]
    use fixture::tiny_checkpoint;

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal", feature = "hardware-tests"))]
    fn native_metal_checkpoint_runs_on_gpu() {
        // This opt-in hardware test must fail when Metal is unavailable; a
        // skipped inference or CPU fallback is not evidence of GPU support.
        let dir = tiny_checkpoint();
        let model = LayaEngine
            .load(dir.path(), DeviceKind::Gpu)
            .expect("hardware-tests requires a working Metal device");
        assert_eq!(model.actual_device(), Some(DeviceKind::Gpu));
        let output = model.infer(
            ModelInput::Text {
                prompt: serde_json::json!({
                    "state": "Please refund the charge",
                    "questions": {"billing": {"type": "noul", "instructions": "Is this billing?"}}
                }).to_string(),
            },
            &GenerationParams::default(),
        ).expect("native Metal inference must succeed");
        let result: serde_json::Value = serde_json::from_str(&output.text.unwrap()).unwrap();
        assert_eq!(result["answers"]["billing"]["noul"], 0.5);
        assert_eq!(result["usage"]["output_tokens"], 0);
    }

    #[test]
    #[cfg(feature = "candle-engine")]
    fn native_cpu_checkpoint_returns_typed_decisions_and_validates_requests() {
        let dir = tiny_checkpoint();
        let model = LayaEngine.load(dir.path(), DeviceKind::Cpu).unwrap();
        assert_eq!(model.actual_device(), Some(DeviceKind::Cpu));
        assert_eq!(
            model.metadata().manifest.family,
            ModelFamily::Custom("laya".into())
        );
        let prompt = serde_json::json!({
            "state": "Please refund the duplicate charge",
            "questions": {
                "billing": {"type":"noul", "instructions":"Is this billing?"},
                "route": {"type":"choice", "instructions":"Select team", "criteria":["billing", "tech"]},
                "urgency": {"type":"score", "instructions":"Rate urgency", "criteria":["low", "high"]}
            }
        }).to_string();
        let output = model
            .infer(
                ModelInput::Text {
                    prompt: prompt.clone(),
                },
                &GenerationParams::default(),
            )
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&output.text.unwrap()).unwrap();
        assert_eq!(result["answers"]["billing"]["noul"], 0.5);
        assert_eq!(result["answers"]["route"]["probabilities"]["billing"], 0.5);
        assert_eq!(result["answers"]["urgency"]["score"], 0.5);
        assert_eq!(result["usage"]["output_tokens"], 0);
        assert!(result["usage"]["input_tokens"].as_u64().unwrap() > 0);
        let mut chunks = Vec::new();
        model
            .infer_stream(
                ModelInput::Text { prompt },
                &GenerationParams::default(),
                &mut |chunk| {
                    chunks.push(chunk);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(matches!(chunks[0], OutputChunk::TextDelta(_)));
        assert!(matches!(chunks[1], OutputChunk::End));
        for (prompt, expected) in [
            ("not JSON", "valid JSON"),
            (r#"{"state":"hello"}"#, "questions"),
            (r#"{"questions":{}}"#, "state"),
            (r#"{"state":"hello","questions":[]}"#, "questions object"),
            (
                r#"{"state":"hello","questions":{"q":{"type":"invalid","instructions":"test"}}}"#,
                "inference failed",
            ),
        ] {
            let error = model
                .infer(
                    ModelInput::Text {
                        prompt: prompt.into(),
                    },
                    &GenerationParams::default(),
                )
                .expect_err("invalid request must fail");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}
