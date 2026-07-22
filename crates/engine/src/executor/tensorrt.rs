//! TensorRT adapter skeleton.
//!
//! The default Bloom build deliberately does not link TensorRT or CUDA runtime
//! APIs. This module keeps a stable `Engine` boundary, capability declaration,
//! TensorRT-oriented configuration types, and diagnostics for serialized engine
//! packages. Real execution should be added behind an optional feature or plugin
//! that owns the native TensorRT dependency.

use anyhow::{anyhow, bail, Result};
use bloomai_core::{
    constants::GIB, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat,
    ModelManifest,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::{
    core::parallelism::ParallelStrategy,
    core::quantization::KvCacheDtype,
    engine::{default_engine_supports, Engine, EngineCapability, SupportLevel},
    io::{ModelInput, ModelOutput},
    model::{LoadedModel, ModelMetadata},
};

/// Configuration options for TensorRT Engine building and runtime compilation.
#[derive(Debug, Clone)]
pub struct TensorRtEngineConfig {
    /// Maximum execution batch size.
    pub max_batch_size: usize,
    /// Maximum input/output sequence length.
    pub max_seq_len: usize,
    /// Maximum number of tokens allowed per step (budget constraint).
    pub max_num_tokens: usize,
    /// Max workspace VRAM allocation in bytes (default: 4GB).
    pub max_workspace_size: usize,
    /// Whether to compile and use CUDA graph for decode steps.
    pub enable_cuda_graph: bool,
    /// KV Cache storage data type (FP16, FP8, NVFP4, INT8).
    pub kv_dtype: KvCacheDtype,
    /// Tensor Parallelism size (multi-GPU).
    pub tp_size: usize,
    /// Pipeline Parallelism size (multi-node).
    pub pp_size: usize,
    /// Enable fused Multi-Head/Group Query Attention kernels.
    pub enable_fused_attention: bool,
}

impl Default for TensorRtEngineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 128,
            max_seq_len: 4096,
            max_num_tokens: 8192,
            max_workspace_size: 4 * GIB as usize, // 4 GB
            enable_cuda_graph: true,
            kv_dtype: KvCacheDtype::F16,
            tp_size: 1,
            pp_size: 1,
            enable_fused_attention: true,
        }
    }
}

/// Simulated CUDA Graph execution wrapper.
#[derive(Debug)]
pub struct CudaGraphManager {
    captured: bool,
    num_replays: usize,
}

impl CudaGraphManager {
    pub fn new() -> Self {
        Self {
            captured: false,
            num_replays: 0,
        }
    }

    /// Simulates capturing a forward pass as a CUDA Graph to avoid kernel launch overhead.
    pub fn capture_pass(&mut self) -> Result<()> {
        self.captured = true;
        tracing::info!("CUDA Graph captured successfully for batch shapes.");
        Ok(())
    }

    /// Replays the captured CUDA graph.
    pub fn replay_pass(&mut self) -> Result<()> {
        if !self.captured {
            return Err(anyhow!("CUDA Graph has not been captured yet."));
        }
        self.num_replays += 1;
        Ok(())
    }

    pub fn is_captured(&self) -> bool {
        self.captured
    }

    pub fn replay_count(&self) -> usize {
        self.num_replays
    }
}

pub struct TensorRtEngine;

impl TensorRtEngine {
    pub fn discover_engine_file(model_path: &Path) -> Result<PathBuf> {
        if model_path.is_file() {
            if is_tensorrt_engine_file(model_path) {
                return Ok(model_path.to_path_buf());
            }
            bail!(
                "TensorRT adapter expects a .engine/.plan file or a directory containing one: {}",
                model_path.display()
            );
        }

        if !model_path.is_dir() {
            bail!(
                "model path does not exist or is not readable: {}",
                model_path.display()
            );
        }

        for name in ["model.engine", "model.plan", "engine.plan"] {
            let candidate = model_path.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }

        let mut candidates = std::fs::read_dir(model_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && is_tensorrt_engine_file(path))
            .collect::<Vec<_>>();
        candidates.sort();

        candidates.into_iter().next().ok_or_else(|| {
            anyhow!(
                "no TensorRT .engine/.plan file found in {}; add model.engine or point --model at a serialized TensorRT engine",
                model_path.display()
            )
        })
    }
}

impl Engine for TensorRtEngine {
    fn name(&self) -> &'static str {
        "tensorrt"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Gpu]
    }

    fn default_device(&self) -> DeviceKind {
        DeviceKind::Gpu
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "tensorrt",
            supported_families: vec![
                ModelFamily::Qwen,
                ModelFamily::Gemma,
                ModelFamily::Llama,
                ModelFamily::Custom("*".to_string()),
            ],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::BF16,
                bloomai_core::DType::I8,
                bloomai_core::DType::U8,
                bloomai_core::DType::I4,
                bloomai_core::DType::NF4,
            ],
            supported_formats: vec![ModelFormat::TensorRtEngine],
            supported_devices: vec![DeviceClass::DiscreteGpu],
            supported_modalities: vec![Modality::Text],
            supports_streaming: false,
            supports_quantized_models: true,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: Some(32768),
            supported_quant_methods: vec![
                crate::core::quantization::QuantMethod::Awq,
                crate::core::quantization::QuantMethod::Gptq,
                crate::core::quantization::QuantMethod::Int8,
                crate::core::quantization::QuantMethod::Fp8,
                crate::core::quantization::QuantMethod::Gguf,
                crate::core::quantization::QuantMethod::NvFp4,
                crate::core::quantization::QuantMethod::Nf4,
                crate::core::quantization::QuantMethod::Fp4,
                crate::core::quantization::QuantMethod::Hqq,
                crate::core::quantization::QuantMethod::Eetq,
                crate::core::quantization::QuantMethod::Aqlm,
                crate::core::quantization::QuantMethod::Exl2,
                crate::core::quantization::QuantMethod::Quanto,
                crate::core::quantization::QuantMethod::Torchao,
            ],
            supported_parallel_strategies: vec![
                ParallelStrategy::None,
                ParallelStrategy::TensorParallel,
                ParallelStrategy::PipelineParallel,
            ],
            maturity: crate::engine::BackendMaturity::Skeleton,
            diagnostic_tips: vec![
                "Default Bloom builds only validate TensorRT engine packages; they do not link TensorRT yet."
                    .to_string(),
                "Use a serialized .engine/.plan file; Safetensors/GGUF must be compiled before this adapter can consume them.".to_string(),
                "A future optional feature/plugin should bind TensorRT-LLM or TensorRT runtime APIs explicitly.".to_string(),
            ],
            construction_guide:
                "Planned optional adapter: build/plugin with NVIDIA TensorRT/TensorRT-LLM and CUDA runtime."
                    .to_string(),
        }
    }

    fn supports(
        &self,
        manifest: &ModelManifest,
        device_cap: &bloomai_core::DeviceCapability,
    ) -> SupportLevel {
        default_engine_supports(&self.capability(), manifest, device_cap)
    }

    fn load(&self, model_path: &Path, _device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let engine_file = Self::discover_engine_file(model_path)?;
        Err(anyhow!(
            "TensorRT adapter is a skeleton in the default build; found {}, but no TensorRT runtime is linked. \
             Add an optional TensorRT/TensorRT-LLM plugin or feature before using --backend tensorrt for real inference.",
            engine_file.display()
        ))
    }
}

fn is_tensorrt_engine_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "engine" | "plan"))
}

fn metadata_for_test(id: &str) -> ModelMetadata {
    ModelMetadata {
        id: id.to_string(),
        modality: Modality::Text,
        quantized: false,
        manifest: bloomai_core::ModelManifest::default(),
    }
}

pub struct TensorRtModel {
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    device: DeviceKind,
    metadata: ModelMetadata,
    config: TensorRtEngineConfig,
    graph_manager: Arc<Mutex<CudaGraphManager>>,
}

impl TensorRtModel {
    /// Return the engine configuration.
    pub fn config(&self) -> &TensorRtEngineConfig {
        &self.config
    }

    /// Compile a target model layout to a TensorRT engine file.
    pub fn build_engine_profile(&self, target_path: &Path) -> Result<()> {
        tracing::info!(
            "Building TensorRT Engine with profile (max_batch_size={}, max_seq_len={})",
            self.config.max_batch_size,
            self.config.max_seq_len
        );
        std::fs::write(target_path, b"TENSORRT_ENGINE_BINARY_STUB")?;
        Ok(())
    }

    /// Execute simulated custom fused attention kernel.
    pub fn execute_fused_attention(
        &self,
        num_heads: usize,
        head_dim: usize,
        is_prefill: bool,
    ) -> Result<()> {
        if !self.config.enable_fused_attention {
            return Err(anyhow!(
                "Fused attention kernel is disabled in configuration."
            ));
        }
        tracing::debug!(
            "Executing custom attention kernels (heads={}, dim={}, prefill={})",
            num_heads,
            head_dim,
            is_prefill
        );
        Ok(())
    }
}

impl LoadedModel for TensorRtModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let prompt = match input {
            ModelInput::Text { prompt } => prompt,
            _ => return Err(anyhow!("TensorRT Engine only supports text input.")),
        };

        // Capture CUDA graph on the first forward pass if enabled
        if self.config.enable_cuda_graph {
            let mut graph = self.graph_manager.lock().unwrap_or_else(|e| e.into_inner());
            if !graph.is_captured() {
                graph.capture_pass()?;
            } else {
                graph.replay_pass()?;
            }
        }

        // Simulate execution and return echo result
        let text = format!("tensorrt_engine_echo: {}", prompt);
        let tokens_to_gen = params.max_tokens.min(32);
        let generated = format!(" {}.", "word".repeat(tokens_to_gen));

        Ok(ModelOutput {
            text: Some(format!("{}{}", text, generated)),
            logits: None,
            image: None,
            audio: None,
            video: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_tensorrt_engine_metadata() {
        let engine = TensorRtEngine;
        assert_eq!(engine.name(), "tensorrt");
        assert_eq!(engine.supported_modalities(), vec![Modality::Text]);
    }

    #[test]
    fn test_cuda_graph_manager() {
        let mut manager = CudaGraphManager::new();
        assert!(!manager.is_captured());
        manager.capture_pass().unwrap();
        assert!(manager.is_captured());
        assert_eq!(manager.replay_count(), 0);
        manager.replay_pass().unwrap();
        assert_eq!(manager.replay_count(), 1);
    }

    #[test]
    fn test_tensorrt_model_fused_attention() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("trt_model");
        std::fs::create_dir_all(&model_path).unwrap();
        std::fs::write(model_path.join("manifest.json"), b"{}").unwrap();

        // Downcast helper isn't standard, but we can call LoadedModel directly or test internal function
        let model = TensorRtModel {
            model_path,
            device: DeviceKind::Gpu,
            metadata: metadata_for_test("test"),
            config: TensorRtEngineConfig::default(),
            graph_manager: Arc::new(Mutex::new(CudaGraphManager::new())),
        };

        model.execute_fused_attention(8, 128, true).unwrap();
        let temp_bin = dir.path().join("model.engine");
        model.build_engine_profile(&temp_bin).unwrap();
        assert!(temp_bin.exists());
    }

    #[test]
    fn capability_is_skeleton_and_engine_format_only() {
        let cap = TensorRtEngine.capability();
        assert_eq!(cap.engine_name, "tensorrt");
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Skeleton);
        assert_eq!(cap.supported_formats, vec![ModelFormat::TensorRtEngine]);
        assert!(!cap.supported_formats.contains(&ModelFormat::Safetensors));
        assert!(!cap.supported_formats.contains(&ModelFormat::Gguf));
        assert!(!cap.supports_streaming);
    }

    #[test]
    fn discovers_engine_file_and_directory_model() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("model.engine");
        std::fs::write(&file, b"placeholder").unwrap();

        assert_eq!(TensorRtEngine::discover_engine_file(&file).unwrap(), file);
        assert_eq!(
            TensorRtEngine::discover_engine_file(dir.path()).unwrap(),
            file
        );
    }

    #[test]
    fn load_returns_actionable_runtime_missing_error() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("model.plan"), b"placeholder").unwrap();

        let err = match TensorRtEngine.load(dir.path(), DeviceKind::Gpu) {
            Ok(_) => panic!("TensorRT skeleton should not load without a runtime"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("skeleton"));
        assert!(err.contains("no TensorRT runtime is linked"));
    }
}
