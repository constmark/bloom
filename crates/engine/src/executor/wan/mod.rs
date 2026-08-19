//! Wan2.x text-to-video generation engine.
//!
//! Implements the Wan2.1 diffusion-based video generation pipeline:
//! - DiT (Diffusion Transformer) for iterative denoising
//! - T5 text encoder for prompt embedding
//! - VAE decoder for latent-to-video conversion
//! - Flow matching scheduler (UniPC) for sampling

pub mod conv3d;
pub mod dit;
pub mod loader;
pub mod model;
pub mod scheduler;
pub mod t5_encoder;
pub mod vae;
pub mod video_encoder;

use std::path::Path;

use anyhow::Result;
use bloomai_core::{DeviceKind, Modality};

use crate::engine::{Engine, EngineCapability};
use crate::model::LoadedModel;

/// Wan2.x text-to-video generation engine.
pub struct WanEngine;

impl Engine for WanEngine {
    fn name(&self) -> &'static str {
        "wan"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            vec![DeviceKind::Cpu, DeviceKind::Gpu]
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            vec![DeviceKind::Cpu]
        }
    }

    fn default_device(&self) -> DeviceKind {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            DeviceKind::Gpu
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            DeviceKind::Cpu
        }
    }

    fn capability(&self) -> EngineCapability {
        use bloomai_core::{DeviceClass, ModelFamily, ModelFormat};

        #[allow(unused_mut)]
        let mut devices = vec![DeviceClass::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(DeviceClass::DiscreteGpu);
        #[cfg(feature = "metal")]
        devices.push(DeviceClass::IntegratedGpu);

        EngineCapability {
            engine_name: "wan",
            supported_families: vec![ModelFamily::Custom("wan".to_string())],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::BF16,
                bloomai_core::DType::Q8,
                bloomai_core::DType::Q4,
            ],
            supported_formats: vec![ModelFormat::Gguf, ModelFormat::Safetensors],
            supported_devices: devices,
            supported_modalities: vec![Modality::Text],
            supports_streaming: false,
            supports_quantized_models: true,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![crate::core::quantization::QuantMethod::Gguf],
            supported_parallel_strategies: vec![crate::core::parallelism::ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Wan video model requires GGUF or safetensors weights.".to_string(),
            ],
            construction_guide:
                "Built-in candle-based Wan video backend. Build with --features candle-engine."
                    .to_string(),
        }
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        model::WanVideoModel::load(model_path, device)
    }
}
