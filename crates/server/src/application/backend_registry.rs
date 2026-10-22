//! Backend registration and serving eligibility, shared by runtime loading and preflight.
use anyhow::{Result, anyhow};
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::candle::CandleEngine;
use bloomai_engine::executor::coreml::CoreMlEngine;
use bloomai_engine::executor::funasr::FunASREngine;
use bloomai_engine::executor::intel_npu::IntelNpuEngine;
use bloomai_engine::executor::laya::LayaEngine;
use bloomai_engine::executor::llamacpp::LlamaCppEngine;
use bloomai_engine::executor::longcat_image_edit::LongCatImageEditEngine;
use bloomai_engine::executor::mlx::MlxEngine;
use bloomai_engine::executor::npu_tts::NpuTtsEngine;
use bloomai_engine::executor::onnx::OnnxRuntimeEngine;
use bloomai_engine::executor::openvino::OpenVINOEngine;
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::qwen3_vl::Qwen3VLEngine;
use bloomai_engine::executor::vulkan::VulkanEngine;
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::wan::WanEngine;
use bloomai_engine::{EngineRegistry, speculative_mode_is_mtp};

#[cfg(not(feature = "candle-engine"))]
use anyhow::Result as AnyResult;
#[cfg(not(feature = "candle-engine"))]
use bloomai_core::{DeviceCapability, DeviceKind, Modality, ModelManifest};
#[cfg(not(feature = "candle-engine"))]
use bloomai_engine::{BackendMaturity, Engine, EngineCapability, LoadedModel, SupportLevel};
#[cfg(not(feature = "candle-engine"))]
use std::path::Path;

/// Diagnostics-only placeholder kept in no-default-features binaries.
///
/// The server can still inspect manifests and return a useful preflight/doctor
/// report when Candle was intentionally omitted from a small deployment build;
/// a missing registration used to turn that expected incompatibility into a
/// 500 response (and even an `unreachable!` panic in startup diagnostics).
#[cfg(not(feature = "candle-engine"))]
struct UnavailableCandleEngine;

#[cfg(not(feature = "candle-engine"))]
impl Engine for UnavailableCandleEngine {
    fn name(&self) -> &'static str {
        "candle"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![
            Modality::Text,
            Modality::Audio,
            Modality::Vision,
            Modality::Multi,
        ]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Npu]
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![bloomai_core::ModelFamily::Custom("*".to_string())],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::BF16,
                bloomai_core::DType::I8,
                bloomai_core::DType::I4,
                bloomai_core::DType::Q8,
                bloomai_core::DType::Q4,
            ],
            supported_formats: vec![
                bloomai_core::ModelFormat::Gguf,
                bloomai_core::ModelFormat::Safetensors,
            ],
            supported_devices: vec![
                bloomai_core::DeviceClass::Cpu,
                bloomai_core::DeviceClass::IntegratedGpu,
                bloomai_core::DeviceClass::DiscreteGpu,
                bloomai_core::DeviceClass::Npu,
            ],
            supported_modalities: self.supported_modalities(),
            supports_streaming: false,
            supports_quantized_models: false,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: Vec::new(),
            supported_parallel_strategies: vec![bloomai_engine::ParallelStrategy::None],
            maturity: BackendMaturity::Skeleton,
            diagnostic_tips: vec![
                "This binary was built without the candle-engine feature.".to_string(),
                "Rebuild with the default features (or --features candle-engine) for native text inference.".to_string(),
            ],
            construction_guide:
                "Enable the candle-engine Cargo feature when building the server.".to_string(),
        }
    }

    fn supports(&self, _: &ModelManifest, _: &DeviceCapability) -> SupportLevel {
        SupportLevel::Unsupported(
            "Candle native inference is not compiled into this binary".to_string(),
        )
    }

    fn load(&self, _: &Path, _: DeviceKind) -> AnyResult<Box<dyn LoadedModel>> {
        Err(anyhow!(
            "Candle engine is unavailable because this binary was built without the candle-engine feature"
        ))
    }
}

pub(crate) fn engine_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::default();
    #[cfg(feature = "candle-engine")]
    registry.register("candle", Box::new(CandleEngine));
    #[cfg(not(feature = "candle-engine"))]
    registry.register("candle", Box::new(UnavailableCandleEngine));
    registry.register("openvino", Box::new(OpenVINOEngine));
    registry.register("funasr", Box::new(FunASREngine));
    registry.register("laya", Box::new(LayaEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("qwen3_vl", Box::new(Qwen3VLEngine));
    registry.register("intel-npu", Box::new(IntelNpuEngine));
    registry.register("npu-tts", Box::new(NpuTtsEngine));
    registry.register("onnxruntime", Box::new(OnnxRuntimeEngine));
    registry.register("coreml", Box::new(CoreMlEngine));
    registry.register("mlx", Box::new(MlxEngine));
    registry.register("vulkan", Box::new(VulkanEngine));
    registry.register("llamacpp", Box::new(LlamaCppEngine));
    registry.register("longcat", Box::new(LongCatImageEditEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("wan", Box::new(WanEngine));
    registry
}

pub(crate) fn select_backend_name(
    backend: &str,
    speculative: &str,
    manifest: &bloomai_core::ModelManifest,
) -> String {
    if backend == "candle" {
        let has_format = |format| manifest.files.iter().any(|file| file.format == format);
        let is_qwen_vl = manifest.family == bloomai_core::ModelFamily::Qwen
            && (manifest.id.to_lowercase().contains("vl")
                || manifest
                    .parameters
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.contains("vl"))
                    .unwrap_or(false));
        if speculative_mode_is_mtp(speculative) {
            "llamacpp".to_string()
        } else if has_format(bloomai_core::ModelFormat::Onnx) {
            "onnxruntime".to_string()
        } else if has_format(bloomai_core::ModelFormat::OpenVinoIr) {
            "openvino".to_string()
        } else if has_format(bloomai_core::ModelFormat::CoreMl) {
            "coreml".to_string()
        } else if has_format(bloomai_core::ModelFormat::Mlx) {
            "mlx".to_string()
        } else if has_format(bloomai_core::ModelFormat::VulkanSpirv) {
            "vulkan".to_string()
        } else if is_qwen_vl {
            "qwen3_vl".to_string()
        } else if matches!(&manifest.family, bloomai_core::ModelFamily::Custom(c) if c == "longcat-image-edit")
        {
            "longcat".to_string()
        } else if manifest.family == bloomai_core::ModelFamily::FunAsr {
            "funasr".to_string()
        } else if matches!(&manifest.family, bloomai_core::ModelFamily::Custom(c) if c == "laya") {
            "laya".to_string()
        } else if matches!(&manifest.family, bloomai_core::ModelFamily::Custom(c) if c == "wan") {
            "wan".to_string()
        } else {
            "candle".to_string()
        }
    } else {
        backend.to_string()
    }
}

pub(crate) fn manifest_param_usize(
    manifest: &bloomai_core::ModelManifest,
    names: &[&str],
    default_value: usize,
) -> usize {
    names
        .iter()
        .find_map(|name| manifest.parameters.get(*name))
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or(default_value)
}

pub(crate) fn div_ceil_usize(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        return 0;
    }
    numerator.saturating_add(denominator - 1) / denominator
}

pub(crate) fn validate_ifb_backend(enable_ifb: bool, backend_name: &str) -> Result<()> {
    if enable_ifb && backend_name != "candle" {
        return Err(anyhow!(
            "in-flight batching requires the verified Candle batch backend; selected backend '{backend_name}' cannot be published with IFB enabled"
        ));
    }
    Ok(())
}

pub(crate) fn validate_strict_runtime_backend(backend_name: &str) -> Result<()> {
    // External bridges must report the exact selected physical device before
    // the server publishes their runtime and charges its manifest footprint.
    // ONNX, FunASR, and MLX satisfy this contract before spawning inference.
    if !matches!(
        backend_name,
        "candle" | "qwen3_vl" | "laya" | "onnxruntime" | "funasr" | "mlx"
    ) {
        return Err(anyhow!(
            "strict aggregate memory admission requires a backend that reports its verified physical device; selected backend '{backend_name}' does not yet provide that contract"
        ));
    }
    Ok(())
}
