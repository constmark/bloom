//! Backend registration and serving eligibility, shared by runtime loading and preflight.
use anyhow::{Result, anyhow};
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::candle::CandleEngine;
use bloomai_engine::executor::coreml::CoreMlEngine;
use bloomai_engine::executor::funasr::FunASREngine;
use bloomai_engine::executor::intel_npu::IntelNpuEngine;
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

pub(crate) fn engine_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::default();
    #[cfg(feature = "candle-engine")]
    registry.register("candle", Box::new(CandleEngine));
    registry.register("openvino", Box::new(OpenVINOEngine));
    registry.register("funasr", Box::new(FunASREngine));
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
    if !matches!(backend_name, "candle" | "qwen3_vl") {
        return Err(anyhow!(
            "strict aggregate memory admission requires a backend that reports its verified physical device; selected backend '{backend_name}' does not yet provide that contract"
        ));
    }
    Ok(())
}
