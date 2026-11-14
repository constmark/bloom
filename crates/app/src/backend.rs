//! Engine registry and backend auto-selection shared by Bloom's native CLIs.
//!
//! `bloom_infer` and `bloom_bench` must pick the same engine for the same
//! model directory; keeping the routing table here prevents the two binaries
//! from drifting apart when a new format or manifest hint is added.

use anyhow::{Result, anyhow};
use bloomai_core::{DType, ModelFamily, ModelFormat, ModelManifest};
use bloomai_engine::{EngineRegistry, model_manifest_supports_embeddings, speculative_mode_is_mtp};

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
use bloomai_engine::executor::tensorrt::TensorRtEngine;
use bloomai_engine::executor::vulkan::VulkanEngine;
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::wan::WanEngine;

/// Human-readable list used in "unknown engine" errors.
pub const SUPPORTED_ENGINES_HINT: &str = "candle, laya, openvino, funasr, qwen3_vl, longcat, intel-npu, npu-tts, onnxruntime, coreml, tensorrt, mlx, vulkan, llamacpp, wan";

/// Register every engine compiled into this build.
pub fn build_engine_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::default();
    #[cfg(feature = "candle-engine")]
    registry.register("candle", Box::new(CandleEngine));
    registry.register("openvino", Box::new(OpenVINOEngine));
    registry.register("funasr", Box::new(FunASREngine));
    registry.register("laya", Box::new(LayaEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("qwen3_vl", Box::new(Qwen3VLEngine));
    registry.register("longcat", Box::new(LongCatImageEditEngine));
    registry.register("intel-npu", Box::new(IntelNpuEngine));
    registry.register("npu-tts", Box::new(NpuTtsEngine));
    registry.register("onnxruntime", Box::new(OnnxRuntimeEngine));
    registry.register("coreml", Box::new(CoreMlEngine));
    registry.register("tensorrt", Box::new(TensorRtEngine));
    registry.register("mlx", Box::new(MlxEngine));
    registry.register("vulkan", Box::new(VulkanEngine));
    registry.register("llamacpp", Box::new(LlamaCppEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("wan", Box::new(WanEngine));
    registry
}

/// Parse a CLI `--dtype` value; `None` for selectors with no fixed label.
pub fn parse_cli_dtype(s: &str) -> Option<DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Some(DType::F32),
        "f16" | "float16" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        "q4" | "int4" => Some(DType::Q4),
        "q8" | "int8" => Some(DType::Q8),
        _ => None,
    }
}

/// Resolve the engine name for a model.
///
/// An explicit `--engine`, or any `--backend` other than the `candle`
/// default, is honoured verbatim. Otherwise the manifest decides: MTP
/// speculation, specialised file formats, manifest `preferred_backends`
/// hints, and finally model family.
pub fn select_backend_name(
    explicit_engine: Option<&str>,
    backend: &str,
    speculative: &str,
    manifest: &ModelManifest,
) -> String {
    if explicit_engine.is_some() || backend != "candle" {
        return explicit_engine.unwrap_or(backend).to_string();
    }
    let has_format = |format| manifest.files.iter().any(|file| file.format == format);
    let prefers = |name: &str| {
        manifest
            .runtime_hints
            .preferred_backends
            .iter()
            .any(|backend| backend == name)
    };
    let is_qwen_vl = manifest.family == ModelFamily::Qwen
        && (manifest.id.to_lowercase().contains("vl")
            || manifest
                .parameters
                .get("model_type")
                .and_then(|v| v.as_str())
                .map(|s| s.contains("vl"))
                .unwrap_or(false));
    let is_custom = |name: &str| matches!(&manifest.family, ModelFamily::Custom(c) if c == name);
    let selected = if speculative_mode_is_mtp(speculative) {
        "llamacpp"
    } else if has_format(ModelFormat::Onnx) {
        "onnxruntime"
    } else if has_format(ModelFormat::OpenVinoIr) {
        "openvino"
    } else if has_format(ModelFormat::CoreMl) {
        "coreml"
    } else if has_format(ModelFormat::TensorRtEngine) {
        "tensorrt"
    } else if has_format(ModelFormat::Mlx) {
        "mlx"
    } else if has_format(ModelFormat::VulkanSpirv) {
        "vulkan"
    } else if prefers("qwen3_vl") {
        "qwen3_vl"
    } else if prefers("llamacpp") {
        // GGUF families such as Phi-3/4, Gemma 3, SmolLM3, Ministral 3,
        // Granite 3.3 and LFM2 have no Candle native loader in this
        // build. Manifest inference marks them for the external bridge.
        "llamacpp"
    } else if is_qwen_vl {
        "qwen3_vl"
    } else if is_custom("longcat-image-edit") {
        "longcat"
    } else if manifest.family == ModelFamily::FunAsr {
        "funasr"
    } else if is_custom("laya") {
        "laya"
    } else if is_custom("wan") {
        "wan"
    } else {
        "candle"
    };
    selected.to_string()
}

/// Reject models whose only task is served by `bloom_server` endpoints.
///
/// `tool` names the calling binary so the error points at the right command.
pub fn validate_cli_model_task(tool: &str, manifest: &ModelManifest) -> Result<()> {
    let tasks = bloomai_engine::model_manifest_tasks(manifest);
    if tasks.contains(&"rerank") && !tasks.contains(&"generation") && !tasks.contains(&"embedding")
    {
        return Err(anyhow!(
            "{tool} cannot generate from a rerank-only cross-encoder; start bloom_server and use /v1/rerank"
        ));
    }
    if model_manifest_supports_embeddings(manifest) {
        return Err(anyhow!(
            "{tool} serves text generation and cannot run an embedding encoder; start bloom_server and use /v1/embeddings, /v1/rerank, /api/embed, or /api/embeddings"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str, format: ModelFormat) -> bloomai_core::ModelFile {
        bloomai_core::ModelFile {
            name: name.to_string(),
            format,
            size_bytes: 1,
            hash_sha256: None,
            required: true,
        }
    }

    #[test]
    fn rerank_only_models_are_rejected_with_the_calling_tool_name() {
        let mut manifest = ModelManifest {
            family: ModelFamily::Bert,
            ..Default::default()
        };
        manifest.parameters.insert(
            "bloom_encoder_kind".into(),
            serde_json::json!("cross_encoder"),
        );
        let error = validate_cli_model_task("bloom_bench", &manifest)
            .unwrap_err()
            .to_string();
        assert!(error.starts_with("bloom_bench "), "{error}");
        assert!(error.contains("rerank-only"), "{error}");
        validate_cli_model_task("bloom_bench", &ModelManifest::default()).unwrap();
    }

    #[test]
    fn cli_dtype_labels_cover_quantized_selectors() {
        assert_eq!(parse_cli_dtype("BF16"), Some(DType::BF16));
        assert_eq!(parse_cli_dtype("q4"), Some(DType::Q4));
        assert_eq!(parse_cli_dtype("int8"), Some(DType::Q8));
        assert_eq!(parse_cli_dtype("auto"), None);
    }

    #[test]
    fn mtp_routes_to_llamacpp_unless_engine_is_explicit() {
        let manifest = ModelManifest::default();
        assert_eq!(
            select_backend_name(None, "candle", "mtp", &manifest),
            "llamacpp"
        );
        assert_eq!(
            select_backend_name(None, "candle", "native-mtp", &manifest),
            "llamacpp"
        );
        assert_eq!(
            select_backend_name(Some("openvino"), "candle", "mtp", &manifest),
            "openvino"
        );
        assert_eq!(
            select_backend_name(None, "openvino", "mtp", &manifest),
            "openvino"
        );
    }

    #[test]
    fn known_formats_route_to_registered_engines() {
        let registry = build_engine_registry();
        for (format, expected) in [
            (ModelFormat::Onnx, "onnxruntime"),
            (ModelFormat::OpenVinoIr, "openvino"),
            (ModelFormat::CoreMl, "coreml"),
            (ModelFormat::TensorRtEngine, "tensorrt"),
            (ModelFormat::Mlx, "mlx"),
            (ModelFormat::VulkanSpirv, "vulkan"),
        ] {
            let mut manifest = ModelManifest::default();
            manifest.files.push(file("model.bin", format));
            assert_eq!(
                select_backend_name(None, "candle", "none", &manifest),
                expected,
                "format {format:?}"
            );
            assert!(
                registry.get(expected).is_ok(),
                "unregistered route: {expected}"
            );
        }
    }

    #[test]
    fn preferred_backend_hints_override_family_routing() {
        // A GGUF architecture without a Candle loader is tagged by manifest
        // inference; the hint must win over the default Candle route.
        let mut manifest = ModelManifest::default();
        manifest.files.push(file("model.gguf", ModelFormat::Gguf));
        manifest.runtime_hints.preferred_backends = vec!["llamacpp".to_string()];
        assert_eq!(
            select_backend_name(None, "candle", "none", &manifest),
            "llamacpp"
        );

        // A Qwen VL id would otherwise reach qwen3_vl via the family check;
        // the hint must still take precedence over that heuristic.
        let mut manifest = ModelManifest {
            family: ModelFamily::Qwen,
            id: "qwen2-vl".to_string(),
            ..Default::default()
        };
        manifest.runtime_hints.preferred_backends = vec!["llamacpp".to_string()];
        assert_eq!(
            select_backend_name(None, "candle", "none", &manifest),
            "llamacpp"
        );

        let mut manifest = ModelManifest {
            family: ModelFamily::Qwen,
            ..Default::default()
        };
        manifest.runtime_hints.preferred_backends = vec!["qwen3_vl".to_string()];
        assert_eq!(
            select_backend_name(None, "candle", "none", &manifest),
            "qwen3_vl"
        );
    }

    #[test]
    fn family_routes_are_preserved() {
        let mut manifest = ModelManifest {
            family: ModelFamily::FunAsr,
            ..Default::default()
        };
        assert_eq!(
            select_backend_name(None, "candle", "none", &manifest),
            "funasr"
        );
        for (family, expected) in [
            ("wan", "wan"),
            ("laya", "laya"),
            ("longcat-image-edit", "longcat"),
        ] {
            manifest.family = ModelFamily::Custom(family.to_string());
            assert_eq!(
                select_backend_name(None, "candle", "none", &manifest),
                expected
            );
        }
        assert_eq!(
            select_backend_name(None, "candle", "none", &ModelManifest::default()),
            "candle"
        );
    }
}
