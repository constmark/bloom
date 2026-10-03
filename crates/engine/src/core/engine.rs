use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow};
use bloomai_core::{
    DType, DeviceCapability, DeviceClass, DeviceKind, Modality, ModelFamily, ModelFormat,
    ModelManifest, QuantScheme,
};

use crate::core::parallelism::ParallelStrategy;
use crate::core::quantization::QuantMethod;
use crate::model::LoadedModel;

/// Maturity level of a backend engine implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackendMaturity {
    /// Fully tested, production-ready.
    Production,
    /// Feature-complete but may have edge cases.
    Beta,
    /// Working but limited testing / known issues.
    Experimental,
    /// Skeleton implementation, not yet functional.
    Skeleton,
}

impl std::fmt::Display for BackendMaturity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Production => write!(f, "production"),
            Self::Beta => write!(f, "beta"),
            Self::Experimental => write!(f, "experimental"),
            Self::Skeleton => write!(f, "skeleton"),
        }
    }
}

/// Declarative capability matrix for an engine implementation.
///
/// Engines declare what model families, dtypes, formats, devices and modalities
/// they support so that the scheduler can make routing decisions *before*
/// attempting to load a model.
#[derive(Debug, Clone)]
pub struct EngineCapability {
    pub engine_name: &'static str,
    pub supported_families: Vec<ModelFamily>,
    pub supported_dtypes: Vec<DType>,
    pub supported_formats: Vec<ModelFormat>,
    pub supported_devices: Vec<DeviceClass>,
    pub supported_modalities: Vec<Modality>,
    pub supports_streaming: bool,
    pub supports_quantized_models: bool,
    /// Whether this engine can produce embedding vectors for embedding models.
    pub supports_embeddings: bool,
    /// Whether this engine can score documents for reranking directly or through embeddings.
    pub supports_rerank: bool,
    /// Whether this engine can enforce schema/grammar-constrained structured output.
    pub supports_structured_output: bool,
    pub max_context_tokens: Option<usize>,
    /// Quantization methods this engine can handle.
    pub supported_quant_methods: Vec<QuantMethod>,
    /// Parallel strategies this engine supports.
    pub supported_parallel_strategies: Vec<ParallelStrategy>,
    /// Maturity level of this backend.
    pub maturity: BackendMaturity,
    /// Diagnostic tips for common issues with this backend.
    pub diagnostic_tips: Vec<String>,
    /// How to build/enable this backend.
    pub construction_guide: String,
}

pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    fn supported_modalities(&self) -> Vec<Modality>;
    fn supported_devices(&self) -> Vec<DeviceKind>;

    /// Return the best device this engine can run on without any external
    /// backend / runtime assistance.  Used by the standalone mode so that
    /// users don't have to pass `--backend` on the command line.
    ///
    /// The default implementation returns `DeviceKind::Cpu`.  Engine
    /// implementations should override this to return `DeviceKind::Gpu`
    /// when compiled with CUDA / Metal support.
    fn default_device(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    /// Return the declarative capability matrix for this engine.
    /// Default returns a minimal capability derived from `supported_modalities()`
    /// and `supported_devices()`. Implementations should override for richer data.
    fn capability(&self) -> EngineCapability {
        // Derive DeviceClass list from legacy DeviceKind list
        let device_classes: Vec<DeviceClass> = self
            .supported_devices()
            .iter()
            .flat_map(|dk| match dk {
                DeviceKind::Cpu => vec![DeviceClass::Cpu],
                DeviceKind::Gpu => vec![DeviceClass::IntegratedGpu, DeviceClass::DiscreteGpu],
                DeviceKind::Npu => vec![DeviceClass::Npu],
            })
            .collect();

        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![],
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16, DType::Q8, DType::Q4],
            supported_formats: vec![],
            supported_devices: device_classes,
            supported_modalities: self.supported_modalities(),
            supports_streaming: false,
            supports_quantized_models: false,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: BackendMaturity::Experimental,
            diagnostic_tips: vec![],
            construction_guide: String::new(),
        }
    }

    /// Determine how well this engine supports a given model + device combination.
    ///
    /// The default implementation checks every dimension declared in `capability()`:
    /// device class, input modalities, model family, dtype and file format.
    fn supports(&self, manifest: &ModelManifest, device_cap: &DeviceCapability) -> SupportLevel {
        default_engine_supports(&self.capability(), manifest, device_cap)
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>>;
}

/// Free function implementing capability-based support checking.
///
/// Engines can call this from their `supports()` override to get the default
/// verdict and then adjust it (e.g. converting Unsupported to Fallback).
pub fn default_engine_supports(
    cap: &EngineCapability,
    manifest: &ModelManifest,
    device_cap: &DeviceCapability,
) -> SupportLevel {
    if cap.maturity == BackendMaturity::Skeleton {
        return SupportLevel::Unsupported(format!(
            "engine '{}' is a skeleton adapter and cannot execute real inference in this build",
            cap.engine_name
        ));
    }

    // 1. Device class check
    if !cap.supported_devices.is_empty()
        && !cap.supported_devices.contains(&device_cap.device_class)
    {
        return SupportLevel::Unsupported(format!(
            "engine '{}' does not support {:?} devices (supported: {:?})",
            cap.engine_name, device_cap.device_class, cap.supported_devices
        ));
    }

    // 2. Input modality check
    if !cap.supported_modalities.is_empty() {
        let unsupported: Vec<_> = manifest
            .io_schema
            .inputs
            .iter()
            .filter(|m| !cap.supported_modalities.contains(m))
            .copied()
            .collect();
        if !unsupported.is_empty() {
            return SupportLevel::Unsupported(format!(
                "engine '{}' does not support input modalities {:?}",
                cap.engine_name, unsupported
            ));
        }
    }

    // 3. Model family check (empty list means "any family")
    if !cap.supported_families.is_empty() {
        let family_match = cap.supported_families.iter().any(|f| match f {
            ModelFamily::Custom(c) if c == "*" => true,
            other => other == &manifest.family,
        });
        if !family_match {
            return SupportLevel::Unsupported(format!(
                "engine '{}' does not have a native adapter for model family {:?}",
                cap.engine_name, manifest.family
            ));
        }
    }

    // 4. Dtype check — fall back if engine dtypes are declared but dtype not listed
    if !cap.supported_dtypes.is_empty() && !cap.supported_dtypes.contains(&manifest.primary_dtype) {
        return SupportLevel::Fallback(format!(
            "engine '{}' does not natively support {:?}; conversion or fallback may be needed",
            cap.engine_name, manifest.primary_dtype
        ));
    }

    // 5. File format check (skip if manifest has no files declared).  A
    // package may contain metadata files represented as `Unknown`, but every
    // executable/weight format must be understood by the selected engine.
    // Checking only `any(...)` here used to route mixed Safetensors+ONNX
    // packages to a Safetensors engine and defer the failure until load time.
    if !cap.supported_formats.is_empty() && !manifest.files.is_empty() {
        let unsupported: Vec<_> = manifest
            .files
            .iter()
            .filter(|f| f.format != ModelFormat::Unknown)
            .filter(|f| !cap.supported_formats.contains(&f.format))
            .map(|f| f.format)
            .collect();
        if !unsupported.is_empty() {
            return SupportLevel::Unsupported(format!(
                "engine '{}' does not support declared formats {:?} (supported: {:?})",
                cap.engine_name, unsupported, cap.supported_formats
            ));
        }
    }

    // 6. Quantization method check.  Engines that advertise a non-empty
    // method list are making a precise compatibility claim; honor it before
    // loading so AWQ/GPTQ/NF4/etc. are never silently treated as GGUF.  An
    // empty list means the engine accepts any quantized representation when
    // `supports_quantized_models` is true (used by opaque vendor runtimes).
    if manifest_is_quantized(manifest) {
        if !cap.supports_quantized_models {
            return SupportLevel::Fallback(format!(
                "engine '{}' does not natively support quantized models{}; conversion or a quantized-capable engine is required",
                cap.engine_name,
                manifest_quant_method(manifest)
                    .map(|method| format!(" ({method})"))
                    .unwrap_or_default()
            ));
        }
        if let Some(method) = manifest_quant_method(manifest)
            && !cap.supported_quant_methods.is_empty()
            && !cap.supported_quant_methods.contains(&method)
        {
            return SupportLevel::Fallback(format!(
                "engine '{}' does not natively support {method} quantization (supported: {:?}); conversion or fallback may be needed",
                cap.engine_name, cap.supported_quant_methods
            ));
        } else if manifest_quant_method(manifest).is_none()
            && !cap.supported_quant_methods.is_empty()
            && manifest_uses_unclassified_quantization(manifest)
        {
            return SupportLevel::Fallback(format!(
                "engine '{}' cannot identify the model's quantization method; supported methods are {:?}",
                cap.engine_name, cap.supported_quant_methods
            ));
        }
    }

    // 7. Backend dtype cross-check (device capability vs manifest)
    if !device_cap.supported_dtypes.is_empty()
        && !device_cap
            .supported_dtypes
            .contains(&manifest.primary_dtype)
    {
        return SupportLevel::Fallback(format!(
            "backend '{}' does not advertise {:?}; engine may need conversion or fallback",
            device_cap.backend_name, manifest.primary_dtype
        ));
    }

    SupportLevel::Native
}

/// Resolve the quantization method represented by a trusted manifest.
///
/// Generic INT4 has no single method identity, so it is left as `None` and is
/// still covered by the `supports_quantized_models` gate through its dtype.
fn manifest_quant_method(manifest: &ModelManifest) -> Option<QuantMethod> {
    if let Some(info) = manifest.quantization.as_ref() {
        let method = match &info.scheme {
            QuantScheme::None => return dtype_quant_method(manifest),
            QuantScheme::GGUF(_) => QuantMethod::Gguf,
            QuantScheme::AWQ => QuantMethod::Awq,
            QuantScheme::GPTQ => QuantMethod::Gptq,
            QuantScheme::INT8 => QuantMethod::Int8,
            QuantScheme::NF4 => QuantMethod::Nf4,
            QuantScheme::INT4 => return None,
        };
        // GGUF metadata with bits == 0 denotes a full precision tensor type
        // (for example F32), not a quantized model.
        if matches!(info.scheme, QuantScheme::GGUF(_)) && info.bits == 0 {
            return None;
        }
        return Some(method);
    }

    // Older callers may only populate `primary_dtype`. Preserve compatibility
    // by deriving a conservative method for the unambiguous 8-bit and NF4
    // forms, and by recognizing GGUF weight files as GGUF quantization.
    dtype_quant_method(manifest)
}

fn dtype_quant_method(manifest: &ModelManifest) -> Option<QuantMethod> {
    match manifest.primary_dtype {
        DType::Q8 | DType::I8 | DType::U8 => Some(QuantMethod::Int8),
        DType::NF4 => Some(QuantMethod::Nf4),
        DType::Q4 | DType::I4 if manifest.files.iter().any(|f| f.format == ModelFormat::Gguf) => {
            Some(QuantMethod::Gguf)
        }
        _ => None,
    }
}

fn manifest_is_quantized(manifest: &ModelManifest) -> bool {
    if let Some(info) = manifest.quantization.as_ref()
        && !matches!(info.scheme, QuantScheme::None)
        && !(matches!(info.scheme, QuantScheme::GGUF(_)) && info.bits == 0)
    {
        return info.bits > 0;
    }
    matches!(
        manifest.primary_dtype,
        DType::Q4 | DType::I4 | DType::NF4 | DType::Q8 | DType::I8 | DType::U8
    )
}

fn manifest_uses_unclassified_quantization(manifest: &ModelManifest) -> bool {
    if let Some(info) = manifest.quantization.as_ref() {
        return matches!(info.scheme, QuantScheme::INT4)
            || (matches!(info.scheme, QuantScheme::None)
                && matches!(manifest.primary_dtype, DType::Q4 | DType::I4));
    }
    matches!(manifest.primary_dtype, DType::Q4 | DType::I4)
        && !manifest
            .files
            .iter()
            .any(|file| file.format == ModelFormat::Gguf)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportLevel {
    Native,
    Fallback(String),
    Unsupported(String),
}

impl SupportLevel {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Native | Self::Fallback(_))
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Native => None,
            Self::Fallback(reason) | Self::Unsupported(reason) => Some(reason.as_str()),
        }
    }
}

/// Convert a `DeviceCapability` to the legacy `DeviceKind` enum.
pub fn device_kind_from_capability(capability: &DeviceCapability) -> DeviceKind {
    match capability.device_class {
        DeviceClass::Cpu => DeviceKind::Cpu,
        DeviceClass::IntegratedGpu | DeviceClass::DiscreteGpu => DeviceKind::Gpu,
        DeviceClass::Npu => DeviceKind::Npu,
        DeviceClass::Dsp | DeviceClass::Remote | DeviceClass::Unknown => DeviceKind::Cpu,
    }
}

#[derive(Default)]
pub struct EngineRegistry {
    engines: HashMap<String, Box<dyn Engine>>,
}

impl EngineRegistry {
    pub fn register(&mut self, name: impl Into<String>, engine: Box<dyn Engine>) {
        self.engines.insert(name.into(), engine);
    }

    pub fn get(&self, name: &str) -> Result<&dyn Engine> {
        self.engines
            .get(name)
            .map(|e| e.as_ref())
            .ok_or_else(|| anyhow!("engine '{}' not found", name))
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.engines.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Iterate over all registered engines.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn Engine)> {
        self.engines
            .iter()
            .map(|(name, engine)| (name.as_str(), engine.as_ref()))
    }
}

// ---------------------------------------------------------------------------
// EngineRouter — selects the best engine for a manifest + device combination
// ---------------------------------------------------------------------------

/// Result of an engine routing decision, including the explanation.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub engine_name: String,
    pub support_level: SupportLevel,
    pub explanation: String,
}

/// Routes inference requests to the best available engine based on manifest
/// requirements and device capabilities.
pub struct EngineRouter {
    /// Engine names in priority order (first = highest priority).
    engines: Vec<String>,
}

impl EngineRouter {
    pub fn new(engines: Vec<String>) -> Self {
        Self { engines }
    }

    /// Build from an `EngineRegistry`, using alphabetical order as priority.
    pub fn from_registry(registry: &EngineRegistry) -> Self {
        Self {
            engines: registry.names().into_iter().map(String::from).collect(),
        }
    }

    /// Select the best engine for the given manifest and device capability.
    ///
    /// Returns the first engine with `Native` support, or the best `Fallback`,
    /// or an error if no engine can handle the request.
    pub fn select_engine(
        &self,
        registry: &EngineRegistry,
        manifest: &ModelManifest,
        device_cap: &DeviceCapability,
    ) -> Result<RoutingDecision> {
        let mut best_fallback: Option<RoutingDecision> = None;
        let mut unsupported_reasons: Vec<String> = Vec::new();

        for engine_name in &self.engines {
            let engine = match registry.get(engine_name) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let level = engine.supports(manifest, device_cap);
            match level {
                SupportLevel::Native => {
                    return Ok(RoutingDecision {
                        explanation: format!(
                            "engine '{}' natively supports model '{}' (family={:?}, dtype={:?}) on backend '{}'",
                            engine_name,
                            manifest.id,
                            manifest.family,
                            manifest.primary_dtype,
                            device_cap.backend_name
                        ),
                        engine_name: engine_name.clone(),
                        support_level: SupportLevel::Native,
                    });
                }
                SupportLevel::Fallback(ref reason) => {
                    if best_fallback.is_none() {
                        best_fallback = Some(RoutingDecision {
                            explanation: format!(
                                "engine '{}' can handle model '{}' with fallback: {}",
                                engine_name, manifest.id, reason
                            ),
                            engine_name: engine_name.clone(),
                            support_level: level,
                        });
                    }
                }
                SupportLevel::Unsupported(ref reason) => {
                    unsupported_reasons.push(format!("{}: {}", engine_name, reason));
                }
            }
        }

        if let Some(decision) = best_fallback {
            return Ok(decision);
        }

        Err(anyhow!(
            "no engine can handle model '{}' (family={:?}, dtype={:?}) on backend '{}': {}",
            manifest.id,
            manifest.family,
            manifest.primary_dtype,
            device_cap.backend_name,
            if unsupported_reasons.is_empty() {
                "no engines registered".to_string()
            } else {
                unsupported_reasons.join("; ")
            }
        ))
    }

    /// Human-readable explanation of why a particular engine would be selected.
    pub fn explain_decision(
        &self,
        registry: &EngineRegistry,
        manifest: &ModelManifest,
        device_cap: &DeviceCapability,
    ) -> String {
        match self.select_engine(registry, manifest, device_cap) {
            Ok(decision) => decision.explanation,
            Err(e) => format!("routing failed: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ModelInput;
    use crate::model::EchoTextModel;
    use crate::pipeline::InferencePipeline;
    use bloomai_backend::CpuBackend;
    use bloomai_core::{
        DType, DeviceCapability, DeviceClass, DeviceKind, GenerationParams, MemoryTopology,
        Modality, ModelFamily, ModelFormat, ModelIoSchema, ModelManifest, ModelMemoryProfile,
        PowerState, QuantScheme, QuantizationInfo, RuntimeHints, ThermalState, constants::GIB,
    };

    struct MockEngine;
    impl Engine for MockEngine {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn supported_modalities(&self) -> Vec<Modality> {
            vec![Modality::Text]
        }
        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }
        fn capability(&self) -> EngineCapability {
            EngineCapability {
                engine_name: "mock",
                supported_families: vec![ModelFamily::Llama, ModelFamily::Qwen],
                supported_dtypes: vec![DType::F32, DType::F16],
                supported_formats: vec![ModelFormat::Safetensors],
                supported_devices: vec![DeviceClass::Cpu],
                supported_modalities: vec![Modality::Text],
                supports_streaming: true,
                supports_quantized_models: false,
                supports_embeddings: true,
                supports_rerank: true,
                supports_structured_output: true,
                max_context_tokens: Some(4096),
                supported_quant_methods: vec![],
                supported_parallel_strategies: vec![
                    crate::core::parallelism::ParallelStrategy::None,
                ],
                maturity: BackendMaturity::Experimental,
                diagnostic_tips: vec![],
                construction_guide: String::new(),
            }
        }
        fn load(&self, _model_path: &Path, _device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
            Ok(Box::new(EchoTextModel::default()))
        }
    }

    fn cpu_capability() -> DeviceCapability {
        DeviceCapability {
            backend_name: "cpu".to_string(),
            vendor: None,
            device_class: DeviceClass::Cpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: 8 * GIB as usize,
            available_memory: 6 * GIB as usize,
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16, DType::Q8, DType::Q4],
            supported_formats: vec![ModelFormat::Safetensors, ModelFormat::Gguf],
            supports_mmap: true,
            has_quantization_kernels: true,
            supports_streaming: false,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: Some(4),
        }
    }

    fn text_manifest(family: ModelFamily, dtype: DType) -> ModelManifest {
        ModelManifest {
            id: "test-model".to_string(),
            family,
            version: "1.0".to_string(),
            io_schema: ModelIoSchema {
                inputs: vec![Modality::Text],
                outputs: vec![Modality::Text],
            },
            primary_dtype: dtype,
            memory_profile: ModelMemoryProfile {
                min_ram_bytes: 0,
                min_vram_bytes: 0,
                recommended_ram_bytes: 0,
                recommended_vram_bytes: 0,
            },
            runtime_hints: RuntimeHints::default(),
            ..ModelManifest::default()
        }
    }

    #[test]
    fn test_engine_registry() {
        let mut registry = EngineRegistry::default();
        assert!(registry.get("mock").is_err());

        registry.register("mock", Box::new(MockEngine));
        assert!(registry.get("mock").is_ok());
        assert_eq!(registry.names(), vec!["mock"]);
    }

    #[test]
    fn test_pipeline_with_mock_engine() {
        let engine = MockEngine;
        let backend = CpuBackend;
        let pipeline = InferencePipeline::load(&engine, &backend, Path::new("")).unwrap();

        let input = ModelInput::Text {
            prompt: "hello".to_string(),
        };
        let params = GenerationParams::default();

        let output = pipeline.run(input, &params).unwrap();
        assert_eq!(output.text.unwrap(), "echo: hello");
    }

    #[test]
    fn pipeline_rejects_invalid_generation_params_before_backend_execution() {
        let engine = MockEngine;
        let backend = CpuBackend;
        let pipeline = InferencePipeline::load(&engine, &backend, Path::new("")).unwrap();
        let input = ModelInput::Text {
            prompt: "hello".to_string(),
        };
        let params = GenerationParams {
            max_tokens: 0,
            ..GenerationParams::default()
        };

        let error = pipeline.run(input, &params).unwrap_err().to_string();
        assert!(error.contains("max_tokens"));
    }

    // ------------------------------------------------------------------
    // EngineCapability tests
    // ------------------------------------------------------------------

    #[test]
    fn test_mock_capability_fields() {
        let engine = MockEngine;
        let cap = engine.capability();
        assert_eq!(cap.engine_name, "mock");
        assert!(cap.supported_families.contains(&ModelFamily::Llama));
        assert!(cap.supported_families.contains(&ModelFamily::Qwen));
        assert!(cap.supported_dtypes.contains(&DType::F32));
        assert!(cap.supported_formats.contains(&ModelFormat::Safetensors));
        assert!(cap.supported_devices.contains(&DeviceClass::Cpu));
        assert!(cap.supports_streaming);
        assert!(!cap.supports_quantized_models);
        assert_eq!(cap.max_context_tokens, Some(4096));
    }

    #[test]
    fn test_supports_native_match() {
        let engine = MockEngine;
        let manifest = text_manifest(ModelFamily::Llama, DType::F32);
        let level = engine.supports(&manifest, &cpu_capability());
        assert_eq!(level, SupportLevel::Native);
    }

    #[test]
    fn test_supports_rejects_skeleton_capability() {
        let engine = MockEngine;
        let mut cap = engine.capability();
        cap.maturity = BackendMaturity::Skeleton;
        let manifest = text_manifest(ModelFamily::Llama, DType::F32);
        let level = default_engine_supports(&cap, &manifest, &cpu_capability());
        assert!(matches!(level, SupportLevel::Unsupported(_)));
        assert!(level.reason().unwrap().contains("skeleton"));
    }

    #[test]
    fn test_supports_unsupported_family() {
        let engine = MockEngine;
        let manifest = text_manifest(ModelFamily::FunAsr, DType::F32);
        let level = engine.supports(&manifest, &cpu_capability());
        assert!(matches!(level, SupportLevel::Unsupported(_)));
        assert!(level.reason().unwrap().contains("FunAsr"));
    }

    #[test]
    fn test_supports_fallback_dtype() {
        let engine = MockEngine;
        let manifest = text_manifest(ModelFamily::Llama, DType::Q4);
        let level = engine.supports(&manifest, &cpu_capability());
        assert!(matches!(level, SupportLevel::Fallback(_)));
        assert!(level.reason().unwrap().contains("Q4"));
    }

    #[test]
    fn test_supports_unsupported_device() {
        let engine = MockEngine;
        let manifest = text_manifest(ModelFamily::Llama, DType::F32);
        let mut gpu_cap = cpu_capability();
        gpu_cap.device_class = DeviceClass::DiscreteGpu;
        let level = engine.supports(&manifest, &gpu_cap);
        assert!(matches!(level, SupportLevel::Unsupported(_)));
    }

    #[test]
    fn test_supports_unsupported_modality() {
        let engine = MockEngine;
        let mut manifest = text_manifest(ModelFamily::Llama, DType::F32);
        manifest.io_schema.inputs = vec![Modality::Audio];
        let level = engine.supports(&manifest, &cpu_capability());
        assert!(matches!(level, SupportLevel::Unsupported(_)));
    }

    #[test]
    fn test_supports_unsupported_format() {
        let engine = MockEngine;
        let mut manifest = text_manifest(ModelFamily::Llama, DType::F32);
        manifest.files = vec![bloomai_core::ModelFile {
            name: "model.onnx".to_string(),
            format: ModelFormat::Onnx,
            size_bytes: 1000,
            hash_sha256: None,
            required: true,
        }];
        let level = engine.supports(&manifest, &cpu_capability());
        assert!(matches!(level, SupportLevel::Unsupported(_)));
    }

    #[test]
    fn test_supports_rejects_mixed_package_with_unsupported_artifact() {
        let engine = MockEngine;
        let mut manifest = text_manifest(ModelFamily::Llama, DType::F32);
        manifest.files = vec![
            bloomai_core::ModelFile {
                name: "model.safetensors".to_string(),
                format: ModelFormat::Safetensors,
                size_bytes: 1000,
                hash_sha256: None,
                required: true,
            },
            bloomai_core::ModelFile {
                name: "fallback.onnx".to_string(),
                format: ModelFormat::Onnx,
                size_bytes: 1000,
                hash_sha256: None,
                required: false,
            },
            bloomai_core::ModelFile {
                name: "config.json".to_string(),
                format: ModelFormat::Unknown,
                size_bytes: 100,
                hash_sha256: None,
                required: true,
            },
        ];
        let level = engine.supports(&manifest, &cpu_capability());
        assert!(matches!(level, SupportLevel::Unsupported(_)));
        assert!(level.reason().unwrap().contains("Onnx"));
    }

    #[test]
    fn test_supports_quantization_method_before_load() {
        let engine = MockEngine;
        let mut manifest = text_manifest(ModelFamily::Llama, DType::F32);
        manifest.quantization = Some(QuantizationInfo {
            scheme: QuantScheme::AWQ,
            bits: 4,
            group_size: Some(128),
            act_order: false,
            kv_cache_dtype: None,
            imatrix: false,
        });
        let level = engine.supports(&manifest, &cpu_capability());
        assert!(matches!(level, SupportLevel::Fallback(_)));
        assert!(level.reason().unwrap().contains("quantized models"));
    }

    // ------------------------------------------------------------------
    // EngineRouter tests
    // ------------------------------------------------------------------

    #[test]
    fn test_engine_router_select_native() {
        let mut registry = EngineRegistry::default();
        registry.register("mock", Box::new(MockEngine));
        let router = EngineRouter::from_registry(&registry);

        let manifest = text_manifest(ModelFamily::Qwen, DType::F32);
        let decision = router
            .select_engine(&registry, &manifest, &cpu_capability())
            .unwrap();
        assert_eq!(decision.engine_name, "mock");
        assert_eq!(decision.support_level, SupportLevel::Native);
        assert!(decision.explanation.contains("natively supports"));
    }

    #[test]
    fn test_engine_router_fallback() {
        let mut registry = EngineRegistry::default();
        registry.register("mock", Box::new(MockEngine));
        let router = EngineRouter::from_registry(&registry);

        // Q4 is not in MockEngine's supported_dtypes → fallback
        let manifest = text_manifest(ModelFamily::Llama, DType::Q4);
        let decision = router
            .select_engine(&registry, &manifest, &cpu_capability())
            .unwrap();
        assert_eq!(decision.engine_name, "mock");
        assert!(matches!(decision.support_level, SupportLevel::Fallback(_)));
    }

    #[test]
    fn test_engine_router_no_engine() {
        let registry = EngineRegistry::default();
        let router = EngineRouter::from_registry(&registry);

        let manifest = text_manifest(ModelFamily::Llama, DType::F32);
        let result = router.select_engine(&registry, &manifest, &cpu_capability());
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_router_explain() {
        let mut registry = EngineRegistry::default();
        registry.register("mock", Box::new(MockEngine));
        let router = EngineRouter::from_registry(&registry);

        let manifest = text_manifest(ModelFamily::Llama, DType::F32);
        let explanation = router.explain_decision(&registry, &manifest, &cpu_capability());
        assert!(explanation.contains("mock"));
        assert!(explanation.contains("natively"));
    }

    #[test]
    fn test_default_capability_derived_from_legacy() {
        // An engine that only overrides supported_modalities/supported_devices
        // should get a derived capability from the default implementation.
        struct LegacyEngine;
        impl Engine for LegacyEngine {
            fn name(&self) -> &'static str {
                "legacy"
            }
            fn supported_modalities(&self) -> Vec<Modality> {
                vec![Modality::Audio]
            }
            fn supported_devices(&self) -> Vec<DeviceKind> {
                vec![DeviceKind::Cpu, DeviceKind::Gpu]
            }
            fn load(&self, _: &Path, _: DeviceKind) -> Result<Box<dyn LoadedModel>> {
                Ok(Box::new(EchoTextModel::default()))
            }
        }

        let engine = LegacyEngine;
        let cap = engine.capability();
        assert_eq!(cap.engine_name, "legacy");
        assert!(cap.supported_modalities.contains(&Modality::Audio));
        assert!(cap.supported_devices.contains(&DeviceClass::Cpu));
        assert!(cap.supported_devices.contains(&DeviceClass::IntegratedGpu));
        assert!(cap.supported_devices.contains(&DeviceClass::DiscreteGpu));
        // Default capability has all dtypes
        assert!(cap.supported_dtypes.contains(&DType::F32));
    }
}
