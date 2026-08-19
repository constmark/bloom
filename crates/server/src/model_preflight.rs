//! Bounded, cached compatibility inspection before a catalog model is loaded.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bloomai_backend::BackendRegistry;
use bloomai_core::{
    DType, DeviceKind, Modality, ModelFamily, ModelFormat, ModelManifest, QuantScheme,
};
use bloomai_engine::{MemoryPreallocationConfig, SupportLevel, model_manifest_tasks};
use serde::Serialize;
use tokio::sync::{Mutex, Semaphore};

use super::model_manager::ModelCatalog;
use super::{engine_registry, select_backend_name};

const PREFLIGHT_CACHE_TTL: Duration = Duration::from_secs(30);
const MAX_PREFLIGHT_CACHE_ENTRIES: usize = 128;
const MAX_PREFLIGHT_ID_CHARS: usize = 256;
const MAX_PREFLIGHT_METADATA_CHARS: usize = 4_096;
const MAX_PREFLIGHT_LIST_ITEMS: usize = 32;
const MAX_PREFLIGHT_LIST_ITEM_CHARS: usize = 2_048;
pub(crate) const MODEL_PREFLIGHT_SCHEMA_VERSION: u32 = 1;
pub(crate) const MODEL_PREFLIGHT_OBJECT: &str = "bloom.model_preflight";

#[derive(Debug, Clone)]
pub(crate) struct ModelPreflightConfig {
    pub(crate) backend: String,
    pub(crate) speculative: String,
    pub(crate) device: DeviceKind,
    pub(crate) context_size: usize,
    pub(crate) max_concurrent: usize,
    pub(crate) memory_utilization: f64,
    pub(crate) reserve_memory_bytes: Option<usize>,
    pub(crate) disable_memory_prealloc: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelManifestSummary {
    pub(crate) id: String,
    pub(crate) family: String,
    pub(crate) version: String,
    pub(crate) model_tasks: Vec<String>,
    pub(crate) description: Option<String>,
    pub(crate) license: Option<String>,
    pub(crate) input_modalities: Vec<String>,
    pub(crate) output_modalities: Vec<String>,
    pub(crate) formats: Vec<String>,
    pub(crate) primary_dtype: String,
    pub(crate) quantization: Option<String>,
    pub(crate) quantization_bits: Option<u8>,
    pub(crate) parameter_count: Option<u64>,
    pub(crate) context_length: Option<u64>,
    pub(crate) num_layers: Option<u64>,
    pub(crate) hidden_size: Option<u64>,
    pub(crate) vocab_size: Option<u64>,
    pub(crate) supports_mmap: bool,
    pub(crate) requires_streaming: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelRuntimeCompatibility {
    pub(crate) configured_engine: String,
    pub(crate) selected_engine: String,
    pub(crate) engine_maturity: String,
    pub(crate) device: String,
    pub(crate) device_backend: String,
    pub(crate) backend_available: bool,
    pub(crate) backend_reason: Option<String>,
    pub(crate) support: String,
    pub(crate) support_reason: Option<String>,
    pub(crate) diagnostic_tips: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ModelMemoryPreflight {
    pub(crate) per_request_context_tokens: u64,
    pub(crate) max_concurrent: u64,
    pub(crate) planned_context_tokens: u64,
    pub(crate) weight_bytes: u64,
    pub(crate) host_weight_bytes: u64,
    pub(crate) device_weight_bytes: u64,
    pub(crate) kv_cache_bytes: u64,
    pub(crate) kv_cache_bytes_per_token: u64,
    pub(crate) temp_tensor_bytes: u64,
    pub(crate) total_bytes: u64,
    pub(crate) available_bytes: Option<u64>,
    pub(crate) budget_bytes: Option<u64>,
    pub(crate) reserve_bytes: u64,
    pub(crate) memory_utilization: f64,
    pub(crate) preallocation_enabled: bool,
    pub(crate) fits_budget: bool,
    pub(crate) scope: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ModelPreflightReport {
    pub(crate) model_id: String,
    pub(crate) inspected_at: u64,
    pub(crate) loadable: bool,
    pub(crate) load_blocker: Option<String>,
    pub(crate) manifest: ModelManifestSummary,
    pub(crate) runtime: ModelRuntimeCompatibility,
    pub(crate) memory: ModelMemoryPreflight,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelPreflightError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Internal(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelFingerprint {
    path: PathBuf,
    length: u64,
    modified_nanos: Option<u128>,
    descriptor_length: Option<u64>,
    descriptor_modified_nanos: Option<u128>,
}

struct CachedPreflight {
    cached_at: Instant,
    fingerprint: ModelFingerprint,
    report: ModelPreflightReport,
}

pub(crate) struct ModelPreflightManager {
    models_root: PathBuf,
    config: ModelPreflightConfig,
    cache: Mutex<HashMap<String, CachedPreflight>>,
    inspection: Semaphore,
}

impl ModelPreflightManager {
    pub(crate) fn new(models_root: PathBuf, config: ModelPreflightConfig) -> Arc<Self> {
        Arc::new(Self {
            models_root,
            config,
            cache: Mutex::new(HashMap::new()),
            inspection: Semaphore::new(1),
        })
    }

    pub(crate) async fn inspect(
        &self,
        model_id: &str,
    ) -> Result<ModelPreflightReport, ModelPreflightError> {
        let model_id = model_id.trim().to_string();
        let fingerprint = resolve_fingerprint(self.models_root.clone(), model_id.clone()).await?;
        if let Some(report) = self.cached(&model_id, &fingerprint).await {
            return Ok(report);
        }

        let _permit = self.inspection.acquire().await.map_err(|error| {
            ModelPreflightError::Internal(format!("model preflight worker stopped: {error}"))
        })?;
        let fingerprint = resolve_fingerprint(self.models_root.clone(), model_id.clone()).await?;
        if let Some(report) = self.cached(&model_id, &fingerprint).await {
            return Ok(report);
        }

        let config = self.config.clone();
        let path = fingerprint.path.clone();
        let report = tokio::task::spawn_blocking(move || build_report(&model_id, &path, &config))
            .await
            .map_err(|error| {
                ModelPreflightError::Internal(format!(
                    "model preflight inspection task failed: {error}"
                ))
            })??;
        self.store(report.model_id.clone(), fingerprint, report.clone())
            .await;
        Ok(report)
    }

    async fn cached(
        &self,
        model_id: &str,
        fingerprint: &ModelFingerprint,
    ) -> Option<ModelPreflightReport> {
        let cache = self.cache.lock().await;
        cache.get(model_id).and_then(|entry| {
            (entry.cached_at.elapsed() <= PREFLIGHT_CACHE_TTL && &entry.fingerprint == fingerprint)
                .then(|| entry.report.clone())
        })
    }

    async fn store(
        &self,
        model_id: String,
        fingerprint: ModelFingerprint,
        report: ModelPreflightReport,
    ) {
        let mut cache = self.cache.lock().await;
        cache.retain(|_, entry| entry.cached_at.elapsed() <= PREFLIGHT_CACHE_TTL);
        if cache.len() >= MAX_PREFLIGHT_CACHE_ENTRIES
            && !cache.contains_key(&model_id)
            && let Some(oldest) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.cached_at)
                .map(|(key, _)| key.clone())
        {
            cache.remove(&oldest);
        }
        cache.insert(
            model_id,
            CachedPreflight {
                cached_at: Instant::now(),
                fingerprint,
                report,
            },
        );
    }
}

async fn resolve_fingerprint(
    root: PathBuf,
    model_id: String,
) -> Result<ModelFingerprint, ModelPreflightError> {
    tokio::task::spawn_blocking(move || {
        let path = ModelCatalog::resolve(&root, &model_id)
            .map_err(|error| ModelPreflightError::Invalid(error.to_string()))?;
        fingerprint(&path).map_err(|error| ModelPreflightError::Invalid(error.to_string()))
    })
    .await
    .map_err(|error| {
        ModelPreflightError::Internal(format!("model preflight catalog task failed: {error}"))
    })?
}

fn fingerprint(path: &Path) -> std::io::Result<ModelFingerprint> {
    let metadata = std::fs::metadata(path)?;
    let descriptor = if metadata.is_dir() {
        ["bloom.json", "config.json"]
            .into_iter()
            .map(|name| path.join(name))
            .find(|candidate| candidate.is_file())
    } else {
        None
    };
    let descriptor_metadata = descriptor.as_ref().map(std::fs::metadata).transpose()?;
    Ok(ModelFingerprint {
        path: path.to_path_buf(),
        length: metadata.len(),
        modified_nanos: modified_nanos(&metadata),
        descriptor_length: descriptor_metadata.as_ref().map(std::fs::Metadata::len),
        descriptor_modified_nanos: descriptor_metadata.as_ref().and_then(modified_nanos),
    })
}

fn modified_nanos(metadata: &std::fs::Metadata) -> Option<u128> {
    metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_nanos())
}

fn build_report(
    model_id: &str,
    path: &Path,
    config: &ModelPreflightConfig,
) -> Result<ModelPreflightReport, ModelPreflightError> {
    let manifest = bloomai_engine::load_manifest(path).map_err(|error| {
        ModelPreflightError::Invalid(format!("Model metadata could not be inspected: {error}"))
    })?;
    let selected_engine = select_backend_name(&config.backend, &config.speculative, &manifest);
    let engines = engine_registry();
    let engine = engines.get(&selected_engine).map_err(|error| {
        ModelPreflightError::Internal(format!("selected engine is unavailable: {error}"))
    })?;
    let engine_capability = engine.capability();

    let device_backend = device_backend_name(config.device);
    let backends = BackendRegistry::default();
    let backend = backends.get(device_backend).map_err(|error| {
        ModelPreflightError::Internal(format!("configured device backend is unavailable: {error}"))
    })?;
    let availability = backend.availability();
    let device_capability = backend.capability();
    let engine_support = engine.supports(&manifest, &device_capability);
    let format_blocker = unsupported_device_format(&manifest, &device_capability.supported_formats);

    let planned_context = config
        .context_size
        .saturating_mul(config.max_concurrent.max(1));
    let estimate =
        bloomai_engine::estimate_memory_for_device(&manifest, planned_context, config.device);
    let memory_plan = bloomai_engine::plan_memory_preallocation(
        estimate.clone(),
        MemoryPreallocationConfig {
            enabled: !config.disable_memory_prealloc,
            memory_utilization: config.memory_utilization,
            reserve_memory_bytes: config.reserve_memory_bytes,
        },
    );
    let memory_error = memory_plan.as_ref().err().map(ToString::to_string);
    let available_bytes = memory_plan
        .as_ref()
        .ok()
        .and_then(|plan| plan.available_bytes)
        .or_else(bloomai_engine::available_system_memory);
    let budget_bytes = memory_plan
        .as_ref()
        .ok()
        .and_then(|plan| plan.budget_bytes)
        .or_else(|| {
            (!config.disable_memory_prealloc)
                .then_some(available_bytes?)
                .and_then(|available| {
                    config
                        .memory_utilization
                        .is_finite()
                        .then_some((available as f64 * config.memory_utilization) as usize)
                })
        });
    let reserve_bytes = memory_plan
        .as_ref()
        .ok()
        .map(|plan| plan.reserve_bytes)
        .unwrap_or_else(|| {
            if config.disable_memory_prealloc {
                0
            } else {
                config.reserve_memory_bytes.unwrap_or_else(|| {
                    estimate
                        .kv_cache_bytes
                        .saturating_add(estimate.temp_tensor_bytes)
                })
            }
        });
    let reported_memory_utilization = if config.memory_utilization.is_finite() {
        config.memory_utilization
    } else {
        0.0
    };

    let support_reason = engine_support.reason().map(ToString::to_string);
    let support = match &engine_support {
        SupportLevel::Native => "native",
        SupportLevel::Fallback(_) => "fallback",
        SupportLevel::Unsupported(_) => "unsupported",
    };
    let load_blocker = if !availability.available {
        Some(
            availability
                .reason
                .clone()
                .unwrap_or_else(|| format!("Device backend '{device_backend}' is unavailable.")),
        )
    } else if !engine_support.is_supported() {
        support_reason.clone()
    } else if let Some(reason) = format_blocker {
        Some(reason)
    } else {
        memory_error.clone()
    };

    let mut warnings = Vec::new();
    if let SupportLevel::Fallback(reason) = &engine_support {
        warnings.push(reason.clone());
    }
    if engine_capability.maturity != bloomai_engine::BackendMaturity::Production {
        warnings.push(format!(
            "The selected '{}' engine is marked {}.",
            selected_engine, engine_capability.maturity
        ));
    }
    if config.disable_memory_prealloc {
        warnings.push(
            "Startup memory preallocation is disabled; the estimate is advisory.".to_string(),
        );
    }

    let report = ModelPreflightReport {
        model_id: model_id.to_string(),
        inspected_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        loadable: load_blocker.is_none(),
        load_blocker,
        manifest: summarize_manifest(&manifest),
        runtime: ModelRuntimeCompatibility {
            configured_engine: config.backend.clone(),
            selected_engine,
            engine_maturity: engine_capability.maturity.to_string(),
            device: device_name(config.device).to_string(),
            device_backend: device_backend.to_string(),
            backend_available: availability.available,
            backend_reason: availability.reason,
            support: support.to_string(),
            support_reason,
            diagnostic_tips: engine_capability.diagnostic_tips,
        },
        memory: ModelMemoryPreflight {
            per_request_context_tokens: as_u64(config.context_size),
            max_concurrent: as_u64(config.max_concurrent.max(1)),
            planned_context_tokens: as_u64(planned_context),
            weight_bytes: as_u64(estimate.weight_bytes),
            host_weight_bytes: as_u64(estimate.host_weight_bytes),
            device_weight_bytes: as_u64(estimate.device_weight_bytes),
            kv_cache_bytes: as_u64(estimate.kv_cache_bytes),
            kv_cache_bytes_per_token: as_u64(estimate.kv_cache_bytes_per_token),
            temp_tensor_bytes: as_u64(estimate.temp_tensor_bytes),
            total_bytes: as_u64(estimate.total_bytes),
            available_bytes: available_bytes.map(as_u64),
            budget_bytes: budget_bytes.map(as_u64),
            reserve_bytes: as_u64(reserve_bytes),
            memory_utilization: reported_memory_utilization,
            preallocation_enabled: !config.disable_memory_prealloc,
            fits_budget: memory_error.is_none(),
            scope: estimate.memory_scope,
        },
        warnings,
    };
    validate_public_report(&report).map_err(ModelPreflightError::Invalid)?;
    Ok(report)
}

fn validate_public_report(report: &ModelPreflightReport) -> Result<(), String> {
    validate_text("catalog model ID", &report.model_id, MAX_PREFLIGHT_ID_CHARS)?;
    if report.inspected_at == 0 {
        return Err("Model preflight produced an invalid inspection time.".to_string());
    }

    validate_text("manifest ID", &report.manifest.id, MAX_PREFLIGHT_ID_CHARS)?;
    validate_text(
        "model family",
        &report.manifest.family,
        MAX_PREFLIGHT_ID_CHARS,
    )?;
    validate_text(
        "model version",
        &report.manifest.version,
        MAX_PREFLIGHT_ID_CHARS,
    )?;
    if !matches!(report.manifest.model_tasks.as_slice(), [generation] if generation == "generation")
        && !matches!(
            report.manifest.model_tasks.as_slice(),
            [embedding, rerank] if embedding == "embedding" && rerank == "rerank"
        )
    {
        return Err("Model preflight produced an invalid task identity.".to_string());
    }
    validate_optional_text("model description", report.manifest.description.as_deref())?;
    validate_optional_text("model license", report.manifest.license.as_deref())?;
    validate_list(
        "input modalities",
        &report.manifest.input_modalities,
        16,
        64,
    )?;
    validate_list(
        "output modalities",
        &report.manifest.output_modalities,
        16,
        64,
    )?;
    if report.manifest.input_modalities.is_empty() || report.manifest.output_modalities.is_empty() {
        return Err("Model preflight produced an empty modality contract.".to_string());
    }
    validate_list("model formats", &report.manifest.formats, 32, 64)?;
    validate_text("primary dtype", &report.manifest.primary_dtype, 64)?;
    if let Some(quantization) = report.manifest.quantization.as_deref() {
        validate_text("quantization", quantization, 128)?;
    }
    if report
        .manifest
        .quantization_bits
        .is_some_and(|bits| !(1..=16).contains(&bits))
    {
        return Err("Model preflight produced invalid quantization bits.".to_string());
    }

    for (label, value) in [
        (
            "configured engine",
            report.runtime.configured_engine.as_str(),
        ),
        ("selected engine", report.runtime.selected_engine.as_str()),
        ("engine maturity", report.runtime.engine_maturity.as_str()),
        ("device", report.runtime.device.as_str()),
        ("device backend", report.runtime.device_backend.as_str()),
        ("runtime support", report.runtime.support.as_str()),
        ("memory scope", report.memory.scope.as_str()),
    ] {
        validate_text(label, value, MAX_PREFLIGHT_ID_CHARS)?;
    }
    if !matches!(
        report.runtime.support.as_str(),
        "native" | "fallback" | "unsupported"
    ) {
        return Err("Model preflight produced an invalid runtime support level.".to_string());
    }
    validate_optional_text("backend reason", report.runtime.backend_reason.as_deref())?;
    validate_optional_text("support reason", report.runtime.support_reason.as_deref())?;
    validate_list(
        "runtime diagnostic tips",
        &report.runtime.diagnostic_tips,
        MAX_PREFLIGHT_LIST_ITEMS,
        MAX_PREFLIGHT_LIST_ITEM_CHARS,
    )?;
    validate_list(
        "warnings",
        &report.warnings,
        MAX_PREFLIGHT_LIST_ITEMS,
        MAX_PREFLIGHT_LIST_ITEM_CHARS,
    )?;
    if report.memory.per_request_context_tokens == 0
        || report.memory.max_concurrent == 0
        || report.memory.planned_context_tokens
            != report
                .memory
                .per_request_context_tokens
                .saturating_mul(report.memory.max_concurrent)
        || !report.memory.memory_utilization.is_finite()
        || !(0.0..=1.0).contains(&report.memory.memory_utilization)
    {
        return Err("Model preflight produced an invalid memory plan.".to_string());
    }
    if report.loadable != report.load_blocker.is_none() {
        return Err("Model preflight produced an inconsistent load decision.".to_string());
    }
    if let Some(load_blocker) = report.load_blocker.as_deref() {
        validate_text("load blocker", load_blocker, MAX_PREFLIGHT_METADATA_CHARS)?;
    }
    if report.loadable
        && (!report.runtime.backend_available
            || report.runtime.support == "unsupported"
            || !report.memory.fits_budget)
    {
        return Err("Model preflight produced an unsafe load decision.".to_string());
    }
    Ok(())
}

fn validate_optional_text(label: &str, value: Option<&str>) -> Result<(), String> {
    value.map_or(Ok(()), |value| {
        validate_text(label, value, MAX_PREFLIGHT_METADATA_CHARS)
    })
}

fn validate_list(
    label: &str,
    values: &[String],
    max_items: usize,
    max_chars: usize,
) -> Result<(), String> {
    if values.len() > max_items {
        return Err(format!("Model preflight produced too many {label}."));
    }
    let unique = values.iter().collect::<BTreeSet<_>>();
    if unique.len() != values.len() {
        return Err(format!("Model preflight produced duplicate {label}."));
    }
    values
        .iter()
        .try_for_each(|value| validate_text(label, value, max_chars))
}

fn validate_text(label: &str, value: &str, max_chars: usize) -> Result<(), String> {
    let length = value.chars().count();
    if length == 0
        || length > max_chars
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(format!("Model preflight produced invalid {label}."))
    } else {
        Ok(())
    }
}

fn unsupported_device_format(
    manifest: &ModelManifest,
    supported: &[ModelFormat],
) -> Option<String> {
    if manifest.files.is_empty() || supported.is_empty() {
        return None;
    }
    manifest
        .files
        .iter()
        .all(|file| !supported.contains(&file.format))
        .then(|| {
            format!(
                "Device backend does not advertise support for model formats {}.",
                manifest
                    .files
                    .iter()
                    .map(|file| format_name(file.format))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

fn summarize_manifest(manifest: &ModelManifest) -> ModelManifestSummary {
    let formats = manifest
        .files
        .iter()
        .map(|file| format_name(file.format).to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    ModelManifestSummary {
        id: manifest.id.clone(),
        family: family_name(&manifest.family),
        version: manifest.version.clone(),
        model_tasks: model_manifest_tasks(manifest)
            .iter()
            .map(|task| (*task).to_string())
            .collect(),
        description: manifest.description.clone(),
        license: manifest.license.clone(),
        input_modalities: manifest
            .io_schema
            .inputs
            .iter()
            .map(|value| modality_name(*value).to_string())
            .collect(),
        output_modalities: manifest
            .io_schema
            .outputs
            .iter()
            .map(|value| modality_name(*value).to_string())
            .collect(),
        formats,
        primary_dtype: dtype_name(manifest.primary_dtype).to_string(),
        quantization: manifest
            .quantization
            .as_ref()
            .map(|value| quantization_name(&value.scheme)),
        quantization_bits: manifest.quantization.as_ref().map(|value| value.bits),
        parameter_count: parameter(manifest, &["parameter_count", "num_parameters", "n_params"]),
        context_length: parameter(
            manifest,
            &[
                "max_position_embeddings",
                "max_sequence_length",
                "context_length",
                "n_ctx",
            ],
        ),
        num_layers: parameter(
            manifest,
            &["num_hidden_layers", "num_layers", "block_count"],
        ),
        hidden_size: parameter(manifest, &["hidden_size", "embedding_length"]),
        vocab_size: parameter(manifest, &["vocab_size", "tokenizer_vocab_size"]),
        supports_mmap: manifest.runtime_hints.supports_mmap,
        requires_streaming: manifest.runtime_hints.requires_streaming,
    }
}

fn parameter(manifest: &ModelManifest, names: &[&str]) -> Option<u64> {
    names.iter().find_map(|name| {
        manifest
            .parameters
            .get(*name)
            .and_then(serde_json::Value::as_u64)
    })
}

fn family_name(value: &ModelFamily) -> String {
    match value {
        ModelFamily::Llama => "llama".to_string(),
        ModelFamily::Qwen => "qwen".to_string(),
        ModelFamily::Gemma => "gemma".to_string(),
        ModelFamily::Bert => "bert".to_string(),
        ModelFamily::Whisper => "whisper".to_string(),
        ModelFamily::FunAsr => "funasr".to_string(),
        ModelFamily::Custom(value) => value.clone(),
    }
}

fn dtype_name(value: DType) -> &'static str {
    match value {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I4 => "i4",
        DType::NF4 => "nf4",
        DType::Q8 => "q8",
        DType::Q4 => "q4",
        DType::Unknown => "unknown",
    }
}

fn format_name(value: ModelFormat) -> &'static str {
    match value {
        ModelFormat::Gguf => "gguf",
        ModelFormat::Safetensors => "safetensors",
        ModelFormat::OpenVinoIr => "openvino_ir",
        ModelFormat::TensorRtEngine => "tensorrt_engine",
        ModelFormat::Onnx => "onnx",
        ModelFormat::CoreMl => "coreml",
        ModelFormat::TorchScript => "torchscript",
        ModelFormat::Mlx => "mlx",
        ModelFormat::VulkanSpirv => "vulkan_spirv",
        ModelFormat::VendorBundle => "vendor_bundle",
        ModelFormat::Unknown => "unknown",
    }
}

fn modality_name(value: Modality) -> &'static str {
    match value {
        Modality::Text => "text",
        Modality::Audio => "audio",
        Modality::Vision => "vision",
        Modality::Multi => "multi",
    }
}

fn quantization_name(value: &QuantScheme) -> String {
    match value {
        QuantScheme::None => "none".to_string(),
        QuantScheme::GGUF(value) => value.to_ascii_lowercase(),
        QuantScheme::AWQ => "awq".to_string(),
        QuantScheme::GPTQ => "gptq".to_string(),
        QuantScheme::INT4 => "int4".to_string(),
        QuantScheme::INT8 => "int8".to_string(),
        QuantScheme::NF4 => "nf4".to_string(),
    }
}

fn device_backend_name(device: DeviceKind) -> &'static str {
    match device {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Gpu if cfg!(target_os = "macos") => "metal",
        DeviceKind::Gpu => "cuda",
        DeviceKind::Npu => "intel-npu",
    }
}

fn device_name(device: DeviceKind) -> &'static str {
    match device {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Gpu => "gpu",
        DeviceKind::Npu => "npu",
    }
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_config() -> ModelPreflightConfig {
        ModelPreflightConfig {
            backend: "candle".to_string(),
            speculative: "none".to_string(),
            device: DeviceKind::Cpu,
            context_size: 128,
            max_concurrent: 1,
            memory_utilization: 0.75,
            reserve_memory_bytes: None,
            disable_memory_prealloc: false,
        }
    }

    fn write_supported_manifest(root: &Path, name: &str) {
        let model = root.join(name);
        std::fs::create_dir(&model).unwrap();
        std::fs::write(
            model.join("bloom.json"),
            r#"{
                "id":"tiny-llama",
                "family":"Llama",
                "version":"1",
                "license":"Apache-2.0",
                "io_schema":{"inputs":["Text"],"outputs":["Text"]},
                "memory_profile":{"min_ram_bytes":1048576,"min_vram_bytes":0,"recommended_ram_bytes":2097152,"recommended_vram_bytes":0},
                "files":[],
                "parameters":{"num_hidden_layers":2,"hidden_size":128,"vocab_size":1000,"max_position_embeddings":4096},
                "runtime_hints":{"preferred_backends":["candle"],"supports_mmap":true,"requires_streaming":false},
                "primary_dtype":"F32"
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn manifest_summary_publishes_encoder_tasks_before_load() {
        let mut manifest = ModelManifest {
            id: "tiny-encoder".to_string(),
            family: ModelFamily::Bert,
            version: "1".to_string(),
            ..ModelManifest::default()
        };
        manifest
            .parameters
            .insert("bloom_task".to_string(), serde_json::json!("embedding"));

        assert_eq!(
            summarize_manifest(&manifest).model_tasks,
            ["embedding", "rerank"]
        );
    }

    #[tokio::test]
    async fn supported_cpu_manifest_is_loadable() {
        let temp = tempfile::tempdir().unwrap();
        write_supported_manifest(temp.path(), "tiny");
        let manager = ModelPreflightManager::new(temp.path().to_path_buf(), cpu_config());

        let report = manager.inspect("tiny").await.unwrap();

        assert!(report.loadable, "{:?}", report.load_blocker);
        assert_eq!(report.manifest.family, "llama");
        assert_eq!(report.manifest.model_tasks, ["generation"]);
        assert_eq!(report.manifest.context_length, Some(4096));
        assert_eq!(report.runtime.selected_engine, "candle");
        assert_eq!(report.runtime.support, "native");
        assert_eq!(report.memory.planned_context_tokens, 128);
    }

    #[tokio::test]
    async fn oversized_public_metadata_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        write_supported_manifest(temp.path(), "tiny");
        let manifest_path = temp.path().join("tiny/bloom.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["description"] = serde_json::json!("x".repeat(MAX_PREFLIGHT_METADATA_CHARS + 1));
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let manager = ModelPreflightManager::new(temp.path().to_path_buf(), cpu_config());

        let error = manager.inspect("tiny").await.unwrap_err().to_string();

        assert!(error.contains("invalid model description"), "{error}");
    }

    #[tokio::test]
    async fn skeleton_onnx_engine_is_reported_before_load() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tiny.onnx"), b"onnx").unwrap();
        let manager = ModelPreflightManager::new(temp.path().to_path_buf(), cpu_config());

        let report = manager.inspect("tiny.onnx").await.unwrap();

        assert!(!report.loadable);
        assert_eq!(report.runtime.selected_engine, "onnxruntime");
        assert_eq!(report.runtime.engine_maturity, "skeleton");
        assert!(
            report
                .load_blocker
                .as_deref()
                .unwrap()
                .contains("skeleton adapter")
        );
    }

    #[tokio::test]
    async fn impossible_memory_reservation_blocks_queue_admission() {
        let temp = tempfile::tempdir().unwrap();
        write_supported_manifest(temp.path(), "tiny");
        let mut config = cpu_config();
        config.reserve_memory_bytes = Some(usize::MAX);
        let manager = ModelPreflightManager::new(temp.path().to_path_buf(), config);

        let report = manager.inspect("tiny").await.unwrap();

        assert!(!report.loadable);
        assert!(!report.memory.fits_budget);
        assert!(
            report
                .load_blocker
                .as_deref()
                .unwrap()
                .contains("reservation")
        );
    }

    #[tokio::test]
    async fn file_fingerprint_invalidates_cached_report() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tiny.onnx");
        std::fs::write(&path, b"one").unwrap();
        let manager = ModelPreflightManager::new(temp.path().to_path_buf(), cpu_config());
        let first = manager.inspect("tiny.onnx").await.unwrap();

        std::fs::write(&path, vec![0_u8; 4096]).unwrap();
        let second = manager.inspect("tiny.onnx").await.unwrap();

        assert_ne!(first.memory.weight_bytes, second.memory.weight_bytes);
    }
}
