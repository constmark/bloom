//! Bounded compatibility inspection before a catalog model is loaded.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bloomai_backend::BackendRegistry;
use bloomai_core::{
    DType, DeviceKind, Modality, ModelFamily, ModelFormat, ModelManifest, QuantScheme,
};
use bloomai_engine::{SupportLevel, model_manifest_tasks};
use serde::Serialize;
use tokio::sync::Semaphore;

use super::model_manager::ModelCatalog;
use super::runtime_memory::{RuntimeMemoryFootprint, RuntimeMemoryPlanner};
use super::{engine_registry, select_backend_name, validate_strict_runtime_backend};

const MAX_PREFLIGHT_ID_CHARS: usize = 256;
const MAX_PREFLIGHT_METADATA_CHARS: usize = 4_096;
const MAX_PREFLIGHT_LIST_ITEMS: usize = 32;
const MAX_PREFLIGHT_LIST_ITEM_CHARS: usize = 2_048;
pub(crate) const MODEL_PREFLIGHT_SCHEMA_VERSION: u32 = 2;
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
    pub(crate) enable_ifb: bool,
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
    /// Legacy single-domain fields. These are `None` for discrete devices,
    /// where host and device bytes cannot be safely combined.
    pub(crate) total_bytes: Option<u64>,
    pub(crate) available_bytes: Option<u64>,
    pub(crate) budget_bytes: Option<u64>,
    pub(crate) host_required_bytes: u64,
    pub(crate) host_available_bytes: Option<u64>,
    pub(crate) host_limit_bytes: u64,
    pub(crate) host_committed_bytes: u64,
    pub(crate) device_required_bytes: u64,
    pub(crate) device_available_bytes: Option<u64>,
    pub(crate) device_limit_bytes: u64,
    pub(crate) device_committed_bytes: u64,
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

pub(crate) struct ModelPreflightManager {
    models_root: PathBuf,
    config: ModelPreflightConfig,
    runtime_memory: RuntimeMemoryPlanner,
    inspection: Semaphore,
}

impl ModelPreflightManager {
    pub(crate) fn new(
        models_root: PathBuf,
        config: ModelPreflightConfig,
        runtime_memory: RuntimeMemoryPlanner,
    ) -> Arc<Self> {
        Arc::new(Self {
            models_root,
            config,
            runtime_memory,
            inspection: Semaphore::new(1),
        })
    }

    pub(crate) async fn inspect(
        &self,
        model_id: &str,
    ) -> Result<ModelPreflightReport, ModelPreflightError> {
        self.inspect_inner(model_id, true).await
    }

    /// Revalidate static compatibility for an already-resident runtime without
    /// treating selection as admission of a second physical model copy.
    pub(crate) async fn inspect_resident(
        &self,
        model_id: &str,
    ) -> Result<ModelPreflightReport, ModelPreflightError> {
        self.inspect_inner(model_id, false).await
    }

    async fn inspect_inner(
        &self,
        model_id: &str,
        require_candidate_memory: bool,
    ) -> Result<ModelPreflightReport, ModelPreflightError> {
        let model_id = model_id.trim().to_string();
        let _permit = self.inspection.acquire().await.map_err(|error| {
            ModelPreflightError::Internal(format!("model preflight worker stopped: {error}"))
        })?;
        let path = resolve_preflight_path(self.models_root.clone(), model_id.clone()).await?;

        let config = self.config.clone();
        let runtime_memory = self.runtime_memory.clone();
        let report = tokio::task::spawn_blocking(move || {
            build_report(
                &model_id,
                &path,
                &config,
                &runtime_memory,
                require_candidate_memory,
            )
        })
        .await
        .map_err(|error| {
            ModelPreflightError::Internal(format!(
                "model preflight inspection task failed: {error}"
            ))
        })??;
        Ok(report)
    }
}

async fn resolve_preflight_path(
    root: PathBuf,
    model_id: String,
) -> Result<PathBuf, ModelPreflightError> {
    tokio::task::spawn_blocking(move || {
        let path = ModelCatalog::resolve(&root, &model_id)
            .map_err(|error| ModelPreflightError::Invalid(error.to_string()))?;
        std::fs::metadata(&path)
            .map_err(|error| ModelPreflightError::Invalid(error.to_string()))?;
        Ok(path)
    })
    .await
    .map_err(|error| {
        ModelPreflightError::Internal(format!("model preflight catalog task failed: {error}"))
    })?
}

fn build_report(
    model_id: &str,
    path: &Path,
    config: &ModelPreflightConfig,
    runtime_memory: &RuntimeMemoryPlanner,
    require_candidate_memory: bool,
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

    let planned_context_overflow = config
        .context_size
        .checked_mul(config.max_concurrent.max(1))
        .is_none();
    let planned_context = config
        .context_size
        .saturating_mul(config.max_concurrent.max(1));
    let fallback_estimate =
        bloomai_engine::estimate_memory_for_device(&manifest, planned_context, config.device);
    let mut budget = runtime_memory.snapshot();
    let mut memory_error = None;
    let (estimate, footprint, available) = if planned_context_overflow {
        memory_error = Some("runtime context memory estimate overflow".to_string());
        let footprint = fallback_footprint(runtime_memory, &manifest, &fallback_estimate, &budget);
        (fallback_estimate, footprint, None)
    } else {
        match runtime_memory.plan_for_context(&manifest, planned_context, config.device) {
            Ok(candidate) => (
                candidate.estimate,
                candidate.footprint,
                Some(candidate.available),
            ),
            Err(error) => {
                memory_error = Some(error.to_string());
                let footprint =
                    fallback_footprint(runtime_memory, &manifest, &fallback_estimate, &budget);
                (fallback_estimate, footprint, None)
            }
        }
    };
    let reserve_bytes = if config.disable_memory_prealloc {
        0
    } else {
        config.reserve_memory_bytes.unwrap_or_else(|| {
            estimate
                .kv_cache_bytes
                .saturating_add(estimate.temp_tensor_bytes)
        })
    };
    if memory_error.is_none()
        && let Some(available) = available
    {
        let evaluation = runtime_memory.evaluate_candidate(footprint, available);
        budget = evaluation.snapshot;
        if let Err(error) = evaluation.verdict {
            memory_error = Some(error.to_string());
        } else if reserve_bytes > available.host_bytes {
            memory_error =
                Some("startup page-touch reservation exceeds available host memory".to_string());
        }
    }
    let discrete = budget.device_limit_bytes > 0;
    let host_budget_available = budget
        .host_limit_bytes
        .saturating_sub(budget.host_used_bytes);
    let legacy_total_bytes = (!discrete).then_some(as_u64(footprint.host_bytes));
    let legacy_available_bytes = if discrete {
        None
    } else {
        available.map(|value| as_u64(value.host_bytes))
    };
    let legacy_budget_bytes = if discrete {
        None
    } else {
        available.map(|value| as_u64(host_budget_available.min(value.host_bytes)))
    };
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
    let ifb_blocker = (config.enable_ifb && selected_engine != "candle").then(|| {
        format!(
            "IFB currently requires the Candle engine, but model selection resolved to '{selected_engine}'."
        )
    });
    let strict_backend_blocker = validate_strict_runtime_backend(&selected_engine)
        .err()
        .map(|error| error.to_string());
    let fits_budget = memory_error.is_none();
    let load_blocker = if !availability.available {
        Some(
            availability
                .reason
                .clone()
                .unwrap_or_else(|| format!("Device backend '{device_backend}' is unavailable.")),
        )
    } else if let Some(reason) = ifb_blocker.clone() {
        Some(reason)
    } else if !engine_support.is_supported() {
        support_reason.clone()
    } else if let Some(reason) = format_blocker {
        Some(reason)
    } else if let Some(reason) = strict_backend_blocker {
        Some(reason)
    } else if require_candidate_memory {
        memory_error.clone()
    } else {
        None
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
            "Startup page-touch preallocation is disabled; strict aggregate runtime memory admission remains enforced."
                .to_string(),
        );
    }
    warnings.push(
        "Memory availability and aggregate usage are a bounded snapshot and are revalidated atomically when loading begins."
            .to_string(),
    );

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
            total_bytes: legacy_total_bytes,
            available_bytes: legacy_available_bytes,
            budget_bytes: legacy_budget_bytes,
            host_required_bytes: as_u64(footprint.host_bytes),
            host_available_bytes: available.map(|value| as_u64(value.host_bytes)),
            host_limit_bytes: as_u64(budget.host_limit_bytes),
            host_committed_bytes: as_u64(budget.host_used_bytes),
            device_required_bytes: as_u64(footprint.device_bytes),
            device_available_bytes: available.map(|value| as_u64(value.device_bytes)),
            device_limit_bytes: as_u64(budget.device_limit_bytes),
            device_committed_bytes: as_u64(budget.device_used_bytes),
            reserve_bytes: as_u64(reserve_bytes),
            memory_utilization: reported_memory_utilization,
            preallocation_enabled: !config.disable_memory_prealloc,
            fits_budget,
            scope: if discrete {
                "aggregate_host_device".to_string()
            } else {
                "aggregate_unified".to_string()
            },
        },
        warnings,
    };
    validate_public_report(&report, require_candidate_memory)
        .map_err(ModelPreflightError::Invalid)?;
    Ok(report)
}

fn fallback_footprint(
    runtime_memory: &RuntimeMemoryPlanner,
    manifest: &ModelManifest,
    estimate: &bloomai_engine::MemoryEstimate,
    budget: &super::runtime_memory::RuntimeMemoryBudgetSnapshot,
) -> RuntimeMemoryFootprint {
    runtime_memory.footprint(manifest, estimate).unwrap_or({
        if budget.device_limit_bytes > 0 {
            RuntimeMemoryFootprint {
                host_bytes: usize::MAX,
                device_bytes: usize::MAX,
            }
        } else {
            RuntimeMemoryFootprint {
                host_bytes: usize::MAX,
                device_bytes: 0,
            }
        }
    })
}

fn validate_public_report(
    report: &ModelPreflightReport,
    require_candidate_memory: bool,
) -> Result<(), String> {
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
    if report.memory.host_committed_bytes > report.memory.host_limit_bytes
        || report.memory.device_committed_bytes > report.memory.device_limit_bytes
    {
        return Err("Model preflight produced inconsistent memory accounting.".to_string());
    }
    let host_fits = report.memory.host_available_bytes.is_some_and(|available| {
        report.memory.host_required_bytes <= available
            && report.memory.host_required_bytes
                <= report
                    .memory
                    .host_limit_bytes
                    .saturating_sub(report.memory.host_committed_bytes)
            && report.memory.reserve_bytes <= available
    });
    let device_fits = report
        .memory
        .device_available_bytes
        .is_some_and(|available| {
            report.memory.device_required_bytes <= available
                && report.memory.device_required_bytes
                    <= report
                        .memory
                        .device_limit_bytes
                        .saturating_sub(report.memory.device_committed_bytes)
        });
    match report.memory.scope.as_str() {
        "aggregate_unified" => {
            let expected_budget = report.memory.host_available_bytes.map(|available| {
                available.min(
                    report
                        .memory
                        .host_limit_bytes
                        .saturating_sub(report.memory.host_committed_bytes),
                )
            });
            if report.memory.total_bytes != Some(report.memory.host_required_bytes)
                || report.memory.available_bytes != report.memory.host_available_bytes
                || report.memory.budget_bytes != expected_budget
                || report.memory.device_required_bytes != 0
                || report.memory.device_limit_bytes != 0
                || report.memory.device_committed_bytes != 0
            {
                return Err(
                    "Model preflight produced invalid unified-memory compatibility fields."
                        .to_string(),
                );
            }
            if report.memory.fits_budget != host_fits {
                return Err(
                    "Model preflight produced an inconsistent unified-memory decision.".to_string(),
                );
            }
        }
        "aggregate_host_device" => {
            if report.memory.total_bytes.is_some()
                || report.memory.available_bytes.is_some()
                || report.memory.budget_bytes.is_some()
                || report.memory.device_limit_bytes == 0
            {
                return Err(
                    "Model preflight produced invalid discrete-memory compatibility fields."
                        .to_string(),
                );
            }
            if report.memory.fits_budget != (host_fits && device_fits) {
                return Err(
                    "Model preflight produced an inconsistent discrete-memory decision."
                        .to_string(),
                );
            }
        }
        _ => return Err("Model preflight produced an invalid memory scope.".to_string()),
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
            || (require_candidate_memory && !report.memory.fits_budget))
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
            enable_ifb: false,
        }
    }

    fn test_runtime_memory() -> RuntimeMemoryPlanner {
        RuntimeMemoryPlanner::for_test(usize::MAX, 0, bloomai_core::MemoryTopology::Unified)
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
        let manager = ModelPreflightManager::new(
            temp.path().to_path_buf(),
            cpu_config(),
            test_runtime_memory(),
        );

        let report = manager.inspect("tiny").await.unwrap();

        assert!(report.loadable, "{:?}", report.load_blocker);
        assert_eq!(report.manifest.family, "llama");
        assert_eq!(report.manifest.model_tasks, ["generation"]);
        assert_eq!(report.manifest.context_length, Some(4096));
        assert_eq!(report.runtime.selected_engine, "candle");
        assert_eq!(report.runtime.support, "native");
        assert_eq!(report.memory.planned_context_tokens, 128);
        assert_eq!(
            report.memory.total_bytes,
            Some(report.memory.host_required_bytes)
        );
        assert_eq!(
            report.memory.budget_bytes,
            report.memory.host_available_bytes.map(|available| {
                available.min(
                    report
                        .memory
                        .host_limit_bytes
                        .saturating_sub(report.memory.host_committed_bytes),
                )
            })
        );
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
        let manager = ModelPreflightManager::new(
            temp.path().to_path_buf(),
            cpu_config(),
            test_runtime_memory(),
        );

        let error = manager.inspect("tiny").await.unwrap_err().to_string();

        assert!(error.contains("invalid model description"), "{error}");
    }

    #[tokio::test]
    async fn skeleton_onnx_engine_is_reported_before_load() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tiny.onnx"), b"onnx").unwrap();
        let manager = ModelPreflightManager::new(
            temp.path().to_path_buf(),
            cpu_config(),
            test_runtime_memory(),
        );

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
        let manager =
            ModelPreflightManager::new(temp.path().to_path_buf(), config, test_runtime_memory());

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
    async fn disabling_page_touch_does_not_bypass_hard_runtime_admission() {
        let temp = tempfile::tempdir().unwrap();
        write_supported_manifest(temp.path(), "tiny");
        let mut config = cpu_config();
        config.disable_memory_prealloc = true;
        let manager = ModelPreflightManager::new(
            temp.path().to_path_buf(),
            config,
            RuntimeMemoryPlanner::for_test(1, 0, bloomai_core::MemoryTopology::Unified),
        );

        let report = manager.inspect("tiny").await.unwrap();

        assert!(!report.loadable);
        assert!(!report.memory.preallocation_enabled);
        assert!(!report.memory.fits_budget);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("remains enforced"))
        );
        assert!(
            report
                .load_blocker
                .as_deref()
                .is_some_and(|blocker| blocker.contains("budget is exhausted"))
        );
        assert!(report.memory.host_required_bytes > 1);
    }

    #[tokio::test]
    async fn model_file_changes_are_reflected_immediately() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tiny.onnx");
        std::fs::write(&path, b"one").unwrap();
        let manager = ModelPreflightManager::new(
            temp.path().to_path_buf(),
            cpu_config(),
            test_runtime_memory(),
        );
        let first = manager.inspect("tiny.onnx").await.unwrap();

        std::fs::write(&path, vec![0_u8; 4096]).unwrap();
        let second = manager.inspect("tiny.onnx").await.unwrap();

        assert_ne!(first.memory.weight_bytes, second.memory.weight_bytes);
    }

    #[tokio::test]
    async fn released_runtime_budget_is_reflected_immediately() {
        let temp = tempfile::tempdir().unwrap();
        write_supported_manifest(temp.path(), "tiny");
        let manifest = bloomai_engine::load_manifest(&temp.path().join("tiny")).unwrap();
        let sizing = test_runtime_memory();
        let estimate = sizing.planned_estimate(&manifest).unwrap();
        let footprint = sizing.footprint(&manifest, &estimate).unwrap();
        let host_limit = footprint
            .host_bytes
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .unwrap();
        let planner =
            RuntimeMemoryPlanner::for_test(host_limit, 0, bloomai_core::MemoryTopology::Unified);
        let resident = planner
            .reserve(
                footprint,
                RuntimeMemoryFootprint {
                    host_bytes: usize::MAX,
                    device_bytes: 0,
                },
            )
            .unwrap();
        let manager = ModelPreflightManager::new(temp.path().to_path_buf(), cpu_config(), planner);

        let blocked = manager.inspect("tiny").await.unwrap();
        assert!(!blocked.loadable);
        assert_eq!(
            blocked.memory.host_required_bytes,
            as_u64(footprint.host_bytes)
        );
        drop(resident);
        let released = manager.inspect("tiny").await.unwrap();

        assert!(released.loadable, "{:?}", released.load_blocker);
        assert_eq!(released.memory.host_committed_bytes, 0);
    }

    #[tokio::test]
    async fn resident_selection_ignores_only_incremental_memory_failure() {
        let temp = tempfile::tempdir().unwrap();
        write_supported_manifest(temp.path(), "tiny");
        let manager = ModelPreflightManager::new(
            temp.path().to_path_buf(),
            cpu_config(),
            RuntimeMemoryPlanner::for_test(1, 0, bloomai_core::MemoryTopology::Unified),
        );

        let candidate = manager.inspect("tiny").await.unwrap();
        let resident = manager.inspect_resident("tiny").await.unwrap();

        assert!(!candidate.loadable);
        assert!(!resident.memory.fits_budget);
        assert!(resident.loadable, "{:?}", resident.load_blocker);

        std::fs::write(temp.path().join("unsupported.onnx"), b"onnx").unwrap();
        let unsupported = manager.inspect_resident("unsupported.onnx").await.unwrap();
        assert!(!unsupported.loadable);
        assert_eq!(unsupported.runtime.support, "unsupported");
    }

    #[tokio::test]
    async fn ifb_rejects_non_candle_engine_before_load() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tiny.onnx"), b"onnx").unwrap();
        let mut config = cpu_config();
        config.enable_ifb = true;
        let manager =
            ModelPreflightManager::new(temp.path().to_path_buf(), config, test_runtime_memory());

        let report = manager.inspect("tiny.onnx").await.unwrap();

        assert!(!report.loadable);
        assert!(report.memory.fits_budget);
        assert!(
            report
                .load_blocker
                .as_deref()
                .is_some_and(|blocker| blocker.contains("IFB currently requires the Candle engine"))
        );
    }

    #[tokio::test]
    async fn discrete_preflight_uses_only_domain_specific_memory_fields() {
        let temp = tempfile::tempdir().unwrap();
        write_supported_manifest(temp.path(), "tiny");
        let manager = ModelPreflightManager::new(
            temp.path().to_path_buf(),
            cpu_config(),
            RuntimeMemoryPlanner::for_test(
                usize::MAX,
                usize::MAX,
                bloomai_core::MemoryTopology::Discrete,
            ),
        );

        let report = manager.inspect("tiny").await.unwrap();

        assert_eq!(report.memory.scope, "aggregate_host_device");
        assert_eq!(report.memory.total_bytes, None);
        assert_eq!(report.memory.available_bytes, None);
        assert_eq!(report.memory.budget_bytes, None);
        assert!(report.memory.host_required_bytes > 0);
        assert!(report.memory.device_required_bytes > 0);
    }
}
