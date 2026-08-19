//! Side-effect-free deployment checks for `bloom_server --doctor`.

use std::net::SocketAddr;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use bloomai_backend::BackendRegistry;
use bloomai_core::DeviceKind;
use bloomai_engine::{MemoryPreallocationConfig, SpeculativeMode, SupportLevel};
use serde::Serialize;
use tokio::sync::Semaphore;

use super::cli::{Args, DoctorFormat};
use super::model_index::validate_configuration as validate_model_index_configuration;
use super::model_index_state::inspect_model_index_watermark_directory;
use super::model_license::ModelLicensePolicy;
use super::{
    BrowserOriginPolicy, MAX_SHUTDOWN_TIMEOUT_SECONDS, ModelCatalog, engine_registry,
    parse_browser_origin_policy, select_backend_name,
};

const DOCTOR_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl CheckStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct DoctorCheck {
    id: &'static str,
    status: CheckStatus,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    remediation: Option<String>,
}

impl DoctorCheck {
    fn pass(id: &'static str, message: impl Into<String>) -> Self {
        Self {
            id,
            status: CheckStatus::Pass,
            message: message.into(),
            remediation: None,
        }
    }

    fn warn(id: &'static str, message: impl Into<String>, remediation: impl Into<String>) -> Self {
        Self {
            id,
            status: CheckStatus::Warn,
            message: message.into(),
            remediation: Some(remediation.into()),
        }
    }

    fn fail(id: &'static str, message: impl Into<String>, remediation: impl Into<String>) -> Self {
        Self {
            id,
            status: CheckStatus::Fail,
            message: message.into(),
            remediation: Some(remediation.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct DoctorSummary {
    passed: usize,
    warnings: usize,
    failures: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct DoctorReport {
    schema_version: u32,
    object: &'static str,
    created: u64,
    bloom_version: &'static str,
    status: &'static str,
    summary: DoctorSummary,
    checks: Vec<DoctorCheck>,
}

impl DoctorReport {
    pub(crate) fn has_failures(&self) -> bool {
        self.summary.failures > 0
    }

    pub(crate) fn render(&self, format: DoctorFormat) -> Result<String> {
        match format {
            DoctorFormat::Text => Ok(self.render_text()),
            DoctorFormat::Json => {
                let json = serde_json::to_string_pretty(self)?;
                Ok(format!("{json}\n"))
            }
        }
    }

    fn render_text(&self) -> String {
        let mut output = format!(
            "Bloom server doctor {}\nResult: {}\n\n",
            self.bloom_version,
            self.status.to_ascii_uppercase()
        );
        for check in &self.checks {
            output.push_str(&format!(
                "[{}] {}: {}\n",
                check.status.label(),
                check.id,
                check.message
            ));
            if let Some(remediation) = &check.remediation {
                output.push_str(&format!("       Next: {remediation}\n"));
            }
        }
        output.push_str(&format!(
            "\nSummary: {} passed, {} warning(s), {} failure(s)\n",
            self.summary.passed, self.summary.warnings, self.summary.failures
        ));
        output
    }
}

pub(crate) fn validate_server_arguments(args: &Args) -> Result<()> {
    let errors = server_argument_errors(args);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "invalid server configuration:\n  - {}",
            errors.join("\n  - ")
        ))
    }
}

pub(crate) fn inspect_server(
    args: &Args,
    config_present: bool,
    models_root: &Path,
) -> DoctorReport {
    let mut checks = Vec::new();
    checks.push(if config_present {
        DoctorCheck::pass(
            "configuration",
            "The configuration file parsed and command-line overrides were applied.",
        )
    } else {
        DoctorCheck::pass(
            "configuration",
            "No configuration file was present; built-in defaults and command-line values are effective.",
        )
    });

    let argument_errors = server_argument_errors(args);
    checks.push(if argument_errors.is_empty() {
        DoctorCheck::pass(
            "arguments",
            "Numeric limits and feature dependencies are internally consistent.",
        )
    } else {
        DoctorCheck::fail(
            "arguments",
            argument_errors.join(" "),
            "Correct the reported flags or configuration fields before starting the server.",
        )
    });
    checks.push(network_check(args));
    checks.push(engine_check(args));

    let (device_kind, device_check) = device_check(args);
    checks.push(device_check);

    let (catalog_count, catalog_check) = catalog_check(models_root);
    checks.push(catalog_check);
    checks.push(startup_model_check(args, device_kind, catalog_count));
    checks.push(storage_check(args, models_root));
    checks.push(license_policy_check(args));
    checks.push(model_index_check(args));
    checks.push(model_index_state_check(args));
    checks.push(if super::ui::embedded_ui_available() {
        DoctorCheck::pass(
            "embedded_ui",
            "This binary contains the Bloom browser UI and can serve it from the root path.",
        )
    } else {
        DoctorCheck::warn(
            "embedded_ui",
            "This binary does not contain the Bloom browser UI.",
            "Use a release built with --features serve-ui after generating ui/dist, or host the standalone UI separately.",
        )
    });

    let summary = DoctorSummary {
        passed: checks
            .iter()
            .filter(|check| check.status == CheckStatus::Pass)
            .count(),
        warnings: checks
            .iter()
            .filter(|check| check.status == CheckStatus::Warn)
            .count(),
        failures: checks
            .iter()
            .filter(|check| check.status == CheckStatus::Fail)
            .count(),
    };
    let status = if summary.failures > 0 {
        "fail"
    } else if summary.warnings > 0 {
        "warn"
    } else {
        "pass"
    };
    DoctorReport {
        schema_version: DOCTOR_SCHEMA_VERSION,
        object: "bloom.server_doctor",
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        bloom_version: env!("CARGO_PKG_VERSION"),
        status,
        summary,
        checks,
    }
}

fn server_argument_errors(args: &Args) -> Vec<String> {
    let mut errors = Vec::new();
    match parse_browser_origin_policy(&args.cors_allow_origin) {
        Ok(BrowserOriginPolicy::Any) if args.strict_security => errors
            .push("Strict security does not allow the wildcard browser origin policy.".to_string()),
        Ok(_) => {}
        Err(error) => errors.push(format!("Browser origin policy is invalid: {error}.")),
    }
    if args.open_browser && !super::ui::embedded_ui_available() {
        errors.push(
            "Opening a browser requires a bloom_server binary built with the embedded UI."
                .to_string(),
        );
    }
    if args.max_concurrent == 0 {
        errors.push("Maximum concurrency must be at least 1.".to_string());
    }
    if args.max_concurrent > Semaphore::MAX_PERMITS {
        errors.push(format!(
            "Maximum concurrency must not exceed this platform's runtime limit of {}.",
            Semaphore::MAX_PERMITS
        ));
    }
    if args.context_size == 0 {
        errors.push("Context size must be at least 1 token.".to_string());
    }
    if args.context_size.checked_mul(args.max_concurrent).is_none() {
        errors.push("Context size multiplied by concurrency overflows this platform.".to_string());
    }
    if !(1..=MAX_SHUTDOWN_TIMEOUT_SECONDS).contains(&args.shutdown_timeout_seconds) {
        errors.push(format!(
            "Shutdown timeout must be between 1 and {MAX_SHUTDOWN_TIMEOUT_SECONDS} seconds."
        ));
    }
    if !args.memory_utilization.is_finite() || !(0.05..=0.95).contains(&args.memory_utilization) {
        errors.push("Memory utilization must be between 0.05 and 0.95.".to_string());
    }
    for (name, value) in [
        ("Maximum tokens per scheduling step", args.max_num_tokens),
        ("Maximum JSON body size", args.max_body_bytes),
        ("Maximum upload size", args.max_upload_bytes),
        (
            "Maximum model import chunk size",
            args.max_model_import_chunk_bytes,
        ),
    ] {
        if value == 0 {
            errors.push(format!("{name} must be greater than zero."));
        }
    }
    if args.enable_model_downloads && args.max_model_download_bytes == 0 {
        errors.push(
            "The model download limit must be greater than zero when downloads are enabled."
                .to_string(),
        );
    }
    if args.enable_model_imports && args.max_model_import_bytes == 0 {
        errors.push(
            "The model import limit must be greater than zero when imports are enabled."
                .to_string(),
        );
    }
    if let Err(error) = ModelLicensePolicy::new(args.allowed_model_licenses.clone()) {
        errors.push(format!("Model license policy is invalid: {error}."));
    }
    if let Err(error) = validate_model_index_configuration(
        args.model_index_file.clone(),
        args.model_index_url.clone(),
        args.model_index_public_key.clone(),
        args.model_index_public_keys.clone(),
        args.model_index_refresh_seconds,
    ) {
        errors.push(format!("Model index configuration is invalid: {error}."));
    }
    let model_index_configured = args.model_index_file.is_some() || args.model_index_url.is_some();
    match args.model_index_state_dir.as_ref() {
        None if model_index_configured => errors.push(
            "A model index state directory is required for persistent rollback protection."
                .to_string(),
        ),
        Some(path) if path.as_os_str().is_empty() => {
            errors.push("The model index state directory path cannot be empty.".to_string())
        }
        _ => {}
    }
    if args
        .api_key
        .as_ref()
        .is_some_and(|key| key.trim().is_empty())
    {
        errors.push("The API key cannot be empty or whitespace-only.".to_string());
    }
    if let Ok(address) = format!("{}:{}", args.host, args.port).parse::<SocketAddr>()
        && !address.ip().is_loopback()
        && !args.api_key.as_ref().is_some_and(|key| !key.is_empty())
    {
        if !args.allow_unauthenticated_network {
            errors.push(
                "A non-loopback listener requires an API key unless the explicit development-only unauthenticated-network override is enabled."
                    .to_string(),
            );
        } else if args.strict_security {
            errors.push(
                "Strict security does not allow the unauthenticated-network override.".to_string(),
            );
        }
    }
    if let Some(dtype) = &args.dtype
        && !matches!(
            dtype.trim().to_ascii_lowercase().as_str(),
            "f32" | "float32" | "f16" | "float16" | "bf16" | "bfloat16"
        )
    {
        errors.push("Dtype must be f32/float32, f16/float16, or bf16/bfloat16.".to_string());
    }
    if let Err(error) = SpeculativeMode::from_parts(
        &args.speculative,
        args.draft_model.clone(),
        args.num_speculative_tokens,
        args.speculative_ngram_order,
    ) {
        errors.push(format!("Speculative decoding is invalid: {error}."));
    }
    if args.enable_chunked_prefill && !args.enable_ifb {
        errors.push("Chunked prefill requires in-flight batching.".to_string());
    }
    if args.enable_chunked_prefill && args.prefill_chunk_size == 0 {
        errors.push("Chunked prefill size must be at least 1 token.".to_string());
    }
    if args.enable_cachemesh && !args.enable_ifb {
        errors.push("CacheMesh requires in-flight batching.".to_string());
    }
    if args.enable_cachemesh && args.cachemesh_l2_capacity_bytes == 0 {
        errors.push("CacheMesh L2 capacity must be greater than zero.".to_string());
    }
    if args.enable_cachemesh_l3 && !args.enable_cachemesh {
        errors.push("CacheMesh L3 requires CacheMesh to be enabled.".to_string());
    }
    if args.cachemesh_write_through_l3 && !args.enable_cachemesh_l3 {
        errors.push("CacheMesh write-through requires CacheMesh L3.".to_string());
    }
    match args
        .long_context_policy
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "full" => {}
        "sliding-window" | "sliding_window" | "sliding" => {
            if args.sliding_window_tokens == 0 {
                errors.push("The sliding-window policy requires a non-zero window.".to_string());
            }
        }
        "context-shift" | "context_shift" | "shift" => {
            let maximum = args.context_shift_max_tokens.unwrap_or(args.context_size);
            if maximum == 0
                || args.context_shift_tokens == 0
                || args.context_shift_tokens >= maximum
            {
                errors.push(
                    "The context-shift policy requires 0 < shift tokens < maximum context tokens."
                        .to_string(),
                );
            }
        }
        "compact-inactive" | "compact_inactive" | "compact" => {
            if args.compact_free_blocks == 0 {
                errors.push(
                    "The compact-inactive policy requires a non-zero free-block target."
                        .to_string(),
                );
            }
        }
        _ => errors.push(
            "Long-context policy must be full, sliding-window, context-shift, or compact-inactive."
                .to_string(),
        ),
    }
    errors
}

fn network_check(args: &Args) -> DoctorCheck {
    let address = match format!("{}:{}", args.host, args.port).parse::<SocketAddr>() {
        Ok(address) => address,
        Err(_) => {
            return DoctorCheck::fail(
                "network_security",
                "The configured host and port do not form a valid socket address.",
                "Use an IPv4 address or a bracketed IPv6 address with a valid port.",
            );
        }
    };
    let origin_policy = match parse_browser_origin_policy(&args.cors_allow_origin) {
        Ok(policy) => policy,
        Err(_) => {
            return DoctorCheck::fail(
                "network_security",
                "The configured browser origin policy is invalid.",
                "Use 'same-origin', one exact HTTP(S) origin without a path, or an explicit '*'.",
            );
        }
    };
    let api_key_set = args.api_key.as_ref().is_some_and(|key| !key.is_empty());
    if !address.ip().is_loopback() && !api_key_set {
        let remediation =
            "Set BLOOM_API_KEY and a narrow BLOOM_CORS_ALLOW_ORIGIN before exposing the server.";
        return if !args.allow_unauthenticated_network || args.strict_security {
            DoctorCheck::fail(
                "network_security",
                "A non-loopback listener has no API key and is not safely admissible.",
                remediation,
            )
        } else {
            DoctorCheck::warn(
                "network_security",
                "A non-loopback listener has no API key under the explicit development-only override.",
                remediation,
            )
        };
    }
    if origin_policy == BrowserOriginPolicy::Any {
        return if args.strict_security {
            DoctorCheck::fail(
                "network_security",
                "The wildcard browser origin policy conflicts with strict security.",
                "Use the default same-origin policy or set BLOOM_CORS_ALLOW_ORIGIN to one exact trusted UI origin.",
            )
        } else {
            DoctorCheck::warn(
                "network_security",
                "Every browser origin is allowed to attempt requests to this listener.",
                "Use the default same-origin policy or set BLOOM_CORS_ALLOW_ORIGIN to one exact trusted UI origin.",
            )
        };
    }
    if (args.enable_model_downloads || args.enable_model_imports) && !api_key_set {
        return DoctorCheck::warn(
            "network_security",
            "Model catalog writes are enabled without an API key.",
            "Set BLOOM_API_KEY even on localhost when untrusted local users or browser origins are possible.",
        );
    }
    DoctorCheck::pass(
        "network_security",
        "Listener authentication and browser-origin settings are appropriate for the configured bind scope.",
    )
}

fn engine_check(args: &Args) -> DoctorCheck {
    match engine_registry().get(&args.backend) {
        Ok(engine) => {
            let capability = engine.capability();
            if capability.maturity == bloomai_engine::BackendMaturity::Skeleton {
                DoctorCheck::warn(
                    "runtime_engine",
                    format!(
                        "The '{}' engine is a discovery-only skeleton in this build.",
                        args.backend
                    ),
                    "Choose an executable engine from the support matrix before loading a production model.",
                )
            } else {
                DoctorCheck::pass(
                    "runtime_engine",
                    format!(
                        "The '{}' engine is registered with {} maturity.",
                        args.backend, capability.maturity
                    ),
                )
            }
        }
        Err(_) => DoctorCheck::fail(
            "runtime_engine",
            format!(
                "The '{}' engine is not registered in this binary.",
                args.backend
            ),
            "Choose a supported --backend value reported by bloom_server --help.",
        ),
    }
}

fn configured_device(args: &Args) -> Result<(DeviceKind, &'static str)> {
    match args.device.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok((DeviceKind::Cpu, "cpu")),
        "gpu" | "cuda" | "metal" => Ok((
            DeviceKind::Gpu,
            if cfg!(target_os = "macos") {
                "metal"
            } else {
                "cuda"
            },
        )),
        "npu" | "intel-npu" => Ok((DeviceKind::Npu, "intel-npu")),
        _ => Err(anyhow!("unsupported device")),
    }
}

fn device_check(args: &Args) -> (Option<DeviceKind>, DoctorCheck) {
    let (device_kind, backend_name) = match configured_device(args) {
        Ok(configured) => configured,
        Err(_) => {
            return (
                None,
                DoctorCheck::fail(
                    "device_backend",
                    format!("The '{}' device selector is unsupported.", args.device),
                    "Choose cpu, gpu, cuda, metal, npu, or intel-npu.",
                ),
            );
        }
    };
    let registry = BackendRegistry::default();
    let check = match registry.get(backend_name) {
        Ok(backend) => {
            let availability = backend.availability();
            if availability.available {
                DoctorCheck::pass(
                    "device_backend",
                    format!("The '{backend_name}' device backend is available."),
                )
            } else {
                DoctorCheck::fail(
                    "device_backend",
                    format!(
                        "The '{backend_name}' device backend is unavailable on this host or build."
                    ),
                    "Choose --device cpu or install and compile the required accelerator runtime.",
                )
            }
        }
        Err(_) => DoctorCheck::fail(
            "device_backend",
            format!("The '{backend_name}' device backend is not registered."),
            "Choose --device cpu or rebuild Bloom with the required backend.",
        ),
    };
    (Some(device_kind), check)
}

fn catalog_check(models_root: &Path) -> (Option<usize>, DoctorCheck) {
    match ModelCatalog::scan(models_root, None) {
        Ok(catalog) if !catalog.root_exists => (
            Some(0),
            DoctorCheck::warn(
                "model_catalog",
                "The model catalog does not exist yet; the server will start with an empty catalog.",
                "Create the catalog, enable a verified acquisition path, or load an explicit --model.",
            ),
        ),
        Ok(catalog) if catalog.models.is_empty() => (
            Some(0),
            DoctorCheck::warn(
                "model_catalog",
                "The model catalog is readable but contains no recognized models.",
                "Import a supported model or point --models-dir at a populated dedicated catalog.",
            ),
        ),
        Ok(catalog) => {
            let count = catalog.models.len();
            (
                Some(count),
                DoctorCheck::pass(
                    "model_catalog",
                    format!("The catalog is readable and contains {count} recognized model(s)."),
                ),
            )
        }
        Err(_) => (
            None,
            DoctorCheck::fail(
                "model_catalog",
                "The model catalog could not be safely scanned.",
                "Ensure --models-dir names a readable directory without unsafe entries.",
            ),
        ),
    }
}

fn startup_model_check(
    args: &Args,
    device_kind: Option<DeviceKind>,
    catalog_count: Option<usize>,
) -> DoctorCheck {
    let Some(path) = args.model.as_ref() else {
        return if catalog_count.is_some_and(|count| count > 0) {
            DoctorCheck::pass(
                "startup_model",
                "No model is preloaded; a recognized catalog model can be selected through the UI or API.",
            )
        } else {
            DoctorCheck::warn(
                "startup_model",
                "No startup model or recognized catalog model is available.",
                "Add a model before expecting readiness; liveness and model-management APIs can still start.",
            )
        };
    };
    if !path.exists() || (!path.is_file() && !path.is_dir()) {
        return DoctorCheck::fail(
            "startup_model",
            "The configured startup model does not exist as a regular file or directory.",
            "Correct --model or the server.model configuration path.",
        );
    }
    let manifest = match bloomai_engine::load_manifest(path) {
        Ok(manifest) => manifest,
        Err(_) => {
            return DoctorCheck::fail(
                "startup_model",
                "The configured startup model metadata could not be validated.",
                "Run inspect_gguf or correct the model manifest and required files.",
            );
        }
    };
    let Some(device_kind) = device_kind else {
        return DoctorCheck::fail(
            "startup_model",
            "Model compatibility cannot be evaluated with an invalid device selector.",
            "Correct --device and run the doctor again.",
        );
    };
    let (_, device_backend_name) = match configured_device(args) {
        Ok(configured) => configured,
        Err(_) => unreachable!("device_kind was already validated"),
    };
    let device_registry = BackendRegistry::default();
    let device_capability = match device_registry.get(device_backend_name) {
        Ok(backend) if backend.availability().available => backend.capability(),
        _ => {
            return DoctorCheck::fail(
                "startup_model",
                "Model compatibility cannot be evaluated because the configured device is unavailable.",
                "Correct the device runtime or choose --device cpu.",
            );
        }
    };
    let selected_engine = select_backend_name(&args.backend, &args.speculative, &manifest);
    let engines = engine_registry();
    let engine = match engines.get(&selected_engine) {
        Ok(engine) => engine,
        Err(_) => {
            return DoctorCheck::fail(
                "startup_model",
                "Automatic routing selected an engine that is not present in this binary.",
                "Choose a supported backend or rebuild Bloom with the required engine.",
            );
        }
    };
    let support = engine.supports(&manifest, &device_capability);
    if let SupportLevel::Unsupported(_) = support {
        return DoctorCheck::fail(
            "startup_model",
            format!(
                "The selected '{selected_engine}' engine cannot execute the startup model on the configured device."
            ),
            "Choose a supported model, engine, or device from the support matrix.",
        );
    }
    let Some(planned_context) = args.context_size.checked_mul(args.max_concurrent) else {
        return DoctorCheck::fail(
            "startup_model",
            "The configured context plan overflows this platform.",
            "Reduce context size or maximum concurrency.",
        );
    };
    let estimate =
        bloomai_engine::estimate_memory_for_device(&manifest, planned_context.max(1), device_kind);
    if bloomai_engine::plan_memory_preallocation(
        estimate,
        MemoryPreallocationConfig {
            enabled: !args.disable_memory_prealloc,
            memory_utilization: args.memory_utilization,
            reserve_memory_bytes: args.reserve_memory_bytes,
        },
    )
    .is_err()
    {
        return DoctorCheck::fail(
            "startup_model",
            "The startup model does not fit the configured conservative memory plan.",
            "Reduce context size or concurrency, use a smaller model, or adjust the documented memory policy.",
        );
    }
    match support {
        SupportLevel::Native => DoctorCheck::pass(
            "startup_model",
            format!(
                "The startup model metadata, '{selected_engine}' route, device capability, and memory plan are compatible."
            ),
        ),
        SupportLevel::Fallback(_) => DoctorCheck::warn(
            "startup_model",
            format!(
                "The startup model requires a fallback path through the '{selected_engine}' engine."
            ),
            "Confirm the fallback with a pinned real-model smoke test before production use.",
        ),
        SupportLevel::Unsupported(_) => unreachable!(),
    }
}

fn storage_check(args: &Args, models_root: &Path) -> DoctorCheck {
    let writable_catalog = args.enable_model_downloads || args.enable_model_imports;
    if models_root.exists() {
        match std::fs::metadata(models_root) {
            Ok(metadata) if !metadata.is_dir() => {
                return DoctorCheck::fail(
                    "storage_policy",
                    "The configured model catalog exists but is not a directory.",
                    "Point --models-dir at a dedicated directory.",
                );
            }
            Ok(metadata) if writable_catalog && metadata.permissions().readonly() => {
                return DoctorCheck::fail(
                    "storage_policy",
                    "Model acquisitions are enabled but the catalog is marked read-only.",
                    "Grant the Bloom process write access or disable model downloads and imports.",
                );
            }
            Err(_) => {
                return DoctorCheck::fail(
                    "storage_policy",
                    "The configured model catalog metadata is unreadable.",
                    "Correct catalog permissions before starting the server.",
                );
            }
            Ok(_) => {}
        }
    }
    if writable_catalog && args.max_model_storage_bytes == 0 {
        return DoctorCheck::warn(
            "storage_policy",
            "Writable model acquisitions have no application-level storage quota.",
            "Set BLOOM_MAX_MODEL_STORAGE_BYTES and retain an operating-system filesystem limit.",
        );
    }
    if writable_catalog && args.staged_model_retention_seconds == 0 {
        return DoctorCheck::warn(
            "storage_policy",
            "Writable model acquisitions have no automatic stale-staging retention policy.",
            "Set BLOOM_STAGED_MODEL_RETENTION_SECONDS or monitor staged data operationally.",
        );
    }
    DoctorCheck::pass(
        "storage_policy",
        if writable_catalog {
            "Writable model acquisition has an explicit quota and stale-staging retention policy."
        } else {
            "Model downloads and browser imports are disabled; the catalog is read-only from the API."
        },
    )
}

fn license_policy_check(args: &Args) -> DoctorCheck {
    let writable_catalog = args.enable_model_downloads || args.enable_model_imports;
    let policy = match ModelLicensePolicy::new(args.allowed_model_licenses.clone()) {
        Ok(policy) => policy,
        Err(error) => {
            return DoctorCheck::fail(
                "model_license_policy",
                format!("The model license policy is invalid: {error}."),
                "Remove empty, oversized, or excessive license declarations.",
            );
        }
    };
    let status = policy.status();
    if !writable_catalog {
        DoctorCheck::pass(
            "model_license_policy",
            "Model downloads and imports are disabled, so acquisition license admission is inactive.",
        )
    } else if !status.enforced {
        DoctorCheck::warn(
            "model_license_policy",
            "Writable model acquisitions record license declarations but do not restrict them.",
            "Set --allowed-model-licenses or BLOOM_ALLOWED_MODEL_LICENSES to an approved comma-separated allowlist.",
        )
    } else {
        DoctorCheck::pass(
            "model_license_policy",
            format!(
                "Writable model acquisitions require one of {} approved license declarations.",
                status.allowed.len()
            ),
        )
    }
}

fn model_index_check(args: &Args) -> DoctorCheck {
    match validate_model_index_configuration(
        args.model_index_file.clone(),
        args.model_index_url.clone(),
        args.model_index_public_key.clone(),
        args.model_index_public_keys.clone(),
        args.model_index_refresh_seconds,
    ) {
        Ok(Some(status)) => {
            let trust = match status.single_key_id.as_deref() {
                Some(key_id) => format!("trusted key {}", &key_id[..12]),
                None => format!(
                    "{} trusted keys (trust set {})",
                    status.trusted_key_count,
                    &status.trust_id[..12]
                ),
            };
            DoctorCheck::pass(
                "model_index",
                format!(
                    "Signed model discovery is configured from a {} source with {trust}.",
                    status.source_kind
                ),
            )
        }
        Ok(None) => DoctorCheck::pass(
            "model_index",
            "No signed model discovery index is configured; manual verified acquisition remains available.",
        ),
        Err(error) => DoctorCheck::fail(
            "model_index",
            format!("The signed model index configuration is invalid: {error}."),
            "Configure exactly one index file or HTTPS URL together with one to eight unique, non-weak Ed25519 public keys.",
        ),
    }
}

fn model_index_state_check(args: &Args) -> DoctorCheck {
    if args.model_index_file.is_none() && args.model_index_url.is_none() {
        return DoctorCheck::pass(
            "model_index_state",
            "Signed model discovery is disabled, so persistent rollback state is inactive.",
        );
    }
    let Some(directory) = args.model_index_state_dir.as_deref() else {
        return DoctorCheck::fail(
            "model_index_state",
            "Persistent signed-index rollback state has no configured directory.",
            "Set --model-index-state-dir or BLOOM_MODEL_INDEX_STATE_DIR to a private durable directory.",
        );
    };
    match inspect_model_index_watermark_directory(directory) {
        Ok(status) if status.exists => DoctorCheck::pass(
            "model_index_state",
            format!(
                "Persistent signed-index rollback state is valid with {} bounded record(s) across {} source identity set(s).",
                status.record_count, status.source_count
            ),
        ),
        Ok(_) => DoctorCheck::warn(
            "model_index_state",
            "The persistent signed-index rollback directory will be created after the first verified generation.",
            "Ensure its parent is private, durable, writable by Bloom, and included in operational backups.",
        ),
        Err(error) => DoctorCheck::fail(
            "model_index_state",
            format!("Persistent signed-index rollback state is invalid: {error}."),
            "Repair the private state directory or select a clean durable directory; do not discard a valid watermark during incident response.",
        ),
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, FromArgMatches, Parser};

    use super::*;

    fn default_args() -> Args {
        Args::try_parse_from(["bloom_server"]).unwrap()
    }

    #[test]
    fn doctor_cli_accepts_text_and_json_modes() {
        let text = Args::try_parse_from(["bloom_server", "--doctor"]).unwrap();
        let json = Args::try_parse_from(["bloom_server", "--doctor=json"]).unwrap();
        assert_eq!(text.doctor, Some(DoctorFormat::Text));
        assert_eq!(json.doctor, Some(DoctorFormat::Json));
        assert_eq!(text.cors_allow_origin, "same-origin");
    }

    #[test]
    fn browser_origin_policy_is_validated_and_wildcard_is_never_silent() {
        let temp = tempfile::tempdir().unwrap();
        let models = temp.path().join("models");

        let default = default_args();
        assert!(validate_server_arguments(&default).is_ok());
        let default_report = inspect_server(&default, false, &models);
        let default_network = default_report
            .checks
            .iter()
            .find(|check| check.id == "network_security")
            .unwrap();
        assert_eq!(default_network.status, CheckStatus::Pass);

        let mut wildcard = default.clone();
        wildcard.cors_allow_origin = "*".to_string();
        assert!(validate_server_arguments(&wildcard).is_ok());
        let wildcard_report = inspect_server(&wildcard, false, &models);
        let wildcard_network = wildcard_report
            .checks
            .iter()
            .find(|check| check.id == "network_security")
            .unwrap();
        assert_eq!(wildcard_network.status, CheckStatus::Warn);
        assert!(wildcard_network.message.contains("Every browser origin"));

        wildcard.strict_security = true;
        assert!(validate_server_arguments(&wildcard).is_err());
        let strict_report = inspect_server(&wildcard, false, &models);
        let strict_network = strict_report
            .checks
            .iter()
            .find(|check| check.id == "network_security")
            .unwrap();
        assert_eq!(strict_network.status, CheckStatus::Fail);

        let mut invalid = default;
        invalid.cors_allow_origin = "https://ui.example/path".to_string();
        assert!(validate_server_arguments(&invalid).is_err());
        let invalid_report = inspect_server(&invalid, false, &models);
        let invalid_network = invalid_report
            .checks
            .iter()
            .find(|check| check.id == "network_security")
            .unwrap();
        assert_eq!(invalid_network.status, CheckStatus::Fail);
    }

    #[test]
    fn unauthenticated_non_loopback_binding_requires_an_explicit_non_strict_override() {
        let temp = tempfile::tempdir().unwrap();
        let models = temp.path().join("models");
        let mut args = default_args();
        args.host = "0.0.0.0".to_string();

        assert!(validate_server_arguments(&args).is_err());
        let rejected = inspect_server(&args, false, &models);
        assert_eq!(
            rejected
                .checks
                .iter()
                .find(|check| check.id == "network_security")
                .unwrap()
                .status,
            CheckStatus::Fail
        );

        args.allow_unauthenticated_network = true;
        assert!(validate_server_arguments(&args).is_ok());
        let overridden = inspect_server(&args, false, &models);
        assert_eq!(
            overridden
                .checks
                .iter()
                .find(|check| check.id == "network_security")
                .unwrap()
                .status,
            CheckStatus::Warn
        );

        args.strict_security = true;
        assert!(validate_server_arguments(&args).is_err());

        args.allow_unauthenticated_network = false;
        args.api_key = Some("configured-secret".to_string());
        assert!(validate_server_arguments(&args).is_ok());
    }

    #[test]
    fn license_allowlist_accepts_comma_separated_cli_values() {
        let args =
            Args::try_parse_from(["bloom_server", "--allowed-model-licenses", "Apache-2.0,MIT"])
                .unwrap();

        assert_eq!(args.allowed_model_licenses, vec!["Apache-2.0", "MIT"]);
        assert!(validate_server_arguments(&args).is_ok());
    }

    #[test]
    fn model_index_keyring_accepts_comma_separated_cli_values() {
        let args = Args::try_parse_from([
            "bloom_server",
            "--model-index-public-keys",
            "key-one,key-two",
        ])
        .unwrap();

        assert_eq!(args.model_index_public_keys, vec!["key-one", "key-two"]);
    }

    #[test]
    fn writable_acquisition_reports_license_policy_enforcement() {
        let temp = tempfile::tempdir().unwrap();
        let mut args = default_args();
        args.enable_model_downloads = true;
        args.allowed_model_licenses = vec!["Apache-2.0".to_string(), "MIT".to_string()];

        let report = inspect_server(&args, false, &temp.path().join("models"));
        let check = report
            .checks
            .iter()
            .find(|check| check.id == "model_license_policy")
            .unwrap();

        assert_eq!(check.status, CheckStatus::Pass);
        assert!(check.message.contains("2 approved"));
        assert!(!check.message.contains("Apache"));
    }

    #[test]
    fn doctor_validates_model_index_pairing_without_disclosing_its_source() {
        let temp = tempfile::tempdir().unwrap();
        let mut args = default_args();
        args.model_index_file = Some(std::path::PathBuf::from("/private/index.json"));
        args.model_index_state_dir = Some(temp.path().join("index-state"));

        let invalid = inspect_server(&args, false, &temp.path().join("models"));
        let invalid_check = invalid
            .checks
            .iter()
            .find(|check| check.id == "model_index")
            .unwrap();
        assert_eq!(invalid_check.status, CheckStatus::Fail);

        args.model_index_public_key =
            Some("ea4a6c63e29c520abef5507b132ec5f9954776aebebe7b92421eea691446d22c".to_string());
        let valid = inspect_server(&args, false, &temp.path().join("models"));
        let valid_check = valid
            .checks
            .iter()
            .find(|check| check.id == "model_index")
            .unwrap();
        assert_eq!(valid_check.status, CheckStatus::Pass);
        assert!(!valid_check.message.contains("/private"));
        assert!(!valid_check.message.contains("ea4a6c63e29c520abef5507b"));

        let second = ed25519_dalek::SigningKey::from_bytes(&[8_u8; 32]);
        args.model_index_public_keys = vec![
            second
                .verifying_key()
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        ];
        let rotating = inspect_server(&args, false, &temp.path().join("models"));
        let rotating_check = rotating
            .checks
            .iter()
            .find(|check| check.id == "model_index")
            .unwrap();
        assert_eq!(rotating_check.status, CheckStatus::Pass);
        assert!(rotating_check.message.contains("2 trusted keys"));
        assert!(!rotating_check.message.contains("ea4a6c63e29c520a"));
        let state_check = rotating
            .checks
            .iter()
            .find(|check| check.id == "model_index_state")
            .unwrap();
        assert_eq!(state_check.status, CheckStatus::Warn);
        assert!(
            !state_check
                .message
                .contains(temp.path().to_string_lossy().as_ref())
        );
    }

    #[test]
    fn doctor_validates_persistent_model_index_state_without_path_disclosure() {
        let temp = tempfile::tempdir().unwrap();
        let state_directory = temp.path().join("private-state");
        std::fs::create_dir(&state_directory).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&state_directory, std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let mut args = default_args();
        args.model_index_file = Some(std::path::PathBuf::from("/private/index.json"));
        args.model_index_public_key =
            Some("ea4a6c63e29c520abef5507b132ec5f9954776aebebe7b92421eea691446d22c".to_string());
        args.model_index_state_dir = Some(state_directory.clone());

        let ready = inspect_server(&args, false, &temp.path().join("models"));
        let ready_check = ready
            .checks
            .iter()
            .find(|check| check.id == "model_index_state")
            .unwrap();
        assert_eq!(ready_check.status, CheckStatus::Pass);
        assert!(ready_check.message.contains("0 bounded record"));

        std::fs::write(state_directory.join("unexpected.txt"), b"invalid").unwrap();
        let invalid = inspect_server(&args, false, &temp.path().join("models"));
        let invalid_check = invalid
            .checks
            .iter()
            .find(|check| check.id == "model_index_state")
            .unwrap();
        assert_eq!(invalid_check.status, CheckStatus::Fail);
        assert!(
            !invalid_check
                .message
                .contains(temp.path().to_string_lossy().as_ref())
        );
    }

    #[test]
    fn server_argument_validation_rejects_zeroes_and_ignored_features() {
        let mut args = default_args();
        args.max_concurrent = 0;
        args.shutdown_timeout_seconds = 0;
        args.memory_utilization = 1.0;
        args.enable_chunked_prefill = true;
        args.prefill_chunk_size = 0;
        args.enable_cachemesh_l3 = true;

        let error = validate_server_arguments(&args).unwrap_err().to_string();
        assert!(error.contains("Maximum concurrency"));
        assert!(error.contains("Shutdown timeout"));
        assert!(error.contains("Memory utilization"));
        assert!(error.contains("Chunked prefill requires"));
        assert!(error.contains("Chunked prefill size"));
        assert!(error.contains("CacheMesh L3 requires"));
    }

    #[test]
    fn chunked_prefill_size_config_respects_cli_precedence_and_validation() {
        let config = bloomai_engine::ServerConfig {
            enable_ifb: Some(true),
            enable_chunked_prefill: Some(true),
            prefill_chunk_size: Some(0),
            ..Default::default()
        };

        let matches = Args::command()
            .try_get_matches_from(["bloom_server"])
            .unwrap();
        let mut configured = Args::from_arg_matches(&matches).unwrap();
        super::super::cli::apply_config(&mut configured, &matches, &config);
        assert_eq!(configured.prefill_chunk_size, 0);
        assert!(validate_server_arguments(&configured).is_err());

        let matches = Args::command()
            .try_get_matches_from(["bloom_server", "--prefill-chunk-size", "256"])
            .unwrap();
        let mut explicit = Args::from_arg_matches(&matches).unwrap();
        super::super::cli::apply_config(&mut explicit, &matches, &config);
        assert_eq!(explicit.prefill_chunk_size, 256);
        assert!(validate_server_arguments(&explicit).is_ok());
    }

    #[test]
    fn maximum_concurrency_is_bounded_before_semaphore_construction() {
        let mut maximum = default_args();
        maximum.context_size = 1;
        maximum.max_concurrent = Semaphore::MAX_PERMITS;
        assert!(validate_server_arguments(&maximum).is_ok());

        let mut oversized = maximum;
        oversized.max_concurrent = Semaphore::MAX_PERMITS + 1;
        let error = validate_server_arguments(&oversized)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Maximum concurrency must not exceed"));
        assert!(error.contains(&Semaphore::MAX_PERMITS.to_string()));
    }

    #[test]
    fn maximum_concurrency_config_respects_cli_precedence_and_validation() {
        let config = bloomai_engine::ServerConfig {
            max_concurrent: Some(Semaphore::MAX_PERMITS + 1),
            ..Default::default()
        };

        let matches = Args::command()
            .try_get_matches_from(["bloom_server"])
            .unwrap();
        let mut configured = Args::from_arg_matches(&matches).unwrap();
        super::super::cli::apply_config(&mut configured, &matches, &config);
        assert_eq!(configured.max_concurrent, Semaphore::MAX_PERMITS + 1);
        assert!(validate_server_arguments(&configured).is_err());

        let matches = Args::command()
            .try_get_matches_from(["bloom_server", "--max-concurrent", "2"])
            .unwrap();
        let mut explicit = Args::from_arg_matches(&matches).unwrap();
        super::super::cli::apply_config(&mut explicit, &matches, &config);
        assert_eq!(explicit.max_concurrent, 2);
        assert!(validate_server_arguments(&explicit).is_ok());
    }

    #[test]
    fn shutdown_timeout_is_bounded_and_available_from_the_cli() {
        let defaults = default_args();
        assert_eq!(defaults.shutdown_timeout_seconds, 30);

        let maximum = Args::try_parse_from([
            "bloom_server",
            "--shutdown-timeout-seconds",
            &MAX_SHUTDOWN_TIMEOUT_SECONDS.to_string(),
        ])
        .unwrap();
        assert!(validate_server_arguments(&maximum).is_ok());

        let mut oversized = defaults;
        oversized.shutdown_timeout_seconds = MAX_SHUTDOWN_TIMEOUT_SECONDS + 1;
        assert!(validate_server_arguments(&oversized).is_err());
    }

    #[test]
    fn shutdown_timeout_config_respects_cli_precedence() {
        let config = bloomai_engine::ServerConfig {
            shutdown_timeout_seconds: Some(45),
            ..Default::default()
        };

        let matches = Args::command()
            .try_get_matches_from(["bloom_server"])
            .unwrap();
        let mut configured = Args::from_arg_matches(&matches).unwrap();
        super::super::cli::apply_config(&mut configured, &matches, &config);
        assert_eq!(configured.shutdown_timeout_seconds, 45);

        let matches = Args::command()
            .try_get_matches_from(["bloom_server", "--shutdown-timeout-seconds", "12"])
            .unwrap();
        let mut explicit = Args::from_arg_matches(&matches).unwrap();
        super::super::cli::apply_config(&mut explicit, &matches, &config);
        assert_eq!(explicit.shutdown_timeout_seconds, 12);
    }

    #[cfg(not(feature = "serve-ui"))]
    #[test]
    fn browser_launch_requires_an_embedded_ui() {
        let mut args = default_args();
        args.open_browser = true;

        let error = validate_server_arguments(&args).unwrap_err().to_string();
        assert!(error.contains("requires a bloom_server binary built with the embedded UI"));
    }

    #[test]
    fn empty_cpu_install_warns_without_failing() {
        let temp = tempfile::tempdir().unwrap();
        let models_root = temp.path().join("models");
        let report = inspect_server(&default_args(), false, &models_root);

        assert_eq!(report.status, "warn");
        assert_eq!(report.summary.failures, 0);
        assert!(report.summary.warnings >= 2);
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.id == "device_backend" && check.status == CheckStatus::Pass)
        );
    }

    #[test]
    fn invalid_catalog_and_arguments_are_failures_without_path_disclosure() {
        let temp = tempfile::tempdir().unwrap();
        let catalog_file = temp.path().join("not-a-directory");
        std::fs::write(&catalog_file, b"catalog").unwrap();
        let mut args = default_args();
        args.max_concurrent = 0;

        let report = inspect_server(&args, true, &catalog_file);
        let json = report.render(DoctorFormat::Json).unwrap();

        assert!(report.has_failures());
        assert!(report.summary.failures >= 2);
        assert!(!json.contains(temp.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn doctor_outputs_are_versioned_and_do_not_serialize_api_keys() {
        let temp = tempfile::tempdir().unwrap();
        let mut args = default_args();
        args.api_key = Some("doctor-secret-value".to_string());
        let report = inspect_server(&args, false, &temp.path().join("models"));

        let json = report.render(DoctorFormat::Json).unwrap();
        let text = report.render(DoctorFormat::Text).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["object"], "bloom.server_doctor");
        assert!(!json.contains("doctor-secret-value"));
        assert!(text.starts_with("Bloom server doctor"));
        assert!(text.contains("Summary:"));
    }
}
