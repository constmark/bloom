#![allow(unused_imports, dead_code)]
use super::*;

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DoctorFormat {
    Text,
    Json,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "bloom_server",
    version,
    about = "Bloom OpenAI-compatible HTTP Server"
)]
pub(crate) struct Args {
    /// Path to Bloom runtime config. Defaults to ~/.bloom/config.json.
    #[arg(long, env = "BLOOM_CONFIG")]
    pub config: Option<PathBuf>,

    /// Write an example config file and exit.
    #[arg(long, default_value_t = false)]
    pub init_config: bool,

    /// Inspect the effective server configuration and exit without loading a model or binding a port.
    #[arg(
        long,
        value_enum,
        num_args = 0..=1,
        default_missing_value = "text",
        conflicts_with = "init_config"
    )]
    pub doctor: Option<DoctorFormat>,

    /// Open the embedded Bloom UI in the system browser after the listener is ready.
    #[arg(
        long,
        env = "BLOOM_OPEN_BROWSER",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    pub open_browser: bool,

    /// Path to the model directory or a GGUF file.
    #[arg(short, long)]
    pub model: Option<PathBuf>,

    /// Root directory scanned by the model-management API.
    #[arg(long, env = "BLOOM_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// Host address to bind the server to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on.
    #[arg(short, long, default_value_t = 3000)]
    pub port: u16,

    /// Selection of backend engine: candle, openvino, funasr, qwen3_vl.
    #[arg(long, default_value = "candle")]
    pub backend: String,

    /// Device to run on: cpu, gpu.
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Precision to load the model with: f32, f16, bf16.
    #[arg(long)]
    pub dtype: Option<String>,

    /// Maximum concurrent requests; bounded by the platform runtime semaphore capacity.
    #[arg(long, default_value_t = 4)]
    pub max_concurrent: usize,

    /// Context size for memory checks.
    #[arg(long, default_value_t = 2048)]
    pub context_size: usize,

    /// Fraction of currently available memory Bloom may plan to use at startup.
    #[arg(long, env = "BLOOM_MEMORY_UTILIZATION", default_value_t = 0.75)]
    pub memory_utilization: f64,

    /// Explicit startup memory reservation size in bytes.
    #[arg(long, env = "BLOOM_RESERVE_MEMORY_BYTES")]
    pub reserve_memory_bytes: Option<usize>,

    /// Disable startup memory preallocation and only log memory estimates.
    #[arg(
        long,
        env = "BLOOM_DISABLE_MEMORY_PREALLOC",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    pub disable_memory_prealloc: bool,

    /// Maximum total tokens per scheduling step (in-flight batching budget).
    #[arg(long, default_value_t = 4096)]
    pub max_num_tokens: usize,

    /// Request timeout in seconds (0 = no timeout).
    #[arg(long, default_value_t = 300)]
    pub timeout: u64,

    /// Maximum graceful shutdown drain time before forced process termination.
    #[arg(long, env = "BLOOM_SHUTDOWN_TIMEOUT_SECONDS", default_value_t = 30)]
    pub shutdown_timeout_seconds: u64,

    /// Optional API key for /v1 routes. Can also be set with BLOOM_API_KEY.
    #[arg(long, env = "BLOOM_API_KEY")]
    pub api_key: Option<String>,

    /// Explicitly allow a non-loopback listener without authentication (unsafe; development only).
    #[arg(
        long,
        env = "BLOOM_ALLOW_UNAUTHENTICATED_NETWORK",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    pub allow_unauthenticated_network: bool,

    /// Browser origin policy: same-origin, one exact HTTP(S) origin, or explicit "*".
    #[arg(long, env = "BLOOM_CORS_ALLOW_ORIGIN", default_value = "same-origin")]
    pub cors_allow_origin: String,

    /// Maximum JSON request body size in bytes.
    #[arg(long, env = "BLOOM_MAX_BODY_BYTES", default_value_t = 1024 * 1024)]
    pub max_body_bytes: usize,

    /// Maximum multipart image upload size in bytes.
    #[arg(long, env = "BLOOM_MAX_UPLOAD_BYTES", default_value_t = 12 * MIB as usize)]
    pub max_upload_bytes: usize,

    /// Allow authenticated downloads from trusted model-hosting domains.
    #[arg(
        long,
        env = "BLOOM_ENABLE_MODEL_DOWNLOADS",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    pub enable_model_downloads: bool,

    /// Maximum size of one downloaded model file in bytes.
    #[arg(long, env = "BLOOM_MAX_MODEL_DOWNLOAD_BYTES", default_value_t = 20 * GIB)]
    pub max_model_download_bytes: u64,

    /// Allow authenticated, chunked local-file imports into the model catalog.
    #[arg(
        long,
        env = "BLOOM_ENABLE_MODEL_IMPORTS",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    pub enable_model_imports: bool,

    /// Maximum size of one imported model file in bytes.
    #[arg(long, env = "BLOOM_MAX_MODEL_IMPORT_BYTES", default_value_t = 20 * GIB)]
    pub max_model_import_bytes: u64,

    /// Maximum request body for one model import chunk in bytes.
    #[arg(long, env = "BLOOM_MAX_MODEL_IMPORT_CHUNK_BYTES", default_value_t = 8 * MIB as usize)]
    pub max_model_import_chunk_bytes: usize,

    /// Allowed model license declarations for downloads and imports (empty = record only).
    #[arg(long, env = "BLOOM_ALLOWED_MODEL_LICENSES", value_delimiter = ',')]
    pub allowed_model_licenses: Vec<String>,

    /// Path to a signed model discovery index.
    #[arg(long, env = "BLOOM_MODEL_INDEX_FILE")]
    pub model_index_file: Option<PathBuf>,

    /// HTTPS URL of a signed model discovery index.
    #[arg(long, env = "BLOOM_MODEL_INDEX_URL")]
    pub model_index_url: Option<String>,

    /// Trusted Ed25519 index public key as 64 hex characters or unpadded base64url.
    #[arg(long, env = "BLOOM_MODEL_INDEX_PUBLIC_KEY")]
    pub model_index_public_key: Option<String>,

    /// Additional trusted Ed25519 index public keys for bounded key rotation.
    #[arg(long, env = "BLOOM_MODEL_INDEX_PUBLIC_KEYS", value_delimiter = ',')]
    pub model_index_public_keys: Vec<String>,

    /// Successful model index refresh interval in seconds.
    #[arg(long, env = "BLOOM_MODEL_INDEX_REFRESH_SECONDS", default_value_t = 300)]
    pub model_index_refresh_seconds: u64,

    /// Directory for persistent signed-index rollback watermarks.
    #[arg(long, env = "BLOOM_MODEL_INDEX_STATE_DIR")]
    pub model_index_state_dir: Option<PathBuf>,

    /// Maximum committed model-catalog bytes across installed and staged data (0 = unlimited).
    #[arg(long, env = "BLOOM_MAX_MODEL_STORAGE_BYTES", default_value_t = 0)]
    pub max_model_storage_bytes: u64,

    /// Remove inactive staged acquisitions older than this (0 = disabled).
    #[arg(
        long,
        env = "BLOOM_STAGED_MODEL_RETENTION_SECONDS",
        default_value_t = 0
    )]
    pub staged_model_retention_seconds: u64,

    /// Speculative decoding mode: none, ngram, draft, mtp.
    #[arg(long, default_value = "none")]
    pub speculative: String,

    /// Path to draft model (for speculative=draft).
    #[arg(long)]
    pub draft_model: Option<PathBuf>,

    /// Number of speculative tokens per step.
    #[arg(long, default_value_t = 5)]
    pub num_speculative_tokens: usize,

    /// N-gram order used when --speculative=ngram.
    #[arg(long, default_value_t = 4)]
    pub speculative_ngram_order: usize,

    /// Enable continuous in-flight batching scheduler.
    #[arg(long, default_value_t = false)]
    pub enable_ifb: bool,

    /// Enable chunked prefill when continuous in-flight batching is enabled.
    #[arg(long, default_value_t = false)]
    pub enable_chunked_prefill: bool,

    /// Maximum tokens per prefill chunk; must be at least 1 when enabled.
    #[arg(long, default_value_t = 512)]
    pub prefill_chunk_size: usize,

    /// Enable CacheMesh L1/L2/L3 KV cache offload for IFB paged attention.
    #[arg(long, default_value_t = false)]
    pub enable_cachemesh: bool,

    /// CacheMesh L2 host-memory capacity in bytes.
    #[arg(long, default_value_t = 2 * GIB as usize)]
    pub cachemesh_l2_capacity_bytes: usize,

    /// Enable CacheMesh L3 remote cache backend.
    #[arg(long, default_value_t = false)]
    pub enable_cachemesh_l3: bool,

    /// Directory-backed CacheMesh L3 path. Use a shared mount for multi-host cache.
    #[arg(long, env = "BLOOM_CACHEMESH_L3_PATH")]
    pub cachemesh_l3_path: Option<PathBuf>,

    /// Write every L2 offload through to L3 so other hosts can restore it immediately.
    #[arg(long, default_value_t = false)]
    pub cachemesh_write_through_l3: bool,

    /// Long-context KV policy for IFB: full, sliding-window, context-shift, compact-inactive.
    #[arg(long, default_value = "full")]
    pub long_context_policy: String,

    /// Visible KV window for --long-context-policy sliding-window.
    #[arg(long, default_value_t = 4096)]
    pub sliding_window_tokens: usize,

    /// Max context for --long-context-policy context-shift.
    #[arg(long)]
    pub context_shift_max_tokens: Option<usize>,

    /// Tokens shifted out when --long-context-policy context-shift is active.
    #[arg(long, default_value_t = 1024)]
    pub context_shift_tokens: usize,

    /// Target free KV blocks for --long-context-policy compact-inactive.
    #[arg(long, default_value_t = 128)]
    pub compact_free_blocks: usize,

    /// Enforce strict memory budget check, failing startup if estimate exceeds available memory.
    #[arg(
        long,
        env = "BLOOM_STRICT_MEMORY_BUDGET",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    pub strict_memory_budget: bool,

    /// Enforce strict security check, failing startup if external scripts/plugins are not allowlisted.
    #[arg(
        long,
        env = "BLOOM_STRICT_SECURITY",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    pub strict_security: bool,
}

macro_rules! apply_config_value {
    ($args:expr_2021, $matches:expr_2021, $config:expr_2021, $field:ident) => {
        if should_use_config($matches, stringify!($field)) {
            if let Some(value) = &$config.$field {
                $args.$field = value.clone();
            }
        }
    };
}

macro_rules! apply_config_option {
    ($args:expr_2021, $matches:expr_2021, $config:expr_2021, $field:ident) => {
        if should_use_config($matches, stringify!($field)) {
            if let Some(value) = &$config.$field {
                $args.$field = Some(value.clone());
            }
        }
    };
}

pub(crate) fn should_use_config(matches: &ArgMatches, id: &str) -> bool {
    matches
        .value_source(id)
        .map(|source| source == ValueSource::DefaultValue)
        .unwrap_or(true)
}

pub(crate) fn parse_args() -> Result<(Args, ArgMatches)> {
    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches)?;
    Ok((args, matches))
}

pub(crate) fn apply_config(
    args: &mut Args,
    matches: &ArgMatches,
    config: &bloomai_engine::ServerConfig,
) {
    apply_config_option!(args, matches, config, model);
    apply_config_option!(args, matches, config, models_dir);
    apply_config_value!(args, matches, config, host);
    apply_config_value!(args, matches, config, port);
    apply_config_value!(args, matches, config, backend);
    apply_config_value!(args, matches, config, device);
    apply_config_option!(args, matches, config, dtype);
    apply_config_value!(args, matches, config, max_concurrent);
    apply_config_value!(args, matches, config, context_size);
    apply_config_value!(args, matches, config, memory_utilization);
    apply_config_option!(args, matches, config, reserve_memory_bytes);
    apply_config_value!(args, matches, config, disable_memory_prealloc);
    apply_config_value!(args, matches, config, max_num_tokens);
    apply_config_value!(args, matches, config, timeout);
    apply_config_value!(args, matches, config, shutdown_timeout_seconds);
    apply_config_value!(args, matches, config, allow_unauthenticated_network);
    apply_config_value!(args, matches, config, max_upload_bytes);
    apply_config_value!(args, matches, config, enable_model_downloads);
    apply_config_value!(args, matches, config, max_model_download_bytes);
    apply_config_value!(args, matches, config, enable_model_imports);
    apply_config_value!(args, matches, config, max_model_import_bytes);
    apply_config_value!(args, matches, config, max_model_import_chunk_bytes);
    apply_config_value!(args, matches, config, allowed_model_licenses);
    apply_config_option!(args, matches, config, model_index_file);
    apply_config_option!(args, matches, config, model_index_url);
    apply_config_option!(args, matches, config, model_index_public_key);
    apply_config_value!(args, matches, config, model_index_public_keys);
    apply_config_value!(args, matches, config, model_index_refresh_seconds);
    apply_config_option!(args, matches, config, model_index_state_dir);
    apply_config_value!(args, matches, config, max_model_storage_bytes);
    apply_config_value!(args, matches, config, staged_model_retention_seconds);
    apply_config_value!(args, matches, config, speculative);
    apply_config_option!(args, matches, config, draft_model);
    apply_config_value!(args, matches, config, num_speculative_tokens);
    apply_config_value!(args, matches, config, speculative_ngram_order);
    apply_config_value!(args, matches, config, enable_ifb);
    apply_config_value!(args, matches, config, enable_chunked_prefill);
    apply_config_value!(args, matches, config, prefill_chunk_size);
    apply_config_value!(args, matches, config, enable_cachemesh);
    apply_config_value!(args, matches, config, cachemesh_l2_capacity_bytes);
    apply_config_value!(args, matches, config, enable_cachemesh_l3);
    apply_config_option!(args, matches, config, cachemesh_l3_path);
    apply_config_value!(args, matches, config, cachemesh_write_through_l3);
    apply_config_value!(args, matches, config, long_context_policy);
    apply_config_value!(args, matches, config, sliding_window_tokens);
    apply_config_option!(args, matches, config, context_shift_max_tokens);
    apply_config_value!(args, matches, config, context_shift_tokens);
    apply_config_value!(args, matches, config, compact_free_blocks);
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

pub(crate) fn build_long_context_policy(args: &Args) -> Result<bloomai_engine::LongContextPolicy> {
    match args.long_context_policy.trim().to_lowercase().as_str() {
        "full" => Ok(bloomai_engine::LongContextPolicy::Full),
        "sliding-window" | "sliding_window" | "sliding" => {
            Ok(bloomai_engine::LongContextPolicy::SlidingWindow {
                window_tokens: args.sliding_window_tokens.max(1),
            })
        }
        "context-shift" | "context_shift" | "shift" => {
            Ok(bloomai_engine::LongContextPolicy::ContextShift {
                max_context_tokens: args.context_shift_max_tokens.unwrap_or(args.context_size),
                shift_tokens: args.context_shift_tokens,
            })
        }
        "compact-inactive" | "compact_inactive" | "compact" => {
            Ok(bloomai_engine::LongContextPolicy::CompactInactive {
                target_free_blocks: args.compact_free_blocks,
            })
        }
        other => Err(anyhow!(
            "unsupported --long-context-policy '{}'; expected full, sliding-window, context-shift, or compact-inactive",
            other
        )),
    }
}
