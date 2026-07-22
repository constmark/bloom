#![allow(unused_imports, dead_code)]
use super::*;

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

    /// Path to the model directory or a GGUF file.
    #[arg(short, long)]
    pub model: Option<PathBuf>,

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

    /// Maximum concurrent requests (backpressure limit).
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
    #[arg(long, env = "BLOOM_DISABLE_MEMORY_PREALLOC", default_value_t = false)]
    pub disable_memory_prealloc: bool,

    /// Maximum total tokens per scheduling step (in-flight batching budget).
    #[arg(long, default_value_t = 4096)]
    pub max_num_tokens: usize,

    /// Request timeout in seconds (0 = no timeout).
    #[arg(long, default_value_t = 300)]
    pub timeout: u64,

    /// Optional API key for /v1 routes. Can also be set with BLOOM_API_KEY.
    #[arg(long, env = "BLOOM_API_KEY")]
    pub api_key: Option<String>,

    /// CORS allowed origin. Use "*" for local development.
    #[arg(long, env = "BLOOM_CORS_ALLOW_ORIGIN", default_value = "*")]
    pub cors_allow_origin: String,

    /// Maximum JSON request body size in bytes.
    #[arg(long, env = "BLOOM_MAX_BODY_BYTES", default_value_t = 1024 * 1024)]
    pub max_body_bytes: usize,

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

    /// Maximum tokens per prefill chunk.
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
    #[arg(long, env = "BLOOM_STRICT_MEMORY_BUDGET", default_value_t = false)]
    pub strict_memory_budget: bool,

    /// Enforce strict security check, failing startup if external scripts/plugins are not allowlisted.
    #[arg(long, env = "BLOOM_STRICT_SECURITY", default_value_t = false)]
    pub strict_security: bool,
}

macro_rules! apply_config_value {
    ($args:expr, $matches:expr, $config:expr, $field:ident) => {
        if should_use_config($matches, stringify!($field)) {
            if let Some(value) = &$config.$field {
                $args.$field = value.clone();
            }
        }
    };
}

macro_rules! apply_config_option {
    ($args:expr, $matches:expr, $config:expr, $field:ident) => {
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

pub(crate) fn apply_config(args: &mut Args, matches: &ArgMatches, config: &bloomai_engine::ServerConfig) {
    apply_config_option!(args, matches, config, model);
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
