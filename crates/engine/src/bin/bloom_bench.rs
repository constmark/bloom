#![allow(clippy::manual_map)]
#![cfg_attr(not(test), warn(clippy::unwrap_used))]
//! bloom_bench — Dedicated benchmarking utility for Bloom engine.
//!
//! Benchmark local models and report TTFT, TBT, throughput (tokens/s), latency,
//! prompt processing speed, and memory usage. Supports multi-prompt benchmarking
//! with cross-prompt statistics.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use anyhow::{Result, anyhow};
use bloomai_core::{BenchmarkResult, DType, DeviceKind, GenerationParams};
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

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
use bloomai_engine::executor::qwen3_vl::Qwen3VLEngine;
use bloomai_engine::executor::vulkan::VulkanEngine;
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::wan::WanEngine;
use bloomai_engine::{
    EngineRegistry, InferencePipeline, MemoryEstimate, ModelInput,
    model_manifest_supports_embeddings,
};

/// Extended benchmark result with additional timing and hardware metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedBenchmarkResult {
    #[serde(flatten)]
    pub base: BenchmarkResult,
    /// Time between tokens (per-token latency, ms).
    pub tbt_ms: Option<f64>,
    /// Prompt processing speed (tokens/second for prefill).
    pub prompt_processing_speed: Option<f64>,
    /// Device/GPU info string.
    pub gpu_info: String,
    /// Detailed quantization label (e.g. "Q4_K_M", "AWQ-4bit").
    pub quantization_detail: String,
    /// Hardware section with device details.
    pub hardware: HardwareInfo,
    /// Timing breakdown section.
    pub timing_breakdown: TimingBreakdown,
    /// Number of prompts tested (multi-prompt mode).
    pub prompts_tested: usize,
    /// Cross-prompt average TTFT (ms).
    pub avg_ttft_ms: Option<f64>,
    /// Cross-prompt average TBT (ms).
    pub avg_tbt_ms: Option<f64>,
    /// KV cache and prompt-cache metrics observed during the run.
    pub cache_metrics: BenchmarkCacheMetrics,
    /// Structured pre-load/post-load memory estimate used for this benchmark.
    pub memory_breakdown: Option<MemoryEstimate>,
    /// Speculative decoding mode used: none, ngram, draft, mtp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_mode: Option<String>,
    /// Number of draft tokens proposed per step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_draft_tokens: Option<usize>,
    /// Number of accepted tokens by speculative decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_accepted_tokens: Option<usize>,
    /// Speculative decoding acceptance rate (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_acceptance_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub device: String,
    pub backend: String,
    pub os: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub model_load_secs: f64,
    pub avg_ttft_ms: Option<f64>,
    pub avg_tbt_ms: Option<f64>,
    pub avg_latency_ms: f64,
    pub throughput_tokens_per_sec: f64,
    pub prompt_processing_tokens_per_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkCacheMetrics {
    pub enabled: bool,
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub active_blocks: usize,
    pub cached_blocks: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub reuses: usize,
    pub utilization: f64,
}

#[derive(Parser, Debug)]
#[command(name = "bloom_bench", version, about = "Bloom benchmarking utility")]
struct Args {
    /// Path to Bloom runtime config. Defaults to ~/.bloom/config.json.
    #[arg(long, env = "BLOOM_CONFIG")]
    config: Option<PathBuf>,

    /// Write an example config file and exit.
    #[arg(long, default_value_t = false)]
    init_config: bool,

    /// Path to the model directory or a GGUF file.
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Input prompt for benchmarking.
    #[arg(
        short,
        long,
        default_value = "Explain quantum computing in simple terms"
    )]
    prompt: String,

    /// Selection of backend engine: candle, openvino, funasr, qwen3_vl.
    #[arg(long, default_value = "candle")]
    backend: String,

    /// Device to run on: cpu, gpu.
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Precision to load the model with: f32, f16, bf16.
    #[arg(long)]
    dtype: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Number of repetition runs.
    #[arg(long, default_value_t = 3)]
    repetitions: usize,

    /// Number of warmup runs.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Context size for memory estimation.
    #[arg(long, default_value_t = 2048)]
    context_size: usize,

    /// Path to a file containing multiple prompts (one per line).
    #[arg(long)]
    prompts_file: Option<PathBuf>,

    /// Speculative decoding mode: none, ngram, draft, mtp.
    #[arg(long, default_value = "none")]
    speculative: String,

    /// Path to draft model (for speculative=draft).
    #[arg(long)]
    draft_model: Option<PathBuf>,

    /// Number of speculative tokens to propose per target verification step.
    #[arg(long, default_value_t = 4)]
    num_speculative_tokens: usize,

    /// N-gram order used when --speculative=ngram.
    #[arg(long, default_value_t = 4)]
    speculative_ngram_order: usize,
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

fn should_use_config(matches: &ArgMatches, id: &str) -> bool {
    matches
        .value_source(id)
        .map(|source| source == ValueSource::DefaultValue)
        .unwrap_or(true)
}

fn parse_args() -> Result<(Args, ArgMatches)> {
    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches)?;
    Ok((args, matches))
}

fn apply_config(args: &mut Args, matches: &ArgMatches, config: &bloomai_engine::BenchConfig) {
    apply_config_option!(args, matches, config, model);
    apply_config_value!(args, matches, config, prompt);
    apply_config_value!(args, matches, config, backend);
    apply_config_value!(args, matches, config, device);
    apply_config_option!(args, matches, config, dtype);
    apply_config_value!(args, matches, config, max_tokens);
    apply_config_value!(args, matches, config, repetitions);
    apply_config_value!(args, matches, config, warmup);
    apply_config_value!(args, matches, config, context_size);
    apply_config_option!(args, matches, config, prompts_file);
    apply_config_value!(args, matches, config, speculative);
    apply_config_option!(args, matches, config, draft_model);
    apply_config_value!(args, matches, config, num_speculative_tokens);
    apply_config_value!(args, matches, config, speculative_ngram_order);
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::new("warn"))
        .init();

    let (mut args, matches) = parse_args()?;
    let config_path = bloomai_engine::resolve_config_path(args.config.as_deref())?;
    if args.init_config {
        bloomai_engine::write_default_config(&config_path)?;
        println!("Wrote Bloom config to {}", config_path.display());
        return Ok(());
    }
    let config = bloomai_engine::load_config(&config_path)?;
    apply_config(&mut args, &matches, &config.bench);

    let model_path = args.model.as_ref().ok_or_else(|| {
        anyhow!(
            "model path is required; pass --model or set bench.model in {}",
            config_path.display()
        )
    })?;

    if !model_path.exists() {
        return Err(anyhow!(
            "model path does not exist: {}",
            model_path.display()
        ));
    }

    if let Some(ref dt) = args.dtype {
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_DTYPE", dt) };
    }

    // FIXME: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("BLOOM_SPECULATIVE", &args.speculative) };
    if let Some(ref path) = args.draft_model {
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_DRAFT_MODEL", path) };
    }
    // FIXME: Audit that the environment access only happens in single-threaded code.
    unsafe {
        std::env::set_var(
            "BLOOM_NUM_SPECULATIVE_TOKENS",
            args.num_speculative_tokens.to_string(),
        )
    };
    // FIXME: Audit that the environment access only happens in single-threaded code.
    unsafe {
        std::env::set_var(
            "BLOOM_SPECULATIVE_NGRAM_ORDER",
            args.speculative_ngram_order.to_string(),
        )
    };

    // --- Load manifest first to assist in engine auto-selection if needed ---
    let manifest = bloomai_engine::load_manifest(model_path)?;
    if model_manifest_supports_embeddings(&manifest) {
        return Err(anyhow!(
            "bloom_bench currently measures token generation and cannot benchmark an embedding encoder; use scripts/test_trained_embedding_runtime.sh for embedding latency and quality evidence"
        ));
    }

    let mut registry = EngineRegistry::default();
    registry.register("candle", Box::new(CandleEngine));
    registry.register("openvino", Box::new(OpenVINOEngine));
    registry.register("funasr", Box::new(FunASREngine));
    registry.register("qwen3_vl", Box::new(Qwen3VLEngine));
    registry.register("intel-npu", Box::new(IntelNpuEngine));
    registry.register("npu-tts", Box::new(NpuTtsEngine));
    registry.register("onnxruntime", Box::new(OnnxRuntimeEngine));
    registry.register("coreml", Box::new(CoreMlEngine));
    registry.register("mlx", Box::new(MlxEngine));
    registry.register("vulkan", Box::new(VulkanEngine));
    registry.register("longcat", Box::new(LongCatImageEditEngine));
    registry.register("llamacpp", Box::new(LlamaCppEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("wan", Box::new(WanEngine));

    let backend_name = if args.backend == "candle" {
        // Auto-select engine based on manifest family and details
        let is_qwen_vl = manifest.family == bloomai_core::ModelFamily::Qwen
            && (manifest.id.to_lowercase().contains("vl")
                || manifest
                    .parameters
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.contains("vl"))
                    .unwrap_or(false));
        if is_qwen_vl {
            "qwen3_vl"
        } else if matches!(&manifest.family, bloomai_core::ModelFamily::Custom(c) if c == "longcat-image-edit")
        {
            "longcat"
        } else if manifest.family == bloomai_core::ModelFamily::FunAsr {
            "funasr"
        } else if matches!(&manifest.family, bloomai_core::ModelFamily::Custom(c) if c == "wan") {
            "wan"
        } else {
            "candle"
        }
    } else {
        args.backend.as_str()
    };

    let engine = registry.get(backend_name).map_err(|e| {
        anyhow!(
            "{}. Supported engines are: candle, openvino, funasr, qwen3_vl, longcat, intel-npu, npu-tts, onnxruntime, coreml, mlx, llamacpp, wan.",
            e
        )
    })?;
    let device_kind = match args.device.to_lowercase().as_str() {
        "cpu" => DeviceKind::Cpu,
        "gpu" | "cuda" | "metal" => DeviceKind::Gpu,
        "npu" | "intel-npu" => DeviceKind::Npu,
        other => return Err(anyhow!("unsupported device: {}", other)),
    };

    let load_start = Instant::now();
    let pipeline = InferencePipeline::load_standalone_with_context(
        engine,
        device_kind,
        model_path,
        args.context_size,
    )?;
    let load_time = load_start.elapsed().as_secs_f64();

    if pipeline.context_size() != args.context_size || pipeline.device() != device_kind {
        tracing::warn!(
            "Model loaded with OOM degradation: requested Context Size {} on {:?}, got {} on {:?}",
            args.context_size,
            device_kind,
            pipeline.context_size(),
            pipeline.device()
        );
    }

    let params = GenerationParams {
        max_tokens: args.max_tokens,
        temperature: 0.7,
        top_p: 0.9,
        seed: Some(42),
        response_format: None,
    };

    // --- Multi-prompt support ---
    let prompts: Vec<String> = if let Some(ref pf) = args.prompts_file {
        let content = std::fs::read_to_string(pf)
            .map_err(|e| anyhow!("failed to read prompts file: {}", e))?;
        content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(String::from)
            .collect()
    } else {
        vec![args.prompt.clone()]
    };

    let device_label = args.device.clone();
    let os_label = if cfg!(target_os = "macos") {
        "macOS".to_string()
    } else if cfg!(target_os = "linux") {
        "Linux".to_string()
    } else if cfg!(target_os = "windows") {
        "Windows".to_string()
    } else {
        "unknown".to_string()
    };

    let quant_detail = pipeline
        .metadata()
        .manifest
        .quantization
        .as_ref()
        .map(|q| format!("{:?}", q.scheme))
        .unwrap_or_else(|| "none".to_string());

    // --- Benchmark each prompt ---
    let mut all_ttft: Vec<f64> = Vec::new();
    let mut all_tbt: Vec<f64> = Vec::new();
    let mut all_speed: Vec<f64> = Vec::new();
    let mut all_latency: Vec<f64> = Vec::new();
    let mut all_prompt_speed: Vec<f64> = Vec::new();
    let mut all_generated_tokens: Vec<usize> = Vec::new();
    let mut all_durations: Vec<f64> = Vec::new();
    let mut all_speculative_draft: Vec<usize> = Vec::new();
    let mut all_speculative_accepted: Vec<usize> = Vec::new();
    let mut all_speculative_rates: Vec<f64> = Vec::new();

    // The speculative mode requested by the user. We only surface speculative
    // fields when a non-`none` mode is requested, mirroring the `Metrics`
    // chunk emitted by the engine (which is only populated when speculative
    // decoding is enabled).
    let spec_mode_requested = args.speculative.trim().to_ascii_lowercase();
    let spec_mode_enabled = !matches!(spec_mode_requested.as_str(), "" | "none" | "off" | "false");

    for (pi, prompt_text) in prompts.iter().enumerate() {
        let input = ModelInput::Text {
            prompt: prompt_text.clone(),
        };

        let prompt_token_count = pipeline.tokenize(prompt_text)?.len();

        // Warmup
        for i in 0..args.warmup {
            if prompts.len() > 1 {
                eprintln!(
                    "Prompt {}/{} warmup {}/{}...",
                    pi + 1,
                    prompts.len(),
                    i + 1,
                    args.warmup
                );
            } else {
                eprintln!("Warmup run {}/{}...", i + 1, args.warmup);
            }
            let _ = pipeline.run(input.clone(), &params)?;
        }

        let mut prompt_ttft: Vec<f64> = Vec::new();
        let mut prompt_tbt: Vec<f64> = Vec::new();
        let mut prompt_speed: Vec<f64> = Vec::new();
        let mut prompt_latency: Vec<f64> = Vec::new();

        eprintln!(
            "Benchmarking prompt {}/{} with {} repetitions...",
            pi + 1,
            prompts.len(),
            args.repetitions
        );

        for rep in 0..args.repetitions {
            let infer_start = Instant::now();
            let first_token_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
            let generated_text = Arc::new(Mutex::new(String::new()));
            // Per-run speculative counters, populated from the engine's
            // `OutputChunk::Metrics` chunk. The engine only emits this chunk
            // when speculative decoding is enabled, so these stay at zero
            // otherwise.
            let run_draft_tokens: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));
            let run_accepted_tokens: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));

            let output = Arc::clone(&generated_text);
            let ftt = Arc::clone(&first_token_time);
            let rd = Arc::clone(&run_draft_tokens);
            let ra = Arc::clone(&run_accepted_tokens);

            pipeline.run_stream(input.clone(), &params, &mut move |chunk| {
                match chunk {
                    bloomai_engine::io::OutputChunk::TextDelta(text) => {
                        if text.is_empty() {
                            return Ok(());
                        }
                        let mut ftt_guard = ftt.lock().unwrap_or_else(|e| e.into_inner());
                        if ftt_guard.is_none() {
                            *ftt_guard = Some(Instant::now());
                        }
                        output
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push_str(&text);
                    }
                    bloomai_engine::io::OutputChunk::Metrics {
                        speculative_draft_tokens,
                        speculative_accepted_tokens,
                        ..
                    } => {
                        // The engine reports the cumulative totals for this
                        // run in the trailing Metrics chunk. Overwrite rather
                        // than add so we always reflect the final tally.
                        if let Some(d) = speculative_draft_tokens {
                            *rd.lock().unwrap_or_else(|e| e.into_inner()) = Some(d);
                        }
                        if let Some(a) = speculative_accepted_tokens {
                            *ra.lock().unwrap_or_else(|e| e.into_inner()) = Some(a);
                        }
                    }
                    _ => {}
                }
                Ok(())
            })?;

            // Fold this run's speculative counters into the cross-run totals.
            if spec_mode_enabled {
                let draft = *run_draft_tokens.lock().unwrap_or_else(|e| e.into_inner());
                let accepted = *run_accepted_tokens
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(d) = draft {
                    all_speculative_draft.push(d);
                }
                if let Some(a) = accepted {
                    all_speculative_accepted.push(a);
                }
                if let (Some(d), Some(a)) = (draft, accepted)
                    && d > 0
                {
                    all_speculative_rates.push(a as f64 / d as f64);
                }
            }

            let duration = infer_start.elapsed();
            let output_text = generated_text
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let generated = pipeline.tokenize(&output_text)?.len();
            let ttft = first_token_time
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .map(|t| t.duration_since(infer_start).as_secs_f64() * 1000.0);

            if generated > 0 {
                let decode_secs = ttft
                    .map(|value| (duration.as_secs_f64() - value / 1000.0).max(0.0))
                    .unwrap_or_else(|| duration.as_secs_f64());
                let speed = if generated > 1 && decode_secs > 0.0 {
                    (generated - 1) as f64 / decode_secs
                } else {
                    generated as f64 / duration.as_secs_f64()
                };
                let avg_lat = duration.as_secs_f64() * 1000.0 / generated as f64;

                // TBT = (total_time - TTFT) / (generated - 1)
                let tbt = if generated > 1 {
                    if let Some(t) = ttft {
                        Some((duration.as_secs_f64() * 1000.0 - t) / (generated as f64 - 1.0))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let prompt_proc_speed =
                    ttft.map(|t| prompt_token_count.max(1) as f64 / (t / 1000.0));

                prompt_speed.push(speed);
                prompt_latency.push(avg_lat);
                all_generated_tokens.push(generated);
                all_durations.push(duration.as_secs_f64());
                if let Some(t) = ttft {
                    prompt_ttft.push(t);
                }
                if let Some(t) = tbt {
                    prompt_tbt.push(t);
                }
                if let Some(ps) = prompt_proc_speed {
                    all_prompt_speed.push(ps);
                }

                eprintln!(
                    "  Run {}: {} tokens in {:.2}s ({:.2} tok/s, TTFT: {}, TBT: {})",
                    rep + 1,
                    generated,
                    duration.as_secs_f64(),
                    speed,
                    ttft.map(|t| format!("{:.2}ms", t))
                        .unwrap_or_else(|| "N/A".into()),
                    tbt.map(|t| format!("{:.2}ms", t))
                        .unwrap_or_else(|| "N/A".into()),
                );
            }
        }

        // Aggregate per-prompt results
        if !prompt_ttft.is_empty() {
            let avg = prompt_ttft.iter().sum::<f64>() / prompt_ttft.len() as f64;
            all_ttft.push(avg);
        }
        if !prompt_tbt.is_empty() {
            let avg = prompt_tbt.iter().sum::<f64>() / prompt_tbt.len() as f64;
            all_tbt.push(avg);
        }
        if !prompt_speed.is_empty() {
            let avg = prompt_speed.iter().sum::<f64>() / prompt_speed.len() as f64;
            all_speed.push(avg);
        }
        if !prompt_latency.is_empty() {
            let avg = prompt_latency.iter().sum::<f64>() / prompt_latency.len() as f64;
            all_latency.push(avg);
        }
    }

    // --- Cross-prompt aggregation ---
    let avg_speed = if !all_speed.is_empty() {
        all_speed.iter().sum::<f64>() / all_speed.len() as f64
    } else {
        0.0
    };
    let avg_ttft = if !all_ttft.is_empty() {
        Some(all_ttft.iter().sum::<f64>() / all_ttft.len() as f64)
    } else {
        None
    };
    let avg_tbt = if !all_tbt.is_empty() {
        Some(all_tbt.iter().sum::<f64>() / all_tbt.len() as f64)
    } else {
        None
    };
    let avg_latency = if !all_latency.is_empty() {
        all_latency.iter().sum::<f64>() / all_latency.len() as f64
    } else {
        0.0
    };
    let avg_prompt_proc = if !all_prompt_speed.is_empty() {
        Some(all_prompt_speed.iter().sum::<f64>() / all_prompt_speed.len() as f64)
    } else {
        None
    };

    // Throughput variance
    let speed_variance = if all_speed.len() > 1 {
        let mean = avg_speed;
        let var =
            all_speed.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / all_speed.len() as f64;
        var.sqrt()
    } else {
        0.0
    };

    // Speculative decoding aggregates. We only populate these when the user
    // explicitly requested a non-`none` mode AND the engine reported at least
    // one `Metrics` chunk with draft/accepted totals. This keeps `none` runs
    // clean (fields omitted via `skip_serializing_if`) while still surfacing
    // "mode requested but engine reported nothing" as `None` per-field so
    // downstream consumers can detect the gap.
    let (spec_mode_field, spec_draft_total, spec_accepted_total, spec_rate) = if spec_mode_enabled {
        let draft_sum: usize = all_speculative_draft.iter().sum();
        let accepted_sum: usize = all_speculative_accepted.iter().sum();
        let rate = if draft_sum > 0 {
            Some(accepted_sum as f64 / draft_sum as f64)
        } else if !all_speculative_rates.is_empty() {
            // Engine reported rates but not absolute counts (shouldn't
            // happen with the current candle path, but stay defensive).
            Some(all_speculative_rates.iter().sum::<f64>() / all_speculative_rates.len() as f64)
        } else {
            None
        };
        let draft_field = if all_speculative_draft.is_empty() {
            None
        } else {
            Some(draft_sum)
        };
        let accepted_field = if all_speculative_accepted.is_empty() {
            None
        } else {
            Some(accepted_sum)
        };
        (
            Some(spec_mode_requested.clone()),
            draft_field,
            accepted_field,
            rate,
        )
    } else {
        (None, None, None, None)
    };

    let memory_breakdown = pipeline.memory_estimate().cloned();
    let observed_peak_rss = peak_rss_bytes();
    let peak_memory = observed_peak_rss
        .or_else(|| {
            memory_breakdown
                .as_ref()
                .map(|estimate| estimate.total_bytes)
        })
        .unwrap_or(0);
    let avg_generated_tokens = if all_generated_tokens.is_empty() {
        0
    } else {
        all_generated_tokens.iter().sum::<usize>() / all_generated_tokens.len()
    };
    let avg_duration = if all_durations.is_empty() {
        0.0
    } else {
        all_durations.iter().sum::<f64>() / all_durations.len() as f64
    };
    let dtype_label = args
        .dtype
        .as_deref()
        .and_then(|s| match s.to_lowercase().as_str() {
            "f32" | "float32" => Some(DType::F32),
            "f16" | "float16" => Some(DType::F16),
            "bf16" | "bfloat16" => Some(DType::BF16),
            _ => None,
        })
        .or_else(|| {
            memory_breakdown
                .as_ref()
                .map(|estimate| estimate.weight_dtype)
        })
        .unwrap_or(pipeline.metadata().manifest.primary_dtype);

    let base = BenchmarkResult {
        backend: device_label.clone(),
        model_id: pipeline.metadata().id.clone(),
        dtype: dtype_label,
        quantization: pipeline
            .metadata()
            .manifest
            .quantization
            .as_ref()
            .map(|q| format!("{:?}", q.scheme)),
        tokens_per_second: avg_speed,
        ttft_ms: avg_ttft,
        avg_latency_ms: avg_latency,
        peak_memory_bytes: peak_memory,
        tokens_generated: avg_generated_tokens,
        duration_secs: avg_duration,
        timestamp: chrono_now(),
        notes: Some(format!(
            "Warmup runs: {}, repetitions: {}, prompts: {}, throughput_stddev: {:.2}, token_count=loaded_model_tokenizer, peak_memory={}",
            args.warmup,
            args.repetitions,
            prompts.len(),
            speed_variance,
            if observed_peak_rss.is_some() {
                "observed_process_rss_hwm"
            } else {
                "model_estimate_fallback"
            }
        )),
    };

    let result = ExtendedBenchmarkResult {
        base,
        tbt_ms: avg_tbt,
        prompt_processing_speed: avg_prompt_proc,
        gpu_info: device_label.clone(),
        quantization_detail: quant_detail.clone(),
        hardware: HardwareInfo {
            device: device_label,
            backend: backend_name.to_string(),
            os: os_label,
        },
        timing_breakdown: TimingBreakdown {
            model_load_secs: load_time,
            avg_ttft_ms: avg_ttft,
            avg_tbt_ms: avg_tbt,
            avg_latency_ms: avg_latency,
            throughput_tokens_per_sec: avg_speed,
            prompt_processing_tokens_per_sec: avg_prompt_proc,
        },
        prompts_tested: prompts.len(),
        avg_ttft_ms: avg_ttft,
        avg_tbt_ms: avg_tbt,
        // Standalone `InferencePipeline` path doesn't use the scheduler
        // or `BloomKvCachePool`, so there are no real cache hits/misses to
        // report. We explicitly mark `enabled: false` so downstream budget
        // checkers can distinguish "scheduler not enabled" from "scheduler
        // ran but had no activity" — the latter would have `enabled: true`
        // with zero counters.
        cache_metrics: BenchmarkCacheMetrics {
            enabled: false,
            ..Default::default()
        },
        memory_breakdown,
        speculative_mode: spec_mode_field,
        speculative_draft_tokens: spec_draft_total,
        speculative_accepted_tokens: spec_accepted_total,
        speculative_acceptance_rate: spec_rate,
    };

    println!("{}", serde_json::to_string_pretty(&result)?);

    Ok(())
}

fn chrono_now() -> String {
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let hours = rem / 3600;
    let mins = (rem % 3600) / 60;
    let s = rem % 60;
    let (y, m, d) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, mins, s
    )
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn peak_rss_bytes() -> Option<usize> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return None;
    }
    let max_rss = unsafe { usage.assume_init() }.ru_maxrss;
    #[cfg(target_os = "linux")]
    return usize::try_from(max_rss).ok()?.checked_mul(1024);
    #[cfg(target_os = "macos")]
    return usize::try_from(max_rss).ok();
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn peak_rss_bytes() -> Option<usize> {
    None
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
