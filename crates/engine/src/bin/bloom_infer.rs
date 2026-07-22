#![allow(clippy::type_complexity)]
//! bloom_infer — Standalone inference CLI for Bloom engine.
//!
//! Load a local model and run inference without any external runtime or
//! scheduling layer.  This is the primary entry-point for using Bloom as
//! an independent inference engine (similar to llama.cpp's `main` or
//! vLLM's `entrypoints`).

use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Result};
use bloomai_core::{BenchmarkResult, DType, DeviceKind, GenerationParams};
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
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
use bloomai_engine::executor::vulkan::VulkanEngine;

use bloomai_engine::executor::qwen3_vl::Qwen3VLEngine;
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::wan::WanEngine;
use bloomai_engine::{
    estimate_memory, speculative_mode_is_mtp, Engine, EngineRegistry, InferencePipeline, ModelInput,
};

/// Chat completion message.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

/// JSON prompt parsing.
#[derive(serde::Deserialize, Debug, Clone)]
struct PromptJson {
    prompt: Option<String>,
    messages: Option<Vec<ChatCompletionMessage>>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
}

/// Resolved input from prompt parsing.
struct ResolvedInput {
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    seed: Option<u64>,
}

#[derive(Debug, Clone)]
struct TextGenerationStats {
    text: String,
    generated_tokens: usize,
    infer_secs: f64,
    ttft_secs: Option<f64>,
    tokens_per_sec: f64,
}

/// Bloom standalone inference CLI.
///
/// Load a local model and perform text generation without requiring
/// any external runtime, scheduler or HTTP gateway.
#[derive(Parser, Debug)]
#[command(
    name = "bloom_infer",
    version,
    about = "Bloom standalone inference engine"
)]
struct Args {
    /// Path to Bloom runtime config. Defaults to ~/.bloom/config.json.
    #[arg(long, env = "BLOOM_CONFIG")]
    config: Option<PathBuf>,

    /// Write an example config file and exit.
    #[arg(long, default_value_t = false)]
    init_config: bool,

    /// Path to the model directory or a single GGUF file.
    /// The directory should contain config.json, tokenizer.json and weight files.
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Input prompt for text generation.
    /// If "-" or empty, reads from stdin.
    #[arg(short, long, default_value = "")]
    prompt: String,

    /// Read the prompt from a file instead of --prompt.
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    /// System prompt prepended to the user prompt (chat-style input).
    #[arg(long)]
    system_prompt: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long, visible_alias = "num-predict", default_value_t = 128)]
    max_tokens: usize,

    /// Sampling temperature (higher = more random).
    #[arg(long, visible_alias = "temp", default_value_t = 0.7)]
    temperature: f64,

    /// Top-p (nucleus sampling) threshold.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Random seed for reproducible generation.
    #[arg(long)]
    seed: Option<u64>,

    /// Enable streaming output (print tokens as they are generated).
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// Device to run on: cpu, gpu.
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Selection of backend engine: candle, openvino, funasr, qwen3_vl, longcat.
    #[arg(long, default_value = "candle")]
    backend: String,

    /// Alias for backend.
    #[arg(long)]
    engine: Option<String>,

    /// Precision to load the model with (affects execution).
    /// One of: f32, f16, bf16, q4, q8.
    #[arg(long)]
    dtype: Option<String>,

    /// Context window size in tokens (used for memory estimation).
    #[arg(long, visible_alias = "ctx-size", default_value_t = 2048)]
    context_size: usize,

    /// Fraction of currently available memory Bloom may plan to use at startup.
    #[arg(long, env = "BLOOM_MEMORY_UTILIZATION", default_value_t = 0.75)]
    memory_utilization: f64,

    /// Explicit startup memory reservation size in bytes.
    #[arg(long, env = "BLOOM_RESERVE_MEMORY_BYTES")]
    reserve_memory_bytes: Option<usize>,

    /// Disable startup memory preallocation and only log memory estimates.
    #[arg(long, env = "BLOOM_DISABLE_MEMORY_PREALLOC", default_value_t = false)]
    disable_memory_prealloc: bool,

    /// Enforce strict memory budget check, failing startup if estimate exceeds available memory.
    #[arg(long, env = "BLOOM_STRICT_MEMORY_BUDGET", default_value_t = false)]
    strict_memory_budget: bool,

    /// Enforce strict security check, failing startup if external scripts/plugins are not allowlisted.
    #[arg(long, env = "BLOOM_STRICT_SECURITY", default_value_t = false)]
    strict_security: bool,

    /// Number of CPU threads to use (passed to the underlying runtime).
    #[arg(long)]
    threads: Option<usize>,

    /// Batch size for inference.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Number of GPU layers to offload (currently informational/diagnostics).
    #[arg(long)]
    gpu_layers: Option<usize>,

    /// Speculative decoding mode: none, ngram, draft, mtp.
    #[arg(long, default_value = "none")]
    speculative: String,

    /// Path to draft model (for speculative=draft).
    #[arg(long)]
    draft_model: Option<PathBuf>,

    /// Number of speculative tokens to propose per target verification step.
    #[arg(long, default_value_t = 5)]
    num_speculative_tokens: usize,

    /// N-gram order used when --speculative=ngram.
    #[arg(long, default_value_t = 4)]
    speculative_ngram_order: usize,

    /// Path to an input image (for multimodal VL models).
    #[arg(long)]
    image: Option<PathBuf>,

    // --- Video generation parameters (for wan backend) ---
    /// Video output width in pixels.
    #[arg(long, default_value_t = 832)]
    width: u32,

    /// Video output height in pixels.
    #[arg(long, default_value_t = 480)]
    height: u32,

    /// Number of video frames to generate.
    #[arg(long, default_value_t = 81)]
    num_frames: u32,

    /// Video frames per second.
    #[arg(long, default_value_t = 16.0)]
    fps: f32,

    /// Classifier-free guidance scale for video generation.
    #[arg(long, default_value_t = 5.0)]
    guidance_scale: f64,

    /// Number of diffusion denoising steps.
    #[arg(long, default_value_t = 50)]
    num_steps: u32,

    /// Negative prompt for video generation (what to avoid).
    #[arg(long)]
    negative_prompt: Option<String>,

    /// Output file path for video generation.
    #[arg(long, default_value = "output.gif")]
    output: PathBuf,

    /// Start an interactive chat-style terminal UI after loading the model.
    #[arg(short = 'i', long, visible_alias = "tui", default_value_t = false)]
    interactive: bool,

    /// Print registered engines and exit.
    #[arg(long, default_value_t = false)]
    list_engines: bool,

    /// Run the benchmarking suite instead of a single prompt.
    #[arg(long, default_value_t = false)]
    bench: bool,

    /// Inspect model metadata, engine capabilities, and memory estimates, then exit.
    #[arg(long, default_value_t = false)]
    inspect: bool,

    /// Number of repetition runs for benchmarking.
    #[arg(long, default_value_t = 3)]
    repetitions: usize,

    /// Number of warmup runs for benchmarking.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Output a JSON benchmark result to stdout instead of human-readable metrics.
    #[arg(long, default_value_t = false)]
    bench_json: bool,

    /// Quiet mode: only print the generated text, no headers or metrics.
    #[arg(long, default_value_t = false)]
    quiet: bool,
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

fn apply_config(args: &mut Args, matches: &ArgMatches, config: &bloomai_engine::InferConfig) {
    apply_config_option!(args, matches, config, model);
    apply_config_value!(args, matches, config, prompt);
    apply_config_option!(args, matches, config, prompt_file);
    apply_config_option!(args, matches, config, system_prompt);
    apply_config_value!(args, matches, config, max_tokens);
    apply_config_value!(args, matches, config, temperature);
    apply_config_value!(args, matches, config, top_p);
    apply_config_option!(args, matches, config, seed);
    apply_config_value!(args, matches, config, stream);
    apply_config_value!(args, matches, config, device);
    apply_config_value!(args, matches, config, backend);
    apply_config_option!(args, matches, config, engine);
    apply_config_option!(args, matches, config, dtype);
    apply_config_value!(args, matches, config, context_size);
    apply_config_value!(args, matches, config, memory_utilization);
    apply_config_option!(args, matches, config, reserve_memory_bytes);
    apply_config_value!(args, matches, config, disable_memory_prealloc);
    apply_config_option!(args, matches, config, threads);
    apply_config_value!(args, matches, config, batch_size);
    apply_config_option!(args, matches, config, gpu_layers);
    apply_config_value!(args, matches, config, speculative);
    apply_config_option!(args, matches, config, draft_model);
    apply_config_value!(args, matches, config, num_speculative_tokens);
    apply_config_value!(args, matches, config, speculative_ngram_order);
    apply_config_option!(args, matches, config, image);
    apply_config_value!(args, matches, config, width);
    apply_config_value!(args, matches, config, height);
    apply_config_value!(args, matches, config, num_frames);
    apply_config_value!(args, matches, config, fps);
    apply_config_value!(args, matches, config, guidance_scale);
    apply_config_value!(args, matches, config, num_steps);
    apply_config_option!(args, matches, config, negative_prompt);
    apply_config_value!(args, matches, config, output);
    apply_config_value!(args, matches, config, interactive);
    apply_config_value!(args, matches, config, list_engines);
    apply_config_value!(args, matches, config, bench);
    apply_config_value!(args, matches, config, inspect);
    apply_config_value!(args, matches, config, repetitions);
    apply_config_value!(args, matches, config, warmup);
    apply_config_value!(args, matches, config, bench_json);
    apply_config_value!(args, matches, config, quiet);
}

fn select_backend_name(
    explicit_engine: Option<&str>,
    backend: &str,
    speculative: &str,
    manifest: &bloomai_core::ModelManifest,
) -> String {
    if explicit_engine.is_none() && backend == "candle" {
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
        explicit_engine.unwrap_or(backend).to_string()
    }
}

fn parse_device(s: &str) -> Result<DeviceKind> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(DeviceKind::Cpu),
        "gpu" | "cuda" | "metal" => Ok(DeviceKind::Gpu),
        "npu" | "intel-npu" => Ok(DeviceKind::Npu),
        other => Err(anyhow!(
            "unsupported device '{}', expected: cpu, gpu, npu",
            other
        )),
    }
}

/// Parse a dtype string from the CLI into a DType enum value.
fn parse_dtype_str(s: &str) -> Option<DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Some(DType::F32),
        "f16" | "float16" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        "q4" | "int4" => Some(DType::Q4),
        "q8" | "int8" => Some(DType::Q8),
        _ => None,
    }
}

/// Build the final prompt string by prepending system prompt if provided.
fn build_prompt(system_prompt: Option<&str>, user_prompt: &str) -> String {
    match system_prompt {
        Some(sys) if !sys.is_empty() => {
            format!("<|system|>\n{}\n<|user|>\n{}", sys, user_prompt)
        }
        _ => user_prompt.to_string(),
    }
}

pub fn chat_prompt(messages: &[ChatCompletionMessage]) -> String {
    if messages.len() == 1 && messages[0].role == "user" {
        return messages[0].content.clone();
    }

    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn resolve_and_parse_input(args: &Args) -> Result<ResolvedInput> {
    let raw_input = if let Some(ref path) = args.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read prompt file '{}': {}", path.display(), e))?
    } else if !args.prompt.is_empty() && args.prompt != "-" {
        args.prompt.clone()
    } else {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() && !args.quiet {
            eprintln!("(Reading prompt from stdin. Press Ctrl+D to end input)");
        }
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| anyhow!("failed to read from stdin: {}", e))?;
        buffer
    };

    let trimmed = raw_input.trim();

    if trimmed.starts_with('{') {
        if let Ok(json_obj) = serde_json::from_str::<PromptJson>(trimmed) {
            let prompt = if let Some(messages) = json_obj.messages {
                chat_prompt(&messages)
            } else {
                json_obj.prompt.unwrap_or_default()
            };
            return Ok(ResolvedInput {
                prompt,
                max_tokens: json_obj.max_tokens.unwrap_or(args.max_tokens),
                temperature: json_obj.temperature.unwrap_or(args.temperature),
                top_p: json_obj.top_p.unwrap_or(args.top_p),
                seed: json_obj.seed.or(args.seed),
            });
        }
    } else if trimmed.starts_with('[') {
        if let Ok(messages) = serde_json::from_str::<Vec<ChatCompletionMessage>>(trimmed) {
            let prompt = chat_prompt(&messages);
            return Ok(ResolvedInput {
                prompt,
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                top_p: args.top_p,
                seed: args.seed,
            });
        }
    }

    let prompt = build_prompt(args.system_prompt.as_deref(), &raw_input);
    Ok(ResolvedInput {
        prompt,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
    })
}

fn resolve_model_path_arg(model: &std::path::Path) -> Result<PathBuf> {
    if model.exists() {
        Ok(model.to_path_buf())
    } else {
        Err(anyhow!("model path does not exist: {}", model.display()))
    }
}

fn build_engine_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::default();
    registry.register("candle", Box::new(CandleEngine));
    registry.register("openvino", Box::new(OpenVINOEngine));
    registry.register("funasr", Box::new(FunASREngine));
    registry.register("qwen3_vl", Box::new(Qwen3VLEngine));
    registry.register("longcat", Box::new(LongCatImageEditEngine));
    registry.register("intel-npu", Box::new(IntelNpuEngine));
    registry.register("npu-tts", Box::new(NpuTtsEngine));
    registry.register("onnxruntime", Box::new(OnnxRuntimeEngine));
    registry.register("coreml", Box::new(CoreMlEngine));
    registry.register("mlx", Box::new(MlxEngine));
    registry.register("vulkan", Box::new(VulkanEngine));
    registry.register("llamacpp", Box::new(LlamaCppEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("wan", Box::new(WanEngine));
    registry
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing (controlled via RUST_LOG env var).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let (mut args, matches) = parse_args()?;
    if args.strict_memory_budget {
        std::env::set_var("BLOOM_STRICT_MEMORY_BUDGET", "1");
    }
    if args.strict_security {
        std::env::set_var("BLOOM_STRICT_SECURITY", "1");
    }
    let config_path = bloomai_engine::resolve_config_path(args.config.as_deref())?;
    if args.init_config {
        bloomai_engine::write_default_config(&config_path)?;
        println!("Wrote Bloom config to {}", config_path.display());
        return Ok(());
    }
    let config = bloomai_engine::load_config(&config_path)?;
    apply_config(&mut args, &matches, &config.infer);

    // --- Set up Engine Registry ---
    let registry = build_engine_registry();
    if args.list_engines {
        return print_engines(&registry);
    }

    // --- Resolve model path (directories and single-file GGUF are both first-class) ---
    let model = args.model.as_ref().ok_or_else(|| {
        anyhow!(
            "model path is required; pass --model or set infer.model in {}",
            config_path.display()
        )
    })?;
    let model_path = resolve_model_path_arg(model)?;

    // --- Load manifest first to assist in engine auto-selection if needed ---
    let manifest = bloomai_engine::load_manifest(&model_path)?;

    let backend_name = select_backend_name(
        args.engine.as_deref(),
        &args.backend,
        &args.speculative,
        &manifest,
    );

    let engine = registry
        .get(&backend_name)
        .map_err(|e| anyhow!("{}. Supported engines are: candle, openvino, funasr, qwen3_vl, longcat, intel-npu, npu-tts, onnxruntime, llamacpp, wan.", e))?;

    if args.inspect {
        return print_inspect(
            &args,
            &backend_name,
            engine,
            &model_path,
            &manifest,
            &registry,
        );
    }

    // --- Apply dtype environment variable ---
    if let Some(ref dt) = args.dtype {
        std::env::set_var("BLOOM_DTYPE", dt);
    }
    if let Some(layers) = args.gpu_layers {
        std::env::set_var("BLOOM_GPU_LAYERS", layers.to_string());
    }
    std::env::set_var("BLOOM_SPECULATIVE", &args.speculative);
    std::env::set_var(
        "BLOOM_NUM_SPECULATIVE_TOKENS",
        args.num_speculative_tokens.to_string(),
    );
    std::env::set_var(
        "BLOOM_SPECULATIVE_NGRAM_ORDER",
        args.speculative_ngram_order.to_string(),
    );
    if let Some(ref draft_model) = args.draft_model {
        std::env::set_var("BLOOM_DRAFT_MODEL", draft_model);
    } else {
        std::env::remove_var("BLOOM_DRAFT_MODEL");
    }

    // --- Configure thread pool if requested ---
    if let Some(threads) = args.threads {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        if !args.quiet && !args.bench_json {
            tracing::info!("CPU thread pool set to {}", threads);
        }
    }

    if args.bench {
        return run_bench(&args, engine, &model_path);
    }

    let device = parse_device(&args.device)?;

    if !args.quiet {
        tracing::info!(
            "Bloom standalone inference engine starting\n  model: {}\n  device: {:?}",
            model_path.display(),
            device
        );
        tracing::info!("Engine: {}", engine.name());
        if args.batch_size > 1 {
            tracing::info!("Batch size: {}", args.batch_size);
        }
        if let Some(layers) = args.gpu_layers {
            tracing::info!("GPU offloaded layers config: {}", layers);
        }
        if args.speculative != "none" {
            tracing::info!(
                "Speculative decoding: mode={}, tokens={}, ngram_order={}",
                args.speculative,
                args.num_speculative_tokens,
                args.speculative_ngram_order
            );
        }
    }

    // Pre-load memory estimation and startup reservation probe.
    let memory_context_size = args.context_size.saturating_mul(args.batch_size.max(1));
    let mem = estimate_memory(&manifest, memory_context_size);
    if !args.quiet {
        tracing::info!("Memory estimate: {}", mem.display_summary());
    }
    let memory_plan = bloomai_engine::plan_memory_preallocation(
        mem.clone(),
        bloomai_engine::MemoryPreallocationConfig {
            enabled: !args.disable_memory_prealloc,
            memory_utilization: args.memory_utilization,
            reserve_memory_bytes: args.reserve_memory_bytes,
        },
    )?;
    if !args.quiet {
        tracing::info!("Startup memory plan: {}", memory_plan.display_summary());
    }
    let startup_reservation = bloomai_engine::reserve_memory_for_plan(&memory_plan)?;
    if !args.quiet && !startup_reservation.is_empty() {
        tracing::info!(
            "Startup memory preallocation probe succeeded: {}",
            bloomai_engine::format_bytes(startup_reservation.bytes())
        );
    }
    drop(startup_reservation);

    // Load model via the standalone pipeline
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load_standalone_with_context(
        engine,
        device,
        &model_path,
        args.context_size,
    )?;
    let load_elapsed = load_start.elapsed();

    if pipeline.context_size() != args.context_size || pipeline.device() != device {
        tracing::warn!(
            "Model loaded with OOM degradation: requested Context Size {} on {:?}, got {} on {:?}",
            args.context_size,
            device,
            pipeline.context_size(),
            pipeline.device()
        );
    }
    if !args.quiet {
        tracing::info!(
            "Model loaded: {} ({:.2}s)",
            pipeline.metadata().id,
            load_elapsed.as_secs_f64()
        );
    }

    if args.interactive {
        let params = GenerationParams {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            seed: args.seed,
            response_format: None,
        };
        return run_interactive(&args, &pipeline, params);
    }

    // --- Resolve and parse prompt / input overrides ---
    let resolved_input = resolve_and_parse_input(&args)?;

    // Build generation parameters
    let params = GenerationParams {
        max_tokens: resolved_input.max_tokens,
        temperature: resolved_input.temperature,
        top_p: resolved_input.top_p,
        seed: resolved_input.seed,
        response_format: None,
    };

    // Build input — use VideoGeneration for Wan backends
    let input = if backend_name == "wan" {
        ModelInput::VideoGeneration {
            prompt: resolved_input.prompt.clone(),
            negative_prompt: args.negative_prompt.clone(),
            width: args.width,
            height: args.height,
            num_frames: args.num_frames,
            fps: args.fps,
            guidance_scale: args.guidance_scale,
            num_steps: args.num_steps,
            seed: resolved_input.seed,
        }
    } else if let Some(ref image_path) = args.image {
        let image_bytes = std::fs::read(image_path)?;
        ModelInput::Multi {
            text: Some(resolved_input.prompt.clone()),
            audio: None,
            image: Some(image_bytes),
        }
    } else {
        ModelInput::Text {
            prompt: resolved_input.prompt.clone(),
        }
    };

    // Determine the dtype label for metrics
    let dtype_label = args
        .dtype
        .as_deref()
        .and_then(parse_dtype_str)
        .unwrap_or(manifest.primary_dtype);

    if !args.quiet && !args.bench_json {
        eprintln!("============================================================");
        eprintln!("Prompt: {}", resolved_input.prompt);
        if let Some(ref sys) = args.system_prompt {
            eprintln!("System: {}", sys);
        }
        eprintln!(
            "Max tokens: {}, Temperature: {}, Top-p: {}",
            resolved_input.max_tokens, resolved_input.temperature, resolved_input.top_p
        );
        eprintln!("============================================================");
    }

    // Run inference with timing
    let infer_start = Instant::now();

    // Shared state for streaming metrics
    let token_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let first_token_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));

    if args.stream {
        // Streaming mode: print tokens as they arrive
        let tc = Arc::clone(&token_count);
        let ftt = Arc::clone(&first_token_time);
        let quiet = args.quiet;
        let bench = args.bench_json;

        // Shared state for collecting video frames in streaming mode
        let video_frames: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
        let video_meta: Arc<Mutex<Option<(u32, u32, f32, u32)>>> = Arc::new(Mutex::new(None)); // (w, h, fps, count)
        let vf = Arc::clone(&video_frames);
        let vm = Arc::clone(&video_meta);
        let output_path = args.output.clone();

        if !quiet && !bench {
            eprintln!("--- Streaming Output ---");
        }

        pipeline.run_stream(
            input,
            &params,
            &mut move |chunk: bloomai_engine::io::OutputChunk| {
                match chunk {
                    bloomai_engine::io::OutputChunk::TextDelta(text) => {
                        // Record first token time
                        {
                            let mut ftt_guard = ftt.lock().unwrap();
                            if ftt_guard.is_none() {
                                *ftt_guard = Some(Instant::now());
                            }
                        }
                        // Count tokens
                        {
                            let mut count = tc.lock().unwrap();
                            *count += 1;
                        }
                        if !bench {
                            print!("{}", text);
                            use std::io::Write;
                            std::io::stdout().flush()?;
                        }
                    }
                    bloomai_engine::io::OutputChunk::DiffusionProgress { step, total_steps } => {
                        if !bench {
                            eprint!("\rDiffusion: {}/{}", step, total_steps);
                            if step == total_steps {
                                eprintln!();
                            }
                        }
                        {
                            let mut count = tc.lock().unwrap();
                            *count = step as usize;
                        }
                    }
                    bloomai_engine::io::OutputChunk::Image(bytes) => {
                        std::fs::write(&output_path, bytes)?;
                        if !bench {
                            eprintln!("Image saved to {}", output_path.display());
                        }
                        *tc.lock().unwrap() = 1;
                        *ftt.lock().unwrap() = Some(infer_start);
                    }
                    bloomai_engine::io::OutputChunk::VideoFrame(rgb) => {
                        vf.lock().unwrap().push(rgb);
                    }
                    bloomai_engine::io::OutputChunk::VideoComplete {
                        width,
                        height,
                        fps,
                        frame_count,
                    } => {
                        *vm.lock().unwrap() = Some((width, height, fps, frame_count));
                        if !bench {
                            eprintln!(
                                "Video complete: {}x{}, {} frames @ {} fps",
                                width, height, frame_count, fps
                            );
                        }
                    }
                    bloomai_engine::io::OutputChunk::End if !bench => {
                        println!(); // final newline
                    }
                    bloomai_engine::io::OutputChunk::End => {}
                    _ => {}
                }
                Ok(())
            },
        )?;

        // Save collected video frames to file
        #[cfg(feature = "candle-engine")]
        {
            let frames = video_frames.lock().unwrap();
            if !frames.is_empty() {
                let meta = video_meta.lock().unwrap();
                if let Some((width, height, fps, frame_count)) = *meta {
                    let video = bloomai_engine::io::VideoOutput {
                        width,
                        height,
                        fps,
                        frame_count,
                        frames: frames.clone(),
                    };
                    bloomai_engine::executor::wan::video_encoder::encode_video(
                        &video,
                        &args.output,
                    )?;
                    if !args.bench_json {
                        println!(
                            "Video saved to {} ({}x{}, {} frames @ {} fps)",
                            args.output.display(),
                            width,
                            height,
                            frame_count,
                            fps
                        );
                    }
                    *token_count.lock().unwrap() = frame_count as usize;
                    *first_token_time.lock().unwrap() = Some(infer_start);
                }
            }
        }

        if !args.quiet && !args.bench_json {
            eprintln!("--- End ---");
        }
    } else {
        // Non-streaming mode: collect full output
        let output = pipeline.run(input, &params)?;
        if let Some(ref image) = output.image {
            std::fs::write(&args.output, image)?;
            *token_count.lock().unwrap() = 1;
            *first_token_time.lock().unwrap() = Some(infer_start);
            if !args.bench_json {
                println!("Image saved to {}", args.output.display());
            }
        } else if let Some(ref video) = output.video {
            // Video output — save to file
            #[cfg(feature = "candle-engine")]
            {
                bloomai_engine::executor::wan::video_encoder::encode_video(video, &args.output)?;
                if !args.bench_json {
                    println!(
                        "Video saved to {} ({}x{}, {} frames @ {} fps)",
                        args.output.display(),
                        video.width,
                        video.height,
                        video.frame_count,
                        video.fps
                    );
                }
            }
            #[cfg(not(feature = "candle-engine"))]
            {
                eprintln!("Video encoding requires candle-engine feature");
            }
            *token_count.lock().unwrap() = video.frame_count as usize;
            *first_token_time.lock().unwrap() = Some(infer_start);
        } else if let Some(ref text) = output.text {
            let approx_tokens = text.split_whitespace().count().max(1);
            *token_count.lock().unwrap() = approx_tokens;
            *first_token_time.lock().unwrap() = Some(infer_start);

            if !args.bench_json {
                println!("--- Output ---");
                println!("{}", text);
                println!("--- End ---");
            }
        } else if !args.bench_json {
            println!("(no text output)");
        }
    }

    let infer_elapsed = infer_start.elapsed();
    let total_elapsed = load_start.elapsed();
    let generated_tokens = *token_count.lock().unwrap();
    let ttft = first_token_time
        .lock()
        .unwrap()
        .map(|t| t.duration_since(infer_start).as_secs_f64());
    let generation_secs = infer_elapsed.as_secs_f64();
    let tokens_per_sec = if generation_secs > 0.0 && generated_tokens > 0 {
        generated_tokens as f64 / generation_secs
    } else {
        0.0
    };

    // Output metrics
    if args.bench_json {
        let result = BenchmarkResult {
            backend: args.device.clone(),
            model_id: pipeline.metadata().id.clone(),
            dtype: dtype_label,
            quantization: manifest
                .quantization
                .as_ref()
                .map(|q| format!("{:?}", q.scheme)),
            tokens_per_second: tokens_per_sec,
            ttft_ms: ttft.map(|t| t * 1000.0),
            avg_latency_ms: if generated_tokens > 0 {
                generation_secs * 1000.0 / generated_tokens as f64
            } else {
                0.0
            },
            peak_memory_bytes: mem.total_bytes,
            tokens_generated: generated_tokens,
            duration_secs: total_elapsed.as_secs_f64(),
            timestamp: chrono_now(),
            notes: None,
        };
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else if !args.quiet {
        eprintln!();
        eprintln!("--- Metrics ---");
        eprintln!(
            "Model: {} | Backend: {} | Dtype: {:?}",
            pipeline.metadata().id,
            args.device,
            dtype_label
        );
        eprintln!(
            "Load: {:.2}s | TTFT: {} | Total: {:.2}s",
            load_elapsed.as_secs_f64(),
            ttft.map(|t| format!("{:.0}ms", t * 1000.0))
                .unwrap_or_else(|| "n/a".to_string()),
            total_elapsed.as_secs_f64()
        );
        eprintln!(
            "Tokens: {} generated | Speed: {:.1} tok/s",
            generated_tokens, tokens_per_sec
        );
        eprintln!("Memory estimate: {}", mem.display_summary());
    }

    Ok(())
}

fn print_inspect(
    args: &Args,
    selected_engine: &str,
    selected: &dyn Engine,
    model_path: &std::path::Path,
    manifest: &bloomai_core::ModelManifest,
    registry: &EngineRegistry,
) -> Result<()> {
    let mem = estimate_memory(manifest, args.context_size);
    let engines = registry
        .iter()
        .map(|(name, engine)| {
            let cap = engine.capability();
            serde_json::json!({
                "name": name,
                "maturity": cap.maturity.to_string(),
                "supports_streaming": cap.supports_streaming,
                "supports_quantized_models": cap.supports_quantized_models,
                "max_context_tokens": cap.max_context_tokens,
                "supported_families": cap.supported_families.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>(),
                "supported_dtypes": cap.supported_dtypes.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>(),
                "supported_formats": cap.supported_formats.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>(),
                "supported_devices": cap.supported_devices.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>(),
                "supported_modalities": cap.supported_modalities.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>(),
                "supported_quant_methods": cap.supported_quant_methods.iter().map(|m| m.label()).collect::<Vec<_>>(),
                "diagnostic_tips": cap.diagnostic_tips,
                "construction_guide": cap.construction_guide,
            })
        })
        .collect::<Vec<_>>();

    let selected_cap = selected.capability();
    let output = serde_json::json!({
        "model_path": model_path.display().to_string(),
        "selected_engine": selected_engine,
        "selected_engine_maturity": selected_cap.maturity.to_string(),
        "context_size": args.context_size,
        "manifest": manifest,
        "memory_estimate": mem,
        "engines": engines,
        "notes": [
            "Inspect output is metadata-only and does not load model weights.",
            "Use --bench --bench-json for measured latency and throughput."
        ]
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn print_engines(registry: &EngineRegistry) -> Result<()> {
    println!("Bloom registered engines:");
    for (name, engine) in registry.iter() {
        let cap = engine.capability();
        let devices = cap
            .supported_devices
            .iter()
            .map(|v| format!("{:?}", v).to_lowercase())
            .collect::<Vec<_>>()
            .join(",");
        let families = cap
            .supported_families
            .iter()
            .map(|v| format!("{:?}", v))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "  {:<12} maturity={:<14} streaming={:<5} devices={:<14} families={}",
            name, cap.maturity, cap.supports_streaming, devices, families
        );
    }
    Ok(())
}

fn resolve_interactive_initial_prompt(args: &Args) -> Result<Option<String>> {
    if let Some(ref path) = args.prompt_file {
        let prompt = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read prompt file '{}': {}", path.display(), e))?;
        return Ok(Some(prompt));
    }
    if !args.prompt.is_empty() && args.prompt != "-" {
        return Ok(Some(args.prompt.clone()));
    }
    Ok(None)
}

fn build_interactive_prompt(
    system_prompt: Option<&str>,
    history: &[ChatCompletionMessage],
    user_prompt: &str,
) -> String {
    let mut messages = Vec::with_capacity(history.len() + 2);
    if let Some(system) = system_prompt {
        if !system.is_empty() {
            messages.push(ChatCompletionMessage {
                role: "system".to_string(),
                content: system.to_string(),
            });
        }
    }
    messages.extend_from_slice(history);
    messages.push(ChatCompletionMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    });
    chat_prompt(&messages)
}

fn run_interactive(
    args: &Args,
    pipeline: &InferencePipeline,
    mut params: GenerationParams,
) -> Result<()> {
    use std::io::Write;

    let mut history: Vec<ChatCompletionMessage> = Vec::new();
    let mut system_prompt = args.system_prompt.clone();
    let mut stream = true;
    let mut last_stats: Option<TextGenerationStats> = None;

    if !args.quiet {
        println!("Bloom interactive TUI");
        println!("Model: {}", pipeline.metadata().id);
        println!("Type /help for commands, /bye to exit.");
    }

    if let Some(initial_prompt) = resolve_interactive_initial_prompt(args)? {
        run_interactive_turn(
            pipeline,
            &mut history,
            system_prompt.as_deref(),
            &initial_prompt,
            &params,
            stream,
            &mut last_stats,
        )?;
    }

    loop {
        print!("bloom> ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        let bytes = std::io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            println!();
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('/') {
            if handle_interactive_command(
                line,
                &mut params,
                &mut stream,
                &mut system_prompt,
                &mut history,
                &last_stats,
            )? {
                break;
            }
            continue;
        }

        run_interactive_turn(
            pipeline,
            &mut history,
            system_prompt.as_deref(),
            line,
            &params,
            stream,
            &mut last_stats,
        )?;
    }

    Ok(())
}

fn run_interactive_turn(
    pipeline: &InferencePipeline,
    history: &mut Vec<ChatCompletionMessage>,
    system_prompt: Option<&str>,
    user_prompt: &str,
    params: &GenerationParams,
    stream: bool,
    last_stats: &mut Option<TextGenerationStats>,
) -> Result<()> {
    let prompt = build_interactive_prompt(system_prompt, history, user_prompt);
    let stats = run_text_generation(pipeline, prompt, params, stream)?;

    history.push(ChatCompletionMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    });
    if !stats.text.trim().is_empty() {
        history.push(ChatCompletionMessage {
            role: "assistant".to_string(),
            content: stats.text.clone(),
        });
    }
    *last_stats = Some(stats);
    Ok(())
}

fn run_text_generation(
    pipeline: &InferencePipeline,
    prompt: String,
    params: &GenerationParams,
    stream: bool,
) -> Result<TextGenerationStats> {
    let infer_start = Instant::now();
    let mut text = String::new();
    let mut generated_tokens = 0usize;
    let mut first_token_time: Option<Instant> = None;

    if stream {
        pipeline.run_stream(
            ModelInput::Text { prompt },
            params,
            &mut |chunk: bloomai_engine::io::OutputChunk| {
                match chunk {
                    bloomai_engine::io::OutputChunk::TextDelta(delta) => {
                        if first_token_time.is_none() {
                            first_token_time = Some(Instant::now());
                        }
                        generated_tokens += 1;
                        text.push_str(&delta);
                        print!("{}", delta);
                        use std::io::Write;
                        std::io::stdout().flush()?;
                    }
                    bloomai_engine::io::OutputChunk::End => {
                        println!();
                    }
                    _ => {}
                }
                Ok(())
            },
        )?;
    } else {
        let output = pipeline.run(ModelInput::Text { prompt }, params)?;
        if let Some(output_text) = output.text {
            generated_tokens = output_text.split_whitespace().count().max(1);
            first_token_time = Some(infer_start);
            println!("{}", output_text);
            text = output_text;
        }
    }

    let infer_secs = infer_start.elapsed().as_secs_f64();
    let tokens_per_sec = if infer_secs > 0.0 && generated_tokens > 0 {
        generated_tokens as f64 / infer_secs
    } else {
        0.0
    };
    Ok(TextGenerationStats {
        text,
        generated_tokens,
        infer_secs,
        ttft_secs: first_token_time.map(|t| t.duration_since(infer_start).as_secs_f64()),
        tokens_per_sec,
    })
}

fn handle_interactive_command(
    line: &str,
    params: &mut GenerationParams,
    stream: &mut bool,
    system_prompt: &mut Option<String>,
    history: &mut Vec<ChatCompletionMessage>,
    last_stats: &Option<TextGenerationStats>,
) -> Result<bool> {
    let mut parts = line.splitn(3, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let key = parts.next().unwrap_or_default();
    let value = parts.next().unwrap_or_default().trim();

    match command {
        "/bye" | "/exit" | "/quit" => Ok(true),
        "/help" => {
            print_interactive_help();
            Ok(false)
        }
        "/clear" => {
            history.clear();
            println!("Conversation history cleared.");
            Ok(false)
        }
        "/stats" => {
            print_interactive_stats(last_stats);
            Ok(false)
        }
        "/set" => {
            set_interactive_option(key, value, params, stream, system_prompt)?;
            Ok(false)
        }
        _ => {
            eprintln!("Unknown command: {}. Type /help for commands.", command);
            Ok(false)
        }
    }
}

fn print_interactive_help() {
    println!("Commands:");
    println!("  /help                         Show this help");
    println!("  /set max_tokens <n>           Set maximum generated tokens");
    println!("  /set temperature <n>          Set sampling temperature");
    println!("  /set top_p <n>                Set nucleus sampling threshold");
    println!("  /set seed <n|none>            Set or clear random seed");
    println!("  /set system <text|none>       Set or clear system prompt");
    println!("  /set stream <on|off>          Toggle streaming output");
    println!("  /stats                        Show last response metrics");
    println!("  /clear                        Clear conversation history");
    println!("  /bye                          Exit");
}

fn print_interactive_stats(last_stats: &Option<TextGenerationStats>) {
    if let Some(stats) = last_stats {
        println!(
            "Last turn: {} chunks/tokens, {:.2}s, {:.2} tok/s, TTFT {}",
            stats.generated_tokens,
            stats.infer_secs,
            stats.tokens_per_sec,
            stats
                .ttft_secs
                .map(|v| format!("{:.0}ms", v * 1000.0))
                .unwrap_or_else(|| "n/a".to_string())
        );
    } else {
        println!("No generation has run yet.");
    }
}

fn set_interactive_option(
    key: &str,
    value: &str,
    params: &mut GenerationParams,
    stream: &mut bool,
    system_prompt: &mut Option<String>,
) -> Result<()> {
    match key {
        "max_tokens" | "max-tokens" | "num_predict" | "num-predict" => {
            params.max_tokens = value.parse()?;
            println!("max_tokens = {}", params.max_tokens);
        }
        "temperature" | "temp" => {
            params.temperature = value.parse()?;
            println!("temperature = {}", params.temperature);
        }
        "top_p" | "top-p" => {
            params.top_p = value.parse()?;
            println!("top_p = {}", params.top_p);
        }
        "seed" => {
            params.seed = parse_optional_seed(value)?;
            println!(
                "seed = {}",
                params
                    .seed
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );
        }
        "stream" => {
            *stream = parse_on_off(value)?;
            println!("stream = {}", if *stream { "on" } else { "off" });
        }
        "system" | "system_prompt" | "system-prompt" => {
            if value.eq_ignore_ascii_case("none") || value.is_empty() {
                *system_prompt = None;
            } else {
                *system_prompt = Some(value.to_string());
            }
            println!("system = {}", system_prompt.as_deref().unwrap_or("none"));
        }
        "" => {
            eprintln!("Usage: /set <max_tokens|temperature|top_p|seed|system|stream> <value>");
        }
        other => {
            eprintln!("Unknown option: {}", other);
        }
    }
    Ok(())
}

fn parse_optional_seed(value: &str) -> Result<Option<u64>> {
    match value.to_lowercase().as_str() {
        "" | "none" | "null" | "off" => Ok(None),
        _ => Ok(Some(value.parse()?)),
    }
}

fn parse_on_off(value: &str) -> Result<bool> {
    match value.to_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        _ => Err(anyhow!("expected on/off")),
    }
}

// ==========================================
// Benchmarking Suite Implementation
// ==========================================

fn run_bench(args: &Args, engine: &dyn Engine, model_path: &std::path::Path) -> Result<()> {
    let device_kind = parse_device(&args.device)?;
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

    let prompt = if args.prompt.is_empty() {
        "Explain quantum computing in simple terms".to_string()
    } else {
        args.prompt.clone()
    };
    let input = ModelInput::Text {
        prompt: prompt.clone(),
    };

    // Warmup
    for i in 0..args.warmup {
        if !args.bench_json {
            eprintln!("Warmup run {}/{}...", i + 1, args.warmup);
        }
        let _ = pipeline.run(input.clone(), &params)?;
    }

    let mut ttft_values = Vec::new();
    let mut speed_values = Vec::new();
    let mut latency_values = Vec::new();
    let mut generated_counts = Vec::new();
    let mut prompt_speeds = Vec::new();
    let mut decode_speeds = Vec::new();
    let mut tbt_values = Vec::new();

    let prompt_tokens_count = pipeline.tokenize(&prompt).unwrap_or_default().len();

    if !args.bench_json {
        eprintln!("Benchmarking with {} repetitions...", args.repetitions);
    }
    for rep in 0..args.repetitions {
        let infer_start = Instant::now();
        let first_token_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
        let token_count = Arc::new(Mutex::new(0));

        let tc = Arc::clone(&token_count);
        let ftt = Arc::clone(&first_token_time);

        pipeline.run_stream(input.clone(), &params, &mut move |chunk| {
            if let bloomai_engine::io::OutputChunk::TextDelta(_) = chunk {
                let mut ftt_guard = ftt.lock().unwrap();
                if ftt_guard.is_none() {
                    *ftt_guard = Some(Instant::now());
                }
                *tc.lock().unwrap() += 1;
            }
            Ok(())
        })?;

        let duration = infer_start.elapsed();
        let generated = *token_count.lock().unwrap();
        let ttft = first_token_time
            .lock()
            .unwrap()
            .map(|t| t.duration_since(infer_start).as_secs_f64() * 1000.0);

        if generated > 0 {
            let speed = generated as f64 / duration.as_secs_f64();
            let avg_lat = duration.as_secs_f64() * 1000.0 / generated as f64;
            speed_values.push(speed);
            latency_values.push(avg_lat);
            generated_counts.push(generated);

            if let Some(t_ms) = ttft {
                ttft_values.push(t_ms);

                let prefill_secs = t_ms / 1000.0;
                let decode_secs = duration.as_secs_f64() - prefill_secs;

                if prefill_secs > 0.0 {
                    prompt_speeds.push(prompt_tokens_count as f64 / prefill_secs);
                }
                if decode_secs > 0.0 && generated > 1 {
                    decode_speeds.push((generated - 1) as f64 / decode_secs);
                    tbt_values.push(decode_secs * 1000.0 / (generated - 1) as f64);
                }
            }

            if !args.bench_json {
                eprintln!(
                    "Run {}: generated {} tokens in {:.2}s (speed: {:.2} tok/s, TTFT: {})",
                    rep + 1,
                    generated,
                    duration.as_secs_f64(),
                    speed,
                    ttft.map(|t| format!("{:.2}ms", t))
                        .unwrap_or_else(|| "N/A".to_string())
                );
            }
        }
    }

    if speed_values.is_empty() || latency_values.is_empty() || generated_counts.is_empty() {
        return Err(anyhow!(
            "benchmark produced no text tokens; cannot calculate throughput"
        ));
    }

    let avg_speed = speed_values.iter().sum::<f64>() / speed_values.len() as f64;
    let avg_ttft = if !ttft_values.is_empty() {
        Some(ttft_values.iter().sum::<f64>() / ttft_values.len() as f64)
    } else {
        None
    };
    let avg_latency = latency_values.iter().sum::<f64>() / latency_values.len() as f64;
    let avg_generated =
        generated_counts.iter().sum::<usize>() as f64 / generated_counts.len() as f64;

    let avg_prompt_speed = if !prompt_speeds.is_empty() {
        prompt_speeds.iter().sum::<f64>() / prompt_speeds.len() as f64
    } else {
        0.0
    };
    let avg_decode_speed = if !decode_speeds.is_empty() {
        decode_speeds.iter().sum::<f64>() / decode_speeds.len() as f64
    } else {
        0.0
    };
    let avg_tbt = if !tbt_values.is_empty() {
        tbt_values.iter().sum::<f64>() / tbt_values.len() as f64
    } else {
        0.0
    };

    let peak_memory = pipeline
        .memory_estimate()
        .map(|e| e.total_bytes)
        .unwrap_or(0);
    let dtype_label = args
        .dtype
        .as_deref()
        .and_then(|s| match s.to_lowercase().as_str() {
            "f32" | "float32" => Some(DType::F32),
            "f16" | "float16" => Some(DType::F16),
            "bf16" | "bfloat16" => Some(DType::BF16),
            _ => None,
        })
        .unwrap_or(pipeline.metadata().manifest.primary_dtype);

    let notes = format!(
        "Warmup runs: {}, repetitions: {}. Prompt tokens: {}. Avg Prompt Speed: {:.2} tok/s, Avg Decode Speed: {:.2} tok/s, Avg TBT: {:.2}ms. Machine: {}, OS: {}, Arch: {}",
        args.warmup,
        args.repetitions,
        prompt_tokens_count,
        avg_prompt_speed,
        avg_decode_speed,
        avg_tbt,
        get_machine_model(),
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let result = BenchmarkResult {
        backend: args.device.clone(),
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
        tokens_generated: avg_generated as usize,
        duration_secs: load_time,
        timestamp: chrono_now(),
        notes: Some(notes),
    };

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn get_machine_model() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                return s.trim().to_string();
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

/// Simple ISO 8601 timestamp without pulling in the `chrono` crate.
fn chrono_now() -> String {
    use std::time::SystemTime;
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

/// Convert days since Unix epoch to (year, month, day).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_prompt_single_user_uses_content() {
        let messages = vec![ChatCompletionMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        assert_eq!(chat_prompt(&messages), "hello");
    }

    #[test]
    fn resolve_model_path_keeps_single_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tiny.gguf");
        std::fs::write(&file, b"GGUF").unwrap();

        let resolved = resolve_model_path_arg(&file).unwrap();
        assert_eq!(resolved, file);
    }

    #[test]
    fn select_backend_auto_routes_mtp_to_llamacpp() {
        let manifest = bloomai_core::ModelManifest::default();
        assert_eq!(
            select_backend_name(None, "candle", "native-mtp", &manifest),
            "llamacpp"
        );
    }

    #[test]
    fn select_backend_keeps_explicit_engine() {
        let manifest = bloomai_core::ModelManifest::default();
        assert_eq!(
            select_backend_name(Some("openvino"), "candle", "mtp", &manifest),
            "openvino"
        );
    }

    #[test]
    fn select_backend_keeps_explicit_backend() {
        let manifest = bloomai_core::ModelManifest::default();
        assert_eq!(
            select_backend_name(None, "openvino", "mtp", &manifest),
            "openvino"
        );
    }

    #[test]
    fn select_backend_preserves_existing_auto_routes() {
        let mut manifest = bloomai_core::ModelManifest {
            family: bloomai_core::ModelFamily::FunAsr,
            ..Default::default()
        };
        assert_eq!(
            select_backend_name(None, "candle", "none", &manifest),
            "funasr"
        );

        manifest.family = bloomai_core::ModelFamily::Custom("wan".to_string());
        assert_eq!(
            select_backend_name(None, "candle", "none", &manifest),
            "wan"
        );
    }
}
