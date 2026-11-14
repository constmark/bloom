#![allow(clippy::type_complexity)]
#![cfg_attr(not(test), warn(clippy::unwrap_used))]
//! bloom_infer — Standalone inference CLI for Bloom engine.
//!
//! Load a local model and run inference without any external runtime or
//! scheduling layer.  This is the primary entry-point for using Bloom as
//! an independent inference engine (similar to llama.cpp's `main` or
//! vLLM's `entrypoints`).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Result, anyhow};
use bloomai_app::text_metrics::TextTokenCounter;
use bloomai_core::{BenchmarkResult, DeviceKind, GenerationParams, ResponseFormat};
use clap::builder::BoolishValueParser;
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
use tracing_subscriber::EnvFilter;

use bloomai_app::{
    SUPPORTED_ENGINES_HINT, build_engine_registry, parse_cli_dtype, select_backend_name,
    validate_cli_model_task,
};
use bloomai_engine::{Engine, EngineRegistry, InferencePipeline, ModelInput};

/// Chat completion message.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

/// JSON prompt parsing.
#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct PromptJson {
    prompt: Option<String>,
    messages: Option<Vec<ChatCompletionMessage>>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: Option<u64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    logit_bias: Option<BTreeMap<u32, f64>>,
    response_format: Option<ResponseFormat>,
    logprobs: Option<usize>,
}

/// Resolved input from prompt parsing.
struct ResolvedInput {
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: Option<usize>,
    seed: Option<u64>,
    presence_penalty: f64,
    frequency_penalty: f64,
    logit_bias: Option<BTreeMap<u32, f64>>,
    response_format: Option<ResponseFormat>,
    logprobs: Option<usize>,
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

    /// Optional top-k sampling cap applied before top-p.
    #[arg(long)]
    top_k: Option<usize>,

    /// Presence penalty in -2..=2, applied before temperature.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    presence_penalty: f64,

    /// Frequency penalty in -2..=2, applied before temperature.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    frequency_penalty: f64,

    /// JSON map of token IDs to logit biases in -100..=100.
    #[arg(long, value_parser = parse_logit_bias)]
    logit_bias: Option<BTreeMap<u32, f64>>,

    /// Typed JSON response format: text, json_object, or json_schema.
    #[arg(long, value_parser = parse_response_format)]
    response_format: Option<ResponseFormat>,

    /// Emit selected-token log probabilities and 0..20 alternatives. Requires
    /// --json; --stream emits TokenLogprobs events, buffered output includes logprobs.
    #[arg(long, requires = "json")]
    logprobs: Option<usize>,

    /// Random seed for reproducible generation.
    #[arg(long)]
    seed: Option<u64>,

    /// Enable streaming output (print tokens as they are generated).
    #[arg(long, default_value_t = false, overrides_with = "no_stream")]
    stream: bool,

    /// Disable streaming output, overriding `infer.stream` in the config.
    #[arg(long, default_value_t = false, overrides_with = "stream")]
    no_stream: bool,

    /// Print the complete ModelOutput as JSON; with --stream, emit one
    /// complete OutputChunk JSON value per line (including the End event).
    #[arg(long, conflicts_with_all = ["bench_json", "bench", "interactive", "inspect", "list_engines"])]
    json: bool,

    /// Device to run on: auto, cpu, gpu, or npu. Auto probes the host and
    /// chooses the best device supported by the selected engine.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Selection of backend engine: candle, laya, openvino, funasr, qwen3_vl, longcat.
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
    #[arg(
        long,
        env = "BLOOM_DISABLE_MEMORY_PREALLOC",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    disable_memory_prealloc: bool,

    /// Enforce strict memory budget check, failing startup if estimate exceeds available memory.
    #[arg(
        long,
        env = "BLOOM_STRICT_MEMORY_BUDGET",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
    strict_memory_budget: bool,

    /// Enforce strict security check, failing startup if external scripts/plugins are not allowlisted.
    #[arg(
        long,
        env = "BLOOM_STRICT_SECURITY",
        default_value_t = false,
        value_parser = BoolishValueParser::new()
    )]
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

    /// Speculative decoding mode: none, ngram, draft, or mtp. MTP uses the
    /// external llama.cpp adapter when the backend is selected automatically.
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
    #[arg(
        short = 'i',
        long,
        visible_alias = "tui",
        default_value_t = false,
        overrides_with = "no_interactive"
    )]
    interactive: bool,

    /// Run a single prompt, overriding `infer.interactive` in the config.
    #[arg(long, default_value_t = false, overrides_with = "interactive")]
    no_interactive: bool,

    /// Print registered engines and exit.
    #[arg(long, default_value_t = false)]
    list_engines: bool,

    /// Run the benchmarking suite instead of a single prompt.
    #[arg(long, default_value_t = false, overrides_with = "no_bench")]
    bench: bool,

    /// Skip the benchmarking suite, overriding `infer.bench` in the config.
    #[arg(long, default_value_t = false, overrides_with = "bench")]
    no_bench: bool,

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
    #[arg(long, default_value_t = false, overrides_with = "no_quiet")]
    quiet: bool,

    /// Print headers and metrics, overriding `infer.quiet` in the config.
    #[arg(long, default_value_t = false, overrides_with = "quiet")]
    no_quiet: bool,
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

/// Apply a config boolean that has a `--no-<flag>` CLI negation. Either CLI
/// spelling takes precedence over the config value; the negation wins last.
macro_rules! apply_config_flag {
    ($args:expr_2021, $matches:expr_2021, $config:expr_2021, $field:ident, $negation:ident) => {
        if should_use_config($matches, stringify!($field))
            && should_use_config($matches, stringify!($negation))
        {
            if let Some(value) = $config.$field {
                $args.$field = value;
            }
        }
        if $args.$negation {
            $args.$field = false;
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

fn apply_config(args: &mut Args, matches: &ArgMatches, config: &bloomai_app::InferConfig) {
    apply_config_option!(args, matches, config, model);
    // `--prompt` and `--prompt-file` are one input choice: naming either on
    // the CLI must not let a config `prompt_file` (which is read first)
    // silently replace the prompt the user typed.
    if should_use_config(matches, "prompt") && should_use_config(matches, "prompt_file") {
        apply_config_value!(args, matches, config, prompt);
        apply_config_option!(args, matches, config, prompt_file);
    }
    apply_config_option!(args, matches, config, system_prompt);
    apply_config_value!(args, matches, config, max_tokens);
    apply_config_value!(args, matches, config, temperature);
    apply_config_value!(args, matches, config, top_p);
    apply_config_option!(args, matches, config, top_k);
    apply_config_value!(args, matches, config, presence_penalty);
    apply_config_value!(args, matches, config, frequency_penalty);
    apply_config_option!(args, matches, config, logit_bias);
    apply_config_option!(args, matches, config, response_format);
    // Logprobs are only representable in --json output, so a config default
    // must not break text, interactive, inspection, or benchmark runs.
    if args.json {
        apply_config_option!(args, matches, config, logprobs);
    }
    apply_config_option!(args, matches, config, seed);
    apply_config_flag!(args, matches, config, stream, no_stream);
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
    apply_config_flag!(args, matches, config, interactive, no_interactive);
    apply_config_value!(args, matches, config, list_engines);
    apply_config_flag!(args, matches, config, bench, no_bench);
    apply_config_value!(args, matches, config, inspect);
    apply_config_value!(args, matches, config, repetitions);
    apply_config_value!(args, matches, config, warmup);
    apply_config_value!(args, matches, config, bench_json);
    apply_config_flag!(args, matches, config, quiet, no_quiet);
}

fn parse_device(s: &str, supported_devices: &[DeviceKind]) -> Result<DeviceKind> {
    bloomai_engine::BackendRegistry::default()
        .resolve_device_selector(s, supported_devices)
        .map_err(|error| anyhow!("could not resolve device '{}': {error}", s))
}

/// Build the final prompt string by prepending system prompt if provided.
#[cfg(test)]
fn build_prompt(system_prompt: Option<&str>, user_prompt: &str) -> String {
    build_prompt_for_model(system_prompt, user_prompt, CliChatTemplate::Legacy)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliChatTemplate {
    Legacy,
    ChatMl,
    SmolLm2,
    Llama2,
    Llama3,
    Gemma,
    Qwen3NoThink,
}

fn build_prompt_for_model(
    system_prompt: Option<&str>,
    user_prompt: &str,
    template: CliChatTemplate,
) -> String {
    match system_prompt {
        Some(sys) if !sys.is_empty() => chat_prompt_for_model(
            &[
                ChatCompletionMessage {
                    role: "system".to_string(),
                    content: sys.to_string(),
                },
                ChatCompletionMessage {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                },
            ],
            template,
        ),
        _ if template != CliChatTemplate::Legacy => chat_prompt_for_model(
            &[ChatCompletionMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }],
            template,
        ),
        _ => user_prompt.to_string(),
    }
}

pub fn chat_prompt(messages: &[ChatCompletionMessage]) -> String {
    chat_prompt_for_model(messages, CliChatTemplate::Legacy)
}

fn chat_prompt_for_model(messages: &[ChatCompletionMessage], template: CliChatTemplate) -> String {
    if messages.len() == 1 && messages[0].role == "user" && template == CliChatTemplate::Legacy {
        return messages[0].content.clone();
    }

    if template == CliChatTemplate::Llama2 {
        return llama2_chat_prompt(messages);
    }
    if template == CliChatTemplate::Llama3 {
        return llama3_chat_prompt(messages);
    }
    if template == CliChatTemplate::Gemma {
        let mut prompt = String::new();
        for message in messages {
            prompt.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                message.role, message.content
            ));
        }
        prompt.push_str("<start_of_turn>model\n");
        return prompt;
    }

    let mut prompt = String::new();
    if template == CliChatTemplate::SmolLm2
        && !messages
            .first()
            .is_some_and(|message| message.role == "system")
    {
        prompt.push_str(concat!(
            "<|im_start|>system\n",
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
            "<|im_end|>\n"
        ));
    }
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    if template == CliChatTemplate::Qwen3NoThink {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}

fn llama2_chat_prompt(messages: &[ChatCompletionMessage]) -> String {
    let mut prompt = String::from("<s>");
    let mut index = 0;
    if let Some(system) = messages.first().filter(|message| message.role == "system") {
        if let Some(user) = messages.get(1).filter(|message| message.role == "user") {
            prompt.push_str(&format!(
                "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                system.content, user.content
            ));
            index = 2;
        } else {
            prompt.push_str(&format!(
                "[INST] <<SYS>>\n{}\n<</SYS>> [/INST]",
                system.content
            ));
            index = 1;
        }
    }
    for message in &messages[index..] {
        match message.role.as_str() {
            "user" => prompt.push_str(&format!("<s>[INST] {} [/INST]", message.content)),
            "assistant" => prompt.push_str(&format!(" {} </s>", message.content)),
            _ => prompt.push_str(&format!("[INST] {} [/INST]", message.content)),
        }
    }
    prompt
}

fn llama3_chat_prompt(messages: &[ChatCompletionMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            message.role, message.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

fn manifest_chat_template(manifest: &bloomai_core::ModelManifest) -> CliChatTemplate {
    let architecture = manifest
        .parameters
        .get("gguf_architecture")
        .or_else(|| manifest.parameters.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if architecture.eq_ignore_ascii_case("qwen3") {
        return CliChatTemplate::Qwen3NoThink;
    }
    match manifest
        .parameters
        .get("chat_template_kind")
        .and_then(serde_json::Value::as_str)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("chatml") => CliChatTemplate::ChatMl,
        Some("smollm2") => CliChatTemplate::SmolLm2,
        Some("llama2") => CliChatTemplate::Llama2,
        Some("llama3") => CliChatTemplate::Llama3,
        Some("gemma") => CliChatTemplate::Gemma,
        _ => CliChatTemplate::Legacy,
    }
}

/// Read the raw prompt text from `--prompt-file`, `--prompt`, or stdin
/// (when `--prompt` is empty or `-`).
fn read_raw_prompt(args: &Args) -> Result<String> {
    if let Some(ref path) = args.prompt_file {
        return std::fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read prompt file '{}': {}", path.display(), e));
    }
    if !args.prompt.is_empty() && args.prompt != "-" {
        return Ok(args.prompt.clone());
    }
    use std::io::IsTerminal;
    if std::io::stdin().is_terminal() && !args.quiet {
        eprintln!("(Reading prompt from stdin. Press Ctrl+D to end input)");
    }
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|e| anyhow!("failed to read from stdin: {}", e))?;
    Ok(buffer)
}

fn resolve_and_parse_input(
    args: &Args,
    template: CliChatTemplate,
    tensor: bool,
) -> Result<ResolvedInput> {
    let raw_input = read_raw_prompt(args)?;
    parse_input(args, &raw_input, template, tensor)
}

/// Merge `--system-prompt` into JSON chat messages. A list that already opens
/// with the same system message is left alone; a different one is ambiguous
/// and rejected instead of silently dropping either instruction.
fn merge_system_prompt(
    system_prompt: Option<&str>,
    mut messages: Vec<ChatCompletionMessage>,
) -> Result<Vec<ChatCompletionMessage>> {
    let Some(system) = system_prompt.filter(|system| !system.is_empty()) else {
        return Ok(messages);
    };
    match messages.first() {
        Some(first) if first.role == "system" && first.content == system => {}
        Some(first) if first.role == "system" => {
            return Err(anyhow!(
                "--system-prompt conflicts with the system message already present in the JSON messages; remove one of them"
            ));
        }
        _ => messages.insert(
            0,
            ChatCompletionMessage {
                role: "system".to_string(),
                content: system.to_string(),
            },
        ),
    }
    Ok(messages)
}

fn parse_input(
    args: &Args,
    raw_input: &str,
    template: CliChatTemplate,
    tensor: bool,
) -> Result<ResolvedInput> {
    let trimmed = raw_input.trim();
    let from_args = |prompt: String| ResolvedInput {
        prompt,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        logit_bias: args.logit_bias.clone(),
        response_format: args.response_format.clone(),
        logprobs: args.logprobs,
    };

    if tensor {
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|error| anyhow!("tensor input must be numeric tensor JSON: {error}"))?;
        if !value.is_array() && !value.is_object() {
            return Err(anyhow!(
                "tensor input must be a JSON array or named tensor object"
            ));
        }
        return Ok(from_args(trimmed.to_string()));
    }

    if trimmed.starts_with('{') {
        // Structured decision engines such as Laya consume the request JSON
        // verbatim. Do this before the chat-oriented PromptJson parser, whose
        // optional fields would otherwise turn {state, questions} into an
        // empty prompt.
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
            && value.get("state").is_some()
            && value.get("questions").is_some()
        {
            return Ok(from_args(trimmed.to_string()));
        }
        if serde_json::from_str::<serde_json::Value>(trimmed)
            .ok()
            .is_some_and(|v| v.get("prompt").is_some() || v.get("messages").is_some())
        {
            let json_obj: PromptJson = serde_json::from_str(trimmed)
                .map_err(|error| anyhow!("invalid generation request JSON: {error}"))?;
            if json_obj.prompt.is_some() && json_obj.messages.is_some() {
                return Err(anyhow!(
                    "generation request must specify either prompt or messages, not both"
                ));
            }
            let system_prompt = args.system_prompt.as_deref();
            let prompt = if let Some(messages) = json_obj.messages {
                chat_prompt_for_model(&merge_system_prompt(system_prompt, messages)?, template)
            } else if system_prompt.is_some_and(|system| !system.is_empty()) {
                // A raw completion prompt has no role slot for the system
                // text, so frame both as a chat turn like plain-text input.
                build_prompt_for_model(
                    system_prompt,
                    &json_obj.prompt.unwrap_or_default(),
                    template,
                )
            } else {
                json_obj.prompt.unwrap_or_default()
            };
            return Ok(ResolvedInput {
                prompt,
                max_tokens: json_obj.max_tokens.unwrap_or(args.max_tokens),
                temperature: json_obj.temperature.unwrap_or(args.temperature),
                top_p: json_obj.top_p.unwrap_or(args.top_p),
                top_k: json_obj.top_k.or(args.top_k),
                seed: json_obj.seed.or(args.seed),
                logprobs: json_obj.logprobs.or(args.logprobs),
                presence_penalty: json_obj.presence_penalty.unwrap_or(args.presence_penalty),
                frequency_penalty: json_obj.frequency_penalty.unwrap_or(args.frequency_penalty),
                logit_bias: json_obj.logit_bias.or_else(|| args.logit_bias.clone()),
                response_format: json_obj
                    .response_format
                    .or_else(|| args.response_format.clone()),
            });
        }
    } else if trimmed.starts_with('[')
        && let Ok(messages) = serde_json::from_str::<Vec<ChatCompletionMessage>>(trimmed)
    {
        let messages = merge_system_prompt(args.system_prompt.as_deref(), messages)?;
        return Ok(from_args(chat_prompt_for_model(&messages, template)));
    }

    Ok(from_args(build_prompt_for_model(
        args.system_prompt.as_deref(),
        raw_input,
        template,
    )))
}

fn resolve_model_path_arg(model: &std::path::Path) -> Result<PathBuf> {
    if model.exists() {
        Ok(model.to_path_buf())
    } else {
        Err(anyhow!("model path does not exist: {}", model.display()))
    }
}

fn configure_process_environment(args: &Args) {
    // SAFETY: `main` calls this exactly once before loading a model or starting
    // any Bloom, Tokio, Rayon, or external-runtime worker threads.
    unsafe {
        if args.strict_memory_budget {
            std::env::set_var("BLOOM_STRICT_MEMORY_BUDGET", "1");
        }
        if args.strict_security {
            std::env::set_var("BLOOM_STRICT_SECURITY", "1");
        }
        if let Some(dtype) = &args.dtype {
            std::env::set_var("BLOOM_DTYPE", dtype);
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
        if let Some(draft_model) = &args.draft_model {
            std::env::set_var("BLOOM_DRAFT_MODEL", draft_model);
        } else {
            std::env::remove_var("BLOOM_DRAFT_MODEL");
        }
        if let Some(threads) = args.threads {
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        }
    }
}

fn format_inference_output(
    output: &bloomai_engine::ModelOutput,
    json: bool,
) -> Result<Option<String>> {
    output.validate().map_err(anyhow::Error::msg)?;
    if json {
        Ok(Some(serde_json::to_string(output)?))
    } else if output.text.is_none()
        && output.image.is_none()
        && output.video.is_none()
        && output.audio.is_none()
    {
        output
            .logits
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(Into::into)
    } else {
        Ok(None)
    }
}

fn format_buffered_generation_output(
    output: &bloomai_engine::ModelOutput,
    generation: Option<bloomai_engine::GenerationMetadata>,
    json: bool,
) -> Result<Option<String>> {
    if json && let Some(generation) = generation {
        output.validate().map_err(anyhow::Error::msg)?;
        let mut value = serde_json::to_value(output)?;
        value["generation"] = serde_json::to_value(generation)?;
        return Ok(Some(serde_json::to_string(&value)?));
    }
    format_inference_output(output, json)
}

fn format_stream_output(
    chunk: &bloomai_engine::io::OutputChunk,
    json: bool,
) -> Result<Option<String>> {
    chunk.validate().map_err(anyhow::Error::msg)?;
    if json {
        Ok(Some(serde_json::to_string(chunk)?))
    } else if let bloomai_engine::io::OutputChunk::Embedding(values) = chunk {
        Ok(Some(serde_json::to_string(values)?))
    } else {
        Ok(None)
    }
}

fn validate_logprob_output_mode(args: &Args, logprobs: Option<usize>) -> Result<()> {
    if logprobs.is_some() && !args.json {
        return Err(anyhow!(
            "logprobs requires --json so token metadata is preserved in the output"
        ));
    }
    Ok(())
}

fn main() -> Result<()> {
    // Initialize tracing (controlled via RUST_LOG env var).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let (mut args, matches) = parse_args()?;
    let config_path = bloomai_app::resolve_config_path(args.config.as_deref())?;
    if args.init_config {
        bloomai_app::write_default_config(&config_path)?;
        println!("Wrote Bloom config to {}", config_path.display());
        return Ok(());
    }
    let config = bloomai_app::load_config(&config_path)?;
    apply_config(&mut args, &matches, &config.infer);
    validate_logprob_output_mode(&args, args.logprobs)?;
    GenerationParams {
        logprobs: args.logprobs,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        logit_bias: args.logit_bias.clone(),
        response_format: args.response_format.clone(),
        seed: args.seed,
    }
    .validate()
    .map_err(anyhow::Error::msg)?;
    if args.json
        && (args.bench || args.bench_json || args.interactive || args.inspect || args.list_engines)
    {
        return Err(anyhow!(
            "--json cannot be combined with benchmark, interactive, inspection, or engine-list modes"
        ));
    }
    configure_process_environment(&args);

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
    let chat_template = manifest_chat_template(&manifest);

    let backend_name = select_backend_name(
        args.engine.as_deref(),
        &args.backend,
        &args.speculative,
        &manifest,
    );

    let engine = registry
        .get(&backend_name)
        .map_err(|e| anyhow!("{e}. Supported engines are: {SUPPORTED_ENGINES_HINT}."))?;

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

    validate_cli_model_task("bloom_infer", &manifest)?;

    let tensor_model = bloomai_engine::model_manifest_tasks(&manifest) == ["tensor"];
    if tensor_model
        && (args.interactive
            || args.bench
            || args.bench_json
            || args.system_prompt.is_some()
            || args.image.is_some())
    {
        return Err(anyhow!(
            "tensor models accept numeric JSON with default generation parameters; interactive chat, generation benchmarks, system prompts, and images are unsupported"
        ));
    }

    // --- Report the already-frozen thread-pool configuration if requested ---
    if let Some(threads) = args.threads
        && !args.quiet
        && !args.bench_json
    {
        tracing::info!("CPU thread pool set to {}", threads);
    }

    if args.bench {
        return run_bench(&args, engine, &model_path, chat_template);
    }

    let device = parse_device(&args.device, &engine.supported_devices())?;

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
    let mem = bloomai_engine::estimate_memory_for_device(&manifest, memory_context_size, device);
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
            logprobs: None,
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            presence_penalty: args.presence_penalty,
            frequency_penalty: args.frequency_penalty,
            logit_bias: args.logit_bias.clone(),
            seed: args.seed,
            response_format: args.response_format.clone(),
        };
        return run_interactive(&args, &pipeline, params, chat_template);
    }

    // --- Resolve and parse prompt / input overrides ---
    let resolved_input = resolve_and_parse_input(&args, chat_template, tensor_model)?;
    validate_logprob_output_mode(&args, resolved_input.logprobs)?;

    // Build generation parameters
    let params = GenerationParams {
        logprobs: resolved_input.logprobs,
        max_tokens: resolved_input.max_tokens,
        temperature: resolved_input.temperature,
        top_p: resolved_input.top_p,
        top_k: resolved_input.top_k,
        presence_penalty: resolved_input.presence_penalty,
        frequency_penalty: resolved_input.frequency_penalty,
        logit_bias: resolved_input.logit_bias,
        seed: resolved_input.seed,
        response_format: resolved_input.response_format,
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
        .and_then(parse_cli_dtype)
        .unwrap_or(mem.weight_dtype);

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
        let text_tokens = Arc::new(Mutex::new(TextTokenCounter::default()));
        let text_tokens_for_sink = Arc::clone(&text_tokens);
        let maximum_tokens = params.max_tokens;
        let mut has_text = false;
        // Streaming mode: print tokens as they arrive
        let tc = Arc::clone(&token_count);
        let ftt = Arc::clone(&first_token_time);
        let quiet = args.quiet;
        let bench = args.bench_json;
        let json_output = args.json;

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
            &mut |chunk: bloomai_engine::io::OutputChunk| {
                has_text |= matches!(
                    chunk,
                    bloomai_engine::io::OutputChunk::TextDelta(_)
                        | bloomai_engine::io::OutputChunk::TokenUsage { .. }
                );
                text_tokens_for_sink
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .observe(&chunk, maximum_tokens)?;
                if !bench && let Some(line) = format_stream_output(&chunk, json_output)? {
                    println!("{line}");
                }
                match chunk {
                    bloomai_engine::io::OutputChunk::TextDelta(text) if !text.is_empty() => {
                        // Record first token time
                        {
                            let mut ftt_guard = ftt.lock().unwrap_or_else(|e| e.into_inner());
                            if ftt_guard.is_none() {
                                *ftt_guard = Some(Instant::now());
                            }
                        }
                        if !bench && !json_output {
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
                            let mut count = tc.lock().unwrap_or_else(|e| e.into_inner());
                            *count = step as usize;
                        }
                    }
                    bloomai_engine::io::OutputChunk::Image(bytes) => {
                        std::fs::write(&output_path, bytes)?;
                        if !bench {
                            eprintln!("Image saved to {}", output_path.display());
                        }
                        *tc.lock().unwrap_or_else(|e| e.into_inner()) = 1;
                        *ftt.lock().unwrap_or_else(|e| e.into_inner()) = Some(infer_start);
                    }
                    bloomai_engine::io::OutputChunk::VideoFrame(rgb) => {
                        vf.lock().unwrap_or_else(|e| e.into_inner()).push(rgb);
                    }
                    bloomai_engine::io::OutputChunk::VideoComplete {
                        width,
                        height,
                        fps,
                        frame_count,
                    } => {
                        *vm.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some((width, height, fps, frame_count));
                        if !bench {
                            eprintln!(
                                "Video complete: {}x{}, {} frames @ {} fps",
                                width, height, frame_count, fps
                            );
                        }
                    }
                    bloomai_engine::io::OutputChunk::End if !bench && !json_output => {
                        println!(); // final newline
                    }
                    bloomai_engine::io::OutputChunk::End => {}
                    _ => {}
                }
                Ok(())
            },
        )?;

        if has_text {
            *token_count.lock().unwrap_or_else(|e| e.into_inner()) = text_tokens
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .count(&pipeline)?;
        }

        // Save collected video frames to file
        #[cfg(feature = "candle-engine")]
        {
            let frames = video_frames.lock().unwrap_or_else(|e| e.into_inner());
            if !frames.is_empty() {
                let meta = video_meta.lock().unwrap_or_else(|e| e.into_inner());
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
                    if !args.bench_json && !args.json {
                        println!(
                            "Video saved to {} ({}x{}, {} frames @ {} fps)",
                            args.output.display(),
                            width,
                            height,
                            frame_count,
                            fps
                        );
                    }
                    *token_count.lock().unwrap_or_else(|e| e.into_inner()) = frame_count as usize;
                    *first_token_time.lock().unwrap_or_else(|e| e.into_inner()) = Some(infer_start);
                }
            }
        }

        if !args.quiet && !args.bench_json {
            eprintln!("--- End ---");
        }
    } else {
        // Non-streaming mode: collect full output
        let (output, generation) =
            bloomai_app::text_metrics::run_buffered(&pipeline, input, &params)?;
        if let Some(metadata) = generation {
            *token_count.lock().unwrap_or_else(|e| e.into_inner()) = metadata.output_tokens;
        }
        if !args.bench_json
            && let Some(line) = format_buffered_generation_output(&output, generation, args.json)?
        {
            println!("{line}");
        }
        if let Some(ref image) = output.image {
            std::fs::write(&args.output, image)?;
            *token_count.lock().unwrap_or_else(|e| e.into_inner()) = 1;
            *first_token_time.lock().unwrap_or_else(|e| e.into_inner()) = Some(infer_start);
            if !args.bench_json && !args.json {
                println!("Image saved to {}", args.output.display());
            }
        } else if let Some(ref video) = output.video {
            // Video output — save to file
            #[cfg(feature = "candle-engine")]
            {
                bloomai_engine::executor::wan::video_encoder::encode_video(video, &args.output)?;
                if !args.bench_json && !args.json {
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
            *token_count.lock().unwrap_or_else(|e| e.into_inner()) = video.frame_count as usize;
            *first_token_time.lock().unwrap_or_else(|e| e.into_inner()) = Some(infer_start);
        } else if let Some(ref text) = output.text {
            *token_count.lock().unwrap_or_else(|e| e.into_inner()) = match generation {
                Some(metadata) => metadata.output_tokens,
                None if text.is_empty() => 0,
                None => pipeline.tokenize(text)?.len(),
            };

            if args.quiet && !args.bench_json && !args.json {
                println!("{}", text);
            } else if !args.bench_json && !args.json {
                println!("--- Output ---");
                println!("{}", text);
                println!("--- End ---");
            }
        } else if !args.bench_json && !args.json && output.logits.is_none() {
            println!("(no text output)");
        }
    }

    let infer_elapsed = infer_start.elapsed();
    let total_elapsed = load_start.elapsed();
    let generated_tokens = *token_count.lock().unwrap_or_else(|e| e.into_inner());
    let ttft = first_token_time
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .map(|t| t.duration_since(infer_start).as_secs_f64());
    let generation_secs = infer_elapsed.as_secs_f64();
    let tokens_per_sec = if generation_secs > 0.0 && generated_tokens > 0 {
        generated_tokens as f64 / generation_secs
    } else {
        0.0
    };
    // Report the device actually used after `auto` resolution and any
    // load-time fallback, rather than echoing the user's selector.
    let selected_device_label = format!("{:?}", pipeline.device()).to_lowercase();

    // Output metrics
    if args.bench_json {
        let result = BenchmarkResult {
            backend: selected_device_label.clone(),
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
            selected_device_label,
            dtype_label
        );
        eprintln!(
            "Load: {:.2}s | TTFT: {} | Total: {:.2}s",
            load_elapsed.as_secs_f64(),
            ttft.map(|t| format!("{:.0}ms", t * 1000.0))
                .unwrap_or_else(|| "n/a".to_string()),
            total_elapsed.as_secs_f64()
        );
        if tensor_model {
            eprintln!("Numeric inference: {:.2}s", generation_secs);
        } else {
            eprintln!(
                "Tokens: {} generated | Speed: {:.1} tok/s",
                generated_tokens, tokens_per_sec
            );
        }
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
    let device = parse_device(&args.device, &selected.supported_devices())?;
    let mem = bloomai_engine::estimate_memory_for_device(manifest, args.context_size, device);
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
    template: CliChatTemplate,
) -> String {
    let mut messages = Vec::with_capacity(history.len() + 2);
    if let Some(system) = system_prompt
        && !system.is_empty()
    {
        messages.push(ChatCompletionMessage {
            role: "system".to_string(),
            content: system.to_string(),
        });
    }
    messages.extend_from_slice(history);
    messages.push(ChatCompletionMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    });
    chat_prompt_for_model(&messages, template)
}

struct InteractiveSession {
    history: Vec<ChatCompletionMessage>,
    system_prompt: Option<String>,
    stream: bool,
    last_stats: Option<TextGenerationStats>,
    chat_template: CliChatTemplate,
}

impl InteractiveSession {
    fn new(args: &Args, chat_template: CliChatTemplate) -> Self {
        Self {
            history: Vec::new(),
            system_prompt: args.system_prompt.clone(),
            stream: args.stream,
            last_stats: None,
            chat_template,
        }
    }

    /// Record a completed turn. An empty reply drops the whole exchange so the
    /// history keeps strict user/assistant alternation; two consecutive user
    /// turns break role-alternating templates such as Llama 2 and Gemma.
    fn record_turn(&mut self, user_prompt: &str, reply: &str) {
        if reply.trim().is_empty() {
            return;
        }
        self.history.push(ChatCompletionMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        });
        self.history.push(ChatCompletionMessage {
            role: "assistant".to_string(),
            content: reply.to_string(),
        });
    }
}

fn run_interactive(
    args: &Args,
    pipeline: &InferencePipeline,
    mut params: GenerationParams,
    chat_template: CliChatTemplate,
) -> Result<()> {
    use std::io::Write;

    let mut session = InteractiveSession::new(args, chat_template);

    if !args.quiet {
        println!("Bloom interactive TUI");
        println!("Model: {}", pipeline.metadata().id);
        println!("Type /help for commands, /bye to exit.");
    }

    if let Some(initial_prompt) = resolve_interactive_initial_prompt(args)?
        && let Err(error) = run_interactive_turn(pipeline, &mut session, &initial_prompt, &params)
    {
        eprintln!("Generation failed: {error}");
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
                &mut session.stream,
                &mut session.system_prompt,
                &mut session.history,
                &session.last_stats,
            )? {
                break;
            }
            continue;
        }

        if let Err(error) = run_interactive_turn(pipeline, &mut session, line, &params) {
            eprintln!("Generation failed: {error}");
        }
    }

    Ok(())
}

fn run_interactive_turn(
    pipeline: &InferencePipeline,
    session: &mut InteractiveSession,
    user_prompt: &str,
    params: &GenerationParams,
) -> Result<()> {
    let prompt = build_interactive_prompt(
        session.system_prompt.as_deref(),
        &session.history,
        user_prompt,
        session.chat_template,
    );
    let stats = run_text_generation(pipeline, prompt, params, session.stream)?;
    session.record_turn(user_prompt, &stats.text);
    session.last_stats = Some(stats);
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
    let mut counter = TextTokenCounter::default();
    let generated_tokens;
    let mut first_token_time: Option<Instant> = None;

    if stream {
        pipeline.run_stream(
            ModelInput::Text { prompt },
            params,
            &mut |chunk: bloomai_engine::io::OutputChunk| {
                counter.observe(&chunk, params.max_tokens)?;
                match chunk {
                    bloomai_engine::io::OutputChunk::TextDelta(delta) if !delta.is_empty() => {
                        if first_token_time.is_none() {
                            first_token_time = Some(Instant::now());
                        }
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
        generated_tokens = counter.count(pipeline)?;
    } else {
        let (output, generation) =
            bloomai_app::text_metrics::run_buffered(pipeline, ModelInput::Text { prompt }, params)?;
        generated_tokens = match generation {
            Some(metadata) => metadata.output_tokens,
            None => match output.text.as_deref().filter(|text| !text.is_empty()) {
                Some(text) => pipeline.tokenize(text)?.len(),
                None => 0,
            },
        };
        if let Some(output_text) = output.text {
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
            if let Err(error) = set_interactive_option(key, value, params, stream, system_prompt) {
                eprintln!("Invalid setting: {error}");
            }
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
    println!("  /set top_k <n|none>           Set or clear top-k sampling cap");
    println!("  /set presence_penalty <n>     Set presence penalty (-2..2)");
    println!("  /set frequency_penalty <n>    Set frequency penalty (-2..2)");
    println!("  /set logit_bias <json|none>    Set token biases");
    println!("  /set response_format <json|none> Set constrained output");
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
            "Last turn: {} tokens, {:.2}s, {:.2} tok/s, TTFT {}",
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
    let mut candidate = params.clone();
    match key {
        "max_tokens" | "max-tokens" | "num_predict" | "num-predict" => {
            candidate.max_tokens = value.parse()?
        }
        "temperature" | "temp" => candidate.temperature = value.parse()?,
        "top_p" | "top-p" => candidate.top_p = value.parse()?,
        "top_k" | "top-k" => {
            candidate.top_k =
                if value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("off") {
                    None
                } else {
                    Some(value.parse()?)
                }
        }
        "presence_penalty" | "presence-penalty" => candidate.presence_penalty = value.parse()?,
        "frequency_penalty" | "frequency-penalty" => candidate.frequency_penalty = value.parse()?,
        "logit_bias" | "logit-bias" => {
            candidate.logit_bias = if value.eq_ignore_ascii_case("none") {
                None
            } else {
                Some(parse_logit_bias(value).map_err(anyhow::Error::msg)?)
            }
        }
        "response_format" | "response-format" => {
            candidate.response_format = if value.eq_ignore_ascii_case("none") {
                None
            } else {
                Some(parse_response_format(value).map_err(anyhow::Error::msg)?)
            }
        }
        _ => {}
    }
    candidate.validate().map_err(anyhow::Error::msg)?;
    match key {
        "presence_penalty" | "presence-penalty" | "frequency_penalty" | "frequency-penalty"
        | "logit_bias" | "logit-bias" | "response_format" | "response-format" => {
            *params = candidate;
            println!("{key} = {value}");
        }
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
        "top_k" | "top-k" => {
            params.top_k =
                if value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("off") {
                    None
                } else {
                    let top_k: usize = value.parse()?;
                    if top_k == 0 {
                        anyhow::bail!("top_k must be greater than zero");
                    }
                    Some(top_k)
                };
            println!(
                "top_k = {}",
                params
                    .top_k
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );
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
            eprintln!(
                "Usage: /set <max_tokens|temperature|top_p|top_k|seed|system|stream> <value>"
            );
        }
        other => {
            eprintln!("Unknown option: {}", other);
        }
    }
    Ok(())
}

fn parse_logit_bias(value: &str) -> std::result::Result<BTreeMap<u32, f64>, String> {
    let bias: BTreeMap<u32, f64> =
        serde_json::from_str(value).map_err(|error| format!("invalid logit_bias JSON: {error}"))?;
    GenerationParams {
        logprobs: None,
        logit_bias: Some(bias.clone()),
        ..Default::default()
    }
    .validate()?;
    Ok(bias)
}

fn parse_response_format(value: &str) -> std::result::Result<ResponseFormat, String> {
    let format: ResponseFormat = serde_json::from_str(value)
        .map_err(|error| format!("invalid response_format JSON: {error}"))?;
    GenerationParams {
        logprobs: None,
        response_format: Some(format.clone()),
        ..Default::default()
    }
    .validate()?;
    Ok(format)
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

/// Prompt benchmarked when neither `--prompt` nor `--prompt-file` is given.
const DEFAULT_BENCH_PROMPT: &str = "Explain quantum computing in simple terms";

/// Resolve the benchmark prompt through the same file/stdin/JSON/system-prompt
/// and chat-template path as a normal run, so `--bench` measures the prompt a
/// regular invocation would send.
fn resolve_bench_input(args: &Args, template: CliChatTemplate) -> Result<ResolvedInput> {
    let raw_input = if args.prompt_file.is_none() && args.prompt.is_empty() {
        DEFAULT_BENCH_PROMPT.to_string()
    } else {
        read_raw_prompt(args)?
    };
    parse_input(args, &raw_input, template, false)
}

fn run_bench(
    args: &Args,
    engine: &dyn Engine,
    model_path: &std::path::Path,
    template: CliChatTemplate,
) -> Result<()> {
    // Read stdin/files before loading weights so input errors fail fast.
    let resolved = resolve_bench_input(args, template)?;
    let device_kind = parse_device(&args.device, &engine.supported_devices())?;
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
        logprobs: None,
        max_tokens: resolved.max_tokens,
        temperature: resolved.temperature,
        top_p: resolved.top_p,
        top_k: resolved.top_k,
        presence_penalty: resolved.presence_penalty,
        frequency_penalty: resolved.frequency_penalty,
        logit_bias: resolved.logit_bias,
        seed: resolved.seed.or(Some(42)),
        response_format: resolved.response_format,
    };
    params.validate().map_err(anyhow::Error::msg)?;

    let prompt = resolved.prompt;
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
    let mut duration_values = Vec::new();
    let mut generated_counts = Vec::new();
    let mut prompt_speeds = Vec::new();
    let mut decode_speeds = Vec::new();
    let mut tbt_values = Vec::new();

    let prompt_tokens_count = pipeline.tokenize(&prompt)?.len();

    if !args.bench_json {
        eprintln!("Benchmarking with {} repetitions...", args.repetitions);
    }
    for rep in 0..args.repetitions {
        let infer_start = Instant::now();
        let first_token_time: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
        let token_count = Arc::new(Mutex::new(TextTokenCounter::default()));

        let tc = Arc::clone(&token_count);
        let ftt = Arc::clone(&first_token_time);

        let maximum_tokens = params.max_tokens;
        pipeline.run_stream(input.clone(), &params, &mut move |chunk| {
            tc.lock()
                .unwrap_or_else(|e| e.into_inner())
                .observe(&chunk, maximum_tokens)?;
            if let bloomai_engine::io::OutputChunk::TextDelta(text) = chunk
                && !text.is_empty()
            {
                let mut ftt_guard = ftt.lock().unwrap_or_else(|e| e.into_inner());
                if ftt_guard.is_none() {
                    *ftt_guard = Some(Instant::now());
                }
            }
            Ok(())
        })?;

        let duration = infer_start.elapsed();
        let generated = token_count
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .count(&pipeline)?;
        let ttft = first_token_time
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map(|t| t.duration_since(infer_start).as_secs_f64() * 1000.0);

        if generated > 0 {
            let speed = generated as f64 / duration.as_secs_f64();
            let avg_lat = duration.as_secs_f64() * 1000.0 / generated as f64;
            speed_values.push(speed);
            latency_values.push(avg_lat);
            duration_values.push(duration.as_secs_f64());
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
        .and_then(parse_cli_dtype)
        .or_else(|| {
            pipeline
                .memory_estimate()
                .map(|estimate| estimate.weight_dtype)
        })
        .unwrap_or(pipeline.metadata().manifest.primary_dtype);

    let notes = format!(
        "Model load: {:.3}s. token_count=native_usage_or_complete_output_tokenizer. Warmup runs: {}, repetitions: {}. Prompt tokens: {}. Avg Prompt Speed: {:.2} tok/s, Avg Decode Speed: {:.2} tok/s, Avg TBT: {:.2}ms. Machine: {}, OS: {}, Arch: {}",
        load_time,
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
        backend: format!("{:?}", pipeline.device()).to_lowercase(),
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
        duration_secs: duration_values.iter().sum::<f64>() / duration_values.len() as f64,
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
            && let Ok(s) = String::from_utf8(output.stdout)
        {
            return s.trim().to_string();
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
    fn cli_rejects_rerank_only_models_before_generation() {
        let mut manifest = bloomai_core::ModelManifest {
            family: bloomai_core::ModelFamily::Bert,
            ..Default::default()
        };
        manifest.parameters.insert(
            "bloom_encoder_kind".into(),
            serde_json::json!("cross_encoder"),
        );
        let error = validate_cli_model_task("bloom_infer", &manifest)
            .unwrap_err()
            .to_string();
        assert!(error.contains("rerank-only"), "{error}");
        assert!(error.contains("/v1/rerank"), "{error}");
        validate_cli_model_task("bloom_infer", &bloomai_core::ModelManifest::default()).unwrap();
    }

    #[test]
    fn buffered_cli_json_preserves_explicit_generation_metadata() {
        let output = bloomai_engine::ModelOutput::from(Vec::<bloomai_engine::OutputChunk>::new());
        let metadata = bloomai_engine::GenerationMetadata {
            input_tokens: 7,
            output_tokens: 1,
            reason: bloomai_engine::GenerationFinishReason::Eos,
        };
        let encoded = format_buffered_generation_output(&output, Some(metadata), true)
            .unwrap()
            .unwrap();
        let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(
            value["generation"],
            serde_json::json!({"input_tokens":7,"output_tokens":1,"reason":"eos"})
        );
        let legacy = format_buffered_generation_output(&output, None, true)
            .unwrap()
            .unwrap();
        assert!(
            serde_json::from_str::<serde_json::Value>(&legacy)
                .unwrap()
                .get("generation")
                .is_none()
        );
    }

    #[test]
    fn logprob_controls_preserve_metadata_and_config_json_precedence() {
        assert!(Args::try_parse_from(["bloom_infer", "--logprobs", "2"]).is_err());
        let matches = Args::command()
            .try_get_matches_from([
                "bloom_infer",
                "--json",
                "--logprobs",
                "2",
                "--prompt",
                r#"{"prompt":"hello","logprobs":0}"#,
            ])
            .unwrap();
        let mut args = Args::from_arg_matches(&matches).unwrap();
        apply_config(
            &mut args,
            &matches,
            &bloomai_app::InferConfig {
                logprobs: Some(5),
                ..Default::default()
            },
        );
        assert_eq!(args.logprobs, Some(2));
        let resolved = resolve_and_parse_input(&args, CliChatTemplate::Legacy, false).unwrap();
        assert_eq!(resolved.logprobs, Some(0));
        validate_logprob_output_mode(&args, resolved.logprobs).unwrap();
        args.json = false;
        assert!(validate_logprob_output_mode(&args, resolved.logprobs).is_err());
        let probability = bloomai_engine::TokenLogprobs {
            token: bloomai_engine::TokenLogprob {
                token_id: 1,
                token: "a".into(),
                bytes: vec![b'a'],
                logprob: -0.1,
            },
            top_logprobs: Vec::new(),
        };
        let encoded = format_stream_output(
            &bloomai_engine::io::OutputChunk::TokenLogprobs(probability.clone()),
            true,
        )
        .unwrap()
        .unwrap();
        assert!(encoded.contains("TokenLogprobs"));
        let output = bloomai_engine::ModelOutput {
            text: Some("a".into()),
            logprobs: Some(vec![probability]),
            logits: None,
            image: None,
            audio: None,
            video: None,
        };
        let encoded: serde_json::Value =
            serde_json::from_str(&format_inference_output(&output, true).unwrap().unwrap())
                .unwrap();
        assert_eq!(
            encoded["logprobs"][0]["token"]["bytes"],
            serde_json::json!([97])
        );
    }

    #[test]
    fn interactive_invalid_command_preserves_session_and_accepts_the_next_command() {
        let mut params = GenerationParams::default();
        let original = params.clone();
        let mut stream = true;
        let mut system = Some("Keep context".to_string());
        let mut history = vec![ChatCompletionMessage {
            role: "user".into(),
            content: "Earlier turn".into(),
        }];
        assert!(
            !handle_interactive_command(
                "/set temperature NaN",
                &mut params,
                &mut stream,
                &mut system,
                &mut history,
                &None
            )
            .unwrap()
        );
        assert_eq!(params, original);
        assert_eq!(history[0].content, "Earlier turn");
        assert_eq!(system.as_deref(), Some("Keep context"));
        assert!(
            !handle_interactive_command(
                "/set temperature 0.4",
                &mut params,
                &mut stream,
                &mut system,
                &mut history,
                &None
            )
            .unwrap()
        );
        assert_eq!(params.temperature, 0.4);
        assert!(
            handle_interactive_command(
                "/bye",
                &mut params,
                &mut stream,
                &mut system,
                &mut history,
                &None
            )
            .unwrap()
        );
    }

    #[test]
    fn sampling_controls_flow_through_flags_config_and_json() {
        let matches = Args::command()
            .try_get_matches_from([
                "bloom_infer",
                "--presence-penalty",
                "-1",
                "--frequency-penalty",
                "0.4",
                "--logit-bias",
                r#"{"7":100}"#,
                "--response-format",
                r#"{"type":"json_object"}"#,
                "--prompt",
                r#"{"prompt":"hello","frequency_penalty":0.9,"logit_bias":{"8":-50}}"#,
            ])
            .unwrap();
        let mut args = Args::from_arg_matches(&matches).unwrap();
        let config = bloomai_app::InferConfig {
            presence_penalty: Some(1.0),
            frequency_penalty: Some(1.0),
            logit_bias: Some(BTreeMap::from([(9, 5.0)])),
            response_format: Some(ResponseFormat::Text),
            ..Default::default()
        };
        apply_config(&mut args, &matches, &config);
        let resolved = resolve_and_parse_input(&args, CliChatTemplate::Legacy, false).unwrap();
        assert_eq!(resolved.presence_penalty, -1.0);
        assert_eq!(resolved.frequency_penalty, 0.9);
        assert_eq!(resolved.logit_bias, Some(BTreeMap::from([(8, -50.0)])));
        assert_eq!(resolved.response_format, Some(ResponseFormat::JsonObject));
        let defaults = Args::command()
            .try_get_matches_from(["bloom_infer", "--prompt", "hello"])
            .unwrap();
        let mut args = Args::from_arg_matches(&defaults).unwrap();
        apply_config(&mut args, &defaults, &config);
        assert_eq!(args.presence_penalty, 1.0);
        assert_eq!(args.logit_bias, config.logit_bias);
        assert_eq!(args.response_format, Some(ResponseFormat::Text));
    }

    #[test]
    fn json_prompts_are_not_silently_erased_and_request_errors_are_explicit() {
        let literal = r#"{"some":"data"}"#;
        let args = Args::parse_from(["bloom_infer", "--prompt", literal]);
        let resolved = resolve_and_parse_input(&args, CliChatTemplate::Legacy, false).unwrap();
        assert_eq!(resolved.prompt, literal);
        for invalid in [
            r#"{"prompt":"hello","temperatur":0.1}"#,
            r#"{"prompt":"hello","temperature":"hot"}"#,
            r#"{"prompt":"hello","messages":[]}"#,
        ] {
            let args = Args::parse_from(["bloom_infer", "--prompt", invalid]);
            assert!(resolve_and_parse_input(&args, CliChatTemplate::Legacy, false).is_err());
        }
    }

    #[test]
    fn interactive_invalid_updates_leave_valid_settings_unchanged() {
        let mut params = GenerationParams::default();
        let expected = params.clone();
        let mut stream = false;
        let mut system = None;
        for (key, value) in [
            ("temperature", "NaN"),
            ("top_p", "0"),
            ("max_tokens", "0"),
            ("presence_penalty", "3"),
            ("frequency_penalty", "inf"),
            ("logit_bias", r#"{"7":101}"#),
            (
                "response_format",
                r#"{"type":"json_schema","json_schema":[]}"#,
            ),
        ] {
            assert!(
                set_interactive_option(key, value, &mut params, &mut stream, &mut system).is_err()
            );
            assert_eq!(params, expected);
        }
        set_interactive_option(
            "presence_penalty",
            "-0.5",
            &mut params,
            &mut stream,
            &mut system,
        )
        .unwrap();
        assert_eq!(params.presence_penalty, -0.5);
        set_interactive_option(
            "response_format",
            r#"{"type":"json_object"}"#,
            &mut params,
            &mut stream,
            &mut system,
        )
        .unwrap();
        assert_eq!(params.response_format, Some(ResponseFormat::JsonObject));
    }

    #[test]
    fn chat_prompt_single_user_uses_content() {
        let messages = vec![ChatCompletionMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        assert_eq!(chat_prompt(&messages), "hello");
    }

    #[test]
    fn system_prompt_uses_the_same_chatml_contract_as_message_input() {
        assert_eq!(
            build_prompt(Some("Follow policy."), "Hello"),
            "<|im_start|>system\nFollow policy.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(build_prompt(None, "Hello"), "Hello");
    }

    #[test]
    fn qwen3_prompt_prefills_disabled_thinking_for_plain_and_message_input() {
        let expected = concat!(
            "<|im_start|>user\nHello<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            build_prompt_for_model(None, "Hello", CliChatTemplate::Qwen3NoThink),
            expected
        );
        assert_eq!(
            chat_prompt_for_model(
                &[ChatCompletionMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                }],
                CliChatTemplate::Qwen3NoThink,
            ),
            expected
        );

        let mut manifest = bloomai_core::ModelManifest::default();
        manifest
            .parameters
            .insert("gguf_architecture".to_string(), serde_json::json!("qwen3"));
        assert_eq!(
            manifest_chat_template(&manifest),
            CliChatTemplate::Qwen3NoThink
        );
    }

    #[test]
    fn classified_llama_templates_are_used_by_cli_inputs() {
        let messages = [ChatCompletionMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let smollm = chat_prompt_for_model(&messages, CliChatTemplate::SmolLm2);
        assert!(smollm.starts_with(concat!(
            "<|im_start|>system\n",
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
            "<|im_end|>\n"
        )));
        assert!(smollm.ends_with("<|im_start|>assistant\n"));

        let llama3 = chat_prompt_for_model(&messages, CliChatTemplate::Llama3);
        assert!(llama3.starts_with("<|begin_of_text|><|start_header_id|>user"));
        assert!(llama3.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

        let mut manifest = bloomai_core::ModelManifest::default();
        manifest.parameters.insert(
            "chat_template_kind".to_string(),
            serde_json::json!("smollm2"),
        );
        assert_eq!(manifest_chat_template(&manifest), CliChatTemplate::SmolLm2);
    }

    #[test]
    fn resolve_model_path_keeps_single_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tiny.gguf");
        std::fs::write(&file, b"GGUF").unwrap();

        let resolved = resolve_model_path_arg(&file).unwrap();
        assert_eq!(resolved, file);
    }

    fn args_with_config(argv: &[&str], config: &bloomai_app::InferConfig) -> Args {
        let matches = Args::command()
            .try_get_matches_from(std::iter::once("bloom_infer").chain(argv.iter().copied()))
            .unwrap();
        let mut args = Args::from_arg_matches(&matches).unwrap();
        apply_config(&mut args, &matches, config);
        args
    }

    #[test]
    fn cli_prompt_choice_suppresses_both_config_prompt_sources() {
        let dir = tempfile::tempdir().unwrap();
        let config_file = dir.path().join("config-prompt.txt");
        let cli_file = dir.path().join("cli-prompt.txt");
        std::fs::write(&config_file, "from config file").unwrap();
        std::fs::write(&cli_file, "from cli file").unwrap();
        let config = bloomai_app::InferConfig {
            prompt: Some("from config".into()),
            prompt_file: Some(config_file.clone()),
            ..Default::default()
        };

        let args = args_with_config(&["--prompt", "from cli"], &config);
        assert_eq!(args.prompt_file, None);
        let resolved = resolve_and_parse_input(&args, CliChatTemplate::Legacy, false).unwrap();
        assert_eq!(resolved.prompt, "from cli");
        assert_eq!(
            resolve_interactive_initial_prompt(&args)
                .unwrap()
                .as_deref(),
            Some("from cli")
        );

        let args = args_with_config(&["--prompt-file", cli_file.to_str().unwrap()], &config);
        assert_eq!(args.prompt, "");
        let resolved = resolve_and_parse_input(&args, CliChatTemplate::Legacy, false).unwrap();
        assert_eq!(resolved.prompt, "from cli file");

        let args = args_with_config(&[], &config);
        assert_eq!(args.prompt, "from config");
        assert_eq!(args.prompt_file, Some(config_file));
    }

    #[test]
    fn config_boolean_modes_can_be_negated_from_the_cli() {
        let config = bloomai_app::InferConfig {
            stream: Some(true),
            quiet: Some(true),
            interactive: Some(true),
            bench: Some(true),
            ..Default::default()
        };
        let args = args_with_config(&[], &config);
        assert!(args.stream && args.quiet && args.interactive && args.bench);

        let args = args_with_config(
            &[
                "--no-stream",
                "--no-quiet",
                "--no-interactive",
                "--no-bench",
            ],
            &config,
        );
        assert!(!args.stream && !args.quiet && !args.interactive && !args.bench);

        // The last spelling wins, and an explicit positive flag beats a
        // config `false`.
        let off = bloomai_app::InferConfig {
            stream: Some(false),
            quiet: Some(false),
            ..Default::default()
        };
        let args = args_with_config(&["--no-stream", "--stream", "--quiet"], &off);
        assert!(args.stream && args.quiet);
        let args = args_with_config(&["--stream", "--no-stream"], &config);
        assert!(!args.stream);
    }

    #[test]
    fn config_logprobs_only_apply_to_json_output() {
        let config = bloomai_app::InferConfig {
            logprobs: Some(3),
            ..Default::default()
        };
        for argv in [
            &["--list-engines"][..],
            &["-i"],
            &["--bench"],
            &["--inspect"],
            &[],
        ] {
            let args = args_with_config(argv, &config);
            assert_eq!(args.logprobs, None, "{argv:?}");
            validate_logprob_output_mode(&args, args.logprobs).unwrap();
        }
        let args = args_with_config(&["--json"], &config);
        assert_eq!(args.logprobs, Some(3));
        validate_logprob_output_mode(&args, args.logprobs).unwrap();
    }

    #[test]
    fn system_prompt_is_merged_into_json_chat_inputs() {
        let expected = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        for prompt in [
            r#"{"messages":[{"role":"user","content":"Hi"}]}"#,
            r#"[{"role":"user","content":"Hi"}]"#,
            r#"{"prompt":"Hi"}"#,
            r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]}"#,
        ] {
            let args = Args::parse_from([
                "bloom_infer",
                "--system-prompt",
                "Be brief.",
                "--prompt",
                prompt,
            ]);
            let resolved = resolve_and_parse_input(&args, CliChatTemplate::ChatMl, false).unwrap();
            assert_eq!(resolved.prompt, expected, "{prompt}");
        }

        for prompt in [
            r#"{"messages":[{"role":"system","content":"Be verbose."},{"role":"user","content":"Hi"}]}"#,
            r#"[{"role":"system","content":"Be verbose."},{"role":"user","content":"Hi"}]"#,
        ] {
            let args = Args::parse_from([
                "bloom_infer",
                "--system-prompt",
                "Be brief.",
                "--prompt",
                prompt,
            ]);
            let error = resolve_and_parse_input(&args, CliChatTemplate::ChatMl, false)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("--system-prompt conflicts"), "{error}");
        }

        // Without --system-prompt a raw JSON completion prompt stays verbatim.
        let args = Args::parse_from(["bloom_infer", "--prompt", r#"{"prompt":"Hi"}"#]);
        let resolved = resolve_and_parse_input(&args, CliChatTemplate::ChatMl, false).unwrap();
        assert_eq!(resolved.prompt, "Hi");
    }

    #[test]
    fn bench_prompt_uses_the_normal_input_resolver() {
        let args = Args::parse_from(["bloom_infer", "--bench", "--system-prompt", "Be brief."]);
        let resolved = resolve_bench_input(&args, CliChatTemplate::ChatMl).unwrap();
        assert_eq!(
            resolved.prompt,
            build_prompt_for_model(
                Some("Be brief."),
                DEFAULT_BENCH_PROMPT,
                CliChatTemplate::ChatMl
            )
        );

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("prompt.json");
        std::fs::write(&file, r#"{"prompt":"From file","max_tokens":7}"#).unwrap();
        let args = Args::parse_from([
            "bloom_infer",
            "--bench",
            "--prompt-file",
            file.to_str().unwrap(),
        ]);
        let resolved = resolve_bench_input(&args, CliChatTemplate::Qwen3NoThink).unwrap();
        assert_eq!(resolved.prompt, "From file");
        assert_eq!(resolved.max_tokens, 7);

        let args = Args::parse_from(["bloom_infer", "--bench", "--prompt", "Hello"]);
        let resolved = resolve_bench_input(&args, CliChatTemplate::Qwen3NoThink).unwrap();
        assert_eq!(
            resolved.prompt,
            build_prompt_for_model(None, "Hello", CliChatTemplate::Qwen3NoThink)
        );
    }

    #[test]
    fn interactive_session_honours_stream_flag_and_keeps_turns_alternating() {
        let args = Args::parse_from(["bloom_infer", "-i"]);
        assert!(!InteractiveSession::new(&args, CliChatTemplate::Legacy).stream);
        let args = Args::parse_from(["bloom_infer", "-i", "--stream"]);
        let mut session = InteractiveSession::new(&args, CliChatTemplate::Legacy);
        assert!(session.stream);

        session.record_turn("first", "  \n");
        assert!(session.history.is_empty());
        session.record_turn("second", "answer");
        session.record_turn("third", "");
        let roles = session
            .history
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>();
        assert_eq!(roles, ["user", "assistant"]);
        assert_eq!(session.history[0].content, "second");
    }

    #[test]
    fn structured_laya_prompt_is_preserved_verbatim() {
        let args = Args::parse_from([
            "bloom_infer",
            "--prompt",
            r#"{"state":"hello","questions":{"q":{"type":"noul","instructions":"is it safe?"}}}"#,
        ]);
        let resolved = resolve_and_parse_input(&args, CliChatTemplate::Legacy, false).unwrap();
        assert_eq!(resolved.prompt, args.prompt);
    }

    #[test]
    fn tensor_input_preserves_numeric_json_and_generation_controls() {
        for prompt in [
            "[1.0,2.0]",
            r#"{"inputs":{"x":{"shape":[1,2],"data":[1,2]}}}"#,
        ] {
            let args = Args::parse_from([
                "bloom_infer",
                "--prompt",
                prompt,
                "--max-tokens",
                "9",
                "--top-k",
                "4",
            ]);
            let resolved = resolve_and_parse_input(&args, CliChatTemplate::ChatMl, true).unwrap();
            assert_eq!(resolved.prompt, prompt);
            assert_eq!(resolved.max_tokens, 9);
            assert_eq!(resolved.top_k, Some(4));
            // Engines see the exact non-default controls and reject them;
            // the CLI must not silently replace them with tensor defaults.
        }
    }

    #[test]
    fn numeric_and_json_output_preserve_every_value_without_changing_text_mode() {
        let output = bloomai_engine::ModelOutput {
            logprobs: None,
            logits: Some(vec![1.25, -2.5, 0.0]),
            text: None,
            audio: None,
            image: None,
            video: None,
        };
        let plain: Vec<f32> =
            serde_json::from_str(&format_inference_output(&output, false).unwrap().unwrap())
                .unwrap();
        assert_eq!(plain, output.logits.clone().unwrap());
        let encoded: serde_json::Value =
            serde_json::from_str(&format_inference_output(&output, true).unwrap().unwrap())
                .unwrap();
        assert_eq!(encoded, serde_json::to_value(&output).unwrap());
        let event = bloomai_engine::io::OutputChunk::Embedding(plain.clone());
        assert_eq!(
            format_stream_output(&event, false).unwrap().unwrap(),
            serde_json::to_string(&plain).unwrap()
        );
        assert_eq!(
            format_stream_output(&event, true).unwrap().unwrap(),
            serde_json::to_string(&event).unwrap()
        );
        assert_eq!(
            format_stream_output(&bloomai_engine::io::OutputChunk::End, true)
                .unwrap()
                .unwrap(),
            "\"End\""
        );
        let text = bloomai_engine::ModelOutput {
            logprobs: None,
            text: Some("unchanged".into()),
            ..output
        };
        assert_eq!(format_inference_output(&text, false).unwrap(), None);
        assert_eq!(
            format_stream_output(
                &bloomai_engine::io::OutputChunk::TextDelta("unchanged".into()),
                false
            )
            .unwrap(),
            None
        );
        assert!(Args::try_parse_from(["bloom_infer", "--json", "--bench-json"]).is_err());
    }
}
