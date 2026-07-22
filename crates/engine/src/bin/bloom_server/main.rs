//! bloom_server — OpenAI-compatible HTTP server for Bloom engine.
//!
//! Exposes /v1/models, /v1/chat/completions, /v1/completions, /health, /ready.
//! Integrates concurrent scheduling with backpressure, request cancellation,
//! graceful shutdown, tower-http middleware (tracing, CORS, request-id, timeout).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use axum::{
    extract::{DefaultBodyLimit, Request as AxumRequest, State},
    http::{header, HeaderValue},
    middleware::{self, Next},
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bloomai_core::{constants::GIB, BloomError, DeviceKind, GenerationParams, TokenSchedulingConfig};
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser};
use futures::StreamExt as _;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::task;
use tokio_stream::wrappers::{ReceiverStream, UnboundedReceiverStream};
use tokio_util::sync::CancellationToken;
use tower_http::{
    cors::{Any, CorsLayer},
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing_subscriber::EnvFilter;

use bloomai_engine::executor::batch_executor::CandleBatchExecutor;
use bloomai_engine::executor::candle::CandleEngine;
use bloomai_engine::executor::candle::ServerKvHook;
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
use bloomai_engine::scheduler::paged_cache::{PagedAttentionCache, PagedCacheConfig};
use bloomai_engine::scheduler::{BloomKvCachePool, InferenceScheduler, Request, RequestState};
use bloomai_engine::{
    speculative_mode_is_mtp, CacheMesh, CacheMeshConfig, EngineRegistry, FileSystemRemoteCache,
    InMemoryRemoteCache, InferencePipeline, InferenceRequest, KvCachePool, ModelInput, OutputChunk,
};

mod chat_template;
mod cli;
mod handlers;
mod helpers;
mod metrics;
mod ui;

use chat_template::{select_template, ChatMessage};
use cli::*;
use handlers::*;
use helpers::*;
use metrics::ServerMetrics;


// ─── Request types ──────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CompletionRequest {
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
enum ResponseFormatMode {
    Text,
    JsonObject,
    JsonSchema(serde_json::Value),
}

#[derive(Deserialize, Debug, Clone)]
pub struct EmbeddingRequest {
    pub input: serde_json::Value,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub return_documents: Option<bool>,
}

// ─── Server state ───────────────────────────────────────────────────────────

struct ServerState {
    pipeline: RwLock<Option<Arc<InferencePipeline>>>,
    model_id: RwLock<String>,
    semaphore: Arc<Semaphore>,
    ready: AtomicBool,
    load_progress: AtomicU8,
    metrics: Arc<ServerMetrics>,
    speculative_mode: String,
    memory_estimate: RwLock<Option<bloomai_engine::MemoryEstimate>>,
    kv_cache_pool: RwLock<Option<Arc<bloomai_engine::BloomKvCachePool>>>,
    cachemesh: RwLock<Option<Arc<bloomai_engine::CacheMesh>>>,
    scheduler: RwLock<Option<Arc<bloomai_engine::scheduler::InferenceScheduler>>>,
    enable_ifb: bool,
    _memory_reservation: RwLock<Option<bloomai_engine::MemoryReservation>>,
    /// Per-request cancellation tokens.
    cancel_tokens: Arc<std::sync::Mutex<HashMap<String, CancellationToken>>>,
    /// Monotonic suffix for OpenAI-compatible request IDs.
    request_counter: AtomicU64,
    /// Model family for chat template selection.
    model_family: RwLock<bloomai_core::ModelFamily>,
    /// Optional shared secret for OpenAI-compatible /v1 endpoints.
    api_key: Option<String>,
}

impl ServerState {
    async fn get_pipeline(&self) -> Result<Arc<InferencePipeline>> {
        let guard = self.pipeline.read().await;
        guard.clone().ok_or_else(|| {
            anyhow!(
                "Model is still loading (progress: {}%)",
                self.load_progress.load(Ordering::Relaxed)
            )
        })
    }
}

struct CancelTokenGuard {
    tokens: Arc<std::sync::Mutex<HashMap<String, CancellationToken>>>,
    request_id: String,
    token: CancellationToken,
    cleanup_on_drop: bool,
}

impl CancelTokenGuard {
    fn register(state: &Arc<ServerState>, request_id: String) -> Self {
        Self::register_with_tokens(Arc::clone(&state.cancel_tokens), request_id)
    }

    fn register_with_tokens(
        tokens: Arc<std::sync::Mutex<HashMap<String, CancellationToken>>>,
        request_id: String,
    ) -> Self {
        let token = CancellationToken::new();
        {
            let mut registrations = tokens.lock().unwrap();
            registrations.insert(request_id.clone(), token.clone());
        }
        Self {
            tokens,
            request_id,
            token,
            cleanup_on_drop: true,
        }
    }

    fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    fn disarm(&mut self) {
        self.cleanup_on_drop = false;
    }
}

impl Drop for CancelTokenGuard {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            if let Ok(mut tokens) = self.tokens.lock() {
                tokens.remove(&self.request_id);
            }
        }
    }
}

// ─── Standardized API error ─────────────────────────────────────────────────

/// OpenAI-compatible error types with HTTP status code mapping.
#[derive(Debug, Clone, Copy)]
pub enum ApiError {
    InvalidRequest,
    AuthenticationError,
    RateLimitExceeded,
    InternalError,
    ServiceUnavailable,
    Timeout,
    NotFound,
}

impl ApiError {
    pub fn status(&self) -> axum::http::StatusCode {
        match self {
            Self::InvalidRequest => axum::http::StatusCode::BAD_REQUEST,
            Self::AuthenticationError => axum::http::StatusCode::UNAUTHORIZED,
            Self::RateLimitExceeded => axum::http::StatusCode::TOO_MANY_REQUESTS,
            Self::InternalError => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Self::ServiceUnavailable => axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Self::Timeout => axum::http::StatusCode::REQUEST_TIMEOUT,
            Self::NotFound => axum::http::StatusCode::NOT_FOUND,
        }
    }

    pub fn error_type(&self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request_error",
            Self::AuthenticationError => "authentication_error",
            Self::RateLimitExceeded => "rate_limit_exceeded",
            Self::InternalError => "internal_error",
            Self::ServiceUnavailable => "service_unavailable",
            Self::Timeout => "timeout_error",
            Self::NotFound => "not_found_error",
        }
    }
}

// ─── Unified error helper ───────────────────────────────────────────────────

fn error_response(
    status: axum::http::StatusCode,
    error_type: &str,
    message: impl std::fmt::Display,
) -> axum::response::Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.to_string(),
                "type": error_type
            }
        })),
    )
        .into_response()
}

fn api_error(err: ApiError, message: impl std::fmt::Display) -> axum::response::Response {
    error_response(err.status(), err.error_type(), message)
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let max_len = left.len().max(right.len());
    let mut diff = left.len() ^ right.len();
    for idx in 0..max_len {
        let l = left.get(idx).copied().unwrap_or(0);
        let r = right.get(idx).copied().unwrap_or(0);
        diff |= (l ^ r) as usize;
    }
    diff == 0
}

async fn require_api_key(
    State(state): State<Arc<ServerState>>,
    req: AxumRequest,
    next: Next,
) -> Response {
    let Some(expected) = state.api_key.as_deref() else {
        return next.run(req).await;
    };

    let bearer = format!("Bearer {}", expected);
    let authorization_ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(|value| constant_time_eq(value.as_bytes(), bearer.as_bytes()))
        .unwrap_or(false);
    let x_api_key_ok = req
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(|value| constant_time_eq(value.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);

    if authorization_ok || x_api_key_ok {
        next.run(req).await
    } else {
        api_error(
            ApiError::AuthenticationError,
            "Missing or invalid API key for /v1 endpoint.",
        )
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
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
    apply_config(&mut args, &matches, &config.server);

    let model_path = args.model.as_ref().ok_or_else(|| {
        anyhow!(
            "model path is required; pass --model or set server.model in {}",
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
        std::env::set_var("BLOOM_DTYPE", dt);
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

    // --- Load manifest first to assist in engine auto-selection if needed ---
    let manifest = bloomai_engine::load_manifest(model_path)?;

    // Initialize engine registry
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
    registry.register("llamacpp", Box::new(LlamaCppEngine));
    registry.register("longcat", Box::new(LongCatImageEditEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("wan", Box::new(WanEngine));

    let backend_name = select_backend_name(&args.backend, &args.speculative, &manifest);

    // Validate the backend_name early so we fail before loading the model.
    registry.get(&backend_name).map_err(|e| {
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

    let state = Arc::new(ServerState {
        pipeline: RwLock::new(None),
        model_id: RwLock::new(String::new()),
        semaphore: Arc::new(Semaphore::new(args.max_concurrent)),
        ready: AtomicBool::new(false),
        load_progress: AtomicU8::new(0),
        metrics: Arc::new(ServerMetrics::new()),
        speculative_mode: args.speculative.clone(),
        memory_estimate: RwLock::new(None),
        kv_cache_pool: RwLock::new(None),
        cachemesh: RwLock::new(None),
        scheduler: RwLock::new(None),
        enable_ifb: args.enable_ifb,
        _memory_reservation: RwLock::new(None),
        cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
        request_counter: AtomicU64::new(0),
        model_family: RwLock::new(manifest.family.clone()),
        api_key: args.api_key.clone().filter(|value| !value.is_empty()),
    });

    let state_clone = Arc::clone(&state);
    let args_clone = args.clone();
    let manifest_clone = manifest.clone();
    let model_path_clone = model_path.clone();
    let backend_name_clone = backend_name.clone();

    tokio::spawn(async move {
        tracing::info!("Starting background model load...");
        state_clone.load_progress.store(5, Ordering::Relaxed);

        let memory_context_size = args_clone
            .context_size
            .saturating_mul(args_clone.max_concurrent.max(1));
        let memory_estimate = bloomai_engine::estimate_memory(&manifest_clone, memory_context_size);
        {
            let mut guard = state_clone.memory_estimate.write().await;
            *guard = Some(memory_estimate.clone());
        }

        state_clone.load_progress.store(15, Ordering::Relaxed);
        let memory_plan_res = bloomai_engine::plan_memory_preallocation(
            memory_estimate.clone(),
            bloomai_engine::MemoryPreallocationConfig {
                enabled: !args_clone.disable_memory_prealloc,
                memory_utilization: args_clone.memory_utilization,
                reserve_memory_bytes: args_clone.reserve_memory_bytes,
            },
        );
        let memory_plan = match memory_plan_res {
            Ok(plan) => plan,
            Err(e) => {
                tracing::error!("Memory preallocation plan failed: {}", e);
                return;
            }
        };

        state_clone.load_progress.store(25, Ordering::Relaxed);
        let startup_reservation = match bloomai_engine::reserve_memory_for_plan(&memory_plan) {
            Ok(res) => res,
            Err(e) => {
                tracing::error!("Memory reservation failed: {}", e);
                return;
            }
        };
        drop(startup_reservation); // prealloc check

        state_clone.load_progress.store(35, Ordering::Relaxed);
        // Load model pipeline in spawn_blocking
        let pipeline_res = task::spawn_blocking(move || {
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
            registry.register("llamacpp", Box::new(LlamaCppEngine));
            registry.register("longcat", Box::new(LongCatImageEditEngine));
            #[cfg(feature = "candle-engine")]
            registry.register("wan", Box::new(WanEngine));

            let engine = registry.get(&backend_name_clone).map_err(|e| {
                anyhow!(
                    "{}. Supported engines are: candle, openvino, funasr, qwen3_vl, longcat, intel-npu, npu-tts, onnxruntime, coreml, mlx, vulkan, llamacpp, wan.",
                    e
                )
            })?;

            InferencePipeline::load_standalone_with_context(
                engine,
                device_kind,
                &model_path_clone,
                args_clone.context_size,
            )
        }).await;

        let pipeline = match pipeline_res {
            Ok(Ok(p)) => Arc::new(p),
            Ok(Err(e)) => {
                tracing::error!("Failed to load model pipeline: {}", e);
                return;
            }
            Err(e) => {
                tracing::error!("Model pipeline load thread panicked: {}", e);
                return;
            }
        };

        let model_id = pipeline.metadata().id.clone();
        tracing::info!("Loaded model pipeline: {}", model_id);
        {
            let mut guard = state_clone.pipeline.write().await;
            *guard = Some(Arc::clone(&pipeline));
        }
        {
            let mut guard = state_clone.model_id.write().await;
            *guard = model_id.clone();
        }

        let actual_context_size = pipeline.context_size();
        let actual_device = pipeline.device();

        // Recalculate memory estimate based on actual degraded context size
        let memory_context_size =
            actual_context_size.saturating_mul(args_clone.max_concurrent.max(1));
        let memory_estimate = bloomai_engine::estimate_memory(&manifest_clone, memory_context_size);
        {
            let mut guard = state_clone.memory_estimate.write().await;
            *guard = Some(memory_estimate.clone());
        }

        state_clone.load_progress.store(65, Ordering::Relaxed);

        let mut kv_cache_pool_val = None;
        let mut cachemesh_val = None;
        let mut scheduler_val = None;
        let mut memory_reservation = None;

        if args_clone.enable_ifb {
            let block_size = 16;
            let total_blocks = div_ceil_usize(memory_context_size, block_size).max(1);
            let num_layers = manifest_param_usize(
                &manifest_clone,
                &["num_hidden_layers", "num_layers", "block_count"],
                28,
            );
            let num_kv_heads = manifest_param_usize(
                &manifest_clone,
                &[
                    "num_key_value_heads",
                    "num_kv_heads",
                    "attention_head_count_kv",
                ],
                8,
            );
            let head_dim = manifest_param_usize(&manifest_clone, &["head_dim"], 128);
            let kv_dim = num_kv_heads.saturating_mul(head_dim).max(1);
            let long_context_policy = match build_long_context_policy(&args_clone) {
                Ok(policy) => policy,
                Err(e) => {
                    tracing::error!("Failed to build long context policy: {}", e);
                    return;
                }
            };

            if !args_clone.disable_memory_prealloc && memory_estimate.kv_cache_bytes > 0 {
                let reservation = match bloomai_engine::MemoryReservation::reserve(
                    memory_estimate.kv_cache_bytes,
                ) {
                    Ok(res) => res,
                    Err(e) => {
                        tracing::error!("Failed to reserve KV cache memory: {}", e);
                        return;
                    }
                };
                memory_reservation = Some(reservation);
            }
            let kv_pool = Arc::new(BloomKvCachePool::new(block_size, total_blocks));
            kv_cache_pool_val = Some(Arc::clone(&kv_pool));

            let device = if actual_device == DeviceKind::Cpu {
                candle_core::Device::Cpu
            } else {
                #[cfg(feature = "cuda")]
                {
                    candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
                }
                #[cfg(feature = "metal")]
                {
                    candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
                }
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                {
                    candle_core::Device::Cpu
                }
            };

            let pipeline_clone = Arc::clone(&pipeline);
            let request_models = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
                usize,
                Arc<std::sync::Mutex<bloomai_engine::executor::candle::QwenModelWrapper>>,
            >::new()));

            let request_models_clone = Arc::clone(&request_models);
            kv_pool.set_on_free(move |handle| {
                if let Ok(mut models) = request_models_clone.lock() {
                    models.remove(&handle);
                    tracing::info!("Cleaned up request model wrapper for handle {}", handle);
                }
            });

            let request_models_clone = Arc::clone(&request_models);
            let forward_fn = Box::new(
                move |input_ids: &candle_core::Tensor,
                      start_pos: usize,
                      kv_handle: Option<usize>|
                      -> Result<candle_core::Tensor> {
                    let handle = kv_handle.unwrap_or(0);
                    let model_arc = {
                        let mut models = request_models_clone.lock().unwrap();
                        use std::collections::hash_map::Entry;
                        Arc::clone(match models.entry(handle) {
                            Entry::Occupied(e) => e.into_mut(),
                            Entry::Vacant(e) => {
                                let wrapper = pipeline_clone.model().create_wrapper()?;
                                let qwen_model = *wrapper
                                    .downcast::<bloomai_engine::executor::candle::QwenModelWrapper>(
                                    )
                                    .map_err(|_| {
                                        BloomError::Engine("Failed to downcast model wrapper".into())
                                    })?;
                                e.insert(Arc::new(std::sync::Mutex::new(qwen_model)))
                            }
                        })
                    };
                    let mut model = model_arc.lock().unwrap();
                    model.forward(input_ids, start_pos)
                },
            );
            let request_models_clone2 = Arc::clone(&request_models);
            let pipeline_clone2 = Arc::clone(&pipeline);
            let forward_batch_fn = Box::new(
                move |input_ids: &candle_core::Tensor,
                      start_positions: &[usize],
                      kv_handles: &[usize],
                      cu_seqlens: &[usize]|
                      -> Result<candle_core::Tensor> {
                    let batch_size = kv_handles.len();
                    if batch_size == 0 {
                        return Ok(candle_core::Tensor::zeros(
                            (0, 0),
                            candle_core::DType::F32,
                            input_ids.device(),
                        )?);
                    }
                    let mut models_to_run = Vec::with_capacity(batch_size);
                    {
                        let mut models = request_models_clone2.lock().unwrap();
                        use std::collections::hash_map::Entry;
                        for &handle in kv_handles {
                            let entry = match models.entry(handle) {
                                Entry::Occupied(e) => e.into_mut(),
                                Entry::Vacant(e) => {
                                    let wrapper = pipeline_clone2.model().create_wrapper()?;
                                    let qwen_model = *wrapper
                                        .downcast::<
                                            bloomai_engine::executor::candle::QwenModelWrapper,
                                        >()
                                        .map_err(|_| {
                                            BloomError::Engine("Failed to downcast model wrapper".into())
                                        })?;
                                    e.insert(Arc::new(std::sync::Mutex::new(qwen_model)))
                                }
                            };
                            models_to_run.push(Arc::clone(entry));
                        }
                    }
                    let mut logits_list = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let start = cu_seqlens[i];
                        let end = cu_seqlens[i + 1];
                        let seq_len = end - start;
                        let start_pos = start_positions.get(i).copied().unwrap_or(0);
                        let sliced_input = input_ids.narrow(0, start, seq_len)?;
                        let req_input = sliced_input.unsqueeze(0)?;
                        let mut model = models_to_run[i].lock().unwrap();
                        let res = model.forward(&req_input, start_pos)?;
                        let squeezed = res.squeeze(0)?;
                        logits_list.push(squeezed);
                    }
                    candle_core::Tensor::cat(&logits_list, 0).map_err(Into::into)
                },
            );

            let cachemesh = if args_clone.enable_cachemesh {
                let config = CacheMeshConfig {
                    enabled: true,
                    namespace: model_id.clone(),
                    l2_capacity_bytes: args_clone.cachemesh_l2_capacity_bytes,
                    l3_enabled: args_clone.enable_cachemesh_l3,
                    write_through_l3: args_clone.cachemesh_write_through_l3,
                };
                let mesh_res = if args_clone.enable_cachemesh_l3 {
                    let remote: Arc<dyn bloomai_engine::RemoteCacheBackend> = if let Some(path) =
                        &args_clone.cachemesh_l3_path
                    {
                        match FileSystemRemoteCache::new(path) {
                            Ok(r) => Arc::new(r),
                            Err(e) => {
                                tracing::error!("Failed to create FileSystemRemoteCache: {}", e);
                                return;
                            }
                        }
                    } else {
                        Arc::new(InMemoryRemoteCache::new())
                    };
                    CacheMesh::with_remote(config, remote)
                } else {
                    CacheMesh::new(config)
                };
                let mesh = Arc::new(mesh_res);
                cachemesh_val = Some(Arc::clone(&mesh));
                Some(mesh)
            } else {
                None
            };

            state_clone.load_progress.store(85, Ordering::Relaxed);
            let paged_cache = Arc::new(PagedAttentionCache::from_pool_and_cachemesh(
                Arc::clone(&kv_pool),
                PagedCacheConfig {
                    block_size,
                    total_blocks,
                    num_layers,
                    kv_dim,
                    kv_dtype: bloomai_engine::core::quantization::KvCacheDtype::F16,
                    long_context_policy,
                },
                cachemesh,
            ));
            let executor = Arc::new({
                let model = pipeline.model();
                let vocab_strings = model.vocab_strings().to_vec();
                let eos_token_ids = model.eos_token_ids().to_vec();
                let tokenizer = model.tokenizer().cloned();
                let base = CandleBatchExecutor::new(forward_fn, device, 4, 32)
                    .with_cache(Arc::clone(&paged_cache))
                    .with_vocab_and_tokenizer(vocab_strings, eos_token_ids, tokenizer)
                    .with_forward_batch_fn(forward_batch_fn);
                if model.supports_paged_kv() {
                    tracing::info!(
                        "Model supports paged KV; attaching PerRequestKvHook to batch executor \
                         (paged KV cache will hold real attention KV, enabling cross-request reuse, \
                         prefix caching and CacheMesh L2/L3 restores)"
                    );
                    let hook = Arc::new(ServerKvHook::new(
                        Arc::clone(&request_models),
                        num_layers,
                        num_kv_heads,
                        head_dim,
                    ));
                    base.with_kv_hook(hook as Arc<dyn bloomai_engine::scheduler::kv_hook::KvHook>)
                } else {
                    tracing::info!(
                        "Model does not support paged KV hook; paged KV cache will operate in \
                         metadata-only mode (block allocation/eviction/metrics still work, but \
                         `paged_attention_forward` has no real KV to gather)"
                    );
                    base
                }
            });

            let mut scheduling_config = TokenSchedulingConfig {
                max_total_tokens_per_step: args_clone.max_num_tokens,
                ..Default::default()
            };
            scheduling_config.chunked_prefill.enabled = args_clone.enable_chunked_prefill;
            scheduling_config.chunked_prefill.chunk_size = args_clone.prefill_chunk_size.max(1);

            let scheduler = Arc::new(InferenceScheduler::with_config(
                executor,
                Arc::clone(&kv_pool) as Arc<dyn KvCachePool>,
                scheduling_config,
            ));
            scheduler_val = Some(Arc::clone(&scheduler));

            // Start background scheduler worker loop
            let scheduler_clone = Arc::clone(&scheduler);
            tokio::spawn(async move {
                tracing::info!("Starting background continuous batching scheduler loop...");
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                    if let Err(e) = scheduler_clone.step() {
                        tracing::error!("Scheduler step error: {}", e);
                    }
                }
            });
        }

        state_clone.load_progress.store(95, Ordering::Relaxed);
        {
            let mut guard = state_clone.kv_cache_pool.write().await;
            *guard = kv_cache_pool_val;
        }
        {
            let mut guard = state_clone.cachemesh.write().await;
            *guard = cachemesh_val;
        }
        {
            let mut guard = state_clone.scheduler.write().await;
            *guard = scheduler_val;
        }
        {
            let mut guard = state_clone._memory_reservation.write().await;
            *guard = memory_reservation;
        }

        state_clone.load_progress.store(100, Ordering::Relaxed);
        state_clone.ready.store(true, Ordering::Relaxed);
        tracing::info!("Background model load completed successfully!");
    });

    // Build middleware stack
    let cors = if args.cors_allow_origin.trim() == "*" {
        CorsLayer::new().allow_origin(Any)
    } else {
        CorsLayer::new().allow_origin(HeaderValue::from_str(args.cors_allow_origin.trim())?)
    }
    .allow_methods(Any)
    .allow_headers(Any);

    let v1_routes = Router::new()
        .route("/observability", get(handle_observability))
        .route("/models", get(handle_models))
        .route("/chat/completions", post(handle_chat_completions))
        .route("/completions", post(handle_completions))
        .route("/embeddings", post(handle_embeddings))
        .route("/rerank", post(handle_rerank))
        .route("/multimodal/stream", post(handle_multimodal_stream))
        .route("/world/step", post(handle_world_step))
        .route("/kv-cache-stats", get(handle_kv_cache_stats))
        .route("/cancel/{request_id}", post(handle_cancel))
        .route("/backends", get(handle_backends))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));

    let mut app = Router::new()
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .route("/metrics", get(handle_metrics))
        .nest("/v1", v1_routes)
        .with_state(state);

    // Optionally serve the embedded UI at `/` (serve-ui feature).
    if let Some(ui) = ui::ui_router() {
        app = app.merge(ui);
    }

    if args.timeout > 0 {
        app = app.layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(args.timeout),
        ));
    }

    app = app
        .layer(DefaultBodyLimit::max(args.max_body_bytes))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .layer(PropagateRequestIdLayer::x_request_id());

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    // Check binding security
    let api_key_set = args
        .api_key
        .as_ref()
        .filter(|value| !value.is_empty())
        .is_some();
    let is_loopback = addr.ip().is_loopback();
    if !is_loopback && !api_key_set {
        let strict_security = std::env::var("BLOOM_STRICT_SECURITY")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let msg = format!(
            "Security Alert: Server is binding to a non-loopback address ({}) without BLOOM_API_KEY or --api-key. \
             This exposes your endpoints to the network without authentication.",
            addr.ip()
        );
        if strict_security {
            return Err(anyhow!("{}", msg));
        } else {
            tracing::warn!("=====================================================================");
            tracing::warn!("{}", msg);
            tracing::warn!("Please set BLOOM_API_KEY or pass --api-key to secure the server.");
            tracing::warn!("To fail-fast on this check, run with BLOOM_STRICT_SECURITY=1.");
            tracing::warn!("=====================================================================");
        }
    }

    tracing::info!("Bloom OpenAI-compatible server running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shut down gracefully.");
    Ok(())
}

// ─── Graceful shutdown ─────────────────────────────────────────────────────

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
    tracing::info!("Received shutdown signal, draining in-flight requests...");
}

// ─── Health / readiness ─────────────────────────────────────────────────────




#[cfg(test)]
mod tests {
    use super::*;
    use bloomai_engine::scheduler::kv_hook::KvHook;

    #[test]
    fn select_backend_auto_routes_mtp_to_llamacpp() {
        let manifest = bloomai_core::ModelManifest::default();
        assert_eq!(
            select_backend_name("candle", "draft-mtp", &manifest),
            "llamacpp"
        );
    }

    #[test]
    fn select_backend_keeps_explicit_backend() {
        let manifest = bloomai_core::ModelManifest::default();
        assert_eq!(
            select_backend_name("openvino", "mtp", &manifest),
            "openvino"
        );
    }

    #[test]
    fn select_backend_preserves_existing_auto_routes() {
        let mut manifest = bloomai_core::ModelManifest {
            family: bloomai_core::ModelFamily::FunAsr,
            ..Default::default()
        };
        assert_eq!(select_backend_name("candle", "none", &manifest), "funasr");

        manifest.family = bloomai_core::ModelFamily::Qwen;
        manifest.id = "qwen3-vl".to_string();
        assert_eq!(select_backend_name("candle", "none", &manifest), "qwen3_vl");
    }

    #[test]
    fn normalize_embedding_input_accepts_string_or_string_array() {
        assert_eq!(
            normalize_embedding_input(&json!("hello")).unwrap(),
            vec!["hello".to_string()]
        );
        assert_eq!(
            normalize_embedding_input(&json!(["hello", "world"])).unwrap(),
            vec!["hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn normalize_embedding_input_rejects_empty_or_token_arrays() {
        assert!(normalize_embedding_input(&json!("")).is_err());
        assert!(normalize_embedding_input(&json!([])).is_err());
        assert!(normalize_embedding_input(&json!([[1, 2, 3]])).is_err());
    }

    #[test]
    fn cosine_similarity_scores_identical_vectors_highest() {
        let same = cosine_similarity(&[1.0, 0.0, 1.0], &[1.0, 0.0, 1.0]);
        let orthogonal = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(same > 0.999);
        assert_eq!(orthogonal, 0.0);
    }

    #[test]
    fn chat_prompt_applies_qwen_template_to_single_user_message() {
        let messages = vec![ChatCompletionMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        assert_eq!(
            chat_prompt(&messages, &bloomai_core::ModelFamily::Qwen),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn response_format_accepts_json_object_and_schema() {
        let json_object = ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        };
        assert_eq!(
            response_format_mode(Some(&json_object)).unwrap(),
            ResponseFormatMode::JsonObject
        );

        let json_schema = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(json!({ "name": "x", "schema": { "type": "object" } })),
        };
        assert!(matches!(
            response_format_mode(Some(&json_schema)).unwrap(),
            ResponseFormatMode::JsonSchema(_)
        ));
    }

    #[test]
    fn structured_output_validation_requires_json_object() {
        assert!(validate_structured_output("plain text", &ResponseFormatMode::JsonObject).is_err());
        assert!(validate_structured_output("[1,2,3]", &ResponseFormatMode::JsonObject).is_err());
        assert!(
            validate_structured_output("{\"ok\":true}", &ResponseFormatMode::JsonObject).is_ok()
        );
    }

    #[test]
    fn json_schema_validation_checks_required_properties_and_extra_fields() {
        let schema = json!({
            "type": "object",
            "required": ["name", "score"],
            "properties": {
                "name": { "type": "string" },
                "score": { "type": "integer" },
                "tags": { "type": "array", "items": { "type": "string" } }
            },
            "additionalProperties": false
        });
        let mode = ResponseFormatMode::JsonSchema(schema);

        assert!(validate_structured_output(
            "{\"name\":\"bloom\",\"score\":3,\"tags\":[\"edge\"]}",
            &mode
        )
        .is_ok());
        assert!(validate_structured_output("{\"name\":\"bloom\"}", &mode).is_err());
        assert!(validate_structured_output(
            "{\"name\":\"bloom\",\"score\":3,\"extra\":true}",
            &mode
        )
        .is_err());
    }

    #[test]
    fn request_ids_are_unique_within_same_process() {
        let counter = AtomicU64::new(0);
        let first = next_request_id_from_counter(&counter, "chatcmpl");
        let second = next_request_id_from_counter(&counter, "chatcmpl");
        assert_ne!(first, second);
        assert!(first.ends_with("-1"));
        assert!(second.ends_with("-2"));
    }

    #[test]
    fn cancel_token_guard_cleans_up_unless_disarmed() {
        let tokens = Arc::new(std::sync::Mutex::new(HashMap::new()));
        {
            let guard =
                CancelTokenGuard::register_with_tokens(Arc::clone(&tokens), "req-1".to_string());
            assert!(!guard.token().is_cancelled());
            assert!(tokens.lock().unwrap().contains_key("req-1"));
        }
        assert!(!tokens.lock().unwrap().contains_key("req-1"));

        {
            let mut guard =
                CancelTokenGuard::register_with_tokens(Arc::clone(&tokens), "req-2".to_string());
            guard.disarm();
        }
        assert!(tokens.lock().unwrap().contains_key("req-2"));
    }

    #[test]
    fn constant_time_eq_matches_byte_equality() {
        assert!(constant_time_eq(b"Bearer secret", b"Bearer secret"));
        assert!(!constant_time_eq(b"Bearer secret", b"Bearer public"));
        assert!(!constant_time_eq(b"short", b"shorter"));
        assert!(!constant_time_eq(b"", b"non-empty"));
    }

    #[tokio::test]
    async fn test_streaming_response_format_validation() {
        use futures::StreamExt;

        let response_format = ResponseFormatMode::JsonObject;

        // Case 1: Valid JSON output
        {
            let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
            let accumulated_text_for_stream = Arc::clone(&accumulated_text);
            let accumulated_text_for_final = Arc::clone(&accumulated_text);
            let response_format_for_final = response_format.clone();

            let tokens_stream = futures::stream::iter(vec![
                "{\"".to_string(),
                "ok".to_string(),
                "\":".to_string(),
                "true".to_string(),
                "}".to_string(),
            ]);

            let sse_stream = tokens_stream.map(move |token| {
                {
                    let mut acc = accumulated_text_for_stream.lock().unwrap();
                    acc.push_str(&token);
                }
                token
            });

            let stop_chunk = "STOP_OK".to_string();

            let final_stream = sse_stream.chain(futures::stream::once(async move {
                let text = {
                    let acc = accumulated_text_for_final.lock().unwrap();
                    acc.clone()
                };
                if let Err(message) = validate_structured_output(&text, &response_format_for_final)
                {
                    format!("ERROR: {}", message)
                } else {
                    stop_chunk
                }
            }));

            let results: Vec<String> = final_stream.collect().await;
            assert_eq!(results.len(), 6);
            assert_eq!(results[5], "STOP_OK");
        }

        // Case 2: Invalid JSON output
        {
            let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
            let accumulated_text_for_stream = Arc::clone(&accumulated_text);
            let accumulated_text_for_final = Arc::clone(&accumulated_text);
            let response_format_for_final = response_format.clone();

            let tokens_stream =
                futures::stream::iter(vec!["invalid".to_string(), " json".to_string()]);

            let sse_stream = tokens_stream.map(move |token| {
                {
                    let mut acc = accumulated_text_for_stream.lock().unwrap();
                    acc.push_str(&token);
                }
                token
            });

            let stop_chunk = "STOP_OK".to_string();

            let final_stream = sse_stream.chain(futures::stream::once(async move {
                let text = {
                    let acc = accumulated_text_for_final.lock().unwrap();
                    acc.clone()
                };
                if let Err(message) = validate_structured_output(&text, &response_format_for_final)
                {
                    format!("ERROR: {}", message)
                } else {
                    stop_chunk
                }
            }));

            let results: Vec<String> = final_stream.collect().await;
            assert_eq!(results.len(), 3);
            assert!(results[2].starts_with("ERROR:"));
        }
    }

    #[test]
    fn test_multimodal_output_chunk_serialization() {
        let chunk1 = OutputChunk::AsrPartial {
            text: "hello".to_string(),
            tokens: vec![1, 2, 3],
        };
        let serialized1 = serde_json::to_string(&chunk1).unwrap();
        assert!(serialized1.contains("AsrPartial"));
        assert!(serialized1.contains("hello"));

        let chunk2 = OutputChunk::TtsAudioChunk {
            samples: vec![0.1, 0.2],
            sample_rate: 24000,
            is_final: true,
        };
        let serialized2 = serde_json::to_string(&chunk2).unwrap();
        assert!(serialized2.contains("TtsAudioChunk"));
        assert!(serialized2.contains("24000"));

        let chunk3 = OutputChunk::VlmToken {
            text: "cat".to_string(),
            bounding_box: Some(vec![0.1, 0.2, 0.3, 0.4]),
        };
        let serialized3 = serde_json::to_string(&chunk3).unwrap();
        assert!(serialized3.contains("VlmToken"));
        assert!(serialized3.contains("cat"));
    }

    #[test]
    fn test_server_kv_hook_lookup_error() {
        let request_models = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let hook = ServerKvHook::new(Arc::clone(&request_models), 28, 8, 128);
        assert_eq!(hook.num_layers(), 28);
        assert_eq!(hook.kv_dim(), 1024);
        let res = hook.extract_kv(999, 0, 0, 10);
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .to_string()
            .contains("No model wrapper found for handle 999"));
    }

    #[test]
    fn test_world_step_request_deserialization() {
        let json_data = json!({
            "observations": [
                {
                    "Scalar": {
                        "name": "sensor_x",
                        "value": 0.5
                    }
                }
            ],
            "horizon": 2,
            "stream": true,
            "cache_config": {
                "max_bytes": 1024,
                "max_entries": 5,
                "default_ttl_ms": 1000,
                "auto_compress": true,
                "compress_after_ms": 500
            }
        });

        let req: WorldStepRequest = serde_json::from_value(json_data).unwrap();
        assert_eq!(req.horizon, 2);
        assert!(req.stream);
        let cache_config = req.cache_config.unwrap();
        assert_eq!(cache_config.max_bytes, Some(1024));
        assert_eq!(cache_config.max_entries, Some(5));
        assert_eq!(cache_config.default_ttl_ms, Some(1000));
        assert_eq!(cache_config.auto_compress, Some(true));
        assert_eq!(cache_config.compress_after_ms, Some(500));
    }
}
