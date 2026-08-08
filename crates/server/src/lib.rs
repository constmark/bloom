//! Bloom's HTTP application layer.
//!
//! Exposes bounded `/v1` and `/api` compatibility surfaces, health, readiness,
//! and local model-management routes. The crate depends on `bloomai-engine`,
//! while the engine has no dependency on this transport layer or its optional
//! browser presentation.
//! Integrates concurrent scheduling with backpressure, request cancellation,
//! graceful shutdown, tower-http middleware (tracing, CORS, request-id, timeout).

#![cfg_attr(not(test), warn(clippy::unwrap_used))]

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, Request as AxumRequest, State},
    http::{header, HeaderValue},
    middleware::{self, Next},
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use bloomai_core::{
    constants::{GIB, MIB},
    BloomError, DeviceKind, GenerationParams, TokenSchedulingConfig,
};
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, ValueEnum};
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::future::IntoFuture as _;
use tokio::sync::{mpsc, watch, Mutex, OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::task;
use tokio_stream::wrappers::{ReceiverStream, UnboundedReceiverStream};
use tokio_util::sync::CancellationToken;
use tower_http::{
    classify::ServerErrorsFailureClass,
    cors::{Any, CorsLayer},
    request_id::{MakeRequestId as _, MakeRequestUuid, RequestId},
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
    speculative_mode_is_mtp, CacheMesh, CacheMeshConfig, DataBlock, EngineRegistry,
    FileSystemRemoteCache, InMemoryRemoteCache, InferenceParams, InferencePipeline,
    InferenceRequest, KvCachePool, ModelInput, OutputChunk,
};

mod catalog_lock;
mod chat_template;
mod cli;
mod doctor;
mod embedding;
mod handlers;
mod helpers;
mod metrics;
mod model_download;
mod model_import;
mod model_index;
mod model_index_state;
mod model_integrity;
mod model_inventory;
mod model_license;
mod model_manager;
mod model_package;
mod model_preflight;
mod model_provenance;
mod model_storage;
mod model_upgrade;
mod ollama;
mod readiness;
mod response_store;
mod tool_calling;
mod ui;

use catalog_lock::ModelCatalogLease;
use chat_template::{select_template_for_metadata, ChatMessage};
use cli::*;
use doctor::{inspect_server, validate_server_arguments};
use embedding::*;
use handlers::*;
use helpers::*;
use metrics::ServerMetrics;
use model_download::{
    ModelDownloadInspectError, ModelDownloadManager, ModelDownloadRequest,
    ModelDownloadSourceRequest, ModelDownloadStartError, ModelPackageDownloadFile,
    ModelPackageDownloadRequest,
};
use model_import::{ModelImportError, ModelImportManager, ModelImportRequest};
use model_index::{ModelIndexError, ModelIndexManager, ModelIndexManagerConfig};
use model_integrity::{ModelIntegrityError, ModelIntegrityManager};
use model_license::ModelLicensePolicy;
use model_manager::ModelCatalog;
use model_preflight::{ModelPreflightConfig, ModelPreflightManager};
use model_storage::ModelStorageManager;
use ollama::*;
use readiness::*;
use response_store::ResponseStore;
use tool_calling::*;

const MODEL_CATALOG_CACHE_TTL: Duration = Duration::from_secs(10);
const MODEL_INDEX_STATE_DIRECTORY: &str = "model-index-watermarks";
const HTTP_REQUEST_ID_HEADER: &str = "x-request-id";
const MAX_HTTP_REQUEST_ID_CHARS: usize = 128;
const DEFAULT_CAPACITY_RETRY_AFTER_SECONDS: &str = "1";
const DEFAULT_BEARER_AUTHENTICATION_CHALLENGE: &str = r#"Bearer realm="Bloom""#;
const DEFAULT_BROWSER_ORIGIN_POLICY: &str = "same-origin";
const MAX_BROWSER_ORIGIN_CHARS: usize = 512;
const MAX_CHAT_REQUEST_MESSAGES: usize = 2_048;
const MAX_CHAT_CONTENT_PARTS: usize = 256;
const MAX_CHAT_CONTENT_BYTES: usize = 768 * 1024;
const MAX_CHAT_USER_MESSAGE_CHARS: usize = 262_144;
const MAX_CHAT_SYSTEM_MESSAGE_CHARS: usize = 65_536;
const MAX_RESPONSES_ADAPTER_BODY_BYTES: usize = 16 * MIB as usize;
const MAX_RESPONSES_STREAM_FRAME_BYTES: usize = MIB as usize;
const MAX_RESPONSES_STREAM_OUTPUT_BYTES: usize = 16 * MIB as usize;
const MAX_RESPONSES_STREAM_EVENTS: u64 = 131_072;
const MAX_RESPONSES_STREAM_ERROR_BYTES: usize = 4 * 1024;
const MAX_GENERATED_TOKENS: usize = 32_768;
const MAX_STOP_SEQUENCES: usize = 4;
const MAX_STOP_SEQUENCE_CHARS: usize = 1_024;
const MAX_STOP_SEQUENCES_BYTES: usize = 16 * 1_024;
const MAX_COMPLETION_PROMPT_CHARS: usize = 262_144;
const MAX_COMPLETION_PROMPT_BYTES: usize = 768 * 1024;
const MAX_EMBEDDING_INPUTS: usize = 256;
const MAX_EMBEDDING_INPUT_CHARS: usize = 262_144;
const MAX_EMBEDDING_CONTENT_BYTES: usize = 768 * 1024;
const MAX_RERANK_DOCUMENTS: usize = 256;
const MAX_RERANK_QUERY_CHARS: usize = 65_536;
const MAX_RERANK_DOCUMENT_CHARS: usize = 262_144;
const MAX_RERANK_CONTENT_BYTES: usize = 768 * 1024;
const MAX_MULTIMODAL_BLOCKS: usize = 3;
const MAX_MULTIMODAL_TEXT_CHARS: usize = 262_144;
const MAX_MULTIMODAL_TEXT_BYTES: usize = 768 * 1024;
const MAX_MULTIMODAL_IMAGE_BYTES: usize = 10 * MIB as usize;
const MIN_MULTIMODAL_AUDIO_SAMPLE_RATE: u32 = 8_000;
const MAX_MULTIMODAL_AUDIO_SAMPLE_RATE: u32 = 48_000;
const MAX_MULTIMODAL_AUDIO_SECONDS: usize = 600;
const MAX_MULTIMODAL_AUDIO_SAMPLES: usize =
    MIN_MULTIMODAL_AUDIO_SAMPLE_RATE as usize * MAX_MULTIMODAL_AUDIO_SECONDS;
const MAX_SHUTDOWN_TIMEOUT_SECONDS: u64 = 3_600;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ValidatedBrowserOrigin {
    serialized: String,
    header: HeaderValue,
    scheme: String,
    authority: String,
    host: String,
}

impl ValidatedBrowserOrigin {
    fn parse(value: &str) -> std::result::Result<Self, String> {
        let value = value.trim();
        if value.is_empty() || value.len() > MAX_BROWSER_ORIGIN_CHARS {
            return Err(format!(
                "browser origin must contain between 1 and {MAX_BROWSER_ORIGIN_CHARS} characters"
            ));
        }
        if value.eq_ignore_ascii_case("null") {
            return Err("opaque browser origins are not allowed".to_string());
        }
        let uri = value
            .parse::<axum::http::Uri>()
            .map_err(|_| "browser origin must be an absolute HTTP(S) origin".to_string())?;
        let scheme = uri
            .scheme_str()
            .filter(|scheme| matches!(*scheme, "http" | "https"))
            .ok_or_else(|| "browser origin scheme must be http or https".to_string())?
            .to_ascii_lowercase();
        let authority = uri
            .authority()
            .ok_or_else(|| "browser origin must include a host".to_string())?;
        if authority.as_str().contains('@') {
            return Err("browser origin must not include user information".to_string());
        }
        if uri
            .path_and_query()
            .is_some_and(|path| path.as_str() != "/")
        {
            return Err("browser origin must not include a path, query, or fragment".to_string());
        }
        let authority = authority.as_str().to_ascii_lowercase();
        let host = uri
            .authority()
            .map(|value| value.host().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "browser origin must include a host".to_string())?;
        let serialized = format!("{scheme}://{authority}");
        let header = HeaderValue::from_str(&serialized)
            .map_err(|_| "browser origin is not a valid HTTP header value".to_string())?;
        Ok(Self {
            serialized,
            header,
            scheme,
            authority,
            host,
        })
    }

    fn has_loopback_host(&self) -> bool {
        let unbracketed = self.host.trim_start_matches('[').trim_end_matches(']');
        unbracketed.eq_ignore_ascii_case("localhost")
            || unbracketed
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BrowserOriginPolicy {
    SameOrigin,
    Exact(ValidatedBrowserOrigin),
    Any,
}

fn parse_browser_origin_policy(value: &str) -> std::result::Result<BrowserOriginPolicy, String> {
    let value = value.trim();
    if value.eq_ignore_ascii_case(DEFAULT_BROWSER_ORIGIN_POLICY) {
        Ok(BrowserOriginPolicy::SameOrigin)
    } else if value == "*" {
        Ok(BrowserOriginPolicy::Any)
    } else {
        ValidatedBrowserOrigin::parse(value).map(BrowserOriginPolicy::Exact)
    }
}

#[derive(Clone, Debug)]
struct BrowserOriginGuard {
    policy: BrowserOriginPolicy,
    loopback_listener: bool,
}

impl BrowserOriginGuard {
    fn permits(&self, request: &AxumRequest) -> bool {
        let mut values = request.headers().get_all(header::ORIGIN).iter();
        let Some(value) = values.next() else {
            return true;
        };
        if values.next().is_some() {
            return false;
        }
        let Ok(value) = value.to_str() else {
            return false;
        };
        let Ok(origin) = ValidatedBrowserOrigin::parse(value) else {
            return false;
        };
        if self.policy == BrowserOriginPolicy::Any {
            return true;
        }
        if let BrowserOriginPolicy::Exact(allowed) = &self.policy {
            if origin.serialized == allowed.serialized {
                return true;
            }
        }
        if origin.scheme != "http" {
            return false;
        }
        if self.loopback_listener && !origin.has_loopback_host() {
            return false;
        }
        request
            .headers()
            .get(header::HOST)
            .and_then(|host| host.to_str().ok())
            .is_some_and(|host| host.eq_ignore_ascii_case(&origin.authority))
    }
}

fn default_model_index_state_directory(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(MODEL_INDEX_STATE_DIRECTORY)
}

// ─── Request types ──────────────────────────────────────────────────────────

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ChatCompletionMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip)]
    pub internal_request_id: Option<String>,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub input: serde_json::Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
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
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
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
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
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
    #[serde(default, flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

// ─── Server state ───────────────────────────────────────────────────────────

struct LoadedRuntime {
    pipeline: Arc<InferencePipeline>,
    model_id: String,
    model_family: bloomai_core::ModelFamily,
    model_architecture: Option<String>,
    model_chat_template: Option<String>,
    input_modalities: Vec<bloomai_core::Modality>,
    memory_estimate: bloomai_engine::MemoryEstimate,
    kv_cache_pool: Option<Arc<bloomai_engine::BloomKvCachePool>>,
    cachemesh: Option<Arc<bloomai_engine::CacheMesh>>,
    scheduler: Option<Arc<bloomai_engine::scheduler::InferenceScheduler>>,
    _memory_reservation: Option<bloomai_engine::MemoryReservation>,
    scheduler_shutdown: CancellationToken,
    published_at: u64,
    source_path: PathBuf,
    catalog_id: Option<String>,
}

impl Drop for LoadedRuntime {
    fn drop(&mut self) {
        self.scheduler_shutdown.cancel();
    }
}

#[derive(Debug)]
struct ModelLoadRequest {
    sequence: u64,
    path: PathBuf,
    catalog_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelLoadOutcome {
    Loading,
    Ready { model_id: String },
    Failed { message: String },
}

#[derive(Debug)]
struct ActiveModelLoad {
    sequence: u64,
    path: PathBuf,
    selector: String,
    completion: watch::Sender<ModelLoadOutcome>,
}

#[derive(Debug, Default)]
struct ModelLifecycle {
    next_sequence: u64,
    active: Option<ActiveModelLoad>,
}

struct OllamaResidencyExpiry {
    runtime: Weak<LoadedRuntime>,
    expires_at: SystemTime,
}

#[derive(Default)]
struct OllamaResidencyState {
    revision: u64,
    expiry: Option<OllamaResidencyExpiry>,
    timer_cancel: CancellationToken,
}

#[derive(Debug)]
enum ModelLoadAdmission {
    AlreadyReady,
    Loading {
        sequence: u64,
        queued: bool,
        completion: watch::Receiver<ModelLoadOutcome>,
    },
}

#[derive(Debug)]
enum ModelLoadAdmissionError {
    Busy,
    Unavailable(String),
}

struct CachedModelCatalog {
    refreshed_at: Instant,
    active_path: Option<PathBuf>,
    download_revision: u64,
    import_revision: u64,
    integrity_revision: u64,
    catalog: ModelCatalog,
}

struct ServerState {
    runtime: RwLock<Option<Arc<LoadedRuntime>>>,
    inference_admission: RwLock<()>,
    semaphore: Arc<Semaphore>,
    ready: AtomicBool,
    load_in_progress: AtomicBool,
    load_progress: AtomicU8,
    load_error: RwLock<Option<String>>,
    requested_model: RwLock<Option<String>>,
    model_lifecycle: Mutex<ModelLifecycle>,
    ollama_residency: Mutex<OllamaResidencyState>,
    models_root: PathBuf,
    model_catalog_cache: RwLock<Option<CachedModelCatalog>>,
    model_storage: Arc<ModelStorageManager>,
    model_downloads: Option<Arc<ModelDownloadManager>>,
    model_imports: Option<Arc<ModelImportManager>>,
    model_index: Option<Arc<ModelIndexManager>>,
    model_integrity: Arc<ModelIntegrityManager>,
    model_preflight: Arc<ModelPreflightManager>,
    model_loader: mpsc::Sender<ModelLoadRequest>,
    metrics: Arc<ServerMetrics>,
    speculative_mode: String,
    enable_ifb: bool,
    /// Per-request cancellation tokens.
    cancel_tokens: Arc<std::sync::Mutex<HashMap<String, CancellationToken>>>,
    /// Monotonic suffix for OpenAI-compatible request IDs.
    request_counter: AtomicU64,
    /// Optional shared secret for OpenAI-compatible /v1 endpoints.
    api_key: Option<String>,
    /// Explicitly retained Responses API state; bounded and process-local.
    response_store: ResponseStore,
}

impl ServerState {
    async fn admit_model_load(
        &self,
        path: PathBuf,
        catalog_id: Option<String>,
        join_matching: bool,
    ) -> std::result::Result<ModelLoadAdmission, ModelLoadAdmissionError> {
        let selector = catalog_id
            .clone()
            .unwrap_or_else(|| model_path_label(&path));
        let mut lifecycle = self.model_lifecycle.lock().await;

        if let Some(active) = lifecycle.active.as_ref() {
            if join_matching && active.path == path && active.selector == selector {
                return Ok(ModelLoadAdmission::Loading {
                    sequence: active.sequence,
                    queued: false,
                    completion: active.completion.subscribe(),
                });
            }
            return Err(ModelLoadAdmissionError::Busy);
        }
        if self.load_in_progress.load(Ordering::Acquire) {
            return Err(ModelLoadAdmissionError::Busy);
        }
        if self
            .runtime
            .read()
            .await
            .as_ref()
            .is_some_and(|runtime| runtime.source_path == path)
        {
            return Ok(ModelLoadAdmission::AlreadyReady);
        }

        lifecycle.next_sequence = lifecycle.next_sequence.saturating_add(1).max(1);
        let sequence = lifecycle.next_sequence;
        let (completion, receiver) = watch::channel(ModelLoadOutcome::Loading);
        lifecycle.active = Some(ActiveModelLoad {
            sequence,
            path: path.clone(),
            selector: selector.clone(),
            completion: completion.clone(),
        });
        self.load_in_progress.store(true, Ordering::Release);
        self.ready.store(false, Ordering::Release);
        self.load_progress.store(0, Ordering::Release);
        *self.load_error.write().await = None;
        *self.requested_model.write().await = Some(selector);

        if let Err(error) = self.model_loader.try_send(ModelLoadRequest {
            sequence,
            path,
            catalog_id,
        }) {
            let message = format!("model loader is unavailable: {error}");
            lifecycle.active = None;
            self.load_in_progress.store(false, Ordering::Release);
            self.ready
                .store(self.runtime.read().await.is_some(), Ordering::Release);
            *self.load_error.write().await = Some(message.clone());
            completion.send_replace(ModelLoadOutcome::Failed {
                message: message.clone(),
            });
            return Err(ModelLoadAdmissionError::Unavailable(message));
        }

        Ok(ModelLoadAdmission::Loading {
            sequence,
            queued: true,
            completion: receiver,
        })
    }

    async fn finish_model_load(&self, sequence: u64, outcome: ModelLoadOutcome) {
        let mut lifecycle = self.model_lifecycle.lock().await;
        if lifecycle
            .active
            .as_ref()
            .is_some_and(|active| active.sequence == sequence)
        {
            if let Some(active) = lifecycle.active.take() {
                active.completion.send_replace(outcome);
            }
        }
    }

    async fn get_runtime(&self) -> Result<Arc<LoadedRuntime>> {
        if let Some(runtime) = self.runtime.read().await.clone() {
            Ok(runtime)
        } else {
            Err(anyhow!(self.model_unavailable().await.1))
        }
    }

    async fn model_unavailable(&self) -> (&'static str, String) {
        if self.load_in_progress.load(Ordering::Acquire) {
            (
                "model_loading",
                format!(
                    "Model is loading (progress: {}%).",
                    self.load_progress.load(Ordering::Acquire)
                ),
            )
        } else if let Some(error) = self.load_error.read().await.as_ref() {
            (
                "model_load_failed",
                format!("The model failed to load: {error}"),
            )
        } else {
            (
                "model_not_loaded",
                "No model is loaded. Choose a catalog model or start the server with --model."
                    .to_string(),
            )
        }
    }

    async fn model_catalog_snapshot(&self) -> Result<(ModelCatalog, Option<Arc<LoadedRuntime>>)> {
        self.model_catalog_snapshot_with_refresh(false).await
    }

    async fn fresh_model_catalog_snapshot(
        &self,
    ) -> Result<(ModelCatalog, Option<Arc<LoadedRuntime>>)> {
        self.model_catalog_snapshot_with_refresh(true).await
    }

    async fn model_catalog_snapshot_with_refresh(
        &self,
        force_refresh: bool,
    ) -> Result<(ModelCatalog, Option<Arc<LoadedRuntime>>)> {
        let runtime = self.runtime.read().await.clone();
        let active_path = runtime.as_ref().map(|runtime| runtime.source_path.clone());
        let download_revision = self
            .model_downloads
            .as_ref()
            .map(|manager| manager.catalog_revision())
            .unwrap_or(0);
        let import_revision = self
            .model_imports
            .as_ref()
            .map(|manager| manager.catalog_revision())
            .unwrap_or(0);
        let integrity_revision = self.model_integrity.catalog_revision();
        if !force_refresh {
            if let Some(cached) = self.model_catalog_cache.read().await.as_ref() {
                if cached.refreshed_at.elapsed() < MODEL_CATALOG_CACHE_TTL
                    && cached.active_path == active_path
                    && cached.download_revision == download_revision
                    && cached.import_revision == import_revision
                    && cached.integrity_revision == integrity_revision
                {
                    return Ok((cached.catalog.clone(), runtime));
                }
            }
        }

        let root = self.models_root.clone();
        let active_for_scan = active_path.clone();
        let catalog =
            task::spawn_blocking(move || ModelCatalog::scan(&root, active_for_scan.as_deref()))
                .await
                .map_err(|error| anyhow!("model catalog scan task failed: {error}"))??;
        *self.model_catalog_cache.write().await = Some(CachedModelCatalog {
            refreshed_at: Instant::now(),
            active_path,
            download_revision,
            import_revision,
            integrity_revision,
            catalog: catalog.clone(),
        });
        Ok((catalog, runtime))
    }
}

struct CancelTokenGuard {
    tokens: Arc<std::sync::Mutex<HashMap<String, CancellationToken>>>,
    request_id: String,
    token: CancellationToken,
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
            let mut registrations = tokens.lock().unwrap_or_else(|e| e.into_inner());
            registrations.insert(request_id.clone(), token.clone());
        }
        Self {
            tokens,
            request_id,
            token,
        }
    }

    fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

impl Drop for CancelTokenGuard {
    fn drop(&mut self) {
        let mut tokens = self.tokens.lock().unwrap_or_else(|e| e.into_inner());
        tokens.remove(&self.request_id);
    }
}

/// Retains request accounting, cancellation registration, and admission until
/// both the client-facing response future and any blocking worker are settled.
struct InferenceLifecycle {
    registration: std::sync::Mutex<Option<CancelTokenGuard>>,
    request_id: String,
    token: CancellationToken,
    metrics: Arc<ServerMetrics>,
    request_start: Instant,
    generated_tokens: Arc<AtomicU64>,
    prompt_tokens: u64,
    execution: StreamExecution,
    permit: std::sync::Mutex<Option<OwnedSemaphorePermit>>,
    worker_done: AtomicBool,
    client_dropped: AtomicBool,
    settled: AtomicBool,
}

struct InferenceLifecycleResources {
    metrics: Arc<ServerMetrics>,
    request_start: Instant,
    generated_tokens: Arc<AtomicU64>,
    prompt_tokens: u64,
    permit: OwnedSemaphorePermit,
}

enum StreamExecution {
    Scheduled(Arc<InferenceScheduler>),
    Blocking,
}

impl InferenceLifecycle {
    fn new(
        registration: CancelTokenGuard,
        resources: InferenceLifecycleResources,
        execution: StreamExecution,
    ) -> Arc<Self> {
        let worker_done = matches!(&execution, StreamExecution::Scheduled(_));
        Arc::new(Self {
            request_id: registration.request_id.clone(),
            token: registration.token(),
            registration: std::sync::Mutex::new(Some(registration)),
            metrics: resources.metrics,
            request_start: resources.request_start,
            generated_tokens: resources.generated_tokens,
            prompt_tokens: resources.prompt_tokens,
            execution,
            permit: std::sync::Mutex::new(Some(resources.permit)),
            worker_done: AtomicBool::new(worker_done),
            client_dropped: AtomicBool::new(false),
            settled: AtomicBool::new(false),
        })
    }

    fn client_guard(self: &Arc<Self>) -> InferenceClientGuard {
        InferenceClientGuard {
            lifecycle: Arc::clone(self),
            completed: false,
        }
    }

    fn worker_guard(self: &Arc<Self>) -> InferenceWorkerGuard {
        InferenceWorkerGuard {
            lifecycle: Arc::clone(self),
        }
    }

    fn finish(&self, success: bool) {
        let success =
            success && !self.token.is_cancelled() && !self.client_dropped.load(Ordering::Acquire);
        self.settle(success);
    }

    fn client_dropped(&self) {
        self.client_dropped.store(true, Ordering::Release);
        self.token.cancel();
        match &self.execution {
            StreamExecution::Scheduled(scheduler) => {
                scheduler.cancel_request(&self.request_id);
                self.settle(false);
            }
            StreamExecution::Blocking if self.worker_done.load(Ordering::Acquire) => {
                self.settle(false);
            }
            StreamExecution::Blocking => {}
        }
    }

    fn worker_finished(&self) {
        self.worker_done.store(true, Ordering::Release);
        if self.client_dropped.load(Ordering::Acquire) {
            self.settle(false);
        }
    }

    fn settle(&self, success: bool) {
        if self.settled.swap(true, Ordering::AcqRel) {
            return;
        }
        self.registration
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        self.metrics.record_request_end(
            success,
            self.request_start.elapsed().as_secs_f64(),
            self.generated_tokens.load(Ordering::Relaxed),
            self.prompt_tokens,
        );
        self.permit.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}

struct InferenceClientGuard {
    lifecycle: Arc<InferenceLifecycle>,
    completed: bool,
}

impl InferenceClientGuard {
    fn finish(&mut self, success: bool) {
        self.lifecycle.finish(success);
        self.completed = true;
    }
}

impl Drop for InferenceClientGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.lifecycle.client_dropped();
        }
    }
}

struct InferenceWorkerGuard {
    lifecycle: Arc<InferenceLifecycle>,
}

impl Drop for InferenceWorkerGuard {
    fn drop(&mut self) {
        self.lifecycle.worker_finished();
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

fn valid_http_request_id(value: &HeaderValue) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_HTTP_REQUEST_ID_CHARS
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

fn normalized_http_request_id(request: &AxumRequest) -> RequestId {
    if let Some(value) = request
        .headers()
        .get(HTTP_REQUEST_ID_HEADER)
        .filter(|value| valid_http_request_id(value))
    {
        return RequestId::new(value.clone());
    }

    MakeRequestUuid
        .make_request_id(request)
        .expect("MakeRequestUuid always returns a request ID")
}

async fn correlate_http_request(mut request: AxumRequest, next: Next) -> Response {
    let request_id = normalized_http_request_id(&request);
    request.headers_mut().insert(
        header::HeaderName::from_static(HTTP_REQUEST_ID_HEADER),
        request_id.header_value().clone(),
    );
    request.extensions_mut().insert(request_id.clone());

    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::HeaderName::from_static(HTTP_REQUEST_ID_HEADER),
        request_id.header_value().clone(),
    );
    response.extensions_mut().insert(request_id);
    response
}

fn requires_no_store(path: &str) -> bool {
    matches!(path, "/health" | "/ready" | "/metrics" | "/v1" | "/api")
        || path.starts_with("/v1/")
        || path.starts_with("/api/")
}

#[derive(Clone, Copy)]
enum ApiProtocolFamily {
    OpenAi,
    Ollama,
}

fn api_protocol_family(path: &str) -> Option<ApiProtocolFamily> {
    if path == "/v1" || path.starts_with("/v1/") {
        Some(ApiProtocolFamily::OpenAi)
    } else if path == "/api" || path.starts_with("/api/") {
        Some(ApiProtocolFamily::Ollama)
    } else {
        None
    }
}

fn has_protocol_error_content_type(response: &Response) -> bool {
    let Some(content_type) = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    media_type == "application/json"
        || media_type.ends_with("+json")
        || media_type == "text/event-stream"
        || media_type == "application/x-ndjson"
}

fn openai_framework_error(status: axum::http::StatusCode) -> (&'static str, &'static str) {
    match status {
        axum::http::StatusCode::REQUEST_TIMEOUT => (
            ApiError::Timeout.error_type(),
            "The request timed out before it could be completed.",
        ),
        axum::http::StatusCode::PAYLOAD_TOO_LARGE => (
            ApiError::InvalidRequest.error_type(),
            "The request body exceeds the configured size limit.",
        ),
        axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE => (
            ApiError::InvalidRequest.error_type(),
            "The request Content-Type is not supported for this API route.",
        ),
        axum::http::StatusCode::UNPROCESSABLE_ENTITY => (
            ApiError::InvalidRequest.error_type(),
            "The request body does not match the endpoint schema.",
        ),
        axum::http::StatusCode::TOO_MANY_REQUESTS => (
            ApiError::RateLimitExceeded.error_type(),
            "The server is temporarily at request capacity.",
        ),
        axum::http::StatusCode::NOT_FOUND => (
            ApiError::NotFound.error_type(),
            "The requested OpenAI-compatible API resource does not exist.",
        ),
        axum::http::StatusCode::UNAUTHORIZED => (
            ApiError::AuthenticationError.error_type(),
            "Authentication is required for this API route.",
        ),
        axum::http::StatusCode::SERVICE_UNAVAILABLE => (
            ApiError::ServiceUnavailable.error_type(),
            "The service is temporarily unavailable.",
        ),
        status if status.is_server_error() => (
            ApiError::InternalError.error_type(),
            "The server could not process the request.",
        ),
        _ => (
            ApiError::InvalidRequest.error_type(),
            "The request is malformed or unsupported.",
        ),
    }
}

fn ollama_framework_error(status: axum::http::StatusCode) -> &'static str {
    match status {
        axum::http::StatusCode::REQUEST_TIMEOUT => "request timed out before it could be completed",
        axum::http::StatusCode::PAYLOAD_TOO_LARGE => {
            "request body exceeds the configured size limit"
        }
        axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE => {
            "request Content-Type is not supported for this API route"
        }
        axum::http::StatusCode::UNPROCESSABLE_ENTITY => {
            "request body does not match the endpoint schema"
        }
        axum::http::StatusCode::TOO_MANY_REQUESTS => "server is temporarily at request capacity",
        axum::http::StatusCode::NOT_FOUND => "requested Ollama-compatible API resource not found",
        axum::http::StatusCode::UNAUTHORIZED => "authentication is required for this API route",
        axum::http::StatusCode::SERVICE_UNAVAILABLE => "service is temporarily unavailable",
        status if status.is_server_error() => "server could not process the request",
        _ => "request is malformed or unsupported",
    }
}

async fn normalize_protocol_error_response(request: AxumRequest, next: Next) -> Response {
    let family = api_protocol_family(request.uri().path());
    let response = next.run(request).await;
    let Some(family) = family else {
        return response;
    };
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error())
        || has_protocol_error_content_type(&response)
    {
        return response;
    }

    let shaped = match family {
        ApiProtocolFamily::OpenAi => {
            let (error_type, message) = openai_framework_error(status);
            error_response(status, error_type, message)
        }
        ApiProtocolFamily::Ollama => ollama_error_response(status, ollama_framework_error(status)),
    };
    let (_, shaped_body) = shaped.into_parts();
    let (mut parts, _) = response.into_parts();
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, shaped_body)
}

async fn prevent_dynamic_response_caching(request: AxumRequest, next: Next) -> Response {
    let no_store = requires_no_store(request.uri().path());
    let mut response = next.run(request).await;
    if no_store {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    response
}

async fn publish_transient_retry_after(request: AxumRequest, next: Next) -> Response {
    let mut response = next.run(request).await;
    if response.status() == axum::http::StatusCode::TOO_MANY_REQUESTS {
        response
            .headers_mut()
            .entry(header::RETRY_AFTER)
            .or_insert(HeaderValue::from_static(
                DEFAULT_CAPACITY_RETRY_AFTER_SECONDS,
            ));
    }
    response
}

async fn publish_authentication_challenge(request: AxumRequest, next: Next) -> Response {
    let mut response = next.run(request).await;
    if response.status() == axum::http::StatusCode::UNAUTHORIZED {
        response
            .headers_mut()
            .entry(header::WWW_AUTHENTICATE)
            .or_insert(HeaderValue::from_static(
                DEFAULT_BEARER_AUTHENTICATION_CHALLENGE,
            ));
    }
    response
}

fn configured_cors_layer(policy: &BrowserOriginPolicy) -> CorsLayer {
    let layer = match policy {
        BrowserOriginPolicy::SameOrigin => CorsLayer::new(),
        BrowserOriginPolicy::Exact(origin) => CorsLayer::new().allow_origin(origin.header.clone()),
        BrowserOriginPolicy::Any => CorsLayer::new().allow_origin(Any),
    };
    layer.allow_methods(Any).allow_headers(Any).expose_headers([
        header::HeaderName::from_static(HTTP_REQUEST_ID_HEADER),
        header::RETRY_AFTER,
        header::WWW_AUTHENTICATE,
    ])
}

async fn enforce_browser_origin(
    State(guard): State<BrowserOriginGuard>,
    request: AxumRequest,
    next: Next,
) -> Response {
    if guard.permits(&request) {
        next.run(request).await
    } else {
        (
            axum::http::StatusCode::FORBIDDEN,
            "The browser origin is not allowed by the Bloom server policy.",
        )
            .into_response()
    }
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
            "Missing or invalid API key for protected API endpoint.",
        )
    }
}

async fn require_ollama_api_key(
    State(state): State<Arc<ServerState>>,
    req: AxumRequest,
    next: Next,
) -> Response {
    let Some(expected) = state.api_key.as_deref() else {
        return next.run(req).await;
    };

    let bearer = format!("Bearer {expected}");
    let authorization_ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| constant_time_eq(value.as_bytes(), bearer.as_bytes()));
    let x_api_key_ok = req
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| constant_time_eq(value.as_bytes(), expected.as_bytes()));

    if authorization_ok || x_api_key_ok {
        next.run(req).await
    } else {
        ollama_error_response(
            axum::http::StatusCode::UNAUTHORIZED,
            "missing or invalid API key for protected API endpoint",
        )
    }
}

async fn handle_openai_route_not_found() -> Response {
    api_error(
        ApiError::NotFound,
        "The requested OpenAI-compatible API route does not exist.",
    )
}

async fn handle_openai_method_not_allowed() -> Response {
    error_response(
        axum::http::StatusCode::METHOD_NOT_ALLOWED,
        ApiError::InvalidRequest.error_type(),
        "The HTTP method is not supported for this OpenAI-compatible API route.",
    )
}

async fn handle_ollama_route_not_found() -> Response {
    ollama_error_response(
        axum::http::StatusCode::NOT_FOUND,
        "Ollama-compatible API route not found",
    )
}

async fn handle_ollama_method_not_allowed() -> Response {
    ollama_error_response(
        axum::http::StatusCode::METHOD_NOT_ALLOWED,
        "HTTP method not allowed for this Ollama-compatible API route",
    )
}

fn ollama_api_router(state: Arc<ServerState>) -> Router<Arc<ServerState>> {
    Router::new()
        .route("/version", get(handle_ollama_version))
        .route("/tags", get(handle_ollama_tags))
        .route("/ps", get(handle_ollama_ps))
        .route("/show", post(handle_ollama_show))
        .route("/pull", post(handle_ollama_pull))
        .route("/delete", delete(handle_ollama_delete))
        .route("/chat", post(handle_ollama_chat))
        .route("/generate", post(handle_ollama_generate))
        .route("/embed", post(handle_ollama_embed))
        .route("/embeddings", post(handle_ollama_legacy_embeddings))
        .route_layer(middleware::from_fn_with_state(
            state,
            require_ollama_api_key,
        ))
        .fallback(handle_ollama_route_not_found)
        .method_not_allowed_fallback(handle_ollama_method_not_allowed)
}

// ─── Main ───────────────────────────────────────────────────────────────────

/// Parse process configuration, assemble the application, and serve requests.
///
/// Keeping the Tokio runtime in the binary entry point lets tests and other
/// launchers reuse the application bootstrap without nesting runtimes.
pub async fn run_cli() -> Result<()> {
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
    if args.model_index_state_dir.is_none() {
        args.model_index_state_dir = Some(default_model_index_state_directory(&config_path));
    }

    let models_root = args
        .models_dir
        .clone()
        .unwrap_or(bloomai_engine::default_config_dir()?.join("models"));
    if let Some(format) = args.doctor {
        let report = inspect_server(&args, config_path.exists(), &models_root);
        print!("{}", report.render(format)?);
        if report.has_failures() {
            return Err(anyhow!(
                "server doctor found blocking configuration or environment failures"
            ));
        }
        return Ok(());
    }
    validate_server_arguments(&args)?;

    let model_path = args.model.clone();
    // Validate runtime selection before any storage cleanup or directory
    // creation can occur.
    let registry = engine_registry();
    registry.get(&args.backend).map_err(|e| {
        anyhow!(
            "{}. Supported engines are: candle, openvino, funasr, qwen3_vl, longcat, intel-npu, npu-tts, onnxruntime, coreml, mlx, vulkan, llamacpp, wan.",
            e
        )
    })?;
    let device_kind = match args.device.to_lowercase().as_str() {
        "cpu" => DeviceKind::Cpu,
        "gpu" | "cuda" | "metal" => DeviceKind::Gpu,
        "npu" | "intel-npu" => DeviceKind::Npu,
        other => return Err(anyhow!("unsupported device: {}", other)),
    };

    let model_license_policy = Arc::new(ModelLicensePolicy::new(
        args.allowed_model_licenses.clone(),
    )?);
    let model_index = ModelIndexManager::from_config(
        ModelIndexManagerConfig {
            file: args.model_index_file.clone(),
            url: args.model_index_url.clone(),
            public_key: args.model_index_public_key.clone(),
            public_keys: args.model_index_public_keys.clone(),
            refresh_seconds: args.model_index_refresh_seconds,
            max_download_bytes: args.max_model_download_bytes,
            state_directory: args
                .model_index_state_dir
                .clone()
                .ok_or_else(|| anyhow!("model index state directory was not resolved"))?,
        },
        Arc::clone(&model_license_policy),
    )?;
    // Hold one operating-system-backed ownership lease through Tokio runtime
    // teardown. It must precede recovery, cleanup, and every catalog mutation.
    let catalog_lease = ModelCatalogLease::acquire(&models_root)?;
    let _catalog_lease_task = tokio::spawn(async move {
        std::future::pending::<()>().await;
        drop(catalog_lease);
    });
    let model_storage = ModelStorageManager::new(
        models_root.clone(),
        args.max_model_storage_bytes,
        args.staged_model_retention_seconds,
    );
    if let Some(recovery) = model_upgrade::recover_model_upgrade(&models_root).await? {
        tracing::warn!(
            recovery = ?recovery,
            "Recovered an interrupted signed-model upgrade transaction"
        );
    }
    // Recovery can restore a configured startup path that was moved into the
    // transaction backup immediately before an interrupted commit.
    if let Some(path) = model_path.as_ref() {
        if !path.exists() {
            return Err(anyhow!("model path does not exist: {}", path.display()));
        }
    }
    let removed_staged_sessions = model_storage.cleanup_stale().await?;
    if removed_staged_sessions > 0 {
        tracing::info!(
            removed_staged_sessions,
            "Removed expired staged model acquisitions"
        );
    }
    let _storage_cleanup_task = if args.staged_model_retention_seconds > 0 {
        let storage = Arc::clone(&model_storage);
        let interval_seconds = (args.staged_model_retention_seconds / 2).clamp(60, 3_600);
        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(interval_seconds));
            interval.tick().await;
            loop {
                interval.tick().await;
                match storage.cleanup_stale().await {
                    Ok(removed) if removed > 0 => {
                        tracing::info!(
                            removed_staged_sessions = removed,
                            "Removed expired staged model acquisitions"
                        );
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(%error, "Failed to clean expired staged model acquisitions");
                    }
                }
            }
        }))
    } else {
        None
    };
    let model_downloads = if args.enable_model_downloads {
        Some(ModelDownloadManager::with_storage_and_license_policy(
            models_root.clone(),
            args.max_model_download_bytes,
            Arc::clone(&model_storage),
            Arc::clone(&model_license_policy),
        )?)
    } else {
        None
    };
    let model_imports = if args.enable_model_imports {
        Some(ModelImportManager::with_storage_and_license_policy(
            models_root.clone(),
            args.max_model_import_bytes,
            args.max_model_import_chunk_bytes,
            Arc::clone(&model_storage),
            Arc::clone(&model_license_policy),
        )?)
    } else {
        None
    };
    let model_integrity = ModelIntegrityManager::new(models_root.clone());

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

    let model_preflight = ModelPreflightManager::new(
        models_root.clone(),
        ModelPreflightConfig {
            backend: args.backend.clone(),
            speculative: args.speculative.clone(),
            device: device_kind,
            context_size: args.context_size,
            max_concurrent: args.max_concurrent,
            memory_utilization: args.memory_utilization,
            reserve_memory_bytes: args.reserve_memory_bytes,
            disable_memory_prealloc: args.disable_memory_prealloc,
        },
    );

    let (model_loader, model_load_requests) = mpsc::channel(1);
    let state = Arc::new(ServerState {
        runtime: RwLock::new(None),
        inference_admission: RwLock::new(()),
        semaphore: Arc::new(Semaphore::new(args.max_concurrent)),
        ready: AtomicBool::new(false),
        load_in_progress: AtomicBool::new(false),
        load_progress: AtomicU8::new(0),
        load_error: RwLock::new(None),
        requested_model: RwLock::new(None),
        model_lifecycle: Mutex::new(ModelLifecycle::default()),
        ollama_residency: Mutex::new(OllamaResidencyState::default()),
        models_root,
        model_catalog_cache: RwLock::new(None),
        model_storage,
        model_downloads,
        model_imports,
        model_index,
        model_integrity,
        model_preflight,
        model_loader,
        metrics: Arc::new(ServerMetrics::new()),
        speculative_mode: args.speculative.clone(),
        enable_ifb: args.enable_ifb,
        cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
        request_counter: AtomicU64::new(0),
        api_key: args.api_key.clone().filter(|value| !value.is_empty()),
        response_store: ResponseStore::default(),
    });

    let state_clone = Arc::clone(&state);
    let args_clone = args.clone();
    tokio::spawn(model_loader_loop(
        state_clone,
        args_clone,
        device_kind,
        model_load_requests,
    ));

    if let Some(path) = model_path {
        state
            .admit_model_load(path, None, false)
            .await
            .map_err(|error| anyhow!("model loader stopped before startup: {error:?}"))?;
    } else {
        tracing::info!(
            models_root = %state.models_root.display(),
            "Server started without an active model; use the model-management API to load one"
        );
    }

    // Build middleware stack
    let browser_origin_policy = parse_browser_origin_policy(&args.cors_allow_origin)
        .map_err(|error| anyhow!("invalid browser origin policy: {error}"))?;
    let configured_addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let browser_origin_guard = BrowserOriginGuard {
        policy: browser_origin_policy.clone(),
        loopback_listener: configured_addr.ip().is_loopback(),
    };
    let cors = configured_cors_layer(&browser_origin_policy);

    let v1_routes = Router::new()
        .route("/observability", get(handle_observability))
        .route("/models", get(handle_models))
        .route("/models/{model}", get(handle_model_retrieve))
        .route("/model-management/models", get(handle_model_catalog))
        .route(
            "/model-management/index",
            get(handle_model_index).post(handle_model_index_refresh),
        )
        .route(
            "/model-management/index/{id}/download",
            post(handle_model_index_download),
        )
        .route("/model-management/inventory", get(handle_model_inventory))
        .route(
            "/model-management/inventory/reconcile",
            post(handle_model_inventory_reconcile).layer(DefaultBodyLimit::max(
                model_inventory::MAX_MODEL_INVENTORY_BYTES,
            )),
        )
        .route(
            "/model-management/inventory/restore/{id}",
            post(handle_model_inventory_restore).layer(DefaultBodyLimit::max(
                model_inventory::MAX_MODEL_INVENTORY_BYTES,
            )),
        )
        .route("/model-management/preflight", post(handle_model_preflight))
        .route("/model-management/switch", post(handle_model_switch))
        .route("/model-management/unload", post(handle_model_unload))
        .route("/model-management/remove", post(handle_model_remove))
        .route(
            "/model-management/integrity",
            post(handle_model_integrity_start).delete(handle_model_integrity_cancel),
        )
        .route(
            "/model-management/downloads",
            post(handle_model_download_start).delete(handle_model_download_cancel),
        )
        .route(
            "/model-management/downloads/inspect",
            post(handle_model_download_source_inspect).layer(DefaultBodyLimit::max(4 * 1024)),
        )
        .route(
            "/model-management/downloads/resume",
            post(handle_model_download_resume),
        )
        .route(
            "/model-management/downloads/discard",
            post(handle_model_download_discard),
        )
        .route("/model-management/imports", post(handle_model_import_begin))
        .route(
            "/model-management/imports/{filename}",
            put(handle_model_import_chunk)
                .delete(handle_model_import_discard)
                .layer(DefaultBodyLimit::max(args.max_model_import_chunk_bytes)),
        )
        .route(
            "/model-management/imports/{filename}/complete",
            post(handle_model_import_complete),
        )
        .route("/chat/completions", post(handle_chat_completions))
        .route("/responses", post(handle_responses))
        .route(
            "/responses/{response_id}",
            get(handle_response_retrieve).delete(handle_response_delete),
        )
        .route(
            "/responses/{response_id}/input_items",
            get(handle_response_input_items),
        )
        .route("/completions", post(handle_completions))
        .route("/embeddings", post(handle_embeddings))
        .route("/rerank", post(handle_rerank))
        .route("/multimodal/stream", post(handle_multimodal_stream))
        .route(
            "/multimodal/upload",
            post(handle_multimodal_upload).layer(DefaultBodyLimit::max(args.max_upload_bytes)),
        )
        .route("/world/step", post(handle_world_step))
        .route("/kv-cache-stats", get(handle_kv_cache_stats))
        .route("/cancel/{request_id}", post(handle_cancel))
        .route("/backends", get(handle_backends))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ))
        .fallback(handle_openai_route_not_found)
        .method_not_allowed_fallback(handle_openai_method_not_allowed);

    let ollama_routes = ollama_api_router(Arc::clone(&state));

    let mut app = Router::new()
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .route("/metrics", get(handle_metrics))
        .nest("/v1", v1_routes)
        .nest("/api", ollama_routes)
        .with_state(Arc::clone(&state));

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
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &AxumRequest| {
                    let request_id = request
                        .extensions()
                        .get::<RequestId>()
                        .and_then(|request_id| request_id.header_value().to_str().ok())
                        .unwrap_or("unavailable");
                    tracing::info_span!(
                        "http_request",
                        request_id,
                        method = %request.method(),
                        path = request.uri().path()
                    )
                })
                .on_failure(
                    |failure: ServerErrorsFailureClass,
                     latency: Duration,
                     _span: &tracing::Span| {
                        match failure {
                            ServerErrorsFailureClass::StatusCode(status)
                                if status == axum::http::StatusCode::SERVICE_UNAVAILABLE =>
                            {
                                tracing::debug!(%status, ?latency, "HTTP service is temporarily unavailable");
                            }
                            failure => {
                                tracing::error!(%failure, ?latency, "HTTP response failed");
                            }
                        }
                    }
                ),
        )
        .layer(cors)
        .layer(middleware::from_fn_with_state(
            browser_origin_guard,
            enforce_browser_origin,
        ))
        .layer(middleware::from_fn(normalize_protocol_error_response))
        .layer(middleware::from_fn(publish_transient_retry_after))
        .layer(middleware::from_fn(publish_authentication_challenge))
        .layer(middleware::from_fn(prevent_dynamic_response_caching))
        .layer(middleware::from_fn(correlate_http_request));

    let addr = configured_addr;

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

    let mut shutdown_signals = ShutdownSignalListener::install()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound_addr = listener.local_addr()?;
    let local_url = ui::browser_url(bound_addr);
    tracing::info!(
        "Bloom OpenAI-compatible server running on http://{}",
        bound_addr
    );
    if ui::embedded_ui_available() {
        tracing::info!("Bloom UI is available at {}", local_url);
    }
    if args.open_browser {
        let url = local_url.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            let launch_url = url.clone();
            match task::spawn_blocking(move || ui::launch_browser(&launch_url)).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    tracing::warn!(%error, %url, "Could not open the system browser; open the URL manually");
                }
                Err(error) => {
                    tracing::warn!(%error, %url, "Browser launcher task failed; open the URL manually");
                }
            }
        });
    }
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let shutdown_deadline_receiver = shutdown_receiver.clone();
    let state_for_shutdown = Arc::clone(&state);
    let signal_task = tokio::spawn(async move {
        match shutdown_signals.recv().await {
            Ok(signal) => {
                tracing::info!(
                    signal = signal.label(),
                    "Received shutdown signal; refusing new inference and draining active HTTP requests"
                );
            }
            Err(error) => {
                tracing::error!(%error, "Shutdown signal listener failed; shutting down safely");
                state_for_shutdown.ready.store(false, Ordering::Release);
                let _ = shutdown_sender.send(true);
                return;
            }
        }
        state_for_shutdown.ready.store(false, Ordering::Release);
        let _ = shutdown_sender.send(true);

        match shutdown_signals.recv().await {
            Ok(signal) => {
                tracing::error!(
                    signal = signal.label(),
                    "Received a repeated shutdown signal; forcing process termination"
                );
                std::process::exit(1);
            }
            Err(error) => {
                tracing::error!(%error, "Shutdown signal listener failed during graceful drain");
            }
        }
    });
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(wait_for_shutdown_notification(shutdown_receiver))
        .into_future();
    match wait_for_server_or_shutdown_timeout(
        server,
        shutdown_deadline_receiver,
        Duration::from_secs(args.shutdown_timeout_seconds),
    )
    .await
    {
        ShutdownWait::Completed(result) => result?,
        ShutdownWait::TimedOut => {
            tracing::error!(
                timeout_seconds = args.shutdown_timeout_seconds,
                "Graceful shutdown deadline expired; forcing process termination"
            );
            std::process::exit(1);
        }
    }
    signal_task.abort();

    tracing::info!("Server shut down gracefully");
    Ok(())
}

fn engine_registry() -> EngineRegistry {
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
    registry
}

async fn model_loader_loop(
    state: Arc<ServerState>,
    args: Args,
    device_kind: DeviceKind,
    mut requests: mpsc::Receiver<ModelLoadRequest>,
) {
    while let Some(request) = requests.recv().await {
        let request_label = request
            .catalog_id
            .clone()
            .unwrap_or_else(|| model_path_label(&request.path));
        tracing::info!(model = %request_label, "Starting model load");
        state.load_progress.store(1, Ordering::Release);
        *state.load_error.write().await = None;

        // Close the admission gate, then let requests that already own the
        // previous runtime finish before it can be replaced.
        let admission_guard = state.inference_admission.write().await;
        while state.metrics.in_flight_requests.load(Ordering::Acquire) > 0 {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        match prepare_loaded_runtime(
            Arc::clone(&state),
            &args,
            device_kind,
            request.path,
            request.catalog_id,
        )
        .await
        {
            Ok(runtime) => {
                let model_id = runtime.model_id.clone();
                let previous = state.runtime.write().await.replace(Arc::new(runtime));
                drop(previous);
                state.load_progress.store(100, Ordering::Release);
                state.ready.store(true, Ordering::Release);
                state.load_in_progress.store(false, Ordering::Release);
                state
                    .finish_model_load(
                        request.sequence,
                        ModelLoadOutcome::Ready {
                            model_id: model_id.clone(),
                        },
                    )
                    .await;
                drop(admission_guard);
                tracing::info!(model = %model_id, "Model load completed");
            }
            Err(error) => {
                let message = error.to_string();
                let requested_model = state
                    .requested_model
                    .read()
                    .await
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                tracing::error!(path = %requested_model, error = %message, "Model load failed");
                *state.load_error.write().await = Some(message.clone());
                state.load_progress.store(0, Ordering::Release);
                let has_fallback = state.runtime.read().await.is_some();
                state.ready.store(has_fallback, Ordering::Release);
                state.load_in_progress.store(false, Ordering::Release);
                state
                    .finish_model_load(
                        request.sequence,
                        ModelLoadOutcome::Failed {
                            message: message.clone(),
                        },
                    )
                    .await;
                drop(admission_guard);
            }
        }
    }
    tracing::error!("Model loader stopped because its request channel was closed");
}

fn model_path_label(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("external model")
        .to_string()
}

async fn prepare_loaded_runtime(
    state: Arc<ServerState>,
    args: &Args,
    device_kind: DeviceKind,
    model_path: PathBuf,
    catalog_id: Option<String>,
) -> Result<LoadedRuntime> {
    state.load_progress.store(5, Ordering::Release);
    let manifest = bloomai_engine::load_manifest(&model_path)?;
    let backend_name = select_backend_name(&args.backend, &args.speculative, &manifest);
    engine_registry().get(&backend_name).map_err(|error| {
        anyhow!(
            "{}. Supported engines are: candle, openvino, funasr, qwen3_vl, longcat, intel-npu, npu-tts, onnxruntime, coreml, mlx, vulkan, llamacpp, wan.",
            error
        )
    })?;

    let planned_context_size = args.context_size.saturating_mul(args.max_concurrent.max(1));
    let planned_memory =
        bloomai_engine::estimate_memory_for_device(&manifest, planned_context_size, device_kind);
    state.load_progress.store(15, Ordering::Release);
    let memory_plan = bloomai_engine::plan_memory_preallocation(
        planned_memory,
        bloomai_engine::MemoryPreallocationConfig {
            enabled: !args.disable_memory_prealloc,
            memory_utilization: args.memory_utilization,
            reserve_memory_bytes: args.reserve_memory_bytes,
        },
    )?;
    state.load_progress.store(25, Ordering::Release);
    drop(bloomai_engine::reserve_memory_for_plan(&memory_plan)?);

    state.load_progress.store(35, Ordering::Release);
    let context_size = args.context_size;
    let pipeline_path = model_path.clone();
    let pipeline = task::spawn_blocking(move || {
        let registry = engine_registry();
        let engine = registry
            .get(&backend_name)
            .map_err(|error| anyhow!(error.to_string()))?;
        InferencePipeline::load_standalone_with_context(
            engine,
            device_kind,
            &pipeline_path,
            context_size,
        )
    })
    .await
    .map_err(|error| anyhow!("model loader task failed: {error}"))??;
    let pipeline = Arc::new(pipeline);
    let model_id = pipeline.metadata().id.clone();
    tracing::info!(model = %model_id, "Model pipeline is loaded; preparing runtime services");

    let actual_context_size = pipeline
        .context_size()
        .saturating_mul(args.max_concurrent.max(1));
    let memory_estimate =
        bloomai_engine::estimate_memory_for_device(&manifest, actual_context_size, device_kind);
    state.load_progress.store(65, Ordering::Release);
    let scheduling = build_scheduling_runtime(
        &state,
        args,
        &manifest,
        Arc::clone(&pipeline),
        &model_id,
        actual_context_size,
        &memory_estimate,
    )?;

    state.load_progress.store(95, Ordering::Release);
    let model_architecture = manifest
        .parameters
        .get("gguf_architecture")
        .or_else(|| manifest.parameters.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let model_chat_template = manifest
        .parameters
        .get("chat_template_kind")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Ok(LoadedRuntime {
        pipeline,
        model_id,
        model_family: manifest.family,
        model_architecture,
        model_chat_template,
        input_modalities: manifest.io_schema.inputs,
        memory_estimate,
        kv_cache_pool: scheduling.kv_cache_pool,
        cachemesh: scheduling.cachemesh,
        scheduler: scheduling.scheduler,
        _memory_reservation: scheduling.memory_reservation,
        scheduler_shutdown: scheduling.shutdown,
        published_at: unix_seconds(),
        source_path: model_path,
        catalog_id,
    })
}

struct SchedulingRuntime {
    kv_cache_pool: Option<Arc<BloomKvCachePool>>,
    cachemesh: Option<Arc<CacheMesh>>,
    scheduler: Option<Arc<InferenceScheduler>>,
    memory_reservation: Option<bloomai_engine::MemoryReservation>,
    shutdown: CancellationToken,
}

fn build_scheduling_runtime(
    state: &ServerState,
    args: &Args,
    manifest: &bloomai_core::ModelManifest,
    pipeline: Arc<InferencePipeline>,
    model_id: &str,
    memory_context_size: usize,
    memory_estimate: &bloomai_engine::MemoryEstimate,
) -> Result<SchedulingRuntime> {
    let shutdown = CancellationToken::new();
    if !args.enable_ifb {
        return Ok(SchedulingRuntime {
            kv_cache_pool: None,
            cachemesh: None,
            scheduler: None,
            memory_reservation: None,
            shutdown,
        });
    }

    let block_size = 16;
    let total_blocks = div_ceil_usize(memory_context_size, block_size).max(1);
    let num_layers = manifest_param_usize(
        manifest,
        &["num_hidden_layers", "num_layers", "block_count"],
        28,
    );
    let num_kv_heads = manifest_param_usize(
        manifest,
        &[
            "num_key_value_heads",
            "num_kv_heads",
            "attention_head_count_kv",
        ],
        8,
    );
    let head_dim = manifest_param_usize(manifest, &["head_dim"], 128);
    let kv_dim = num_kv_heads.saturating_mul(head_dim).max(1);
    let long_context_policy = build_long_context_policy(args)?;
    let memory_reservation = if !args.disable_memory_prealloc && memory_estimate.kv_cache_bytes > 0
    {
        Some(bloomai_engine::MemoryReservation::reserve(
            memory_estimate.kv_cache_bytes,
        )?)
    } else {
        None
    };
    let kv_pool = Arc::new(BloomKvCachePool::new(block_size, total_blocks));

    let device = if pipeline.device() == DeviceKind::Cpu {
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

    let request_models = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
        usize,
        Arc<std::sync::Mutex<bloomai_engine::executor::candle::QwenModelWrapper>>,
    >::new()));
    let request_models_for_free = Arc::clone(&request_models);
    kv_pool.set_on_free(move |handle| {
        if let Ok(mut models) = request_models_for_free.lock() {
            models.remove(&handle);
        }
    });

    let pipeline_for_forward = Arc::clone(&pipeline);
    let request_models_for_forward = Arc::clone(&request_models);
    let forward_fn = Box::new(
        move |input_ids: &candle_core::Tensor,
              start_pos: usize,
              kv_handle: Option<usize>|
              -> Result<candle_core::Tensor> {
            let handle = kv_handle.ok_or_else(|| {
                BloomError::Engine("scheduler request is missing its KV cache handle".into())
            })?;
            let model = {
                let mut models = request_models_for_forward
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                use std::collections::hash_map::Entry;
                Arc::clone(match models.entry(handle) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => {
                        let wrapper = pipeline_for_forward.model().create_wrapper()?;
                        let model = *wrapper
                            .downcast::<bloomai_engine::executor::candle::QwenModelWrapper>()
                            .map_err(|_| {
                                BloomError::Engine("failed to downcast model wrapper".into())
                            })?;
                        entry.insert(Arc::new(std::sync::Mutex::new(model)))
                    }
                })
            };
            let result = model
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .forward(input_ids, start_pos);
            result
        },
    );

    let pipeline_for_batch = Arc::clone(&pipeline);
    let request_models_for_batch = Arc::clone(&request_models);
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
            if cu_seqlens.len() != batch_size + 1 {
                return Err(anyhow!("invalid continuous-batching sequence offsets"));
            }
            let mut models_to_run = Vec::with_capacity(batch_size);
            {
                let mut models = request_models_for_batch
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                use std::collections::hash_map::Entry;
                for &handle in kv_handles {
                    let model = match models.entry(handle) {
                        Entry::Occupied(entry) => entry.into_mut(),
                        Entry::Vacant(entry) => {
                            let wrapper = pipeline_for_batch.model().create_wrapper()?;
                            let model = *wrapper
                                .downcast::<bloomai_engine::executor::candle::QwenModelWrapper>()
                                .map_err(|_| {
                                    BloomError::Engine("failed to downcast model wrapper".into())
                                })?;
                            entry.insert(Arc::new(std::sync::Mutex::new(model)))
                        }
                    };
                    models_to_run.push(Arc::clone(model));
                }
            }
            let mut logits = Vec::with_capacity(batch_size);
            for (index, model) in models_to_run.iter().enumerate() {
                let start = cu_seqlens[index];
                let end = cu_seqlens[index + 1];
                let sequence_len = end.checked_sub(start).ok_or_else(|| {
                    anyhow!("continuous-batching sequence offsets are not ordered")
                })?;
                let start_pos = start_positions.get(index).copied().unwrap_or(0);
                let request_input = input_ids.narrow(0, start, sequence_len)?.unsqueeze(0)?;
                let result = model
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .forward(&request_input, start_pos)?;
                logits.push(result.squeeze(0)?);
            }
            candle_core::Tensor::cat(&logits, 0).map_err(Into::into)
        },
    );

    let cachemesh = if args.enable_cachemesh {
        let config = CacheMeshConfig {
            enabled: true,
            namespace: model_id.to_string(),
            l2_capacity_bytes: args.cachemesh_l2_capacity_bytes,
            l3_enabled: args.enable_cachemesh_l3,
            write_through_l3: args.cachemesh_write_through_l3,
        };
        let mesh = if args.enable_cachemesh_l3 {
            let remote: Arc<dyn bloomai_engine::RemoteCacheBackend> =
                if let Some(path) = &args.cachemesh_l3_path {
                    Arc::new(FileSystemRemoteCache::new(path)?)
                } else {
                    Arc::new(InMemoryRemoteCache::new())
                };
            CacheMesh::with_remote(config, remote)
        } else {
            CacheMesh::new(config)
        };
        Some(Arc::new(mesh))
    } else {
        None
    };

    state.load_progress.store(85, Ordering::Release);
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
        cachemesh.clone(),
    ));
    let executor = Arc::new({
        let model = pipeline.model();
        let base = CandleBatchExecutor::new(forward_fn, device, 4, 32)
            .with_cache(Arc::clone(&paged_cache))
            .with_vocab_and_tokenizer(
                model.vocab_strings().to_vec(),
                model.eos_token_ids().to_vec(),
                model.tokenizer().cloned(),
            )
            .with_forward_batch_fn(forward_batch_fn);
        if model.supports_paged_kv() {
            let hook = Arc::new(ServerKvHook::new(
                Arc::clone(&request_models),
                num_layers,
                num_kv_heads,
                head_dim,
            ));
            base.with_kv_hook(hook as Arc<dyn bloomai_engine::scheduler::kv_hook::KvHook>)
        } else {
            base
        }
    });
    let mut scheduling_config = TokenSchedulingConfig {
        max_total_tokens_per_step: args.max_num_tokens,
        ..Default::default()
    };
    scheduling_config.chunked_prefill.enabled = args.enable_chunked_prefill;
    scheduling_config.chunked_prefill.chunk_size = args.prefill_chunk_size;
    let scheduler = Arc::new(InferenceScheduler::with_config(
        executor,
        Arc::clone(&kv_pool) as Arc<dyn KvCachePool>,
        scheduling_config,
    ));

    let scheduler_for_worker = Arc::clone(&scheduler);
    let worker_shutdown = shutdown.clone();
    tokio::spawn(async move {
        tracing::info!("Starting continuous-batching scheduler worker");
        loop {
            tokio::select! {
                _ = worker_shutdown.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_millis(5)) => {
                    if let Err(error) = scheduler_for_worker.step() {
                        tracing::error!(%error, "Scheduler step failed");
                    }
                }
            }
        }
        tracing::info!("Continuous-batching scheduler worker stopped");
    });

    Ok(SchedulingRuntime {
        kv_cache_pool: Some(kv_pool),
        cachemesh,
        scheduler: Some(scheduler),
        memory_reservation,
        shutdown,
    })
}

// ─── Graceful shutdown ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShutdownSignal {
    Interrupt,
    #[cfg(unix)]
    Terminate,
}

impl ShutdownSignal {
    fn label(self) -> &'static str {
        match self {
            Self::Interrupt => "interrupt",
            #[cfg(unix)]
            Self::Terminate => "terminate",
        }
    }
}

#[cfg(unix)]
struct ShutdownSignalListener {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl ShutdownSignalListener {
    fn install() -> std::io::Result<Self> {
        Ok(Self {
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
        })
    }

    async fn recv(&mut self) -> std::io::Result<ShutdownSignal> {
        tokio::select! {
            signal = self.interrupt.recv() => signal
                .map(|()| ShutdownSignal::Interrupt)
                .ok_or_else(|| std::io::Error::other("SIGINT listener closed unexpectedly")),
            signal = self.terminate.recv() => signal
                .map(|()| ShutdownSignal::Terminate)
                .ok_or_else(|| std::io::Error::other("SIGTERM listener closed unexpectedly")),
        }
    }
}

#[cfg(windows)]
struct ShutdownSignalListener {
    interrupt: tokio::signal::windows::CtrlC,
}

#[cfg(windows)]
impl ShutdownSignalListener {
    fn install() -> std::io::Result<Self> {
        Ok(Self {
            interrupt: tokio::signal::windows::ctrl_c()?,
        })
    }

    async fn recv(&mut self) -> std::io::Result<ShutdownSignal> {
        self.interrupt
            .recv()
            .await
            .map(|()| ShutdownSignal::Interrupt)
            .ok_or_else(|| std::io::Error::other("Ctrl-C listener closed unexpectedly"))
    }
}

#[cfg(not(any(unix, windows)))]
struct ShutdownSignalListener;

#[cfg(not(any(unix, windows)))]
impl ShutdownSignalListener {
    fn install() -> std::io::Result<Self> {
        Ok(Self)
    }

    async fn recv(&mut self) -> std::io::Result<ShutdownSignal> {
        tokio::signal::ctrl_c().await?;
        Ok(ShutdownSignal::Interrupt)
    }
}

async fn wait_for_shutdown_notification(mut receiver: watch::Receiver<bool>) {
    while !*receiver.borrow() {
        if receiver.changed().await.is_err() {
            break;
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ShutdownWait<T> {
    Completed(T),
    TimedOut,
}

async fn wait_for_server_or_shutdown_timeout<F, T>(
    server: F,
    shutdown_receiver: watch::Receiver<bool>,
    timeout: Duration,
) -> ShutdownWait<T>
where
    F: std::future::Future<Output = T>,
{
    let deadline = async move {
        wait_for_shutdown_notification(shutdown_receiver).await;
        tokio::time::sleep(timeout).await;
    };
    tokio::pin!(server);
    tokio::pin!(deadline);
    tokio::select! {
        result = &mut server => ShutdownWait::Completed(result),
        () = &mut deadline => ShutdownWait::TimedOut,
    }
}

// ─── Health / readiness ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_download::ModelDownloadPhase;
    use bloomai_engine::scheduler::kv_hook::KvHook;
    use sha2::{Digest as _, Sha256};
    use tower::ServiceExt as _;

    struct StreamDropFlag(Arc<AtomicBool>);

    impl Drop for StreamDropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    struct TestEmbeddingEngine {
        native_batch_calls: Arc<AtomicU64>,
    }

    impl bloomai_engine::Engine for TestEmbeddingEngine {
        fn name(&self) -> &'static str {
            "test-embedding"
        }

        fn supported_modalities(&self) -> Vec<bloomai_core::Modality> {
            vec![bloomai_core::Modality::Text]
        }

        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }

        fn load(
            &self,
            _model_path: &Path,
            _device: DeviceKind,
        ) -> Result<Box<dyn bloomai_engine::LoadedModel>> {
            let mut manifest = bloomai_core::ModelManifest {
                id: "test-embed-model".to_string(),
                ..bloomai_core::ModelManifest::default()
            };
            manifest.parameters.insert(
                "bloom_task".to_string(),
                serde_json::Value::String("embedding".to_string()),
            );
            Ok(Box::new(TestEmbeddingModel {
                native_batch_calls: Arc::clone(&self.native_batch_calls),
                metadata: bloomai_engine::ModelMetadata {
                    id: "test-embed-model".to_string(),
                    modality: bloomai_core::Modality::Text,
                    quantized: false,
                    manifest,
                },
            }))
        }
    }

    struct TestEmbeddingModel {
        native_batch_calls: Arc<AtomicU64>,
        metadata: bloomai_engine::ModelMetadata,
    }

    fn test_embedding_values(prompt: &str) -> Vec<f32> {
        if prompt.contains("banana") {
            vec![0.0, 2.0, 0.0]
        } else {
            vec![3.0, 0.0, 0.0]
        }
    }

    impl bloomai_engine::LoadedModel for TestEmbeddingModel {
        fn metadata(&self) -> &bloomai_engine::ModelMetadata {
            &self.metadata
        }

        fn supports_native_embedding_batch(&self) -> bool {
            true
        }

        fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            self.native_batch_calls.fetch_add(1, Ordering::Relaxed);
            Ok(inputs
                .iter()
                .map(|prompt| test_embedding_values(prompt))
                .collect())
        }

        fn infer(
            &self,
            _input: ModelInput,
            _params: &GenerationParams,
        ) -> Result<bloomai_engine::ModelOutput> {
            Ok(bloomai_engine::ModelOutput {
                text: None,
                logits: None,
                image: None,
                audio: None,
                video: None,
            })
        }

        fn infer_stream(
            &self,
            input: ModelInput,
            _params: &GenerationParams,
            sink: &mut dyn bloomai_engine::model::OutputSink,
        ) -> Result<()> {
            let ModelInput::Text { prompt } = input else {
                return Err(anyhow!("test embedding model requires text"));
            };
            let embedding = test_embedding_values(&prompt);
            sink.on_chunk(OutputChunk::Embedding(embedding))?;
            sink.on_chunk(OutputChunk::End)?;
            Ok(())
        }
    }

    struct TestTextEngine {
        emitted_chunks: Arc<AtomicU64>,
    }

    impl bloomai_engine::Engine for TestTextEngine {
        fn name(&self) -> &'static str {
            "test-text"
        }

        fn supported_modalities(&self) -> Vec<bloomai_core::Modality> {
            vec![bloomai_core::Modality::Text]
        }

        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }

        fn load(
            &self,
            _model_path: &Path,
            _device: DeviceKind,
        ) -> Result<Box<dyn bloomai_engine::LoadedModel>> {
            let manifest = bloomai_core::ModelManifest {
                id: "test-text-model".to_string(),
                ..bloomai_core::ModelManifest::default()
            };
            Ok(Box::new(TestTextModel {
                metadata: bloomai_engine::ModelMetadata {
                    id: "test-text-model".to_string(),
                    modality: bloomai_core::Modality::Text,
                    quantized: false,
                    manifest,
                },
                emitted_chunks: Arc::clone(&self.emitted_chunks),
            }))
        }
    }

    struct TestTextModel {
        metadata: bloomai_engine::ModelMetadata,
        emitted_chunks: Arc<AtomicU64>,
    }

    impl bloomai_engine::LoadedModel for TestTextModel {
        fn metadata(&self) -> &bloomai_engine::ModelMetadata {
            &self.metadata
        }

        fn infer(
            &self,
            _input: ModelInput,
            _params: &GenerationParams,
        ) -> Result<bloomai_engine::ModelOutput> {
            Ok(bloomai_engine::ModelOutput {
                text: Some("visible STOP hidden".to_string()),
                logits: None,
                image: None,
                audio: None,
                video: None,
            })
        }

        fn infer_stream(
            &self,
            _input: ModelInput,
            _params: &GenerationParams,
            sink: &mut dyn bloomai_engine::model::OutputSink,
        ) -> Result<()> {
            for text in ["visible ST", "OP hidden", " never emitted"] {
                self.emitted_chunks.fetch_add(1, Ordering::Relaxed);
                sink.on_chunk(OutputChunk::TextDelta(text.to_string()))?;
            }
            sink.on_chunk(OutputChunk::End)?;
            Ok(())
        }
    }

    #[test]
    fn shutdown_signal_labels_are_bounded_and_stable() {
        assert_eq!(ShutdownSignal::Interrupt.label(), "interrupt");
        #[cfg(unix)]
        assert_eq!(ShutdownSignal::Terminate.label(), "terminate");
    }

    #[tokio::test]
    async fn shutdown_wait_returns_when_the_server_finishes_without_a_signal() {
        let (_sender, receiver) = watch::channel(false);
        let outcome = wait_for_server_or_shutdown_timeout(
            futures::future::ready(7_u8),
            receiver,
            Duration::from_secs(1),
        )
        .await;

        assert_eq!(outcome, ShutdownWait::Completed(7));
    }

    #[tokio::test]
    async fn shutdown_wait_enforces_the_deadline_after_notification() {
        let (sender, receiver) = watch::channel(false);
        sender.send(true).unwrap();
        let outcome = wait_for_server_or_shutdown_timeout(
            futures::future::pending::<()>(),
            receiver,
            Duration::from_millis(1),
        )
        .await;

        assert_eq!(outcome, ShutdownWait::TimedOut);
    }

    #[test]
    fn model_index_state_defaults_next_to_the_effective_configuration() {
        assert_eq!(
            default_model_index_state_directory(Path::new("/srv/bloom/config.json")),
            PathBuf::from("/srv/bloom/model-index-watermarks")
        );
        assert_eq!(
            default_model_index_state_directory(Path::new("config.json")),
            PathBuf::from("model-index-watermarks")
        );
    }

    #[test]
    fn chat_stream_options_decode_and_usage_payload_matches_openai_shape() {
        let request = serde_json::from_value::<ChatRequest>(serde_json::json!({
            "model": "tiny.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .unwrap();
        assert_eq!(request.model.as_deref(), Some("tiny.gguf"));
        assert!(request.stream_options.unwrap().include_usage);

        let completion = serde_json::from_value::<CompletionRequest>(serde_json::json!({
            "model": "default",
            "prompt": "Hello"
        }))
        .unwrap();
        assert_eq!(completion.model.as_deref(), Some("default"));

        let payload =
            helpers::chat_usage_payload("chatcmpl-1".into(), "tiny".into(), 1_700_000_000, 12, 4);
        assert_eq!(payload["object"], "chat.completion.chunk");
        assert_eq!(payload["created"], 1_700_000_000);
        assert_eq!(payload["choices"], serde_json::json!([]));
        assert_eq!(payload["usage"]["prompt_tokens"], 12);
        assert_eq!(payload["usage"]["completion_tokens"], 4);
        assert_eq!(payload["usage"]["total_tokens"], 16);
    }

    #[test]
    fn chat_message_admission_enforces_roles_counts_and_content_budgets() {
        let valid = vec![
            NormalizedChatMessage {
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            NormalizedChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        handlers::validate_chat_messages(&valid).unwrap();
        assert!(handlers::validate_chat_messages(&[]).is_err());

        let invalid_role = vec![NormalizedChatMessage {
            role: "tool".to_string(),
            content: "output".to_string(),
        }];
        assert!(handlers::validate_chat_messages(&invalid_role).is_err());

        let too_many = vec![
            NormalizedChatMessage {
                role: "assistant".to_string(),
                content: String::new(),
            };
            MAX_CHAT_REQUEST_MESSAGES + 1
        ];
        assert!(handlers::validate_chat_messages(&too_many).is_err());

        let oversized_user = vec![NormalizedChatMessage {
            role: "user".to_string(),
            content: "x".repeat(MAX_CHAT_USER_MESSAGE_CHARS + 1),
        }];
        assert!(handlers::validate_chat_messages(&oversized_user).is_err());

        let oversized_system = vec![NormalizedChatMessage {
            role: "system".to_string(),
            content: "x".repeat(MAX_CHAT_SYSTEM_MESSAGE_CHARS + 1),
        }];
        assert!(handlers::validate_chat_messages(&oversized_system).is_err());

        let oversized_history = vec![
            NormalizedChatMessage {
                role: "assistant".to_string(),
                content: "x".repeat(MAX_CHAT_CONTENT_BYTES / 2 + 1),
            },
            NormalizedChatMessage {
                role: "assistant".to_string(),
                content: "x".repeat(MAX_CHAT_CONTENT_BYTES / 2 + 1),
            },
        ];
        assert!(handlers::validate_chat_messages(&oversized_history).is_err());
    }

    #[test]
    fn chat_content_normalization_accepts_bounded_text_parts_only() {
        let request = serde_json::from_value::<ChatRequest>(json!({
            "messages": [
                {"role": "system", "content": "Be concise."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello", "future": null},
                        {"type": "text", "text": " world"}
                    ]
                }
            ]
        }))
        .unwrap();
        let normalized = helpers::normalize_chat_messages(&request.messages).unwrap();
        assert_eq!(
            normalized,
            vec![
                NormalizedChatMessage {
                    role: "system".to_string(),
                    content: "Be concise.".to_string(),
                },
                NormalizedChatMessage {
                    role: "user".to_string(),
                    content: "Hello world".to_string(),
                },
            ]
        );
        handlers::validate_chat_messages(&normalized).unwrap();

        for content in [
            json!(null),
            json!({"type": "text", "text": "Hello"}),
            json!([]),
            json!(["Hello"]),
            json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}]),
            json!([{"type": "text", "text": 7}]),
            json!([{"type": "text", "text": "Hello", "cache_control": {}}]),
        ] {
            let request = serde_json::from_value::<ChatRequest>(json!({
                "messages": [{"role": "user", "content": content}]
            }))
            .unwrap();
            assert!(helpers::normalize_chat_messages(&request.messages).is_err());
        }

        let request = serde_json::from_value::<ChatRequest>(json!({
            "messages": [{
                "role": "user",
                "content": vec![json!({"type": "text", "text": "x"}); MAX_CHAT_CONTENT_PARTS + 1]
            }]
        }))
        .unwrap();
        assert!(helpers::normalize_chat_messages(&request.messages).is_err());

        let request = serde_json::from_value::<ChatRequest>(json!({
            "messages": [
                {"role": "assistant", "content": "x".repeat(MAX_CHAT_CONTENT_BYTES / 2 + 1)},
                {"role": "assistant", "content": "x".repeat(MAX_CHAT_CONTENT_BYTES / 2 + 1)}
            ]
        }))
        .unwrap();
        assert!(helpers::normalize_chat_messages(&request.messages).is_err());
    }

    #[test]
    fn developer_messages_are_explicit_leading_system_aliases() {
        let request = serde_json::from_value::<ChatRequest>(json!({
            "messages": [
                {"role": "developer", "content": "Follow the local policy."},
                {"role": "user", "content": "Hello"}
            ]
        }))
        .unwrap();
        let normalized = helpers::normalize_chat_messages(&request.messages).unwrap();
        assert_eq!(normalized[0].role, "system");
        assert_eq!(normalized[0].content, "Follow the local policy.");
        handlers::validate_chat_messages(&normalized).unwrap();

        let late = serde_json::from_value::<ChatRequest>(json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "developer", "content": "Late policy"}
            ]
        }))
        .unwrap();
        assert!(helpers::normalize_chat_messages(&late.messages)
            .unwrap_err()
            .contains("must appear before"));
    }

    #[test]
    fn responses_text_subset_normalizes_and_rejects_active_semantics() {
        let neutral = serde_json::from_value::<ResponsesRequest>(json!({
            "model": "default",
            "instructions": "Be concise.",
            "input": [{
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": " world", "future": null}
                ],
                "future": null
            }],
            "max_output_tokens": 32,
            "background": false,
            "store": false,
            "tools": [],
            "tool_choice": "none",
            "parallel_tool_calls": false,
            "metadata": {},
            "include": [],
            "top_logprobs": 0,
            "truncation": "disabled",
            "service_tier": "default",
            "text": {"format": {"type": "text"}},
            "safety_identifier": "local-client"
        }))
        .unwrap();
        helpers::validate_responses_request_compatibility(&neutral).unwrap();
        assert_eq!(
            helpers::responses_metadata(neutral.metadata.as_ref()).unwrap(),
            json!({})
        );
        let (text_response_format, normalized_text_format) =
            helpers::responses_text_format(neutral.text.as_ref()).unwrap();
        assert!(text_response_format.is_none());
        assert_eq!(normalized_text_format, json!({"type": "text"}));
        let messages =
            helpers::responses_chat_messages(neutral.input.clone(), neutral.instructions.clone())
                .unwrap();
        let normalized = helpers::normalize_chat_messages(&messages).unwrap();
        assert_eq!(normalized[0].role, "system");
        assert_eq!(normalized[0].content, "Be concise.");
        assert_eq!(normalized[1].role, "user");
        assert_eq!(normalized[1].content, "Hello world");
        let input_items = helpers::responses_input_items(&messages[1..], "resp-example").unwrap();
        assert_eq!(input_items[0]["id"], "msg-example-input-0");
        assert_eq!(input_items[0]["status"], "completed");
        assert_eq!(input_items[0]["content"][0]["type"], "input_text");

        let streaming = serde_json::from_value::<ResponsesRequest>(json!({
            "input": "Hello",
            "stream": true
        }))
        .unwrap();
        helpers::validate_responses_request_compatibility(&streaming).unwrap();

        let structured = serde_json::from_value::<ResponsesRequest>(json!({
            "input": "Return one answer.",
            "text": {"format": {
                "type": "json_schema",
                "name": "answer",
                "description": "One bounded answer.",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                },
                "strict": true
            }}
        }))
        .unwrap();
        let (chat_format, normalized_format) =
            helpers::responses_text_format(structured.text.as_ref()).unwrap();
        assert!(matches!(
            helpers::response_format_mode(chat_format.as_ref()).unwrap(),
            ResponseFormatMode::JsonSchema(_)
        ));
        assert_eq!(normalized_format["type"], "json_schema");
        assert_eq!(normalized_format["name"], "answer");
        assert_eq!(normalized_format["strict"], true);

        for text in [
            json!("json"),
            json!({"verbosity": "high"}),
            json!({"format": {"type": "xml"}}),
            json!({"format": {"type": "json_object", "schema": {}}}),
            json!({"format": {"type": "json_schema", "schema": {"type": "object"}}}),
            json!({"format": {
                "type": "json_schema",
                "name": "bad name",
                "schema": {"type": "object"}
            }}),
            json!({"format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "strict": "true"
            }}),
            json!({"format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object", "minProperties": 1}
            }}),
        ] {
            assert!(helpers::responses_text_format(Some(&text)).is_err());
        }

        let invalid_tool = serde_json::from_value::<ResponsesRequest>(
            json!({"input": "Hello", "tools": [{"type": "function"}]}),
        )
        .unwrap();
        helpers::validate_responses_request_compatibility(&invalid_tool).unwrap();
        assert!(tool_calling::responses_tool_bridge(
            invalid_tool.tools.as_ref(),
            invalid_tool.tool_choice.as_ref(),
            invalid_tool.parallel_tool_calls,
        )
        .is_err());
        let unsupported = serde_json::from_value::<ResponsesRequest>(
            json!({"input": "Hello", "truncation": "auto"}),
        )
        .unwrap();
        assert!(helpers::validate_responses_request_compatibility(&unsupported).is_err());
        for body in [
            json!({"input": "Hello", "store": true}),
            json!({"input": "Hello", "previous_response_id": "resp-1"}),
        ] {
            let request = serde_json::from_value::<ResponsesRequest>(body).unwrap();
            helpers::validate_responses_request_compatibility(&request).unwrap();
        }

        for input in [
            json!(null),
            json!([]),
            json!([{"type": "function_call", "name": "lookup"}]),
            json!([{"type": "message", "role": "tool", "content": "output"}]),
            json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,"}]
            }]),
        ] {
            assert!(helpers::responses_chat_messages(input, None).is_err());
        }
        for metadata in [
            json!([]),
            json!({"key": 1}),
            json!({"": "value"}),
            json!({"key": "x".repeat(513)}),
            serde_json::Value::Object(
                (0..17)
                    .map(|index| (format!("key-{index}"), json!("value")))
                    .collect(),
            ),
        ] {
            assert!(helpers::responses_metadata(Some(&metadata)).is_err());
        }
    }

    #[test]
    fn responses_payload_is_sdk_shaped_and_reports_length_limits() {
        let chat = json!({
            "id": "chatcmpl-1700000000-7",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": "tiny.gguf",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            }
        });
        let response = helpers::responses_payload_from_chat(
            &chat,
            Some("Be concise."),
            32,
            0.2,
            0.9,
            &json!({"type": "text"}),
            &helpers::ResponsesStateOptions::default(),
        )
        .unwrap();
        assert_eq!(response["id"], "resp-1700000000-7");
        assert_eq!(response["object"], "response");
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"][0]["id"], "msg-1700000000-7");
        assert_eq!(response["output"][0]["content"][0]["text"], "Hello");
        assert_eq!(response["usage"]["input_tokens"], 3);
        assert_eq!(response["usage"]["output_tokens"], 2);
        assert_eq!(response["usage"]["total_tokens"], 5);
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            0
        );
        assert!(response["incomplete_details"].is_null());
        assert_eq!(response["text"]["format"]["type"], "text");
        assert_eq!(response["store"], false);
        assert!(response["previous_response_id"].is_null());

        let retained = helpers::responses_payload_from_chat(
            &chat,
            None,
            32,
            0.2,
            0.9,
            &json!({"type": "text"}),
            &helpers::ResponsesStateOptions::new(true, Some("resp-previous".to_string()))
                .with_metadata(json!({"session": "local"})),
        )
        .unwrap();
        assert_eq!(retained["store"], true);
        assert_eq!(retained["previous_response_id"], "resp-previous");
        assert_eq!(retained["metadata"], json!({"session": "local"}));

        let mut limited = chat;
        limited["choices"][0]["finish_reason"] = json!("length");
        let response = helpers::responses_payload_from_chat(
            &limited,
            None,
            2,
            0.7,
            0.9,
            &json!({"type": "text"}),
            &helpers::ResponsesStateOptions::default(),
        )
        .unwrap();
        assert_eq!(response["status"], "incomplete");
        assert_eq!(response["output"][0]["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );

        assert_eq!(helpers::generation_finish_reason(1, 2), "stop");
        assert_eq!(helpers::generation_finish_reason(2, 2), "length");
    }

    #[test]
    fn responses_function_tools_bridge_native_items_and_history() {
        let tools = json!([{
            "type": "function",
            "name": "lookup_weather",
            "description": "Look up local weather data.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            },
            "strict": true
        }]);
        let choice = json!({"type": "function", "name": "lookup_weather"});
        let bridge =
            tool_calling::responses_tool_bridge(Some(&tools), Some(&choice), Some(false)).unwrap();
        assert_eq!(bridge.chat_tools.as_ref().unwrap()[0]["type"], "function");
        assert_eq!(
            bridge.chat_tools.as_ref().unwrap()[0]["function"]["name"],
            "lookup_weather"
        );
        assert_eq!(
            bridge.chat_tool_choice.as_ref().unwrap()["function"]["name"],
            "lookup_weather"
        );
        assert_eq!(bridge.response_tools, tools);
        assert_eq!(bridge.response_tool_choice, choice);
        assert!(!bridge.response_parallel_tool_calls);

        let input = helpers::normalize_responses_input(
            json!([
                {"type": "message", "role": "user", "content": "Compare two cities."},
                {
                    "id": "fc-first",
                    "type": "function_call",
                    "call_id": "call_first",
                    "name": "lookup_weather",
                    "arguments": "{\"city\":\"Paris\"}"
                },
                {
                    "id": "fc-second",
                    "type": "function_call",
                    "call_id": "call_second",
                    "name": "lookup_weather",
                    "arguments": "{\"city\":\"Berlin\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_first",
                    "output": "{\"temperature_c\":20}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_second",
                    "output": "{\"temperature_c\":18}"
                }
            ]),
            "resp-native",
        )
        .unwrap();
        assert_eq!(input.messages.len(), 4);
        assert_eq!(
            input.messages[1].extensions["tool_calls"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(input.items[1]["id"], "fc-first");
        assert_eq!(input.items[3]["type"], "function_call_output");
        helpers::normalize_chat_messages(&input.messages).unwrap();

        let chat = json!({
            "id": "resp-native-output",
            "object": "chat.completion",
            "created": 15_u64,
            "model": "tiny.gguf",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_native_0",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": "{\"city\":\"Paris\"}"
                            }
                        },
                        {
                            "id": "call_native_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": "{\"city\":\"Berlin\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
        });
        let response = helpers::responses_payload_from_chat(
            &chat,
            None,
            32,
            0.7,
            0.9,
            &json!({"type": "text"}),
            &helpers::ResponsesStateOptions::default().with_tools(&bridge),
        )
        .unwrap();
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"].as_array().unwrap().len(), 2);
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["id"], "fc-native-output-0");
        assert_eq!(response["output"][0]["call_id"], "call_native_0");
        assert_eq!(response["output"][1]["name"], "lookup_weather");
        assert_eq!(response["tool_choice"], choice);
        assert_eq!(response["tools"], tools);

        let mut history = Vec::new();
        helpers::append_responses_output_to_history(&mut history, &response).unwrap();
        assert_eq!(history.len(), 1);
        assert!(history[0].content.is_null());
        assert_eq!(
            history[0].extensions["tool_calls"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let store = response_store::ResponseStore::default();
        response_store::PendingResponseStorage::new(
            store.clone(),
            vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: json!("Compare two cities."),
                extensions: BTreeMap::new(),
            }],
            vec![json!({
                "id": "msg-native-user",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": "Compare two cities."}]
            })],
        )
        .commit(&response)
        .unwrap();
        let stored = store.get("resp-native-output").unwrap();
        assert_eq!(stored.history.len(), 2);
        let continuation = helpers::normalize_responses_input(
            json!([
                {
                    "type": "function_call_output",
                    "call_id": "call_native_0",
                    "output": "{\"temperature_c\":20}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_native_1",
                    "output": "{\"temperature_c\":18}"
                }
            ]),
            "resp-continuation",
        )
        .unwrap();
        let mut chained_history = stored.history;
        chained_history.extend(continuation.messages);
        helpers::normalize_chat_messages(&chained_history).unwrap();
        assert_eq!(continuation.items[0]["type"], "function_call_output");

        for invalid in [
            json!([{"type": "function_call", "call_id": "call_1", "name": "bad name", "arguments": "{}"}]),
            json!([{"type": "function_call", "call_id": "call_1", "name": "lookup_weather", "arguments": "[]"}]),
            json!([{"type": "function_call_output", "call_id": "call_1", "output": {}}]),
        ] {
            assert!(helpers::normalize_responses_input(invalid, "resp-invalid").is_err());
        }
        assert!(tool_calling::responses_tool_bridge(
            Some(&json!([{"type": "web_search"}])),
            None,
            None,
        )
        .is_err());
        assert!(tool_calling::responses_tool_bridge(
            Some(&tools),
            Some(&json!({"type": "function", "name": "missing"})),
            None,
        )
        .is_err());
    }

    #[test]
    fn responses_structured_format_is_preserved_and_stream_validation_fails_closed() {
        let text_format = json!({
            "type": "json_schema",
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            },
            "strict": true
        });
        let chat = json!({
            "id": "resp-structured",
            "object": "chat.completion",
            "created": 12_u64,
            "model": "tiny.gguf",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "{\"answer\":\"Bloom\"}"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        });
        let response = helpers::responses_payload_from_chat(
            &chat,
            None,
            8,
            0.7,
            0.9,
            &text_format,
            &helpers::ResponsesStateOptions::default(),
        )
        .unwrap();
        assert_eq!(response["text"]["format"], text_format);

        let mut valid_adapter = helpers::ResponsesStreamAdapter::new(
            "resp-structured-valid".to_string(),
            None,
            8,
            0.7,
            0.9,
            text_format.clone(),
            helpers::ResponsesStateOptions::default(),
        );
        let opening = valid_adapter
            .ingest_chat_payload(json!({
                "id": "resp-structured-valid",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            }))
            .unwrap();
        assert_eq!(opening[0].data["response"]["text"]["format"], text_format);
        for payload in [
            json!({
                "id": "resp-structured-valid",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "{\"answer\":\"Bloom\"}"},
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "resp-structured-valid",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }),
            json!({
                "id": "resp-structured-valid",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
            }),
        ] {
            valid_adapter.ingest_chat_payload(payload).unwrap();
        }
        let completed = valid_adapter.finish().unwrap().pop().unwrap();
        assert_eq!(completed.event_type, "response.completed");
        assert_eq!(completed.data["response"]["text"]["format"], text_format);

        let mut adapter = helpers::ResponsesStreamAdapter::new(
            "resp-structured".to_string(),
            None,
            8,
            0.7,
            0.9,
            text_format.clone(),
            helpers::ResponsesStateOptions::default(),
        );
        for payload in [
            json!({
                "id": "resp-structured",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            }),
            json!({
                "id": "resp-structured",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {"content": "not JSON"}, "finish_reason": null}]
            }),
            json!({
                "id": "resp-structured",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }),
            json!({
                "id": "resp-structured",
                "object": "chat.completion.chunk",
                "created": 12_u64,
                "model": "tiny.gguf",
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            }),
        ] {
            adapter.ingest_chat_payload(payload).unwrap();
        }
        let error = adapter.finish().unwrap_err();
        assert!(error.contains("structured output validation failed"));
        let failed = adapter.failure_events(&error).unwrap().pop().unwrap();
        assert_eq!(failed.event_type, "response.failed");
        assert_eq!(failed.data["response"]["status"], "failed");
        assert_eq!(failed.data["response"]["text"]["format"], text_format);
        assert_eq!(
            failed.data["response"]["output"][0]["content"][0]["text"],
            "not JSON"
        );
    }

    #[test]
    fn responses_stream_adapter_emits_ordered_complete_and_incomplete_events() {
        fn chat_chunk(choice: serde_json::Value) -> serde_json::Value {
            json!({
                "id": "resp-1700000000-9",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000_u64,
                "model": "tiny.gguf",
                "choices": [choice]
            })
        }

        let mut adapter = helpers::ResponsesStreamAdapter::new(
            "resp-1700000000-9".to_string(),
            Some("Be concise.".to_string()),
            8,
            0.2,
            0.9,
            json!({"type": "text"}),
            helpers::ResponsesStateOptions::default(),
        );
        let mut events = adapter
            .ingest_chat_payload(chat_chunk(json!({
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null
            })))
            .unwrap();
        events.extend(
            adapter
                .ingest_chat_payload(chat_chunk(json!({
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": null
                })))
                .unwrap(),
        );
        assert!(adapter
            .ingest_chat_payload(chat_chunk(json!({
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            })))
            .unwrap()
            .is_empty());
        assert!(adapter
            .ingest_chat_payload(json!({
                "id": "resp-1700000000-9",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000_u64,
                "model": "tiny.gguf",
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5
                }
            }))
            .unwrap()
            .is_empty());
        events.extend(adapter.finish().unwrap());

        let event_types = events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        assert_eq!(
            event_types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed"
            ]
        );
        for (sequence, event) in events.iter().enumerate() {
            assert_eq!(event.data["sequence_number"], sequence as u64);
            assert_eq!(event.data["type"], event.event_type);
        }
        assert_eq!(events[4].data["delta"], "Hello");
        let terminal = &events.last().unwrap().data["response"];
        assert_eq!(terminal["id"], "resp-1700000000-9");
        assert_eq!(terminal["status"], "completed");
        assert_eq!(terminal["output"][0]["content"][0]["text"], "Hello");
        assert_eq!(terminal["usage"]["total_tokens"], 5);
        assert!(adapter.is_terminal());

        let mut limited = helpers::ResponsesStreamAdapter::new(
            "resp-limited".to_string(),
            None,
            1,
            0.7,
            0.9,
            json!({"type": "text"}),
            helpers::ResponsesStateOptions::default(),
        );
        limited
            .ingest_chat_payload(json!({
                "id": "resp-limited",
                "object": "chat.completion.chunk",
                "created": 8,
                "model": "tiny.gguf",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null
                }]
            }))
            .unwrap();
        limited
            .ingest_chat_payload(json!({
                "id": "resp-limited",
                "object": "chat.completion.chunk",
                "created": 8,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
            }))
            .unwrap();
        limited
            .ingest_chat_payload(json!({
                "id": "resp-limited",
                "object": "chat.completion.chunk",
                "created": 8,
                "model": "tiny.gguf",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }))
            .unwrap();
        let terminal = limited.finish().unwrap().pop().unwrap();
        assert_eq!(terminal.event_type, "response.incomplete");
        assert_eq!(terminal.data["response"]["status"], "incomplete");
        assert_eq!(
            terminal.data["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    #[test]
    fn responses_stream_adapter_emits_native_parallel_function_events() {
        let tools = json!([{
            "type": "function",
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            },
            "strict": true
        }]);
        let bridge =
            tool_calling::responses_tool_bridge(Some(&tools), Some(&json!("required")), Some(true))
                .unwrap();
        let mut adapter = helpers::ResponsesStreamAdapter::new(
            "resp-function-stream".to_string(),
            None,
            32,
            0.7,
            0.9,
            json!({"type": "text"}),
            helpers::ResponsesStateOptions::default().with_tools(&bridge),
        );
        let mut events = adapter
            .ingest_chat_payload(json!({
                "id": "resp-function-stream",
                "object": "chat.completion.chunk",
                "created": 20_u64,
                "model": "tiny.gguf",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null
                }]
            }))
            .unwrap();
        events.extend(
            adapter
                .ingest_chat_payload(json!({
                    "id": "resp-function-stream",
                    "object": "chat.completion.chunk",
                    "created": 20_u64,
                    "model": "tiny.gguf",
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [
                            {
                                "index": 0,
                                "id": "call_stream_0",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{\"city\":\"Paris\"}"}
                            },
                            {
                                "index": 1,
                                "id": "call_stream_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{\"city\":\"Berlin\"}"}
                            }
                        ]},
                        "finish_reason": "tool_calls"
                    }]
                }))
                .unwrap(),
        );
        assert!(adapter
            .ingest_chat_payload(json!({
                "id": "resp-function-stream",
                "object": "chat.completion.chunk",
                "created": 20_u64,
                "model": "tiny.gguf",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
            }))
            .unwrap()
            .is_empty());
        events.extend(adapter.finish().unwrap());

        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type)
                .collect::<Vec<_>>(),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.output_item.done",
                "response.completed"
            ]
        );
        for (sequence, event) in events.iter().enumerate() {
            assert_eq!(event.data["sequence_number"], sequence as u64);
        }
        assert_eq!(events[2].data["item"]["arguments"], "");
        assert_eq!(events[3].data["delta"], "{\"city\":\"Paris\"}");
        assert_eq!(events[4].data["arguments"], "{\"city\":\"Paris\"}");
        let terminal = &events.last().unwrap().data["response"];
        assert_eq!(terminal["output"].as_array().unwrap().len(), 2);
        assert_eq!(terminal["output"][0]["type"], "function_call");
        assert_eq!(terminal["output"][1]["call_id"], "call_stream_1");
        assert_eq!(terminal["parallel_tool_calls"], true);
        assert_eq!(terminal["tool_choice"], "required");
        assert_eq!(terminal["tools"], tools);
    }

    #[test]
    fn responses_stream_decoder_and_failures_are_bounded_and_fail_closed() {
        let mut decoder = helpers::ChatSseDecoder::default();
        assert!(decoder
            .push(b"event: ignored\r\ndata: {\"value\":")
            .unwrap()
            .is_empty());
        assert_eq!(
            decoder.push(b"1}\r\n\r\ndata: [DO").unwrap(),
            vec!["{\"value\":1}".to_string()]
        );
        assert_eq!(decoder.push(b"NE]\n\n").unwrap(), vec!["[DONE]"]);
        decoder.finish().unwrap();

        let mut incomplete = helpers::ChatSseDecoder::default();
        incomplete.push(b"data: unfinished").unwrap();
        assert!(incomplete.finish().is_err());
        assert!(helpers::ChatSseDecoder::default()
            .push(&vec![b'x'; MAX_RESPONSES_STREAM_FRAME_BYTES + 1])
            .is_err());

        let mut adapter = helpers::ResponsesStreamAdapter::new(
            "resp-error".to_string(),
            None,
            8,
            0.7,
            0.9,
            json!({"type": "text"}),
            helpers::ResponsesStateOptions::default(),
        );
        adapter
            .ingest_chat_payload(json!({
                "id": "resp-error",
                "object": "chat.completion.chunk",
                "created": 9,
                "model": "tiny.gguf",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": null
                }]
            }))
            .unwrap();
        let events = adapter
            .ingest_chat_payload(json!({
                "error": {"message": "generation\nfailed", "type": "internal_error"}
            }))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "response.failed");
        assert_eq!(events[0].data["response"]["status"], "failed");
        assert_eq!(
            events[0].data["response"]["error"]["message"],
            "generation failed"
        );
        assert!(adapter.is_terminal());
    }

    #[tokio::test]
    async fn responses_stream_http_adapter_emits_named_sse_through_completion() {
        let chunk = |choices: serde_json::Value| {
            json!({
                "id": "resp-http-1",
                "object": "chat.completion.chunk",
                "created": 11,
                "model": "tiny.gguf",
                "choices": choices
            })
        };
        let frames = [
            chunk(json!([{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null
            }])),
            chunk(json!([{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": null
            }])),
            chunk(json!([{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }])),
            json!({
                "id": "resp-http-1",
                "object": "chat.completion.chunk",
                "created": 11,
                "model": "tiny.gguf",
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            }),
        ];
        let mut body = String::new();
        for frame in frames {
            body.push_str("data: ");
            body.push_str(&frame.to_string());
            body.push_str("\n\n");
        }
        body.push_str("data: [DONE]\n\n");
        let internal = axum::http::Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
            .body(axum::body::Body::from(body))
            .unwrap();
        let response_store = response_store::ResponseStore::default();
        let pending_storage = response_store::PendingResponseStorage::new(
            response_store.clone(),
            vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: json!("Hello"),
                extensions: BTreeMap::new(),
            }],
            vec![json!({
                "id": "msg-http-input-0",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            })],
        );
        let response = handlers::responses_stream_from_chat_response(
            internal,
            helpers::ResponsesStreamAdapter::new(
                "resp-http-1".to_string(),
                None,
                8,
                0.7,
                0.9,
                json!({"type": "text"}),
                helpers::ResponsesStateOptions::new(true, None)
                    .with_metadata(json!({"session": "stream"})),
            ),
            Some(pending_storage),
        );
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert!(response
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let bytes = axum::body::to_bytes(response.into_body(), 4 * MIB as usize)
            .await
            .unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(!text.contains("[DONE]"));
        let event_types = text
            .lines()
            .filter_map(|line| line.strip_prefix("event:"))
            .map(str::trim)
            .collect::<Vec<_>>();
        assert_eq!(
            event_types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed"
            ]
        );
        let data = text
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim)
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(data.len(), event_types.len());
        for (sequence, event) in data.iter().enumerate() {
            assert_eq!(event["sequence_number"], sequence as u64);
        }
        assert_eq!(data.last().unwrap()["response"]["status"], "completed");
        assert_eq!(
            data.last().unwrap()["response"]["output"][0]["content"][0]["text"],
            "Hello"
        );
        let stored = response_store.get("resp-http-1").unwrap();
        assert_eq!(stored.response["store"], true);
        assert_eq!(stored.response["metadata"]["session"], "stream");
        assert_eq!(stored.history.len(), 2);
        assert_eq!(stored.history[1].role, "assistant");
    }

    #[tokio::test]
    async fn responses_stream_disconnect_drops_the_owned_internal_body() {
        let dropped = Arc::new(AtomicBool::new(false));
        let stream = futures::stream::unfold(StreamDropFlag(Arc::clone(&dropped)), |guard| async {
            futures::future::pending::<()>().await;
            Some((
                Ok::<axum::body::Bytes, std::convert::Infallible>(axum::body::Bytes::new()),
                guard,
            ))
        });
        let internal = axum::http::Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(axum::body::Body::from_stream(stream))
            .unwrap();
        let response_store = response_store::ResponseStore::default();
        let pending_storage = response_store::PendingResponseStorage::new(
            response_store.clone(),
            vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: json!("Hello"),
                extensions: BTreeMap::new(),
            }],
            Vec::new(),
        );
        let response = handlers::responses_stream_from_chat_response(
            internal,
            helpers::ResponsesStreamAdapter::new(
                "resp-disconnect".to_string(),
                None,
                8,
                0.7,
                0.9,
                json!({"type": "text"}),
                helpers::ResponsesStateOptions::new(true, None),
            ),
            Some(pending_storage),
        );
        drop(response);

        tokio::time::timeout(Duration::from_secs(1), async {
            while !dropped.load(Ordering::Acquire) {
                task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(response_store.get("resp-disconnect").is_none());
    }

    #[test]
    fn chat_token_limit_alias_is_unambiguous() {
        assert_eq!(helpers::resolve_chat_max_tokens(None, None).unwrap(), 128);
        assert_eq!(
            helpers::resolve_chat_max_tokens(Some(32), None).unwrap(),
            32
        );
        assert_eq!(
            helpers::resolve_chat_max_tokens(None, Some(48)).unwrap(),
            48
        );
        assert_eq!(
            helpers::resolve_chat_max_tokens(Some(64), Some(64)).unwrap(),
            64
        );
        assert!(helpers::resolve_chat_max_tokens(Some(64), Some(65)).is_err());
    }

    #[test]
    fn stop_sequences_are_bounded_and_do_not_leak_across_chunks() {
        assert_eq!(
            helpers::normalize_stop_sequences(Some(&json!("END"))).unwrap(),
            vec!["END"]
        );
        assert_eq!(
            helpers::normalize_stop_sequences(Some(&json!(["END", "DONE✓"]))).unwrap(),
            vec!["END", "DONE✓"]
        );
        for invalid in [
            json!(""),
            json!(["a", "b", "c", "d", "e"]),
            json!(["ok", 3]),
            json!({"sequence": "END"}),
        ] {
            assert!(helpers::normalize_stop_sequences(Some(&invalid)).is_err());
        }

        let mut filter = helpers::StopSequenceFilter::new(vec!["STOP".into(), "DONE✓".into()]);
        assert_eq!(filter.push("alpha ST").text, "alpha ");
        let matched = filter.push("OP hidden");
        assert!(matched.stopped);
        assert!(matched.text.is_empty());
        assert!(filter.finish().is_empty());

        let mut unicode = helpers::StopSequenceFilter::new(vec!["DONE✓".into()]);
        assert_eq!(unicode.push("answer DONE").text, "answer ");
        assert!(unicode.push("✓ trailing").stopped);

        let mut natural = helpers::StopSequenceFilter::new(vec!["STOP".into()]);
        assert_eq!(natural.push("keep ST").text, "keep ");
        assert_eq!(natural.finish(), "ST");
    }

    #[test]
    fn openai_extension_admission_accepts_neutral_defaults_and_rejects_semantics() {
        let neutral = serde_json::from_value::<ChatRequest>(json!({
            "model": "default",
            "messages": [{
                "role": "user",
                "content": "Hello",
                "tool_calls": [],
                "future_message_field": null
            }],
            "stream": true,
            "stream_options": {
                "include_usage": true,
                "include_obfuscation": false,
                "future_stream_field": null
            },
            "n": 1,
            "best_of": 1,
            "stop": [],
            "tools": [],
            "functions": [],
            "tool_choice": "none",
            "function_call": "none",
            "parallel_tool_calls": false,
            "frequency_penalty": 0,
            "presence_penalty": 0.0,
            "logit_bias": {},
            "logprobs": false,
            "top_logprobs": 0,
            "store": false,
            "metadata": {},
            "user": "local-client",
            "future_request_field": null
        }))
        .unwrap();
        helpers::validate_chat_request_compatibility(&neutral).unwrap();

        let active = serde_json::from_value::<ChatRequest>(json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2,
            "stop": "END",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "frequency_penalty": 0.5,
            "unknown_semantic": true
        }))
        .unwrap();
        let error = helpers::validate_chat_request_compatibility(&active).unwrap_err();
        for field in ["frequency_penalty", "n", "unknown_semantic"] {
            assert!(error.contains(field));
        }
        assert_eq!(
            helpers::normalize_stop_sequences(active.stop.as_ref()).unwrap(),
            vec!["END"]
        );
        assert!(error.contains("instead of silently ignoring"));

        let message_semantics = serde_json::from_value::<ChatRequest>(json!({
            "messages": [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}]
            }]
        }))
        .unwrap();
        assert!(
            helpers::validate_chat_request_compatibility(&message_semantics)
                .unwrap_err()
                .contains("assistant tool call 0")
        );

        let stream_semantics = serde_json::from_value::<ChatRequest>(json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "stream_options": {"include_obfuscation": true}
        }))
        .unwrap();
        assert!(
            helpers::validate_chat_request_compatibility(&stream_semantics)
                .unwrap_err()
                .contains("stream_options")
        );

        let completion_semantics = serde_json::from_value::<CompletionRequest>(json!({
            "prompt": "Hello",
            "best_of": 2,
            "echo": true
        }))
        .unwrap();
        let completion_error =
            helpers::validate_completion_request_compatibility(&completion_semantics).unwrap_err();
        assert!(completion_error.contains("best_of"));
        assert!(completion_error.contains("echo"));

        let mut bounded = neutral;
        let oversized_field = "x".repeat(1_000);
        bounded
            .extensions
            .insert(oversized_field.clone(), json!(true));
        let bounded_error = helpers::validate_chat_request_compatibility(&bounded).unwrap_err();
        assert!(bounded_error.len() < 1_024);
        assert!(!bounded_error.contains(&oversized_field));
        assert!(bounded_error.contains("<invalid-field-name>"));
    }

    #[tokio::test]
    async fn chat_route_rejects_invalid_requests_before_runtime_availability() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route("/", post(handle_chat_completions))
            .with_state(Arc::clone(&state));
        for body in [
            json!({}),
            json!({"model": "default", "messages": []}),
            json!({"messages": [{"content": "Hello"}]}),
            json!({"messages": [{"role": "user"}]}),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": MAX_GENERATED_TOKENS + 1
            }),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 16,
                "max_completion_tokens": 17
            }),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": null}]
            }),
            json!({
                "model": "default",
                "messages": [{
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}]
                }]
            }),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "response_format": {"type": "unsupported"}
            }),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "response_format": {"type": "text", "future": true}
            }),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "custom", "custom": {"name": "lookup"}}]
            }),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "tool_choice": "required"
            }),
            json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "response_format": {"type": "json_object"}
            }),
            json!({
                "model": "default",
                "messages": [{
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-1", "type": "function"}]
                }]
            }),
            json!({
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "developer", "content": "Late policy"}
                ]
            }),
        ] {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/")
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();

            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
            assert_eq!(
                response
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "application/json"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }

        let admitted = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                json!({
                    "model": "default",
                    "messages": [
                        {"role": "developer", "content": "Be concise."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "text", "text": " world"}
                            ]
                        }
                    ],
                    "max_completion_tokens": 16
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(admitted).await.unwrap();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );

        let admitted_tool = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                json!({
                    "model": "default",
                    "messages": [{"role": "user", "content": "Look up Paris."}],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": false
                            }
                        }
                    }],
                    "tool_choice": "required",
                    "parallel_tool_calls": false
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.oneshot(admitted_tool).await.unwrap();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(state.metrics.in_flight_requests.load(Ordering::Relaxed), 0);
        assert_eq!(state.semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn responses_route_admits_only_the_bounded_text_subset() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route("/", post(handle_responses))
            .with_state(Arc::clone(&state));

        for body in [
            json!({}),
            json!({"input": []}),
            json!({"input": "Hello", "max_output_tokens": 0}),
            json!({
                "instructions": "x".repeat(MAX_CHAT_SYSTEM_MESSAGE_CHARS + 1),
                "input": "Hello"
            }),
            json!({"input": "Hello", "metadata": {"key": 1}}),
            json!({"input": "Hello", "text": "json"}),
            json!({"input": "Hello", "text": {"verbosity": "high"}}),
            json!({
                "input": "Hello",
                "text": {"format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "array"}
                }}
            }),
            json!({
                "input": [{
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.invalid/x.png"}]
                }]
            }),
        ] {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/")
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }

        let missing_previous = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                json!({"input": "Hello", "previous_response_id": "resp-1"}).to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(missing_previous).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);

        for stream in [false, true] {
            let admitted = axum::http::Request::builder()
                .method("POST")
                .uri("/")
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(
                    json!({
                        "model": "default",
                        "instructions": "Be concise.",
                        "input": [{
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Hello"}]
                        }],
                        "max_output_tokens": 16,
                        "stream": stream,
                        "metadata": {"session": "route-test"},
                        "text": {"format": {
                            "type": "json_schema",
                            "name": "answer",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": false
                            },
                            "strict": true
                        }}
                    })
                    .to_string(),
                ))
                .unwrap();
            let response = app.clone().oneshot(admitted).await.unwrap();
            assert_eq!(
                response.status(),
                axum::http::StatusCode::SERVICE_UNAVAILABLE
            );
        }
        let admitted_tool = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                json!({
                    "model": "default",
                    "input": "Look up Paris.",
                    "tools": [{
                        "type": "function",
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": false
                        },
                        "strict": true
                    }],
                    "tool_choice": {"type": "function", "name": "lookup"},
                    "parallel_tool_calls": false
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.oneshot(admitted_tool).await.unwrap();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(state.metrics.in_flight_requests.load(Ordering::Relaxed), 0);
        assert_eq!(state.semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn stored_response_routes_retrieve_page_chain_and_delete_local_state() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let stored_payload = json!({
            "id": "resp-stored",
            "object": "response",
            "model": "tiny.gguf",
            "store": true,
            "previous_response_id": null,
            "output": [{
                "id": "msg-output",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Prior answer",
                    "annotations": [],
                    "logprobs": []
                }]
            }]
        });
        let history = vec![
            ChatCompletionMessage {
                role: "user".to_string(),
                content: json!("Prior question"),
                extensions: BTreeMap::new(),
            },
            ChatCompletionMessage {
                role: "assistant".to_string(),
                content: json!("Prior answer"),
                extensions: BTreeMap::new(),
            },
        ];
        let input_items = ["one", "two", "three"]
            .into_iter()
            .map(|suffix| {
                json!({
                    "id": format!("msg-{suffix}"),
                    "type": "message",
                    "status": "completed",
                    "role": "user",
                    "content": [{"type": "input_text", "text": suffix}]
                })
            })
            .collect();
        state
            .response_store
            .insert(stored_payload.clone(), history, input_items)
            .unwrap();

        let app = Router::new()
            .route("/responses", post(handle_responses))
            .route(
                "/responses/{response_id}",
                get(handle_response_retrieve).delete(handle_response_delete),
            )
            .route(
                "/responses/{response_id}/input_items",
                get(handle_response_input_items),
            )
            .with_state(Arc::clone(&state));

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/responses/resp-stored")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            stored_payload
        );

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/responses/resp-stored/input_items?order=asc&limit=2")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let page: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(page["object"], "list");
        assert_eq!(page["data"][0]["id"], "msg-one");
        assert_eq!(page["last_id"], "msg-two");
        assert_eq!(page["has_more"], true);

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/responses/resp-stored/input_items?order=asc&limit=2&after=msg-two")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let page: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(page["data"][0]["id"], "msg-three");
        assert_eq!(page["has_more"], false);

        for uri in [
            "/responses/resp-stored?unknown=true",
            "/responses/resp-stored/input_items?limit=not-a-number",
            "/responses/resp-stored/input_items?after=msg-missing",
        ] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(uri)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
            assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&body).unwrap()["error"]["type"],
                "invalid_request_error"
            );
        }

        let wrong_model = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/responses")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "other.gguf",
                            "input": "Next question",
                            "previous_response_id": "resp-stored"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_model.status(), axum::http::StatusCode::BAD_REQUEST);

        let admitted_chain = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/responses")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "input": "Next question",
                            "previous_response_id": "resp-stored",
                            "store": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            admitted_chain.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/responses/resp-stored")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let deleted: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            deleted,
            json!({
                "id": "resp-stored",
                "object": "response",
                "deleted": true
            })
        );

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/responses/resp-stored")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn non_chat_routes_reject_invalid_input_before_runtime_admission() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route("/completions", post(handle_completions))
            .route("/embeddings", post(handle_embeddings))
            .route("/rerank", post(handle_rerank))
            .route("/multimodal", post(handle_multimodal_stream))
            .with_state(Arc::clone(&state));
        let local_path_request = test_multimodal_request(vec![DataBlock::AudioFile {
            path: "/etc/passwd".to_string(),
            language: None,
        }]);
        let cases = [
            ("/completions", br#"{"prompt":" "}"#.to_vec()),
            (
                "/completions",
                br#"{"prompt":"Hello","stop":[""]}"#.to_vec(),
            ),
            ("/embeddings", br#"{"input":[]}"#.to_vec()),
            (
                "/embeddings",
                br#"{"input":"Hello","dimensions":0}"#.to_vec(),
            ),
            (
                "/embeddings",
                br#"{"input":"Hello","future":true}"#.to_vec(),
            ),
            ("/rerank", br#"{"query":"","documents":[]}"#.to_vec()),
            (
                "/rerank",
                br#"{"query":"Hello","documents":["Hello"],"future":true}"#.to_vec(),
            ),
            (
                "/multimodal",
                serde_json::to_vec(&local_path_request).unwrap(),
            ),
        ];

        for (path, body) in cases {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();

            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
            assert_eq!(
                response
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "application/json"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert!(!body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("/etc/passwd"));
        }

        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(state.metrics.in_flight_requests.load(Ordering::Relaxed), 0);
        assert_eq!(state.semaphore.available_permits(), 1);
    }

    #[test]
    fn requested_models_bind_to_the_active_runtime() {
        use helpers::RequestedModelError::{Invalid, NotLoaded};

        assert!(helpers::validate_requested_model(None, "tiny.gguf").is_ok());
        assert!(helpers::validate_requested_model(Some("default"), "tiny.gguf").is_ok());
        assert!(helpers::validate_requested_model(Some("tiny.gguf"), "tiny.gguf").is_ok());
        assert_eq!(
            helpers::validate_requested_model(Some("other.gguf"), "tiny.gguf"),
            Err(NotLoaded)
        );
        for invalid in ["", " tiny.gguf", "tiny.gguf ", "tiny\ngguf"] {
            assert_eq!(
                helpers::validate_requested_model(Some(invalid), "tiny.gguf"),
                Err(Invalid)
            );
        }

        let maximum = "m".repeat(helpers::MAX_REQUESTED_MODEL_ID_CHARS);
        assert!(helpers::validate_requested_model(Some(&maximum), &maximum).is_ok());
        let oversized = "m".repeat(helpers::MAX_REQUESTED_MODEL_ID_CHARS + 1);
        assert_eq!(
            helpers::validate_requested_model(Some(&oversized), "tiny.gguf"),
            Err(Invalid)
        );
    }

    #[tokio::test]
    async fn requested_model_errors_are_bounded_and_machine_readable() {
        let invalid =
            helpers::requested_model_error_response(helpers::RequestedModelError::Invalid);
        assert_eq!(invalid.status(), axum::http::StatusCode::BAD_REQUEST);

        let missing =
            helpers::requested_model_error_response(helpers::RequestedModelError::NotLoaded);
        assert_eq!(missing.status(), axum::http::StatusCode::NOT_FOUND);
        let bytes = axum::body::to_bytes(missing.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "model_not_found");
        assert!(!body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("other.gguf"));
    }

    #[tokio::test]
    async fn model_retrieve_returns_stable_active_identity_and_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_server_state_with_text_runtime(
            temp.path().to_path_buf(),
            Arc::new(AtomicU64::new(0)),
        )
        .await;
        let (model_id, published_at) = {
            let runtime = state.runtime.read().await;
            let runtime = runtime.as_ref().unwrap();
            (runtime.model_id.clone(), runtime.published_at)
        };
        let app = Router::new()
            .route("/models", get(handle_models))
            .route("/models/{model}", get(handle_model_retrieve))
            .with_state(state);

        let listed = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let listed: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(listed.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(listed["data"][0]["id"], model_id);
        assert_eq!(listed["data"][0]["created"], published_at);

        for selector in [model_id.as_str(), "default"] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(format!("/models/{selector}"))
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let body: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(body["id"], model_id);
            assert_eq!(body["object"], "model");
            assert_eq!(body["created"], published_at);
            assert_eq!(body["owned_by"], "bloom");
        }

        for (uri, expected_status) in [
            ("/models/missing-model", axum::http::StatusCode::NOT_FOUND),
            ("/models/%20default", axum::http::StatusCode::BAD_REQUEST),
            (
                "/models/default?future=true",
                axum::http::StatusCode::BAD_REQUEST,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(uri)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status);
            let body: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert!(body["error"]["type"].is_string());
        }
    }

    #[tokio::test]
    async fn model_retrieve_reports_not_found_without_an_active_runtime() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route("/models/{model}", get(handle_model_retrieve))
            .with_state(state);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/models/default")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["type"], "model_not_found");
    }

    #[test]
    fn generation_admission_validates_controls_prompt_shape_and_context_budget() {
        assert!(helpers::validate_generation_controls(128, 0.7, 0.9).is_ok());
        assert!(helpers::validate_generation_controls(MAX_GENERATED_TOKENS, 0.7, 0.9).is_ok());
        assert!(helpers::validate_generation_controls(0, 0.7, 0.9).is_err());
        assert!(helpers::validate_generation_controls(MAX_GENERATED_TOKENS + 1, 0.7, 0.9).is_err());
        assert!(helpers::validate_generation_controls(128, f64::NAN, 0.9).is_err());
        assert!(helpers::validate_generation_controls(128, 0.7, 0.0).is_err());

        assert_eq!(
            helpers::single_completion_prompt(&serde_json::json!(["one"])).unwrap(),
            "one"
        );
        assert!(helpers::single_completion_prompt(&serde_json::json!(["one", "two"])).is_err());
        assert!(helpers::single_completion_prompt(&serde_json::json!([1, 2])).is_err());
        assert!(helpers::single_completion_prompt(&serde_json::json!("   ")).is_err());
        assert!(helpers::single_completion_prompt(&serde_json::json!(
            "x".repeat(MAX_COMPLETION_PROMPT_CHARS + 1)
        ))
        .is_err());
        assert!(
            helpers::single_completion_prompt(&serde_json::json!("\u{1f600}".repeat(200_000)))
                .is_err()
        );

        assert!(helpers::validate_context_budget(3_000, 1_096, 4_096).is_ok());
        let error = helpers::validate_context_budget(3_000, 1_097, 4_096).unwrap_err();
        assert!(error.contains("3000 prompt tokens"));
        assert!(error.contains("4096 tokens"));
        assert!(helpers::validate_context_budget(usize::MAX, 1, usize::MAX).is_err());
    }

    fn test_server_state(
        models_root: PathBuf,
    ) -> (Arc<ServerState>, mpsc::Receiver<ModelLoadRequest>) {
        test_server_state_with_imports(models_root, None)
    }

    async fn test_server_state_with_embedding_runtime(
        models_root: PathBuf,
    ) -> (Arc<ServerState>, Arc<AtomicU64>) {
        let (state, _receiver) = test_server_state(models_root.clone());
        let model_path = models_root.join("test-embed-model.fixture");
        std::fs::write(&model_path, b"bounded embedding fixture").unwrap();
        let native_batch_calls = Arc::new(AtomicU64::new(0));
        let pipeline = Arc::new(
            InferencePipeline::load_standalone_with_context(
                &TestEmbeddingEngine {
                    native_batch_calls: Arc::clone(&native_batch_calls),
                },
                DeviceKind::Cpu,
                &model_path,
                128,
            )
            .unwrap(),
        );
        let manifest = pipeline.metadata().manifest.clone();
        let runtime = Arc::new(LoadedRuntime {
            model_id: pipeline.metadata().id.clone(),
            model_family: manifest.family.clone(),
            model_architecture: None,
            model_chat_template: None,
            input_modalities: vec![bloomai_core::Modality::Text],
            memory_estimate: bloomai_engine::estimate_memory(&manifest, 128),
            pipeline,
            kv_cache_pool: None,
            cachemesh: None,
            scheduler: None,
            _memory_reservation: None,
            scheduler_shutdown: CancellationToken::new(),
            published_at: unix_seconds(),
            source_path: model_path,
            catalog_id: None,
        });
        *state.runtime.write().await = Some(runtime);
        state.ready.store(true, Ordering::Release);
        (state, native_batch_calls)
    }

    async fn test_server_state_with_text_runtime(
        models_root: PathBuf,
        emitted_chunks: Arc<AtomicU64>,
    ) -> Arc<ServerState> {
        let (state, _receiver) = test_server_state(models_root.clone());
        let model_path = models_root.join("test-text-model.fixture");
        std::fs::write(&model_path, b"bounded text fixture").unwrap();
        let pipeline = Arc::new(
            InferencePipeline::load_standalone_with_context(
                &TestTextEngine { emitted_chunks },
                DeviceKind::Cpu,
                &model_path,
                128,
            )
            .unwrap(),
        );
        let manifest = pipeline.metadata().manifest.clone();
        let runtime = Arc::new(LoadedRuntime {
            model_id: pipeline.metadata().id.clone(),
            model_family: manifest.family.clone(),
            model_architecture: None,
            model_chat_template: None,
            input_modalities: vec![bloomai_core::Modality::Text],
            memory_estimate: bloomai_engine::estimate_memory(&manifest, 128),
            pipeline,
            kv_cache_pool: None,
            cachemesh: None,
            scheduler: None,
            _memory_reservation: None,
            scheduler_shutdown: CancellationToken::new(),
            published_at: unix_seconds(),
            source_path: model_path,
            catalog_id: None,
        });
        *state.runtime.write().await = Some(runtime);
        state.ready.store(true, Ordering::Release);
        state
    }

    fn test_multimodal_request(blocks: Vec<DataBlock>) -> InferenceRequest {
        InferenceRequest {
            blocks,
            params: InferenceParams {
                max_tokens: 128,
                ..InferenceParams::default()
            },
        }
    }

    fn test_server_state_with_imports(
        models_root: PathBuf,
        model_imports: Option<Arc<ModelImportManager>>,
    ) -> (Arc<ServerState>, mpsc::Receiver<ModelLoadRequest>) {
        test_server_state_with_acquisitions(models_root, None, model_imports)
    }

    fn test_server_state_with_acquisitions(
        models_root: PathBuf,
        model_downloads: Option<Arc<ModelDownloadManager>>,
        model_imports: Option<Arc<ModelImportManager>>,
    ) -> (Arc<ServerState>, mpsc::Receiver<ModelLoadRequest>) {
        test_server_state_with_services(models_root, model_downloads, model_imports, None)
    }

    fn test_server_state_with_services(
        models_root: PathBuf,
        model_downloads: Option<Arc<ModelDownloadManager>>,
        model_imports: Option<Arc<ModelImportManager>>,
        model_index: Option<Arc<ModelIndexManager>>,
    ) -> (Arc<ServerState>, mpsc::Receiver<ModelLoadRequest>) {
        let (model_loader, receiver) = mpsc::channel(1);
        let model_storage = ModelStorageManager::new(models_root.clone(), 0, 0);
        let model_integrity = ModelIntegrityManager::new(models_root.clone());
        let model_preflight = ModelPreflightManager::new(
            models_root.clone(),
            ModelPreflightConfig {
                backend: "candle".to_string(),
                speculative: "none".to_string(),
                device: DeviceKind::Cpu,
                context_size: 128,
                max_concurrent: 1,
                memory_utilization: 0.75,
                reserve_memory_bytes: None,
                disable_memory_prealloc: false,
            },
        );
        (
            Arc::new(ServerState {
                runtime: RwLock::new(None),
                inference_admission: RwLock::new(()),
                semaphore: Arc::new(Semaphore::new(1)),
                ready: AtomicBool::new(false),
                load_in_progress: AtomicBool::new(false),
                load_progress: AtomicU8::new(0),
                load_error: RwLock::new(None),
                requested_model: RwLock::new(None),
                model_lifecycle: Mutex::new(ModelLifecycle::default()),
                ollama_residency: Mutex::new(OllamaResidencyState::default()),
                models_root,
                model_catalog_cache: RwLock::new(None),
                model_storage,
                model_downloads,
                model_imports,
                model_index,
                model_integrity,
                model_preflight,
                model_loader,
                metrics: Arc::new(ServerMetrics::new()),
                speculative_mode: "none".to_string(),
                enable_ifb: false,
                cancel_tokens: Arc::new(std::sync::Mutex::new(HashMap::new())),
                request_counter: AtomicU64::new(0),
                api_key: None,
                response_store: ResponseStore::default(),
            }),
            receiver,
        )
    }

    fn write_test_model_manifest(root: &std::path::Path, name: &str) -> PathBuf {
        let model_path = root.join(name);
        std::fs::create_dir(&model_path).unwrap();
        std::fs::write(
            model_path.join("bloom.json"),
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
        model_path
    }

    #[test]
    fn select_backend_auto_routes_mtp_to_llamacpp() {
        let manifest = bloomai_core::ModelManifest::default();
        assert_eq!(
            select_backend_name("candle", "draft-mtp", &manifest),
            "llamacpp"
        );
    }

    #[tokio::test]
    async fn model_catalog_endpoint_supports_an_empty_runtime() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tiny.gguf"), b"gguf").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_catalog(State(Arc::clone(&state))).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["schema_version"], 1);
        assert_eq!(body["object"], "bloom.model_catalog");
        assert_eq!(body["load"]["phase"], "idle");
        assert_eq!(body["data"][0]["id"], "tiny.gguf");
        assert!(body["active_model"].is_null());
        assert_eq!(body["download"]["enabled"], false);
        assert_eq!(body["download"]["license_policy"]["enforced"], false);
        assert_eq!(body["import"]["enabled"], false);
        assert_eq!(body["import"]["license_policy"]["allowed"], json!([]));
        assert_eq!(body["index"]["enabled"], false);
        assert!(body["index"]["key_id"].is_null());
        assert!(body["index"]["trust_id"].is_null());
        assert_eq!(body["index"]["trusted_key_count"], 0);
        assert_eq!(body["index"]["refresh_seconds"], 0);
        assert_eq!(body["index"]["persistent_rollback_protection"], false);
        assert_eq!(body["integrity"]["phase"], "idle");
        assert_eq!(body["storage"]["quota_enabled"], false);
        assert_eq!(body["storage"]["used_bytes"], 4);
        assert_eq!(body["storage"]["committed_bytes"], 4);

        let response = handle_model_index(State(state)).await;
        assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn protocol_fallbacks_return_shaped_404_and_405_without_bypassing_auth() {
        let temp = tempfile::tempdir().unwrap();
        let (mut state, _receiver) = test_server_state(temp.path().to_path_buf());
        Arc::get_mut(&mut state).unwrap().api_key = Some("test-secret".to_string());

        let openai_routes = Router::new()
            .route(
                "/models",
                get(|| async { axum::http::StatusCode::NO_CONTENT }),
            )
            .route_layer(middleware::from_fn_with_state(
                Arc::clone(&state),
                require_api_key,
            ))
            .fallback(handle_openai_route_not_found)
            .method_not_allowed_fallback(handle_openai_method_not_allowed);
        let app = Router::new()
            .nest("/v1", openai_routes)
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(state)
            .layer(middleware::from_fn(publish_authentication_challenge));

        let openai_missing = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/unknown")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(openai_missing.status(), axum::http::StatusCode::NOT_FOUND);
        assert_eq!(
            openai_missing.headers()[header::CONTENT_TYPE],
            "application/json"
        );
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(openai_missing.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["type"], "not_found_error");
        assert_eq!(
            body["error"]["message"],
            "The requested OpenAI-compatible API route does not exist."
        );

        let openai_wrong_method = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            openai_wrong_method.status(),
            axum::http::StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(openai_wrong_method.headers()[header::ALLOW], "GET,HEAD");
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(openai_wrong_method.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");

        let ollama_missing = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/unknown")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ollama_missing.status(), axum::http::StatusCode::NOT_FOUND);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(ollama_missing.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"], "Ollama-compatible API route not found");

        let ollama_wrong_method = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/show")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            ollama_wrong_method.status(),
            axum::http::StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(ollama_wrong_method.headers()[header::ALLOW], "POST");
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(ollama_wrong_method.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            body["error"],
            "HTTP method not allowed for this Ollama-compatible API route"
        );

        for path in ["/v1/models", "/api/version"] {
            let unauthorized = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(path)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(unauthorized.status(), axum::http::StatusCode::UNAUTHORIZED);
            assert_eq!(
                unauthorized.headers()[header::WWW_AUTHENTICATE],
                DEFAULT_BEARER_AUTHENTICATION_CHALLENGE
            );
        }
    }

    #[tokio::test]
    async fn framework_rejections_are_protocol_shaped_without_touching_static_or_json_bodies() {
        #[derive(Deserialize)]
        struct NumberPayload {
            value: u8,
        }

        for content_type in [
            "application/json; charset=utf-8",
            "application/problem+json",
            "text/event-stream",
            "application/x-ndjson",
        ] {
            let response = (
                axum::http::StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, content_type)],
                "protocol body",
            )
                .into_response();
            assert!(has_protocol_error_content_type(&response));
        }

        let app = Router::new()
            .route(
                "/v1/json",
                post(|Json(payload): Json<serde_json::Value>| async move {
                    let _ = payload;
                    axum::http::StatusCode::NO_CONTENT
                }),
            )
            .route(
                "/v1/typed",
                post(|Json(payload): Json<NumberPayload>| async move {
                    let _ = payload.value;
                    axum::http::StatusCode::NO_CONTENT
                }),
            )
            .route(
                "/v1/upload",
                post(|_multipart: Multipart| async { axum::http::StatusCode::NO_CONTENT }),
            )
            .route(
                "/v1/plain",
                get(|| async {
                    (
                        axum::http::StatusCode::BAD_GATEWAY,
                        [(header::RETRY_AFTER, "5")],
                        "private upstream detail",
                    )
                }),
            )
            .route(
                "/v1/already-json",
                get(|| async {
                    error_response(
                        axum::http::StatusCode::BAD_REQUEST,
                        "specific_error",
                        "specific safe message",
                    )
                }),
            )
            .route(
                "/assets/plain",
                get(|| async {
                    (
                        axum::http::StatusCode::BAD_REQUEST,
                        "static response detail",
                    )
                }),
            )
            .layer(DefaultBodyLimit::max(32))
            .layer(middleware::from_fn(normalize_protocol_error_response));

        for (path, content_type, body, expected_status, expected_type) in [
            (
                "/v1/json",
                None,
                "{}",
                axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "invalid_request_error",
            ),
            (
                "/v1/json",
                Some("application/json"),
                "{\"secret\":\"bloom-private-value\"",
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                "/v1/typed",
                Some("application/json"),
                "{\"value\":\"bloom-private-value\"}",
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request_error",
            ),
            (
                "/v1/json",
                Some("application/json"),
                "                                  ",
                axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
            ),
            (
                "/v1/upload",
                Some("multipart/form-data"),
                "bloom-private-value",
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
        ] {
            let mut request = axum::http::Request::builder().method("POST").uri(path);
            if let Some(content_type) = content_type {
                request = request.header(header::CONTENT_TYPE, content_type);
            }
            let response = app
                .clone()
                .oneshot(request.body(axum::body::Body::from(body)).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status, "path: {path}");
            assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(payload["error"]["type"], expected_type);
            assert!(!String::from_utf8_lossy(&bytes).contains("bloom-private-value"));
        }

        let plain = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/plain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(plain.status(), axum::http::StatusCode::BAD_GATEWAY);
        assert_eq!(plain.headers()[header::RETRY_AFTER], "5");
        let plain_body = axum::body::to_bytes(plain.into_body(), usize::MAX)
            .await
            .unwrap();
        let plain_body: serde_json::Value = serde_json::from_slice(&plain_body).unwrap();
        assert_eq!(plain_body["error"]["type"], "internal_error");
        assert!(!plain_body.to_string().contains("private upstream detail"));

        let already_json = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/already-json")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let already_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(already_json.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(already_json["error"]["type"], "specific_error");
        assert_eq!(already_json["error"]["message"], "specific safe message");

        let static_response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/assets/plain")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            axum::body::to_bytes(static_response.into_body(), usize::MAX)
                .await
                .unwrap(),
            "static response detail"
        );

        let timed = Router::new()
            .route(
                "/v1/slow",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                    axum::http::StatusCode::NO_CONTENT
                }),
            )
            .layer(TimeoutLayer::with_status_code(
                axum::http::StatusCode::REQUEST_TIMEOUT,
                Duration::from_millis(1),
            ))
            .layer(middleware::from_fn(normalize_protocol_error_response));
        let timed = timed
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/slow")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(timed.status(), axum::http::StatusCode::REQUEST_TIMEOUT);
        let timed: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(timed.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(timed["error"]["type"], "timeout_error");

        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let ollama = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(state)
            .layer(DefaultBodyLimit::max(32))
            .layer(middleware::from_fn(normalize_protocol_error_response));
        let response = ollama
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        "{\"model\":\"default\",\"padding\":\"0123456789\"}",
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            body["error"],
            "request body exceeds the configured size limit"
        );
    }

    #[tokio::test]
    async fn ollama_routes_expose_discovery_and_fail_closed_generation() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tiny.gguf"), b"gguf").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(state);

        let version = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/version")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(version.status(), axum::http::StatusCode::OK);
        assert_eq!(
            version.headers().get(header::CACHE_CONTROL).unwrap(),
            "no-store"
        );
        let version: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(version.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(version["version"], env!("CARGO_PKG_VERSION"));

        let tags = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/tags")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tags.status(), axum::http::StatusCode::OK);
        let tags: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(tags.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(tags["models"][0]["name"], "tiny.gguf");
        assert_eq!(tags["models"][0]["details"]["format"], "gguf");

        let ps = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/ps")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let ps: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(ps.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(ps["models"], json!([]));

        let show = axum::http::Request::builder()
            .method("POST")
            .uri("/api/show")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                json!({"model": "tiny.gguf"}).to_string(),
            ))
            .unwrap();
        let show = app.clone().oneshot(show).await.unwrap();
        assert_eq!(show.status(), axum::http::StatusCode::OK);
        let show: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(show.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(show["details"]["format"], "gguf");
        assert_eq!(show["capabilities"], json!(["completion", "tools"]));

        for (path, body) in [
            (
                "/api/chat",
                json!({
                    "model": "default",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": false,
                    "options": {"top_k": 40}
                }),
            ),
            (
                "/api/generate",
                json!({"model": "default", "prompt": "Hello", "raw": true}),
            ),
            (
                "/api/embed",
                json!({"model": "default", "input": "Hello", "options": {"num_ctx": 1024}}),
            ),
            (
                "/api/embed",
                json!({"model": "default", "input": "Hello", "dimensions": 16385}),
            ),
            (
                "/api/embeddings",
                json!({"model": "default", "prompt": "Hello", "future": true}),
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri(path)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
            let body: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert!(body["error"].is_string());
        }

        for (path, body, expected_status) in [
            (
                "/api/chat",
                json!({
                    "model": "default",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": false
                }),
                axum::http::StatusCode::NOT_FOUND,
            ),
            (
                "/api/embed",
                json!({"model": "default", "input": ["Hello", "World"]}),
                axum::http::StatusCode::NOT_FOUND,
            ),
            (
                "/api/embeddings",
                json!({"model": "default", "prompt": "Hello"}),
                axum::http::StatusCode::NOT_FOUND,
            ),
        ] {
            let admitted = axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let admitted = app.clone().oneshot(admitted).await.unwrap();
            assert_eq!(admitted.status(), expected_status);
            let admitted: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(admitted.into_body(), usize::MAX)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert!(admitted["error"].is_string());
            assert!(admitted.get("type").is_none());
        }
    }

    #[tokio::test]
    async fn ollama_routes_use_ollama_shaped_api_key_errors() {
        let temp = tempfile::tempdir().unwrap();
        let protected_model = temp.path().join("protected.gguf");
        std::fs::write(&protected_model, b"gguf").unwrap();
        let (mut state, _receiver) = test_server_state(temp.path().to_path_buf());
        Arc::get_mut(&mut state).unwrap().api_key = Some("test-secret".to_string());
        let app = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(state)
            .layer(middleware::from_fn(publish_authentication_challenge));

        let unauthorized = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/version")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), axum::http::StatusCode::UNAUTHORIZED);
        assert_eq!(
            unauthorized.headers()[header::WWW_AUTHENTICATE],
            DEFAULT_BEARER_AUTHENTICATION_CHALLENGE
        );
        let unauthorized: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(unauthorized.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(unauthorized["error"].is_string());
        assert!(unauthorized.get("type").is_none());

        let unauthorized_embed = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/embed")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({"model": "default", "input": "Hello"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            unauthorized_embed.status(),
            axum::http::StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            unauthorized_embed.headers()[header::WWW_AUTHENTICATE],
            DEFAULT_BEARER_AUTHENTICATION_CHALLENGE
        );
        let unauthorized_embed: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(unauthorized_embed.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(unauthorized_embed["error"].is_string());
        assert!(unauthorized_embed.get("type").is_none());

        let unauthorized_delete = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/api/delete")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({"model": "protected.gguf"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            unauthorized_delete.status(),
            axum::http::StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            unauthorized_delete.headers()[header::WWW_AUTHENTICATE],
            DEFAULT_BEARER_AUTHENTICATION_CHALLENGE
        );
        assert!(protected_model.exists());

        for (name, value) in [
            (header::AUTHORIZATION.as_str(), "Bearer test-secret"),
            ("x-api-key", "test-secret"),
        ] {
            let authorized = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri("/api/version")
                        .header(name, value)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(authorized.status(), axum::http::StatusCode::OK);
        }

        let authorized_delete = app
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/api/delete")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::AUTHORIZATION, "Bearer test-secret")
                    .body(axum::body::Body::from(
                        json!({"model": "protected.gguf"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized_delete.status(), axum::http::StatusCode::OK);
        assert!(!protected_model.exists());
    }

    #[tokio::test]
    async fn ollama_empty_prompt_keep_alive_zero_unloads_the_active_runtime() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_server_state_with_text_runtime(
            temp.path().to_path_buf(),
            Arc::new(AtomicU64::new(0)),
        )
        .await;
        let app = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(Arc::clone(&state));

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "default",
                            "prompt": "",
                            "keep_alive": 0,
                            "stream": false
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["model"], "default");
        assert_eq!(body["response"], "");
        assert_eq!(body["done"], true);
        assert_eq!(body["done_reason"], "unload");
        assert!(state.runtime.read().await.is_none());
        assert!(!state.ready.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn ollama_timed_keep_alive_expires_and_new_policy_cancels_the_timer() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_server_state_with_text_runtime(
            temp.path().to_path_buf(),
            Arc::new(AtomicU64::new(0)),
        )
        .await;
        let app = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(Arc::clone(&state));

        let preload = |keep_alive: serde_json::Value| {
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/generate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(
                    json!({
                        "model": "default",
                        "prompt": "",
                        "keep_alive": keep_alive,
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap()
        };

        let timed = app.clone().oneshot(preload(json!("40ms"))).await.unwrap();
        assert_eq!(timed.status(), axum::http::StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(5)).await;
        let processes = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/ps")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let processes: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(processes.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_ne!(processes["models"][0]["expires_at"], "9999-12-31T23:59:59Z");

        let indefinite = app.clone().oneshot(preload(json!(-1))).await.unwrap();
        assert_eq!(indefinite.status(), axum::http::StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(state.runtime.read().await.is_some());

        let expiring = app.oneshot(preload(json!("20ms"))).await.unwrap();
        assert_eq!(expiring.status(), axum::http::StatusCode::OK);
        let deadline = Instant::now() + Duration::from_secs(1);
        while state.runtime.read().await.is_some() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(state.runtime.read().await.is_none());
        assert!(!state.ready.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn ollama_failed_activation_preserves_the_existing_residency_deadline() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_server_state_with_text_runtime(
            temp.path().to_path_buf(),
            Arc::new(AtomicU64::new(0)),
        )
        .await;
        let app = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(Arc::clone(&state));
        let request = |model: &str, keep_alive: serde_json::Value| {
            axum::http::Request::builder()
                .method("POST")
                .uri("/api/generate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(
                    json!({
                        "model": model,
                        "prompt": "",
                        "keep_alive": keep_alive,
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap()
        };

        let timed = app
            .clone()
            .oneshot(request("default", json!("40ms")))
            .await
            .unwrap();
        assert_eq!(timed.status(), axum::http::StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(5)).await;

        let failed = app
            .oneshot(request("missing-model", json!(-1)))
            .await
            .unwrap();
        assert_eq!(failed.status(), axum::http::StatusCode::NOT_FOUND);

        let deadline = Instant::now() + Duration::from_secs(1);
        while state.runtime.read().await.is_some() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(state.runtime.read().await.is_none());
        assert!(!state.ready.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn ollama_delete_is_bounded_and_preserves_models_on_rejection() {
        let temp = tempfile::tempdir().unwrap();
        let removable = temp.path().join("remove.gguf");
        let retained = temp.path().join("keep.gguf");
        std::fs::write(&removable, b"remove").unwrap();
        std::fs::write(&retained, b"keep").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(Arc::clone(&state));

        for (body, expected_status) in [
            (
                json!({"model": "missing.gguf"}),
                axum::http::StatusCode::NOT_FOUND,
            ),
            (
                json!({"model": "../keep.gguf"}),
                axum::http::StatusCode::BAD_REQUEST,
            ),
            (
                json!({"model": "keep.gguf", "future": true}),
                axum::http::StatusCode::BAD_REQUEST,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("DELETE")
                        .uri("/api/delete")
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status);
            let body: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert!(body["error"].is_string());
            assert!(body.get("type").is_none());
            assert!(retained.exists());
        }

        let malformed = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/api/delete")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(malformed.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(retained.exists());

        state.load_in_progress.store(true, Ordering::Release);
        let conflicted = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/api/delete")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({"model": "keep.gguf"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(conflicted.status(), axum::http::StatusCode::CONFLICT);
        assert!(retained.exists());
        state.load_in_progress.store(false, Ordering::Release);

        let removed = app
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/api/delete")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({"model": "remove.gguf"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(removed.status(), axum::http::StatusCode::OK);
        assert!(axum::body::to_bytes(removed.into_body(), usize::MAX)
            .await
            .unwrap()
            .is_empty());
        assert!(!removable.exists());
        assert!(retained.exists());
        assert!(state.model_catalog_cache.read().await.is_none());
    }

    #[tokio::test]
    async fn ollama_delete_refuses_the_active_catalog_model() {
        let temp = tempfile::tempdir().unwrap();
        let active_model = temp.path().join("active.gguf");
        std::fs::write(&active_model, b"active").unwrap();
        let state = test_server_state_with_text_runtime(
            temp.path().to_path_buf(),
            Arc::new(AtomicU64::new(0)),
        )
        .await;
        {
            let mut runtime = state.runtime.write().await;
            let runtime = Arc::get_mut(runtime.as_mut().unwrap()).unwrap();
            runtime.source_path = active_model.clone();
            runtime.catalog_id = Some("active.gguf".to_string());
        }
        let app = Router::new()
            .nest("/api", ollama_api_router(Arc::clone(&state)))
            .with_state(state);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/api/delete")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({"model": "active.gguf"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body["error"].as_str().unwrap().contains("active model"));
        assert!(active_model.exists());
    }

    #[tokio::test]
    async fn model_catalog_exposes_the_bounded_acquisition_license_policy() {
        let temp = tempfile::tempdir().unwrap();
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);
        let policy = Arc::new(
            ModelLicensePolicy::new(vec!["Apache-2.0".to_string(), "MIT".to_string()]).unwrap(),
        );
        let downloads = ModelDownloadManager::with_storage_and_license_policy(
            temp.path().to_path_buf(),
            1024,
            storage,
            policy,
        )
        .unwrap();
        let (state, _receiver) =
            test_server_state_with_acquisitions(temp.path().to_path_buf(), Some(downloads), None);

        let response = handle_model_catalog(State(state)).await;
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(body["download"]["license_policy"]["enforced"], true);
        assert_eq!(
            body["download"]["license_policy"]["allowed"],
            json!(["Apache-2.0", "MIT"])
        );
    }

    #[tokio::test]
    async fn signed_model_index_endpoint_returns_a_versioned_path_free_snapshot() {
        use base64::Engine as _;
        use ed25519_dalek::{Signer as _, SigningKey};

        let temp = tempfile::tempdir().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let key = SigningKey::from_bytes(&[9_u8; 32]);
        let revision = "ab".repeat(20);
        let payload = serde_json::to_vec(&json!({
            "schema_version": 1,
            "object": "bloom.model_index",
            "name": "Test Models",
            "generated_at": now.saturating_sub(1),
            "expires_at": now.saturating_add(3600),
            "models": [{
                "id": "tiny-q4",
                "name": "Tiny Q4",
                "description": "A small API fixture.",
                "download_url": format!("https://huggingface.co/acme/tiny/resolve/{revision}/tiny.gguf"),
                "filename": "tiny.gguf",
                "size_bytes": 4096,
                "sha256": "cd".repeat(32),
                "license": "Apache-2.0"
            }]
        }))
        .unwrap();
        let mut message = b"bloom.model_index.v1\0".to_vec();
        message.extend_from_slice(&payload);
        let envelope = serde_json::to_vec(&json!({
            "schema_version": 1,
            "object": "bloom.signed_model_index",
            "algorithm": "ed25519",
            "key_id": format!("{:x}", sha2::Sha256::digest(key.verifying_key().as_bytes())),
            "payload": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload),
            "signature": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key.sign(&message).to_bytes())
        }))
        .unwrap();
        let index_path = temp.path().join("private-signed-index.json");
        tokio::fs::write(&index_path, envelope).await.unwrap();
        let public_key = key
            .verifying_key()
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let index = ModelIndexManager::from_config(
            ModelIndexManagerConfig {
                file: Some(index_path.clone()),
                url: None,
                public_key: Some(public_key),
                public_keys: vec![],
                refresh_seconds: 300,
                max_download_bytes: 8192,
                state_directory: temp.path().join("index-state"),
            },
            Arc::new(ModelLicensePolicy::default()),
        )
        .unwrap()
        .unwrap();
        let (state, _receiver) =
            test_server_state_with_services(temp.path().to_path_buf(), None, None, Some(index));

        let catalog_response = handle_model_catalog(State(Arc::clone(&state))).await;
        let catalog_bytes = axum::body::to_bytes(catalog_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes).unwrap();
        assert_eq!(catalog["index"]["enabled"], true);
        assert_eq!(catalog["index"]["trusted_key_count"], 1);
        assert_eq!(catalog["index"]["refresh_seconds"], 300);
        assert_eq!(catalog["index"]["persistent_rollback_protection"], true);
        assert!(catalog["index"]["key_id"].is_string());
        assert!(catalog["index"]["trust_id"].is_string());

        let response = handle_model_index(State(state)).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(body["schema_version"], 1);
        assert_eq!(body["object"], "bloom.model_index");
        assert_eq!(body["cache_status"], "fresh");
        assert_eq!(body["data"][0]["id"], "tiny-q4");
        assert!(body.get("generation_id").is_none());
        assert!(!String::from_utf8(bytes.to_vec())
            .unwrap()
            .contains(index_path.to_string_lossy().as_ref()));
    }

    async fn test_signed_index(
        root: &Path,
        model_id: &str,
        filename: &str,
        size_bytes: u64,
        sha256: &str,
    ) -> Arc<ModelIndexManager> {
        use base64::Engine as _;
        use ed25519_dalek::{Signer as _, SigningKey};

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let key = SigningKey::from_bytes(&[11_u8; 32]);
        let revision = "de".repeat(20);
        let download_url =
            format!("https://huggingface.co/acme/tiny/resolve/{revision}/{filename}");
        let payload = serde_json::to_vec(&json!({
            "schema_version": 1,
            "object": "bloom.model_index",
            "name": "Ollama Pull Test Models",
            "generated_at": now.saturating_sub(1),
            "expires_at": now.saturating_add(3600),
            "models": [{
                "id": model_id,
                "name": "Tiny Pull Fixture",
                "description": "A deterministic Ollama pull fixture.",
                "download_url": download_url,
                "filename": filename,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "license": "Apache-2.0"
            }]
        }))
        .unwrap();
        let mut message = b"bloom.model_index.v1\0".to_vec();
        message.extend_from_slice(&payload);
        let key_id = format!("{:x}", sha2::Sha256::digest(key.verifying_key().as_bytes()));
        let envelope = serde_json::to_vec(&json!({
            "schema_version": 1,
            "object": "bloom.signed_model_index",
            "algorithm": "ed25519",
            "key_id": key_id,
            "payload": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload),
            "signature": base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(key.sign(&message).to_bytes())
        }))
        .unwrap();
        let index_path = root.join("ollama-pull-index.json");
        tokio::fs::write(&index_path, envelope).await.unwrap();
        let public_key = key
            .verifying_key()
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        ModelIndexManager::from_config(
            ModelIndexManagerConfig {
                file: Some(index_path),
                url: None,
                public_key: Some(public_key),
                public_keys: vec![],
                refresh_seconds: 300,
                max_download_bytes: 8192,
                state_directory: root.join("ollama-pull-index-state"),
            },
            Arc::new(ModelLicensePolicy::default()),
        )
        .unwrap()
        .unwrap()
    }

    #[tokio::test]
    async fn signed_index_download_endpoint_installs_an_atomic_model_package() {
        async fn download_package_file(
            State(files): State<Arc<std::collections::BTreeMap<String, Vec<u8>>>>,
            axum::extract::Path(path): axum::extract::Path<String>,
        ) -> axum::response::Response {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let Some(bytes) = files.get(&path) else {
                return axum::http::StatusCode::NOT_FOUND.into_response();
            };
            (
                [(header::CONTENT_LENGTH, bytes.len().to_string())],
                bytes.clone(),
            )
                .into_response()
        }

        let files = std::collections::BTreeMap::from([
            (
                "config.json".to_string(),
                br#"{"model_type":"qwen2"}"#.to_vec(),
            ),
            (
                "model.safetensors".to_string(),
                b"server-authoritative package fixture".repeat(1024),
            ),
        ]);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let fixture_app = Router::new()
            .route("/{*path}", get(download_package_file))
            .with_state(Arc::new(files.clone()));
        let fixture_server = tokio::spawn(async move {
            axum::serve(listener, fixture_app).await.unwrap();
        });

        let descriptors = files
            .iter()
            .map(|(filename, bytes)| model_package::ModelPackageFile {
                filename: filename.clone(),
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            })
            .collect::<Vec<_>>();
        let expected_size = descriptors.iter().map(|file| file.size_bytes).sum();
        let expected_digest = model_package::package_digest(&descriptors).unwrap();
        let entry = model_index::ModelIndexEntry {
            id: "endpoint-package".to_string(),
            name: "Endpoint Package".to_string(),
            description: "A server-authoritative package acquisition fixture.".to_string(),
            download_url: None,
            filename: "endpoint-package".to_string(),
            format: "transformers".to_string(),
            size_bytes: expected_size,
            sha256: expected_digest.clone(),
            files: descriptors
                .iter()
                .map(|file| model_index::ModelIndexFile {
                    download_url: format!("http://{address}/{}", file.filename),
                    filename: file.filename.clone(),
                    size_bytes: file.size_bytes,
                    sha256: file.sha256.clone(),
                })
                .collect(),
            license: "Apache-2.0".to_string(),
            family: Some("qwen2".to_string()),
            parameter_count: None,
            quantization: None,
            tags: vec!["test".to_string()],
            downloadable: true,
            blocking_reasons: Vec::new(),
        };
        let temp = tempfile::tempdir().unwrap();
        let downloads =
            ModelDownloadManager::with_loopback_test_policy(temp.path().to_path_buf(), 1024 * 1024)
                .unwrap();
        let index = ModelIndexManager::from_test_entry(
            entry,
            temp.path().join("endpoint-package-index-state"),
        )
        .unwrap();
        let (state, _receiver) = test_server_state_with_services(
            temp.path().to_path_buf(),
            Some(Arc::clone(&downloads)),
            None,
            Some(index),
        );

        let response = handle_model_index_download(
            State(Arc::clone(&state)),
            axum::extract::Path("endpoint-package".to_string()),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["object"], "bloom.model_index_download");
        assert_eq!(body["model_index_id"], "endpoint-package");
        assert_eq!(body["accepted"], true);
        assert_eq!(body["already_installed"], false);
        assert_eq!(body["already_in_progress"], false);

        let response = handle_model_index_download(
            State(Arc::clone(&state)),
            axum::extract::Path("endpoint-package".to_string()),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["accepted"], false);
        assert_eq!(body["already_installed"], false);
        assert_eq!(body["already_in_progress"], true);

        let status = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = downloads.status().await;
                if matches!(
                    status.phase,
                    ModelDownloadPhase::Complete | ModelDownloadPhase::Error
                ) {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(status.phase, ModelDownloadPhase::Complete);
        for (filename, expected) in files {
            assert_eq!(
                tokio::fs::read(temp.path().join("endpoint-package").join(filename))
                    .await
                    .unwrap(),
                expected
            );
        }
        let provenance =
            model_provenance::read_provenance(temp.path(), "endpoint-package", expected_size)
                .unwrap()
                .unwrap();
        assert_eq!(provenance.sha256, expected_digest);
        assert_eq!(provenance.file_count, Some(descriptors.len()));
        assert_eq!(
            provenance.model_index_id.as_deref(),
            Some("endpoint-package")
        );

        let response = handle_model_index_download(
            State(Arc::clone(&state)),
            axum::extract::Path("endpoint-package".to_string()),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["accepted"], false);
        assert_eq!(body["already_installed"], true);
        assert_eq!(body["already_in_progress"], false);
        assert_eq!(body["status"]["phase"], "complete");

        model_provenance::remove_provenance(temp.path(), "endpoint-package").unwrap();
        let response = handle_model_index_download(
            State(state),
            axum::extract::Path("endpoint-package".to_string()),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["type"], "model_index_entry_conflict");
        fixture_server.abort();
    }

    #[tokio::test]
    async fn signed_index_download_endpoint_transactionally_upgrades_an_installed_alias() {
        async fn download_upgrade(State(bytes): State<Arc<Vec<u8>>>) -> axum::response::Response {
            tokio::time::sleep(Duration::from_millis(200)).await;
            (
                [(header::CONTENT_LENGTH, bytes.len().to_string())],
                bytes.as_ref().clone(),
            )
                .into_response()
        }

        let old = b"previous signed endpoint model";
        let new = b"replacement signed endpoint model".repeat(128);
        let index_id = "upgrade-endpoint";
        let filename = "upgrade.gguf";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let fixture_app = Router::new()
            .route("/upgrade.gguf", get(download_upgrade))
            .with_state(Arc::new(new.clone()));
        let fixture_server = tokio::spawn(async move {
            axum::serve(listener, fixture_app).await.unwrap();
        });

        let temp = tempfile::tempdir().unwrap();
        tokio::fs::write(temp.path().join(filename), old)
            .await
            .unwrap();
        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: Some(index_id.to_string()),
                filename: filename.to_string(),
                size_bytes: old.len() as u64,
                source_url: Some(format!(
                    "https://huggingface.co/acme/old/resolve/{}/{}",
                    "11".repeat(20),
                    filename
                )),
                source_host: Some("huggingface.co".to_string()),
                sha256: format!("{:x}", Sha256::digest(old)),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let entry = model_index::ModelIndexEntry {
            id: index_id.to_string(),
            name: "Upgrade Endpoint".to_string(),
            description: "A transactional signed upgrade fixture.".to_string(),
            download_url: Some(format!("http://{address}/upgrade.gguf")),
            filename: filename.to_string(),
            format: "gguf".to_string(),
            size_bytes: new.len() as u64,
            sha256: format!("{:x}", Sha256::digest(&new)),
            files: Vec::new(),
            license: "Apache-2.0".to_string(),
            family: Some("qwen2".to_string()),
            parameter_count: None,
            quantization: None,
            tags: vec!["test".to_string()],
            downloadable: true,
            blocking_reasons: Vec::new(),
        };
        let downloads =
            ModelDownloadManager::with_loopback_test_policy(temp.path().to_path_buf(), 1024 * 1024)
                .unwrap();
        let index = ModelIndexManager::from_test_entry(
            entry,
            temp.path().join("upgrade-endpoint-index-state"),
        )
        .unwrap();
        let (state, _receiver) = test_server_state_with_services(
            temp.path().to_path_buf(),
            Some(Arc::clone(&downloads)),
            None,
            Some(index),
        );

        state.load_in_progress.store(true, Ordering::Release);
        let blocked = handle_model_index_download(
            State(Arc::clone(&state)),
            axum::extract::Path(index_id.to_string()),
        )
        .await;
        assert_eq!(blocked.status(), axum::http::StatusCode::CONFLICT);
        let blocked: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(blocked.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(blocked["error"]["type"], "model_index_upgrade_blocked");
        assert_eq!(
            tokio::fs::read(temp.path().join(filename)).await.unwrap(),
            old
        );
        state.load_in_progress.store(false, Ordering::Release);

        let response = handle_model_index_download(
            State(Arc::clone(&state)),
            axum::extract::Path(index_id.to_string()),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["accepted"], true);
        assert_eq!(body["upgrading"], true);

        let load_error = prepare_catalog_model_load(&state, filename)
            .await
            .unwrap_err();
        assert_eq!(load_error.code, "model_upgrade_in_progress");
        let removal_error = remove_catalog_model(&state, filename).await.unwrap_err();
        assert!(matches!(
            removal_error,
            ModelRemovalError::Conflict {
                code: "model_upgrade_in_progress",
                ..
            }
        ));
        assert_eq!(
            tokio::fs::read(temp.path().join(filename)).await.unwrap(),
            old
        );

        let joined = handle_model_index_download(
            State(Arc::clone(&state)),
            axum::extract::Path(index_id.to_string()),
        )
        .await;
        assert_eq!(joined.status(), axum::http::StatusCode::ACCEPTED);
        let joined: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(joined.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(joined["accepted"], false);
        assert_eq!(joined["already_in_progress"], true);
        assert_eq!(joined["upgrading"], true);

        let status = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = downloads.status().await;
                if matches!(
                    status.phase,
                    ModelDownloadPhase::Complete | ModelDownloadPhase::Error
                ) {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(status.phase, ModelDownloadPhase::Complete, "{status:?}");
        assert_eq!(
            tokio::fs::read(temp.path().join(filename)).await.unwrap(),
            new
        );
        let provenance = model_provenance::read_provenance(temp.path(), filename, new.len() as u64)
            .unwrap()
            .unwrap();
        assert_eq!(provenance.model_index_id.as_deref(), Some(index_id));
        assert_eq!(provenance.sha256, format!("{:x}", Sha256::digest(&new)));
        assert!(!temp.path().join(model_upgrade::UPGRADE_DIRECTORY).exists());

        let installed =
            handle_model_index_download(State(state), axum::extract::Path(index_id.to_string()))
                .await;
        assert_eq!(installed.status(), axum::http::StatusCode::OK);
        let installed: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(installed.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(installed["already_installed"], true);
        assert_eq!(installed["upgrading"], false);
        fixture_server.abort();
    }

    #[tokio::test]
    async fn ollama_pull_transactionally_upgrades_and_reuses_a_signed_model_package() {
        async fn download_package_file(
            State(files): State<Arc<std::collections::BTreeMap<String, Vec<u8>>>>,
            axum::extract::Path(path): axum::extract::Path<String>,
        ) -> axum::response::Response {
            let Some(bytes) = files.get(&path) else {
                return axum::http::StatusCode::NOT_FOUND.into_response();
            };
            (
                [(header::CONTENT_LENGTH, bytes.len().to_string())],
                bytes.clone(),
            )
                .into_response()
        }

        let files = std::collections::BTreeMap::from([
            (
                "config.json".to_string(),
                br#"{"model_type":"qwen2"}"#.to_vec(),
            ),
            (
                "model.safetensors".to_string(),
                b"Ollama package pull fixture".repeat(1024),
            ),
            (
                "tokenizer.json".to_string(),
                br#"{"version":"1.0"}"#.to_vec(),
            ),
        ]);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let fixture_app = Router::new()
            .route("/{*path}", get(download_package_file))
            .with_state(Arc::new(files.clone()));
        let fixture_server = tokio::spawn(async move {
            axum::serve(listener, fixture_app).await.unwrap();
        });

        let descriptors = files
            .iter()
            .map(|(filename, bytes)| model_package::ModelPackageFile {
                filename: filename.clone(),
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            })
            .collect::<Vec<_>>();
        let expected_size = descriptors.iter().map(|file| file.size_bytes).sum();
        let expected_digest = model_package::package_digest(&descriptors).unwrap();
        let entry = model_index::ModelIndexEntry {
            id: "ollama-package".to_string(),
            name: "Ollama Package".to_string(),
            description: "A verified Ollama multi-file acquisition fixture.".to_string(),
            download_url: None,
            filename: "ollama-package".to_string(),
            format: "transformers".to_string(),
            size_bytes: expected_size,
            sha256: expected_digest.clone(),
            files: descriptors
                .iter()
                .map(|file| model_index::ModelIndexFile {
                    download_url: format!("http://{address}/{}", file.filename),
                    filename: file.filename.clone(),
                    size_bytes: file.size_bytes,
                    sha256: file.sha256.clone(),
                })
                .collect(),
            license: "Apache-2.0".to_string(),
            family: Some("qwen2".to_string()),
            parameter_count: None,
            quantization: None,
            tags: vec!["test".to_string()],
            downloadable: true,
            blocking_reasons: Vec::new(),
        };
        let temp = tempfile::tempdir().unwrap();
        let previous_id = "ollama-package-old.gguf";
        let previous = b"previous Ollama signed model";
        tokio::fs::write(temp.path().join(previous_id), previous)
            .await
            .unwrap();
        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: Some("ollama-package".to_string()),
                filename: previous_id.to_string(),
                size_bytes: previous.len() as u64,
                source_url: Some(format!(
                    "https://huggingface.co/acme/old/resolve/{}/{}",
                    "11".repeat(20),
                    previous_id
                )),
                source_host: Some("huggingface.co".to_string()),
                sha256: format!("{:x}", Sha256::digest(previous)),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let downloads =
            ModelDownloadManager::with_loopback_test_policy(temp.path().to_path_buf(), 1024 * 1024)
                .unwrap();
        let index = ModelIndexManager::from_test_entry(
            entry,
            temp.path().join("ollama-package-index-state"),
        )
        .unwrap();
        let (state, _receiver) = test_server_state_with_services(
            temp.path().to_path_buf(),
            Some(downloads),
            None,
            Some(index),
        );

        let request = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "ollama-package",
            "stream": false
        }))
        .unwrap();
        let response = handle_ollama_pull(State(Arc::clone(&state)), Ok(Json(request))).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body, json!({"status": "success"}));
        assert!(!temp.path().join(previous_id).exists());
        for (filename, expected) in &files {
            assert_eq!(
                tokio::fs::read(temp.path().join("ollama-package").join(filename))
                    .await
                    .unwrap(),
                *expected
            );
        }
        let provenance =
            model_provenance::read_provenance(temp.path(), "ollama-package", expected_size)
                .unwrap()
                .unwrap();
        assert_eq!(provenance.sha256, expected_digest);
        assert_eq!(provenance.file_count, Some(descriptors.len()));
        assert_eq!(provenance.model_index_id.as_deref(), Some("ollama-package"));

        let repeat = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "ollama-package",
            "stream": true
        }))
        .unwrap();
        let response = handle_ollama_pull(State(state), Ok(Json(repeat))).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .as_ref(),
            b"{\"status\":\"success\"}\n"
        );
        fixture_server.abort();
    }

    #[tokio::test]
    async fn ollama_pull_requires_verified_acquisition_services() {
        let temp = tempfile::tempdir().unwrap();
        let request = || {
            serde_json::from_value::<OllamaPullRequest>(json!({
                "model": "tiny-pull",
                "stream": false
            }))
            .unwrap()
        };
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_ollama_pull(State(state), Ok(Json(request()))).await;
        assert_eq!(response.status(), axum::http::StatusCode::FORBIDDEN);

        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);
        let downloads =
            ModelDownloadManager::with_storage(temp.path().to_path_buf(), 8192, storage).unwrap();
        let (state, _receiver) =
            test_server_state_with_acquisitions(temp.path().to_path_buf(), Some(downloads), None);
        let response = handle_ollama_pull(State(state), Ok(Json(request()))).await;
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );

        let invalid = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "tiny-pull",
            "insecure": true
        }))
        .unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let response = handle_ollama_pull(State(state), Ok(Json(invalid))).await;
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn ollama_pull_is_idempotent_for_a_matching_signed_catalog_entry() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = b"already verified Ollama pull fixture";
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        let filename = "tiny-pull.gguf";
        let model_path = temp.path().join(filename);
        tokio::fs::write(&model_path, bytes).await.unwrap();
        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: Some("tiny-pull".to_string()),
                filename: filename.to_string(),
                size_bytes: bytes.len() as u64,
                source_url: Some(
                    "https://huggingface.co/acme/tiny/resolve/dededededededededededededededededededede/tiny-pull.gguf"
                        .to_string(),
                ),
                source_host: Some("huggingface.co".to_string()),
                sha256: sha256.clone(),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let index = test_signed_index(
            temp.path(),
            "tiny-pull",
            filename,
            bytes.len() as u64,
            &sha256,
        )
        .await;
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);
        let downloads =
            ModelDownloadManager::with_storage(temp.path().to_path_buf(), 8192, storage).unwrap();
        let (state, _receiver) = test_server_state_with_services(
            temp.path().to_path_buf(),
            Some(downloads),
            None,
            Some(index),
        );

        let non_stream = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "tiny-pull",
            "stream": false
        }))
        .unwrap();
        let response = handle_ollama_pull(State(Arc::clone(&state)), Ok(Json(non_stream))).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body, json!({"status": "success"}));

        let streaming = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "tiny-pull",
            "stream": true
        }))
        .unwrap();
        let response = handle_ollama_pull(State(Arc::clone(&state)), Ok(Json(streaming))).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/x-ndjson"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), b"{\"status\":\"success\"}\n");

        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: None,
                filename: filename.to_string(),
                size_bytes: bytes.len() as u64,
                source_url: None,
                source_host: Some("huggingface.co".to_string()),
                sha256: sha256.clone(),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let missing_alias = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "tiny-pull",
            "stream": false
        }))
        .unwrap();
        let response = handle_ollama_pull(State(Arc::clone(&state)), Ok(Json(missing_alias))).await;
        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);

        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: Some("different-signed-id".to_string()),
                filename: filename.to_string(),
                size_bytes: bytes.len() as u64,
                source_url: None,
                source_host: Some("huggingface.co".to_string()),
                sha256,
                license: Some("MIT".to_string()),
            },
        )
        .await
        .unwrap();
        let occupied_by_different_alias = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "tiny-pull",
            "stream": false
        }))
        .unwrap();
        let response =
            handle_ollama_pull(State(state), Ok(Json(occupied_by_different_alias))).await;
        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn ollama_pull_stream_installs_a_locally_served_verified_fixture() {
        async fn download_fixture(State(bytes): State<Arc<Vec<u8>>>) -> axum::response::Response {
            (
                [(header::CONTENT_LENGTH, bytes.len().to_string())],
                bytes.as_ref().clone(),
            )
                .into_response()
        }

        let bytes = Arc::new(b"Ollama verified pull stream fixture".repeat(2048));
        let sha256 = format!("{:x}", Sha256::digest(bytes.as_slice()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let fixture_app = Router::new()
            .route("/stream.gguf", get(download_fixture))
            .with_state(Arc::clone(&bytes));
        let fixture_server = tokio::spawn(async move {
            axum::serve(listener, fixture_app).await.unwrap();
        });

        let temp = tempfile::tempdir().unwrap();
        let downloads =
            ModelDownloadManager::with_loopback_test_policy(temp.path().to_path_buf(), 1024 * 1024)
                .unwrap();
        let entry = model_index::ModelIndexEntry {
            id: "stream-pull".to_string(),
            name: "Stream Pull".to_string(),
            description: "A local verified pull stream fixture.".to_string(),
            download_url: Some(format!("http://{address}/stream.gguf")),
            filename: "stream.gguf".to_string(),
            format: "gguf".to_string(),
            size_bytes: bytes.len() as u64,
            sha256: sha256.clone(),
            files: Vec::new(),
            license: "Apache-2.0".to_string(),
            family: None,
            parameter_count: None,
            quantization: None,
            tags: vec!["test".to_string()],
            downloadable: true,
            blocking_reasons: Vec::new(),
        };
        let index =
            ModelIndexManager::from_test_entry(entry, temp.path().join("stream-pull-index-state"))
                .unwrap();
        let (state, _receiver) = test_server_state_with_services(
            temp.path().to_path_buf(),
            Some(downloads),
            None,
            Some(index),
        );
        let request = serde_json::from_value::<OllamaPullRequest>(json!({
            "model": "stream-pull",
            "stream": true
        }))
        .unwrap();

        let response = handle_ollama_pull(State(state), Ok(Json(request))).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/x-ndjson"
        );
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let events = body
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(!events.is_empty());
        assert_eq!(events.last().unwrap(), &json!({"status": "success"}));
        assert!(events.iter().all(|event| event.get("error").is_none()));
        assert_eq!(
            tokio::fs::read(temp.path().join("stream.gguf"))
                .await
                .unwrap(),
            bytes.as_ref().clone()
        );
        let provenance =
            model_provenance::read_provenance(temp.path(), "stream.gguf", bytes.len() as u64)
                .unwrap()
                .unwrap();
        assert_eq!(provenance.sha256, sha256);
        assert_eq!(provenance.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(provenance.model_index_id.as_deref(), Some("stream-pull"));
        fixture_server.abort();
    }

    #[tokio::test]
    async fn readiness_is_versioned_and_advertises_ui_compatibility() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_ready(State(state)).await;
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["schema_version"], READINESS_SCHEMA_VERSION);
        assert_eq!(body["object"], READINESS_OBJECT);
        assert_eq!(body["protocol_version"], READINESS_PROTOCOL_VERSION);
        assert_eq!(
            body["minimum_ui_protocol_version"],
            MINIMUM_UI_PROTOCOL_VERSION
        );
        assert_eq!(
            body["maximum_ui_protocol_version"],
            MAXIMUM_UI_PROTOCOL_VERSION
        );
        let minimum_ui_protocol = body["minimum_ui_protocol_version"].as_u64().unwrap();
        let maximum_ui_protocol = body["maximum_ui_protocol_version"].as_u64().unwrap();
        let server_protocol = body["protocol_version"].as_u64().unwrap();
        assert!(minimum_ui_protocol <= maximum_ui_protocol);
        assert!((minimum_ui_protocol..=maximum_ui_protocol).contains(&server_protocol));
        assert_eq!(body["server_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(body["status"], "not_ready");
        for field in [
            "progress",
            "model",
            "loading",
            "load_error",
            "input_modalities",
            "model_tasks",
            "context_window",
            "in_flight_requests",
            "available_permits",
            "memory_pressure_high",
            "ram_utilization",
        ] {
            assert!(body.get(field).is_some(), "missing readiness field {field}");
        }
        assert_eq!(body["model"], "not loaded");
        assert_eq!(body["loading"], false);
        assert!(body["load_error"].is_null());
        assert!(body["input_modalities"].as_array().unwrap().is_empty());
        assert!(body["model_tasks"].as_array().unwrap().is_empty());
        assert!(body["available_permits"].as_u64().unwrap() > 0);
        assert!(body["ram_utilization"].as_f64().unwrap() >= 0.0);
        assert!(body["ram_utilization"].as_f64().unwrap() <= 1.0);
        assert!(body.get("context_window").is_some());
        assert!(body["context_window"].is_null());
    }

    #[tokio::test]
    async fn readiness_publishes_bounded_active_model_tasks() {
        let embedding_temp = tempfile::tempdir().unwrap();
        let (embedding_state, _) =
            test_server_state_with_embedding_runtime(embedding_temp.path().to_path_buf()).await;
        let embedding_response = handle_ready(State(embedding_state)).await;
        assert_eq!(embedding_response.status(), axum::http::StatusCode::OK);
        let embedding_body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(embedding_response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            embedding_body["model_tasks"],
            json!(["embedding", "rerank"])
        );

        let text_temp = tempfile::tempdir().unwrap();
        let text_state = test_server_state_with_text_runtime(
            text_temp.path().to_path_buf(),
            Arc::new(AtomicU64::new(0)),
        )
        .await;
        let text_response = handle_ready(State(text_state)).await;
        assert_eq!(text_response.status(), axum::http::StatusCode::OK);
        let text_body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(text_response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(text_body["model_tasks"], json!(["generation"]));
    }

    #[tokio::test]
    async fn observability_snapshot_is_versioned_and_path_free() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        *state.requested_model.write().await = Some("tiny.gguf".to_string());
        state.load_in_progress.store(true, Ordering::Release);
        state.load_progress.store(37, Ordering::Release);
        state.metrics.record_request_start();
        state.metrics.record_request_end(true, 0.1, 3, 5);

        let response = handle_observability(State(state)).await.into_response();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(body["schema_version"], 1);
        assert_eq!(body["object"], "bloom.observability_snapshot");
        assert_eq!(body["server"]["version"], env!("CARGO_PKG_VERSION"));
        assert!(body["server"]["uptime_seconds"].is_u64());
        assert_eq!(body["model"], "not loaded");
        assert_eq!(body["load"]["phase"], "loading");
        assert_eq!(body["load"]["progress"], 37);
        assert_eq!(body["load"]["requested_model"], "tiny.gguf");
        assert_eq!(body["load"]["failure_present"], false);
        assert_eq!(body["requests"]["total"], 1);
        assert_eq!(body["requests"]["completed"], 1);
        assert_eq!(body["tokens"]["prompt_total"], 5);
        assert_eq!(body["tokens"]["generated_total"], 3);
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(!text.contains(temp.path().to_string_lossy().as_ref()));
    }

    #[tokio::test]
    async fn model_inventory_endpoint_is_path_free_and_downloadable() {
        let temp = tempfile::tempdir().unwrap();
        let model_bytes = b"inventory model";
        std::fs::write(temp.path().join("tiny.gguf"), model_bytes).unwrap();
        let revision = "a".repeat(40);
        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: None,
                filename: "tiny.gguf".to_string(),
                size_bytes: model_bytes.len() as u64,
                source_url: Some(format!(
                    "https://huggingface.co/acme/tiny/resolve/{revision}/tiny.gguf?token=secret"
                )),
                source_host: Some("huggingface.co".to_string()),
                sha256: "ab".repeat(32),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_inventory(State(state)).await;

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(
            response.headers()[header::CONTENT_DISPOSITION],
            "attachment; filename=\"bloom-model-inventory.json\""
        );
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["schema_version"],
            model_inventory::MODEL_INVENTORY_SCHEMA_VERSION
        );
        assert_eq!(body["object"], "bloom.model_inventory");
        assert_eq!(body["summary"]["model_count"], 1);
        assert_eq!(body["summary"]["source_locked_count"], 1);
        assert_eq!(body["models"][0]["id"], "tiny.gguf");
        assert_eq!(body["models"][0]["source"]["revision"], revision);
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(!text.contains(temp.path().to_string_lossy().as_ref()));
        assert!(!text.contains("secret"));
    }

    #[tokio::test]
    async fn model_inventory_reconciliation_reports_an_in_sync_catalog() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tiny.gguf"), b"gguf").unwrap();
        let expected = model_inventory::ModelInventory::from_catalog(
            &ModelCatalog::scan(temp.path(), None).unwrap(),
        );
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_inventory_reconcile(State(state), Json(expected)).await;

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "bloom.model_inventory_reconciliation");
        assert_eq!(body["in_sync"], true);
        assert_eq!(body["summary"]["matching_count"], 1);
        assert_eq!(body["summary"]["drift_count"], 0);
        assert_eq!(body["drift"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn model_inventory_reconciliation_rejects_an_unknown_schema() {
        let temp = tempfile::tempdir().unwrap();
        let mut expected = model_inventory::ModelInventory::from_catalog(
            &ModelCatalog::scan(temp.path(), None).unwrap(),
        );
        expected.schema_version = 3;
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_inventory_reconcile(State(state), Json(expected)).await;

        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "invalid_model_inventory");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("version"));
    }

    #[tokio::test]
    async fn model_inventory_reconciliation_honors_its_route_body_limit() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route(
                "/",
                post(handle_model_inventory_reconcile).layer(DefaultBodyLimit::max(32)),
            )
            .with_state(state);
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from("x".repeat(33)))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn model_inventory_restore_is_explicitly_opt_in() {
        let temp = tempfile::tempdir().unwrap();
        let expected = model_inventory::ModelInventory::from_catalog(
            &ModelCatalog::scan(temp.path(), None).unwrap(),
        );
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_inventory_restore(
            State(state),
            axum::extract::Path("missing.gguf".to_string()),
            Json(expected),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn model_inventory_restore_queues_only_a_locked_verified_download() {
        let temp = tempfile::tempdir().unwrap();
        let model_bytes = b"inventory restore fixture";
        std::fs::write(temp.path().join("restore.gguf"), model_bytes).unwrap();
        let revision = "a".repeat(40);
        let source_url =
            format!("https://huggingface.co/acme/restore/resolve/{revision}/restore.gguf");
        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: None,
                filename: "restore.gguf".to_string(),
                size_bytes: model_bytes.len() as u64,
                source_url: Some(source_url.clone()),
                source_host: Some("huggingface.co".to_string()),
                sha256: format!("{:x}", Sha256::digest(model_bytes)),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let expected = model_inventory::ModelInventory::from_catalog(
            &ModelCatalog::scan(temp.path(), None).unwrap(),
        );
        std::fs::remove_file(temp.path().join("restore.gguf")).unwrap();
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);
        let manager =
            ModelDownloadManager::with_storage(temp.path().to_path_buf(), 1024, storage).unwrap();
        let (state, _receiver) = test_server_state_with_acquisitions(
            temp.path().to_path_buf(),
            Some(Arc::clone(&manager)),
            None,
        );

        let response = handle_model_inventory_restore(
            State(state),
            axum::extract::Path("restore.gguf".to_string()),
            Json(expected),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "bloom.model_inventory_restore");
        assert_eq!(body["accepted"], true);
        assert_eq!(body["model"], "restore.gguf");
        assert_eq!(body["status"]["phase"], "queued");
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(!text.contains(&source_url));
        assert!(!text.contains(&format!("{:x}", Sha256::digest(model_bytes))));
        assert!(manager.cancel().await);
    }

    #[tokio::test]
    async fn model_inventory_restore_honors_its_route_body_limit() {
        let temp = tempfile::tempdir().unwrap();
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);
        let manager =
            ModelDownloadManager::with_storage(temp.path().to_path_buf(), 1024, storage).unwrap();
        let (state, _receiver) =
            test_server_state_with_acquisitions(temp.path().to_path_buf(), Some(manager), None);
        let app = Router::new()
            .route(
                "/{id}",
                post(handle_model_inventory_restore).layer(DefaultBodyLimit::max(32)),
            )
            .with_state(state);
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/missing.gguf")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from("x".repeat(33)))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn model_download_endpoint_is_explicitly_opt_in() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_download_start(
            State(state),
            Json(ModelDownloadRequest {
                url: "https://huggingface.co/acme/model/resolve/main/model.gguf".to_string(),
                filename: "model.gguf".to_string(),
                sha256: "00".repeat(32),
                license: None,
                expected_size_bytes: None,
                model_index_id: None,
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn model_download_source_inspection_is_explicitly_opt_in() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_download_source_inspect(
            State(state),
            Json(ModelDownloadSourceRequest {
                url: "https://huggingface.co/acme/model/resolve/main/model.gguf".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn model_import_endpoint_is_explicitly_opt_in() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_import_begin(
            State(state),
            Json(ModelImportRequest {
                filename: "model.gguf".to_string(),
                total_bytes: 4,
                sha256: "00".repeat(32),
                source_url: None,
                license: None,
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn model_import_routes_install_a_verified_chunked_file() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = b"GGUF handler import fixture";
        let manager = ModelImportManager::new(temp.path().to_path_buf(), 1024, 64).unwrap();
        let (state, _receiver) =
            test_server_state_with_imports(temp.path().to_path_buf(), Some(manager));
        let app = Router::new()
            .route("/imports", post(handle_model_import_begin))
            .route(
                "/imports/{filename}",
                put(handle_model_import_chunk).layer(DefaultBodyLimit::max(64)),
            )
            .route(
                "/imports/{filename}/complete",
                post(handle_model_import_complete),
            )
            .with_state(Arc::clone(&state));
        let begin = axum::http::Request::builder()
            .method("POST")
            .uri("/imports")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                json!({
                    "filename": "local.gguf",
                    "total_bytes": bytes.len(),
                    "sha256": format!("{:x}", Sha256::digest(bytes))
                })
                .to_string(),
            ))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(begin).await.unwrap().status(),
            axum::http::StatusCode::OK
        );
        let wrong_offset = axum::http::Request::builder()
            .method("PUT")
            .uri("/imports/local.gguf")
            .header("upload-offset", "1")
            .body(axum::body::Body::from("x"))
            .unwrap();
        let wrong_offset = app.clone().oneshot(wrong_offset).await.unwrap();
        assert_eq!(wrong_offset.status(), axum::http::StatusCode::CONFLICT);
        let wrong_offset = axum::body::to_bytes(wrong_offset.into_body(), usize::MAX)
            .await
            .unwrap();
        let wrong_offset: serde_json::Value = serde_json::from_slice(&wrong_offset).unwrap();
        assert_eq!(wrong_offset["error"]["expected_offset"], 0);
        let chunk = axum::http::Request::builder()
            .method("PUT")
            .uri("/imports/local.gguf")
            .header("upload-offset", "0")
            .body(axum::body::Body::from(bytes.as_slice()))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(chunk).await.unwrap().status(),
            axum::http::StatusCode::OK
        );
        let complete = axum::http::Request::builder()
            .method("POST")
            .uri("/imports/local.gguf/complete")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(complete).await.unwrap().status(),
            axum::http::StatusCode::OK
        );
        assert_eq!(
            std::fs::read(temp.path().join("local.gguf")).unwrap(),
            bytes
        );

        let response = handle_model_catalog(State(state)).await;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["import"]["enabled"], true);
        assert_eq!(body["import"]["max_bytes"], 1024);
        assert_eq!(body["import"]["max_chunk_bytes"], 64);
        assert_eq!(body["data"][0]["id"], "local.gguf");
    }

    #[tokio::test]
    async fn model_import_chunk_honors_its_route_body_limit() {
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelImportManager::new(temp.path().to_path_buf(), 1024, 64).unwrap();
        manager
            .begin(ModelImportRequest {
                filename: "limit.gguf".to_string(),
                total_bytes: 5,
                sha256: format!("{:x}", Sha256::digest(b"12345")),
                source_url: None,
                license: None,
            })
            .await
            .unwrap();
        let (state, _receiver) =
            test_server_state_with_imports(temp.path().to_path_buf(), Some(manager));
        let app = Router::new()
            .route(
                "/imports/{filename}",
                put(handle_model_import_chunk).layer(DefaultBodyLimit::max(4)),
            )
            .with_state(state);
        let request = axum::http::Request::builder()
            .method("PUT")
            .uri("/imports/limit.gguf")
            .header("upload-offset", "0")
            .body(axum::body::Body::from("12345"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn model_remove_endpoint_deletes_an_inactive_catalog_entry() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("remove.gguf");
        std::fs::write(&model, b"gguf").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_remove(
            State(state),
            Json(ModelRemoveRequest {
                id: "remove.gguf".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert!(!model.exists());
    }

    #[tokio::test]
    async fn model_remove_endpoint_rejects_ambiguous_and_missing_ids() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("keep.gguf");
        std::fs::write(&model, b"gguf").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let ambiguous = handle_model_remove(
            State(Arc::clone(&state)),
            Json(ModelRemoveRequest {
                id: " keep.gguf ".to_string(),
            }),
        )
        .await;
        assert_eq!(ambiguous.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(model.exists());

        let missing = handle_model_remove(
            State(state),
            Json(ModelRemoveRequest {
                id: "missing.gguf".to_string(),
            }),
        )
        .await;
        assert_eq!(missing.status(), axum::http::StatusCode::NOT_FOUND);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(missing.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["type"], "model_not_found");
        assert!(model.exists());
    }

    #[tokio::test]
    async fn model_remove_endpoint_blocks_during_a_lifecycle_operation() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("keep.gguf");
        std::fs::write(&model, b"gguf").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        state.load_in_progress.store(true, Ordering::Release);

        let response = handle_model_remove(
            State(state),
            Json(ModelRemoveRequest {
                id: "keep.gguf".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        assert!(model.exists());
    }

    #[tokio::test]
    async fn model_integrity_endpoint_reports_a_verified_catalog_file() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = b"verified endpoint model".repeat(4096);
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        tokio::fs::write(temp.path().join("verified.gguf"), &bytes)
            .await
            .unwrap();
        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Import,
                model_index_id: None,
                filename: "verified.gguf".to_string(),
                size_bytes: bytes.len() as u64,
                source_url: None,
                source_host: None,
                sha256,
                license: None,
            },
        )
        .await
        .unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_integrity_start(
            State(Arc::clone(&state)),
            Json(ModelIntegrityRequest {
                id: "verified.gguf".to_string(),
            }),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if state.model_integrity.status().await.matches_expected == Some(true) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let response = handle_model_catalog(State(state)).await;
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["integrity"]["phase"], "complete");
        assert_eq!(body["integrity"]["matches_expected"], true);
        assert_eq!(body["integrity"]["model_id"], "verified.gguf");
    }

    #[tokio::test]
    async fn model_integrity_endpoint_blocks_during_model_loading() {
        let temp = tempfile::tempdir().unwrap();
        tokio::fs::write(temp.path().join("loading.gguf"), b"gguf")
            .await
            .unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        state.load_in_progress.store(true, Ordering::Release);

        let response = handle_model_integrity_start(
            State(state),
            Json(ModelIntegrityRequest {
                id: "loading.gguf".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn checksum_mismatch_blocks_model_loading() {
        let temp = tempfile::tempdir().unwrap();
        let original = b"original model";
        let modified = b"modified model";
        assert_eq!(original.len(), modified.len());
        tokio::fs::write(temp.path().join("changed.gguf"), original)
            .await
            .unwrap();
        model_provenance::write_provenance(
            temp.path(),
            model_provenance::ModelProvenanceDraft {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: None,
                filename: "changed.gguf".to_string(),
                size_bytes: original.len() as u64,
                source_url: None,
                source_host: None,
                sha256: format!("{:x}", Sha256::digest(original)),
                license: None,
            },
        )
        .await
        .unwrap();
        tokio::fs::write(temp.path().join("changed.gguf"), modified)
            .await
            .unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        state.model_integrity.start("changed.gguf").await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if state.model_integrity.status().await.matches_expected == Some(false) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let (restarted_state, mut restarted_receiver) =
            test_server_state(temp.path().to_path_buf());

        let response = handle_model_switch(
            State(restarted_state),
            Json(ModelSwitchRequest {
                id: "changed.gguf".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        assert!(restarted_receiver.try_recv().is_err());
    }

    #[tokio::test]
    async fn model_catalog_cache_avoids_repeated_directory_walks() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("one.gguf"), b"one").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        assert_eq!(
            state.model_catalog_snapshot().await.unwrap().0.models.len(),
            1
        );
        std::fs::write(temp.path().join("two.gguf"), b"two").unwrap();
        assert_eq!(
            state.model_catalog_snapshot().await.unwrap().0.models.len(),
            1
        );

        state
            .model_catalog_cache
            .write()
            .await
            .as_mut()
            .unwrap()
            .refreshed_at = Instant::now() - MODEL_CATALOG_CACHE_TTL;
        assert_eq!(
            state.model_catalog_snapshot().await.unwrap().0.models.len(),
            2
        );
    }

    #[tokio::test]
    async fn fresh_model_catalog_snapshot_observes_external_changes() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("one.gguf"), b"one").unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        assert_eq!(
            state.model_catalog_snapshot().await.unwrap().0.models.len(),
            1
        );
        std::fs::write(temp.path().join("two.gguf"), b"two").unwrap();
        assert_eq!(
            state
                .fresh_model_catalog_snapshot()
                .await
                .unwrap()
                .0
                .models
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn model_switch_endpoint_queues_only_catalog_entries() {
        let temp = tempfile::tempdir().unwrap();
        let model_path = write_test_model_manifest(temp.path(), "tiny");
        let (state, mut receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_switch(
            State(Arc::clone(&state)),
            Json(ModelSwitchRequest {
                id: "tiny".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
        assert!(state.load_in_progress.load(Ordering::Acquire));
        assert!(!state.ready.load(Ordering::Acquire));
        let queued = receiver.recv().await.unwrap();
        assert_eq!(queued.path, model_path.canonicalize().unwrap());
        assert_eq!(queued.catalog_id.as_deref(), Some("tiny"));
    }

    #[tokio::test]
    async fn model_load_admission_joins_only_the_exact_active_sequence() {
        let temp = tempfile::tempdir().unwrap();
        let model_path = write_test_model_manifest(temp.path(), "tiny")
            .canonicalize()
            .unwrap();
        let other_path = write_test_model_manifest(temp.path(), "other")
            .canonicalize()
            .unwrap();
        let (state, mut receiver) = test_server_state(temp.path().to_path_buf());

        let first = state
            .admit_model_load(model_path.clone(), Some("tiny".to_string()), true)
            .await
            .unwrap();
        let ModelLoadAdmission::Loading {
            sequence,
            queued,
            completion: first_completion,
        } = first
        else {
            panic!("the first admission must queue a load");
        };
        assert!(queued);

        let joined = state
            .admit_model_load(model_path.clone(), Some("tiny".to_string()), true)
            .await
            .unwrap();
        let ModelLoadAdmission::Loading {
            sequence: joined_sequence,
            queued: joined_queued,
            completion: joined_completion,
        } = joined
        else {
            panic!("the matching admission must join the active load");
        };
        assert_eq!(joined_sequence, sequence);
        assert!(!joined_queued);
        assert!(matches!(
            state
                .admit_model_load(other_path, Some("other".to_string()), true)
                .await,
            Err(ModelLoadAdmissionError::Busy)
        ));

        let queued_request = receiver.recv().await.unwrap();
        assert_eq!(queued_request.sequence, sequence);
        assert_eq!(queued_request.path, model_path);
        assert!(receiver.try_recv().is_err());

        state.load_in_progress.store(false, Ordering::Release);
        state
            .finish_model_load(
                sequence,
                ModelLoadOutcome::Ready {
                    model_id: "tiny-runtime".to_string(),
                },
            )
            .await;
        assert_eq!(
            ollama::wait_for_model_activation(first_completion)
                .await
                .unwrap(),
            "tiny-runtime"
        );
        assert_eq!(
            ollama::wait_for_model_activation(joined_completion)
                .await
                .unwrap(),
            "tiny-runtime"
        );

        let failed = state
            .admit_model_load(
                write_test_model_manifest(temp.path(), "failed")
                    .canonicalize()
                    .unwrap(),
                Some("failed".to_string()),
                true,
            )
            .await
            .unwrap();
        let ModelLoadAdmission::Loading {
            sequence: failed_sequence,
            completion: failed_completion,
            ..
        } = failed
        else {
            panic!("the later admission must receive a new load sequence");
        };
        assert!(failed_sequence > sequence);
        let failed_request = receiver.recv().await.unwrap();
        assert_eq!(failed_request.sequence, failed_sequence);
        state.load_in_progress.store(false, Ordering::Release);
        state
            .finish_model_load(
                failed_sequence,
                ModelLoadOutcome::Failed {
                    message: "deterministic loader failure".to_string(),
                },
            )
            .await;
        let error = ollama::wait_for_model_activation(failed_completion)
            .await
            .unwrap_err();
        assert_eq!(error.status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(error.message.contains("deterministic loader failure"));
    }

    #[tokio::test]
    async fn ollama_activation_queues_an_inactive_catalog_model_and_waits_for_completion() {
        let temp = tempfile::tempdir().unwrap();
        let model_path = write_test_model_manifest(temp.path(), "tiny")
            .canonicalize()
            .unwrap();
        let (state, mut receiver) = test_server_state(temp.path().to_path_buf());
        let activation_state = Arc::clone(&state);
        let activation =
            tokio::spawn(
                async move { ollama::activate_ollama_model(&activation_state, "tiny").await },
            );

        let queued = receiver.recv().await.unwrap();
        assert_eq!(queued.path, model_path);
        assert_eq!(queued.catalog_id.as_deref(), Some("tiny"));
        assert!(!activation.is_finished());

        state.load_in_progress.store(false, Ordering::Release);
        state
            .finish_model_load(
                queued.sequence,
                ModelLoadOutcome::Ready {
                    model_id: "tiny-llama".to_string(),
                },
            )
            .await;
        assert_eq!(activation.await.unwrap().unwrap(), "tiny-llama");
    }

    #[tokio::test]
    async fn model_preflight_endpoint_returns_bounded_runtime_details() {
        let temp = tempfile::tempdir().unwrap();
        write_test_model_manifest(temp.path(), "tiny");
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_preflight(
            State(state),
            Json(ModelPreflightRequest {
                id: "tiny".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["schema_version"], 1);
        assert_eq!(body["object"], "bloom.model_preflight");
        assert_eq!(body["data"]["model_id"], "tiny");
        assert_eq!(body["data"]["manifest"]["family"], "llama");
        assert_eq!(
            body["data"]["manifest"]["model_tasks"],
            json!(["generation"])
        );
        assert_eq!(body["data"]["runtime"]["selected_engine"], "candle");
        assert_eq!(body["data"]["loadable"], true);
        assert!(body["data"].get("path").is_none());
    }

    #[tokio::test]
    async fn model_switch_rejects_an_unsupported_engine_before_queueing() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("tiny.onnx"), b"onnx").unwrap();
        let (state, mut receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_switch(
            State(state),
            Json(ModelSwitchRequest {
                id: "tiny.onnx".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
        assert!(receiver.try_recv().is_err());
    }

    #[tokio::test]
    async fn model_switch_endpoint_rejects_path_traversal() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        let response = handle_model_switch(
            State(Arc::clone(&state)),
            Json(ModelSwitchRequest {
                id: "../outside.gguf".to_string(),
            }),
        )
        .await;

        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(!state.load_in_progress.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn multimodal_upload_parses_a_bounded_image_request() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route("/", post(handle_multimodal_upload))
            .with_state(state);
        let boundary = "bloom-test-boundary";
        let mut body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nWhat is shown?\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"tiny.png\"\r\nContent-Type: image/png\r\n\r\n"
        )
        .into_bytes();
        body.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn multimodal_upload_rejects_requests_without_an_image() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route("/", post(handle_multimodal_upload))
            .with_state(state);
        let boundary = "bloom-test-boundary";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nNo image\r\n--{boundary}--\r\n"
        );
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn multimodal_upload_honors_its_route_body_limit() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());
        let app = Router::new()
            .route(
                "/",
                post(handle_multimodal_upload).layer(DefaultBodyLimit::max(32)),
            )
            .with_state(state);
        let boundary = "bloom-test-boundary";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nThis exceeds the tiny route limit.\r\n--{boundary}--\r\n"
        );
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
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
    fn select_backend_routes_specialized_file_formats() {
        for (format, expected) in [
            (bloomai_core::ModelFormat::Onnx, "onnxruntime"),
            (bloomai_core::ModelFormat::OpenVinoIr, "openvino"),
            (bloomai_core::ModelFormat::CoreMl, "coreml"),
            (bloomai_core::ModelFormat::Mlx, "mlx"),
            (bloomai_core::ModelFormat::VulkanSpirv, "vulkan"),
        ] {
            let mut manifest = bloomai_core::ModelManifest::default();
            manifest.files.push(bloomai_core::ModelFile {
                name: "model".to_string(),
                format,
                size_bytes: 0,
                hash_sha256: None,
                required: true,
            });
            assert_eq!(select_backend_name("candle", "none", &manifest), expected);
        }
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
        assert!(normalize_embedding_input(&json!("   ")).is_err());
        assert!(normalize_embedding_input(&json!([])).is_err());
        assert!(normalize_embedding_input(&json!([[1, 2, 3]])).is_err());
        assert!(normalize_embedding_input(&json!(vec!["x"; MAX_EMBEDDING_INPUTS + 1])).is_err());
        assert!(
            normalize_embedding_input(&json!("x".repeat(MAX_EMBEDDING_INPUT_CHARS + 1))).is_err()
        );
        assert!(normalize_embedding_input(&json!([
            "x".repeat(MAX_EMBEDDING_INPUT_CHARS),
            "x".repeat(MAX_EMBEDDING_INPUT_CHARS),
            "x".repeat(MAX_EMBEDDING_INPUT_CHARS),
            "x"
        ]))
        .is_err());
    }

    #[test]
    fn rerank_admission_bounds_query_documents_content_and_results() {
        let valid = RerankRequest {
            query: "local inference".to_string(),
            documents: vec!["first".to_string(), "second".to_string()],
            model: None,
            top_n: Some(1),
            return_documents: Some(true),
            extensions: BTreeMap::new(),
        };
        validate_rerank_request(&valid).unwrap();

        let mut invalid = valid.clone();
        invalid.extensions.insert("future".to_string(), json!(true));
        assert!(validate_rerank_request(&invalid).is_err());
        invalid = valid.clone();
        invalid.query = " ".to_string();
        assert!(validate_rerank_request(&invalid).is_err());
        invalid = valid.clone();
        invalid.documents.clear();
        assert!(validate_rerank_request(&invalid).is_err());
        invalid = valid.clone();
        invalid.documents = vec!["x".to_string(); MAX_RERANK_DOCUMENTS + 1];
        assert!(validate_rerank_request(&invalid).is_err());
        invalid = valid.clone();
        invalid.documents[0] = "x".repeat(MAX_RERANK_DOCUMENT_CHARS + 1);
        assert!(validate_rerank_request(&invalid).is_err());
        invalid = valid.clone();
        invalid.top_n = Some(0);
        assert!(validate_rerank_request(&invalid).is_err());
        invalid = valid.clone();
        invalid.top_n = Some(3);
        assert!(validate_rerank_request(&invalid).is_err());
        invalid = valid;
        invalid.query = "x".repeat(MAX_RERANK_QUERY_CHARS);
        invalid.documents = vec![
            "x".repeat(MAX_RERANK_DOCUMENT_CHARS),
            "x".repeat(MAX_RERANK_DOCUMENT_CHARS),
            "x".repeat(MAX_RERANK_DOCUMENT_CHARS),
        ];
        invalid.top_n = None;
        assert!(validate_rerank_request(&invalid).is_err());
    }

    #[tokio::test]
    async fn embedding_and_rerank_routes_use_bounded_native_batches() {
        let temp = tempfile::tempdir().unwrap();
        let (state, native_batch_calls) =
            test_server_state_with_embedding_runtime(temp.path().to_path_buf()).await;
        let app = Router::new()
            .route("/rerank", post(handle_rerank))
            .route("/embeddings", post(handle_embeddings))
            .with_state(Arc::clone(&state))
            .layer(middleware::from_fn(publish_transient_retry_after));

        let over_context = axum::http::Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                json!({
                    "model": "test-embed-model",
                    "query": "word ".repeat(129),
                    "documents": ["short document"]
                })
                .to_string(),
            ))
            .unwrap();
        let over_context = app.clone().oneshot(over_context).await.unwrap();
        assert_eq!(over_context.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 0);

        let body = json!({
            "model": "test-embed-model",
            "query": "local AI runtime",
            "documents": [
                "local AI runtime",
                "banana orchard",
                "local AI runtime"
            ],
            "top_n": 2,
            "return_documents": true
        });
        let held_permit = Arc::clone(&state.semaphore).acquire_owned().await.unwrap();
        let capacity_response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/rerank")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            capacity_response.status(),
            axum::http::StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            capacity_response.headers()[header::RETRY_AFTER],
            DEFAULT_CAPACITY_RETRY_AFTER_SECONDS
        );
        drop(held_permit);

        let mut response_ids = Vec::new();
        for _ in 0..2 {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/rerank")
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let response: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(response["object"], "rerank");
            assert_eq!(response["model"], "test-embed-model");
            assert_eq!(response["results"].as_array().unwrap().len(), 2);
            assert_eq!(response["results"][0]["index"], 0);
            assert_eq!(response["results"][1]["index"], 2);
            assert_eq!(
                response["results"][0]["document"],
                json!({"text": "local AI runtime"})
            );
            assert!(response["results"][0]["relevance_score"].as_f64().unwrap() > 0.999_999);
            assert_eq!(response["usage"]["prompt_tokens"], 11);
            assert_eq!(response["usage"]["total_tokens"], 11);
            response_ids.push(response["id"].as_str().unwrap().to_string());
        }

        assert_ne!(response_ids[0], response_ids[1]);
        assert!(response_ids.iter().all(|id| id.starts_with("rerank-")));
        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 2);
        assert_eq!(state.metrics.requests_completed.load(Ordering::Relaxed), 2);
        assert_eq!(state.metrics.requests_failed.load(Ordering::Relaxed), 0);
        assert_eq!(state.metrics.in_flight_requests.load(Ordering::Relaxed), 0);
        assert_eq!(
            state.metrics.prompt_tokens_total.load(Ordering::Relaxed),
            22
        );
        assert_eq!(state.semaphore.available_permits(), 1);
        assert_eq!(native_batch_calls.load(Ordering::Relaxed), 2);
        assert!(state
            .cancel_tokens
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty());

        let inputs = (0..17)
            .map(|index| {
                if index % 2 == 0 {
                    "local".to_string()
                } else {
                    "banana".to_string()
                }
            })
            .collect::<Vec<_>>();
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/embeddings")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "test-embed-model",
                            "input": inputs
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let response: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let vectors = response["data"].as_array().unwrap();
        assert_eq!(vectors.len(), 17);
        for (index, vector) in vectors.iter().enumerate() {
            assert_eq!(vector["index"], index);
            assert_eq!(
                vector["embedding"],
                if index % 2 == 0 {
                    json!([1.0, 0.0, 0.0])
                } else {
                    json!([0.0, 1.0, 0.0])
                }
            );
        }
        assert_eq!(native_batch_calls.load(Ordering::Relaxed), 4);
        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 3);
        assert_eq!(state.metrics.requests_completed.load(Ordering::Relaxed), 3);
        assert_eq!(
            state.metrics.prompt_tokens_total.load(Ordering::Relaxed),
            39
        );
    }

    #[tokio::test]
    async fn stop_sequences_truncate_http_outputs_and_end_inference_early() {
        let temp = tempfile::tempdir().unwrap();
        let emitted_chunks = Arc::new(AtomicU64::new(0));
        let state = test_server_state_with_text_runtime(
            temp.path().to_path_buf(),
            Arc::clone(&emitted_chunks),
        )
        .await;
        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/completions", post(handle_completions))
            .route("/api/chat", post(handle_ollama_chat))
            .route("/api/generate", post(handle_ollama_generate))
            .with_state(Arc::clone(&state));

        let chat_body = json!({
            "model": "test-text-model",
            "messages": [{"role": "user", "content": "Continue"}],
            "max_tokens": 8,
            "stop": ["STOP"]
        });
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(chat_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "visible ");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(emitted_chunks.load(Ordering::Relaxed), 2);

        let mut streaming_body = chat_body;
        streaming_body["stream"] = json!(true);
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(streaming_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let stream = String::from_utf8(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(stream.contains("visible "));
        assert!(stream.contains("\"finish_reason\":\"stop\""));
        assert!(!stream.contains("STOP"));
        assert!(!stream.contains("hidden"));
        assert!(!stream.contains("never emitted"));
        assert_eq!(emitted_chunks.load(Ordering::Relaxed), 4);

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "test-text-model",
                            "prompt": "Continue",
                            "max_tokens": 8,
                            "stop": "STOP"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["choices"][0]["text"], "visible ");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(emitted_chunks.load(Ordering::Relaxed), 6);

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "test-text-model",
                            "messages": [{"role": "user", "content": "Continue"}],
                            "stream": false,
                            "options": {"num_predict": 8, "stop": ["STOP"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["message"]["content"], "visible ");
        assert_eq!(body["done_reason"], "stop");
        assert_eq!(emitted_chunks.load(Ordering::Relaxed), 8);

        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "test-text-model",
                            "prompt": "Continue",
                            "stream": false,
                            "options": {"num_predict": 8, "stop": ["STOP"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["response"], "visible ");
        assert_eq!(body["done_reason"], "stop");
        assert_eq!(emitted_chunks.load(Ordering::Relaxed), 10);

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/generate")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "test-text-model",
                            "prompt": "Continue",
                            "stream": true,
                            "options": {"num_predict": 8, "stop": ["STOP"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let stream = String::from_utf8(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(stream.contains("visible "));
        assert!(stream.contains("\"done_reason\":\"stop\""));
        assert!(!stream.contains("STOP"));
        assert!(!stream.contains("hidden"));
        assert_eq!(emitted_chunks.load(Ordering::Relaxed), 12);

        assert_eq!(state.metrics.requests_total.load(Ordering::Relaxed), 6);
        assert_eq!(state.metrics.requests_completed.load(Ordering::Relaxed), 6);
        assert_eq!(state.semaphore.available_permits(), 1);
        assert!(state
            .cancel_tokens
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty());
    }

    #[test]
    fn chat_prompt_applies_qwen_template_to_single_user_message() {
        let messages = vec![NormalizedChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        assert_eq!(
            chat_prompt(&messages, &bloomai_core::ModelFamily::Qwen),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chat_prompt_disables_raw_qwen3_thinking_output() {
        let messages = vec![NormalizedChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        assert_eq!(
            chat_prompt_for_architecture(
                &messages,
                &bloomai_core::ModelFamily::Qwen,
                Some("qwen3"),
            ),
            concat!(
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        );
    }

    #[test]
    fn chat_prompt_uses_classified_smollm2_template_for_llama_architecture() {
        let messages = vec![NormalizedChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        assert_eq!(
            chat_prompt_for_metadata(
                &messages,
                &bloomai_core::ModelFamily::Llama,
                Some("llama"),
                Some("smollm2"),
            ),
            concat!(
                "<|im_start|>system\n",
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
                "<|im_end|>\n",
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n"
            )
        );
    }

    #[test]
    fn response_format_accepts_json_object_and_schema() {
        let json_object = ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
            extensions: BTreeMap::new(),
        };
        assert_eq!(
            response_format_mode(Some(&json_object)).unwrap(),
            ResponseFormatMode::JsonObject
        );

        let json_schema = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(json!({ "name": "x", "schema": { "type": "object" } })),
            extensions: BTreeMap::new(),
        };
        assert!(matches!(
            response_format_mode(Some(&json_schema)).unwrap(),
            ResponseFormatMode::JsonSchema(_)
        ));

        let unsupported_keyword = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(json!({
                "name": "x",
                "strict": true,
                "schema": {"type": "object", "minProperties": 1}
            })),
            extensions: BTreeMap::new(),
        };
        assert!(response_format_mode(Some(&unsupported_keyword))
            .unwrap_err()
            .contains("unsupported keyword"));

        let unsupported_type = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(json!({
                "name": "x",
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "date"}}
                }
            })),
            extensions: BTreeMap::new(),
        };
        assert!(response_format_mode(Some(&unsupported_type))
            .unwrap_err()
            .contains("unsupported type"));

        let mut extended = json_object;
        extended
            .extensions
            .insert("future".to_string(), json!(true));
        assert!(response_format_mode(Some(&extended))
            .unwrap_err()
            .contains("response_format"));
    }

    #[test]
    fn json_schema_admission_is_bounded_and_fail_closed() {
        let mut nested = json!({"type": "string"});
        for _ in 0..=helpers::MAX_JSON_SCHEMA_DEPTH {
            nested = json!({"type": "array", "items": nested});
        }
        let deep = json!({
            "type": "object",
            "properties": {"nested": nested}
        });
        assert!(helpers::validate_supported_json_schema(&deep)
            .unwrap_err()
            .contains("maximum depth"));

        let oversized = json!({
            "type": "object",
            "description": "x".repeat(helpers::MAX_JSON_SCHEMA_BYTES)
        });
        assert!(helpers::validate_supported_json_schema(&oversized)
            .unwrap_err()
            .contains("byte limit"));

        let unknown_required = json!({
            "type": "object",
            "properties": {"known": {"type": "string"}},
            "required": ["missing"]
        });
        assert!(helpers::validate_supported_json_schema(&unknown_required)
            .unwrap_err()
            .contains("unknown property"));
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
        assert!(validate_request_id(&first).is_ok());
        assert!(validate_request_id(&second).is_ok());
    }

    #[test]
    fn browser_origin_policy_accepts_only_explicit_web_origins() {
        assert_eq!(
            parse_browser_origin_policy("same-origin").unwrap(),
            BrowserOriginPolicy::SameOrigin
        );
        assert_eq!(
            parse_browser_origin_policy("*").unwrap(),
            BrowserOriginPolicy::Any
        );

        let BrowserOriginPolicy::Exact(origin) =
            parse_browser_origin_policy(" HTTPS://UI.Example:8443/ ").unwrap()
        else {
            panic!("expected one exact browser origin");
        };
        assert_eq!(origin.serialized, "https://ui.example:8443");
        assert_eq!(origin.header, "https://ui.example:8443");
        assert!(!origin.has_loopback_host());

        let BrowserOriginPolicy::Exact(ipv6) =
            parse_browser_origin_policy("http://[::1]:3000").unwrap()
        else {
            panic!("expected one exact IPv6 browser origin");
        };
        assert!(ipv6.has_loopback_host());

        for invalid in [
            "",
            "null",
            "file:///tmp/ui.html",
            "https://user@example.com",
            "https://example.com/path",
            "https://example.com?query=1",
            "https://",
        ] {
            assert!(
                parse_browser_origin_policy(invalid).is_err(),
                "unexpectedly accepted {invalid:?}"
            );
        }
    }

    #[tokio::test]
    async fn browser_origin_guard_is_fail_closed_and_cors_aligned() {
        let policy = parse_browser_origin_policy("https://ui.example").unwrap();
        let guard = BrowserOriginGuard {
            policy: policy.clone(),
            loopback_listener: true,
        };
        let app = Router::new()
            .route(
                "/health",
                get(|| async { axum::http::StatusCode::NO_CONTENT }),
            )
            .route(
                "/v1/models",
                get(|| async { axum::http::StatusCode::NO_CONTENT }),
            )
            .route(
                "/api/version",
                get(|| async { axum::http::StatusCode::NO_CONTENT }),
            )
            .layer(configured_cors_layer(&policy))
            .layer(middleware::from_fn_with_state(
                guard,
                enforce_browser_origin,
            ))
            .layer(middleware::from_fn(normalize_protocol_error_response))
            .layer(middleware::from_fn(prevent_dynamic_response_caching))
            .layer(middleware::from_fn(correlate_http_request));

        let allowed = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .header(header::ORIGIN, "https://ui.example")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(allowed.status(), axum::http::StatusCode::NO_CONTENT);
        assert_eq!(
            allowed.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            "https://ui.example"
        );

        let same_origin = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .header(header::ORIGIN, "http://127.0.0.1:3000")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(same_origin.status(), axum::http::StatusCode::NO_CONTENT);
        assert_eq!(
            same_origin.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            "https://ui.example"
        );

        let without_origin = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(without_origin.status(), axum::http::StatusCode::NO_CONTENT);

        for (path, family) in [("/v1/models", "openai"), ("/api/version", "ollama")] {
            let rejected = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(path)
                        .header(header::HOST, "127.0.0.1:3000")
                        .header(header::ORIGIN, "https://malicious.example")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(rejected.status(), axum::http::StatusCode::FORBIDDEN);
            assert_eq!(rejected.headers()[header::CACHE_CONTROL], "no-store");
            assert!(rejected.headers().contains_key(HTTP_REQUEST_ID_HEADER));
            assert!(!rejected
                .headers()
                .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
            let body: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(rejected.into_body(), usize::MAX)
                    .await
                    .unwrap(),
            )
            .unwrap();
            if family == "openai" {
                assert_eq!(body["error"]["type"], "invalid_request_error");
            } else {
                assert!(body["error"].is_string());
            }
        }

        let preflight = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("OPTIONS")
                    .uri("/v1/models")
                    .header(header::HOST, "127.0.0.1:3000")
                    .header(header::ORIGIN, "https://malicious.example")
                    .header(header::ACCESS_CONTROL_REQUEST_METHOD, "GET")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(preflight.status(), axum::http::StatusCode::FORBIDDEN);

        for origin in ["null", "http://attacker.example"] {
            let rebound = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri("/health")
                        .header(header::HOST, "attacker.example")
                        .header(header::ORIGIN, origin)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(rebound.status(), axum::http::StatusCode::FORBIDDEN);
        }

        let duplicate = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .header(header::ORIGIN, "https://ui.example")
                    .header(header::ORIGIN, "https://malicious.example")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(duplicate.status(), axum::http::StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn http_request_ids_are_generated_bounded_and_authoritative() {
        let app = Router::new()
            .route("/ok", get(|| async { axum::http::StatusCode::NO_CONTENT }))
            .layer(configured_cors_layer(&BrowserOriginPolicy::Any))
            .layer(middleware::from_fn(correlate_http_request));

        let generated = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/ok")
                    .header(header::ORIGIN, "https://ui.example")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let generated_id = generated
            .headers()
            .get(HTTP_REQUEST_ID_HEADER)
            .unwrap()
            .clone();
        assert!(valid_http_request_id(&generated_id));
        assert_eq!(generated_id.as_bytes().len(), 36);
        let exposed_headers = generated
            .headers()
            .get(header::ACCESS_CONTROL_EXPOSE_HEADERS)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(exposed_headers
            .split(',')
            .any(|value| value.trim() == HTTP_REQUEST_ID_HEADER));
        assert!(exposed_headers
            .split(',')
            .any(|value| value.trim() == header::RETRY_AFTER.as_str()));
        assert!(exposed_headers
            .split(',')
            .any(|value| value.trim() == header::WWW_AUTHENTICATE.as_str()));

        let second = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/ok")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(
            second.headers().get(HTTP_REQUEST_ID_HEADER).unwrap(),
            &generated_id
        );

        let accepted = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/ok")
                    .header(HTTP_REQUEST_ID_HEADER, "proxy.node_1:request-42")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            accepted.headers().get(HTTP_REQUEST_ID_HEADER).unwrap(),
            "proxy.node_1:request-42"
        );

        for invalid in [
            HeaderValue::from_static("contains spaces"),
            HeaderValue::from_str(&"r".repeat(MAX_HTTP_REQUEST_ID_CHARS + 1)).unwrap(),
        ] {
            let rejected = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri("/missing")
                        .header(HTTP_REQUEST_ID_HEADER, invalid.clone())
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(rejected.status(), axum::http::StatusCode::NOT_FOUND);
            let replacement = rejected.headers().get(HTTP_REQUEST_ID_HEADER).unwrap();
            assert!(valid_http_request_id(replacement));
            assert_ne!(replacement, &invalid);
            assert_eq!(replacement.as_bytes().len(), 36);
        }
    }

    #[tokio::test]
    async fn transient_capacity_errors_publish_a_retry_hint_without_rewriting_one() {
        let app = Router::new()
            .route(
                "/capacity",
                get(|| async { axum::http::StatusCode::TOO_MANY_REQUESTS }),
            )
            .route(
                "/custom",
                get(|| async {
                    (
                        axum::http::StatusCode::TOO_MANY_REQUESTS,
                        [(header::RETRY_AFTER, "7")],
                    )
                }),
            )
            .route(
                "/unavailable",
                get(|| async { axum::http::StatusCode::SERVICE_UNAVAILABLE }),
            )
            .layer(configured_cors_layer(&BrowserOriginPolicy::Any))
            .layer(middleware::from_fn(publish_transient_retry_after));

        let capacity = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/capacity")
                    .header(header::ORIGIN, "https://ui.example")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            capacity.headers()[header::RETRY_AFTER],
            DEFAULT_CAPACITY_RETRY_AFTER_SECONDS
        );
        assert!(capacity.headers()[header::ACCESS_CONTROL_EXPOSE_HEADERS]
            .to_str()
            .unwrap()
            .split(',')
            .any(|value| value.trim() == header::RETRY_AFTER.as_str()));

        let custom = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/custom")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(custom.headers()[header::RETRY_AFTER], "7");

        let unavailable = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/unavailable")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!unavailable.headers().contains_key(header::RETRY_AFTER));
    }

    #[tokio::test]
    async fn authentication_failures_publish_a_bearer_challenge_without_rewriting_one() {
        let app = Router::new()
            .route(
                "/authentication",
                get(|| async { axum::http::StatusCode::UNAUTHORIZED }),
            )
            .route(
                "/custom",
                get(|| async {
                    (
                        axum::http::StatusCode::UNAUTHORIZED,
                        [(header::WWW_AUTHENTICATE, "Bearer realm=\"Custom\"")],
                    )
                }),
            )
            .route(
                "/forbidden",
                get(|| async { axum::http::StatusCode::FORBIDDEN }),
            )
            .layer(configured_cors_layer(&BrowserOriginPolicy::Any))
            .layer(middleware::from_fn(publish_authentication_challenge));

        let authentication = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/authentication")
                    .header(header::ORIGIN, "https://ui.example")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            authentication.headers()[header::WWW_AUTHENTICATE],
            DEFAULT_BEARER_AUTHENTICATION_CHALLENGE
        );
        assert!(
            authentication.headers()[header::ACCESS_CONTROL_EXPOSE_HEADERS]
                .to_str()
                .unwrap()
                .split(',')
                .any(|value| value.trim() == header::WWW_AUTHENTICATE.as_str())
        );

        let custom = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/custom")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            custom.headers()[header::WWW_AUTHENTICATE],
            "Bearer realm=\"Custom\""
        );

        let forbidden = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/forbidden")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!forbidden.headers().contains_key(header::WWW_AUTHENTICATE));
    }

    #[tokio::test]
    async fn dynamic_http_responses_are_never_cacheable() {
        let app = Router::new()
            .route(
                "/v1/value",
                get(|| async { ([(header::CACHE_CONTROL, "public, max-age=3600")], "dynamic") }),
            )
            .route(
                "/assets/app.js",
                get(|| async { ([(header::CACHE_CONTROL, "public, max-age=3600")], "static") }),
            )
            .layer(middleware::from_fn(prevent_dynamic_response_caching));

        for path in [
            "/health",
            "/ready",
            "/metrics",
            "/v1",
            "/v1/value",
            "/v1/not-found",
            "/api",
            "/api/not-found",
        ] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(path)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.headers().get(header::CACHE_CONTROL).unwrap(),
                "no-store",
                "path: {path}"
            );
        }

        let static_asset = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/assets/app.js")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            static_asset.headers().get(header::CACHE_CONTROL).unwrap(),
            "public, max-age=3600"
        );
        assert!(!requires_no_store("/v1-compatible-browser-route"));
        assert!(!requires_no_store("/apiary"));
    }

    #[tokio::test]
    async fn cancellation_rejects_unsafe_or_oversized_request_ids() {
        let temp = tempfile::tempdir().unwrap();
        let (state, _receiver) = test_server_state(temp.path().to_path_buf());

        for invalid in [
            "",
            "../model-management/unload",
            "chatcmpl?admin=true",
            "chatcmpl%2Funload",
        ] {
            let response = handle_cancel(
                State(Arc::clone(&state)),
                axum::extract::Path(invalid.to_string()),
            )
            .await;
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
        }

        let response = handle_cancel(
            State(state),
            axum::extract::Path("r".repeat(MAX_REQUEST_ID_CHARS + 1)),
        )
        .await;
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn cancel_token_guard_removes_its_registration() {
        let tokens = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let guard =
            CancelTokenGuard::register_with_tokens(Arc::clone(&tokens), "req-1".to_string());
        let token = guard.token();
        assert!(!token.is_cancelled());
        assert!(tokens.lock().unwrap().contains_key("req-1"));
        drop(guard);
        assert!(!tokens.lock().unwrap().contains_key("req-1"));
        assert!(!token.is_cancelled());
    }

    #[test]
    fn inference_lifecycle_finishes_once_and_holds_admission_until_completion() {
        let tokens = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let registration =
            CancelTokenGuard::register_with_tokens(Arc::clone(&tokens), "req-1".to_string());
        let token = registration.token();
        let metrics = Arc::new(ServerMetrics::new());
        metrics.record_request_start();
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&semaphore).try_acquire_owned().unwrap();
        let generated = Arc::new(AtomicU64::new(4));
        let lifecycle = InferenceLifecycle::new(
            registration,
            InferenceLifecycleResources {
                metrics: Arc::clone(&metrics),
                request_start: Instant::now(),
                generated_tokens: generated,
                prompt_tokens: 7,
                permit,
            },
            StreamExecution::Blocking,
        );
        let worker = lifecycle.worker_guard();
        let mut client = lifecycle.client_guard();

        assert_eq!(semaphore.available_permits(), 0);
        drop(worker);
        assert_eq!(metrics.in_flight_requests.load(Ordering::Relaxed), 1);
        assert_eq!(semaphore.available_permits(), 0);

        client.finish(true);
        client.finish(false);
        drop(client);
        assert!(!token.is_cancelled());
        assert!(!tokens.lock().unwrap().contains_key("req-1"));
        assert_eq!(metrics.in_flight_requests.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.requests_failed.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.tokens_generated_total.load(Ordering::Relaxed), 4);
        assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), 7);
        assert_eq!(semaphore.available_permits(), 1);
    }

    #[test]
    fn dropped_client_cancels_but_drains_worker_before_releasing_admission() {
        let tokens = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let registration =
            CancelTokenGuard::register_with_tokens(Arc::clone(&tokens), "req-2".to_string());
        let token = registration.token();
        let metrics = Arc::new(ServerMetrics::new());
        metrics.record_request_start();
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&semaphore).try_acquire_owned().unwrap();
        let generated = Arc::new(AtomicU64::new(3));
        let lifecycle = InferenceLifecycle::new(
            registration,
            InferenceLifecycleResources {
                metrics: Arc::clone(&metrics),
                request_start: Instant::now(),
                generated_tokens: generated,
                prompt_tokens: 5,
                permit,
            },
            StreamExecution::Blocking,
        );
        let worker = lifecycle.worker_guard();
        let client = lifecycle.client_guard();

        drop(client);
        assert!(token.is_cancelled());
        assert!(tokens.lock().unwrap().contains_key("req-2"));
        assert_eq!(metrics.in_flight_requests.load(Ordering::Relaxed), 1);
        assert_eq!(semaphore.available_permits(), 0);

        drop(worker);
        assert!(!tokens.lock().unwrap().contains_key("req-2"));
        assert_eq!(metrics.in_flight_requests.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_failed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.tokens_generated_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), 5);
        assert_eq!(semaphore.available_permits(), 1);
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
    fn multimodal_input_requires_declared_vision_support() {
        let blocks = vec![DataBlock::Image {
            bytes: vec![1, 2, 3],
            mime: "image/png".to_string(),
        }];

        assert!(validate_multimodal_modalities(&[bloomai_core::Modality::Text], &blocks).is_err());
        assert!(validate_multimodal_modalities(
            &[bloomai_core::Modality::Text, bloomai_core::Modality::Vision],
            &blocks
        )
        .is_ok());

        let audio = vec![DataBlock::AudioPcm {
            samples: vec![0.0],
            sample_rate: 16_000,
        }];
        assert!(validate_multimodal_modalities(&[bloomai_core::Modality::Text], &audio).is_err());
        assert!(validate_multimodal_modalities(&[bloomai_core::Modality::Multi], &audio).is_ok());
    }

    #[test]
    fn public_multimodal_admission_accepts_only_bounded_inline_data() {
        let valid = test_multimodal_request(vec![
            DataBlock::Text("Describe this image.".to_string()),
            DataBlock::Image {
                bytes: b"\x89PNG\r\n\x1a\n".to_vec(),
                mime: "image/png".to_string(),
            },
        ]);
        validate_multimodal_request(&valid).unwrap();

        let empty = test_multimodal_request(Vec::new());
        assert!(validate_multimodal_request(&empty).is_err());

        let duplicate = test_multimodal_request(vec![
            DataBlock::Text("one".to_string()),
            DataBlock::Text("two".to_string()),
        ]);
        assert!(validate_multimodal_request(&duplicate).is_err());

        let local_path = test_multimodal_request(vec![DataBlock::AudioFile {
            path: "/etc/passwd".to_string(),
            language: None,
        }]);
        let error = validate_multimodal_request(&local_path).unwrap_err();
        assert!(error.contains("server-local audio paths"));

        for block in [
            DataBlock::Tokens(vec![1]),
            DataBlock::Tensor(vec![0.0]),
            DataBlock::VideoFrames(vec![vec![0]]),
            DataBlock::WorldState {
                state_id: "state".to_string(),
                latent: None,
                step: 0,
            },
            DataBlock::Action {
                action_space: "test".to_string(),
                values: vec![0.0],
            },
        ] {
            assert!(validate_multimodal_request(&test_multimodal_request(vec![block])).is_err());
        }
    }

    #[test]
    fn public_multimodal_admission_bounds_controls_audio_and_images() {
        let mut request = test_multimodal_request(vec![DataBlock::AudioPcm {
            samples: vec![0.0, 0.5, -0.5],
            sample_rate: 16_000,
        }]);
        validate_multimodal_request(&request).unwrap();

        request.params.max_tokens = MAX_GENERATED_TOKENS + 1;
        assert!(validate_multimodal_request(&request).is_err());
        request.params.max_tokens = 128;
        request.params.response_format = Some(bloomai_core::ResponseFormat::JsonObject);
        assert!(validate_multimodal_request(&request).is_err());

        request = test_multimodal_request(vec![DataBlock::AudioPcm {
            samples: vec![1.1],
            sample_rate: 16_000,
        }]);
        assert!(validate_multimodal_request(&request).is_err());
        request = test_multimodal_request(vec![DataBlock::AudioPcm {
            samples: vec![0.0],
            sample_rate: MIN_MULTIMODAL_AUDIO_SAMPLE_RATE - 1,
        }]);
        assert!(validate_multimodal_request(&request).is_err());
        request = test_multimodal_request(vec![DataBlock::AudioPcm {
            samples: vec![
                0.0;
                MIN_MULTIMODAL_AUDIO_SAMPLE_RATE as usize * MAX_MULTIMODAL_AUDIO_SECONDS
                    + 1
            ],
            sample_rate: MIN_MULTIMODAL_AUDIO_SAMPLE_RATE,
        }]);
        assert!(validate_multimodal_request(&request).is_err());

        request = test_multimodal_request(vec![DataBlock::Image {
            bytes: vec![0; MAX_MULTIMODAL_IMAGE_BYTES + 1],
            mime: "image/png".to_string(),
        }]);
        assert!(validate_multimodal_request(&request).is_err());
    }

    #[test]
    fn image_upload_signature_must_match_declared_mime() {
        assert!(validate_uploaded_image(b"\x89PNG\r\n\x1a\n", "image/png").is_ok());
        assert!(validate_uploaded_image(b"\x89PNG\r\n\x1a\n", "image/jpeg").is_err());
        assert!(validate_uploaded_image(b"not an image", "image/png").is_err());
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
