//! Bounded adapters for the commonly used local Ollama HTTP API surface.

use super::model_download::{ModelDownloadPhase, ModelDownloadStatus};
use super::model_index::{
    ModelIndexEntry, ModelIndexInstallationState as InstalledPullState,
    model_index_installation_state, validate_index_id,
};
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::collections::VecDeque;
use std::convert::Infallible;

const MAX_OLLAMA_STREAM_BYTES: usize = 16 * MIB as usize;
const MAX_OLLAMA_STREAM_EVENTS: usize = 131_072;
const MAX_OLLAMA_IMAGE_BASE64_BYTES: usize = MAX_MULTIMODAL_IMAGE_BYTES.div_ceil(3) * 4;
const OLLAMA_CONTENT_TYPE: &str = "application/x-ndjson";
const OLLAMA_PULL_POLL_INTERVAL: Duration = Duration::from_millis(100);
const DEFAULT_OLLAMA_KEEP_ALIVE: Duration = Duration::from_secs(5 * 60);
const MAX_OLLAMA_KEEP_ALIVE: Duration = Duration::from_secs(365 * 24 * 60 * 60);
const OLLAMA_RESIDENCY_RETRY_INTERVAL: Duration = Duration::from_millis(25);
const INDEFINITE_OLLAMA_EXPIRY: &str = "9999-12-31T23:59:59Z";

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<OllamaMessage>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    format: Option<serde_json::Value>,
    #[serde(default)]
    options: Option<serde_json::Value>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    think: Option<serde_json::Value>,
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    images: Option<serde_json::Value>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug)]
struct OllamaImageInput {
    bytes: Vec<u8>,
    mime: String,
}

struct OllamaPreparedImageRequest {
    prompt: Option<String>,
    image: OllamaImageInput,
    params: InferenceParams,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaGenerateRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    images: Option<serde_json::Value>,
    #[serde(default)]
    format: Option<serde_json::Value>,
    #[serde(default)]
    options: Option<serde_json::Value>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    think: Option<serde_json::Value>,
    #[serde(default)]
    raw: Option<bool>,
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaShowRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    verbose: bool,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaDeleteRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaPullRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    insecure: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaEmbedRequest {
    #[serde(default)]
    model: Option<String>,
    input: serde_json::Value,
    #[serde(default)]
    truncate: Option<bool>,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    options: Option<serde_json::Value>,
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OllamaLegacyEmbeddingRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    options: Option<serde_json::Value>,
    #[serde(default)]
    keep_alive: Option<serde_json::Value>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum OllamaOutputKind {
    Chat,
    Generate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OllamaLifecycleAction {
    Load(OllamaKeepAlive),
    Unload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OllamaKeepAlive {
    Indefinite,
    Unload,
    Timed(Duration),
}

struct OllamaResidencyLease {
    state: Arc<ServerState>,
    revision: u64,
    runtime: Weak<LoadedRuntime>,
    keep_alive: OllamaKeepAlive,
    timer_cancel: CancellationToken,
}

impl Drop for OllamaResidencyLease {
    fn drop(&mut self) {
        let duration = match self.keep_alive {
            OllamaKeepAlive::Timed(duration) => duration,
            OllamaKeepAlive::Unload => Duration::ZERO,
            OllamaKeepAlive::Indefinite => return,
        };
        let state = Arc::clone(&self.state);
        let revision = self.revision;
        let runtime = self.runtime.clone();
        let timer_cancel = self.timer_cancel.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                arm_ollama_residency_expiry(state, revision, runtime, duration, timer_cancel).await;
            });
        }
    }
}

impl OllamaResidencyState {
    fn runtime_index(&self, runtime: &Arc<LoadedRuntime>) -> Option<usize> {
        let runtime = Arc::downgrade(runtime);
        self.weak_runtime_index(&runtime)
    }

    fn weak_runtime_index(&self, runtime: &Weak<LoadedRuntime>) -> Option<usize> {
        self.runtimes
            .iter()
            .position(|entry| Weak::ptr_eq(&entry.runtime, runtime))
    }

    fn prune_released(&mut self) {
        self.runtimes
            .retain(|entry| entry.runtime.strong_count() > 0);
    }

    fn commit(&mut self, runtime: &Arc<LoadedRuntime>) -> (u64, CancellationToken) {
        self.prune_released();
        if let Some(index) = self.runtime_index(runtime) {
            let entry = &mut self.runtimes[index];
            entry.timer_cancel.cancel();
            entry.timer_cancel = CancellationToken::new();
            entry.revision = entry.revision.wrapping_add(1).max(1);
            entry.expiry = None;
            return (entry.revision, entry.timer_cancel.clone());
        }

        let timer_cancel = CancellationToken::new();
        self.runtimes.push(OllamaRuntimeResidency {
            runtime: Arc::downgrade(runtime),
            revision: 1,
            expiry: None,
            timer_cancel: timer_cancel.clone(),
        });
        (1, timer_cancel)
    }

    fn set_expiry(
        &mut self,
        runtime: &Weak<LoadedRuntime>,
        revision: u64,
        expires_at: SystemTime,
    ) -> bool {
        let Some(index) = self.weak_runtime_index(runtime) else {
            return false;
        };
        let entry = &mut self.runtimes[index];
        if entry.revision != revision {
            return false;
        }
        entry.expiry = Some(expires_at);
        true
    }

    fn matches(&self, runtime: &Weak<LoadedRuntime>, revision: u64) -> bool {
        self.weak_runtime_index(runtime)
            .is_some_and(|index| self.runtimes[index].revision == revision)
    }

    fn clear_if_current(&mut self, runtime: &Weak<LoadedRuntime>, revision: u64) {
        let Some(index) = self.weak_runtime_index(runtime) else {
            return;
        };
        if self.runtimes[index].revision != revision {
            return;
        }
        let entry = self.runtimes.remove(index);
        entry.timer_cancel.cancel();
    }

    fn cancel_runtime(&mut self, runtime: &Arc<LoadedRuntime>) {
        let Some(index) = self.runtime_index(runtime) else {
            return;
        };
        let entry = self.runtimes.remove(index);
        entry.timer_cancel.cancel();
    }

    pub(super) fn expiry_for(&self, runtime: &Arc<LoadedRuntime>) -> Option<SystemTime> {
        self.runtime_index(runtime)
            .and_then(|index| self.runtimes[index].expiry)
    }
}

fn commit_ollama_residency_request(
    residency: &mut OllamaResidencyState,
    runtime: &Arc<LoadedRuntime>,
) -> (u64, CancellationToken) {
    residency.commit(runtime)
}

async fn ollama_residency_lease(
    state: Arc<ServerState>,
    runtime: &Arc<LoadedRuntime>,
    revision: u64,
    timer_cancel: CancellationToken,
    keep_alive: OllamaKeepAlive,
) -> std::result::Result<OllamaResidencyLease, String> {
    Ok(OllamaResidencyLease {
        state,
        revision,
        runtime: Arc::downgrade(runtime),
        keep_alive,
        timer_cancel,
    })
}

async fn arm_ollama_residency_expiry(
    state: Arc<ServerState>,
    revision: u64,
    runtime: Weak<LoadedRuntime>,
    duration: Duration,
    timer_cancel: CancellationToken,
) {
    let expires_at = SystemTime::now()
        .checked_add(duration)
        .unwrap_or(SystemTime::UNIX_EPOCH + MAX_OLLAMA_KEEP_ALIVE);
    {
        let mut residency = state.ollama_residency.lock().await;
        if !residency.set_expiry(&runtime, revision, expires_at) {
            return;
        }
    }
    tokio::select! {
        _ = timer_cancel.cancelled() => return,
        _ = tokio::time::sleep(duration) => {}
    }

    loop {
        let mut residency = state.ollama_residency.lock().await;
        if !residency.matches(&runtime, revision) {
            return;
        }
        let Some(expected_runtime) = runtime.upgrade() else {
            residency.clear_if_current(&runtime, revision);
            return;
        };
        if !state
            .runtime_pool
            .read()
            .await
            .contains_exact(&expected_runtime)
        {
            residency.clear_if_current(&runtime, revision);
            return;
        }

        let response =
            handle_model_unload_exact_if_idle(Arc::clone(&state), expected_runtime).await;
        if response.status().is_success() {
            residency.clear_if_current(&runtime, revision);
            tracing::info!("Ollama keep_alive deadline unloaded its resident model");
            return;
        }
        if response.status() == axum::http::StatusCode::NOT_FOUND {
            residency.clear_if_current(&runtime, revision);
            return;
        }
        if response.status() != axum::http::StatusCode::CONFLICT {
            residency.clear_if_current(&runtime, revision);
            tracing::warn!(
                status = %response.status(),
                "Ollama keep_alive deadline could not unload its resident model"
            );
            return;
        }
        drop(residency);
        tokio::select! {
            _ = timer_cancel.cancelled() => return,
            _ = tokio::time::sleep(OLLAMA_RESIDENCY_RETRY_INTERVAL) => {}
        }
    }
}

async fn ollama_runtime_expiry(state: &ServerState, runtime: &Arc<LoadedRuntime>) -> String {
    let residency = state.ollama_residency.lock().await;
    residency
        .expiry_for(runtime)
        .and_then(|expires_at| expires_at.duration_since(UNIX_EPOCH).ok())
        .map(|elapsed| {
            let rounded_seconds = elapsed
                .as_secs()
                .saturating_add(u64::from(elapsed.subsec_nanos() > 0));
            iso8601_from_unix(rounded_seconds)
        })
        .unwrap_or_else(|| INDEFINITE_OLLAMA_EXPIRY.to_string())
}

#[derive(Debug, Default)]
struct OllamaGenerationOptions {
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
    stop: Vec<String>,
}

pub(crate) async fn handle_ollama_version() -> axum::response::Response {
    ollama_json(json!({"version": env!("CARGO_PKG_VERSION")}))
}

pub(crate) async fn handle_ollama_tags(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    let (catalog, _) = match state.model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(_) => {
            return ollama_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "failed to inspect the local model catalog",
            );
        }
    };
    let resident = state.runtime_pool.read().await.snapshot();
    let mut models = catalog
        .models
        .iter()
        .map(|entry| {
            ollama_catalog_model(
                entry,
                resident
                    .entries()
                    .iter()
                    .map(|(_, runtime)| runtime)
                    .find(|runtime| catalog_entry_matches_runtime(&catalog, entry, runtime)),
            )
        })
        .collect::<Vec<_>>();
    for (_, runtime) in resident.entries() {
        if catalog
            .models
            .iter()
            .all(|entry| !catalog_entry_matches_runtime(&catalog, entry, runtime))
        {
            models.push(ollama_runtime_model(runtime));
        }
    }
    ollama_json(json!({"models": models}))
}

pub(crate) async fn handle_ollama_ps(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    let (catalog, _) = match state.model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(_) => {
            return ollama_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "failed to inspect the active local model",
            );
        }
    };
    let resident = state.runtime_pool.read().await.snapshot();
    let default = resident.default_runtime();
    let mut runtimes = Vec::with_capacity(resident.entries().len());
    if let Some(runtime) = default.as_ref() {
        runtimes.push(Arc::clone(runtime));
    }
    runtimes.extend(
        resident
            .entries()
            .iter()
            .map(|(_, runtime)| runtime)
            .filter(|runtime| {
                default
                    .as_ref()
                    .is_none_or(|default| !Arc::ptr_eq(runtime, default))
            })
            .cloned(),
    );
    let mut models = Vec::with_capacity(runtimes.len());
    for runtime in runtimes {
        let expires_at = ollama_runtime_expiry(&state, &runtime).await;
        let entry = catalog
            .models
            .iter()
            .find(|entry| catalog_entry_matches_runtime(&catalog, entry, &runtime));
        let selector = entry
            .map(ollama_catalog_selector)
            .unwrap_or(runtime.model_id.as_str());
        models.push(json!({
            "name": selector,
            "model": selector,
            "size": runtime.memory_estimate.weight_bytes,
            "digest": runtime_digest(entry),
            "details": ollama_details(
                runtime_format(&runtime),
                model_family_name(&runtime.model_family),
                runtime_quantization(&runtime),
            ),
            "expires_at": expires_at,
            "size_vram": runtime.memory_estimate.device_weight_bytes,
            "context_length": runtime.pipeline.context_size()
        }));
    }
    ollama_json(json!({"models": models}))
}

pub(crate) async fn handle_ollama_show(
    State(state): State<Arc<ServerState>>,
    payload: std::result::Result<Json<OllamaShowRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return ollama_json_rejection_response(error),
    };
    if let Err(message) = reject_ollama_extensions("show request", &payload.extensions) {
        return ollama_bad_request(message);
    }
    if payload.verbose {
        return ollama_bad_request(
            "verbose model metadata is not supported; Bloom returns bounded metadata only",
        );
    }
    let requested = match (payload.model.as_deref(), payload.name.as_deref()) {
        (Some(model), Some(name)) if model != name => {
            return ollama_bad_request("model and legacy name selectors must match");
        }
        (Some(model), _) | (_, Some(model)) if !model.is_empty() => model,
        _ => return ollama_bad_request("model is required"),
    };
    if validate_ollama_model_selector(requested).is_err() {
        return ollama_bad_request("model selector is invalid");
    }
    let (catalog, _) = match state.model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(_) => {
            return ollama_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "failed to inspect the local model catalog",
            );
        }
    };
    let resident = state.runtime_pool.read().await.snapshot();
    let (entry, matching_runtime) = if requested == "default" {
        let Some(runtime) = resident.default_runtime() else {
            return ollama_error_response(
                axum::http::StatusCode::NOT_FOUND,
                "model 'default' not found",
            );
        };
        let entry = catalog
            .models
            .iter()
            .find(|entry| catalog_entry_matches_runtime(&catalog, entry, &runtime));
        (entry, Some(runtime))
    } else {
        let matching_entries = catalog
            .models
            .iter()
            .filter(|entry| ollama_catalog_entry_matches_selector(entry, requested))
            .collect::<Vec<_>>();
        if matching_entries.len() > 1 {
            return ollama_error_response(
                axum::http::StatusCode::CONFLICT,
                format!("model selector {requested:?} resolves to multiple local catalog entries"),
            );
        }
        let entry = matching_entries.first().copied();
        let matching_runtime = match resolve_resident_runtime(&catalog, &resident, requested, entry)
        {
            Ok(runtime) => runtime,
            Err(()) => {
                return ollama_error_response(
                    axum::http::StatusCode::CONFLICT,
                    format!("model selector {requested:?} resolves to multiple loaded runtimes"),
                );
            }
        };
        (entry, matching_runtime)
    };
    if entry.is_none() && matching_runtime.is_none() {
        return ollama_error_response(
            axum::http::StatusCode::NOT_FOUND,
            format!("model {requested:?} not found"),
        );
    }

    let format = entry
        .map(|entry| entry.format.as_str())
        .or_else(|| {
            matching_runtime
                .as_ref()
                .map(|runtime| runtime_format(runtime))
        })
        .unwrap_or("unknown");
    let family = matching_runtime
        .as_ref()
        .map(|runtime| model_family_name(&runtime.model_family))
        .unwrap_or("unknown");
    let quantization = matching_runtime
        .as_ref()
        .and_then(|runtime| runtime_quantization(runtime))
        .or_else(|| entry.and_then(|entry| quantization_from_name(&entry.id)));
    let modified_at = entry
        .and_then(|entry| entry.modified_at)
        .or_else(|| {
            matching_runtime
                .as_ref()
                .and_then(|runtime| runtime_modified_at(runtime))
        })
        .unwrap_or_default();
    let license = entry
        .and_then(|entry| entry.provenance.as_ref())
        .and_then(|provenance| provenance.license.clone())
        .unwrap_or_default();
    let mut model_info = serde_json::Map::new();
    model_info.insert("general.architecture".to_string(), json!(family));
    if let Some(runtime) = matching_runtime.as_ref() {
        model_info.insert(
            format!("{family}.context_length"),
            json!(runtime.pipeline.context_size()),
        );
    }
    let capabilities = if matching_runtime
        .as_ref()
        .is_some_and(|runtime| model_supports_embeddings(&runtime.pipeline))
    {
        vec!["embedding"]
    } else {
        vec!["completion", "tools"]
    };
    ollama_json(json!({
        "parameters": "temperature 0.7\ntop_p 0.9\nnum_predict 128",
        "license": license,
        "modified_at": iso8601_from_unix(modified_at),
        "details": ollama_details(format, family, quantization),
        "template": "",
        "capabilities": capabilities,
        "model_info": model_info
    }))
}

pub(crate) async fn handle_ollama_delete(
    State(state): State<Arc<ServerState>>,
    payload: std::result::Result<
        Json<OllamaDeleteRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> axum::response::Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return ollama_json_rejection_response(error),
    };
    if let Err(message) = reject_ollama_extensions("delete request", &payload.extensions) {
        return ollama_bad_request(message);
    }
    let model = match required_ollama_model(payload.model) {
        Ok(Some(model)) => model,
        Ok(None) => return ollama_bad_request("model is required"),
        Err(message) => return ollama_bad_request(message),
    };
    if validate_ollama_model_selector(&model).is_err() {
        return ollama_bad_request("model selector is invalid");
    }

    let (catalog, _) = match state.fresh_model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(_) => {
            return ollama_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "failed to inspect the local model catalog",
            );
        }
    };
    let matches = catalog
        .models
        .iter()
        .filter(|entry| ollama_catalog_entry_matches_selector(entry, &model))
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return ollama_error_response(
            axum::http::StatusCode::CONFLICT,
            format!("model selector {model:?} resolves to multiple local catalog entries"),
        );
    }
    let Some(catalog_id) = matches.first().map(|entry| entry.id.clone()) else {
        return ollama_error_response(
            axum::http::StatusCode::NOT_FOUND,
            format!("model {model:?} not found"),
        );
    };

    match remove_catalog_model(&state, &catalog_id).await {
        Ok(_) => axum::http::StatusCode::OK.into_response(),
        Err(error) => ollama_error_response(error.status(), error.message()),
    }
}

pub(crate) async fn handle_ollama_pull(
    State(state): State<Arc<ServerState>>,
    payload: std::result::Result<Json<OllamaPullRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return ollama_json_rejection_response(error),
    };
    if let Err(message) = reject_ollama_extensions("pull request", &payload.extensions) {
        return ollama_bad_request(message);
    }
    if payload.insecure == Some(true) {
        return ollama_bad_request(
            "insecure pulls are not supported; Bloom requires its signed model index and verified HTTPS downloads",
        );
    }
    let model = match required_ollama_model(payload.model) {
        Ok(Some(model)) => model,
        Ok(None) => return ollama_bad_request("model is required"),
        Err(message) => return ollama_bad_request(message),
    };
    if validate_index_id(&model).is_err() {
        return ollama_bad_request("model must be an exact Bloom signed-index ID");
    }
    let stream = payload.stream.unwrap_or(true);
    let Some(downloads) = state.model_downloads.as_ref().cloned() else {
        return ollama_error_response(
            axum::http::StatusCode::FORBIDDEN,
            "verified model downloads are disabled",
        );
    };
    let Some(index) = state.model_index.as_ref() else {
        return ollama_error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "a trusted signed model index is not configured",
        );
    };
    let snapshot = match index.snapshot(false).await {
        Ok(snapshot) => snapshot,
        Err(ModelIndexError::Invalid(_)) => {
            return ollama_error_response(
                axum::http::StatusCode::BAD_GATEWAY,
                "the configured signed model index is invalid",
            );
        }
        Err(ModelIndexError::Unavailable(_)) => {
            return ollama_error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "the configured signed model index is unavailable",
            );
        }
        Err(ModelIndexError::Internal(_)) => {
            return ollama_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "the signed model index could not be inspected",
            );
        }
    };
    let Some(entry) = snapshot.data.into_iter().find(|entry| entry.id == model) else {
        return ollama_error_response(
            axum::http::StatusCode::NOT_FOUND,
            format!("model {model:?} was not found in the trusted signed index"),
        );
    };
    if !entry.downloadable {
        return ollama_error_response(
            axum::http::StatusCode::FORBIDDEN,
            format!("model {model:?} is blocked by the configured size or license policy"),
        );
    }

    match admit_ollama_pull(&state, &downloads, &entry).await {
        Ok(OllamaPullAdmission::Complete) => ollama_pull_success_response(stream),
        Ok(OllamaPullAdmission::Active) if stream => {
            ollama_pull_stream(Arc::clone(&state), downloads, entry)
        }
        Ok(OllamaPullAdmission::Active) => {
            match wait_for_ollama_pull(&state, &downloads, &entry).await {
                Ok(()) => ollama_pull_success_response(false),
                Err(error) => ollama_error_response(error.status, error.message),
            }
        }
        Err(error) => ollama_error_response(error.status, error.message),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OllamaPullAdmission {
    Complete,
    Active,
}

#[derive(Debug)]
struct OllamaPullError {
    status: axum::http::StatusCode,
    message: String,
}

impl OllamaPullError {
    fn new(status: axum::http::StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

async fn admit_ollama_pull(
    state: &Arc<ServerState>,
    downloads: &Arc<ModelDownloadManager>,
    entry: &ModelIndexEntry,
) -> std::result::Result<OllamaPullAdmission, OllamaPullError> {
    let _storage_guard = state.model_storage.serial().await;
    let (catalog, _) = state.fresh_model_catalog_snapshot().await.map_err(|_| {
        OllamaPullError::new(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the local model catalog could not be inspected",
        )
    })?;
    let replacement = match model_index_installation_state(&catalog, entry) {
        InstalledPullState::Verified => return Ok(OllamaPullAdmission::Complete),
        InstalledPullState::Conflict => {
            return Err(OllamaPullError::new(
                axum::http::StatusCode::CONFLICT,
                format!(
                    "catalog entry {:?} already exists but does not match the signed model entry",
                    entry.filename
                ),
            ));
        }
        InstalledPullState::Missing => None,
        InstalledPullState::Upgradable => {
            let source = super::model_index::model_index_upgrade_source(&catalog, entry)
                .ok_or_else(|| {
                    OllamaPullError::new(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "the signed-model upgrade source could not be resolved",
                    )
                })?;
            if source.active {
                return Err(OllamaPullError::new(
                    axum::http::StatusCode::CONFLICT,
                    "unload or switch away from the installed model before pulling its upgrade",
                ));
            }
            if state.load_in_progress.load(Ordering::Acquire) {
                return Err(OllamaPullError::new(
                    axum::http::StatusCode::CONFLICT,
                    "wait for the current model lifecycle operation before pulling an upgrade",
                ));
            }
            if state.model_integrity.is_active(&source.id).await {
                return Err(OllamaPullError::new(
                    axum::http::StatusCode::CONFLICT,
                    "finish or cancel the installed model integrity check before pulling an upgrade",
                ));
            }
            Some(
                super::model_index::model_index_upgrade_descriptor(&catalog, entry).ok_or_else(
                    || {
                        OllamaPullError::new(
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "the signed-model upgrade identity could not be prepared",
                        )
                    },
                )?,
            )
        }
    };

    let current = downloads.status().await;
    if active_download_phase(current.phase) {
        return if current.filename.as_deref() == Some(entry.filename.as_str())
            && downloads
                .active_matches(
                    &entry.filename,
                    &entry.sha256,
                    Some(entry.size_bytes),
                    Some(&entry.id),
                )
                .await
        {
            Ok(OllamaPullAdmission::Active)
        } else {
            Err(OllamaPullError::new(
                axum::http::StatusCode::CONFLICT,
                "another model download is already in progress",
            ))
        };
    }

    let start = if entry.is_package() {
        let request = ModelPackageDownloadRequest {
            directory: entry.filename.clone(),
            size_bytes: entry.size_bytes,
            sha256: entry.sha256.clone(),
            files: entry
                .files
                .iter()
                .map(|file| ModelPackageDownloadFile {
                    url: file.download_url.clone(),
                    filename: file.filename.clone(),
                    size_bytes: file.size_bytes,
                    sha256: file.sha256.clone(),
                })
                .collect(),
            license: Some(entry.license.clone()),
            model_index_id: entry.id.clone(),
        };
        if let Some(replacement) = replacement.clone() {
            downloads.start_package_upgrade(request, replacement).await
        } else {
            downloads.start_package(request).await
        }
    } else {
        let url = entry.download_url.clone().ok_or_else(|| {
            OllamaPullError::new(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "the signed single-file entry has no download URL",
            )
        })?;
        let request = ModelDownloadRequest {
            url,
            filename: entry.filename.clone(),
            sha256: entry.sha256.clone(),
            license: Some(entry.license.clone()),
            expected_size_bytes: Some(entry.size_bytes),
            model_index_id: Some(entry.id.clone()),
        };
        if let Some(replacement) = replacement {
            downloads.start_upgrade(request, replacement).await
        } else {
            downloads.start(request).await
        }
    };
    match start {
        Ok(_) => Ok(OllamaPullAdmission::Active),
        Err(ModelDownloadStartError::Invalid(message)) => Err(OllamaPullError::new(
            axum::http::StatusCode::BAD_REQUEST,
            message,
        )),
        Err(ModelDownloadStartError::NotFound(message)) => Err(OllamaPullError::new(
            axum::http::StatusCode::NOT_FOUND,
            message,
        )),
        Err(ModelDownloadStartError::Internal(_)) => Err(OllamaPullError::new(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the verified model download could not be started",
        )),
        Err(ModelDownloadStartError::Conflict(message)) => {
            match installed_pull_state(state, entry).await? {
                InstalledPullState::Verified => Ok(OllamaPullAdmission::Complete),
                InstalledPullState::Conflict => Err(OllamaPullError::new(
                    axum::http::StatusCode::CONFLICT,
                    format!(
                        "catalog entry {:?} already exists but does not match the signed model entry",
                        entry.filename
                    ),
                )),
                InstalledPullState::Missing | InstalledPullState::Upgradable => {
                    let current = downloads.status().await;
                    if active_download_phase(current.phase)
                        && current.filename.as_deref() == Some(entry.filename.as_str())
                        && downloads
                            .active_matches(
                                &entry.filename,
                                &entry.sha256,
                                Some(entry.size_bytes),
                                Some(&entry.id),
                            )
                            .await
                    {
                        Ok(OllamaPullAdmission::Active)
                    } else {
                        Err(OllamaPullError::new(
                            axum::http::StatusCode::CONFLICT,
                            message,
                        ))
                    }
                }
            }
        }
    }
}

async fn installed_pull_state(
    state: &Arc<ServerState>,
    entry: &ModelIndexEntry,
) -> std::result::Result<InstalledPullState, OllamaPullError> {
    let (catalog, _) = state.fresh_model_catalog_snapshot().await.map_err(|_| {
        OllamaPullError::new(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the local model catalog could not be inspected",
        )
    })?;
    Ok(model_index_installation_state(&catalog, entry))
}

fn active_download_phase(phase: ModelDownloadPhase) -> bool {
    matches!(
        phase,
        ModelDownloadPhase::Queued
            | ModelDownloadPhase::Downloading
            | ModelDownloadPhase::Verifying
    )
}

async fn wait_for_ollama_pull(
    state: &Arc<ServerState>,
    downloads: &Arc<ModelDownloadManager>,
    entry: &ModelIndexEntry,
) -> std::result::Result<(), OllamaPullError> {
    loop {
        let status = downloads.status().await;
        if status.filename.as_deref() != Some(entry.filename.as_str()) {
            if installed_pull_state(state, entry).await? == InstalledPullState::Verified {
                return Ok(());
            }
            return Err(OllamaPullError::new(
                axum::http::StatusCode::CONFLICT,
                "the shared download slot moved to another model",
            ));
        }
        match status.phase {
            ModelDownloadPhase::Queued
            | ModelDownloadPhase::Downloading
            | ModelDownloadPhase::Verifying => {
                tokio::time::sleep(OLLAMA_PULL_POLL_INTERVAL).await;
            }
            ModelDownloadPhase::Complete => {
                return match installed_pull_state(state, entry).await? {
                    InstalledPullState::Verified => Ok(()),
                    _ => Err(OllamaPullError::new(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "the verified download completed without a matching catalog entry",
                    )),
                };
            }
            ModelDownloadPhase::Cancelled => {
                return Err(OllamaPullError::new(
                    axum::http::StatusCode::CONFLICT,
                    "the model download was cancelled and can be resumed",
                ));
            }
            ModelDownloadPhase::Error => {
                return Err(OllamaPullError::new(
                    axum::http::StatusCode::BAD_GATEWAY,
                    status
                        .error
                        .unwrap_or_else(|| "the verified model download failed".to_string()),
                ));
            }
            ModelDownloadPhase::Idle => {
                return Err(OllamaPullError::new(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "the verified model download ended without a terminal status",
                ));
            }
        }
    }
}

fn ollama_pull_success_response(stream: bool) -> axum::response::Response {
    if !stream {
        return ollama_json(json!({"status": "success"}));
    }
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static(OLLAMA_CONTENT_TYPE),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        axum::body::Body::from("{\"status\":\"success\"}\n"),
    )
        .into_response()
}

fn ollama_pull_stream(
    state: Arc<ServerState>,
    downloads: Arc<ModelDownloadManager>,
    entry: ModelIndexEntry,
) -> axum::response::Response {
    let (tx, rx) = mpsc::channel::<std::result::Result<axum::body::Bytes, Infallible>>(32);
    task::spawn(async move {
        let mut previous = None;
        let mut event_count = 0_usize;
        loop {
            let status = downloads.status().await;
            if status.filename.as_deref() != Some(entry.filename.as_str()) {
                match installed_pull_state(&state, &entry).await {
                    Ok(InstalledPullState::Verified) => {
                        let _ = send_ollama_line(&tx, json!({"status": "success"})).await;
                    }
                    Ok(_) => {
                        send_ollama_stream_error(
                            &tx,
                            "the shared download slot moved to another model",
                        )
                        .await;
                    }
                    Err(error) => send_ollama_stream_error(&tx, &error.message).await,
                }
                return;
            }
            match status.phase {
                ModelDownloadPhase::Queued
                | ModelDownloadPhase::Downloading
                | ModelDownloadPhase::Verifying => {
                    if previous.as_ref() != Some(&status) {
                        event_count = event_count.saturating_add(1);
                        if event_count > MAX_OLLAMA_STREAM_EVENTS {
                            send_ollama_stream_error(
                                &tx,
                                "the pull stream exceeded its event limit",
                            )
                            .await;
                            return;
                        }
                        if !send_ollama_line(&tx, ollama_pull_progress(&status, &entry.sha256))
                            .await
                        {
                            return;
                        }
                        previous = Some(status);
                    }
                }
                ModelDownloadPhase::Complete => {
                    match installed_pull_state(&state, &entry).await {
                        Ok(InstalledPullState::Verified) => {
                            let _ = send_ollama_line(&tx, json!({"status": "success"})).await;
                        }
                        Ok(_) => {
                            send_ollama_stream_error(
                                &tx,
                                "the verified download completed without a matching catalog entry",
                            )
                            .await;
                        }
                        Err(error) => send_ollama_stream_error(&tx, &error.message).await,
                    }
                    return;
                }
                ModelDownloadPhase::Cancelled => {
                    send_ollama_stream_error(
                        &tx,
                        "the model download was cancelled and can be resumed",
                    )
                    .await;
                    return;
                }
                ModelDownloadPhase::Error => {
                    send_ollama_stream_error(
                        &tx,
                        status
                            .error
                            .as_deref()
                            .unwrap_or("the verified model download failed"),
                    )
                    .await;
                    return;
                }
                ModelDownloadPhase::Idle => {
                    send_ollama_stream_error(
                        &tx,
                        "the verified model download ended without a terminal status",
                    )
                    .await;
                    return;
                }
            }
            tokio::select! {
                _ = tx.closed() => return,
                _ = tokio::time::sleep(OLLAMA_PULL_POLL_INTERVAL) => {}
            }
        }
    });
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static(OLLAMA_CONTENT_TYPE),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        axum::body::Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response()
}

fn ollama_pull_progress(status: &ModelDownloadStatus, sha256: &str) -> serde_json::Value {
    let label = match status.phase {
        ModelDownloadPhase::Queued => "pulling manifest",
        ModelDownloadPhase::Downloading => "pulling model",
        ModelDownloadPhase::Verifying => "verifying sha256 digest",
        _ => "pulling model",
    };
    let mut progress = serde_json::Map::from_iter([
        ("status".to_string(), json!(label)),
        ("digest".to_string(), json!(format!("sha256:{sha256}"))),
        ("completed".to_string(), json!(status.downloaded_bytes)),
    ]);
    if let Some(total) = status.total_bytes {
        progress.insert("total".to_string(), json!(total));
    }
    serde_json::Value::Object(progress)
}

#[derive(Debug)]
pub(crate) struct OllamaActivationError {
    pub(crate) status: axum::http::StatusCode,
    pub(crate) message: String,
}

impl OllamaActivationError {
    fn new(status: axum::http::StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

#[cfg(test)]
pub(crate) async fn activate_ollama_model(
    state: &Arc<ServerState>,
    requested: &str,
) -> std::result::Result<String, OllamaActivationError> {
    activate_ollama_model_with_permission(state, requested, true)
        .await
        .map(|runtime| runtime.model_id.clone())
}

async fn activate_ollama_model_with_permission(
    state: &Arc<ServerState>,
    requested: &str,
    allow_lifecycle_change: bool,
) -> std::result::Result<Arc<LoadedRuntime>, OllamaActivationError> {
    if validate_ollama_model_selector(requested).is_err() {
        return Err(OllamaActivationError::new(
            axum::http::StatusCode::BAD_REQUEST,
            "model must be a bounded, trimmed identifier without control characters",
        ));
    }

    let _storage_guard = state.model_storage.serial().await;
    let (catalog, _) = state.fresh_model_catalog_snapshot().await.map_err(|_| {
        OllamaActivationError::new(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the local model catalog could not be inspected",
        )
    })?;
    let resident = state.runtime_pool.read().await.snapshot();

    if requested == "default" {
        return resident.default_runtime().ok_or_else(|| {
            OllamaActivationError::new(
                axum::http::StatusCode::NOT_FOUND,
                "model \"default\" is not loaded",
            )
        });
    }

    let mut candidates = catalog
        .models
        .iter()
        .filter(|entry| ollama_catalog_entry_matches_selector(entry, requested))
        .collect::<Vec<_>>();
    if candidates.len() > 1 {
        return Err(OllamaActivationError::new(
            axum::http::StatusCode::CONFLICT,
            format!("model selector {requested:?} resolves to multiple local catalog entries"),
        ));
    }
    let resident_match =
        resolve_resident_runtime(&catalog, &resident, requested, candidates.first().copied())
            .map_err(|()| {
                OllamaActivationError::new(
                    axum::http::StatusCode::CONFLICT,
                    format!("model selector {requested:?} resolves to multiple loaded runtimes"),
                )
            })?;
    if let Some(runtime) = resident_match {
        if allow_lifecycle_change && !state.runtime_pool.write().await.promote_exact(&runtime) {
            return Err(OllamaActivationError::new(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "the selected model was unloaded before inference admission",
            ));
        }
        return Ok(runtime);
    }
    if !allow_lifecycle_change {
        return Err(OllamaActivationError::new(
            axum::http::StatusCode::FORBIDDEN,
            "the inference API key cannot load or switch the active model",
        ));
    }

    if candidates.is_empty()
        && validate_index_id(requested).is_ok()
        && let Some(index) = state.model_index.as_ref()
    {
        let snapshot = index.snapshot(false).await.map_err(|_| {
            OllamaActivationError::new(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "the signed model index is temporarily unavailable",
            )
        })?;
        if let Some(index_entry) = snapshot.data.iter().find(|entry| entry.id == requested)
            && let Some(entry) = catalog
                .models
                .iter()
                .find(|entry| entry.id == index_entry.filename)
                .filter(|entry| catalog_entry_matches_signed_index(entry, index_entry))
        {
            candidates.push(entry);
        }
    }

    if candidates.len() > 1 {
        return Err(OllamaActivationError::new(
            axum::http::StatusCode::CONFLICT,
            format!("model selector {requested:?} resolves to multiple local catalog entries"),
        ));
    }
    let Some(entry) = candidates.first() else {
        return Err(OllamaActivationError::new(
            axum::http::StatusCode::NOT_FOUND,
            format!("model {requested:?} not found"),
        ));
    };
    let resident_match = resolve_resident_runtime(&catalog, &resident, requested, Some(entry))
        .map_err(|()| {
            OllamaActivationError::new(
                axum::http::StatusCode::CONFLICT,
                format!("model selector {requested:?} resolves to multiple loaded runtimes"),
            )
        })?;
    if let Some(runtime) = resident_match {
        if !state.runtime_pool.write().await.promote_exact(&runtime) {
            return Err(OllamaActivationError::new(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "the selected model was unloaded before inference admission",
            ));
        }
        return Ok(runtime);
    }

    let catalog_id = entry.id.clone();
    let path = prepare_catalog_model_load(state, &catalog_id)
        .await
        .map_err(|error| OllamaActivationError::new(error.status, error.message))?;
    let admission = state
        .admit_model_load(path, Some(catalog_id), true)
        .await
        .map_err(|error| match error {
            ModelLoadAdmissionError::Busy => OllamaActivationError::new(
                axum::http::StatusCode::CONFLICT,
                "another model lifecycle operation is already in progress",
            ),
            ModelLoadAdmissionError::Unavailable(message) => {
                OllamaActivationError::new(axum::http::StatusCode::SERVICE_UNAVAILABLE, message)
            }
        })?;
    drop(_storage_guard);

    match admission {
        ModelLoadAdmission::AlreadyReady { runtime } => Ok(runtime),
        ModelLoadAdmission::Loading {
            sequence,
            queued,
            completion,
        } => {
            tracing::debug!(
                sequence,
                queued,
                model = requested,
                "Waiting for model activation"
            );
            wait_for_model_activation(completion).await
        }
    }
}

fn validate_ollama_model_selector(selector: &str) -> std::result::Result<(), ()> {
    if validate_model_selector(selector).is_err()
        || (selector != "default"
            && model_manager::validate_catalog_id(selector).is_err()
            && validate_index_id(selector).is_err())
    {
        Err(())
    } else {
        Ok(())
    }
}

pub(crate) async fn wait_for_model_activation(
    mut completion: watch::Receiver<ModelLoadOutcome>,
) -> std::result::Result<Arc<LoadedRuntime>, OllamaActivationError> {
    loop {
        match completion.borrow().clone() {
            ModelLoadOutcome::Loading => {}
            ModelLoadOutcome::Ready { runtime } => return Ok(runtime),
            ModelLoadOutcome::Failed { message } => {
                return Err(OllamaActivationError::new(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    format!("model activation failed: {message}"),
                ));
            }
        }
        completion.changed().await.map_err(|_| {
            OllamaActivationError::new(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model activation ended without a terminal result",
            )
        })?;
    }
}

fn catalog_entry_matches_signed_index(
    entry: &model_manager::ModelCatalogEntry,
    signed: &ModelIndexEntry,
) -> bool {
    entry.size_complete
        && entry.size_bytes == signed.size_bytes
        && entry.provenance_error.is_none()
        && entry.provenance.as_ref().is_some_and(|provenance| {
            provenance.sha256.eq_ignore_ascii_case(&signed.sha256)
                && provenance
                    .license
                    .as_deref()
                    .is_some_and(|license| license.eq_ignore_ascii_case(&signed.license))
                && provenance.integrity_mismatch_at.is_none()
                && provenance
                    .model_index_id
                    .as_deref()
                    .is_none_or(|id| id == signed.id)
        })
}

fn ollama_chat_lifecycle_action(
    payload: &OllamaChatRequest,
) -> std::result::Result<Option<OllamaLifecycleAction>, String> {
    if !payload.messages.is_empty() {
        return Ok(None);
    }
    reject_ollama_extensions("chat request", &payload.extensions)?;
    validate_ollama_lifecycle_controls(
        payload.think.as_ref(),
        payload.keep_alive.as_ref(),
        payload.logprobs,
        payload.top_logprobs,
    )?;
    if !ollama_neutral_value(payload.tools.as_ref())
        || !ollama_neutral_value(payload.format.as_ref())
        || !ollama_neutral_value(payload.options.as_ref())
    {
        return Err(
            "empty-message lifecycle requests cannot include generation options, tools, or output formats"
                .to_string(),
        );
    }
    Ok(Some(
        match parse_ollama_keep_alive(payload.keep_alive.as_ref())? {
            OllamaKeepAlive::Unload => OllamaLifecycleAction::Unload,
            keep_alive => OllamaLifecycleAction::Load(keep_alive),
        },
    ))
}

fn ollama_generate_lifecycle_action(
    payload: &OllamaGenerateRequest,
) -> std::result::Result<Option<OllamaLifecycleAction>, String> {
    if payload
        .prompt
        .as_deref()
        .is_some_and(|prompt| !prompt.trim().is_empty())
    {
        return Ok(None);
    }
    reject_ollama_extensions("generate request", &payload.extensions)?;
    validate_ollama_lifecycle_controls(
        payload.think.as_ref(),
        payload.keep_alive.as_ref(),
        payload.logprobs,
        payload.top_logprobs,
    )?;
    if payload.raw == Some(true)
        || payload
            .system
            .as_deref()
            .is_some_and(|value| !value.is_empty())
        || payload
            .suffix
            .as_deref()
            .is_some_and(|value| !value.is_empty())
        || !ollama_neutral_value(payload.images.as_ref())
        || !ollama_neutral_value(payload.format.as_ref())
        || !ollama_neutral_value(payload.options.as_ref())
    {
        return Err(
            "empty-prompt lifecycle requests cannot include generation inputs or options"
                .to_string(),
        );
    }
    Ok(Some(
        match parse_ollama_keep_alive(payload.keep_alive.as_ref())? {
            OllamaKeepAlive::Unload => OllamaLifecycleAction::Unload,
            keep_alive => OllamaLifecycleAction::Load(keep_alive),
        },
    ))
}

fn ollama_neutral_value(value: Option<&serde_json::Value>) -> bool {
    value.is_none_or(|value| {
        value.is_null()
            || value.as_array().is_some_and(Vec::is_empty)
            || value.as_object().is_some_and(serde_json::Map::is_empty)
    })
}

async fn handle_ollama_lifecycle(
    state: &Arc<ServerState>,
    model: Option<String>,
    action: OllamaLifecycleAction,
    kind: OllamaOutputKind,
    started: Instant,
) -> axum::response::Response {
    let model = match required_ollama_model(model) {
        Ok(Some(model)) => model,
        Ok(None) => return ollama_bad_request("model is required"),
        Err(message) => return ollama_bad_request(message),
    };
    if validate_ollama_model_selector(&model).is_err() {
        return ollama_bad_request("model selector is invalid");
    }

    // Resolve or load without holding the residency registry. Only policy
    // commit and exact unload are serialized with expiry, so work for one
    // runtime cannot stall every other resident timer or `/api/ps` query.
    let mut residency_lease = None;
    let done_reason = match action {
        OllamaLifecycleAction::Load(keep_alive) => {
            let runtime = match activate_ollama_model_with_permission(state, &model, true).await {
                Ok(runtime) => runtime,
                Err(error) => return ollama_error_response(error.status, error.message),
            };
            let mut residency = state.ollama_residency.lock().await;
            if !state.runtime_pool.read().await.contains_exact(&runtime) {
                return ollama_error_response(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "the selected model was unloaded before its residency policy could be committed",
                );
            }
            let (residency_revision, residency_timer_cancel) =
                commit_ollama_residency_request(&mut residency, &runtime);
            drop(residency);
            let lease = match ollama_residency_lease(
                Arc::clone(state),
                &runtime,
                residency_revision,
                residency_timer_cancel,
                keep_alive,
            )
            .await
            {
                Ok(lease) => lease,
                Err(message) => {
                    return ollama_error_response(
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        message,
                    );
                }
            };
            residency_lease = Some(lease);
            "load"
        }
        OllamaLifecycleAction::Unload => {
            if state.load_in_progress.load(Ordering::Acquire) {
                return ollama_error_response(
                    axum::http::StatusCode::CONFLICT,
                    "another model lifecycle operation is already in progress",
                );
            }
            let (catalog, _) = match state.fresh_model_catalog_snapshot().await {
                Ok(snapshot) => snapshot,
                Err(_) => {
                    return ollama_error_response(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "failed to inspect the active local model",
                    );
                }
            };
            let resident = state.runtime_pool.read().await.snapshot();
            let runtime = if model == "default" {
                resident.default_runtime()
            } else {
                let matching_entries = catalog
                    .models
                    .iter()
                    .filter(|entry| ollama_catalog_entry_matches_selector(entry, &model))
                    .collect::<Vec<_>>();
                if matching_entries.len() > 1 {
                    return ollama_error_response(
                        axum::http::StatusCode::CONFLICT,
                        format!(
                            "model selector {model:?} resolves to multiple local catalog entries"
                        ),
                    );
                }
                match resolve_resident_runtime(
                    &catalog,
                    &resident,
                    &model,
                    matching_entries.first().copied(),
                ) {
                    Ok(runtime) => runtime,
                    Err(()) => {
                        return ollama_error_response(
                            axum::http::StatusCode::CONFLICT,
                            format!(
                                "model selector {model:?} resolves to multiple loaded runtimes"
                            ),
                        );
                    }
                }
            };
            let Some(runtime) = runtime else {
                return ollama_error_response(
                    axum::http::StatusCode::NOT_FOUND,
                    format!("model {model:?} is not loaded"),
                );
            };
            let mut residency = state.ollama_residency.lock().await;
            let response = handle_model_unload_exact(Arc::clone(state), Arc::clone(&runtime)).await;
            if !response.status().is_success() {
                return adapt_ollama_error_response(response).await;
            }
            residency.cancel_runtime(&runtime);
            drop(residency);
            "unload"
        }
    };

    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let mut payload = ollama_base_payload(&model, unix_seconds(), kind);
    payload["done"] = json!(true);
    payload["done_reason"] = json!(done_reason);
    payload["total_duration"] = json!(elapsed);
    payload["load_duration"] = json!(elapsed);
    payload["prompt_eval_count"] = json!(0);
    payload["eval_count"] = json!(0);
    let response = ollama_json(payload);
    drop(residency_lease);
    response
}

fn ollama_has_operator_scope(scope: Option<axum::extract::Extension<CredentialScope>>) -> bool {
    scope
        .map(|axum::extract::Extension(scope)| scope == CredentialScope::Operator)
        .unwrap_or(true)
}

fn ollama_operator_permission_error(operation: &str) -> axum::response::Response {
    ollama_error_response(
        axum::http::StatusCode::FORBIDDEN,
        format!("the inference API key cannot {operation}"),
    )
}

async fn activate_ollama_model_for_request(
    state: &Arc<ServerState>,
    requested_model: &str,
    operator_scope: bool,
    keep_alive: OllamaKeepAlive,
) -> std::result::Result<(Arc<LoadedRuntime>, Option<OllamaResidencyLease>), OllamaActivationError>
{
    let runtime =
        activate_ollama_model_with_permission(state, requested_model, operator_scope).await?;
    if !operator_scope {
        return Ok((runtime, None));
    }

    let mut residency = state.ollama_residency.lock().await;
    if !state.runtime_pool.read().await.contains_exact(&runtime) {
        return Err(OllamaActivationError::new(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "the selected model was unloaded before its residency policy could be committed",
        ));
    }
    let (residency_revision, residency_timer_cancel) =
        commit_ollama_residency_request(&mut residency, &runtime);
    drop(residency);
    let lease = ollama_residency_lease(
        Arc::clone(state),
        &runtime,
        residency_revision,
        residency_timer_cancel,
        keep_alive,
    )
    .await
    .map_err(|message| {
        OllamaActivationError::new(axum::http::StatusCode::SERVICE_UNAVAILABLE, message)
    })?;
    Ok((runtime, Some(lease)))
}

pub(crate) async fn handle_ollama_chat(
    State(state): State<Arc<ServerState>>,
    scope: Option<axum::extract::Extension<CredentialScope>>,
    payload: std::result::Result<Json<OllamaChatRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    let operator_scope = ollama_has_operator_scope(scope);
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return ollama_json_rejection_response(error),
    };
    let started = Instant::now();
    match ollama_chat_lifecycle_action(&payload) {
        Ok(Some(action)) => {
            if !operator_scope {
                return ollama_operator_permission_error("change model residency");
            }
            return handle_ollama_lifecycle(
                &state,
                payload.model.clone(),
                action,
                OllamaOutputKind::Chat,
                started,
            )
            .await;
        }
        Ok(None) => {}
        Err(message) => return ollama_bad_request(message),
    }
    if !operator_scope
        && payload
            .keep_alive
            .as_ref()
            .is_some_and(|value| !value.is_null())
    {
        return ollama_operator_permission_error("set keep_alive");
    }
    let stream = payload.stream.unwrap_or(true);
    let keep_alive = match parse_ollama_keep_alive(payload.keep_alive.as_ref()) {
        Ok(keep_alive) => keep_alive,
        Err(message) => return ollama_bad_request(message),
    };
    let (chat_request, image) = match ollama_chat_request(payload, stream) {
        Ok(request) => request,
        Err(message) => return ollama_bad_request(message),
    };
    let image_request = match image
        .map(|image| prepare_ollama_image_request(&chat_request, image))
        .transpose()
    {
        Ok(request) => request,
        Err(message) => return ollama_bad_request(message),
    };
    let requested_model = chat_request.model.clone().unwrap_or_default();
    let (runtime, residency_lease) = match activate_ollama_model_for_request(
        &state,
        &requested_model,
        operator_scope,
        keep_alive,
    )
    .await
    {
        Ok(activation) => activation,
        Err(error) => return ollama_error_response(error.status, error.message),
    };
    if let Some(image_request) = image_request {
        let request = image_request.into_inference_request();
        let response = run_multimodal_request_for_runtime(state, request, runtime).await;
        return ollama_from_multimodal_response(
            response,
            OllamaOutputKind::Chat,
            stream,
            started,
            requested_model,
            residency_lease,
        )
        .await;
    }
    let response = handle_chat_completions_for_runtime(state, chat_request, runtime).await;
    ollama_from_chat_response(
        response,
        OllamaOutputKind::Chat,
        stream,
        started,
        requested_model,
        residency_lease,
    )
    .await
}

pub(crate) async fn handle_ollama_generate(
    State(state): State<Arc<ServerState>>,
    scope: Option<axum::extract::Extension<CredentialScope>>,
    payload: std::result::Result<
        Json<OllamaGenerateRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> axum::response::Response {
    let operator_scope = ollama_has_operator_scope(scope);
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return ollama_json_rejection_response(error),
    };
    let started = Instant::now();
    match ollama_generate_lifecycle_action(&payload) {
        Ok(Some(action)) => {
            if !operator_scope {
                return ollama_operator_permission_error("change model residency");
            }
            return handle_ollama_lifecycle(
                &state,
                payload.model.clone(),
                action,
                OllamaOutputKind::Generate,
                started,
            )
            .await;
        }
        Ok(None) => {}
        Err(message) => return ollama_bad_request(message),
    }
    if !operator_scope
        && payload
            .keep_alive
            .as_ref()
            .is_some_and(|value| !value.is_null())
    {
        return ollama_operator_permission_error("set keep_alive");
    }
    let stream = payload.stream.unwrap_or(true);
    let keep_alive = match parse_ollama_keep_alive(payload.keep_alive.as_ref()) {
        Ok(keep_alive) => keep_alive,
        Err(message) => return ollama_bad_request(message),
    };
    let (chat_request, image) = match ollama_generate_request(payload, stream) {
        Ok(request) => request,
        Err(message) => return ollama_bad_request(message),
    };
    let image_request = match image
        .map(|image| prepare_ollama_image_request(&chat_request, image))
        .transpose()
    {
        Ok(request) => request,
        Err(message) => return ollama_bad_request(message),
    };
    let requested_model = chat_request.model.clone().unwrap_or_default();
    let (runtime, residency_lease) = match activate_ollama_model_for_request(
        &state,
        &requested_model,
        operator_scope,
        keep_alive,
    )
    .await
    {
        Ok(activation) => activation,
        Err(error) => return ollama_error_response(error.status, error.message),
    };
    if let Some(image_request) = image_request {
        let request = image_request.into_inference_request();
        let response = run_multimodal_request_for_runtime(state, request, runtime).await;
        return ollama_from_multimodal_response(
            response,
            OllamaOutputKind::Generate,
            stream,
            started,
            requested_model,
            residency_lease,
        )
        .await;
    }
    let response = handle_chat_completions_for_runtime(state, chat_request, runtime).await;
    ollama_from_chat_response(
        response,
        OllamaOutputKind::Generate,
        stream,
        started,
        requested_model,
        residency_lease,
    )
    .await
}

pub(crate) async fn handle_ollama_embed(
    State(state): State<Arc<ServerState>>,
    scope: Option<axum::extract::Extension<CredentialScope>>,
    payload: std::result::Result<Json<OllamaEmbedRequest>, axum::extract::rejection::JsonRejection>,
) -> axum::response::Response {
    let operator_scope = ollama_has_operator_scope(scope);
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return ollama_json_rejection_response(error),
    };
    if !operator_scope
        && payload
            .keep_alive
            .as_ref()
            .is_some_and(|value| !value.is_null())
    {
        return ollama_operator_permission_error("set keep_alive");
    }
    let keep_alive = match parse_ollama_keep_alive(payload.keep_alive.as_ref()) {
        Ok(keep_alive) => keep_alive,
        Err(message) => return ollama_bad_request(message),
    };
    if let Err(message) =
        reject_ollama_extensions("embed request", &payload.extensions).and_then(|()| {
            validate_ollama_embedding_controls(
                payload.options.as_ref(),
                payload.keep_alive.as_ref(),
            )
        })
    {
        return ollama_bad_request(message);
    }
    if payload
        .dimensions
        .is_some_and(|dimensions| dimensions > MAX_EMBEDDING_DIMENSIONS)
    {
        return ollama_bad_request(format!(
            "dimensions cannot exceed {MAX_EMBEDDING_DIMENSIONS}"
        ));
    }
    let model = match required_ollama_model(payload.model) {
        Ok(Some(model)) => model,
        Ok(None) => return ollama_bad_request("model is required"),
        Err(message) => return ollama_bad_request(message),
    };
    let inputs = match normalize_embedding_input(&payload.input) {
        Ok(inputs) => inputs,
        Err(message) => return ollama_bad_request(message),
    };
    let (runtime, _residency_lease) =
        match activate_ollama_model_for_request(&state, &model, operator_scope, keep_alive).await {
            Ok(activation) => activation,
            Err(error) => return ollama_error_response(error.status, error.message),
        };
    let mut result = match execute_embedding_batch_for_runtime(
        state,
        runtime,
        inputs,
        payload.truncate.unwrap_or(true),
        EmbeddingProjection::L2Normalized {
            dimensions: payload.dimensions.filter(|dimensions| *dimensions > 0),
            require_exact_dimensions: false,
        },
    )
    .await
    {
        Ok(result) => result,
        Err(error) => return ollama_error_response(error.status, error.message),
    };
    result.model_id = model;
    match ollama_embed_payload(result) {
        Ok(payload) => ollama_json(payload),
        Err(message) => {
            ollama_error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, message)
        }
    }
}

pub(crate) async fn handle_ollama_legacy_embeddings(
    State(state): State<Arc<ServerState>>,
    scope: Option<axum::extract::Extension<CredentialScope>>,
    payload: std::result::Result<
        Json<OllamaLegacyEmbeddingRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> axum::response::Response {
    let operator_scope = ollama_has_operator_scope(scope);
    let Json(payload) = match payload {
        Ok(payload) => payload,
        Err(error) => return ollama_json_rejection_response(error),
    };
    if !operator_scope
        && payload
            .keep_alive
            .as_ref()
            .is_some_and(|value| !value.is_null())
    {
        return ollama_operator_permission_error("set keep_alive");
    }
    let keep_alive = match parse_ollama_keep_alive(payload.keep_alive.as_ref()) {
        Ok(keep_alive) => keep_alive,
        Err(message) => return ollama_bad_request(message),
    };
    if let Err(message) = reject_ollama_extensions("legacy embeddings request", &payload.extensions)
        .and_then(|()| {
            validate_ollama_embedding_controls(
                payload.options.as_ref(),
                payload.keep_alive.as_ref(),
            )
        })
    {
        return ollama_bad_request(message);
    }
    let model = match required_ollama_model(payload.model) {
        Ok(Some(model)) => model,
        Ok(None) => return ollama_bad_request("model is required"),
        Err(message) => return ollama_bad_request(message),
    };
    let prompt = serde_json::Value::String(payload.prompt.unwrap_or_default());
    let inputs = match normalize_embedding_input(&prompt) {
        Ok(inputs) => inputs,
        Err(message) => return ollama_bad_request(message),
    };
    let (runtime, _residency_lease) =
        match activate_ollama_model_for_request(&state, &model, operator_scope, keep_alive).await {
            Ok(activation) => activation,
            Err(error) => return ollama_error_response(error.status, error.message),
        };
    let result = match execute_embedding_batch_for_runtime(
        state,
        runtime,
        inputs,
        false,
        EmbeddingProjection::L2Normalized {
            dimensions: None,
            require_exact_dimensions: true,
        },
    )
    .await
    {
        Ok(result) => result,
        Err(error) => return ollama_error_response(error.status, error.message),
    };
    let EmbeddingBatchOutput::Embeddings(embeddings) = result.output else {
        return ollama_error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the embedding worker returned an unexpected result type",
        );
    };
    let Some(embedding) = embeddings.into_iter().next() else {
        return ollama_error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the embedding worker returned no vectors",
        );
    };
    ollama_json(json!({"embedding": embedding}))
}

fn ollama_embed_payload(
    result: EmbeddingBatchResult,
) -> std::result::Result<serde_json::Value, String> {
    let duration = u64::try_from(result.total_duration.as_nanos()).unwrap_or(u64::MAX);
    let EmbeddingBatchOutput::Embeddings(embeddings) = result.output else {
        return Err("the embedding worker returned an unexpected result type".to_string());
    };
    Ok(json!({
        "model": result.model_id,
        "embeddings": embeddings,
        "total_duration": duration,
        "load_duration": 0,
        "prompt_eval_count": result.prompt_tokens
    }))
}

fn ollama_chat_request(
    payload: OllamaChatRequest,
    stream: bool,
) -> std::result::Result<(ChatRequest, Option<OllamaImageInput>), String> {
    reject_ollama_extensions("chat request", &payload.extensions)?;
    validate_ollama_neutral_controls(
        payload.think.as_ref(),
        payload.keep_alive.as_ref(),
        payload.logprobs,
        payload.top_logprobs,
    )?;
    if payload.messages.is_empty() {
        return Err("messages must contain at least one entry".to_string());
    }
    let options = parse_ollama_options(payload.options.as_ref())?;
    let (messages, image) = ollama_chat_messages(payload.messages)?;
    let response_format = ollama_response_format(payload.format.as_ref())?;
    Ok((
        ChatRequest {
            model: required_ollama_model(payload.model)?,
            messages,
            stream,
            stream_options: stream.then_some(StreamOptions {
                include_usage: true,
                extensions: BTreeMap::new(),
            }),
            max_tokens: None,
            max_completion_tokens: Some(options.max_tokens.unwrap_or(128)),
            temperature: Some(options.temperature.unwrap_or(0.7)),
            top_p: Some(options.top_p.unwrap_or(0.9)),
            seed: options.seed,
            stop: (!options.stop.is_empty()).then(|| json!(options.stop)),
            response_format,
            tools: payload.tools,
            tool_choice: None,
            parallel_tool_calls: None,
            internal_request_id: None,
            extensions: BTreeMap::new(),
        },
        image,
    ))
}

fn ollama_generate_request(
    payload: OllamaGenerateRequest,
    stream: bool,
) -> std::result::Result<(ChatRequest, Option<OllamaImageInput>), String> {
    reject_ollama_extensions("generate request", &payload.extensions)?;
    validate_ollama_neutral_controls(
        payload.think.as_ref(),
        payload.keep_alive.as_ref(),
        payload.logprobs,
        payload.top_logprobs,
    )?;
    if payload.raw == Some(true) {
        return Err("raw prompt mode is not supported by Bloom's Ollama adapter".to_string());
    }
    if payload
        .suffix
        .as_ref()
        .is_some_and(|suffix| !suffix.is_empty())
    {
        return Err("fill-in-the-middle suffix generation is not supported".to_string());
    }
    let image = parse_ollama_images(payload.images.as_ref(), "generate request")?;
    let prompt = match payload.prompt {
        Some(prompt) if !prompt.trim().is_empty() => Some(prompt),
        _ if image.is_some() => None,
        _ => return Err("prompt is required and must not be blank".to_string()),
    };
    let options = parse_ollama_options(payload.options.as_ref())?;
    let mut messages = Vec::with_capacity(2);
    if let Some(system) = payload.system.filter(|system| !system.is_empty()) {
        messages.push(ChatCompletionMessage {
            role: "system".to_string(),
            content: json!(system),
            extensions: BTreeMap::new(),
        });
    }
    messages.push(ChatCompletionMessage {
        role: "user".to_string(),
        content: json!(prompt.unwrap_or_default()),
        extensions: BTreeMap::new(),
    });
    Ok((
        ChatRequest {
            model: required_ollama_model(payload.model)?,
            messages,
            stream,
            stream_options: stream.then_some(StreamOptions {
                include_usage: true,
                extensions: BTreeMap::new(),
            }),
            max_tokens: None,
            max_completion_tokens: Some(options.max_tokens.unwrap_or(128)),
            temperature: Some(options.temperature.unwrap_or(0.7)),
            top_p: Some(options.top_p.unwrap_or(0.9)),
            seed: options.seed,
            stop: (!options.stop.is_empty()).then(|| json!(options.stop)),
            response_format: ollama_response_format(payload.format.as_ref())?,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            internal_request_id: None,
            extensions: BTreeMap::new(),
        },
        image,
    ))
}

fn parse_ollama_images(
    value: Option<&serde_json::Value>,
    context: &str,
) -> std::result::Result<Option<OllamaImageInput>, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let images = value
        .as_array()
        .ok_or_else(|| format!("{context} images must be an array of base64 strings"))?;
    if images.is_empty() {
        return Ok(None);
    }
    if images.len() != 1 {
        return Err("Ollama requests can contain at most one image".to_string());
    }
    let encoded = images[0]
        .as_str()
        .ok_or_else(|| format!("{context} image must be a base64 string"))?;
    if encoded.is_empty() {
        return Err(format!("{context} image must not be empty"));
    }
    if encoded.len() > MAX_OLLAMA_IMAGE_BASE64_BYTES {
        return Err(format!(
            "{context} image exceeds the {MAX_MULTIMODAL_IMAGE_BYTES}-byte decoded limit"
        ));
    }
    if encoded.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(format!(
            "{context} image must use canonical base64 without whitespace"
        ));
    }
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|_| format!("{context} image contains invalid base64"))?;
    if STANDARD.encode(&bytes) != encoded {
        return Err(format!("{context} image must use canonical padded base64"));
    }
    if bytes.is_empty() || bytes.len() > MAX_MULTIMODAL_IMAGE_BYTES {
        return Err(format!(
            "{context} image must decode to between 1 and {MAX_MULTIMODAL_IMAGE_BYTES} bytes"
        ));
    }
    let mime = match image::guess_format(&bytes) {
        Ok(image::ImageFormat::Jpeg) => "image/jpeg",
        Ok(image::ImageFormat::Png) => "image/png",
        Ok(_) => return Err(format!("{context} image must be JPEG or PNG")),
        Err(_) => return Err(format!("{context} image has an invalid file signature")),
    };
    validate_uploaded_image(&bytes, mime)
        .map_err(|message| format!("{context} image is invalid: {message}"))?;
    Ok(Some(OllamaImageInput {
        bytes,
        mime: mime.to_string(),
    }))
}

fn prepare_ollama_image_request(
    request: &ChatRequest,
    image: OllamaImageInput,
) -> std::result::Result<OllamaPreparedImageRequest, String> {
    if !ollama_neutral_value(request.tools.as_ref()) {
        return Err("tools cannot be combined with image input".to_string());
    }
    if request.response_format.is_some() {
        return Err("format cannot be combined with image input".to_string());
    }
    if request.stop.is_some() {
        return Err("stop sequences cannot be combined with image input".to_string());
    }
    let messages = normalize_chat_messages(&request.messages)?;
    validate_chat_messages(&messages)?;
    if messages.len() != 1 || messages[0].role != "user" {
        return Err(
            "image requests currently support exactly one user message and no system or history messages"
                .to_string(),
        );
    }
    let max_tokens = resolve_chat_max_tokens(request.max_tokens, request.max_completion_tokens)?;
    let params = InferenceParams {
        max_tokens,
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        seed: request.seed,
        response_format: None,
    };
    validate_generation_controls(params.max_tokens, params.temperature, params.top_p)?;
    Ok(OllamaPreparedImageRequest {
        prompt: (!messages[0].content.trim().is_empty()).then(|| messages[0].content.clone()),
        image,
        params,
    })
}

impl OllamaPreparedImageRequest {
    fn into_inference_request(self) -> InferenceRequest {
        let mut blocks = Vec::with_capacity(2);
        if let Some(prompt) = self.prompt {
            blocks.push(DataBlock::Text(prompt));
        }
        blocks.push(DataBlock::Image {
            bytes: self.image.bytes,
            mime: self.image.mime,
        });
        InferenceRequest {
            blocks,
            params: self.params,
        }
    }
}

fn required_ollama_model(model: Option<String>) -> std::result::Result<Option<String>, String> {
    let model = model
        .filter(|model| !model.is_empty())
        .ok_or_else(|| "model is required".to_string())?;
    Ok(Some(model))
}

fn ollama_chat_messages(
    messages: Vec<OllamaMessage>,
) -> std::result::Result<(Vec<ChatCompletionMessage>, Option<OllamaImageInput>), String> {
    if messages.len() > MAX_CHAT_REQUEST_MESSAGES {
        return Err(format!(
            "messages cannot contain more than {MAX_CHAT_REQUEST_MESSAGES} entries"
        ));
    }
    let mut converted = Vec::with_capacity(messages.len());
    let mut outstanding = VecDeque::<(String, String)>::new();
    let mut request_image = None;
    for (message_index, message) in messages.into_iter().enumerate() {
        reject_ollama_extensions(
            &format!("message at index {message_index}"),
            &message.extensions,
        )?;
        let image = parse_ollama_images(
            message.images.as_ref(),
            &format!("message at index {message_index}"),
        )?;
        let has_image = image.is_some();
        if has_image && message.role != "user" {
            return Err(format!(
                "message at index {message_index} may attach images only to the user role"
            ));
        }
        if let Some(image) = image
            && request_image.replace(image).is_some()
        {
            return Err("Ollama requests can contain at most one image".to_string());
        }
        if message
            .thinking
            .as_ref()
            .is_some_and(|thinking| !thinking.is_empty())
        {
            return Err(format!(
                "message at index {message_index} contains unsupported thinking history"
            ));
        }
        match message.role.as_str() {
            "system" | "user" => {
                if !outstanding.is_empty() {
                    return Err(format!(
                        "message at index {message_index} appears before all tool calls received results"
                    ));
                }
                if message.tool_calls.as_ref().is_some_and(|calls| {
                    !calls.is_null() && calls.as_array().is_none_or(|calls| !calls.is_empty())
                }) || message
                    .tool_name
                    .as_ref()
                    .is_some_and(|name| !name.is_empty())
                {
                    return Err(format!(
                        "message at index {message_index} contains tool fields for role {:?}",
                        message.role
                    ));
                }
                let content = match message.content {
                    Some(content) => content,
                    None if message.role == "user" && has_image => String::new(),
                    _ => {
                        return Err(format!(
                            "message at index {message_index} requires string content"
                        ));
                    }
                };
                converted.push(ChatCompletionMessage {
                    role: message.role,
                    content: json!(content),
                    extensions: BTreeMap::new(),
                });
            }
            "assistant" => {
                if !outstanding.is_empty() {
                    return Err(format!(
                        "assistant message at index {message_index} appears before all tool calls received results"
                    ));
                }
                if message
                    .tool_name
                    .as_ref()
                    .is_some_and(|name| !name.is_empty())
                {
                    return Err(format!(
                        "assistant message at index {message_index} cannot contain tool_name"
                    ));
                }
                let calls = match message.tool_calls.as_ref().filter(|calls| !calls.is_null()) {
                    Some(calls) => ollama_historical_tool_calls(calls, message_index)?,
                    None => Vec::new(),
                };
                let mut extensions = BTreeMap::new();
                if !calls.is_empty() {
                    for call in &calls {
                        let id = call
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .expect("converted Ollama call has an ID")
                            .to_string();
                        let name = call
                            .pointer("/function/name")
                            .and_then(serde_json::Value::as_str)
                            .expect("converted Ollama call has a name")
                            .to_string();
                        outstanding.push_back((id, name));
                    }
                    extensions.insert("tool_calls".to_string(), serde_json::Value::Array(calls));
                }
                let content = match message.content {
                    Some(content) => json!(content),
                    None if extensions.contains_key("tool_calls") => serde_json::Value::Null,
                    None => {
                        return Err(format!(
                            "assistant message at index {message_index} requires content or tool_calls"
                        ));
                    }
                };
                converted.push(ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content,
                    extensions,
                });
            }
            "tool" => {
                if message.tool_calls.as_ref().is_some_and(|calls| {
                    !calls.is_null() && calls.as_array().is_none_or(|calls| !calls.is_empty())
                }) {
                    return Err(format!(
                        "tool message at index {message_index} cannot contain tool_calls"
                    ));
                }
                let name = message
                    .tool_name
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        format!("tool message at index {message_index} requires tool_name")
                    })?;
                validate_tool_name(&name, "Ollama tool_name")?;
                let position = outstanding
                    .iter()
                    .position(|(_, expected)| expected == &name)
                    .ok_or_else(|| {
                        format!(
                            "tool message at index {message_index} references unknown or already-resolved function {name:?}"
                        )
                    })?;
                let (call_id, _) = outstanding
                    .remove(position)
                    .expect("the outstanding tool position was found");
                let content = message.content.ok_or_else(|| {
                    format!("tool message at index {message_index} requires string content")
                })?;
                converted.push(ChatCompletionMessage {
                    role: "tool".to_string(),
                    content: json!(content),
                    extensions: BTreeMap::from([
                        ("tool_call_id".to_string(), json!(call_id)),
                        ("name".to_string(), json!(name)),
                    ]),
                });
            }
            _ => {
                return Err(format!(
                    "message at index {message_index} must use role system, user, assistant, or tool"
                ));
            }
        }
    }
    if !outstanding.is_empty() {
        return Err(
            "every assistant tool call must be followed by a matching tool result".to_string(),
        );
    }
    Ok((converted, request_image))
}

fn ollama_historical_tool_calls(
    value: &serde_json::Value,
    message_index: usize,
) -> std::result::Result<Vec<serde_json::Value>, String> {
    let calls = value.as_array().ok_or_else(|| {
        format!("assistant tool_calls at message index {message_index} must be an array")
    })?;
    if calls.is_empty() || calls.len() > MAX_PARALLEL_TOOL_CALLS {
        return Err(format!(
            "assistant tool_calls at message index {message_index} must contain between 1 and {MAX_PARALLEL_TOOL_CALLS} calls"
        ));
    }
    calls
        .iter()
        .enumerate()
        .map(|(call_index, call)| {
            let call = call
                .as_object()
                .ok_or_else(|| format!("assistant tool call {call_index} must be an object"))?;
            reject_ollama_object_fields(
                call,
                &["type", "function"],
                &format!("assistant tool call {call_index}"),
            )?;
            if call
                .get("type")
                .filter(|value| !value.is_null())
                .is_some_and(|value| value.as_str() != Some("function"))
            {
                return Err(format!(
                    "assistant tool call {call_index} must have type function"
                ));
            }
            let function = call
                .get("function")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| {
                    format!("assistant tool call {call_index} requires a function object")
                })?;
            reject_ollama_object_fields(
                function,
                &["index", "name", "arguments"],
                &format!("assistant tool call {call_index} function"),
            )?;
            if function
                .get("index")
                .filter(|value| !value.is_null())
                .is_some_and(|value| value.as_u64() != Some(call_index as u64))
            {
                return Err(format!(
                    "assistant tool call {call_index} has an invalid function index"
                ));
            }
            let name = function
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("assistant tool call {call_index} requires a name"))?;
            validate_tool_name(name, "Ollama assistant tool-call name")?;
            let arguments = function
                .get("arguments")
                .filter(|arguments| arguments.is_object())
                .ok_or_else(|| {
                    format!("assistant tool call {call_index} arguments must be an object")
                })?;
            let arguments = serde_json::to_string(arguments).map_err(|error| {
                format!("failed to encode assistant tool-call arguments: {error}")
            })?;
            Ok(json!({
                "id": format!("ollama_call_{message_index}_{call_index}"),
                "type": "function",
                "function": {"name": name, "arguments": arguments}
            }))
        })
        .collect()
}

fn parse_ollama_options(
    options: Option<&serde_json::Value>,
) -> std::result::Result<OllamaGenerationOptions, String> {
    let Some(options) = options.filter(|options| !options.is_null()) else {
        return Ok(OllamaGenerationOptions::default());
    };
    let options = options
        .as_object()
        .ok_or_else(|| "options must be an object".to_string())?;
    reject_ollama_object_fields(
        options,
        &["num_predict", "temperature", "top_p", "seed", "stop"],
        "options",
    )?;
    let max_tokens = options
        .get("num_predict")
        .filter(|value| !value.is_null())
        .map(|value| {
            let value = value
                .as_i64()
                .ok_or_else(|| "options.num_predict must be an integer".to_string())?;
            usize::try_from(value)
                .ok()
                .filter(|value| (1..=32_768).contains(value))
                .ok_or_else(|| "options.num_predict must be between 1 and 32768".to_string())
        })
        .transpose()?;
    let temperature = optional_f64(options.get("temperature"), "options.temperature")?;
    let top_p = optional_f64(options.get("top_p"), "options.top_p")?;
    let seed = options
        .get("seed")
        .filter(|value| !value.is_null())
        .map(|value| {
            let value = value
                .as_i64()
                .ok_or_else(|| "options.seed must be an integer".to_string())?;
            if value == -1 {
                Ok(None)
            } else {
                u64::try_from(value)
                    .map(Some)
                    .map_err(|_| "options.seed must be -1 or a non-negative integer".to_string())
            }
        })
        .transpose()?
        .flatten();
    let stop = normalize_stop_sequences(options.get("stop"))?;
    Ok(OllamaGenerationOptions {
        max_tokens,
        temperature,
        top_p,
        seed,
        stop,
    })
}

fn optional_f64(
    value: Option<&serde_json::Value>,
    label: &str,
) -> std::result::Result<Option<f64>, String> {
    value
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("{label} must be a finite number"))
        })
        .transpose()
}

fn ollama_response_format(
    format: Option<&serde_json::Value>,
) -> std::result::Result<Option<ResponseFormat>, String> {
    let Some(format) = format.filter(|format| !format.is_null()) else {
        return Ok(None);
    };
    if let Some(format) = format.as_str() {
        return match format {
            "" => Ok(None),
            "json" => Ok(Some(ResponseFormat {
                format_type: "json_object".to_string(),
                json_schema: None,
                extensions: BTreeMap::new(),
            })),
            _ => Err("format must be `json` or a supported JSON Schema object".to_string()),
        };
    }
    if !format.is_object() {
        return Err("format must be `json` or a supported JSON Schema object".to_string());
    }
    Ok(Some(ResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: Some(json!({
            "name": "ollama_response",
            "schema": format,
            "strict": true
        })),
        extensions: BTreeMap::new(),
    }))
}

fn validate_ollama_neutral_controls(
    think: Option<&serde_json::Value>,
    keep_alive: Option<&serde_json::Value>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
) -> std::result::Result<(), String> {
    if think.is_some_and(|value| !value.is_null() && value.as_bool() != Some(false)) {
        return Err("thinking output is not supported".to_string());
    }
    parse_ollama_keep_alive(keep_alive)?;
    if logprobs == Some(true) || top_logprobs.is_some_and(|value| value != 0) {
        return Err("token log probabilities are not supported".to_string());
    }
    Ok(())
}

fn validate_ollama_lifecycle_controls(
    think: Option<&serde_json::Value>,
    keep_alive: Option<&serde_json::Value>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
) -> std::result::Result<(), String> {
    if think.is_some_and(|value| !value.is_null() && value.as_bool() != Some(false)) {
        return Err("thinking output is not supported".to_string());
    }
    parse_ollama_keep_alive(keep_alive)?;
    if logprobs == Some(true) || top_logprobs.is_some_and(|value| value != 0) {
        return Err("token log probabilities are not supported".to_string());
    }
    Ok(())
}

fn parse_ollama_keep_alive(
    keep_alive: Option<&serde_json::Value>,
) -> std::result::Result<OllamaKeepAlive, String> {
    let Some(value) = keep_alive.filter(|value| !value.is_null()) else {
        return Ok(OllamaKeepAlive::Timed(DEFAULT_OLLAMA_KEEP_ALIVE));
    };
    if let Some(number) = value.as_f64() {
        return ollama_keep_alive_from_seconds(number);
    }
    if let Some(duration) = value.as_str() {
        return parse_ollama_duration(duration);
    }
    Err("keep_alive must be a number or duration string".to_string())
}

fn ollama_keep_alive_from_seconds(seconds: f64) -> std::result::Result<OllamaKeepAlive, String> {
    if !seconds.is_finite() {
        return Err("keep_alive must be finite".to_string());
    }
    if seconds < 0.0 {
        return Ok(OllamaKeepAlive::Indefinite);
    }
    if seconds == 0.0 {
        return Ok(OllamaKeepAlive::Unload);
    }
    let duration = Duration::try_from_secs_f64(seconds)
        .map_err(|_| "keep_alive duration is out of range".to_string())?;
    if duration > MAX_OLLAMA_KEEP_ALIVE {
        return Err("keep_alive cannot exceed 365 days".to_string());
    }
    Ok(OllamaKeepAlive::Timed(duration))
}

fn parse_ollama_duration(value: &str) -> std::result::Result<OllamaKeepAlive, String> {
    let value = value.trim();
    if value.is_empty() || value.len() > 128 {
        return Err("keep_alive duration string is empty or too long".to_string());
    }
    if let Ok(seconds) = value.parse::<f64>() {
        return ollama_keep_alive_from_seconds(seconds);
    }

    let (negative, mut remaining) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        _ => (false, value),
    };
    if remaining.is_empty() {
        return Err("keep_alive duration string is invalid".to_string());
    }

    let mut total_seconds = 0.0_f64;
    let mut components = 0_usize;
    while !remaining.is_empty() {
        let number_end = remaining
            .char_indices()
            .take_while(|(_, character)| character.is_ascii_digit() || *character == '.')
            .map(|(index, character)| index + character.len_utf8())
            .last()
            .ok_or_else(|| "keep_alive duration component requires a number".to_string())?;
        let number = remaining[..number_end]
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite() && *number >= 0.0)
            .ok_or_else(|| "keep_alive duration contains an invalid number".to_string())?;
        remaining = &remaining[number_end..];
        let (unit, unit_seconds) = [
            ("ns", 0.000_000_001_f64),
            ("us", 0.000_001_f64),
            ("µs", 0.000_001_f64),
            ("μs", 0.000_001_f64),
            ("ms", 0.001_f64),
            ("s", 1.0_f64),
            ("m", 60.0_f64),
            ("h", 3_600.0_f64),
        ]
        .into_iter()
        .find(|(unit, _)| remaining.starts_with(unit))
        .ok_or_else(|| "keep_alive duration contains an unsupported unit".to_string())?;
        remaining = &remaining[unit.len()..];
        total_seconds += number * unit_seconds;
        if !total_seconds.is_finite() {
            return Err("keep_alive duration is out of range".to_string());
        }
        components = components.saturating_add(1);
        if components > 32 {
            return Err("keep_alive duration contains too many components".to_string());
        }
    }
    if negative && total_seconds > 0.0 {
        return Ok(OllamaKeepAlive::Indefinite);
    }
    ollama_keep_alive_from_seconds(total_seconds)
}

fn validate_ollama_embedding_controls(
    options: Option<&serde_json::Value>,
    keep_alive: Option<&serde_json::Value>,
) -> std::result::Result<(), String> {
    parse_ollama_keep_alive(keep_alive)?;
    let Some(options) = options.filter(|options| !options.is_null()) else {
        return Ok(());
    };
    let options = options
        .as_object()
        .ok_or_else(|| "options must be an object".to_string())?;
    if let Some(field) = options
        .iter()
        .find(|(_, value)| !value.is_null())
        .map(|(field, _)| reported_extension_field(field))
    {
        return Err(format!(
            "embedding options contain unsupported non-neutral field {field:?}"
        ));
    }
    Ok(())
}

async fn ollama_from_chat_response(
    response: axum::response::Response,
    kind: OllamaOutputKind,
    stream: bool,
    started: Instant,
    requested_model: String,
    residency_lease: Option<OllamaResidencyLease>,
) -> axum::response::Response {
    if !response.status().is_success() {
        return adapt_ollama_error_response(response).await;
    }
    if stream {
        return ollama_stream_from_chat(response, kind, started, requested_model, residency_lease);
    }
    let body = match axum::body::to_bytes(response.into_body(), MAX_OLLAMA_ADAPTER_BODY_BYTES).await
    {
        Ok(body) => body,
        Err(_) => {
            return ollama_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "failed to read the bounded internal generation response",
            );
        }
    };
    let mut chat = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(chat) => chat,
        Err(_) => {
            return ollama_error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "the internal generation response was invalid",
            );
        }
    };
    chat["model"] = json!(requested_model);
    match ollama_payload_from_chat(&chat, kind, started.elapsed()) {
        Ok(payload) => ollama_json(payload),
        Err(message) => {
            ollama_error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, message)
        }
    }
}

async fn ollama_from_multimodal_response(
    response: axum::response::Response,
    kind: OllamaOutputKind,
    stream: bool,
    started: Instant,
    requested_model: String,
    residency_lease: Option<OllamaResidencyLease>,
) -> axum::response::Response {
    if !response.status().is_success() {
        return adapt_ollama_error_response(response).await;
    }
    if !ollama_internal_response_is_sse(&response) {
        return ollama_error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the internal multimodal response did not use SSE",
        );
    }
    if stream {
        return ollama_stream_from_multimodal(
            response,
            kind,
            started,
            requested_model,
            residency_lease,
        );
    }

    let _residency_lease = residency_lease;
    let mut body = response.into_body().into_data_stream();
    let mut decoder = ChatSseDecoder::default();
    let mut state = OllamaMultimodalStreamState::new(kind, started, requested_model);
    let mut transport_bytes = 0_usize;
    while let Some(chunk) = body.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(_) => {
                return ollama_error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "the internal multimodal stream body failed",
                );
            }
        };
        transport_bytes = match transport_bytes.checked_add(chunk.len()) {
            Some(bytes) if bytes <= MAX_OLLAMA_STREAM_BYTES => bytes,
            _ => {
                return ollama_error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "the internal multimodal stream exceeded its byte limit",
                );
            }
        };
        let frames = match decoder.push(&chunk) {
            Ok(frames) => frames,
            Err(message) => {
                return ollama_error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                );
            }
        };
        let frame_count = frames.len();
        for (index, frame) in frames.into_iter().enumerate() {
            if frame == "[DONE]" {
                if index + 1 != frame_count || decoder.finish().is_err() {
                    return ollama_error_response(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "the internal multimodal stream ended with invalid framing",
                    );
                }
                return match state.finish(true) {
                    Ok(payload) => ollama_json(payload),
                    Err(message) => ollama_error_response(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        message,
                    ),
                };
            }
            let payload = match serde_json::from_str::<serde_json::Value>(&frame) {
                Ok(payload) => payload,
                Err(_) => {
                    return ollama_error_response(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "the internal multimodal stream emitted invalid JSON",
                    );
                }
            };
            if let Err(message) = state.ingest(payload, true) {
                return ollama_error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                );
            }
        }
    }
    ollama_error_response(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "the internal multimodal stream ended before its terminal marker",
    )
}

fn ollama_internal_response_is_sse(response: &axum::response::Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
}

fn ollama_stream_from_multimodal(
    response: axum::response::Response,
    kind: OllamaOutputKind,
    started: Instant,
    requested_model: String,
    residency_lease: Option<OllamaResidencyLease>,
) -> axum::response::Response {
    let mut body = response.into_body().into_data_stream();
    let (tx, rx) = mpsc::channel::<std::result::Result<axum::body::Bytes, Infallible>>(32);
    task::spawn(async move {
        let _residency_lease = residency_lease;
        let mut decoder = ChatSseDecoder::default();
        let mut state = OllamaMultimodalStreamState::new(kind, started, requested_model);
        let mut transport_bytes = 0_usize;
        loop {
            let chunk = tokio::select! {
                _ = tx.closed() => return,
                chunk = body.next() => chunk,
            };
            let Some(chunk) = chunk else {
                send_ollama_stream_error(
                    &tx,
                    "the internal multimodal stream ended before its terminal marker",
                )
                .await;
                return;
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) => {
                    send_ollama_stream_error(&tx, "the internal multimodal stream body failed")
                        .await;
                    return;
                }
            };
            transport_bytes = match transport_bytes.checked_add(chunk.len()) {
                Some(bytes) if bytes <= MAX_OLLAMA_STREAM_BYTES => bytes,
                _ => {
                    send_ollama_stream_error(
                        &tx,
                        "the internal multimodal stream exceeded its byte limit",
                    )
                    .await;
                    return;
                }
            };
            let frames = match decoder.push(&chunk) {
                Ok(frames) => frames,
                Err(message) => {
                    send_ollama_stream_error(&tx, &message).await;
                    return;
                }
            };
            let frame_count = frames.len();
            for (index, frame) in frames.into_iter().enumerate() {
                if frame == "[DONE]" {
                    if index + 1 != frame_count || decoder.finish().is_err() {
                        send_ollama_stream_error(
                            &tx,
                            "the internal multimodal stream ended with invalid framing",
                        )
                        .await;
                        return;
                    }
                    match state.finish(false) {
                        Ok(payload) => {
                            let _ = send_ollama_line(&tx, payload).await;
                        }
                        Err(message) => send_ollama_stream_error(&tx, &message).await,
                    }
                    return;
                }
                let payload = match serde_json::from_str::<serde_json::Value>(&frame) {
                    Ok(payload) => payload,
                    Err(_) => {
                        send_ollama_stream_error(
                            &tx,
                            "the internal multimodal stream emitted invalid JSON",
                        )
                        .await;
                        return;
                    }
                };
                match state.ingest(payload, false) {
                    Ok(Some(payload)) => {
                        if !send_ollama_line(&tx, payload).await {
                            return;
                        }
                    }
                    Ok(None) => {}
                    Err(message) => {
                        send_ollama_stream_error(&tx, &message).await;
                        return;
                    }
                }
            }
        }
    });
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static(OLLAMA_CONTENT_TYPE),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        axum::body::Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response()
}

struct OllamaMultimodalStreamState {
    kind: OllamaOutputKind,
    started: Instant,
    requested_model: String,
    id: Option<String>,
    internal_model: Option<String>,
    created: Option<u64>,
    saw_start: bool,
    saw_end: bool,
    events: usize,
    output_bytes: usize,
    buffered: String,
}

impl OllamaMultimodalStreamState {
    fn new(kind: OllamaOutputKind, started: Instant, requested_model: String) -> Self {
        Self {
            kind,
            started,
            requested_model,
            id: None,
            internal_model: None,
            created: None,
            saw_start: false,
            saw_end: false,
            events: 0,
            output_bytes: 0,
            buffered: String::new(),
        }
    }

    fn ingest(
        &mut self,
        payload: serde_json::Value,
        collect: bool,
    ) -> std::result::Result<Option<serde_json::Value>, String> {
        self.events = self.events.saturating_add(1);
        if self.events > MAX_OLLAMA_STREAM_EVENTS {
            return Err("the internal multimodal stream emitted too many events".to_string());
        }
        if let Some(error) = payload.get("error") {
            return Err(error
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("the local multimodal generation stream failed")
                .to_string());
        }
        let object = payload
            .as_object()
            .ok_or_else(|| "the internal multimodal stream emitted a non-object".to_string())?;
        reject_ollama_object_fields(
            object,
            &["id", "object", "created", "model", "chunk"],
            "internal multimodal stream event",
        )?;
        if payload.get("object").and_then(serde_json::Value::as_str) != Some("multimodal.chunk") {
            return Err(
                "the internal multimodal stream used an unexpected object type".to_string(),
            );
        }
        self.validate_identity(&payload)?;
        let chunk = payload
            .get("chunk")
            .ok_or_else(|| "the internal multimodal stream omitted its chunk".to_string())?;
        if chunk.is_null() {
            if self.saw_start || self.saw_end || self.output_bytes != 0 {
                return Err(
                    "the internal multimodal stream emitted an invalid start event".to_string(),
                );
            }
            self.saw_start = true;
            return Ok(None);
        }
        if !self.saw_start {
            return Err(
                "the internal multimodal stream emitted output before its start event".to_string(),
            );
        }
        if self.saw_end {
            return Err(
                "the internal multimodal stream emitted output after its end event".to_string(),
            );
        }
        if chunk.as_str() == Some("End") {
            self.saw_end = true;
            return Ok(None);
        }
        let chunk = chunk
            .as_object()
            .filter(|chunk| chunk.len() == 1)
            .ok_or_else(|| "the internal multimodal stream emitted an invalid chunk".to_string())?;
        if chunk.contains_key("Metrics") {
            if !chunk["Metrics"].is_object() {
                return Err("the internal multimodal stream emitted invalid metrics".to_string());
            }
            return Ok(None);
        }
        let text = if let Some(text) = chunk.get("TextDelta") {
            text.as_str().ok_or_else(|| {
                "the internal multimodal stream emitted a non-text delta".to_string()
            })?
        } else if let Some(token) = chunk.get("VlmToken") {
            let token = token.as_object().ok_or_else(|| {
                "the internal multimodal stream emitted an invalid VLM token".to_string()
            })?;
            reject_ollama_object_fields(
                token,
                &["text", "bounding_box"],
                "internal multimodal VLM token",
            )?;
            token
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "the internal multimodal stream omitted VLM token text".to_string()
                })?
        } else {
            return Err(
                "the internal multimodal stream emitted an unsupported output chunk".to_string(),
            );
        };
        self.output_bytes = self
            .output_bytes
            .checked_add(text.len())
            .filter(|bytes| *bytes <= MAX_OLLAMA_STREAM_BYTES)
            .ok_or_else(|| "the internal multimodal output exceeded its byte limit".to_string())?;
        if collect {
            self.buffered.push_str(text);
            return Ok(None);
        }
        let mut output = self.base_payload()?;
        match self.kind {
            OllamaOutputKind::Chat => {
                output["message"] = json!({"role": "assistant", "content": text});
            }
            OllamaOutputKind::Generate => output["response"] = json!(text),
        }
        Ok(Some(output))
    }

    fn validate_identity(
        &mut self,
        payload: &serde_json::Value,
    ) -> std::result::Result<(), String> {
        let id = payload
            .get("id")
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| "the internal multimodal stream omitted its id".to_string())?;
        let model = payload
            .get("model")
            .and_then(serde_json::Value::as_str)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| "the internal multimodal stream omitted its model".to_string())?;
        let created = payload
            .get("created")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                "the internal multimodal stream omitted its creation time".to_string()
            })?;
        if self.id.as_deref().is_some_and(|expected| expected != id)
            || self
                .internal_model
                .as_deref()
                .is_some_and(|expected| expected != model)
            || self.created.is_some_and(|expected| expected != created)
        {
            return Err("the internal multimodal stream changed identity".to_string());
        }
        self.id.get_or_insert_with(|| id.to_string());
        self.internal_model.get_or_insert_with(|| model.to_string());
        self.created.get_or_insert(created);
        Ok(())
    }

    fn base_payload(&self) -> std::result::Result<serde_json::Value, String> {
        let created = self.created.ok_or_else(|| {
            "the internal multimodal stream omitted its creation time".to_string()
        })?;
        Ok(ollama_base_payload(
            &self.requested_model,
            created,
            self.kind,
        ))
    }

    fn finish(&self, buffered: bool) -> std::result::Result<serde_json::Value, String> {
        if !self.saw_start || !self.saw_end {
            return Err(
                "the internal multimodal stream omitted its start or end event".to_string(),
            );
        }
        let mut payload = self.base_payload()?;
        if buffered {
            match self.kind {
                OllamaOutputKind::Chat => {
                    payload["message"] = json!({"role": "assistant", "content": self.buffered});
                }
                OllamaOutputKind::Generate => payload["response"] = json!(self.buffered),
            }
        }
        add_ollama_terminal_without_usage(&mut payload, self.started.elapsed())?;
        Ok(payload)
    }
}

pub(crate) fn ollama_payload_from_chat(
    chat: &serde_json::Value,
    kind: OllamaOutputKind,
    elapsed: Duration,
) -> std::result::Result<serde_json::Value, String> {
    if chat.get("object").and_then(serde_json::Value::as_str) != Some("chat.completion") {
        return Err("the internal response used an unexpected object type".to_string());
    }
    let model = chat
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "the internal response omitted its model".to_string())?;
    let created = chat
        .get("created")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "the internal response omitted its creation time".to_string())?;
    let choice = chat
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .filter(|choices| choices.len() == 1)
        .and_then(|choices| choices.first())
        .ok_or_else(|| "the internal response omitted its output choice".to_string())?;
    if choice.get("index").and_then(serde_json::Value::as_u64) != Some(0) {
        return Err("the internal response used an invalid choice index".to_string());
    }
    let finish_reason = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
        .filter(|reason| matches!(*reason, "stop" | "length" | "tool_calls"))
        .ok_or_else(|| "the internal response omitted its finish reason".to_string())?;
    let usage = chat
        .get("usage")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "the internal response omitted usage".to_string())?;
    let mut payload = ollama_base_payload(model, created, kind);
    match kind {
        OllamaOutputKind::Chat => {
            payload["message"] = ollama_message_from_openai(
                choice
                    .get("message")
                    .ok_or_else(|| "the internal response omitted its message".to_string())?,
            )?;
        }
        OllamaOutputKind::Generate => {
            if finish_reason == "tool_calls" {
                return Err("generate unexpectedly produced function calls".to_string());
            }
            let content = choice
                .pointer("/message/content")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "the internal response omitted generated text".to_string())?;
            payload["response"] = json!(content);
        }
    }
    add_ollama_terminal_fields(&mut payload, finish_reason, usage, elapsed)?;
    Ok(payload)
}

fn ollama_message_from_openai(
    message: &serde_json::Value,
) -> std::result::Result<serde_json::Value, String> {
    if message.get("role").and_then(serde_json::Value::as_str) != Some("assistant") {
        return Err("the internal response used a non-assistant role".to_string());
    }
    let content = message
        .get("content")
        .filter(|content| !content.is_null())
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let tool_calls = message
        .get("tool_calls")
        .filter(|calls| !calls.is_null())
        .map(ollama_tool_calls_from_openai)
        .transpose()?
        .unwrap_or_default();
    let mut result = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        result["tool_calls"] = serde_json::Value::Array(tool_calls);
    }
    Ok(result)
}

fn ollama_tool_calls_from_openai(
    value: &serde_json::Value,
) -> std::result::Result<Vec<serde_json::Value>, String> {
    let calls = value
        .as_array()
        .ok_or_else(|| "the internal response emitted invalid tool calls".to_string())?;
    if calls.len() > MAX_PARALLEL_TOOL_CALLS {
        return Err("the internal response emitted too many tool calls".to_string());
    }
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let name = call
                .pointer("/function/name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("internal tool call {index} omitted its name"))?;
            validate_tool_name(name, "internal tool-call name")?;
            let arguments = call
                .pointer("/function/arguments")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("internal tool call {index} omitted its arguments"))?;
            let arguments =
                serde_json::from_str::<serde_json::Value>(arguments).map_err(|error| {
                    format!("internal tool call {index} arguments were invalid JSON: {error}")
                })?;
            if !arguments.is_object() {
                return Err(format!(
                    "internal tool call {index} arguments were not an object"
                ));
            }
            Ok(json!({
                "type": "function",
                "function": {"index": index, "name": name, "arguments": arguments}
            }))
        })
        .collect()
}

fn ollama_stream_from_chat(
    response: axum::response::Response,
    kind: OllamaOutputKind,
    started: Instant,
    requested_model: String,
    residency_lease: Option<OllamaResidencyLease>,
) -> axum::response::Response {
    let is_sse = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"));
    if !is_sse {
        return ollama_error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "the internal streaming response did not use SSE",
        );
    }
    let mut body = response.into_body().into_data_stream();
    let (tx, rx) = mpsc::channel::<std::result::Result<axum::body::Bytes, Infallible>>(32);
    task::spawn(async move {
        let _residency_lease = residency_lease;
        let mut decoder = ChatSseDecoder::default();
        let mut state = OllamaStreamState::new_for_model(kind, started, requested_model);
        loop {
            let chunk = tokio::select! {
                _ = tx.closed() => return,
                chunk = body.next() => chunk,
            };
            let Some(chunk) = chunk else {
                send_ollama_stream_error(
                    &tx,
                    "the internal stream ended before its terminal marker",
                )
                .await;
                return;
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) => {
                    send_ollama_stream_error(&tx, "the internal stream body failed").await;
                    return;
                }
            };
            let frames = match decoder.push(&chunk) {
                Ok(frames) => frames,
                Err(message) => {
                    send_ollama_stream_error(&tx, &message).await;
                    return;
                }
            };
            for frame in frames {
                if frame == "[DONE]" {
                    if decoder.finish().is_err() {
                        send_ollama_stream_error(
                            &tx,
                            "the internal stream ended with incomplete framing",
                        )
                        .await;
                        return;
                    }
                    match state.finish() {
                        Ok(payload) => {
                            let _ = send_ollama_line(&tx, payload).await;
                        }
                        Err(message) => send_ollama_stream_error(&tx, &message).await,
                    }
                    return;
                }
                let payload = match serde_json::from_str::<serde_json::Value>(&frame) {
                    Ok(payload) => payload,
                    Err(_) => {
                        send_ollama_stream_error(&tx, "the internal stream emitted invalid JSON")
                            .await;
                        return;
                    }
                };
                match state.ingest(payload) {
                    Ok(Some(payload)) => {
                        if !send_ollama_line(&tx, payload).await {
                            return;
                        }
                    }
                    Ok(None) => {}
                    Err(message) => {
                        send_ollama_stream_error(&tx, &message).await;
                        return;
                    }
                }
            }
        }
    });
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static(OLLAMA_CONTENT_TYPE),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        axum::body::Body::from_stream(ReceiverStream::new(rx)),
    )
        .into_response()
}

struct OllamaStreamState {
    kind: OllamaOutputKind,
    started: Instant,
    requested_model: Option<String>,
    id: Option<String>,
    model: Option<String>,
    created: Option<u64>,
    finish_reason: Option<String>,
    usage: Option<serde_json::Value>,
    events: usize,
    output_bytes: usize,
}

impl OllamaStreamState {
    fn new(kind: OllamaOutputKind, started: Instant) -> Self {
        Self {
            kind,
            started,
            requested_model: None,
            id: None,
            model: None,
            created: None,
            finish_reason: None,
            usage: None,
            events: 0,
            output_bytes: 0,
        }
    }

    fn new_for_model(kind: OllamaOutputKind, started: Instant, requested_model: String) -> Self {
        let mut state = Self::new(kind, started);
        state.requested_model = Some(requested_model);
        state
    }

    fn ingest(
        &mut self,
        payload: serde_json::Value,
    ) -> std::result::Result<Option<serde_json::Value>, String> {
        if let Some(error) = payload.get("error") {
            return Err(error
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("the local generation stream failed")
                .to_string());
        }
        if payload.get("object").and_then(serde_json::Value::as_str)
            != Some("chat.completion.chunk")
        {
            return Err("the internal stream emitted an unexpected object type".to_string());
        }
        self.validate_identity(&payload)?;
        let choices = payload
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "the internal stream omitted its choices".to_string())?;
        if choices.is_empty() {
            if self.usage.is_some() {
                return Err("the internal stream emitted duplicate usage".to_string());
            }
            if self.finish_reason.is_none() {
                return Err(
                    "the internal stream emitted usage before its finish reason".to_string()
                );
            }
            self.usage = Some(
                payload
                    .get("usage")
                    .filter(|usage| usage.is_object())
                    .cloned()
                    .ok_or_else(|| "the internal stream omitted terminal usage".to_string())?,
            );
            return Ok(None);
        }
        if choices.len() != 1 {
            return Err("the internal stream emitted multiple choices".to_string());
        }
        if self.finish_reason.is_some() || self.usage.is_some() {
            return Err("the internal stream emitted output after its finish reason".to_string());
        }
        let choice = &choices[0];
        if choice.get("index").and_then(serde_json::Value::as_u64) != Some(0) {
            return Err("the internal stream emitted an invalid choice index".to_string());
        }
        if let Some(finish_reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
            let finish_reason = finish_reason
                .as_str()
                .filter(|reason| matches!(*reason, "stop" | "length" | "tool_calls"))
                .ok_or_else(|| {
                    "the internal stream emitted an invalid finish reason".to_string()
                })?;
            if self
                .finish_reason
                .replace(finish_reason.to_string())
                .is_some()
            {
                return Err("the internal stream emitted duplicate finish reasons".to_string());
            }
        }
        let delta = choice
            .get("delta")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| "the internal stream omitted its delta object".to_string())?;
        reject_ollama_object_fields(
            delta,
            &["role", "content", "tool_calls"],
            "internal stream delta",
        )?;
        if delta
            .get("role")
            .filter(|role| !role.is_null())
            .is_some_and(|role| role.as_str() != Some("assistant"))
        {
            return Err("the internal stream emitted a non-assistant role".to_string());
        }
        let mut output = None;
        if let Some(content) = delta.get("content") {
            let content = content
                .as_str()
                .ok_or_else(|| "the internal stream emitted non-text content".to_string())?;
            if !content.is_empty() {
                self.record_output(content.len())?;
                let mut payload = self.base_payload()?;
                match self.kind {
                    OllamaOutputKind::Chat => {
                        payload["message"] = json!({"role": "assistant", "content": content});
                    }
                    OllamaOutputKind::Generate => payload["response"] = json!(content),
                }
                output = Some(payload);
            }
        }
        if let Some(tool_calls) = delta.get("tool_calls") {
            if !matches!(self.kind, OllamaOutputKind::Chat) || output.is_some() {
                return Err("the internal stream mixed incompatible output types".to_string());
            }
            let tool_calls = ollama_stream_tool_calls_from_openai(tool_calls)?;
            let bytes = serde_json::to_vec(&tool_calls)
                .map_err(|error| format!("failed to encode streamed tool calls: {error}"))?
                .len();
            self.record_output(bytes)?;
            let mut payload = self.base_payload()?;
            payload["message"] = json!({
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls
            });
            output = Some(payload);
        }
        Ok(output)
    }

    fn finish(&mut self) -> std::result::Result<serde_json::Value, String> {
        let finish_reason = self
            .finish_reason
            .clone()
            .ok_or_else(|| "the internal stream omitted its finish reason".to_string())?;
        let usage = self
            .usage
            .as_ref()
            .and_then(serde_json::Value::as_object)
            .cloned()
            .ok_or_else(|| "the internal stream omitted usage".to_string())?;
        let mut payload = self.base_payload()?;
        match self.kind {
            OllamaOutputKind::Chat => {
                payload["message"] = json!({"role": "assistant", "content": ""});
            }
            OllamaOutputKind::Generate => payload["response"] = json!(""),
        }
        add_ollama_terminal_fields(&mut payload, &finish_reason, &usage, self.started.elapsed())?;
        Ok(payload)
    }

    fn validate_identity(
        &mut self,
        payload: &serde_json::Value,
    ) -> std::result::Result<(), String> {
        let id = payload
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "the internal stream omitted its ID".to_string())?;
        let model = payload
            .get("model")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "the internal stream omitted its model".to_string())?;
        let created = payload
            .get("created")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "the internal stream omitted its creation time".to_string())?;
        if let (Some(expected_id), Some(expected_model), Some(expected_created)) =
            (&self.id, &self.model, self.created)
        {
            if expected_id != id || expected_model != model || expected_created != created {
                return Err("the internal stream changed its identity metadata".to_string());
            }
        } else {
            self.id = Some(id.to_string());
            self.model = Some(model.to_string());
            self.created = Some(created);
        }
        Ok(())
    }

    fn base_payload(&mut self) -> std::result::Result<serde_json::Value, String> {
        self.events = self.events.saturating_add(1);
        if self.events > MAX_OLLAMA_STREAM_EVENTS {
            return Err("the Ollama stream exceeded its event limit".to_string());
        }
        Ok(ollama_base_payload(
            self.requested_model
                .as_deref()
                .or(self.model.as_deref())
                .ok_or_else(|| "the internal stream omitted its model".to_string())?,
            self.created
                .ok_or_else(|| "the internal stream omitted its creation time".to_string())?,
            self.kind,
        ))
    }

    fn record_output(&mut self, bytes: usize) -> std::result::Result<(), String> {
        self.output_bytes = self.output_bytes.saturating_add(bytes);
        if self.output_bytes > MAX_OLLAMA_STREAM_BYTES {
            return Err("the Ollama stream exceeded its output byte limit".to_string());
        }
        Ok(())
    }
}

fn ollama_stream_tool_calls_from_openai(
    value: &serde_json::Value,
) -> std::result::Result<Vec<serde_json::Value>, String> {
    let calls = value
        .as_array()
        .ok_or_else(|| "the internal stream emitted invalid tool_calls".to_string())?;
    if calls.is_empty() || calls.len() > MAX_PARALLEL_TOOL_CALLS {
        return Err(format!(
            "the internal stream must emit between 1 and {MAX_PARALLEL_TOOL_CALLS} tool calls"
        ));
    }
    calls
        .iter()
        .enumerate()
        .map(|(expected_index, call)| {
            let index = call
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("streamed tool call {expected_index} omitted its index"))?;
            if index != expected_index as u64 {
                return Err(format!(
                    "streamed tool call {expected_index} changed its index"
                ));
            }
            let name = call
                .pointer("/function/name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("streamed tool call {expected_index} omitted its name"))?;
            validate_tool_name(name, "streamed tool-call name")?;
            let arguments = call
                .pointer("/function/arguments")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    format!("streamed tool call {expected_index} omitted its arguments")
                })?;
            let arguments =
                serde_json::from_str::<serde_json::Value>(arguments).map_err(|error| {
                    format!("streamed tool call {expected_index} arguments were invalid: {error}")
                })?;
            if !arguments.is_object() {
                return Err(format!(
                    "streamed tool call {expected_index} arguments were not an object"
                ));
            }
            Ok(json!({
                "type": "function",
                "function": {"index": index, "name": name, "arguments": arguments}
            }))
        })
        .collect()
}

fn ollama_base_payload(model: &str, created: u64, kind: OllamaOutputKind) -> serde_json::Value {
    match kind {
        OllamaOutputKind::Chat => json!({
            "model": model,
            "created_at": iso8601_from_unix(created),
            "message": {"role": "assistant", "content": ""},
            "done": false
        }),
        OllamaOutputKind::Generate => json!({
            "model": model,
            "created_at": iso8601_from_unix(created),
            "response": "",
            "done": false
        }),
    }
}

fn add_ollama_terminal_fields(
    payload: &mut serde_json::Value,
    finish_reason: &str,
    usage: &serde_json::Map<String, serde_json::Value>,
    elapsed: Duration,
) -> std::result::Result<(), String> {
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "the internal response omitted prompt token usage".to_string())?;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "the internal response omitted completion token usage".to_string())?;
    let expected_total = prompt_tokens
        .checked_add(completion_tokens)
        .ok_or_else(|| "the internal response token usage overflowed".to_string())?;
    if usage
        .get("total_tokens")
        .and_then(serde_json::Value::as_u64)
        != Some(expected_total)
    {
        return Err("the internal response reported inconsistent token usage".to_string());
    }
    let duration = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    let object = payload
        .as_object_mut()
        .ok_or_else(|| "the Ollama adapter constructed an invalid payload".to_string())?;
    object.insert("done".to_string(), json!(true));
    object.insert(
        "done_reason".to_string(),
        json!(if finish_reason == "tool_calls" {
            "stop"
        } else {
            finish_reason
        }),
    );
    object.insert("total_duration".to_string(), json!(duration));
    object.insert("load_duration".to_string(), json!(0));
    object.insert("prompt_eval_count".to_string(), json!(prompt_tokens));
    object.insert("prompt_eval_duration".to_string(), json!(0));
    object.insert("eval_count".to_string(), json!(completion_tokens));
    object.insert("eval_duration".to_string(), json!(duration));
    Ok(())
}

fn add_ollama_terminal_without_usage(
    payload: &mut serde_json::Value,
    elapsed: Duration,
) -> std::result::Result<(), String> {
    let duration = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    let object = payload
        .as_object_mut()
        .ok_or_else(|| "the Ollama adapter constructed an invalid payload".to_string())?;
    object.insert("done".to_string(), json!(true));
    object.insert("done_reason".to_string(), json!("stop"));
    object.insert("total_duration".to_string(), json!(duration));
    Ok(())
}

async fn adapt_ollama_error_response(
    response: axum::response::Response,
) -> axum::response::Response {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), MAX_OLLAMA_ADAPTER_BODY_BYTES).await;
    let message = body
        .ok()
        .and_then(|body| serde_json::from_slice::<serde_json::Value>(&body).ok())
        .and_then(|body| {
            body.pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "the local request failed".to_string());
    ollama_error_response(status, message)
}

fn ollama_json(payload: serde_json::Value) -> axum::response::Response {
    (
        [(header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
        Json(payload),
    )
        .into_response()
}

fn ollama_bad_request(message: impl AsRef<str>) -> axum::response::Response {
    ollama_error_response(axum::http::StatusCode::BAD_REQUEST, message)
}

fn ollama_json_rejection_response(
    error: axum::extract::rejection::JsonRejection,
) -> axum::response::Response {
    if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        ollama_error_response(
            axum::http::StatusCode::PAYLOAD_TOO_LARGE,
            "request body exceeds the configured size limit",
        )
    } else {
        ollama_bad_request("invalid JSON request body")
    }
}

pub(crate) fn ollama_error_response(
    status: axum::http::StatusCode,
    message: impl AsRef<str>,
) -> axum::response::Response {
    let message = bounded_ollama_error(message.as_ref());
    (
        status,
        [(header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
        Json(json!({"error": message})),
    )
        .into_response()
}

async fn send_ollama_line(
    tx: &mpsc::Sender<std::result::Result<axum::body::Bytes, Infallible>>,
    payload: serde_json::Value,
) -> bool {
    let mut encoded = match serde_json::to_vec(&payload) {
        Ok(encoded) => encoded,
        Err(_) => return false,
    };
    encoded.push(b'\n');
    tx.send(Ok(axum::body::Bytes::from(encoded))).await.is_ok()
}

async fn send_ollama_stream_error(
    tx: &mpsc::Sender<std::result::Result<axum::body::Bytes, Infallible>>,
    message: &str,
) {
    let _ = send_ollama_line(tx, json!({"error": bounded_ollama_error(message)})).await;
}

fn bounded_ollama_error(message: &str) -> String {
    let mut bounded = String::new();
    for character in message.chars() {
        let character = if character.is_control() {
            ' '
        } else {
            character
        };
        if bounded.len().saturating_add(character.len_utf8()) > 4 * 1024 {
            break;
        }
        bounded.push(character);
    }
    if bounded.trim().is_empty() {
        "the local request failed".to_string()
    } else {
        bounded
    }
}

fn reject_ollama_extensions(
    scope: &str,
    extensions: &BTreeMap<String, serde_json::Value>,
) -> std::result::Result<(), String> {
    if let Some(field) = extensions
        .iter()
        .find(|(_, value)| !value.is_null())
        .map(|(field, _)| field)
    {
        return Err(format!(
            "{scope} contains unsupported field {:?}",
            reported_extension_field(field)
        ));
    }
    Ok(())
}

fn reject_ollama_object_fields(
    object: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
    scope: &str,
) -> std::result::Result<(), String> {
    if let Some(field) = object
        .iter()
        .find(|(field, value)| !value.is_null() && !allowed.contains(&field.as_str()))
        .map(|(field, _)| field)
    {
        return Err(format!(
            "{scope} contains unsupported field {:?}",
            reported_extension_field(field)
        ));
    }
    Ok(())
}

fn ollama_catalog_model(
    entry: &model_manager::ModelCatalogEntry,
    runtime: Option<&Arc<LoadedRuntime>>,
) -> serde_json::Value {
    let family = runtime
        .map(|runtime| model_family_name(&runtime.model_family))
        .unwrap_or("unknown");
    let selector = ollama_catalog_selector(entry);
    json!({
        "name": selector,
        "model": selector,
        "modified_at": iso8601_from_unix(entry.modified_at.unwrap_or_default()),
        "size": entry.size_bytes,
        "digest": entry.provenance.as_ref().map(|value| value.sha256.as_str()).unwrap_or(""),
        "details": ollama_details(
            &entry.format,
            family,
            runtime.and_then(|runtime| runtime_quantization(runtime))
                .or_else(|| quantization_from_name(&entry.id)),
        )
    })
}

fn ollama_runtime_model(runtime: &Arc<LoadedRuntime>) -> serde_json::Value {
    json!({
        "name": runtime.model_id,
        "model": runtime.model_id,
        "modified_at": iso8601_from_unix(runtime_modified_at(runtime).unwrap_or_default()),
        "size": runtime.memory_estimate.weight_bytes,
        "digest": runtime_digest(None),
        "details": ollama_details(
            runtime_format(runtime),
            model_family_name(&runtime.model_family),
            runtime_quantization(runtime),
        )
    })
}

fn ollama_details(format: &str, family: &str, quantization: Option<&str>) -> serde_json::Value {
    json!({
        "parent_model": "",
        "format": format,
        "family": family,
        "families": if family == "unknown" { Vec::<&str>::new() } else { vec![family] },
        "parameter_size": "",
        "quantization_level": quantization.unwrap_or("")
    })
}

fn runtime_digest(entry: Option<&model_manager::ModelCatalogEntry>) -> String {
    entry
        .and_then(|entry| entry.provenance.as_ref())
        .map(|provenance| provenance.sha256.clone())
        .unwrap_or_default()
}

fn resolve_resident_runtime(
    catalog: &model_manager::ModelCatalog,
    resident: &crate::runtime_pool::RuntimePoolSnapshot<LoadedRuntime>,
    selector: &str,
    catalog_entry: Option<&model_manager::ModelCatalogEntry>,
) -> std::result::Result<Option<Arc<LoadedRuntime>>, ()> {
    let mut matched = None;
    for (_, runtime) in resident.entries() {
        let matches = runtime.model_id == selector
            || runtime.catalog_id.as_deref() == Some(selector)
            || catalog_entry
                .is_some_and(|entry| catalog_entry_matches_runtime(catalog, entry, runtime));
        if !matches {
            continue;
        }
        if matched
            .as_ref()
            .is_some_and(|previous| !Arc::ptr_eq(previous, runtime))
        {
            return Err(());
        }
        matched = Some(Arc::clone(runtime));
    }
    Ok(matched)
}

fn catalog_entry_matches_runtime(
    catalog: &model_manager::ModelCatalog,
    entry: &model_manager::ModelCatalogEntry,
    runtime: &LoadedRuntime,
) -> bool {
    if runtime.catalog_id.as_deref() == Some(entry.id.as_str()) {
        return true;
    }
    if !entry.active {
        return false;
    }
    let catalog_path = Path::new(&catalog.root).join(&entry.id);
    runtime.source_path == catalog_path
        || runtime
            .source_path
            .canonicalize()
            .is_ok_and(|source| source == catalog_path)
}

fn ollama_catalog_selector(entry: &model_manager::ModelCatalogEntry) -> &str {
    entry
        .provenance
        .as_ref()
        .and_then(|provenance| provenance.model_index_id.as_deref())
        .unwrap_or(entry.id.as_str())
}

fn ollama_catalog_entry_matches_selector(
    entry: &model_manager::ModelCatalogEntry,
    selector: &str,
) -> bool {
    entry.id == selector || ollama_catalog_selector(entry) == selector
}

fn runtime_format(runtime: &LoadedRuntime) -> &str {
    runtime
        .source_path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| match extension.to_ascii_lowercase().as_str() {
            "gguf" => "gguf",
            "onnx" => "onnx",
            "mlmodel" | "mlpackage" => "coreml",
            _ => "directory",
        })
        .unwrap_or("directory")
}

fn runtime_modified_at(runtime: &LoadedRuntime) -> Option<u64> {
    std::fs::metadata(&runtime.source_path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn runtime_quantization(runtime: &LoadedRuntime) -> Option<&str> {
    match runtime.memory_estimate.quantization.as_ref()?.scheme {
        bloomai_core::QuantScheme::GGUF(ref scheme) => Some(scheme.as_str()),
        bloomai_core::QuantScheme::INT4 => Some("Q4"),
        bloomai_core::QuantScheme::INT8 => Some("Q8"),
        bloomai_core::QuantScheme::GPTQ => Some("GPTQ"),
        bloomai_core::QuantScheme::AWQ => Some("AWQ"),
        bloomai_core::QuantScheme::NF4 => Some("NF4"),
        bloomai_core::QuantScheme::None => None,
    }
}

fn quantization_from_name(name: &str) -> Option<&'static str> {
    let upper = name.to_ascii_uppercase();
    [
        "Q2_K", "Q3_K", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
    ]
    .into_iter()
    .find(|quantization| upper.contains(quantization))
}

fn model_family_name(family: &bloomai_core::ModelFamily) -> &str {
    match family {
        bloomai_core::ModelFamily::Llama => "llama",
        bloomai_core::ModelFamily::Qwen => "qwen",
        bloomai_core::ModelFamily::Gemma => "gemma",
        bloomai_core::ModelFamily::Bert => "bert",
        bloomai_core::ModelFamily::Whisper => "whisper",
        bloomai_core::ModelFamily::FunAsr => "funasr",
        bloomai_core::ModelFamily::Custom(_) => "custom",
    }
}

fn iso8601_from_unix(seconds: u64) -> String {
    let days = seconds / 86_400;
    let remainder = seconds % 86_400;
    let hours = remainder / 3_600;
    let minutes = (remainder % 3_600) / 60;
    let seconds = remainder % 60;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let day_of_year = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = year + u64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder as _;

    fn test_png_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(&[0, 0, 0, 255], 1, 1, image::ColorType::Rgba8)
            .unwrap();
        bytes
    }

    fn test_jpeg_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut bytes)
            .encode(&[0, 0, 0], 1, 1, image::ColorType::Rgb8)
            .unwrap();
        bytes
    }

    #[test]
    fn signed_acquisition_alias_is_the_stable_ollama_catalog_identity() {
        let entry = model_manager::ModelCatalogEntry {
            id: "tiny-q4.gguf".to_string(),
            name: "Tiny Q4".to_string(),
            kind: "file".to_string(),
            format: "gguf".to_string(),
            size_bytes: 42,
            size_complete: true,
            modified_at: Some(1),
            active: false,
            provenance: Some(model_provenance::ModelProvenance {
                acquisition: model_provenance::ModelAcquisitionKind::Download,
                model_index_id: Some("tiny-q4".to_string()),
                source_url: None,
                source_host: Some("huggingface.co".to_string()),
                sha256: "ab".repeat(32),
                file_count: None,
                license: Some("Apache-2.0".to_string()),
                installed_at: 1,
                last_verified_at: None,
                integrity_mismatch_at: None,
            }),
            provenance_error: None,
        };

        assert_eq!(ollama_catalog_selector(&entry), "tiny-q4");
        assert!(ollama_catalog_entry_matches_selector(&entry, "tiny-q4"));
        assert!(ollama_catalog_entry_matches_selector(
            &entry,
            "tiny-q4.gguf"
        ));
        let payload = ollama_catalog_model(&entry, None);
        assert_eq!(payload["name"], "tiny-q4");
        assert_eq!(payload["model"], "tiny-q4");
    }

    #[test]
    fn timestamp_and_options_are_bounded() {
        assert_eq!(iso8601_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_from_unix(1_700_000_000), "2023-11-14T22:13:20Z");
        let options = parse_ollama_options(Some(&json!({
            "num_predict": 32,
            "temperature": 0.2,
            "top_p": 0.8,
            "seed": 7,
            "stop": ["END"]
        })))
        .unwrap();
        assert_eq!(options.max_tokens, Some(32));
        assert_eq!(options.seed, Some(7));
        assert_eq!(options.stop, vec!["END"]);
        for invalid in [
            json!({"num_predict": 0}),
            json!({"seed": -2}),
            json!({"stop": [""]}),
            json!({"stop": ["a", "b", "c", "d", "e"]}),
            json!({"top_k": 40}),
        ] {
            assert!(parse_ollama_options(Some(&invalid)).is_err());
        }
    }

    #[test]
    fn ollama_images_accept_only_one_bounded_canonical_png_or_jpeg() {
        for (bytes, expected_mime) in [
            (test_png_bytes(), "image/png"),
            (test_jpeg_bytes(), "image/jpeg"),
        ] {
            let encoded = STANDARD.encode(&bytes);
            let image = parse_ollama_images(Some(&json!([encoded])), "test")
                .unwrap()
                .unwrap();
            assert_eq!(image.bytes, bytes);
            assert_eq!(image.mime, expected_mime);
        }
        assert!(parse_ollama_images(None, "test").unwrap().is_none());
        assert!(
            parse_ollama_images(Some(&serde_json::Value::Null), "test")
                .unwrap()
                .is_none()
        );
        assert!(
            parse_ollama_images(Some(&json!([])), "test")
                .unwrap()
                .is_none()
        );
        for invalid in [
            json!({}),
            json!("not-an-array"),
            json!([1]),
            json!(["", "extra"]),
            json!(["aGVs bG8="]),
            json!(["data:image/png;base64,aGVsbG8="]),
            json!([STANDARD.encode(b"not an image")]),
        ] {
            assert!(parse_ollama_images(Some(&invalid), "test").is_err());
        }
        let padded = STANDARD.encode(test_png_bytes());
        let unpadded = padded.trim_end_matches('=');
        assert_ne!(unpadded, padded);
        assert!(parse_ollama_images(Some(&json!([unpadded])), "test").is_err());

        let mut at_limit = test_png_bytes();
        at_limit.resize(MAX_MULTIMODAL_IMAGE_BYTES, 0);
        let encoded = STANDARD.encode(&at_limit);
        let parsed = parse_ollama_images(Some(&json!([encoded])), "test")
            .unwrap()
            .unwrap();
        assert_eq!(parsed.bytes.len(), MAX_MULTIMODAL_IMAGE_BYTES);

        at_limit.push(0);
        let encoded = STANDARD.encode(at_limit);
        assert!(parse_ollama_images(Some(&json!([encoded])), "test").is_err());
    }

    #[test]
    fn ollama_image_requests_are_single_turn_and_preserve_image_only_input() {
        let encoded = STANDARD.encode(test_png_bytes());
        for content in [None, Some(""), Some("   ")] {
            let mut message = json!({"role": "user", "images": [encoded.clone()]});
            if let Some(content) = content {
                message["content"] = json!(content);
            }
            let payload = serde_json::from_value::<OllamaChatRequest>(json!({
                "model": "vision",
                "messages": [message],
                "stream": false
            }))
            .unwrap();
            let (request, image) = ollama_chat_request(payload, false).unwrap();
            let prepared = prepare_ollama_image_request(&request, image.unwrap()).unwrap();
            let inference = prepared.into_inference_request();
            assert_eq!(inference.blocks.len(), 1);
            assert!(matches!(inference.blocks[0], DataBlock::Image { .. }));
        }

        let payload = serde_json::from_value::<OllamaGenerateRequest>(json!({
            "model": "vision",
            "images": [encoded.clone()],
            "stream": false
        }))
        .unwrap();
        let (request, image) = ollama_generate_request(payload, false).unwrap();
        let inference = prepare_ollama_image_request(&request, image.unwrap())
            .unwrap()
            .into_inference_request();
        assert_eq!(inference.blocks.len(), 1);
        assert!(matches!(inference.blocks[0], DataBlock::Image { .. }));

        let payload = serde_json::from_value::<OllamaChatRequest>(json!({
            "model": "vision",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second", "images": [encoded]}
            ]
        }))
        .unwrap();
        let (request, image) = ollama_chat_request(payload, true).unwrap();
        assert!(prepare_ollama_image_request(&request, image.unwrap()).is_err());
    }

    #[test]
    fn ollama_image_requests_reject_incompatible_controls_before_activation() {
        let encoded = STANDARD.encode(test_png_bytes());
        for extra in [
            json!({"tools": [{"type": "function"}]}),
            json!({"format": "json"}),
            json!({"options": {"stop": ["END"]}}),
        ] {
            let mut payload = json!({
                "model": "vision",
                "messages": [{
                    "role": "user",
                    "content": "describe",
                    "images": [encoded.clone()]
                }]
            });
            payload
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let payload = serde_json::from_value::<OllamaChatRequest>(payload).unwrap();
            let (request, image) = ollama_chat_request(payload, true).unwrap();
            assert!(prepare_ollama_image_request(&request, image.unwrap()).is_err());
        }
    }

    #[test]
    fn multimodal_stream_state_is_ordered_bounded_and_uses_the_requested_alias() {
        let event = |chunk: serde_json::Value| {
            json!({
                "id": "mms-1",
                "object": "multimodal.chunk",
                "created": 1_700_000_000_u64,
                "model": "internal-vision",
                "chunk": chunk
            })
        };
        let mut state = OllamaMultimodalStreamState::new(
            OllamaOutputKind::Chat,
            Instant::now(),
            "vision-alias".to_string(),
        );
        assert!(
            state
                .ingest(event(serde_json::Value::Null), true)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .ingest(event(json!({"TextDelta": "a"})), true)
                .unwrap()
                .is_none()
        );
        assert!(
            state
                .ingest(
                    event(json!({"VlmToken": {"text": "b", "bounding_box": null}})),
                    true
                )
                .unwrap()
                .is_none()
        );
        state
            .ingest(event(json!({"Metrics": {"compute_ms": 1}})), true)
            .unwrap();
        state.ingest(event(json!("End")), true).unwrap();
        let terminal = state.finish(true).unwrap();
        assert_eq!(terminal["model"], "vision-alias");
        assert_eq!(terminal["message"]["content"], "ab");
        assert_eq!(terminal["done"], true);
        assert!(terminal.get("eval_count").is_none());
        assert!(terminal.get("prompt_eval_count").is_none());

        let mut missing_start = OllamaMultimodalStreamState::new(
            OllamaOutputKind::Generate,
            Instant::now(),
            "vision".to_string(),
        );
        assert!(
            missing_start
                .ingest(event(json!({"TextDelta": "early"})), false)
                .is_err()
        );
        let mut missing_end = OllamaMultimodalStreamState::new(
            OllamaOutputKind::Generate,
            Instant::now(),
            "vision".to_string(),
        );
        missing_end
            .ingest(event(serde_json::Value::Null), false)
            .unwrap();
        assert!(missing_end.finish(false).is_err());
    }

    #[test]
    fn empty_generation_requests_map_to_bounded_model_lifecycle_actions() {
        let preload = serde_json::from_value::<OllamaGenerateRequest>(json!({
            "model": "tiny-q4",
            "prompt": "",
            "keep_alive": -1
        }))
        .unwrap();
        assert_eq!(
            ollama_generate_lifecycle_action(&preload).unwrap(),
            Some(OllamaLifecycleAction::Load(OllamaKeepAlive::Indefinite))
        );

        let unload = serde_json::from_value::<OllamaChatRequest>(json!({
            "model": "tiny-q4",
            "messages": [],
            "keep_alive": "0s"
        }))
        .unwrap();
        assert_eq!(
            ollama_chat_lifecycle_action(&unload).unwrap(),
            Some(OllamaLifecycleAction::Unload)
        );

        let generation = serde_json::from_value::<OllamaGenerateRequest>(json!({
            "model": "tiny-q4",
            "prompt": "Hello"
        }))
        .unwrap();
        assert_eq!(ollama_generate_lifecycle_action(&generation).unwrap(), None);
        assert_eq!(
            parse_ollama_keep_alive(None).unwrap(),
            OllamaKeepAlive::Timed(Duration::from_secs(300))
        );
        for (value, expected) in [
            (json!(300), Duration::from_secs(300)),
            (json!(0.25), Duration::from_millis(250)),
            (json!("5m"), Duration::from_secs(300)),
            (json!("1h30m"), Duration::from_secs(5_400)),
            (json!("250ms"), Duration::from_millis(250)),
        ] {
            assert_eq!(
                parse_ollama_keep_alive(Some(&value)).unwrap(),
                OllamaKeepAlive::Timed(expected)
            );
        }
        assert_eq!(
            parse_ollama_keep_alive(Some(&json!(0))).unwrap(),
            OllamaKeepAlive::Unload
        );
        assert_eq!(
            parse_ollama_keep_alive(Some(&json!("-1m"))).unwrap(),
            OllamaKeepAlive::Indefinite
        );
        for invalid in [json!("1d"), json!("1..2s"), json!("8761h"), json!({})] {
            assert!(parse_ollama_keep_alive(Some(&invalid)).is_err());
        }
        assert!(validate_ollama_neutral_controls(None, Some(&json!(-1)), None, None).is_ok());
        assert!(validate_ollama_neutral_controls(None, Some(&json!(0)), None, None).is_ok());
    }

    #[test]
    fn embedding_controls_and_payload_are_bounded() {
        for options in [None, Some(&serde_json::Value::Null), Some(&json!({}))] {
            validate_ollama_embedding_controls(options, None).unwrap();
        }
        validate_ollama_embedding_controls(Some(&json!({"num_ctx": null})), None).unwrap();
        for options in [json!(12), json!({"num_ctx": 1024})] {
            assert!(validate_ollama_embedding_controls(Some(&options), None).is_err());
        }
        assert!(validate_ollama_embedding_controls(None, Some(&json!("5m"))).is_ok());

        let result = EmbeddingBatchResult {
            model_id: "embed.gguf".to_string(),
            output: EmbeddingBatchOutput::Embeddings(vec![vec![0.6, 0.8]]),
            prompt_tokens: 3,
            total_duration: Duration::from_millis(2),
        };
        let payload = ollama_embed_payload(result).unwrap();
        assert_eq!(payload["model"], "embed.gguf");
        let first_value = payload["embeddings"][0][0].as_f64().unwrap();
        assert!((first_value - 0.6).abs() < 1e-6);
        assert_eq!(payload["prompt_eval_count"], 3);
        assert_eq!(payload["total_duration"], 2_000_000_u64);
        assert_eq!(payload["load_duration"], 0);
    }

    #[test]
    fn chat_history_maps_name_correlated_tool_calls_to_strict_internal_ids() {
        let messages = serde_json::from_value::<Vec<OllamaMessage>>(json!([
            {"role": "user", "content": "Look up two cities."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"index": 0, "name": "lookup", "arguments": {"city": "Paris"}}},
                    {"type": "function", "function": {"index": 1, "name": "lookup", "arguments": {"city": "Berlin"}}}
                ]
            },
            {"role": "tool", "tool_name": "lookup", "content": "20 C"},
            {"role": "tool", "tool_name": "lookup", "content": "18 C"}
        ]))
        .unwrap();
        let (messages, image) = ollama_chat_messages(messages).unwrap();
        assert!(image.is_none());
        assert_eq!(messages.len(), 4);
        assert_eq!(
            messages[1].extensions["tool_calls"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        normalize_chat_messages(&messages).unwrap();

        let unknown = serde_json::from_value::<Vec<OllamaMessage>>(json!([
            {"role": "tool", "tool_name": "missing", "content": "result"}
        ]))
        .unwrap();
        assert!(ollama_chat_messages(unknown).is_err());
    }

    #[test]
    fn nonstreaming_chat_and_generate_payloads_use_ollama_shapes() {
        let chat = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": "tiny.gguf",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        });
        let payload =
            ollama_payload_from_chat(&chat, OllamaOutputKind::Chat, Duration::from_millis(10))
                .unwrap();
        assert_eq!(payload["message"]["content"], "Hello");
        assert_eq!(payload["done"], true);
        assert_eq!(payload["prompt_eval_count"], 3);
        assert_eq!(payload["eval_count"], 2);
        assert_eq!(payload["total_duration"], 10_000_000_u64);

        let payload =
            ollama_payload_from_chat(&chat, OllamaOutputKind::Generate, Duration::from_millis(10))
                .unwrap();
        assert_eq!(payload["response"], "Hello");
        assert!(payload.get("message").is_none());
    }

    #[test]
    fn function_outputs_become_argument_objects_without_internal_ids() {
        let chat = json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "created": 10_u64,
            "model": "tiny.gguf",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_private",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        let payload =
            ollama_payload_from_chat(&chat, OllamaOutputKind::Chat, Duration::from_millis(1))
                .unwrap();
        assert_eq!(payload["message"]["content"], "");
        assert_eq!(
            payload["message"]["tool_calls"][0]["function"]["arguments"]["city"],
            "Paris"
        );
        assert!(payload["message"]["tool_calls"][0].get("id").is_none());
        assert_eq!(payload["done_reason"], "stop");
    }

    #[tokio::test]
    async fn streaming_chat_is_adapted_from_sse_to_terminal_ndjson() {
        let frames = [
            json!({
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            }),
            json!({
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
            }),
            json!({
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000_u64,
                "model": "tiny.gguf",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }),
            json!({
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000_u64,
                "model": "tiny.gguf",
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            }),
        ];
        let mut sse = frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>();
        sse.push_str("data: [DONE]\n\n");
        let internal = axum::http::Response::builder()
            .status(axum::http::StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(axum::body::Body::from(sse))
            .unwrap();

        let response = ollama_stream_from_chat(
            internal,
            OllamaOutputKind::Chat,
            Instant::now() - Duration::from_millis(1),
            "signed-tiny".to_string(),
            None,
        );
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            OLLAMA_CONTENT_TYPE
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let lines = std::str::from_utf8(&body)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["model"], "signed-tiny");
        assert_eq!(lines[0]["message"]["content"], "Hello");
        assert_eq!(lines[0]["done"], false);
        assert_eq!(lines[1]["message"]["content"], "");
        assert_eq!(lines[1]["done"], true);
        assert_eq!(lines[1]["done_reason"], "stop");
        assert_eq!(lines[1]["model"], "signed-tiny");
        assert_eq!(lines[1]["prompt_eval_count"], 3);
        assert_eq!(lines[1]["eval_count"], 1);
    }

    #[test]
    fn streaming_adapter_rejects_out_of_order_or_extended_internal_events() {
        let mut state = OllamaStreamState::new(OllamaOutputKind::Chat, Instant::now());
        let early_usage = json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1_u64,
            "model": "tiny.gguf",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        assert!(
            state
                .ingest(early_usage)
                .unwrap_err()
                .contains("before its finish reason")
        );

        let mut state = OllamaStreamState::new(OllamaOutputKind::Chat, Instant::now());
        let extended_delta = json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1_u64,
            "model": "tiny.gguf",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello", "future": true},
                "finish_reason": null
            }]
        });
        assert!(
            state
                .ingest(extended_delta)
                .unwrap_err()
                .contains("unsupported field")
        );
    }
}
