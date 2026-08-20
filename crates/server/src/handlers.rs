#![allow(unused_imports, dead_code)]
use super::*;

async fn model_unavailable_response(state: &ServerState) -> axum::response::Response {
    let (error_type, message) = state.model_unavailable().await;
    error_response(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        error_type,
        message,
    )
}

fn embedding_model_generation_response(model_id: &str) -> axum::response::Response {
    error_response(
        axum::http::StatusCode::UNPROCESSABLE_ENTITY,
        "unsupported_operation",
        format!(
            "Model '{model_id}' is an embedding encoder and cannot serve text-generation requests. Use /v1/embeddings, /v1/rerank, /api/embed, or /api/embeddings."
        ),
    )
}

pub(crate) async fn handle_health(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let in_flight = state
        .metrics
        .in_flight_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let requests_total = state
        .metrics
        .requests_total
        .load(std::sync::atomic::Ordering::Relaxed);
    let runtime = state.runtime_pool.read().await.default_runtime();
    let model_id = runtime
        .as_ref()
        .map(|runtime| runtime.model_id.as_str())
        .unwrap_or("loading...");
    Json(json!({
        "status": "ok",
        "model": model_id,
        "in_flight_requests": in_flight,
        "requests_total": requests_total,
        "speculative_mode": state.speculative_mode,
    }))
}

// ─── /metrics ──────────────────────────────────────────────────────────────

pub(crate) async fn handle_metrics(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let runtime = state.runtime_pool.read().await.default_runtime();
    let kv_metrics = {
        if let Some(pool) = runtime
            .as_ref()
            .and_then(|runtime| runtime.kv_cache_pool.as_ref())
        {
            pool.get_metrics()
        } else {
            bloomai_engine::KvCacheMetrics::default()
        }
    };
    let queue_stats = {
        if let Some(scheduler) = runtime
            .as_ref()
            .and_then(|runtime| runtime.scheduler.as_ref())
        {
            scheduler.queue_stats()
        } else {
            (0, 0, 0)
        }
    };
    let cachemesh_metrics = {
        runtime
            .as_ref()
            .and_then(|runtime| runtime.cachemesh.as_ref())
            .map(|mesh| mesh.metrics())
    };
    let body =
        state
            .metrics
            .render_prometheus(&kv_metrics, cachemesh_metrics.as_ref(), queue_stats);
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

// ─── /v1/observability ─────────────────────────────────────────────────────

pub(crate) async fn handle_observability(
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    let (runtime, resident_generations, draining_generations, draining_request_leases) = {
        let runtime_pool = state.runtime_pool.read().await;
        let (draining_generations, draining_request_leases) = state.draining_runtime_stats();
        (
            runtime_pool.default_runtime(),
            runtime_pool.len(),
            draining_generations,
            draining_request_leases,
        )
    };
    let kv_metrics = {
        if let Some(pool) = runtime
            .as_ref()
            .and_then(|runtime| runtime.kv_cache_pool.as_ref())
        {
            pool.get_metrics()
        } else {
            bloomai_engine::KvCacheMetrics::default()
        }
    };
    let kv_utilization = if kv_metrics.total_blocks > 0 {
        (kv_metrics.total_blocks - kv_metrics.free_blocks) as f64 / kv_metrics.total_blocks as f64
    } else {
        0.0
    };
    let queue_stats = {
        if let Some(scheduler) = runtime
            .as_ref()
            .and_then(|runtime| runtime.scheduler.as_ref())
        {
            scheduler.queue_stats()
        } else {
            (0, 0, 0)
        }
    };
    let mut memory = bloomai_engine::MemoryTelemetry::new();
    memory.refresh_ram();

    let model_id = runtime
        .as_ref()
        .map(|runtime| runtime.model_id.as_str())
        .unwrap_or("not loaded");
    let startup_memory_estimate = runtime
        .as_ref()
        .map(|runtime| runtime.memory_estimate.clone());
    let cachemesh = runtime
        .as_ref()
        .and_then(|runtime| runtime.cachemesh.as_ref())
        .map(|mesh| mesh.metrics());
    let loading = state.load_in_progress.load(Ordering::Relaxed);
    let ready = state.ready.load(Ordering::Acquire);
    let load_failed = state.load_error.read().await.is_some();
    let load_phase = if loading {
        "loading"
    } else if load_failed {
        "failed"
    } else if ready {
        "ready"
    } else {
        "idle"
    };
    let requested_model = state.requested_model.read().await.clone();

    (
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        Json(json!({
            "schema_version": 1,
            "object": "bloom.observability_snapshot",
            "created": unix_seconds(),
            "server": {
                "version": env!("CARGO_PKG_VERSION"),
                "uptime_seconds": state.metrics.uptime_seconds(),
            },
            "model": model_id,
            "ready": ready,
            "runtime_pool": {
                "resident_generations": resident_generations,
                "draining_generations": draining_generations,
                "draining_request_leases": draining_request_leases,
            },
            "load": {
                "phase": load_phase,
                "progress": state.load_progress.load(Ordering::Relaxed),
                "requested_model": requested_model,
                "failure_present": load_phase == "failed",
            },
            "speculative_mode": state.speculative_mode,
            "requests": {
                "total": state.metrics.requests_total.load(Ordering::Relaxed),
                "completed": state.metrics.requests_completed.load(Ordering::Relaxed),
                "failed": state.metrics.requests_failed.load(Ordering::Relaxed),
                "in_flight": state.metrics.in_flight_requests.load(Ordering::Relaxed)
            },
            "tokens": {
                "prompt_total": state.metrics.prompt_tokens_total.load(Ordering::Relaxed),
                "generated_total": state.metrics.tokens_generated_total.load(Ordering::Relaxed)
            },
            "scheduler": {
                "ifb_enabled": state.enable_ifb,
                "prefill_queue": queue_stats.0,
                "decoding_queue": queue_stats.1,
                "active_requests": queue_stats.2
            },
            "startup_memory_estimate": startup_memory_estimate,
            "kv_cache": {
                "total_blocks": kv_metrics.total_blocks,
                "free_blocks": kv_metrics.free_blocks,
                "active_blocks": kv_metrics.active_blocks,
                "cached_blocks": kv_metrics.cached_blocks,
                "hits": kv_metrics.hits,
                "misses": kv_metrics.misses,
                "evictions": kv_metrics.evictions,
                "reuses": kv_metrics.reuses,
                "utilization": kv_utilization
            },
            "cachemesh": cachemesh,
            "memory": memory,
        })),
    )
}

// ─── /v1/kv-cache-stats ────────────────────────────────────────────────────

pub(crate) async fn handle_kv_cache_stats(
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    let runtime = state.runtime_pool.read().await.default_runtime();
    let kv_metrics = {
        if let Some(pool) = runtime
            .as_ref()
            .and_then(|runtime| runtime.kv_cache_pool.as_ref())
        {
            pool.get_metrics()
        } else {
            bloomai_engine::KvCacheMetrics::default()
        }
    };
    let utilization = if kv_metrics.total_blocks > 0 {
        (kv_metrics.total_blocks - kv_metrics.free_blocks) as f64 / kv_metrics.total_blocks as f64
    } else {
        0.0
    };
    let cachemesh = runtime
        .as_ref()
        .and_then(|runtime| runtime.cachemesh.as_ref())
        .map(|mesh| mesh.metrics());
    Json(json!({
        "total_blocks": kv_metrics.total_blocks,
        "free_blocks": kv_metrics.free_blocks,
        "active_blocks": kv_metrics.active_blocks,
        "cached_blocks": kv_metrics.cached_blocks,
        "hits": kv_metrics.hits,
        "misses": kv_metrics.misses,
        "evictions": kv_metrics.evictions,
        "reuses": kv_metrics.reuses,
        "utilization": utilization,
        "cachemesh": cachemesh,
    }))
}

// ─── /v1/models ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ModelResourceQuery {
    #[serde(default, flatten)]
    parameters: BTreeMap<String, String>,
}

fn openai_model_resource(runtime: &LoadedRuntime) -> serde_json::Value {
    json!({
        "id": runtime.model_id,
        "object": "model",
        "created": runtime.published_at,
        "owned_by": "bloom"
    })
}

pub(crate) async fn handle_models(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let snapshot = state.runtime_pool.read().await.snapshot();
    let default = snapshot.default_runtime();
    let mut models = Vec::with_capacity(snapshot.entries().len());
    if let Some(runtime) = default.as_ref() {
        models.push(openai_model_resource(runtime));
    }
    models.extend(
        snapshot
            .entries()
            .iter()
            .map(|(_, runtime)| runtime)
            .filter(|runtime| {
                default
                    .as_ref()
                    .is_none_or(|default| !Arc::ptr_eq(runtime, default))
            })
            .map(|runtime| openai_model_resource(runtime)),
    );
    Json(json!({
        "object": "list",
        "data": models
    }))
}

pub(crate) async fn handle_model_retrieve(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(model): axum::extract::Path<String>,
    query: std::result::Result<
        axum::extract::Query<ModelResourceQuery>,
        axum::extract::rejection::QueryRejection,
    >,
) -> axum::response::Response {
    let query = match query {
        Ok(axum::extract::Query(query)) => query,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "The model query parameters are invalid.",
            );
        }
    };
    if !query.parameters.is_empty() {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "The model retrieval endpoint does not accept query parameters.",
        );
    }
    if let Err(error) = validate_model_selector(&model) {
        return requested_model_error_response(error);
    }

    let runtime = match state.runtime_pool.read().await.resolve(Some(&model)) {
        Ok(Some(runtime)) => runtime,
        Ok(None) => return requested_model_error_response(RequestedModelError::NotLoaded),
        Err(error) => return requested_model_error_response(error),
    };
    Json(openai_model_resource(&runtime)).into_response()
}

// ─── /v1/model-management ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct ModelSwitchRequest {
    pub(crate) id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelPreflightRequest {
    pub(crate) id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelRemoveRequest {
    pub(crate) id: String,
}

#[derive(Debug)]
pub(crate) struct ModelActivationError {
    pub(crate) status: axum::http::StatusCode,
    pub(crate) code: &'static str,
    pub(crate) message: String,
}

impl ModelActivationError {
    fn new(status: axum::http::StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum ModelRemovalError {
    InvalidRequest(String),
    NotFound(String),
    Conflict { code: &'static str, message: String },
    Internal(String),
}

impl ModelRemovalError {
    pub(crate) fn status(&self) -> axum::http::StatusCode {
        match self {
            Self::InvalidRequest(_) => axum::http::StatusCode::BAD_REQUEST,
            Self::NotFound(_) => axum::http::StatusCode::NOT_FOUND,
            Self::Conflict { .. } => axum::http::StatusCode::CONFLICT,
            Self::Internal(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub(crate) fn message(&self) -> &str {
        match self {
            Self::InvalidRequest(message)
            | Self::NotFound(message)
            | Self::Internal(message)
            | Self::Conflict { message, .. } => message,
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelIntegrityRequest {
    pub(crate) id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelDownloadControlRequest {
    pub(crate) filename: String,
    #[serde(default)]
    pub(crate) license: Option<String>,
}

pub(crate) async fn handle_model_catalog(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    let (catalog, runtime) = match state.model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(error) => return api_error(ApiError::InternalError, error),
    };

    let loading = state.load_in_progress.load(Ordering::Acquire);
    let ready = state.ready.load(Ordering::Acquire);
    let error = state.load_error.read().await.clone();
    let requested_model = state.requested_model.read().await.clone();
    let phase = if loading {
        "loading"
    } else if ready {
        "ready"
    } else if error.is_some() {
        "error"
    } else {
        "idle"
    };
    let active_model = runtime.as_ref().map(|runtime| {
        json!({
            "id": runtime.model_id,
            "catalog_id": runtime.catalog_id,
            "source": if runtime.catalog_id.is_some() { "catalog" } else { "external" },
            "input_modalities": runtime.input_modalities
        })
    });
    let download_status = match state.model_downloads.as_ref() {
        Some(manager) => {
            let (status, staged) = tokio::join!(manager.status(), manager.staged());
            json!({
                "enabled": true,
                "license_policy": manager.license_policy(),
                "status": status,
                "staged": staged
            })
        }
        None => json!({
            "enabled": false,
            "license_policy": ModelLicensePolicy::default().status(),
            "status": model_download::ModelDownloadStatus::default(),
            "staged": []
        }),
    };
    let import_status = match state.model_imports.as_ref() {
        Some(manager) => {
            let (status, staged) = tokio::join!(manager.status(), manager.staged());
            json!({
                "enabled": true,
                "max_bytes": manager.max_bytes(),
                "max_chunk_bytes": manager.max_chunk_bytes(),
                "license_policy": manager.license_policy(),
                "status": status,
                "staged": staged
            })
        }
        None => json!({
            "enabled": false,
            "max_bytes": 0,
            "max_chunk_bytes": 0,
            "license_policy": ModelLicensePolicy::default().status(),
            "status": model_import::ModelImportStatus::default(),
            "staged": []
        }),
    };
    let storage_status = match state.model_storage.snapshot().await {
        Ok(status) => status,
        Err(error) => return api_error(ApiError::InternalError, error),
    };
    let integrity_status = state.model_integrity.status().await;
    let index_status = json!({
        "enabled": state.model_index.is_some(),
        "key_id": state.model_index.as_ref().and_then(|manager| manager.single_key_id()),
        "trust_id": state.model_index.as_ref().map(|manager| manager.trust_id()),
        "trusted_key_count": state.model_index.as_ref().map_or(0, |manager| manager.trusted_key_count()),
        "refresh_seconds": state.model_index.as_ref().map_or(0, |manager| manager.refresh_seconds()),
        "persistent_rollback_protection": state.model_index.as_ref().is_some_and(|manager| manager.persistent_rollback_protection()),
    });

    Json(json!({
        "schema_version": model_manager::MODEL_CATALOG_SCHEMA_VERSION,
        "object": model_manager::MODEL_CATALOG_OBJECT,
        "root": catalog.root,
        "root_exists": catalog.root_exists,
        "data": catalog.models,
        "active_model": active_model,
        "download": download_status,
        "import": import_status,
        "index": index_status,
        "storage": storage_status,
        "integrity": integrity_status,
        "load": {
            "phase": phase,
            "progress": state.load_progress.load(Ordering::Acquire),
            "requested_model": requested_model,
            "error": error
        }
    }))
    .into_response()
}

pub(crate) async fn handle_model_index(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    model_index_response(&state, false).await
}

pub(crate) async fn handle_model_index_refresh(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    model_index_response(&state, true).await
}

async fn model_index_response(
    state: &ServerState,
    force_refresh: bool,
) -> axum::response::Response {
    let Some(manager) = state.model_index.as_ref() else {
        return error_response(
            axum::http::StatusCode::NOT_FOUND,
            "model_index_not_configured",
            "No signed model discovery index is configured.",
        );
    };
    match manager.snapshot(force_refresh).await {
        Ok(snapshot) => (
            [(header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
            Json(snapshot),
        )
            .into_response(),
        Err(ModelIndexError::Invalid(message)) => error_response(
            axum::http::StatusCode::BAD_GATEWAY,
            "invalid_model_index",
            message,
        ),
        Err(ModelIndexError::Unavailable(message)) => error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "model_index_unavailable",
            message,
        ),
        Err(ModelIndexError::Internal(message)) => error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "model_index_error",
            message,
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelIndexDownloadAdmission {
    Started,
    StartedUpgrade,
    Joined,
    JoinedUpgrade,
    Installed,
}

fn model_index_download_response(
    entry: &model_index::ModelIndexEntry,
    status: model_download::ModelDownloadStatus,
    admission: ModelIndexDownloadAdmission,
) -> axum::response::Response {
    let accepted = matches!(
        admission,
        ModelIndexDownloadAdmission::Started | ModelIndexDownloadAdmission::StartedUpgrade
    );
    let already_in_progress = matches!(
        admission,
        ModelIndexDownloadAdmission::Joined | ModelIndexDownloadAdmission::JoinedUpgrade
    );
    let already_installed = admission == ModelIndexDownloadAdmission::Installed;
    let upgrading = matches!(
        admission,
        ModelIndexDownloadAdmission::StartedUpgrade | ModelIndexDownloadAdmission::JoinedUpgrade
    );
    let status_code = if already_installed {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::ACCEPTED
    };
    (
        status_code,
        [(header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
        Json(json!({
            "object": "bloom.model_index_download",
            "accepted": accepted,
            "already_installed": already_installed,
            "already_in_progress": already_in_progress,
            "upgrading": upgrading,
            "model_index_id": entry.id,
            "model": entry.filename,
            "status": status
        })),
    )
        .into_response()
}

fn installed_model_index_download_response(
    entry: &model_index::ModelIndexEntry,
) -> axum::response::Response {
    model_index_download_response(
        entry,
        model_download::ModelDownloadStatus {
            phase: model_download::ModelDownloadPhase::Complete,
            filename: Some(entry.filename.clone()),
            source_host: None,
            downloaded_bytes: entry.size_bytes,
            total_bytes: Some(entry.size_bytes),
            resumable: false,
            error: None,
        },
        ModelIndexDownloadAdmission::Installed,
    )
}

fn conflicting_model_index_download_response(
    entry: &model_index::ModelIndexEntry,
) -> axum::response::Response {
    error_response(
        axum::http::StatusCode::CONFLICT,
        "model_index_entry_conflict",
        format!(
            "Catalog entry '{}' exists but does not exactly match signed index entry '{}'. Remove or rename the local entry before downloading.",
            entry.filename, entry.id
        ),
    )
}

fn blocked_model_index_upgrade_response(message: impl Into<String>) -> axum::response::Response {
    error_response(
        axum::http::StatusCode::CONFLICT,
        "model_index_upgrade_blocked",
        message.into(),
    )
}

pub(crate) async fn handle_model_index_download(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    if model_index::validate_index_id(&id).is_err() {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_model_index_id",
            "The model index ID is invalid.",
        );
    }
    let Some(downloads) = state.model_downloads.as_ref() else {
        return error_response(
            axum::http::StatusCode::FORBIDDEN,
            "model_downloads_disabled",
            "Model downloads are disabled. Start bloom_server with --enable-model-downloads to enable signed acquisitions.",
        );
    };
    let Some(index) = state.model_index.as_ref() else {
        return error_response(
            axum::http::StatusCode::NOT_FOUND,
            "model_index_not_configured",
            "No signed model discovery index is configured.",
        );
    };
    let snapshot = match index.snapshot(false).await {
        Ok(snapshot) => snapshot,
        Err(ModelIndexError::Invalid(message)) => {
            return error_response(
                axum::http::StatusCode::BAD_GATEWAY,
                "invalid_model_index",
                message,
            );
        }
        Err(ModelIndexError::Unavailable(message)) => {
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model_index_unavailable",
                message,
            );
        }
        Err(ModelIndexError::Internal(message)) => {
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "model_index_error",
                message,
            );
        }
    };
    let Some(entry) = snapshot.data.into_iter().find(|entry| entry.id == id) else {
        return error_response(
            axum::http::StatusCode::NOT_FOUND,
            "model_index_entry_not_found",
            "The requested signed model index entry was not found.",
        );
    };
    if !entry.downloadable {
        return error_response(
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            "model_index_entry_blocked",
            "The requested signed model conflicts with the server acquisition policy.",
        );
    }

    let _storage_guard = state.model_storage.serial().await;
    let (catalog, _) = match state.fresh_model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(error) => return api_error(ApiError::InternalError, error),
    };
    let replacement = match model_index::model_index_installation_state(&catalog, &entry) {
        model_index::ModelIndexInstallationState::Verified => {
            return installed_model_index_download_response(&entry);
        }
        model_index::ModelIndexInstallationState::Conflict => {
            return conflicting_model_index_download_response(&entry);
        }
        model_index::ModelIndexInstallationState::Missing => None,
        model_index::ModelIndexInstallationState::Upgradable => {
            let Some(source) = model_index::model_index_upgrade_source(&catalog, &entry) else {
                return api_error(
                    ApiError::InternalError,
                    "The signed-model upgrade source could not be resolved.",
                );
            };
            if source.active {
                return blocked_model_index_upgrade_response(
                    "Unload or switch away from the installed model before upgrading it.",
                );
            }
            if state.load_in_progress.load(Ordering::Acquire) {
                return blocked_model_index_upgrade_response(
                    "Wait for the current model lifecycle operation before upgrading.",
                );
            }
            if state.model_integrity.is_active(&source.id).await {
                return blocked_model_index_upgrade_response(
                    "Cancel or finish the installed model integrity check before upgrading.",
                );
            }
            match model_index::model_index_upgrade_descriptor(&catalog, &entry) {
                Some(replacement) => Some(replacement),
                None => {
                    return api_error(
                        ApiError::InternalError,
                        "The signed-model upgrade identity could not be prepared.",
                    );
                }
            }
        }
    };

    let current = downloads.status().await;
    if matches!(
        current.phase,
        model_download::ModelDownloadPhase::Queued
            | model_download::ModelDownloadPhase::Downloading
            | model_download::ModelDownloadPhase::Verifying
    ) && current.filename.as_deref() == Some(entry.filename.as_str())
        && downloads
            .active_matches(
                &entry.filename,
                &entry.sha256,
                Some(entry.size_bytes),
                Some(&entry.id),
            )
            .await
    {
        return model_index_download_response(
            &entry,
            current,
            if replacement.is_some() {
                ModelIndexDownloadAdmission::JoinedUpgrade
            } else {
                ModelIndexDownloadAdmission::Joined
            },
        );
    }

    let result = if entry.is_package() {
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
        let Some(url) = entry.download_url.clone() else {
            return api_error(
                ApiError::InternalError,
                "The verified single-file index entry has no download URL.",
            );
        };
        let request = ModelDownloadRequest {
            url,
            filename: entry.filename.clone(),
            sha256: entry.sha256.clone(),
            license: Some(entry.license.clone()),
            expected_size_bytes: Some(entry.size_bytes),
            model_index_id: Some(entry.id.clone()),
        };
        if let Some(replacement) = replacement.clone() {
            downloads.start_upgrade(request, replacement).await
        } else {
            downloads.start(request).await
        }
    };
    match result {
        Ok(status) => model_index_download_response(
            &entry,
            status,
            if replacement.is_some() {
                ModelIndexDownloadAdmission::StartedUpgrade
            } else {
                ModelIndexDownloadAdmission::Started
            },
        ),
        Err(error @ ModelDownloadStartError::Conflict(_)) => {
            let (catalog, _) = match state.fresh_model_catalog_snapshot().await {
                Ok(snapshot) => snapshot,
                Err(scan_error) => return api_error(ApiError::InternalError, scan_error),
            };
            match model_index::model_index_installation_state(&catalog, &entry) {
                model_index::ModelIndexInstallationState::Verified => {
                    installed_model_index_download_response(&entry)
                }
                model_index::ModelIndexInstallationState::Conflict => {
                    conflicting_model_index_download_response(&entry)
                }
                model_index::ModelIndexInstallationState::Missing
                | model_index::ModelIndexInstallationState::Upgradable => {
                    let current = downloads.status().await;
                    if matches!(
                        current.phase,
                        model_download::ModelDownloadPhase::Queued
                            | model_download::ModelDownloadPhase::Downloading
                            | model_download::ModelDownloadPhase::Verifying
                    ) && current.filename.as_deref() == Some(entry.filename.as_str())
                        && downloads
                            .active_matches(
                                &entry.filename,
                                &entry.sha256,
                                Some(entry.size_bytes),
                                Some(&entry.id),
                            )
                            .await
                    {
                        model_index_download_response(
                            &entry,
                            current,
                            if replacement.is_some() {
                                ModelIndexDownloadAdmission::JoinedUpgrade
                            } else {
                                ModelIndexDownloadAdmission::Joined
                            },
                        )
                    } else {
                        model_download_error_response(error)
                    }
                }
            }
        }
        Err(error) => model_download_error_response(error),
    }
}

pub(crate) async fn handle_model_inventory(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    let (catalog, _) = match state.fresh_model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(error) => return api_error(ApiError::InternalError, error),
    };
    let inventory = model_inventory::ModelInventory::from_catalog(&catalog);
    (
        [
            (
                header::CONTENT_DISPOSITION,
                HeaderValue::from_static(model_inventory::MODEL_INVENTORY_CONTENT_DISPOSITION),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        Json(inventory),
    )
        .into_response()
}

pub(crate) async fn handle_model_inventory_reconcile(
    State(state): State<Arc<ServerState>>,
    Json(expected): Json<model_inventory::ModelInventory>,
) -> axum::response::Response {
    let (catalog, _) = match state.fresh_model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(error) => return api_error(ApiError::InternalError, error),
    };
    let current = model_inventory::ModelInventory::from_catalog(&catalog);
    match model_inventory::ModelInventory::reconcile(&expected, &current) {
        Ok(report) => (
            [(header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
            Json(report),
        )
            .into_response(),
        Err(message) => error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_model_inventory",
            message,
        ),
    }
}

pub(crate) async fn handle_model_inventory_restore(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(expected): Json<model_inventory::ModelInventory>,
) -> axum::response::Response {
    let Some(manager) = state.model_downloads.as_ref() else {
        return error_response(
            axum::http::StatusCode::FORBIDDEN,
            "model_downloads_disabled",
            "Inventory restore requires verified model downloads. Start bloom_server with --enable-model-downloads.",
        );
    };
    let (catalog, _) = match state.fresh_model_catalog_snapshot().await {
        Ok(snapshot) => snapshot,
        Err(error) => return api_error(ApiError::InternalError, error),
    };
    let current = model_inventory::ModelInventory::from_catalog(&catalog);
    let candidate =
        match model_inventory::ModelInventory::restore_candidate(&expected, &current, &id) {
            Ok(candidate) => candidate,
            Err(model_inventory::ModelInventoryRestoreError::InvalidInventory(message)) => {
                return error_response(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_model_inventory",
                    message,
                );
            }
            Err(model_inventory::ModelInventoryRestoreError::Invalid(message)) => {
                return error_response(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_model_inventory_restore",
                    message,
                );
            }
            Err(model_inventory::ModelInventoryRestoreError::NotFound(message)) => {
                return error_response(
                    axum::http::StatusCode::NOT_FOUND,
                    "model_inventory_restore_not_found",
                    message,
                );
            }
            Err(model_inventory::ModelInventoryRestoreError::Conflict(message)) => {
                return error_response(
                    axum::http::StatusCode::CONFLICT,
                    "model_inventory_restore_conflict",
                    message,
                );
            }
            Err(model_inventory::ModelInventoryRestoreError::Unavailable(message)) => {
                return error_response(
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    "model_inventory_restore_unavailable",
                    message,
                );
            }
        };
    if candidate.size_bytes > manager.max_bytes() {
        return error_response(
            axum::http::StatusCode::PAYLOAD_TOO_LARGE,
            "model_inventory_restore_too_large",
            format!(
                "The expected model size exceeds the configured {} byte download limit.",
                manager.max_bytes()
            ),
        );
    }
    let filename = candidate.filename.clone();
    match manager
        .start(ModelDownloadRequest {
            url: candidate.url,
            filename: candidate.filename,
            sha256: candidate.sha256,
            license: candidate.license,
            expected_size_bytes: Some(candidate.size_bytes),
            model_index_id: candidate.model_index_id,
        })
        .await
    {
        Ok(status) => (
            axum::http::StatusCode::ACCEPTED,
            [(header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
            Json(json!({
                "object": "bloom.model_inventory_restore",
                "accepted": true,
                "model": filename,
                "status": status
            })),
        )
            .into_response(),
        Err(error) => model_download_error_response(error),
    }
}

pub(crate) async fn handle_model_preflight(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelPreflightRequest>,
) -> axum::response::Response {
    match state.model_preflight.inspect(&payload.id).await {
        Ok(report) => Json(json!({
            "schema_version": model_preflight::MODEL_PREFLIGHT_SCHEMA_VERSION,
            "object": model_preflight::MODEL_PREFLIGHT_OBJECT,
            "data": report
        }))
        .into_response(),
        Err(model_preflight::ModelPreflightError::Invalid(message)) => error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "model_preflight_failed",
            message,
        ),
        Err(model_preflight::ModelPreflightError::Internal(message)) => {
            api_error(ApiError::InternalError, message)
        }
    }
}

pub(crate) async fn handle_model_download_start(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelDownloadRequest>,
) -> axum::response::Response {
    let Some(manager) = state.model_downloads.as_ref() else {
        return error_response(
            axum::http::StatusCode::FORBIDDEN,
            "model_downloads_disabled",
            "Model downloads are disabled. Start bloom_server with --enable-model-downloads to enable trusted, verified downloads.",
        );
    };
    match manager.start(payload).await {
        Ok(status) => (
            axum::http::StatusCode::ACCEPTED,
            Json(json!({
                "object": "bloom.model_download",
                "accepted": true,
                "status": status
            })),
        )
            .into_response(),
        Err(error) => model_download_error_response(error),
    }
}

pub(crate) async fn handle_model_download_source_inspect(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelDownloadSourceRequest>,
) -> axum::response::Response {
    let Some(manager) = state.model_downloads.as_ref() else {
        return error_response(
            axum::http::StatusCode::FORBIDDEN,
            "model_downloads_disabled",
            "Model downloads are disabled. Start bloom_server with --enable-model-downloads to inspect trusted download sources.",
        );
    };
    match manager.inspect_source(payload).await {
        Ok(source) => (
            [(header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
            Json(json!({
                "object": "bloom.model_download_source",
                "download_url": source.download_url,
                "filename": source.filename,
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
                "commit_hash": source.commit_hash,
                "verification_ready": source.verification_ready,
                "warning": source.warning
            })),
        )
            .into_response(),
        Err(ModelDownloadInspectError::Invalid(message)) => error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_model_download_source",
            message,
        ),
        Err(ModelDownloadInspectError::TooLarge(message)) => error_response(
            axum::http::StatusCode::PAYLOAD_TOO_LARGE,
            "model_download_source_too_large",
            message,
        ),
        Err(ModelDownloadInspectError::Unavailable(message)) => error_response(
            axum::http::StatusCode::BAD_GATEWAY,
            "model_download_source_unavailable",
            message,
        ),
    }
}

pub(crate) async fn handle_model_download_resume(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelDownloadControlRequest>,
) -> axum::response::Response {
    let Some(manager) = state.model_downloads.as_ref() else {
        return error_response(
            axum::http::StatusCode::FORBIDDEN,
            "model_downloads_disabled",
            "Model downloads are disabled.",
        );
    };
    match manager
        .resume(payload.filename.trim(), payload.license)
        .await
    {
        Ok(status) => (
            axum::http::StatusCode::ACCEPTED,
            Json(json!({
                "object": "bloom.model_download",
                "accepted": true,
                "resumed": true,
                "status": status
            })),
        )
            .into_response(),
        Err(error) => model_download_error_response(error),
    }
}

pub(crate) async fn handle_model_download_discard(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelDownloadControlRequest>,
) -> axum::response::Response {
    let Some(manager) = state.model_downloads.as_ref() else {
        return error_response(
            axum::http::StatusCode::FORBIDDEN,
            "model_downloads_disabled",
            "Model downloads are disabled.",
        );
    };
    let filename = payload.filename.trim().to_string();
    match manager.discard(&filename).await {
        Ok(()) => Json(json!({
            "object": "bloom.model_download_discard",
            "discarded": true,
            "filename": filename
        }))
        .into_response(),
        Err(error) => model_download_error_response(error),
    }
}

fn model_download_error_response(error: ModelDownloadStartError) -> axum::response::Response {
    match error {
        ModelDownloadStartError::Invalid(message) => error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_model_download",
            message,
        ),
        ModelDownloadStartError::Conflict(message) => error_response(
            axum::http::StatusCode::CONFLICT,
            "model_download_conflict",
            message,
        ),
        ModelDownloadStartError::NotFound(message) => error_response(
            axum::http::StatusCode::NOT_FOUND,
            "model_download_not_found",
            message,
        ),
        ModelDownloadStartError::Internal(message) => error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "model_download_error",
            message,
        ),
    }
}

pub(crate) async fn handle_model_import_begin(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelImportRequest>,
) -> axum::response::Response {
    let Some(manager) = state.model_imports.as_ref() else {
        return model_imports_disabled_response();
    };
    match manager.begin(payload).await {
        Ok(status) => Json(json!({
            "object": "bloom.model_import",
            "status": status
        }))
        .into_response(),
        Err(error) => model_import_error_response(error),
    }
}

pub(crate) async fn handle_model_import_chunk(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(filename): axum::extract::Path<String>,
    headers: axum::http::HeaderMap,
    bytes: axum::body::Bytes,
) -> axum::response::Response {
    let Some(manager) = state.model_imports.as_ref() else {
        return model_imports_disabled_response();
    };
    let offset = match headers
        .get("upload-offset")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        Some(offset) => offset,
        None => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_model_import",
                "A valid Upload-Offset header is required.",
            );
        }
    };
    match manager.append_chunk(&filename, offset, &bytes).await {
        Ok(status) => Json(json!({
            "object": "bloom.model_import",
            "status": status
        }))
        .into_response(),
        Err(error) => model_import_error_response(error),
    }
}

pub(crate) async fn handle_model_import_complete(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> axum::response::Response {
    let Some(manager) = state.model_imports.as_ref() else {
        return model_imports_disabled_response();
    };
    match manager.complete(&filename).await {
        Ok(status) => Json(json!({
            "object": "bloom.model_import",
            "installed": true,
            "status": status
        }))
        .into_response(),
        Err(error) => model_import_error_response(error),
    }
}

pub(crate) async fn handle_model_import_discard(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> axum::response::Response {
    let Some(manager) = state.model_imports.as_ref() else {
        return model_imports_disabled_response();
    };
    match manager.discard(&filename).await {
        Ok(()) => Json(json!({
            "object": "bloom.model_import_discard",
            "discarded": true,
            "filename": filename
        }))
        .into_response(),
        Err(error) => model_import_error_response(error),
    }
}

fn model_imports_disabled_response() -> axum::response::Response {
    error_response(
        axum::http::StatusCode::FORBIDDEN,
        "model_imports_disabled",
        "Model imports are disabled. Start bloom_server with --enable-model-imports to enable bounded local-file imports.",
    )
}

fn model_import_error_response(error: ModelImportError) -> axum::response::Response {
    match error {
        ModelImportError::Invalid(message) => error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_model_import",
            message,
        ),
        ModelImportError::Conflict(message) => error_response(
            axum::http::StatusCode::CONFLICT,
            "model_import_conflict",
            message,
        ),
        ModelImportError::NotFound(message) => error_response(
            axum::http::StatusCode::NOT_FOUND,
            "model_import_not_found",
            message,
        ),
        ModelImportError::OffsetMismatch { expected } => (
            axum::http::StatusCode::CONFLICT,
            Json(json!({
                "error": {
                    "message": format!("Upload offset does not match the staged file; expected {expected}."),
                    "type": "model_import_offset_mismatch",
                    "expected_offset": expected
                }
            })),
        )
            .into_response(),
        ModelImportError::Internal(message) => error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "model_import_error",
            message,
        ),
    }
}

pub(crate) async fn handle_model_download_cancel(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    let Some(manager) = state.model_downloads.as_ref() else {
        return error_response(
            axum::http::StatusCode::FORBIDDEN,
            "model_downloads_disabled",
            "Model downloads are disabled.",
        );
    };
    if manager.cancel().await {
        (
            axum::http::StatusCode::ACCEPTED,
            Json(json!({
                "object": "bloom.model_download_cancel",
                "cancelled": true,
                "partial_download_retained": true
            })),
        )
            .into_response()
    } else {
        error_response(
            axum::http::StatusCode::CONFLICT,
            "no_model_download",
            "No model download is in progress.",
        )
    }
}

pub(crate) async fn prepare_catalog_model_load(
    state: &Arc<ServerState>,
    model_id: &str,
) -> std::result::Result<PathBuf, ModelActivationError> {
    if let Some(downloads) = state.model_downloads.as_ref()
        && downloads.upgrade_source_active(model_id).await
    {
        return Err(ModelActivationError::new(
            axum::http::StatusCode::CONFLICT,
            "model_upgrade_in_progress",
            "Wait for the signed-model upgrade to finish before loading this model.",
        ));
    }
    if state.model_integrity.is_active(model_id).await {
        return Err(ModelActivationError::new(
            axum::http::StatusCode::CONFLICT,
            "model_integrity_in_progress",
            "Wait for the model integrity verification to finish before loading this model.",
        ));
    }
    let integrity = state.model_integrity.status().await;
    if integrity.model_id.as_deref() == Some(model_id) && integrity.matches_expected == Some(false)
    {
        return Err(ModelActivationError::new(
            axum::http::StatusCode::CONFLICT,
            "model_integrity_mismatch",
            "The model does not match its verified acquisition checksum and cannot be loaded.",
        ));
    }
    let root = state.models_root.clone();
    let resolve_id = model_id.to_string();
    let (path, recorded_mismatch) = match task::spawn_blocking(move || -> anyhow::Result<_> {
        let path = ModelCatalog::resolve(&root, &resolve_id)?;
        let metadata = std::fs::metadata(&path)?;
        let recorded_mismatch = if metadata.is_file() {
            model_provenance::read_provenance(&root, &resolve_id, metadata.len())?
                .is_some_and(|provenance| provenance.integrity_mismatch_at.is_some())
        } else {
            false
        };
        Ok((path, recorded_mismatch))
    })
    .await
    {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            return Err(ModelActivationError::new(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_model",
                error.to_string(),
            ));
        }
        Err(error) => {
            return Err(ModelActivationError::new(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "model_catalog_error",
                format!("model catalog resolution failed: {error}"),
            ));
        }
    };
    if recorded_mismatch {
        return Err(ModelActivationError::new(
            axum::http::StatusCode::CONFLICT,
            "model_integrity_mismatch",
            "The model has a recorded integrity mismatch and cannot be loaded until it passes verification.",
        ));
    }

    let preflight = match state.model_preflight.inspect(model_id).await {
        Ok(report) => report,
        Err(model_preflight::ModelPreflightError::Invalid(message)) => {
            return Err(ModelActivationError::new(
                axum::http::StatusCode::BAD_REQUEST,
                "model_preflight_failed",
                message,
            ));
        }
        Err(model_preflight::ModelPreflightError::Internal(message)) => {
            return Err(ModelActivationError::new(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "model_preflight_error",
                message,
            ));
        }
    };
    if !preflight.loadable {
        return Err(ModelActivationError::new(
            axum::http::StatusCode::CONFLICT,
            "model_preflight_failed",
            preflight.load_blocker.unwrap_or_else(|| {
                "The model is not compatible with the configured runtime.".to_string()
            }),
        ));
    }
    Ok(path)
}

pub(crate) async fn handle_model_switch(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelSwitchRequest>,
) -> axum::response::Response {
    let _storage_guard = state.model_storage.serial().await;
    let model_id = payload.id.trim().to_string();
    let path = match prepare_catalog_model_load(&state, &model_id).await {
        Ok(path) => path,
        Err(error) => return error_response(error.status, error.code, error.message),
    };

    match state
        .admit_model_load(path, Some(model_id.clone()), false)
        .await
    {
        Ok(ModelLoadAdmission::AlreadyReady { .. }) => Json(json!({
            "object": "bloom.model_load",
            "accepted": false,
            "unchanged": true,
            "model": model_id
        }))
        .into_response(),
        Ok(ModelLoadAdmission::Loading { sequence, .. }) => (
            axum::http::StatusCode::ACCEPTED,
            Json(json!({
                "object": "bloom.model_load",
                "accepted": true,
                "model": model_id,
                "sequence": sequence
            })),
        )
            .into_response(),
        Err(ModelLoadAdmissionError::Busy) => error_response(
            axum::http::StatusCode::CONFLICT,
            "model_load_in_progress",
            "Another model lifecycle operation is already in progress.",
        ),
        Err(ModelLoadAdmissionError::Unavailable(message)) => {
            api_error(ApiError::ServiceUnavailable, message)
        }
    }
}

pub(crate) async fn handle_model_remove(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelRemoveRequest>,
) -> axum::response::Response {
    match remove_catalog_model(&state, &payload.id).await {
        Ok(id) => Json(json!({
            "object": "bloom.model_remove",
            "removed": true,
            "model": id
        }))
        .into_response(),
        Err(ModelRemovalError::InvalidRequest(message)) => error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        ),
        Err(ModelRemovalError::NotFound(message)) => error_response(
            axum::http::StatusCode::NOT_FOUND,
            "model_not_found",
            message,
        ),
        Err(ModelRemovalError::Conflict { code, message }) => {
            error_response(axum::http::StatusCode::CONFLICT, code, message)
        }
        Err(ModelRemovalError::Internal(message)) => error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            message,
        ),
    }
}

pub(crate) async fn remove_catalog_model(
    state: &Arc<ServerState>,
    requested_id: &str,
) -> std::result::Result<String, ModelRemovalError> {
    const MAX_MODEL_CATALOG_ID_BYTES: usize = 255;

    if requested_id != requested_id.trim()
        || requested_id.is_empty()
        || requested_id.len() > MAX_MODEL_CATALOG_ID_BYTES
        || requested_id.chars().any(char::is_control)
        || model_manager::validate_catalog_id(requested_id).is_err()
    {
        return Err(ModelRemovalError::InvalidRequest(
            "Model must be a bounded direct catalog entry ID without surrounding whitespace or control characters."
                .to_string(),
        ));
    }

    let id = requested_id.to_string();
    let _storage_guard = state.model_storage.serial().await;
    if let Some(downloads) = state.model_downloads.as_ref()
        && downloads.upgrade_source_active(&id).await
    {
        return Err(ModelRemovalError::Conflict {
            code: "model_upgrade_in_progress",
            message: "Wait for the signed-model upgrade to finish before removing this model."
                .to_string(),
        });
    }
    if state.load_in_progress.load(Ordering::Acquire) {
        return Err(ModelRemovalError::Conflict {
            code: "model_load_in_progress",
            message: "Models cannot be removed while a lifecycle operation is in progress."
                .to_string(),
        });
    }

    if state.model_integrity.is_active(&id).await {
        return Err(ModelRemovalError::Conflict {
            code: "model_integrity_in_progress",
            message: "Cancel the active integrity verification before removing this model."
                .to_string(),
        });
    }

    let (catalog, _) = state
        .fresh_model_catalog_snapshot()
        .await
        .map_err(|error| {
            tracing::error!(%error, "Failed to refresh the model catalog before removal");
            ModelRemovalError::Internal(
                "The local model catalog could not be inspected safely.".to_string(),
            )
        })?;
    let entry = catalog
        .models
        .iter()
        .find(|entry| entry.id == id)
        .ok_or_else(|| {
            ModelRemovalError::NotFound(format!("Model '{id}' was not found in the local catalog."))
        })?;
    if entry.active {
        return Err(ModelRemovalError::Conflict {
            code: "model_is_active",
            message: "Unload or switch away from the active model before removing it.".to_string(),
        });
    }

    let root = state.models_root.clone();
    let resolve_id = id.clone();
    let resolved =
        match task::spawn_blocking(move || ModelCatalog::resolve(&root, &resolve_id)).await {
            Ok(Ok(path)) => path,
            Ok(Err(error)) => {
                tracing::warn!(model = %id, %error, "Catalog entry changed before removal");
                return Err(ModelRemovalError::Conflict {
                    code: "model_catalog_changed",
                    message: "The model catalog entry changed while removal was being prepared."
                        .to_string(),
                });
            }
            Err(error) => {
                tracing::error!(model = %id, %error, "Model catalog resolution task failed");
                return Err(ModelRemovalError::Internal(
                    "The model catalog entry could not be inspected safely.".to_string(),
                ));
            }
        };
    if state.source_is_resident_or_draining(&resolved).await {
        return Err(ModelRemovalError::Conflict {
            code: "model_is_active",
            message: "Unload or switch away from the active model before removing it.".to_string(),
        });
    }

    let root = state.models_root.clone();
    let remove_id = id.clone();
    match task::spawn_blocking(move || ModelCatalog::remove(&root, &remove_id, &resolved)).await {
        Ok(Ok(())) => {
            *state.model_catalog_cache.write().await = None;
            Ok(id)
        }
        Ok(Err(error)) => {
            tracing::warn!(model = %id, %error, "Model removal did not complete safely");
            Err(ModelRemovalError::Conflict {
                code: "model_catalog_changed",
                message: "The model catalog entry changed or could not be removed safely."
                    .to_string(),
            })
        }
        Err(error) => {
            tracing::error!(model = %id, %error, "Model removal task failed");
            Err(ModelRemovalError::Internal(
                "The model removal task failed.".to_string(),
            ))
        }
    }
}

pub(crate) async fn handle_model_integrity_start(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ModelIntegrityRequest>,
) -> axum::response::Response {
    let _storage_guard = state.model_storage.serial().await;
    if state.load_in_progress.load(Ordering::Acquire) {
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "model_load_in_progress",
            "Wait for the model lifecycle operation to finish before verifying integrity.",
        );
    }
    let model_id = payload.id.trim().to_string();
    if let Some(downloads) = state.model_downloads.as_ref()
        && downloads.upgrade_source_active(&model_id).await
    {
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "model_upgrade_in_progress",
            "Wait for the signed-model upgrade to finish before verifying this model.",
        );
    }
    let root = state.models_root.clone();
    let resolve_id = model_id.clone();
    let resolved =
        match task::spawn_blocking(move || ModelCatalog::resolve(&root, &resolve_id)).await {
            Ok(Ok(path)) => path,
            Ok(Err(error)) => return api_error(ApiError::InvalidRequest, error),
            Err(error) => {
                return api_error(
                    ApiError::InternalError,
                    format!("model catalog resolution failed: {error}"),
                );
            }
        };
    if state.source_is_resident_or_draining(&resolved).await {
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "model_is_active",
            "Unload or switch away from the active model before verifying its on-disk integrity.",
        );
    }
    match state.model_integrity.start(&model_id).await {
        Ok(status) => (
            axum::http::StatusCode::ACCEPTED,
            Json(json!({
                "object": "bloom.model_integrity",
                "accepted": true,
                "status": status
            })),
        )
            .into_response(),
        Err(error) => model_integrity_error_response(error),
    }
}

pub(crate) async fn handle_model_integrity_cancel(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    if state.model_integrity.cancel().await {
        (
            axum::http::StatusCode::ACCEPTED,
            Json(json!({
                "object": "bloom.model_integrity_cancel",
                "cancelled": true
            })),
        )
            .into_response()
    } else {
        error_response(
            axum::http::StatusCode::CONFLICT,
            "no_model_integrity_verification",
            "No model integrity verification is in progress.",
        )
    }
}

fn model_integrity_error_response(error: ModelIntegrityError) -> axum::response::Response {
    match error {
        ModelIntegrityError::Invalid(message) => error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_model_integrity_request",
            message,
        ),
        ModelIntegrityError::Conflict(message) => error_response(
            axum::http::StatusCode::CONFLICT,
            "model_integrity_conflict",
            message,
        ),
        ModelIntegrityError::Internal(message) => error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "model_integrity_error",
            message,
        ),
    }
}

pub(crate) async fn handle_model_unload(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    unload_model_runtime(state, None, false).await
}

pub(crate) async fn handle_model_unload_exact(
    state: Arc<ServerState>,
    expected: Arc<LoadedRuntime>,
) -> axum::response::Response {
    unload_model_runtime(state, Some(expected), false).await
}

pub(crate) async fn handle_model_unload_exact_if_idle(
    state: Arc<ServerState>,
    expected: Arc<LoadedRuntime>,
) -> axum::response::Response {
    unload_model_runtime(state, Some(expected), true).await
}

async fn unload_model_runtime(
    state: Arc<ServerState>,
    expected: Option<Arc<LoadedRuntime>>,
    only_if_idle: bool,
) -> axum::response::Response {
    let _lifecycle_guard = state.model_lifecycle.lock().await;
    if state.load_in_progress.swap(true, Ordering::AcqRel) {
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "model_load_in_progress",
            "Another model lifecycle operation is already in progress.",
        );
    }

    let (removed, fallback, busy) = {
        let mut runtime_pool = state.runtime_pool.write().await;
        // Request leases are acquired while holding the pool read lock. This
        // write-side check therefore makes timer-driven idle eviction atomic
        // with new inference admission.
        let busy = only_if_idle
            && expected.as_ref().is_some_and(|expected| {
                runtime_pool.contains_exact(expected)
                    && expected.active_request_leases.load(Ordering::Acquire) > 0
            });
        let removed = if busy {
            None
        } else {
            match expected.as_ref() {
                Some(expected) => runtime_pool.remove_exact(expected),
                None => runtime_pool.remove_default(),
            }
        };
        if let Some(runtime) = removed.as_ref() {
            state.track_draining_runtimes(std::slice::from_ref(runtime));
        }
        let fallback = runtime_pool.default_runtime();
        if !busy {
            state.ready.store(fallback.is_some(), Ordering::Release);
        }
        (removed, fallback, busy)
    };
    if busy {
        state.load_in_progress.store(false, Ordering::Release);
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "requests_in_flight",
            "The selected runtime still has requests in flight.",
        );
    }
    if expected.is_some() && removed.is_none() {
        state.load_in_progress.store(false, Ordering::Release);
        return error_response(
            axum::http::StatusCode::NOT_FOUND,
            "model_not_loaded",
            "The selected runtime generation is no longer loaded.",
        );
    }
    drop(removed);
    *state.requested_model.write().await = fallback.as_ref().map(|runtime| {
        runtime
            .catalog_id
            .clone()
            .unwrap_or_else(|| runtime.model_id.clone())
    });
    *state.load_error.write().await = None;
    state
        .load_progress
        .store(if fallback.is_some() { 100 } else { 0 }, Ordering::Release);
    state.load_in_progress.store(false, Ordering::Release);

    Json(json!({
        "object": "bloom.model_unload",
        "unloaded": true
    }))
    .into_response()
}

// ─── /v1/chat/completions ───────────────────────────────────────────────────

pub(crate) fn validate_chat_messages(messages: &[NormalizedChatMessage]) -> Result<(), String> {
    if messages.is_empty() {
        return Err("Messages must contain at least one entry.".to_string());
    }
    if messages.len() > MAX_CHAT_REQUEST_MESSAGES {
        return Err(format!(
            "Messages cannot contain more than {MAX_CHAT_REQUEST_MESSAGES} entries."
        ));
    }
    let mut content_bytes = 0_usize;
    for message in messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err("Message roles must be one of system, user, or assistant.".to_string());
        }
        if message.role == "user" && message.content.chars().count() > MAX_CHAT_USER_MESSAGE_CHARS {
            return Err(format!(
                "User message content cannot exceed {MAX_CHAT_USER_MESSAGE_CHARS} characters."
            ));
        }
        if message.role == "system"
            && message.content.chars().count() > MAX_CHAT_SYSTEM_MESSAGE_CHARS
        {
            return Err(format!(
                "System message content cannot exceed {MAX_CHAT_SYSTEM_MESSAGE_CHARS} characters."
            ));
        }
        content_bytes = content_bytes
            .checked_add(message.content.len())
            .ok_or_else(|| "Message content byte count overflowed.".to_string())?;
        if content_bytes > MAX_CHAT_CONTENT_BYTES {
            return Err(format!(
                "Combined message content cannot exceed {MAX_CHAT_CONTENT_BYTES} bytes."
            ));
        }
    }
    Ok(())
}

struct GeneratedText {
    text: String,
    stopped: bool,
}

async fn collect_scheduled_text(
    receiver: &mut mpsc::UnboundedReceiver<Result<u32, String>>,
    pipeline: &InferencePipeline,
    scheduler: &InferenceScheduler,
    request_id: &str,
    generated_count: &AtomicU64,
    stop_sequences: Vec<String>,
) -> std::result::Result<GeneratedText, String> {
    let has_stop_sequences = !stop_sequences.is_empty();
    let mut filter = StopSequenceFilter::new(stop_sequences);
    let mut generated_tokens = Vec::new();
    let mut text = String::new();
    while let Some(result) = receiver.recv().await {
        let token = result?;
        generated_tokens.push(token);
        generated_count.fetch_add(1, Ordering::Relaxed);
        if has_stop_sequences {
            let delta = pipeline.detokenize(&[token]).unwrap_or_default();
            let update = filter.push(&delta);
            text.push_str(&update.text);
            if update.stopped {
                scheduler.cancel_request(request_id);
                return Ok(GeneratedText {
                    text,
                    stopped: true,
                });
            }
        }
    }
    if has_stop_sequences {
        text.push_str(&filter.finish());
    } else {
        text = pipeline.detokenize(&generated_tokens).unwrap_or_default();
    }
    Ok(GeneratedText {
        text,
        stopped: false,
    })
}

fn run_cancellable_text_inference(
    pipeline: &InferencePipeline,
    input: ModelInput,
    params: &GenerationParams,
    cancel_token: &CancellationToken,
    stop_sequences: Vec<String>,
) -> Result<GeneratedText> {
    if cancel_token.is_cancelled() {
        return Err(anyhow!("request cancelled"));
    }
    let mut filter = StopSequenceFilter::new(stop_sequences);
    let mut generated_text = String::new();
    let run_result = pipeline.run_stream(input, params, &mut |chunk| {
        if cancel_token.is_cancelled() {
            return Err(anyhow!("request cancelled"));
        }
        if let OutputChunk::TextDelta(text) = chunk {
            let update = filter.push(&text);
            generated_text.push_str(&update.text);
            if update.stopped {
                return Err(anyhow!("stop sequence reached"));
            }
        }
        Ok(())
    });
    let stopped = filter.stopped();
    if let Err(error) = run_result
        && !stopped
    {
        return Err(error);
    }
    if cancel_token.is_cancelled() {
        return Err(anyhow!("request cancelled"));
    }
    generated_text.push_str(&filter.finish());
    Ok(GeneratedText {
        text: generated_text,
        stopped,
    })
}

pub(crate) async fn handle_chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ChatRequest>,
) -> axum::response::Response {
    handle_chat_completions_inner(state, payload, None).await
}

/// Execute a chat request against the exact runtime generation selected by an
/// internal protocol adapter. This prevents an unload/reload with the same
/// public model id from redirecting the request after adapter activation.
pub(crate) async fn handle_chat_completions_for_runtime(
    state: Arc<ServerState>,
    payload: ChatRequest,
    runtime: Arc<LoadedRuntime>,
) -> axum::response::Response {
    handle_chat_completions_inner(state, payload, Some(runtime)).await
}

async fn handle_chat_completions_inner(
    state: Arc<ServerState>,
    payload: ChatRequest,
    exact_runtime: Option<Arc<LoadedRuntime>>,
) -> axum::response::Response {
    if let Err(message) = validate_chat_request_compatibility(&payload) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let stop_sequences = match normalize_stop_sequences(payload.stop.as_ref()) {
        Ok(sequences) => sequences,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let tool_config = match chat_tool_config(&payload) {
        Ok(config) => config,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let messages = match normalize_chat_messages(&payload.messages) {
        Ok(messages) => messages,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    if let Err(message) = validate_chat_messages(&messages) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let response_format = match response_format_mode(payload.response_format.as_ref()) {
        Ok(mode) => mode,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    if tool_config.is_some() && !matches!(response_format, ResponseFormatMode::Text) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "response_format cannot be combined with active function tools in Bloom. Tool arguments are validated against each function's parameters schema.",
        );
    }
    let core_response_format = match (&tool_config, &response_format) {
        (Some(_), _) => bloomai_core::ResponseFormat::JsonObject,
        (None, ResponseFormatMode::Text) => bloomai_core::ResponseFormat::Text,
        (None, ResponseFormatMode::JsonObject) => bloomai_core::ResponseFormat::JsonObject,
        (None, ResponseFormatMode::JsonSchema(schema)) => {
            bloomai_core::ResponseFormat::JsonSchema(schema.clone())
        }
    };
    let max_tokens =
        match resolve_chat_max_tokens(payload.max_tokens, payload.max_completion_tokens) {
            Ok(max_tokens) => max_tokens,
            Err(message) => {
                return error_response(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    message,
                );
            }
        };
    let temperature = payload.temperature.unwrap_or(0.7);
    let top_p = payload.top_p.unwrap_or(0.9);
    if let Err(message) = validate_generation_controls(max_tokens, temperature, top_p) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    if !state.ready.load(Ordering::Acquire) {
        return model_unavailable_response(&state).await;
    }

    let runtime_lease = match exact_runtime {
        Some(runtime) => match state.lease_exact_runtime(&runtime).await {
            Some(lease) => lease,
            None => {
                return error_response(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "model_unavailable",
                    "The selected model was unloaded before inference admission.",
                );
            }
        },
        None => match state.lease_runtime(payload.model.as_deref()).await {
            Ok(Some(lease)) => lease,
            Ok(None) => return model_unavailable_response(&state).await,
            Err(error) => return requested_model_error_response(error),
        },
    };
    let runtime = Arc::clone(runtime_lease.runtime());
    let model_id = runtime.model_id.clone();
    if model_supports_embeddings(&runtime.pipeline) {
        return embedding_model_generation_response(&model_id);
    }
    let pipeline = Arc::clone(&runtime.pipeline);
    let model_family = runtime.model_family.clone();
    let model_architecture = runtime.model_architecture.clone();
    let model_chat_template = runtime.model_chat_template.clone();
    let scheduler_opt = runtime.scheduler.clone();
    let include_usage = payload
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);

    let permit = match Arc::clone(&state.semaphore).try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = std::time::Instant::now();
    let params = GenerationParams {
        max_tokens,
        temperature,
        top_p,
        seed: payload.seed,
        response_format: match core_response_format {
            bloomai_core::ResponseFormat::Text => None,
            constrained => Some(constrained),
        },
    };

    let prompt = match &tool_config {
        Some(config) => apply_tool_instruction(
            chat_prompt_for_metadata(
                &messages,
                &model_family,
                model_architecture.as_deref(),
                model_chat_template.as_deref(),
            ),
            config,
        ),
        None => apply_response_format_instruction(
            chat_prompt_for_metadata(
                &messages,
                &model_family,
                model_architecture.as_deref(),
                model_chat_template.as_deref(),
            ),
            &response_format,
        ),
    };
    let prompt_tokens_vec = match pipeline.tokenize(&prompt) {
        Ok(tokens) => tokens,
        Err(error) => {
            state
                .metrics
                .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "tokenization_error",
                format!("Prompt tokenization failed: {error}"),
            );
        }
    };
    let prompt_tokens = prompt_tokens_vec.len();
    if let Err(message) =
        validate_context_budget(prompt_tokens, params.max_tokens, pipeline.context_size())
    {
        state.metrics.record_request_end(
            false,
            request_start.elapsed().as_secs_f64(),
            0,
            prompt_tokens as u64,
        );
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            message,
        );
    }
    let input = ModelInput::Text { prompt };
    let request_id = match payload.internal_request_id.clone() {
        Some(request_id) if validate_request_id(&request_id).is_ok() => request_id,
        Some(_) => {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "The internal generation request ID is invalid.",
            );
        }
        None => next_request_id(&state, "chatcmpl"),
    };
    let created = unix_seconds();
    let cancel_scheduler = if state.enable_ifb {
        scheduler_opt.clone()
    } else {
        None
    };
    let Some(cancel_guard) =
        CancelTokenGuard::register(&state, request_id.clone(), cancel_scheduler)
    else {
        state.metrics.record_request_end(
            false,
            request_start.elapsed().as_secs_f64(),
            0,
            prompt_tokens as u64,
        );
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "request_id_conflict",
            "A request with the same ID is already active.",
        );
    };
    let cancel_token = cancel_guard.token();

    if state.enable_ifb {
        let Some(scheduler) = scheduler_opt else {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "scheduler_unavailable",
                "Continuous batching is enabled, but the scheduler is unavailable.",
            );
        };
        let (tx, mut rx) = mpsc::unbounded_channel::<Result<u32, String>>();
        scheduler
            .token_senders
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id.clone(), tx);

        let req = Request {
            id: request_id.clone(),
            model_id: model_id.clone(),
            prompt_tokens: prompt_tokens_vec,
            generated_tokens: Vec::new(),
            params: params.clone(),
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        if let Err(e) = scheduler.submit_with_execution_guard(req, runtime_lease.execution_guard())
        {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("Failed to submit to scheduler: {}", e),
            );
        }
        // Cancellation can win after the request ID is registered but before
        // the scheduler owns the request. Recheck after submission so a
        // successful cancellation can never leave a newly queued orphan.
        if cancel_token.is_cancelled() {
            scheduler.cancel_request(&request_id);
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::REQUEST_TIMEOUT,
                "request_cancelled",
                "The generation request was cancelled.",
            );
        }

        if !payload.stream {
            let generated_count = Arc::new(AtomicU64::new(0));
            let lifecycle = InferenceLifecycle::new(
                cancel_guard,
                InferenceLifecycleResources {
                    metrics: Arc::clone(&state.metrics),
                    request_start,
                    generated_tokens: Arc::clone(&generated_count),
                    prompt_tokens: prompt_tokens as u64,
                    permit,
                    runtime_lease: runtime_lease.clone(),
                },
                StreamExecution::Scheduled(Arc::clone(&scheduler)),
            );
            let mut client_guard = lifecycle.client_guard();
            let generated_output = match collect_scheduled_text(
                &mut rx,
                &pipeline,
                &scheduler,
                &request_id,
                &generated_count,
                stop_sequences.clone(),
            )
            .await
            {
                Ok(output) => output,
                Err(error) => {
                    client_guard.finish(false);
                    return error_response(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "internal_error",
                        format!("Scheduler execution failed: {error}"),
                    );
                }
            };
            if cancel_token.is_cancelled() {
                client_guard.finish(false);
                return error_response(
                    axum::http::StatusCode::REQUEST_TIMEOUT,
                    "request_cancelled",
                    "The generation request was cancelled.",
                );
            }
            let generated_text = generated_output.text;
            let parsed_output =
                match parse_chat_output(&generated_text, &response_format, tool_config.as_ref()) {
                    Ok(output) => output,
                    Err(message) => {
                        client_guard.finish(false);
                        return error_response(
                            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                            if tool_config.is_some() {
                                "invalid_tool_call"
                            } else {
                                "invalid_response_format"
                            },
                            message,
                        );
                    }
                };
            client_guard.finish(true);
            let completion_tokens = generated_count.load(Ordering::Relaxed) as usize;
            let finish_reason = if generated_output.stopped {
                "stop"
            } else {
                chat_output_finish_reason(&parsed_output, completion_tokens, max_tokens)
            };
            let message = chat_output_message(&request_id, &parsed_output);

            return Json(json!({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }))
            .into_response();
        }

        // Streaming under IFB scheduler
        let req_id = request_id.clone();
        let model = model_id.clone();
        let state_clone = Arc::clone(&state);
        let generated_count = Arc::new(AtomicU64::new(0));
        let first_token_seen = Arc::new(AtomicBool::new(false));
        let last_token_time = Arc::new(std::sync::Mutex::new(None));
        let stream_failed = Arc::new(AtomicBool::new(false));
        let generated_count_for_stream = Arc::clone(&generated_count);
        let first_token_for_stream = Arc::clone(&first_token_seen);
        let last_token_for_stream = Arc::clone(&last_token_time);
        let stream_failed_for_stream = Arc::clone(&stream_failed);

        let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
        let accumulated_text_for_stream = Arc::clone(&accumulated_text);
        let stop_filter = Arc::new(std::sync::Mutex::new(StopSequenceFilter::new(
            stop_sequences.clone(),
        )));
        let stop_filter_for_stream = Arc::clone(&stop_filter);
        let stop_sequence_hit = Arc::new(AtomicBool::new(false));
        let stop_sequence_hit_for_stream = Arc::clone(&stop_sequence_hit);
        let scheduler_for_stop = Arc::clone(&scheduler);
        let request_id_for_stop = request_id.clone();

        let pipeline_for_stream = Arc::clone(&pipeline);
        let expose_text_deltas = tool_config.is_none();
        let sse_stream = UnboundedReceiverStream::new(rx)
            .map(move |item| {
                let chunk = match item {
                    Ok(tok) => {
                        record_stream_tokens(
                            &state_clone,
                            request_start,
                            &first_token_for_stream,
                            &last_token_for_stream,
                            &generated_count_for_stream,
                            1,
                        );
                        let text = pipeline_for_stream.detokenize(&[tok]).unwrap_or_default();
                        let update = {
                            let mut filter = stop_filter_for_stream
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            filter.push(&text)
                        };
                        if update.stopped {
                            stop_sequence_hit_for_stream.store(true, Ordering::Release);
                            scheduler_for_stop.cancel_request(&request_id_for_stop);
                        }
                        {
                            let mut acc = accumulated_text_for_stream
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            acc.push_str(&update.text);
                        }
                        if !expose_text_deltas || update.text.is_empty() {
                            return None;
                        }
                        json!({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": update.text },
                                "finish_reason": null
                            }]
                        })
                    }
                    Err(message) => {
                        stream_failed_for_stream.store(true, Ordering::Relaxed);
                        json!({
                            "error": {
                                "message": message,
                                "type": "internal_error"
                            }
                        })
                    }
                };
                Some(Ok::<Event, std::convert::Infallible>(json_event(chunk)))
            })
            .filter_map(futures::future::ready);

        let final_request_id = request_id.clone();
        let final_model_id = model_id.clone();
        let generated_count_for_finish = Arc::clone(&generated_count);
        let stop_sequence_hit_for_final = Arc::clone(&stop_sequence_hit);
        let stream_failed_for_final = Arc::clone(&stream_failed);
        let stream_failed_for_validation = Arc::clone(&stream_failed);
        let accumulated_text_for_final = Arc::clone(&accumulated_text);
        let response_format_for_final = response_format.clone();
        let tool_config_for_final = tool_config.clone();
        let usage_request_id = request_id.clone();
        let usage_model_id = model_id.clone();
        let start_request_id = request_id.clone();
        let start_model_id = model_id.clone();
        let generated_count_for_usage = Arc::clone(&generated_count);
        let stream_failed_for_usage = Arc::clone(&stream_failed);
        let usage_stream = futures::stream::once(async move {
            (include_usage && !stream_failed_for_usage.load(Ordering::Acquire)).then(|| {
                Ok::<Event, std::convert::Infallible>(chat_usage_chunk(
                    usage_request_id,
                    usage_model_id,
                    created,
                    prompt_tokens as u64,
                    generated_count_for_usage.load(Ordering::Relaxed),
                ))
            })
        })
        .filter_map(futures::future::ready);
        let start_stream = futures::stream::once(async move {
            Ok::<Event, std::convert::Infallible>(chat_start_chunk(
                start_request_id,
                start_model_id,
                created,
            ))
        });
        let stop_filter_for_flush = Arc::clone(&stop_filter);
        let accumulated_text_for_flush = Arc::clone(&accumulated_text);
        let stream_failed_for_flush = Arc::clone(&stream_failed);
        let flush_request_id = request_id.clone();
        let flush_model_id = model_id.clone();
        let flush_stream = futures::stream::once(async move {
            if stream_failed_for_flush.load(Ordering::Acquire) {
                return None;
            }
            let tail = stop_filter_for_flush
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .finish();
            if !tail.is_empty() {
                accumulated_text_for_flush
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push_str(&tail);
            }
            (expose_text_deltas && !tail.is_empty()).then(|| {
                Ok::<Event, std::convert::Infallible>(json_event(json!({
                    "id": flush_request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": flush_model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": tail},
                        "finish_reason": null
                    }]
                })))
            })
        })
        .filter_map(futures::future::ready);
        let lifecycle = InferenceLifecycle::new(
            cancel_guard,
            InferenceLifecycleResources {
                metrics: Arc::clone(&state.metrics),
                request_start,
                generated_tokens: Arc::clone(&generated_count),
                prompt_tokens: prompt_tokens as u64,
                permit,
                runtime_lease: runtime_lease.clone(),
            },
            StreamExecution::Scheduled(Arc::clone(&scheduler)),
        );
        let mut lifecycle_for_final = lifecycle.client_guard();

        let final_stream = start_stream
            .chain(sse_stream)
            .chain(flush_stream)
            .chain(futures::stream::once(async move {
                if stream_failed_for_validation.load(Ordering::Acquire) {
                    return None;
                }
                let text = {
                    let acc = accumulated_text_for_final
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    acc.clone()
                };
                match parse_chat_output(
                    &text,
                    &response_format_for_final,
                    tool_config_for_final.as_ref(),
                ) {
                    Err(message) => {
                        stream_failed_for_validation.store(true, Ordering::Relaxed);
                        let err_chunk = json!({
                            "error": {
                                "message": format!("Stream terminal output validation failed: {message}"),
                                "type": if tool_config_for_final.is_some() {
                                    "invalid_tool_call"
                                } else {
                                    "invalid_response_format"
                                }
                            }
                        });
                        Some(Ok::<Event, std::convert::Infallible>(json_event(err_chunk)))
                    }
                    Ok(output) if tool_config_for_final.is_some() => {
                        Some(Ok::<Event, std::convert::Infallible>(chat_tool_output_chunk(
                            final_request_id,
                            final_model_id,
                            created,
                            &output,
                            generated_count_for_finish.load(Ordering::Relaxed) as usize,
                            max_tokens,
                        )))
                    }
                    Ok(_) => {
                        let finish_reason = if stop_sequence_hit_for_final.load(Ordering::Acquire) {
                            "stop"
                        } else {
                            generation_finish_reason(
                                generated_count_for_finish.load(Ordering::Relaxed) as usize,
                                max_tokens,
                            )
                        };
                        Some(Ok::<Event, std::convert::Infallible>(chat_stop_chunk(
                            final_request_id,
                            final_model_id,
                            created,
                            finish_reason,
                        )))
                    }
                }
            }).filter_map(futures::future::ready))
            .chain(usage_stream)
            .chain(futures::stream::once(async move {
                lifecycle_for_final.finish(!stream_failed_for_final.load(Ordering::Relaxed));
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }));

        return Sse::new(final_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    if !payload.stream {
        let generated_count = Arc::new(AtomicU64::new(0));
        let lifecycle = InferenceLifecycle::new(
            cancel_guard,
            InferenceLifecycleResources {
                metrics: Arc::clone(&state.metrics),
                request_start,
                generated_tokens: Arc::clone(&generated_count),
                prompt_tokens: prompt_tokens as u64,
                permit,
                runtime_lease: runtime_lease.clone(),
            },
            StreamExecution::Blocking,
        );
        let worker_lifecycle = lifecycle.worker_guard();
        let mut client_guard = lifecycle.client_guard();
        let pipeline_for_run = Arc::clone(&pipeline);
        let cancel_token_for_run = cancel_token.clone();
        let stop_sequences_for_run = stop_sequences.clone();
        let inference_start = std::time::Instant::now();
        let res = task::spawn_blocking(move || {
            let _worker_lifecycle = worker_lifecycle;
            run_cancellable_text_inference(
                &pipeline_for_run,
                input,
                &params,
                &cancel_token_for_run,
                stop_sequences_for_run,
            )
        })
        .await;
        state
            .metrics
            .record_inference_latency(inference_start.elapsed().as_secs_f64());

        let generated_output = match res {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                client_guard.finish(false);
                if cancel_token.is_cancelled() {
                    return error_response(
                        axum::http::StatusCode::REQUEST_TIMEOUT,
                        "request_cancelled",
                        "The generation request was cancelled.",
                    );
                }
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Inference failed: {}", e),
                );
            }
            Err(e) => {
                client_guard.finish(false);
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Task join failed: {}", e),
                );
            }
        };

        let generated_text = generated_output.text;
        let parsed_output =
            match parse_chat_output(&generated_text, &response_format, tool_config.as_ref()) {
                Ok(output) => output,
                Err(message) => {
                    client_guard.finish(false);
                    return error_response(
                        axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                        if tool_config.is_some() {
                            "invalid_tool_call"
                        } else {
                            "invalid_response_format"
                        },
                        message,
                    );
                }
            };
        let completion_tokens = pipeline.tokenize(&generated_text).unwrap_or_default().len();
        generated_count.store(completion_tokens as u64, Ordering::Relaxed);

        client_guard.finish(true);
        let finish_reason = if generated_output.stopped {
            "stop"
        } else {
            chat_output_finish_reason(&parsed_output, completion_tokens, max_tokens)
        };
        let message = chat_output_message(&request_id, &parsed_output);

        return Json(json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id.clone(),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }))
        .into_response();
    }

    // Streaming
    let (tx, rx) = mpsc::channel::<std::result::Result<String, String>>(100);
    let pipeline_for_stream_run = Arc::clone(&pipeline);
    let cancel_token_clone = cancel_token.clone();
    let stop_sequence_hit = Arc::new(AtomicBool::new(false));
    let stop_sequence_hit_for_worker = Arc::clone(&stop_sequence_hit);
    let stop_sequences_for_stream = stop_sequences.clone();
    let generated_count = Arc::new(AtomicU64::new(0));
    let lifecycle = InferenceLifecycle::new(
        cancel_guard,
        InferenceLifecycleResources {
            metrics: Arc::clone(&state.metrics),
            request_start,
            generated_tokens: Arc::clone(&generated_count),
            prompt_tokens: prompt_tokens as u64,
            permit,
            runtime_lease: runtime_lease.clone(),
        },
        StreamExecution::Blocking,
    );
    let worker_lifecycle = lifecycle.worker_guard();
    task::spawn_blocking(move || {
        let _worker_lifecycle = worker_lifecycle;
        let tx_clone = tx.clone();
        let mut stop_filter = StopSequenceFilter::new(stop_sequences_for_stream);
        let run_res = pipeline_for_stream_run.run_stream(
            input,
            &params,
            &mut |chunk: bloomai_engine::io::OutputChunk| {
                // Check for cancellation
                if cancel_token_clone.is_cancelled() {
                    return Err(anyhow!("request cancelled"));
                }
                if let bloomai_engine::io::OutputChunk::TextDelta(text) = chunk {
                    let update = stop_filter.push(&text);
                    if !update.text.is_empty() && tx_clone.blocking_send(Ok(update.text)).is_err() {
                        return Err(anyhow!("client disconnected"));
                    }
                    if update.stopped {
                        stop_sequence_hit_for_worker.store(true, Ordering::Release);
                        return Err(anyhow!("stop sequence reached"));
                    }
                }
                Ok(())
            },
        );
        if run_res.is_ok() {
            let tail = stop_filter.finish();
            if !tail.is_empty() {
                let _ = tx.blocking_send(Ok(tail));
            }
        } else if !stop_sequence_hit_for_worker.load(Ordering::Acquire) {
            let e = run_res.expect_err("failed stream result must contain an error");
            let _ = tx.blocking_send(Err(e.to_string()));
        }
    });

    let req_id = request_id.clone();
    let model = model_id.clone();
    let state_for_stream = Arc::clone(&state);
    let first_token_seen = Arc::new(AtomicBool::new(false));
    let last_token_time = Arc::new(std::sync::Mutex::new(None));
    let stream_failed = Arc::new(AtomicBool::new(false));
    let generated_count_for_stream = Arc::clone(&generated_count);
    let first_token_for_stream = Arc::clone(&first_token_seen);
    let last_token_for_stream = Arc::clone(&last_token_time);
    let stream_failed_for_stream = Arc::clone(&stream_failed);

    let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
    let accumulated_text_for_stream = Arc::clone(&accumulated_text);

    let pipeline_for_stream = Arc::clone(&pipeline);
    let expose_text_deltas = tool_config.is_none();
    let sse_stream = ReceiverStream::new(rx)
        .map(move |item| {
            let chunk = match item {
                Ok(token) => {
                    {
                        let mut acc = accumulated_text_for_stream
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        acc.push_str(&token);
                    }
                    let token_count = estimate_delta_tokens(&pipeline_for_stream, &token);
                    record_stream_tokens(
                        &state_for_stream,
                        request_start,
                        &first_token_for_stream,
                        &last_token_for_stream,
                        &generated_count_for_stream,
                        token_count,
                    );
                    if !expose_text_deltas {
                        return None;
                    }
                    json!({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": token },
                            "finish_reason": null
                        }]
                    })
                }
                Err(message) => {
                    stream_failed_for_stream.store(true, Ordering::Relaxed);
                    json!({
                        "error": {
                            "message": message,
                            "type": "internal_error"
                        }
                    })
                }
            };
            Some(Ok::<Event, std::convert::Infallible>(json_event(chunk)))
        })
        .filter_map(futures::future::ready);

    let final_request_id = request_id.clone();
    let final_model_id = model_id.clone();
    let stream_failed_for_final = Arc::clone(&stream_failed);
    let stream_failed_for_validation = Arc::clone(&stream_failed);
    let accumulated_text_for_final = Arc::clone(&accumulated_text);
    let response_format_for_final = response_format.clone();
    let tool_config_for_final = tool_config.clone();
    let generated_count_for_validation = Arc::clone(&generated_count);
    let stop_sequence_hit_for_final = Arc::clone(&stop_sequence_hit);
    let pipeline_for_validation = Arc::clone(&pipeline);
    let usage_request_id = request_id.clone();
    let usage_model_id = model_id.clone();
    let start_request_id = request_id.clone();
    let start_model_id = model_id.clone();
    let generated_count_for_usage = Arc::clone(&generated_count);
    let usage_stream = futures::stream::once(async move {
        include_usage.then(|| {
            Ok::<Event, std::convert::Infallible>(chat_usage_chunk(
                usage_request_id,
                usage_model_id,
                created,
                prompt_tokens as u64,
                generated_count_for_usage.load(Ordering::Relaxed),
            ))
        })
    })
    .filter_map(futures::future::ready);
    let start_stream = futures::stream::once(async move {
        Ok::<Event, std::convert::Infallible>(chat_start_chunk(
            start_request_id,
            start_model_id,
            created,
        ))
    });
    let mut lifecycle_for_final = lifecycle.client_guard();

    let final_stream = start_stream
        .chain(sse_stream)
        .chain(futures::stream::once(async move {
            let text = {
                let acc = accumulated_text_for_final
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                acc.clone()
            };
            generated_count_for_validation.store(
                pipeline_for_validation
                    .tokenize(&text)
                    .map(|tokens| tokens.len() as u64)
                    .unwrap_or_else(|_| generated_count_for_validation.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
            match parse_chat_output(
                &text,
                &response_format_for_final,
                tool_config_for_final.as_ref(),
            ) {
                Err(message) => {
                    stream_failed_for_validation.store(true, Ordering::Relaxed);
                    let err_chunk = json!({
                        "error": {
                            "message": format!("Stream terminal output validation failed: {message}"),
                            "type": if tool_config_for_final.is_some() {
                                "invalid_tool_call"
                            } else {
                                "invalid_response_format"
                            }
                        }
                    });
                    Ok::<Event, std::convert::Infallible>(json_event(err_chunk))
                }
                Ok(output) if tool_config_for_final.is_some() => {
                    Ok::<Event, std::convert::Infallible>(chat_tool_output_chunk(
                        final_request_id,
                        final_model_id,
                        created,
                        &output,
                        generated_count_for_validation.load(Ordering::Relaxed) as usize,
                        max_tokens,
                    ))
                }
                Ok(_) => {
                    let finish_reason = if stop_sequence_hit_for_final.load(Ordering::Acquire) {
                        "stop"
                    } else {
                        generation_finish_reason(
                            generated_count_for_validation.load(Ordering::Relaxed) as usize,
                            max_tokens,
                        )
                    };
                    Ok::<Event, std::convert::Infallible>(chat_stop_chunk(
                        final_request_id,
                        final_model_id,
                        created,
                        finish_reason,
                    ))
                }
            }
        }))
        .chain(usage_stream)
        .chain(futures::stream::once(async move {
            lifecycle_for_final.finish(!stream_failed_for_final.load(Ordering::Relaxed));
            Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
        }));

    Sse::new(final_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ─── /v1/responses ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ResponseInputItemsQuery {
    #[serde(default)]
    after: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    order: Option<String>,
    #[serde(default, flatten)]
    extensions: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ResponseResourceQuery {
    #[serde(default, flatten)]
    parameters: BTreeMap<String, String>,
}

pub(crate) async fn handle_response_retrieve(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
    query: std::result::Result<
        axum::extract::Query<ResponseResourceQuery>,
        axum::extract::rejection::QueryRejection,
    >,
) -> axum::response::Response {
    let query = match query {
        Ok(axum::extract::Query(query)) => query,
        Err(_) => return invalid_responses_query(),
    };
    if !query.parameters.is_empty() {
        return invalid_responses_query();
    }
    if !valid_stored_response_id(&response_id) {
        return invalid_stored_response_id();
    }
    match state.response_store.get(&response_id) {
        Some(stored) => private_responses_json(stored.response),
        None => stored_response_not_found(&response_id),
    }
}

pub(crate) async fn handle_response_delete(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
    query: std::result::Result<
        axum::extract::Query<ResponseResourceQuery>,
        axum::extract::rejection::QueryRejection,
    >,
) -> axum::response::Response {
    let query = match query {
        Ok(axum::extract::Query(query)) => query,
        Err(_) => return invalid_responses_query(),
    };
    if !query.parameters.is_empty() {
        return invalid_responses_query();
    }
    if !valid_stored_response_id(&response_id) {
        return invalid_stored_response_id();
    }
    if !state.response_store.delete(&response_id) {
        return stored_response_not_found(&response_id);
    }
    private_responses_json(json!({
        "id": response_id,
        "object": "response",
        "deleted": true
    }))
}

pub(crate) async fn handle_response_input_items(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
    query: std::result::Result<
        axum::extract::Query<ResponseInputItemsQuery>,
        axum::extract::rejection::QueryRejection,
    >,
) -> axum::response::Response {
    let query = match query {
        Ok(axum::extract::Query(query)) => query,
        Err(_) => return invalid_responses_query(),
    };
    if !valid_stored_response_id(&response_id) {
        return invalid_stored_response_id();
    }
    if !query.extensions.is_empty() {
        return responses_bad_request("The input-items query contains unsupported parameters.");
    }
    let limit = query.limit.unwrap_or(20);
    if !(1..=100).contains(&limit) {
        return responses_bad_request("The input-items limit must be between 1 and 100.");
    }
    let order = query.order.as_deref().unwrap_or("desc");
    if !matches!(order, "asc" | "desc") {
        return responses_bad_request("The input-items order must be `asc` or `desc`.");
    }
    if let Some(after) = query.after.as_deref()
        && validate_request_id(after).is_err()
    {
        return responses_bad_request("The input-items cursor is invalid.");
    }

    let Some(stored) = state.response_store.get(&response_id) else {
        return stored_response_not_found(&response_id);
    };
    let mut items = stored.input_items;
    if order == "desc" {
        items.reverse();
    }
    let start = match query.after.as_deref() {
        Some(after) => match items
            .iter()
            .position(|item| item.get("id").and_then(serde_json::Value::as_str) == Some(after))
        {
            Some(position) => position + 1,
            None => {
                return responses_bad_request(
                    "The input-items cursor does not belong to this response.",
                );
            }
        },
        None => 0,
    };
    let end = start.saturating_add(limit).min(items.len());
    let data = items[start..end].to_vec();
    let first_id = data
        .first()
        .and_then(|item| item.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let last_id = data
        .last()
        .and_then(|item| item.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    private_responses_json(json!({
        "object": "list",
        "data": data,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": end < items.len()
    }))
}

fn valid_stored_response_id(response_id: &str) -> bool {
    response_id.starts_with("resp-") && validate_request_id(response_id).is_ok()
}

fn invalid_stored_response_id() -> axum::response::Response {
    responses_bad_request("The response ID is invalid.")
}

fn stored_response_not_found(response_id: &str) -> axum::response::Response {
    private_responses_response(error_response(
        axum::http::StatusCode::NOT_FOUND,
        "invalid_request_error",
        format!("No stored response exists for ID {response_id:?}."),
    ))
}

fn invalid_responses_query() -> axum::response::Response {
    private_responses_response(error_response(
        axum::http::StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "The Responses query parameters are invalid or unsupported.",
    ))
}

fn responses_bad_request(message: &'static str) -> axum::response::Response {
    private_responses_response(error_response(
        axum::http::StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
    ))
}

fn private_responses_json(payload: serde_json::Value) -> axum::response::Response {
    private_responses_response(Json(payload).into_response())
}

fn private_responses_response(mut response: axum::response::Response) -> axum::response::Response {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

pub(crate) async fn handle_responses(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ResponsesRequest>,
) -> axum::response::Response {
    if let Err(message) = validate_responses_request_compatibility(&payload) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }

    let max_output_tokens = payload.max_output_tokens.unwrap_or(128);
    let temperature = payload.temperature.unwrap_or(0.7);
    let top_p = payload.top_p.unwrap_or(0.9);
    let stream = payload.stream;
    if let Err(message) = validate_generation_controls(max_output_tokens, temperature, top_p) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let (response_format, response_text_format) = match responses_text_format(payload.text.as_ref())
    {
        Ok(format) => format,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let response_metadata = match responses_metadata(payload.metadata.as_ref()) {
        Ok(metadata) => metadata,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let tool_bridge = match responses_tool_bridge(
        payload.tools.as_ref(),
        payload.tool_choice.as_ref(),
        payload.parallel_tool_calls,
    ) {
        Ok(bridge) => bridge,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };

    if payload.instructions.as_ref().is_some_and(|instructions| {
        instructions.chars().count() > MAX_CHAT_SYSTEM_MESSAGE_CHARS
            || instructions.len() > MAX_CHAT_CONTENT_BYTES
    }) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "Responses instructions cannot exceed {MAX_CHAT_SYSTEM_MESSAGE_CHARS} characters or {MAX_CHAT_CONTENT_BYTES} bytes."
            ),
        );
    }
    let store = payload.store.unwrap_or(false);
    if let Some(previous_response_id) = payload.previous_response_id.as_deref()
        && !valid_stored_response_id(previous_response_id)
    {
        return invalid_stored_response_id();
    }
    let previous_response_id = payload.previous_response_id.clone();
    let response_state = ResponsesStateOptions::new(store, previous_response_id.clone())
        .with_metadata(response_metadata)
        .with_tools(&tool_bridge);
    let previous = match previous_response_id.as_deref() {
        Some(response_id) => match state.response_store.get(response_id) {
            Some(stored) => Some(stored),
            None => return stored_response_not_found(response_id),
        },
        None => None,
    };
    if let Some(previous) = previous.as_ref()
        && payload
            .model
            .as_deref()
            .is_some_and(|model| model != "default" && model != previous.model)
    {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "A chained local response must use the same model as its previous response.",
        );
    }

    let response_instructions = payload.instructions.clone();
    let response_request_id = next_request_id(&state, "resp");
    let current_input = match normalize_responses_input(payload.input, &response_request_id) {
        Ok(input) => input,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let current_messages = current_input.messages;
    let mut inherited_history = previous
        .as_ref()
        .map(|stored| stored.history.clone())
        .unwrap_or_default();
    inherited_history.extend(current_messages.iter().cloned());
    let mut messages = Vec::with_capacity(
        inherited_history
            .len()
            .saturating_add(usize::from(response_instructions.is_some())),
    );
    if let Some(instructions) = response_instructions.clone() {
        messages.push(ChatCompletionMessage {
            role: "developer".to_string(),
            content: serde_json::Value::String(instructions),
            extensions: BTreeMap::new(),
        });
    }
    messages.extend(inherited_history.iter().cloned());

    let pending_storage = if store {
        let mut input_items = previous
            .as_ref()
            .map(|stored| {
                let mut items = stored.input_items.clone();
                if let Some(output) = stored
                    .response
                    .get("output")
                    .and_then(serde_json::Value::as_array)
                {
                    items.extend(output.iter().cloned());
                }
                items
            })
            .unwrap_or_default();
        input_items.extend(current_input.items);
        Some(response_store::PendingResponseStorage::new(
            state.response_store.clone(),
            inherited_history,
            input_items,
        ))
    } else {
        None
    };
    let model = previous
        .as_ref()
        .map(|stored| stored.model.clone())
        .or(payload.model);
    let chat_request = ChatRequest {
        model,
        messages,
        stream,
        stream_options: stream.then_some(StreamOptions {
            include_usage: true,
            extensions: std::collections::BTreeMap::new(),
        }),
        max_tokens: None,
        max_completion_tokens: Some(max_output_tokens),
        temperature: Some(temperature),
        top_p: Some(top_p),
        seed: None,
        stop: None,
        response_format,
        tools: tool_bridge.chat_tools,
        tool_choice: tool_bridge.chat_tool_choice,
        parallel_tool_calls: tool_bridge.parallel_tool_calls,
        internal_request_id: Some(response_request_id.clone()),
        extensions: std::collections::BTreeMap::new(),
    };
    let response = handle_chat_completions(State(state), Json(chat_request))
        .await
        .into_response();
    if !response.status().is_success() {
        return response;
    }
    if stream {
        return responses_stream_from_chat_response(
            response,
            ResponsesStreamAdapter::new(
                response_request_id,
                response_instructions,
                max_output_tokens,
                temperature,
                top_p,
                response_text_format,
                response_state,
            ),
            pending_storage,
        );
    }

    let body =
        match axum::body::to_bytes(response.into_body(), MAX_RESPONSES_ADAPTER_BODY_BYTES).await {
            Ok(body) => body,
            Err(_) => {
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "The bounded Responses adapter could not read the generated completion.",
                );
            }
        };
    let chat_payload = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(payload) => payload,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "The Responses adapter received an invalid internal completion payload.",
            );
        }
    };
    match responses_payload_from_chat(
        &chat_payload,
        response_instructions.as_deref(),
        max_output_tokens,
        temperature,
        top_p,
        &response_text_format,
        &response_state,
    ) {
        Ok(payload) => {
            if let Some(pending_storage) = pending_storage
                && let Err(error) = pending_storage.commit(&payload)
            {
                return error_response(
                    axum::http::StatusCode::INSUFFICIENT_STORAGE,
                    "server_error",
                    error.to_string(),
                );
            }
            private_responses_json(payload)
        }
        Err(message) => error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            message,
        ),
    }
}

pub(crate) fn responses_stream_from_chat_response(
    response: axum::response::Response,
    mut adapter: ResponsesStreamAdapter,
    pending_storage: Option<response_store::PendingResponseStorage>,
) -> axum::response::Response {
    let is_event_stream = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("text/event-stream"));
    if !is_event_stream {
        return error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "The Responses adapter expected an internal SSE completion stream.",
        );
    }

    let mut body = response.into_body().into_data_stream();
    let (tx, rx) = mpsc::channel::<std::result::Result<Event, std::convert::Infallible>>(32);
    task::spawn(async move {
        let mut pending_storage = pending_storage;
        let mut decoder = ChatSseDecoder::default();
        loop {
            let chunk = tokio::select! {
                _ = tx.closed() => return,
                chunk = body.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) => {
                    send_responses_stream_failure(
                        &tx,
                        &mut adapter,
                        "The internal chat response body failed during streaming.",
                    )
                    .await;
                    return;
                }
            };
            let frames = match decoder.push(&chunk) {
                Ok(frames) => frames,
                Err(message) => {
                    send_responses_stream_failure(&tx, &mut adapter, &message).await;
                    return;
                }
            };
            for frame in frames {
                if frame == "[DONE]" {
                    if let Err(message) = decoder.finish() {
                        send_responses_stream_failure(&tx, &mut adapter, &message).await;
                        return;
                    }
                    let terminal_response = match adapter.build_terminal_response() {
                        Ok(response) => response,
                        Err(message) => {
                            send_responses_stream_failure(&tx, &mut adapter, &message).await;
                            return;
                        }
                    };
                    if let Some(storage) = pending_storage.take()
                        && let Err(error) = storage.commit(&terminal_response)
                    {
                        send_responses_stream_failure(&tx, &mut adapter, &error.to_string()).await;
                        return;
                    }
                    let events = match adapter.finish_with_response(terminal_response) {
                        Ok(events) => events,
                        Err(message) => {
                            send_responses_stream_failure(&tx, &mut adapter, &message).await;
                            return;
                        }
                    };
                    send_responses_sse_events(&tx, events).await;
                    return;
                }
                let payload = match serde_json::from_str::<serde_json::Value>(&frame) {
                    Ok(payload) => payload,
                    Err(_) => {
                        send_responses_stream_failure(
                            &tx,
                            &mut adapter,
                            "The internal chat stream emitted invalid JSON data.",
                        )
                        .await;
                        return;
                    }
                };
                let events = match adapter.ingest_chat_payload(payload) {
                    Ok(events) => events,
                    Err(message) => {
                        send_responses_stream_failure(&tx, &mut adapter, &message).await;
                        return;
                    }
                };
                if !send_responses_sse_events(&tx, events).await {
                    return;
                }
                if adapter.is_terminal() {
                    return;
                }
            }
        }

        let message = decoder.finish().err().unwrap_or_else(|| {
            "The internal chat stream ended before its [DONE] marker.".to_string()
        });
        send_responses_stream_failure(&tx, &mut adapter, &message).await;
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

async fn send_responses_sse_events(
    tx: &mpsc::Sender<std::result::Result<Event, std::convert::Infallible>>,
    events: Vec<ResponsesSseEvent>,
) -> bool {
    for event in events {
        let event_type = event.event_type;
        let serialized = Event::default()
            .event(event_type)
            .json_data(event.data)
            .unwrap_or_else(|_| {
                Event::default().event("error").data(
                    r#"{"type":"error","code":"server_error","message":"The Responses adapter could not serialize a stream event.","param":null,"sequence_number":0}"#,
                )
            });
        if tx.send(Ok(serialized)).await.is_err() {
            return false;
        }
    }
    true
}

async fn send_responses_stream_failure(
    tx: &mpsc::Sender<std::result::Result<Event, std::convert::Infallible>>,
    adapter: &mut ResponsesStreamAdapter,
    message: &str,
) {
    let events = adapter.failure_events(message).unwrap_or_else(|_| {
        vec![ResponsesSseEvent {
            event_type: "error",
            data: json!({
                "type": "error",
                "code": "server_error",
                "message": "The Responses adapter failed while reporting a stream error.",
                "param": null,
                "sequence_number": 0
            }),
        }]
    });
    send_responses_sse_events(tx, events).await;
}

// ─── /v1/completions ────────────────────────────────────────────────────────

pub(crate) async fn handle_completions(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<CompletionRequest>,
) -> impl IntoResponse {
    if let Err(message) = validate_completion_request_compatibility(&payload) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let stop_sequences = match normalize_stop_sequences(payload.stop.as_ref()) {
        Ok(sequences) => sequences,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let max_tokens = payload.max_tokens.unwrap_or(128);
    let temperature = payload.temperature.unwrap_or(0.7);
    let top_p = payload.top_p.unwrap_or(0.9);
    if let Err(message) = validate_generation_controls(max_tokens, temperature, top_p) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let response_format = match response_format_mode(payload.response_format.as_ref()) {
        Ok(mode) => mode,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let prompt = match single_completion_prompt(&payload.prompt) {
        Ok(prompt) => prompt,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    if !state.ready.load(Ordering::Acquire) {
        return model_unavailable_response(&state).await;
    }

    let runtime_lease = match state.lease_runtime(payload.model.as_deref()).await {
        Ok(Some(lease)) => lease,
        Ok(None) => return model_unavailable_response(&state).await,
        Err(error) => return requested_model_error_response(error),
    };
    let runtime = Arc::clone(runtime_lease.runtime());
    let model_id = runtime.model_id.clone();
    if model_supports_embeddings(&runtime.pipeline) {
        return embedding_model_generation_response(&model_id);
    }
    let pipeline = Arc::clone(&runtime.pipeline);
    let scheduler_opt = runtime.scheduler.clone();

    let permit = match Arc::clone(&state.semaphore).try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = std::time::Instant::now();

    let core_response_format = match &response_format {
        ResponseFormatMode::Text => bloomai_core::ResponseFormat::Text,
        ResponseFormatMode::JsonObject => bloomai_core::ResponseFormat::JsonObject,
        ResponseFormatMode::JsonSchema(schema) => {
            bloomai_core::ResponseFormat::JsonSchema(schema.clone())
        }
    };

    let params = GenerationParams {
        max_tokens,
        temperature,
        top_p,
        seed: payload.seed,
        response_format: match core_response_format {
            bloomai_core::ResponseFormat::Text => None,
            constrained => Some(constrained),
        },
    };

    let prompt = apply_response_format_instruction(prompt, &response_format);

    let prompt_tokens_vec = match pipeline.tokenize(&prompt) {
        Ok(tokens) => tokens,
        Err(error) => {
            state
                .metrics
                .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "tokenization_error",
                format!("Prompt tokenization failed: {error}"),
            );
        }
    };
    let prompt_tokens = prompt_tokens_vec.len();
    if let Err(message) =
        validate_context_budget(prompt_tokens, params.max_tokens, pipeline.context_size())
    {
        state.metrics.record_request_end(
            false,
            request_start.elapsed().as_secs_f64(),
            0,
            prompt_tokens as u64,
        );
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            message,
        );
    }
    let input = ModelInput::Text { prompt };
    let request_id = next_request_id(&state, "cmpl");
    let cancel_scheduler = if state.enable_ifb {
        scheduler_opt.clone()
    } else {
        None
    };
    let Some(cancel_guard) =
        CancelTokenGuard::register(&state, request_id.clone(), cancel_scheduler)
    else {
        state.metrics.record_request_end(
            false,
            request_start.elapsed().as_secs_f64(),
            0,
            prompt_tokens as u64,
        );
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "request_id_conflict",
            "A request with the same ID is already active.",
        );
    };
    let cancel_token = cancel_guard.token();

    if state.enable_ifb {
        let Some(scheduler) = scheduler_opt else {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "scheduler_unavailable",
                "Continuous batching is enabled, but the scheduler is unavailable.",
            );
        };
        let (tx, mut rx) = mpsc::unbounded_channel::<Result<u32, String>>();
        scheduler
            .token_senders
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id.clone(), tx);

        let req = Request {
            id: request_id.clone(),
            model_id: model_id.clone(),
            prompt_tokens: prompt_tokens_vec,
            generated_tokens: Vec::new(),
            params: params.clone(),
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        if let Err(e) = scheduler.submit_with_execution_guard(req, runtime_lease.execution_guard())
        {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("Failed to submit to scheduler: {}", e),
            );
        }
        // See the chat path above: cancellation may be claimed before the
        // scheduler has a queue entry to remove.
        if cancel_token.is_cancelled() {
            scheduler.cancel_request(&request_id);
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::REQUEST_TIMEOUT,
                "request_cancelled",
                "The generation request was cancelled.",
            );
        }

        if !payload.stream {
            let generated_count = Arc::new(AtomicU64::new(0));
            let lifecycle = InferenceLifecycle::new(
                cancel_guard,
                InferenceLifecycleResources {
                    metrics: Arc::clone(&state.metrics),
                    request_start,
                    generated_tokens: Arc::clone(&generated_count),
                    prompt_tokens: prompt_tokens as u64,
                    permit,
                    runtime_lease: runtime_lease.clone(),
                },
                StreamExecution::Scheduled(Arc::clone(&scheduler)),
            );
            let mut client_guard = lifecycle.client_guard();
            let generated_output = match collect_scheduled_text(
                &mut rx,
                &pipeline,
                &scheduler,
                &request_id,
                &generated_count,
                stop_sequences.clone(),
            )
            .await
            {
                Ok(output) => output,
                Err(error) => {
                    client_guard.finish(false);
                    return error_response(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "internal_error",
                        format!("Scheduler execution failed: {error}"),
                    );
                }
            };
            if cancel_token.is_cancelled() {
                client_guard.finish(false);
                return error_response(
                    axum::http::StatusCode::REQUEST_TIMEOUT,
                    "request_cancelled",
                    "The generation request was cancelled.",
                );
            }
            let generated_text = generated_output.text;
            if let Err(message) = validate_structured_output(&generated_text, &response_format) {
                client_guard.finish(false);
                return error_response(
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    "invalid_response_format",
                    message,
                );
            }
            client_guard.finish(true);
            let completion_tokens = generated_count.load(Ordering::Relaxed) as usize;
            let finish_reason = if generated_output.stopped {
                "stop"
            } else {
                generation_finish_reason(completion_tokens, max_tokens)
            };

            return Json(json!({
                "id": request_id,
                "object": "text_completion",
                "created": unix_seconds(),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "text": generated_text,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }))
            .into_response();
        }

        // Streaming under IFB scheduler
        let req_id = request_id.clone();
        let model = model_id.clone();
        let state_clone = Arc::clone(&state);
        let generated_count = Arc::new(AtomicU64::new(0));
        let first_token_seen = Arc::new(AtomicBool::new(false));
        let last_token_time = Arc::new(std::sync::Mutex::new(None));
        let stream_failed = Arc::new(AtomicBool::new(false));
        let generated_count_for_stream = Arc::clone(&generated_count);
        let first_token_for_stream = Arc::clone(&first_token_seen);
        let last_token_for_stream = Arc::clone(&last_token_time);
        let stream_failed_for_stream = Arc::clone(&stream_failed);

        let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
        let accumulated_text_for_stream = Arc::clone(&accumulated_text);
        let stop_filter = Arc::new(std::sync::Mutex::new(StopSequenceFilter::new(
            stop_sequences.clone(),
        )));
        let stop_filter_for_stream = Arc::clone(&stop_filter);
        let stop_sequence_hit = Arc::new(AtomicBool::new(false));
        let stop_sequence_hit_for_stream = Arc::clone(&stop_sequence_hit);
        let scheduler_for_stop = Arc::clone(&scheduler);
        let request_id_for_stop = request_id.clone();

        let pipeline_for_stream = Arc::clone(&pipeline);
        let sse_stream = UnboundedReceiverStream::new(rx)
            .map(move |item| {
                let chunk = match item {
                    Ok(tok) => {
                        record_stream_tokens(
                            &state_clone,
                            request_start,
                            &first_token_for_stream,
                            &last_token_for_stream,
                            &generated_count_for_stream,
                            1,
                        );
                        let text = pipeline_for_stream.detokenize(&[tok]).unwrap_or_default();
                        let update = {
                            let mut filter = stop_filter_for_stream
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            filter.push(&text)
                        };
                        if update.stopped {
                            stop_sequence_hit_for_stream.store(true, Ordering::Release);
                            scheduler_for_stop.cancel_request(&request_id_for_stop);
                        }
                        {
                            let mut acc = accumulated_text_for_stream
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            acc.push_str(&update.text);
                        }
                        if update.text.is_empty() {
                            return None;
                        }
                        json!({
                            "id": req_id,
                            "object": "text_completion.chunk",
                            "created": unix_seconds(),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": update.text,
                                "finish_reason": null
                            }]
                        })
                    }
                    Err(message) => {
                        stream_failed_for_stream.store(true, Ordering::Relaxed);
                        json!({
                            "error": {
                                "message": message,
                                "type": "internal_error"
                            }
                        })
                    }
                };
                Some(Ok::<Event, std::convert::Infallible>(json_event(chunk)))
            })
            .filter_map(futures::future::ready);

        let stream_failed_for_final = Arc::clone(&stream_failed);
        let stream_failed_for_validation = Arc::clone(&stream_failed);
        let accumulated_text_for_final = Arc::clone(&accumulated_text);
        let response_format_for_final = response_format.clone();
        let stop_sequence_hit_for_final = Arc::clone(&stop_sequence_hit);
        let generated_count_for_final = Arc::clone(&generated_count);
        let final_request_id = request_id.clone();
        let final_model_id = model_id.clone();
        let stop_filter_for_flush = Arc::clone(&stop_filter);
        let accumulated_text_for_flush = Arc::clone(&accumulated_text);
        let stream_failed_for_flush = Arc::clone(&stream_failed);
        let flush_request_id = request_id.clone();
        let flush_model_id = model_id.clone();
        let flush_stream = futures::stream::once(async move {
            if stream_failed_for_flush.load(Ordering::Acquire) {
                return None;
            }
            let tail = stop_filter_for_flush
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .finish();
            if !tail.is_empty() {
                accumulated_text_for_flush
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push_str(&tail);
            }
            (!tail.is_empty()).then(|| {
                Ok::<Event, std::convert::Infallible>(json_event(json!({
                    "id": flush_request_id,
                    "object": "text_completion.chunk",
                    "created": unix_seconds(),
                    "model": flush_model_id,
                    "choices": [{
                        "index": 0,
                        "text": tail,
                        "finish_reason": null
                    }]
                })))
            })
        })
        .filter_map(futures::future::ready);
        let lifecycle = InferenceLifecycle::new(
            cancel_guard,
            InferenceLifecycleResources {
                metrics: Arc::clone(&state.metrics),
                request_start,
                generated_tokens: Arc::clone(&generated_count),
                prompt_tokens: prompt_tokens as u64,
                permit,
                runtime_lease: runtime_lease.clone(),
            },
            StreamExecution::Scheduled(Arc::clone(&scheduler)),
        );
        let mut lifecycle_for_final = lifecycle.client_guard();

        let final_stream = sse_stream
            .chain(flush_stream)
            .chain(futures::stream::once(async move {
                if stream_failed_for_validation.load(Ordering::Acquire) {
                    return None;
                }
                let text = {
                    let acc = accumulated_text_for_final
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    acc.clone()
                };
                if let Err(message) = validate_structured_output(&text, &response_format_for_final) {
                    stream_failed_for_validation.store(true, Ordering::Relaxed);
                    let err_chunk = json!({
                        "error": {
                            "message": format!("Stream structured output validation failed: {}", message),
                            "type": "invalid_response_format"
                        }
                    });
                    Some(Ok::<Event, std::convert::Infallible>(json_event(err_chunk)))
                } else {
                    let finish_reason = if stop_sequence_hit_for_final.load(Ordering::Acquire) {
                        "stop"
                    } else {
                        generation_finish_reason(
                            generated_count_for_final.load(Ordering::Relaxed) as usize,
                            max_tokens,
                        )
                    };
                    Some(Ok::<Event, std::convert::Infallible>(json_event(json!({
                        "id": final_request_id,
                        "object": "text_completion.chunk",
                        "created": unix_seconds(),
                        "model": final_model_id,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "finish_reason": finish_reason
                        }]
                    }))))
                }
            }).filter_map(futures::future::ready))
            .chain(futures::stream::once(async move {
                lifecycle_for_final.finish(!stream_failed_for_final.load(Ordering::Relaxed));
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }));

        return Sse::new(final_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    if !payload.stream {
        let generated_count = Arc::new(AtomicU64::new(0));
        let lifecycle = InferenceLifecycle::new(
            cancel_guard,
            InferenceLifecycleResources {
                metrics: Arc::clone(&state.metrics),
                request_start,
                generated_tokens: Arc::clone(&generated_count),
                prompt_tokens: prompt_tokens as u64,
                permit,
                runtime_lease: runtime_lease.clone(),
            },
            StreamExecution::Blocking,
        );
        let worker_lifecycle = lifecycle.worker_guard();
        let mut client_guard = lifecycle.client_guard();
        let pipeline_for_run = Arc::clone(&pipeline);
        let cancel_token_for_run = cancel_token.clone();
        let stop_sequences_for_run = stop_sequences.clone();
        let inference_start = std::time::Instant::now();
        let res = task::spawn_blocking(move || {
            let _worker_lifecycle = worker_lifecycle;
            run_cancellable_text_inference(
                &pipeline_for_run,
                input,
                &params,
                &cancel_token_for_run,
                stop_sequences_for_run,
            )
        })
        .await;
        state
            .metrics
            .record_inference_latency(inference_start.elapsed().as_secs_f64());

        let generated_output = match res {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                client_guard.finish(false);
                if cancel_token.is_cancelled() {
                    return error_response(
                        axum::http::StatusCode::REQUEST_TIMEOUT,
                        "request_cancelled",
                        "The generation request was cancelled.",
                    );
                }
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Inference failed: {}", e),
                );
            }
            Err(e) => {
                client_guard.finish(false);
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Task join failed: {}", e),
                );
            }
        };

        let generated_text = generated_output.text;
        if let Err(message) = validate_structured_output(&generated_text, &response_format) {
            client_guard.finish(false);
            return error_response(
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_response_format",
                message,
            );
        }
        let completion_tokens = pipeline.tokenize(&generated_text).unwrap_or_default().len();
        generated_count.store(completion_tokens as u64, Ordering::Relaxed);

        client_guard.finish(true);
        let finish_reason = if generated_output.stopped {
            "stop"
        } else {
            generation_finish_reason(completion_tokens, max_tokens)
        };

        return Json(json!({
            "id": request_id,
            "object": "text_completion",
            "created": unix_seconds(),
            "model": model_id.clone(),
            "choices": [{
                "index": 0,
                "text": generated_text,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }))
        .into_response();
    }

    // Streaming
    let (tx, rx) = mpsc::channel::<std::result::Result<String, String>>(100);
    let pipeline_for_stream_run = Arc::clone(&pipeline);
    let cancel_token_clone = cancel_token.clone();
    let stop_sequence_hit = Arc::new(AtomicBool::new(false));
    let stop_sequence_hit_for_worker = Arc::clone(&stop_sequence_hit);
    let stop_sequences_for_stream = stop_sequences.clone();
    let generated_count = Arc::new(AtomicU64::new(0));
    let lifecycle = InferenceLifecycle::new(
        cancel_guard,
        InferenceLifecycleResources {
            metrics: Arc::clone(&state.metrics),
            request_start,
            generated_tokens: Arc::clone(&generated_count),
            prompt_tokens: prompt_tokens as u64,
            permit,
            runtime_lease: runtime_lease.clone(),
        },
        StreamExecution::Blocking,
    );
    let worker_lifecycle = lifecycle.worker_guard();
    task::spawn_blocking(move || {
        let _worker_lifecycle = worker_lifecycle;
        let tx_clone = tx.clone();
        let mut stop_filter = StopSequenceFilter::new(stop_sequences_for_stream);
        let run_res = pipeline_for_stream_run.run_stream(
            input,
            &params,
            &mut |chunk: bloomai_engine::io::OutputChunk| {
                if cancel_token_clone.is_cancelled() {
                    return Err(anyhow!("request cancelled"));
                }
                if let bloomai_engine::io::OutputChunk::TextDelta(text) = chunk {
                    let update = stop_filter.push(&text);
                    if !update.text.is_empty() && tx_clone.blocking_send(Ok(update.text)).is_err() {
                        return Err(anyhow!("client disconnected"));
                    }
                    if update.stopped {
                        stop_sequence_hit_for_worker.store(true, Ordering::Release);
                        return Err(anyhow!("stop sequence reached"));
                    }
                }
                Ok(())
            },
        );
        if run_res.is_ok() {
            let tail = stop_filter.finish();
            if !tail.is_empty() {
                let _ = tx.blocking_send(Ok(tail));
            }
        } else if !stop_sequence_hit_for_worker.load(Ordering::Acquire) {
            let e = run_res.expect_err("failed stream result must contain an error");
            let _ = tx.blocking_send(Err(e.to_string()));
        }
    });

    let req_id = request_id.clone();
    let model = model_id.clone();
    let state_for_stream = Arc::clone(&state);
    let first_token_seen = Arc::new(AtomicBool::new(false));
    let last_token_time = Arc::new(std::sync::Mutex::new(None));
    let stream_failed = Arc::new(AtomicBool::new(false));
    let generated_count_for_stream = Arc::clone(&generated_count);
    let first_token_for_stream = Arc::clone(&first_token_seen);
    let last_token_for_stream = Arc::clone(&last_token_time);
    let stream_failed_for_stream = Arc::clone(&stream_failed);

    let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
    let accumulated_text_for_stream = Arc::clone(&accumulated_text);

    let pipeline_for_stream = Arc::clone(&pipeline);
    let sse_stream = ReceiverStream::new(rx).map(move |item| {
        let chunk = match item {
            Ok(token) => {
                {
                    let mut acc = accumulated_text_for_stream
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    acc.push_str(&token);
                }
                let token_count = estimate_delta_tokens(&pipeline_for_stream, &token);
                record_stream_tokens(
                    &state_for_stream,
                    request_start,
                    &first_token_for_stream,
                    &last_token_for_stream,
                    &generated_count_for_stream,
                    token_count,
                );
                json!({
                    "id": req_id,
                    "object": "text_completion.chunk",
                    "created": unix_seconds(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "text": token,
                        "finish_reason": null
                    }]
                })
            }
            Err(message) => {
                stream_failed_for_stream.store(true, Ordering::Relaxed);
                json!({
                    "error": {
                        "message": message,
                        "type": "internal_error"
                    }
                })
            }
        };
        Ok::<Event, std::convert::Infallible>(json_event(chunk))
    });

    let stream_failed_for_final = Arc::clone(&stream_failed);
    let stream_failed_for_validation = Arc::clone(&stream_failed);
    let accumulated_text_for_final = Arc::clone(&accumulated_text);
    let response_format_for_final = response_format.clone();
    let stop_sequence_hit_for_final = Arc::clone(&stop_sequence_hit);
    let generated_count_for_final = Arc::clone(&generated_count);
    let final_request_id = request_id.clone();
    let final_model_id = model_id.clone();
    let mut lifecycle_for_final = lifecycle.client_guard();

    let final_stream = sse_stream
        .chain(futures::stream::once(async move {
            let text = {
                let acc = accumulated_text_for_final
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                acc.clone()
            };
            if let Err(message) = validate_structured_output(&text, &response_format_for_final) {
                stream_failed_for_validation.store(true, Ordering::Relaxed);
                let err_chunk = json!({
                    "error": {
                        "message": format!("Stream structured output validation failed: {}", message),
                        "type": "invalid_response_format"
                    }
                });
                Ok::<Event, std::convert::Infallible>(json_event(err_chunk))
            } else {
                let finish_reason = if stop_sequence_hit_for_final.load(Ordering::Acquire) {
                    "stop"
                } else {
                    generation_finish_reason(
                        generated_count_for_final.load(Ordering::Relaxed) as usize,
                        max_tokens,
                    )
                };
                Ok::<Event, std::convert::Infallible>(json_event(json!({
                    "id": final_request_id,
                    "object": "text_completion.chunk",
                    "created": unix_seconds(),
                    "model": final_model_id,
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "finish_reason": finish_reason
                    }]
                })))
            }
        }))
        .chain(futures::stream::once(async move {
            lifecycle_for_final.finish(!stream_failed_for_final.load(Ordering::Relaxed));
            Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
        }));

    Sse::new(final_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ─── /v1/embeddings ────────────────────────────────────────────────────────

pub(crate) async fn handle_embeddings(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    if let Err(message) = validate_openai_embedding_request(&payload) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let inputs = match normalize_embedding_input(&payload.input) {
        Ok(inputs) => inputs,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };
    let result = match execute_embedding_batch(
        state,
        payload.model,
        inputs,
        false,
        EmbeddingProjection::L2Normalized {
            dimensions: payload.dimensions,
            require_exact_dimensions: true,
        },
    )
    .await
    {
        Ok(result) => result,
        Err(error) => return error.into_openai_response(),
    };
    let EmbeddingBatchOutput::Embeddings(embeddings) = result.output else {
        return error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "The embedding executor returned an unexpected result type.",
        );
    };
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "embedding": embedding,
                "index": index
            })
        })
        .collect::<Vec<_>>();

    Json(json!({
        "object": "list",
        "data": data,
        "model": result.model_id,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "total_tokens": result.prompt_tokens
        }
    }))
    .into_response()
}

// ─── /v1/rerank ────────────────────────────────────────────────────────────

pub(crate) async fn handle_rerank(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<RerankRequest>,
) -> impl IntoResponse {
    if let Err(message) = validate_rerank_request(&payload) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let mut texts = Vec::with_capacity(payload.documents.len() + 1);
    texts.push(payload.query.clone());
    texts.extend(payload.documents.iter().cloned());
    let result = match execute_embedding_batch(
        Arc::clone(&state),
        payload.model,
        texts,
        false,
        EmbeddingProjection::Rerank {
            top_n: payload.top_n.unwrap_or(payload.documents.len()),
        },
    )
    .await
    {
        Ok(result) => result,
        Err(error) => return error.into_openai_response(),
    };
    let EmbeddingBatchOutput::Rerank(scores) = result.output else {
        return error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "The embedding executor returned an unexpected rerank result type.",
        );
    };
    let include_documents = payload.return_documents.unwrap_or(false);
    let results = scores
        .into_iter()
        .map(|score| {
            let document = payload.documents.get(score.index).ok_or_else(|| {
                "The rerank executor returned a document index outside the request.".to_string()
            })?;
            let mut item = json!({
                "index": score.index,
                "relevance_score": score.relevance_score
            });
            if include_documents {
                item["document"] = json!({"text": document});
            }
            Ok(item)
        })
        .collect::<std::result::Result<Vec<_>, String>>();
    let results = match results {
        Ok(results) => results,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                message,
            );
        }
    };
    let response_id = next_request_id(&state, "rerank");

    Json(json!({
        "id": response_id,
        "object": "rerank",
        "model": result.model_id,
        "results": results,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "total_tokens": result.prompt_tokens
        }
    }))
    .into_response()
}

// ─── /v1/multimodal/stream ──────────────────────────────────────────────────

pub(crate) async fn handle_multimodal_stream(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<InferenceRequest>,
) -> axum::response::Response {
    run_multimodal_request(state, payload, None).await
}

pub(crate) async fn handle_multimodal_upload(
    State(state): State<Arc<ServerState>>,
    mut multipart: Multipart,
) -> axum::response::Response {
    let mut prompt = None;
    let mut image = None;
    let mut requested_model = None;
    let mut params = InferenceParams {
        max_tokens: 512,
        ..InferenceParams::default()
    };

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) => {
                return multipart_error_response("invalid multipart upload", error);
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "prompt" => match field.text().await {
                Ok(value) => prompt = Some(value),
                Err(error) => {
                    return multipart_error_response("failed to read prompt field", error);
                }
            },
            "model" => match field.text().await {
                Ok(value) => requested_model = Some(value),
                Err(error) => {
                    return multipart_error_response("failed to read model field", error);
                }
            },
            "image" => {
                if image.is_some() {
                    return api_error(
                        ApiError::InvalidRequest,
                        "Only one image attachment is supported per request.",
                    );
                }
                let mime = field
                    .content_type()
                    .unwrap_or("application/octet-stream")
                    .to_ascii_lowercase();
                if !matches!(mime.as_str(), "image/jpeg" | "image/png") {
                    return api_error(
                        ApiError::InvalidRequest,
                        "Image attachment must be a JPEG or PNG file.",
                    );
                }
                let bytes = match field.bytes().await {
                    Ok(bytes) if !bytes.is_empty() => bytes.to_vec(),
                    Ok(_) => {
                        return api_error(ApiError::InvalidRequest, "Image attachment is empty.");
                    }
                    Err(error) => {
                        return multipart_error_response("failed to read image attachment", error);
                    }
                };
                if let Err(message) = validate_uploaded_image(&bytes, &mime) {
                    return api_error(ApiError::InvalidRequest, message);
                }
                image = Some(DataBlock::Image { bytes, mime });
            }
            "max_tokens" => match parse_multipart_field::<usize>(field, "max_tokens").await {
                Ok(value) => params.max_tokens = value,
                Err(response) => return response,
            },
            "temperature" => match parse_multipart_field::<f64>(field, "temperature").await {
                Ok(value) => params.temperature = value,
                Err(response) => return response,
            },
            "top_p" => match parse_multipart_field::<f64>(field, "top_p").await {
                Ok(value) => params.top_p = value,
                Err(response) => return response,
            },
            "seed" => match parse_multipart_field::<u64>(field, "seed").await {
                Ok(value) => params.seed = Some(value),
                Err(response) => return response,
            },
            _ => {}
        }
    }

    let Some(image) = image else {
        return api_error(ApiError::InvalidRequest, "An image attachment is required.");
    };
    if !(1..=32_768).contains(&params.max_tokens)
        || !params.temperature.is_finite()
        || !(0.0..=2.0).contains(&params.temperature)
        || !params.top_p.is_finite()
        || !(0.0 < params.top_p && params.top_p <= 1.0)
    {
        return api_error(
            ApiError::InvalidRequest,
            "Invalid generation settings for multimodal upload.",
        );
    }
    let prompt = prompt
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "Describe this image.".to_string());
    let payload = InferenceRequest {
        blocks: vec![DataBlock::Text(prompt), image],
        params,
    };
    run_multimodal_request(state, payload, requested_model).await
}

async fn parse_multipart_field<T>(
    field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> std::result::Result<T, axum::response::Response>
where
    T: std::str::FromStr,
{
    let text = field
        .text()
        .await
        .map_err(|error| multipart_error_response(&format!("failed to read {name}"), error))?;
    text.parse::<T>().map_err(|_| {
        api_error(
            ApiError::InvalidRequest,
            format!("{name} has an invalid value"),
        )
    })
}

fn multipart_error_response(
    context: &str,
    error: axum::extract::multipart::MultipartError,
) -> axum::response::Response {
    let status = error.status();
    let message = if status == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        error.body_text()
    } else {
        format!("{context}: {}", error.body_text())
    };
    error_response(status, "invalid_request_error", message)
}

async fn run_multimodal_request(
    state: Arc<ServerState>,
    payload: InferenceRequest,
    requested_model: Option<String>,
) -> axum::response::Response {
    run_multimodal_request_inner(state, payload, requested_model, None).await
}

/// Run a multimodal request against the exact runtime selected by a protocol
/// adapter. The runtime identity is rechecked while inference admission is
/// held so a concurrent unload cannot silently redirect the request to a
/// newer runtime with the same public model id.
pub(crate) async fn run_multimodal_request_for_runtime(
    state: Arc<ServerState>,
    payload: InferenceRequest,
    runtime: Arc<LoadedRuntime>,
) -> axum::response::Response {
    run_multimodal_request_inner(state, payload, None, Some(runtime)).await
}

async fn run_multimodal_request_inner(
    state: Arc<ServerState>,
    payload: InferenceRequest,
    requested_model: Option<String>,
    exact_runtime: Option<Arc<LoadedRuntime>>,
) -> axum::response::Response {
    if let Err(message) = validate_multimodal_request(&payload) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    if !state.ready.load(Ordering::Acquire) {
        return model_unavailable_response(&state).await;
    }

    let runtime_lease = match exact_runtime {
        Some(runtime) => match state.lease_exact_runtime(&runtime).await {
            Some(lease) => lease,
            None => {
                return error_response(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "model_unavailable",
                    "The selected model was unloaded before inference admission.",
                );
            }
        },
        None => match state.lease_runtime(requested_model.as_deref()).await {
            Ok(Some(lease)) => lease,
            Ok(None) => return model_unavailable_response(&state).await,
            Err(error) => return requested_model_error_response(error),
        },
    };
    let runtime = Arc::clone(runtime_lease.runtime());
    let model_id = runtime.model_id.clone();
    let pipeline = Arc::clone(&runtime.pipeline);
    if let Err(message) = validate_multimodal_modalities(&runtime.input_modalities, &payload.blocks)
    {
        return error_response(
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_modality",
            message,
        );
    }
    let prompt_tokens = match estimate_multimodal_context_tokens(&pipeline, &payload.blocks) {
        Ok(tokens) => tokens,
        Err(message) => {
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "tokenization_error",
                message,
            );
        }
    };
    if let Err(message) = validate_context_budget(
        prompt_tokens,
        payload.params.max_tokens,
        pipeline.context_size(),
    ) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            message,
        );
    }

    let permit = match Arc::clone(&state.semaphore).try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = std::time::Instant::now();
    let request_id = next_request_id(&state, "mms");
    let Some(cancel_guard) = CancelTokenGuard::register(&state, request_id.clone(), None) else {
        state.metrics.record_request_end(
            false,
            request_start.elapsed().as_secs_f64(),
            0,
            u64::try_from(prompt_tokens).unwrap_or(u64::MAX),
        );
        return error_response(
            axum::http::StatusCode::CONFLICT,
            "request_id_conflict",
            "A request with the same ID is already active.",
        );
    };
    let cancel_token = cancel_guard.token();

    let (tx, rx) = mpsc::channel::<std::result::Result<OutputChunk, String>>(100);
    let pipeline_clone = Arc::clone(&pipeline);
    let cancel_token_clone = cancel_token.clone();
    let generated_count = Arc::new(AtomicU64::new(0));
    let lifecycle = InferenceLifecycle::new(
        cancel_guard,
        InferenceLifecycleResources {
            metrics: Arc::clone(&state.metrics),
            request_start,
            generated_tokens: generated_count,
            prompt_tokens: u64::try_from(prompt_tokens).unwrap_or(u64::MAX),
            permit,
            runtime_lease: runtime_lease.clone(),
        },
        StreamExecution::Blocking,
    );
    let worker_lifecycle = lifecycle.worker_guard();
    let runtime_for_worker = Arc::clone(&runtime);

    task::spawn_blocking(move || {
        let _runtime_guard = runtime_for_worker;
        let _worker_lifecycle = worker_lifecycle;
        let tx_clone = tx.clone();
        let run_res = pipeline_clone.run_request(payload, &mut |chunk: OutputChunk| {
            if cancel_token_clone.is_cancelled() {
                return Err(anyhow!("request cancelled"));
            }
            if tx_clone.blocking_send(Ok(chunk)).is_err() {
                return Err(anyhow!("client disconnected"));
            }
            Ok(())
        });
        if let Err(e) = run_res {
            let _ = tx.blocking_send(Err(e.to_string()));
        }
    });

    let stream_failed = Arc::new(AtomicBool::new(false));
    let stream_failed_for_stream = Arc::clone(&stream_failed);
    let req_id_for_stream = request_id.clone();
    let model_id_clone = model_id.clone();
    let created = unix_seconds();

    let sse_stream = ReceiverStream::new(rx).map(move |item| {
        let chunk = match item {
            Ok(out_chunk) => {
                json!({
                    "id": req_id_for_stream.clone(),
                    "object": "multimodal.chunk",
                    "created": created,
                    "model": model_id_clone.clone(),
                    "chunk": out_chunk,
                })
            }
            Err(message) => {
                stream_failed_for_stream.store(true, Ordering::Relaxed);
                json!({
                    "error": {
                        "message": message,
                        "type": "internal_error"
                    }
                })
            }
        };
        Ok::<Event, std::convert::Infallible>(json_event(chunk))
    });

    let stream_failed_for_final = Arc::clone(&stream_failed);
    let start_event = json_event(json!({
        "id": request_id,
        "object": "multimodal.chunk",
        "created": created,
        "model": model_id,
        "chunk": null
    }));
    let mut lifecycle_for_final = lifecycle.client_guard();
    let final_stream =
        futures::stream::once(async move { Ok::<Event, std::convert::Infallible>(start_event) })
            .chain(sse_stream)
            .chain(futures::stream::once(async move {
                lifecycle_for_final.finish(!stream_failed_for_final.load(Ordering::Relaxed));
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }));

    Sse::new(final_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

pub(crate) fn validate_multimodal_request(
    payload: &InferenceRequest,
) -> std::result::Result<(), String> {
    validate_generation_controls(
        payload.params.max_tokens,
        payload.params.temperature,
        payload.params.top_p,
    )?;
    if !matches!(
        payload.params.response_format.as_ref(),
        None | Some(bloomai_core::ResponseFormat::Text)
    ) {
        return Err(
            "Structured response_format modes are supported by text completion endpoints only."
                .to_string(),
        );
    }
    if payload.blocks.is_empty() || payload.blocks.len() > MAX_MULTIMODAL_BLOCKS {
        return Err(format!(
            "Multimodal input must contain between 1 and {MAX_MULTIMODAL_BLOCKS} inline blocks."
        ));
    }

    let mut has_text = false;
    let mut has_audio = false;
    let mut has_image = false;
    for block in &payload.blocks {
        match block {
            DataBlock::Text(text) => {
                if has_text {
                    return Err(
                        "Multimodal input cannot contain duplicate Text blocks.".to_string()
                    );
                }
                has_text = true;
                if text.trim().is_empty() {
                    return Err(
                        "Multimodal Text blocks must not be empty or whitespace-only.".to_string(),
                    );
                }
                if text.chars().count() > MAX_MULTIMODAL_TEXT_CHARS
                    || text.len() > MAX_MULTIMODAL_TEXT_BYTES
                {
                    return Err(format!(
                        "Multimodal text cannot exceed {MAX_MULTIMODAL_TEXT_CHARS} characters or {MAX_MULTIMODAL_TEXT_BYTES} bytes."
                    ));
                }
            }
            DataBlock::AudioPcm {
                samples,
                sample_rate,
            } => {
                if has_audio {
                    return Err(
                        "Multimodal input cannot contain duplicate AudioPcm blocks.".to_string()
                    );
                }
                has_audio = true;
                if !(MIN_MULTIMODAL_AUDIO_SAMPLE_RATE..=MAX_MULTIMODAL_AUDIO_SAMPLE_RATE)
                    .contains(sample_rate)
                {
                    return Err(format!(
                        "Inline audio sample_rate must be between {MIN_MULTIMODAL_AUDIO_SAMPLE_RATE} and {MAX_MULTIMODAL_AUDIO_SAMPLE_RATE} Hz."
                    ));
                }
                let max_samples = (*sample_rate as usize)
                    .checked_mul(MAX_MULTIMODAL_AUDIO_SECONDS)
                    .ok_or_else(|| "Inline audio duration overflowed.".to_string())?
                    .min(MAX_MULTIMODAL_AUDIO_SAMPLES);
                if samples.is_empty() || samples.len() > max_samples {
                    return Err(format!(
                        "Inline audio must contain between 1 and {MAX_MULTIMODAL_AUDIO_SAMPLES} samples and cannot exceed {MAX_MULTIMODAL_AUDIO_SECONDS} seconds at its declared sample rate."
                    ));
                }
                if samples
                    .iter()
                    .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
                {
                    return Err(
                        "Inline audio samples must be finite normalized values between -1 and 1."
                            .to_string(),
                    );
                }
            }
            DataBlock::Image { bytes, mime } => {
                if has_image {
                    return Err(
                        "Multimodal input cannot contain duplicate Image blocks.".to_string()
                    );
                }
                has_image = true;
                if bytes.is_empty() || bytes.len() > MAX_MULTIMODAL_IMAGE_BYTES {
                    return Err(format!(
                        "Inline image data must be between 1 byte and {MAX_MULTIMODAL_IMAGE_BYTES} bytes."
                    ));
                }
                validate_uploaded_image(bytes, mime).map_err(str::to_string)?;
            }
            DataBlock::AudioFile { .. } => {
                return Err(
                    "Public multimodal requests cannot reference server-local audio paths; send bounded inline AudioPcm data instead."
                        .to_string(),
                );
            }
            DataBlock::Tokens(_)
            | DataBlock::VideoFrames(_)
            | DataBlock::Tensor(_)
            | DataBlock::WorldState { .. }
            | DataBlock::Action { .. } => {
                return Err(
                    "Public multimodal requests support only inline Text, AudioPcm, and JPEG or PNG Image blocks."
                        .to_string(),
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_multimodal_modalities(
    input_modalities: &[bloomai_core::Modality],
    blocks: &[DataBlock],
) -> Result<(), &'static str> {
    let supports_multi = input_modalities.contains(&bloomai_core::Modality::Multi);
    if blocks
        .iter()
        .any(|block| matches!(block, DataBlock::Text(_)))
        && !supports_multi
        && !input_modalities.contains(&bloomai_core::Modality::Text)
    {
        return Err("The active model does not declare Text input support.");
    }
    if blocks
        .iter()
        .any(|block| matches!(block, DataBlock::AudioPcm { .. }))
        && !supports_multi
        && !input_modalities.contains(&bloomai_core::Modality::Audio)
    {
        return Err("The active model does not declare Audio input support.");
    }
    if blocks
        .iter()
        .any(|block| matches!(block, DataBlock::Image { .. }))
        && !supports_multi
        && !input_modalities.contains(&bloomai_core::Modality::Vision)
    {
        return Err("The active model does not declare Vision input support.");
    }
    Ok(())
}

fn estimate_multimodal_context_tokens(
    pipeline: &InferencePipeline,
    blocks: &[DataBlock],
) -> std::result::Result<usize, String> {
    let mut tokens = 0_usize;
    for block in blocks {
        match block {
            DataBlock::Text(text) => {
                let text_tokens = pipeline
                    .tokenize(text)
                    .map_err(|error| format!("Prompt tokenization failed: {error}"))?
                    .len();
                tokens = tokens
                    .checked_add(text_tokens)
                    .ok_or_else(|| "Multimodal context token count overflowed.".to_string())?;
            }
            DataBlock::Image { bytes, mime } => {
                let (width, height) = uploaded_image_dimensions(bytes, mime)
                    .map_err(|message| message.to_string())?;
                let vision_tokens = estimate_multimodal_visual_tokens(width, height)?;
                tokens = tokens
                    .checked_add(vision_tokens)
                    .ok_or_else(|| "Multimodal context token count overflowed.".to_string())?;
            }
            _ => {}
        }
    }
    Ok(tokens)
}

pub(crate) fn estimate_multimodal_visual_tokens(
    width: u32,
    height: u32,
) -> std::result::Result<usize, String> {
    const VISION_RESIZE_ALIGNMENT: usize = 32;
    if width == 0 || height == 0 {
        return Err("Image dimensions must be non-zero.".to_string());
    }
    let align = |dimension: u32| {
        usize::try_from(dimension)
            .map_err(|_| "Image dimension exceeded this platform.".to_string())?
            .checked_add(VISION_RESIZE_ALIGNMENT - 1)
            .map(|value| value / VISION_RESIZE_ALIGNMENT * VISION_RESIZE_ALIGNMENT)
            .ok_or_else(|| "Aligned image dimension overflowed this platform.".to_string())
    };
    let aligned_pixels = align(width)?
        .checked_mul(align(height)?)
        .ok_or_else(|| "Aligned image pixel count overflowed this platform.".to_string())?;
    let resized_pixels =
        aligned_pixels.clamp(MIN_MULTIMODAL_VISION_PIXELS, MAX_MULTIMODAL_VISION_PIXELS);
    resized_pixels
        .div_ceil(MULTIMODAL_VISION_PIXELS_PER_TOKEN)
        .checked_add(MULTIMODAL_VISION_TOKEN_OVERHEAD)
        .ok_or_else(|| "Multimodal visual token estimate overflowed.".to_string())
}

pub(crate) fn validate_uploaded_image(bytes: &[u8], mime: &str) -> Result<(), &'static str> {
    let expected = match mime {
        "image/jpeg" => image::ImageFormat::Jpeg,
        "image/png" => image::ImageFormat::Png,
        _ => return Err("Image attachment must be a JPEG or PNG file."),
    };
    match image::guess_format(bytes) {
        Ok(actual) if actual == expected => Ok(()),
        Ok(_) => Err("Image content does not match its declared media type."),
        Err(_) => Err("Image attachment has an invalid or unsupported file signature."),
    }?;
    let (width, height) = uploaded_image_dimensions(bytes, mime)?;
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or("Image dimensions overflow the supported pixel count.")?;
    if width == 0 || height == 0 || pixels > MAX_MULTIMODAL_IMAGE_PIXELS {
        return Err("Image dimensions exceed the supported pixel count.");
    }
    if width > MAX_MULTIMODAL_IMAGE_DIMENSION || height > MAX_MULTIMODAL_IMAGE_DIMENSION {
        return Err("Image dimensions exceed the supported width or height.");
    }
    let shorter = width.min(height);
    let longer = width.max(height);
    if u64::from(longer)
        > u64::from(shorter).saturating_mul(u64::from(MAX_MULTIMODAL_IMAGE_ASPECT_RATIO))
    {
        return Err("Image aspect ratio exceeds the supported limit.");
    }
    Ok(())
}

fn uploaded_image_dimensions(bytes: &[u8], mime: &str) -> Result<(u32, u32), &'static str> {
    let format = match mime {
        "image/jpeg" => image::ImageFormat::Jpeg,
        "image/png" => image::ImageFormat::Png,
        _ => return Err("Image attachment must be a JPEG or PNG file."),
    };
    image::io::Reader::with_format(std::io::Cursor::new(bytes), format)
        .into_dimensions()
        .map_err(|_| "Image attachment has an invalid or incomplete header.")
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorldCacheConfig {
    pub max_bytes: Option<usize>,
    pub max_entries: Option<usize>,
    pub default_ttl_ms: Option<u64>,
    pub auto_compress: Option<bool>,
    pub compress_after_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorldStepRequest {
    pub observations: Vec<bloomai_core::WorldObservation>,
    #[serde(default = "default_horizon")]
    pub horizon: u32,
    pub state_schema: Option<bloomai_engine::WorldStateSchema>,
    pub action_schema: Option<bloomai_engine::ActionSchema>,
    pub cache_config: Option<WorldCacheConfig>,
    pub thermal_state: Option<bloomai_core::ThermalState>,
    pub power_state: Option<bloomai_core::PowerState>,
    #[serde(default)]
    pub stream: bool,
}

pub(crate) fn default_horizon() -> u32 {
    1
}

pub(crate) async fn handle_world_step(
    State(_state): State<Arc<ServerState>>,
    Json(payload): Json<WorldStepRequest>,
) -> impl IntoResponse {
    let wm = Box::new(bloomai_engine::MockWorldModel::new("mock-world-model"));
    let policy = Box::new(bloomai_engine::MockPolicyEngine::new(
        "mock-policy",
        "robot_velocity",
        2,
    ));

    let mut cache_config = bloomai_core::StateCacheConfig::default();
    if let Some(ref c) = payload.cache_config {
        if let Some(mb) = c.max_bytes {
            cache_config.max_bytes = mb;
        }
        if let Some(me) = c.max_entries {
            cache_config.max_entries = me;
        }
        if let Some(ttl) = c.default_ttl_ms {
            cache_config.default_ttl_ms = ttl;
        }
        if let Some(ac) = c.auto_compress {
            cache_config.auto_compress = ac;
        }
        if let Some(cam) = c.compress_after_ms {
            cache_config.compress_after_ms = cam;
        }
    }

    let mut world_loop = bloomai_engine::WorldModelLoop::new(wm, policy, cache_config);
    world_loop.set_schemas(payload.state_schema, payload.action_schema);

    if let (Some(t), Some(p)) = (payload.thermal_state, payload.power_state) {
        world_loop.set_environment(t, p);
    }

    let chunks = match world_loop.step(payload.observations, payload.horizon) {
        Ok(c) => c,
        Err(e) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_observation_or_action",
                e.to_string(),
            );
        }
    };

    if payload.stream {
        // SSE stream response
        let (tx, rx) = mpsc::channel::<
            std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
        >(10);

        tokio::spawn(async move {
            for chunk in chunks {
                let event = json_event(&chunk);
                if tx.send(Ok(event)).await.is_err() {
                    break;
                }
            }
        });

        Sse::new(ReceiverStream::new(rx))
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    } else {
        Json(chunks).into_response()
    }
}

// ─── /v1/cancel/:request_id ──────────────────────────────────────────────

pub(crate) async fn handle_cancel(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(request_id): axum::extract::Path<String>,
) -> axum::response::Response {
    if let Err(message) = validate_request_id(&request_id) {
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        );
    }
    let (registration, cancelled) = {
        let registrations = state
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(registration) = registrations.get(&request_id).cloned() {
            if registration
                .cancelling
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                registration.token.cancel();
                (Some(registration), true)
            } else {
                (None, true)
            }
        } else {
            (None, false)
        }
    };
    if let Some(registration) = registration {
        if let Some(scheduler) = &registration.scheduler {
            scheduler.cancel_request(&request_id);
        }
        let mut registrations = state
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if registrations
            .get(&request_id)
            .is_some_and(|active| Arc::ptr_eq(active, &registration))
        {
            registrations.remove(&request_id);
        }
    }

    if cancelled {
        Json(json!({
            "id": request_id,
            "object": "cancellation",
            "cancelled": true
        }))
        .into_response()
    } else {
        Json(json!({
            "id": request_id,
            "object": "cancellation",
            "cancelled": false,
            "error": "request not found or already completed"
        }))
        .into_response()
    }
}

// ─── /v1/backends ───────────────────────────────────────────────────────────

pub(crate) async fn handle_backends(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let runtime = state.runtime_pool.read().await.default_runtime();
    let model_id = runtime
        .as_ref()
        .map(|runtime| runtime.model_id.clone())
        .unwrap_or_default();
    let registry = engine_registry();

    let backends: Vec<serde_json::Value> = registry
        .iter()
        .map(|(name, engine)| {
            let cap = engine.capability();
            json!({
                "name": name,
                "maturity": cap.maturity.to_string(),
                "supported_families": cap.supported_families.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>(),
                "supported_formats": cap.supported_formats.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>(),
                "supported_devices": cap.supported_devices.iter().map(|d| format!("{:?}", d)).collect::<Vec<_>>(),
                "supports_streaming": cap.supports_streaming,
                "supports_quantized_models": cap.supports_quantized_models,
                "supports_embeddings": cap.supports_embeddings,
                "supports_rerank": cap.supports_rerank,
                "supports_structured_output": cap.supports_structured_output,
                "max_context_tokens": cap.max_context_tokens,
                "diagnostic_tips": cap.diagnostic_tips,
                "construction_guide": cap.construction_guide,
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": backends,
        "active_model": model_id,
    }))
}
