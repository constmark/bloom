//! Exact, bounded tokenizer operations for local clients. These are Bloom
//! extensions, not OpenAI endpoints, and never expose approximate token IDs.
use super::*;
use axum::extract::RawQuery;
use axum::http::StatusCode;

pub(crate) const MAX_TOKENIZATION_BODY_BYTES: usize = 1024 * 1024;
const MAX_TEXT_BYTES: usize = 256 * 1024;
const MAX_TOKEN_IDS: usize = 65_536;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TokenizeRequest {
    model: Option<String>,
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DetokenizeRequest {
    model: Option<String>,
    tokens: Vec<u32>,
}

enum Operation {
    Encode(String),
    Decode(Vec<u32>),
}

enum WorkerError {
    Unsupported,
    InvalidOutput,
    Failed,
    Cancelled,
}

pub(crate) async fn handle_tokenize(
    State(state): State<Arc<ServerState>>,
    RawQuery(query): RawQuery,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    if query.is_some() || request.text.len() > MAX_TEXT_BYTES {
        return invalid_request(
            "Tokenization accepts no query parameters and at most 262144 UTF-8 input bytes.",
        );
    }
    execute(state, request.model, Operation::Encode(request.text)).await
}

pub(crate) async fn handle_detokenize(
    State(state): State<Arc<ServerState>>,
    RawQuery(query): RawQuery,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    if query.is_some() || request.tokens.len() > MAX_TOKEN_IDS {
        return invalid_request(
            "Detokenization accepts no query parameters and at most 65536 unsigned 32-bit token IDs.",
        );
    }
    execute(state, request.model, Operation::Decode(request.tokens)).await
}

fn invalid_request(message: &'static str) -> Response {
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

async fn execute(state: Arc<ServerState>, model: Option<String>, operation: Operation) -> Response {
    // Reject malformed selectors even if the server is currently unloaded.
    if model
        .as_deref()
        .is_some_and(|model| validate_model_selector(model).is_err())
    {
        return invalid_request(
            "The model field must contain 1 to 256 characters without surrounding whitespace or control characters.",
        );
    }
    if !state.app.ready.load(Ordering::Acquire) {
        let (kind, message) = state.model_unavailable().await;
        return error_response(StatusCode::SERVICE_UNAVAILABLE, kind, message);
    }
    let runtime_lease = match state.app.lease_runtime(model.as_deref()).await {
        Ok(Some(lease)) => lease,
        Ok(None) => {
            let (kind, message) = state.model_unavailable().await;
            return error_response(StatusCode::SERVICE_UNAVAILABLE, kind, message);
        }
        Err(RequestedModelError::Invalid) => return invalid_request("Invalid model selector."),
        Err(RequestedModelError::NotLoaded) => {
            return error_response(
                StatusCode::NOT_FOUND,
                "model_not_found",
                "The requested model is not loaded.",
            );
        }
        Err(RequestedModelError::Revoked) => {
            return error_response(
                StatusCode::GONE,
                "model_version_revoked",
                "The loaded model version has been revoked.",
            );
        }
    };
    let model_id = runtime_lease.runtime().model_id.clone();
    let pipeline = Arc::clone(&runtime_lease.runtime().pipeline);
    let permit = match Arc::clone(&state.app.semaphore).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };
    let request_id = next_request_id(&state, "tokenize");
    let Some(registration) = state.app.cancellations.register(request_id.clone(), None) else {
        return error_response(
            StatusCode::CONFLICT,
            "request_id_conflict",
            "A request with the same ID is already active.",
        );
    };
    let cancellation = registration.token();
    state.app.metrics.record_request_start();
    let lifecycle = InferenceLifecycle::new(
        registration,
        InferenceLifecycleResources {
            metrics: Arc::clone(&state.app.metrics),
            request_start: Instant::now(),
            generated_tokens: Arc::new(AtomicU64::new(0)),
            // Tokenizer utilities do not perform prompt/generation inference.
            prompt_tokens: 0,
            permit,
            runtime_lease,
        },
        StreamExecution::Blocking,
    );
    let worker_guard = lifecycle.worker_guard();
    let mut client_guard = lifecycle.client_guard();
    let worker_cancellation = cancellation.clone();
    let worker = task::spawn_blocking(move || {
        let _worker_guard = worker_guard;
        if worker_cancellation.is_cancelled() {
            return Err(WorkerError::Cancelled);
        }
        let output = match operation {
            Operation::Encode(text) => {
                let tokens = pipeline
                    .tokenize_exact(&text)
                    .ok_or(WorkerError::Unsupported)?
                    .map_err(|_| WorkerError::Failed)?;
                if tokens.len() > MAX_TOKEN_IDS {
                    return Err(WorkerError::InvalidOutput);
                }
                json!({"schema_version": 1, "object": "bloom.tokenization", "id": request_id,
                    "model": model_id, "count": tokens.len(), "tokens": tokens})
            }
            Operation::Decode(tokens) => {
                let text = pipeline
                    .detokenize_exact(&tokens)
                    .ok_or(WorkerError::Unsupported)?
                    .map_err(|_| WorkerError::Failed)?;
                if text.len() > MAX_TEXT_BYTES {
                    return Err(WorkerError::InvalidOutput);
                }
                json!({"schema_version": 1, "object": "bloom.detokenization", "id": request_id,
                    "model": model_id, "count": tokens.len(), "text": text})
            }
        };
        if worker_cancellation.is_cancelled() {
            return Err(WorkerError::Cancelled);
        }
        Ok(output)
    });
    let result = tokio::select! {
        biased;
        _ = cancellation.cancelled() => Ok(Err(WorkerError::Cancelled)),
        result = worker => result,
    };
    match result {
        Ok(Ok(output)) => {
            client_guard.finish(true);
            Json(output).into_response()
        }
        result => {
            client_guard.finish(false);
            match result {
                Ok(Err(WorkerError::Unsupported)) => error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "unsupported_operation",
                    "The loaded model does not expose this exact tokenizer operation.",
                ),
                Ok(Err(WorkerError::InvalidOutput)) => error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "invalid_tokenizer_output",
                    "Tokenizer output exceeded the public token or text limit.",
                ),
                Ok(Err(WorkerError::Failed)) => error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "tokenization_failed",
                    "The exact tokenizer could not process this input; token IDs must belong to the loaded tokenizer.",
                ),
                Ok(Err(WorkerError::Cancelled)) => error_response(
                    StatusCode::REQUEST_TIMEOUT,
                    "request_cancelled",
                    "The tokenizer request was cancelled.",
                ),
                Err(_) => error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "The tokenizer worker failed.",
                ),
                Ok(Ok(_)) => unreachable!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloomai_core::{DeviceKind, Modality};
    use bloomai_engine::{Engine, LoadedModel, ModelMetadata, ModelOutput};
    use tower::ServiceExt as _;

    #[derive(Default)]
    struct Blocker {
        released: std::sync::Mutex<bool>,
        changed: std::sync::Condvar,
    }
    impl Blocker {
        fn wait(&self) {
            drop(
                self.changed
                    .wait_timeout_while(
                        self.released.lock().unwrap(),
                        Duration::from_secs(5),
                        |released| !*released,
                    )
                    .unwrap(),
            );
        }
        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.changed.notify_all();
        }
    }

    struct TokenizerEngine {
        exact: bool,
        blocked: Option<Arc<Blocker>>,
        entered: Arc<AtomicBool>,
    }
    struct TokenizerModel {
        metadata: ModelMetadata,
        exact: bool,
        blocked: Option<Arc<Blocker>>,
        entered: Arc<AtomicBool>,
    }
    impl Engine for TokenizerEngine {
        fn name(&self) -> &'static str {
            "tokenizer-fixture"
        }
        fn supported_modalities(&self) -> Vec<Modality> {
            vec![Modality::Text]
        }
        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }
        fn load(&self, _: &Path, _: DeviceKind) -> Result<Box<dyn LoadedModel>> {
            Ok(Box::new(TokenizerModel {
                metadata: ModelMetadata {
                    id: "exact-tokenizer".into(),
                    modality: Modality::Text,
                    quantized: false,
                    manifest: bloomai_core::ModelManifest::default(),
                },
                exact: self.exact,
                blocked: self.blocked.clone(),
                entered: Arc::clone(&self.entered),
            }))
        }
    }
    impl LoadedModel for TokenizerModel {
        fn metadata(&self) -> &ModelMetadata {
            &self.metadata
        }
        fn infer(&self, _: ModelInput, _: &GenerationParams) -> Result<ModelOutput> {
            anyhow::bail!("tokenizer endpoint must never perform inference")
        }
        fn tokenize(&self, text: &str) -> Option<Result<Vec<u32>>> {
            if !self.exact {
                return None;
            }
            self.entered.store(true, Ordering::Release);
            if let Some(barrier) = &self.blocked {
                barrier.wait();
            }
            if text == "oversized" {
                return Some(Ok(vec![1; MAX_TOKEN_IDS + 1]));
            }
            if text == "failure" {
                return Some(Err(anyhow!("private backend detail")));
            }
            Some(Ok(text.chars().map(u32::from).collect()))
        }
        fn detokenize(&self, tokens: &[u32]) -> Option<Result<String>> {
            if !self.exact {
                return None;
            }
            Some(
                tokens
                    .iter()
                    .map(|id| char::from_u32(*id).ok_or_else(|| anyhow!("private invalid ID")))
                    .collect(),
            )
        }
    }
    async fn fixture(
        exact: bool,
        blocked: Option<Arc<Blocker>>,
        entered: Arc<AtomicBool>,
    ) -> (tempfile::TempDir, Arc<ServerState>, Router) {
        let dir = tempfile::tempdir().unwrap();
        let (state, _receiver) = crate::tests::test_server_state(dir.path().to_path_buf());
        let runtime = crate::tests::test_text_runtime_with_engine(
            dir.path().join("tokenizer.fixture"),
            &TokenizerEngine {
                exact,
                blocked,
                entered,
            },
        );
        state
            .app
            .runtime_pool
            .write()
            .await
            .publish_default(runtime);
        state.app.ready.store(true, Ordering::Release);
        state.app.load_progress.store(100, Ordering::Release);
        let router = Router::new()
            .route("/tokenize", post(handle_tokenize))
            .route("/detokenize", post(handle_detokenize))
            .layer(DefaultBodyLimit::max(MAX_TOKENIZATION_BODY_BYTES))
            .with_state(Arc::clone(&state));
        (dir, state, router)
    }
    async fn call(
        router: Router,
        uri: &str,
        input: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let response = router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(input.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * MAX_TOKENIZATION_BODY_BYTES)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }
    #[tokio::test]
    async fn exact_round_trip_empty_input_and_model_binding() {
        let (_dir, state, router) = fixture(true, None, Arc::new(AtomicBool::new(false))).await;
        let (status, body) = call(
            router.clone(),
            "/tokenize",
            json!({"model":"default","text":"Hi🌸"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model"], "exact-tokenizer");
        assert_eq!(body["tokens"], json!([72, 105, 127800]));
        assert_eq!(body["count"], 3);
        let (status, decoded) = call(
            router.clone(),
            "/detokenize",
            json!({"tokens": body["tokens"]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(decoded["text"], "Hi🌸");
        assert_eq!(
            call(router.clone(), "/tokenize", json!({"text":""}))
                .await
                .1["count"],
            0
        );
        assert_eq!(
            call(router.clone(), "/detokenize", json!({"tokens":[]}))
                .await
                .1["text"],
            ""
        );
        assert_eq!(
            call(
                router.clone(),
                "/tokenize",
                json!({"model":"other","text":"hi"})
            )
            .await
            .0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            call(router, "/tokenize", json!({"model":" bad ","text":"hi"}))
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            state.app.metrics.requests_completed.load(Ordering::Relaxed),
            4
        );
        assert_eq!(
            state
                .app
                .metrics
                .tokens_generated_total
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            state.app.metrics.in_flight_requests.load(Ordering::Relaxed),
            0
        );
    }
    #[tokio::test]
    async fn bounds_unknown_fields_errors_and_no_approximation() {
        let (_dir, _, router) = fixture(true, None, Arc::new(AtomicBool::new(false))).await;
        for (uri, input, status) in [
            (
                "/tokenize?extra=1",
                json!({"text":"hi"}),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/tokenize",
                json!({"text":"hi","add_special":false}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/tokenize",
                json!({"text":"x".repeat(MAX_TEXT_BYTES+1)}),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/detokenize",
                json!({"tokens":vec![1; MAX_TOKEN_IDS+1]}),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/detokenize",
                json!({"tokens":[-1]}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/detokenize",
                json!({"tokens":[u32::MAX]}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/tokenize",
                json!({"text":"oversized"}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/tokenize",
                json!({"text":"failure"}),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
        ] {
            let (actual, body) = call(router.clone(), uri, input).await;
            assert_eq!(actual, status, "{uri}");
            assert!(!body.to_string().contains("private"));
        }
        let (_dir, _, unsupported) = fixture(false, None, Arc::new(AtomicBool::new(false))).await;
        let (status, body) = call(unsupported, "/tokenize", json!({"text":"two words"})).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["error"]["type"], "unsupported_operation");
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disconnect_retains_runtime_and_capacity_until_worker_settles() {
        let barrier = Arc::new(Blocker::default());
        let entered = Arc::new(AtomicBool::new(false));
        let (_dir, state, router) =
            fixture(true, Some(Arc::clone(&barrier)), Arc::clone(&entered)).await;
        let initial = state.app.semaphore.available_permits();
        let runtime = state
            .app
            .runtime_pool
            .read()
            .await
            .default_runtime()
            .unwrap();
        let task = tokio::spawn(call(router, "/tokenize", json!({"text":"waiting"})));
        tokio::time::timeout(Duration::from_secs(5), async {
            while !entered.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        let _ = task.await;
        assert_eq!(state.app.semaphore.available_permits(), initial - 1);
        assert_eq!(runtime.active_request_leases.load(Ordering::Acquire), 1);
        assert_eq!(
            state.app.metrics.in_flight_requests.load(Ordering::Relaxed),
            1
        );
        barrier.release();
        tokio::time::timeout(Duration::from_secs(5), async {
            while state.app.metrics.in_flight_requests.load(Ordering::Relaxed) != 0
                || state.app.semaphore.available_permits() != initial
                || runtime.active_request_leases.load(Ordering::Acquire) != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(state.app.semaphore.available_permits(), initial);
        assert_eq!(runtime.active_request_leases.load(Ordering::Acquire), 0);
        assert_eq!(state.app.metrics.requests_failed.load(Ordering::Relaxed), 1);
    }
}
