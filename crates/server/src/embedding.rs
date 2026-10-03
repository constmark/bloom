//! HTTP-facing embedding execution and response projection.
//!
//! The bounded, protocol-neutral preparation and vector algorithms live in
//! [`crate::application::embedding`]. This module owns only runtime admission,
//! cancellation, metrics, and translating service outcomes for HTTP adapters.

use super::*;
use crate::application::embedding::{
    EmbeddingBatchOutput, EmbeddingProjection, MAX_EMBEDDING_DIMENSIONS,
    MAX_NATIVE_EMBEDDING_MICROBATCH_ITEMS, collect_embedding, normalize_embedding_batch,
    prepare_embedding_inputs, rank_embedding_documents, validate_embedding_output,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};

#[derive(Debug)]
pub(crate) struct EmbeddingBatchResult {
    pub(crate) model_id: String,
    pub(crate) output: EmbeddingBatchOutput,
    pub(crate) prompt_tokens: usize,
    pub(crate) total_duration: Duration,
}

#[derive(Debug)]
pub(crate) struct EmbeddingExecutionError {
    pub(crate) status: axum::http::StatusCode,
    pub(crate) error_type: &'static str,
    pub(crate) message: String,
}

impl EmbeddingExecutionError {
    fn new(
        status: axum::http::StatusCode,
        error_type: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            error_type,
            message: message.into(),
        }
    }

    pub(crate) fn into_openai_response(self) -> axum::response::Response {
        error_response(self.status, self.error_type, self.message)
    }
}

#[derive(Debug)]
enum EmbeddingWorkerError {
    Cancelled,
    InvalidRequest(String),
    Inference(String),
    InvalidOutput(String),
}

pub(crate) fn validate_openai_embedding_request(
    request: &EmbeddingRequest,
) -> std::result::Result<(), String> {
    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| !matches!(format, "float" | "base64"))
    {
        return Err("encoding_format must be either 'float' or 'base64'.".to_string());
    }
    if request
        .dimensions
        .is_some_and(|dimensions| !(1..=MAX_EMBEDDING_DIMENSIONS).contains(&dimensions))
    {
        return Err(format!(
            "dimensions must be between 1 and {MAX_EMBEDDING_DIMENSIONS}."
        ));
    }
    if let Some(user) = request.user.as_deref()
        && (user.is_empty() || user.chars().count() > 256 || user.chars().any(char::is_control))
    {
        return Err(
            "user must contain between 1 and 256 characters without control characters."
                .to_string(),
        );
    }
    if let Some(field) = request
        .extensions
        .iter()
        .find(|(_, value)| !value.is_null())
        .map(|(field, _)| reported_extension_field(field))
    {
        return Err(format!(
            "Embedding request contains unsupported non-neutral field {field}. Bloom rejects unsupported request semantics instead of silently ignoring them."
        ));
    }
    Ok(())
}

/// Encode an embedding vector using OpenAI's `encoding_format: "base64"`
/// representation. The wire format is the contiguous little-endian IEEE-754
/// binary32 representation of each value, wrapped in standard padded base64.
/// Keeping this conversion here makes the HTTP adapter's encoding explicit and
/// avoids relying on the host's native float byte order.
pub(crate) fn encode_embedding_base64(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(std::mem::size_of::<f32>()));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BASE64_STANDARD.encode(bytes)
}

pub(crate) async fn execute_embedding_batch(
    state: Arc<ServerState>,
    requested_model: Option<String>,
    inputs: Vec<String>,
    truncate_inputs: bool,
    projection: EmbeddingProjection,
) -> std::result::Result<EmbeddingBatchResult, EmbeddingExecutionError> {
    execute_embedding_batch_inner(
        state,
        requested_model,
        None,
        inputs,
        truncate_inputs,
        projection,
    )
    .await
}

/// Execute embeddings against the exact runtime generation selected by an
/// internal protocol adapter. This prevents an unload/reload with the same
/// public model id from redirecting the request after adapter activation.
pub(crate) async fn execute_embedding_batch_for_runtime(
    state: Arc<ServerState>,
    runtime: Arc<LoadedRuntime>,
    inputs: Vec<String>,
    truncate_inputs: bool,
    projection: EmbeddingProjection,
) -> std::result::Result<EmbeddingBatchResult, EmbeddingExecutionError> {
    execute_embedding_batch_inner(
        state,
        None,
        Some(runtime),
        inputs,
        truncate_inputs,
        projection,
    )
    .await
}

async fn execute_embedding_batch_inner(
    state: Arc<ServerState>,
    requested_model: Option<String>,
    exact_runtime: Option<Arc<LoadedRuntime>>,
    inputs: Vec<String>,
    truncate_inputs: bool,
    projection: EmbeddingProjection,
) -> std::result::Result<EmbeddingBatchResult, EmbeddingExecutionError> {
    if !state.app.ready.load(Ordering::Acquire) {
        let (error_type, message) = state.model_unavailable().await;
        return Err(EmbeddingExecutionError::new(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            error_type,
            message,
        ));
    }

    let runtime_lease = match exact_runtime {
        Some(runtime) if state.app.runtime_is_revoked(&runtime) => {
            return Err(EmbeddingExecutionError::new(
                axum::http::StatusCode::GONE,
                "model_version_revoked",
                "The loaded signed-index model version has been permanently revoked. Install a replacement with a different digest before retrying.",
            ));
        }
        Some(runtime) => match state.app.lease_exact_runtime(&runtime).await {
            Some(runtime_lease) => runtime_lease,
            None => {
                return Err(EmbeddingExecutionError::new(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "model_unavailable",
                    "The selected model was unloaded before inference admission.",
                ));
            }
        },
        None => match state.app.lease_runtime(requested_model.as_deref()).await {
            Ok(Some(runtime_lease)) => runtime_lease,
            Ok(None) => {
                let (error_type, message) = state.model_unavailable().await;
                return Err(EmbeddingExecutionError::new(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    error_type,
                    message,
                ));
            }
            Err(RequestedModelError::Invalid) => {
                return Err(EmbeddingExecutionError::new(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "The model field must contain 1 to 256 characters without surrounding whitespace or control characters.",
                ));
            }
            Err(RequestedModelError::NotLoaded) => {
                return Err(EmbeddingExecutionError::new(
                    axum::http::StatusCode::NOT_FOUND,
                    "model_not_found",
                    "The requested model is not loaded. Query the model discovery endpoint or switch the active runtime before retrying.",
                ));
            }
            Err(RequestedModelError::Revoked) => {
                return Err(EmbeddingExecutionError::new(
                    axum::http::StatusCode::GONE,
                    "model_version_revoked",
                    "The loaded signed-index model version has been permanently revoked. Install a replacement with a different digest before retrying.",
                ));
            }
        },
    };
    let runtime = runtime_lease.runtime();
    let model_id = runtime.model_id.clone();
    let pipeline = Arc::clone(&runtime.pipeline);
    if !model_supports_embeddings(&pipeline) {
        return Err(EmbeddingExecutionError::new(
            axum::http::StatusCode::NOT_IMPLEMENTED,
            "unsupported_operation",
            format!(
                "Model '{model_id}' does not advertise embedding/rerank support. Load a supported encoder model or declare bloom_task=embedding in its trusted manifest metadata."
            ),
        ));
    }
    let (inputs, prompt_tokens) = prepare_embedding_inputs(&pipeline, inputs, truncate_inputs)
        .map_err(|message| {
            EmbeddingExecutionError::new(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            )
        })?;
    let permit = Arc::clone(&state.app.semaphore)
        .try_acquire_owned()
        .map_err(|_| {
            EmbeddingExecutionError::new(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            )
        })?;

    state.app.metrics.record_request_start();
    let request_start = Instant::now();
    let request_id = next_request_id(&state, "embed");
    let Some(cancel_guard) = state.app.cancellations.register(request_id, None) else {
        state.app.metrics.record_request_end(
            false,
            request_start.elapsed().as_secs_f64(),
            0,
            u64::try_from(prompt_tokens).unwrap_or(u64::MAX),
        );
        return Err(EmbeddingExecutionError::new(
            axum::http::StatusCode::CONFLICT,
            "request_id_conflict",
            "A request with the same ID is already active.",
        ));
    };
    let cancel_token = cancel_guard.token();
    let lifecycle = InferenceLifecycle::new(
        cancel_guard,
        InferenceLifecycleResources {
            metrics: Arc::clone(&state.app.metrics),
            request_start,
            generated_tokens: Arc::new(AtomicU64::new(0)),
            prompt_tokens: u64::try_from(prompt_tokens).unwrap_or(u64::MAX),
            permit,
            runtime_lease,
        },
        StreamExecution::Blocking,
    );
    let worker_guard = lifecycle.worker_guard();
    let mut client_guard = lifecycle.client_guard();
    let worker_token = cancel_token.clone();
    let inference_started = Instant::now();
    let worker = task::spawn_blocking(move || {
        let _worker_guard = worker_guard;
        let mut embeddings = Vec::with_capacity(inputs.len());
        let mut expected_dimensions = None;
        let mut total_values = 0_usize;
        if pipeline.supports_native_embedding_batch() {
            for batch in inputs.chunks(MAX_NATIVE_EMBEDDING_MICROBATCH_ITEMS) {
                if worker_token.is_cancelled() {
                    return Err(EmbeddingWorkerError::Cancelled);
                }
                let batch_embeddings = pipeline
                    .run_embedding_batch(batch)
                    .map_err(|error| EmbeddingWorkerError::Inference(error.to_string()))?;
                if batch_embeddings.len() != batch.len() {
                    return Err(EmbeddingWorkerError::InvalidOutput(format!(
                        "Native embedding batch returned {} vectors for {} inputs.",
                        batch_embeddings.len(),
                        batch.len()
                    )));
                }
                for embedding in batch_embeddings {
                    validate_embedding_output(
                        &embedding,
                        &mut expected_dimensions,
                        &mut total_values,
                    )
                    .map_err(EmbeddingWorkerError::InvalidOutput)?;
                    embeddings.push(embedding);
                }
            }
        } else {
            for text in inputs {
                if worker_token.is_cancelled() {
                    return Err(EmbeddingWorkerError::Cancelled);
                }
                let embedding = collect_embedding(Arc::clone(&pipeline), text)
                    .map_err(|error| EmbeddingWorkerError::Inference(error.to_string()))?;
                validate_embedding_output(&embedding, &mut expected_dimensions, &mut total_values)
                    .map_err(EmbeddingWorkerError::InvalidOutput)?;
                embeddings.push(embedding);
            }
        }
        if worker_token.is_cancelled() {
            return Err(EmbeddingWorkerError::Cancelled);
        }
        match projection {
            EmbeddingProjection::L2Normalized {
                dimensions,
                require_exact_dimensions,
            } => {
                if require_exact_dimensions
                    && dimensions.is_some_and(|dimensions| {
                        expected_dimensions.is_some_and(|native| dimensions > native)
                    })
                {
                    return Err(EmbeddingWorkerError::InvalidRequest(format!(
                        "Requested {} embedding dimensions, but the active model produces {}.",
                        dimensions.unwrap_or_default(),
                        expected_dimensions.unwrap_or_default()
                    )));
                }
                normalize_embedding_batch(&embeddings, dimensions, require_exact_dimensions)
                    .map(EmbeddingBatchOutput::Embeddings)
                    .map_err(EmbeddingWorkerError::InvalidOutput)
            }
            EmbeddingProjection::Rerank { top_n } => rank_embedding_documents(&embeddings, top_n)
                .map(EmbeddingBatchOutput::Rerank)
                .map_err(EmbeddingWorkerError::InvalidOutput),
        }
    })
    .await;
    state
        .app
        .metrics
        .record_inference_latency(inference_started.elapsed().as_secs_f64());

    let output = match worker {
        Ok(Ok(output)) => output,
        Ok(Err(EmbeddingWorkerError::Cancelled)) => {
            client_guard.finish(false);
            return Err(EmbeddingExecutionError::new(
                axum::http::StatusCode::REQUEST_TIMEOUT,
                "request_cancelled",
                "The embedding request was cancelled.",
            ));
        }
        Ok(Err(EmbeddingWorkerError::InvalidRequest(message))) => {
            client_guard.finish(false);
            return Err(EmbeddingExecutionError::new(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            ));
        }
        Ok(Err(EmbeddingWorkerError::Inference(message))) => {
            client_guard.finish(false);
            return Err(EmbeddingExecutionError::new(
                axum::http::StatusCode::NOT_IMPLEMENTED,
                "unsupported_operation",
                format!("Embedding inference failed: {message}"),
            ));
        }
        Ok(Err(EmbeddingWorkerError::InvalidOutput(message))) => {
            client_guard.finish(false);
            return Err(EmbeddingExecutionError::new(
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_embedding_output",
                message,
            ));
        }
        Err(error) => {
            client_guard.finish(false);
            return Err(EmbeddingExecutionError::new(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("Embedding task join failed: {error}"),
            ));
        }
    };
    let total_duration = request_start.elapsed();
    client_guard.finish(true);
    Ok(EmbeddingBatchResult {
        model_id,
        output,
        prompt_tokens,
        total_duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_embedding_request_is_fail_closed_and_bounded() {
        let valid = serde_json::from_value::<EmbeddingRequest>(json!({
            "model": "default",
            "input": ["one", "two"],
            "encoding_format": "base64",
            "dimensions": 2,
            "user": "local-client",
            "future": null
        }))
        .unwrap();
        validate_openai_embedding_request(&valid).unwrap();

        for invalid in [
            json!({"input": "one", "encoding_format": "binary"}),
            json!({"input": "one", "dimensions": 0}),
            json!({"input": "one", "user": ""}),
            json!({"input": "one", "future": true}),
        ] {
            let invalid = serde_json::from_value::<EmbeddingRequest>(invalid).unwrap();
            assert!(validate_openai_embedding_request(&invalid).is_err());
        }
    }

    #[test]
    fn base64_embedding_encoding_is_little_endian_float32() {
        // 1.0f32 and -2.5f32 in IEEE-754 little-endian bytes.
        assert_eq!(encode_embedding_base64(&[1.0, -2.5]), "AACAPwAAIMA=");
    }
}
