//! Bounded numeric inference over the existing tensor-task engine contract.
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use axum::{
    Json,
    extract::{RawQuery, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bloomai_core::GenerationParams;
use bloomai_engine::{ModelInput, ModelOutput};
use serde::{
    Deserialize, Serialize,
    de::{MapAccess, Visitor},
};
use serde_json::Number;

use crate::application::inference::{
    InferenceLifecycle, InferenceLifecycleResources, StreamExecution,
};
use crate::application::model_selector::{RequestedModelError, validate_model_selector};
use crate::{ServerState, error_response, next_request_id};

pub(crate) const MAX_TENSOR_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_TENSOR_VALUES: usize = 1_048_576;
const MAX_NAMED_INPUTS: usize = 32;
const MAX_TENSOR_RANK: usize = 8;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorRequest {
    model: Option<String>,
    input: TensorInput,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum TensorInput {
    Flat(Vec<Number>),
    Named(NamedTensorInput),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NamedTensorInput {
    #[serde(deserialize_with = "unique_inputs")]
    inputs: BTreeMap<String, TensorValues>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TensorValues {
    shape: Vec<usize>,
    data: Vec<Number>,
}

fn unique_inputs<'de, D>(deserializer: D) -> Result<BTreeMap<String, TensorValues>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct UniqueInputs;
    impl<'de> Visitor<'de> for UniqueInputs {
        type Value = BTreeMap<String, TensorValues>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a bounded map of unique tensor input names")
        }
        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut inputs = BTreeMap::new();
            while let Some((name, value)) = map.next_entry::<String, TensorValues>()? {
                if inputs.len() == MAX_NAMED_INPUTS || inputs.insert(name, value).is_some() {
                    return Err(serde::de::Error::custom(
                        "too many or duplicate tensor inputs",
                    ));
                }
            }
            Ok(inputs)
        }
    }
    deserializer.deserialize_map(UniqueInputs)
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.trim() == name
        && name.chars().count() <= 256
        && !name.chars().any(char::is_control)
}

fn validate_values(values: &[Number]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_TENSOR_VALUES
        && values.iter().all(|value| {
            value
                .as_f64()
                .is_some_and(|number| number.is_finite() && number.abs() <= f32::MAX as f64)
        })
}

impl TensorInput {
    fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Flat(values) if validate_values(values) => Ok(()),
            Self::Flat(_) => {
                Err("input must contain 1 to 1048576 finite numbers representable as float32.")
            }
            Self::Named(input) => {
                if input.inputs.is_empty()
                    || input.inputs.len() > MAX_NAMED_INPUTS
                    || input
                        .output
                        .as_deref()
                        .is_some_and(|name| !valid_name(name))
                {
                    return Err(
                        "Named tensor input requires 1 to 32 inputs and a valid optional output name.",
                    );
                }
                let mut total = 0usize;
                for (name, tensor) in &input.inputs {
                    if !valid_name(name)
                        || tensor.shape.len() > MAX_TENSOR_RANK
                        || !validate_values(&tensor.data)
                    {
                        return Err(
                            "Tensor names, ranks, or numeric values exceed the supported bounds.",
                        );
                    }
                    let elements = tensor.shape.iter().try_fold(1usize, |count, &dimension| {
                        count
                            .checked_mul(dimension)
                            .filter(|&n| dimension > 0 && n <= MAX_TENSOR_VALUES)
                    });
                    if elements != Some(tensor.data.len()) {
                        return Err(
                            "Each tensor shape must have positive dimensions and exactly match its flat data length.",
                        );
                    }
                    total = total
                        .checked_add(tensor.data.len())
                        .filter(|&n| n <= MAX_TENSOR_VALUES)
                        .ok_or("Named tensor inputs exceed the aggregate value limit.")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Serialize)]
struct TensorResponse {
    schema_version: u32,
    object: &'static str,
    id: String,
    model: String,
    output: TensorOutput,
}

#[derive(Serialize)]
struct TensorOutput {
    element_count: usize,
    data: Vec<f32>,
}

fn numeric_output(output: ModelOutput) -> Result<TensorOutput, ()> {
    if output.text.is_some()
        || output.image.is_some()
        || output.audio.is_some()
        || output.video.is_some()
    {
        return Err(());
    }
    let data = output.logits.ok_or(())?;
    if data.is_empty()
        || data.len() > MAX_TENSOR_VALUES
        || data.iter().any(|value| !value.is_finite())
    {
        return Err(());
    }
    Ok(TensorOutput {
        element_count: data.len(),
        data,
    })
}

pub(crate) async fn handle_tensor_infer(
    State(state): State<Arc<ServerState>>,
    RawQuery(query): RawQuery,
    Json(request): Json<TensorRequest>,
) -> Response {
    if query.is_some() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Tensor inference does not accept query parameters.",
        );
    }
    if request
        .model
        .as_deref()
        .is_some_and(|model| validate_model_selector(model).is_err())
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "The model field must contain 1 to 256 characters without surrounding whitespace or control characters.",
        );
    }
    if let Err(message) = request.input.validate() {
        return error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message);
    }
    let prompt = match serde_json::to_string(&request.input) {
        Ok(prompt) if prompt.len() <= MAX_TENSOR_BODY_BYTES => prompt,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "The serialized tensor input exceeds its byte limit.",
            );
        }
    };
    if !state.app.ready.load(Ordering::Acquire) {
        let (error_type, message) = state.model_unavailable().await;
        return error_response(StatusCode::SERVICE_UNAVAILABLE, error_type, message);
    }
    let runtime_lease = match state.app.lease_runtime(request.model.as_deref()).await {
        Ok(Some(lease)) => lease,
        Ok(None) => {
            let (error_type, message) = state.model_unavailable().await;
            return error_response(StatusCode::SERVICE_UNAVAILABLE, error_type, message);
        }
        Err(RequestedModelError::Invalid) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Invalid model selector.",
            );
        }
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
                "The loaded model version has been revoked. Install a replacement before retrying.",
            );
        }
    };
    let runtime = runtime_lease.runtime();
    let pipeline = Arc::clone(&runtime.pipeline);
    let model_id = runtime.model_id.clone();
    if bloomai_engine::model_manifest_tasks(&pipeline.metadata().manifest) != ["tensor"] {
        return error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_operation",
            "The loaded model does not advertise numeric tensor inference.",
        );
    }
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
    state.app.metrics.record_request_start();
    let request_start = Instant::now();
    let request_id = next_request_id(&state, "tensor");
    let Some(registration) = state.app.cancellations.register(request_id.clone(), None) else {
        state
            .app
            .metrics
            .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
        return error_response(
            StatusCode::CONFLICT,
            "request_id_conflict",
            "A request with the same ID is already active.",
        );
    };
    let cancel_token = registration.token();
    let lifecycle = InferenceLifecycle::new(
        registration,
        InferenceLifecycleResources {
            metrics: Arc::clone(&state.app.metrics),
            request_start,
            generated_tokens: Arc::new(AtomicU64::new(0)),
            prompt_tokens: 0,
            permit,
            runtime_lease,
        },
        StreamExecution::Blocking,
    );
    let worker_guard = lifecycle.worker_guard();
    let mut client_guard = lifecycle.client_guard();
    let worker_token = cancel_token.clone();
    let metrics = Arc::clone(&state.app.metrics);
    let worker = tokio::task::spawn_blocking(move || {
        let _worker_guard = worker_guard;
        if worker_token.is_cancelled() {
            return Err(());
        }
        let started = Instant::now();
        let result = pipeline.run(ModelInput::Text { prompt }, &GenerationParams::default());
        metrics.record_inference_latency(started.elapsed().as_secs_f64());
        // Runtime errors can contain private paths or input data. Keep the
        // transport error fixed; engine logs remain the operator's diagnostic.
        if worker_token.is_cancelled() {
            return Err(());
        }
        numeric_output(result.map_err(|_| ())?)
    });
    let result = tokio::select! {
        result = worker => result,
        _ = cancel_token.cancelled() => {
            client_guard.finish(false);
            return error_response(StatusCode::REQUEST_TIMEOUT, "request_cancelled", "The tensor request was cancelled.");
        }
    };
    if cancel_token.is_cancelled() {
        client_guard.finish(false);
        return error_response(
            StatusCode::REQUEST_TIMEOUT,
            "request_cancelled",
            "The tensor request was cancelled.",
        );
    }
    match result {
        Ok(Ok(output)) => {
            client_guard.finish(true);
            Json(TensorResponse {
                schema_version: 1,
                object: "bloom.tensor_inference",
                id: request_id,
                model: model_id,
                output,
            })
            .into_response()
        }
        Ok(Err(())) => {
            client_guard.finish(false);
            error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "tensor_inference_failed",
                "The tensor request could not be executed or produced invalid numeric output. Check the loaded model's input shape, names, and numeric types.",
            )
        }
        Err(_) => {
            client_guard.finish(false);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "The tensor inference worker failed.",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::DefaultBodyLimit,
        http::Request,
        routing::post,
    };
    use bloomai_core::{DeviceKind, Modality, ModelManifest};
    use bloomai_engine::{Engine, LoadedModel, ModelMetadata};
    use serde_json::{Value, json};
    use std::path::Path;
    use std::sync::{Mutex, atomic::AtomicBool};
    use tower::ServiceExt;

    #[derive(Clone, Default)]
    struct NumericEngine {
        calls: Arc<Mutex<Vec<String>>>,
        kind: &'static str,
        started: Arc<tokio::sync::Notify>,
        release: Arc<AtomicBool>,
    }

    impl Engine for NumericEngine {
        fn name(&self) -> &'static str {
            "test-tensor"
        }
        fn supported_modalities(&self) -> Vec<Modality> {
            vec![Modality::Text]
        }
        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }
        fn load(&self, _: &Path, _: DeviceKind) -> anyhow::Result<Box<dyn LoadedModel>> {
            let mut manifest = ModelManifest {
                id: "numeric-model".into(),
                ..Default::default()
            };
            if self.kind != "text" {
                manifest
                    .parameters
                    .insert("bloom_task".into(), "tensor".into());
            }
            Ok(Box::new(NumericModel {
                engine: self.clone(),
                metadata: ModelMetadata {
                    id: manifest.id.clone(),
                    modality: Modality::Text,
                    quantized: false,
                    manifest,
                },
            }))
        }
    }

    struct NumericModel {
        engine: NumericEngine,
        metadata: ModelMetadata,
    }
    impl LoadedModel for NumericModel {
        fn metadata(&self) -> &ModelMetadata {
            &self.metadata
        }
        fn infer(
            &self,
            input: ModelInput,
            params: &GenerationParams,
        ) -> anyhow::Result<ModelOutput> {
            assert_eq!(*params, GenerationParams::default());
            let ModelInput::Text { prompt } = input else {
                panic!("numeric JSON text input expected")
            };
            self.engine.calls.lock().unwrap().push(prompt);
            self.engine.started.notify_one();
            if self.engine.kind == "blocked" {
                let deadline = Instant::now() + std::time::Duration::from_secs(5);
                while !self.engine.release.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
            if self.engine.kind == "failure" {
                anyhow::bail!("private model path /secret/weights; private input");
            }
            Ok(ModelOutput {
                logits: Some(match self.engine.kind {
                    "nan" => vec![f32::NAN],
                    "empty" => vec![],
                    _ => vec![3.25, -2.0],
                }),
                text: (self.engine.kind == "text-output").then(|| "private output".into()),
                image: None,
                audio: None,
                video: None,
            })
        }
    }

    async fn state(engine: &NumericEngine) -> (tempfile::TempDir, Arc<ServerState>) {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = crate::tests::test_server_state(dir.path().to_path_buf());
        let runtime =
            crate::tests::test_text_runtime_with_engine(dir.path().join("numeric.fixture"), engine);
        state
            .app
            .runtime_pool
            .write()
            .await
            .publish_default(runtime);
        state.app.ready.store(true, Ordering::Release);
        state.app.load_progress.store(100, Ordering::Release);
        (dir, state)
    }

    fn router(state: Arc<ServerState>) -> Router {
        Router::new()
            .route(
                "/v1/tensors/infer",
                post(handle_tensor_infer).layer(DefaultBodyLimit::max(MAX_TENSOR_BODY_BYTES)),
            )
            .with_state(state)
    }

    async fn request(app: Router, uri: &str, body: Value) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), MAX_TENSOR_BODY_BYTES)
            .await
            .unwrap();
        let value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| json!({"raw": String::from_utf8_lossy(&bytes)}));
        (status, value)
    }

    #[tokio::test]
    async fn numeric_requests_preserve_input_and_return_exact_model_without_token_usage() {
        let engine = NumericEngine::default();
        let (_dir, state) = state(&engine).await;
        let app = router(Arc::clone(&state));
        let (status, first) = request(
            app.clone(),
            "/v1/tensors/infer",
            json!({"model":"default", "input":[1, 2.5]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(first["schema_version"], 1);
        assert_eq!(first["object"], "bloom.tensor_inference");
        assert_eq!(first["model"], "numeric-model");
        assert_eq!(
            first["output"],
            json!({"data":[3.25,-2.0],"element_count":2})
        );
        let input = json!({"inputs":{"value":{"shape":[1,2],"data":[9007199254740993i64, 3]}},"output":"score"});
        let (status, second) = request(
            app,
            "/v1/tensors/infer",
            json!({"model":"numeric-model", "input":input}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_ne!(first["id"], second["id"]);
        assert_eq!(
            serde_json::from_str::<Value>(&engine.calls.lock().unwrap()[1]).unwrap(),
            input
        );
        assert_eq!(
            state.app.metrics.requests_completed.load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            state
                .app
                .metrics
                .prompt_tokens_total
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            state
                .app
                .metrics
                .tokens_generated_total
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(state.app.semaphore.available_permits(), 1);
    }

    #[test]
    fn validates_tensor_shapes_sizes_names_and_duplicate_fields() {
        for input in [
            json!([]),
            json!([1e100]),
            json!({"inputs":{}}),
            json!({"inputs":{"x":{"shape":[2],"data":[1]}}}),
            json!({"inputs":{"x":{"shape":[0],"data":[1]}}}),
            json!({"inputs":{"x":{"shape":[usize::MAX,2],"data":[1]}}}),
            json!({"inputs":{"x":{"shape":[1,1,1,1,1,1,1,1,1],"data":[1]}}}),
            json!({"inputs":{"bad\nname":{"shape":[1],"data":[1]}}}),
            json!({"inputs":{"x":{"shape":[1],"data":[1]}},"output":""}),
        ] {
            let decoded: TensorInput = serde_json::from_value(input).unwrap();
            assert!(decoded.validate().is_err());
        }
        let scalar: TensorInput =
            serde_json::from_value(json!({"inputs":{"x":{"shape":[],"data":[1]}}})).unwrap();
        scalar.validate().unwrap();
        for invalid in [
            r#"{"input":[true]}"#,
            r#"{"input":[1],"temperature":0.7}"#,
            r#"{"input":{"inputs":{"x":{"shape":[1],"data":[1]},"x":{"shape":[1],"data":[2]}}}}"#,
            r#"{"input":{"inputs":{"x":{"shape":[1],"data":[1],"dtype":"float32"}}}}"#,
        ] {
            assert!(
                serde_json::from_str::<TensorRequest>(invalid).is_err(),
                "{invalid}"
            );
        }
        assert!(!validate_values(&vec![
            Number::from(0);
            MAX_TENSOR_VALUES + 1
        ]));
        assert!(
            numeric_output(ModelOutput {
                text: None,
                logits: Some(vec![0.0; MAX_TENSOR_VALUES + 1]),
                image: None,
                audio: None,
                video: None
            })
            .is_err()
        );
    }

    #[tokio::test]
    async fn rejects_invalid_requests_before_runtime_admission() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = crate::tests::test_server_state(dir.path().to_path_buf());
        let app = router(Arc::clone(&state));
        for payload in [
            json!({"input":[]}),
            json!({"model":" bad ","input":[1]}),
            json!({"input":[1e100]}),
        ] {
            assert_eq!(
                request(app.clone(), "/v1/tensors/infer", payload).await.0,
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(
            request(
                app.clone(),
                "/v1/tensors/infer?ignored=1",
                json!({"input":[1]})
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            request(
                app.clone(),
                "/v1/tensors/infer",
                json!({"input":[1],"stream":false})
            )
            .await
            .0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(
            request(app, "/v1/tensors/infer", json!({"input":[1]}))
                .await
                .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(state.app.metrics.requests_total.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn rejects_non_tensor_models_and_wrong_selectors() {
        let engine = NumericEngine {
            kind: "text",
            ..Default::default()
        };
        let (_dir, state) = state(&engine).await;
        let app = router(state);
        assert_eq!(
            request(app.clone(), "/v1/tensors/infer", json!({"input":[1]}))
                .await
                .0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(
            request(
                app,
                "/v1/tensors/infer",
                json!({"model":"other-model","input":[1]})
            )
            .await
            .0,
            StatusCode::NOT_FOUND
        );
        assert!(engine.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn rejects_invalid_outputs_and_redacts_engine_failures() {
        for kind in ["nan", "empty", "text-output", "failure"] {
            let engine = NumericEngine {
                kind,
                ..Default::default()
            };
            let (_dir, state) = state(&engine).await;
            let (status, value) = request(
                router(Arc::clone(&state)),
                "/v1/tensors/infer",
                json!({"input":[1]}),
            )
            .await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{kind}");
            assert_eq!(value["error"]["type"], "tensor_inference_failed");
            assert!(!value.to_string().contains("private"));
            assert_eq!(state.app.metrics.requests_failed.load(Ordering::Relaxed), 1);
            assert_eq!(state.app.semaphore.available_permits(), 1);
        }
    }

    #[tokio::test]
    async fn disconnect_retains_admission_until_blocking_worker_finishes() {
        let engine = NumericEngine {
            kind: "blocked",
            ..Default::default()
        };
        let (_dir, state) = state(&engine).await;
        let app = router(Arc::clone(&state));
        let first = tokio::spawn(request(
            app.clone(),
            "/v1/tensors/infer",
            json!({"input":[1]}),
        ));
        tokio::time::timeout(std::time::Duration::from_secs(2), engine.started.notified())
            .await
            .unwrap();
        first.abort();
        let _ = first.await;
        assert_eq!(state.app.semaphore.available_permits(), 0);
        let runtime = state
            .app
            .runtime_pool
            .read()
            .await
            .default_runtime()
            .unwrap();
        assert_eq!(runtime.active_request_leases.load(Ordering::Acquire), 1);
        assert_eq!(
            request(app, "/v1/tensors/infer", json!({"input":[2]}))
                .await
                .0,
            StatusCode::TOO_MANY_REQUESTS
        );
        engine.release.store(true, Ordering::Release);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while state.app.semaphore.available_permits() == 0
                || runtime.active_request_leases.load(Ordering::Acquire) != 0
                || !state.app.cancellations.is_empty()
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(runtime.active_request_leases.load(Ordering::Acquire), 0);
        assert_eq!(state.app.metrics.requests_failed.load(Ordering::Relaxed), 1);
        assert_eq!(
            state.app.metrics.in_flight_requests.load(Ordering::Relaxed),
            0
        );
        assert!(state.app.cancellations.is_empty());
    }
}
