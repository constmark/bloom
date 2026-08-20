//! Strongly typed public readiness and UI/server compatibility projection.

use super::*;

pub(crate) const READINESS_SCHEMA_VERSION: u32 = 3;
pub(crate) const READINESS_PROTOCOL_VERSION: u32 = 3;
pub(crate) const MINIMUM_UI_PROTOCOL_VERSION: u32 = 3;
pub(crate) const MAXIMUM_UI_PROTOCOL_VERSION: u32 = 3;
pub(crate) const READINESS_OBJECT: &str = "bloom.readiness";

const _: () = assert!(
    MINIMUM_UI_PROTOCOL_VERSION > 0
        && MINIMUM_UI_PROTOCOL_VERSION <= MAXIMUM_UI_PROTOCOL_VERSION
        && READINESS_PROTOCOL_VERSION >= MINIMUM_UI_PROTOCOL_VERSION
        && READINESS_PROTOCOL_VERSION <= MAXIMUM_UI_PROTOCOL_VERSION
);

#[derive(Serialize)]
struct ReadinessSnapshot<'a> {
    schema_version: u32,
    object: &'static str,
    protocol_version: u32,
    minimum_ui_protocol_version: u32,
    maximum_ui_protocol_version: u32,
    server_version: &'static str,
    status: &'static str,
    progress: u8,
    model: &'a str,
    loading: bool,
    load_error: Option<&'static str>,
    input_modalities: &'a [bloomai_core::Modality],
    model_tasks: Vec<&'static str>,
    context_window: Option<u64>,
    in_flight_requests: u64,
    available_permits: u64,
    memory_pressure_high: bool,
    ram_utilization: f64,
}

pub(crate) async fn handle_ready(
    State(state): State<Arc<ServerState>>,
) -> axum::response::Response {
    let in_flight = state.metrics.in_flight_requests.load(Ordering::Relaxed);
    let available_permits = state.semaphore.available_permits();
    let mut memory = bloomai_engine::MemoryTelemetry::new();
    memory.refresh_ram();
    let memory_pressure_high = memory.is_high_pressure();
    let ready =
        state.ready.load(Ordering::Relaxed) && available_permits > 0 && !memory_pressure_high;
    let runtime = state.runtime_pool.read().await.default_runtime();
    let model = runtime
        .as_ref()
        .map(|runtime| runtime.model_id.as_str())
        .unwrap_or("not loaded");
    let input_modalities = runtime
        .as_ref()
        .map(|runtime| runtime.input_modalities.as_slice())
        .unwrap_or_default();
    let model_tasks = runtime
        .as_ref()
        .map(|runtime| {
            if model_supports_embeddings(&runtime.pipeline) {
                vec!["embedding", "rerank"]
            } else {
                vec!["generation"]
            }
        })
        .unwrap_or_default();
    let context_window = runtime
        .as_ref()
        .map(|runtime| runtime.pipeline.context_size() as u64);
    let load_failed = state.load_error.read().await.is_some();
    let body = ReadinessSnapshot {
        schema_version: READINESS_SCHEMA_VERSION,
        object: READINESS_OBJECT,
        protocol_version: READINESS_PROTOCOL_VERSION,
        minimum_ui_protocol_version: MINIMUM_UI_PROTOCOL_VERSION,
        maximum_ui_protocol_version: MAXIMUM_UI_PROTOCOL_VERSION,
        server_version: env!("CARGO_PKG_VERSION"),
        status: if ready { "ready" } else { "not_ready" },
        progress: state.load_progress.load(Ordering::Relaxed),
        model,
        loading: state.load_in_progress.load(Ordering::Relaxed),
        load_error: load_failed.then_some(
            "Model load failed. See the authenticated model-management endpoint for details.",
        ),
        input_modalities,
        model_tasks,
        context_window,
        in_flight_requests: in_flight,
        available_permits: available_permits as u64,
        memory_pressure_high,
        ram_utilization: memory.ram_utilization(),
    };

    if ready {
        Json(body).into_response()
    } else {
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
    }
}
