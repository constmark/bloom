//! Transport state: authentication, protocol limits, response storage, and adapter residency.
use crate::application::runtime::LoadedRuntime;
use crate::application::runtime_service::{ModelAvailability, RuntimeService};
use crate::response_store::ResponseStore;
use std::sync::{Arc, Weak, atomic::AtomicU64};
use std::time::SystemTime;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

pub(crate) struct ServerState {
    pub(crate) app: Arc<RuntimeService>,
    pub(crate) ollama_residency: Mutex<OllamaResidencyState>,
    pub(crate) max_ollama_body_bytes: usize,
    pub(crate) request_counter: AtomicU64,
    pub(crate) api_key: Option<String>,
    pub(crate) operator_api_key: Option<String>,
    pub(crate) response_store: ResponseStore,
}

impl ServerState {
    pub(crate) async fn model_unavailable(&self) -> (&'static str, String) {
        match self.app.model_availability().await {
            ModelAvailability::Loading(progress) => (
                "model_loading",
                format!("Model is loading (progress: {progress}%)."),
            ),
            ModelAvailability::Failed(error) => (
                "model_load_failed",
                format!("The model failed to load: {error}"),
            ),
            ModelAvailability::NotLoaded => (
                "model_not_loaded",
                "No model is loaded. Choose a catalog model or start the server with --model."
                    .to_string(),
            ),
        }
    }
}

pub(crate) struct OllamaRuntimeResidency {
    pub(crate) runtime: Weak<LoadedRuntime>,
    pub(crate) revision: u64,
    pub(crate) expiry: Option<SystemTime>,
    pub(crate) timer_cancel: CancellationToken,
}

#[derive(Default)]
pub(crate) struct OllamaResidencyState {
    pub(crate) runtimes: Vec<OllamaRuntimeResidency>,
}
