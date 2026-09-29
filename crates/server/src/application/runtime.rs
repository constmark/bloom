//! Runtime ownership, generation identity, and request leases.
use super::memory::RuntimeMemoryPermit;
use bloomai_engine::InferencePipeline;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio_util::sync::CancellationToken;

pub(crate) struct LoadedRuntime {
    pub(crate) pipeline: Arc<InferencePipeline>,
    pub(crate) model_id: String,
    pub(crate) model_family: bloomai_core::ModelFamily,
    pub(crate) model_architecture: Option<String>,
    pub(crate) model_chat_template: Option<String>,
    pub(crate) input_modalities: Vec<bloomai_core::Modality>,
    pub(crate) memory_estimate: bloomai_engine::MemoryEstimate,
    pub(crate) kv_cache_pool: Option<Arc<bloomai_engine::BloomKvCachePool>>,
    pub(crate) cachemesh: Option<Arc<bloomai_engine::CacheMesh>>,
    pub(crate) scheduler: Option<Arc<bloomai_engine::scheduler::InferenceScheduler>>,
    pub(crate) _memory_reservation: Option<bloomai_engine::MemoryReservation>,
    pub(crate) scheduler_shutdown: CancellationToken,
    pub(crate) published_at: u64,
    pub(crate) source_path: PathBuf,
    pub(crate) catalog_id: Option<String>,
    pub(crate) signed_model_version: Option<SignedModelVersion>,
    pub(crate) _runtime_memory_permit: Option<RuntimeMemoryPermit>,
    /// Declared last so its final strong reference is released only after all
    /// heavyweight runtime fields have finished teardown.
    pub(crate) active_request_leases: Arc<AtomicU64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SignedModelVersion {
    pub(crate) model_index_id: String,
    pub(crate) sha256: String,
}

impl Drop for LoadedRuntime {
    fn drop(&mut self) {
        self.scheduler_shutdown.cancel();
    }
}

struct RuntimeRequestLeaseInner {
    runtime: Arc<LoadedRuntime>,
}

impl Drop for RuntimeRequestLeaseInner {
    fn drop(&mut self) {
        self.runtime
            .active_request_leases
            .fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeRequestLease {
    inner: Arc<RuntimeRequestLeaseInner>,
}

impl RuntimeRequestLease {
    pub(crate) fn try_new(runtime: Arc<LoadedRuntime>) -> Option<Self> {
        runtime
            .active_request_leases
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |leases| {
                leases.checked_add(1)
            })
            .ok()?;
        Some(Self {
            inner: Arc::new(RuntimeRequestLeaseInner { runtime }),
        })
    }

    pub(crate) fn runtime(&self) -> &Arc<LoadedRuntime> {
        &self.inner.runtime
    }

    pub(crate) fn execution_guard(&self) -> Arc<dyn std::any::Any + Send + Sync> {
        Arc::clone(&self.inner) as Arc<dyn std::any::Any + Send + Sync>
    }
}
