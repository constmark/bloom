//! Model admission, publication, retirement, and catalog snapshots.
//! This service knows no HTTP state, wire errors, credentials, or response types.
use super::inference::CancellationRegistry;
use super::memory::{RuntimeMemoryPermit, RuntimeMemoryPlanner};
use super::model_selector::{RequestedModelError, model_path_label};
use super::pool::RuntimePool;
use super::runtime::{LoadedRuntime, RuntimeRequestLease};
use crate::metrics::ServerMetrics;
use crate::model_download::ModelDownloadManager;
use crate::model_import::ModelImportManager;
use crate::model_index::ModelIndexManager;
use crate::model_integrity::ModelIntegrityManager;
use crate::model_manager::ModelCatalog;
use crate::model_preflight::ModelPreflightManager;
use crate::model_storage::ModelStorageManager;
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore, mpsc, watch};
use tokio::task;

const MODEL_CATALOG_CACHE_TTL: Duration = Duration::from_secs(10);

#[derive(Debug)]
pub(crate) struct ModelLoadRequest {
    pub(crate) sequence: u64,
    pub(crate) path: PathBuf,
    pub(crate) catalog_id: Option<String>,
    pub(crate) memory_permit: Option<RuntimeMemoryPermit>,
}

#[derive(Clone)]
pub(crate) enum ModelLoadOutcome {
    Loading,
    Ready { runtime: Arc<LoadedRuntime> },
    Failed { message: String },
}

impl std::fmt::Debug for ModelLoadOutcome {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loading => formatter.write_str("Loading"),
            Self::Ready { runtime } => formatter
                .debug_struct("Ready")
                .field("model_id", &runtime.model_id)
                .field("source_path", &runtime.source_path)
                .finish(),
            Self::Failed { message } => formatter
                .debug_struct("Failed")
                .field("message", message)
                .finish(),
        }
    }
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

pub(crate) enum ModelLoadAdmission {
    AlreadyReady {
        runtime: Arc<LoadedRuntime>,
    },
    Loading {
        sequence: u64,
        queued: bool,
        completion: watch::Receiver<ModelLoadOutcome>,
    },
}

impl std::fmt::Debug for ModelLoadAdmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyReady { runtime } => formatter
                .debug_struct("AlreadyReady")
                .field("model_id", &runtime.model_id)
                .finish(),
            Self::Loading {
                sequence, queued, ..
            } => formatter
                .debug_struct("Loading")
                .field("sequence", sequence)
                .field("queued", queued)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum ModelLoadAdmissionError {
    Busy,
    Unavailable(String),
}

/// One physical generation beyond the discoverable pool capacity lets a
/// replacement become available while already-admitted requests finish,
/// without allowing repeated switches to retain unbounded model weights.
pub(crate) const RUNTIME_DRAINING_HEADROOM: usize = 1;
pub(crate) const RUNTIME_DRAINING_CAPACITY_ERROR: &str =
    "runtime capacity is temporarily exhausted while a retired model generation is draining";
pub(crate) const RUNTIME_SOURCE_DRAINING_ERROR: &str =
    "the selected model source is still draining requests from an earlier unload";

struct CachedModelCatalog {
    refreshed_at: Instant,
    active_paths: Vec<PathBuf>,
    download_revision: u64,
    import_revision: u64,
    integrity_revision: u64,
    catalog: ModelCatalog,
}

struct DrainingRuntime {
    runtime: Weak<LoadedRuntime>,
    source_path: PathBuf,
    /// This weak lifetime marker outlives heavyweight field teardown. It lets
    /// registry inspection avoid upgrading/dropping the runtime under locks.
    active_request_leases: Weak<AtomicU64>,
}

pub(crate) struct RuntimeService {
    pub(crate) runtime_pool: RwLock<RuntimePool>,
    pub(crate) runtime_memory: RuntimeMemoryPlanner,
    /// Runtime generations removed from discovery but still retained by an
    /// admitted request. Weak references preserve storage safety without
    /// extending their physical lifetime.
    draining_runtimes: std::sync::Mutex<Vec<DrainingRuntime>>,
    pub(crate) semaphore: Arc<Semaphore>,
    pub(crate) ready: AtomicBool,
    pub(crate) load_in_progress: AtomicBool,
    pub(crate) load_progress: AtomicU8,
    pub(crate) load_error: RwLock<Option<String>>,
    pub(crate) requested_model: RwLock<Option<String>>,
    model_lifecycle: Mutex<ModelLifecycle>,
    pub(crate) models_root: PathBuf,
    model_catalog_cache: RwLock<Option<CachedModelCatalog>>,
    pub(crate) model_storage: Arc<ModelStorageManager>,
    pub(crate) model_downloads: Option<Arc<ModelDownloadManager>>,
    pub(crate) model_imports: Option<Arc<ModelImportManager>>,
    pub(crate) model_index: Option<Arc<ModelIndexManager>>,
    pub(crate) model_integrity: Arc<ModelIntegrityManager>,
    pub(crate) model_preflight: Arc<ModelPreflightManager>,
    model_loader: mpsc::Sender<ModelLoadRequest>,
    pub(crate) metrics: Arc<ServerMetrics>,
    pub(crate) speculative_mode: String,
    pub(crate) enable_ifb: bool,
    /// Per-request cancellation tokens.
    pub(crate) cancellations: CancellationRegistry,
}

pub(crate) struct RuntimeServiceConfig {
    pub(crate) runtime_pool_capacity: std::num::NonZeroUsize,
    pub(crate) max_concurrent: usize,
    pub(crate) models_root: PathBuf,
    pub(crate) speculative_mode: String,
    pub(crate) enable_ifb: bool,
}

pub(crate) struct ModelServices {
    pub(crate) storage: Arc<ModelStorageManager>,
    pub(crate) downloads: Option<Arc<ModelDownloadManager>>,
    pub(crate) imports: Option<Arc<ModelImportManager>>,
    pub(crate) index: Option<Arc<ModelIndexManager>>,
    pub(crate) integrity: Arc<ModelIntegrityManager>,
    pub(crate) preflight: Arc<ModelPreflightManager>,
}

impl RuntimeService {
    pub(crate) fn new(
        config: RuntimeServiceConfig,
        models: ModelServices,
        runtime_memory: RuntimeMemoryPlanner,
        model_loader: mpsc::Sender<ModelLoadRequest>,
    ) -> Self {
        Self {
            runtime_pool: RwLock::new(RuntimePool::with_capacity(config.runtime_pool_capacity)),
            runtime_memory,
            draining_runtimes: std::sync::Mutex::new(Vec::new()),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            ready: AtomicBool::new(false),
            load_in_progress: AtomicBool::new(false),
            load_progress: AtomicU8::new(0),
            load_error: RwLock::new(None),
            requested_model: RwLock::new(None),
            model_lifecycle: Mutex::new(ModelLifecycle::default()),
            models_root: config.models_root,
            model_catalog_cache: RwLock::new(None),
            model_storage: models.storage,
            model_downloads: models.downloads,
            model_imports: models.imports,
            model_index: models.index,
            model_integrity: models.integrity,
            model_preflight: models.preflight,
            model_loader,
            metrics: Arc::new(ServerMetrics::new()),
            speculative_mode: config.speculative_mode,
            enable_ifb: config.enable_ifb,
            cancellations: CancellationRegistry::default(),
        }
    }

    pub(crate) async fn invalidate_catalog_cache(&self) {
        *self.model_catalog_cache.write().await = None;
    }

    #[cfg(test)]
    pub(crate) async fn has_active_model_load(&self) -> bool {
        self.model_lifecycle.lock().await.active.is_some()
    }

    #[cfg(test)]
    pub(crate) async fn has_cached_catalog(&self) -> bool {
        self.model_catalog_cache.read().await.is_some()
    }

    #[cfg(test)]
    pub(crate) async fn expire_catalog_cache(&self) {
        self.model_catalog_cache
            .write()
            .await
            .as_mut()
            .unwrap()
            .refreshed_at = Instant::now() - MODEL_CATALOG_CACHE_TTL;
    }

    fn track_draining_runtimes(&self, runtimes: &[Arc<LoadedRuntime>]) {
        if runtimes.is_empty() {
            return;
        }
        let mut draining = self
            .draining_runtimes
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        draining.retain(|runtime| runtime.active_request_leases.strong_count() > 0);
        for runtime in runtimes {
            let weak_runtime = Arc::downgrade(runtime);
            if !draining
                .iter()
                .any(|current| Weak::ptr_eq(&current.runtime, &weak_runtime))
            {
                draining.push(DrainingRuntime {
                    runtime: weak_runtime,
                    source_path: runtime.source_path.clone(),
                    active_request_leases: Arc::downgrade(&runtime.active_request_leases),
                });
            }
        }
    }

    pub(crate) fn discard_unpublished_runtime(&self, runtime: Arc<LoadedRuntime>) {
        // Runtime preparation may already have spawned an IFB worker. Register
        // the candidate before dropping its owning Arc so that the worker's
        // source marker and memory permit remain visible to delete/reload and
        // physical-generation admission until the task has fully exited.
        self.track_draining_runtimes(std::slice::from_ref(&runtime));
        drop(runtime);
    }

    pub(crate) fn append_draining_sources(&self, sources: &mut Vec<PathBuf>) {
        let mut draining = self
            .draining_runtimes
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        draining.retain(|runtime| {
            let alive = runtime.active_request_leases.strong_count() > 0;
            if alive {
                sources.push(runtime.source_path.clone());
            }
            alive
        });
    }

    fn inspect_draining_runtimes(&self, source: Option<&Path>) -> (usize, u64, bool) {
        let mut draining = self
            .draining_runtimes
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut generations = 0usize;
        let mut request_leases = 0u64;
        let mut source_is_draining = false;
        draining.retain(|runtime| {
            let Some(active_request_leases) = runtime.active_request_leases.upgrade() else {
                return false;
            };
            generations = generations.saturating_add(1);
            request_leases =
                request_leases.saturating_add(active_request_leases.load(Ordering::Acquire));
            source_is_draining |= source.is_some_and(|source| runtime.source_path == source);
            true
        });
        (generations, request_leases, source_is_draining)
    }

    pub(crate) fn draining_runtime_stats(&self) -> (usize, u64) {
        let (generations, request_leases, _) = self.inspect_draining_runtimes(None);
        (generations, request_leases)
    }

    pub(crate) async fn source_is_resident_or_draining(&self, source: &Path) -> bool {
        let runtime_pool = self.runtime_pool.read().await;
        if runtime_pool.contains_source(source) {
            return true;
        }
        let mut draining = self
            .draining_runtimes
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut found = false;
        draining.retain(|runtime| {
            let alive = runtime.active_request_leases.strong_count() > 0;
            if alive {
                found |= runtime.source_path == source;
            }
            alive
        });
        found
    }

    pub(crate) async fn admit_model_load(
        &self,
        path: PathBuf,
        catalog_id: Option<String>,
        join_matching: bool,
    ) -> std::result::Result<ModelLoadAdmission, ModelLoadAdmissionError> {
        // Catalog paths are already canonical. Normalizing an explicitly
        // configured startup path here gives resident and draining identity
        // checks the same source representation without doing I/O under a
        // lifecycle or pool lock. A missing path is left for the loader to
        // report through the existing asynchronous failure contract.
        let resolved_path = tokio::fs::canonicalize(&path).await;
        let source_exists = resolved_path.is_ok();
        let path = resolved_path.unwrap_or(path);
        let selector = catalog_id
            .clone()
            .unwrap_or_else(|| model_path_label(&path));

        // Manifest parsing and hardware probes may touch storage or invoke a
        // backend-specific capability probe, so perform them before entering
        // the lifecycle/pool linearization boundary. An unreadable manifest
        // deliberately remains an asynchronous loader failure for backwards
        // compatibility; once a manifest is readable, memory admission is a
        // fail-closed prerequisite that cannot be disabled with page-touch
        // preallocation controls.
        let planning_path = path.clone();
        let planner = self.runtime_memory.clone();
        let planned_reservation = task::spawn_blocking(move || {
            source_exists
                .then(|| bloomai_engine::load_manifest(&planning_path).ok())
                .flatten()
                .map(|manifest| {
                    let estimate = planner.planned_estimate(&manifest)?;
                    planner
                        .footprint(&manifest, &estimate)
                        .and_then(|footprint| {
                            planner.available().map(|available| (footprint, available))
                        })
                })
        })
        .await
        .map_err(|error| {
            ModelLoadAdmissionError::Unavailable(format!("model memory planning failed: {error}"))
        })?;
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
        let (resident, memory_permit) = {
            let mut runtime_pool = self.runtime_pool.write().await;
            if let Some(runtime) = runtime_pool.find_source(&path) {
                runtime_pool.promote_exact(&runtime);
                (Some(runtime), None)
            } else {
                let (draining_generations, _, source_is_draining) =
                    self.inspect_draining_runtimes(Some(&path));
                let physical_limit = runtime_pool
                    .capacity()
                    .saturating_add(RUNTIME_DRAINING_HEADROOM);
                let has_physical_capacity =
                    runtime_pool.len().saturating_add(draining_generations) < physical_limit;
                if source_is_draining {
                    return Err(ModelLoadAdmissionError::Unavailable(
                        RUNTIME_SOURCE_DRAINING_ERROR.to_string(),
                    ));
                }
                if !has_physical_capacity {
                    return Err(ModelLoadAdmissionError::Unavailable(
                        RUNTIME_DRAINING_CAPACITY_ERROR.to_string(),
                    ));
                }

                let memory_permit = match planned_reservation {
                    Some(Ok((footprint, available))) => {
                        Some(self.runtime_memory.reserve(footprint, available).map_err(
                            |error| ModelLoadAdmissionError::Unavailable(error.to_string()),
                        )?)
                    }
                    Some(Err(error)) => {
                        return Err(ModelLoadAdmissionError::Unavailable(error.to_string()));
                    }
                    None => None,
                };
                (None, memory_permit)
            }
        };
        if let Some(runtime) = resident {
            *self.requested_model.write().await = Some(selector);
            self.ready.store(true, Ordering::Release);
            return Ok(ModelLoadAdmission::AlreadyReady { runtime });
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
        self.ready.store(
            !self.runtime_pool.read().await.is_empty(),
            Ordering::Release,
        );
        self.load_progress.store(0, Ordering::Release);
        *self.load_error.write().await = None;
        *self.requested_model.write().await = Some(selector);

        if let Err(error) = self.model_loader.try_send(ModelLoadRequest {
            sequence,
            path,
            catalog_id,
            memory_permit,
        }) {
            let message = format!("model loader is unavailable: {error}");
            lifecycle.active = None;
            self.load_in_progress.store(false, Ordering::Release);
            self.ready.store(
                !self.runtime_pool.read().await.is_empty(),
                Ordering::Release,
            );
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

    pub(crate) async fn finish_model_load(&self, sequence: u64, outcome: ModelLoadOutcome) {
        let mut lifecycle = self.model_lifecycle.lock().await;
        if lifecycle
            .active
            .as_ref()
            .is_some_and(|active| active.sequence == sequence)
            && let Some(active) = lifecycle.active.take()
        {
            self.load_in_progress.store(false, Ordering::Release);
            active.completion.send_replace(outcome);
        }
    }

    pub(crate) async fn lease_runtime(
        &self,
        requested: Option<&str>,
    ) -> std::result::Result<Option<RuntimeRequestLease>, RequestedModelError> {
        let runtime_pool = self.runtime_pool.read().await;
        let runtime = runtime_pool.resolve(requested)?;
        if runtime
            .as_ref()
            .is_some_and(|runtime| self.runtime_is_revoked(runtime))
        {
            return Err(RequestedModelError::Revoked);
        }
        Ok(runtime.and_then(RuntimeRequestLease::try_new))
    }

    pub(crate) async fn lease_exact_runtime(
        &self,
        expected: &Arc<LoadedRuntime>,
    ) -> Option<RuntimeRequestLease> {
        let runtime_pool = self.runtime_pool.read().await;
        (runtime_pool.contains_exact(expected) && !self.runtime_is_revoked(expected))
            .then(|| Arc::clone(expected))
            .and_then(RuntimeRequestLease::try_new)
    }

    pub(crate) fn runtime_is_revoked(&self, runtime: &LoadedRuntime) -> bool {
        let Some(index) = self.model_index.as_ref() else {
            return false;
        };
        runtime
            .signed_model_version
            .as_ref()
            .is_some_and(|version| index.is_revoked(&version.model_index_id, &version.sha256))
    }

    pub(crate) async fn publish_default_runtime(
        &self,
        runtime: Arc<LoadedRuntime>,
    ) -> std::result::Result<Vec<Arc<LoadedRuntime>>, String> {
        if self.runtime_is_revoked(&runtime) {
            return Err(
                "The verified signed-index model version was revoked before runtime publication. Install a replacement with a different digest."
                    .to_string(),
            );
        }
        let mut runtime_pool = self.runtime_pool.write().await;
        let (draining_generations, _, _) = self.inspect_draining_runtimes(None);
        let physical_limit = runtime_pool
            .capacity()
            .saturating_add(RUNTIME_DRAINING_HEADROOM);
        if runtime_pool.len().saturating_add(draining_generations) >= physical_limit {
            return Err(RUNTIME_DRAINING_CAPACITY_ERROR.to_string());
        }
        let retired = runtime_pool.publish_default(runtime);
        self.track_draining_runtimes(&retired);
        self.ready.store(true, Ordering::Release);
        Ok(retired)
    }

    pub(crate) async fn unload_runtime(
        &self,
        expected: Option<Arc<LoadedRuntime>>,
        only_if_idle: bool,
    ) -> Result<(), ModelUnloadError> {
        let _lifecycle_guard = self.model_lifecycle.lock().await;
        if self.load_in_progress.swap(true, Ordering::AcqRel) {
            return Err(ModelUnloadError::LifecycleBusy);
        }

        let (removed, fallback, busy) = {
            let mut runtime_pool = self.runtime_pool.write().await;
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
                self.track_draining_runtimes(std::slice::from_ref(runtime));
            }
            let fallback = runtime_pool.default_runtime();
            if !busy {
                self.ready.store(fallback.is_some(), Ordering::Release);
            }
            (removed, fallback, busy)
        };
        if busy {
            self.load_in_progress.store(false, Ordering::Release);
            return Err(ModelUnloadError::RequestsInFlight);
        }
        if expected.is_some() && removed.is_none() {
            self.load_in_progress.store(false, Ordering::Release);
            return Err(ModelUnloadError::NotLoaded);
        }
        drop(removed);
        *self.requested_model.write().await = fallback.as_ref().map(|runtime| {
            runtime
                .catalog_id
                .clone()
                .unwrap_or_else(|| runtime.model_id.clone())
        });
        *self.load_error.write().await = None;
        self.load_progress
            .store(if fallback.is_some() { 100 } else { 0 }, Ordering::Release);
        self.load_in_progress.store(false, Ordering::Release);

        Ok(())
    }

    pub(crate) async fn model_availability(&self) -> ModelAvailability {
        if self.load_in_progress.load(Ordering::Acquire) {
            ModelAvailability::Loading(self.load_progress.load(Ordering::Acquire))
        } else if let Some(error) = self.load_error.read().await.as_ref() {
            ModelAvailability::Failed(error.clone())
        } else {
            ModelAvailability::NotLoaded
        }
    }

    pub(crate) async fn model_catalog_snapshot(
        &self,
    ) -> Result<(ModelCatalog, Option<Arc<LoadedRuntime>>)> {
        self.model_catalog_snapshot_with_refresh(false).await
    }

    pub(crate) async fn fresh_model_catalog_snapshot(
        &self,
    ) -> Result<(ModelCatalog, Option<Arc<LoadedRuntime>>)> {
        self.model_catalog_snapshot_with_refresh(true).await
    }

    async fn model_catalog_snapshot_with_refresh(
        &self,
        force_refresh: bool,
    ) -> Result<(ModelCatalog, Option<Arc<LoadedRuntime>>)> {
        let runtime_pool = self.runtime_pool.read().await;
        let runtime = runtime_pool.default_runtime();
        let mut active_paths = runtime_pool
            .active_sources()
            .map(Path::to_path_buf)
            .collect::<Vec<_>>();
        self.append_draining_sources(&mut active_paths);
        drop(runtime_pool);
        active_paths.sort();
        active_paths.dedup();
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
        if !force_refresh
            && let Some(cached) = self.model_catalog_cache.read().await.as_ref()
            && cached.refreshed_at.elapsed() < MODEL_CATALOG_CACHE_TTL
            && cached.active_paths == active_paths
            && cached.download_revision == download_revision
            && cached.import_revision == import_revision
            && cached.integrity_revision == integrity_revision
        {
            return Ok((cached.catalog.clone(), runtime));
        }

        let root = self.models_root.clone();
        let active_for_scan = active_paths.clone();
        let catalog = task::spawn_blocking(move || {
            ModelCatalog::scan_with_active_paths(&root, &active_for_scan)
        })
        .await
        .map_err(|error| anyhow!("model catalog scan task failed: {error}"))??;
        *self.model_catalog_cache.write().await = Some(CachedModelCatalog {
            refreshed_at: Instant::now(),
            active_paths,
            download_revision,
            import_revision,
            integrity_revision,
            catalog: catalog.clone(),
        });
        Ok((catalog, runtime))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ModelAvailability {
    Loading(u8),
    Failed(String),
    NotLoaded,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ModelUnloadError {
    LifecycleBusy,
    RequestsInFlight,
    NotLoaded,
}
