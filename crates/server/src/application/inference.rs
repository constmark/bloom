//! Request cancellation and resource retention until both client and worker settle.
use super::runtime::RuntimeRequestLease;
use crate::metrics::ServerMetrics;
use bloomai_engine::scheduler::InferenceScheduler;
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
};
use std::time::Instant;
use tokio::sync::OwnedSemaphorePermit;
use tokio_util::sync::CancellationToken;

/// Owns cancellation identity and its linearization with scheduler cleanup.
#[derive(Clone, Default)]
pub(crate) struct CancellationRegistry {
    registrations: Arc<std::sync::Mutex<HashMap<String, Arc<RequestCancellation>>>>,
}

impl CancellationRegistry {
    pub(crate) fn register(
        &self,
        request_id: String,
        scheduler: Option<Arc<InferenceScheduler>>,
    ) -> Option<CancelTokenGuard> {
        CancelTokenGuard::register_with_tokens(
            Arc::clone(&self.registrations),
            request_id,
            scheduler,
        )
    }

    pub(crate) fn cancel(&self, request_id: &str) -> bool {
        let (registration, cancelled) = {
            let registrations = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(registration) = registrations.get(request_id).cloned() {
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
                scheduler.cancel_request(request_id);
            }
            let mut registrations = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
            if registrations
                .get(request_id)
                .is_some_and(|active| Arc::ptr_eq(active, &registration))
            {
                registrations.remove(request_id);
            }
        }

        cancelled
    }

    #[cfg(test)]
    pub(crate) fn with_lock<T>(&self, action: impl FnOnce() -> T) -> T {
        let _lock = self.registrations.lock().unwrap();
        action()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.registrations.lock().unwrap().is_empty()
    }
}

struct RequestCancellation {
    token: CancellationToken,
    scheduler: Option<Arc<InferenceScheduler>>,
    cancelling: AtomicBool,
}

pub(crate) struct CancelTokenGuard {
    registrations: Arc<std::sync::Mutex<HashMap<String, Arc<RequestCancellation>>>>,
    request_id: String,
    registration: Arc<RequestCancellation>,
}

impl CancelTokenGuard {
    fn register_with_tokens(
        registrations: Arc<std::sync::Mutex<HashMap<String, Arc<RequestCancellation>>>>,
        request_id: String,
        scheduler: Option<Arc<InferenceScheduler>>,
    ) -> Option<Self> {
        let registration = Arc::new(RequestCancellation {
            token: CancellationToken::new(),
            scheduler,
            cancelling: AtomicBool::new(false),
        });
        {
            let mut active = registrations.lock().unwrap_or_else(|e| e.into_inner());
            match active.entry(request_id.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(Arc::clone(&registration));
                }
                std::collections::hash_map::Entry::Occupied(_) => return None,
            }
        }
        Some(Self {
            registrations,
            request_id,
            registration,
        })
    }

    pub(crate) fn token(&self) -> CancellationToken {
        self.registration.token.clone()
    }
}

impl Drop for CancelTokenGuard {
    fn drop(&mut self) {
        let mut registrations = self.registrations.lock().unwrap_or_else(|e| e.into_inner());
        if !self.registration.cancelling.load(Ordering::Acquire)
            && registrations
                .get(&self.request_id)
                .is_some_and(|active| Arc::ptr_eq(active, &self.registration))
        {
            registrations.remove(&self.request_id);
        }
    }
}

/// Retains request accounting, cancellation registration, and admission until
/// both the client-facing response future and any blocking worker are settled.
pub(crate) struct InferenceLifecycle {
    registration: std::sync::Mutex<Option<CancelTokenGuard>>,
    request_id: String,
    token: CancellationToken,
    metrics: Arc<ServerMetrics>,
    request_start: Instant,
    generated_tokens: Arc<AtomicU64>,
    prompt_tokens: u64,
    execution: StreamExecution,
    permit: std::sync::Mutex<Option<OwnedSemaphorePermit>>,
    runtime_lease: std::sync::Mutex<Option<RuntimeRequestLease>>,
    worker_done: AtomicBool,
    client_outcome: AtomicU8,
    settled: AtomicBool,
}

pub(crate) struct InferenceLifecycleResources {
    pub(crate) metrics: Arc<ServerMetrics>,
    pub(crate) request_start: Instant,
    pub(crate) generated_tokens: Arc<AtomicU64>,
    pub(crate) prompt_tokens: u64,
    pub(crate) permit: OwnedSemaphorePermit,
    pub(crate) runtime_lease: RuntimeRequestLease,
}

pub(crate) enum StreamExecution {
    Scheduled(Arc<InferenceScheduler>),
    Blocking,
}

impl InferenceLifecycle {
    #[cfg(test)]
    pub(crate) fn client_has_finished(&self) -> bool {
        self.client_outcome.load(Ordering::Acquire) != 0
    }

    pub(crate) fn new(
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
            runtime_lease: std::sync::Mutex::new(Some(resources.runtime_lease)),
            worker_done: AtomicBool::new(worker_done),
            client_outcome: AtomicU8::new(0),
            settled: AtomicBool::new(false),
        })
    }

    pub(crate) fn client_guard(self: &Arc<Self>) -> InferenceClientGuard {
        InferenceClientGuard {
            lifecycle: Arc::clone(self),
            completed: false,
        }
    }

    pub(crate) fn worker_guard(self: &Arc<Self>) -> InferenceWorkerGuard {
        InferenceWorkerGuard {
            lifecycle: Arc::clone(self),
        }
    }

    pub(crate) fn finish(&self, success: bool) {
        let outcome = if success && !self.token.is_cancelled() {
            1
        } else {
            2
        };
        if self
            .client_outcome
            .compare_exchange(0, outcome, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.maybe_settle();
        }
    }

    fn client_dropped(&self) {
        if self
            .client_outcome
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        self.token.cancel();
        if let StreamExecution::Scheduled(scheduler) = &self.execution {
            scheduler.cancel_request(&self.request_id);
        }
        self.maybe_settle();
    }

    fn worker_finished(&self) {
        self.worker_done.store(true, Ordering::Release);
        self.maybe_settle();
    }

    fn maybe_settle(&self) {
        let client_outcome = self.client_outcome.load(Ordering::Acquire);
        if client_outcome != 0 && self.worker_done.load(Ordering::Acquire) {
            self.settle(client_outcome == 1 && !self.token.is_cancelled());
        }
    }

    fn settle(&self, success: bool) {
        if self.settled.swap(true, Ordering::AcqRel) {
            return;
        }
        let registration = self
            .registration
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop(registration);
        let success = success && !self.token.is_cancelled();
        self.metrics.record_request_end(
            success,
            self.request_start.elapsed().as_secs_f64(),
            self.generated_tokens.load(Ordering::Relaxed),
            self.prompt_tokens,
        );
        self.permit.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.runtime_lease
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }
}

pub(crate) struct InferenceClientGuard {
    lifecycle: Arc<InferenceLifecycle>,
    completed: bool,
}

impl InferenceClientGuard {
    pub(crate) fn finish(&mut self, success: bool) {
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

pub(crate) struct InferenceWorkerGuard {
    lifecycle: Arc<InferenceLifecycle>,
}

impl Drop for InferenceWorkerGuard {
    fn drop(&mut self) {
        self.lifecycle.worker_finished();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancel_token_guard_removes_its_registration() {
        let tokens = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let guard =
            CancelTokenGuard::register_with_tokens(Arc::clone(&tokens), "req-1".to_string(), None)
                .unwrap();
        let token = guard.token();
        assert!(!token.is_cancelled());
        assert!(tokens.lock().unwrap().contains_key("req-1"));
        drop(guard);
        assert!(!tokens.lock().unwrap().contains_key("req-1"));
        assert!(!token.is_cancelled());
    }

    #[test]
    fn duplicate_active_request_id_registration_is_rejected() {
        let registrations = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let older = CancelTokenGuard::register_with_tokens(
            Arc::clone(&registrations),
            "req-shared".to_string(),
            None,
        )
        .unwrap();
        let newer = CancelTokenGuard::register_with_tokens(
            Arc::clone(&registrations),
            "req-shared".to_string(),
            None,
        );
        assert!(newer.is_none());
        assert!(registrations.lock().unwrap().contains_key("req-shared"));

        drop(older);
        assert!(!registrations.lock().unwrap().contains_key("req-shared"));
    }

    #[test]
    fn claimed_cancellation_keeps_the_request_id_reserved_until_cleanup() {
        let registrations = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let guard = CancelTokenGuard::register_with_tokens(
            Arc::clone(&registrations),
            "req-cancelling".to_string(),
            None,
        )
        .unwrap();
        let registration = Arc::clone(registrations.lock().unwrap().get("req-cancelling").unwrap());
        registration.cancelling.store(true, Ordering::Release);

        drop(guard);
        assert!(registrations.lock().unwrap().contains_key("req-cancelling"));

        registrations.lock().unwrap().remove("req-cancelling");
        assert!(
            CancelTokenGuard::register_with_tokens(
                Arc::clone(&registrations),
                "req-cancelling".to_string(),
                None,
            )
            .is_some()
        );
    }
}
