//! Ownership boundary for loaded inference runtimes.
//!
//! The first iteration intentionally retains exactly one default runtime.  It
//! centralizes publication and lookup without changing the server's existing
//! single-runtime API or lifecycle semantics.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::helpers::{RequestedModelError, validate_requested_model};

use super::LoadedRuntime;

/// Runtime metadata needed by the pool.
///
/// Keeping the container generic makes its ownership and selection behavior
/// independently testable without constructing a complete inference backend.
pub(crate) trait RuntimePoolEntry {
    type Scheduler;

    fn model_id(&self) -> &str;
    fn source_path(&self) -> &Path;
    fn scheduler(&self) -> Option<Arc<Self::Scheduler>>;
}

impl RuntimePoolEntry for LoadedRuntime {
    type Scheduler = bloomai_engine::scheduler::InferenceScheduler;

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn source_path(&self) -> &Path {
        &self.source_path
    }

    fn scheduler(&self) -> Option<Arc<Self::Scheduler>> {
        self.scheduler.clone()
    }
}

/// Capacity-one runtime registry.
///
/// Replacing or removing a runtime returns its `Arc` to the caller so teardown
/// can happen after the caller releases any surrounding pool lock.
pub(crate) struct RuntimePool<R = LoadedRuntime> {
    default: Option<Arc<R>>,
}

impl<R> RuntimePool<R> {
    pub(crate) const fn new() -> Self {
        Self { default: None }
    }

    pub(crate) fn default_runtime(&self) -> Option<Arc<R>> {
        self.default.clone()
    }

    pub(crate) fn default_ref(&self) -> Option<&Arc<R>> {
        self.default.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn default_mut(&mut self) -> Option<&mut Arc<R>> {
        self.default.as_mut()
    }

    pub(crate) fn publish_default(&mut self, runtime: Arc<R>) -> Option<Arc<R>> {
        self.default.replace(runtime)
    }

    pub(crate) fn remove_default(&mut self) -> Option<Arc<R>> {
        self.default.take()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.default.is_none()
    }
}

impl<R> Default for RuntimePool<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RuntimePoolEntry> RuntimePool<R> {
    /// Resolves an OpenAI-compatible selector against the default runtime.
    ///
    /// An empty pool produces `Ok(None)` before selector validation, matching
    /// the existing request path's model-unavailable precedence.  Once a
    /// runtime exists, omitted, `default`, and exact model IDs retain their
    /// established behavior through `validate_requested_model`.
    pub(crate) fn resolve(
        &self,
        requested: Option<&str>,
    ) -> Result<Option<Arc<R>>, RequestedModelError> {
        let Some(runtime) = self.default.as_ref() else {
            return Ok(None);
        };
        validate_requested_model(requested, runtime.model_id())?;
        Ok(Some(Arc::clone(runtime)))
    }

    pub(crate) fn contains_source(&self, source: &Path) -> bool {
        self.default
            .as_ref()
            .is_some_and(|runtime| runtime.source_path() == source)
    }

    pub(crate) fn default_source(&self) -> Option<PathBuf> {
        self.default
            .as_ref()
            .map(|runtime| runtime.source_path().to_path_buf())
    }

    pub(crate) fn schedulers(&self) -> impl Iterator<Item = Arc<R::Scheduler>> + '_ {
        self.default
            .iter()
            .filter_map(|runtime| runtime.scheduler())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Debug)]
    struct FakeRuntime {
        model_id: String,
        source_path: PathBuf,
        scheduler: Option<Arc<u64>>,
        drops: Arc<AtomicUsize>,
    }

    impl FakeRuntime {
        fn new(
            model_id: &str,
            source_path: &str,
            scheduler: Option<u64>,
            drops: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                model_id: model_id.to_string(),
                source_path: PathBuf::from(source_path),
                scheduler: scheduler.map(Arc::new),
                drops,
            }
        }
    }

    impl Drop for FakeRuntime {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl RuntimePoolEntry for FakeRuntime {
        type Scheduler = u64;

        fn model_id(&self) -> &str {
            &self.model_id
        }

        fn source_path(&self) -> &Path {
            &self.source_path
        }

        fn scheduler(&self) -> Option<Arc<Self::Scheduler>> {
            self.scheduler.clone()
        }
    }

    #[test]
    fn empty_pool_preserves_model_unavailable_precedence() {
        let pool = RuntimePool::<FakeRuntime>::default();

        assert!(pool.is_empty());
        assert!(pool.default_runtime().is_none());
        assert!(pool.default_ref().is_none());
        assert_eq!(pool.default_source(), None);
        assert!(!pool.contains_source(Path::new("/models/missing")));
        assert_eq!(pool.schedulers().count(), 0);
        assert!(pool.resolve(Some(" invalid ")).unwrap().is_none());
    }

    #[test]
    fn publishes_and_resolves_only_the_default_runtime() {
        let drops = Arc::new(AtomicUsize::new(0));
        let runtime = Arc::new(FakeRuntime::new(
            "bloom-test",
            "/models/bloom-test",
            Some(7),
            Arc::clone(&drops),
        ));
        let mut pool = RuntimePool::new();

        assert!(pool.publish_default(Arc::clone(&runtime)).is_none());
        assert!(!pool.is_empty());
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("default runtime"),
            &runtime
        ));
        assert!(Arc::ptr_eq(
            pool.default_ref().expect("default runtime reference"),
            &runtime
        ));
        assert!(pool.contains_source(Path::new("/models/bloom-test")));
        assert!(!pool.contains_source(Path::new("/models/other")));
        assert_eq!(
            pool.default_source(),
            Some(PathBuf::from("/models/bloom-test"))
        );
        assert_eq!(
            pool.schedulers()
                .map(|scheduler| *scheduler)
                .collect::<Vec<_>>(),
            vec![7]
        );

        for selector in [None, Some("default"), Some("bloom-test")] {
            let resolved = pool
                .resolve(selector)
                .expect("valid selector")
                .expect("loaded runtime");
            assert!(Arc::ptr_eq(&resolved, &runtime));
        }
        assert_eq!(
            pool.resolve(Some("other")).unwrap_err(),
            RequestedModelError::NotLoaded
        );
        assert_eq!(
            pool.resolve(Some(" bloom-test")).unwrap_err(),
            RequestedModelError::Invalid
        );
    }

    #[test]
    fn default_mut_supports_unique_runtime_test_updates() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut pool = RuntimePool::new();
        pool.publish_default(Arc::new(FakeRuntime::new(
            "before",
            "/models/before",
            None,
            drops,
        )));

        let runtime = Arc::get_mut(pool.default_mut().expect("default runtime must exist"))
            .expect("pool owns the only runtime reference");
        runtime.model_id = "after".to_string();

        assert!(pool.resolve(Some("after")).unwrap().is_some());
    }

    #[test]
    fn replacement_and_removal_return_runtime_ownership() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut pool = RuntimePool::new();
        pool.publish_default(Arc::new(FakeRuntime::new(
            "first",
            "/models/first",
            None,
            Arc::clone(&drops),
        )));
        let in_flight = pool
            .resolve(Some("first"))
            .expect("valid selector")
            .expect("loaded runtime");

        let replaced = pool
            .publish_default(Arc::new(FakeRuntime::new(
                "second",
                "/models/second",
                None,
                Arc::clone(&drops),
            )))
            .expect("first runtime must be returned");
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(replaced);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(in_flight);
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let removed = pool
            .remove_default()
            .expect("second runtime must be returned");
        assert!(pool.is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(removed);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}
