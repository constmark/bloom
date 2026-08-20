//! Ownership boundary for loaded inference runtimes.
//!
//! Runtimes are keyed by their resolved source path and retained behind
//! `Arc`s so an admitted request keeps executing the exact generation it
//! resolved, even when a newer generation is published concurrently.  Pool
//! mutation returns retired `Arc`s to the caller; their potentially expensive
//! teardown can therefore happen after the surrounding pool lock is released.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::helpers::{RequestedModelError, validate_model_selector};

use super::LoadedRuntime;

pub(crate) const DEFAULT_RUNTIME_POOL_CAPACITY: usize = 1;

/// Stable identity for a logical runtime across reload generations.
///
/// The loader is responsible for resolving aliases before publication.  The
/// pool deliberately does not canonicalize paths because doing filesystem I/O
/// while holding the pool lock would violate its ownership boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RuntimeKey(PathBuf);

impl RuntimeKey {
    pub(crate) fn from_source(source: &Path) -> Self {
        Self(source.to_path_buf())
    }

    pub(crate) fn source(&self) -> &Path {
        &self.0
    }
}

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

struct RuntimeSlot<R> {
    key: RuntimeKey,
    runtime: Arc<R>,
}

/// Result of publishing one runtime.
///
/// A same-key reload appears in `replaced`; capacity victims appear in
/// `evicted`.  The caller owns both collections and should drop them only after
/// releasing the pool write lock.
pub(crate) struct RuntimePublication<R> {
    #[cfg(test)]
    key: RuntimeKey,
    replaced: Option<Arc<R>>,
    evicted: Vec<Arc<R>>,
}

impl<R> RuntimePublication<R> {
    #[cfg(test)]
    pub(crate) fn key(&self) -> &RuntimeKey {
        &self.key
    }

    #[cfg(test)]
    pub(crate) fn replaced(&self) -> Option<&Arc<R>> {
        self.replaced.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn evicted(&self) -> &[Arc<R>] {
        &self.evicted
    }

    pub(crate) fn into_retired(self) -> Vec<Arc<R>> {
        let mut retired =
            Vec::with_capacity(usize::from(self.replaced.is_some()) + self.evicted.len());
        retired.extend(self.replaced);
        retired.extend(self.evicted);
        retired
    }
}

/// One immutable view of the pool, with `Arc` ownership detached from its lock.
pub(crate) struct RuntimePoolSnapshot<R> {
    default_key: Option<RuntimeKey>,
    entries: Vec<(RuntimeKey, Arc<R>)>,
}

impl<R> RuntimePoolSnapshot<R> {
    #[cfg(test)]
    pub(crate) fn default_key(&self) -> Option<&RuntimeKey> {
        self.default_key.as_ref()
    }

    pub(crate) fn entries(&self) -> &[(RuntimeKey, Arc<R>)] {
        &self.entries
    }

    pub(crate) fn default_runtime(&self) -> Option<Arc<R>> {
        let default = self.default_key.as_ref()?;
        self.entries
            .iter()
            .find(|(key, _)| key == default)
            .map(|(_, runtime)| Arc::clone(runtime))
    }
}

/// Bounded registry of loaded runtime generations.
///
/// Entries are ordered from oldest to newest publication.  When full, a new
/// publication evicts the oldest non-default entry.  Capacity one necessarily
/// evicts the previous default.  This insertion-order policy is deterministic
/// and keeps the actively selected runtime pinned whenever another slot exists.
pub(crate) struct RuntimePool<R = LoadedRuntime> {
    capacity: NonZeroUsize,
    entries: VecDeque<RuntimeSlot<R>>,
    default_key: Option<RuntimeKey>,
}

impl<R> RuntimePool<R> {
    pub(crate) fn new() -> Self {
        Self::with_capacity(
            NonZeroUsize::new(DEFAULT_RUNTIME_POOL_CAPACITY)
                .expect("default runtime pool capacity must be non-zero"),
        )
    }

    pub(crate) fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity.get()),
            default_key: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity.get()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn default_key(&self) -> Option<&RuntimeKey> {
        self.default_key.as_ref()
    }

    pub(crate) fn default_runtime(&self) -> Option<Arc<R>> {
        self.default_ref().cloned()
    }

    pub(crate) fn default_ref(&self) -> Option<&Arc<R>> {
        let index = self.default_index()?;
        Some(&self.entries[index].runtime)
    }

    #[cfg(test)]
    pub(crate) fn default_mut(&mut self) -> Option<&mut Arc<R>> {
        let index = self.default_index()?;
        Some(&mut self.entries[index].runtime)
    }

    pub(crate) fn snapshot(&self) -> RuntimePoolSnapshot<R> {
        RuntimePoolSnapshot {
            default_key: self.default_key.clone(),
            entries: self
                .entries
                .iter()
                .map(|slot| (slot.key.clone(), Arc::clone(&slot.runtime)))
                .collect(),
        }
    }

    /// Clones every active generation in deterministic publication order.
    #[cfg(test)]
    pub(crate) fn snapshots(&self) -> Vec<Arc<R>> {
        self.entries
            .iter()
            .map(|slot| Arc::clone(&slot.runtime))
            .collect()
    }

    pub(crate) fn remove_default(&mut self) -> Option<Arc<R>> {
        let index = self.default_index()?;
        self.remove_index(index)
    }

    /// Removes only the exact runtime generation represented by `expected`.
    ///
    /// This prevents an expiry timer or delayed unload for an older generation
    /// from accidentally removing a newer same-source replacement.
    pub(crate) fn remove_exact(&mut self, expected: &Arc<R>) -> Option<Arc<R>> {
        let index = self
            .entries
            .iter()
            .position(|slot| Arc::ptr_eq(&slot.runtime, expected))?;
        self.remove_index(index)
    }

    pub(crate) fn contains_exact(&self, expected: &Arc<R>) -> bool {
        self.entries
            .iter()
            .any(|slot| Arc::ptr_eq(&slot.runtime, expected))
    }

    pub(crate) fn promote_exact(&mut self, expected: &Arc<R>) -> bool {
        let Some(slot) = self
            .entries
            .iter()
            .find(|slot| Arc::ptr_eq(&slot.runtime, expected))
        else {
            return false;
        };
        self.default_key = Some(slot.key.clone());
        true
    }

    fn default_index(&self) -> Option<usize> {
        let key = self.default_key.as_ref()?;
        self.entries.iter().position(|slot| &slot.key == key)
    }

    fn remove_index(&mut self, index: usize) -> Option<Arc<R>> {
        let removed = self.entries.remove(index)?;
        if self.default_key.as_ref() == Some(&removed.key) {
            self.default_key = self.entries.back().map(|slot| slot.key.clone());
        }
        Some(removed.runtime)
    }
}

impl<R> Default for RuntimePool<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RuntimePoolEntry> RuntimePool<R> {
    /// Publishes a runtime and optionally makes it the default selection.
    pub(crate) fn publish(&mut self, runtime: Arc<R>, make_default: bool) -> RuntimePublication<R> {
        let key = RuntimeKey::from_source(runtime.source_path());
        let model_id = runtime.model_id();
        let previous_default = self.default_key.clone();
        let mut replaced = None;
        let mut evicted = Vec::new();
        let mut inherits_default = false;
        let mut index = 0;
        while index < self.entries.len() {
            let slot = &self.entries[index];
            if slot.key != key && slot.runtime.model_id() != model_id {
                index += 1;
                continue;
            }

            let slot = self
                .entries
                .remove(index)
                .expect("matched runtime slot must remain present");
            inherits_default |= previous_default.as_ref() == Some(&slot.key);
            if slot.key == key && replaced.is_none() {
                replaced = Some(slot.runtime);
            } else {
                evicted.push(slot.runtime);
            }
        }

        self.entries.push_back(RuntimeSlot {
            key: key.clone(),
            runtime,
        });

        if make_default || inherits_default || self.entries.len() == 1 {
            self.default_key = Some(key.clone());
        }

        while self.entries.len() > self.capacity.get() {
            let eviction_index = self.eviction_index();
            if let Some(evicted_runtime) = self.remove_index(eviction_index) {
                evicted.push(evicted_runtime);
            }
        }

        RuntimePublication {
            #[cfg(test)]
            key,
            replaced,
            evicted,
        }
    }

    /// Publishes a runtime as the default and returns every retired generation.
    ///
    /// Retirement can include a same-source replacement, a conflicting model
    /// ID, and a capacity victim. Returning all of them keeps teardown outside
    /// the pool's write lock.
    pub(crate) fn publish_default(&mut self, runtime: Arc<R>) -> Vec<Arc<R>> {
        self.publish(runtime, true).into_retired()
    }

    /// Resolves an OpenAI-compatible selector.
    ///
    /// An empty pool produces `Ok(None)` before selector validation, preserving
    /// model-unavailable precedence. Omitted and `default` select the promoted
    /// runtime. Exact model IDs are unambiguous because publication retires any
    /// older entry carrying the same ID.
    pub(crate) fn resolve(
        &self,
        requested: Option<&str>,
    ) -> Result<Option<Arc<R>>, RequestedModelError> {
        if self.entries.is_empty() {
            return Ok(None);
        }
        let Some(requested) = requested else {
            return Ok(self.default_runtime());
        };
        validate_model_selector(requested)?;
        if requested == "default" {
            return Ok(self.default_runtime());
        }
        self.entries
            .iter()
            .find(|slot| slot.runtime.model_id() == requested)
            .map(|slot| Some(Arc::clone(&slot.runtime)))
            .ok_or(RequestedModelError::NotLoaded)
    }

    pub(crate) fn find_source(&self, source: &Path) -> Option<Arc<R>> {
        let key = RuntimeKey::from_source(source);
        self.entries
            .iter()
            .find(|slot| slot.key == key)
            .map(|slot| Arc::clone(&slot.runtime))
    }

    pub(crate) fn contains_source(&self, source: &Path) -> bool {
        self.find_source(source).is_some()
    }

    #[cfg(test)]
    pub(crate) fn default_source(&self) -> Option<PathBuf> {
        self.default_key
            .as_ref()
            .map(|key| key.source().to_path_buf())
    }

    pub(crate) fn active_sources(&self) -> impl ExactSizeIterator<Item = &Path> {
        self.entries.iter().map(|slot| slot.key.source())
    }

    pub(crate) fn schedulers(&self) -> impl Iterator<Item = Arc<R::Scheduler>> + '_ {
        self.entries
            .iter()
            .filter_map(|slot| slot.runtime.scheduler())
    }

    #[cfg(test)]
    pub(crate) fn promote_source(&mut self, source: &Path) -> bool {
        let key = RuntimeKey::from_source(source);
        if self.entries.iter().any(|slot| slot.key == key) {
            self.default_key = Some(key);
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    pub(crate) fn remove_source(&mut self, source: &Path) -> Option<Arc<R>> {
        let key = RuntimeKey::from_source(source);
        let index = self.entries.iter().position(|slot| slot.key == key)?;
        self.remove_index(index)
    }

    fn eviction_index(&self) -> usize {
        if self.capacity.get() == 1 {
            return 0;
        }
        self.entries
            .iter()
            .position(|slot| self.default_key.as_ref() != Some(&slot.key))
            .unwrap_or(0)
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

    fn runtime(
        model_id: &str,
        source_path: &str,
        scheduler: Option<u64>,
        drops: &Arc<AtomicUsize>,
    ) -> Arc<FakeRuntime> {
        Arc::new(FakeRuntime::new(
            model_id,
            source_path,
            scheduler,
            Arc::clone(drops),
        ))
    }

    fn pool_with_capacity(capacity: usize) -> RuntimePool<FakeRuntime> {
        RuntimePool::with_capacity(NonZeroUsize::new(capacity).expect("test capacity is non-zero"))
    }

    #[test]
    fn empty_pool_preserves_model_unavailable_precedence() {
        let pool = RuntimePool::<FakeRuntime>::default();

        assert_eq!(pool.capacity(), DEFAULT_RUNTIME_POOL_CAPACITY);
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert!(pool.default_key().is_none());
        assert!(pool.default_runtime().is_none());
        assert!(pool.default_ref().is_none());
        assert_eq!(pool.default_source(), None);
        assert!(!pool.contains_source(Path::new("/models/missing")));
        assert!(pool.find_source(Path::new("/models/missing")).is_none());
        assert_eq!(pool.active_sources().count(), 0);
        assert_eq!(pool.schedulers().count(), 0);
        assert!(pool.resolve(Some(" invalid ")).unwrap().is_none());
    }

    #[test]
    fn default_capacity_preserves_single_runtime_behavior() {
        let drops = Arc::new(AtomicUsize::new(0));
        let first = runtime("first", "/models/first", Some(1), &drops);
        let second = runtime("second", "/models/second", Some(2), &drops);
        let mut pool = RuntimePool::new();

        assert!(pool.publish_default(Arc::clone(&first)).is_empty());
        let retired = pool.publish_default(Arc::clone(&second));

        assert_eq!(retired.len(), 1);
        assert!(Arc::ptr_eq(&retired[0], &first));
        assert_eq!(pool.len(), 1);
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("default runtime"),
            &second
        ));
        assert!(!pool.contains_source(Path::new("/models/first")));
        assert!(!pool.contains_exact(&first));
        assert!(pool.contains_exact(&second));
    }

    #[test]
    fn bounded_publication_protects_default_and_evicts_oldest_non_default() {
        let drops = Arc::new(AtomicUsize::new(0));
        let first = runtime("first", "/models/first", None, &drops);
        let second = runtime("second", "/models/second", None, &drops);
        let third = runtime("third", "/models/third", None, &drops);
        let mut pool = pool_with_capacity(2);

        pool.publish(Arc::clone(&first), true);
        pool.publish(Arc::clone(&second), false);
        let publication = pool.publish(Arc::clone(&third), false);

        assert!(publication.replaced().is_none());
        assert_eq!(publication.evicted().len(), 1);
        assert!(Arc::ptr_eq(&publication.evicted()[0], &second));
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("pinned default"),
            &first
        ));
        assert_eq!(
            pool.active_sources().collect::<Vec<_>>(),
            vec![Path::new("/models/first"), Path::new("/models/third")]
        );
    }

    #[test]
    fn promoting_changes_default_without_changing_insertion_order() {
        let drops = Arc::new(AtomicUsize::new(0));
        let first = runtime("first", "/models/first", None, &drops);
        let second = runtime("second", "/models/second", None, &drops);
        let mut pool = pool_with_capacity(2);
        pool.publish(Arc::clone(&first), true);
        pool.publish(Arc::clone(&second), false);

        assert!(pool.promote_source(Path::new("/models/second")));
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("promoted runtime"),
            &second
        ));
        assert!(pool.promote_exact(&first));
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("promoted exact runtime"),
            &first
        ));
        assert!(!pool.promote_source(Path::new("/models/missing")));
        assert!(!pool.promote_exact(&runtime("missing", "/models/missing", None, &drops)));
        assert_eq!(
            pool.active_sources().collect::<Vec<_>>(),
            vec![Path::new("/models/first"), Path::new("/models/second")]
        );
    }

    #[test]
    fn resolves_default_model_id_and_source_with_stable_arc_identity() {
        let drops = Arc::new(AtomicUsize::new(0));
        let first = runtime("first", "/models/first", Some(1), &drops);
        let second = runtime("second", "/models/second", Some(2), &drops);
        let mut pool = pool_with_capacity(2);
        pool.publish(Arc::clone(&first), true);
        pool.publish(Arc::clone(&second), false);

        for selector in [None, Some("default"), Some("first")] {
            let resolved = pool
                .resolve(selector)
                .expect("valid selector")
                .expect("loaded runtime");
            assert!(Arc::ptr_eq(&resolved, &first));
        }
        assert!(Arc::ptr_eq(
            &pool
                .find_source(Path::new("/models/second"))
                .expect("source match"),
            &second
        ));
        assert!(Arc::ptr_eq(
            &pool
                .resolve(Some("second"))
                .expect("valid selector")
                .expect("model match"),
            &second
        ));
        assert_eq!(
            pool.resolve(Some("missing")).unwrap_err(),
            RequestedModelError::NotLoaded
        );
        assert_eq!(
            pool.resolve(Some(" shared")).unwrap_err(),
            RequestedModelError::Invalid
        );
        assert_eq!(
            pool.schedulers()
                .map(|scheduler| *scheduler)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn publishing_duplicate_model_id_retires_old_entry_and_inherits_default() {
        let drops = Arc::new(AtomicUsize::new(0));
        let older = runtime("duplicate", "/models/older", None, &drops);
        let newer = runtime("duplicate", "/models/newer", None, &drops);
        let mut pool = pool_with_capacity(3);
        pool.publish(Arc::clone(&older), true);
        let publication = pool.publish(Arc::clone(&newer), false);

        assert!(publication.replaced().is_none());
        assert_eq!(publication.evicted().len(), 1);
        assert!(Arc::ptr_eq(&publication.evicted()[0], &older));
        assert_eq!(pool.len(), 1);
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("inherited default"),
            &newer
        ));
        let resolved = pool
            .resolve(Some("duplicate"))
            .expect("valid selector")
            .expect("matching runtime");
        assert!(Arc::ptr_eq(&resolved, &newer));
        assert!(!pool.contains_exact(&older));
        assert!(!pool.contains_source(Path::new("/models/older")));
    }

    #[test]
    fn source_and_model_collisions_retire_each_old_entry_exactly_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let same_source = runtime("alpha", "/models/shared-source", None, &drops);
        let same_model = runtime("beta", "/models/other-source", None, &drops);
        let replacement = runtime("beta", "/models/shared-source", None, &drops);
        let mut pool = pool_with_capacity(3);
        pool.publish(Arc::clone(&same_source), true);
        pool.publish(Arc::clone(&same_model), false);

        let publication = pool.publish(Arc::clone(&replacement), false);

        assert!(Arc::ptr_eq(
            publication.replaced().expect("same-source replacement"),
            &same_source
        ));
        assert_eq!(publication.evicted().len(), 1);
        assert!(Arc::ptr_eq(&publication.evicted()[0], &same_model));
        let retired = publication.into_retired();
        assert_eq!(retired.len(), 2);
        assert!(!Arc::ptr_eq(&retired[0], &retired[1]));
        assert_eq!(pool.len(), 1);
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("replacement default"),
            &replacement
        ));
    }

    #[test]
    fn same_source_reload_returns_old_generation_without_dropping_in_flight_arc() {
        let drops = Arc::new(AtomicUsize::new(0));
        let old = runtime("old", "/models/model", None, &drops);
        let new = runtime("new", "/models/model", None, &drops);
        let mut pool = pool_with_capacity(2);
        pool.publish(Arc::clone(&old), true);
        let in_flight = pool
            .resolve(Some("old"))
            .expect("valid selector")
            .expect("old generation");

        let publication = pool.publish(Arc::clone(&new), false);
        assert_eq!(publication.key().source(), Path::new("/models/model"));
        assert!(publication.evicted().is_empty());
        assert!(Arc::ptr_eq(
            publication.replaced().expect("old generation returned"),
            &old
        ));
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("same-key default remains"),
            &new
        ));

        drop(publication);
        drop(old);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(in_flight);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn remove_exact_cannot_remove_a_newer_same_source_generation() {
        let drops = Arc::new(AtomicUsize::new(0));
        let old = runtime("old", "/models/model", None, &drops);
        let new = runtime("new", "/models/model", None, &drops);
        let mut pool = pool_with_capacity(2);
        pool.publish(Arc::clone(&old), true);
        let publication = pool.publish(Arc::clone(&new), true);

        assert!(pool.remove_exact(&old).is_none());
        let removed = pool.remove_exact(&new).expect("new generation removed");
        assert!(Arc::ptr_eq(&removed, &new));
        assert!(pool.is_empty());
        drop(publication);
    }

    #[test]
    fn removing_default_promotes_newest_remaining_runtime() {
        let drops = Arc::new(AtomicUsize::new(0));
        let first = runtime("first", "/models/first", None, &drops);
        let second = runtime("second", "/models/second", None, &drops);
        let third = runtime("third", "/models/third", None, &drops);
        let mut pool = pool_with_capacity(3);
        pool.publish(Arc::clone(&first), true);
        pool.publish(Arc::clone(&second), false);
        pool.publish(Arc::clone(&third), false);

        assert!(Arc::ptr_eq(
            &pool
                .remove_source(Path::new("/models/first"))
                .expect("default removed"),
            &first
        ));
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("new fallback default"),
            &third
        ));
        assert!(Arc::ptr_eq(
            &pool.remove_default().expect("fallback removed"),
            &third
        ));
        assert!(Arc::ptr_eq(
            &pool.default_runtime().expect("remaining fallback"),
            &second
        ));
    }

    #[test]
    fn snapshot_retains_exact_generations_after_pool_mutation() {
        let drops = Arc::new(AtomicUsize::new(0));
        let first = runtime("first", "/models/first", None, &drops);
        let second = runtime("second", "/models/second", None, &drops);
        let mut pool = pool_with_capacity(2);
        pool.publish(Arc::clone(&first), true);
        pool.publish(Arc::clone(&second), false);
        let snapshot = pool.snapshot();

        assert_eq!(
            snapshot.default_key().map(RuntimeKey::source),
            Some(Path::new("/models/first"))
        );
        assert_eq!(snapshot.entries().len(), 2);
        let generations = pool.snapshots();
        assert_eq!(generations.len(), 2);
        assert!(Arc::ptr_eq(&generations[0], &first));
        assert!(Arc::ptr_eq(&generations[1], &second));
        assert!(Arc::ptr_eq(
            &snapshot.default_runtime().expect("snapshot default"),
            &first
        ));

        let retired = pool.remove_default().expect("live default removed");
        drop(retired);
        drop(first);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(snapshot);
        drop(generations);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn default_mut_supports_unique_runtime_test_updates() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut pool = RuntimePool::new();
        pool.publish_default(runtime("before", "/models/before", None, &drops));

        let runtime = Arc::get_mut(pool.default_mut().expect("default runtime must exist"))
            .expect("pool owns the only runtime reference");
        runtime.model_id = "after".to_string();

        assert!(pool.resolve(Some("after")).unwrap().is_some());
    }
}
