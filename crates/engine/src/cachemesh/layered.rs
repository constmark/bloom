//! A backend-independent, byte-oriented three-tier block cache.
//!
//! [`LayeredBlockCache`] is deliberately separate from the model-specific
//! [`super::CacheMesh`] adapter.  It is useful for KV blocks, embeddings, or
//! other immutable inference artifacts and only requires a stable string key
//! and bytes.  L1 and L2 are bounded in-process LRU tiers; L3 is an optional
//! persistent backend.  All public operations are thread-safe.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_MAX_ENTRY_BYTES: usize = 256 * 1024 * 1024;
const FILE_MAGIC: &[u8; 8] = b"BLOOMTC1";
const FILE_VERSION: u8 = 1;

/// Stable key used by the generic layered cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayeredCacheKey(String);

impl LayeredCacheKey {
    pub fn new(key: impl Into<String>) -> Result<Self> {
        let key = key.into();
        if key.is_empty() {
            bail!("layered cache key must not be empty");
        }
        if key.len() > 4096 {
            bail!("layered cache key exceeds 4096 bytes");
        }
        Ok(Self(key))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for LayeredCacheKey {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl TryFrom<&str> for LayeredCacheKey {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<String> for LayeredCacheKey {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<&LayeredCacheKey> for LayeredCacheKey {
    type Error = anyhow::Error;

    fn try_from(value: &LayeredCacheKey) -> Result<Self> {
        Self::new(value.0.clone())
    }
}

/// Configuration for [`LayeredBlockCache`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LayeredCacheConfig {
    /// Maximum resident bytes in the fastest tier. Zero disables L1.
    pub l1_capacity_bytes: usize,
    /// Maximum resident bytes in the host-memory tier. Zero disables L2.
    pub l2_capacity_bytes: usize,
    /// Hard limit for one value, applied before any backend is called.
    pub max_entry_bytes: usize,
    /// Write new values to L3 before returning from `put`.
    pub write_through: bool,
}

impl Default for LayeredCacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity_bytes: 64 * 1024 * 1024,
            l2_capacity_bytes: 512 * 1024 * 1024,
            max_entry_bytes: DEFAULT_MAX_ENTRY_BYTES,
            write_through: false,
        }
    }
}

impl LayeredCacheConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_entry_bytes == 0 {
            bail!("layered cache max_entry_bytes must be greater than zero");
        }
        Ok(())
    }
}

/// Backend used for the persistent tier.
pub trait LayeredCacheBackend: Send + Sync {
    fn get(&self, key: &LayeredCacheKey) -> Result<Option<Vec<u8>>>;
    fn put(&self, key: &LayeredCacheKey, value: &[u8]) -> Result<()>;
    fn remove(&self, key: &LayeredCacheKey) -> Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn bytes(&self) -> Option<usize>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayeredCacheTier {
    L1,
    L2,
    L3,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayeredTierMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub items: usize,
    pub bytes: usize,
    pub hit_rate: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayeredCacheMetrics {
    pub enabled: bool,
    pub gets: u64,
    pub puts: u64,
    pub removes: u64,
    pub backend_errors: u64,
    pub l1: LayeredTierMetrics,
    pub l2: LayeredTierMetrics,
    pub l3: LayeredTierMetrics,
}

#[derive(Default)]
struct Counters {
    gets: AtomicU64,
    puts: AtomicU64,
    removes: AtomicU64,
    backend_errors: AtomicU64,
    l1_hits: AtomicU64,
    l1_misses: AtomicU64,
    l1_evictions: AtomicU64,
    l1_promotions: AtomicU64,
    l1_demotions: AtomicU64,
    l2_hits: AtomicU64,
    l2_misses: AtomicU64,
    l2_evictions: AtomicU64,
    l2_promotions: AtomicU64,
    l2_demotions: AtomicU64,
    l3_hits: AtomicU64,
    l3_misses: AtomicU64,
    l3_promotions: AtomicU64,
}

impl Counters {
    fn tier(
        hits: &AtomicU64,
        misses: &AtomicU64,
        evictions: &AtomicU64,
        promotions: &AtomicU64,
        demotions: &AtomicU64,
        items: usize,
        bytes: usize,
    ) -> LayeredTierMetrics {
        let hits = hits.load(Ordering::Relaxed);
        let misses = misses.load(Ordering::Relaxed);
        let lookups = hits.saturating_add(misses);
        LayeredTierMetrics {
            hits,
            misses,
            evictions: evictions.load(Ordering::Relaxed),
            promotions: promotions.load(Ordering::Relaxed),
            demotions: demotions.load(Ordering::Relaxed),
            items,
            bytes,
            hit_rate: if lookups == 0 {
                0.0
            } else {
                hits as f64 / lookups as f64
            },
        }
    }
}

struct Entry {
    value: Arc<[u8]>,
    pins: usize,
}

struct TierState {
    entries: HashMap<LayeredCacheKey, Entry>,
    lru: VecDeque<LayeredCacheKey>,
    bytes: usize,
}

/// A value evicted from an in-process tier, together with the generation of
/// the value that was resident when it was evicted.  The generation is used
/// to prevent an older asynchronous spill from overwriting a newer value in
/// the persistent tier.
type CacheSpill = (LayeredCacheKey, Arc<[u8]>, u64);
type CacheSpills = Vec<CacheSpill>;
type RawCacheSpill = (LayeredCacheKey, Arc<[u8]>);
type RawCacheSpills = Vec<RawCacheSpill>;

impl TierState {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            bytes: 0,
        }
    }

    fn touch(&mut self, key: &LayeredCacheKey) {
        if let Some(pos) = self.lru.iter().position(|candidate| candidate == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key.clone());
    }

    fn insert(&mut self, key: LayeredCacheKey, value: Arc<[u8]>, pins: usize) {
        if let Some(old) = self.entries.remove(&key) {
            self.bytes = self.bytes.saturating_sub(old.value.len());
            if let Some(pos) = self.lru.iter().position(|candidate| candidate == &key) {
                self.lru.remove(pos);
            }
        }
        self.bytes = self.bytes.saturating_add(value.len());
        self.entries.insert(key.clone(), Entry { value, pins });
        self.touch(&key);
    }

    fn remove(&mut self, key: &LayeredCacheKey) -> Option<Entry> {
        let entry = self.entries.remove(key)?;
        self.bytes = self.bytes.saturating_sub(entry.value.len());
        if let Some(pos) = self.lru.iter().position(|candidate| candidate == key) {
            self.lru.remove(pos);
        }
        Some(entry)
    }

    fn evict_until(&mut self, capacity: usize) -> (RawCacheSpills, u64) {
        let mut evicted = Vec::new();
        let mut count = 0;
        while self.bytes > capacity {
            let Some(key) = self.lru.pop_front() else {
                break;
            };
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if entry.pins > 0 {
                self.lru.push_back(key);
                if self.lru.iter().all(|candidate| {
                    self.entries
                        .get(candidate)
                        .is_none_or(|candidate| candidate.pins > 0)
                }) {
                    break;
                }
                continue;
            }
            let Some(entry) = self.remove(&key) else {
                continue;
            };
            evicted.push((key, entry.value));
            count += 1;
        }
        (evicted, count)
    }
}

struct CacheState {
    l1: TierState,
    l2: TierState,
    /// Latest value generation for every key seen by this cache.  Entries are
    /// retained after eviction so a delayed spill can be discarded safely.
    generations: HashMap<LayeredCacheKey, u64>,
}

/// Thread-safe L1/L2/L3 cache with LRU admission and explicit pinning.
pub struct LayeredBlockCache {
    config: LayeredCacheConfig,
    state: Mutex<CacheState>,
    backend: Option<Arc<dyn LayeredCacheBackend>>,
    /// Serializes mutations so a completed put cannot be followed by an
    /// older spill from another put for the same key.
    write_lock: Mutex<()>,
    /// Serializes persistent writes/removes.  Combined with generations this
    /// gives same-key put operations a monotonic last-writer-wins order even
    /// though backend I/O happens outside the cache-state lock.
    persist_lock: Mutex<()>,
    generation: AtomicU64,
    counters: Counters,
}

impl LayeredBlockCache {
    pub fn new(config: LayeredCacheConfig) -> Result<Self> {
        Self::with_backend(config, None)
    }

    pub fn with_backend(
        config: LayeredCacheConfig,
        backend: Option<Arc<dyn LayeredCacheBackend>>,
    ) -> Result<Self> {
        config.validate()?;
        if config.l1_capacity_bytes == 0 && config.l2_capacity_bytes == 0 && backend.is_none() {
            bail!("layered cache needs an in-process capacity or an L3 backend");
        }
        if config.write_through && backend.is_none() {
            bail!("layered cache write_through requires an L3 backend");
        }
        Ok(Self {
            config,
            state: Mutex::new(CacheState {
                l1: TierState::new(),
                l2: TierState::new(),
                generations: HashMap::new(),
            }),
            backend,
            write_lock: Mutex::new(()),
            persist_lock: Mutex::new(()),
            generation: AtomicU64::new(0),
            counters: Counters::default(),
        })
    }

    pub fn config(&self) -> &LayeredCacheConfig {
        &self.config
    }

    pub fn backend(&self) -> Option<&Arc<dyn LayeredCacheBackend>> {
        self.backend.as_ref()
    }

    /// Insert or replace a value. Values are admitted to L1 when they fit;
    /// larger values go directly to L2 or L3.
    pub fn put<K: TryInto<LayeredCacheKey>>(
        &self,
        key: K,
        value: Vec<u8>,
    ) -> Result<LayeredCacheTier>
    where
        K::Error: Into<anyhow::Error>,
    {
        let key = key.try_into().map_err(Into::into)?;
        let _write_guard = lock_recover(&self.write_lock);
        if value.len() > self.config.max_entry_bytes {
            bail!(
                "layered cache value is {} bytes, above max_entry_bytes {}",
                value.len(),
                self.config.max_entry_bytes
            );
        }
        if value.len() > self.config.l1_capacity_bytes
            && value.len() > self.config.l2_capacity_bytes
            && self.backend.is_none()
        {
            bail!(
                "layered cache value is larger than both in-process tiers and no L3 backend is configured"
            );
        }
        self.counters.puts.fetch_add(1, Ordering::Relaxed);
        let value: Arc<[u8]> = Arc::from(value);
        let generation = self.next_generation();
        self.set_generation(&key, generation);
        if self.config.write_through
            && let Err(error) = self.persist_value(&key, &value, generation)
        {
            self.counters.backend_errors.fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }

        let (tier, spills) = self
            .admit(key, value, generation)
            .ok_or_else(|| anyhow!("layered cache put was superseded by a newer value"))?;
        self.persist_spills(spills)?;
        Ok(tier)
    }

    fn next_generation(&self) -> u64 {
        // Zero is reserved for values loaded from a backend that predates the
        // in-process generation map.  Saturating keeps the cache usable after
        // an extremely long-running process exhausts the counter.
        self.generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_add(1))
            })
            .unwrap_or(u64::MAX)
            .saturating_add(1)
    }

    fn set_generation(&self, key: &LayeredCacheKey, generation: u64) {
        let mut state = lock_recover(&self.state);
        state.generations.insert(key.clone(), generation);
    }

    fn current_generation(&self, key: &LayeredCacheKey) -> u64 {
        let state = lock_recover(&self.state);
        state.generations.get(key).copied().unwrap_or(0)
    }

    fn resident_value(&self, key: &LayeredCacheKey) -> Option<Vec<u8>> {
        let mut state = lock_recover(&self.state);
        if let Some(entry) = state.l1.entries.get(key) {
            let value = entry.value.to_vec();
            state.l1.touch(key);
            return Some(value);
        }
        if let Some(entry) = state.l2.entries.get(key) {
            let value = entry.value.to_vec();
            state.l2.touch(key);
            return Some(value);
        }
        None
    }

    fn resident_snapshot(&self, key: &LayeredCacheKey) -> Option<(Arc<[u8]>, u64)> {
        let state = lock_recover(&self.state);
        let value = state
            .l1
            .entries
            .get(key)
            .or_else(|| state.l2.entries.get(key))
            .map(|entry| entry.value.clone())?;
        let generation = state.generations.get(key).copied().unwrap_or(0);
        Some((value, generation))
    }

    fn tag_spills(state: &CacheState, spills: RawCacheSpills) -> CacheSpills {
        spills
            .into_iter()
            .map(|(key, value)| {
                let generation = state.generations.get(&key).copied().unwrap_or(0);
                (key, value, generation)
            })
            .collect()
    }

    fn admit(
        &self,
        key: LayeredCacheKey,
        value: Arc<[u8]>,
        generation: u64,
    ) -> Option<(LayeredCacheTier, CacheSpills)> {
        let mut state = lock_recover(&self.state);
        if state.generations.get(&key).copied().unwrap_or(0) != generation {
            return None;
        }
        state.generations.insert(key.clone(), generation);
        let pins = state
            .l1
            .remove(&key)
            .or_else(|| state.l2.remove(&key))
            .map(|entry| entry.pins)
            .unwrap_or(0);
        let (tier, spills) = if value.len() <= self.config.l1_capacity_bytes
            && self.config.l1_capacity_bytes > 0
        {
            state.l1.insert(key, value, pins);
            let (raw_spills, count) = state.l1.evict_until(self.config.l1_capacity_bytes);
            let spills = Self::tag_spills(&state, raw_spills);
            self.counters
                .l1_evictions
                .fetch_add(count, Ordering::Relaxed);
            self.counters.l1_demotions.fetch_add(
                u64::try_from(spills.len()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            (LayeredCacheTier::L1, spills)
        } else if value.len() <= self.config.l2_capacity_bytes && self.config.l2_capacity_bytes > 0
        {
            state.l2.insert(key, value, pins);
            let (raw_spills, count) = state.l2.evict_until(self.config.l2_capacity_bytes);
            let spills = Self::tag_spills(&state, raw_spills);
            self.counters
                .l2_evictions
                .fetch_add(count, Ordering::Relaxed);
            (LayeredCacheTier::L2, spills)
        } else {
            (LayeredCacheTier::L3, vec![(key, value, generation)])
        };

        // L1 evictions are demoted into L2. L2 evictions go directly to L3;
        // reinserting them would reverse the LRU order and retain the wrong
        // victim under pressure.
        let mut l3_spills: CacheSpills = Vec::new();
        if tier == LayeredCacheTier::L1 {
            for (spill_key, spill_value, spill_generation) in spills {
                if spill_value.len() <= self.config.l2_capacity_bytes
                    && self.config.l2_capacity_bytes > 0
                {
                    state.l2.insert(spill_key, spill_value, 0);
                    let (raw_more, count) = state.l2.evict_until(self.config.l2_capacity_bytes);
                    let more = Self::tag_spills(&state, raw_more);
                    self.counters
                        .l2_evictions
                        .fetch_add(count, Ordering::Relaxed);
                    self.counters.l2_demotions.fetch_add(
                        u64::try_from(more.len()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                    l3_spills.extend(more);
                } else {
                    l3_spills.push((spill_key, spill_value, spill_generation));
                }
            }
        } else if tier == LayeredCacheTier::L2 {
            self.counters.l2_demotions.fetch_add(
                u64::try_from(spills.len()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            l3_spills = spills;
        } else {
            l3_spills = spills;
        }
        Some((tier, l3_spills))
    }

    fn persist_value(&self, key: &LayeredCacheKey, value: &[u8], generation: u64) -> Result<()> {
        let Some(backend) = &self.backend else {
            return Ok(());
        };
        let _persist_guard = lock_recover(&self.persist_lock);
        if self.current_generation(key) != generation {
            return Ok(());
        }
        backend.put(key, value)
    }

    fn persist_spills(&self, spills: CacheSpills) -> Result<()> {
        if spills.is_empty() || self.config.write_through && self.backend.is_none() {
            return Ok(());
        }
        let Some(backend) = &self.backend else {
            return Ok(());
        };
        // Do not let an older spill overtake a newer spill for the same key.
        // The generation check is deliberately performed while holding the
        // persistence lock, so a remove cannot race a backend write either.
        let _persist_guard = lock_recover(&self.persist_lock);
        for (key, value, generation) in spills {
            let mut candidate = value;
            let mut candidate_generation = generation;
            loop {
                if self.current_generation(&key) != candidate_generation {
                    let Some((latest, latest_generation)) = self.resident_snapshot(&key) else {
                        if let Err(error) = backend.remove(&key) {
                            self.counters.backend_errors.fetch_add(1, Ordering::Relaxed);
                            return Err(error);
                        }
                        break;
                    };
                    candidate = latest;
                    candidate_generation = latest_generation;
                    continue;
                }
                if let Err(error) = backend.put(&key, &candidate) {
                    self.counters.backend_errors.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
                if self.current_generation(&key) == candidate_generation {
                    break;
                }
                let Some((latest, latest_generation)) = self.resident_snapshot(&key) else {
                    if let Err(error) = backend.remove(&key) {
                        self.counters.backend_errors.fetch_add(1, Ordering::Relaxed);
                        return Err(error);
                    }
                    break;
                };
                candidate = latest;
                candidate_generation = latest_generation;
            }
        }
        Ok(())
    }

    /// Read and promote a value to L1 when it fits.
    pub fn get<K: TryInto<LayeredCacheKey>>(&self, key: K) -> Result<Option<Vec<u8>>>
    where
        K::Error: Into<anyhow::Error>,
    {
        let key = key.try_into().map_err(Into::into)?;
        self.counters.gets.fetch_add(1, Ordering::Relaxed);
        let observed_generation;
        {
            let mut state = lock_recover(&self.state);
            if let Some(entry) = state.l1.entries.get(&key) {
                let value = entry.value.to_vec();
                state.l1.touch(&key);
                self.counters.l1_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Some(value));
            }
            self.counters.l1_misses.fetch_add(1, Ordering::Relaxed);
            if let Some(entry) = state.l2.remove(&key) {
                let value = entry.value.clone();
                let pins = entry.pins;
                self.counters.l2_hits.fetch_add(1, Ordering::Relaxed);
                if value.len() <= self.config.l1_capacity_bytes && self.config.l1_capacity_bytes > 0
                {
                    state.l1.insert(key.clone(), value.clone(), pins);
                    let (raw_spills, count) = state.l1.evict_until(self.config.l1_capacity_bytes);
                    let spills = Self::tag_spills(&state, raw_spills);
                    self.counters
                        .l1_evictions
                        .fetch_add(count, Ordering::Relaxed);
                    self.counters.l1_promotions.fetch_add(1, Ordering::Relaxed);
                    drop(state);
                    self.persist_spills(spills)?;
                } else {
                    // Keep a value in L2 when it does not fit L1. A lookup
                    // must not silently discard the only resident copy.
                    state.l2.insert(key.clone(), value.clone(), pins);
                    drop(state);
                }
                return Ok(Some(value.to_vec()));
            }
            self.counters.l2_misses.fetch_add(1, Ordering::Relaxed);
            observed_generation = state.generations.get(&key).copied().unwrap_or(0);
        }

        let Some(backend) = &self.backend else {
            return Ok(None);
        };
        let (value, spills) = {
            // Serialize backend reads with writes/removes so the generation
            // check and promotion form one linearizable cache operation.
            let _persist_guard = lock_recover(&self.persist_lock);
            let value = match backend.get(&key) {
                Ok(value) => value,
                Err(error) => {
                    self.counters.backend_errors.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
            };
            let Some(value) = value else {
                self.counters.l3_misses.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            };
            if self.current_generation(&key) != observed_generation {
                // A newer put/remove won while the backend read was in
                // flight. Never resurrect the stale backend value.
                return Ok(self.resident_value(&key));
            }
            if value.len() > self.config.max_entry_bytes {
                bail!("layered cache backend returned an oversized value");
            }
            let value_arc: Arc<[u8]> = Arc::from(value.clone());
            let Some((_tier, spills)) = self.admit(key.clone(), value_arc, observed_generation)
            else {
                return Ok(self.resident_value(&key));
            };
            self.counters.l3_hits.fetch_add(1, Ordering::Relaxed);
            self.counters.l3_promotions.fetch_add(1, Ordering::Relaxed);
            (value, spills)
        };
        self.persist_spills(spills)?;
        Ok(Some(value))
    }

    pub fn pin<K: TryInto<LayeredCacheKey>>(&self, key: K) -> Result<bool>
    where
        K::Error: Into<anyhow::Error>,
    {
        let key = key.try_into().map_err(Into::into)?;
        let mut state = lock_recover(&self.state);
        if let Some(entry) = state.l1.entries.get_mut(&key) {
            entry.pins = entry.pins.saturating_add(1);
            state.l1.touch(&key);
            return Ok(true);
        }
        if let Some(entry) = state.l2.entries.get_mut(&key) {
            entry.pins = entry.pins.saturating_add(1);
            state.l2.touch(&key);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn unpin<K: TryInto<LayeredCacheKey>>(&self, key: K) -> Result<bool>
    where
        K::Error: Into<anyhow::Error>,
    {
        let key = key.try_into().map_err(Into::into)?;
        let mut state = lock_recover(&self.state);
        if let Some(entry) = state.l1.entries.get_mut(&key) {
            entry.pins = entry.pins.saturating_sub(1);
            return Ok(true);
        }
        if let Some(entry) = state.l2.entries.get_mut(&key) {
            entry.pins = entry.pins.saturating_sub(1);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn remove<K: TryInto<LayeredCacheKey>>(&self, key: K) -> Result<bool>
    where
        K::Error: Into<anyhow::Error>,
    {
        let key = key.try_into().map_err(Into::into)?;
        let _write_guard = lock_recover(&self.write_lock);
        self.counters.removes.fetch_add(1, Ordering::Relaxed);
        let generation = self.next_generation();
        let removed = {
            let mut state = lock_recover(&self.state);
            // Invalidate any spill produced before this remove.  The
            // persistence lock below ensures a concurrent backend write is
            // ordered before the remove itself.
            state.generations.insert(key.clone(), generation);
            state.l1.remove(&key).is_some() || state.l2.remove(&key).is_some()
        };
        let backend_removed = if let Some(backend) = &self.backend {
            let _persist_guard = lock_recover(&self.persist_lock);
            if self.current_generation(&key) != generation {
                false
            } else {
                match backend.remove(&key) {
                    Ok(()) => true,
                    Err(error) => {
                        self.counters.backend_errors.fetch_add(1, Ordering::Relaxed);
                        return Err(error);
                    }
                }
            }
        } else {
            false
        };
        Ok(removed || backend_removed)
    }

    pub fn clear_memory(&self) {
        let mut state = lock_recover(&self.state);
        state.l1 = TierState::new();
        state.l2 = TierState::new();
    }

    pub fn metrics(&self) -> LayeredCacheMetrics {
        let (l1_items, l1_bytes, l2_items, l2_bytes) = {
            let state = lock_recover(&self.state);
            (
                state.l1.entries.len(),
                state.l1.bytes,
                state.l2.entries.len(),
                state.l2.bytes,
            )
        };
        let l3_items = self
            .backend
            .as_ref()
            .map(|backend| backend.len())
            .unwrap_or(0);
        let l3_bytes = self
            .backend
            .as_ref()
            .and_then(|backend| backend.bytes())
            .unwrap_or(0);
        LayeredCacheMetrics {
            enabled: true,
            gets: self.counters.gets.load(Ordering::Relaxed),
            puts: self.counters.puts.load(Ordering::Relaxed),
            removes: self.counters.removes.load(Ordering::Relaxed),
            backend_errors: self.counters.backend_errors.load(Ordering::Relaxed),
            l1: Counters::tier(
                &self.counters.l1_hits,
                &self.counters.l1_misses,
                &self.counters.l1_evictions,
                &self.counters.l1_promotions,
                &self.counters.l1_demotions,
                l1_items,
                l1_bytes,
            ),
            l2: Counters::tier(
                &self.counters.l2_hits,
                &self.counters.l2_misses,
                &self.counters.l2_evictions,
                &self.counters.l2_promotions,
                &self.counters.l2_demotions,
                l2_items,
                l2_bytes,
            ),
            l3: Counters::tier(
                &self.counters.l3_hits,
                &self.counters.l3_misses,
                &AtomicU64::new(0),
                &self.counters.l3_promotions,
                &AtomicU64::new(0),
                l3_items,
                l3_bytes,
            ),
        }
    }
}

/// A bounded filesystem implementation for the L3 tier.
pub struct FileSystemLayeredCache {
    root: PathBuf,
    max_entry_bytes: usize,
}

impl FileSystemLayeredCache {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        Self::with_max_entry_bytes(root, DEFAULT_MAX_ENTRY_BYTES)
    }

    pub fn with_max_entry_bytes(root: impl Into<PathBuf>, max_entry_bytes: usize) -> Result<Self> {
        if max_entry_bytes == 0 {
            bail!("filesystem layered cache max_entry_bytes must be greater than zero");
        }
        let root = root.into();
        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create layered cache root '{}'", root.display()))?;
        let root = root.canonicalize()?;
        Ok(Self {
            root,
            max_entry_bytes,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for(&self, key: &LayeredCacheKey) -> PathBuf {
        let digest = Sha256::digest(key.as_str().as_bytes());
        let hex = hex_digest(&digest);
        self.root.join(&hex[..2]).join(format!("{hex}.bin"))
    }

    fn validate_regular_parent(&self, path: &Path) -> Result<()> {
        let Some(parent) = path.parent() else {
            bail!("layered cache path has no parent");
        };
        if let Ok(metadata) = fs::symlink_metadata(parent)
            && (metadata.file_type().is_symlink() || !metadata.is_dir())
        {
            bail!("layered cache parent must be a regular directory");
        }
        Ok(())
    }
}

impl LayeredCacheBackend for FileSystemLayeredCache {
    fn get(&self, key: &LayeredCacheKey) -> Result<Option<Vec<u8>>> {
        let path = self.path_for(key);
        self.validate_regular_parent(&path)?;
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            bail!("layered cache entry must be a regular file");
        }
        let max_file_bytes = self
            .max_entry_bytes
            .checked_add(FILE_MAGIC.len() + 1 + 8 + 32)
            .ok_or_else(|| anyhow!("layered cache file size overflows usize"))?;
        if metadata.len() > u64::try_from(max_file_bytes).unwrap_or(u64::MAX) {
            bail!("layered cache entry exceeds the configured read limit");
        }
        let mut file = File::open(&path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != FILE_MAGIC {
            bail!("layered cache entry has an unsupported format");
        }
        let mut version = [0u8; 1];
        file.read_exact(&mut version)?;
        if version[0] != FILE_VERSION {
            bail!("layered cache entry has unsupported version {}", version[0]);
        }
        let mut length = [0u8; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        let length =
            usize::try_from(length).context("layered cache entry length overflows usize")?;
        if length > self.max_entry_bytes {
            bail!("layered cache entry exceeds the configured read limit");
        }
        let mut checksum = [0u8; 32];
        file.read_exact(&mut checksum)?;
        let mut value = vec![0u8; length];
        file.read_exact(&mut value)?;
        let mut trailing = [0u8; 1];
        if file.read(&mut trailing)? != 0 {
            bail!("layered cache entry has trailing bytes");
        }
        if Sha256::digest(&value).as_slice() != checksum {
            bail!("layered cache entry checksum mismatch");
        }
        Ok(Some(value))
    }

    fn put(&self, key: &LayeredCacheKey, value: &[u8]) -> Result<()> {
        if value.len() > self.max_entry_bytes {
            bail!("layered cache value exceeds the configured write limit");
        }
        let path = self.path_for(key);
        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("layered cache path has no parent"))?;
        fs::create_dir_all(parent)?;
        self.validate_regular_parent(&path)?;
        if let Ok(metadata) = fs::symlink_metadata(&path)
            && (metadata.file_type().is_symlink() || !metadata.is_file())
        {
            bail!("layered cache entry must be a regular file");
        }
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        let file = temporary.as_file_mut();
        file.write_all(FILE_MAGIC)?;
        file.write_all(&[FILE_VERSION])?;
        file.write_all(&u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes())?;
        file.write_all(Sha256::digest(value).as_slice())?;
        file.write_all(value)?;
        file.sync_all()?;
        temporary.persist(&path).map_err(|error| error.error)?;
        Ok(())
    }

    fn remove(&self, key: &LayeredCacheKey) -> Result<()> {
        let path = self.path_for(key);
        self.validate_regular_parent(&path)?;
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    fn len(&self) -> usize {
        count_files(&self.root).unwrap_or(0)
    }

    fn bytes(&self) -> Option<usize> {
        sum_file_bytes(&self.root).ok()
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn count_files(root: &Path) -> Result<usize> {
    let mut count: usize = 0;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            count = count.saturating_add(count_files(&entry.path())?);
        } else if file_type.is_file() {
            count += 1;
        }
    }
    Ok(count)
}

fn sum_file_bytes(root: &Path) -> Result<usize> {
    let mut bytes: usize = 0;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            bytes = bytes.saturating_add(sum_file_bytes(&entry.path())?);
        } else if file_type.is_file() {
            bytes = bytes
                .saturating_add(usize::try_from(entry.metadata()?.len()).unwrap_or(usize::MAX));
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailingBackend;

    impl LayeredCacheBackend for FailingBackend {
        fn get(&self, _key: &LayeredCacheKey) -> Result<Option<Vec<u8>>> {
            Err(anyhow!("test backend get failure"))
        }

        fn put(&self, _key: &LayeredCacheKey, _value: &[u8]) -> Result<()> {
            Err(anyhow!("test backend put failure"))
        }

        fn remove(&self, _key: &LayeredCacheKey) -> Result<()> {
            Err(anyhow!("test backend remove failure"))
        }

        fn len(&self) -> usize {
            0
        }

        fn bytes(&self) -> Option<usize> {
            Some(0)
        }
    }

    fn cache(config: LayeredCacheConfig) -> LayeredBlockCache {
        LayeredBlockCache::new(config).expect("valid cache config")
    }

    #[test]
    fn promotes_and_spills_between_memory_tiers() {
        let cache = cache(LayeredCacheConfig {
            l1_capacity_bytes: 4,
            l2_capacity_bytes: 8,
            max_entry_bytes: 16,
            ..Default::default()
        });
        assert_eq!(
            cache.put("a", vec![1, 2, 3, 4]).unwrap(),
            LayeredCacheTier::L1
        );
        assert_eq!(
            cache.put("b", vec![5, 6, 7, 8]).unwrap(),
            LayeredCacheTier::L1
        );
        let metrics = cache.metrics();
        assert_eq!(metrics.l1.items, 1);
        assert_eq!(metrics.l2.items, 1);
        assert_eq!(cache.get("a").unwrap(), Some(vec![1, 2, 3, 4]));
        assert_eq!(cache.metrics().l2.hits, 1);
    }

    #[test]
    fn l3_spill_survives_memory_clear_and_concurrent_restore() {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(FileSystemLayeredCache::new(directory.path()).unwrap());
        let cache = Arc::new(
            LayeredBlockCache::with_backend(
                LayeredCacheConfig {
                    l1_capacity_bytes: 4,
                    l2_capacity_bytes: 4,
                    max_entry_bytes: 16,
                    ..Default::default()
                },
                Some(backend),
            )
            .unwrap(),
        );

        // `a` is demoted to L2 by `b`, then spilled to L3 by `c` when L2 is
        // full. This verifies that L3 is populated by normal eviction rather
        // than only by write-through puts.
        cache.put("a", vec![1, 2, 3, 4]).unwrap();
        cache.put("b", vec![5, 6, 7, 8]).unwrap();
        cache.put("c", vec![9, 10, 11, 12]).unwrap();
        assert_eq!(cache.metrics().l3.items, 1);

        cache.clear_memory();
        let workers = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for _ in 0..16 {
                        assert_eq!(cache.get("a").unwrap(), Some(vec![1, 2, 3, 4]));
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        let metrics = cache.metrics();
        assert!(metrics.l3.hits >= 1);
        assert!(metrics.l3.promotions >= 1);
        assert_eq!(metrics.l1.items, 1);
        assert_eq!(metrics.l1.bytes, 4);
    }

    #[test]
    fn pinned_entries_survive_capacity_pressure() {
        let cache = cache(LayeredCacheConfig {
            l1_capacity_bytes: 4,
            l2_capacity_bytes: 8,
            max_entry_bytes: 16,
            ..Default::default()
        });
        cache.put("a", vec![1, 2, 3, 4]).unwrap();
        assert!(cache.pin("a").unwrap());
        cache.put("b", vec![5, 6, 7, 8]).unwrap();
        assert_eq!(cache.get("a").unwrap(), Some(vec![1, 2, 3, 4]));
        // The pinned entry survives; the unpinned value is evicted to keep
        // the tier within its byte budget.
        assert_eq!(cache.metrics().l1.items, 1);
    }

    #[test]
    fn oversized_value_without_l3_fails_before_mutating_memory() {
        let cache = cache(LayeredCacheConfig {
            l1_capacity_bytes: 4,
            l2_capacity_bytes: 8,
            max_entry_bytes: 16,
            ..Default::default()
        });
        assert!(cache.put("too-large-for-memory", vec![0; 9]).is_err());
        let metrics = cache.metrics();
        assert_eq!(metrics.puts, 0);
        assert_eq!(metrics.l1.items, 0);
        assert_eq!(metrics.l2.items, 0);
        assert_eq!(cache.get("too-large-for-memory").unwrap(), None);
    }

    #[test]
    fn l2_eviction_keeps_newest_values_and_drops_oldest() {
        let cache = cache(LayeredCacheConfig {
            l1_capacity_bytes: 0,
            l2_capacity_bytes: 8,
            max_entry_bytes: 16,
            ..Default::default()
        });
        cache.put("a", vec![1, 2, 3, 4]).unwrap();
        cache.put("b", vec![5, 6, 7, 8]).unwrap();
        cache.put("c", vec![9, 10, 11, 12]).unwrap();
        assert_eq!(cache.get("a").unwrap(), None);
        assert_eq!(cache.get("b").unwrap(), Some(vec![5, 6, 7, 8]));
        assert_eq!(cache.get("c").unwrap(), Some(vec![9, 10, 11, 12]));
    }

    #[test]
    fn write_through_requires_an_l3_backend() {
        assert!(
            LayeredBlockCache::new(LayeredCacheConfig {
                write_through: true,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn concurrent_put_and_get_keeps_values_well_formed() {
        let cache = Arc::new(cache(LayeredCacheConfig {
            l1_capacity_bytes: 64,
            l2_capacity_bytes: 256,
            max_entry_bytes: 64,
            ..Default::default()
        }));
        let workers = (0..8)
            .map(|worker| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for sequence in 0..64 {
                        let key = format!("worker-{worker}/block-{}", sequence % 8);
                        let value = vec![worker as u8; 8];
                        cache.put(key.as_str(), value.clone()).unwrap();
                        // Another worker may evict this key between put and
                        // get; any resident value must still be intact.
                        if let Some(actual) = cache.get(key.as_str()).unwrap() {
                            assert_eq!(actual, value);
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
        let metrics = cache.metrics();
        assert_eq!(metrics.puts, 8 * 64);
        assert_eq!(metrics.gets, 8 * 64);
    }

    #[test]
    fn backend_spill_failure_is_returned_and_counted() {
        let cache = LayeredBlockCache::with_backend(
            LayeredCacheConfig {
                l1_capacity_bytes: 4,
                l2_capacity_bytes: 4,
                max_entry_bytes: 16,
                ..Default::default()
            },
            Some(Arc::new(FailingBackend)),
        )
        .unwrap();
        cache.put("a", vec![1, 2, 3, 4]).unwrap();
        cache.put("b", vec![5, 6, 7, 8]).unwrap();

        // Inserting a third block forces the oldest value through L3. A
        // backend failure must be visible to the caller and metrics.
        assert!(cache.put("c", vec![9, 10, 11, 12]).is_err());
        assert_eq!(cache.metrics().backend_errors, 1);
        assert!(cache.get("a").is_err());
    }

    #[test]
    fn filesystem_backend_round_trips_and_detects_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let backend = FileSystemLayeredCache::new(directory.path()).unwrap();
        let key = LayeredCacheKey::new("model/layer/0/block/1").unwrap();
        backend.put(&key, b"payload").unwrap();
        assert_eq!(backend.get(&key).unwrap(), Some(b"payload".to_vec()));
        let path = backend.path_for(&key);
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(path, bytes).unwrap();
        assert!(backend.get(&key).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_backend_rejects_symlinked_shard_directories() {
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let backend = FileSystemLayeredCache::new(directory.path()).unwrap();
        let key = LayeredCacheKey::new("symlink-test").unwrap();
        let path = backend.path_for(&key);
        std::os::unix::fs::symlink(outside.path(), path.parent().unwrap()).unwrap();
        assert!(backend.get(&key).is_err());
        assert!(backend.remove(&key).is_err());
    }
}
