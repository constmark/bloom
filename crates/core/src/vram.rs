use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::constants::GIB;
#[allow(unused_imports)]
use crate::error::ResourceError;
use crate::resource::{
    BackendLease, CacheHandle, ModelResidencyRecord, ModelResourceSnapshot, OffloadCallback,
    ResourceSnapshot, ResourceTicket,
};
use crate::types::{CacheKind, MemoryTopology, ResidencyStrategy, ResourcePriority};
use crate::unified_memory::UnifiedMemoryConfig;

/// Backward-compatible alias for ResourceCoordinator.
pub type VRAMCoordinator = ResourceCoordinator;

static GLOBAL_COORDINATOR: OnceLock<ResourceCoordinator> = OnceLock::new();

/// Unified resource coordinator covering RAM, VRAM, cache and state budgets.
pub struct ResourceCoordinator {
    ram_budget: usize,
    vram_budget: usize,
    memory_topology: MemoryTopology,
    inner: Mutex<ResourceCoordinatorInner>,
    next_lease_id: AtomicU64,
    #[allow(dead_code)]
    next_cache_id: AtomicU64,
}

struct ResourceCoordinatorInner {
    ram_allocated: usize,
    vram_allocated: usize,
    residencies: HashMap<String, ModelResidencyRecord>,
    leases: HashMap<u64, LeaseRecord>,
    caches: HashMap<u64, CacheRecord>,
}

/// Internal lightweight lease record (the public BackendLease is constructed on demand).
struct LeaseRecord {
    model_id: String,
    #[allow(dead_code)]
    backend_name: String,
    granted_ram: usize,
    granted_vram: usize,
    #[allow(dead_code)]
    created_at: Instant,
}

/// Internal cache record.
struct CacheRecord {
    #[allow(dead_code)]
    model_id: String,
    #[allow(dead_code)]
    cache_kind: CacheKind,
    bytes: usize,
    priority: ResourcePriority,
}

impl ResourceCoordinator {
    /// Create a new ResourceCoordinator with explicit budgets and memory topology.
    pub fn new(ram_budget: usize, vram_budget: usize, memory_topology: MemoryTopology) -> Self {
        Self {
            ram_budget,
            vram_budget,
            memory_topology,
            inner: Mutex::new(ResourceCoordinatorInner {
                ram_allocated: 0,
                vram_allocated: 0,
                residencies: HashMap::new(),
                leases: HashMap::new(),
                caches: HashMap::new(),
            }),
            next_lease_id: AtomicU64::new(1),
            next_cache_id: AtomicU64::new(1),
        }
    }

    /// Create with a single unified budget (convenience for UMA).
    pub fn new_unified(budget: usize) -> Self {
        Self::new(budget, budget, MemoryTopology::Unified)
    }

    /// Create a coordinator from unified-memory runtime configuration.
    pub fn from_unified_memory_config(
        config: &UnifiedMemoryConfig,
        detected_ram_bytes: usize,
        detected_vram_bytes: usize,
        detected_topology: MemoryTopology,
    ) -> Self {
        let topology = config.topology.unwrap_or(detected_topology);
        let ram_budget = config.effective_ram_budget(detected_ram_bytes);
        let vram_budget = config.effective_vram_budget(detected_vram_bytes);
        Self::new(ram_budget, vram_budget, topology)
    }

    /// Unified effective budget (for UMA: single pool, for Discrete: separate).
    fn effective_budget(&self) -> (usize, usize) {
        match self.memory_topology {
            MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
                // UMA / shared system memory: single pool, use max of ram and vram budgets
                let unified = self.ram_budget.max(self.vram_budget);
                (unified, unified)
            }
            MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
                (self.ram_budget, self.vram_budget)
            }
        }
    }

    /// Reserve resources for a model. Returns a BackendLease on success.
    pub fn reserve(
        &self,
        ticket: ResourceTicket,
        offload_cb: OffloadCallback,
    ) -> Result<BackendLease, ResourceError> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        // Check if already loaded
        if let Some(existing) = inner.residencies.get(&ticket.model_id) {
            return Err(ResourceError::AlreadyLoaded {
                model_id: ticket.model_id.clone(),
                lease_id: existing.lease_id,
            });
        }

        let (ram_budget, vram_budget) = self.effective_budget();
        let needed_ram = ticket.ram_bytes;
        let needed_vram = ticket.vram_bytes;
        let needed_cache = ticket.cache_bytes;

        // For UMA: check total footprint against unified budget
        let total_needed = needed_ram + needed_vram + needed_cache;
        let total_used = inner.ram_allocated + inner.vram_allocated;

        let can_fit = match self.memory_topology {
            MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
                total_used + total_needed <= ram_budget
            }
            MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
                inner.ram_allocated + needed_ram <= ram_budget
                    && inner.vram_allocated + needed_vram <= vram_budget
            }
        };

        let mut evicted_models: Vec<String> = Vec::new();

        if !can_fit {
            // Try eviction
            evicted_models = self.try_evict(&mut inner, needed_ram, needed_vram, needed_cache)?;

            // Re-check after eviction
            let total_used_after = inner.ram_allocated + inner.vram_allocated;
            let can_fit_after = match self.memory_topology {
                MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
                    total_used_after + total_needed <= ram_budget
                }
                MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
                    inner.ram_allocated + needed_ram <= ram_budget
                        && inner.vram_allocated + needed_vram <= vram_budget
                }
            };

            if !can_fit_after {
                let deficit = match self.memory_topology {
                    MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
                        (total_used_after + total_needed).saturating_sub(ram_budget)
                    }
                    MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
                        let ram_deficit =
                            (inner.ram_allocated + needed_ram).saturating_sub(ram_budget);
                        let vram_deficit =
                            (inner.vram_allocated + needed_vram).saturating_sub(vram_budget);
                        ram_deficit.max(vram_deficit)
                    }
                };
                return Err(ResourceError::BudgetExceeded { deficit });
            }
        }

        // Allocate
        let lease_id = self.next_lease_id.fetch_add(1, Ordering::SeqCst);
        let backend_name = ticket
            .preferred_backend
            .clone()
            .unwrap_or_else(|| "default".to_string());

        inner.ram_allocated += needed_ram;
        inner.vram_allocated += needed_vram;

        let now = Instant::now();

        let record = ModelResidencyRecord {
            model_id: ticket.model_id.clone(),
            lease_id,
            ram_bytes: needed_ram,
            vram_bytes: needed_vram,
            cache_bytes: needed_cache,
            priority: ticket.priority,
            strategy: ticket.strategy,
            backend_name: backend_name.clone(),
            offload_cb,
            last_accessed: now,
        };
        inner.residencies.insert(ticket.model_id.clone(), record);

        inner.leases.insert(
            lease_id,
            LeaseRecord {
                model_id: ticket.model_id.clone(),
                backend_name: backend_name.clone(),
                granted_ram: needed_ram,
                granted_vram: needed_vram,
                created_at: now,
            },
        );

        tracing::info!(
            "Reserved resources for model '{}': ram={} vram={} cache={} lease_id={}",
            ticket.model_id,
            needed_ram,
            needed_vram,
            needed_cache,
            lease_id
        );

        Ok(BackendLease {
            lease_id,
            ticket,
            granted_backend: backend_name,
            granted_ram: needed_ram,
            granted_vram: needed_vram,
            degraded: false,
            degraded_reason: None,
            evicted_models,
            created_at: now,
        })
    }

    /// Release a lease by lease_id.
    pub fn release(&self, lease_id: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(lease) = inner.leases.remove(&lease_id) {
            inner.ram_allocated = inner.ram_allocated.saturating_sub(lease.granted_ram);
            inner.vram_allocated = inner.vram_allocated.saturating_sub(lease.granted_vram);
            inner.residencies.remove(&lease.model_id);
            tracing::info!(
                "Released lease {} for model '{}': freed ram={} vram={}",
                lease_id,
                lease.model_id,
                lease.granted_ram,
                lease.granted_vram
            );
        }
    }

    /// Release a lease by model_id (backward compatible with record_unload).
    pub fn release_by_model(&self, model_id: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(record) = inner.residencies.remove(model_id) {
            inner.ram_allocated = inner.ram_allocated.saturating_sub(record.ram_bytes);
            inner.vram_allocated = inner.vram_allocated.saturating_sub(record.vram_bytes);
            inner.leases.remove(&record.lease_id);
            tracing::info!(
                "Released model '{}': freed ram={} vram={}",
                model_id,
                record.ram_bytes,
                record.vram_bytes
            );
        }
    }

    /// Register a cache handle in the coordinator.
    pub fn register_cache(&self, handle: CacheHandle) -> Result<(), ResourceError> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (ram_budget, _) = self.effective_budget();
        let total_used = inner.ram_allocated + inner.vram_allocated;

        if matches!(
            self.memory_topology,
            MemoryTopology::Unified | MemoryTopology::SharedSystemMemory
        ) && total_used + handle.bytes > ram_budget
        {
            return Err(ResourceError::InsufficientUnifiedMemory {
                requested: handle.bytes,
                available: ram_budget.saturating_sub(total_used),
            });
        }

        inner.ram_allocated += handle.bytes;
        inner.caches.insert(
            handle.handle_id,
            CacheRecord {
                model_id: handle.model_id,
                cache_kind: handle.cache_kind,
                bytes: handle.bytes,
                priority: handle.priority,
            },
        );
        Ok(())
    }

    /// Release a cache handle.
    pub fn release_cache(&self, handle_id: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cache) = inner.caches.remove(&handle_id) {
            inner.ram_allocated = inner.ram_allocated.saturating_sub(cache.bytes);
        }
    }

    /// Get a snapshot of current resource state.
    pub fn snapshot(&self) -> ResourceSnapshot {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        ResourceSnapshot {
            ram_budget: self.ram_budget,
            vram_budget: self.vram_budget,
            ram_allocated: inner.ram_allocated,
            vram_allocated: inner.vram_allocated,
            memory_topology: self.memory_topology,
            model_count: inner.residencies.len(),
            cache_count: inner.caches.len(),
            lease_count: inner.leases.len(),
        }
    }

    /// Get per-model residency snapshots sorted by model id.
    pub fn model_snapshots(&self) -> Vec<ModelResourceSnapshot> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut models = inner
            .residencies
            .values()
            .map(|record| ModelResourceSnapshot {
                model_id: record.model_id.clone(),
                lease_id: record.lease_id,
                ram_bytes: record.ram_bytes,
                vram_bytes: record.vram_bytes,
                cache_bytes: record.cache_bytes,
                priority: record.priority,
                strategy: record.strategy,
                backend_name: record.backend_name.clone(),
            })
            .collect::<Vec<_>>();
        models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        models
    }

    /// Look up lease_id for a given model_id.
    pub fn lease_id_for_model(&self, model_id: &str) -> Option<u64> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.residencies.get(model_id).map(|r| r.lease_id)
    }

    /// Look up residency strategy for a given model_id.
    pub fn residency_strategy_for_model(&self, model_id: &str) -> Option<ResidencyStrategy> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.residencies.get(model_id).map(|r| r.strategy)
    }

    /// Touch a model's last_accessed timestamp (for LRU eviction).
    pub fn touch(&self, model_id: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(record) = inner.residencies.get_mut(model_id) {
            record.last_accessed = Instant::now();
        }
    }

    /// Evict low-priority entries to free the requested amount.
    /// Returns list of evicted model IDs.
    fn try_evict(
        &self,
        inner: &mut ResourceCoordinatorInner,
        needed_ram: usize,
        needed_vram: usize,
        needed_cache: usize,
    ) -> Result<Vec<String>, ResourceError> {
        let _span =
            tracing::info_span!("vram.evict", needed_ram, needed_vram, needed_cache).entered();
        // Collect evictable candidates: skip Critical and Resident strategy
        let mut candidates: Vec<(String, ResourcePriority, Instant, usize, usize, bool)> =
            Vec::new(); // (id, priority, last_accessed, ram, vram, is_cache)

        for (id, record) in &inner.residencies {
            if record.priority == ResourcePriority::Critical
                || record.strategy == ResidencyStrategy::Resident
            {
                continue;
            }
            candidates.push((
                id.clone(),
                record.priority,
                record.last_accessed,
                record.ram_bytes,
                record.vram_bytes,
                false,
            ));
        }

        for (hid, record) in &inner.caches {
            if record.priority == ResourcePriority::Critical {
                continue;
            }
            candidates.push((
                hid.to_string(),
                record.priority,
                Instant::now(), // caches don't track LRU, evict early
                record.bytes,
                0,
                true,
            ));
        }

        // Sort: low priority first, then oldest last_accessed
        candidates.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)).then_with(|| {
                // Within same priority and time, prefer evicting caches before models
                b.5.cmp(&a.5)
            })
        });

        let mut evicted = Vec::new();
        let mut freed_ram: usize = 0;
        let mut freed_vram: usize = 0;

        let (ram_budget, vram_budget) = self.effective_budget();

        let need_more =
            |inner: &ResourceCoordinatorInner, freed_ram: usize, freed_vram: usize| -> bool {
                match self.memory_topology {
                    MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
                        let total_used = inner.ram_allocated + inner.vram_allocated;
                        let total_needed = needed_ram + needed_vram + needed_cache;
                        total_used.saturating_sub(freed_ram + freed_vram) + total_needed
                            > ram_budget
                    }
                    MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
                        (inner.ram_allocated.saturating_sub(freed_ram) + needed_ram > ram_budget)
                            || (inner.vram_allocated.saturating_sub(freed_vram) + needed_vram
                                > vram_budget)
                    }
                }
            };

        for candidate in &candidates {
            if !need_more(inner, freed_ram, freed_vram) {
                break;
            }

            let (ref id, priority, _, ram, vram, is_cache) = *candidate;

            if is_cache {
                // id is a cache handle_id string
                if let Ok(handle_id) = id.parse::<u64>() {
                    if let Some(cache) = inner.caches.remove(&handle_id) {
                        inner.ram_allocated = inner.ram_allocated.saturating_sub(cache.bytes);
                        freed_ram += cache.bytes;
                        tracing::info!(
                            "Evicted cache handle {} ({} bytes)",
                            handle_id,
                            cache.bytes
                        );
                    }
                }
            } else if let Some(record) = inner.residencies.remove(id) {
                tracing::info!(
                    "Evicting model '{}' (priority={:?}, ram={}, vram={})",
                    id,
                    priority,
                    ram,
                    vram
                );
                if let Err(e) = (record.offload_cb)() {
                    tracing::error!("Failed to evict model '{}': {}", id, e);
                }
                inner.ram_allocated = inner.ram_allocated.saturating_sub(record.ram_bytes);
                inner.vram_allocated = inner.vram_allocated.saturating_sub(record.vram_bytes);
                freed_ram += record.ram_bytes;
                freed_vram += record.vram_bytes;
                inner.leases.remove(&record.lease_id);
                evicted.push(id.clone());
            }
        }

        Ok(evicted)
    }

    // ====================================================================
    // Backward-compatible API
    // ====================================================================

    /// Legacy API: request to load a model with simple size-based budget.
    /// In UMA mode, bypasses budget checks (backward-compatible behavior).
    pub fn request_load(
        &self,
        id: &str,
        size_bytes: usize,
        is_uma: bool,
        offload_cb: OffloadCallback,
    ) -> Result<(), String> {
        // UMA backward-compat: skip budget, just track residency
        if is_uma {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.residencies.contains_key(id) {
                return Ok(());
            }
            let lease_id = self.next_lease_id.fetch_add(1, Ordering::SeqCst);
            let now = Instant::now();
            let record = ModelResidencyRecord {
                model_id: id.to_string(),
                lease_id,
                ram_bytes: size_bytes,
                vram_bytes: 0,
                cache_bytes: 0,
                priority: ResourcePriority::Normal,
                strategy: ResidencyStrategy::Mmap,
                backend_name: "uma".to_string(),
                offload_cb,
                last_accessed: now,
            };
            inner.residencies.insert(id.to_string(), record);
            inner.leases.insert(
                lease_id,
                LeaseRecord {
                    model_id: id.to_string(),
                    backend_name: "uma".to_string(),
                    granted_ram: size_bytes,
                    granted_vram: 0,
                    created_at: now,
                },
            );
            return Ok(());
        }

        let ticket = ResourceTicket {
            model_id: id.to_string(),
            ram_bytes: 0,
            vram_bytes: size_bytes,
            cache_bytes: 0,
            priority: ResourcePriority::Normal,
            strategy: ResidencyStrategy::OnDemand,
            preferred_backend: None,
            fallback_backends: vec![],
        };
        match self.reserve(ticket, offload_cb) {
            Ok(_) => Ok(()),
            Err(ResourceError::AlreadyLoaded { .. }) => Ok(()), // backward compat
            Err(e) => Err(e.to_string()),
        }
    }

    /// Legacy API: record model unload.
    pub fn record_unload(&self, id: &str) {
        self.release_by_model(id);
    }
}

/// Global singleton coordinator (backward-compatible name).
pub fn global_vram_coordinator() -> &'static ResourceCoordinator {
    global_resource_coordinator()
}

/// Global singleton coordinator (recommended name).
pub fn global_resource_coordinator() -> &'static ResourceCoordinator {
    GLOBAL_COORDINATOR.get_or_init(|| {
        let budget = std::env::var("BLOOM_MEMORY_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5 * GIB as usize); // 5GB default
        ResourceCoordinator::new(budget, budget, MemoryTopology::Unified)
    })
}

/// Initialize the global coordinator once, typically during server startup.
#[allow(clippy::result_large_err)]
pub fn init_global_resource_coordinator(
    coordinator: ResourceCoordinator,
) -> Result<(), ResourceCoordinator> {
    GLOBAL_COORDINATOR.set(coordinator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn make_ticket(
        id: &str,
        ram: usize,
        vram: usize,
        priority: ResourcePriority,
    ) -> ResourceTicket {
        ResourceTicket {
            model_id: id.to_string(),
            ram_bytes: ram,
            vram_bytes: vram,
            cache_bytes: 0,
            priority,
            strategy: ResidencyStrategy::OnDemand,
            preferred_backend: Some("cpu".to_string()),
            fallback_backends: vec![],
        }
    }

    fn noop_cb() -> OffloadCallback {
        Arc::new(|| Ok(()))
    }

    #[test]
    fn test_coordinator_basic_reserve_release() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);
        let ticket = make_ticket("m1", 400, 0, ResourcePriority::Normal);
        let lease = coord.reserve(ticket, noop_cb()).unwrap();
        assert_eq!(lease.lease_id, 1);
        assert_eq!(lease.granted_ram, 400);

        let snap = coord.snapshot();
        assert_eq!(snap.ram_allocated, 400);
        assert_eq!(snap.model_count, 1);

        coord.release(lease.lease_id);
        let snap2 = coord.snapshot();
        assert_eq!(snap2.ram_allocated, 0);
        assert_eq!(snap2.model_count, 0);
    }

    #[test]
    fn test_coordinator_eviction_by_priority() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);

        // Load a Low priority model (600 bytes)
        let t1 = make_ticket("low-model", 600, 0, ResourcePriority::Low);
        coord.reserve(t1, noop_cb()).unwrap();

        // Load a Normal priority model (300 bytes)
        let t2 = make_ticket("normal-model", 300, 0, ResourcePriority::Normal);
        coord.reserve(t2, noop_cb()).unwrap();

        // Now try to load another model (200 bytes) - should evict low-model first
        let t3 = make_ticket("new-model", 200, 0, ResourcePriority::Normal);
        let lease = coord.reserve(t3, noop_cb()).unwrap();
        assert!(lease.evicted_models.contains(&"low-model".to_string()));

        let snap = coord.snapshot();
        // normal-model (300) + new-model (200) = 500
        assert_eq!(snap.ram_allocated, 500);
    }

    #[test]
    fn test_coordinator_critical_not_evicted() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);

        // Load a Critical model (800 bytes)
        let t1 = make_ticket("critical", 800, 0, ResourcePriority::Critical);
        coord.reserve(t1, noop_cb()).unwrap();

        // Try to load another (300 bytes) - should fail, Critical cannot be evicted
        let t2 = make_ticket("other", 300, 0, ResourcePriority::Normal);
        let result = coord.reserve(t2, noop_cb());
        assert!(result.is_err());
        match result.unwrap_err() {
            ResourceError::BudgetExceeded { .. } => {}
            e => panic!("Expected BudgetExceeded, got {:?}", e),
        }
    }

    #[test]
    fn test_coordinator_uma_unified_budget() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Unified);

        // UMA: ram + vram share same pool
        let t1 = make_ticket("m1", 500, 300, ResourcePriority::Normal);
        coord.reserve(t1, noop_cb()).unwrap();

        let snap = coord.snapshot();
        assert_eq!(snap.ram_allocated, 500);
        assert_eq!(snap.vram_allocated, 300);

        // Total used = 800, budget = 1000, try 300 more -> eviction succeeds (m1 evicted)
        let t2 = make_ticket("m2", 300, 0, ResourcePriority::Normal);
        let lease = coord.reserve(t2, noop_cb()).unwrap();
        assert!(lease.evicted_models.contains(&"m1".to_string()));

        let snap2 = coord.snapshot();
        assert_eq!(snap2.ram_allocated, 300);
        assert_eq!(snap2.vram_allocated, 0);

        // Now try loading 1200 bytes -> should fail even after evicting m2 (300 freed, 0 + 1200 = 1200 > 1000)
        let t3 = make_ticket("m3", 1200, 0, ResourcePriority::Normal);
        let result = coord.reserve(t3, noop_cb());
        assert!(result.is_err());
    }

    #[test]
    fn test_coordinator_uma_no_vram_only_limit() {
        // UMA device: model requests only vram, should not be rejected by "no vram budget"
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Unified);

        let t = ResourceTicket {
            model_id: "gpu-only".to_string(),
            ram_bytes: 0,
            vram_bytes: 800,
            cache_bytes: 0,
            priority: ResourcePriority::Normal,
            strategy: ResidencyStrategy::Mmap,
            preferred_backend: Some("metal".to_string()),
            fallback_backends: vec![],
        };
        let lease = coord.reserve(t, noop_cb()).unwrap();
        assert_eq!(lease.granted_vram, 800);
    }

    #[test]
    fn test_coordinator_discrete_separate_budget() {
        let coord = ResourceCoordinator::new(2000, 1000, MemoryTopology::Discrete);

        // RAM 1500, VRAM 0 -> OK (within RAM budget)
        let t1 = make_ticket("m1", 1500, 0, ResourcePriority::Normal);
        coord.reserve(t1, noop_cb()).unwrap();

        // RAM 0, VRAM 800 -> OK (within VRAM budget)
        let t2 = make_ticket("m2", 0, 800, ResourcePriority::Normal);
        coord.reserve(t2, noop_cb()).unwrap();

        // RAM 600, VRAM 0 -> eviction: m1 is Normal priority and can be evicted
        // After evicting m1 (1500 ram freed), m3 (600 ram) fits
        let t3 = make_ticket("m3", 600, 0, ResourcePriority::Normal);
        let lease = coord.reserve(t3, noop_cb()).unwrap();
        assert!(lease.evicted_models.contains(&"m1".to_string()));

        let snap = coord.snapshot();
        // m2 (0 ram, 800 vram) + m3 (600 ram, 0 vram)
        assert_eq!(snap.ram_allocated, 600);
        assert_eq!(snap.vram_allocated, 800);

        // RAM 1500, VRAM 0 -> should fail: 600 + 1500 = 2100 > 2000, only m3 can be evicted (600 freed)
        // After evicting m3: 0 + 1500 = 1500 <= 2000 -> fits!
        let t4 = make_ticket("m4", 1500, 0, ResourcePriority::Normal);
        let lease2 = coord.reserve(t4, noop_cb()).unwrap();
        assert!(lease2.evicted_models.contains(&"m3".to_string()));

        // Now try to load RAM 600 -> should fail (1500 + 600 > 2000, nothing evictable except m4)
        let t5 = make_ticket("m5", 600, 0, ResourcePriority::Normal);
        // m4 has Normal priority and can be evicted -> 0 + 600 = 600 <= 2000 -> fits
        let lease3 = coord.reserve(t5, noop_cb()).unwrap();
        assert!(lease3.evicted_models.contains(&"m4".to_string()));
    }

    #[test]
    fn test_coordinator_already_loaded() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);

        let t1 = make_ticket("m1", 100, 0, ResourcePriority::Normal);
        coord.reserve(t1, noop_cb()).unwrap();

        let t2 = make_ticket("m1", 100, 0, ResourcePriority::Normal);
        let result = coord.reserve(t2, noop_cb());
        assert!(matches!(result, Err(ResourceError::AlreadyLoaded { .. })));
    }

    #[test]
    fn test_coordinator_release_by_model() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);

        let t = make_ticket("m1", 400, 0, ResourcePriority::Normal);
        coord.reserve(t, noop_cb()).unwrap();
        assert_eq!(coord.snapshot().model_count, 1);

        coord.release_by_model("m1");
        assert_eq!(coord.snapshot().model_count, 0);
        assert_eq!(coord.snapshot().ram_allocated, 0);
    }

    #[test]
    fn test_coordinator_cache_management() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);

        let handle = CacheHandle {
            handle_id: 1,
            model_id: "m1".to_string(),
            cache_kind: CacheKind::KvCache,
            bytes: 200,
            priority: ResourcePriority::Normal,
        };
        coord.register_cache(handle).unwrap();
        assert_eq!(coord.snapshot().cache_count, 1);
        assert_eq!(coord.snapshot().ram_allocated, 200);

        coord.release_cache(1);
        assert_eq!(coord.snapshot().cache_count, 0);
        assert_eq!(coord.snapshot().ram_allocated, 0);
    }

    #[test]
    fn test_coordinator_lease_id_for_model() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);
        assert!(coord.lease_id_for_model("m1").is_none());

        let t = make_ticket("m1", 100, 0, ResourcePriority::Normal);
        let lease = coord.reserve(t, noop_cb()).unwrap();
        assert_eq!(coord.lease_id_for_model("m1"), Some(lease.lease_id));
    }

    #[test]
    fn test_coordinator_model_snapshots() {
        let coord = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);
        let t = make_ticket("snap-model", 100, 200, ResourcePriority::High);
        let lease = coord.reserve(t, noop_cb()).unwrap();

        let snapshots = coord.model_snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].model_id, "snap-model");
        assert_eq!(snapshots[0].lease_id, lease.lease_id);
        assert_eq!(snapshots[0].ram_bytes, 100);
        assert_eq!(snapshots[0].vram_bytes, 200);
        assert_eq!(snapshots[0].priority, ResourcePriority::High);
    }

    // ====================================================================
    // Backward-compatible API tests (originally from vram.rs)
    // ====================================================================

    #[test]
    fn test_vram_coordinator_basic_load_unload() {
        let coordinator = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);
        let evict_counter = Arc::new(AtomicUsize::new(0));

        let cb_counter = Arc::clone(&evict_counter);
        let cb = Arc::new(move || {
            cb_counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        assert!(coordinator
            .request_load("model_1", 400, false, cb.clone())
            .is_ok());
        assert_eq!(coordinator.snapshot().vram_allocated, 400);
        assert_eq!(evict_counter.load(Ordering::SeqCst), 0);

        assert!(coordinator
            .request_load("model_2", 300, false, cb.clone())
            .is_ok());
        assert_eq!(coordinator.snapshot().vram_allocated, 700);

        assert!(coordinator
            .request_load("model_1", 400, false, cb.clone())
            .is_ok());
        assert_eq!(coordinator.snapshot().vram_allocated, 700);

        coordinator.record_unload("model_2");
        assert_eq!(coordinator.snapshot().vram_allocated, 400);
    }

    #[test]
    fn test_vram_coordinator_eviction() {
        let coordinator = ResourceCoordinator::new(1000, 1000, MemoryTopology::Discrete);

        let evict_1 = Arc::new(AtomicUsize::new(0));
        let evict_2 = Arc::new(AtomicUsize::new(0));

        let c1 = Arc::clone(&evict_1);
        let cb1 = Arc::new(move || {
            c1.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        let c2 = Arc::clone(&evict_2);
        let cb2 = Arc::new(move || {
            c2.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        assert!(coordinator.request_load("m1", 600, false, cb1).is_ok());
        assert_eq!(coordinator.snapshot().vram_allocated, 600);

        assert!(coordinator.request_load("m2", 500, false, cb2).is_ok());

        assert_eq!(evict_1.load(Ordering::SeqCst), 1);
        assert_eq!(coordinator.snapshot().vram_allocated, 500);
    }

    #[test]
    fn test_vram_coordinator_uma_bypass() {
        let coordinator = ResourceCoordinator::new(1000, 1000, MemoryTopology::Unified);
        let evict_counter = Arc::new(AtomicUsize::new(0));
        let cb_counter = Arc::clone(&evict_counter);
        let cb = Arc::new(move || {
            cb_counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        // Under UMA mode, memory budget checks and eviction should be bypassed completely.
        assert!(coordinator.request_load("m1", 1200, true, cb).is_ok());
        // UMA bypass does not update standard ram/vram allocated counters
        assert_eq!(coordinator.snapshot().ram_allocated, 0);
        assert_eq!(coordinator.snapshot().vram_allocated, 0);
        assert_eq!(evict_counter.load(Ordering::SeqCst), 0);
        // But the model is still tracked in residencies
        assert_eq!(coordinator.snapshot().model_count, 1);
    }
}
