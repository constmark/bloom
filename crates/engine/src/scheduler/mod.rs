// Scheduler queues expose callback and completion-channel ownership in their types.
#![allow(clippy::type_complexity)]

use anyhow::Result;
use bloomai_core::{
    BloomError, CacheHandle, DeviceCapability, GenerationParams, PowerState, ResourcePriority,
    ResourceTicket, ThermalState,
    token_scheduling::{
        TokenSchedulingConfig as CoreTokenSchedulingConfig,
        chunked_prefill::ChunkedPrefillQueue,
        preemption::{PreemptibleRequest, PreemptionManager},
        priority_eviction::{AdmissionResult, KvEvictionManager, KvSessionInfo},
        rate_limiter::{RateLimitDecision, TokenBucketRateLimiter},
    },
};
use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering as AtomicOrdering},
};
use std::time::Instant;

use crate::io::DataBlock;

pub mod kv_hook;
pub mod paged_cache;

/// Return whether `requested` tokens fit without overflowing the counter.
///
/// Keeping this check subtraction-based makes oversized public configuration
/// and accounting state fail closed in both debug and release builds.
pub(crate) fn token_budget_allows(used: usize, requested: usize, limit: usize) -> bool {
    used <= limit && requested <= limit - used
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenSchedulingConfig {
    pub max_prefill_tokens_per_step: usize,
    pub max_decode_tokens_per_step: usize,
    pub max_total_tokens_per_step: usize,
}

impl Default for TokenSchedulingConfig {
    fn default() -> Self {
        Self {
            max_prefill_tokens_per_step: 4096,
            max_decode_tokens_per_step: 4096,
            max_total_tokens_per_step: 4096,
        }
    }
}

#[derive(Default)]
pub struct TokenAdmission {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
}

impl TokenAdmission {
    pub fn try_reserve(
        &mut self,
        config: &TokenSchedulingConfig,
        phase: TokenPhase,
        tokens: usize,
    ) -> bool {
        let Some(total_used) = self.prefill_tokens.checked_add(self.decode_tokens) else {
            return false;
        };
        if !token_budget_allows(total_used, tokens, config.max_total_tokens_per_step) {
            return false;
        }
        match phase {
            TokenPhase::Prefill => {
                if !token_budget_allows(
                    self.prefill_tokens,
                    tokens,
                    config.max_prefill_tokens_per_step,
                ) {
                    return false;
                }
                self.prefill_tokens += tokens;
                true
            }
            TokenPhase::Decode => {
                if !token_budget_allows(
                    self.decode_tokens,
                    tokens,
                    config.max_decode_tokens_per_step,
                ) {
                    return false;
                }
                self.decode_tokens += tokens;
                true
            }
        }
    }
}

/// Unique identifier for a request
pub type RequestId = String;

const EXECUTION_FAILED_CLIENT_MESSAGE: &str = "inference execution failed";

/// Opaque ownership retained while a request is executing.
///
/// The scheduler never inspects this value. Callers can use it to keep a
/// runtime, model lease, or other request-scoped resource alive through the
/// executor's synchronous model call without requiring that resource to
/// implement `Debug` or serialization traits.
pub type ExecutionGuard = Arc<dyn Any + Send + Sync>;

/// State of a request in the lifecycle
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    Pending,
    Prefill,
    Decoding { current_step: usize },
    Finished,
}

/// Represents a single request in the scheduler
pub struct Request {
    pub id: RequestId,
    pub model_id: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub params: GenerationParams,
    pub state: RequestState,
    pub priority: u32,
    pub kv_handle: Option<usize>, // Index into KV cache pool
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub preemption_count: usize,
    pub decode_started_at: Option<Instant>,
    pub last_scheduled_at: Option<Instant>,
    pub multimodal_hash: Option<String>,
}

/// Scheduler-owned request metadata that is intentionally kept out of the
/// public [`Request`] struct-literal API.
struct ScheduledRequest {
    request: Request,
    execution_guard: Option<ExecutionGuard>,
}

impl Deref for ScheduledRequest {
    type Target = Request;

    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

impl DerefMut for ScheduledRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.request
    }
}

#[derive(Debug, Clone)]
pub struct KvCacheAllocation {
    pub handle: usize,
    pub matched_tokens: usize,
    pub allocated_blocks: Vec<usize>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct KvCacheMetrics {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub active_blocks: usize,
    pub cached_blocks: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub reuses: usize,
}

/// Represents a physical KV cache pool on a device
pub trait KvCachePool: Send + Sync {
    fn allocate(&self, num_tokens: usize) -> Result<usize>;
    fn free(&self, handle: usize);

    /// Allocate blocks for a sequence with optional prefix matching/reuse
    fn allocate_paged(
        &self,
        request_id: &str,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        multimodal_hash: Option<&str>,
    ) -> Result<KvCacheAllocation>;

    /// Mark a sequence as inactive/finished, allowing its blocks to be reused or evicted
    fn free_paged(&self, request_id: &str);

    /// Get current cache metrics (hits, misses, evictions, active blocks, etc.)
    fn get_metrics(&self) -> KvCacheMetrics;

    /// Update metadata for a sequence (used for priority eviction score computation)
    fn update_request_metadata(
        &self,
        _request_id: &str,
        _priority: u32,
        _generated_tokens: usize,
        _estimated_token_value: f64,
    ) {
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixCacheKey {
    pub tokens: Vec<u32>,
    pub multimodal_hash: Option<String>,
}

struct ActiveRequestRecord {
    blocks: Vec<usize>,
    prefixes: Vec<PrefixCacheKey>,
    last_accessed: Instant,
    active: bool,
    created_at: Instant,
    priority: u32,
    generated_tokens: usize,
    estimated_token_value: f64,
}

pub(crate) struct PoolState {
    free_blocks: VecDeque<usize>,
    block_table: HashMap<PrefixCacheKey, usize>, // maps prefix key to block ID
    block_ref_counts: HashMap<usize, usize>,     // tracks reference count for each block
    active_requests: HashMap<String, ActiveRequestRecord>,
    lru_list: VecDeque<String>, // request IDs of inactive requests
    next_handle: usize,
    handle_to_request_id: HashMap<usize, String>,
    pub(crate) metrics: KvCacheMetrics,
}

impl PoolState {
    fn update_block_counts(&mut self, total_blocks: usize) {
        let mut active_set = std::collections::HashSet::new();
        for record in self.active_requests.values() {
            if record.active {
                for &b in &record.blocks {
                    active_set.insert(b);
                }
            }
        }
        self.metrics.free_blocks = self.free_blocks.len();
        self.metrics.active_blocks = active_set.len();
        let allocated_count = total_blocks - self.free_blocks.len();
        self.metrics.cached_blocks = allocated_count.saturating_sub(active_set.len());
    }
}

pub struct BloomKvCachePool {
    block_size: usize,
    total_blocks: usize,
    legacy_request_sequence: AtomicU64,
    pub(crate) state: Mutex<PoolState>,
    on_free: Mutex<Option<Box<dyn Fn(usize) + Send + Sync + 'static>>>,
    on_evict: Mutex<Option<Box<dyn Fn(&[usize]) + Send + Sync + 'static>>>,
}

impl BloomKvCachePool {
    pub fn new(block_size: usize, total_blocks: usize) -> Self {
        let mut free_blocks = VecDeque::new();
        for i in 0..total_blocks {
            free_blocks.push_back(i);
        }
        Self {
            block_size,
            total_blocks,
            legacy_request_sequence: AtomicU64::new(1),
            state: Mutex::new(PoolState {
                free_blocks,
                block_table: HashMap::new(),
                block_ref_counts: HashMap::new(),
                active_requests: HashMap::new(),
                lru_list: VecDeque::new(),
                next_handle: 1,
                handle_to_request_id: HashMap::new(),
                metrics: KvCacheMetrics {
                    total_blocks,
                    free_blocks: total_blocks,
                    ..Default::default()
                },
            }),
            on_free: Mutex::new(None),
            on_evict: Mutex::new(None),
        }
    }

    pub fn set_on_free<F>(&self, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        *self.on_free.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(f));
    }

    pub fn set_on_evict<F>(&self, f: F)
    where
        F: Fn(&[usize]) + Send + Sync + 'static,
    {
        *self.on_evict.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(f));
    }

    /// Resolve the block id that owns `token_pos` for the request identified by
    /// `handle`. Returns `None` when the handle is unknown or the position is
    /// outside the allocated range. Used by the batch executor's KV bridge to
    /// route extracted model KV into the correct paged-cache block.
    pub fn block_for_handle(&self, handle: usize, token_pos: usize) -> Option<usize> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let request_id = state.handle_to_request_id.get(&handle)?;
        let record = state.active_requests.get(request_id)?;
        let block_idx = token_pos / self.block_size;
        record.blocks.get(block_idx).copied()
    }

    /// Block size (tokens per block) configured for this pool.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    fn notify_evicted(&self, blocks: &[usize]) {
        if blocks.is_empty() {
            return;
        }
        if let Some(ref f) = *self.on_evict.lock().unwrap_or_else(|e| e.into_inner()) {
            f(blocks);
        }
    }

    /// Evict inactive cached requests until at least `target_free_blocks` are
    /// available. Active streaming/decoding requests are never compacted here.
    pub fn compact_inactive(&self, target_free_blocks: usize) -> usize {
        let mut evicted_blocks = Vec::new();
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.free_blocks.len() < target_free_blocks {
                let Some(evict_id) = state.lru_list.pop_front() else {
                    break;
                };
                let Some(record) = state.active_requests.remove(&evict_id) else {
                    continue;
                };
                for (block_id, prefix) in record.blocks.iter().zip(record.prefixes.iter()) {
                    let count = state.block_ref_counts.entry(*block_id).or_insert(1);
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        if !prefix.tokens.is_empty() {
                            state.block_table.remove(prefix);
                        }
                        state.free_blocks.push_back(*block_id);
                        evicted_blocks.push(*block_id);
                        state.block_ref_counts.remove(block_id);
                    }
                }
                state.metrics.evictions += 1;
            }
            state.update_block_counts(self.total_blocks);
        }
        self.notify_evicted(&evicted_blocks);
        evicted_blocks.len()
    }

    /// Prune low attention/inactive blocks within a specific active request to free up memory.
    /// This performs dynamic attention sensitivity analysis on the request's blocks.
    pub fn prune_low_attention_blocks(
        &self,
        handle: usize,
        attention_weights: &[f32],
        target_free: usize,
    ) -> usize {
        let mut evicted_blocks = Vec::new();
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let Some(request_id) = state.handle_to_request_id.get(&handle).cloned() else {
                return 0;
            };

            // 1. Immutable/non-overlapping inspection of the target record to copy its blocks and prefixes
            let (record_blocks, record_prefixes) = {
                let Some(record) = state.active_requests.get(&request_id) else {
                    return 0;
                };
                if record.blocks.len() <= 4 {
                    return 0;
                }
                (record.blocks.clone(), record.prefixes.clone())
            };

            // Map each block index in `record_blocks` to its attention weight.
            let mut block_weights: Vec<(usize, usize, f32)> = record_blocks
                .iter()
                .enumerate()
                .map(|(idx, &block_id)| {
                    let weight = attention_weights
                        .get(idx)
                        .copied()
                        .unwrap_or(0.1f32 + (idx as f32 * 0.01f32));
                    (idx, block_id, weight)
                })
                .collect();

            // Sort by weight ascending (lowest weight first)
            block_weights
                .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            let mut pruned_indices = std::collections::HashSet::new();
            for &(idx, block_id, _) in &block_weights {
                if state.free_blocks.len() >= target_free
                    || record_blocks.len() - pruned_indices.len() <= 4
                {
                    break;
                }

                // Verify block is not shared or referenced elsewhere before evicting
                let count = state.block_ref_counts.entry(block_id).or_insert(1);
                *count = count.saturating_sub(1);
                if *count == 0 {
                    if let Some(prefix) = record_prefixes.get(idx)
                        && !prefix.tokens.is_empty()
                    {
                        state.block_table.remove(prefix);
                    }
                    state.free_blocks.push_back(block_id);
                    evicted_blocks.push(block_id);
                    state.block_ref_counts.remove(&block_id);
                }
                pruned_indices.insert(idx);
            }

            // 2. Mutably borrow request record to update its blocks/prefixes
            if let Some(record) = state.active_requests.get_mut(&request_id) {
                let mut new_blocks = Vec::new();
                let mut new_prefixes = Vec::new();
                for idx in 0..record.blocks.len() {
                    if !pruned_indices.contains(&idx) {
                        new_blocks.push(record.blocks[idx]);
                        new_prefixes.push(record.prefixes[idx].clone());
                    }
                }
                record.blocks = new_blocks;
                record.prefixes = new_prefixes;
            }
            state.update_block_counts(self.total_blocks);
        }

        self.notify_evicted(&evicted_blocks);
        evicted_blocks.len()
    }

    /// Compact duplicate/redundant prefix segments across all requests and
    /// defragment the block layout to optimize contiguous memory access.
    pub fn defragment_and_compact_prefixes(&self) -> usize {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.defragment_and_compact_prefixes_impl(&mut state)
    }

    fn defragment_and_compact_prefixes_impl(&self, state: &mut PoolState) -> usize {
        let mut merged_count = 0;

        // 1. Cross-segment Prefix Compaction (Deduplication)
        let mut prefix_to_block = std::collections::HashMap::new();
        let mut updates = Vec::new();

        for (request_id, record) in &state.active_requests {
            for (idx, prefix) in record.prefixes.iter().enumerate() {
                if prefix.tokens.is_empty() {
                    continue;
                }
                if let Some(&canonical_block_id) = prefix_to_block.get(&prefix.tokens) {
                    let current_block_id = record.blocks[idx];
                    if current_block_id != canonical_block_id {
                        updates.push((
                            request_id.clone(),
                            idx,
                            canonical_block_id,
                            current_block_id,
                        ));
                    }
                } else {
                    prefix_to_block.insert(prefix.tokens.clone(), record.blocks[idx]);
                }
            }
        }

        // Apply updates
        for (request_id, idx, canonical_block_id, current_block_id) in updates {
            // First perform the ref counts updates on state
            *state
                .block_ref_counts
                .entry(canonical_block_id)
                .or_insert(1) += 1;

            let count = state.block_ref_counts.entry(current_block_id).or_insert(1);
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.free_blocks.push_back(current_block_id);
                state.block_ref_counts.remove(&current_block_id);
            }

            // Then mutably borrow record in a separate step to update block mapping
            if let Some(record) = state.active_requests.get_mut(&request_id) {
                record.blocks[idx] = canonical_block_id;
                merged_count += 1;
            }
        }

        // 2. In-place defragmentation (GC)
        let mut sorted_free: Vec<usize> = state.free_blocks.iter().copied().collect();
        sorted_free.sort_unstable();
        state.free_blocks = sorted_free.into_iter().collect();
        state.update_block_counts(self.total_blocks);

        merged_count
    }
}

impl KvCachePool for BloomKvCachePool {
    fn allocate(&self, num_tokens: usize) -> Result<usize> {
        let sequence = self
            .legacy_request_sequence
            .fetch_update(
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
                |current| current.checked_add(1),
            )
            .map_err(|_| {
                BloomError::SchedulingFailed("legacy KV request sequence exhausted".into())
            })?;
        let request_id = format!("legacy-request-{sequence}");
        let alloc = self.allocate_paged(&request_id, &[], num_tokens, None)?;
        Ok(alloc.handle)
    }

    fn free(&self, handle: usize) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(request_id) = state.handle_to_request_id.remove(&handle)
                && let Some(record) = state.active_requests.get_mut(&request_id)
            {
                record.active = false;
                record.last_accessed = Instant::now();
                if let Some(pos) = state.lru_list.iter().position(|r| r == &request_id) {
                    state.lru_list.remove(pos);
                }
                state.lru_list.push_back(request_id.clone());
            }
            state.update_block_counts(self.total_blocks);
        }

        // Call hook
        if let Some(ref f) = *self.on_free.lock().unwrap_or_else(|e| e.into_inner()) {
            f(handle);
        }
    }

    fn allocate_paged(
        &self,
        request_id: &str,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        multimodal_hash: Option<&str>,
    ) -> Result<KvCacheAllocation> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut evicted_blocks = Vec::new();

        if state.active_requests.contains_key(request_id) {
            let handle = state
                .handle_to_request_id
                .iter()
                .find(|(_, rid)| *rid == request_id)
                .map(|(&h, _)| h)
                .ok_or_else(|| {
                    BloomError::SchedulingFailed(format!(
                        "active KV cache handle disappeared for request {request_id}"
                    ))
                })?;
            if let Some(pos) = state.lru_list.iter().position(|r| r == request_id) {
                state.lru_list.remove(pos);
            }
            let allocated_blocks = {
                let Some(record) = state.active_requests.get_mut(request_id) else {
                    return Err(BloomError::SchedulingFailed(format!(
                        "active KV cache record disappeared for request {request_id}"
                    ))
                    .into());
                };
                record.active = true;
                record.last_accessed = Instant::now();
                record.blocks.clone()
            };

            return Ok(KvCacheAllocation {
                handle,
                matched_tokens: 0,
                allocated_blocks,
            });
        }

        let total_tokens = prompt_tokens.len() + max_new_tokens;
        let needed_blocks = total_tokens.div_ceil(self.block_size);
        let num_prompt_blocks = prompt_tokens.len() / self.block_size;
        let mut reused_blocks = Vec::new();
        let mut reused_prefixes = Vec::new();
        for i in 0..num_prompt_blocks {
            let prefix_len = (i + 1) * self.block_size;
            let prefix = &prompt_tokens[0..prefix_len];
            let key = PrefixCacheKey {
                tokens: prefix.to_vec(),
                multimodal_hash: multimodal_hash.map(String::from),
            };
            if let Some(&block_id) = state.block_table.get(&key) {
                reused_blocks.push(block_id);
                reused_prefixes.push(key);
            } else {
                break;
            }
        }

        let matched_tokens = reused_blocks.len() * self.block_size;
        // Pin prefix hits before reserving additional capacity. Otherwise an
        // inactive request that owns the matched prefix could be selected as
        // the LRU victim below, freeing that same block and allowing it to be
        // reserved a second time for a different sequence position.
        for &block_id in &reused_blocks {
            let count = state.block_ref_counts.entry(block_id).or_insert(0);
            *count += 1;
        }
        let blocks_to_allocate = needed_blocks - reused_blocks.len();
        let mut newly_allocated = Vec::new();
        let mut allocation_error = None;

        for _ in 0..blocks_to_allocate {
            while state.free_blocks.is_empty() {
                if state.lru_list.is_empty() {
                    // Automatically trigger prefix cache compaction and defragmentation to reclaim duplicate space
                    let merged = self.defragment_and_compact_prefixes_impl(&mut state);
                    if merged > 0 && !state.free_blocks.is_empty() {
                        continue;
                    }
                    allocation_error = Some(BloomError::SchedulingFailed(
                        "KV Cache Pool is full and no inactive sequences can be evicted".into(),
                    ));
                    break;
                }
                let victim_idx = {
                    let mut best_idx = 0;
                    let mut lowest_score = f64::MAX;
                    for (idx, evict_id) in state.lru_list.iter().enumerate() {
                        if let Some(record) = state.active_requests.get(evict_id) {
                            let age = record.created_at.elapsed().as_secs_f64().max(0.001);
                            let value = record.estimated_token_value;
                            let cost = (record.blocks.len() * 16) as f64;
                            let score = age * value * cost;
                            if score < lowest_score {
                                lowest_score = score;
                                best_idx = idx;
                            }
                        }
                    }
                    best_idx
                };
                let Some(evict_id) = state.lru_list.remove(victim_idx) else {
                    allocation_error = Some(BloomError::SchedulingFailed(
                        "KV cache eviction candidate disappeared".into(),
                    ));
                    break;
                };
                if let Some(record) = state.active_requests.remove(&evict_id) {
                    state
                        .handle_to_request_id
                        .retain(|_, request_id| request_id != &evict_id);
                    for (block_id, prefix) in record.blocks.iter().zip(record.prefixes.iter()) {
                        let count = state.block_ref_counts.entry(*block_id).or_insert(1);
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            if !prefix.tokens.is_empty() {
                                state.block_table.remove(prefix);
                            }
                            state.free_blocks.push_back(*block_id);
                            evicted_blocks.push(*block_id);
                            state.block_ref_counts.remove(block_id);
                        }
                    }
                    state.metrics.evictions += 1;
                }
            }

            if allocation_error.is_some() {
                break;
            }
            let Some(block_id) = state.free_blocks.pop_front() else {
                allocation_error = Some(BloomError::SchedulingFailed(
                    "KV cache free list became empty during allocation".into(),
                ));
                break;
            };
            newly_allocated.push(block_id);
        }

        if let Some(error) = allocation_error {
            // Only reservation of these blocks belongs to the failed request.
            // Evictions/compaction performed to obtain capacity remain valid
            // external side effects and are reported after releasing the lock.
            for block_id in newly_allocated.into_iter().rev() {
                state.free_blocks.push_front(block_id);
            }
            for (block_id, prefix) in reused_blocks.iter().zip(reused_prefixes.iter()) {
                let release_block = state
                    .block_ref_counts
                    .get_mut(block_id)
                    .is_some_and(|count| {
                        *count = count.saturating_sub(1);
                        *count == 0
                    });
                if release_block {
                    state.block_ref_counts.remove(block_id);
                    if state
                        .block_table
                        .get(prefix)
                        .is_some_and(|mapped| mapped == block_id)
                    {
                        state.block_table.remove(prefix);
                    }
                    state.free_blocks.push_back(*block_id);
                    evicted_blocks.push(*block_id);
                }
            }
            state.update_block_counts(self.total_blocks);
            drop(state);
            self.notify_evicted(&evicted_blocks);
            return Err(error.into());
        }

        if !reused_blocks.is_empty() {
            state.metrics.hits += 1;
            state.metrics.reuses += reused_blocks.len();
        } else {
            state.metrics.misses += 1;
        }

        // Capacity is fully reserved. Commit refcounts and prefix mappings in
        // one infallible pass so no failed request can leak partial metadata.
        let mut new_prefixes = Vec::new();
        let mut next_prompt_block_idx = reused_blocks.len();
        for &block_id in &newly_allocated {
            state.block_ref_counts.insert(block_id, 1);

            let key = if next_prompt_block_idx < num_prompt_blocks {
                let prefix_len = (next_prompt_block_idx + 1) * self.block_size;
                let prefix = prompt_tokens[0..prefix_len].to_vec();
                let key = PrefixCacheKey {
                    tokens: prefix,
                    multimodal_hash: multimodal_hash.map(String::from),
                };
                state.block_table.insert(key.clone(), block_id);
                next_prompt_block_idx += 1;
                key
            } else {
                PrefixCacheKey {
                    tokens: Vec::new(),
                    multimodal_hash: None,
                }
            };
            new_prefixes.push(key);
        }

        // The reservation pin above becomes this request's committed prefix
        // reference on success.
        let mut final_blocks = reused_blocks;
        final_blocks.extend(newly_allocated);

        let mut final_prefixes = reused_prefixes;
        final_prefixes.extend(new_prefixes);

        let handle = state.next_handle;
        state.next_handle += 1;
        state
            .handle_to_request_id
            .insert(handle, request_id.to_string());

        state.active_requests.insert(
            request_id.to_string(),
            ActiveRequestRecord {
                blocks: final_blocks.clone(),
                prefixes: final_prefixes,
                last_accessed: Instant::now(),
                active: true,
                created_at: Instant::now(),
                priority: 1,
                generated_tokens: 0,
                estimated_token_value: 1.0,
            },
        );

        state.update_block_counts(self.total_blocks);

        let allocation = KvCacheAllocation {
            handle,
            matched_tokens,
            allocated_blocks: final_blocks,
        };
        drop(state);
        self.notify_evicted(&evicted_blocks);
        Ok(allocation)
    }

    fn free_paged(&self, request_id: &str) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(record) = state.active_requests.get_mut(request_id) {
            record.active = false;
            record.last_accessed = Instant::now();

            if let Some(pos) = state.lru_list.iter().position(|r| r == request_id) {
                state.lru_list.remove(pos);
            }
            state.lru_list.push_back(request_id.to_string());
        }
        state.update_block_counts(self.total_blocks);
    }

    fn get_metrics(&self) -> KvCacheMetrics {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.metrics.clone()
    }

    fn update_request_metadata(
        &self,
        request_id: &str,
        priority: u32,
        generated_tokens: usize,
        estimated_token_value: f64,
    ) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(record) = state.active_requests.get_mut(request_id) {
            record.priority = priority;
            record.generated_tokens = generated_tokens;
            record.estimated_token_value = estimated_token_value;
        }
    }
}

/// Disaggregated execution phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPhase {
    Load,
    Prefill,
    Decode,
    Decoding,
    Encode,
    Generate,
    Simulate,
    Postprocess,
}

/// A batch of tokens to be executed by the engine
pub struct ExecutionBatch {
    pub phase: ExecutionPhase,
    pub request_ids: Vec<RequestId>,
    pub tokens: Vec<u32>,
    pub cu_seqlens: Vec<usize>,
    pub kv_handles: Vec<usize>,
    /// Per-request start positions for decode phase (prompt_len + generated_tokens).
    pub start_positions: Vec<usize>,
    /// Per-request generation parameters (temperature, top_p, seed) for sampling.
    pub params: Vec<GenerationParams>,
    /// Per-request generated tokens so far (needed for grammar/schema validation).
    pub generated_tokens: Vec<Vec<u32>>,
}

/// The result of a batch execution
pub struct BatchResult {
    pub next_tokens: Vec<u32>,
    /// Optional speculative/accepted tokens beyond the first token.
    pub speculative_tokens: Option<Vec<Vec<u32>>>,
}

// ---------------------------------------------------------------------------
// Generic segment scheduling types (F05)
// ---------------------------------------------------------------------------

/// Classification of a scheduling request, controlling fairness and priority.
/// Ordered by priority: Maintenance (lowest) < BackgroundBatch < RealtimeStream < ForegroundInteractive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RequestClass {
    /// Maintenance tasks (GC, warmup, model eviction) — lowest priority.
    Maintenance,
    /// Background / batch jobs — low priority.
    BackgroundBatch,
    /// Realtime streaming (ASR, TTS, live inference) — high priority.
    RealtimeStream,
    /// Foreground interactive (chat, user-facing API) — highest priority.
    ForegroundInteractive,
}

impl RequestClass {
    /// Map RequestClass to the ResourcePriority used for resource budgeting.
    pub fn to_resource_priority(self) -> ResourcePriority {
        match self {
            Self::BackgroundBatch => ResourcePriority::Low,
            Self::Maintenance => ResourcePriority::Speculative,
            Self::RealtimeStream => ResourcePriority::High,
            Self::ForegroundInteractive => ResourcePriority::Critical,
        }
    }
}

/// Result produced after a segment completes execution.
#[derive(Debug, Clone)]
pub enum SegmentResult {
    /// Segment produced output data blocks.
    Success { outputs: Vec<DataBlock> },
    /// Segment needs to continue in a different phase.
    Continue { next_phase: ExecutionPhase },
    /// Segment failed with an error description.
    Failed { reason: String },
}

/// Reason for switching from one model/backend to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSwitchReason {
    BackendUnavailable,
    InsufficientMemory,
    ThermalLimit,
    PowerSaving,
    UserPolicy,
    BetterLatencyEstimate,
}

/// Describes the routing decision for a segment.
#[derive(Debug, Clone)]
pub struct ModelRoute {
    pub engine_name: String,
    pub backend_name: String,
    pub model_id: String,
    pub model_variant: Option<String>,
    pub switch_reason: Option<ModelSwitchReason>,
    pub degraded: bool,
}

/// A scheduled unit of work for the engine layer.
#[derive(Debug, Clone)]
pub struct ScheduledSegment {
    pub request_id: String,
    pub model_id: String,
    pub engine_name: String,
    pub backend_name: String,
    pub phase: ExecutionPhase,
    pub inputs: Vec<DataBlock>,
    pub cache_handle: Option<CacheHandle>,
    pub ticket: ResourceTicket,
    pub class: RequestClass,
    pub route: ModelRoute,
    pub deadline_ms: Option<u64>,
    pub created_at: Instant,
}

/// Trait for generic segment schedulers.
///
/// Implementations can be LLM-token-level (priority-aware), multi-modal pipeline
/// schedulers, or world-model realtime loop schedulers.
pub trait Scheduler: Send + Sync {
    /// Submit a new inference request to the scheduler.
    fn submit_segment(
        &self,
        request_id: String,
        model_id: String,
        phase: ExecutionPhase,
        inputs: Vec<DataBlock>,
        class: RequestClass,
        ticket: ResourceTicket,
    ) -> Result<()>;

    /// Pick the next batch of segments to execute, given current device capabilities.
    fn next_segments(&self, devices: &[DeviceCapability]) -> Result<Vec<ScheduledSegment>>;

    /// Report completion of a segment and optionally advance to the next phase.
    fn complete_segment(
        &self,
        request_id: &str,
        result: SegmentResult,
    ) -> Result<Option<ScheduledSegment>>;

    /// Query current queue depths per RequestClass.
    fn queue_depths(&self) -> HashMap<RequestClass, usize>;

    /// Number of in-flight (active) segments.
    fn active_count(&self) -> usize;
}

/// Decoupled execution engine that performs the actual model forward pass
pub trait EngineExecutor: Send + Sync {
    fn execute(&self, batch: ExecutionBatch) -> Result<BatchResult>;
    fn max_batch_size(&self, phase: ExecutionPhase) -> usize;
}

/// Fairness strategy for balancing request classes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FairnessStrategy {
    /// Strict priority — always drain higher-class queues first.
    #[default]
    StrictPriority,
    /// Weighted fair queuing — each class gets a configurable share of slots.
    WeightedFairQueue,
}

/// Environment constraints that affect scheduling decisions.
#[derive(Debug, Clone)]
pub struct EnvironmentConstraints {
    pub thermal_state: ThermalState,
    pub power_state: PowerState,
    /// Max concurrent segments when thermal is Serious.
    pub thermal_batch_limit: usize,
    /// Max concurrent segments when on battery.
    pub power_batch_limit: usize,
}

impl Default for EnvironmentConstraints {
    fn default() -> Self {
        Self {
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            thermal_batch_limit: 2,
            power_batch_limit: 4,
        }
    }
}

impl EnvironmentConstraints {
    /// Effective max batch size after applying thermal / power limits.
    pub fn effective_max_batch(&self, base_max: usize) -> usize {
        let mut limit = base_max;
        match self.thermal_state {
            ThermalState::Nominal | ThermalState::Fair => {}
            ThermalState::Serious => limit = limit.min(self.thermal_batch_limit),
            ThermalState::Critical => limit = limit.min(1),
        }
        if self.power_state == PowerState::Battery {
            limit = limit.min(self.power_batch_limit);
        }
        limit.max(1)
    }

    /// Whether thermal or power conditions require degradation.
    pub fn is_degraded(&self) -> bool {
        matches!(
            self.thermal_state,
            ThermalState::Serious | ThermalState::Critical
        ) || self.power_state == PowerState::Battery
    }
}

// ---------------------------------------------------------------------------
// BloomScheduler — generic segment scheduler (F05)
// ---------------------------------------------------------------------------

/// Internal record for a submitted segment request.
#[derive(Debug, Clone)]
struct SegmentRecord {
    request_id: String,
    model_id: String,
    engine_name: String,
    backend_name: String,
    phase: ExecutionPhase,
    inputs: Vec<DataBlock>,
    cache_handle: Option<CacheHandle>,
    ticket: ResourceTicket,
    class: RequestClass,
    deadline_ms: Option<u64>,
    created_at: Instant,
    token_cost: usize,
}

/// Generic segment scheduler supporting multi-class fairness, thermal/power
/// awareness, and cross-engine routing.
pub struct BloomScheduler {
    /// Per-class submission queues.
    queues: Mutex<HashMap<RequestClass, VecDeque<SegmentRecord>>>,
    /// Currently executing segments.
    active: Mutex<HashMap<String, SegmentRecord>>,
    /// Completed segments awaiting result collection.
    completed: Mutex<VecDeque<(String, SegmentResult)>>,
    /// Fairness strategy.
    fairness: FairnessStrategy,
    /// Environment constraints (thermal, power).
    constraints: Mutex<EnvironmentConstraints>,
    /// Max segments per `next_segments` call (before constraint adjustment).
    base_max_batch: usize,
    /// Token-level budget for prefill/decode segment admission.
    token_config: TokenSchedulingConfig,
    /// Monotonic round-robin counter for WFQ.
    rr_counter: Mutex<usize>,
}

impl BloomScheduler {
    pub fn new(base_max_batch: usize) -> Self {
        let mut queues = HashMap::new();
        queues.insert(RequestClass::ForegroundInteractive, VecDeque::new());
        queues.insert(RequestClass::RealtimeStream, VecDeque::new());
        queues.insert(RequestClass::BackgroundBatch, VecDeque::new());
        queues.insert(RequestClass::Maintenance, VecDeque::new());

        Self {
            queues: Mutex::new(queues),
            active: Mutex::new(HashMap::new()),
            completed: Mutex::new(VecDeque::new()),
            fairness: FairnessStrategy::default(),
            constraints: Mutex::new(EnvironmentConstraints::default()),
            base_max_batch,
            token_config: TokenSchedulingConfig::default(),
            rr_counter: Mutex::new(0),
        }
    }

    pub fn with_token_config(base_max_batch: usize, token_config: TokenSchedulingConfig) -> Self {
        let mut scheduler = Self::new(base_max_batch);
        scheduler.token_config = token_config;
        scheduler
    }

    /// Set the fairness strategy.
    pub fn set_fairness(&mut self, strategy: FairnessStrategy) {
        self.fairness = strategy;
    }

    /// Update environment constraints (thermal state, power state, etc.).
    pub fn update_constraints(&self, constraints: EnvironmentConstraints) {
        *self.constraints.lock().unwrap_or_else(|e| e.into_inner()) = constraints;
    }

    /// Get current environment constraints.
    pub fn current_constraints(&self) -> EnvironmentConstraints {
        self.constraints
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Build a ModelRoute for a segment record.
    fn build_route(record: &SegmentRecord, degraded: bool) -> ModelRoute {
        ModelRoute {
            engine_name: record.engine_name.clone(),
            backend_name: record.backend_name.clone(),
            model_id: record.model_id.clone(),
            model_variant: None,
            switch_reason: if degraded {
                Some(ModelSwitchReason::ThermalLimit)
            } else {
                None
            },
            degraded,
        }
    }

    fn record_to_segment(record: &SegmentRecord, degraded: bool) -> ScheduledSegment {
        ScheduledSegment {
            request_id: record.request_id.clone(),
            model_id: record.model_id.clone(),
            engine_name: record.engine_name.clone(),
            backend_name: record.backend_name.clone(),
            phase: record.phase,
            inputs: record.inputs.clone(),
            cache_handle: record.cache_handle.clone(),
            ticket: record.ticket.clone(),
            class: record.class,
            route: Self::build_route(record, degraded),
            deadline_ms: record.deadline_ms,
            created_at: record.created_at,
        }
    }

    fn estimate_input_tokens(inputs: &[DataBlock], phase: ExecutionPhase) -> usize {
        let input_tokens = inputs
            .iter()
            .map(|block| match block {
                DataBlock::Text(text) => {
                    let chars = text.chars().count();
                    if !text.is_ascii() {
                        chars.max(1)
                    } else {
                        (chars / 4).max(1)
                    }
                }
                DataBlock::Tokens(tokens) => tokens.len().max(1),
                DataBlock::AudioPcm { samples, .. } => samples.len().max(1),
                DataBlock::AudioFile { .. } | DataBlock::Image { .. } => 1,
                DataBlock::VideoFrames(frames) => frames.len().max(1),
                DataBlock::Tensor(data) => data.len().max(1),
                DataBlock::WorldState { .. } | DataBlock::Action { .. } => 1,
            })
            .sum::<usize>()
            .max(1);

        match phase {
            ExecutionPhase::Decode | ExecutionPhase::Decoding => 1,
            _ => input_tokens,
        }
    }

    fn token_phase(phase: ExecutionPhase) -> TokenPhase {
        match phase {
            ExecutionPhase::Decode | ExecutionPhase::Decoding => TokenPhase::Decode,
            _ => TokenPhase::Prefill,
        }
    }

    fn pop_if_admitted(
        &self,
        q: &mut VecDeque<SegmentRecord>,
        out_is_empty: bool,
        admission: &mut TokenAdmission,
    ) -> Option<SegmentRecord> {
        let record = q.front()?;
        let phase = Self::token_phase(record.phase);
        if admission.try_reserve(&self.token_config, phase, record.token_cost) || out_is_empty {
            q.pop_front()
        } else {
            None
        }
    }

    /// Pop segments from queues using strict-priority strategy.
    fn pop_strict_priority(
        &self,
        queues: &mut HashMap<RequestClass, VecDeque<SegmentRecord>>,
        max: usize,
    ) -> Vec<SegmentRecord> {
        let priority_order = [
            RequestClass::ForegroundInteractive,
            RequestClass::RealtimeStream,
            RequestClass::BackgroundBatch,
            RequestClass::Maintenance,
        ];
        let mut out = Vec::new();
        let mut admission = TokenAdmission::default();
        for class in &priority_order {
            if out.len() >= max {
                break;
            }
            if let Some(q) = queues.get_mut(class) {
                while out.len() < max {
                    let out_is_empty = out.is_empty();
                    match self.pop_if_admitted(q, out_is_empty, &mut admission) {
                        Some(r) => out.push(r),
                        None => break,
                    }
                }
            }
        }
        out
    }

    /// Pop segments using weighted fair queuing.
    fn pop_wfq(
        &self,
        queues: &mut HashMap<RequestClass, VecDeque<SegmentRecord>>,
        max: usize,
    ) -> Vec<SegmentRecord> {
        // Weights: FG=4, RT=3, BG=2, MT=1
        let weights: [(RequestClass, usize); 4] = [
            (RequestClass::ForegroundInteractive, 4),
            (RequestClass::RealtimeStream, 3),
            (RequestClass::BackgroundBatch, 2),
            (RequestClass::Maintenance, 1),
        ];
        let mut rr = self.rr_counter.lock().unwrap_or_else(|e| e.into_inner());
        let total_weight: usize = weights.iter().map(|(_, w)| *w).sum();
        let start = *rr % total_weight;
        *rr = rr.wrapping_add(1);

        let mut out = Vec::new();
        let mut remaining = max;
        let mut admission = TokenAdmission::default();

        // Cycle through classes starting from round-robin position
        for offset in 0..weights.len() {
            if remaining == 0 {
                break;
            }
            let idx = (start + offset) % weights.len();
            let (class, weight) = weights[idx];
            let share = weight.min(remaining);
            if let Some(q) = queues.get_mut(&class) {
                for _ in 0..share {
                    let out_is_empty = out.is_empty();
                    match self.pop_if_admitted(q, out_is_empty, &mut admission) {
                        Some(r) => {
                            out.push(r);
                            remaining -= 1;
                        }
                        None => break,
                    }
                }
            }
        }
        // Second pass: fill remaining slots from any non-empty queue
        for (class, _) in &weights {
            if remaining == 0 {
                break;
            }
            if let Some(q) = queues.get_mut(class) {
                while remaining > 0 {
                    let out_is_empty = out.is_empty();
                    match self.pop_if_admitted(q, out_is_empty, &mut admission) {
                        Some(r) => {
                            out.push(r);
                            remaining -= 1;
                        }
                        None => break,
                    }
                }
            }
        }
        out
    }
}

impl Scheduler for BloomScheduler {
    fn submit_segment(
        &self,
        request_id: String,
        model_id: String,
        phase: ExecutionPhase,
        inputs: Vec<DataBlock>,
        class: RequestClass,
        ticket: ResourceTicket,
    ) -> Result<()> {
        let token_cost = Self::estimate_input_tokens(&inputs, phase);
        let engine_name = ticket.preferred_backend.clone().unwrap_or_default();
        let backend_name = ticket
            .preferred_backend
            .clone()
            .or_else(|| ticket.fallback_backends.first().cloned())
            .unwrap_or_else(|| "cpu".to_string());

        let record = SegmentRecord {
            request_id: request_id.clone(),
            model_id,
            engine_name,
            backend_name,
            phase,
            inputs,
            cache_handle: None,
            ticket,
            class,
            deadline_ms: None,
            created_at: Instant::now(),
            token_cost,
        };

        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        queues
            .get_mut(&class)
            .ok_or_else(|| BloomError::InvalidInput(format!("Unknown request class: {:?}", class)))?
            .push_back(record);
        Ok(())
    }

    fn next_segments(&self, devices: &[DeviceCapability]) -> Result<Vec<ScheduledSegment>> {
        // Update constraints from device list FIRST, before calculating batch size
        if let Some(dev) = devices.first() {
            let mut c = self.constraints.lock().unwrap_or_else(|e| e.into_inner());
            c.thermal_state = dev.thermal_state;
            c.power_state = dev.power_state;
        }

        let constraints = self
            .constraints
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        // Derive effective batch size from (now updated) constraints
        let effective_max = constraints.effective_max_batch(self.base_max_batch);
        let degraded = constraints.is_degraded();

        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        let records = match self.fairness {
            FairnessStrategy::StrictPriority => {
                self.pop_strict_priority(&mut queues, effective_max)
            }
            FairnessStrategy::WeightedFairQueue => self.pop_wfq(&mut queues, effective_max),
        };

        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        let mut segments = Vec::new();
        for record in records {
            let seg = Self::record_to_segment(&record, degraded);
            active.insert(record.request_id.clone(), record);
            segments.push(seg);
        }
        Ok(segments)
    }

    fn complete_segment(
        &self,
        request_id: &str,
        result: SegmentResult,
    ) -> Result<Option<ScheduledSegment>> {
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        let mut completed = self.completed.lock().unwrap_or_else(|e| e.into_inner());

        let record = active.remove(request_id);
        completed.push_back((request_id.to_string(), result.clone()));

        // If segment wants to continue in a new phase, create a follow-up segment
        if let (Some(rec), SegmentResult::Continue { next_phase }) = (record, result) {
            let degraded = self
                .constraints
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_degraded();
            let new_rec = SegmentRecord {
                phase: next_phase,
                token_cost: Self::estimate_input_tokens(&rec.inputs, next_phase),
                ..rec
            };
            let seg = Self::record_to_segment(&new_rec, degraded);
            // Put back into the appropriate queue
            let class = new_rec.class;
            let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(q) = queues.get_mut(&class) {
                q.push_front(new_rec);
            }
            return Ok(Some(seg));
        }

        Ok(None)
    }

    fn queue_depths(&self) -> HashMap<RequestClass, usize> {
        let queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        queues.iter().map(|(k, v)| (*k, v.len())).collect()
    }

    fn active_count(&self) -> usize {
        self.active.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

// ---------------------------------------------------------------------------
// InferenceScheduler — LLM token-level scheduler
// ---------------------------------------------------------------------------

/// Engine-local token scheduler for continuous batching and chunked prefill.
pub struct InferenceScheduler {
    executor: Arc<dyn EngineExecutor>,
    /// Serializes batch construction/execution with destructive cancellation
    /// so a KV handle cannot be freed or reused while an executor is using it.
    execution_gate: Mutex<()>,
    prefill_queue: Mutex<VecDeque<ScheduledRequest>>,
    decoding_queue: Mutex<VecDeque<ScheduledRequest>>,
    active_requests: Mutex<HashMap<RequestId, ScheduledRequest>>,
    kv_pool: Arc<dyn KvCachePool>,
    /// Maximum total tokens per scheduling step (prefill + decode).
    max_num_tokens: usize,
    /// Channel senders for active request token streams.
    pub token_senders:
        Arc<Mutex<HashMap<RequestId, tokio::sync::mpsc::UnboundedSender<Result<u32, String>>>>>,
    // priority-aware components
    config: CoreTokenSchedulingConfig,
    rate_limiter: Mutex<TokenBucketRateLimiter>,
    chunked_prefill_queue: Mutex<ChunkedPrefillQueue>,
    preemption_manager: Mutex<PreemptionManager>,
    kv_eviction_manager: Mutex<KvEvictionManager>,
    pending_prefill_requests: Mutex<HashMap<RequestId, ScheduledRequest>>,
}

impl InferenceScheduler {
    pub fn new(executor: Arc<dyn EngineExecutor>, kv_pool: Arc<dyn KvCachePool>) -> Self {
        Self::with_config(executor, kv_pool, CoreTokenSchedulingConfig::default())
    }

    pub fn with_max_tokens(
        executor: Arc<dyn EngineExecutor>,
        kv_pool: Arc<dyn KvCachePool>,
        max_num_tokens: usize,
    ) -> Self {
        let config = CoreTokenSchedulingConfig {
            max_total_tokens_per_step: max_num_tokens,
            ..Default::default()
        };
        Self::with_config(executor, kv_pool, config)
    }

    pub fn with_config(
        executor: Arc<dyn EngineExecutor>,
        kv_pool: Arc<dyn KvCachePool>,
        config: CoreTokenSchedulingConfig,
    ) -> Self {
        let rate_limiter = Mutex::new(TokenBucketRateLimiter::new(config.rate_limiter.clone()));
        let chunked_prefill_queue =
            Mutex::new(ChunkedPrefillQueue::new(config.chunked_prefill.clone()));
        let preemption_manager = Mutex::new(PreemptionManager::new(config.preemption.clone()));
        let kv_eviction_manager = Mutex::new(KvEvictionManager::new(config.kv_eviction.clone()));

        Self {
            executor,
            execution_gate: Mutex::new(()),
            prefill_queue: Mutex::new(VecDeque::new()),
            decoding_queue: Mutex::new(VecDeque::new()),
            active_requests: Mutex::new(HashMap::new()),
            kv_pool,
            max_num_tokens: config.max_total_tokens_per_step,
            token_senders: Arc::new(Mutex::new(HashMap::new())),
            config,
            rate_limiter,
            chunked_prefill_queue,
            preemption_manager,
            kv_eviction_manager,
            pending_prefill_requests: Mutex::new(HashMap::new()),
        }
    }

    /// Cancel a request by ID, freeing its KV cache allocation.
    pub fn cancel_request(&self, request_id: &str) -> bool {
        let _execution = self
            .execution_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.cancel_request_inner(request_id)
    }

    /// Cancel while the caller already owns `execution_gate`.
    fn cancel_request_inner(&self, request_id: &str) -> bool {
        // A response body can disappear independently of queue ownership. Drop
        // the client sender first so cancellation never leaves an orphaned
        // channel registration, including races before or after scheduling.
        let sender_removed = self
            .token_senders
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(request_id)
            .is_some();

        // Submit and scheduling use the same chunked -> pending lock order.
        // Remove both representations atomically so a concurrent submit can
        // never expose a half-cancelled chunked request. KV release stays
        // outside both locks because it may invoke backend callbacks.
        let (pending_request, chunked_removed) = {
            let mut chunked = self
                .chunked_prefill_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut pending = self
                .pending_prefill_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let chunked_removed = chunked.cancel(request_id);
            (pending.remove(request_id), chunked_removed)
        };
        if let Some(req) = pending_request {
            if let Some(handle) = req.kv_handle {
                self.kv_pool.free(handle);
            }
            return true;
        }

        // Check prefill queue
        let prefill_request = {
            let mut prefill = self.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            prefill
                .iter()
                .position(|r| r.id == request_id)
                .and_then(|pos| prefill.remove(pos))
        };
        if let Some(req) = prefill_request {
            if let Some(handle) = req.kv_handle {
                self.kv_pool.free(handle);
            }
            return true;
        }

        // Check decoding queue
        let decoding_request = {
            let mut decoding = self
                .decoding_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            decoding
                .iter()
                .position(|r| r.id == request_id)
                .and_then(|pos| decoding.remove(pos))
        };
        if let Some(req) = decoding_request {
            if let Some(handle) = req.kv_handle {
                self.kv_pool.free(handle);
            }
            return true;
        }

        // Check active requests
        let active_request = {
            let mut active = self
                .active_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            active.remove(request_id)
        };
        if let Some(req) = active_request {
            if let Some(handle) = req.kv_handle {
                self.kv_pool.free(handle);
            }
            return true;
        }
        sender_removed || chunked_removed
    }

    /// Return queue depths for monitoring.
    pub fn queue_stats(&self) -> (usize, usize, usize) {
        let prefill = if self.config.chunked_prefill.enabled {
            self.chunked_prefill_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pending_count()
        } else {
            self.prefill_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len()
        };
        let decoding = self
            .decoding_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        let active = self
            .active_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        (prefill, decoding, active)
    }

    pub fn submit(&self, request: Request) -> Result<()> {
        self.submit_scheduled_request(ScheduledRequest {
            request,
            execution_guard: None,
        })
    }

    /// Submit a request while retaining opaque ownership through every model
    /// invocation scheduled for it.
    pub fn submit_with_execution_guard(
        &self,
        request: Request,
        execution_guard: ExecutionGuard,
    ) -> Result<()> {
        self.submit_scheduled_request(ScheduledRequest {
            request,
            execution_guard: Some(execution_guard),
        })
    }

    fn submit_scheduled_request(&self, request: ScheduledRequest) -> Result<()> {
        let request_id = request.id.clone();
        let result = self.submit_inner(request);
        if result.is_err() {
            self.token_senders
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&request_id);
        }
        result
    }

    fn submit_inner(&self, mut request: ScheduledRequest) -> Result<()> {
        // HBM Admission Control & Eviction
        if self.config.enabled {
            let metrics = self.kv_pool.get_metrics();
            let block_size = 16;
            let active_tokens = metrics.active_blocks * block_size;
            let total_tokens = metrics.total_blocks * block_size;

            // Collect KvSessionInfo for eviction decisions
            let mut sessions = Vec::new();
            {
                let active = self
                    .active_requests
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                for r in active.values() {
                    sessions.push(KvSessionInfo {
                        request_id: r.id.clone(),
                        model_id: r.model_id.clone(),
                        priority: r.priority,
                        created_at: r.created_at,
                        last_accessed: r.last_accessed,
                        kv_cache_tokens: r.prompt_tokens.len() + r.generated_tokens.len(),
                        generated_tokens: r.generated_tokens.len(),
                        estimated_token_value: Some(1.0),
                        is_active: matches!(
                            r.state,
                            RequestState::Decoding { .. } | RequestState::Prefill
                        ),
                    });
                }
            }
            {
                let decoding = self
                    .decoding_queue
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                for r in decoding.iter() {
                    sessions.push(KvSessionInfo {
                        request_id: r.id.clone(),
                        model_id: r.model_id.clone(),
                        priority: r.priority,
                        created_at: r.created_at,
                        last_accessed: r.last_accessed,
                        kv_cache_tokens: r.prompt_tokens.len() + r.generated_tokens.len(),
                        generated_tokens: r.generated_tokens.len(),
                        estimated_token_value: Some(1.0),
                        is_active: false,
                    });
                }
            }

            let mut eviction_mgr = self
                .kv_eviction_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let num_tokens = request.prompt_tokens.len() + request.params.max_tokens;

            // Check admission
            match eviction_mgr.check_admission(active_tokens, total_tokens) {
                AdmissionResult::Rejected { reason, .. } => {
                    // Try to evict inactive sessions to make room
                    let victims = eviction_mgr.select_eviction_victims(&sessions, num_tokens);
                    if victims.is_empty() {
                        return Err(BloomError::SchedulingFailed(format!(
                            "Admission rejected: {} and no inactive sessions can be evicted",
                            reason
                        ))
                        .into());
                    }
                    // Free evicted requests
                    for victim in victims {
                        self.cancel_request(&victim.request_id);
                    }
                }
                AdmissionResult::Admitted => {}
            }
        }

        // Inference scheduler: allocate KV cache at submission time.
        // Use `allocate_paged` (not the legacy `allocate`) so the pool
        // registers the real `request_id` and `prompt_tokens` — this is
        // what makes `free_paged(request_id)` resolve and what populates
        // the prefix → block_id `block_table` for cross-request prefix
        // cache hits. Calling `allocate(num_tokens)` here would internally
        // synthesize a `legacy-handle-{n}` request_id and pass `&[]` as
        // prompt_tokens, silently breaking both eviction and prefix reuse.
        let max_new_tokens = request.params.max_tokens;
        let prompt_tokens = request.prompt_tokens.clone();
        let alloc = self.kv_pool.allocate_paged(
            &request.id,
            &prompt_tokens,
            max_new_tokens,
            request.multimodal_hash.as_deref(),
        )?;
        let slot = alloc.handle;

        request.kv_handle = Some(slot);
        request.state = RequestState::Pending;

        if self.config.chunked_prefill.enabled {
            let mut cp_queue = self
                .chunked_prefill_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut pending = self
                .pending_prefill_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cp_queue.submit(request.id.clone(), request.prompt_tokens.clone());
            pending.insert(request.id.clone(), request);
        } else {
            let mut prefill = self.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            prefill.push_back(request);
        }
        Ok(())
    }

    pub fn step(&self) -> Result<()> {
        let _execution = self
            .execution_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Check and perform preemption if high-priority requests are waiting
        let preempted_ids = self.check_and_perform_preemption()?;

        // Token budget tracking for in-flight batching. Like vLLM V1, running
        // decode work is admitted first, then waiting/chunked prefill uses the
        // remaining budget. `decode_quantum_tokens` mirrors SGLang's continuous
        // decode steps knob while keeping the default at one token step.
        let mut tokens_used = 0usize;
        let mut prefill_tokens_used = 0usize;
        let mut decode_tokens_used = 0usize;
        let mut scheduled_decode = false;

        let decode_microsteps = self.config.budget().decode_quantum_tokens;
        for _ in 0..decode_microsteps {
            if tokens_used >= self.max_num_tokens {
                break;
            }
            let Some(batch) =
                self.schedule_decoding_budgeted(&mut tokens_used, &mut decode_tokens_used)?
            else {
                break;
            };
            scheduled_decode = true;
            self.execute_scheduled_batch(batch, ExecutionPhase::Decode)?;
        }

        // Mixed chunked prefill follows SGLang's `enable_mixed_chunk` semantics:
        // decode runs first, and prefill chunks may fill leftover budget only
        // when interleaving is enabled.
        let can_interleave_prefill = !scheduled_decode
            || !self.config.chunked_prefill.enabled
            || self.config.chunked_prefill.interleave_with_decode;
        if can_interleave_prefill
            && tokens_used < self.max_num_tokens
            && let Some(batch) = self.schedule_prefill_budgeted(
                &mut tokens_used,
                &mut prefill_tokens_used,
                &preempted_ids,
            )?
        {
            self.execute_scheduled_batch(batch, ExecutionPhase::Prefill)?;
        }

        Ok(())
    }

    fn execute_scheduled_batch(
        &self,
        scheduled: ScheduledBatch,
        phase: ExecutionPhase,
    ) -> Result<()> {
        let ScheduledBatch {
            request_ids,
            batch,
            is_final,
            execution_guards,
        } = scheduled;

        // Cancellation waits on `execution_gate`, while these cloned guards
        // additionally retain opaque runtime ownership through executor
        // completion and scheduler bookkeeping.
        let result = match self.executor.execute(batch) {
            Ok(result) => result,
            Err(error) => {
                self.fail_scheduled_requests(&request_ids);
                drop(execution_guards);
                return Err(error);
            }
        };

        if result.next_tokens.len() != request_ids.len() || is_final.len() != request_ids.len() {
            let error = BloomError::SchedulingFailed(format!(
                "executor returned {} next tokens and {} completion flags for {} requests",
                result.next_tokens.len(),
                is_final.len(),
                request_ids.len()
            ));
            self.fail_scheduled_requests(&request_ids);
            drop(execution_guards);
            return Err(error.into());
        }

        if let Err(error) = self.process_result(&request_ids, result, phase, &is_final) {
            self.fail_scheduled_requests(&request_ids);
            drop(execution_guards);
            return Err(error);
        }
        drop(execution_guards);
        Ok(())
    }

    fn fail_scheduled_requests(&self, request_ids: &[RequestId]) {
        // Keep client-facing failures generic and bounded. The original
        // executor error is returned to the caller for internal logging.
        {
            let senders = self.token_senders.lock().unwrap_or_else(|e| e.into_inner());
            for request_id in request_ids {
                if let Some(sender) = senders.get(request_id) {
                    let _ = sender.send(Err(EXECUTION_FAILED_CLIENT_MESSAGE.to_string()));
                }
            }
        }

        for request_id in request_ids {
            self.cancel_request_inner(request_id);
        }
    }

    fn prefill_tokens_for(&self, req: &Request) -> Vec<u32> {
        let mut tokens = req.prompt_tokens.clone();
        tokens.extend_from_slice(&req.generated_tokens);
        tokens
    }

    fn check_and_perform_preemption(&self) -> Result<Vec<RequestId>> {
        let mut preempted_ids = Vec::new();
        if !self.config.preemption.enabled {
            return Ok(preempted_ids);
        }

        // Get the highest waiting priority and its wait time
        let mut max_wait_ms = 0;
        let mut highest_waiting_priority = 0;

        if self.config.chunked_prefill.enabled {
            let cp_queue = self
                .chunked_prefill_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let pending = self
                .pending_prefill_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for state in cp_queue.queue.iter() {
                if let Some(req) = pending.get(&state.request_id) {
                    let wait = req.created_at.elapsed().as_millis() as u64;
                    if wait > max_wait_ms {
                        max_wait_ms = wait;
                    }
                    if req.priority > highest_waiting_priority {
                        highest_waiting_priority = req.priority;
                    }
                }
            }
        } else {
            let prefill = self.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            for req in prefill.iter() {
                let wait = req.created_at.elapsed().as_millis() as u64;
                if wait > max_wait_ms {
                    max_wait_ms = wait;
                }
                if req.priority > highest_waiting_priority {
                    highest_waiting_priority = req.priority;
                }
            }
        }

        let mut pm = self
            .preemption_manager
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if pm.should_preempt(max_wait_ms) {
            // Find candidate active requests to preempt
            let mut preemptible = Vec::new();
            {
                let decoding = self
                    .decoding_queue
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                for r in decoding.iter() {
                    if r.priority < highest_waiting_priority {
                        preemptible.push(PreemptibleRequest {
                            request_id: r.id.clone(),
                            model_id: r.model_id.clone(),
                            priority: r.priority,
                            generated_tokens: r.generated_tokens.len(),
                            kv_cache_tokens: r.prompt_tokens.len() + r.generated_tokens.len(),
                            preemption_count: r.preemption_count,
                            decode_started_at: r.decode_started_at.unwrap_or_else(Instant::now),
                            last_scheduled_at: r.last_scheduled_at.unwrap_or_else(Instant::now),
                        });
                    }
                }
            }

            if !preemptible.is_empty()
                && let Some(decision) = pm.select_victim(&preemptible)
            {
                // Actually preempt the victim
                let mut decoding = self
                    .decoding_queue
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(mut req) = decoding
                    .iter()
                    .position(|r| r.id == decision.preempted_request_id)
                    .and_then(|pos| decoding.remove(pos))
                {
                    preempted_ids.push(req.id.clone());

                    // Free KV cache
                    if let Some(handle) = req.kv_handle.take() {
                        self.kv_pool.free(handle);
                    }

                    // Update request state
                    req.preemption_count = decision.preemption_count;
                    req.state = RequestState::Pending;

                    // Move back to prefill
                    if self.config.chunked_prefill.enabled {
                        let mut cp_queue = self
                            .chunked_prefill_queue
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        let mut pending = self
                            .pending_prefill_requests
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        let full_prompt = self.prefill_tokens_for(&req);
                        cp_queue.submit(req.id.clone(), full_prompt);
                        pending.insert(req.id.clone(), req);
                    } else {
                        let mut prefill =
                            self.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
                        prefill.push_back(req);
                    }
                }
            }
        }

        Ok(preempted_ids)
    }

    /// Schedule a prefill batch that respects the token budget.
    fn schedule_prefill_budgeted(
        &self,
        tokens_used: &mut usize,
        prefill_tokens_used: &mut usize,
        preempted_ids: &[RequestId],
    ) -> Result<Option<ScheduledBatch>> {
        if self.config.chunked_prefill.enabled {
            let mut cp_queue = self
                .chunked_prefill_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut pending = self
                .pending_prefill_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut active = self
                .active_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut rate_limiter = self.rate_limiter.lock().unwrap_or_else(|e| e.into_inner());

            if cp_queue.queue.is_empty() {
                return Ok(None);
            }

            let max_batch = self
                .executor
                .max_batch_size(ExecutionPhase::Prefill)
                .min(self.config.budget().concurrent_segments);
            let mut batch_request_ids = Vec::new();
            let mut tokens = Vec::new();
            let mut kv_handles = Vec::new();
            let mut cu_seqlens = vec![0];
            let mut start_positions = Vec::new();
            let mut params = Vec::new();
            let mut is_final_vec = Vec::new();
            let mut generated_tokens = Vec::new();
            let mut execution_guards = Vec::new();

            let mut i = 0;
            while i < cp_queue.queue.len() && batch_request_ids.len() < max_batch {
                let state = &mut cp_queue.queue[i];
                let req_id = state.request_id.clone();
                if preempted_ids.contains(&req_id) {
                    i += 1;
                    continue;
                }

                let chunk_size = state.next_chunk_size();
                let phase_budget = self.config.budget().prefill_tokens.min(self.max_num_tokens);
                let remaining_budget = self
                    .max_num_tokens
                    .saturating_sub(*tokens_used)
                    .min(phase_budget.saturating_sub(*prefill_tokens_used));
                let scheduled_chunk_size = chunk_size.min(remaining_budget);
                if scheduled_chunk_size == 0 {
                    i += 1;
                    continue;
                }

                // Get request info
                let (model_id, req_params, kv_handle, execution_guard) = {
                    let req_opt = if pending.contains_key(&req_id) {
                        pending.get_mut(&req_id)
                    } else {
                        active.get_mut(&req_id)
                    };
                    if let Some(req) = req_opt {
                        if req.kv_handle.is_none() {
                            // Fallback allocation when submit didn't
                            // allocate (e.g. legacy code path). Use
                            // `allocate_paged` with the real request_id
                            // and prompt_tokens so prefix caching and
                            // `free_paged` keep working.
                            let alloc = self.kv_pool.allocate_paged(
                                &req.id,
                                &req.prompt_tokens,
                                req.params.max_tokens,
                                req.multimodal_hash.as_deref(),
                            );
                            match alloc {
                                Ok(a) => {
                                    req.kv_handle = Some(a.handle);
                                }
                                Err(_) => {
                                    i += 1;
                                    continue;
                                }
                            }
                        }
                        let Some(kv_handle) = req.kv_handle else {
                            i += 1;
                            continue;
                        };
                        (
                            req.model_id.clone(),
                            req.params.clone(),
                            kv_handle,
                            req.execution_guard.clone(),
                        )
                    } else {
                        i += 1;
                        continue;
                    }
                };

                // Check rate limiter
                if self.config.rate_limiter.enabled {
                    match rate_limiter.try_acquire(&model_id, scheduled_chunk_size) {
                        RateLimitDecision::Throttled { .. } => {
                            i += 1;
                            continue; // Skip this request for this step to avoid head-of-line blocking
                        }
                        RateLimitDecision::Allowed => {}
                    }
                }

                // Schedule this chunk
                let start = state.filled_tokens;
                let end = start + scheduled_chunk_size;
                let chunk_tokens = state.prompt_tokens[start..end].to_vec();
                *tokens_used += scheduled_chunk_size;
                *prefill_tokens_used += scheduled_chunk_size;
                batch_request_ids.push(req_id.clone());
                tokens.extend_from_slice(&chunk_tokens);
                kv_handles.push(kv_handle);
                cu_seqlens.push(tokens.len());
                start_positions.push(start);
                params.push(req_params);
                if let Some(guard) = execution_guard {
                    execution_guards.push(guard);
                }

                // Advance chunk state
                state.filled_tokens += scheduled_chunk_size;
                let is_final = state.filled_tokens >= state.prompt_tokens.len();
                if is_final {
                    state.finished = true;
                }
                is_final_vec.push(is_final);

                // Move from pending to active if needed
                if let Some(mut req) = pending.remove(&req_id) {
                    req.state = RequestState::Prefill;
                    active.insert(req_id.clone(), req);
                } else if let Some(req) = active.get_mut(&req_id) {
                    req.state = RequestState::Prefill;
                }

                let req_gen = active
                    .get(&req_id)
                    .map(|r| r.generated_tokens.clone())
                    .unwrap_or_default();
                generated_tokens.push(req_gen);

                if is_final {
                    cp_queue.queue.remove(i);
                } else {
                    i += 1;
                }
            }

            if batch_request_ids.is_empty() {
                return Ok(None);
            }

            return Ok(Some(ScheduledBatch {
                request_ids: batch_request_ids.clone(),
                batch: ExecutionBatch {
                    phase: ExecutionPhase::Prefill,
                    request_ids: batch_request_ids,
                    tokens,
                    cu_seqlens,
                    kv_handles,
                    start_positions,
                    params,
                    generated_tokens,
                },
                is_final: is_final_vec,
                execution_guards,
            }));
        }

        // Non-chunked prefill path
        let mut prefill = self.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
        if prefill.is_empty() {
            return Ok(None);
        }

        let max_batch = self
            .executor
            .max_batch_size(ExecutionPhase::Prefill)
            .min(self.config.budget().concurrent_segments);
        let mut batch_request_ids = Vec::new();
        let mut tokens = Vec::new();
        let mut kv_handles = Vec::new();
        let mut cu_seqlens = vec![0];
        let mut start_positions = Vec::new();
        let mut params = Vec::new();
        let mut is_final_vec = Vec::new();
        let mut generated_tokens = Vec::new();
        let mut execution_guards = Vec::new();

        let mut rate_limiter = self.rate_limiter.lock().unwrap_or_else(|e| e.into_inner());
        let mut i = 0;
        while i < prefill.len() && batch_request_ids.len() < max_batch {
            let req = &prefill[i];
            if preempted_ids.contains(&req.id) {
                i += 1;
                continue;
            }
            let req_tokens = self.prefill_tokens_for(req);
            let req_token_count = req_tokens.len();
            let phase_budget = self.config.budget().prefill_tokens.min(self.max_num_tokens);
            if !token_budget_allows(*tokens_used, req_token_count, self.max_num_tokens)
                || !token_budget_allows(*prefill_tokens_used, req_token_count, phase_budget)
            {
                i += 1;
                continue;
            }

            // Check rate limiter
            if self.config.rate_limiter.enabled {
                match rate_limiter.try_acquire(&req.model_id, req_token_count) {
                    RateLimitDecision::Throttled { .. } => {
                        i += 1;
                        continue;
                    }
                    RateLimitDecision::Allowed => {}
                }
            }

            // Pop request and schedule
            let Some(mut req) = prefill.remove(i) else {
                break;
            };
            if req.kv_handle.is_none() {
                // Fallback allocation when submit didn't allocate.
                // Use `allocate_paged` so prefix caching and
                // `free_paged` keep working with the real request_id.
                match self.kv_pool.allocate_paged(
                    &req.id,
                    &req.prompt_tokens,
                    req.params.max_tokens,
                    req.multimodal_hash.as_deref(),
                ) {
                    Ok(a) => {
                        req.kv_handle = Some(a.handle);
                    }
                    Err(_) => {
                        prefill.insert(i, req);
                        i += 1;
                        continue;
                    }
                }
            }

            let Some(kv_handle) = req.kv_handle else {
                prefill.insert(i, req);
                i += 1;
                continue;
            };

            *tokens_used += req_token_count;
            *prefill_tokens_used += req_token_count;
            req.state = RequestState::Prefill;
            batch_request_ids.push(req.id.clone());
            tokens.extend_from_slice(&req_tokens);
            kv_handles.push(kv_handle);
            cu_seqlens.push(tokens.len());
            start_positions.push(0);
            params.push(req.params.clone());
            is_final_vec.push(true);
            generated_tokens.push(req.generated_tokens.clone());
            if let Some(guard) = req.execution_guard.clone() {
                execution_guards.push(guard);
            }

            let mut active = self
                .active_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            active.insert(req.id.clone(), req);
        }

        if batch_request_ids.is_empty() {
            return Ok(None);
        }

        Ok(Some(ScheduledBatch {
            request_ids: batch_request_ids.clone(),
            batch: ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: batch_request_ids,
                tokens,
                cu_seqlens,
                kv_handles,
                start_positions,
                params,
                generated_tokens,
            },
            is_final: is_final_vec,
            execution_guards,
        }))
    }

    /// Schedule a decoding batch that respects the remaining token budget.
    fn schedule_decoding_budgeted(
        &self,
        tokens_used: &mut usize,
        decode_tokens_used: &mut usize,
    ) -> Result<Option<ScheduledBatch>> {
        let mut decoding = self
            .decoding_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if decoding.is_empty() {
            return Ok(None);
        }

        let max_batch = self
            .executor
            .max_batch_size(ExecutionPhase::Decode)
            .min(self.config.budget().concurrent_segments);
        let mut batch_request_ids = Vec::new();
        let mut tokens = Vec::new();
        let mut kv_handles = Vec::new();
        let mut cu_seqlens = vec![0];
        let mut start_positions = Vec::new();
        let mut params = Vec::new();
        let mut is_final_vec = Vec::new();
        let mut generated_tokens = Vec::new();
        let mut execution_guards = Vec::new();

        let mut rate_limiter = self.rate_limiter.lock().unwrap_or_else(|e| e.into_inner());
        let mut i = 0;
        while i < decoding.len() && batch_request_ids.len() < max_batch {
            // Each decode step adds 1 token per request
            if !token_budget_allows(*tokens_used, 1, self.max_num_tokens) {
                break;
            }
            if !token_budget_allows(*decode_tokens_used, 1, self.config.budget().decode_tokens) {
                break;
            }

            let req = &decoding[i];

            // Check rate limiter for decode (1 token)
            if self.config.rate_limiter.enabled {
                match rate_limiter.try_acquire(&req.model_id, 1) {
                    RateLimitDecision::Throttled { .. } => {
                        i += 1;
                        continue;
                    }
                    RateLimitDecision::Allowed => {}
                }
            }

            let last_token = req
                .generated_tokens
                .last()
                .or_else(|| req.prompt_tokens.last())
                .copied()
                .ok_or_else(|| {
                    BloomError::SchedulingFailed(format!(
                        "request {} has no token available for decoding",
                        req.id
                    ))
                })?;
            let kv_handle = req.kv_handle.ok_or_else(|| {
                BloomError::SchedulingFailed(format!(
                    "request {} has no KV cache allocation",
                    req.id
                ))
            })?;

            // Pop request and schedule
            let Some(mut req) = decoding.remove(i) else {
                break;
            };
            *tokens_used += 1;
            *decode_tokens_used += 1;
            req.state = RequestState::Decoding {
                current_step: req.generated_tokens.len(),
            };
            req.last_scheduled_at = Some(Instant::now());
            if req.decode_started_at.is_none() {
                req.decode_started_at = Some(Instant::now());
            }

            batch_request_ids.push(req.id.clone());

            tokens.push(last_token);
            kv_handles.push(kv_handle);
            cu_seqlens.push(tokens.len());
            // Real start position: prompt_len + generated_so_far
            start_positions.push(req.prompt_tokens.len() + req.generated_tokens.len());
            params.push(req.params.clone());
            is_final_vec.push(true);
            generated_tokens.push(req.generated_tokens.clone());
            if let Some(guard) = req.execution_guard.clone() {
                execution_guards.push(guard);
            }

            // Update metadata in kv cache pool
            self.kv_pool.update_request_metadata(
                &req.id,
                req.priority,
                req.generated_tokens.len(),
                1.0,
            );

            let mut active = self
                .active_requests
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            active.insert(req.id.clone(), req);
        }

        if batch_request_ids.is_empty() {
            return Ok(None);
        }

        Ok(Some(ScheduledBatch {
            request_ids: batch_request_ids.clone(),
            batch: ExecutionBatch {
                phase: ExecutionPhase::Decode,
                request_ids: batch_request_ids,
                tokens,
                cu_seqlens,
                kv_handles,
                start_positions,
                params,
                generated_tokens,
            },
            is_final: is_final_vec,
            execution_guards,
        }))
    }

    fn process_result(
        &self,
        request_ids: &[RequestId],
        result: BatchResult,
        _phase: ExecutionPhase,
        is_final: &[bool],
    ) -> Result<()> {
        let mut active = self
            .active_requests
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut decoding = self
            .decoding_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        for (i, ((id, &next_token), &final_chunk)) in request_ids
            .iter()
            .zip(result.next_tokens.iter())
            .zip(is_final.iter())
            .enumerate()
        {
            if !final_chunk {
                // For non-final chunked prefill, we don't process next_token or move the request.
                // It remains in active_requests (in Prefill state).
                continue;
            }

            if let Some(mut req) = active.remove(id) {
                req.generated_tokens.push(next_token);
                req.last_accessed = Instant::now();

                // Send generated token to the corresponding client stream
                if let Ok(senders) = self.token_senders.lock()
                    && let Some(sender) = senders.get(id)
                {
                    let _ = sender.send(Ok(next_token));
                }

                // Process extra speculative tokens if any
                if let Some(ref spec_tokens_list) = result.speculative_tokens
                    && let Some(spec_tokens) = spec_tokens_list.get(i)
                {
                    for &tok in spec_tokens {
                        req.generated_tokens.push(tok);
                        if let Ok(senders) = self.token_senders.lock()
                            && let Some(sender) = senders.get(id)
                        {
                            let _ = sender.send(Ok(tok));
                        }
                    }
                }

                let last_token = result
                    .speculative_tokens
                    .as_ref()
                    .and_then(|list| list.get(i))
                    .and_then(|toks| toks.last().copied())
                    .unwrap_or(next_token);

                let is_finished =
                    last_token == 2 || req.generated_tokens.len() >= req.params.max_tokens;

                if is_finished {
                    req.state = RequestState::Finished;
                    if let Some(handle) = req.kv_handle {
                        self.kv_pool.free(handle);
                    }
                    // Clean up sender
                    if let Ok(mut senders) = self.token_senders.lock() {
                        senders.remove(id);
                    }
                } else {
                    req.state = RequestState::Decoding {
                        current_step: req.generated_tokens.len(),
                    };
                    self.kv_pool.update_request_metadata(
                        id,
                        req.priority,
                        req.generated_tokens.len(),
                        1.0,
                    );
                    decoding.push_back(req);
                }
            }
        }
        Ok(())
    }
}

struct ScheduledBatch {
    request_ids: Vec<RequestId>,
    batch: ExecutionBatch,
    is_final: Vec<bool>,
    execution_guards: Vec<ExecutionGuard>,
}

#[cfg(test)]
mod scheduler_test;
