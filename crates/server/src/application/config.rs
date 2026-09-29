//! Resolved runtime options. CLI parsing and environment merging stay at entry points.
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(crate) struct RuntimeConfig {
    pub(crate) backend: String,
    pub(crate) cachemesh_l2_capacity_bytes: usize,
    pub(crate) cachemesh_l3_path: Option<PathBuf>,
    pub(crate) cachemesh_write_through_l3: bool,
    pub(crate) context_size: usize,
    pub(crate) disable_memory_prealloc: bool,
    pub(crate) enable_cachemesh: bool,
    pub(crate) enable_cachemesh_l3: bool,
    pub(crate) enable_chunked_prefill: bool,
    pub(crate) enable_ifb: bool,
    pub(crate) max_concurrent: usize,
    pub(crate) max_num_tokens: usize,
    pub(crate) memory_utilization: f64,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) reserve_memory_bytes: Option<usize>,
    pub(crate) speculative: String,
    pub(crate) long_context_policy: bloomai_engine::LongContextPolicy,
}
