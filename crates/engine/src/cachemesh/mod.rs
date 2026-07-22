//! Multi-tier KV cache offload for Bloom.
//!
//! CacheMesh mirrors the shape of SGLang-style hierarchical cache designs:
//! L1 is the executor-local paged KV cache, L2 is a host-memory backing store,
//! and L3 is a pluggable remote/distributed backing store.

mod store;

pub use store::{
    CacheMesh, CacheMeshBlock, CacheMeshConfig, CacheMeshKey, CacheMeshMetrics, CacheMeshSnapshot,
    CacheMeshTier, FileSystemRemoteCache, InMemoryRemoteCache, RemoteCacheBackend, TierMetrics,
};
