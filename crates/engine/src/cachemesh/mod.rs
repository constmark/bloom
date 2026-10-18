//! Multi-tier KV cache offload for Bloom.
//!
//! CacheMesh mirrors the shape of SGLang-style hierarchical cache designs:
//! L1 is the executor-local paged KV cache, L2 is a host-memory backing store,
//! and L3 is a pluggable remote/distributed backing store.
//! [`LayeredBlockCache`] in this module provides the same tiering contract for
//! arbitrary immutable byte blocks without depending on a model executor.

mod layered;
mod store;

pub use layered::{
    FileSystemLayeredCache, LayeredBlockCache, LayeredCacheBackend, LayeredCacheConfig,
    LayeredCacheKey, LayeredCacheMetrics, LayeredCacheTier, LayeredTierMetrics,
};

pub use store::{
    CacheMesh, CacheMeshBlock, CacheMeshConfig, CacheMeshKey, CacheMeshMetrics, CacheMeshSnapshot,
    CacheMeshTier, FileSystemRemoteCache, InMemoryRemoteCache, RemoteCacheBackend, TierMetrics,
};
