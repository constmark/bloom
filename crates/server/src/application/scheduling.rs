//! Construction and lifetime ownership for optional continuous batching.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Result, anyhow};
use bloomai_core::TokenSchedulingConfig;
use bloomai_engine::{
    BloomKvCachePool, CacheMesh, CacheMeshConfig, FileSystemRemoteCache, InferencePipeline,
    InferenceScheduler, KvCachePool,
};
use tokio_util::sync::CancellationToken;

use super::backend_registry::{div_ceil_usize, manifest_param_usize};
use super::config::RuntimeConfig;
use super::runtime_service::RuntimeService;
use crate::application::memory::RuntimeMemoryPermit;

pub(super) struct SchedulingRuntime {
    pub(super) kv_cache_pool: Option<Arc<BloomKvCachePool>>,
    pub(super) cachemesh: Option<Arc<CacheMesh>>,
    pub(super) scheduler: Option<Arc<InferenceScheduler>>,
    pub(super) memory_reservation: Option<bloomai_engine::MemoryReservation>,
    pub(super) shutdown: CancellationToken,
}

/// Owns every resource whose physical lifetime may extend past logical
/// runtime unload. Declaration order is intentional: scheduler-owned model
/// wrappers are destroyed before the accounting permit, and the source marker
/// is released last even when the worker future unwinds or is aborted.
struct SchedulingWorkerLifetime {
    scheduler: Arc<InferenceScheduler>,
    _runtime_memory_permit: Option<RuntimeMemoryPermit>,
    _runtime_lifetime_marker: Arc<AtomicU64>,
}

pub(super) struct SchedulingRuntimeBuildContext<'a> {
    pub(super) memory_context_size: usize,
    pub(super) memory_estimate: &'a bloomai_engine::MemoryEstimate,
    pub(super) runtime_lifetime_marker: Arc<AtomicU64>,
    pub(super) runtime_memory_permit: Option<RuntimeMemoryPermit>,
}

pub(super) fn build_scheduling_runtime(
    state: &RuntimeService,
    args: &RuntimeConfig,
    manifest: &bloomai_core::ModelManifest,
    pipeline: Arc<InferencePipeline>,
    model_id: &str,
    build: SchedulingRuntimeBuildContext<'_>,
) -> Result<SchedulingRuntime> {
    let SchedulingRuntimeBuildContext {
        memory_context_size,
        memory_estimate,
        runtime_lifetime_marker,
        runtime_memory_permit,
    } = build;
    let shutdown = CancellationToken::new();
    if !args.enable_ifb {
        return Ok(SchedulingRuntime {
            kv_cache_pool: None,
            cachemesh: None,
            scheduler: None,
            memory_reservation: None,
            shutdown,
        });
    }

    let block_size = 16;
    let total_blocks = div_ceil_usize(memory_context_size, block_size).max(1);
    let num_layers = manifest_param_usize(
        manifest,
        &["num_hidden_layers", "num_layers", "block_count"],
        28,
    );
    let num_kv_heads = manifest_param_usize(
        manifest,
        &[
            "num_key_value_heads",
            "num_kv_heads",
            "attention_head_count_kv",
        ],
        8,
    );
    let head_dim = manifest_param_usize(manifest, &["head_dim"], 128);
    let long_context_policy = args.long_context_policy.clone();
    let memory_reservation = if !args.disable_memory_prealloc && memory_estimate.kv_cache_bytes > 0
    {
        Some(bloomai_engine::MemoryReservation::reserve(
            memory_estimate.kv_cache_bytes,
        )?)
    } else {
        None
    };
    let kv_pool = Arc::new(BloomKvCachePool::new(block_size, total_blocks));

    let cachemesh = if args.enable_cachemesh {
        let config = CacheMeshConfig {
            enabled: true,
            namespace: model_id.to_string(),
            l2_capacity_bytes: args.cachemesh_l2_capacity_bytes,
            l3_enabled: args.enable_cachemesh_l3,
            write_through_l3: args.cachemesh_write_through_l3,
        };
        let mesh = if args.enable_cachemesh_l3 {
            let path = args
                .cachemesh_l3_path
                .as_ref()
                .filter(|path| !path.as_os_str().is_empty())
                .ok_or_else(|| {
                    anyhow!(
                        "CacheMesh L3 requires a persistent directory configured with --cachemesh-l3-path"
                    )
                })?;
            let remote: Arc<dyn bloomai_engine::RemoteCacheBackend> =
                Arc::new(FileSystemRemoteCache::new(path)?);
            CacheMesh::with_remote(config, remote)
        } else {
            CacheMesh::new(config)
        };
        Some(Arc::new(mesh))
    } else {
        None
    };

    state.load_progress.store(85, Ordering::Release);
    let executor = bloomai_engine::build_batch_executor(
        pipeline,
        Arc::clone(&kv_pool),
        cachemesh.clone(),
        bloomai_engine::BatchExecutorConfig {
            block_size,
            total_blocks,
            num_layers,
            num_kv_heads,
            head_dim,
            long_context_policy,
        },
    )?;
    let mut scheduling_config = TokenSchedulingConfig {
        max_total_tokens_per_step: args.max_num_tokens,
        ..Default::default()
    };
    scheduling_config.chunked_prefill.enabled = args.enable_chunked_prefill;
    scheduling_config.chunked_prefill.chunk_size = args.prefill_chunk_size;
    let scheduler = Arc::new(InferenceScheduler::with_config(
        executor,
        Arc::clone(&kv_pool) as Arc<dyn KvCachePool>,
        scheduling_config,
    ));

    let worker_lifetime = SchedulingWorkerLifetime {
        scheduler: Arc::clone(&scheduler),
        _runtime_memory_permit: runtime_memory_permit,
        _runtime_lifetime_marker: runtime_lifetime_marker,
    };
    let worker_shutdown = shutdown.clone();
    tokio::spawn(async move {
        tracing::info!("Starting continuous-batching scheduler worker");
        loop {
            tokio::select! {
                _ = worker_shutdown.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_millis(5)) => {
                    if let Err(error) = worker_lifetime.scheduler.step() {
                        tracing::error!(%error, "Scheduler step failed");
                    }
                }
            }
        }
        tracing::info!("Continuous-batching scheduler worker stopped");
    });

    Ok(SchedulingRuntime {
        kv_cache_pool: Some(kv_pool),
        cachemesh,
        scheduler: Some(scheduler),
        memory_reservation,
        shutdown,
    })
}
