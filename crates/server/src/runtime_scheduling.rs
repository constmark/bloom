//! Construction and lifetime ownership for optional continuous batching.

use super::*;

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
    state: &ServerState,
    args: &Args,
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
    let kv_dim = num_kv_heads.saturating_mul(head_dim).max(1);
    let long_context_policy = build_long_context_policy(args)?;
    let memory_reservation = if !args.disable_memory_prealloc && memory_estimate.kv_cache_bytes > 0
    {
        Some(bloomai_engine::MemoryReservation::reserve(
            memory_estimate.kv_cache_bytes,
        )?)
    } else {
        None
    };
    let kv_pool = Arc::new(BloomKvCachePool::new(block_size, total_blocks));

    let device = build_ifb_scheduler_device(&pipeline)?;

    let request_models = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
        usize,
        Arc<std::sync::Mutex<bloomai_engine::executor::candle::QwenModelWrapper>>,
    >::new()));
    let request_models_for_free = Arc::clone(&request_models);
    kv_pool.set_on_free(move |handle| {
        if let Ok(mut models) = request_models_for_free.lock() {
            models.remove(&handle);
        }
    });

    let pipeline_for_forward = Arc::clone(&pipeline);
    let request_models_for_forward = Arc::clone(&request_models);
    let forward_fn = Box::new(
        move |input_ids: &candle_core::Tensor,
              start_pos: usize,
              kv_handle: Option<usize>|
              -> Result<candle_core::Tensor> {
            let handle = kv_handle.ok_or_else(|| {
                BloomError::Engine("scheduler request is missing its KV cache handle".into())
            })?;
            let model = {
                let mut models = request_models_for_forward
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                use std::collections::hash_map::Entry;
                Arc::clone(match models.entry(handle) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => {
                        let wrapper = pipeline_for_forward.model().create_wrapper()?;
                        let model = *wrapper
                            .downcast::<bloomai_engine::executor::candle::QwenModelWrapper>()
                            .map_err(|_| {
                                BloomError::Engine("failed to downcast model wrapper".into())
                            })?;
                        entry.insert(Arc::new(std::sync::Mutex::new(model)))
                    }
                })
            };

            model
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .forward(input_ids, start_pos)
        },
    );

    let pipeline_for_batch = Arc::clone(&pipeline);
    let request_models_for_batch = Arc::clone(&request_models);
    let forward_batch_fn = Box::new(
        move |input_ids: &candle_core::Tensor,
              start_positions: &[usize],
              kv_handles: &[usize],
              cu_seqlens: &[usize]|
              -> Result<candle_core::Tensor> {
            let batch_size = kv_handles.len();
            if batch_size == 0 {
                return Ok(candle_core::Tensor::zeros(
                    (0, 0),
                    candle_core::DType::F32,
                    input_ids.device(),
                )?);
            }
            if cu_seqlens.len() != batch_size + 1 {
                return Err(anyhow!("invalid continuous-batching sequence offsets"));
            }
            let mut models_to_run = Vec::with_capacity(batch_size);
            {
                let mut models = request_models_for_batch
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                use std::collections::hash_map::Entry;
                for &handle in kv_handles {
                    let model = match models.entry(handle) {
                        Entry::Occupied(entry) => entry.into_mut(),
                        Entry::Vacant(entry) => {
                            let wrapper = pipeline_for_batch.model().create_wrapper()?;
                            let model = *wrapper
                                .downcast::<bloomai_engine::executor::candle::QwenModelWrapper>()
                                .map_err(|_| {
                                    BloomError::Engine("failed to downcast model wrapper".into())
                                })?;
                            entry.insert(Arc::new(std::sync::Mutex::new(model)))
                        }
                    };
                    models_to_run.push(Arc::clone(model));
                }
            }
            let mut logits = Vec::with_capacity(batch_size);
            for (index, model) in models_to_run.iter().enumerate() {
                let start = cu_seqlens[index];
                let end = cu_seqlens[index + 1];
                let sequence_len = end.checked_sub(start).ok_or_else(|| {
                    anyhow!("continuous-batching sequence offsets are not ordered")
                })?;
                let start_pos = start_positions.get(index).copied().unwrap_or(0);
                let request_input = input_ids.narrow(0, start, sequence_len)?.unsqueeze(0)?;
                let result = model
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .forward(&request_input, start_pos)?;
                logits.push(result.squeeze(0)?);
            }
            candle_core::Tensor::cat(&logits, 0).map_err(Into::into)
        },
    );

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
    let paged_cache = Arc::new(PagedAttentionCache::from_pool_and_cachemesh(
        Arc::clone(&kv_pool),
        PagedCacheConfig {
            block_size,
            total_blocks,
            num_layers,
            kv_dim,
            kv_dtype: bloomai_engine::core::quantization::KvCacheDtype::F16,
            long_context_policy,
        },
        cachemesh.clone(),
    ));
    let executor = Arc::new({
        let model = pipeline.model();
        let base = CandleBatchExecutor::new(forward_fn, device, 4, 32)
            .with_cache(Arc::clone(&paged_cache))
            .with_vocab_and_tokenizer(
                model.vocab_strings().to_vec(),
                model.eos_token_ids().to_vec(),
                model.tokenizer().cloned(),
            )
            .with_forward_batch_fn(forward_batch_fn);
        if model.supports_paged_kv() {
            let hook = Arc::new(ServerKvHook::new(
                Arc::clone(&request_models),
                num_layers,
                num_kv_heads,
                head_dim,
            ));
            base.with_kv_hook(hook as Arc<dyn bloomai_engine::scheduler::kv_hook::KvHook>)
        } else {
            base
        }
    });
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

fn build_ifb_scheduler_device(pipeline: &InferencePipeline) -> Result<candle_core::Device> {
    #[cfg(feature = "candle-engine")]
    {
        let device = pipeline.model().candle_device().ok_or_else(|| {
            anyhow!(
                "in-flight batching requires the exact Candle device identity owned by the loaded model"
            )
        })?;
        let actual_kind = if device.is_cpu() {
            DeviceKind::Cpu
        } else {
            DeviceKind::Gpu
        };
        if actual_kind != pipeline.device() {
            return Err(anyhow!(
                "the IFB scheduler device does not match the admitted pipeline device"
            ));
        }
        Ok(device)
    }
    #[cfg(not(feature = "candle-engine"))]
    {
        let _ = pipeline;
        Err(anyhow!(
            "in-flight batching requires a server built with Candle support"
        ))
    }
}
