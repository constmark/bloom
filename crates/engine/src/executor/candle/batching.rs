use super::{CandleKvHook, CandleTextModel, QwenModelWrapper};
use crate::batching::BatchExecutorConfig;
use crate::core::quantization::KvCacheDtype;
use crate::executor::batch_executor::CandleBatchExecutor;
use crate::executor::qwen_streaming::QwenStreamingModelForCausalLM;
use crate::scheduler::paged_cache::{PagedAttentionCache, PagedCacheConfig};
use crate::{BloomKvCachePool, CacheMesh, EngineExecutor};
use anyhow::Result;
use anyhow::anyhow;
use bloomai_core::BloomError;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod dense_batch_tests;

pub(super) fn memory_plan(model: &CandleTextModel) -> Result<crate::batching::DenseMemoryPlan> {
    anyhow::ensure!(
        model.device.is_cpu()
            && !candle_core::utils::has_accelerate()
            && !candle_core::utils::has_mkl(),
        "dense memory admission currently supports audited CPU/F32 GEMM, not opaque GPU/BLAS workspaces"
    );
    anyhow::ensure!(
        matches!(
            model.model_type,
            super::ModelType::Qwen2 | super::ModelType::Qwen3
        ) && model.gguf_file_path.is_none()
            && model
                .config
                .get("quantization_config")
                .is_none_or(serde_json::Value::is_null),
        "dense memory admission requires unquantized Safetensors Qwen2/Qwen3"
    );
    anyhow::ensure!(
        !KvCacheDtype::from_env().needs_dequant()
            && std::env::var("BLOOM_KV_CACHE_DTYPE")
                .unwrap_or_default()
                .to_lowercase()
                != "int4",
        "dense memory admission requires unquantized KV"
    );
    anyhow::ensure!(
        model.config["hidden_act"].as_str() == Some("silu"),
        "dense memory admission currently audits the SiLU Qwen feed-forward path"
    );
    model.weight_artifacts.verify()?;
    let paths = crate::core::manifest::resolve_hf_safetensors_files(&model.model_path)?;
    anyhow::ensure!(!paths.is_empty(), "dense model has no safetensors files");
    let file_bytes = paths.iter().try_fold(0usize, |total, path| {
        let bytes = usize::try_from(std::fs::metadata(path)?.len())?;
        total
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("weight file size overflows"))
    })?;
    let vocabulary_bytes = model
        .vocab_strings
        .iter()
        .try_fold(0usize, |total, token| {
            total
                .checked_add(token.len() + std::mem::size_of::<String>())
                .ok_or_else(|| anyhow!("vocabulary size overflows"))
        })?;
    crate::batching::DenseMemoryPlan::cpu_qwen(&model.config, file_bytes, vocabulary_bytes)
}

pub(super) fn build(
    model: Arc<CandleTextModel>,
    kv_pool: Arc<BloomKvCachePool>,
    cachemesh: Option<Arc<CacheMesh>>,
    config: BatchExecutorConfig,
) -> Result<Arc<dyn EngineExecutor>> {
    let BatchExecutorConfig {
        block_size,
        total_blocks,
        num_layers,
        num_kv_heads,
        head_dim,
        long_context_policy,
        experimental_dense_batch: experimental_dense,
    } = config;
    // Clone the exact device used by the weights; a fresh GPU device with the
    // same ordinal does not necessarily have the same Candle identity.
    let device = model.device.clone();
    let kv_dim = num_kv_heads * head_dim; // Checked by the capability entry point.
    let use_paged_kv = cachemesh.is_some();
    if use_paged_kv
        && (!matches!(
            model.model_type,
            super::ModelType::Qwen2 | super::ModelType::Qwen3 | super::ModelType::Qwen2Moe
        ) || crate::core::manifest::resolve_hf_safetensors_files(&model.model_path)?.is_empty())
    {
        anyhow::bail!(
            "CacheMesh requires a native Qwen2/Qwen3/Qwen2-MoE Safetensors model with an extractable KV cache"
        );
    }

    // A streaming model keeps immutable weights in shared tensors and
    // owns only its KV state mutably.  Forking one empty-KV template per
    // request shares those weights instead of reopening every safetensors
    // shard for each admitted sequence.  This is the same residency model a
    // single-node serving engine needs for continuous batching.
    let native_dense = experimental_dense
        && matches!(
            model.model_type,
            super::ModelType::Qwen2 | super::ModelType::Qwen3
        )
        && !crate::core::manifest::resolve_hf_safetensors_files(&model.model_path)?.is_empty();
    let streaming_template = if model.uses_streaming_variant() || use_paged_kv || native_dense {
        model.weight_artifacts.verify()?;
        let loaded = {
            let source = model
                .model
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            match source.as_ref() {
                Some(QwenModelWrapper::Streaming(streaming)) => {
                    Some(streaming.fork_with_empty_kv())
                }
                // The startup verification wrapper may be a regular Candle
                // model even when IFB/CacheMesh later selects the streaming
                // adapter. Reload the correct adapter below in that case.
                Some(_) | None => None,
            }
        };
        let template = match loaded {
            Some(template) => template,
            // Callers can release the startup wrapper before constructing the
            // executor. In that case reload once here, before accepting work.
            None => match model.reload_with_paged_kv()? {
                QwenModelWrapper::Streaming(streaming) => streaming.fork_with_empty_kv(),
                _ => anyhow::bail!("native paged-KV batching requires a Streaming wrapper"),
            },
        };
        model.weight_artifacts.verify()?;
        tracing::info!("Sharing native streaming weights across independent request KV caches");
        Some(Arc::new(template))
    } else {
        None
    };

    let has_native_kv = streaming_template.is_some();
    let dense_batch = experimental_dense
        && streaming_template
            .as_ref()
            .is_some_and(|model| model.supports_dense_batch());
    let request_models = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
        usize,
        Arc<std::sync::Mutex<QwenModelWrapper>>,
    >::new()));
    let request_models_for_free = Arc::clone(&request_models);
    let restored_prefix_handles = Arc::new(Mutex::new(HashSet::new()));
    let restored_prefix_handles_for_free = Arc::clone(&restored_prefix_handles);

    let model_for_forward = Arc::clone(&model);
    let template_for_forward = streaming_template.clone();
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
                        let model = new_request_model(
                            &model_for_forward,
                            template_for_forward.as_deref(),
                            use_paged_kv,
                        )?;
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

    let model_for_batch = Arc::clone(&model);
    let template_for_batch = streaming_template.clone();
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
            validate_batch_layout(input_ids, start_positions, kv_handles, cu_seqlens)?;
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
                            let model = new_request_model(
                                &model_for_batch,
                                template_for_batch.as_deref(),
                                use_paged_kv,
                            )?;
                            entry.insert(Arc::new(std::sync::Mutex::new(model)))
                        }
                    };
                    models_to_run.push(Arc::clone(model));
                }
            }
            forward_request_batch(
                input_ids,
                start_positions,
                kv_handles,
                cu_seqlens,
                &models_to_run,
                dense_batch,
            )
        },
    );

    let model_for_prepare = Arc::clone(&model);
    let template_for_prepare = streaming_template;
    let request_models_for_prepare = Arc::clone(&request_models);
    let prepare_request = Box::new(move |handle| -> Result<()> {
        let mut models = request_models_for_prepare
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let std::collections::hash_map::Entry::Vacant(entry) = models.entry(handle) {
            let model = new_request_model(
                &model_for_prepare,
                template_for_prepare.as_deref(),
                use_paged_kv,
            )?;
            entry.insert(Arc::new(std::sync::Mutex::new(model)));
        }
        Ok(())
    });

    let paged_cache = Arc::new(PagedAttentionCache::from_pool_and_cachemesh(
        Arc::clone(&kv_pool),
        PagedCacheConfig {
            block_size,
            total_blocks,
            num_layers,
            kv_dim,
            kv_dtype: KvCacheDtype::from_env(),
            long_context_policy,
        },
        cachemesh.clone(),
    ));
    let executor = Arc::new({
        let base = CandleBatchExecutor::new(forward_fn, device, 4, 32)
            .with_cache(Arc::clone(&paged_cache))
            .with_prepare_request(prepare_request)
            .with_restored_prefix_handles(Arc::clone(&restored_prefix_handles))
            .with_shared_vocab_and_tokenizer(
                model.vocab_strings.clone(),
                model.eos_token_ids.clone(),
                Some(Arc::clone(&model.tokenizer)),
            );
        let base = if dense_batch {
            base.with_dense_forward_batch_fn(forward_batch_fn)
        } else {
            base.with_forward_batch_fn(forward_batch_fn)
        };
        if has_native_kv {
            let hook = Arc::new(CandleKvHook::new(
                Arc::clone(&request_models),
                num_layers,
                num_kv_heads,
                head_dim,
            ));
            base.with_kv_hook(hook as Arc<dyn crate::scheduler::kv_hook::KvHook>)
        } else {
            base
        }
    });

    // KV handles are recycled by the pool. Clear every per-request state store
    // when a handle is released so stale speculative context (or a loaded
    // draft model) cannot leak into the next request that reuses that handle.
    let executor_for_free = Arc::downgrade(&executor);
    kv_pool.set_on_free(move |handle| {
        request_models_for_free
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&handle);
        restored_prefix_handles_for_free
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&handle);
        if let Some(executor) = executor_for_free.upgrade() {
            executor.clear_request_state(handle);
        }
    });

    Ok(executor)
}

fn validate_batch_layout(
    input: &candle_core::Tensor,
    positions: &[usize],
    handles: &[usize],
    offsets: &[usize],
) -> Result<()> {
    if positions.len() != handles.len()
        || offsets.len() != handles.len() + 1
        || offsets.first() != Some(&0)
        || offsets.last().copied() != Some(input.dim(0)?)
        || offsets.windows(2).any(|pair| pair[0] >= pair[1])
        || handles.iter().copied().collect::<HashSet<_>>().len() != handles.len()
    {
        anyhow::bail!("invalid continuous-batching layout or duplicate KV handles");
    }
    Ok(())
}

fn forward_request_batch(
    input_ids: &candle_core::Tensor,
    start_positions: &[usize],
    kv_handles: &[usize],
    cu_seqlens: &[usize],
    models: &[Arc<Mutex<QwenModelWrapper>>],
    dense_batch: bool,
) -> Result<candle_core::Tensor> {
    validate_batch_layout(input_ids, start_positions, kv_handles, cu_seqlens)?;
    let batch_size = models.len();
    if batch_size != kv_handles.len() {
        anyhow::bail!("continuous-batching model count mismatch");
    }
    if dense_batch
        && batch_size > 1
        && start_positions
            .iter()
            .all(|position| *position == start_positions[0])
        && cu_seqlens
            .windows(2)
            .all(|pair| pair[1] - pair[0] == cu_seqlens[1])
    {
        // Deterministic lock ordering also protects callers outside the
        // executor's serialization lock. Duplicate handles are rejected above.
        let mut order = (0..batch_size).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&index| kv_handles[index]);
        let mut guards = order
            .iter()
            .map(|&index| {
                (
                    index,
                    models[index]
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()),
                )
            })
            .collect::<Vec<_>>();
        guards.sort_unstable_by_key(|(index, _)| *index);
        let mut streaming = guards
            .iter_mut()
            .filter_map(|(_, model)| match &mut **model {
                QwenModelWrapper::Streaming(model) if model.supports_dense_batch() => Some(model),
                _ => None,
            })
            .collect::<Vec<_>>();
        if streaming.len() == batch_size {
            let input = input_ids.reshape((batch_size, cu_seqlens[1]))?;
            let logits = QwenStreamingModelForCausalLM::forward_dense_batch(
                &mut streaming,
                &input,
                start_positions[0],
            )?;
            tracing::debug!(
                batch_size,
                tokens_per_request = cu_seqlens[1],
                "executed one dense model forward"
            );
            return Ok(logits.squeeze(1)?);
        }
    }

    let run_request = |index: usize,
                       model: &Arc<std::sync::Mutex<QwenModelWrapper>>|
     -> Result<candle_core::Tensor> {
        let start = cu_seqlens[index];
        let end = cu_seqlens[index + 1];
        let sequence_len = end
            .checked_sub(start)
            .ok_or_else(|| anyhow!("continuous-batching sequence offsets are not ordered"))?;
        let start_pos = start_positions.get(index).copied().unwrap_or(0);
        let request_input = input_ids.narrow(0, start, sequence_len)?.unsqueeze(0)?;
        model
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .forward(&request_input, start_pos)
    };

    // Candle does not expose a ragged Qwen forward kernel here. CPU
    // requests can still make real progress concurrently because each
    // handle owns a private KV state while immutable streaming weights
    // are shared. Keep the GPU path serialized: parallel CUDA launches
    // would compete for one admitted context and generally regress
    // latency without a dedicated stream/graph scheduler.
    let request_logits = if input_ids.device().is_cpu()
        && batch_size > 1
        && batch_size
            <= std::thread::available_parallelism()
                .map(|parallelism| parallelism.get().saturating_mul(2))
                .unwrap_or(2)
    {
        let slots = (0..batch_size)
            .map(|_| Mutex::new(None))
            .collect::<Vec<Mutex<Option<Result<candle_core::Tensor>>>>>();
        std::thread::scope(|scope| {
            for (index, model) in models.iter().enumerate() {
                let slot = &slots[index];
                scope.spawn(move || {
                    let result = run_request(index, model);
                    *slot.lock().unwrap_or_else(|error| error.into_inner()) = Some(result);
                });
            }
        });
        slots
            .into_iter()
            .map(|slot| {
                slot.into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .ok_or_else(|| anyhow!("CPU batch worker did not finish"))?
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        models
            .iter()
            .enumerate()
            .map(|(index, model)| run_request(index, model))
            .collect::<Result<Vec<_>>>()?
    };

    let mut logits = Vec::with_capacity(batch_size);
    for result in request_logits {
        logits.push(final_token_logits(result)?);
    }
    // Each request contributes one vocabulary vector, regardless of
    // whether its model returns [1, vocab] or [1, sequence, vocab].
    candle_core::Tensor::stack(&logits, 0).map_err(Into::into)
}

fn new_request_model(
    model: &CandleTextModel,
    moe_template: Option<&QwenStreamingModelForCausalLM>,
    use_paged_kv: bool,
) -> Result<QwenModelWrapper> {
    if let Some(template) = moe_template {
        // Preserve the reload path's file identity checks for each request,
        // even though immutable in-memory weights are now shared.
        model.weight_artifacts.verify()?;
        let wrapper = QwenModelWrapper::Streaming(template.fork_with_empty_kv());
        model.weight_artifacts.verify()?;
        Ok(wrapper)
    } else if use_paged_kv {
        model.reload_with_paged_kv()
    } else {
        model.reload()
    }
}

pub(super) fn final_token_logits(logits: candle_core::Tensor) -> Result<candle_core::Tensor> {
    match logits.dims() {
        [1, vocab] if *vocab > 0 => logits.squeeze(0).map_err(Into::into),
        [1, sequence, vocab] if *sequence > 0 && *vocab > 0 => {
            logits.get(0)?.get(sequence - 1).map_err(Into::into)
        }
        _ => Err(anyhow!(
            "batch model must return [1, vocab] or [1, sequence, vocab] logits"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::kv_hook::KvHook;
    #[test]
    fn native_model_logits_keep_one_vocabulary_vector_per_request() {
        use candle_core::{Device, Tensor};
        let matrix = Tensor::new(&[[1f32, 2., 3.]], &Device::Cpu).unwrap();
        let sequence = Tensor::new(&[[[9f32, 8., 7.], [4., 5., 6.]]], &Device::Cpu).unwrap();
        let rows = [
            final_token_logits(matrix).unwrap(),
            final_token_logits(sequence).unwrap(),
        ];
        let batch = Tensor::stack(&rows, 0).unwrap();
        assert_eq!(
            batch.to_vec2::<f32>().unwrap(),
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        );
        let invalid = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu).unwrap();
        assert!(final_token_logits(invalid).is_err());
    }

    #[test]
    fn test_candle_kv_hook_lookup_error() {
        let request_models = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let hook = CandleKvHook::new(Arc::clone(&request_models), 28, 8, 128);
        assert_eq!(hook.num_layers(), 28);
        assert_eq!(hook.kv_dim(), 1024);
        let res = hook.extract_kv(999, 0, 0, 10);
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("No model wrapper found for handle 999")
        );
    }
}
