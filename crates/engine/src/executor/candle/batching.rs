use super::{CandleKvHook, CandleTextModel, QwenModelWrapper};
use crate::batching::BatchExecutorConfig;
use crate::executor::batch_executor::CandleBatchExecutor;
use crate::scheduler::paged_cache::{PagedAttentionCache, PagedCacheConfig};
use crate::{BloomKvCachePool, CacheMesh, EngineExecutor};
use anyhow::Result;
use anyhow::anyhow;
use bloomai_core::BloomError;
use std::sync::Arc;

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
    } = config;
    // Clone the exact device used by the weights; a fresh GPU device with the
    // same ordinal does not necessarily have the same Candle identity.
    let device = model.device.clone();
    let kv_dim = num_kv_heads * head_dim; // Checked by the capability entry point.

    let request_models = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
        usize,
        Arc<std::sync::Mutex<QwenModelWrapper>>,
    >::new()));
    let request_models_for_free = Arc::clone(&request_models);
    kv_pool.set_on_free(move |handle| {
        request_models_for_free
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&handle);
    });

    let model_for_forward = Arc::clone(&model);
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
                        let model = model_for_forward.reload()?;
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
                            let model = model_for_batch.reload()?;
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
                logits.push(final_token_logits(result)?);
            }
            // Each request contributes one vocabulary vector, regardless of
            // whether its model returns [1, vocab] or [1, sequence, vocab].
            candle_core::Tensor::stack(&logits, 0).map_err(Into::into)
        },
    );

    let paged_cache = Arc::new(PagedAttentionCache::from_pool_and_cachemesh(
        Arc::clone(&kv_pool),
        PagedCacheConfig {
            block_size,
            total_blocks,
            num_layers,
            kv_dim,
            kv_dtype: crate::core::quantization::KvCacheDtype::F16,
            long_context_policy,
        },
        cachemesh.clone(),
    ));
    let executor = Arc::new({
        let base = CandleBatchExecutor::new(forward_fn, device, 4, 32)
            .with_cache(Arc::clone(&paged_cache))
            .with_vocab_and_tokenizer(
                model.vocab_strings.clone(),
                model.eos_token_ids.clone(),
                Some(model.tokenizer.clone()),
            )
            .with_forward_batch_fn(forward_batch_fn);
        if model.uses_streaming_variant() {
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
    Ok(executor)
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
