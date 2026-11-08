use super::*;
use crate::executor::qwen_streaming::dense_batch_tests::{close, model, write_fixture};
use crate::scheduler::{ExecutionBatch, ExecutionPhase, KvCachePool};
use bloomai_core::GenerationParams;
use candle_core::{Device, Tensor};
use std::sync::atomic::Ordering;

#[test]
fn dense_batch_matches_original_candle_qwen2_and_qwen3() -> Result<()> {
    for qwen2 in [true, false] {
        let directory = tempfile::tempdir()?;
        crate::executor::qwen_streaming::dense_batch_tests::write_model_fixture(
            directory.path(),
            qwen2,
        )?;
        let source = super::super::CandleEngine
            .load_text_model(directory.path(), bloomai_core::DeviceKind::Cpu)?;
        let QwenModelWrapper::Streaming(template) = source.reload_with_paged_kv()? else {
            panic!("expected streaming template");
        };
        let mut requests = [template.fork_with_empty_kv(), template.fork_with_empty_kv()];
        let mut originals = [source.reload()?, source.reload()?];
        let mut position = 0;
        for input in [
            Tensor::new(&[[1u32, 2], [8, 9]], &Device::Cpu)?,
            Tensor::new(&[[3u32, 4], [10, 11]], &Device::Cpu)?,
            Tensor::new(&[[5u32], [12]], &Device::Cpu)?,
        ] {
            let mut references = requests.iter_mut().collect::<Vec<_>>();
            let actual = QwenStreamingModelForCausalLM::forward_dense_batch(
                &mut references,
                &input,
                position,
            )?;
            for (index, original) in originals.iter_mut().enumerate() {
                let expected = original.forward(&input.narrow(0, index, 1)?, position)?;
                close(
                    &final_token_logits(actual.narrow(0, index, 1)?)?,
                    &final_token_logits(expected)?,
                );
            }
            position += input.dim(1)?;
        }
    }
    Ok(())
}

#[test]
fn dense_batch_dispatch_uses_one_forward_and_mixed_shapes_use_independent_forwards() -> Result<()> {
    let template = model();
    let models = (0..2)
        .map(|_| {
            Arc::new(Mutex::new(QwenModelWrapper::Streaming(
                template.fork_with_empty_kv(),
            )))
        })
        .collect::<Vec<_>>();
    let mut references = [template.fork_with_empty_kv(), template.fork_with_empty_kv()];
    // Reversed handle order must not reorder output rows.
    let input = Tensor::new(&[1u32, 2, 8, 9], &Device::Cpu)?;
    let calls = template.forward_calls.load(Ordering::Relaxed);
    let actual = forward_request_batch(&input, &[0, 0], &[9, 3], &[0, 2, 4], &models, true)?;
    assert_eq!(template.forward_calls.load(Ordering::Relaxed), calls + 1);
    for (index, reference) in references.iter_mut().enumerate() {
        let expected = reference.forward(&input.narrow(0, index * 2, 2)?.unsqueeze(0)?, 0)?;
        close(&actual.get(index)?, &final_token_logits(expected)?);
    }

    // Different chunk lengths are explicitly scheduling-only batches.
    let input = Tensor::new(&[3u32, 10, 11], &Device::Cpu)?;
    let calls = template.forward_calls.load(Ordering::Relaxed);
    let actual = forward_request_batch(&input, &[2, 2], &[9, 3], &[0, 1, 3], &models, true)?;
    assert_eq!(template.forward_calls.load(Ordering::Relaxed), calls + 2);
    for (index, reference) in references.iter_mut().enumerate() {
        let expected = reference.forward(&input.narrow(0, index, index + 1)?.unsqueeze(0)?, 2)?;
        close(&actual.get(index)?, &final_token_logits(expected)?);
    }
    // Same token length with different cached positions must also fall back.
    let input = Tensor::new(&[4u32, 12], &Device::Cpu)?;
    let calls = template.forward_calls.load(Ordering::Relaxed);
    let actual = forward_request_batch(&input, &[3, 4], &[9, 3], &[0, 1, 2], &models, true)?;
    assert_eq!(template.forward_calls.load(Ordering::Relaxed), calls + 2);
    for (index, reference) in references.iter_mut().enumerate() {
        let expected = reference.forward(&input.narrow(0, index, 1)?.unsqueeze(0)?, index + 3)?;
        close(&actual.get(index)?, &final_token_logits(expected)?);
    }
    Ok(())
}

#[test]
fn dense_batch_invalid_layout_never_runs_and_generic_closure_does_not_advertise_fusion()
-> Result<()> {
    let template = model();
    let models = (0..2)
        .map(|_| {
            Arc::new(Mutex::new(QwenModelWrapper::Streaming(
                template.fork_with_empty_kv(),
            )))
        })
        .collect::<Vec<_>>();
    let input = Tensor::new(&[1u32, 8], &Device::Cpu)?;
    for (positions, handles, offsets) in [
        (vec![0, 0], vec![3, 3], vec![0, 1, 2]),
        (vec![0], vec![3, 4], vec![0, 1, 2]),
        (vec![0, 0], vec![3, 4], vec![0, 0, 2]),
        (vec![0, 0], vec![3, 4], vec![0, 1, 3]),
    ] {
        assert!(
            forward_request_batch(&input, &positions, &handles, &offsets, &models, true).is_err()
        );
    }
    assert_eq!(template.forward_calls.load(Ordering::Relaxed), 0);
    // Disabling the experimental capability disables the execution path too.
    forward_request_batch(&input, &[0, 0], &[3, 4], &[0, 1, 2], &models, false)?;
    assert_eq!(template.forward_calls.load(Ordering::Relaxed), 2);
    let executor = CandleBatchExecutor::new(
        Box::new(|input, _, _| Ok(input.clone())),
        Device::Cpu,
        4,
        32,
    )
    .with_forward_batch_fn(Box::new(|input, _, _, _| Ok(input.clone())));
    assert!(!executor.supports_dense_batch());
    Ok(())
}

#[test]
fn dense_batch_native_builder_executes_and_discard_keeps_other_request_alive() -> Result<()> {
    let directory = tempfile::tempdir()?;
    write_fixture(directory.path())?;
    let model = Arc::new(
        super::super::CandleEngine
            .load_text_model(directory.path(), bloomai_core::DeviceKind::Cpu)?,
    );
    let pool = Arc::new(BloomKvCachePool::new(4, 16));
    let disabled = build(
        Arc::clone(&model),
        Arc::new(BloomKvCachePool::new(4, 16)),
        None,
        BatchExecutorConfig {
            block_size: 4,
            total_blocks: 16,
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 4,
            long_context_policy: crate::LongContextPolicy::Full,
            experimental_dense_batch: false,
        },
    )?;
    assert!(!disabled.supports_dense_batch());
    let executor = build(
        model,
        Arc::clone(&pool),
        None,
        BatchExecutorConfig {
            block_size: 4,
            total_blocks: 16,
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 4,
            long_context_policy: crate::LongContextPolicy::Full,
            experimental_dense_batch: true,
        },
    )?;
    assert!(executor.supports_dense_batch());
    let first = pool.allocate_paged("first", &[1, 2], 8, None)?.handle;
    let second = pool.allocate_paged("second", &[8, 9], 8, None)?.handle;
    let execute =
        |tokens: Vec<u32>, offsets: Vec<usize>, handles: Vec<usize>, positions: Vec<usize>| {
            let count = handles.len();
            executor.execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: handles
                    .iter()
                    .map(|handle| format!("request-{handle}"))
                    .collect(),
                tokens,
                cu_seqlens: offsets,
                kv_handles: handles,
                start_positions: positions,
                params: (0..count)
                    .map(|index| GenerationParams {
                        temperature: if index == 0 { 0.0 } else { 1.1 },
                        top_k: Some(2),
                        top_p: 0.7,
                        seed: Some(index as u64 + 17),
                        logit_bias: Some([(index as u32 + 3, 100.0)].into()),
                        ..Default::default()
                    })
                    .collect(),
                generated_tokens: vec![vec![]; count],
            })
        };
    let initial = execute(
        vec![1, 2, 8, 9],
        vec![0, 2, 4],
        vec![first, second],
        vec![0, 0],
    )?;
    assert_eq!(
        initial.next_tokens,
        vec![3, 4],
        "request sampling parameters must stay independent"
    );
    let next = execute(
        initial.next_tokens,
        vec![0, 1, 2],
        vec![first, second],
        vec![2, 2],
    )?;
    assert_eq!(next.next_tokens, vec![3, 4]);
    pool.discard(first);
    let remaining = execute(vec![next.next_tokens[1]], vec![0, 1], vec![second], vec![3])?;
    assert_eq!(remaining.next_tokens.len(), 1);
    let fresh = pool.allocate_paged("fresh", &[12, 13], 8, None)?.handle;
    execute(vec![12, 13], vec![0, 2], vec![fresh], vec![0])?;
    pool.discard(fresh);
    pool.discard(second);
    assert_eq!(pool.get_metrics().active_blocks, 0);
    assert_eq!(pool.get_metrics().cached_blocks, 0);
    Ok(())
}
