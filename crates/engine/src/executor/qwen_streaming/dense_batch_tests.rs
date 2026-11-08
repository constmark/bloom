use super::*;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Nonzero, nonuniform weights make cross-request KV contamination observable.
fn weights() -> (Config, HashMap<String, Tensor>) {
    let config = Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_position_embeddings: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        head_dim: 4,
        hidden_act: Activation::Silu,
        tie_word_embeddings: false,
        use_sliding_window: false,
        sliding_window: None,
        max_window_layers: 0,
        attention_bias: false,
    };
    let mut tensors = HashMap::new();
    let mut insert = |name: String, shape: Vec<usize>| {
        let seed = name.bytes().map(usize::from).sum::<usize>();
        let data = (0..shape.iter().product::<usize>())
            .map(|index| {
                if name.contains("norm") {
                    1.0
                } else {
                    (((index * 17 + seed) % 113) as f32 - 56.0) / 160.0
                }
            })
            .collect::<Vec<_>>();
        tensors.insert(name, Tensor::from_vec(data, shape, &Device::Cpu).unwrap());
    };
    insert("model.embed_tokens.weight".into(), vec![32, 8]);
    insert("lm_head.weight".into(), vec![32, 8]);
    insert("model.norm.weight".into(), vec![8]);
    for layer in 0..2 {
        for (name, shape) in [
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("self_attn.q_proj.weight", vec![8, 8]),
            ("self_attn.k_proj.weight", vec![4, 8]),
            ("self_attn.v_proj.weight", vec![4, 8]),
            ("self_attn.o_proj.weight", vec![8, 8]),
            ("self_attn.q_norm.weight", vec![4]),
            ("self_attn.k_norm.weight", vec![4]),
            ("mlp.gate_proj.weight", vec![16, 8]),
            ("mlp.up_proj.weight", vec![16, 8]),
            ("mlp.down_proj.weight", vec![8, 16]),
        ] {
            insert(format!("model.layers.{layer}.{name}"), shape);
        }
    }
    (config, tensors)
}

pub(crate) fn model() -> QwenStreamingModelForCausalLM {
    let (config, tensors) = weights();
    let mut model = QwenStreamingModelForCausalLM::new(
        &config,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
        &Device::Cpu,
    )
    .unwrap();
    // Tests do not mutate global KV-format environment variables.
    for layer in &mut model.base.layers {
        layer.self_attn.kv_cache_dtype = KvCacheDType::F16;
    }
    model
}

pub(crate) fn write_fixture(path: &std::path::Path) -> anyhow::Result<()> {
    write_model_fixture(path, false)
}

pub(crate) fn write_model_fixture(path: &std::path::Path, qwen2: bool) -> anyhow::Result<()> {
    let (config, mut tensors) = weights();
    if qwen2 {
        tensors
            .retain(|name, _| !name.ends_with("q_norm.weight") && !name.ends_with("k_norm.weight"));
        for layer in 0..2 {
            for (projection, size) in [("q_proj", 8), ("k_proj", 4), ("v_proj", 4)] {
                tensors.insert(
                    format!("model.layers.{layer}.self_attn.{projection}.bias"),
                    Tensor::from_vec(
                        (0..size)
                            .map(|index| index as f32 / 100.0)
                            .collect::<Vec<_>>(),
                        size,
                        &Device::Cpu,
                    )?,
                );
            }
        }
    }
    candle_core::safetensors::save(&tensors, path.join("model.safetensors"))?;
    std::fs::write(
        path.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "model_type": if qwen2 { "qwen2" } else { "qwen3" },
            "architectures": [if qwen2 { "Qwen2ForCausalLM" } else { "Qwen3ForCausalLM" }],
            "vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size, "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads, "num_key_value_heads": config.num_key_value_heads,
            "max_position_embeddings": config.max_position_embeddings, "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps, "head_dim": config.head_dim,
            "hidden_act": "silu", "tie_word_embeddings": false, "use_sliding_window": false,
            "sliding_window": null, "max_window_layers": 0, "attention_bias": qwen2,
            "eos_token_id": 31
        }))?,
    )?;
    let vocabulary = (0..32u32)
        .map(|index| (format!("t{index}"), index))
        .collect();
    let wordlevel = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("t0".into())
        .build()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let tokenizer = tokenizers::Tokenizer::new(wordlevel);
    tokenizer
        .save(path.join("tokenizer.json"), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}

pub(crate) fn close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
    }
}

fn snapshot(model: &QwenStreamingModelForCausalLM) -> Vec<(Vec<f32>, Vec<f32>)> {
    model
        .base
        .layers
        .iter()
        .map(|layer| {
            let cache = &layer.self_attn.kv_cache;
            (
                cache.k().unwrap().flatten_all().unwrap().to_vec1().unwrap(),
                cache.v().unwrap().flatten_all().unwrap().to_vec1().unwrap(),
            )
        })
        .collect()
}

#[test]
fn dense_batch_chunked_prefill_and_decode_match_independent_requests() -> Result<()> {
    let template = model();
    let mut first = template.fork_with_empty_kv();
    let mut second = template.fork_with_empty_kv();
    let mut reference_first = template.fork_with_empty_kv();
    let mut reference_second = template.fork_with_empty_kv();
    let mut inputs = vec![vec![1u32, 2], vec![8u32, 9]];
    let mut offset = 0;
    for step in 0..5 {
        let input = Tensor::new(inputs.clone(), &Device::Cpu)?;
        let calls = template.forward_calls.load(Ordering::Relaxed);
        let actual = QwenStreamingModelForCausalLM::forward_dense_batch(
            &mut [&mut first, &mut second],
            &input,
            offset,
        )?;
        assert_eq!(template.forward_calls.load(Ordering::Relaxed), calls + 1);
        for (index, reference) in [&mut reference_first, &mut reference_second]
            .into_iter()
            .enumerate()
        {
            let expected = reference.forward(&input.narrow(0, index, 1)?, offset)?;
            close(&actual.narrow(0, index, 1)?, &expected);
        }
        offset += inputs[0].len();
        for (actual, reference) in [(&first, &reference_first), (&second, &reference_second)] {
            for layer in 0..2 {
                let (ak, av) = actual.extract_kv_tensor(layer, 0, offset)?;
                let (rk, rv) = reference.extract_kv_tensor(layer, 0, offset)?;
                close(&ak, &rk);
                close(&av, &rv);
            }
        }
        inputs = if step == 0 {
            vec![vec![3, 4], vec![10, 11]]
        } else {
            actual
                .squeeze(1)?
                .argmax(1)?
                .to_vec1::<u32>()?
                .into_iter()
                .map(|token| vec![token])
                .collect()
        };
    }
    assert_ne!(snapshot(&first), snapshot(&second));
    for request in [&first, &second] {
        for layer in &request.base.layers {
            for tensor in [
                layer.self_attn.kv_cache.k().unwrap(),
                layer.self_attn.kv_cache.v().unwrap(),
            ] {
                let (storage, layout) = tensor.storage_and_layout();
                let candle_core::Storage::Cpu(candle_core::CpuStorage::F32(values)) = &*storage
                else {
                    panic!("expected CPU F32 cache");
                };
                assert_eq!(
                    values.len(),
                    tensor.elem_count(),
                    "private cache must not clone other batch rows"
                );
                assert_eq!(layout.start_offset(), 0);
            }
        }
    }
    let (first_storage, _) = first.base.layers[0]
        .self_attn
        .kv_cache
        .k()
        .unwrap()
        .storage_and_layout();
    let (second_storage, _) = second.base.layers[0]
        .self_attn
        .kv_cache
        .k()
        .unwrap()
        .storage_and_layout();
    assert!(!std::ptr::eq(&*first_storage, &*second_storage));
    drop(first_storage);
    drop(second_storage);

    // A cancellation releases only that request's complete private caches.
    first.clear_kv_cache();
    assert!(
        first
            .base
            .layers
            .iter()
            .all(|layer| layer.self_attn.kv_cache.is_empty())
    );
    let input = Tensor::new(&[[15u32]], &Device::Cpu)?;
    close(
        &second.forward(&input, offset)?,
        &reference_second.forward(&input, offset)?,
    );
    close(
        &first.forward(&input, 0)?,
        &template.fork_with_empty_kv().forward(&input, 0)?,
    );
    Ok(())
}

#[test]
fn dense_batch_failed_forward_never_commits_partial_kv() -> Result<()> {
    let template = model();
    let mut first = template.fork_with_empty_kv();
    let mut second = template.fork_with_empty_kv();
    QwenStreamingModelForCausalLM::forward_dense_batch(
        &mut [&mut first, &mut second],
        &Tensor::new(&[[1u32, 2], [8, 9]], &Device::Cpu)?,
        0,
    )?;
    let before = [snapshot(&first), snapshot(&second)];
    // Fail after every attention layer has appended to the temporary cache.
    first.lm_head = StreamingLinear::new(Tensor::zeros((32, 7), DType::F32, &Device::Cpu)?, None);
    let error = QwenStreamingModelForCausalLM::forward_dense_batch(
        &mut [&mut first, &mut second],
        &Tensor::new(&[[3u32], [10]], &Device::Cpu)?,
        2,
    );
    assert!(error.is_err());
    assert_eq!(before, [snapshot(&first), snapshot(&second)]);
    // A mismatched offset must also leave the existing histories untouched.
    assert!(
        QwenStreamingModelForCausalLM::forward_dense_batch(
            &mut [&mut first, &mut second],
            &Tensor::new(&[[3u32], [10]], &Device::Cpu)?,
            1,
        )
        .is_err()
    );
    assert_eq!(before, [snapshot(&first), snapshot(&second)]);
    Ok(())
}

#[test]
fn dense_batch_rejects_unrelated_weights_and_unsupported_kv() -> Result<()> {
    let mut first = model();
    let mut unrelated = model();
    let input = Tensor::new(&[[1u32], [2]], &Device::Cpu)?;
    assert!(
        QwenStreamingModelForCausalLM::forward_dense_batch(
            &mut [&mut first, &mut unrelated],
            &input,
            0
        )
        .is_err()
    );
    assert_eq!(first.forward_calls.load(Ordering::Relaxed), 0);
    let mut second = first.fork_with_empty_kv();
    second.base.layers[0].self_attn.kv_cache_dtype = KvCacheDType::Int8;
    assert!(!second.supports_dense_batch());
    assert!(
        QwenStreamingModelForCausalLM::forward_dense_batch(
            &mut [&mut first, &mut second],
            &input,
            0
        )
        .is_err()
    );
    assert!(first.base.layers[0].self_attn.kv_cache.is_empty());
    Ok(())
}

#[test]
fn paged_kv_extraction_does_not_retain_a_complete_historical_cache() {
    let mut model = model();
    model
        .forward(
            &Tensor::new(&[[1u32, 2, 3, 4, 5, 6]], &Device::Cpu).unwrap(),
            0,
        )
        .unwrap();
    let (keys, values) = model.extract_kv_tensor(0, 2, 2).unwrap();
    for tensor in [keys, values] {
        let (storage, layout) = tensor.storage_and_layout();
        let candle_core::Storage::Cpu(candle_core::CpuStorage::F32(values)) = &*storage else {
            panic!("expected CPU F32 cache");
        };
        assert_eq!(values.len(), tensor.elem_count());
        assert_eq!(layout.start_offset(), 0);
        assert_eq!(tensor.dim(2).unwrap(), 2);
    }
}
