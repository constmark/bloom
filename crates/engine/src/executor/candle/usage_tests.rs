use super::*;
use crate::io::OutputChunk;

/// Deterministic next-token cycle 1 -> 2 -> EOS(31) -> 0 -> 1. The prompt
/// contains that cycle so ngram speculation proposes EOS and a valid tail.
fn cycle_fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    crate::executor::qwen_streaming::dense_batch_tests::write_fixture(directory.path()).unwrap();
    let weights = directory.path().join("model.safetensors");
    let original = candle_core::safetensors::load(&weights, &Device::Cpu).unwrap();
    let mut tensors = std::collections::HashMap::new();
    for (name, tensor) in original {
        let value = if name.contains("norm") {
            Tensor::ones(tensor.shape(), DType::F32, &Device::Cpu).unwrap()
        } else {
            Tensor::zeros(tensor.shape(), DType::F32, &Device::Cpu).unwrap()
        };
        tensors.insert(name, value);
    }
    let mut embedding = vec![0f32; 32 * 8];
    for (id, dimension) in [(0, 0), (1, 1), (2, 2), (31, 3)] {
        embedding[id * 8 + dimension] = 1.;
    }
    let mut head = vec![0f32; 32 * 8];
    for (next, dimension) in [(1, 0), (2, 1), (31, 2), (0, 3)] {
        head[next * 8 + dimension] = 1.;
    }
    tensors.insert(
        "model.embed_tokens.weight".into(),
        Tensor::from_vec(embedding, (32, 8), &Device::Cpu).unwrap(),
    );
    tensors.insert(
        "lm_head.weight".into(),
        Tensor::from_vec(head, (32, 8), &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&tensors, &weights).unwrap();
    directory
}

#[test]
fn accepted_speculative_eos_stops_output_usage_and_invalidates_verifier_tail() {
    let directory = cycle_fixture();
    let mut model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    model.residency_strategy = ResidencyStrategy::Resident;
    let params = GenerationParams {
        max_tokens: 6,
        temperature: 0.0,
        ..Default::default()
    };
    let prompt = vec![1, 2, 31, 0, 1];
    let mut events = Vec::new();
    model
        .infer_tokens_stream_with_speculation(
            prompt.clone(),
            &params,
            &mut |event| {
                events.push(event);
                Ok(())
            },
            SpeculativeMode::NGram {
                ngram_order: 1,
                num_speculative: 3,
            },
        )
        .unwrap();
    let text: String = events
        .iter()
        .filter_map(|event| match event {
            OutputChunk::TextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        text, "t2",
        "EOS and accepted drafts after EOS must never be emitted"
    );
    let usage = events
        .iter()
        .filter_map(|event| match event {
            OutputChunk::TokenUsage {
                input_tokens,
                output_tokens,
            } => Some((*input_tokens, *output_tokens)),
            _ => None,
        })
        .next_back();
    assert_eq!(
        usage,
        Some((5, 2)),
        "one visible token plus EOS are generated"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        OutputChunk::Metrics {
            speculative_draft_tokens: Some(3),
            speculative_accepted_tokens: Some(1),
            ..
        }
    )));
    assert!(
        model.current_prefix.lock().unwrap().is_empty(),
        "verifier tail must not be reused"
    );
    let mut next_text = String::new();
    model
        .infer_tokens_stream_with_speculation(
            prompt,
            &params,
            &mut |event| {
                if let OutputChunk::TextDelta(text) = event {
                    next_text.push_str(&text);
                }
                Ok(())
            },
            SpeculativeMode::None,
        )
        .unwrap();
    assert_eq!(
        next_text, "t2",
        "the next request must start with a clean KV cache"
    );
}

#[test]
fn sampling_adjusts_logits_before_temperature_without_clipping_or_underflow() {
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use std::collections::BTreeMap;

    // Top-k=1 makes the regression deterministic, even at nonzero temperature.
    for (temperature, logits, history, presence, frequency, bias, expected) in [
        (0.1, [2.0, 1.0], vec![0], 2.0, 0.0, None, 1),
        (2.0, [3.0, 0.0], vec![], 0.0, 0.0, Some((1, 2.0)), 0),
        (1.0, [40.0, 0.0], vec![], 0.0, 0.0, Some((1, 100.0)), 1),
        (0.01, [50.0, 0.0], vec![], 0.0, 0.0, Some((1, 100.0)), 1),
        (1.0, [50.0, 0.0], vec![0; 30], 0.0, 2.0, None, 1),
        (1.0, [50.0, 0.0], vec![], 0.0, 0.0, Some((0, -100.0)), 1),
    ] {
        let params = GenerationParams {
            temperature,
            top_k: Some(1),
            top_p: 1.0,
            presence_penalty: presence,
            frequency_penalty: frequency,
            logit_bias: bias.map(|(id, value)| BTreeMap::from([(id, value)])),
            ..Default::default()
        };
        let mut sampler = LogitsProcessor::from_sampling(42, Sampling::TopK { k: 1, temperature });
        let logits = Tensor::new(&logits, &Device::Cpu).unwrap();
        assert_eq!(
            CandleTextModel::sample_with_penalties(&mut sampler, &logits, &params, &history)
                .unwrap(),
            expected
        );
    }
}

#[test]
fn speculative_ngram_preserves_bias_by_using_the_regular_sampler() {
    let directory = cycle_fixture();
    let model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    let params = GenerationParams {
        max_tokens: 6,
        temperature: 0.0,
        logit_bias: Some(std::collections::BTreeMap::from([(31, -100.0)])),
        ..Default::default()
    };
    let mut outputs = Vec::new();
    for mode in [
        SpeculativeMode::None,
        SpeculativeMode::NGram {
            ngram_order: 1,
            num_speculative: 3,
        },
    ] {
        let mut text = String::new();
        model
            .infer_tokens_stream_with_speculation(
                vec![1, 2, 31, 0, 1],
                &params,
                &mut |event| {
                    if let OutputChunk::TextDelta(delta) = event {
                        text.push_str(&delta);
                    }
                    Ok(())
                },
                mode,
            )
            .unwrap();
        outputs.push(text);
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_ne!(
        outputs[0], "t2",
        "the negative EOS bias must survive speculative mode"
    );
}

#[test]
fn draft_proposals_sample_the_declared_distribution_reproducibly() {
    use crate::executor::speculative::DraftModelStrategy;
    let directory = cycle_fixture();
    let draft = || {
        DraftModelStrategy::new(
            directory.path().to_string_lossy().into_owned(),
            16,
            DeviceKind::Cpu,
        )
        .with_seed(7)
    };
    let first = draft();
    let second = draft();
    let draws = first.propose_with_logits(&[1, 2], 16, 100.0).unwrap();
    let replay = second.propose_with_logits(&[1, 2], 16, 100.0).unwrap();
    assert_eq!(draws, replay);
    assert!(
        draws.iter().any(|(token, logits)| logits[*token as usize]
            < logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
        "nonzero-temperature draft proposals must not always be argmax"
    );
    let next = first.propose_with_logits(&[1, 2], 16, 100.0).unwrap();
    assert_ne!(
        draws, next,
        "proposal RNG must advance across verifier rounds"
    );
}

#[test]
fn native_logprobs_match_buffered_streaming_bytes_and_exclude_eos() {
    let directory = cycle_fixture();
    let model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    assert!(model.supports_logprobs());
    let params = GenerationParams {
        max_tokens: 6,
        temperature: 0.,
        logprobs: Some(3),
        ..Default::default()
    };
    let input = crate::ModelInput::Text {
        prompt: "t1".into(),
    };
    let buffered = model.infer(input.clone(), &params).unwrap();
    buffered.validate().unwrap();
    let mut chunks = Vec::new();
    model
        .infer_stream(input, &params, &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap();
    let streamed = crate::ModelOutput::from(chunks);
    assert_eq!(buffered.text, streamed.text);
    assert_eq!(buffered.logprobs, streamed.logprobs);
    let records = buffered.logprobs.unwrap();
    assert!(!records.is_empty());
    assert!(
        records
            .iter()
            .all(|record| record.token.token_id != 31 && record.top_logprobs.len() == 3)
    );
    assert!(
        model
            .infer_tokens_stream_with_speculation(
                vec![1],
                &params,
                &mut |_| Ok(()),
                SpeculativeMode::NGram {
                    ngram_order: 1,
                    num_speculative: 2
                }
            )
            .is_err()
    );
}
