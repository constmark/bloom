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
