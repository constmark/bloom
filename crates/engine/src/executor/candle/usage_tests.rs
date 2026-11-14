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
        OutputChunk::GenerationFinished(crate::GenerationMetadata {
            input_tokens: 5,
            output_tokens: 2,
            reason: crate::GenerationFinishReason::Eos,
        })
    )));
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
fn speculative_correction_eos_and_budget_report_explicit_metadata() {
    let directory = cycle_fixture();
    let model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    for suppress_eos in [false, true] {
        let mut events = Vec::new();
        model
            .infer_tokens_stream_with_speculation(
                // The previous continuation of t2 is wrong, forcing correction.
                vec![1, 2, 0, 1],
                &GenerationParams {
                    max_tokens: 3,
                    temperature: 0.0,
                    logit_bias: suppress_eos
                        .then(|| std::collections::BTreeMap::from([(31, -100.0)])),
                    ..Default::default()
                },
                &mut |event| {
                    events.push(event);
                    Ok(())
                },
                SpeculativeMode::NGram {
                    ngram_order: 1,
                    num_speculative: 2,
                },
            )
            .unwrap();
        assert!(events.iter().any(|event| matches!(event,
            OutputChunk::GenerationFinished(metadata)
                if metadata.input_tokens == 4
                    && metadata.output_tokens == if suppress_eos { 3 } else { 2 }
                    && metadata.reason == if suppress_eos { crate::GenerationFinishReason::Length } else { crate::GenerationFinishReason::Eos }
        )), "{events:?}");
        if !suppress_eos {
            assert!(events.iter().any(|event| matches!(event,
                OutputChunk::Metrics { speculative_draft_tokens: Some(drafts), speculative_accepted_tokens: Some(0), .. } if *drafts > 0
            )), "fixture must exercise a rejected draft: {events:?}");
        }
    }
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
fn resident_prefix_tracks_only_tokens_already_forwarded_at_the_generation_limit() {
    let directory = cycle_fixture();
    let mut model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    model.residency_strategy = ResidencyStrategy::Resident;
    for max_tokens in [1, 2, 5] {
        model.current_prefix.lock().unwrap().clear();
        let params = GenerationParams {
            max_tokens,
            temperature: 0.0,
            logit_bias: Some(std::collections::BTreeMap::from([(31, -100.0)])),
            ..Default::default()
        };
        model
            .infer_tokens_stream_with_speculation(
                vec![1],
                &params,
                &mut |_| Ok(()),
                SpeculativeMode::None,
            )
            .unwrap();
        assert_eq!(
            model.current_prefix.lock().unwrap().len(),
            max_tokens,
            "the final sampled token has not been forwarded into KV"
        );
    }
}

#[test]
fn native_generation_metadata_distinguishes_budget_eos_and_visible_token_limits() {
    use crate::{GenerationFinishReason as Reason, GenerationMetadata};
    let directory = cycle_fixture();
    let model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    for (prompt, maximum, count, reason) in [
        (vec![2], 1, 1, Reason::Eos),
        (vec![1], 2, 2, Reason::Eos),
        (vec![1], 1, 1, Reason::Length),
        (vec![0], 8, 3, Reason::Eos),
    ] {
        let mut events = Vec::new();
        model
            .infer_tokens_stream_with_speculation(
                prompt.clone(),
                &GenerationParams {
                    max_tokens: maximum,
                    temperature: 0.0,
                    ..Default::default()
                },
                &mut |event| {
                    events.push(event);
                    Ok(())
                },
                SpeculativeMode::None,
            )
            .unwrap();
        let metadata = events
            .iter()
            .filter_map(|event| match event {
                OutputChunk::GenerationFinished(metadata) => Some(*metadata),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            metadata,
            [GenerationMetadata {
                input_tokens: prompt.len(),
                output_tokens: count,
                reason
            }]
        );
        let final_index = events
            .iter()
            .position(|event| matches!(event, OutputChunk::GenerationFinished(_)))
            .unwrap();
        assert!(
            events[final_index + 1..]
                .iter()
                .all(|event| matches!(event, OutputChunk::Metrics { .. } | OutputChunk::End))
        );
        assert!(matches!(events.last(), Some(OutputChunk::End)));
    }
    let mut events = Vec::new();
    assert!(
        model
            .infer_tokens_stream_with_speculation(
                vec![0],
                &GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    ..Default::default()
                },
                &mut |event| {
                    if matches!(event, OutputChunk::TextDelta(_)) {
                        anyhow::bail!("cancelled by client stop");
                    }
                    events.push(event);
                    Ok(())
                },
                SpeculativeMode::None
            )
            .is_err()
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, OutputChunk::GenerationFinished(_) | OutputChunk::End))
    );
    assert!(model.current_prefix.lock().unwrap().is_empty());
}

#[test]
fn resident_prefix_reuse_preserves_seeded_sampling_across_prompt_suffix_lengths() {
    let directory = cycle_fixture();
    let mut model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    model.residency_strategy = ResidencyStrategy::Resident;
    let warmup = GenerationParams {
        max_tokens: 1,
        temperature: 0.0,
        ..Default::default()
    };
    let params = GenerationParams {
        max_tokens: 4,
        temperature: 2.0,
        top_p: 1.0,
        seed: Some(7),
        logprobs: Some(0),
        logit_bias: Some(std::collections::BTreeMap::from([(31, -100.0)])),
        ..Default::default()
    };
    for suffix in [vec![3, 4], vec![3, 4, 5, 6]] {
        let mut prompt = vec![1];
        prompt.extend(suffix);
        model.current_prefix.lock().unwrap().clear();
        model
            .infer_tokens_stream_with_speculation(
                vec![1],
                &warmup,
                &mut |_| Ok(()),
                SpeculativeMode::None,
            )
            .unwrap();
        // Keep this sampling regression independent of prefix bookkeeping:
        // warmup forwarded exactly token 1, regardless of its generated token.
        *model.current_prefix.lock().unwrap() = vec![1];
        let mut outputs = Vec::new();
        for cached in [true, false] {
            if !cached {
                model.current_prefix.lock().unwrap().clear();
            }
            let mut tokens = Vec::new();
            model
                .infer_tokens_stream_with_speculation(
                    prompt.clone(),
                    &params,
                    &mut |event| {
                        if let OutputChunk::TokenLogprobs(record) = event {
                            tokens.push(record.token.token_id);
                        }
                        Ok(())
                    },
                    SpeculativeMode::None,
                )
                .unwrap();
            outputs.push(tokens);
        }
        assert_eq!(outputs[0], outputs[1], "prompt: {prompt:?}");
    }
}

#[test]
fn direct_stream_buffers_split_unicode_and_flushes_the_terminal_decoder_policy() {
    let directory = cycle_fixture();
    let mut model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    // The deterministic cycle generates IDs 1, 2, 31, 0. Give the first
    // three their ByteLevel encodings for UTF-8 E2 82 AC (the euro sign).
    let vocabulary = (0..32u32)
        .map(|id| {
            let text = match id {
                0 => "A".into(),
                1 => "â".into(),
                2 => "Ĥ".into(),
                31 => "¬".into(),
                _ => format!("t{id}"),
            };
            (text, id)
        })
        .collect();
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("A".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    model.tokenizer = Arc::new(tokenizer);
    model.eos_token_ids = vec![30];
    for (max_tokens, expected) in [(3, "€"), (4, "€A"), (2, "�")] {
        let params = GenerationParams {
            max_tokens,
            temperature: 0.0,
            ..Default::default()
        };
        let mut deltas = Vec::new();
        model
            .infer_tokens_stream_with_speculation(
                vec![0],
                &params,
                &mut |event| {
                    if let OutputChunk::TextDelta(delta) = event {
                        deltas.push(delta);
                    }
                    Ok(())
                },
                SpeculativeMode::None,
            )
            .unwrap();
        assert_eq!(deltas.concat(), expected, "max_tokens: {max_tokens}");
    }
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

#[test]
fn presence_and_frequency_penalties_count_only_generated_tokens() {
    let directory = cycle_fixture();
    let model = CandleEngine
        .load_text_model(directory.path(), DeviceKind::Cpu)
        .unwrap();
    // Every prompt token, including EOS, appears in the prompt. Penalizing
    // prompt history would suppress both the natural continuation and EOS.
    let params = GenerationParams {
        max_tokens: 6,
        temperature: 0.0,
        presence_penalty: 2.0,
        frequency_penalty: 2.0,
        logprobs: Some(1),
        ..Default::default()
    };
    let generate = |params: &GenerationParams, mode| {
        let mut events = Vec::new();
        model
            .infer_tokens_stream_with_speculation(
                vec![1, 2, 31, 0, 1],
                params,
                &mut |event| {
                    events.push(event);
                    Ok(())
                },
                mode,
            )
            .unwrap();
        events
    };
    let unpenalized = generate(
        &GenerationParams {
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            ..params.clone()
        },
        SpeculativeMode::None,
    )
    .into_iter()
    .find_map(|event| match event {
        OutputChunk::TokenLogprobs(record) => Some(record.token.logprob),
        _ => None,
    })
    .unwrap();
    for mode in [
        SpeculativeMode::None,
        SpeculativeMode::NGram {
            ngram_order: 1,
            num_speculative: 3,
        },
    ] {
        let speculative = !matches!(mode, SpeculativeMode::None);
        let request = GenerationParams {
            logprobs: (!speculative).then_some(1),
            ..params.clone()
        };
        let events = generate(&request, mode);
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                OutputChunk::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "t2", "prompt tokens must not be penalized");
        assert!(events.iter().any(|event| matches!(
            event,
            OutputChunk::GenerationFinished(crate::GenerationMetadata {
                output_tokens: 2,
                reason: crate::GenerationFinishReason::Eos,
                ..
            })
        )));
        let selected: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                OutputChunk::TokenLogprobs(record) => Some(record.token.logprob),
                _ => None,
            })
            .collect();
        if speculative {
            assert!(selected.is_empty());
            continue;
        }
        assert_eq!(selected.len(), 1);
        assert!(
            (selected[0] - unpenalized).abs() < 1e-9,
            "logprobs must describe the unpenalized first-token distribution: {} vs {unpenalized}",
            selected[0]
        );
    }
}

#[test]
fn direct_top_k_then_top_p_renormalizes_like_the_batch_sampler() {
    // Four equal logits with k=2 leave two tokens at probability 0.5 each
    // after renormalization, so top_p=0.3 keeps only one of them.
    let logits = Tensor::new(&[0f32, 0., 0., 0.], &Device::Cpu).unwrap();
    let mut direct = std::collections::BTreeSet::new();
    for seed in 0..64 {
        let params = GenerationParams {
            temperature: 1.0,
            top_p: 0.3,
            top_k: Some(2),
            seed: Some(seed),
            ..Default::default()
        };
        let mut processor = CandleTextModel::logits_processor(&params);
        direct.insert(
            CandleTextModel::sample_with_penalties(&mut processor, &logits, &params, &[]).unwrap(),
        );
    }
    assert_eq!(direct.len(), 1, "direct nucleus: {direct:?}");
}
