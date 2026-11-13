//! Exact CPU text token probabilities and byte-preserving emission.
//!
//! The distribution is log-softmax of the model logits after the same
//! presence/frequency penalties and logit biases used by the sampler, before
//! temperature, top-k and top-p. Greedy output therefore still has a real
//! model probability. EOS does not produce an output record. Metadata precedes
//! its text delta; ByteLevel may need multiple records to complete UTF-8.
//!
//! Only a direct ByteLevel decoder and decoder-free WordLevel are admitted.
//! CPU direct text supports at most 4096 generated records, 20 alternatives,
//! 1024 raw bytes per token and 8 MiB of metadata. Other decoders/devices,
//! structured output, in-flight batching and speculative decoding fail closed.
//! An invalid UTF-8 sequence, including an unfinished codepoint at max_tokens
//! or EOS, returns an error rather than fabricating replacement bytes or a
//! successful terminal event. Earlier streamed metadata may already be visible.
use super::CandleTextModel;
use crate::io::{MAX_TEXT_BYTES, OutputChunk, TokenLogprob, TokenLogprobs};
use crate::model::OutputSink;
use anyhow::{Context, Result, ensure};
use bloomai_core::GenerationParams;
use candle_core::{DType, Tensor};
use std::collections::{HashMap, HashSet};
use tokenizers::Tokenizer;

pub(super) fn supports(tokenizer: &Tokenizer) -> bool {
    match tokenizer.get_decoder() {
        Some(decoder) => serde_json::to_value(decoder)
            .ok()
            .is_some_and(|value| value["type"] == "ByteLevel"),
        None => matches!(
            tokenizer.get_model(),
            tokenizers::models::ModelWrapper::WordLevel(_)
        ),
    }
}

pub(super) struct Emitter<'a> {
    tokenizer: &'a Tokenizer,
    byte_level: bool,
    byte_decoder: HashMap<char, u8>,
    special: HashSet<u32>,
    has_visible_token: bool,
    pending_utf8: Vec<u8>,
    metadata_bytes: usize,
}

impl<'a> Emitter<'a> {
    pub(super) fn new(tokenizer: &'a Tokenizer) -> Result<Self> {
        ensure!(
            supports(tokenizer),
            "logprobs requires ByteLevel or decoder-free WordLevel tokenization"
        );
        let mut byte_decoder = HashMap::new();
        let mut next = 256u32;
        for byte in 0..=255u8 {
            let scalar = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                u32::from(byte)
            } else {
                let scalar = next;
                next += 1;
                scalar
            };
            byte_decoder.insert(
                char::from_u32(scalar).context("invalid byte alphabet")?,
                byte,
            );
        }
        Ok(Self {
            tokenizer,
            byte_level: tokenizer.get_decoder().is_some(),
            byte_decoder,
            special: tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .filter_map(|(id, token)| token.special.then_some(id))
                .collect(),
            has_visible_token: false,
            pending_utf8: Vec::new(),
            metadata_bytes: 0,
        })
    }

    fn token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let text = self
            .tokenizer
            .id_to_token(id)
            .context("logprob token is outside the tokenizer vocabulary")?;
        if self.special.contains(&id) {
            return Ok(Vec::new());
        }
        let bytes = if self.byte_level {
            // Match Tokenizers' ByteLevel decoder, preserving incomplete UTF-8
            // rather than replacing it before neighboring tokens arrive.
            text.chars()
                .map(|c| self.byte_decoder.get(&c).copied())
                .collect::<Option<Vec<_>>>()
                .unwrap_or_else(|| text.into_bytes())
        } else if self.has_visible_token {
            format!(" {text}").into_bytes()
        } else {
            text.into_bytes()
        };
        ensure!(bytes.len() <= 1024, "logprob token exceeds 1024 bytes");
        Ok(bytes)
    }

    pub(super) fn emit(
        &mut self,
        logits: &Tensor,
        selected: u32,
        params: &GenerationParams,
        history: &[u32],
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let adjusted = CandleTextModel::adjusted_logits(logits, params, history)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let scores = log_probabilities(&adjusted)?;
        let selected_score = *scores
            .get(selected as usize)
            .context("selected token is outside logits")?;
        ensure!(
            selected_score.is_finite(),
            "selected token has zero or invalid model probability"
        );
        let record = |id: u32, score: f64| -> Result<TokenLogprob> {
            let bytes = self.token_bytes(id)?;
            Ok(TokenLogprob {
                token_id: id,
                token: String::from_utf8_lossy(&bytes).into_owned(),
                bytes,
                logprob: score,
            })
        };
        let count = params.logprobs.context("missing logprobs request")?;
        let mut candidates: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter_map(|(id, p)| p.is_finite().then_some(id))
            .collect();
        let compare = |a: &usize, b: &usize| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b));
        if candidates.len() > count && count > 0 {
            candidates.select_nth_unstable_by(count - 1, compare);
        }
        candidates.truncate(count);
        candidates.sort_by(compare);
        let metadata = TokenLogprobs {
            token: record(selected, selected_score)?,
            top_logprobs: candidates
                .into_iter()
                .map(|id| record(id as u32, scores[id]))
                .collect::<Result<_>>()?,
        };
        metadata.validate().map_err(anyhow::Error::msg)?;
        self.metadata_bytes = self
            .metadata_bytes
            .checked_add(metadata.byte_size())
            .context("logprobs output size overflow")?;
        ensure!(
            self.metadata_bytes <= MAX_TEXT_BYTES,
            "logprobs output exceeds 8 MiB"
        );
        if !self.special.contains(&selected) {
            self.has_visible_token = true;
        }
        self.pending_utf8.extend_from_slice(&metadata.token.bytes);
        sink.on_chunk(OutputChunk::TokenLogprobs(metadata))?;
        self.emit_text(sink)
    }

    fn emit_text(&mut self, sink: &mut dyn OutputSink) -> Result<()> {
        let complete = match std::str::from_utf8(&self.pending_utf8) {
            Ok(_) => self.pending_utf8.len(),
            Err(error) => {
                ensure!(
                    error.error_len().is_none(),
                    "logprob token bytes do not form valid UTF-8 text"
                );
                error.valid_up_to()
            }
        };
        if complete > 0 {
            let text = std::str::from_utf8(&self.pending_utf8[..complete])?.to_owned();
            sink.on_chunk(OutputChunk::TextDelta(text))?;
            self.pending_utf8.drain(..complete);
        }
        Ok(())
    }

    pub(super) fn finish(&self) -> Result<()> {
        ensure!(
            self.pending_utf8.is_empty(),
            "generation ended with an incomplete UTF-8 token sequence"
        );
        Ok(())
    }
}

fn log_probabilities(logits: &[f32]) -> Result<Vec<f64>> {
    ensure!(
        !logits.is_empty()
            && logits
                .iter()
                .all(|value| !value.is_nan() && *value != f32::INFINITY),
        "invalid model logits for logprobs"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    ensure!(max.is_finite(), "all model logits are masked");
    let sum: f64 = logits
        .iter()
        .map(|value| (f64::from(*value) - max).exp())
        .sum();
    let normalizer = sum.ln();
    Ok(logits
        .iter()
        .map(|value| ((f64::from(*value) - max) - normalizer).min(0.0))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn tokenizer(tokens: &[&str], byte_level: bool) -> Tokenizer {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                tokens
                    .iter()
                    .enumerate()
                    .map(|(id, text)| (text.to_string(), id as u32))
                    .collect(),
            )
            .unk_token(tokens[0].into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        if byte_level {
            tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        }
        tokenizer
    }

    #[test]
    fn logprob_math_is_stable_and_applies_real_bias_and_penalties_before_sampling_controls() {
        let logits = Tensor::new(&[10000f32, 10001., 9999.], &Device::Cpu).unwrap();
        let params = GenerationParams {
            temperature: 0.,
            top_p: 0.1,
            top_k: Some(1),
            presence_penalty: 1.,
            frequency_penalty: 0.5,
            logit_bias: Some(std::collections::BTreeMap::from([(2, 3.0)])),
            logprobs: Some(2),
            ..Default::default()
        };
        let adjusted = CandleTextModel::adjusted_logits(&logits, &params, &[1, 1])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(adjusted, vec![10000., 9999., 10002.]);
        let actual = log_probabilities(&adjusted).unwrap();
        let norm = (1.0f64 + (-2.0f64).exp() + (-3.0f64).exp()).ln();
        assert!((actual[2] + norm).abs() < 1e-12);
        assert!(
            actual[2] < 0.0,
            "greedy sampling must not fabricate logprob 0"
        );
        assert!((actual.iter().map(|x| x.exp()).sum::<f64>() - 1.0).abs() < 1e-12);
        for invalid in [
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
            vec![],
        ] {
            assert!(log_probabilities(&invalid).is_err());
        }
    }

    #[test]
    fn wordlevel_logprobs_preserve_joiners_and_top_candidates() {
        let tokenizer = tokenizer(&["hello", "world", "other"], false);
        let mut emitter = Emitter::new(&tokenizer).unwrap();
        let logits = Tensor::new(&[3f32, 2., 1.], &Device::Cpu).unwrap();
        let params = GenerationParams {
            logprobs: Some(2),
            ..Default::default()
        };
        let mut chunks = Vec::new();
        for token in [0, 1] {
            emitter
                .emit(&logits, token, &params, &[], &mut |chunk| {
                    chunks.push(chunk);
                    Ok(())
                })
                .unwrap();
        }
        emitter.finish().unwrap();
        let output = crate::ModelOutput::from(chunks);
        assert_eq!(output.text.as_deref(), Some("hello world"));
        output.validate().unwrap();
        let records = output.logprobs.unwrap();
        assert_eq!(records[1].token.bytes, b" world");
        assert_eq!(records[1].top_logprobs[0].bytes, b" hello");
    }

    #[test]
    fn bytelevel_logprobs_buffer_utf8_without_replacement_and_reject_incomplete_tail() {
        // ByteLevel encodings for UTF-8 E2 82 AC (the euro symbol).
        let tokenizer = tokenizer(&["â", "Ĥ", "¬"], true);
        let mut emitter = Emitter::new(&tokenizer).unwrap();
        let logits = Tensor::new(&[1f32, 1., 1.], &Device::Cpu).unwrap();
        let params = GenerationParams {
            logprobs: Some(0),
            ..Default::default()
        };
        let mut chunks = Vec::new();
        for token in [0, 1, 2] {
            emitter
                .emit(&logits, token, &params, &[], &mut |chunk| {
                    chunks.push(chunk);
                    Ok(())
                })
                .unwrap();
            if token < 2 {
                assert!(emitter.finish().is_err());
            }
        }
        emitter.finish().unwrap();
        assert_eq!(chunks.len(), 4);
        assert!(matches!(&chunks[3],OutputChunk::TextDelta(value) if value=="€"));
        let output = crate::ModelOutput::from(chunks);
        output.validate().unwrap();
        assert_eq!(
            output
                .logprobs
                .unwrap()
                .iter()
                .flat_map(|record| record.token.bytes.clone())
                .collect::<Vec<_>>(),
            "€".as_bytes()
        );
    }
}
