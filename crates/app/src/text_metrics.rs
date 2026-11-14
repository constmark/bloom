//! Text generation accounting shared by the command-line applications.

use anyhow::{Result, anyhow};
use bloomai_engine::{InferencePipeline, io::OutputChunk};

/// Prefer the model's cumulative accepted-token count (including EOS). For
/// adapters without usage metadata, tokenize the complete output once; stream
/// chunks and whitespace-separated words are not token boundaries.
#[derive(Default)]
pub struct TextTokenCounter {
    text: String,
    generation: bloomai_engine::GenerationStreamState,
}

impl TextTokenCounter {
    pub fn observe(&mut self, chunk: &OutputChunk, max_tokens: usize) -> Result<()> {
        self.generation
            .observe(chunk, max_tokens)
            .map_err(anyhow::Error::msg)?;
        if let OutputChunk::TextDelta(text) = chunk {
            if text.len() > bloomai_engine::io::MAX_TEXT_BYTES.saturating_sub(self.text.len()) {
                return Err(anyhow!("text output exceeded the byte limit"));
            }
            self.text.push_str(text);
        }
        Ok(())
    }

    pub fn count(&self, pipeline: &InferencePipeline) -> Result<usize> {
        self.count_with(|text| Ok(pipeline.tokenize(text)?.len()))
    }

    pub fn metadata(&self) -> Option<bloomai_engine::GenerationMetadata> {
        self.generation.metadata()
    }

    fn count_with(&self, tokenize: impl FnOnce(&str) -> Result<usize>) -> Result<usize> {
        match self.generation.output_tokens() {
            Some(count) => Ok(count),
            None if self.text.is_empty() => Ok(0),
            None => tokenize(&self.text),
        }
    }
}

/// Buffered CLI generation still consumes native metadata; legacy adapters
/// retain their existing buffered implementation and output shape.
pub fn run_buffered(
    pipeline: &InferencePipeline,
    input: bloomai_engine::ModelInput,
    params: &bloomai_core::GenerationParams,
) -> Result<(
    bloomai_engine::ModelOutput,
    Option<bloomai_engine::GenerationMetadata>,
)> {
    if !pipeline.model().supports_generation_metadata() {
        return Ok((pipeline.run(input, params)?, None));
    }
    let mut counter = TextTokenCounter::default();
    let mut chunks = Vec::new();
    pipeline.run_stream(input, params, &mut |chunk: OutputChunk| {
        counter.observe(&chunk, params.max_tokens)?;
        if matches!(
            chunk,
            OutputChunk::TextDelta(_) | OutputChunk::TokenLogprobs(_)
        ) {
            chunks.push(chunk);
        }
        Ok(())
    })?;
    let mut output = bloomai_engine::ModelOutput::from(chunks);
    if params.logprobs.is_some() && output.logprobs.is_none() {
        output.logprobs = Some(Vec::new());
    }
    output.validate().map_err(anyhow::Error::msg)?;
    Ok((output, counter.metadata()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounting_preserves_native_usage_and_is_independent_of_chunk_boundaries() {
        for chunks in [vec!["你好 Bloom"], vec!["你", "好 ", "Bloom"]] {
            let mut counter = TextTokenCounter::default();
            for text in chunks {
                counter
                    .observe(&OutputChunk::TextDelta(text.into()), 8)
                    .unwrap();
            }
            assert_eq!(
                counter.count_with(|text| Ok(text.chars().count())).unwrap(),
                8
            );
            counter
                .observe(
                    &OutputChunk::TokenUsage {
                        input_tokens: 2,
                        output_tokens: 5,
                    },
                    8,
                )
                .unwrap();
            assert_eq!(
                counter
                    .count_with(|_| panic!("must use native usage"))
                    .unwrap(),
                5
            );
        }
        let mut eos = TextTokenCounter::default();
        assert_eq!(
            eos.count_with(|_| panic!("empty output has zero tokens"))
                .unwrap(),
            0
        );
        eos.observe(
            &OutputChunk::TokenUsage {
                input_tokens: 2,
                output_tokens: 1,
            },
            8,
        )
        .unwrap();
        assert_eq!(eos.count_with(|_| panic!("EOS has no text")).unwrap(), 1);
    }

    #[test]
    fn malformed_usage_does_not_replace_the_last_valid_count() {
        let mut counter = TextTokenCounter::default();
        counter
            .observe(
                &OutputChunk::TokenUsage {
                    input_tokens: 2,
                    output_tokens: 3,
                },
                8,
            )
            .unwrap();
        for (input_tokens, output_tokens) in [(2, 2), (3, 4), (2, 9)] {
            assert!(
                counter
                    .observe(
                        &OutputChunk::TokenUsage {
                            input_tokens,
                            output_tokens
                        },
                        8
                    )
                    .is_err()
            );
            assert_eq!(counter.count_with(|_| panic!()).unwrap(), 3);
        }
    }
}
