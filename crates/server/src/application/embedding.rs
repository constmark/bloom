//! Protocol-neutral embedding preparation and projection.
//!
//! This module deliberately has no Axum, HTTP DTO, or server-state dependency.
//! Protocol adapters provide request validation and map these bounded domain
//! errors to their own wire envelopes.

use anyhow::{Result, anyhow};
use bloomai_engine::{InferencePipeline, ModelInput, OutputChunk};
use std::sync::Arc;

pub(crate) const MAX_EMBEDDING_DIMENSIONS: usize = 16_384;
pub(crate) const MAX_EMBEDDING_VALUES: usize = 1_048_576;
pub(crate) const MAX_NATIVE_EMBEDDING_MICROBATCH_ITEMS: usize = 16;

#[derive(Debug)]
pub(crate) enum EmbeddingBatchOutput {
    Embeddings(Vec<Vec<f32>>),
    Rerank(Vec<RerankScore>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum EmbeddingProjection {
    L2Normalized {
        dimensions: Option<usize>,
        require_exact_dimensions: bool,
    },
    Rerank {
        top_n: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RerankScore {
    pub(crate) index: usize,
    pub(crate) relevance_score: f64,
}

pub(crate) fn prepare_embedding_inputs(
    pipeline: &InferencePipeline,
    inputs: Vec<String>,
    truncate_inputs: bool,
) -> std::result::Result<(Vec<String>, usize), String> {
    let context_size = pipeline.context_size();
    if context_size == 0 {
        return Err("The active model reports a zero-sized context window.".to_string());
    }
    let mut prepared = Vec::with_capacity(inputs.len());
    let mut prompt_tokens = 0_usize;
    for (index, input) in inputs.into_iter().enumerate() {
        let tokens = pipeline
            .tokenize(&input)
            .map_err(|error| format!("Failed to tokenize embedding input {index}: {error}"))?;
        let (input, token_count) = if tokens.len() > context_size {
            if !truncate_inputs {
                return Err(format!(
                    "Embedding input {index} contains {} tokens and exceeds the active context window of {context_size} tokens.",
                    tokens.len()
                ));
            }
            truncate_embedding_input(pipeline, &tokens, context_size, index)?
        } else {
            (input, tokens.len())
        };
        prompt_tokens = prompt_tokens
            .checked_add(token_count)
            .ok_or_else(|| "Embedding prompt token count overflowed.".to_string())?;
        prepared.push(input);
    }
    Ok((prepared, prompt_tokens))
}

fn truncate_embedding_input(
    pipeline: &InferencePipeline,
    tokens: &[u32],
    context_size: usize,
    index: usize,
) -> std::result::Result<(String, usize), String> {
    const MAX_ROUND_TRIPS: usize = 16;

    let mut retained_tokens = context_size.min(tokens.len());
    for _ in 0..MAX_ROUND_TRIPS {
        let truncated = pipeline
            .detokenize(&tokens[..retained_tokens])
            .map_err(|error| format!("Failed to truncate embedding input {index}: {error}"))?;
        if truncated.trim().is_empty() {
            return Err(format!(
                "Embedding input {index} became empty after context truncation."
            ));
        }
        let round_trip_tokens = pipeline.tokenize(&truncated).map_err(|error| {
            format!("Failed to validate truncated embedding input {index}: {error}")
        })?;
        if round_trip_tokens.len() <= context_size {
            return Ok((truncated, round_trip_tokens.len()));
        }
        let excess = round_trip_tokens.len().saturating_sub(context_size).max(1);
        retained_tokens = retained_tokens.saturating_sub(excess);
        if retained_tokens == 0 {
            break;
        }
    }
    Err(format!(
        "Embedding input {index} could not be safely truncated to the active context window of {context_size} tokens."
    ))
}

pub(crate) fn collect_embedding(
    pipeline: Arc<InferencePipeline>,
    text: String,
) -> Result<Vec<f32>> {
    let mut embedding = None;
    let params = bloomai_core::GenerationParams {
        max_tokens: 1,
        temperature: 0.0,
        top_p: 1.0,
        seed: None,
        response_format: None,
    };
    pipeline.run_stream(
        ModelInput::Text { prompt: text },
        &params,
        &mut |chunk: OutputChunk| {
            if let OutputChunk::Embedding(values) = chunk {
                embedding = Some(values);
            }
            Ok(())
        },
    )?;

    embedding.ok_or_else(|| anyhow!("model did not produce OutputChunk::Embedding"))
}

pub(crate) fn validate_embedding_output(
    embedding: &[f32],
    expected_dimensions: &mut Option<usize>,
    total_values: &mut usize,
) -> std::result::Result<(), String> {
    if embedding.is_empty() || embedding.len() > MAX_EMBEDDING_DIMENSIONS {
        return Err(format!(
            "Embedding output must contain between 1 and {MAX_EMBEDDING_DIMENSIONS} dimensions."
        ));
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err("Embedding output contains a non-finite value.".to_string());
    }
    let norm_squared = embedding.iter().fold(0.0_f64, |sum, value| {
        sum + f64::from(*value) * f64::from(*value)
    });
    if !norm_squared.is_finite() || norm_squared <= f64::EPSILON {
        return Err("Embedding output must have a finite non-zero norm.".to_string());
    }
    if let Some(expected) = *expected_dimensions {
        if embedding.len() != expected {
            return Err(format!(
                "Embedding batch changed dimensions from {expected} to {}.",
                embedding.len()
            ));
        }
    } else {
        *expected_dimensions = Some(embedding.len());
    }
    *total_values = total_values
        .checked_add(embedding.len())
        .ok_or_else(|| "Embedding output value count overflowed.".to_string())?;
    if *total_values > MAX_EMBEDDING_VALUES {
        return Err(format!(
            "Embedding batch cannot contain more than {MAX_EMBEDDING_VALUES} values."
        ));
    }
    Ok(())
}

pub(crate) fn normalize_embedding_batch(
    embeddings: &[Vec<f32>],
    dimensions: Option<usize>,
    require_exact_dimensions: bool,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    if dimensions.is_some_and(|dimensions| dimensions > MAX_EMBEDDING_DIMENSIONS) {
        return Err(format!(
            "dimensions cannot exceed {MAX_EMBEDDING_DIMENSIONS}."
        ));
    }
    embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            let target = dimensions
                .filter(|dimensions| *dimensions > 0)
                .unwrap_or(embedding.len());
            if require_exact_dimensions && target > embedding.len() {
                return Err(format!(
                    "Requested {target} embedding dimensions, but the active model produces {}.",
                    embedding.len()
                ));
            }
            let target = target.min(embedding.len());
            let norm_squared = embedding[..target].iter().try_fold(0.0_f64, |sum, value| {
                if !value.is_finite() {
                    return Err(format!(
                        "Embedding output {index} contains a non-finite value."
                    ));
                }
                Ok(sum + f64::from(*value) * f64::from(*value))
            })?;
            if !norm_squared.is_finite() || norm_squared <= f64::EPSILON {
                return Err(format!(
                    "Embedding output {index} cannot be normalized because its norm is zero."
                ));
            }
            let inverse_norm = norm_squared.sqrt().recip();
            Ok(embedding[..target]
                .iter()
                .map(|value| (f64::from(*value) * inverse_norm) as f32)
                .collect())
        })
        .collect()
}

pub(crate) fn rank_embedding_documents(
    embeddings: &[Vec<f32>],
    top_n: usize,
) -> std::result::Result<Vec<RerankScore>, String> {
    let Some((_, documents)) = embeddings.split_first() else {
        return Err("Rerank output omitted the query embedding.".to_string());
    };
    if documents.is_empty() {
        return Err("Rerank output omitted document embeddings.".to_string());
    }
    if top_n == 0 || top_n > documents.len() {
        return Err(format!(
            "Rerank top_n must be between 1 and {}.",
            documents.len()
        ));
    }
    let normalized = normalize_embedding_batch(embeddings, None, true)?;
    let query = &normalized[0];
    let mut scores = normalized[1..]
        .iter()
        .enumerate()
        .map(|(index, document)| {
            if document.len() != query.len() {
                return Err(format!(
                    "Rerank document embedding {index} has {} dimensions; expected {}.",
                    document.len(),
                    query.len()
                ));
            }
            let relevance_score = query
                .iter()
                .zip(document)
                .fold(0.0_f64, |score, (left, right)| {
                    score + f64::from(*left) * f64::from(*right)
                });
            if !relevance_score.is_finite() {
                return Err(format!(
                    "Rerank document embedding {index} produced a non-finite score."
                ));
            }
            Ok(RerankScore {
                index,
                relevance_score: relevance_score.clamp(-1.0, 1.0),
            })
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    scores.sort_by(|left, right| {
        right
            .relevance_score
            .total_cmp(&left.relevance_score)
            .then_with(|| left.index.cmp(&right.index))
    });
    scores.truncate(top_n);
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_projection_truncates_and_rejects_invalid_vectors() {
        let normalized = normalize_embedding_batch(&[vec![3.0, 4.0, 12.0]], Some(2), true).unwrap();
        assert_eq!(normalized, vec![vec![0.6, 0.8]]);
        assert!(normalize_embedding_batch(&[vec![0.0, 0.0]], None, false).is_err());
        assert!(normalize_embedding_batch(&[vec![1.0, 2.0]], Some(3), true).is_err());
        let native = normalize_embedding_batch(&[vec![3.0, 4.0]], Some(3), false).unwrap();
        assert_eq!(native, vec![vec![0.6, 0.8]]);
    }

    #[test]
    fn embedding_output_validation_enforces_shape_and_aggregate_bounds() {
        let mut expected = None;
        let mut total = 0;
        validate_embedding_output(&[1.0, 2.0], &mut expected, &mut total).unwrap();
        assert_eq!(expected, Some(2));
        assert_eq!(total, 2);
        assert!(validate_embedding_output(&[1.0], &mut expected, &mut total).is_err());
        let mut expected = None;
        let mut total = 0;
        assert!(validate_embedding_output(&[f32::NAN, 1.0], &mut expected, &mut total).is_err());
        assert!(validate_embedding_output(&[0.0, 0.0], &mut expected, &mut total).is_err());
    }

    #[test]
    fn rerank_scoring_is_normalized_bounded_and_stable() {
        let scores = rank_embedding_documents(
            &[
                vec![2.0, 0.0],
                vec![1.0, 0.0],
                vec![3.0, 0.0],
                vec![0.0, 4.0],
                vec![-1.0, 0.0],
            ],
            3,
        )
        .unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].index, 0);
        assert_eq!(scores[1].index, 1);
        assert!(scores[0].relevance_score > 0.999_999);
        assert_eq!(scores[2].index, 2);
        assert!(scores[2].relevance_score.abs() < 1e-9);
    }

    #[test]
    fn rerank_scoring_rejects_invalid_shapes_and_limits() {
        assert!(rank_embedding_documents(&[], 1).is_err());
        assert!(rank_embedding_documents(&[vec![1.0]], 1).is_err());
        assert!(rank_embedding_documents(&[vec![1.0], vec![1.0]], 0).is_err());
        assert!(rank_embedding_documents(&[vec![1.0], vec![1.0]], 2).is_err());
        assert!(rank_embedding_documents(&[vec![1.0, 0.0], vec![1.0]], 1).is_err());
        assert!(rank_embedding_documents(&[vec![1.0, 0.0], vec![f32::NAN, 1.0]], 1).is_err());
    }
}
