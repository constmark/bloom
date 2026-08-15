// Callback and KV bridge types stay explicit because they encode scheduler ownership.
#![allow(clippy::type_complexity)]

//! Batch executor that bridges the InferenceScheduler with the Candle model forward pass.
//!
//! Implements `EngineExecutor` to accept `ExecutionBatch` from the scheduler
//! and run actual model inference with packed multi-request tokens.
//!
//! ## KV cache bridge status
//!
//! The scheduler manages block allocation via `BloomKvCachePool` and the per-block
//! KV tensors via [`crate::scheduler::paged_cache::PagedAttentionCache`]. The model's
//! attention kernel (e.g. `StreamingQwenAttention`) holds its own internal KV cache
//! (candle_nn's `ConcatKvCache`) which is the source of truth during forward. To
//! let the paged cache participate in cross-request KV reuse, prefix caching and
//! the `paged_attention_forward` path, the model must expose per-layer KV through
//! [`crate::scheduler::kv_hook::KvHook`].
//!
//! When `kv_hook` is `None`, the executor does **not** touch the paged cache's KV
//! tensors — leaving them empty is more honest than writing placeholder zeros,
//! because empty blocks return zero KV from `read_kv` and remain available for
//! real data. When `kv_hook` is `Some`, the executor extracts KV from the model
//! after prefill, writes it into the paged cache, and (on decode for restored
//! blocks) injects KV back into the model before the forward pass.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use bloomai_core::{GenerationParams, ResponseFormat};
use candle_core::{Device, Tensor};

use super::speculative::{
    verify_greedy_tokens, DraftModelStrategy, NGramStrategy, SpeculativeMode, SpeculativeStrategy,
};
use crate::scheduler::kv_hook::KvHook;
use crate::scheduler::paged_cache::PagedAttentionCache;
use crate::scheduler::{
    token_budget_allows, BatchResult, EngineExecutor, ExecutionBatch, ExecutionPhase,
};

/// Shared state between the batch executor and the loaded model.
///
/// Holds a reference to the model wrapper and the device it runs on.
/// The model wrapper is behind a `Mutex` to allow safe access from
/// the scheduler loop running on a blocking thread.
pub struct CandleBatchExecutor {
    /// The underlying Candle model wrapper, shared with `CandleTextModel`.
    model: Arc<Mutex<Option<BatchableModel>>>,
    /// Device to run inference on.
    device: Device,
    /// Maximum batch size for prefill phase.
    max_prefill_batch: usize,
    /// Maximum batch size for decode phase.
    max_decode_batch: usize,
    /// Optional paged attention cache storage.
    cache: Option<Arc<PagedAttentionCache>>,
    /// Optional per-layer KV bridge. When absent the model's internal KV cache
    /// is the sole source of truth and the paged cache stays metadata-only.
    kv_hook: Option<Arc<dyn KvHook>>,
    /// Decoded vocabulary strings indexed by token id, used for token-level
    /// grammar constraints on the IFB hot path. Empty when no grammar
    /// support is configured.
    vocab_strings: Vec<String>,
    /// End-of-sequence token ids, used by grammar filtering to decide when
    /// a structured response is allowed to terminate.
    eos_token_ids: Vec<u32>,
    /// Explicit speculative mode used by deterministic executor tests. Normal
    /// runtime construction retains environment-backed configuration.
    #[cfg(test)]
    speculative_mode: Option<SpeculativeMode>,
    /// Optional tokenizer for token-to-text decoding during generation.
    tokenizer: Option<tokenizers::Tokenizer>,
    /// Speculative decoding strategies indexed by request KV handle.
    speculative_strategies: Arc<Mutex<HashMap<usize, Box<dyn SpeculativeStrategy>>>>,
}

/// A model wrapper that supports batched forward passes.
///
/// Unlike `QwenModelWrapper` (which holds model weights and caches internally),
/// this wrapper is designed for the scheduler-driven path where the KV cache
/// is managed externally by `BloomKvCachePool`.
pub struct BatchableModel {
    /// Underlying model — for now we hold the full `QwenModelWrapper`-equivalent
    /// forward function as a boxed closure to decouple from the candle model types.
    pub forward_fn: Box<dyn Fn(&Tensor, usize, Option<usize>) -> Result<Tensor> + Send + Sync>,
    /// Optional batched forward function for packed prefill and stacked decode.
    pub forward_batch_fn:
        Option<Box<dyn Fn(&Tensor, &[usize], &[usize], &[usize]) -> Result<Tensor> + Send + Sync>>,
    /// Whether this is a streaming (layer-wise) model.
    pub is_streaming: bool,
}

impl CandleBatchExecutor {
    /// Create a new batch executor with a model forward function.
    pub fn new(
        forward_fn: Box<dyn Fn(&Tensor, usize, Option<usize>) -> Result<Tensor> + Send + Sync>,
        device: Device,
        max_prefill_batch: usize,
        max_decode_batch: usize,
    ) -> Self {
        let model = BatchableModel {
            forward_fn,
            forward_batch_fn: None,
            is_streaming: false,
        };
        Self {
            model: Arc::new(Mutex::new(Some(model))),
            device,
            max_prefill_batch,
            max_decode_batch,
            cache: None,
            kv_hook: None,
            vocab_strings: Vec::new(),
            eos_token_ids: Vec::new(),
            #[cfg(test)]
            speculative_mode: None,
            tokenizer: None,
            speculative_strategies: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set an optional batched forward function.
    pub fn with_forward_batch_fn(
        self,
        forward_batch_fn: Box<
            dyn Fn(&Tensor, &[usize], &[usize], &[usize]) -> Result<Tensor> + Send + Sync,
        >,
    ) -> Self {
        if let Some(ref mut model) = *self.model.lock().unwrap_or_else(|e| e.into_inner()) {
            model.forward_batch_fn = Some(forward_batch_fn);
        }
        self
    }

    /// Set an optional paged attention cache.
    pub fn with_cache(mut self, cache: Arc<PagedAttentionCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Attach a [`KvHook`] so the executor can extract KV from the model after
    /// prefill and write it into the paged cache, then read it back and inject
    /// it into the model before decode. Without this the paged cache stays
    /// metadata-only — block allocation, eviction and metrics still work, but
    /// `paged_attention_forward` has no real KV to gather.
    pub fn with_kv_hook(mut self, hook: Arc<dyn KvHook>) -> Self {
        self.kv_hook = Some(hook);
        self
    }

    /// Set vocabulary and tokenizer for structured generation / grammar filtering.
    pub fn with_vocab_and_tokenizer(
        mut self,
        vocab_strings: Vec<String>,
        eos_token_ids: Vec<u32>,
        tokenizer: Option<tokenizers::Tokenizer>,
    ) -> Self {
        self.vocab_strings = vocab_strings;
        self.eos_token_ids = eos_token_ids;
        self.tokenizer = tokenizer;
        self
    }

    /// Attach vocabulary and eos tokens so the executor can apply
    /// token-level grammar constraints (`response_format` = `JsonObject` /
    /// `JsonSchema`) on the IFB hot path before sampling.
    ///
    /// Without this, `response_format` requests on the batched path fall back
    /// to unconstrained sampling (matching the single-request path's behaviour
    /// when vocab is unavailable).
    pub fn with_grammar_support(
        mut self,
        vocab_strings: Vec<String>,
        eos_token_ids: Vec<u32>,
    ) -> Self {
        self.vocab_strings = vocab_strings;
        self.eos_token_ids = eos_token_ids;
        self
    }

    #[cfg(test)]
    fn with_speculative_mode(mut self, mode: SpeculativeMode) -> Self {
        self.speculative_mode = Some(mode);
        self
    }

    /// Create a batch executor from a `CandleTextModel`'s shared model state.
    ///
    /// This allows the scheduler to drive the same model that `CandleTextModel`
    /// uses for single-request inference.
    pub fn from_shared_model(
        model: Arc<Mutex<Option<BatchableModel>>>,
        device: Device,
        max_prefill_batch: usize,
        max_decode_batch: usize,
    ) -> Self {
        Self {
            model,
            device,
            max_prefill_batch,
            max_decode_batch,
            cache: None,
            kv_hook: None,
            vocab_strings: Vec::new(),
            eos_token_ids: Vec::new(),
            #[cfg(test)]
            speculative_mode: None,
            tokenizer: None,
            speculative_strategies: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Run a prefill forward pass with packed multi-request tokens.
    ///
    /// `tokens` contains concatenated prompt tokens from multiple requests,
    /// `cu_seqlens` marks the cumulative sequence boundaries.
    fn forward_prefill(
        &self,
        tokens: &[u32],
        cu_seqlens: &[usize],
        kv_handles: &[usize],
        start_positions: &[usize],
        params: &[GenerationParams],
        generated_tokens: &[Vec<u32>],
    ) -> Result<Vec<u32>> {
        let mut model_guard = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("model not loaded in batch executor"))?;

        if let Some(ref mut batch_forward) = model.forward_batch_fn {
            let input_tensor = Tensor::new(tokens, &self.device)?;
            let logits = (batch_forward)(&input_tensor, start_positions, kv_handles, cu_seqlens)?;
            let batch_size = kv_handles.len();
            let mut next_tokens = Vec::with_capacity(batch_size);

            for i in 0..batch_size {
                let start = cu_seqlens[i];
                let end = cu_seqlens[i + 1];
                let seq_len = end - start;
                let start_pos = start_positions.get(i).copied().unwrap_or(0);

                let last_token_idx = end - 1;
                let req_logits = if logits.rank() >= 2 {
                    if logits.dim(0)? == batch_size {
                        let b_logits = logits.get(i)?;
                        let last_idx = b_logits.dim(0)? - 1;
                        b_logits.get(last_idx)?
                    } else {
                        logits.get(last_token_idx)?
                    }
                } else {
                    logits.clone()
                };

                let req_params = params.get(i);
                let prior_tokens = generated_tokens.get(i).map(Vec::as_slice).unwrap_or(&[]);
                let next_tok = self.sample_request_token(&req_logits, req_params, prior_tokens)?;
                next_tokens.push(next_tok);

                let handle = kv_handles[i];
                self.write_request_kv(handle, start_pos, seq_len)?;
            }

            return Ok(next_tokens);
        }

        // For prefill, we process each request's prompt tokens sequentially
        // and collect the last logit for each to determine the first generated token.
        // Future optimization: use a single packed forward with cu_seqlens masking.
        let mut next_tokens = Vec::with_capacity(kv_handles.len());

        for (i, &handle) in kv_handles.iter().enumerate() {
            let start = cu_seqlens[i];
            let end = cu_seqlens[i + 1];
            let request_tokens = &tokens[start..end];
            let start_pos = start_positions.get(i).copied().unwrap_or(0);
            let seq_len = end - start;

            let input_ids = Tensor::new(request_tokens.to_vec(), &self.device)?.unsqueeze(0)?;
            let logits = (model.forward_fn)(&input_ids, start_pos, Some(handle))?;
            let logits = logits.squeeze(0)?;
            let logits = if logits.rank() >= 2 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };

            // Sample using per-request params (temperature/top_p/seed) with
            // optional grammar constraints from `response_format`.
            let req_params = params.get(i);
            let prior_tokens = generated_tokens.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let next_tok = self.sample_request_token(&logits, req_params, prior_tokens)?;

            next_tokens.push(next_tok);

            // Bridge model KV -> paged cache. Only when a KvHook is attached
            // do we have access to the model's internal KV tensors; without
            // the hook we intentionally leave the paged cache empty rather
            // than writing misleading placeholder zeros.
            self.write_request_kv(handle, start_pos, seq_len)?;
        }
        Ok(next_tokens)
    }

    /// Run a decode forward pass with one token per request.
    ///
    /// `tokens` contains one token per request (batch_size = tokens.len()),
    /// all at their respective `start_pos` positions.
    fn forward_decode(
        &self,
        tokens: &[u32],
        _cu_seqlens: &[usize],
        kv_handles: &[usize],
        start_positions: &[usize],
        params: &[GenerationParams],
        generated_tokens: &[Vec<u32>],
    ) -> Result<(Vec<u32>, Option<Vec<Vec<u32>>>)> {
        let mut model_guard = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("model not loaded in batch executor"))?;

        if let Some(ref mut batch_forward) = model.forward_batch_fn {
            for (i, &handle) in kv_handles.iter().enumerate() {
                let start_pos = start_positions.get(i).copied().unwrap_or(0);
                self.restore_request_kv(Some(handle), start_pos)?;
            }

            let input_tensor = Tensor::new(tokens, &self.device)?;
            let batch_size = tokens.len();
            let mut cu_seqlens = Vec::with_capacity(batch_size + 1);
            for i in 0..=batch_size {
                cu_seqlens.push(i);
            }
            let logits = (batch_forward)(&input_tensor, start_positions, kv_handles, &cu_seqlens)?;
            let mut next_tokens = Vec::with_capacity(batch_size);

            for i in 0..batch_size {
                let mut req_logits = logits.get(i)?;
                if req_logits.rank() >= 2 {
                    let last_idx = req_logits.dim(0)? - 1;
                    req_logits = req_logits.get(last_idx)?;
                }

                let req_params = params.get(i);
                let prior_tokens = generated_tokens.get(i).map(Vec::as_slice).unwrap_or(&[]);
                let next_tok = self.sample_request_token(&req_logits, req_params, prior_tokens)?;
                next_tokens.push(next_tok);

                let handle = kv_handles.get(i).copied().unwrap_or(0);
                let start_pos = start_positions.get(i).copied().unwrap_or(0);
                self.write_request_kv(handle, start_pos, 1)?;
            }

            return Ok((next_tokens, None));
        }

        let mut next_tokens = vec![0; tokens.len()];
        let mut speculative_tokens_list = vec![Vec::new(); tokens.len()];
        let mut has_speculative = false;

        let forward_fn = &model.forward_fn;

        // Since the multi-threaded path supports batch size >= 1 perfectly, we
        // run all decode passes inside scoped threads to keep the code unified.
        let mut results = Vec::new();
        for _ in 0..tokens.len() {
            results.push(Mutex::new(None));
        }

        #[cfg(test)]
        let speculative_mode = match &self.speculative_mode {
            Some(mode) => mode.clone(),
            None => SpeculativeMode::from_env()?,
        };
        #[cfg(not(test))]
        let speculative_mode = SpeculativeMode::from_env()?;

        std::thread::scope(|s| {
            for (i, &tok) in tokens.iter().enumerate() {
                let results_ref = &results;
                let speculative_mode_ref = &speculative_mode;
                s.spawn(move || {
                    let start_pos = if i < start_positions.len() {
                        start_positions[i]
                    } else {
                        0
                    };

                    let run = || -> Result<(u32, Vec<u32>)> {
                        let req_params = params.get(i);

                        let structured_output = req_params
                            .and_then(|params| params.response_format.as_ref())
                            .is_some_and(|format| {
                                matches!(
                                    format,
                                    ResponseFormat::JsonObject | ResponseFormat::JsonSchema(_)
                                )
                            });
                        let mut use_speculative = false;
                        let mut num_speculative = 0;
                        if structured_output
                            || matches!(speculative_mode_ref, SpeculativeMode::None)
                        {
                            // None
                        } else {
                            use_speculative = true;
                            num_speculative = match speculative_mode_ref {
                                SpeculativeMode::NGram {
                                    num_speculative, ..
                                } => *num_speculative,
                                SpeculativeMode::DraftModel {
                                    num_speculative, ..
                                } => *num_speculative,
                                SpeculativeMode::Mtp {
                                    num_speculative, ..
                                } => *num_speculative,
                                _ => 1,
                            };
                        }

                        self.restore_request_kv(kv_handles.get(i).copied(), start_pos)?;

                        if !use_speculative || num_speculative == 0 {
                            let input_ids = Tensor::new(&[[tok]], &self.device)?;
                            let handle_opt = kv_handles.get(i).copied();
                            let logits = (forward_fn)(&input_ids, start_pos, handle_opt)?;
                            let logits = logits.squeeze(0)?;
                            let logits = if logits.rank() >= 2 {
                                logits.get(logits.dim(0)? - 1)?
                            } else {
                                logits
                            };
                            let prior_tokens =
                                generated_tokens.get(i).map(Vec::as_slice).unwrap_or(&[]);
                            let next_tok =
                                self.sample_request_token(&logits, req_params, prior_tokens)?;

                            self.write_request_kv(
                                kv_handles.get(i).copied().unwrap_or(0),
                                start_pos,
                                1,
                            )?;
                            return Ok((next_tok, Vec::new()));
                        }

                        let handle = kv_handles.get(i).copied().unwrap_or(0);

                        let mut strategy_map = self
                            .speculative_strategies
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        let strategy =
                            strategy_map.entry(handle).or_insert_with(
                                || match speculative_mode_ref {
                                    SpeculativeMode::NGram { ngram_order, .. } => {
                                        Box::new(NGramStrategy::new(*ngram_order))
                                    }
                                    SpeculativeMode::DraftModel {
                                        model_path,
                                        num_speculative,
                                    } => {
                                        let dev_kind = match self.device {
                                            Device::Cpu => bloomai_core::DeviceKind::Cpu,
                                            _ => bloomai_core::DeviceKind::Gpu,
                                        };
                                        Box::new(DraftModelStrategy::new(
                                            model_path.clone(),
                                            *num_speculative,
                                            dev_kind,
                                        ))
                                    }
                                    _ => Box::new(NGramStrategy::new(3)),
                                },
                            );

                        let mut all_tokens = generated_tokens.get(i).cloned().unwrap_or_default();
                        if all_tokens.last().copied() != Some(tok) {
                            all_tokens.push(tok);
                        }

                        strategy.update_context(&all_tokens);

                        let dynamic_limit = strategy.current_speculative_limit(num_speculative);
                        let proposed = strategy.propose(&all_tokens, dynamic_limit)?;
                        if proposed.is_empty() {
                            let input_ids = Tensor::new(&[[tok]], &self.device)?;
                            let handle_opt = kv_handles.get(i).copied();
                            let logits = (forward_fn)(&input_ids, start_pos, handle_opt)?;
                            let logits = logits.squeeze(0)?;
                            let logits = if logits.rank() >= 2 {
                                logits.get(logits.dim(0)? - 1)?
                            } else {
                                logits
                            };
                            let prior_tokens =
                                generated_tokens.get(i).map(Vec::as_slice).unwrap_or(&[]);
                            let next_tok =
                                self.sample_request_token(&logits, req_params, prior_tokens)?;

                            self.write_request_kv(handle, start_pos, 1)?;
                            return Ok((next_tok, Vec::new()));
                        }

                        let mut verify_tokens = Vec::with_capacity(proposed.len() + 1);
                        verify_tokens.push(tok);
                        verify_tokens.extend_from_slice(&proposed);

                        let input_ids =
                            Tensor::new(verify_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                        let verifier_logits = (forward_fn)(&input_ids, start_pos, Some(handle))?;
                        let verifier_logits = verifier_logits.squeeze(0)?;

                        let mut verifier_greedy = Vec::with_capacity(proposed.len());
                        for idx in 0..proposed.len() {
                            let logits_row = verifier_logits.get(idx)?;
                            let logits_f32 = logits_row.to_dtype(candle_core::DType::F32)?;
                            let logits_vec = logits_f32.to_vec1::<f32>()?;
                            let max_tok = logits_vec
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(idx, _)| idx as u32)
                                .unwrap_or(0);
                            verifier_greedy.push(max_tok);
                        }

                        let accepted = verify_greedy_tokens(proposed.len(), &proposed, |idx| {
                            verifier_greedy.get(idx).copied()
                        });
                        strategy.record_acceptance(accepted, proposed.len());

                        let mut final_accepted_tokens = Vec::new();
                        for &t in proposed.iter().take(accepted) {
                            final_accepted_tokens.push(t);
                        }

                        let correction_idx = accepted;
                        let last_logits = verifier_logits.get(correction_idx)?;
                        let prior_tokens =
                            generated_tokens.get(i).map(Vec::as_slice).unwrap_or(&[]);
                        let next_tok =
                            self.sample_request_token(&last_logits, req_params, prior_tokens)?;

                        let final_len = accepted + 1;
                        self.write_request_kv(handle, start_pos, final_len)?;

                        if accepted < proposed.len() {
                            if let Some(ref hook) = self.kv_hook {
                                let rollback_len = start_pos + accepted + 1;
                                let _ = hook.rollback_kv_cache(handle, rollback_len);
                            }
                        }

                        let mut all_yielded = final_accepted_tokens;
                        all_yielded.push(next_tok);

                        let first_tok = all_yielded[0];
                        let extra_toks = all_yielded[1..].to_vec();
                        Ok((first_tok, extra_toks))
                    };

                    let res = run();
                    *results_ref[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(res);
                });
            }
        });

        for (i, slot) in results.into_iter().enumerate() {
            let (first_tok, extra_toks) = slot
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .ok_or_else(|| anyhow!("Thread did not finish task"))??;
            next_tokens[i] = first_tok;
            if !extra_toks.is_empty() {
                speculative_tokens_list[i] = extra_toks;
                has_speculative = true;
            }
        }

        let spec_tokens_res = if has_speculative {
            Some(speculative_tokens_list)
        } else {
            None
        };

        Ok((next_tokens, spec_tokens_res))
    }

    /// Sample a single token from `logits_tensor`, applying token-level
    /// grammar constraints when `response_format` is set on `req_params`
    /// and a vocabulary is available on the executor.
    ///
    /// `generated_tokens` is the scheduler-owned authoritative response
    /// history. Deriving grammar state from it on every decode keeps batching,
    /// preemption, and request-ID reuse from leaking or losing parser state.
    fn sample_request_token(
        &self,
        logits_tensor: &Tensor,
        req_params: Option<&GenerationParams>,
        generated_tokens: &[u32],
    ) -> Result<u32> {
        let format = req_params.and_then(|p| p.response_format.as_ref());
        let grammar_format = format
            .filter(|format| {
                matches!(
                    format,
                    ResponseFormat::JsonObject | ResponseFormat::JsonSchema(_)
                )
            })
            .filter(|_| !self.vocab_strings.is_empty());
        // Only apply grammar filtering for structured formats, and only when
        // we actually have a vocabulary to decode tokens against.
        let logits_vec = if let Some(fmt) = grammar_format {
            let generated_text = if generated_tokens.is_empty() {
                String::new()
            } else if let Some(tokenizer) = &self.tokenizer {
                tokenizer.decode(generated_tokens, false).map_err(|error| {
                    anyhow!("failed to decode structured generation state: {error}")
                })?
            } else {
                let mut text = String::new();
                for token in generated_tokens {
                    let decoded = self.vocab_strings.get(*token as usize).ok_or_else(|| {
                        anyhow!("structured generation token {token} is outside the vocabulary")
                    })?;
                    text.push_str(decoded);
                }
                text
            };
            let filtered = super::candle::filter_logits_by_grammar(
                logits_tensor,
                &generated_text,
                fmt,
                &self.vocab_strings,
                &self.eos_token_ids,
                &self.device,
            )?;
            filtered
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?
        } else {
            logits_tensor
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?
        };

        let next_tok = sample_logits(&logits_vec, req_params);
        Ok(next_tok)
    }

    /// Extract KV for `[start_pos, start_pos + seq_len)` from every layer of
    /// the attached model and write it into the paged cache's blocks.
    ///
    /// No-op when no `KvHook` or no `PagedAttentionCache` is attached. Returns
    /// an error if the hook fails (so the scheduler can surface it instead of
    /// silently producing mismatched KV).
    fn write_request_kv(&self, handle: usize, start_pos: usize, seq_len: usize) -> Result<()> {
        let (Some(cache), Some(hook)) = (self.cache.as_ref(), self.kv_hook.as_ref()) else {
            return Ok(());
        };
        if seq_len == 0 {
            return Ok(());
        }
        let config = cache.config();
        let block_size = config.block_size;
        let kv_dim = config.kv_dim;
        if hook.kv_dim() != kv_dim {
            return Err(anyhow!(
                "KvHook kv_dim {} != paged cache kv_dim {}",
                hook.kv_dim(),
                kv_dim
            ));
        }
        if hook.num_layers() != config.num_layers {
            return Err(anyhow!(
                "KvHook num_layers {} != paged cache num_layers {}",
                hook.num_layers(),
                config.num_layers
            ));
        }

        // Try tensor-based path first to keep tensors on GPU and avoid CPU round-trip
        let mut tensor_path_succeeded = true;
        for layer_idx in 0..config.num_layers {
            match hook.extract_kv_tensor(handle, layer_idx, start_pos, seq_len) {
                Ok(Some((k_tensor, v_tensor))) => {
                    // Write block-by-block using tensor slicing (narrow along sequence dim 2)
                    for offset in (0..seq_len).step_by(block_size) {
                        let chunk = block_size.min(seq_len - offset);
                        let block_token_idx = start_pos + offset;
                        let Some(block_id) = cache.block_for_handle(handle, block_token_idx) else {
                            continue;
                        };
                        if let (Ok(k_chunk), Ok(v_chunk)) = (
                            k_tensor.narrow(2, offset, chunk),
                            v_tensor.narrow(2, offset, chunk),
                        ) {
                            if let Err(e) =
                                cache.write_kv_tensor(layer_idx, block_id, k_chunk, v_chunk, chunk)
                            {
                                tracing::warn!(
                                    "Failed to write KV tensor for layer {} block {}: {:?}",
                                    layer_idx,
                                    block_id,
                                    e
                                );
                                tensor_path_succeeded = false;
                                break;
                            }
                        } else {
                            tensor_path_succeeded = false;
                            break;
                        }
                    }
                }
                _ => {
                    tensor_path_succeeded = false;
                    break;
                }
            }
            if !tensor_path_succeeded {
                break;
            }
        }

        if tensor_path_succeeded {
            return Ok(());
        }

        for layer_idx in 0..config.num_layers {
            let (keys, values) = hook.extract_kv(handle, layer_idx, start_pos, seq_len)?;
            // Write block-by-block so partial blocks are still persisted.
            for offset in (0..seq_len).step_by(block_size) {
                let chunk = block_size.min(seq_len - offset);
                let block_token_idx = start_pos + offset;
                let Some(block_id) = cache.block_for_handle(handle, block_token_idx) else {
                    // Block not tracked for this handle at this position — skip
                    // rather than failing the whole forward. The pool's metrics
                    // already reflect the allocation.
                    continue;
                };
                let k_start = offset * kv_dim;
                let k_end = (offset + chunk) * kv_dim;
                // write_kv expects `chunk * kv_dim` elements (not the full
                // block_size * kv_dim) — partial writes only persist the
                // actually-computed tokens.
                let mut block_keys = vec![0.0; chunk * kv_dim];
                let mut block_values = vec![0.0; chunk * kv_dim];
                block_keys.copy_from_slice(&keys[k_start..k_end]);
                block_values.copy_from_slice(&values[k_start..k_end]);
                cache.write_kv(layer_idx, block_id, block_keys, block_values, chunk)?;
            }
        }
        Ok(())
    }

    /// Read KV for `[start_pos, start_pos + 1)` from the paged cache and inject
    /// it back into the model via the attached hook.
    ///
    /// This is the inverse of [`Self::write_request_kv`]. It is intentionally
    /// narrow (one token at a time) for the decode hot path; the prefill path
    /// relies on the model's internal cache being freshly populated.
    fn restore_request_kv(&self, handle: Option<usize>, start_pos: usize) -> Result<()> {
        let (Some(cache), Some(hook), Some(&handle)) =
            (self.cache.as_ref(), self.kv_hook.as_ref(), handle.as_ref())
        else {
            return Ok(());
        };
        let config = cache.config();
        let block_size = config.block_size;
        let kv_dim = config.kv_dim;
        let Some(block_id) = cache.block_for_handle(handle, start_pos) else {
            return Ok(());
        };

        // Try tensor-based path first to avoid host-device copying
        let mut tensor_path_succeeded = true;
        for layer_idx in 0..config.num_layers {
            match cache.read_kv_tensor(layer_idx, block_id) {
                Ok(Some((k_cached, v_cached))) => {
                    let slot = start_pos % block_size;
                    if let (Ok(k_slot), Ok(v_slot)) =
                        (k_cached.narrow(2, slot, 1), v_cached.narrow(2, slot, 1))
                    {
                        if let Err(e) =
                            hook.inject_kv_tensor(handle, layer_idx, start_pos, &k_slot, &v_slot)
                        {
                            tracing::warn!(
                                "Failed to inject KV tensor for layer {}: {:?}",
                                layer_idx,
                                e
                            );
                            tensor_path_succeeded = false;
                            break;
                        }
                    } else {
                        tensor_path_succeeded = false;
                        break;
                    }
                }
                _ => {
                    tensor_path_succeeded = false;
                    break;
                }
            }
        }

        if tensor_path_succeeded {
            return Ok(());
        }

        for layer_idx in 0..config.num_layers {
            let (keys, values) = cache.read_kv(layer_idx, &[block_id])?;
            // `read_kv` returns `block_size * kv_dim` elements; we only want
            // the slot for `start_pos` within this block.
            let slot = start_pos % block_size;
            let k_start = slot * kv_dim;
            let k_end = k_start + kv_dim;
            let mut single_k = vec![0.0; kv_dim];
            let mut single_v = vec![0.0; kv_dim];
            if k_end <= keys.len() {
                single_k.copy_from_slice(&keys[k_start..k_end]);
                single_v.copy_from_slice(&values[k_start..k_end]);
            }
            hook.inject_kv(handle, layer_idx, start_pos, &single_k, &single_v, 1)?;
        }
        Ok(())
    }
}

impl EngineExecutor for CandleBatchExecutor {
    fn execute(&self, batch: ExecutionBatch) -> Result<BatchResult> {
        match batch.phase {
            ExecutionPhase::Prefill => {
                let next_tokens = self.forward_prefill(
                    &batch.tokens,
                    &batch.cu_seqlens,
                    &batch.kv_handles,
                    &batch.start_positions,
                    &batch.params,
                    &batch.generated_tokens,
                )?;
                if let Some(ref cache) = self.cache {
                    cache.maintain_long_context();
                }
                Ok(BatchResult {
                    next_tokens,
                    speculative_tokens: None,
                })
            }
            ExecutionPhase::Decode | ExecutionPhase::Decoding => {
                // Use real start_positions from the batch (prompt_len + generated_tokens)
                let start_positions = if !batch.start_positions.is_empty() {
                    batch.start_positions.clone()
                } else {
                    // Fallback: derive from cu_seqlens for backward compatibility
                    batch.cu_seqlens.iter().skip(1).copied().collect()
                };
                let (next_tokens, speculative_tokens) = self.forward_decode(
                    &batch.tokens,
                    &batch.cu_seqlens,
                    &batch.kv_handles,
                    &start_positions,
                    &batch.params,
                    &batch.generated_tokens,
                )?;
                if let Some(ref cache) = self.cache {
                    cache.maintain_long_context();
                }
                Ok(BatchResult {
                    next_tokens,
                    speculative_tokens,
                })
            }
            _ => Err(anyhow!(
                "batch executor does not support phase {:?}",
                batch.phase
            )),
        }
    }

    fn max_batch_size(&self, phase: ExecutionPhase) -> usize {
        match phase {
            ExecutionPhase::Prefill => self.max_prefill_batch,
            ExecutionPhase::Decode | ExecutionPhase::Decoding => self.max_decode_batch,
            _ => 1,
        }
    }
}

/// Token budget tracker for in-flight batching.
///
/// Ensures that the total number of tokens (prefill + decode) in a single
/// scheduling step does not exceed `max_num_tokens`.
pub struct TokenBudget {
    /// Maximum tokens per scheduling step.
    pub max_num_tokens: usize,
}

impl TokenBudget {
    pub fn new(max_num_tokens: usize) -> Self {
        Self { max_num_tokens }
    }

    /// Check if adding `num_prefill_tokens` prefill tokens and `num_decode_tokens`
    /// decode tokens would exceed the budget.
    pub fn fits(&self, num_prefill_tokens: usize, num_decode_tokens: usize) -> bool {
        token_budget_allows(num_prefill_tokens, num_decode_tokens, self.max_num_tokens)
    }

    /// Remaining token budget given current usage.
    pub fn remaining(&self, used_prefill: usize, used_decode: usize) -> usize {
        self.max_num_tokens
            .saturating_sub(used_prefill)
            .saturating_sub(used_decode)
    }
}

/// Sample a token from logits using temperature, top-p, and optional seed.
///
/// When `params` is `None` or temperature is 0, falls back to greedy (argmax).
fn sample_logits(logits: &[f32], params: Option<&GenerationParams>) -> u32 {
    let (temperature, top_p, seed) = match params {
        Some(p) => (p.temperature, p.top_p, p.seed),
        None => (1.0, 1.0, None),
    };

    // Greedy when temperature is effectively 0
    if temperature <= 1e-6 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
    }

    // Apply temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature as f32).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum_exp: f32 = exp.iter().sum();
    let mut probs: Vec<f32> = exp.iter().map(|&v| (v / sum_exp).max(0.0)).collect();

    // Top-p (nucleus) filtering
    if top_p < 1.0 && top_p > 0.0 {
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative > top_p as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out probabilities beyond cutoff
        let allowed: std::collections::HashSet<usize> =
            indexed[..cutoff_idx].iter().map(|&(idx, _)| idx).collect();
        for (i, p) in probs.iter_mut().enumerate() {
            if !allowed.contains(&i) {
                *p = 0.0;
            }
        }

        // Re-normalize
        let new_sum: f32 = probs.iter().sum();
        if new_sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= new_sum;
            }
        }
    }

    // Weighted random sampling with optional seed
    let mut rng_state = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    });

    // Simple LCG PRNG for deterministic sampling with seed
    let r = {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f32 / (u32::MAX as f32)
    };

    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i as u32;
        }
    }

    // Fallback to argmax
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::kv_hook::{InMemoryKvHook, KvHook};
    use crate::scheduler::paged_cache::{LongContextPolicy, PagedCacheConfig};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_token_budget() {
        let budget = TokenBudget::new(4096);
        assert!(budget.fits(1000, 500));
        assert!(budget.fits(2048, 2048));
        assert!(!budget.fits(3000, 2000));
        assert_eq!(budget.remaining(1000, 500), 2596);
        assert_eq!(budget.remaining(4000, 100), 0);
    }

    #[test]
    fn token_budget_fails_closed_on_counter_overflow() {
        let budget = TokenBudget::new(usize::MAX);

        assert!(budget.fits(usize::MAX, 0));
        assert!(!budget.fits(usize::MAX, 1));
        assert_eq!(budget.remaining(usize::MAX, 1), 0);
        assert_eq!(budget.remaining(1, usize::MAX), 0);
    }

    #[test]
    fn test_batch_executor_max_batch() {
        let forward_fn = Box::new(
            |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                Err(anyhow!("stub"))
            },
        );
        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32);
        assert_eq!(executor.max_batch_size(ExecutionPhase::Prefill), 4);
        assert_eq!(executor.max_batch_size(ExecutionPhase::Decode), 32);
    }

    #[test]
    fn test_prefill_uses_per_request_start_positions() {
        let observed_positions = Arc::new(Mutex::new(Vec::new()));
        let observed_positions_clone = Arc::clone(&observed_positions);
        let forward_fn = Box::new(
            move |input: &Tensor, start_pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                observed_positions_clone
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(start_pos);
                let seq_len = input.dim(1)?;
                Tensor::new(vec![0.0f32, 1.0], &Device::Cpu)?
                    .reshape((1, 1, 2))?
                    .broadcast_as((1, seq_len, 2))
                    .map_err(Into::into)
            },
        );
        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32);

        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-1".to_string(), "req-2".to_string()],
                tokens: vec![1, 2, 3, 4, 5, 6],
                cu_seqlens: vec![0, 4, 6],
                kv_handles: vec![10, 11],
                start_positions: vec![0, 4],
                params: vec![
                    GenerationParams {
                        temperature: 0.0,
                        ..Default::default()
                    },
                    GenerationParams {
                        temperature: 0.0,
                        ..Default::default()
                    },
                ],
                generated_tokens: vec![vec![], vec![]],
            })
            .unwrap();

        assert_eq!(
            *observed_positions.lock().unwrap_or_else(|e| e.into_inner()),
            vec![0, 4]
        );
    }

    #[test]
    fn test_executor_runs_long_context_maintenance() {
        let forward_fn = Box::new(
            |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                Tensor::new(vec![0.0f32, 1.0], &Device::Cpu)?
                    .reshape((1, 1, 2))
                    .map_err(Into::into)
            },
        );
        let config = PagedCacheConfig {
            block_size: 4,
            total_blocks: 2,
            num_layers: 1,
            kv_dim: 2,
            kv_dtype: crate::core::quantization::KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::CompactInactive {
                target_free_blocks: 2,
            },
        };
        let cache = Arc::new(PagedAttentionCache::new(config));
        let allocation = cache.allocate("old-req", &[1, 2, 3, 4], 4).unwrap();
        assert_eq!(allocation.allocated_blocks.len(), 2);
        cache.free("old-req");

        let executor =
            CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32).with_cache(Arc::clone(&cache));
        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Decode,
                request_ids: vec!["req-1".to_string()],
                tokens: vec![1],
                cu_seqlens: vec![0, 1],
                kv_handles: vec![allocation.handle],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![vec![]],
            })
            .unwrap();

        assert_eq!(cache.metrics().free_blocks, 2);
    }

    /// Regression test for the placeholder-zero bug: when no `KvHook` is
    /// attached, the executor must NOT write misleading zero KV into the
    /// paged cache. The pool's block allocation still advances normally.
    #[test]
    fn test_executor_without_kv_hook_leaves_paged_cache_empty() {
        let forward_fn = Box::new(
            |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                Tensor::new(vec![0.0f32, 1.0], &Device::Cpu)?
                    .reshape((1, 1, 2))
                    .map_err(Into::into)
            },
        );
        let config = PagedCacheConfig {
            block_size: 4,
            total_blocks: 8,
            num_layers: 2,
            kv_dim: 4,
            kv_dtype: crate::core::quantization::KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::Full,
        };
        let cache = Arc::new(PagedAttentionCache::new(config));
        let alloc = cache.allocate("req-1", &[10, 20, 30, 40], 4).unwrap();
        let handle = alloc.handle;

        let executor =
            CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32).with_cache(Arc::clone(&cache));
        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-1".to_string()],
                tokens: vec![10, 20, 30, 40],
                cu_seqlens: vec![0, 4],
                kv_handles: vec![handle],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![vec![]],
            })
            .unwrap();

        // The pool tracks block allocation regardless of hook. 4 prompt tokens
        // + 4 max_new_tokens = 8 tokens, divided by block_size 4 = 2 blocks.
        assert_eq!(cache.metrics().active_blocks, 2);
        // But the per-layer KV storage must remain empty — no placeholder zeros.
        assert_eq!(cache.layer_block_count(0), 0);
        assert_eq!(cache.layer_block_count(1), 0);
    }

    /// End-to-end proof that the `KvHook` path populates the paged cache with
    /// real KV extracted from the model, and that `paged_attention_forward`
    /// then consumes that KV to produce a numerically meaningful output.
    #[test]
    fn test_executor_with_kv_hook_writes_real_kv() {
        let num_layers = 2;
        let kv_dim = 4;
        let block_size = 4;
        let config = PagedCacheConfig {
            block_size,
            total_blocks: 16,
            num_layers,
            kv_dim,
            kv_dtype: crate::core::quantization::KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::Full,
        };
        let cache = Arc::new(PagedAttentionCache::new(config));
        let hook = Arc::new(InMemoryKvHook::new(num_layers, kv_dim));

        // The forward function simulates a model: it writes deterministic KV
        // into the hook for each layer during forward, mimicking what a real
        // attention layer does internally. The executor's write_request_kv
        // then extracts that KV and persists it into the paged cache.
        let hook_for_forward = Arc::clone(&hook);
        let forward_fn = Box::new(
            move |input: &Tensor, start_pos: usize, handle: Option<usize>| -> Result<Tensor> {
                let seq_len = input.dim(1)?;
                let handle = handle.unwrap_or(0);
                for layer in 0..num_layers {
                    // Layer 0 -> key 1.0, value 2.0
                    // Layer 1 -> key 3.0, value 4.0
                    let key_seed = 1.0 + 2.0 * layer as f32;
                    let value_seed = 2.0 + 2.0 * layer as f32;
                    hook_for_forward.populate_deterministic(
                        handle, layer, start_pos, seq_len, key_seed, value_seed,
                    );
                }
                Tensor::new(vec![0.0f32, 1.0], &Device::Cpu)?
                    .reshape((1, 1, 2))?
                    .broadcast_as((1, seq_len, 2))
                    .map_err(Into::into)
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_cache(Arc::clone(&cache))
            .with_kv_hook(Arc::clone(&hook) as Arc<dyn KvHook>);

        let prompt_tokens = vec![10, 20, 30, 40];
        let alloc = cache.allocate("req-1", &prompt_tokens, 0).unwrap();
        let handle = alloc.handle;

        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-1".to_string()],
                tokens: prompt_tokens,
                cu_seqlens: vec![0, 4],
                kv_handles: vec![handle],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![vec![]],
            })
            .unwrap();

        // Layer 0 should now have one populated block with real KV.
        assert_eq!(cache.layer_block_count(0), 1);
        let block_id = alloc.allocated_blocks[0];

        let (keys_l0, values_l0) = cache.read_kv(0, &[block_id]).unwrap();
        assert_eq!(keys_l0.len(), block_size * kv_dim);
        // Layer 0 key[0] should match the seed (1.0) used by the mock model.
        assert!(
            (keys_l0[0] - 1.0).abs() < 1e-3,
            "layer 0 key[0] expected ~1.0, got {}",
            keys_l0[0]
        );
        assert!(
            (values_l0[0] - 2.0).abs() < 1e-3,
            "layer 0 value[0] expected ~2.0, got {}",
            values_l0[0]
        );

        // Layer 1 should use seed 3.0 for keys and 4.0 for values.
        let (keys_l1, values_l1) = cache.read_kv(1, &[block_id]).unwrap();
        assert!(
            (keys_l1[0] - 3.0).abs() < 1e-3,
            "layer 1 key[0] expected ~3.0, got {}",
            keys_l1[0]
        );
        assert!(
            (values_l1[0] - 4.0).abs() < 1e-3,
            "layer 1 value[0] expected ~4.0, got {}",
            values_l1[0]
        );
    }

    /// Proves the full round-trip: hook -> paged cache -> `paged_attention_forward`.
    /// With K and V all near-constant across tokens and a query that matches the
    /// key direction, attention collapses to roughly `mean(V)`.
    #[test]
    fn test_paged_attention_forward_with_hooked_kv() {
        let num_layers = 1;
        let kv_dim = 4;
        let block_size = 4;
        let config = PagedCacheConfig {
            block_size,
            total_blocks: 16,
            num_layers,
            kv_dim,
            kv_dtype: crate::core::quantization::KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::Full,
        };
        let cache = Arc::new(PagedAttentionCache::new(config));
        let hook = Arc::new(InMemoryKvHook::new(num_layers, kv_dim));

        let hook_for_forward = Arc::clone(&hook);
        let forward_fn = Box::new(
            move |input: &Tensor, start_pos: usize, handle: Option<usize>| -> Result<Tensor> {
                let seq_len = input.dim(1)?;
                let handle = handle.unwrap_or(0);
                hook_for_forward.populate_deterministic(handle, 0, start_pos, seq_len, 1.0, 5.0);
                Tensor::new(vec![0.0f32], &Device::Cpu)?
                    .reshape((1, 1, 1))
                    .map_err(Into::into)
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_cache(Arc::clone(&cache))
            .with_kv_hook(Arc::clone(&hook) as Arc<dyn KvHook>);

        let prompt_tokens = vec![1, 2, 3, 4];
        let alloc = cache.allocate("req-1", &prompt_tokens, 0).unwrap();
        let handle = alloc.handle;

        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-1".to_string()],
                tokens: prompt_tokens,
                cu_seqlens: vec![0, 4],
                kv_handles: vec![handle],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![vec![]],
            })
            .unwrap();

        let block_id = alloc.allocated_blocks[0];

        // Query with a vector that aligns with the key direction (≈1.0 per dim).
        let query = vec![1.0; kv_dim];
        let scale = 1.0 / (kv_dim as f32).sqrt();
        let output = cache
            .paged_attention_forward(0, &query, &[block_id], scale)
            .unwrap();

        assert_eq!(output.len(), kv_dim);

        // All four tokens share nearly-equal K (≈1.0) and V (≈5.0). Uniform
        // softmax means the output is approximately the mean of V vectors,
        // i.e. ~5.0 in every dimension.
        let mean_v = output.iter().sum::<f32>() / output.len() as f32;
        assert!(
            (mean_v - 5.0).abs() < 0.1,
            "expected ~5.0 (mean of V across uniform attention), got {} (output={:?})",
            mean_v,
            output
        );

        // Sanity: if we query with all-zeros (no alignment with K), attention
        // is still uniform because softmax over equal scores is uniform. The
        // output should again approximate mean(V).
        let zero_query = vec![0.0; kv_dim];
        let zero_out = cache
            .paged_attention_forward(0, &zero_query, &[block_id], scale)
            .unwrap();
        let zero_mean = zero_out.iter().sum::<f32>() / zero_out.len() as f32;
        assert!(
            (zero_mean - 5.0).abs() < 0.1,
            "zero-query attention should still collapse to mean(V), got {}",
            zero_mean
        );
    }

    /// The decode path must round-trip KV back into the model via `inject_kv`
    /// when the paged cache holds the data. Verifies the hook receives the
    /// same KV that was extracted during prefill.
    #[test]
    fn test_decode_round_trips_kv_into_hook() {
        let num_layers = 1;
        let kv_dim = 4;
        let block_size = 4;
        let config = PagedCacheConfig {
            block_size,
            total_blocks: 16,
            num_layers,
            kv_dim,
            kv_dtype: crate::core::quantization::KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::Full,
        };
        let cache = Arc::new(PagedAttentionCache::new(config));
        let hook = Arc::new(InMemoryKvHook::new(num_layers, kv_dim));

        let hook_for_forward = Arc::clone(&hook);
        let forward_fn = Box::new(
            move |input: &Tensor, start_pos: usize, handle: Option<usize>| -> Result<Tensor> {
                let seq_len = input.dim(1)?;
                let handle = handle.unwrap_or(0);
                hook_for_forward.populate_deterministic(handle, 0, start_pos, seq_len, 7.0, 9.0);
                Tensor::new(vec![0.0f32], &Device::Cpu)?
                    .reshape((1, 1, 1))
                    .map_err(Into::into)
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_cache(Arc::clone(&cache))
            .with_kv_hook(Arc::clone(&hook) as Arc<dyn KvHook>);

        let prompt_tokens = vec![10, 20, 30, 40];
        let alloc = cache.allocate("req-1", &prompt_tokens, 4).unwrap();
        let handle = alloc.handle;

        // Prefill: writes KV for tokens [0, 4).
        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-1".to_string()],
                tokens: prompt_tokens.clone(),
                cu_seqlens: vec![0, 4],
                kv_handles: vec![handle],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![vec![]],
            })
            .unwrap();

        // Decode: position 4. The executor will read KV for the block owning
        // token_pos=4 (block 1) from the paged cache, slice slot 0, and inject
        // it back into the hook at position 4. With block_size=4, slot=0, the
        // injected KV should equal what we wrote during prefill into block 1's
        // first slot (which was zeros — prefill only ran for tokens [0,4) which
        // is block 0). So we expect zeros injected at pos 4, then the forward
        // overwrites pos 4 with seed 7.0.
        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Decode,
                request_ids: vec!["req-1".to_string()],
                tokens: vec![99],
                cu_seqlens: vec![0, 1],
                kv_handles: vec![handle],
                start_positions: vec![4],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![vec![]],
            })
            .unwrap();

        // After decode, the hook should have KV at position 4 written by the
        // forward pass (key_seed=7.0, value_seed=9.0).
        let (k4, v4) = hook.extract_kv(handle, 0, 4, 1).unwrap();
        assert_eq!(k4.len(), kv_dim);
        assert_eq!(v4.len(), kv_dim);
        assert!(
            (k4[0] - 7.0).abs() < 1e-3,
            "decode should have written key seed 7.0 at pos 4, got {}",
            k4[0]
        );
        assert!(
            (v4[0] - 9.0).abs() < 1e-3,
            "decode should have written value seed 9.0 at pos 4, got {}",
            v4[0]
        );

        // The paged cache should now also hold the decode KV at block 1, slot 0.
        let block1 = alloc.allocated_blocks[1];
        let (bk, bv) = cache.read_kv(0, &[block1]).unwrap();
        let slot0_k = &bk[0..kv_dim];
        let slot0_v = &bv[0..kv_dim];
        assert!(
            (slot0_k[0] - 7.0).abs() < 1e-3,
            "paged cache should hold decode KV at block 1 slot 0, got {}",
            slot0_k[0]
        );
        assert!(
            (slot0_v[0] - 9.0).abs() < 1e-3,
            "paged cache should hold decode value at block 1 slot 0, got {}",
            slot0_v[0]
        );
    }

    #[test]
    fn test_batch_executor_true_batched_forward_prefill_and_decode() {
        let num_layers = 1;
        let kv_dim = 4;
        let block_size = 4;
        let config = PagedCacheConfig {
            block_size,
            total_blocks: 16,
            num_layers,
            kv_dim,
            kv_dtype: crate::core::quantization::KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::Full,
        };
        let cache = Arc::new(PagedAttentionCache::new(config));
        let hook = Arc::new(InMemoryKvHook::new(num_layers, kv_dim));

        // Stub forward_fn (should not be called when forward_batch_fn is present)
        let forward_fn = Box::new(
            |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                panic!("sequential forward_fn called unexpectedly!");
            },
        );

        let observed_shapes = Arc::new(Mutex::new(Vec::new()));
        let observed_shapes_clone = Arc::clone(&observed_shapes);
        let forward_batch_fn = Box::new(
            move |input: &Tensor,
                  start_positions: &[usize],
                  kv_handles: &[usize],
                  _cu_seqlens: &[usize]|
                  -> Result<Tensor> {
                observed_shapes_clone
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((
                        input.dims().to_vec(),
                        start_positions.to_vec(),
                        kv_handles.to_vec(),
                    ));
                let num_tokens = input.dim(0)?;
                let data = vec![0.0f32; num_tokens * 2]; // 2 vocab classes
                Tensor::new(data, &Device::Cpu)?
                    .reshape((num_tokens, 2))
                    .map_err(Into::into)
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_cache(Arc::clone(&cache))
            .with_kv_hook(Arc::clone(&hook) as Arc<dyn KvHook>)
            .with_forward_batch_fn(forward_batch_fn);

        let alloc1 = cache.allocate("req-1", &[10, 20], 0).unwrap();
        let alloc2 = cache.allocate("req-2", &[30, 40, 50], 0).unwrap();

        // 1. Prefill verification
        let batch = ExecutionBatch {
            phase: ExecutionPhase::Prefill,
            request_ids: vec!["req-1".to_string(), "req-2".to_string()],
            tokens: vec![10, 20, 30, 40, 50],
            cu_seqlens: vec![0, 2, 5],
            kv_handles: vec![alloc1.handle, alloc2.handle],
            start_positions: vec![0, 0],
            params: vec![
                GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                },
            ],
            generated_tokens: vec![vec![], vec![]],
        };

        let result = executor.execute(batch).unwrap();
        assert_eq!(result.next_tokens.len(), 2);

        // Verify shape received in forward_batch_fn for prefill: should be [5] (flat packed tokens)
        {
            let observed = observed_shapes.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(observed.len(), 1);
            let (ref dims, ref positions, ref handles) = observed[0];
            assert_eq!(dims, &vec![5]);
            assert_eq!(positions, &vec![0, 0]);
            assert_eq!(handles, &vec![alloc1.handle, alloc2.handle]);
        }

        // 2. Decode verification
        observed_shapes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        let batch_decode = ExecutionBatch {
            phase: ExecutionPhase::Decode,
            request_ids: vec!["req-1".to_string(), "req-2".to_string()],
            tokens: vec![99, 100],
            cu_seqlens: vec![0, 1, 2],
            kv_handles: vec![alloc1.handle, alloc2.handle],
            start_positions: vec![2, 3],
            params: vec![
                GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                },
            ],
            generated_tokens: vec![vec![], vec![]],
        };

        let result_decode = executor.execute(batch_decode).unwrap();
        assert_eq!(result_decode.next_tokens.len(), 2);

        // Verify shape received in forward_batch_fn for decode: should be [2] (flat stacked tokens)
        {
            let observed = observed_shapes.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(observed.len(), 1);
            let (ref dims, ref positions, ref handles) = observed[0];
            assert_eq!(dims, &vec![2]);
            assert_eq!(positions, &vec![2, 3]);
            assert_eq!(handles, &vec![alloc1.handle, alloc2.handle]);
        }
    }

    #[test]
    fn test_speculative_decoding_ngram_verification() {
        let forward_fn = Box::new(
            move |input: &Tensor, _start_pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                let seq_len = input.dim(1)?;
                if seq_len > 1 {
                    let input_vec = input.to_vec2::<u32>()?[0].clone();
                    let mut data = vec![0.0f32; seq_len * 20];
                    for idx in 0..seq_len {
                        let target_tok = if idx + 1 < seq_len {
                            input_vec[idx + 1]
                        } else {
                            13
                        };
                        data[idx * 20 + target_tok as usize] = 10.0;
                    }
                    Tensor::new(data, &Device::Cpu)?
                        .reshape((1, seq_len, 20))
                        .map_err(Into::into)
                } else {
                    let mut data = vec![0.0f32; 20];
                    data[11] = 10.0;
                    Tensor::new(data, &Device::Cpu)?
                        .reshape((1, 1, 20))
                        .map_err(Into::into)
                }
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_speculative_mode(SpeculativeMode::NGram {
                ngram_order: 3,
                num_speculative: 2,
            });
        let batch = ExecutionBatch {
            phase: ExecutionPhase::Decode,
            request_ids: vec!["req-1".to_string()],
            tokens: vec![10],
            cu_seqlens: vec![0, 1],
            kv_handles: vec![99],
            start_positions: vec![8],
            params: vec![GenerationParams {
                temperature: 0.0,
                ..Default::default()
            }],
            // The authoritative history ends with the current token. Its
            // suffix [2, 3, 10] has an earlier continuation [11, 12].
            generated_tokens: vec![vec![2, 3, 10, 11, 12, 1, 2, 3, 10]],
        };

        let result = executor.execute(batch).unwrap();

        assert_eq!(result.next_tokens, vec![11]);
        assert_eq!(result.speculative_tokens, Some(vec![vec![12, 13]]));
    }

    /// Verify that `response_format = JsonObject` on the IFB hot path
    /// filters out tokens that cannot start a valid JSON object.
    ///
    /// The forward_fn returns logits that strongly favour token 1 ("a"),
    /// but "a" is not a valid JSON object start. With grammar support
    /// attached, the executor should sample token 0 ("{") instead.
    #[test]
    fn test_batch_executor_grammar_filtering_prefill() {
        // vocab: 0="{", 1="a", 2="}", 3..9=""
        let vocab: Vec<String> = vec![
            "{".to_string(),
            "a".to_string(),
            "}".to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ];
        let eos: Vec<u32> = vec![99]; // out of vocab, no EOS effect

        let forward_fn = Box::new(
            |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                // seq_len inferred from input dim(1); return logits favouring "a" (token 1)
                let seq_len = _input.dim(1)?;
                // logits: [0.0, 10.0, 0.0, 0.0, ...] for each position
                let mut row = [0.0f32; 10];
                row[1] = 10.0; // "a" has the highest logit
                let all: Vec<f32> = row.repeat(seq_len);
                Tensor::new(all, &Device::Cpu)?
                    .reshape((1, seq_len, 10))
                    .map_err(Into::into)
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_grammar_support(vocab, eos);

        // Without grammar: token 1 ("a") would be sampled (greedy, temp=0)
        let result_no_grammar = {
            let forward_fn2 = Box::new(
                |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                    let seq_len = _input.dim(1)?;
                    let mut row = [0.0f32; 10];
                    row[1] = 10.0;
                    let all: Vec<f32> = row.repeat(seq_len);
                    Tensor::new(all, &Device::Cpu)?
                        .reshape((1, seq_len, 10))
                        .map_err(Into::into)
                },
            );
            let exec = CandleBatchExecutor::new(forward_fn2, Device::Cpu, 4, 32);
            exec.execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-no-grammar".to_string()],
                tokens: vec![1, 2],
                cu_seqlens: vec![0, 2],
                kv_handles: vec![0],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![],
            })
            .unwrap()
        };
        assert_eq!(result_no_grammar.next_tokens, vec![1]); // "a"

        // With grammar: token 1 ("a") is filtered out, token 0 ("{") wins
        let result = executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-grammar".to_string()],
                tokens: vec![1, 2],
                cu_seqlens: vec![0, 2],
                kv_handles: vec![0],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    response_format: Some(ResponseFormat::JsonObject),
                    ..Default::default()
                }],
                generated_tokens: vec![],
            })
            .unwrap();
        assert_eq!(result.next_tokens, vec![0]); // "{"
    }

    #[test]
    fn resumed_prefill_uses_authoritative_structured_history() {
        let vocab = vec![
            "{".to_string(),
            "a".to_string(),
            "}".to_string(),
            String::new(),
            "\"".to_string(),
            " ".to_string(),
        ];
        let forward_fn = Box::new(
            |input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                let mut row = [0.0_f32; 6];
                row[0] = 5.0;
                row[1] = 10.0;
                row[2] = 2.0;
                Tensor::new(row.repeat(input.dim(1)?), &Device::Cpu)?
                    .reshape((1, input.dim(1)?, 6))
                    .map_err(Into::into)
            },
        );
        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_grammar_support(vocab, vec![99]);

        let result = executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["resumed-structured".to_string()],
                tokens: vec![8, 9, 0],
                cu_seqlens: vec![0, 3],
                kv_handles: vec![13],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    response_format: Some(ResponseFormat::JsonObject),
                    ..Default::default()
                }],
                generated_tokens: vec![vec![0]],
            })
            .unwrap();

        assert_eq!(result.next_tokens, vec![2]);
    }

    /// Verify grammar state machine carries across decode steps: after
    /// prefill emits "{", the decode step should only allow tokens that
    /// extend the JSON object (e.g. "}", not "a" outside a string).
    #[test]
    fn test_batch_executor_grammar_filtering_decode_chain() {
        // vocab: 0="{", 1="a", 2="}", 3=":", 4="\"", 5=" "
        let vocab: Vec<String> = vec![
            "{".to_string(),
            "a".to_string(),
            "}".to_string(),
            ":".to_string(),
            "\"".to_string(),
            " ".to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ];
        let eos: Vec<u32> = vec![99];

        // forward_fn returns logits favouring "a" (token 1) at every step,
        // but "{" (token 0) has a high logit too so it wins after "a" is
        // filtered out.
        let forward_fn = Box::new(
            |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                let seq_len = _input.dim(1).unwrap_or(1).max(1);
                let mut row = [0.0f32; 10];
                row[0] = 5.0; // "{" — high, wins after "a" is filtered
                row[1] = 10.0; // "a" — highest, but invalid JSON start
                row[2] = 2.0; // "}"
                row[4] = 1.0; // "\""
                let all: Vec<f32> = row.repeat(seq_len);
                Tensor::new(all, &Device::Cpu)?
                    .reshape((1, seq_len, 10))
                    .map_err(Into::into)
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_grammar_support(vocab, eos);

        // Prefill: should emit "{" (token 0) since "a" is filtered
        let prefill_result = executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-chain".to_string()],
                tokens: vec![1, 2],
                cu_seqlens: vec![0, 2],
                kv_handles: vec![0],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    response_format: Some(ResponseFormat::JsonObject),
                    ..Default::default()
                }],
                generated_tokens: vec![],
            })
            .unwrap();
        assert_eq!(prefill_result.next_tokens, vec![0]); // "{"

        // Decode: generated_text is now "{". "a" is still not valid outside
        // a string in JSON (would need a key in quotes first). The valid
        // continuations are "}" (2) or "\"" (4). "a" (1) is filtered.
        let decode_result = executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Decode,
                request_ids: vec!["req-chain".to_string()],
                tokens: vec![0], // the "{" token from prefill
                cu_seqlens: vec![0, 1],
                kv_handles: vec![0],
                start_positions: vec![3], // prompt_len(2) + generated(1)
                params: vec![GenerationParams {
                    temperature: 0.0,
                    response_format: Some(ResponseFormat::JsonObject),
                    ..Default::default()
                }],
                generated_tokens: vec![vec![0]], // "{" from prefill
            })
            .unwrap();
        // After "{", "a" (1) is invalid. The highest valid token is
        // "}" (2, logit 2.0) — "\""(4, logit 1.0) is lower.
        assert_ne!(decode_result.next_tokens[0], 1); // not "a"
        assert_eq!(decode_result.next_tokens[0], 2); // "}"
    }

    #[test]
    fn batched_structured_decode_keeps_authoritative_histories_independent() {
        // vocab: 0="{", 1="a", 2="}", 3=":", 4="\"", 5=" "
        let vocab = vec!["{", "a", "}", ":", "\"", " "]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let forward_fn = Box::new(
            |_input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                Err(anyhow!("scalar forward must not run"))
            },
        );
        let forward_batch_fn = Box::new(
            |input: &Tensor,
             _positions: &[usize],
             _handles: &[usize],
             _cu_seqlens: &[usize]|
             -> Result<Tensor> {
                let mut row = [0.0_f32; 6];
                row[0] = 9.0; // invalid after both prefixes
                row[1] = 10.0; // invalid outside a JSON string
                row[2] = 7.0; // valid after "{"
                row[3] = 8.0; // valid after "{\"a\""
                row[4] = 6.0;
                Tensor::new(row.repeat(input.dim(0)?), &Device::Cpu)?
                    .reshape((input.dim(0)?, 6))
                    .map_err(Into::into)
            },
        );
        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_grammar_support(vocab, vec![99])
            .with_forward_batch_fn(forward_batch_fn);

        let result = executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Decode,
                request_ids: vec!["object-end".to_string(), "object-value".to_string()],
                tokens: vec![0, 4],
                cu_seqlens: vec![0, 1, 2],
                kv_handles: vec![10, 11],
                start_positions: vec![1, 4],
                params: vec![
                    GenerationParams {
                        temperature: 0.0,
                        response_format: Some(ResponseFormat::JsonObject),
                        ..Default::default()
                    },
                    GenerationParams {
                        temperature: 0.0,
                        response_format: Some(ResponseFormat::JsonObject),
                        ..Default::default()
                    },
                ],
                generated_tokens: vec![vec![0], vec![0, 4, 1, 4]],
            })
            .unwrap();

        assert_eq!(result.next_tokens, vec![2, 3]);
        assert_eq!(result.speculative_tokens, None);
    }

    #[test]
    fn structured_decode_disables_unconstrained_speculative_tokens() {
        let vocab = vec!["{", "a", "}", "\"", ":", " "]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let forward_fn = Box::new(
            |input: &Tensor, _pos: usize, _handle: Option<usize>| -> Result<Tensor> {
                let seq_len = input.dim(1)?;
                if seq_len > 1 {
                    let input_tokens = input.to_vec2::<u32>()?[0].clone();
                    let mut logits = vec![0.0_f32; seq_len * 6];
                    for position in 0..seq_len {
                        let target = input_tokens.get(position + 1).copied().unwrap_or(3);
                        logits[position * 6 + target as usize] = 10.0;
                    }
                    Tensor::new(logits, &Device::Cpu)?
                        .reshape((1, seq_len, 6))
                        .map_err(Into::into)
                } else {
                    let mut logits = vec![0.0_f32; 6];
                    logits[2] = 10.0; // EOS is invalid before the object closes
                    logits[1] = 8.0; // continue the current string
                    Tensor::new(logits, &Device::Cpu)?
                        .reshape((1, 1, 6))
                        .map_err(Into::into)
                }
            },
        );
        let executor = CandleBatchExecutor::new(forward_fn, Device::Cpu, 4, 32)
            .with_grammar_support(vocab, vec![2])
            .with_speculative_mode(SpeculativeMode::NGram {
                ngram_order: 3,
                num_speculative: 2,
            });

        let result = executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Decode,
                request_ids: vec!["structured-speculation".to_string()],
                tokens: vec![1],
                cu_seqlens: vec![0, 1],
                kv_handles: vec![12],
                start_positions: vec![7],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    response_format: Some(ResponseFormat::JsonObject),
                    ..Default::default()
                }],
                // Repeated string content would produce an n-gram draft if
                // structured generation did not force single-token checking.
                generated_tokens: vec![vec![0, 3, 1, 1, 1, 1, 1]],
            })
            .unwrap();

        assert_eq!(result.next_tokens, vec![1]);
        assert_eq!(result.speculative_tokens, None);
    }
}
