//! Speculative decoding strategies for accelerating LLM inference.
//!
//! Provides a framework for draft-then-verify decoding with support for:
//! - N-gram speculative decoding (no draft model required)
//! - Draft model speculative decoding (uses a smaller model for proposals)

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use bloomai_core::DeviceKind;

/// Result of a speculative decoding step.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Tokens accepted by verification.
    pub accepted_tokens: Vec<u32>,
    /// Number of draft tokens that were accepted.
    pub num_accepted: usize,
    /// Whether the generation is complete (EOS or max_tokens reached).
    pub is_finished: bool,
}

/// Trait for speculative decoding strategies.
///
/// A strategy proposes candidate tokens that are then verified by the
/// target (large) model. If accepted, multiple tokens are emitted per
/// target model forward pass.
pub trait SpeculativeStrategy: Send + Sync {
    /// Propose `n` candidate tokens given the current context.
    fn propose(&self, context: &[u32], n: usize) -> Result<Vec<u32>>;

    /// Propose tokens with the draft model's raw logits for each position.
    ///
    /// Returns `(token_id, logits)` pairs. The logits are the draft model's
    /// raw output for that position (before temperature/softmax). The caller
    /// uses these to compute draft probabilities `q = softmax(logits/T)` for
    /// rejection sampling via [`verify_with_rejection_sampling`].
    ///
    /// When the logits vector is empty, no distribution is available and the
    /// caller should fall back to greedy verification ([`verify_greedy_tokens`]).
    /// This is the default behaviour — strategies that can compute real
    /// probabilities (e.g. [`DraftModelStrategy`]) should override this.
    ///
    /// `temperature` should match the target model's sampling temperature so
    /// that draft and target distributions are comparable.
    fn propose_with_logits(
        &self,
        context: &[u32],
        n: usize,
        _temperature: f64,
    ) -> Result<Vec<(u32, Vec<f32>)>> {
        let tokens = self.propose(context, n)?;
        Ok(tokens.into_iter().map(|t| (t, Vec::new())).collect())
    }

    /// Update the strategy's internal context/state with newly generated tokens.
    fn update_context(&self, _new_tokens: &[u32]) {}

    /// Record the acceptance rate of the last proposal to dynamically adjust speculative length.
    fn record_acceptance(&self, _accepted_count: usize, _proposed_count: usize) {}

    /// Get the current dynamically adjusted number of speculative tokens to propose.
    fn current_speculative_limit(&self, default_limit: usize) -> usize {
        default_limit
    }

    /// Name of this strategy for logging/metrics.
    fn name(&self) -> &'static str;
}

/// N-gram speculative decoding strategy.
///
/// Scans the prompt + generated tokens for matching n-gram suffixes and
/// proposes the tokens that followed the match in the original context.
/// This requires no additional model and works well for repetitive or
/// structured text (code, templates, etc.).
pub struct NGramStrategy {
    /// Context window to search in (prompt + generated tokens).
    context: Mutex<Vec<u32>>,
    /// N-gram order to match (default: 4).
    ngram_order: usize,
    /// Exponential moving average of acceptance rate (0.0 to 1.0).
    acceptance_rate_ema: Mutex<f32>,
    /// Dynamically adjusted limit of speculative tokens to propose.
    dynamic_limit: Mutex<usize>,
}

impl NGramStrategy {
    pub fn new(ngram_order: usize) -> Self {
        Self {
            context: Mutex::new(Vec::new()),
            ngram_order: ngram_order.max(2),
            acceptance_rate_ema: Mutex::new(0.8),
            dynamic_limit: Mutex::new(5), // default default_limit
        }
    }

    /// Update the context with newly generated tokens.
    pub fn update_context(&self, new_tokens: &[u32]) {
        let mut ctx = self.context.lock().unwrap_or_else(|e| e.into_inner());
        ctx.extend_from_slice(new_tokens);
    }

    /// Set the initial context (prompt tokens).
    pub fn set_context(&self, tokens: &[u32]) {
        let mut ctx = self.context.lock().unwrap_or_else(|e| e.into_inner());
        *ctx = tokens.to_vec();
    }

    /// Search for matching n-gram in the context and return following tokens.
    fn find_ngram_continuation(&self, suffix: &[u32], max_tokens: usize) -> Vec<u32> {
        let ctx = self.context.lock().unwrap_or_else(|e| e.into_inner());
        if ctx.len() < self.ngram_order + 1 {
            return Vec::new();
        }

        let search_len = suffix.len().min(self.ngram_order);
        let search_suffix = &suffix[suffix.len() - search_len..];

        // Search backwards from the end for the most recent match
        for window_start in (0..ctx.len() - search_len).rev() {
            let window = &ctx[window_start..window_start + search_len];
            if window == search_suffix {
                // Found a match — collect continuation tokens
                let cont_start = window_start + search_len;
                let cont_end = (cont_start + max_tokens).min(ctx.len());
                if cont_start < ctx.len() {
                    return ctx[cont_start..cont_end].to_vec();
                }
            }
        }

        Vec::new()
    }
}

impl SpeculativeStrategy for NGramStrategy {
    fn propose(&self, context: &[u32], n: usize) -> Result<Vec<u32>> {
        let proposed = self.find_ngram_continuation(context, n);
        Ok(proposed)
    }

    fn update_context(&self, new_tokens: &[u32]) {
        self.update_context(new_tokens);
    }

    fn record_acceptance(&self, accepted_count: usize, proposed_count: usize) {
        if proposed_count == 0 {
            return;
        }
        let rate = accepted_count as f32 / proposed_count as f32;
        let mut ema = self
            .acceptance_rate_ema
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *ema = 0.8 * (*ema) + 0.2 * rate;

        let mut limit = self.dynamic_limit.lock().unwrap_or_else(|e| e.into_inner());
        if *ema < 0.3 {
            *limit = (*limit).saturating_sub(1).max(1);
        } else if *ema > 0.7 {
            // Let's cap the dynamic limit at 10 to prevent infinite growth
            *limit = (*limit + 1).min(10);
        }
    }

    fn current_speculative_limit(&self, default_limit: usize) -> usize {
        let limit = *self.dynamic_limit.lock().unwrap_or_else(|e| e.into_inner());
        limit.min(default_limit)
    }

    fn name(&self) -> &'static str {
        "ngram"
    }
}

/// Draft model speculative decoding strategy.
///
/// Uses a smaller "draft" model to propose tokens, which are then verified
/// by the larger target model. The draft model runs autoregressive generation
/// for `n` steps, and the target model verifies all `n` tokens in a single
/// forward pass using rejection sampling.
pub struct DraftModelStrategy {
    /// Number of speculative tokens to propose per step.
    num_speculative: usize,
    /// Path to the draft model.
    draft_model_path: String,
    /// Whether the draft model is loaded.
    is_loaded: Mutex<bool>,
    /// The loaded draft model instance.
    draft_model: Mutex<Option<Box<dyn crate::core::model::LoadedModel>>>,
    /// Device kind to load the draft model on.
    device_kind: DeviceKind,
    /// Exponential moving average of acceptance rate (0.0 to 1.0).
    acceptance_rate_ema: Mutex<f32>,
    /// Dynamically adjusted limit of speculative tokens to propose.
    dynamic_limit: Mutex<usize>,
}

impl DraftModelStrategy {
    pub fn new(draft_model_path: String, num_speculative: usize, device_kind: DeviceKind) -> Self {
        Self {
            num_speculative,
            draft_model_path,
            is_loaded: Mutex::new(false),
            draft_model: Mutex::new(None),
            device_kind,
            acceptance_rate_ema: Mutex::new(0.8),
            dynamic_limit: Mutex::new(num_speculative),
        }
    }

    /// Mark the draft model as loaded.
    pub fn mark_loaded(&self) {
        *self.is_loaded.lock().unwrap_or_else(|e| e.into_inner()) = true;
    }

    /// Whether the draft model is loaded and ready.
    pub fn is_ready(&self) -> bool {
        *self.is_loaded.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Path to the draft model.
    pub fn model_path(&self) -> &str {
        &self.draft_model_path
    }

    /// Number of speculative tokens per step.
    pub fn num_speculative(&self) -> usize {
        self.num_speculative
    }
}

#[cfg(feature = "candle-engine")]
fn get_candle_device(kind: DeviceKind) -> Result<candle_core::Device> {
    match kind {
        DeviceKind::Cpu => Ok(candle_core::Device::Cpu),
        DeviceKind::Gpu => {
            #[cfg(feature = "cuda")]
            {
                candle_core::Device::new_cuda(0)
                    .map_err(|e| anyhow!("failed to initialize CUDA device: {}", e))
            }
            #[cfg(feature = "metal")]
            {
                candle_core::Device::new_metal(0)
                    .map_err(|e| anyhow!("failed to initialize Metal device: {}", e))
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            {
                Err(anyhow!(
                    "GPU backend (CUDA/Metal) not compiled; please rebuild with features"
                ))
            }
        }
        _ => Err(anyhow!("unsupported device kind for Candle draft model")),
    }
}

#[cfg(feature = "candle-engine")]
fn get_greedy_token(logits: &candle_core::Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    let last_logits = if logits.rank() >= 2 {
        logits.get(logits.dim(0)? - 1)?
    } else {
        logits
    };
    let logits_f32 = last_logits.to_dtype(candle_core::DType::F32)?;
    let logits_vec = logits_f32.to_vec1::<f32>()?;
    logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| anyhow!("empty logits tensor"))
}

/// Extract the last position's logits as a `Vec<f32>` from a forward output.
///
/// The forward output typically has shape `[1, seq_len, vocab_size]` or
/// `[1, 1, vocab_size]`. This squeezes the batch dim and takes the last
/// position along the sequence dim.
#[cfg(feature = "candle-engine")]
fn get_last_logits_vec(logits: &candle_core::Tensor) -> Result<Vec<f32>> {
    let logits = logits.squeeze(0)?;
    let last = if logits.rank() >= 2 {
        logits.get(logits.dim(0)? - 1)?
    } else {
        logits
    };
    let last_f32 = last.to_dtype(candle_core::DType::F32)?;
    Ok(last_f32.to_vec1::<f32>()?)
}

/// Compute `softmax(logits / temperature)` and return as `Vec<f32>`.
fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = if temperature > 0.0 { temperature } else { 1.0 };
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temp).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum_exp: f32 = exp.iter().sum();
    if sum_exp <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exp.iter().map(|&v| (v / sum_exp).max(0.0)).collect()
}

/// Return the argmax index of `logits` (ties resolved to the first occurrence).
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Advance the LCG PRNG state and return a `Uniform(0, 1)` float.
///
/// Uses the same LCG constants as `batch_executor::sample_logits` for
/// consistency across the codebase.
fn lcg_next_f32(rng_state: &mut u64) -> f32 {
    *rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*rng_state >> 33) as f32 / (u32::MAX as f32)
}

/// Sample a token from raw logits using temperature scaling and softmax.
///
/// When `temperature <= 0`, falls back to argmax (greedy).
fn sample_from_logits(logits: &[f32], temperature: f64, rng_state: &mut u64) -> u32 {
    if temperature <= 1e-6 {
        return argmax(logits);
    }
    let probs = softmax(logits, temperature as f32);
    let r = lcg_next_f32(rng_state);
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i as u32;
        }
    }
    argmax(&probs)
}

impl SpeculativeStrategy for DraftModelStrategy {
    fn propose(&self, context: &[u32], n: usize) -> Result<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        // Lazy load the model on first call
        {
            let mut model_guard = self.draft_model.lock().unwrap_or_else(|e| e.into_inner());
            if model_guard.is_none() {
                #[cfg(feature = "candle-engine")]
                {
                    use crate::engine::Engine;
                    use crate::executor::candle::CandleEngine;
                    use std::path::Path;
                    let engine = CandleEngine;
                    let loaded =
                        engine.load(Path::new(&self.draft_model_path), self.device_kind)?;
                    *model_guard = Some(loaded);
                    *self.is_loaded.lock().unwrap_or_else(|e| e.into_inner()) = true;
                }
                #[cfg(not(feature = "candle-engine"))]
                {
                    bail!(
                        "Candle engine feature is not enabled; draft model loading is unsupported."
                    );
                }
            }
        }

        #[cfg(feature = "candle-engine")]
        {
            use candle_core::Tensor;
            let device = get_candle_device(self.device_kind)?;
            let model_guard = self.draft_model.lock().unwrap_or_else(|e| e.into_inner());
            let draft_model = model_guard
                .as_ref()
                .ok_or_else(|| anyhow!("draft model not loaded"))?;

            // Clear KV cache for a fresh proposal step
            draft_model.clear_kv_cache();

            let mut proposed_tokens = Vec::with_capacity(n);
            let mut current_pos = 0;

            // Step 0: Prefill the context
            let input_ids = Tensor::new(context, &device)?.unsqueeze(0)?;
            let logits = draft_model.forward(&input_ids, current_pos)?;
            let mut last_token = get_greedy_token(&logits)?;
            proposed_tokens.push(last_token);
            current_pos += context.len();

            // Step 1 to n-1: Autoregressive decoding
            for _ in 1..n {
                let input_ids = Tensor::new(&[[last_token]], &device)?;
                let logits = draft_model.forward(&input_ids, current_pos)?;
                last_token = get_greedy_token(&logits)?;
                proposed_tokens.push(last_token);
                current_pos += 1;
            }

            Ok(proposed_tokens)
        }
        #[cfg(not(feature = "candle-engine"))]
        {
            bail!("Candle engine feature is not enabled; draft model speculative decoding is unsupported.");
        }
    }

    fn record_acceptance(&self, accepted_count: usize, proposed_count: usize) {
        if proposed_count == 0 {
            return;
        }
        let rate = accepted_count as f32 / proposed_count as f32;
        let mut ema = self
            .acceptance_rate_ema
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *ema = 0.8 * (*ema) + 0.2 * rate;

        let mut limit = self.dynamic_limit.lock().unwrap_or_else(|e| e.into_inner());
        if *ema < 0.3 {
            *limit = (*limit).saturating_sub(1).max(1);
        } else if *ema > 0.7 {
            *limit = (*limit + 1).min(self.num_speculative);
        }
    }

    fn current_speculative_limit(&self, default_limit: usize) -> usize {
        let limit = *self.dynamic_limit.lock().unwrap_or_else(|e| e.into_inner());
        limit.min(default_limit)
    }

    fn name(&self) -> &'static str {
        "draft_model"
    }

    /// Propose tokens with the draft model's per-position raw logits.
    ///
    /// Mirrors [`Self::propose`] but, instead of discarding the logits, returns
    /// the full last-position logits vector for each proposed token. The caller
    /// (target-side rejection sampler) computes `q = softmax(logits / T)` and
    /// compares against the target distribution `p`.
    ///
    /// The proposed token id at each position is the argmax of that position's
    /// logits, so the behaviour matches `propose` when the caller only consumes
    /// the token ids. When `temperature <= 0`, the caller falls back to greedy
    /// verification, so empty logits are returned to avoid wasted work.
    fn propose_with_logits(
        &self,
        context: &[u32],
        n: usize,
        temperature: f64,
    ) -> Result<Vec<(u32, Vec<f32>)>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        // Greedy path: defer to `propose` and return empty logits so the caller
        // uses `verify_greedy_tokens` rather than rejection sampling.
        if temperature <= 1e-6 {
            let tokens = self.propose(context, n)?;
            return Ok(tokens.into_iter().map(|t| (t, Vec::new())).collect());
        }

        // Lazy load the model on first call (mirrors `propose`).
        {
            let mut model_guard = self.draft_model.lock().unwrap_or_else(|e| e.into_inner());
            if model_guard.is_none() {
                #[cfg(feature = "candle-engine")]
                {
                    use crate::engine::Engine;
                    use crate::executor::candle::CandleEngine;
                    use std::path::Path;
                    let engine = CandleEngine;
                    let loaded =
                        engine.load(Path::new(&self.draft_model_path), self.device_kind)?;
                    *model_guard = Some(loaded);
                    *self.is_loaded.lock().unwrap_or_else(|e| e.into_inner()) = true;
                }
                #[cfg(not(feature = "candle-engine"))]
                {
                    bail!(
                        "Candle engine feature is not enabled; draft model loading is unsupported."
                    );
                }
            }
        }

        #[cfg(feature = "candle-engine")]
        {
            use candle_core::Tensor;
            let device = get_candle_device(self.device_kind)?;
            let model_guard = self.draft_model.lock().unwrap_or_else(|e| e.into_inner());
            let draft_model = model_guard
                .as_ref()
                .ok_or_else(|| anyhow!("draft model not loaded"))?;

            draft_model.clear_kv_cache();

            let mut proposed: Vec<(u32, Vec<f32>)> = Vec::with_capacity(n);
            let mut current_pos = 0;

            // Step 0: Prefill the context.
            let input_ids = Tensor::new(context, &device)?.unsqueeze(0)?;
            let logits = draft_model.forward(&input_ids, current_pos)?;
            let logits_vec = get_last_logits_vec(&logits)?;
            let mut last_token = argmax(&logits_vec);
            proposed.push((last_token, logits_vec));
            current_pos += context.len();

            // Steps 1..n-1: autoregressive decoding, keeping full logits.
            for _ in 1..n {
                let input_ids = Tensor::new(&[[last_token]], &device)?;
                let logits = draft_model.forward(&input_ids, current_pos)?;
                let logits_vec = get_last_logits_vec(&logits)?;
                last_token = argmax(&logits_vec);
                proposed.push((last_token, logits_vec));
                current_pos += 1;
            }

            Ok(proposed)
        }
        #[cfg(not(feature = "candle-engine"))]
        {
            bail!(
                "Candle engine feature is not enabled; draft model speculative decoding is unsupported."
            );
        }
    }
}

/// Verify speculative tokens against the target model.
///
/// Given draft tokens proposed by a speculative strategy, runs the target
/// model forward on the full sequence (prompt + draft tokens) and determines
/// how many tokens to accept using rejection sampling.
///
/// Returns the number of accepted tokens (0 to draft_tokens.len()).
///
/// # Deprecated
///
/// This function predates [`verify_with_rejection_sampling`] and only returns
/// the accepted count — it cannot surface the residual-sampled correction or
/// target-sampled bonus token, which are required for a correct speculative
/// decoding loop. The previous stochastic path used an ad-hoc
/// `exp(-1/T)` threshold that has no theoretical basis; it now delegates to
/// [`verify_with_rejection_sampling`] with a fixed seed so existing callers
/// get correct behaviour, but new code should call
/// [`verify_with_rejection_sampling`] directly to obtain the correction token.
#[deprecated(
    since = "0.2.0",
    note = "use `verify_with_rejection_sampling` instead — it also returns the correction/bonus token"
)]
pub fn verify_speculative_tokens(
    target_logits: &[Vec<f32>],
    draft_tokens: &[u32],
    temperature: f64,
) -> usize {
    if target_logits.is_empty() || draft_tokens.is_empty() {
        return 0;
    }

    // Greedy path: argmax comparison (no draft logits → fallback inside
    // verify_with_rejection_sampling would do the same, but we keep an inline
    // fast path so the deprecation doesn't add observable overhead when T=0).
    if temperature <= 0.0 {
        let mut num_accepted = 0;
        for (i, &draft_tok) in draft_tokens.iter().enumerate() {
            if i >= target_logits.len() || target_logits[i].is_empty() {
                break;
            }
            let target_tok = argmax(&target_logits[i]);
            if draft_tok == target_tok {
                num_accepted += 1;
            } else {
                break;
            }
        }
        return num_accepted;
    }

    // Stochastic path: delegate to the proper rejection sampler. We pass empty
    // draft logits (no draft distribution available through this API), which
    // makes `verify_with_rejection_sampling` fall back to per-position greedy
    // comparison at each step — the only correct behaviour when the draft
    // distribution is unknown. A fixed seed keeps the function deterministic.
    let draft: Vec<(u32, Vec<f32>)> = draft_tokens.iter().map(|t| (*t, Vec::new())).collect();
    let mut rng_state = 0x9E3779B97F4A7C15u64;
    let (accepted, _bonus) =
        verify_with_rejection_sampling(target_logits, &draft, temperature, &mut rng_state);
    accepted
}

/// Verify speculative tokens via standard rejection sampling.
///
/// Implements the canonical speculative-decoding acceptance rule (Leviathan
/// et al., 2022; Chen et al., 2023): for each draft token `x_i` sampled from
/// the draft distribution `q_i`, draw `r ~ Uniform(0, 1)` and accept iff
/// `r < min(1, p_i(x_i) / q_i(x_i))`, where `p_i` is the target distribution
/// at the same position. On rejection, a correction token is sampled from the
/// residual `norm(max(0, p_i - q_i))`. If all draft tokens are accepted, a
/// bonus token is sampled from `p_{n}` (the target distribution one step past
/// the last draft token), when available.
///
/// # Arguments
/// * `target_logits` - Raw target-model logits at each draft position. Must
///   have length `>= draft.len()`. If it has length `draft.len() + 1`, the
///   extra entry is used for the bonus token on full acceptance.
/// * `draft` - `(token_id, draft_logits)` pairs from the draft strategy.
///   When `draft_logits` is empty (e.g. from [`NGramStrategy`]), the position
///   falls back to greedy acceptance against the target argmax.
/// * `temperature` - Sampling temperature; must match between draft and
///   target. When `<= 1e-6`, falls back to greedy verification.
/// * `rng_state` - Mutable LCG PRNG state (see [`lcg_next_f32`]).
///
/// # Returns
/// `(accepted, correction_or_bonus)` where:
/// - `accepted` is the number of draft tokens accepted (0..=draft.len()).
/// - `correction_or_bonus` is `Some(token)` when a token should be emitted
///   beyond the accepted draft prefix: either a residual-sampled correction
///   (on rejection) or a target-sampled bonus (on full acceptance). `None`
///   means no extra token is available (e.g. `target_logits` has no bonus
///   entry, or all inputs were empty).
pub fn verify_with_rejection_sampling(
    target_logits: &[Vec<f32>],
    draft: &[(u32, Vec<f32>)],
    temperature: f64,
    rng_state: &mut u64,
) -> (usize, Option<u32>) {
    if draft.is_empty() || target_logits.is_empty() {
        return (0, None);
    }

    // Greedy fallback: when temperature is effectively zero, stochastic
    // acceptance degenerates to exact-match against the target argmax. The
    // caller should normally use `verify_greedy_tokens` in this regime, but
    // we handle it here for safety.
    if temperature <= 1e-6 {
        let mut accepted = 0;
        for (i, (draft_tok, _)) in draft.iter().enumerate() {
            if i >= target_logits.len() || target_logits[i].is_empty() {
                break;
            }
            let target_tok = argmax(&target_logits[i]);
            if target_tok == *draft_tok {
                accepted += 1;
            } else {
                return (accepted, Some(target_tok));
            }
        }
        if target_logits.len() > accepted && !target_logits[accepted].is_empty() {
            return (accepted, Some(argmax(&target_logits[accepted])));
        }
        return (accepted, None);
    }

    let n = draft.len();
    let mut accepted = 0;

    for i in 0..n {
        if i >= target_logits.len() {
            break;
        }

        let draft_tok = draft[i].0;
        let target_logit_vec = &target_logits[i];

        if target_logit_vec.is_empty() {
            break;
        }

        let draft_logit_vec = &draft[i].1;

        // If draft logits are unavailable (e.g. n-gram strategy) or the draft
        // token index is out of range, fall back to greedy comparison at this
        // position. This keeps the verifier correct without a draft distribution.
        if draft_logit_vec.is_empty()
            || (draft_tok as usize) >= draft_logit_vec.len()
            || (draft_tok as usize) >= target_logit_vec.len()
        {
            let target_tok = argmax(target_logit_vec);
            if target_tok == draft_tok {
                accepted += 1;
                continue;
            } else {
                return (accepted, Some(target_tok));
            }
        }

        let p = softmax(target_logit_vec, temperature as f32);
        let q = softmax(draft_logit_vec, temperature as f32);

        let p_x = p[draft_tok as usize];
        let q_x = q[draft_tok as usize];

        let r = lcg_next_f32(rng_state);
        let accept_prob = if q_x > 0.0 { (p_x / q_x).min(1.0) } else { 1.0 };

        if r < accept_prob {
            accepted += 1;
        } else {
            // Reject: sample correction from the residual `norm(max(0, p - q))`.
            // When the residual sums to zero (draft dominates target everywhere),
            // fall back to sampling from the target distribution directly.
            let residual: Vec<f32> = p
                .iter()
                .zip(q.iter())
                .map(|(&pi, &qi)| (pi - qi).max(0.0))
                .collect();
            let sum: f32 = residual.iter().sum();
            let correction = if sum > 0.0 {
                let r2 = lcg_next_f32(rng_state) * sum;
                let mut cum = 0.0f32;
                let mut chosen = argmax(&residual);
                for (idx, &val) in residual.iter().enumerate() {
                    cum += val;
                    if r2 <= cum {
                        chosen = idx as u32;
                        break;
                    }
                }
                chosen
            } else {
                sample_from_logits(target_logit_vec, temperature, rng_state)
            };
            return (accepted, Some(correction));
        }
    }

    // All draft tokens accepted: sample a bonus token from the target
    // distribution one step past the last accepted draft token, if available.
    if target_logits.len() > accepted && !target_logits[accepted].is_empty() {
        let bonus = sample_from_logits(&target_logits[accepted], temperature, rng_state);
        return (accepted, Some(bonus));
    }

    (accepted, None)
}

/// Speculative decoding mode configuration.
#[derive(Debug, Clone)]
pub enum SpeculativeMode {
    /// No speculative decoding.
    None,
    /// N-gram based speculative decoding.
    NGram {
        /// N-gram order to match.
        ngram_order: usize,
        /// Number of speculative tokens per step.
        num_speculative: usize,
    },
    /// Draft model based speculative decoding.
    DraftModel {
        /// Path to the draft model.
        model_path: String,
        /// Number of speculative tokens per step.
        num_speculative: usize,
    },
    /// Native multi-token-prediction heads attached to the target model.
    Mtp {
        /// Number of auxiliary future tokens to verify per step.
        num_speculative: usize,
    },
}

impl Default for SpeculativeMode {
    fn default() -> Self {
        Self::None
    }
}

impl SpeculativeMode {
    pub fn from_parts(
        mode: &str,
        draft_model: Option<PathBuf>,
        num_speculative: usize,
        ngram_order: usize,
    ) -> Result<Self> {
        let num_speculative = num_speculative.max(1);
        match mode.trim().to_ascii_lowercase().as_str() {
            "" | "none" | "off" | "false" => Ok(Self::None),
            "ngram" | "n-gram" => Ok(Self::NGram {
                ngram_order: ngram_order.max(2),
                num_speculative,
            }),
            "draft" | "draft_model" | "draft-model" => {
                let Some(path) = draft_model else {
                    bail!("--draft-model is required when --speculative=draft");
                };
                Ok(Self::DraftModel {
                    model_path: path.display().to_string(),
                    num_speculative,
                })
            }
            "mtp" | "native-mtp" | "draft-mtp" => Ok(Self::Mtp { num_speculative }),
            other => Err(anyhow!(
                "unsupported speculative decoding mode '{}'; expected none, ngram, draft, or mtp",
                other
            )),
        }
    }

    pub fn from_env() -> Result<Self> {
        let mode = std::env::var("BLOOM_SPECULATIVE").unwrap_or_else(|_| "none".to_string());
        let draft_model = std::env::var_os("BLOOM_DRAFT_MODEL").map(PathBuf::from);
        let num_speculative = std::env::var("BLOOM_NUM_SPECULATIVE_TOKENS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5);
        let ngram_order = std::env::var("BLOOM_SPECULATIVE_NGRAM_ORDER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);
        Self::from_parts(&mode, draft_model, num_speculative, ngram_order)
    }

    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NGram { .. } => "ngram",
            Self::DraftModel { .. } => "draft",
            Self::Mtp { .. } => "mtp",
        }
    }
}

/// Returns true when a user-facing speculative mode string requests native
/// multi-token prediction heads.
pub fn speculative_mode_is_mtp(mode: &str) -> bool {
    matches!(
        mode.trim().to_ascii_lowercase().as_str(),
        "mtp" | "native-mtp" | "draft-mtp"
    )
}

/// Whether a model config advertises native MTP/next-n auxiliary heads.
pub fn config_supports_mtp(config: &serde_json::Value) -> bool {
    fn value_has_mtp_keys(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(map) => map.iter().any(|(key, value)| {
                let key = key.to_ascii_lowercase();
                key.contains("mtp")
                    || key.contains("nextn")
                    || key.contains("next_n")
                    || key.contains("medusa")
                    || key.contains("eagle")
                    || value_has_mtp_keys(value)
            }),
            serde_json::Value::Array(values) => values.iter().any(value_has_mtp_keys),
            _ => false,
        }
    }

    value_has_mtp_keys(config)
}

/// Greedy acceptance count for candidate tokens from verifier logits.
pub fn verify_greedy_tokens<F>(
    num_candidates: usize,
    draft_tokens: &[u32],
    mut token_at: F,
) -> usize
where
    F: FnMut(usize) -> Option<u32>,
{
    let mut accepted = 0;
    for (idx, &draft) in draft_tokens.iter().take(num_candidates).enumerate() {
        if token_at(idx) == Some(draft) {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_strategy_basic() {
        let strategy = NGramStrategy::new(3);
        // Set context with a repeating pattern
        strategy.set_context(&[1, 2, 3, 4, 5, 1, 2, 3, 10, 11]);

        // Given context ending in [1, 2, 3], should propose [10, 11]
        // (matching the earlier [1, 2, 3] -> 10, 11 pattern)
        let proposed = strategy.propose(&[1, 2, 3], 5).unwrap();
        assert_eq!(proposed, vec![10, 11]);
    }

    #[test]
    fn test_ngram_no_match() {
        let strategy = NGramStrategy::new(3);
        strategy.set_context(&[1, 2, 3]);

        // No matching n-gram suffix
        let proposed = strategy.propose(&[99, 98, 97], 5).unwrap();
        assert!(proposed.is_empty());
    }

    #[test]
    fn test_ngram_update_context() {
        let strategy = NGramStrategy::new(2);
        strategy.set_context(&[1, 2, 3]);
        strategy.update_context(&[4, 5]);

        let ctx = strategy.context.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(*ctx, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[allow(deprecated)]
    fn test_verify_greedy_accept_all() {
        // Target model agrees with all draft tokens
        let target_logits = vec![
            make_logits(0, 10), // token 0 has highest logit
            make_logits(1, 10), // token 1 has highest logit
            make_logits(2, 10), // token 2 has highest logit
        ];
        let draft_tokens = vec![0, 1, 2];
        let accepted = verify_speculative_tokens(&target_logits, &draft_tokens, 0.0);
        assert_eq!(accepted, 3);
    }

    #[test]
    #[allow(deprecated)]
    fn test_verify_greedy_reject_second() {
        let target_logits = vec![
            make_logits(0, 10), // agrees with draft token 0
            make_logits(5, 10), // disagrees: target wants 5, draft has 1
        ];
        let draft_tokens = vec![0, 1];
        let accepted = verify_speculative_tokens(&target_logits, &draft_tokens, 0.0);
        assert_eq!(accepted, 1);
    }

    #[test]
    #[allow(deprecated)]
    fn test_verify_empty_input() {
        let accepted = verify_speculative_tokens(&[], &[], 0.0);
        assert_eq!(accepted, 0);
    }

    #[test]
    fn test_speculative_mode_default() {
        let mode = SpeculativeMode::default();
        assert!(matches!(mode, SpeculativeMode::None));
    }

    #[test]
    fn test_speculative_mode_parse_mtp() {
        let mode = SpeculativeMode::from_parts("mtp", None, 3, 4).unwrap();
        assert!(matches!(mode, SpeculativeMode::Mtp { num_speculative: 3 }));
        assert_eq!(mode.label(), "mtp");
    }

    #[test]
    fn test_speculative_mode_is_mtp_accepts_aliases() {
        assert!(speculative_mode_is_mtp("mtp"));
        assert!(speculative_mode_is_mtp(" native-mtp "));
        assert!(speculative_mode_is_mtp("DRAFT-MTP"));
        assert!(!speculative_mode_is_mtp("ngram"));
        assert!(!speculative_mode_is_mtp("none"));
    }

    #[test]
    fn test_config_supports_mtp() {
        let cfg = serde_json::json!({
            "model_type": "qwen3",
            "num_nextn_predict_layers": 1
        });
        assert!(config_supports_mtp(&cfg));

        let plain = serde_json::json!({ "model_type": "qwen3" });
        assert!(!config_supports_mtp(&plain));
    }

    #[test]
    fn test_verify_greedy_tokens_stops_on_first_reject() {
        let accepted = verify_greedy_tokens(4, &[10, 11, 12, 13], |idx| {
            [10, 11, 99, 13].get(idx).copied()
        });
        assert_eq!(accepted, 2);
    }

    #[test]
    fn test_draft_model_strategy() {
        let strategy = DraftModelStrategy::new("/path/to/draft".to_string(), 5, DeviceKind::Cpu);
        assert!(!strategy.is_ready());
        strategy.mark_loaded();
        assert!(strategy.is_ready());
        assert_eq!(strategy.num_speculative(), 5);
        assert_eq!(strategy.name(), "draft_model");
    }

    #[test]
    #[cfg(feature = "candle-engine")]
    fn test_draft_model_strategy_propose() {
        use crate::core::model::{LoadedModel, ModelMetadata};
        use crate::io::{ModelInput, ModelOutput};
        use bloomai_core::{GenerationParams, Modality, ModelManifest};
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        struct MockLoadedModel {
            clear_calls: Arc<AtomicUsize>,
            forward_result: Mutex<Vec<u32>>,
            metadata: ModelMetadata,
        }

        impl LoadedModel for MockLoadedModel {
            fn forward(
                &self,
                _input_ids: &candle_core::Tensor,
                _start_pos: usize,
            ) -> Result<candle_core::Tensor> {
                let mut res = self
                    .forward_result
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let next_tok = if !res.is_empty() { res.remove(0) } else { 0 };
                let mut logits = vec![-10.0f32; 100];
                if (next_tok as usize) < 100 {
                    logits[next_tok as usize] = 10.0;
                }
                let tensor =
                    candle_core::Tensor::new(logits.as_slice(), &candle_core::Device::Cpu)?;
                Ok(tensor.reshape((1, 1, logits.len()))?)
            }

            fn clear_kv_cache(&self) {
                self.clear_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }

            fn metadata(&self) -> &ModelMetadata {
                &self.metadata
            }

            fn infer(&self, _input: ModelInput, _params: &GenerationParams) -> Result<ModelOutput> {
                Ok(ModelOutput {
                    text: None,
                    logits: None,
                    image: None,
                    audio: None,
                    video: None,
                })
            }
        }

        let metadata = ModelMetadata {
            id: "mock".to_string(),
            modality: Modality::Text,
            quantized: false,
            manifest: ModelManifest::default(),
        };

        let clear_calls = Arc::new(AtomicUsize::new(0));
        let mock_model = Box::new(MockLoadedModel {
            clear_calls: clear_calls.clone(),
            forward_result: Mutex::new(vec![42, 43, 44]),
            metadata,
        });

        let strategy = DraftModelStrategy::new("dummy_path".to_string(), 3, DeviceKind::Cpu);
        *strategy
            .draft_model
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(mock_model);
        strategy.mark_loaded();

        let proposed = strategy.propose(&[1, 2, 3], 3).unwrap();
        assert_eq!(proposed, vec![42, 43, 44]);
        assert_eq!(clear_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn test_draft_model_strategy_adaptive_acceptance() {
        let strategy = DraftModelStrategy::new("dummy_path".to_string(), 5, DeviceKind::Cpu);
        assert_eq!(strategy.current_speculative_limit(5), 5);

        // Record zero acceptance multiple times to decay the EMA
        for _ in 0..5 {
            strategy.record_acceptance(0, 5);
        }
        // EMA should decrease, and dynamic limit should decrease
        let limit = strategy.current_speculative_limit(5);
        assert!(limit < 5);

        // Record full acceptance multiple times
        for _ in 0..10 {
            strategy.record_acceptance(5, 5);
        }
        // EMA and dynamic limit should increase back to max
        let limit = strategy.current_speculative_limit(5);
        assert_eq!(limit, 5);
    }

    /// Helper: create logits where token `peak_idx` has the highest value.
    fn make_logits(peak_idx: u32, vocab_size: usize) -> Vec<f32> {
        let mut logits = vec![-10.0f32; vocab_size];
        if (peak_idx as usize) < vocab_size {
            logits[peak_idx as usize] = 10.0;
        }
        logits
    }

    // ---- Rejection sampling tests ----

    #[test]
    fn test_rejection_sampling_empty_inputs() {
        let mut rng = 42;
        let (accepted, bonus) = verify_with_rejection_sampling(&[], &[], 1.0, &mut rng);
        assert_eq!(accepted, 0);
        assert_eq!(bonus, None);
    }

    #[test]
    fn test_rejection_sampling_greedy_fallback() {
        // temperature = 0 → greedy exact-match path
        let target = vec![make_logits(5, 10), make_logits(7, 10)];
        let draft = vec![(5u32, Vec::new()), (7u32, Vec::new())];
        let mut rng = 42;
        let (accepted, bonus) = verify_with_rejection_sampling(&target, &draft, 0.0, &mut rng);
        assert_eq!(accepted, 2);
        // No bonus position available (target.len() == draft.len())
        assert_eq!(bonus, None);
    }

    #[test]
    fn test_rejection_sampling_greedy_reject() {
        // temperature = 0, target disagrees on second token
        let target = vec![make_logits(5, 10), make_logits(9, 10)];
        let draft = vec![(5u32, Vec::new()), (7u32, Vec::new())];
        let mut rng = 42;
        let (accepted, bonus) = verify_with_rejection_sampling(&target, &draft, 0.0, &mut rng);
        assert_eq!(accepted, 1);
        // Bonus/correction from target argmax at rejection point (token 9)
        assert_eq!(bonus, Some(9));
    }

    #[test]
    fn test_rejection_sampling_identical_distributions_accept_all() {
        // When p == q, accept_prob = min(1, p/p) = 1 → all accepted.
        // 4-token vocab, uniform draft + target distributions.
        let uniform = vec![0.0f32; 4];
        let target = vec![uniform.clone(), uniform.clone(), uniform.clone()];
        let draft = vec![
            (0u32, uniform.clone()),
            (1u32, uniform.clone()),
            (2u32, uniform.clone()),
        ];
        let mut rng = 100;
        let (accepted, bonus) = verify_with_rejection_sampling(&target, &draft, 1.0, &mut rng);
        assert_eq!(accepted, 3);
        // Bonus token sampled from target[3] — but there's no target[3]!
        // target.len() == draft.len() == 3, so no bonus.
        assert_eq!(bonus, None);
    }

    #[test]
    fn test_rejection_sampling_identical_with_bonus() {
        // Same as above but with an extra target position → bonus sampled.
        let uniform = vec![0.0f32; 4];
        let target = vec![
            uniform.clone(),
            uniform.clone(),
            uniform.clone(),
            make_logits(2, 4), // bonus position: token 2 dominates
        ];
        let draft = vec![
            (0u32, uniform.clone()),
            (1u32, uniform.clone()),
            (2u32, uniform.clone()),
        ];
        let mut rng = 100;
        let (accepted, bonus) = verify_with_rejection_sampling(&target, &draft, 1.0, &mut rng);
        assert_eq!(accepted, 3);
        // Bonus sampled from target[3] where token 2 has the highest logit
        assert_eq!(bonus, Some(2));
    }

    #[test]
    fn test_rejection_sampling_ngram_fallback_to_greedy() {
        // Draft logits empty (n-gram strategy) → per-position greedy fallback
        let target = vec![make_logits(3, 10), make_logits(4, 10)];
        let draft = vec![
            (3u32, Vec::new()), // matches target argmax → accepted
            (9u32, Vec::new()), // doesn't match → rejected, correction = 4
        ];
        let mut rng = 42;
        let (accepted, bonus) = verify_with_rejection_sampling(&target, &draft, 1.0, &mut rng);
        assert_eq!(accepted, 1);
        assert_eq!(bonus, Some(4));
    }

    #[test]
    fn test_rejection_sampling_draft_dominates_then_reject() {
        // Draft puts high probability on token 0, target puts it on token 1.
        // p(0) ≈ 0, q(0) ≈ 1 → accept_prob = min(1, 0/1) = 0 → always rejected.
        // Correction from residual norm(max(0, p - q)) → token 1.
        let target_logits = vec![10.0f32, 0.0, 0.0, 0.0]; // p ≈ [1, 0, 0, 0]
        let draft_logits = vec![0.0f32, 10.0, 0.0, 0.0]; // q ≈ [0, 1, 0, 0]
        let target = vec![target_logits];
        let draft = vec![(1u32, draft_logits)]; // draft proposes token 1
        let mut rng = 42;
        let (accepted, bonus) = verify_with_rejection_sampling(&target, &draft, 1.0, &mut rng);
        assert_eq!(accepted, 0);
        // Correction should be token 0 (target's argmax, residual mass on 0)
        assert_eq!(bonus, Some(0));
    }

    #[test]
    fn test_rejection_sampling_target_dominates_accept() {
        // Target puts high probability on token 1, draft also proposes token 1
        // but with lower confidence. p(1) >> q(1) → accept_prob = min(1, p/q) = 1.
        let target_logits = vec![0.0f32, 10.0, 0.0, 0.0]; // p ≈ [0, 1, 0, 0]
        let draft_logits = vec![0.0f32, 1.0, 0.0, 0.0]; // q ≈ [0, ~1, 0, 0]
        let target = vec![target_logits];
        let draft = vec![(1u32, draft_logits)];
        let mut rng = 42;
        let (accepted, _bonus) = verify_with_rejection_sampling(&target, &draft, 1.0, &mut rng);
        assert_eq!(accepted, 1);
    }

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs = softmax(&logits, 1.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[], 1.0);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_argmax_simple() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 5.0]), 2);
    }

    #[test]
    fn test_lcg_next_f32_in_range() {
        let mut state = 42u64;
        for _ in 0..100 {
            let r = lcg_next_f32(&mut state);
            assert!((0.0..=1.0).contains(&r), "rng out of range: {}", r);
        }
    }
}
