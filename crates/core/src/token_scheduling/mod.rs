//! Token-level scheduling — Aegaeon-inspired scheduling primitives.
//!
//! This module provides the building blocks for fine-grained token-level scheduling:
//!
//! - **Chunked Prefill** (`chunked_prefill`): Split long prompts into chunks so prefill
//!   can interleave with decode tokens in the same forward pass.
//! - **Preemption** (`preemption`): Suspend low-priority decode requests to make room
//!   for high-priority arrivals.
//! - **Rate Limiter** (`rate_limiter`): Per-model token-bucket rate limiting for
//!   noisy-neighbor protection.
//! - **Priority Eviction** (`priority_eviction`): Aegaeon-style KV cache eviction with
//!   admission control based on HBM utilization.

pub mod chunked_prefill;
pub mod preemption;
pub mod priority_eviction;
pub mod rate_limiter;

use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

fn default_max_prefill_tokens_per_step() -> usize {
    4096
}

fn default_max_decode_tokens_per_step() -> usize {
    256
}

fn default_max_total_tokens_per_step() -> usize {
    4096
}

fn default_max_concurrent_segments() -> usize {
    8
}

fn default_decode_quantum_tokens() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenSchedulingConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_max_prefill_tokens_per_step")]
    pub max_prefill_tokens_per_step: usize,
    #[serde(default = "default_max_decode_tokens_per_step")]
    pub max_decode_tokens_per_step: usize,
    #[serde(default = "default_max_total_tokens_per_step")]
    pub max_total_tokens_per_step: usize,
    #[serde(default = "default_max_concurrent_segments")]
    pub max_concurrent_segments: usize,
    #[serde(default = "default_decode_quantum_tokens")]
    pub decode_quantum_tokens: usize,

    /// Chunked prefill configuration from Aegaeon section 4.1.
    #[serde(default)]
    pub chunked_prefill: chunked_prefill::ChunkedPrefillConfig,
    /// Preemption configuration from Aegaeon section 4.
    #[serde(default)]
    pub preemption: preemption::PreemptionConfig,
    /// Per-model token-bucket rate limiting from Aegaeon section 3.
    #[serde(default)]
    pub rate_limiter: rate_limiter::RateLimiterConfig,
    /// KV-cache priority-eviction configuration from Aegaeon section 5.2.
    #[serde(default)]
    pub kv_eviction: priority_eviction::KvEvictionConfig,
}

impl Default for TokenSchedulingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_prefill_tokens_per_step: default_max_prefill_tokens_per_step(),
            max_decode_tokens_per_step: default_max_decode_tokens_per_step(),
            max_total_tokens_per_step: default_max_total_tokens_per_step(),
            max_concurrent_segments: default_max_concurrent_segments(),
            decode_quantum_tokens: default_decode_quantum_tokens(),
            chunked_prefill: chunked_prefill::ChunkedPrefillConfig::default(),
            preemption: preemption::PreemptionConfig::default(),
            rate_limiter: rate_limiter::RateLimiterConfig::default(),
            kv_eviction: priority_eviction::KvEvictionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBudget {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub total_tokens: usize,
    pub concurrent_segments: usize,
    pub decode_quantum_tokens: usize,
}

impl TokenSchedulingConfig {
    pub fn budget(&self) -> TokenBudget {
        if !self.enabled {
            return TokenBudget {
                prefill_tokens: usize::MAX,
                decode_tokens: usize::MAX,
                total_tokens: usize::MAX,
                concurrent_segments: default_max_concurrent_segments(),
                decode_quantum_tokens: usize::MAX,
            };
        }

        TokenBudget {
            prefill_tokens: self
                .max_prefill_tokens_per_step
                .min(self.max_total_tokens_per_step)
                .max(1),
            decode_tokens: self
                .max_decode_tokens_per_step
                .min(self.max_total_tokens_per_step)
                .max(1),
            total_tokens: self.max_total_tokens_per_step.max(1),
            concurrent_segments: self.max_concurrent_segments.max(1),
            decode_quantum_tokens: self.decode_quantum_tokens.max(1),
        }
    }

    pub fn phase_budget(&self, phase: TokenPhase) -> usize {
        let budget = self.budget();
        match phase {
            TokenPhase::Prefill => budget.prefill_tokens,
            TokenPhase::Decode => budget.decode_tokens,
        }
    }

    pub fn clamp_decode_request(&self, requested_tokens: usize) -> usize {
        if !self.enabled {
            requested_tokens
        } else {
            requested_tokens.min(self.budget().decode_quantum_tokens)
        }
    }
}

#[derive(Debug, Default)]
pub struct TokenAdmission {
    used_prefill_tokens: usize,
    used_decode_tokens: usize,
}

impl TokenAdmission {
    pub fn reset(&mut self) {
        self.used_prefill_tokens = 0;
        self.used_decode_tokens = 0;
    }

    pub fn try_reserve(
        &mut self,
        config: &TokenSchedulingConfig,
        phase: TokenPhase,
        tokens: usize,
    ) -> bool {
        let tokens = tokens.max(1);
        if !config.enabled {
            return true;
        }

        let budget = config.budget();
        let used_total = self.used_prefill_tokens + self.used_decode_tokens;
        if used_total + tokens > budget.total_tokens {
            return false;
        }

        match phase {
            TokenPhase::Prefill => {
                if self.used_prefill_tokens + tokens > budget.prefill_tokens {
                    return false;
                }
                self.used_prefill_tokens += tokens;
            }
            TokenPhase::Decode => {
                if self.used_decode_tokens + tokens > budget.decode_tokens {
                    return false;
                }
                self.used_decode_tokens += tokens;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_clamps_phase_limits_to_total() {
        let config = TokenSchedulingConfig {
            max_prefill_tokens_per_step: 100,
            max_decode_tokens_per_step: 80,
            max_total_tokens_per_step: 64,
            ..Default::default()
        };

        let budget = config.budget();
        assert_eq!(budget.prefill_tokens, 64);
        assert_eq!(budget.decode_tokens, 64);
        assert_eq!(budget.concurrent_segments, 8);
    }

    #[test]
    fn admission_tracks_phase_and_total_budget() {
        let config = TokenSchedulingConfig {
            max_prefill_tokens_per_step: 8,
            max_decode_tokens_per_step: 4,
            max_total_tokens_per_step: 10,
            ..Default::default()
        };
        let mut admission = TokenAdmission::default();

        assert!(admission.try_reserve(&config, TokenPhase::Prefill, 8));
        assert!(!admission.try_reserve(&config, TokenPhase::Decode, 4));
        assert!(admission.try_reserve(&config, TokenPhase::Decode, 2));
    }
}
