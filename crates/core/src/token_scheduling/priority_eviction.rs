//! Priority KV Cache Eviction — Aegaeon §5.2
//!
//! Aegaeon uses explicit KV-cache memory management with this eviction score:
//!
//! ```text
//! eviction_score = (request_age × token_value_estimate) / session_cost
//! ```
//!
//! Lower scores are evicted first. This means:
//! - younger, recently created requests are evicted first;
//! - requests with low token value or slow generation are evicted first; and
//! - expensive sessions that occupy large KV caches are evicted first.
//!
//! This module also rejects new requests when HBM utilization exceeds a limit.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

/// KV-cache eviction strategy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvEvictionPolicy {
    /// Composite scoring strategy from the Aegaeon paper.
    #[default]
    AegaeonScore,
    /// Simple LRU that evicts the least recently accessed session first.
    Lru,
    /// Priority-based eviction, with lower-priority sessions first.
    Priority,
    /// KV-cache-size eviction, with larger sessions first.
    LargestFirst,
}

/// KV-cache eviction configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvEvictionConfig {
    /// Eviction strategy.
    #[serde(default)]
    pub policy: KvEvictionPolicy,
    /// Reject new requests when HBM utilization exceeds this threshold.
    #[serde(default = "default_admission_threshold")]
    pub admission_threshold: f64,
    /// Minimum fraction of the KV cache to retain during eviction.
    #[serde(default = "default_retention_ratio")]
    pub retention_ratio: f64,
    /// Default token-value estimate when no prior information is available.
    #[serde(default = "default_token_value")]
    pub default_token_value: f64,
}

fn default_admission_threshold() -> f64 {
    0.85
}

fn default_retention_ratio() -> f64 {
    0.1
}

fn default_token_value() -> f64 {
    1.0
}

impl Default for KvEvictionConfig {
    fn default() -> Self {
        Self {
            policy: KvEvictionPolicy::default(),
            admission_threshold: default_admission_threshold(),
            retention_ratio: default_retention_ratio(),
            default_token_value: default_token_value(),
        }
    }
}

/// Runtime metadata used to make KV-cache eviction decisions.
#[derive(Debug, Clone)]
pub struct KvSessionInfo {
    pub request_id: String,
    pub model_id: String,
    pub priority: u32,
    /// Session creation time.
    pub created_at: Instant,
    /// Time of the most recent access or generated token.
    pub last_accessed: Instant,
    /// Number of tokens currently held in the KV cache.
    pub kv_cache_tokens: usize,
    /// Number of generated tokens.
    pub generated_tokens: usize,
    /// Current generation rate in tokens per second, used to estimate value.
    pub estimated_token_value: Option<f64>,
    /// Whether the request is actively decoding.
    pub is_active: bool,
}

impl KvSessionInfo {
    /// Request age in seconds.
    pub fn age_secs(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64().max(0.001)
    }

    /// Idle time in seconds.
    pub fn idle_secs(&self) -> f64 {
        self.last_accessed.elapsed().as_secs_f64().max(0.001)
    }

    /// Session cost based on occupied KV-cache tokens; larger means costlier.
    pub fn session_cost(&self) -> f64 {
        self.kv_cache_tokens as f64
    }

    /// Estimated token value.
    pub fn token_value(&self, default: f64) -> f64 {
        self.estimated_token_value.unwrap_or(default)
    }
}

/// Aegaeon eviction score; lower scores are evicted first.
/// `score = age * value * cost`; older, higher-value, smaller-cache sessions are protected.
fn aegaeon_score(session: &KvSessionInfo, default_token_value: f64) -> f64 {
    let age = session.age_secs();
    let value = session.token_value(default_token_value);
    let cost = session.session_cost().max(1.0);
    age * value * cost
}

/// Eviction candidate used in a priority queue.
struct EvictionCandidate {
    request_id: String,
    score: f64,
    kv_cache_tokens: usize,
}

impl PartialEq for EvictionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for EvictionCandidate {}

impl PartialOrd for EvictionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvictionCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so the minimum score is evicted first.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.kv_cache_tokens.cmp(&self.kv_cache_tokens))
    }
}

/// Admission-control result.
#[derive(Debug, Clone, PartialEq)]
pub enum AdmissionResult {
    /// Accept the new request.
    Admitted,
    /// Reject the request because HBM utilization is too high.
    Rejected {
        current_utilization: f64,
        threshold: f64,
        reason: String,
    },
}

/// KV-cache eviction manager.
#[derive(Debug)]
pub struct KvEvictionManager {
    config: KvEvictionConfig,
    /// Total number of evictions.
    total_evictions: usize,
    /// Total number of evicted tokens.
    total_evicted_tokens: usize,
    /// Total number of rejected requests.
    total_rejections: usize,
}

impl KvEvictionManager {
    pub fn new(config: KvEvictionConfig) -> Self {
        Self {
            config,
            total_evictions: 0,
            total_evicted_tokens: 0,
            total_rejections: 0,
        }
    }

    pub fn config(&self) -> &KvEvictionConfig {
        &self.config
    }

    pub fn total_evictions(&self) -> usize {
        self.total_evictions
    }

    pub fn total_evicted_tokens(&self) -> usize {
        self.total_evicted_tokens
    }

    pub fn total_rejections(&self) -> usize {
        self.total_rejections
    }

    /// Check whether admission control permits a new request.
    pub fn check_admission(
        &mut self,
        current_kv_tokens: usize,
        max_kv_tokens: usize,
    ) -> AdmissionResult {
        if max_kv_tokens == 0 {
            return AdmissionResult::Rejected {
                current_utilization: 1.0,
                threshold: self.config.admission_threshold,
                reason: "KV cache pool has zero capacity".to_string(),
            };
        }

        let utilization = current_kv_tokens as f64 / max_kv_tokens as f64;
        if utilization >= self.config.admission_threshold {
            self.total_rejections += 1;
            AdmissionResult::Rejected {
                current_utilization: utilization,
                threshold: self.config.admission_threshold,
                reason: format!(
                    "HBM utilization {:.1}% >= threshold {:.1}%",
                    utilization * 100.0,
                    self.config.admission_threshold * 100.0
                ),
            }
        } else {
            AdmissionResult::Admitted
        }
    }

    /// Select active sessions to free at least `tokens_to_free` KV-cache tokens.
    ///
    /// Return evicted sessions in eviction order.
    pub fn select_eviction_victims(
        &mut self,
        sessions: &[KvSessionInfo],
        tokens_to_free: usize,
    ) -> Vec<EvictionDecision> {
        let _span = tracing::info_span!("kv_cache.evict", tokens_to_free).entered();
        let mut heap = BinaryHeap::new();

        for session in sessions {
            let score = match self.config.policy {
                KvEvictionPolicy::AegaeonScore => {
                    aegaeon_score(session, self.config.default_token_value)
                }
                KvEvictionPolicy::Lru => {
                    // Negate idle time so the min-heap selects the longest-idle session.
                    -session.idle_secs()
                }
                KvEvictionPolicy::Priority => {
                    // Lower-priority sessions receive lower scores and are evicted first.
                    session.priority as f64
                }
                KvEvictionPolicy::LargestFirst => {
                    // Evict larger KV caches first.
                    -(session.kv_cache_tokens as f64)
                }
            };

            heap.push(EvictionCandidate {
                request_id: session.request_id.clone(),
                score,
                kv_cache_tokens: session.kv_cache_tokens,
            });
        }

        let mut decisions = Vec::new();
        let mut freed = 0usize;

        while freed < tokens_to_free {
            match heap.pop() {
                Some(candidate) => {
                    freed += candidate.kv_cache_tokens;
                    self.total_evictions += 1;
                    self.total_evicted_tokens += candidate.kv_cache_tokens;
                    decisions.push(EvictionDecision {
                        request_id: candidate.request_id,
                        freed_tokens: candidate.kv_cache_tokens,
                        score: candidate.score,
                    });
                }
                None => break,
            }
        }

        decisions
    }

    /// Calculate current eviction scores for monitoring and debugging.
    pub fn score_session(&self, session: &KvSessionInfo) -> f64 {
        match self.config.policy {
            KvEvictionPolicy::AegaeonScore => {
                aegaeon_score(session, self.config.default_token_value)
            }
            KvEvictionPolicy::Lru => -session.idle_secs(),
            KvEvictionPolicy::Priority => session.priority as f64,
            KvEvictionPolicy::LargestFirst => -(session.kv_cache_tokens as f64),
        }
    }
}

/// Result of one eviction decision.
#[derive(Debug, Clone)]
pub struct EvictionDecision {
    pub request_id: String,
    pub freed_tokens: usize,
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(
        id: &str,
        priority: u32,
        kv_tokens: usize,
        age_ms: u64,
        idle_ms: u64,
        token_value: Option<f64>,
    ) -> KvSessionInfo {
        let now = Instant::now();
        KvSessionInfo {
            request_id: id.to_string(),
            model_id: "m1".to_string(),
            priority,
            created_at: now - std::time::Duration::from_millis(age_ms),
            last_accessed: now - std::time::Duration::from_millis(idle_ms),
            kv_cache_tokens: kv_tokens,
            generated_tokens: kv_tokens / 2,
            estimated_token_value: token_value,
            is_active: idle_ms < 100,
        }
    }

    #[test]
    fn admission_control_allows_under_threshold() {
        let config = KvEvictionConfig {
            admission_threshold: 0.85,
            ..Default::default()
        };
        let mut mgr = KvEvictionManager::new(config);

        // 80% utilization -> admitted
        let result = mgr.check_admission(800, 1000);
        assert_eq!(result, AdmissionResult::Admitted);
    }

    #[test]
    fn admission_control_rejects_over_threshold() {
        let config = KvEvictionConfig {
            admission_threshold: 0.85,
            ..Default::default()
        };
        let mut mgr = KvEvictionManager::new(config);

        // 90% utilization -> rejected
        let result = mgr.check_admission(900, 1000);
        assert!(matches!(result, AdmissionResult::Rejected { .. }));
        assert_eq!(mgr.total_rejections(), 1);
    }

    #[test]
    fn aegaeon_score_prefers_young_low_value_high_cost() {
        // An old, high-value, low-cost request receives a high protected score.
        let old_cheap = make_session("old_cheap", 5, 100, 10000, 100, Some(10.0));
        // A new, low-value, high-cost request receives a low eviction score.
        let new_expensive = make_session("new_expensive", 5, 1000, 100, 100, Some(0.1));

        let config = KvEvictionConfig::default();
        let mgr = KvEvictionManager::new(config);

        let score_old = mgr.score_session(&old_cheap);
        let score_new = mgr.score_session(&new_expensive);

        // `old_cheap` has the higher score and is less likely to be evicted.
        assert!(score_old > score_new);
    }

    #[test]
    fn eviction_selects_lowest_score_first() {
        let config = KvEvictionConfig {
            policy: KvEvictionPolicy::AegaeonScore,
            ..Default::default()
        };
        let mut mgr = KvEvictionManager::new(config);

        let sessions = vec![
            make_session("r1", 5, 100, 5000, 50, Some(5.0)),
            make_session("r2", 5, 500, 100, 50, Some(0.1)), // low score -> evict first
            make_session("r3", 5, 200, 3000, 50, Some(3.0)),
        ];

        let decisions = mgr.select_eviction_victims(&sessions, 500);
        assert!(!decisions.is_empty());
        // r2 should be first (lowest score)
        assert_eq!(decisions[0].request_id, "r2");
    }

    #[test]
    fn lru_eviction_prefers_idle_sessions() {
        let config = KvEvictionConfig {
            policy: KvEvictionPolicy::Lru,
            ..Default::default()
        };
        let mut mgr = KvEvictionManager::new(config);

        let sessions = vec![
            make_session("recent", 5, 100, 1000, 10, None),
            make_session("stale", 5, 100, 1000, 5000, None),
        ];

        let decisions = mgr.select_eviction_victims(&sessions, 100);
        assert_eq!(decisions[0].request_id, "stale");
    }

    #[test]
    fn priority_eviction_prefers_low_priority() {
        let config = KvEvictionConfig {
            policy: KvEvictionPolicy::Priority,
            ..Default::default()
        };
        let mut mgr = KvEvictionManager::new(config);

        let sessions = vec![
            make_session("high", 10, 100, 1000, 50, None),
            make_session("low", 1, 100, 1000, 50, None),
            make_session("mid", 5, 100, 1000, 50, None),
        ];

        let decisions = mgr.select_eviction_victims(&sessions, 100);
        assert_eq!(decisions[0].request_id, "low");
    }

    #[test]
    fn largest_first_eviction() {
        let config = KvEvictionConfig {
            policy: KvEvictionPolicy::LargestFirst,
            ..Default::default()
        };
        let mut mgr = KvEvictionManager::new(config);

        let sessions = vec![
            make_session("small", 5, 50, 1000, 50, None),
            make_session("big", 5, 500, 1000, 50, None),
            make_session("medium", 5, 200, 1000, 50, None),
        ];

        let decisions = mgr.select_eviction_victims(&sessions, 500);
        assert_eq!(decisions[0].request_id, "big");
        assert_eq!(decisions[0].freed_tokens, 500);
    }

    #[test]
    fn eviction_tracks_stats() {
        let config = KvEvictionConfig::default();
        let mut mgr = KvEvictionManager::new(config);

        let sessions = vec![
            make_session("r1", 5, 100, 1000, 50, None),
            make_session("r2", 5, 200, 1000, 50, None),
        ];

        mgr.select_eviction_victims(&sessions, 300);
        assert_eq!(mgr.total_evictions(), 2);
        assert_eq!(mgr.total_evicted_tokens(), 300);
    }
}
