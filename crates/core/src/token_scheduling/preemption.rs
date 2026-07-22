//! Preemption — Aegaeon §4
//!
//! When a high-priority request arrives while the GPU batch is full, a
//! lower-priority decode request can be preempted and returned to the waiting
//! queue. It resumes during a later scheduling pass at an additional TTFT cost.
//!
//! Preemption strategies:
//! - `Priority`: preempt the lowest-priority request.
//! - `Oldest`: preempt the request with the most generated tokens to reduce stragglers.
//! - `KvCost`: preempt the request with the smallest KV cache to reduce recovery cost.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

/// Preemption strategy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreemptionPolicy {
    /// Preempt the lowest-priority request.
    #[default]
    Priority,
    /// Preempt the request with the most generated tokens.
    Oldest,
    /// Preempt the request with the smallest KV cache and recovery cost.
    KvCost,
}

/// Preemption configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreemptionConfig {
    /// Whether preemption is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Preemption strategy.
    #[serde(default)]
    pub policy: PreemptionPolicy,
    /// Maximum number of times a request may be preempted to prevent starvation.
    #[serde(default = "default_max_preemptions")]
    pub max_preemptions_per_request: usize,
    /// Trigger preemption only after a high-priority request waits this many milliseconds.
    #[serde(default = "default_preemption_threshold_ms")]
    pub preemption_threshold_ms: u64,
}

fn default_max_preemptions() -> usize {
    3
}

fn default_preemption_threshold_ms() -> u64 {
    500
}

impl Default for PreemptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy: PreemptionPolicy::default(),
            max_preemptions_per_request: default_max_preemptions(),
            preemption_threshold_ms: default_preemption_threshold_ms(),
        }
    }
}

/// An active decode request that may be preempted.
#[derive(Debug, Clone)]
pub struct PreemptibleRequest {
    pub request_id: String,
    pub model_id: String,
    pub priority: u32,
    pub generated_tokens: usize,
    pub kv_cache_tokens: usize,
    pub preemption_count: usize,
    pub decode_started_at: Instant,
    pub last_scheduled_at: Instant,
}

impl PreemptibleRequest {
    /// Calculate the preemption cost. Lower scores are preempted first.
    /// A reversed `Ord` creates a min-heap that selects the lowest-cost victim.
    fn preemption_cost(&self, policy: PreemptionPolicy) -> f64 {
        match policy {
            // Lower priority produces a lower cost and earlier preemption.
            PreemptionPolicy::Priority => self.priority as f64,
            // More generated tokens produce a lower cost for straggler-aware preemption.
            PreemptionPolicy::Oldest => -(self.generated_tokens as f64),
            // A smaller KV cache has a lower recovery cost and is preempted first.
            PreemptionPolicy::KvCost => self.kv_cache_tokens as f64,
        }
    }
}

/// Preemption candidate used for priority-queue ordering.
struct PreemptionCandidate {
    request_id: String,
    cost: f64,
}

impl PartialEq for PreemptionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for PreemptionCandidate {}

impl PartialOrd for PreemptionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PreemptionCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the ordering so the minimum cost is preempted first.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

/// Result of a preemption decision.
#[derive(Debug, Clone)]
pub struct PreemptionDecision {
    /// ID of the preempted request.
    pub preempted_request_id: String,
    /// Number of times the request has been preempted, including this event.
    pub preemption_count: usize,
    /// Number of tokens generated when the request was preempted.
    pub tokens_generated_before_preemption: usize,
}

/// Preemption manager.
#[derive(Debug)]
pub struct PreemptionManager {
    config: PreemptionConfig,
    /// Total number of preemptions.
    total_preemptions: usize,
}

impl PreemptionManager {
    pub fn new(config: PreemptionConfig) -> Self {
        Self {
            config,
            total_preemptions: 0,
        }
    }

    pub fn config(&self) -> &PreemptionConfig {
        &self.config
    }

    pub fn total_preemptions(&self) -> usize {
        self.total_preemptions
    }

    /// Determine whether preemption should be triggered.
    /// `high_priority_wait_ms` is the high-priority request's wait time.
    pub fn should_preempt(&self, high_priority_wait_ms: u64) -> bool {
        self.config.enabled && high_priority_wait_ms >= self.config.preemption_threshold_ms
    }

    /// Select the best preemption candidate from active decode requests.
    ///
    /// Returns `None` when the queue is empty or every request reached the limit.
    pub fn select_victim(
        &mut self,
        active_requests: &[PreemptibleRequest],
    ) -> Option<PreemptionDecision> {
        if !self.config.enabled || active_requests.is_empty() {
            return None;
        }

        let mut heap = BinaryHeap::new();

        for req in active_requests {
            // Skip requests that reached the preemption limit.
            if req.preemption_count >= self.config.max_preemptions_per_request {
                continue;
            }
            let cost = req.preemption_cost(self.config.policy);
            heap.push(PreemptionCandidate {
                request_id: req.request_id.clone(),
                cost,
            });
        }

        let candidate = heap.pop()?;

        // Find the selected request and update its state.
        let victim = active_requests
            .iter()
            .find(|r| r.request_id == candidate.request_id)?;

        self.total_preemptions += 1;

        Some(PreemptionDecision {
            preempted_request_id: victim.request_id.clone(),
            preemption_count: victim.preemption_count + 1,
            tokens_generated_before_preemption: victim.generated_tokens,
        })
    }

    /// Select enough victims to free `slots_needed` positions.
    pub fn select_victims(
        &mut self,
        active_requests: &[PreemptibleRequest],
        slots_needed: usize,
    ) -> Vec<PreemptionDecision> {
        let mut decisions = Vec::new();
        let mut remaining = slots_needed;
        let mut excluded: std::collections::HashSet<String> = std::collections::HashSet::new();

        while remaining > 0 {
            let available: Vec<PreemptibleRequest> = active_requests
                .iter()
                .filter(|r| !excluded.contains(&r.request_id))
                .cloned()
                .collect();

            if available.is_empty() {
                break;
            }

            match self.select_victim(&available) {
                Some(decision) => {
                    excluded.insert(decision.preempted_request_id.clone());
                    decisions.push(decision);
                    remaining -= 1;
                }
                None => break,
            }
        }

        decisions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(
        id: &str,
        priority: u32,
        generated: usize,
        kv_tokens: usize,
    ) -> PreemptibleRequest {
        PreemptibleRequest {
            request_id: id.to_string(),
            model_id: "m1".to_string(),
            priority,
            generated_tokens: generated,
            kv_cache_tokens: kv_tokens,
            preemption_count: 0,
            decode_started_at: Instant::now(),
            last_scheduled_at: Instant::now(),
        }
    }

    #[test]
    fn preemption_disabled_returns_none() {
        let config = PreemptionConfig {
            enabled: false,
            ..Default::default()
        };
        let mut mgr = PreemptionManager::new(config);
        let reqs = vec![make_request("r1", 1, 10, 100)];
        assert!(mgr.select_victim(&reqs).is_none());
    }

    #[test]
    fn priority_policy_selects_lowest_priority() {
        let config = PreemptionConfig {
            enabled: true,
            policy: PreemptionPolicy::Priority,
            ..Default::default()
        };
        let mut mgr = PreemptionManager::new(config);
        let reqs = vec![
            make_request("high", 10, 5, 50),
            make_request("low", 1, 20, 200),
            make_request("mid", 5, 10, 100),
        ];
        let decision = mgr.select_victim(&reqs).unwrap();
        assert_eq!(decision.preempted_request_id, "low");
    }

    #[test]
    fn oldest_policy_selects_most_generated() {
        let config = PreemptionConfig {
            enabled: true,
            policy: PreemptionPolicy::Oldest,
            ..Default::default()
        };
        let mut mgr = PreemptionManager::new(config);
        let reqs = vec![
            make_request("r1", 5, 10, 100),
            make_request("r2", 5, 50, 100),
            make_request("r3", 5, 5, 100),
        ];
        let decision = mgr.select_victim(&reqs).unwrap();
        assert_eq!(decision.preempted_request_id, "r2");
    }

    #[test]
    fn kv_cost_policy_selects_smallest_cache() {
        let config = PreemptionConfig {
            enabled: true,
            policy: PreemptionPolicy::KvCost,
            ..Default::default()
        };
        let mut mgr = PreemptionManager::new(config);
        let reqs = vec![
            make_request("r1", 5, 10, 500),
            make_request("r2", 5, 10, 10),
            make_request("r3", 5, 10, 200),
        ];
        let decision = mgr.select_victim(&reqs).unwrap();
        assert_eq!(decision.preempted_request_id, "r2");
    }

    #[test]
    fn respects_max_preemptions() {
        let config = PreemptionConfig {
            enabled: true,
            policy: PreemptionPolicy::Priority,
            max_preemptions_per_request: 2,
            ..Default::default()
        };
        let mut mgr = PreemptionManager::new(config);
        let mut req = make_request("r1", 1, 10, 100);
        req.preemption_count = 2; // already at max

        assert!(mgr.select_victim(&[req]).is_none());
    }

    #[test]
    fn should_preempt_respects_threshold() {
        let config = PreemptionConfig {
            enabled: true,
            preemption_threshold_ms: 100,
            ..Default::default()
        };
        let mgr = PreemptionManager::new(config);
        assert!(!mgr.should_preempt(50));
        assert!(mgr.should_preempt(100));
        assert!(mgr.should_preempt(200));
    }

    #[test]
    fn batch_victim_selection() {
        let config = PreemptionConfig {
            enabled: true,
            policy: PreemptionPolicy::Priority,
            ..Default::default()
        };
        let mut mgr = PreemptionManager::new(config);
        let reqs = vec![
            make_request("r1", 1, 10, 100),
            make_request("r2", 2, 10, 100),
            make_request("r3", 3, 10, 100),
            make_request("r4", 4, 10, 100),
        ];
        let decisions = mgr.select_victims(&reqs, 2);
        assert_eq!(decisions.len(), 2);
        let ids: Vec<&str> = decisions
            .iter()
            .map(|d| d.preempted_request_id.as_str())
            .collect();
        assert!(ids.contains(&"r1"));
        assert!(ids.contains(&"r2"));
    }
}
