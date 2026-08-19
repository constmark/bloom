//! World model engine traits and state cache management.
//!
//! World models operate in a closed loop: observation → state → prediction/action.
//! This module defines the engine interfaces, a state cache manager, and mock
//! implementations for testing without real model weights.

use std::collections::HashMap;

use anyhow::Result;
use bloomai_core::{
    Action, PowerState, PredictedFuture, StateCacheConfig, StateCacheEntry, StateCachePriority,
    StateDelta, ThermalState, WorldObservation, WorldState,
};
use serde::{Deserialize, Serialize};

use crate::io::OutputChunk;

// ---------------------------------------------------------------------------
// WorldModelEngine trait
// ---------------------------------------------------------------------------

/// Engine for world models: observation → state delta / predicted futures.
///
/// A world model ingests observations and produces updated world states
/// with optional predicted future states. Unlike standard engines that
/// follow a prompt-response pattern, world models maintain internal state.
pub trait WorldModelEngine: Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Process new observations and produce a state delta.
    fn observe(
        &self,
        current_state: Option<&WorldState>,
        observations: Vec<WorldObservation>,
        horizon: u32,
    ) -> Result<StateDelta>;

    /// Predict future states from a given world state without new observations.
    fn predict(&self, state: &WorldState, horizon: u32) -> Result<Vec<PredictedFuture>>;
}

// ---------------------------------------------------------------------------
// PolicyEngine trait
// ---------------------------------------------------------------------------

/// Policy model: state → action.
///
/// Given a world state (and optionally predicted futures), the policy engine
/// decides on an action. This is the "brain" in a closed-loop world model
/// system.
pub trait PolicyEngine: Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Decide on an action given the current world state.
    fn decide(&self, state: &WorldState) -> Result<Action>;

    /// Decide using state + predicted futures for lookahead planning.
    fn decide_with_futures(
        &self,
        state: &WorldState,
        _futures: &[PredictedFuture],
    ) -> Result<Action> {
        // Default: ignore futures, use current state.
        self.decide(state)
    }
}

// ---------------------------------------------------------------------------
// StateCacheManager
// ---------------------------------------------------------------------------

/// Manages cached world states with retention, compression, expiry and eviction.
pub struct StateCacheManager {
    config: StateCacheConfig,
    entries: HashMap<String, StateCacheEntry>,
    /// Map from state_id to the actual WorldState data.
    states: HashMap<String, WorldState>,
    total_bytes: usize,
}

impl StateCacheManager {
    pub fn new(config: StateCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            states: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Insert a new world state into the cache.
    /// Evicts lower-priority entries if budget is exceeded.
    pub fn insert(&mut self, state: WorldState, priority: StateCachePriority) -> Result<()> {
        let size = state.latent_bytes;
        let now = state.timestamp_ms;

        // Evict if over budget
        while self.total_bytes + size > self.config.max_bytes
            || self.entries.len() >= self.config.max_entries
        {
            if !self.evict_one(now) {
                anyhow::bail!(
                    "state cache full ({} bytes / {} max, {} entries / {} max); cannot evict",
                    self.total_bytes,
                    self.config.max_bytes,
                    self.entries.len(),
                    self.config.max_entries
                );
            }
        }

        let entry = StateCacheEntry {
            state_id: state.state_id.clone(),
            size_bytes: size,
            created_ms: now,
            last_access_ms: now,
            access_count: 0,
            compressed: false,
            priority,
            ttl_ms: self.config.default_ttl_ms,
        };

        self.total_bytes += size;
        self.entries.insert(state.state_id.clone(), entry);
        self.states.insert(state.state_id.clone(), state);
        Ok(())
    }

    /// Retrieve a cached state by ID, updating access stats.
    pub fn get(&mut self, state_id: &str) -> Option<&WorldState> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if let Some(entry) = self.entries.get_mut(state_id) {
            entry.last_access_ms = now;
            entry.access_count += 1;
        }
        self.states.get(state_id)
    }

    /// Remove a state from the cache.
    pub fn remove(&mut self, state_id: &str) -> bool {
        if let Some(entry) = self.entries.remove(state_id) {
            self.total_bytes = self.total_bytes.saturating_sub(entry.size_bytes);
            self.states.remove(state_id);
            true
        } else {
            false
        }
    }

    /// Expire entries past their TTL.
    pub fn expire(&mut self, now_ms: u64) -> Vec<String> {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.ttl_ms > 0 && now_ms.saturating_sub(e.created_ms) > e.ttl_ms)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            self.remove(id);
        }
        expired
    }

    /// Compress old entries (simulated: mark as compressed, halve size).
    pub fn compress_old(&mut self, now_ms: u64) -> usize {
        if !self.config.auto_compress {
            return 0;
        }
        let mut compressed_count = 0;
        let ids: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                !e.compressed && now_ms.saturating_sub(e.created_ms) > self.config.compress_after_ms
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            if let Some(entry) = self.entries.get_mut(&id) {
                let saved = entry.size_bytes / 2;
                entry.size_bytes -= saved;
                entry.compressed = true;
                self.total_bytes = self.total_bytes.saturating_sub(saved);
                compressed_count += 1;
            }
        }
        compressed_count
    }

    /// Current total bytes used.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict the lowest-priority, least-recently-used entry.
    fn evict_one(&mut self, _now_ms: u64) -> bool {
        let victim = self
            .entries
            .iter()
            .min_by_key(|(_, e)| (e.priority, e.last_access_ms))
            .map(|(id, _)| id.clone());

        if let Some(id) = victim {
            tracing::info!(state_id = %id, "evicting world state from cache");
            self.remove(&id);
            true
        } else {
            false
        }
    }

    /// Get all entry metadata (for diagnostics).
    pub fn entries(&self) -> Vec<&StateCacheEntry> {
        self.entries.values().collect()
    }
}

/// Schema defining constraints on world observations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorldStateSchema {
    /// Allowed range (min, max) for scalar observations by name.
    pub scalar_ranges: HashMap<String, (f64, f64)>,
    /// Allowed image MIME types. If empty, all are allowed.
    pub allowed_image_mimes: Vec<String>,
    /// Expected tensor shapes. If empty, all are allowed.
    pub tensor_shapes: Vec<Vec<usize>>,
    /// Whether text observations are allowed.
    pub allow_text: bool,
    /// Whether audio observations are allowed.
    pub allow_audio: bool,
}

impl Default for WorldStateSchema {
    fn default() -> Self {
        Self {
            scalar_ranges: HashMap::new(),
            allowed_image_mimes: Vec::new(),
            tensor_shapes: Vec::new(),
            allow_text: true,
            allow_audio: true,
        }
    }
}

/// Schema defining constraints on policy actions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ActionSchema {
    /// Allowed action spaces.
    pub allowed_action_spaces: Vec<String>,
    /// Expected dimension (length of values vector) for each action space.
    pub action_dimensions: HashMap<String, usize>,
    /// Range constraints (min, max) for action values.
    pub value_range: Option<(f32, f32)>,
}

/// Orchestrates a closed-loop world model execution cycle.
///
/// Supports adaptive degradation under thermal / power constraints:
/// when the device overheats or enters low-power mode, the loop can
/// reduce observation frequency (skip steps), shorten the prediction
/// horizon, or signal that a smaller model should be used.
pub struct WorldModelLoop {
    world_model: Box<dyn WorldModelEngine>,
    policy: Box<dyn PolicyEngine>,
    cache: StateCacheManager,
    constraints: WorldModelConstraints,
    /// Internal step counter used for observation skipping.
    internal_step: u64,
    pub state_schema: Option<WorldStateSchema>,
    pub action_schema: Option<ActionSchema>,
}

/// Environmental constraints for world model adaptive degradation.
#[derive(Debug, Clone)]
pub struct WorldModelConstraints {
    pub thermal_state: ThermalState,
    pub power_state: PowerState,
    /// Skip observations every N steps when degraded (0 = never skip).
    /// At thermal=Serious: skip every other step.
    /// At thermal=Critical: skip 2 out of 3 steps.
    pub observation_skip_ratio: u32,
    /// Maximum prediction horizon under degraded conditions.
    pub max_degraded_horizon: u32,
    /// Whether the runtime should consider switching to a smaller model.
    pub suggest_smaller_model: bool,
}

impl Default for WorldModelConstraints {
    fn default() -> Self {
        Self {
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            observation_skip_ratio: 0,
            max_degraded_horizon: u32::MAX,
            suggest_smaller_model: false,
        }
    }
}

impl WorldModelConstraints {
    /// Build constraints from thermal and power state, auto-deriving
    /// skip ratio and horizon limits.
    pub fn from_thermal_power(thermal: ThermalState, power: PowerState) -> Self {
        let (skip_ratio, max_horizon, suggest_smaller) = match thermal {
            ThermalState::Nominal | ThermalState::Fair => (0u32, u32::MAX, false),
            ThermalState::Serious => (2, 4, false),
            ThermalState::Critical => (3, 1, true),
        };
        // Battery mode further restricts horizon
        let max_horizon = if power == PowerState::Battery {
            max_horizon.min(2)
        } else {
            max_horizon
        };
        let suggest_smaller =
            suggest_smaller || (thermal == ThermalState::Critical && power == PowerState::Battery);

        Self {
            thermal_state: thermal,
            power_state: power,
            observation_skip_ratio: skip_ratio,
            max_degraded_horizon: max_horizon,
            suggest_smaller_model: suggest_smaller,
        }
    }

    /// Whether current conditions require degradation.
    pub fn is_degraded(&self) -> bool {
        self.observation_skip_ratio > 0
            || self.max_degraded_horizon < u32::MAX
            || self.power_state == PowerState::Battery
    }

    /// Effective horizon after applying constraints.
    pub fn effective_horizon(&self, requested: u32) -> u32 {
        requested.min(self.max_degraded_horizon)
    }

    /// Whether this step should skip observation (to reduce compute).
    pub fn should_skip_observation(&self, step: u64) -> bool {
        if self.observation_skip_ratio == 0 {
            return false;
        }
        !step.is_multiple_of(self.observation_skip_ratio as u64)
    }
}

impl WorldModelLoop {
    pub fn new(
        world_model: Box<dyn WorldModelEngine>,
        policy: Box<dyn PolicyEngine>,
        cache_config: StateCacheConfig,
    ) -> Self {
        Self {
            world_model,
            policy,
            cache: StateCacheManager::new(cache_config),
            constraints: WorldModelConstraints::default(),
            internal_step: 0,
            state_schema: None,
            action_schema: None,
        }
    }

    pub fn set_schemas(
        &mut self,
        state_schema: Option<WorldStateSchema>,
        action_schema: Option<ActionSchema>,
    ) {
        self.state_schema = state_schema;
        self.action_schema = action_schema;
    }

    pub fn validate_observation(&self, obs: &WorldObservation) -> Result<()> {
        let schema = match &self.state_schema {
            Some(s) => s,
            None => return Ok(()),
        };
        match obs {
            WorldObservation::Text(_) => {
                if !schema.allow_text {
                    anyhow::bail!("Text observations are not allowed by schema");
                }
            }
            WorldObservation::Image { mime, .. } => {
                if !schema.allowed_image_mimes.is_empty()
                    && !schema.allowed_image_mimes.contains(mime)
                {
                    anyhow::bail!("Image MIME type '{}' is not allowed by schema", mime);
                }
            }
            WorldObservation::AudioPcm { .. } => {
                if !schema.allow_audio {
                    anyhow::bail!("Audio observations are not allowed by schema");
                }
            }
            WorldObservation::Tensor { shape, .. } => {
                if !schema.tensor_shapes.is_empty() && !schema.tensor_shapes.contains(shape) {
                    anyhow::bail!("Tensor shape {:?} is not allowed by schema", shape);
                }
            }
            WorldObservation::Scalar { name, value } => {
                if let Some(&(min, max)) = schema.scalar_ranges.get(name)
                    && (*value < min || *value > max)
                {
                    anyhow::bail!(
                        "Scalar observation '{}' value {} is out of range [{}, {}]",
                        name,
                        value,
                        min,
                        max
                    );
                }
            }
        }
        Ok(())
    }

    pub fn validate_action(&self, action: &Action) -> Result<()> {
        let schema = match &self.action_schema {
            Some(s) => s,
            None => return Ok(()),
        };
        if !schema.allowed_action_spaces.is_empty()
            && !schema.allowed_action_spaces.contains(&action.action_space)
        {
            anyhow::bail!(
                "Action space '{}' is not allowed by schema",
                action.action_space
            );
        }
        if let Some(&expected_dim) = schema.action_dimensions.get(&action.action_space)
            && action.values.len() != expected_dim
        {
            anyhow::bail!(
                "Action space '{}' expected dimension {}, got {}",
                action.action_space,
                expected_dim,
                action.values.len()
            );
        }
        if let Some((min, max)) = schema.value_range {
            for &val in &action.values {
                if val < min || val > max {
                    anyhow::bail!("Action value {} is out of range [{}, {}]", val, min, max);
                }
            }
        }
        Ok(())
    }

    /// Update environment constraints for adaptive degradation.
    pub fn update_constraints(&mut self, constraints: WorldModelConstraints) {
        self.constraints = constraints;
    }

    /// Get current constraints.
    pub fn constraints(&self) -> &WorldModelConstraints {
        &self.constraints
    }

    /// Set constraints from thermal and power state.
    pub fn set_environment(&mut self, thermal: ThermalState, power: PowerState) {
        self.constraints = WorldModelConstraints::from_thermal_power(thermal, power);
    }

    /// Run one step: observe → update state → predict → decide.
    ///
    /// Under degraded conditions (high temperature / low power):
    /// - Some steps may be skipped (observation frequency reduction)
    /// - The prediction horizon is shortened
    /// - A `suggest_smaller_model` flag is emitted via metrics
    ///
    /// Returns the output chunks (state delta, predicted states, action).
    pub fn step(
        &mut self,
        observations: Vec<WorldObservation>,
        horizon: u32,
    ) -> Result<Vec<OutputChunk>> {
        // Validate observations first
        for obs in &observations {
            self.validate_observation(obs)?;
        }

        self.internal_step += 1;

        // --- Adaptive degradation: skip observation step ---
        if self.constraints.should_skip_observation(self.internal_step) {
            tracing::debug!(
                step = self.internal_step,
                thermal = ?self.constraints.thermal_state,
                "world model loop: skipping observation step (degraded)",
            );
            // Return a minimal End chunk — caller can detect the skip
            return Ok(vec![
                OutputChunk::Metrics {
                    compute_ms: 0,
                    speculative_acceptance_rate: None,
                    speculative_draft_tokens: None,
                    speculative_accepted_tokens: None,
                },
                OutputChunk::End,
            ]);
        }

        // --- Apply effective horizon ---
        let effective_horizon = self.constraints.effective_horizon(horizon);
        if effective_horizon < horizon {
            tracing::info!(
                requested = horizon,
                effective = effective_horizon,
                "world model loop: horizon reduced (degraded)",
            );
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Expire old states
        self.cache.expire(now_ms);

        // Get current state (most recent)
        let current_state = self.latest_state();

        // Observe: produce state delta
        let delta =
            self.world_model
                .observe(current_state.as_ref(), observations, effective_horizon)?;

        let new_state_id = delta.to_state_id.clone();
        let from_state_id = delta.from_state_id.clone();

        // Build new state from delta
        let step = current_state.as_ref().map(|s| s.step + 1).unwrap_or(0);
        let new_state = WorldState {
            state_id: new_state_id.clone(),
            observations: delta.new_observations.clone(),
            latent: delta.latent_update.clone(),
            step,
            timestamp_ms: now_ms,
            latent_bytes: delta.latent_update.as_ref().map(|l| l.len()).unwrap_or(0),
        };

        // Cache the new state
        let _ = self
            .cache
            .insert(new_state.clone(), StateCachePriority::Normal);

        // Predict futures
        let futures = self.world_model.predict(&new_state, effective_horizon)?;

        // Decide action
        let action = if futures.is_empty() {
            self.policy.decide(&new_state)?
        } else {
            self.policy.decide_with_futures(&new_state, &futures)?
        };

        // Validate action
        self.validate_action(&action)?;

        // Build output chunks
        let mut chunks = Vec::new();

        // State delta
        chunks.push(OutputChunk::StateDelta {
            from_state_id,
            to_state_id: new_state_id,
            latent_update: delta.latent_update,
        });

        // Predicted states
        for pred in &futures {
            chunks.push(OutputChunk::PredictedState {
                state_id: pred.state_id.clone(),
                confidence: pred.confidence,
                horizon_step: pred.horizon_step,
            });
        }

        // Action
        chunks.push(OutputChunk::Action {
            action_space: action.action_space,
            values: action.values,
        });

        chunks.push(OutputChunk::Metrics {
            compute_ms: 0,
            speculative_acceptance_rate: None,
            speculative_draft_tokens: None,
            speculative_accepted_tokens: None,
        });

        // Signal degraded state if applicable
        if self.constraints.suggest_smaller_model {
            tracing::warn!(
                thermal = ?self.constraints.thermal_state,
                power = ?self.constraints.power_state,
                "world model loop: suggesting smaller model due to environment constraints",
            );
        }

        chunks.push(OutputChunk::End);

        Ok(chunks)
    }

    /// Run multiple steps in sequence.
    pub fn run_steps(
        &mut self,
        observation_sequence: Vec<Vec<WorldObservation>>,
        horizon: u32,
    ) -> Result<Vec<Vec<OutputChunk>>> {
        let mut all_outputs = Vec::new();
        for obs in observation_sequence {
            let output = self.step(obs, horizon)?;
            all_outputs.push(output);
        }
        Ok(all_outputs)
    }

    fn latest_state(&self) -> Option<WorldState> {
        // Find the state with the highest step number (most recent).
        self.cache
            .entries()
            .iter()
            .filter_map(|e| self.cache.states.get(&e.state_id))
            .max_by_key(|s| s.step)
            .cloned()
    }

    /// Access the state cache for inspection.
    pub fn cache(&self) -> &StateCacheManager {
        &self.cache
    }

    /// Access the state cache mutably.
    pub fn cache_mut(&mut self) -> &mut StateCacheManager {
        &mut self.cache
    }
}

// ---------------------------------------------------------------------------
// Mock implementations for testing
// ---------------------------------------------------------------------------

/// Mock world model that produces simple deterministic state transitions.
pub struct MockWorldModel {
    name: &'static str,
}

impl MockWorldModel {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl WorldModelEngine for MockWorldModel {
    fn name(&self) -> &'static str {
        self.name
    }

    fn observe(
        &self,
        current_state: Option<&WorldState>,
        observations: Vec<WorldObservation>,
        horizon: u32,
    ) -> Result<StateDelta> {
        let from_id = current_state
            .map(|s| s.state_id.clone())
            .unwrap_or_else(|| "init".to_string());
        let step = current_state.map(|s| s.step + 1).unwrap_or(0);
        let to_id = format!("state-{}", step);

        // Generate predicted futures
        let mut predicted_futures = Vec::new();
        for h in 1..=horizon {
            predicted_futures.push(PredictedFuture {
                state_id: format!("{}-pred-{}", to_id, h),
                latent: None,
                confidence: 1.0 / (h as f32),
                horizon_step: h,
            });
        }

        Ok(StateDelta {
            from_state_id: from_id,
            to_state_id: to_id,
            new_observations: observations,
            latent_update: Some(vec![0u8; 64]), // 64-byte mock latent
            predicted_futures,
        })
    }

    fn predict(&self, state: &WorldState, horizon: u32) -> Result<Vec<PredictedFuture>> {
        let mut futures = Vec::new();
        for h in 1..=horizon {
            futures.push(PredictedFuture {
                state_id: format!("{}-pred-{}", state.state_id, h),
                latent: None,
                confidence: 0.9 / (h as f32),
                horizon_step: h,
            });
        }
        Ok(futures)
    }
}

/// Mock policy engine that produces deterministic actions.
pub struct MockPolicyEngine {
    name: &'static str,
    action_space: String,
    action_dim: usize,
}

impl MockPolicyEngine {
    pub fn new(name: &'static str, action_space: &str, action_dim: usize) -> Self {
        Self {
            name,
            action_space: action_space.to_string(),
            action_dim,
        }
    }
}

impl PolicyEngine for MockPolicyEngine {
    fn name(&self) -> &'static str {
        self.name
    }

    fn decide(&self, state: &WorldState) -> Result<Action> {
        // Deterministic: action values = step number repeated
        let val = state.step as f32 * 0.1;
        Ok(Action {
            action_space: self.action_space.clone(),
            values: vec![val; self.action_dim],
            metadata: Some(format!("step_{}", state.step)),
        })
    }

    fn decide_with_futures(
        &self,
        state: &WorldState,
        futures: &[PredictedFuture],
    ) -> Result<Action> {
        // Use the most confident future to adjust action
        let best_future = futures.iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let base_val = state.step as f32 * 0.1;
        let bonus = best_future.map(|f| f.confidence).unwrap_or(0.0);

        Ok(Action {
            action_space: self.action_space.clone(),
            values: vec![base_val + bonus; self.action_dim],
            metadata: Some(format!("step_{}_with_future", state.step)),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_obs(name: &str, value: f64) -> WorldObservation {
        WorldObservation::Scalar {
            name: name.into(),
            value,
        }
    }

    #[test]
    fn test_state_cache_insert_and_get() {
        let mut cache = StateCacheManager::new(StateCacheConfig::default());
        let state = WorldState {
            state_id: "s1".into(),
            observations: vec![],
            latent: Some(vec![0u8; 128]),
            step: 0,
            timestamp_ms: 1000,
            latent_bytes: 128,
        };
        cache.insert(state, StateCachePriority::Normal).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), 128);

        let retrieved = cache.get("s1").unwrap();
        assert_eq!(retrieved.state_id, "s1");
    }

    #[test]
    fn test_state_cache_eviction() {
        let config = StateCacheConfig {
            max_bytes: 200,
            max_entries: 2,
            default_ttl_ms: 0,
            auto_compress: false,
            compress_after_ms: u64::MAX,
        };
        let mut cache = StateCacheManager::new(config);

        // Insert 2 states at 100 bytes each
        for i in 0..2 {
            let state = WorldState {
                state_id: format!("s{}", i),
                observations: vec![],
                latent: Some(vec![0u8; 100]),
                step: i,
                timestamp_ms: 1000 + i * 100,
                latent_bytes: 100,
            };
            cache.insert(state, StateCachePriority::Normal).unwrap();
        }
        assert_eq!(cache.len(), 2);

        // Insert a third state — should evict the oldest
        let state3 = WorldState {
            state_id: "s2".into(),
            observations: vec![],
            latent: Some(vec![0u8; 100]),
            step: 2,
            timestamp_ms: 1200,
            latent_bytes: 100,
        };
        cache.insert(state3, StateCachePriority::Normal).unwrap();
        assert_eq!(cache.len(), 2);
        // s0 should have been evicted
        assert!(cache.get("s0").is_none());
    }

    #[test]
    fn test_state_cache_expiry() {
        let config = StateCacheConfig {
            max_bytes: 1024 * 1024,
            max_entries: 100,
            default_ttl_ms: 500,
            auto_compress: false,
            compress_after_ms: u64::MAX,
        };
        let mut cache = StateCacheManager::new(config);

        let state = WorldState {
            state_id: "exp".into(),
            observations: vec![],
            latent: Some(vec![0u8; 64]),
            step: 0,
            timestamp_ms: 1000,
            latent_bytes: 64,
        };
        cache.insert(state, StateCachePriority::Normal).unwrap();
        assert_eq!(cache.len(), 1);

        // Expire at t=1400 (within TTL)
        let expired = cache.expire(1400);
        assert!(expired.is_empty());
        assert_eq!(cache.len(), 1);

        // Expire at t=1600 (past TTL)
        let expired = cache.expire(1600);
        assert_eq!(expired.len(), 1);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_state_cache_compression() {
        let config = StateCacheConfig {
            max_bytes: 1024 * 1024,
            max_entries: 100,
            default_ttl_ms: 0,
            auto_compress: true,
            compress_after_ms: 100,
        };
        let mut cache = StateCacheManager::new(config);

        let state = WorldState {
            state_id: "comp".into(),
            observations: vec![],
            latent: Some(vec![0u8; 200]),
            step: 0,
            timestamp_ms: 1000,
            latent_bytes: 200,
        };
        cache.insert(state, StateCachePriority::Normal).unwrap();
        assert_eq!(cache.total_bytes(), 200);

        // Compress after 200ms
        let count = cache.compress_old(1200);
        assert_eq!(count, 1);
        assert_eq!(cache.total_bytes(), 100); // halved
    }

    #[test]
    fn test_mock_world_model_observe() {
        let wm = MockWorldModel::new("mock-wm");
        let delta = wm.observe(None, vec![scalar_obs("temp", 25.0)], 3).unwrap();
        assert_eq!(delta.from_state_id, "init");
        assert_eq!(delta.to_state_id, "state-0");
        assert_eq!(delta.predicted_futures.len(), 3);
    }

    #[test]
    fn test_mock_policy_engine_decide() {
        let policy = MockPolicyEngine::new("mock-policy", "continuous", 4);
        let state = WorldState {
            state_id: "s5".into(),
            observations: vec![],
            latent: None,
            step: 5,
            timestamp_ms: 5000,
            latent_bytes: 0,
        };
        let action = policy.decide(&state).unwrap();
        assert_eq!(action.action_space, "continuous");
        assert_eq!(action.values.len(), 4);
        assert!((action.values[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_world_model_loop_single_step() {
        let wm = Box::new(MockWorldModel::new("mock-wm"));
        let policy = Box::new(MockPolicyEngine::new("mock-policy", "discrete", 2));
        let config = StateCacheConfig::default();
        let mut loop_ = WorldModelLoop::new(wm, policy, config);

        let output = loop_.step(vec![scalar_obs("x", 1.0)], 2).unwrap();

        // Should have: StateDelta, PredictedState(s), Action, Metrics, End
        assert!(output.len() >= 4);
        assert!(matches!(&output[0], OutputChunk::StateDelta { .. }));
        assert!(matches!(&output[output.len() - 1], OutputChunk::End));

        // Find the action
        let action_chunk = output
            .iter()
            .find(|c| matches!(c, OutputChunk::Action { .. }));
        assert!(action_chunk.is_some());

        if let OutputChunk::Action {
            action_space,
            values,
        } = action_chunk.unwrap()
        {
            assert_eq!(action_space, "discrete");
            assert_eq!(values.len(), 2);
        }
    }

    #[test]
    fn test_world_model_loop_multi_step() {
        let wm = Box::new(MockWorldModel::new("mock-wm"));
        let policy = Box::new(MockPolicyEngine::new("mock-policy", "continuous", 3));
        let config = StateCacheConfig::default();
        let mut loop_ = WorldModelLoop::new(wm, policy, config);

        let obs_seq = vec![
            vec![scalar_obs("x", 1.0)],
            vec![scalar_obs("x", 2.0)],
            vec![scalar_obs("x", 3.0)],
        ];

        let all_outputs = loop_.run_steps(obs_seq, 2).unwrap();
        assert_eq!(all_outputs.len(), 3);

        // Cache should have 3 states
        assert_eq!(loop_.cache().len(), 3);
    }

    #[test]
    fn test_world_model_loop_cache_eviction_under_budget() {
        let wm = Box::new(MockWorldModel::new("mock-wm"));
        let policy = Box::new(MockPolicyEngine::new("mock-policy", "discrete", 1));
        let config = StateCacheConfig {
            max_bytes: 200, // very small budget
            max_entries: 2,
            default_ttl_ms: 0,
            auto_compress: false,
            compress_after_ms: u64::MAX,
        };
        let mut loop_ = WorldModelLoop::new(wm, policy, config);

        // Run 4 steps — cache should evict old states
        for i in 0..4 {
            loop_.step(vec![scalar_obs("x", i as f64)], 1).unwrap();
        }

        // Cache should not exceed max_entries
        assert!(loop_.cache().len() <= 2);
    }

    // ------------------------------------------------------------------
    // Adaptive degradation tests
    // ------------------------------------------------------------------

    #[test]
    fn test_world_model_constraints_nominal() {
        use bloomai_core::{PowerState, ThermalState};
        let c =
            WorldModelConstraints::from_thermal_power(ThermalState::Nominal, PowerState::PluggedIn);
        assert!(!c.is_degraded());
        assert_eq!(c.effective_horizon(10), 10);
        assert!(!c.should_skip_observation(1));
        assert!(!c.should_skip_observation(2));
    }

    #[test]
    fn test_world_model_constraints_thermal_serious() {
        use bloomai_core::{PowerState, ThermalState};
        let c =
            WorldModelConstraints::from_thermal_power(ThermalState::Serious, PowerState::PluggedIn);
        assert!(c.is_degraded());
        assert_eq!(c.effective_horizon(10), 4); // max_degraded_horizon = 4
        // Skip every other step
        assert!(!c.should_skip_observation(2)); // step 2 % 2 == 0 -> no skip
        assert!(c.should_skip_observation(1)); // step 1 % 2 != 0 -> skip
    }

    #[test]
    fn test_world_model_constraints_thermal_critical_battery() {
        use bloomai_core::{PowerState, ThermalState};
        let c =
            WorldModelConstraints::from_thermal_power(ThermalState::Critical, PowerState::Battery);
        assert!(c.is_degraded());
        assert!(c.suggest_smaller_model);
        // Critical: max horizon = 1, battery further limits to min(1,2) = 1
        assert_eq!(c.effective_horizon(10), 1);
        // Skip 2 out of 3
        assert!(!c.should_skip_observation(3)); // step 3 % 3 == 0
        assert!(c.should_skip_observation(1));
        assert!(c.should_skip_observation(2));
    }

    #[test]
    fn test_world_model_constraints_battery_only() {
        use bloomai_core::{PowerState, ThermalState};
        let c =
            WorldModelConstraints::from_thermal_power(ThermalState::Nominal, PowerState::Battery);
        assert!(c.is_degraded()); // battery counts as degraded
        assert!(!c.suggest_smaller_model);
        // Battery caps horizon at 2
        assert_eq!(c.effective_horizon(10), 2);
        // No observation skipping for battery alone
        assert!(!c.should_skip_observation(1));
    }

    #[test]
    fn test_world_model_loop_degraded_step_skip() {
        use bloomai_core::{PowerState, ThermalState};
        let wm = Box::new(MockWorldModel::new("mock-wm"));
        let policy = Box::new(MockPolicyEngine::new("mock-policy", "discrete", 2));
        let config = StateCacheConfig::default();
        let mut loop_ = WorldModelLoop::new(wm, policy, config);

        // Set Critical thermal — skip 2 out of 3 observations
        loop_.set_environment(ThermalState::Critical, PowerState::PluggedIn);

        // Run 3 steps
        let mut full_steps = 0;
        let mut skipped_steps = 0;
        for i in 0..3 {
            let output = loop_.step(vec![scalar_obs("x", i as f64)], 5).unwrap();
            // A skipped step only returns Metrics + End (2 chunks)
            if output.len() <= 2 {
                skipped_steps += 1;
            } else {
                full_steps += 1;
            }
        }
        // Critical: skip ratio=3, so step 1,2 skipped, step 3 processed
        assert_eq!(skipped_steps, 2);
        assert_eq!(full_steps, 1);
    }

    #[test]
    fn test_world_model_loop_horizon_reduced() {
        use bloomai_core::{PowerState, ThermalState};
        let wm = Box::new(MockWorldModel::new("mock-wm"));
        let policy = Box::new(MockPolicyEngine::new("mock-policy", "continuous", 2));
        let config = StateCacheConfig::default();
        let mut loop_ = WorldModelLoop::new(wm, policy, config);

        // Serious thermal limits horizon to 4
        loop_.set_environment(ThermalState::Serious, PowerState::PluggedIn);

        // Request horizon=10, should be capped at 4
        let output = loop_.step(vec![scalar_obs("x", 1.0)], 10).unwrap();

        // Count predicted states — should be at most 4 (effective horizon)
        let predicted_count = output
            .iter()
            .filter(|c| matches!(c, OutputChunk::PredictedState { .. }))
            .count();
        assert!(
            predicted_count <= 4,
            "expected <= 4 predicted states, got {}",
            predicted_count
        );
    }

    #[test]
    fn test_world_model_loop_schema_validation() {
        let wm = Box::new(MockWorldModel::new("mock-wm"));
        let policy = Box::new(MockPolicyEngine::new("mock-policy", "discrete", 2));
        let config = StateCacheConfig::default();
        let mut loop_ = WorldModelLoop::new(wm, policy, config);

        let mut state_schema = WorldStateSchema::default();
        state_schema
            .scalar_ranges
            .insert("temp".to_string(), (0.0, 50.0));
        state_schema.allow_text = false;

        let mut action_schema = ActionSchema::default();
        action_schema
            .allowed_action_spaces
            .push("discrete".to_string());
        action_schema
            .action_dimensions
            .insert("discrete".to_string(), 2);
        action_schema.value_range = Some((-1.0, 1.0));

        loop_.set_schemas(Some(state_schema), Some(action_schema));

        // 1. Invalid scalar temp (too high)
        let obs_invalid_scalar = vec![scalar_obs("temp", 60.0)];
        assert!(loop_.step(obs_invalid_scalar, 1).is_err());

        // 2. Valid scalar temp
        let obs_valid_scalar = vec![scalar_obs("temp", 25.0)];
        let output = loop_.step(obs_valid_scalar, 1);
        assert!(output.is_ok());

        // 3. Disallowed text
        let obs_invalid_text = vec![WorldObservation::Text("hello".to_string())];
        assert!(loop_.step(obs_invalid_text, 1).is_err());
    }
}
