//! World model types: state, action, delta, and cache for closed-loop
//! observation → state → prediction/action execution models.

use serde::{Deserialize, Serialize};

use crate::DType;

/// Snapshot of the world at a given step.
///
/// A world state captures observations (sensor readings, text, images, audio)
/// and an optional latent representation produced by a world model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    /// Unique identifier for this state snapshot.
    pub state_id: String,
    /// Observations that produced or describe this state.
    pub observations: Vec<WorldObservation>,
    /// Optional latent representation (opaque bytes from the world model).
    pub latent: Option<Vec<u8>>,
    /// Monotonic step counter.
    pub step: u64,
    /// Wall-clock timestamp (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Size of the latent representation in bytes (for budgeting).
    pub latent_bytes: usize,
}

/// A single observation fed into the world model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldObservation {
    /// Raw text observation (e.g. caption, log line).
    Text(String),
    /// Image bytes with MIME type.
    Image { bytes: Vec<u8>, mime: String },
    /// Audio PCM samples.
    AudioPcm { samples: Vec<f32>, sample_rate: u32 },
    /// Generic tensor observation.
    Tensor {
        dtype: DType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    },
    /// Scalar sensor reading (e.g. temperature, velocity).
    Scalar { name: String, value: f64 },
}

/// An action produced by a policy model given a world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Name of the action space (e.g. "discrete", "continuous", "joint_torque").
    pub action_space: String,
    /// Action values (interpretation depends on action_space).
    pub values: Vec<f32>,
    /// Optional metadata or labels.
    pub metadata: Option<String>,
}

/// Delta between two world states, used for incremental updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDelta {
    /// Source state this delta is relative to.
    pub from_state_id: String,
    /// Target state after applying this delta.
    pub to_state_id: String,
    /// New observations since the source state.
    pub new_observations: Vec<WorldObservation>,
    /// Updated latent (if the world model produces incremental latents).
    pub latent_update: Option<Vec<u8>>,
    /// Predicted future states from the world model.
    pub predicted_futures: Vec<PredictedFuture>,
}

/// A predicted future state from the world model's horizon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedFuture {
    /// Predicted state (lightweight, no full observations).
    pub state_id: String,
    /// Latent of the predicted state.
    pub latent: Option<Vec<u8>>,
    /// Confidence score [0, 1].
    pub confidence: f32,
    /// Steps ahead from current state.
    pub horizon_step: u32,
}

/// Metadata for a cached world state managed by `StateCacheManager`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCacheEntry {
    /// Reference to the world state.
    pub state_id: String,
    /// Size in bytes (latent + overhead).
    pub size_bytes: usize,
    /// When this entry was created (ms since epoch).
    pub created_ms: u64,
    /// Last access time (ms since epoch).
    pub last_access_ms: u64,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Whether the latent has been compressed.
    pub compressed: bool,
    /// Priority for eviction (higher = keep longer).
    pub priority: StateCachePriority,
    /// TTL in milliseconds (0 = no expiry).
    pub ttl_ms: u64,
}

/// Priority levels for state cache entries.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateCachePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Configuration for the state cache manager.
#[derive(Debug, Clone)]
pub struct StateCacheConfig {
    /// Maximum total bytes for all cached states.
    pub max_bytes: usize,
    /// Maximum number of cached states.
    pub max_entries: usize,
    /// Default TTL for new entries (0 = no expiry).
    pub default_ttl_ms: u64,
    /// Whether to enable automatic compression of old entries.
    pub auto_compress: bool,
    /// Entries older than this (ms) are candidates for compression.
    pub compress_after_ms: u64,
}

impl Default for StateCacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024, // 256 MB
            max_entries: 64,
            default_ttl_ms: 0,
            auto_compress: true,
            compress_after_ms: 60_000, // 1 minute
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(id: &str, step: u64, latent_bytes: usize) -> WorldState {
        WorldState {
            state_id: id.to_string(),
            observations: vec![WorldObservation::Scalar {
                name: "sensor".into(),
                value: step as f64,
            }],
            latent: if latent_bytes > 0 {
                Some(vec![0u8; latent_bytes])
            } else {
                None
            },
            step,
            timestamp_ms: 1000 + step * 100,
            latent_bytes,
        }
    }

    #[test]
    fn test_world_state_serde() {
        let state = make_state("s1", 0, 128);
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("s1"));
        let deser: WorldState = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.state_id, "s1");
        assert_eq!(deser.step, 0);
    }

    #[test]
    fn test_action_serde() {
        let action = Action {
            action_space: "continuous".into(),
            values: vec![0.5, -0.3, 1.0],
            metadata: Some("move_forward".into()),
        };
        let json = serde_json::to_string(&action).unwrap();
        let deser: Action = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.values.len(), 3);
        assert_eq!(deser.action_space, "continuous");
    }

    #[test]
    fn test_state_delta_serde() {
        let delta = StateDelta {
            from_state_id: "s0".into(),
            to_state_id: "s1".into(),
            new_observations: vec![WorldObservation::Text("hello".into())],
            latent_update: Some(vec![1, 2, 3]),
            predicted_futures: vec![PredictedFuture {
                state_id: "s2_pred".into(),
                latent: None,
                confidence: 0.85,
                horizon_step: 1,
            }],
        };
        let json = serde_json::to_string(&delta).unwrap();
        let deser: StateDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.predicted_futures.len(), 1);
        assert!((deser.predicted_futures[0].confidence - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_state_cache_entry_priority_ordering() {
        assert!(StateCachePriority::Critical > StateCachePriority::High);
        assert!(StateCachePriority::High > StateCachePriority::Normal);
        assert!(StateCachePriority::Normal > StateCachePriority::Low);
    }

    #[test]
    fn test_state_cache_config_defaults() {
        let cfg = StateCacheConfig::default();
        assert_eq!(cfg.max_bytes, 256 * 1024 * 1024);
        assert_eq!(cfg.max_entries, 64);
        assert!(cfg.auto_compress);
    }
}
