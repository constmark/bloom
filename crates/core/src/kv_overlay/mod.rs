use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

fn default_block_size_tokens() -> usize {
    16
}

fn default_max_overlay_entries() -> usize {
    1024
}

fn default_sync_granularity_tokens() -> usize {
    16
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvOverlayConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_block_size_tokens")]
    pub block_size_tokens: usize,
    #[serde(default = "default_max_overlay_entries")]
    pub max_overlay_entries: usize,
    #[serde(default = "default_sync_granularity_tokens")]
    pub sync_granularity_tokens: usize,
    #[serde(default)]
    pub retain_finished_sequences: bool,
}

impl Default for KvOverlayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            block_size_tokens: default_block_size_tokens(),
            max_overlay_entries: default_max_overlay_entries(),
            sync_granularity_tokens: default_sync_granularity_tokens(),
            retain_finished_sequences: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvOverlayEntry {
    pub request_id: String,
    pub model_id: String,
    pub token_count: usize,
    pub synced_tokens: usize,
    pub block_count: usize,
    pub last_synced_at: Option<Instant>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct KvOverlayMetrics {
    pub entries: usize,
    pub syncs: usize,
    pub evictions: usize,
    pub retained_entries: usize,
}

#[derive(Debug, Default)]
struct KvOverlayState {
    entries: HashMap<String, KvOverlayEntry>,
    metrics: KvOverlayMetrics,
}

#[derive(Debug)]
pub struct KvOverlayManager {
    config: KvOverlayConfig,
    state: Mutex<KvOverlayState>,
}

impl KvOverlayManager {
    pub fn new(config: KvOverlayConfig) -> Self {
        Self {
            config,
            state: Mutex::new(KvOverlayState::default()),
        }
    }

    pub fn disabled() -> Self {
        Self::new(KvOverlayConfig::default())
    }

    pub fn config(&self) -> &KvOverlayConfig {
        &self.config
    }

    pub fn attach_sequence(&self, request_id: &str, model_id: &str, token_count: usize) {
        if !self.config.enabled {
            return;
        }

        let block_size = self.config.block_size_tokens.max(1);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.entries.len() >= self.config.max_overlay_entries.max(1) {
            if let Some(oldest_key) = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_synced_at.unwrap_or_else(Instant::now))
                .map(|(key, _)| key.clone())
            {
                state.entries.remove(&oldest_key);
                state.metrics.evictions += 1;
            }
        }

        state.entries.insert(
            request_id.to_string(),
            KvOverlayEntry {
                request_id: request_id.to_string(),
                model_id: model_id.to_string(),
                token_count,
                synced_tokens: 0,
                block_count: token_count.div_ceil(block_size),
                last_synced_at: None,
            },
        );
        state.metrics.entries = state.entries.len();
    }

    pub fn record_tokens(&self, request_id: &str, new_tokens: usize) -> bool {
        if !self.config.enabled {
            return false;
        }

        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let granularity = self.config.sync_granularity_tokens.max(1);
        let should_sync = if let Some(entry) = state.entries.get_mut(request_id) {
            entry.token_count = entry.token_count.saturating_add(new_tokens);
            entry.token_count.saturating_sub(entry.synced_tokens) >= granularity
        } else {
            false
        };

        if should_sync {
            if let Some(entry) = state.entries.get_mut(request_id) {
                entry.synced_tokens = entry.token_count;
                entry.last_synced_at = Some(Instant::now());
            }
            state.metrics.syncs += 1;
        }
        should_sync
    }

    pub fn release_sequence(&self, request_id: &str) {
        if !self.config.enabled || self.config.retain_finished_sequences {
            return;
        }

        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.entries.remove(request_id);
        state.metrics.entries = state.entries.len();
    }

    pub fn metrics(&self) -> KvOverlayMetrics {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        KvOverlayMetrics {
            entries: state.entries.len(),
            retained_entries: state.entries.len(),
            ..state.metrics
        }
    }

    pub fn entry(&self, request_id: &str) -> Option<KvOverlayEntry> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .get(request_id)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_syncs_at_granularity() {
        let overlay = KvOverlayManager::new(KvOverlayConfig {
            enabled: true,
            sync_granularity_tokens: 2,
            ..Default::default()
        });

        overlay.attach_sequence("r1", "m1", 1);
        assert!(!overlay.record_tokens("r1", 0));
        assert!(overlay.record_tokens("r1", 1));
        assert_eq!(overlay.metrics().syncs, 1);
    }

    #[test]
    fn release_removes_finished_sequence_by_default() {
        let overlay = KvOverlayManager::new(KvOverlayConfig {
            enabled: true,
            ..Default::default()
        });

        overlay.attach_sequence("r1", "m1", 4);
        overlay.release_sequence("r1");
        assert!(overlay.entry("r1").is_none());
    }
}
