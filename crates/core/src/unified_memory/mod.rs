use serde::{Deserialize, Serialize};

use crate::{MemoryTopology, ResourcePriority};

fn default_true() -> bool {
    true
}

fn default_cache_budget_bytes() -> usize {
    512 * 1024 * 1024
}

fn default_safety_margin_percent() -> u8 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnifiedMemoryConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub ram_budget_bytes: Option<usize>,
    #[serde(default)]
    pub vram_budget_bytes: Option<usize>,
    #[serde(default = "default_cache_budget_bytes")]
    pub cache_budget_bytes: usize,
    #[serde(default = "default_safety_margin_percent")]
    pub safety_margin_percent: u8,
    #[serde(default)]
    pub topology: Option<MemoryTopology>,
}

impl Default for UnifiedMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ram_budget_bytes: None,
            vram_budget_bytes: None,
            cache_budget_bytes: default_cache_budget_bytes(),
            safety_margin_percent: default_safety_margin_percent(),
            topology: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryReservation {
    pub model_bytes: usize,
    pub cache_bytes: usize,
    pub priority: ResourcePriority,
}

impl UnifiedMemoryConfig {
    pub fn effective_ram_budget(&self, detected_ram_bytes: usize) -> usize {
        if !self.enabled {
            return detected_ram_bytes;
        }
        let raw = self.ram_budget_bytes.unwrap_or(detected_ram_bytes);
        apply_margin(raw, self.safety_margin_percent)
    }

    pub fn effective_vram_budget(&self, detected_vram_bytes: usize) -> usize {
        if !self.enabled {
            return detected_vram_bytes;
        }
        let raw = self.vram_budget_bytes.unwrap_or(detected_vram_bytes);
        apply_margin(raw, self.safety_margin_percent)
    }

    pub fn estimate_kv_cache_bytes(
        &self,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        tokens: usize,
        bytes_per_element: usize,
    ) -> usize {
        if !self.enabled {
            return 0;
        }
        let per_token = layers
            .saturating_mul(kv_heads)
            .saturating_mul(head_dim)
            .saturating_mul(bytes_per_element)
            .saturating_mul(2);
        per_token
            .saturating_mul(tokens)
            .min(self.cache_budget_bytes)
    }
}

fn apply_margin(bytes: usize, margin_percent: u8) -> usize {
    let margin = margin_percent.min(90) as usize;
    bytes.saturating_mul(100usize.saturating_sub(margin)) / 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_applies_safety_margin() {
        let config = UnifiedMemoryConfig {
            safety_margin_percent: 25,
            ..Default::default()
        };

        assert_eq!(config.effective_ram_budget(1000), 750);
    }

    #[test]
    fn cache_estimate_is_capped() {
        let config = UnifiedMemoryConfig {
            cache_budget_bytes: 64,
            ..Default::default()
        };

        assert_eq!(config.estimate_kv_cache_bytes(2, 2, 2, 10, 2), 64);
    }
}
