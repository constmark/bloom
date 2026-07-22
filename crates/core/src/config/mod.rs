use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::kv_overlay::KvOverlayConfig;
use crate::online_switching::OnlineSwitchingConfig;
use crate::token_scheduling::TokenSchedulingConfig;
use crate::unified_memory::UnifiedMemoryConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct BloomConfig {
    #[serde(default)]
    pub token_scheduling: TokenSchedulingConfig,
    #[serde(default)]
    pub unified_memory: UnifiedMemoryConfig,
    #[serde(default)]
    pub kv_overlay: KvOverlayConfig,
    #[serde(default)]
    pub online_switching: OnlineSwitchingConfig,
}

impl BloomConfig {
    pub fn from_json_str(input: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(input)?)
    }

    pub fn from_path(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_missing_sections() {
        let config = BloomConfig::from_json_str(r#"{"kv_overlay":{"enabled":true}}"#).unwrap();
        assert!(config.token_scheduling.enabled);
        assert!(config.unified_memory.enabled);
        assert!(config.kv_overlay.enabled);
        assert!(!config.online_switching.enabled);
    }
}
