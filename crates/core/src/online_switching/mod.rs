use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ResourceSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSwitchCandidate {
    #[serde(default)]
    pub id: Option<String>,
    pub model_path: PathBuf,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub warm: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SwitchPolicy {
    #[default]
    PreferActive,
    RequestedModel,
    LowestMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OnlineSwitchingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub policy: SwitchPolicy,
    #[serde(default)]
    pub allow_request_model_override: bool,
    #[serde(default)]
    pub candidates: Vec<ModelSwitchCandidate>,
}

impl Default for OnlineSwitchingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy: SwitchPolicy::PreferActive,
            allow_request_model_override: true,
            candidates: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchDecision {
    UseActive { model_id: String },
    Switch { model_id: String },
    Reject { reason: String },
}

#[derive(Debug, Clone)]
pub struct OnlineSwitchingPolicy {
    config: OnlineSwitchingConfig,
}

impl OnlineSwitchingPolicy {
    pub fn new(config: OnlineSwitchingConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &OnlineSwitchingConfig {
        &self.config
    }

    pub fn decide(
        &self,
        active_model_id: &str,
        requested_model_id: Option<&str>,
        loaded_model_ids: &[String],
        _snapshot: Option<&ResourceSnapshot>,
    ) -> SwitchDecision {
        if !self.config.enabled {
            return match requested_model_id {
                Some(requested) if requested != active_model_id => SwitchDecision::Reject {
                    reason: format!(
                        "model switching is disabled; active model is '{}'",
                        active_model_id
                    ),
                },
                _ => SwitchDecision::UseActive {
                    model_id: active_model_id.to_string(),
                },
            };
        }

        if let Some(requested) = requested_model_id {
            if requested == active_model_id {
                return SwitchDecision::UseActive {
                    model_id: active_model_id.to_string(),
                };
            }
            if !self.config.allow_request_model_override {
                return SwitchDecision::Reject {
                    reason: "request model override is disabled".to_string(),
                };
            }
            if loaded_model_ids.iter().any(|id| id == requested) {
                return SwitchDecision::Switch {
                    model_id: requested.to_string(),
                };
            }
            return SwitchDecision::Reject {
                reason: format!("requested model '{}' is not loaded", requested),
            };
        }

        match self.config.policy {
            SwitchPolicy::PreferActive | SwitchPolicy::RequestedModel => {
                SwitchDecision::UseActive {
                    model_id: active_model_id.to_string(),
                }
            }
            SwitchPolicy::LowestMemory => loaded_model_ids
                .first()
                .map(|id| {
                    if id == active_model_id {
                        SwitchDecision::UseActive {
                            model_id: active_model_id.to_string(),
                        }
                    } else {
                        SwitchDecision::Switch {
                            model_id: id.clone(),
                        }
                    }
                })
                .unwrap_or_else(|| SwitchDecision::UseActive {
                    model_id: active_model_id.to_string(),
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_policy_rejects_non_active_model() {
        let policy = OnlineSwitchingPolicy::new(OnlineSwitchingConfig::default());
        let decision = policy.decide("active", Some("other"), &["active".to_string()], None);
        assert!(matches!(decision, SwitchDecision::Reject { .. }));
    }

    #[test]
    fn enabled_policy_switches_to_loaded_requested_model() {
        let policy = OnlineSwitchingPolicy::new(OnlineSwitchingConfig {
            enabled: true,
            ..Default::default()
        });
        let decision = policy.decide(
            "active",
            Some("other"),
            &["active".to_string(), "other".to_string()],
            None,
        );
        assert_eq!(
            decision,
            SwitchDecision::Switch {
                model_id: "other".to_string()
            }
        );
    }
}
