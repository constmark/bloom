//! Deployer-defined license admission for writable model acquisitions.

use anyhow::{anyhow, Result};
use serde::Serialize;

use super::model_provenance::normalize_license;

const MAX_ALLOWED_LICENSES: usize = 64;

#[derive(Debug, Clone, Default)]
pub(crate) struct ModelLicensePolicy {
    allowed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelLicensePolicyStatus {
    pub enforced: bool,
    pub allowed: Vec<String>,
}

impl ModelLicensePolicy {
    pub(crate) fn new(values: Vec<String>) -> Result<Self> {
        if values.len() > MAX_ALLOWED_LICENSES {
            return Err(anyhow!(
                "model license policy must not contain more than {MAX_ALLOWED_LICENSES} declarations"
            ));
        }
        let mut allowed = Vec::with_capacity(values.len());
        for value in values {
            let value = normalize_license(Some(value))?
                .ok_or_else(|| anyhow!("model license policy entries must not be empty"))?;
            if !allowed
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(&value))
            {
                allowed.push(value);
            }
        }
        Ok(Self { allowed })
    }

    pub(crate) fn enforce(&self, value: Option<String>) -> Result<Option<String>> {
        let value = normalize_license(value)?;
        if self.allowed.is_empty() {
            return Ok(value);
        }
        let value = value.ok_or_else(|| {
            anyhow!("a model license is required by the server acquisition policy")
        })?;
        self.allowed
            .iter()
            .find(|allowed| allowed.eq_ignore_ascii_case(&value))
            .cloned()
            .map(Some)
            .ok_or_else(|| {
                anyhow!("the model license is not allowed by the server acquisition policy")
            })
    }

    pub(crate) fn status(&self) -> ModelLicensePolicyStatus {
        ModelLicensePolicyStatus {
            enforced: !self.allowed.is_empty(),
            allowed: self.allowed.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_policy_preserves_optional_normalized_declarations() {
        let policy = ModelLicensePolicy::default();

        assert_eq!(policy.enforce(None).unwrap(), None);
        assert_eq!(
            policy.enforce(Some(" Apache-2.0 ".to_string())).unwrap(),
            Some("Apache-2.0".to_string())
        );
        assert!(!policy.status().enforced);
    }

    #[test]
    fn allowlist_requires_a_match_and_returns_configured_casing() {
        let policy = ModelLicensePolicy::new(vec![
            "Apache-2.0".to_string(),
            "MIT".to_string(),
            "apache-2.0".to_string(),
        ])
        .unwrap();

        assert_eq!(policy.status().allowed, vec!["Apache-2.0", "MIT"]);
        assert_eq!(
            policy.enforce(Some("apache-2.0".to_string())).unwrap(),
            Some("Apache-2.0".to_string())
        );
        assert!(policy.enforce(None).is_err());
        assert!(policy.enforce(Some("GPL-3.0-only".to_string())).is_err());
    }

    #[test]
    fn policy_configuration_is_bounded_and_rejects_empty_entries() {
        assert!(ModelLicensePolicy::new(vec![String::new()]).is_err());
        assert!(
            ModelLicensePolicy::new(vec!["MIT".to_string(); MAX_ALLOWED_LICENSES + 1]).is_err()
        );
    }
}
