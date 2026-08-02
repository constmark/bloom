//! Stable, path-free export of the local model catalog and acquisition state.

use reqwest::Url;
use serde::{Deserialize, Serialize};

use super::model_index::validate_index_id;
use super::model_manager::{validate_catalog_id, ModelCatalog, ModelCatalogEntry};
use super::model_provenance::{normalize_license, ModelAcquisitionKind, ModelProvenance};

pub(crate) const MODEL_INVENTORY_SCHEMA_VERSION: u8 = 2;
const MIN_MODEL_INVENTORY_SCHEMA_VERSION: u8 = 1;
const MODEL_INVENTORY_RECONCILIATION_SCHEMA_VERSION: u8 = 1;
pub(crate) const MODEL_INVENTORY_OBJECT: &str = "bloom.model_inventory";
pub(crate) const MODEL_INVENTORY_CONTENT_DISPOSITION: &str =
    "attachment; filename=\"bloom-model-inventory.json\"";
pub(crate) const MAX_MODEL_INVENTORY_BYTES: usize = 16 * 1024 * 1024;
const MAX_MODEL_INVENTORY_ENTRIES: usize = 20_000;
const MAX_RECONCILIATION_DRIFT_ENTRIES: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelInventory {
    pub(crate) schema_version: u8,
    pub(crate) object: String,
    pub(crate) summary: ModelInventorySummary,
    pub(crate) models: Vec<ModelInventoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelInventorySummary {
    pub(crate) model_count: usize,
    pub(crate) provenance_count: usize,
    pub(crate) source_locked_count: usize,
    pub(crate) quarantined_count: usize,
    pub(crate) invalid_provenance_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelInventoryEntry {
    pub(crate) id: String,
    pub(crate) kind: String,
    pub(crate) format: String,
    pub(crate) size_bytes: u64,
    pub(crate) size_complete: bool,
    pub(crate) provenance_status: String,
    pub(crate) acquisition: Option<ModelAcquisitionKind>,
    #[serde(default)]
    pub(crate) model_index_id: Option<String>,
    pub(crate) source: Option<ModelInventorySource>,
    pub(crate) sha256: Option<String>,
    pub(crate) license: Option<String>,
    pub(crate) installed_at: Option<u64>,
    pub(crate) last_verified_at: Option<u64>,
    pub(crate) integrity: String,
    pub(crate) source_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelInventorySource {
    pub(crate) url: Option<String>,
    pub(crate) host: Option<String>,
    pub(crate) revision: Option<String>,
    pub(crate) immutable_revision: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelInventoryReconciliation {
    pub(crate) schema_version: u8,
    pub(crate) object: String,
    pub(crate) in_sync: bool,
    pub(crate) truncated: bool,
    pub(crate) summary: ModelInventoryReconciliationSummary,
    pub(crate) drift: Vec<ModelInventoryDrift>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelInventoryReconciliationSummary {
    pub(crate) expected_model_count: usize,
    pub(crate) current_model_count: usize,
    pub(crate) matching_count: usize,
    pub(crate) missing_count: usize,
    pub(crate) unexpected_count: usize,
    pub(crate) changed_count: usize,
    pub(crate) blocking_count: usize,
    pub(crate) restorable_count: usize,
    pub(crate) drift_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelInventoryDrift {
    pub(crate) id: String,
    pub(crate) status: String,
    pub(crate) severity: String,
    pub(crate) changes: Vec<String>,
    pub(crate) restore_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelInventoryRestoreCandidate {
    pub(crate) filename: String,
    pub(crate) url: String,
    pub(crate) sha256: String,
    pub(crate) license: Option<String>,
    pub(crate) size_bytes: u64,
    pub(crate) model_index_id: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum ModelInventoryRestoreError {
    #[error("{0}")]
    InvalidInventory(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Unavailable(String),
}

impl ModelInventory {
    pub(crate) fn from_catalog(catalog: &ModelCatalog) -> Self {
        let mut models = catalog
            .models
            .iter()
            .map(ModelInventoryEntry::from_catalog_entry)
            .collect::<Vec<_>>();
        models.sort_by(|left, right| left.id.cmp(&right.id));
        let summary = ModelInventorySummary {
            model_count: models.len(),
            provenance_count: models
                .iter()
                .filter(|model| model.provenance_status == "recorded")
                .count(),
            source_locked_count: models.iter().filter(|model| model.source_locked).count(),
            quarantined_count: models
                .iter()
                .filter(|model| model.integrity == "quarantined")
                .count(),
            invalid_provenance_count: models
                .iter()
                .filter(|model| model.provenance_status == "invalid")
                .count(),
        };
        Self {
            schema_version: MODEL_INVENTORY_SCHEMA_VERSION,
            object: MODEL_INVENTORY_OBJECT.to_string(),
            summary,
            models,
        }
    }

    pub(crate) fn reconcile(
        expected: &Self,
        current: &Self,
    ) -> Result<ModelInventoryReconciliation, String> {
        expected.validate_import()?;
        let expected_by_id = expected
            .models
            .iter()
            .map(|model| (model.id.as_str(), model))
            .collect::<std::collections::BTreeMap<_, _>>();
        let current_by_id = current
            .models
            .iter()
            .map(|model| (model.id.as_str(), model))
            .collect::<std::collections::BTreeMap<_, _>>();
        let ids = expected_by_id
            .keys()
            .chain(current_by_id.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>();

        let mut all_drift = Vec::new();
        let mut matching_count = 0;
        let mut missing_count = 0;
        let mut unexpected_count = 0;
        let mut changed_count = 0;
        let mut blocking_count = 0;
        let mut restorable_count = 0;
        for id in ids {
            let drift = match (expected_by_id.get(id), current_by_id.get(id)) {
                (Some(expected), None) => {
                    missing_count += 1;
                    let restore_available = inventory_entry_restore_available(expected);
                    if restore_available {
                        restorable_count += 1;
                    }
                    Some(ModelInventoryDrift {
                        id: id.to_string(),
                        status: "missing".to_string(),
                        severity: "blocking".to_string(),
                        changes: vec!["model_missing".to_string()],
                        restore_available,
                    })
                }
                (None, Some(current)) => {
                    unexpected_count += 1;
                    let severity = if current.integrity == "quarantined" {
                        "blocking"
                    } else {
                        "warning"
                    };
                    Some(ModelInventoryDrift {
                        id: id.to_string(),
                        status: "unexpected".to_string(),
                        severity: severity.to_string(),
                        changes: vec!["model_unexpected".to_string()],
                        restore_available: false,
                    })
                }
                (Some(expected), Some(current)) => {
                    let changes = inventory_changes(expected, current);
                    if changes.is_empty() {
                        matching_count += 1;
                        None
                    } else {
                        changed_count += 1;
                        let severity = drift_severity(&changes, current);
                        Some(ModelInventoryDrift {
                            id: id.to_string(),
                            status: "changed".to_string(),
                            severity: severity.to_string(),
                            changes,
                            restore_available: false,
                        })
                    }
                }
                (None, None) => None,
            };
            if let Some(drift) = drift {
                if drift.severity == "blocking" {
                    blocking_count += 1;
                }
                all_drift.push(drift);
            }
        }
        let drift_count = all_drift.len();
        let truncated = drift_count > MAX_RECONCILIATION_DRIFT_ENTRIES;
        all_drift.truncate(MAX_RECONCILIATION_DRIFT_ENTRIES);
        Ok(ModelInventoryReconciliation {
            schema_version: MODEL_INVENTORY_RECONCILIATION_SCHEMA_VERSION,
            object: "bloom.model_inventory_reconciliation".to_string(),
            in_sync: drift_count == 0,
            truncated,
            summary: ModelInventoryReconciliationSummary {
                expected_model_count: expected.models.len(),
                current_model_count: current.models.len(),
                matching_count,
                missing_count,
                unexpected_count,
                changed_count,
                blocking_count,
                restorable_count,
                drift_count,
            },
            drift: all_drift,
        })
    }

    pub(crate) fn restore_candidate(
        expected: &Self,
        current: &Self,
        id: &str,
    ) -> Result<ModelInventoryRestoreCandidate, ModelInventoryRestoreError> {
        expected
            .validate_import()
            .map_err(ModelInventoryRestoreError::InvalidInventory)?;
        validate_catalog_id(id).map_err(|_| {
            ModelInventoryRestoreError::Invalid("model inventory restore ID is invalid".to_string())
        })?;
        let model = expected
            .models
            .binary_search_by(|model| model.id.as_str().cmp(id))
            .ok()
            .and_then(|index| expected.models.get(index))
            .ok_or_else(|| {
                ModelInventoryRestoreError::NotFound(format!(
                    "Model '{id}' is not present in the expected inventory."
                ))
            })?;
        if current
            .models
            .binary_search_by(|model| model.id.as_str().cmp(id))
            .is_ok()
        {
            return Err(ModelInventoryRestoreError::Conflict(format!(
                "A current catalog entry named '{id}' already exists."
            )));
        }
        if !inventory_entry_restore_available(model) {
            return Err(ModelInventoryRestoreError::Unavailable(format!(
                "Model '{id}' does not have a restorable exact-commit download record."
            )));
        }
        let source = model.source.as_ref().ok_or_else(|| {
            ModelInventoryRestoreError::Unavailable(format!(
                "Model '{id}' does not have restorable source metadata."
            ))
        })?;
        Ok(ModelInventoryRestoreCandidate {
            filename: model.id.clone(),
            url: source.url.clone().ok_or_else(|| {
                ModelInventoryRestoreError::Unavailable(format!(
                    "Model '{id}' does not have a restorable source URL."
                ))
            })?,
            sha256: model.sha256.clone().ok_or_else(|| {
                ModelInventoryRestoreError::Unavailable(format!(
                    "Model '{id}' does not have a restorable checksum."
                ))
            })?,
            license: model.license.clone(),
            size_bytes: model.size_bytes,
            model_index_id: model.model_index_id.clone(),
        })
    }

    fn validate_import(&self) -> Result<(), String> {
        if !(MIN_MODEL_INVENTORY_SCHEMA_VERSION..=MODEL_INVENTORY_SCHEMA_VERSION)
            .contains(&self.schema_version)
        {
            return Err(format!(
                "unsupported model inventory schema version: {}",
                self.schema_version
            ));
        }
        if self.object != MODEL_INVENTORY_OBJECT {
            return Err("model inventory object is invalid".to_string());
        }
        if self.models.len() > MAX_MODEL_INVENTORY_ENTRIES {
            return Err(format!(
                "model inventory must not contain more than {MAX_MODEL_INVENTORY_ENTRIES} entries"
            ));
        }
        if self.summary.model_count != self.models.len() {
            return Err("model inventory summary count is inconsistent".to_string());
        }
        if self.models.windows(2).any(|pair| pair[0].id >= pair[1].id) {
            return Err("model inventory IDs must be unique and sorted".to_string());
        }
        for model in &self.models {
            validate_inventory_entry(model)?;
            if self.schema_version == 1 && model.model_index_id.is_some() {
                return Err(
                    "model inventory schema version 1 cannot contain model_index_id".to_string(),
                );
            }
        }
        let expected_summary = ModelInventorySummary {
            model_count: self.models.len(),
            provenance_count: self
                .models
                .iter()
                .filter(|model| model.provenance_status == "recorded")
                .count(),
            source_locked_count: self
                .models
                .iter()
                .filter(|model| model.source_locked)
                .count(),
            quarantined_count: self
                .models
                .iter()
                .filter(|model| model.integrity == "quarantined")
                .count(),
            invalid_provenance_count: self
                .models
                .iter()
                .filter(|model| model.provenance_status == "invalid")
                .count(),
        };
        if self.summary != expected_summary {
            return Err("model inventory summary is inconsistent".to_string());
        }
        Ok(())
    }
}

impl ModelInventoryEntry {
    fn from_catalog_entry(entry: &ModelCatalogEntry) -> Self {
        let provenance_status = if entry.provenance.is_some() {
            "recorded"
        } else if entry.provenance_error.is_some() {
            "invalid"
        } else {
            "not_recorded"
        };
        let source = entry.provenance.as_ref().and_then(inventory_source);
        let source_locked = source
            .as_ref()
            .is_some_and(|source| source.immutable_revision);
        let integrity = match entry.provenance.as_ref() {
            Some(provenance) if provenance.integrity_mismatch_at.is_some() => "quarantined",
            Some(provenance) if provenance.last_verified_at.is_some() => "verified",
            Some(_) => "verified_at_acquisition",
            None => "untracked",
        };
        Self {
            id: entry.id.clone(),
            kind: entry.kind.clone(),
            format: entry.format.clone(),
            size_bytes: entry.size_bytes,
            size_complete: entry.size_complete,
            provenance_status: provenance_status.to_string(),
            acquisition: entry
                .provenance
                .as_ref()
                .map(|provenance| provenance.acquisition),
            model_index_id: entry
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.model_index_id.clone()),
            source,
            sha256: entry
                .provenance
                .as_ref()
                .map(|provenance| provenance.sha256.clone()),
            license: entry
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.license.clone()),
            installed_at: entry
                .provenance
                .as_ref()
                .map(|provenance| provenance.installed_at),
            last_verified_at: entry
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.last_verified_at),
            integrity: integrity.to_string(),
            source_locked,
        }
    }
}

fn validate_inventory_entry(model: &ModelInventoryEntry) -> Result<(), String> {
    validate_catalog_id(&model.id).map_err(|_| "model inventory ID is invalid".to_string())?;
    if model.id.len() > 255 || model.id.chars().any(char::is_control) {
        return Err("model inventory ID is too long or contains control characters".to_string());
    }
    if !matches!(model.kind.as_str(), "file" | "directory") {
        return Err(format!(
            "model inventory kind is invalid for '{}'",
            model.id
        ));
    }
    if model.format.is_empty()
        || model.format.len() > 64
        || !model
            .format
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(format!(
            "model inventory format is invalid for '{}'",
            model.id
        ));
    }
    if !matches!(
        model.provenance_status.as_str(),
        "recorded" | "not_recorded" | "invalid"
    ) {
        return Err(format!(
            "model inventory provenance status is invalid for '{}'",
            model.id
        ));
    }
    if !matches!(
        model.integrity.as_str(),
        "untracked" | "verified_at_acquisition" | "verified" | "quarantined"
    ) {
        return Err(format!(
            "model inventory integrity status is invalid for '{}'",
            model.id
        ));
    }
    if let Some(sha256) = model.sha256.as_deref() {
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!(
                "model inventory SHA-256 is invalid for '{}'",
                model.id
            ));
        }
    }
    let normalized_license = normalize_license(model.license.clone())
        .map_err(|_| format!("model inventory license is invalid for '{}'", model.id))?;
    if normalized_license != model.license {
        return Err(format!(
            "model inventory license is not normalized for '{}'",
            model.id
        ));
    }
    if let Some(model_index_id) = model.model_index_id.as_deref() {
        validate_index_id(model_index_id).map_err(|_| {
            format!(
                "model inventory signed-index ID is invalid for '{}'",
                model.id
            )
        })?;
        if model.provenance_status != "recorded"
            || model.acquisition != Some(ModelAcquisitionKind::Download)
        {
            return Err(format!(
                "model inventory signed-index ID requires recorded download provenance for '{}'",
                model.id
            ));
        }
    }
    if model.installed_at == Some(0) || model.last_verified_at == Some(0) {
        return Err(format!(
            "model inventory timestamp is invalid for '{}'",
            model.id
        ));
    }
    validate_inventory_source(model)?;

    if model.provenance_status == "recorded" {
        if model.kind != "file" || !model.size_complete || model.size_bytes == 0 {
            return Err(format!(
                "recorded model provenance requires a complete non-empty file for '{}'",
                model.id
            ));
        }
        if model.acquisition.is_none() || model.sha256.is_none() || model.installed_at.is_none() {
            return Err(format!(
                "recorded model provenance is incomplete for '{}'",
                model.id
            ));
        }
        if model.integrity == "untracked" {
            return Err(format!(
                "recorded model provenance cannot be untracked for '{}'",
                model.id
            ));
        }
        if model.integrity == "verified" && model.last_verified_at.is_none() {
            return Err(format!(
                "verified model inventory entry lacks a verification time for '{}'",
                model.id
            ));
        }
        if model.integrity == "verified_at_acquisition" && model.last_verified_at.is_some() {
            return Err(format!(
                "acquisition-only model inventory entry has a later verification time for '{}'",
                model.id
            ));
        }
    } else if model.acquisition.is_some()
        || model.model_index_id.is_some()
        || model.source.is_some()
        || model.sha256.is_some()
        || model.license.is_some()
        || model.installed_at.is_some()
        || model.last_verified_at.is_some()
        || model.integrity != "untracked"
        || model.source_locked
    {
        return Err(format!(
            "unrecorded model inventory entry contains provenance for '{}'",
            model.id
        ));
    }
    Ok(())
}

fn inventory_entry_restore_available(model: &ModelInventoryEntry) -> bool {
    model.kind == "file"
        && model.size_complete
        && model.size_bytes > 0
        && model.provenance_status == "recorded"
        && model.acquisition == Some(ModelAcquisitionKind::Download)
        && model.sha256.is_some()
        && model.source_locked
        && model.source.as_ref().is_some_and(|source| {
            source.url.is_some() && source.revision.is_some() && source.immutable_revision
        })
}

fn validate_inventory_source(model: &ModelInventoryEntry) -> Result<(), String> {
    let Some(source) = model.source.as_ref() else {
        if model.source_locked {
            return Err(format!(
                "source lock is missing source metadata for '{}'",
                model.id
            ));
        }
        return Ok(());
    };
    if source.host.as_deref().is_some_and(|host| {
        host.is_empty()
            || host.len() > 253
            || host.chars().any(char::is_control)
            || host != host.to_ascii_lowercase()
    }) {
        return Err(format!(
            "model inventory source host is invalid for '{}'",
            model.id
        ));
    }
    let parsed = match source.url.as_deref() {
        Some(value) => {
            if value.len() > 2_048 {
                return Err(format!(
                    "model inventory source URL is too long for '{}'",
                    model.id
                ));
            }
            let url = Url::parse(value)
                .map_err(|_| format!("model inventory source URL is invalid for '{}'", model.id))?;
            if url.scheme() != "https"
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err(format!(
                    "model inventory source URL is unsafe for '{}'",
                    model.id
                ));
            }
            let host = url.host_str().ok_or_else(|| {
                format!("model inventory source URL lacks a host for '{}'", model.id)
            })?;
            if source
                .host
                .as_deref()
                .is_some_and(|recorded| !recorded.eq_ignore_ascii_case(host))
            {
                return Err(format!(
                    "model inventory source host does not match its URL for '{}'",
                    model.id
                ));
            }
            Some(url)
        }
        None => None,
    };
    if source.url.is_none() && source.host.is_none() {
        return Err(format!(
            "model inventory source is empty for '{}'",
            model.id
        ));
    }
    let revision = parsed.as_ref().and_then(hugging_face_revision);
    if source.revision != revision {
        return Err(format!(
            "model inventory source revision is inconsistent for '{}'",
            model.id
        ));
    }
    let immutable_revision = revision.as_deref().is_some_and(is_commit_revision);
    if source.immutable_revision != immutable_revision || model.source_locked != immutable_revision
    {
        return Err(format!(
            "model inventory source lock is inconsistent for '{}'",
            model.id
        ));
    }
    Ok(())
}

fn inventory_changes(expected: &ModelInventoryEntry, current: &ModelInventoryEntry) -> Vec<String> {
    let mut changes = Vec::new();
    if expected.kind != current.kind {
        changes.push("kind".to_string());
    }
    if expected.format != current.format {
        changes.push("format".to_string());
    }
    if expected.size_bytes != current.size_bytes {
        changes.push("size_bytes".to_string());
    }
    if expected.size_complete != current.size_complete {
        changes.push("size_complete".to_string());
    }
    if expected.provenance_status != current.provenance_status {
        changes.push("provenance_status".to_string());
    }
    if expected.acquisition != current.acquisition {
        changes.push("acquisition".to_string());
    }
    if expected.model_index_id != current.model_index_id {
        changes.push("model_index_id".to_string());
    }
    let expected_source = expected.source.as_ref();
    let current_source = current.source.as_ref();
    if expected_source.and_then(|source| source.url.as_ref())
        != current_source.and_then(|source| source.url.as_ref())
    {
        changes.push("source_url".to_string());
    }
    if expected_source.and_then(|source| source.host.as_ref())
        != current_source.and_then(|source| source.host.as_ref())
    {
        changes.push("source_host".to_string());
    }
    if expected_source.and_then(|source| source.revision.as_ref())
        != current_source.and_then(|source| source.revision.as_ref())
    {
        changes.push("source_revision".to_string());
    }
    if expected.source_locked != current.source_locked
        || expected_source.map(|source| source.immutable_revision)
            != current_source.map(|source| source.immutable_revision)
    {
        changes.push("source_lock".to_string());
    }
    if expected.sha256 != current.sha256 {
        changes.push("sha256".to_string());
    }
    if expected.license != current.license {
        changes.push("license".to_string());
    }
    if expected.installed_at != current.installed_at {
        changes.push("installed_at".to_string());
    }
    if expected.last_verified_at != current.last_verified_at {
        changes.push("last_verified_at".to_string());
    }
    if expected.integrity != current.integrity {
        changes.push("integrity".to_string());
    }
    changes
}

fn drift_severity(changes: &[String], current: &ModelInventoryEntry) -> &'static str {
    if current.integrity == "quarantined"
        || changes.iter().any(|change| {
            matches!(
                change.as_str(),
                "kind"
                    | "format"
                    | "size_bytes"
                    | "size_complete"
                    | "provenance_status"
                    | "model_index_id"
                    | "sha256"
            )
        })
    {
        "blocking"
    } else {
        "warning"
    }
}

fn inventory_source(provenance: &ModelProvenance) -> Option<ModelInventorySource> {
    let mut parsed = provenance
        .source_url
        .as_deref()
        .and_then(|value| Url::parse(value).ok())
        .filter(|url| {
            url.scheme() == "https" && url.username().is_empty() && url.password().is_none()
        });
    if let Some(url) = parsed.as_mut() {
        url.set_query(None);
        url.set_fragment(None);
    }
    let url = parsed.as_ref().map(ToString::to_string);
    let host = parsed
        .as_ref()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .or_else(|| provenance.source_host.clone());
    let revision = parsed.as_ref().and_then(hugging_face_revision);
    let immutable_revision = revision.as_deref().is_some_and(is_commit_revision);

    (url.is_some() || host.is_some()).then_some(ModelInventorySource {
        url,
        host,
        revision,
        immutable_revision,
    })
}

fn hugging_face_revision(url: &Url) -> Option<String> {
    let host = url.host_str()?.to_ascii_lowercase();
    if host != "huggingface.co" && host != "www.huggingface.co" {
        return None;
    }
    let segments = url.path_segments()?.collect::<Vec<_>>();
    let marker = segments.iter().position(|segment| *segment == "resolve")?;
    if marker != 2
        || segments.first().is_none_or(|segment| segment.is_empty())
        || segments.get(1).is_none_or(|segment| segment.is_empty())
        || segments
            .get(marker + 2)
            .is_none_or(|segment| segment.is_empty())
    {
        return None;
    }
    segments
        .get(marker + 1)
        .filter(|revision| !revision.is_empty())
        .map(|revision| revision.to_string())
}

fn is_commit_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, source_url: Option<String>) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: id.to_string(),
            name: id.to_string(),
            kind: "file".to_string(),
            format: "gguf".to_string(),
            size_bytes: 42,
            size_complete: true,
            modified_at: Some(1),
            active: true,
            provenance: Some(ModelProvenance {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: None,
                source_url,
                source_host: Some("huggingface.co".to_string()),
                sha256: "ab".repeat(32),
                file_count: None,
                license: Some("Apache-2.0".to_string()),
                installed_at: 10,
                last_verified_at: Some(11),
                integrity_mismatch_at: None,
            }),
            provenance_error: None,
        }
    }

    #[test]
    fn inventory_is_stable_sorted_and_excludes_local_runtime_state() {
        let revision = "a".repeat(40);
        let catalog = ModelCatalog {
            root: "/private/catalog".to_string(),
            root_exists: true,
            models: vec![
                entry(
                    "zeta.gguf",
                    Some(format!(
                        "https://huggingface.co/acme/model/resolve/{revision}/zeta.gguf?token=secret#fragment"
                    )),
                ),
                entry(
                    "alpha.gguf",
                    Some(
                        "https://huggingface.co/acme/model/resolve/main/alpha.gguf".to_string(),
                    ),
                ),
            ],
        };

        let inventory = ModelInventory::from_catalog(&catalog);
        let first = serde_json::to_string_pretty(&inventory).unwrap();
        let second = serde_json::to_string_pretty(&ModelInventory::from_catalog(&catalog)).unwrap();

        assert_eq!(first, second);
        assert_eq!(inventory.models[0].id, "alpha.gguf");
        assert_eq!(inventory.models[1].id, "zeta.gguf");
        assert!(!inventory.models[0].source_locked);
        assert!(inventory.models[1].source_locked);
        assert_eq!(
            inventory.models[1]
                .source
                .as_ref()
                .and_then(|source| source.revision.as_deref()),
            Some(revision.as_str())
        );
        assert!(!first.contains("/private/catalog"));
        assert!(!first.contains("secret"));
        assert!(!first.contains("fragment"));
        assert!(!first.contains("active"));
        assert!(!first.contains("modified_at"));
    }

    #[test]
    fn inventory_summarizes_missing_invalid_and_quarantined_provenance() {
        let mut quarantined = entry("quarantined.gguf", None);
        quarantined
            .provenance
            .as_mut()
            .unwrap()
            .integrity_mismatch_at = Some(12);
        let mut missing = entry("manual.gguf", None);
        missing.provenance = None;
        let mut invalid = entry("invalid.gguf", None);
        invalid.provenance = None;
        invalid.provenance_error = Some("hidden detail".to_string());
        let catalog = ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: vec![quarantined, missing, invalid],
        };

        let inventory = ModelInventory::from_catalog(&catalog);

        assert_eq!(inventory.summary.model_count, 3);
        assert_eq!(inventory.summary.provenance_count, 1);
        assert_eq!(inventory.summary.quarantined_count, 1);
        assert_eq!(inventory.summary.invalid_provenance_count, 1);
        assert_eq!(inventory.models[0].provenance_status, "invalid");
        assert_eq!(inventory.models[1].integrity, "untracked");
        assert_eq!(inventory.models[2].integrity, "quarantined");
    }

    #[test]
    fn reconciliation_reports_deterministic_drift_without_echoing_values() {
        let matching = entry("alpha.gguf", None);
        let expected_changed = entry("changed.gguf", None);
        let expected_missing = entry("missing.gguf", None);
        let mut current_changed = expected_changed.clone();
        current_changed.size_bytes = 84;
        current_changed.provenance.as_mut().unwrap().sha256 = "cd".repeat(32);
        current_changed.provenance.as_mut().unwrap().license = Some("MIT".to_string());
        let expected = ModelInventory::from_catalog(&ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: vec![matching.clone(), expected_changed, expected_missing],
        });
        let current = ModelInventory::from_catalog(&ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: vec![matching, current_changed, entry("unexpected.gguf", None)],
        });

        let report = ModelInventory::reconcile(&expected, &current).unwrap();

        assert!(!report.in_sync);
        assert!(!report.truncated);
        assert_eq!(report.summary.matching_count, 1);
        assert_eq!(report.summary.missing_count, 1);
        assert_eq!(report.summary.unexpected_count, 1);
        assert_eq!(report.summary.changed_count, 1);
        assert_eq!(report.summary.blocking_count, 2);
        assert_eq!(report.summary.restorable_count, 0);
        assert_eq!(report.summary.drift_count, 3);
        assert_eq!(report.drift[0].id, "changed.gguf");
        assert_eq!(report.drift[0].status, "changed");
        assert_eq!(report.drift[0].severity, "blocking");
        assert!(!report.drift[0].restore_available);
        assert_eq!(
            report.drift[0].changes,
            vec!["size_bytes", "sha256", "license"]
        );
        assert_eq!(report.drift[1].id, "missing.gguf");
        assert_eq!(report.drift[2].id, "unexpected.gguf");
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains(&"cd".repeat(32)));
        assert!(!json.contains("MIT"));
    }

    #[test]
    fn imported_inventory_rejects_inconsistent_or_unsafe_documents() {
        let mut inventory = ModelInventory::from_catalog(&ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: vec![entry(
                "model.gguf",
                Some("https://huggingface.co/acme/model/resolve/main/model.gguf".to_string()),
            )],
        });
        assert!(inventory.validate_import().is_ok());

        inventory.schema_version = 3;
        assert!(inventory.validate_import().unwrap_err().contains("version"));
        inventory.schema_version = 2;
        inventory.summary.model_count = 2;
        assert!(inventory.validate_import().unwrap_err().contains("summary"));
        inventory.summary.model_count = 1;
        inventory.models[0].source.as_mut().unwrap().url = Some(
            "https://huggingface.co/acme/model/resolve/main/model.gguf?token=secret".to_string(),
        );
        let error = inventory.validate_import().unwrap_err();
        assert!(error.contains("unsafe"));
        assert!(!error.contains("secret"));

        inventory.models[0].source.as_mut().unwrap().url =
            Some("https://huggingface.co/acme/model/resolve/main/model.gguf".to_string());
        inventory.models[0].kind = "directory".to_string();
        assert!(inventory
            .validate_import()
            .unwrap_err()
            .contains("complete non-empty file"));
    }

    #[test]
    fn reconciliation_truncates_details_but_preserves_complete_counts() {
        let expected = ModelInventory::from_catalog(&ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: (0..=MAX_RECONCILIATION_DRIFT_ENTRIES)
                .map(|index| entry(&format!("model-{index:03}.gguf"), None))
                .collect(),
        });
        let current = ModelInventory::from_catalog(&ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: Vec::new(),
        });

        let report = ModelInventory::reconcile(&expected, &current).unwrap();

        assert!(report.truncated);
        assert_eq!(
            report.summary.drift_count,
            MAX_RECONCILIATION_DRIFT_ENTRIES + 1
        );
        assert_eq!(report.drift.len(), MAX_RECONCILIATION_DRIFT_ENTRIES);
        assert_eq!(report.summary.blocking_count, report.summary.drift_count);
    }

    #[test]
    fn restore_candidate_requires_a_missing_exact_commit_download() {
        let revision = "a".repeat(40);
        let mut restorable = entry(
            "restorable.gguf",
            Some(format!(
                "https://huggingface.co/acme/model/resolve/{revision}/restorable.gguf"
            )),
        );
        restorable.provenance.as_mut().unwrap().model_index_id = Some("restorable-q4".to_string());
        let expected = ModelInventory::from_catalog(&ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: vec![restorable],
        });
        let empty = ModelInventory::from_catalog(&ModelCatalog {
            root: "ignored".to_string(),
            root_exists: true,
            models: Vec::new(),
        });

        let report = ModelInventory::reconcile(&expected, &empty).unwrap();
        assert_eq!(report.summary.restorable_count, 1);
        assert!(report.drift[0].restore_available);

        let candidate =
            ModelInventory::restore_candidate(&expected, &empty, "restorable.gguf").unwrap();
        assert_eq!(candidate.filename, "restorable.gguf");
        assert_eq!(candidate.sha256, "ab".repeat(32));
        assert_eq!(candidate.size_bytes, 42);
        assert_eq!(candidate.model_index_id.as_deref(), Some("restorable-q4"));
        assert!(candidate.url.contains(&revision));

        assert!(matches!(
            ModelInventory::restore_candidate(&expected, &expected, "restorable.gguf"),
            Err(ModelInventoryRestoreError::Conflict(_))
        ));

        let mut mutable = expected.clone();
        mutable.models[0].source.as_mut().unwrap().url =
            Some("https://huggingface.co/acme/model/resolve/main/restorable.gguf".to_string());
        mutable.models[0].source.as_mut().unwrap().revision = Some("main".to_string());
        mutable.models[0]
            .source
            .as_mut()
            .unwrap()
            .immutable_revision = false;
        mutable.models[0].source_locked = false;
        mutable.summary.source_locked_count = 0;
        assert!(matches!(
            ModelInventory::restore_candidate(&mutable, &empty, "restorable.gguf"),
            Err(ModelInventoryRestoreError::Unavailable(_))
        ));

        let mut imported = expected;
        imported.models[0].acquisition = Some(ModelAcquisitionKind::Import);
        imported.models[0].model_index_id = None;
        assert!(matches!(
            ModelInventory::restore_candidate(&imported, &empty, "restorable.gguf"),
            Err(ModelInventoryRestoreError::Unavailable(_))
        ));
    }

    #[test]
    fn source_lock_requires_a_complete_repository_resolve_path() {
        let revision = "a".repeat(40);
        let complete = Url::parse(&format!(
            "https://huggingface.co/acme/model/resolve/{revision}/model.gguf"
        ))
        .unwrap();
        let missing_repository = Url::parse(&format!(
            "https://huggingface.co/resolve/{revision}/model.gguf"
        ))
        .unwrap();
        let missing_file = Url::parse(&format!(
            "https://huggingface.co/acme/model/resolve/{revision}"
        ))
        .unwrap();

        assert_eq!(hugging_face_revision(&complete), Some(revision));
        assert_eq!(hugging_face_revision(&missing_repository), None);
        assert_eq!(hugging_face_revision(&missing_file), None);
    }
}
