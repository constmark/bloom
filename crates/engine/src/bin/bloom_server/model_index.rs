//! Signed, bounded model discovery indexes for verified acquisitions.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
use futures::StreamExt as _;
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncReadExt as _;
use tokio::sync::{Mutex, RwLock};

use super::model_index_state::{
    ModelIndexWatermark, ModelIndexWatermarkAdmission, ModelIndexWatermarkStore,
};
use super::model_license::ModelLicensePolicy;
use super::model_manager::{validate_model_filename, ModelCatalog, ModelCatalogEntry};
use super::model_package::{
    normalize_package_files, package_digest, validate_package_id, ModelPackageFile,
};
use super::model_provenance::{normalize_license, ModelAcquisitionKind};
use super::model_upgrade::{ModelUpgradeEntryKind, ModelUpgradeSource};

pub(crate) const MAX_MODEL_INDEX_ENVELOPE_BYTES: usize = 512 * 1024;
const MAX_MODEL_INDEX_PAYLOAD_BYTES: usize = 384 * 1024;
const MAX_MODEL_INDEX_ENTRIES: usize = 200;
const MAX_MODEL_INDEX_URL_BYTES: usize = 2_048;
const MAX_REDIRECTS: usize = 3;
const MAX_TRUSTED_MODEL_INDEX_KEYS: usize = 8;
const MAX_CLOCK_SKEW_SECONDS: u64 = 3_600;
const MAX_INDEX_LIFETIME_SECONDS: u64 = 366 * 24 * 60 * 60;
const SIGNATURE_DOMAIN_V1: &[u8] = b"bloom.model_index.v1\0";
const SIGNATURE_DOMAIN_V2: &[u8] = b"bloom.model_index.v2\0";
const GENERATION_ID_DOMAIN: &[u8] = b"bloom.model_index.generation.v1\0";
const GENERATION_ID_DOMAIN_V2: &[u8] = b"bloom.model_index.generation.v2\0";
const SOURCE_ID_DOMAIN: &[u8] = b"bloom.model_index.source.v1\0";

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SignedModelIndexEnvelope {
    schema_version: u8,
    object: String,
    algorithm: String,
    key_id: String,
    payload: String,
    signature: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ModelIndexPayload {
    schema_version: u8,
    object: String,
    name: String,
    generated_at: u64,
    expires_at: u64,
    #[serde(default)]
    models: Vec<ModelIndexPayloadEntry>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ModelIndexPayloadEntry {
    id: String,
    name: String,
    description: String,
    #[serde(default)]
    download_url: Option<String>,
    filename: String,
    size_bytes: u64,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    files: Vec<ModelIndexPayloadFile>,
    license: String,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    parameter_count: Option<u64>,
    #[serde(default)]
    quantization: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ModelIndexPayloadFile {
    download_url: String,
    filename: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelIndexSnapshot {
    pub schema_version: u8,
    pub object: &'static str,
    pub key_id: String,
    pub name: String,
    pub generated_at: u64,
    pub expires_at: u64,
    pub source_kind: &'static str,
    pub cache_status: &'static str,
    pub warning: Option<String>,
    pub data: Vec<ModelIndexEntry>,
    #[serde(skip_serializing)]
    generation_id: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelIndexEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,
    pub filename: String,
    pub format: String,
    pub size_bytes: u64,
    pub sha256: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<ModelIndexFile>,
    pub license: String,
    pub family: Option<String>,
    pub parameter_count: Option<u64>,
    pub quantization: Option<String>,
    pub tags: Vec<String>,
    pub downloadable: bool,
    pub blocking_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelIndexFile {
    pub download_url: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

impl ModelIndexEntry {
    pub(crate) fn is_package(&self) -> bool {
        !self.files.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelIndexInstallationState {
    Missing,
    Verified,
    Upgradable,
    Conflict,
}

/// Compare one signed index entry with the current catalog without trusting a
/// client-supplied installation claim. A verified match requires the exact
/// destination, signed digest, size, license, package shape, and persistent
/// signed-index alias recorded by Bloom's download pipeline.
pub(crate) fn model_index_installation_state(
    catalog: &ModelCatalog,
    entry: &ModelIndexEntry,
) -> ModelIndexInstallationState {
    let aliases = catalog
        .models
        .iter()
        .filter(|model| {
            model
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.model_index_id.as_deref())
                == Some(entry.id.as_str())
        })
        .collect::<Vec<_>>();
    if aliases.len() > 1 {
        return ModelIndexInstallationState::Conflict;
    }
    let destination = catalog
        .models
        .iter()
        .find(|model| model.id == entry.filename);
    if destination.is_some_and(|installed| exact_installation_matches(installed, entry)) {
        return ModelIndexInstallationState::Verified;
    }
    let Some(previous) = aliases.first().copied() else {
        return if destination.is_some() {
            ModelIndexInstallationState::Conflict
        } else {
            ModelIndexInstallationState::Missing
        };
    };
    if destination.is_some_and(|installed| installed.id != previous.id)
        || !upgrade_source_is_trusted(previous, &entry.id)
    {
        return ModelIndexInstallationState::Conflict;
    }
    ModelIndexInstallationState::Upgradable
}

pub(crate) fn model_index_upgrade_source<'a>(
    catalog: &'a ModelCatalog,
    entry: &ModelIndexEntry,
) -> Option<&'a ModelCatalogEntry> {
    (model_index_installation_state(catalog, entry) == ModelIndexInstallationState::Upgradable)
        .then(|| {
            catalog.models.iter().find(|model| {
                model
                    .provenance
                    .as_ref()
                    .and_then(|provenance| provenance.model_index_id.as_deref())
                    == Some(entry.id.as_str())
            })
        })
        .flatten()
}

pub(crate) fn model_index_upgrade_descriptor(
    catalog: &ModelCatalog,
    entry: &ModelIndexEntry,
) -> Option<ModelUpgradeSource> {
    let source = model_index_upgrade_source(catalog, entry)?;
    let provenance = source.provenance.as_ref()?;
    Some(ModelUpgradeSource {
        catalog_id: source.id.clone(),
        kind: match source.kind.as_str() {
            "file" => ModelUpgradeEntryKind::File,
            "directory" => ModelUpgradeEntryKind::Directory,
            _ => return None,
        },
        size_bytes: source.size_bytes,
        sha256: provenance.sha256.clone(),
        model_index_id: entry.id.clone(),
    })
}

fn exact_installation_matches(installed: &ModelCatalogEntry, entry: &ModelIndexEntry) -> bool {
    let expected_kind = if entry.is_package() {
        "directory"
    } else {
        "file"
    };
    let expected_file_count = entry.is_package().then_some(entry.files.len());
    let format_matches = installed.format == entry.format
        || (installed.format == "coreml" && entry.format == "mlmodel");
    installed.kind == expected_kind
        && format_matches
        && installed.size_complete
        && installed.size_bytes == entry.size_bytes
        && installed.provenance_error.is_none()
        && installed.provenance.as_ref().is_some_and(|provenance| {
            provenance.acquisition == ModelAcquisitionKind::Download
                && provenance.model_index_id.as_deref() == Some(entry.id.as_str())
                && provenance.sha256.eq_ignore_ascii_case(&entry.sha256)
                && provenance.file_count == expected_file_count
                && provenance
                    .license
                    .as_deref()
                    .is_some_and(|license| license.eq_ignore_ascii_case(&entry.license))
                && provenance.integrity_mismatch_at.is_none()
        })
}

fn upgrade_source_is_trusted(installed: &ModelCatalogEntry, model_index_id: &str) -> bool {
    installed.size_complete
        && installed.size_bytes > 0
        && installed.provenance_error.is_none()
        && installed.provenance.as_ref().is_some_and(|provenance| {
            provenance.acquisition == ModelAcquisitionKind::Download
                && provenance.model_index_id.as_deref() == Some(model_index_id)
                && provenance.integrity_mismatch_at.is_none()
        })
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelIndexError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    Internal(String),
}

#[derive(Clone)]
enum ModelIndexSource {
    File(PathBuf),
    Https(Url),
}

impl ModelIndexSource {
    fn kind(&self) -> &'static str {
        match self {
            Self::File(_) => "file",
            Self::Https(_) => "https",
        }
    }
}

#[derive(Clone, Default)]
struct ModelIndexUrlPolicy {
    allow_http: bool,
    allow_loopback: bool,
}

struct PreparedModelIndexConfig {
    source: ModelIndexSource,
    source_id: String,
    verifying_keys: BTreeMap<String, VerifyingKey>,
    trust_id: String,
    refresh_interval: Duration,
}

pub(crate) struct ModelIndexConfigurationStatus {
    pub source_kind: &'static str,
    pub trusted_key_count: usize,
    pub trust_id: String,
    pub single_key_id: Option<String>,
}

pub(crate) struct ModelIndexManagerConfig {
    pub file: Option<PathBuf>,
    pub url: Option<String>,
    pub public_key: Option<String>,
    pub public_keys: Vec<String>,
    pub refresh_seconds: u64,
    pub max_download_bytes: u64,
    pub state_directory: PathBuf,
}

#[derive(Clone)]
struct CachedModelIndex {
    refreshed_at: Instant,
    snapshot: ModelIndexSnapshot,
}

pub(crate) struct ModelIndexManager {
    source: ModelIndexSource,
    verifying_keys: BTreeMap<String, VerifyingKey>,
    trust_id: String,
    refresh_interval: Duration,
    max_download_bytes: u64,
    license_policy: Arc<ModelLicensePolicy>,
    watermark_store: ModelIndexWatermarkStore,
    client: Client,
    refresh_gate: Mutex<()>,
    cache: RwLock<Option<CachedModelIndex>>,
}

impl ModelIndexManager {
    #[cfg(test)]
    pub(crate) fn from_test_entry(
        entry: ModelIndexEntry,
        state_directory: PathBuf,
    ) -> Result<Arc<Self>> {
        let now = unix_time()?;
        let source_id = "01".repeat(32);
        let snapshot = ModelIndexSnapshot {
            schema_version: if entry.is_package() { 2 } else { 1 },
            object: "bloom.model_index",
            key_id: "02".repeat(32),
            name: "Test Models".to_string(),
            generated_at: now.saturating_sub(1),
            expires_at: now.saturating_add(3600),
            source_kind: "file",
            cache_status: "fresh",
            warning: None,
            data: vec![entry],
            generation_id: "03".repeat(32),
        };
        Ok(Arc::new(Self {
            source: ModelIndexSource::File(state_directory.join("unused-test-index.json")),
            verifying_keys: BTreeMap::new(),
            trust_id: "04".repeat(32),
            refresh_interval: Duration::from_secs(3600),
            max_download_bytes: u64::MAX,
            license_policy: Arc::new(ModelLicensePolicy::default()),
            watermark_store: ModelIndexWatermarkStore::new(state_directory, source_id)?,
            client: Client::new(),
            refresh_gate: Mutex::new(()),
            cache: RwLock::new(Some(CachedModelIndex {
                refreshed_at: Instant::now(),
                snapshot,
            })),
        }))
    }

    pub(crate) fn from_config(
        manager_config: ModelIndexManagerConfig,
        license_policy: Arc<ModelLicensePolicy>,
    ) -> Result<Option<Arc<Self>>> {
        let url_policy = ModelIndexUrlPolicy::default();
        let Some(config) = prepare_configuration(
            manager_config.file,
            manager_config.url,
            manager_config.public_key,
            manager_config.public_keys,
            manager_config.refresh_seconds,
            &url_policy,
        )?
        else {
            return Ok(None);
        };
        let watermark_store =
            ModelIndexWatermarkStore::new(manager_config.state_directory, config.source_id)?;
        let redirect_policy = url_policy;
        let client = Client::builder()
            .user_agent(concat!("bloom/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(20))
            .redirect(reqwest::redirect::Policy::custom(move |attempt| {
                if attempt.previous().len() >= MAX_REDIRECTS {
                    return attempt.error("model index exceeded the redirect limit");
                }
                if validate_remote_index_url(attempt.url(), &redirect_policy).is_err() {
                    return attempt.error("model index redirected to an invalid URL");
                }
                attempt.follow()
            }))
            .build()
            .context("failed to create the model index HTTP client")?;
        Ok(Some(Arc::new(Self {
            source: config.source,
            verifying_keys: config.verifying_keys,
            trust_id: config.trust_id,
            refresh_interval: config.refresh_interval,
            max_download_bytes: manager_config.max_download_bytes,
            license_policy,
            watermark_store,
            client,
            refresh_gate: Mutex::new(()),
            cache: RwLock::new(None),
        })))
    }

    pub(crate) fn trust_id(&self) -> &str {
        &self.trust_id
    }

    pub(crate) fn trusted_key_count(&self) -> usize {
        self.verifying_keys.len()
    }

    pub(crate) fn refresh_seconds(&self) -> u64 {
        self.refresh_interval.as_secs()
    }

    pub(crate) fn persistent_rollback_protection(&self) -> bool {
        true
    }

    pub(crate) fn single_key_id(&self) -> Option<&str> {
        if self.verifying_keys.len() == 1 {
            self.verifying_keys.keys().next().map(String::as_str)
        } else {
            None
        }
    }

    pub(crate) async fn snapshot(
        &self,
        force_refresh: bool,
    ) -> std::result::Result<ModelIndexSnapshot, ModelIndexError> {
        let now = unix_time().map_err(|error| ModelIndexError::Internal(error.to_string()))?;
        if !force_refresh {
            if let Some(cached) = self.cache.read().await.as_ref() {
                if cached.refreshed_at.elapsed() < self.refresh_interval
                    && cached.snapshot.expires_at > now
                {
                    let mut snapshot = cached.snapshot.clone();
                    snapshot.cache_status = "cached";
                    return Ok(snapshot);
                }
            }
        }

        let _refresh_guard = self.refresh_gate.lock().await;
        let now = unix_time().map_err(|error| ModelIndexError::Internal(error.to_string()))?;
        if !force_refresh {
            if let Some(cached) = self.cache.read().await.as_ref() {
                if cached.refreshed_at.elapsed() < self.refresh_interval
                    && cached.snapshot.expires_at > now
                {
                    let mut snapshot = cached.snapshot.clone();
                    snapshot.cache_status = "cached";
                    return Ok(snapshot);
                }
            }
        }

        match self.load_snapshot().await {
            Ok(snapshot) => {
                let cached_rejection = self.cache.read().await.as_ref().and_then(|cached| {
                    let rollback = snapshot.generated_at < cached.snapshot.generated_at;
                    let conflict = snapshot.generated_at == cached.snapshot.generated_at
                        && snapshot.generation_id != cached.snapshot.generation_id;
                    (rollback || conflict).then(|| ModelIndexWatermark {
                        generated_at: cached.snapshot.generated_at,
                        generation_id: cached.snapshot.generation_id.clone(),
                    })
                });
                if let Some(current) = cached_rejection {
                    return self.reject_persisted_generation(current).await;
                }
                let admission = self
                    .watermark_store
                    .admit(ModelIndexWatermark {
                        generated_at: snapshot.generated_at,
                        generation_id: snapshot.generation_id.clone(),
                    })
                    .await
                    .map_err(|error| {
                        tracing::error!(error = %error, "Persistent model index rollback state failed");
                        ModelIndexError::Internal(
                            "The persistent model index rollback state is unavailable."
                                .to_string(),
                        )
                    })?;
                if let ModelIndexWatermarkAdmission::Rollback(current)
                | ModelIndexWatermarkAdmission::Conflict(current) = admission
                {
                    return self.reject_persisted_generation(current).await;
                }
                *self.cache.write().await = Some(CachedModelIndex {
                    refreshed_at: Instant::now(),
                    snapshot: snapshot.clone(),
                });
                Ok(snapshot)
            }
            Err(error) => {
                let fallback_now =
                    unix_time().map_err(|clock| ModelIndexError::Internal(clock.to_string()))?;
                if let Some(cached) = self.cache.read().await.as_ref() {
                    if cached.snapshot.expires_at > fallback_now {
                        let mut snapshot = cached.snapshot.clone();
                        snapshot.cache_status = "stale";
                        snapshot.warning = Some(
                            "The configured index could not be refreshed. Bloom is showing the last verified, unexpired snapshot."
                                .to_string(),
                        );
                        return Ok(snapshot);
                    }
                }
                Err(error)
            }
        }
    }

    async fn reject_persisted_generation(
        &self,
        current: ModelIndexWatermark,
    ) -> std::result::Result<ModelIndexSnapshot, ModelIndexError> {
        let now = unix_time().map_err(|error| ModelIndexError::Internal(error.to_string()))?;
        if let Some(cached) = self.cache.read().await.as_ref() {
            if cached.snapshot.generated_at == current.generated_at
                && cached.snapshot.generation_id == current.generation_id
                && cached.snapshot.expires_at > now
            {
                let mut retained = cached.snapshot.clone();
                retained.cache_status = "stale";
                retained.warning = Some(
                    "A signed index rollback or conflicting generation was rejected. Bloom is showing the newer verified, unexpired snapshot."
                        .to_string(),
                );
                return Ok(retained);
            }
        }
        Err(ModelIndexError::Invalid(
            "The signed model index is older than, or conflicts with, the persisted verified generation."
                .to_string(),
        ))
    }

    async fn load_snapshot(&self) -> std::result::Result<ModelIndexSnapshot, ModelIndexError> {
        let envelope = self.read_envelope().await?;
        let now = unix_time().map_err(|error| ModelIndexError::Internal(error.to_string()))?;
        decode_signed_index(
            &envelope,
            &self.verifying_keys,
            self.source.kind(),
            self.max_download_bytes,
            &self.license_policy,
            now,
        )
        .map_err(|error| ModelIndexError::Invalid(error.to_string()))
    }

    async fn read_envelope(&self) -> std::result::Result<Vec<u8>, ModelIndexError> {
        match &self.source {
            ModelIndexSource::File(path) => read_bounded_file(path).await,
            ModelIndexSource::Https(url) => read_bounded_response(&self.client, url).await,
        }
    }
}

pub(crate) fn validate_configuration(
    file: Option<PathBuf>,
    url: Option<String>,
    public_key: Option<String>,
    public_keys: Vec<String>,
    refresh_seconds: u64,
) -> Result<Option<ModelIndexConfigurationStatus>> {
    Ok(prepare_configuration(
        file,
        url,
        public_key,
        public_keys,
        refresh_seconds,
        &ModelIndexUrlPolicy::default(),
    )?
    .map(|config| {
        let single_key_id = if config.verifying_keys.len() == 1 {
            config.verifying_keys.keys().next().cloned()
        } else {
            None
        };
        ModelIndexConfigurationStatus {
            source_kind: config.source.kind(),
            trusted_key_count: config.verifying_keys.len(),
            trust_id: config.trust_id,
            single_key_id,
        }
    }))
}

fn prepare_configuration(
    file: Option<PathBuf>,
    url: Option<String>,
    public_key: Option<String>,
    public_keys: Vec<String>,
    refresh_seconds: u64,
    url_policy: &ModelIndexUrlPolicy,
) -> Result<Option<PreparedModelIndexConfig>> {
    if file.is_some() && url.is_some() {
        return Err(anyhow!(
            "configure either a model index file or URL, not both"
        ));
    }
    let source = match (file, url) {
        (Some(path), None) => {
            if path.as_os_str().is_empty() {
                return Err(anyhow!("model index file path must not be empty"));
            }
            Some(ModelIndexSource::File(path))
        }
        (None, Some(value)) => {
            if value.trim().is_empty() || value.len() > MAX_MODEL_INDEX_URL_BYTES {
                return Err(anyhow!(
                    "model index URL must be between 1 and {MAX_MODEL_INDEX_URL_BYTES} bytes"
                ));
            }
            let url = Url::parse(value.trim()).context("model index URL is invalid")?;
            validate_remote_index_url(&url, url_policy)?;
            Some(ModelIndexSource::Https(url))
        }
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!(),
    };
    let mut configured_keys = public_keys;
    if let Some(public_key) = public_key {
        configured_keys.insert(0, public_key);
    }
    match (source, configured_keys.is_empty()) {
        (None, true) => Ok(None),
        (None, false) => Err(anyhow!(
            "model index public keys require a configured index file or URL"
        )),
        (Some(_), true) => Err(anyhow!(
            "at least one model index public key is required when an index is configured"
        )),
        (Some(source), false) => {
            if refresh_seconds == 0 || refresh_seconds > 24 * 60 * 60 {
                return Err(anyhow!(
                    "model index refresh interval must be between 1 and 86400 seconds"
                ));
            }
            if configured_keys.len() > MAX_TRUSTED_MODEL_INDEX_KEYS {
                return Err(anyhow!(
                    "no more than {MAX_TRUSTED_MODEL_INDEX_KEYS} model index public keys may be trusted"
                ));
            }
            let mut verifying_keys = BTreeMap::new();
            for public_key in configured_keys {
                let verifying_key = parse_public_key(&public_key)?;
                if verifying_key.is_weak() {
                    return Err(anyhow!("model index public keys must not be weak"));
                }
                let key_id = key_id(&verifying_key);
                if verifying_keys.insert(key_id, verifying_key).is_some() {
                    return Err(anyhow!("model index public keys must be unique"));
                }
            }
            let trust_id = trust_id(&verifying_keys);
            let source_id = source_id(&source)?;
            Ok(Some(PreparedModelIndexConfig {
                source,
                source_id,
                verifying_keys,
                trust_id,
                refresh_interval: Duration::from_secs(refresh_seconds),
            }))
        }
    }
}

fn validate_remote_index_url(url: &Url, policy: &ModelIndexUrlPolicy) -> Result<()> {
    let valid_scheme = url.scheme() == "https" || (policy.allow_http && url.scheme() == "http");
    if !valid_scheme {
        return Err(anyhow!("model index URLs require HTTPS"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(anyhow!("model index URLs must not contain credentials"));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(anyhow!(
            "model index URLs must not contain a query string or fragment"
        ));
    }
    if url.scheme() == "https" && url.port().is_some_and(|port| port != 443) {
        return Err(anyhow!("model index URLs must use the default HTTPS port"));
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("model index URL must include a host"))?
        .to_ascii_lowercase();
    if !policy.allow_loopback && matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return Err(anyhow!("model index URL must not target a loopback host"));
    }
    Ok(())
}

fn parse_public_key(value: &str) -> Result<VerifyingKey> {
    let value = value.trim();
    let bytes = if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        decode_hex::<32>(value).context("model index public key is invalid")?
    } else {
        let bytes = URL_SAFE_NO_PAD
            .decode(value)
            .context("model index public key must be 64 hex characters or unpadded base64url")?;
        bytes
            .try_into()
            .map_err(|_| anyhow!("model index public key must decode to exactly 32 bytes"))?
    };
    VerifyingKey::from_bytes(&bytes).context("model index public key is invalid")
}

fn key_id(key: &VerifyingKey) -> String {
    format!("{:x}", Sha256::digest(key.as_bytes()))
}

fn trust_id(keys: &BTreeMap<String, VerifyingKey>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"bloom.model_index.trust.v1\0");
    for key_id in keys.keys() {
        digest.update(key_id.as_bytes());
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

fn source_id(source: &ModelIndexSource) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(SOURCE_ID_DOMAIN);
    match source {
        ModelIndexSource::File(path) => {
            digest.update(b"file\0");
            let absolute = if path.is_absolute() {
                path.clone()
            } else {
                std::env::current_dir()
                    .context("failed to resolve the model index file path")?
                    .join(path)
            };
            digest.update(absolute.as_os_str().as_encoded_bytes());
        }
        ModelIndexSource::Https(url) => {
            digest.update(b"https\0");
            digest.update(url.as_str().as_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn generation_id(schema_version: u8, key_id: &str, payload: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(if schema_version == 1 {
        GENERATION_ID_DOMAIN
    } else {
        GENERATION_ID_DOMAIN_V2
    });
    digest.update(key_id.as_bytes());
    digest.update([0]);
    digest.update(payload);
    format!("{:x}", digest.finalize())
}

fn decode_signed_index(
    envelope_bytes: &[u8],
    verifying_keys: &BTreeMap<String, VerifyingKey>,
    source_kind: &'static str,
    max_download_bytes: u64,
    license_policy: &ModelLicensePolicy,
    now: u64,
) -> Result<ModelIndexSnapshot> {
    if envelope_bytes.is_empty() || envelope_bytes.len() > MAX_MODEL_INDEX_ENVELOPE_BYTES {
        return Err(anyhow!(
            "model index envelope must be between 1 and {MAX_MODEL_INDEX_ENVELOPE_BYTES} bytes"
        ));
    }
    let envelope = serde_json::from_slice::<SignedModelIndexEnvelope>(envelope_bytes)
        .context("model index envelope is invalid")?;
    if !matches!(envelope.schema_version, 1 | 2) || envelope.object != "bloom.signed_model_index" {
        return Err(anyhow!("unsupported model index envelope identity"));
    }
    if envelope.algorithm != "ed25519" {
        return Err(anyhow!("unsupported model index signature algorithm"));
    }
    if envelope.key_id.len() != 64
        || !envelope
            .key_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(anyhow!("model index envelope contains an invalid key ID"));
    }
    let verifying_key = verifying_keys
        .get(&envelope.key_id)
        .ok_or_else(|| anyhow!("model index was not signed by a configured trusted key"))?;
    let payload = URL_SAFE_NO_PAD
        .decode(envelope.payload)
        .context("model index payload is not valid unpadded base64url")?;
    if payload.is_empty() || payload.len() > MAX_MODEL_INDEX_PAYLOAD_BYTES {
        return Err(anyhow!(
            "model index payload must be between 1 and {MAX_MODEL_INDEX_PAYLOAD_BYTES} bytes"
        ));
    }
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(envelope.signature)
        .context("model index signature is not valid unpadded base64url")?;
    let signature = Signature::from_slice(&signature_bytes)
        .context("model index signature must contain exactly 64 bytes")?;
    let signature_domain = if envelope.schema_version == 1 {
        SIGNATURE_DOMAIN_V1
    } else {
        SIGNATURE_DOMAIN_V2
    };
    let mut signed_message = Vec::with_capacity(signature_domain.len() + payload.len());
    signed_message.extend_from_slice(signature_domain);
    signed_message.extend_from_slice(&payload);
    verifying_key
        .verify_strict(&signed_message, &signature)
        .context("model index signature verification failed")?;

    let generation_id = generation_id(envelope.schema_version, &envelope.key_id, &payload);
    let payload = serde_json::from_slice::<ModelIndexPayload>(&payload)
        .context("verified model index payload is invalid")?;
    if payload.schema_version != envelope.schema_version {
        return Err(anyhow!(
            "model index envelope and payload schema versions do not match"
        ));
    }
    validate_payload(
        payload,
        &envelope.key_id,
        source_kind,
        max_download_bytes,
        license_policy,
        now,
        generation_id,
    )
}

fn validate_payload(
    payload: ModelIndexPayload,
    expected_key_id: &str,
    source_kind: &'static str,
    max_download_bytes: u64,
    license_policy: &ModelLicensePolicy,
    now: u64,
    generation_id: String,
) -> Result<ModelIndexSnapshot> {
    if !matches!(payload.schema_version, 1 | 2) || payload.object != "bloom.model_index" {
        return Err(anyhow!("unsupported model index payload identity"));
    }
    validate_text(&payload.name, "model index name", 1, 80)?;
    if payload.generated_at == 0
        || payload.generated_at > now.saturating_add(MAX_CLOCK_SKEW_SECONDS)
    {
        return Err(anyhow!("model index generation time is invalid"));
    }
    if payload.expires_at <= payload.generated_at || payload.expires_at <= now {
        return Err(anyhow!("model index is expired or has an invalid expiry"));
    }
    if payload.expires_at.saturating_sub(payload.generated_at) > MAX_INDEX_LIFETIME_SECONDS {
        return Err(anyhow!("model index validity period exceeds 366 days"));
    }
    if payload.models.len() > MAX_MODEL_INDEX_ENTRIES {
        return Err(anyhow!(
            "model index contains more than {MAX_MODEL_INDEX_ENTRIES} entries"
        ));
    }

    let mut ids = HashSet::with_capacity(payload.models.len());
    let mut filenames = HashSet::with_capacity(payload.models.len());
    let mut data = Vec::with_capacity(payload.models.len());
    let schema_version = payload.schema_version;
    for model in payload.models {
        validate_index_id(&model.id)?;
        if !ids.insert(model.id.to_ascii_lowercase()) {
            return Err(anyhow!("model index contains a duplicate model ID"));
        }
        validate_text(&model.name, "model index entry name", 1, 80)?;
        validate_text(&model.description, "model index entry description", 1, 400)?;
        let is_package = !model.files.is_empty();
        if is_package {
            if schema_version != 2 {
                return Err(anyhow!(
                    "multi-file model packages require model index schema version 2"
                ));
            }
            validate_package_id(&model.filename)?;
            if model.download_url.is_some() || model.sha256.is_some() {
                return Err(anyhow!(
                    "multi-file model packages must declare verification metadata per file"
                ));
            }
        } else {
            validate_model_filename(&model.filename)?;
            if model.download_url.is_none() || model.sha256.is_none() {
                return Err(anyhow!(
                    "single-file model entries require a download URL and SHA-256"
                ));
            }
        }
        if model.filename.len() > 255 || model.filename.contains('%') {
            return Err(anyhow!("model index entry filename is invalid"));
        }
        if !filenames.insert(model.filename.to_ascii_lowercase()) {
            return Err(anyhow!(
                "model index contains duplicate destination filenames"
            ));
        }
        if model.size_bytes == 0 {
            return Err(anyhow!("model index entry size must be greater than zero"));
        }
        let license = normalize_license(Some(model.license))?
            .ok_or_else(|| anyhow!("model index entries must declare a license"))?;
        let (download_url, sha256, files, format) = if is_package {
            let mut source_identity = None;
            let package_files = model
                .files
                .into_iter()
                .map(|file| {
                    let download_url =
                        immutable_hugging_face_source(&file.download_url, &file.filename)?;
                    let identity = hugging_face_repository_revision(&download_url)?;
                    if source_identity
                        .as_ref()
                        .is_some_and(|expected| expected != &identity)
                    {
                        return Err(anyhow!(
                            "all model package files must use the same repository commit"
                        ));
                    }
                    source_identity.get_or_insert(identity);
                    Ok((
                        ModelPackageFile {
                            filename: file.filename.clone(),
                            size_bytes: file.size_bytes,
                            sha256: file.sha256,
                        },
                        download_url,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let urls = package_files
                .iter()
                .map(|(file, url)| (file.filename.clone(), url.clone()))
                .collect::<BTreeMap<_, _>>();
            let normalized = normalize_package_files(
                package_files.into_iter().map(|(file, _)| file).collect(),
                model.size_bytes,
            )?;
            let digest = package_digest(&normalized)?;
            let files = normalized
                .into_iter()
                .map(|file| ModelIndexFile {
                    download_url: urls[&file.filename].clone(),
                    filename: file.filename,
                    size_bytes: file.size_bytes,
                    sha256: file.sha256,
                })
                .collect();
            (None, digest, files, "transformers".to_string())
        } else {
            let sha256 = model
                .sha256
                .as_deref()
                .map(str::trim)
                .unwrap_or_default()
                .to_ascii_lowercase();
            if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(anyhow!(
                    "model index entry SHA-256 must contain 64 hexadecimal characters"
                ));
            }
            let download_url = immutable_hugging_face_source(
                model.download_url.as_deref().unwrap_or_default(),
                &model.filename,
            )?;
            let format = model
                .filename
                .rsplit_once('.')
                .map(|(_, extension)| extension.to_ascii_lowercase())
                .ok_or_else(|| anyhow!("model index entry filename has no format"))?;
            (Some(download_url), sha256, Vec::new(), format)
        };
        let family = validate_optional_text(model.family, "model family", 64)?;
        let quantization = validate_optional_text(model.quantization, "model quantization", 32)?;
        if model.parameter_count == Some(0) {
            return Err(anyhow!(
                "model index parameter count must be greater than zero"
            ));
        }
        if model.tags.len() > 12 {
            return Err(anyhow!(
                "model index entries must not contain more than 12 tags"
            ));
        }
        let mut tags = Vec::with_capacity(model.tags.len());
        let mut seen_tags = HashSet::with_capacity(model.tags.len());
        for tag in model.tags {
            validate_text(&tag, "model index tag", 1, 32)?;
            if !tag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            {
                return Err(anyhow!("model index tags contain unsupported characters"));
            }
            let normalized = tag.to_ascii_lowercase();
            if seen_tags.insert(normalized.clone()) {
                tags.push(normalized);
            }
        }
        let mut blocking_reasons = Vec::new();
        if model.size_bytes > max_download_bytes {
            blocking_reasons.push("size_limit".to_string());
        }
        if license_policy.enforce(Some(license.clone())).is_err() {
            blocking_reasons.push("license_policy".to_string());
        }
        data.push(ModelIndexEntry {
            id: model.id,
            name: model.name,
            description: model.description,
            download_url,
            filename: model.filename,
            format,
            size_bytes: model.size_bytes,
            sha256,
            files,
            license,
            family,
            parameter_count: model.parameter_count,
            quantization,
            tags,
            downloadable: blocking_reasons.is_empty(),
            blocking_reasons,
        });
    }
    data.sort_by(|left, right| left.name.cmp(&right.name).then(left.id.cmp(&right.id)));

    Ok(ModelIndexSnapshot {
        schema_version,
        object: "bloom.model_index",
        key_id: expected_key_id.to_string(),
        name: payload.name,
        generated_at: payload.generated_at,
        expires_at: payload.expires_at,
        source_kind,
        cache_status: "fresh",
        warning: None,
        data,
        generation_id,
    })
}

fn immutable_hugging_face_source(value: &str, expected_filename: &str) -> Result<String> {
    if value.is_empty() || value.len() > MAX_MODEL_INDEX_URL_BYTES {
        return Err(anyhow!("model index download URL is invalid"));
    }
    let url = Url::parse(value).context("model index download URL is invalid")?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some_and(|port| port != 443)
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(anyhow!(
            "model index download URLs require public default-port HTTPS without credentials, query strings, or fragments"
        ));
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("model index download URL must include a host"))?
        .to_ascii_lowercase();
    if !matches!(host.as_str(), "huggingface.co" | "www.huggingface.co") {
        return Err(anyhow!(
            "model index download URLs must use a Hugging Face repository host"
        ));
    }
    let segments = url
        .path_segments()
        .ok_or_else(|| anyhow!("model index download URL has no file path"))?
        .collect::<Vec<_>>();
    if segments.len() < 5
        || segments[0].is_empty()
        || segments[1].is_empty()
        || segments[2] != "resolve"
        || !matches!(segments[3].len(), 40 | 64)
        || !segments[3].bytes().all(|byte| byte.is_ascii_hexdigit())
        || segments[4..]
            .iter()
            .any(|segment| segment.is_empty() || segment.contains('%'))
        || segments[4..].join("/") != expected_filename
    {
        return Err(anyhow!(
            "model index download URL must identify a file at an immutable Hugging Face commit"
        ));
    }
    Ok(url.to_string())
}

fn hugging_face_repository_revision(value: &str) -> Result<String> {
    let url = Url::parse(value).context("model package download URL is invalid")?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("model package download URL must include a host"))?
        .to_ascii_lowercase();
    let segments = url
        .path_segments()
        .ok_or_else(|| anyhow!("model package download URL has no file path"))?
        .collect::<Vec<_>>();
    if segments.len() < 5 || segments[2] != "resolve" {
        return Err(anyhow!("model package download URL is not immutable"));
    }
    Ok(format!(
        "{host}/{}/{}/{}",
        segments[0], segments[1], segments[3]
    ))
}

pub(crate) fn validate_index_id(value: &str) -> Result<()> {
    validate_text(value, "model index entry ID", 1, 64)?;
    if !value.bytes().enumerate().all(|(index, byte)| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || (index > 0 && matches!(byte, b'-' | b'_' | b'.'))
    }) {
        return Err(anyhow!(
            "model index entry IDs must use lowercase ASCII letters, digits, dots, dashes, or underscores"
        ));
    }
    Ok(())
}

fn validate_text(value: &str, field: &str, min: usize, max: usize) -> Result<()> {
    let length = value.chars().count();
    if length < min || length > max || value.trim() != value || value.chars().any(char::is_control)
    {
        return Err(anyhow!(
            "{field} must contain {min} to {max} trimmed, non-control characters"
        ));
    }
    Ok(())
}

fn validate_optional_text(
    value: Option<String>,
    field: &str,
    max: usize,
) -> Result<Option<String>> {
    match value {
        Some(value) => {
            validate_text(&value, field, 1, max)?;
            Ok(Some(value))
        }
        None => Ok(None),
    }
}

async fn read_bounded_file(path: &Path) -> std::result::Result<Vec<u8>, ModelIndexError> {
    let metadata = fs::symlink_metadata(path).await.map_err(|_| {
        ModelIndexError::Unavailable("The configured model index file is unavailable.".to_string())
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ModelIndexError::Invalid(
            "The configured model index path must be a regular, non-symlink file.".to_string(),
        ));
    }
    if metadata.len() == 0 || metadata.len() > MAX_MODEL_INDEX_ENVELOPE_BYTES as u64 {
        return Err(ModelIndexError::Invalid(format!(
            "The configured model index file must be between 1 and {MAX_MODEL_INDEX_ENVELOPE_BYTES} bytes."
        )));
    }
    let file = fs::File::open(path).await.map_err(|_| {
        ModelIndexError::Unavailable("The configured model index file is unavailable.".to_string())
    })?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_MODEL_INDEX_ENVELOPE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|_| {
            ModelIndexError::Unavailable(
                "The configured model index file could not be read.".to_string(),
            )
        })?;
    if bytes.is_empty() || bytes.len() > MAX_MODEL_INDEX_ENVELOPE_BYTES {
        return Err(ModelIndexError::Invalid(
            "The configured model index file changed size while it was read.".to_string(),
        ));
    }
    Ok(bytes)
}

async fn read_bounded_response(
    client: &Client,
    url: &Url,
) -> std::result::Result<Vec<u8>, ModelIndexError> {
    let response = client.get(url.clone()).send().await.map_err(|_| {
        ModelIndexError::Unavailable("The configured model index request failed.".to_string())
    })?;
    if response.status() != StatusCode::OK {
        return Err(ModelIndexError::Unavailable(format!(
            "The configured model index returned HTTP {}.",
            response.status().as_u16()
        )));
    }
    if response
        .content_length()
        .is_some_and(|length| length == 0 || length > MAX_MODEL_INDEX_ENVELOPE_BYTES as u64)
    {
        return Err(ModelIndexError::Invalid(
            "The configured model index response declared an invalid size.".to_string(),
        ));
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| {
            ModelIndexError::Unavailable(
                "The configured model index response was interrupted.".to_string(),
            )
        })?;
        if bytes.len().saturating_add(chunk.len()) > MAX_MODEL_INDEX_ENVELOPE_BYTES {
            return Err(ModelIndexError::Invalid(format!(
                "The configured model index response exceeds {MAX_MODEL_INDEX_ENVELOPE_BYTES} bytes."
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.is_empty() {
        return Err(ModelIndexError::Invalid(
            "The configured model index response was empty.".to_string(),
        ));
    }
    Ok(bytes)
}

fn unix_time() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

fn decode_hex<const N: usize>(value: &str) -> Result<[u8; N]> {
    if value.len() != N * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(anyhow!("hex value has an invalid length or character"));
    }
    let mut output = [0_u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer as _, SigningKey};

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7_u8; 32])
    }

    fn payload(now: u64, revision: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "object": "bloom.model_index",
            "name": "Bloom Test Models",
            "generated_at": now.saturating_sub(60),
            "expires_at": now.saturating_add(3600),
            "models": [{
                "id": "tiny-q4",
                "name": "Tiny Q4",
                "description": "A small deterministic test entry.",
                "download_url": format!("https://huggingface.co/acme/tiny/resolve/{revision}/tiny-q4.gguf"),
                "filename": "tiny-q4.gguf",
                "size_bytes": 4096,
                "sha256": "ab".repeat(32),
                "license": "Apache-2.0",
                "family": "Llama",
                "parameter_count": 1_100_000_000_u64,
                "quantization": "Q4_K_M",
                "tags": ["chat", "small"]
            }]
        }))
        .unwrap()
    }

    fn package_payload(now: u64, revision: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 2,
            "object": "bloom.model_index",
            "name": "Bloom Test Packages",
            "generated_at": now.saturating_sub(60),
            "expires_at": now.saturating_add(3600),
            "models": [{
                "id": "tiny-package",
                "name": "Tiny Package",
                "description": "A deterministic multi-file test package.",
                "filename": "tiny-package",
                "size_bytes": 10,
                "license": "Apache-2.0",
                "family": "Qwen2",
                "tags": ["chat", "package"],
                "files": [
                    {
                        "download_url": format!("https://huggingface.co/acme/tiny/resolve/{revision}/model.safetensors"),
                        "filename": "model.safetensors",
                        "size_bytes": 7,
                        "sha256": "ab".repeat(32)
                    },
                    {
                        "download_url": format!("https://huggingface.co/acme/tiny/resolve/{revision}/config.json"),
                        "filename": "config.json",
                        "size_bytes": 3,
                        "sha256": "cd".repeat(32)
                    }
                ]
            }]
        }))
        .unwrap()
    }

    fn payload_generation(now: u64, generated_at: u64, revision: &str, name: &str) -> Vec<u8> {
        let mut value =
            serde_json::from_slice::<serde_json::Value>(&payload(now, revision)).unwrap();
        value["generated_at"] = serde_json::Value::from(generated_at);
        value["expires_at"] = serde_json::Value::from(now.saturating_add(3600));
        value["name"] = serde_json::Value::from(name);
        serde_json::to_vec(&value).unwrap()
    }

    fn envelope(payload: &[u8], key: &SigningKey) -> Vec<u8> {
        let schema_version = serde_json::from_slice::<serde_json::Value>(payload).unwrap()
            ["schema_version"]
            .as_u64()
            .unwrap() as u8;
        let mut message = if schema_version == 1 {
            SIGNATURE_DOMAIN_V1.to_vec()
        } else {
            SIGNATURE_DOMAIN_V2.to_vec()
        };
        message.extend_from_slice(payload);
        let signature = key.sign(&message);
        serde_json::to_vec(&serde_json::json!({
            "schema_version": schema_version,
            "object": "bloom.signed_model_index",
            "algorithm": "ed25519",
            "key_id": key_id(&key.verifying_key()),
            "payload": URL_SAFE_NO_PAD.encode(payload),
            "signature": URL_SAFE_NO_PAD.encode(signature.to_bytes())
        }))
        .unwrap()
    }

    fn key_hex(key: &SigningKey) -> String {
        key.verifying_key()
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn keyring(keys: &[VerifyingKey]) -> BTreeMap<String, VerifyingKey> {
        keys.iter()
            .map(|key| (key_id(key), *key))
            .collect::<BTreeMap<_, _>>()
    }

    #[test]
    fn configuration_requires_one_source_and_a_bounded_unique_non_weak_keyring() {
        let key = signing_key();
        assert!(validate_configuration(None, None, None, vec![], 300)
            .unwrap()
            .is_none());
        assert!(validate_configuration(
            Some(PathBuf::from("index.json")),
            Some("https://example.com/index.json".to_string()),
            Some(key_hex(&key)),
            vec![],
            300,
        )
        .is_err());
        assert!(validate_configuration(
            None,
            Some("http://example.com/index.json".to_string()),
            Some(key_hex(&key)),
            vec![],
            300,
        )
        .is_err());
        assert!(
            validate_configuration(Some(PathBuf::from("index.json")), None, None, vec![], 300,)
                .is_err()
        );
        assert!(validate_configuration(
            Some(PathBuf::from("index.json")),
            None,
            Some("00".repeat(32)),
            vec![],
            300,
        )
        .is_err());
        assert!(validate_configuration(
            Some(PathBuf::from("index.json")),
            None,
            Some(key_hex(&key)),
            vec![key_hex(&key)],
            300,
        )
        .is_err());
        assert!(validate_configuration(
            Some(PathBuf::from("index.json")),
            None,
            None,
            vec![key_hex(&key); MAX_TRUSTED_MODEL_INDEX_KEYS + 1],
            300,
        )
        .is_err());
    }

    #[test]
    fn trust_set_is_order_independent_and_accepts_each_configured_signer() {
        let now = unix_time().unwrap();
        let first = signing_key();
        let second = SigningKey::from_bytes(&[8_u8; 32]);
        let first_hex = key_hex(&first);
        let second_hex = key_hex(&second);
        let source = Some(PathBuf::from("index.json"));
        let left = validate_configuration(
            source.clone(),
            None,
            Some(first_hex.clone()),
            vec![second_hex.clone()],
            300,
        )
        .unwrap()
        .unwrap();
        let right = validate_configuration(source, None, None, vec![second_hex, first_hex], 300)
            .unwrap()
            .unwrap();

        assert_eq!(left.trusted_key_count, 2);
        assert!(left.single_key_id.is_none());
        assert_eq!(left.trust_id, right.trust_id);
        let keys = keyring(&[first.verifying_key(), second.verifying_key()]);
        assert!(decode_signed_index(
            &envelope(&payload(now, &"56".repeat(20)), &first),
            &keys,
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .is_ok());
        assert!(decode_signed_index(
            &envelope(&payload(now, &"56".repeat(20)), &second),
            &keys,
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .is_ok());
        let unknown = SigningKey::from_bytes(&[9_u8; 32]);
        assert!(decode_signed_index(
            &envelope(&payload(now, &"56".repeat(20)), &unknown),
            &keys,
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .is_err());
    }

    #[test]
    fn signed_index_is_strictly_verified_and_normalized() {
        let now = unix_time().unwrap();
        let key = signing_key();
        let revision = "cd".repeat(20);
        let envelope = envelope(&payload(now, &revision), &key);

        let snapshot = decode_signed_index(
            &envelope,
            &keyring(&[key.verifying_key()]),
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .unwrap();

        assert_eq!(snapshot.object, "bloom.model_index");
        assert_eq!(snapshot.data.len(), 1);
        assert_eq!(snapshot.data[0].format, "gguf");
        assert!(snapshot.data[0].downloadable);
        assert!(snapshot.data[0]
            .download_url
            .as_deref()
            .unwrap()
            .contains(&format!("/resolve/{revision}/")));
    }

    #[test]
    fn signed_v2_index_normalizes_one_atomic_model_package() {
        let now = unix_time().unwrap();
        let key = signing_key();
        let revision = "12".repeat(20);
        let snapshot = decode_signed_index(
            &envelope(&package_payload(now, &revision), &key),
            &keyring(&[key.verifying_key()]),
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .unwrap();

        assert_eq!(snapshot.schema_version, 2);
        assert_eq!(snapshot.data.len(), 1);
        let entry = &snapshot.data[0];
        assert!(entry.is_package());
        assert_eq!(entry.filename, "tiny-package");
        assert_eq!(entry.format, "transformers");
        assert_eq!(entry.download_url, None);
        assert_eq!(entry.files.len(), 2);
        assert_eq!(entry.files[0].filename, "config.json");
        assert_eq!(entry.sha256.len(), 64);
        assert!(entry.downloadable);
    }

    #[test]
    fn signed_v2_index_rejects_non_atomic_package_manifests() {
        let now = unix_time().unwrap();
        let key = signing_key();
        let revision = "34".repeat(20);
        let base =
            serde_json::from_slice::<serde_json::Value>(&package_payload(now, &revision)).unwrap();
        for mutation in ["size", "path", "commit"] {
            let mut payload = base.clone();
            match mutation {
                "size" => payload["models"][0]["size_bytes"] = serde_json::Value::from(11),
                "path" => {
                    payload["models"][0]["files"][0]["filename"] =
                        serde_json::Value::from("../model.safetensors")
                }
                "commit" => {
                    payload["models"][0]["files"][0]["download_url"] =
                        serde_json::Value::from(format!(
                            "https://huggingface.co/acme/tiny/resolve/{}/config.json",
                            "56".repeat(20)
                        ))
                }
                _ => unreachable!(),
            }
            let payload = serde_json::to_vec(&payload).unwrap();
            assert!(decode_signed_index(
                &envelope(&payload, &key),
                &keyring(&[key.verifying_key()]),
                "file",
                8192,
                &ModelLicensePolicy::default(),
                now,
            )
            .is_err());
        }
    }

    #[test]
    fn bundled_signed_example_matches_its_documented_public_key() {
        let verifying_key =
            parse_public_key("ea4a6c63e29c520abef5507b132ec5f9954776aebebe7b92421eea691446d22c")
                .unwrap();
        let snapshot = decode_signed_index(
            include_bytes!("../../../examples/fixtures/model-index.signed.json"),
            &keyring(&[verifying_key]),
            "file",
            2 * 1024 * 1024,
            &ModelLicensePolicy::default(),
            1_785_513_601,
        )
        .unwrap();

        assert_eq!(snapshot.name, "Bloom Example Models");
        assert_eq!(snapshot.data[0].id, "example-tiny-q4");
    }

    #[test]
    fn tampering_expiry_and_mutable_sources_fail_closed() {
        let now = unix_time().unwrap();
        let key = signing_key();
        let revision = "ef".repeat(20);
        let mut signed =
            serde_json::from_slice::<serde_json::Value>(&envelope(&payload(now, &revision), &key))
                .unwrap();
        let encoded = signed["payload"].as_str().unwrap();
        signed["payload"] = serde_json::Value::String(format!(
            "{}{}",
            if encoded.starts_with('A') { "B" } else { "A" },
            &encoded[1..]
        ));
        let signed = serde_json::to_vec(&signed).unwrap();
        assert!(decode_signed_index(
            &signed,
            &keyring(&[key.verifying_key()]),
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .is_err());

        let mutable = envelope(&payload(now, "main"), &key);
        assert!(decode_signed_index(
            &mutable,
            &keyring(&[key.verifying_key()]),
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .is_err());

        let expired_payload = payload(now.saturating_sub(7200), &revision);
        let expired = envelope(&expired_payload, &key);
        assert!(decode_signed_index(
            &expired,
            &keyring(&[key.verifying_key()]),
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .is_err());

        let mut boundary_payload =
            serde_json::from_slice::<serde_json::Value>(&payload(now, &revision)).unwrap();
        boundary_payload["expires_at"] = serde_json::Value::from(now);
        let boundary_payload = serde_json::to_vec(&boundary_payload).unwrap();
        assert!(decode_signed_index(
            &envelope(&boundary_payload, &key),
            &keyring(&[key.verifying_key()]),
            "file",
            8192,
            &ModelLicensePolicy::default(),
            now,
        )
        .is_err());
    }

    #[tokio::test]
    async fn manager_falls_back_only_to_an_unexpired_verified_snapshot() {
        let now = unix_time().unwrap();
        let key = signing_key();
        let revision = "12".repeat(20);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index.json");
        fs::write(&path, envelope(&payload(now, &revision), &key))
            .await
            .unwrap();
        let manager = ModelIndexManager::from_config(
            ModelIndexManagerConfig {
                file: Some(path.clone()),
                url: None,
                public_key: Some(key_hex(&key)),
                public_keys: vec![],
                refresh_seconds: 300,
                max_download_bytes: 8192,
                state_directory: temp.path().join("index-state"),
            },
            Arc::new(ModelLicensePolicy::default()),
        )
        .unwrap()
        .unwrap();

        let fresh = manager.snapshot(false).await.unwrap();
        assert_eq!(fresh.cache_status, "fresh");
        let repeated = manager.snapshot(true).await.unwrap();
        assert_eq!(repeated.cache_status, "fresh");
        assert_eq!(repeated, fresh);
        fs::write(&path, b"tampered").await.unwrap();
        let stale = manager.snapshot(true).await.unwrap();

        assert_eq!(stale.cache_status, "stale");
        assert!(stale.warning.is_some());
        assert_eq!(stale.data, fresh.data);
    }

    #[tokio::test]
    async fn key_rotation_accepts_a_newer_signer_and_rejects_a_signed_rollback() {
        let now = unix_time().unwrap();
        let old_key = signing_key();
        let new_key = SigningKey::from_bytes(&[8_u8; 32]);
        let revision = "78".repeat(20);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index.json");
        fs::write(
            &path,
            envelope(
                &payload_generation(now, now.saturating_sub(20), &revision, "Old Key Generation"),
                &old_key,
            ),
        )
        .await
        .unwrap();
        let manager = ModelIndexManager::from_config(
            ModelIndexManagerConfig {
                file: Some(path.clone()),
                url: None,
                public_key: Some(key_hex(&old_key)),
                public_keys: vec![key_hex(&new_key)],
                refresh_seconds: 300,
                max_download_bytes: 8192,
                state_directory: temp.path().join("index-state"),
            },
            Arc::new(ModelLicensePolicy::default()),
        )
        .unwrap()
        .unwrap();

        let old = manager.snapshot(false).await.unwrap();
        assert_eq!(old.name, "Old Key Generation");
        fs::write(
            &path,
            envelope(
                &payload_generation(now, now.saturating_sub(10), &revision, "New Key Generation"),
                &new_key,
            ),
        )
        .await
        .unwrap();
        let rotated = manager.snapshot(true).await.unwrap();
        assert_eq!(rotated.name, "New Key Generation");
        assert_eq!(rotated.key_id, key_id(&new_key.verifying_key()));
        assert_eq!(rotated.cache_status, "fresh");

        fs::remove_dir_all(temp.path().join("index-state"))
            .await
            .unwrap();

        fs::write(
            &path,
            envelope(
                &payload_generation(now, now.saturating_sub(30), &revision, "Signed Rollback"),
                &old_key,
            ),
        )
        .await
        .unwrap();
        let retained = manager.snapshot(true).await.unwrap();
        assert_eq!(retained.name, "New Key Generation");
        assert_eq!(retained.key_id, key_id(&new_key.verifying_key()));
        assert_eq!(retained.cache_status, "stale");
        assert!(retained.warning.unwrap().contains("rollback"));
    }

    #[tokio::test]
    async fn persisted_watermark_survives_restart_and_trust_set_reduction() {
        let now = unix_time().unwrap();
        let old_key = signing_key();
        let new_key = SigningKey::from_bytes(&[8_u8; 32]);
        let revision = "90".repeat(20);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index.json");
        let state_directory = temp.path().join("index-state");
        let newer_generation = now.saturating_sub(10);
        fs::write(
            &path,
            envelope(
                &payload_generation(now, newer_generation, &revision, "New Signer"),
                &new_key,
            ),
        )
        .await
        .unwrap();
        let manager = ModelIndexManager::from_config(
            ModelIndexManagerConfig {
                file: Some(path.clone()),
                url: None,
                public_key: Some(key_hex(&old_key)),
                public_keys: vec![key_hex(&new_key)],
                refresh_seconds: 300,
                max_download_bytes: 8192,
                state_directory: state_directory.clone(),
            },
            Arc::new(ModelLicensePolicy::default()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(manager.snapshot(false).await.unwrap().name, "New Signer");
        drop(manager);

        fs::write(
            &path,
            envelope(
                &payload_generation(
                    now,
                    now.saturating_sub(20),
                    &revision,
                    "Old Signer Rollback",
                ),
                &old_key,
            ),
        )
        .await
        .unwrap();
        let restarted = ModelIndexManager::from_config(
            ModelIndexManagerConfig {
                file: Some(path.clone()),
                url: None,
                public_key: Some(key_hex(&old_key)),
                public_keys: vec![],
                refresh_seconds: 300,
                max_download_bytes: 8192,
                state_directory: state_directory.clone(),
            },
            Arc::new(ModelLicensePolicy::default()),
        )
        .unwrap()
        .unwrap();
        assert!(restarted.snapshot(false).await.is_err());

        fs::write(
            &path,
            envelope(
                &payload_generation(now, newer_generation, &revision, "Equal-Time Conflict"),
                &old_key,
            ),
        )
        .await
        .unwrap();
        let conflicting = ModelIndexManager::from_config(
            ModelIndexManagerConfig {
                file: Some(path),
                url: None,
                public_key: Some(key_hex(&old_key)),
                public_keys: vec![],
                refresh_seconds: 300,
                max_download_bytes: 8192,
                state_directory,
            },
            Arc::new(ModelLicensePolicy::default()),
        )
        .unwrap()
        .unwrap();
        assert!(conflicting.snapshot(false).await.is_err());
    }

    #[test]
    fn installation_state_distinguishes_verified_upgradable_and_conflicting_aliases() {
        let entry = ModelIndexEntry {
            id: "stable-model".to_string(),
            name: "Stable Model".to_string(),
            description: "A signed upgrade fixture.".to_string(),
            download_url: Some(format!(
                "https://huggingface.co/acme/new/resolve/{}/model.gguf",
                "22".repeat(20)
            )),
            filename: "model.gguf".to_string(),
            format: "gguf".to_string(),
            size_bytes: 20,
            sha256: "bb".repeat(32),
            files: Vec::new(),
            license: "Apache-2.0".to_string(),
            family: None,
            parameter_count: None,
            quantization: None,
            tags: Vec::new(),
            downloadable: true,
            blocking_reasons: Vec::new(),
        };
        let previous = ModelCatalogEntry {
            id: "model.gguf".to_string(),
            name: "Model".to_string(),
            kind: "file".to_string(),
            format: "gguf".to_string(),
            size_bytes: 10,
            size_complete: true,
            modified_at: Some(1),
            active: false,
            provenance: Some(super::super::model_provenance::ModelProvenance {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(entry.id.clone()),
                source_url: None,
                source_host: Some("huggingface.co".to_string()),
                sha256: "aa".repeat(32),
                file_count: None,
                license: Some("Apache-2.0".to_string()),
                installed_at: 1,
                last_verified_at: None,
                integrity_mismatch_at: None,
            }),
            provenance_error: None,
        };
        let catalog = |models| ModelCatalog {
            root: "/models".to_string(),
            root_exists: true,
            models,
        };

        assert_eq!(
            model_index_installation_state(&catalog(Vec::new()), &entry),
            ModelIndexInstallationState::Missing
        );
        assert_eq!(
            model_index_installation_state(&catalog(vec![previous.clone()]), &entry),
            ModelIndexInstallationState::Upgradable
        );
        let descriptor =
            model_index_upgrade_descriptor(&catalog(vec![previous.clone()]), &entry).unwrap();
        assert_eq!(descriptor.catalog_id, "model.gguf");
        assert_eq!(descriptor.sha256, "aa".repeat(32));

        let mut verified = previous.clone();
        verified.size_bytes = entry.size_bytes;
        let provenance = verified.provenance.as_mut().unwrap();
        provenance.sha256 = entry.sha256.clone();
        assert_eq!(
            model_index_installation_state(&catalog(vec![verified]), &entry),
            ModelIndexInstallationState::Verified
        );

        let mut moved = previous.clone();
        moved.id = "old-model.gguf".to_string();
        assert_eq!(
            model_index_installation_state(&catalog(vec![moved.clone()]), &entry),
            ModelIndexInstallationState::Upgradable
        );
        let occupied = ModelCatalogEntry {
            id: entry.filename.clone(),
            provenance: None,
            ..previous.clone()
        };
        assert_eq!(
            model_index_installation_state(&catalog(vec![moved.clone(), occupied]), &entry),
            ModelIndexInstallationState::Conflict
        );

        moved.provenance.as_mut().unwrap().integrity_mismatch_at = Some(2);
        assert_eq!(
            model_index_installation_state(&catalog(vec![moved]), &entry),
            ModelIndexInstallationState::Conflict
        );

        let mut duplicate = previous.clone();
        duplicate.id = "duplicate.gguf".to_string();
        assert_eq!(
            model_index_installation_state(&catalog(vec![previous, duplicate]), &entry),
            ModelIndexInstallationState::Conflict
        );
    }

    #[test]
    fn deployment_limits_annotate_verified_entries_without_rejecting_the_index() {
        let now = unix_time().unwrap();
        let key = signing_key();
        let revision = "34".repeat(20);
        let policy = ModelLicensePolicy::new(vec!["MIT".to_string()]).unwrap();
        let snapshot = decode_signed_index(
            &envelope(&payload(now, &revision), &key),
            &keyring(&[key.verifying_key()]),
            "file",
            1024,
            &policy,
            now,
        )
        .unwrap();

        assert!(!snapshot.data[0].downloadable);
        assert_eq!(
            snapshot.data[0].blocking_reasons,
            vec!["size_limit", "license_policy"]
        );
    }
}
