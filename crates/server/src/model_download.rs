//! Verified, resumable downloads into the local model catalog.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use futures::StreamExt as _;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_LENGTH, CONTENT_RANGE, ETAG, LOCATION, RANGE};
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs::{self, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use super::model_license::{ModelLicensePolicy, ModelLicensePolicyStatus};
use super::model_manager::validate_model_filename;
use super::model_package::{
    decode_sha256 as decode_package_sha256, normalize_package_files, package_digest,
    validate_package_id, validate_package_path, ModelPackageFile,
};
use super::model_provenance::{
    remove_provenance, sanitized_download_source, write_package_provenance, write_provenance,
    ModelAcquisitionKind, ModelPackageProvenanceDraft, ModelProvenanceDraft,
};
use super::model_storage::ModelStorageManager;
use super::model_upgrade::{commit_model_upgrade, ModelUpgradeSource, ModelUpgradeTarget};

const STAGING_DIRECTORY: &str = ".bloom-downloads";
const STATUS_UPDATE_BYTES: u64 = 1024 * 1024;
const STATUS_UPDATE_INTERVAL: Duration = Duration::from_millis(250);
const MAX_REDIRECTS: usize = 5;
const MAX_STAGED_DOWNLOADS: usize = 1_000;
const MAX_SOURCE_URL_BYTES: usize = 2_048;
const LINKED_ETAG_HEADER: &str = "x-linked-etag";
const LINKED_SIZE_HEADER: &str = "x-linked-size";
const REPO_COMMIT_HEADER: &str = "x-repo-commit";

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelDownloadSourceRequest {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelDownloadSource {
    pub download_url: String,
    pub filename: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub commit_hash: Option<String>,
    pub verification_ready: bool,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct ModelDownloadRequest {
    pub url: String,
    pub filename: String,
    pub sha256: String,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(skip)]
    pub expected_size_bytes: Option<u64>,
    #[serde(skip)]
    pub model_index_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelPackageDownloadFile {
    pub url: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelPackageDownloadRequest {
    pub directory: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub files: Vec<ModelPackageDownloadFile>,
    pub license: Option<String>,
    pub model_index_id: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelDownloadPhase {
    Idle,
    Queued,
    Downloading,
    Verifying,
    Complete,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelDownloadStatus {
    pub phase: ModelDownloadPhase,
    pub filename: Option<String>,
    pub source_host: Option<String>,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub resumable: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct StagedModelDownload {
    pub filename: String,
    pub source_host: String,
    pub downloaded_bytes: u64,
    pub modified_at: Option<u64>,
}

impl Default for ModelDownloadStatus {
    fn default() -> Self {
        Self {
            phase: ModelDownloadPhase::Idle,
            filename: None,
            source_host: None,
            downloaded_bytes: 0,
            total_bytes: None,
            resumable: false,
            error: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelDownloadStartError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Internal(String),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelDownloadInspectError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    TooLarge(String),
    #[error("{0}")]
    Unavailable(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct StagedDownload {
    version: u8,
    url: String,
    filename: String,
    sha256: String,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    expected_size_bytes: Option<u64>,
    #[serde(default)]
    model_index_id: Option<String>,
    #[serde(default)]
    replacement: Option<ModelUpgradeSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedPackageDownload {
    version: u8,
    directory: String,
    size_bytes: u64,
    sha256: String,
    files: Vec<StagedPackageFile>,
    #[serde(default)]
    license: Option<String>,
    model_index_id: String,
    #[serde(default)]
    replacement: Option<ModelUpgradeSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedPackageFile {
    url: String,
    filename: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone)]
struct PreparedDownload {
    url: Url,
    filename: String,
    sha256: [u8; 32],
    sha256_hex: String,
    source_host: String,
    source_url: Option<String>,
    license: Option<String>,
    expected_size_bytes: Option<u64>,
    model_index_id: Option<String>,
    replacement: Option<ModelUpgradeSource>,
}

#[derive(Debug, Clone)]
struct PreparedPackageDownload {
    directory: String,
    size_bytes: u64,
    sha256: String,
    files: Vec<PreparedPackageFile>,
    source_host: String,
    source_url: Option<String>,
    license: Option<String>,
    model_index_id: String,
    replacement: Option<ModelUpgradeSource>,
}

#[derive(Debug, Clone)]
struct PreparedPackageFile {
    url: Url,
    filename: String,
    size_bytes: u64,
    sha256: [u8; 32],
    sha256_hex: String,
}

#[derive(Debug)]
struct DownloadFailure {
    message: String,
    resumable: bool,
}

struct ActiveDownload {
    cancellation: CancellationToken,
    filename: String,
    sha256: String,
    expected_size_bytes: Option<u64>,
    model_index_id: Option<String>,
    replacement_source: Option<String>,
}

impl DownloadFailure {
    fn resumable(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
            resumable: true,
        }
    }

    fn terminal(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
            resumable: false,
        }
    }
}

#[derive(Clone, Default)]
struct DownloadUrlPolicy {
    allow_http: bool,
    allow_loopback: bool,
}

pub(crate) struct ModelDownloadManager {
    models_root: PathBuf,
    max_bytes: u64,
    client: Client,
    metadata_client: Client,
    policy: DownloadUrlPolicy,
    status: RwLock<ModelDownloadStatus>,
    active: Mutex<Option<ActiveDownload>>,
    catalog_revision: AtomicU64,
    storage: Arc<ModelStorageManager>,
    license_policy: Arc<ModelLicensePolicy>,
}

impl ModelDownloadManager {
    #[cfg(test)]
    pub(crate) fn with_storage(
        models_root: PathBuf,
        max_bytes: u64,
        storage: Arc<ModelStorageManager>,
    ) -> Result<Arc<Self>> {
        Self::with_policy_and_storage(
            models_root,
            max_bytes,
            DownloadUrlPolicy::default(),
            storage,
        )
    }

    pub(crate) fn with_storage_and_license_policy(
        models_root: PathBuf,
        max_bytes: u64,
        storage: Arc<ModelStorageManager>,
        license_policy: Arc<ModelLicensePolicy>,
    ) -> Result<Arc<Self>> {
        Self::with_policy_storage_and_license(
            models_root,
            max_bytes,
            DownloadUrlPolicy::default(),
            storage,
            license_policy,
        )
    }

    #[cfg(test)]
    fn with_policy(
        models_root: PathBuf,
        max_bytes: u64,
        policy: DownloadUrlPolicy,
    ) -> Result<Arc<Self>> {
        let storage = ModelStorageManager::new(models_root.clone(), 0, 0);
        Self::with_policy_and_storage(models_root, max_bytes, policy, storage)
    }

    #[cfg(test)]
    pub(crate) fn with_loopback_test_policy(
        models_root: PathBuf,
        max_bytes: u64,
    ) -> Result<Arc<Self>> {
        Self::with_policy(
            models_root,
            max_bytes,
            DownloadUrlPolicy {
                allow_http: true,
                allow_loopback: true,
            },
        )
    }

    #[cfg(test)]
    fn with_policy_and_storage(
        models_root: PathBuf,
        max_bytes: u64,
        policy: DownloadUrlPolicy,
        storage: Arc<ModelStorageManager>,
    ) -> Result<Arc<Self>> {
        Self::with_policy_storage_and_license(
            models_root,
            max_bytes,
            policy,
            storage,
            Arc::new(ModelLicensePolicy::default()),
        )
    }

    fn with_policy_storage_and_license(
        models_root: PathBuf,
        max_bytes: u64,
        policy: DownloadUrlPolicy,
        storage: Arc<ModelStorageManager>,
        license_policy: Arc<ModelLicensePolicy>,
    ) -> Result<Arc<Self>> {
        if max_bytes == 0 {
            return Err(anyhow!(
                "maximum model download size must be greater than zero"
            ));
        }
        let redirect_policy = policy.clone();
        let client = Client::builder()
            .user_agent(concat!("bloom/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(15))
            .read_timeout(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::custom(move |attempt| {
                if attempt.previous().len() >= MAX_REDIRECTS {
                    return attempt.error("model download exceeded the redirect limit");
                }
                if validate_download_url(attempt.url(), &redirect_policy).is_err() {
                    return attempt.error("model download redirected to an untrusted URL");
                }
                attempt.follow()
            }))
            .build()
            .context("failed to create the model download HTTP client")?;
        let metadata_client = Client::builder()
            .user_agent(concat!("bloom/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("failed to create the model metadata HTTP client")?;
        Ok(Arc::new(Self {
            models_root,
            max_bytes,
            client,
            metadata_client,
            policy,
            status: RwLock::new(ModelDownloadStatus::default()),
            active: Mutex::new(None),
            catalog_revision: AtomicU64::new(0),
            storage,
            license_policy,
        }))
    }

    pub(crate) async fn status(&self) -> ModelDownloadStatus {
        self.status.read().await.clone()
    }

    pub(crate) async fn active_matches(
        &self,
        filename: &str,
        sha256: &str,
        expected_size_bytes: Option<u64>,
        model_index_id: Option<&str>,
    ) -> bool {
        self.active.lock().await.as_ref().is_some_and(|active| {
            active.filename == filename
                && active.sha256.eq_ignore_ascii_case(sha256)
                && active.expected_size_bytes == expected_size_bytes
                && active.model_index_id.as_deref() == model_index_id
        })
    }

    pub(crate) async fn upgrade_source_active(&self, catalog_id: &str) -> bool {
        self.active
            .lock()
            .await
            .as_ref()
            .and_then(|active| active.replacement_source.as_deref())
            == Some(catalog_id)
    }

    pub(crate) fn catalog_revision(&self) -> u64 {
        self.catalog_revision.load(Ordering::Acquire)
    }

    pub(crate) fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub(crate) fn license_policy(&self) -> ModelLicensePolicyStatus {
        self.license_policy.status()
    }

    pub(crate) async fn inspect_source(
        &self,
        request: ModelDownloadSourceRequest,
    ) -> std::result::Result<ModelDownloadSource, ModelDownloadInspectError> {
        let (source_url, filename) = prepare_source_url(&request.url, &self.policy)
            .map_err(|error| ModelDownloadInspectError::Invalid(error.to_string()))?;
        let mut request_url = source_url.clone();
        let mut sha256 = None;
        let mut commit_hash = None;
        let mut size_bytes = None;

        for redirect_count in 0..=MAX_REDIRECTS {
            let response = self
                .metadata_client
                .head(request_url.clone())
                .header(ACCEPT_ENCODING, "identity")
                .send()
                .await
                .map_err(|_| {
                    ModelDownloadInspectError::Unavailable(
                        "The source metadata request failed.".to_string(),
                    )
                })?;
            let status = response.status();
            let headers = response.headers();

            if sha256.is_none() {
                sha256 = metadata_sha256(headers);
            }
            if commit_hash.is_none() {
                commit_hash = metadata_commit(headers);
            }
            let reported_size = metadata_size(headers, status)
                .map_err(|message| ModelDownloadInspectError::Unavailable(message.to_string()))?;
            if let Some(reported_size) = reported_size {
                if reported_size > self.max_bytes {
                    return Err(ModelDownloadInspectError::TooLarge(format!(
                        "The source declares {reported_size} bytes, exceeding the configured model download limit of {} bytes.",
                        self.max_bytes
                    )));
                }
                size_bytes = Some(reported_size);
            }

            if status.is_success() || (status.is_redirection() && sha256.is_some()) {
                break;
            }
            if !status.is_redirection() {
                return Err(ModelDownloadInspectError::Unavailable(format!(
                    "The source metadata request returned HTTP {}.",
                    status.as_u16()
                )));
            }
            if redirect_count == MAX_REDIRECTS {
                return Err(ModelDownloadInspectError::Unavailable(
                    "The source metadata request exceeded the redirect limit.".to_string(),
                ));
            }
            let location = headers
                .get(LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    ModelDownloadInspectError::Unavailable(
                        "The source metadata redirect did not include a valid location."
                            .to_string(),
                    )
                })?;
            let next_url = request_url.join(location).map_err(|_| {
                ModelDownloadInspectError::Unavailable(
                    "The source metadata redirect location was invalid.".to_string(),
                )
            })?;
            validate_download_url(&next_url, &self.policy).map_err(|_| {
                ModelDownloadInspectError::Unavailable(
                    "The source metadata request redirected to an untrusted URL.".to_string(),
                )
            })?;
            request_url = next_url;
        }

        let download_url = commit_hash
            .as_deref()
            .map(|commit| pin_source_revision(&source_url, commit))
            .transpose()
            .map_err(|error| ModelDownloadInspectError::Invalid(error.to_string()))?
            .unwrap_or(source_url);
        let verification_ready = sha256.is_some();
        let warning = match (verification_ready, commit_hash.is_some()) {
            (false, true) => Some(
                "The source was pinned to a commit, but it did not publish a 64-character SHA-256 digest. Enter an independently obtained SHA-256 before downloading."
                    .to_string(),
            ),
            (false, false) => Some(
                "The source did not publish a SHA-256 digest or immutable commit. Enter an independently obtained SHA-256 before downloading."
                    .to_string(),
            ),
            (true, false) => Some(
                "Checksum metadata is available, but the source could not be pinned to an immutable commit. Bloom will still verify the exact downloaded bytes."
                    .to_string(),
            ),
            (true, true) => None,
        };

        Ok(ModelDownloadSource {
            download_url: download_url.to_string(),
            filename,
            size_bytes,
            sha256,
            commit_hash,
            verification_ready,
            warning,
        })
    }

    pub(crate) async fn start(
        self: &Arc<Self>,
        request: ModelDownloadRequest,
    ) -> std::result::Result<ModelDownloadStatus, ModelDownloadStartError> {
        self.start_with_replacement(request, None).await
    }

    pub(crate) async fn start_upgrade(
        self: &Arc<Self>,
        request: ModelDownloadRequest,
        replacement: ModelUpgradeSource,
    ) -> std::result::Result<ModelDownloadStatus, ModelDownloadStartError> {
        self.start_with_replacement(request, Some(replacement))
            .await
    }

    async fn start_with_replacement(
        self: &Arc<Self>,
        request: ModelDownloadRequest,
        replacement: Option<ModelUpgradeSource>,
    ) -> std::result::Result<ModelDownloadStatus, ModelDownloadStartError> {
        if request
            .expected_size_bytes
            .is_some_and(|size| size == 0 || size > self.max_bytes)
        {
            return Err(ModelDownloadStartError::Invalid(format!(
                "expected model size must be between 1 and {} bytes",
                self.max_bytes
            )));
        }
        let mut prepared = prepare_download(request, &self.policy, &self.license_policy)
            .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        validate_replacement_identity(
            replacement.as_ref(),
            &prepared.filename,
            prepared.model_index_id.as_deref(),
        )
        .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        prepared.replacement = replacement;
        let destination = self.models_root.join(&prepared.filename);
        if destination.exists()
            && prepared
                .replacement
                .as_ref()
                .is_none_or(|source| source.catalog_id != prepared.filename)
        {
            return Err(ModelDownloadStartError::Conflict(format!(
                "A catalog entry named '{}' already exists.",
                prepared.filename
            )));
        }

        fs::create_dir_all(&self.models_root)
            .await
            .map_err(|error| {
                ModelDownloadStartError::Internal(format!(
                    "failed to create the model catalog directory: {error}"
                ))
            })?;
        let staging_root = self.staging_root();
        fs::create_dir_all(&staging_root).await.map_err(|error| {
            ModelDownloadStartError::Internal(format!(
                "failed to create the model download staging directory: {error}"
            ))
        })?;
        secure_staging_directory(&staging_root)
            .await
            .map_err(|error| {
                ModelDownloadStartError::Internal(format!(
                    "failed to secure the model download staging directory: {error}"
                ))
            })?;

        let mut active = self.active.lock().await;
        if active.is_some() {
            return Err(ModelDownloadStartError::Conflict(
                "Another model download is already in progress.".to_string(),
            ));
        }
        let cancellation = CancellationToken::new();
        *active = Some(ActiveDownload {
            cancellation: cancellation.clone(),
            filename: prepared.filename.clone(),
            sha256: prepared.sha256_hex.clone(),
            expected_size_bytes: prepared.expected_size_bytes,
            model_index_id: prepared.model_index_id.clone(),
            replacement_source: prepared
                .replacement
                .as_ref()
                .map(|source| source.catalog_id.clone()),
        });
        let status = ModelDownloadStatus {
            phase: ModelDownloadPhase::Queued,
            filename: Some(prepared.filename.clone()),
            source_host: Some(prepared.source_host.clone()),
            downloaded_bytes: 0,
            total_bytes: None,
            resumable: false,
            error: None,
        };
        *self.status.write().await = status.clone();
        self.storage.mark_download_active(&prepared.filename).await;
        drop(active);

        let manager = Arc::clone(self);
        tokio::spawn(async move {
            manager.run(prepared, cancellation).await;
        });
        Ok(status)
    }

    pub(crate) async fn start_package(
        self: &Arc<Self>,
        request: ModelPackageDownloadRequest,
    ) -> std::result::Result<ModelDownloadStatus, ModelDownloadStartError> {
        self.start_package_with_replacement(request, None).await
    }

    pub(crate) async fn start_package_upgrade(
        self: &Arc<Self>,
        request: ModelPackageDownloadRequest,
        replacement: ModelUpgradeSource,
    ) -> std::result::Result<ModelDownloadStatus, ModelDownloadStartError> {
        self.start_package_with_replacement(request, Some(replacement))
            .await
    }

    async fn start_package_with_replacement(
        self: &Arc<Self>,
        request: ModelPackageDownloadRequest,
        replacement: Option<ModelUpgradeSource>,
    ) -> std::result::Result<ModelDownloadStatus, ModelDownloadStartError> {
        if request.size_bytes == 0 || request.size_bytes > self.max_bytes {
            return Err(ModelDownloadStartError::Invalid(format!(
                "expected model package size must be between 1 and {} bytes",
                self.max_bytes
            )));
        }
        let mut prepared = prepare_package_download(request, &self.policy, &self.license_policy)
            .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        validate_replacement_identity(
            replacement.as_ref(),
            &prepared.directory,
            Some(&prepared.model_index_id),
        )
        .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        prepared.replacement = replacement;
        let destination = self.models_root.join(&prepared.directory);
        if destination.exists()
            && prepared
                .replacement
                .as_ref()
                .is_none_or(|source| source.catalog_id != prepared.directory)
        {
            return Err(ModelDownloadStartError::Conflict(format!(
                "A catalog entry named '{}' already exists.",
                prepared.directory
            )));
        }

        fs::create_dir_all(&self.models_root)
            .await
            .map_err(|error| {
                ModelDownloadStartError::Internal(format!(
                    "failed to create the model catalog directory: {error}"
                ))
            })?;
        let staging_root = self.staging_root();
        fs::create_dir_all(&staging_root).await.map_err(|error| {
            ModelDownloadStartError::Internal(format!(
                "failed to create the model download staging directory: {error}"
            ))
        })?;
        secure_staging_directory(&staging_root)
            .await
            .map_err(|error| {
                ModelDownloadStartError::Internal(format!(
                    "failed to secure the model download staging directory: {error}"
                ))
            })?;

        let mut active = self.active.lock().await;
        if active.is_some() {
            return Err(ModelDownloadStartError::Conflict(
                "Another model download is already in progress.".to_string(),
            ));
        }
        let cancellation = CancellationToken::new();
        *active = Some(ActiveDownload {
            cancellation: cancellation.clone(),
            filename: prepared.directory.clone(),
            sha256: prepared.sha256.clone(),
            expected_size_bytes: Some(prepared.size_bytes),
            model_index_id: Some(prepared.model_index_id.clone()),
            replacement_source: prepared
                .replacement
                .as_ref()
                .map(|source| source.catalog_id.clone()),
        });
        let status = ModelDownloadStatus {
            phase: ModelDownloadPhase::Queued,
            filename: Some(prepared.directory.clone()),
            source_host: Some(prepared.source_host.clone()),
            downloaded_bytes: 0,
            total_bytes: Some(prepared.size_bytes),
            resumable: false,
            error: None,
        };
        *self.status.write().await = status.clone();
        self.storage.mark_download_active(&prepared.directory).await;
        drop(active);

        let manager = Arc::clone(self);
        tokio::spawn(async move {
            manager.run_package(prepared, cancellation).await;
        });
        Ok(status)
    }

    pub(crate) async fn cancel(&self) -> bool {
        let active = self.active.lock().await;
        if let Some(active) = active.as_ref() {
            active.cancellation.cancel();
            true
        } else {
            false
        }
    }

    pub(crate) async fn staged(&self) -> Vec<StagedModelDownload> {
        match self.scan_staged().await {
            Ok(staged) => staged,
            Err(error) => {
                tracing::warn!(%error, "Failed to scan staged model downloads");
                Vec::new()
            }
        }
    }

    pub(crate) async fn resume(
        self: &Arc<Self>,
        filename: &str,
        license: Option<String>,
    ) -> std::result::Result<ModelDownloadStatus, ModelDownloadStartError> {
        if validate_model_filename(filename).is_ok() {
            let (mut request, replacement) = self.read_staged_request(filename).await?;
            if license
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty())
            {
                request.license = license;
            }
            self.start_with_replacement(request, replacement).await
        } else {
            let (mut request, replacement) = self.read_staged_package_request(filename).await?;
            if license
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty())
            {
                request.license = license;
            }
            self.start_package_with_replacement(request, replacement)
                .await
        }
    }

    pub(crate) async fn discard(
        &self,
        filename: &str,
    ) -> std::result::Result<(), ModelDownloadStartError> {
        let single_file = validate_model_filename(filename).is_ok();
        if !single_file {
            validate_package_id(filename)
                .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        }
        let status = self.status.read().await.clone();
        if self.active.lock().await.is_some() && status.filename.as_deref() == Some(filename) {
            return Err(ModelDownloadStartError::Conflict(
                "Cancel the active model download before discarding its partial data.".to_string(),
            ));
        }
        let staging_root = self.staging_root();
        let _storage = self.storage.serial().await;
        require_safe_staging_directory(&staging_root, filename).await?;
        let mut removed = false;
        if single_file {
            let part_path = staging_root.join(format!("{filename}.part"));
            let metadata_path = staging_root.join(format!("{filename}.json"));
            for path in [
                part_path,
                metadata_path.clone(),
                metadata_path.with_extension("json.tmp"),
            ] {
                removed |= remove_staged_file(&path).await.map_err(|error| {
                    ModelDownloadStartError::Internal(format!(
                        "failed to discard staged model download: {error}"
                    ))
                })?;
            }
        } else {
            let package_root = staging_root.join(format!("{filename}.package"));
            let metadata_path = staging_root.join(format!("{filename}.package.json"));
            removed = fs::symlink_metadata(&package_root).await.is_ok()
                || fs::symlink_metadata(&metadata_path).await.is_ok();
            cleanup_package_staging(&package_root, &metadata_path).await;
        }
        if !removed {
            return Err(ModelDownloadStartError::NotFound(format!(
                "No staged download exists for '{filename}'."
            )));
        }
        if status.filename.as_deref() == Some(filename) {
            *self.status.write().await = ModelDownloadStatus::default();
        }
        Ok(())
    }

    fn staging_root(&self) -> PathBuf {
        self.models_root.join(STAGING_DIRECTORY)
    }

    async fn scan_staged(&self) -> Result<Vec<StagedModelDownload>> {
        let staging_root = self.staging_root();
        match fs::symlink_metadata(&staging_root).await {
            Ok(_) => secure_staging_directory(&staging_root).await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).context("failed to inspect model download staging directory")
            }
        }
        let mut entries = match fs::read_dir(&staging_root).await {
            Ok(entries) => entries,
            Err(error) => {
                return Err(error).context("failed to open model download staging directory")
            }
        };
        let mut staged = Vec::new();
        let mut inspected = 0_usize;
        while inspected < MAX_STAGED_DOWNLOADS {
            let Some(entry) = entries
                .next_entry()
                .await
                .context("failed to read model download staging entry")?
            else {
                break;
            };
            inspected += 1;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            if reject_symlink(&path).await.is_err() {
                continue;
            }
            let Ok(bytes) = fs::read(&path).await else {
                continue;
            };
            if path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.ends_with(".package.json"))
            {
                let Ok(metadata) = serde_json::from_slice::<StagedPackageDownload>(&bytes) else {
                    continue;
                };
                if let Ok(package) = inspect_staged_package(
                    &staging_root,
                    &path,
                    metadata,
                    &self.policy,
                    self.max_bytes,
                )
                .await
                {
                    staged.push(package);
                }
                continue;
            }
            let Ok(metadata) = serde_json::from_slice::<StagedDownload>(&bytes) else {
                continue;
            };
            if validate_model_filename(&metadata.filename).is_err()
                || path.file_stem().and_then(|value| value.to_str())
                    != Some(metadata.filename.as_str())
                || validate_replacement_identity(
                    metadata.replacement.as_ref(),
                    &metadata.filename,
                    metadata.model_index_id.as_deref(),
                )
                .is_err()
            {
                continue;
            }
            let Ok(url) = Url::parse(&metadata.url) else {
                continue;
            };
            if validate_download_url(&url, &self.policy).is_err() {
                continue;
            }
            let part_path = staging_root.join(format!("{}.part", metadata.filename));
            if reject_symlink(&part_path).await.is_err() {
                continue;
            }
            let Ok(part_metadata) = fs::metadata(&part_path).await else {
                continue;
            };
            if !part_metadata.is_file() || part_metadata.len() > self.max_bytes {
                continue;
            }
            let source_host = url.host_str().unwrap_or("unknown").to_string();
            let modified_at = part_metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs());
            staged.push(StagedModelDownload {
                filename: metadata.filename,
                source_host,
                downloaded_bytes: part_metadata.len(),
                modified_at,
            });
        }
        staged.sort_by(|left, right| left.filename.cmp(&right.filename));
        Ok(staged)
    }

    async fn read_staged_request(
        &self,
        filename: &str,
    ) -> std::result::Result<
        (ModelDownloadRequest, Option<ModelUpgradeSource>),
        ModelDownloadStartError,
    > {
        validate_model_filename(filename)
            .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        let staging_root = self.staging_root();
        require_safe_staging_directory(&staging_root, filename).await?;
        let metadata_path = staging_root.join(format!("{filename}.json"));
        reject_symlink(&metadata_path).await.map_err(|error| {
            ModelDownloadStartError::Invalid(format!(
                "staged model download metadata is unsafe: {error}"
            ))
        })?;
        let bytes = fs::read(&metadata_path).await.map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ModelDownloadStartError::NotFound(format!(
                    "No staged download exists for '{filename}'."
                ))
            } else {
                ModelDownloadStartError::Internal(format!(
                    "failed to read staged model download metadata: {error}"
                ))
            }
        })?;
        let metadata: StagedDownload = serde_json::from_slice(&bytes).map_err(|error| {
            ModelDownloadStartError::Invalid(format!(
                "staged model download metadata is invalid: {error}"
            ))
        })?;
        if metadata.filename != filename {
            return Err(ModelDownloadStartError::Invalid(
                "staged model download metadata does not match its filename".to_string(),
            ));
        }
        validate_replacement_identity(
            metadata.replacement.as_ref(),
            &metadata.filename,
            metadata.model_index_id.as_deref(),
        )
        .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        Ok((
            ModelDownloadRequest {
                url: metadata.url,
                filename: metadata.filename,
                sha256: metadata.sha256,
                license: metadata.license,
                expected_size_bytes: metadata.expected_size_bytes,
                model_index_id: metadata.model_index_id,
            },
            metadata.replacement,
        ))
    }

    async fn read_staged_package_request(
        &self,
        directory: &str,
    ) -> std::result::Result<
        (ModelPackageDownloadRequest, Option<ModelUpgradeSource>),
        ModelDownloadStartError,
    > {
        validate_package_id(directory)
            .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        let staging_root = self.staging_root();
        require_safe_staging_directory(&staging_root, directory).await?;
        let metadata_path = staging_root.join(format!("{directory}.package.json"));
        reject_symlink(&metadata_path).await.map_err(|error| {
            ModelDownloadStartError::Invalid(format!(
                "staged model package metadata is unsafe: {error}"
            ))
        })?;
        let bytes = fs::read(&metadata_path).await.map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ModelDownloadStartError::NotFound(format!(
                    "No staged download exists for '{directory}'."
                ))
            } else {
                ModelDownloadStartError::Internal(format!(
                    "failed to read staged model package metadata: {error}"
                ))
            }
        })?;
        let metadata: StagedPackageDownload = serde_json::from_slice(&bytes).map_err(|error| {
            ModelDownloadStartError::Invalid(format!(
                "staged model package metadata is invalid: {error}"
            ))
        })?;
        if metadata.version != 1 || metadata.directory != directory {
            return Err(ModelDownloadStartError::Invalid(
                "staged model package metadata does not match its directory".to_string(),
            ));
        }
        validate_replacement_identity(
            metadata.replacement.as_ref(),
            &metadata.directory,
            Some(&metadata.model_index_id),
        )
        .map_err(|error| ModelDownloadStartError::Invalid(error.to_string()))?;
        Ok((
            ModelPackageDownloadRequest {
                directory: metadata.directory,
                size_bytes: metadata.size_bytes,
                sha256: metadata.sha256,
                files: metadata
                    .files
                    .into_iter()
                    .map(|file| ModelPackageDownloadFile {
                        url: file.url,
                        filename: file.filename,
                        size_bytes: file.size_bytes,
                        sha256: file.sha256,
                    })
                    .collect(),
                license: metadata.license,
                model_index_id: metadata.model_index_id,
            },
            metadata.replacement,
        ))
    }

    async fn run(&self, prepared: PreparedDownload, cancellation: CancellationToken) {
        let result = self.execute(&prepared, &cancellation).await;
        self.storage.release_download(&prepared.filename).await;
        match result {
            Ok(()) => {
                self.catalog_revision.fetch_add(1, Ordering::AcqRel);
                let mut status = self.status.write().await;
                status.phase = ModelDownloadPhase::Complete;
                status.resumable = false;
                status.error = None;
            }
            Err(_failure) if cancellation.is_cancelled() => {
                let mut status = self.status.write().await;
                status.phase = ModelDownloadPhase::Cancelled;
                status.resumable = true;
                status.error = None;
            }
            Err(failure) => {
                let mut status = self.status.write().await;
                status.phase = ModelDownloadPhase::Error;
                status.resumable = failure.resumable;
                status.error = Some(failure.message);
            }
        }
        self.active.lock().await.take();
    }

    async fn run_package(
        &self,
        prepared: PreparedPackageDownload,
        cancellation: CancellationToken,
    ) {
        let result = self.execute_package(&prepared, &cancellation).await;
        self.storage.release_download(&prepared.directory).await;
        match result {
            Ok(()) => {
                self.catalog_revision.fetch_add(1, Ordering::AcqRel);
                let mut status = self.status.write().await;
                status.phase = ModelDownloadPhase::Complete;
                status.downloaded_bytes = prepared.size_bytes;
                status.total_bytes = Some(prepared.size_bytes);
                status.resumable = false;
                status.error = None;
            }
            Err(_failure) if cancellation.is_cancelled() => {
                let mut status = self.status.write().await;
                status.phase = ModelDownloadPhase::Cancelled;
                status.resumable = true;
                status.error = None;
            }
            Err(failure) => {
                let mut status = self.status.write().await;
                status.phase = ModelDownloadPhase::Error;
                status.resumable = failure.resumable;
                status.error = Some(failure.message);
            }
        }
        self.active.lock().await.take();
    }

    async fn execute_package(
        &self,
        prepared: &PreparedPackageDownload,
        cancellation: &CancellationToken,
    ) -> std::result::Result<(), DownloadFailure> {
        let staging_root = self.staging_root();
        let package_root = staging_root.join(format!("{}.package", prepared.directory));
        let metadata_path = staging_root.join(format!("{}.package.json", prepared.directory));
        let metadata = StagedPackageDownload {
            version: 1,
            directory: prepared.directory.clone(),
            size_bytes: prepared.size_bytes,
            sha256: prepared.sha256.clone(),
            files: prepared
                .files
                .iter()
                .map(|file| StagedPackageFile {
                    url: file.url.as_str().to_string(),
                    filename: file.filename.clone(),
                    size_bytes: file.size_bytes,
                    sha256: file.sha256_hex.clone(),
                })
                .collect(),
            license: prepared.license.clone(),
            model_index_id: prepared.model_index_id.clone(),
            replacement: prepared.replacement.clone(),
        };
        {
            let _storage = self.storage.serial().await;
            prepare_package_staging(&package_root, &metadata_path, &metadata)
                .await
                .map_err(DownloadFailure::resumable)?;
        }

        let staged_bytes = match package_staged_bytes(&package_root, &prepared.files).await {
            Ok(bytes) => bytes,
            Err(error) => {
                cleanup_package_staging(&package_root, &metadata_path).await;
                return Err(DownloadFailure::terminal(error));
            }
        };
        {
            let mut storage = self.storage.serial().await;
            storage
                .reserve_download(
                    &prepared.directory,
                    Some(prepared.size_bytes),
                    staged_bytes,
                    staged_bytes,
                )
                .await
                .map_err(DownloadFailure::resumable)?;
        }
        self.update_progress(
            ModelDownloadPhase::Downloading,
            staged_bytes,
            Some(prepared.size_bytes),
            staged_bytes > 0,
        )
        .await;

        let mut completed_bytes = 0_u64;
        for file in &prepared.files {
            if cancellation.is_cancelled() {
                return Err(DownloadFailure::resumable(
                    "model package download cancelled",
                ));
            }
            let final_path = safe_package_file_path(&package_root, &file.filename)
                .await
                .map_err(DownloadFailure::terminal)?;
            let extension = final_path
                .extension()
                .and_then(|value| value.to_str())
                .ok_or_else(|| DownloadFailure::terminal("model package file has no extension"))?;
            let part_path = final_path.with_extension(format!("{extension}.part"));
            reject_symlink(&final_path)
                .await
                .map_err(DownloadFailure::terminal)?;
            reject_symlink(&part_path)
                .await
                .map_err(DownloadFailure::terminal)?;

            if file_size(&final_path)
                .await
                .map_err(DownloadFailure::terminal)?
                > 0
            {
                let metadata = fs::metadata(&final_path)
                    .await
                    .map_err(DownloadFailure::terminal)?;
                let hash = sha256_file(&final_path, cancellation)
                    .await
                    .map_err(DownloadFailure::resumable)?;
                if metadata.len() != file.size_bytes || hash != file.sha256 {
                    cleanup_package_staging(&package_root, &metadata_path).await;
                    return Err(DownloadFailure::terminal(format!(
                        "A verified staged file changed for '{}'.",
                        file.filename
                    )));
                }
                completed_bytes = completed_bytes.saturating_add(file.size_bytes);
                continue;
            }

            let mut offset = file_size(&part_path)
                .await
                .map_err(DownloadFailure::resumable)?;
            if offset > file.size_bytes {
                cleanup_package_staging(&package_root, &metadata_path).await;
                return Err(DownloadFailure::terminal(format!(
                    "Staged file '{}' exceeds its signed size.",
                    file.filename
                )));
            }
            let mut request = self.client.get(file.url.clone());
            if offset > 0 {
                request = request.header(RANGE, format!("bytes={offset}-"));
            }
            let response = tokio::select! {
                _ = cancellation.cancelled() => return Err(DownloadFailure::resumable("model package download cancelled")),
                response = request.send() => response.map_err(|error| DownloadFailure::resumable(format!("model package request failed: {}", error.without_url())))?,
            };
            let response_status = response.status();
            let already_complete = offset > 0
                && response_status == StatusCode::RANGE_NOT_SATISFIABLE
                && parse_unsatisfied_content_range(response.headers().get(CONTENT_RANGE))
                    .is_ok_and(|total| total == offset);
            let total_bytes = if already_complete {
                Some(offset)
            } else if offset > 0 && response_status == StatusCode::PARTIAL_CONTENT {
                parse_content_range(response.headers().get(CONTENT_RANGE), offset)
                    .map_err(DownloadFailure::resumable)?
            } else if response_status == StatusCode::OK {
                if offset > 0 {
                    offset = 0;
                }
                response.content_length()
            } else {
                return Err(DownloadFailure::resumable(format!(
                    "model package download returned HTTP {response_status}"
                )));
            };
            if total_bytes.is_some_and(|total| total != file.size_bytes) {
                cleanup_package_staging(&package_root, &metadata_path).await;
                return Err(DownloadFailure::terminal(format!(
                    "File '{}' does not match its signed size.",
                    file.filename
                )));
            }

            let mut options = OpenOptions::new();
            options.create(true).write(true);
            if offset == 0 {
                options.truncate(true);
            } else {
                options.append(true);
            }
            let mut output = options.open(&part_path).await.map_err(|error| {
                DownloadFailure::resumable(format!(
                    "failed to open model package staging file: {error}"
                ))
            })?;
            let mut downloaded = offset;
            let mut last_reported = offset;
            let mut last_update = Instant::now();
            if !already_complete {
                let mut stream = response.bytes_stream();
                loop {
                    let next = tokio::select! {
                        _ = cancellation.cancelled() => return Err(DownloadFailure::resumable("model package download cancelled")),
                        next = stream.next() => next,
                    };
                    let Some(chunk) = next else {
                        break;
                    };
                    let chunk = chunk.map_err(|error| {
                        DownloadFailure::resumable(format!(
                            "model package stream failed: {}",
                            error.without_url()
                        ))
                    })?;
                    downloaded = downloaded.saturating_add(chunk.len() as u64);
                    if downloaded > file.size_bytes {
                        drop(output);
                        cleanup_package_staging(&package_root, &metadata_path).await;
                        return Err(DownloadFailure::terminal(format!(
                            "File '{}' exceeded its signed size.",
                            file.filename
                        )));
                    }
                    output.write_all(&chunk).await.map_err(|error| {
                        DownloadFailure::resumable(format!(
                            "failed to write model package staging file: {error}"
                        ))
                    })?;
                    self.storage
                        .serial()
                        .await
                        .consume_download_reservation(&prepared.directory, chunk.len() as u64);
                    if downloaded.saturating_sub(last_reported) >= STATUS_UPDATE_BYTES
                        || last_update.elapsed() >= STATUS_UPDATE_INTERVAL
                    {
                        self.update_progress(
                            ModelDownloadPhase::Downloading,
                            completed_bytes.saturating_add(downloaded),
                            Some(prepared.size_bytes),
                            true,
                        )
                        .await;
                        last_reported = downloaded;
                        last_update = Instant::now();
                    }
                }
            }
            output.sync_all().await.map_err(|error| {
                DownloadFailure::resumable(format!(
                    "failed to sync model package staging file: {error}"
                ))
            })?;
            drop(output);
            if downloaded != file.size_bytes {
                return Err(DownloadFailure::resumable(format!(
                    "File '{}' ended at {downloaded} bytes; expected {} bytes.",
                    file.filename, file.size_bytes
                )));
            }
            self.update_progress(
                ModelDownloadPhase::Verifying,
                completed_bytes.saturating_add(downloaded),
                Some(prepared.size_bytes),
                false,
            )
            .await;
            let actual_hash = sha256_file(&part_path, cancellation)
                .await
                .map_err(DownloadFailure::resumable)?;
            if actual_hash != file.sha256 {
                cleanup_package_staging(&package_root, &metadata_path).await;
                return Err(DownloadFailure::terminal(format!(
                    "SHA-256 verification failed for package file '{}'.",
                    file.filename
                )));
            }
            fs::rename(&part_path, &final_path).await.map_err(|error| {
                DownloadFailure::terminal(format!(
                    "failed to commit verified package file '{}': {error}",
                    file.filename
                ))
            })?;
            completed_bytes = completed_bytes.saturating_add(file.size_bytes);
            self.update_progress(
                ModelDownloadPhase::Downloading,
                completed_bytes,
                Some(prepared.size_bytes),
                true,
            )
            .await;
        }

        validate_complete_package_tree(&package_root, &prepared.files)
            .await
            .map_err(DownloadFailure::terminal)?;
        if let Err(error) = validate_package_runtime_layout(&package_root).await {
            cleanup_package_staging(&package_root, &metadata_path).await;
            return Err(DownloadFailure::terminal(format!(
                "Model package runtime layout is invalid: {error}"
            )));
        }
        sync_package_tree_for_install(&package_root, &prepared.files)
            .await
            .map_err(|error| {
                DownloadFailure::resumable(format!(
                    "failed to durably sync the verified model package: {error}"
                ))
            })?;
        if cancellation.is_cancelled() {
            return Err(DownloadFailure::resumable(
                "model package download cancelled",
            ));
        }
        self.update_progress(
            ModelDownloadPhase::Verifying,
            prepared.size_bytes,
            Some(prepared.size_bytes),
            false,
        )
        .await;

        let destination = self.models_root.join(&prepared.directory);
        let package_files = prepared
            .files
            .iter()
            .map(|file| ModelPackageFile {
                filename: file.filename.clone(),
                size_bytes: file.size_bytes,
                sha256: file.sha256_hex.clone(),
            })
            .collect::<Vec<_>>();
        if let Some(replacement) = prepared.replacement.clone() {
            let _storage = self.storage.serial().await;
            commit_model_upgrade(
                &self.models_root,
                &package_root,
                replacement,
                ModelUpgradeTarget {
                    catalog_id: prepared.directory.clone(),
                    size_bytes: prepared.size_bytes,
                    sha256: prepared.sha256.clone(),
                    source_url: prepared.source_url.clone(),
                    source_host: Some(prepared.source_host.clone()),
                    license: prepared.license.clone(),
                    model_index_id: prepared.model_index_id.clone(),
                    files: package_files,
                },
            )
            .await
            .map_err(|error| {
                DownloadFailure::resumable(format!(
                    "failed to commit verified model package upgrade: {error}"
                ))
            })?;
            cleanup_package_metadata(&metadata_path).await;
            return Ok(());
        }
        let _storage = self.storage.serial().await;
        if fs::symlink_metadata(&destination).await.is_ok() {
            return Err(DownloadFailure::terminal(format!(
                "A catalog entry named '{}' appeared during installation.",
                prepared.directory
            )));
        }
        write_package_provenance(
            &self.models_root,
            ModelPackageProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(prepared.model_index_id.clone()),
                directory: prepared.directory.clone(),
                size_bytes: prepared.size_bytes,
                source_url: prepared.source_url.clone(),
                source_host: Some(prepared.source_host.clone()),
                sha256: prepared.sha256.clone(),
                license: prepared.license.clone(),
                files: package_files,
            },
        )
        .await
        .map_err(|error| {
            DownloadFailure::terminal(format!(
                "failed to persist verified model package provenance: {error}"
            ))
        })?;
        if let Err(error) = atomic_install_directory(&package_root, &destination).await {
            if let Err(remove_error) = remove_provenance(&self.models_root, &prepared.directory) {
                tracing::error!(%remove_error, "Failed to roll back model package provenance");
            }
            return Err(DownloadFailure::terminal(format!(
                "failed to atomically install verified model package: {error}"
            )));
        }
        cleanup_package_metadata(&metadata_path).await;
        Ok(())
    }

    async fn execute(
        &self,
        prepared: &PreparedDownload,
        cancellation: &CancellationToken,
    ) -> std::result::Result<(), DownloadFailure> {
        let staging_root = self.staging_root();
        let part_path = staging_root.join(format!("{}.part", prepared.filename));
        let metadata_path = staging_root.join(format!("{}.json", prepared.filename));
        let metadata = StagedDownload {
            version: 1,
            url: prepared.url.as_str().to_string(),
            filename: prepared.filename.clone(),
            sha256: prepared.sha256_hex.clone(),
            license: prepared.license.clone(),
            expected_size_bytes: prepared.expected_size_bytes,
            model_index_id: prepared.model_index_id.clone(),
            replacement: prepared.replacement.clone(),
        };
        {
            let _storage = self.storage.serial().await;
            self.prepare_staging(&part_path, &metadata_path, &metadata)
                .await
                .map_err(DownloadFailure::resumable)?;
        }

        let mut offset = file_size(&part_path)
            .await
            .map_err(DownloadFailure::resumable)?;
        let replaced_bytes = offset;
        if offset > self.max_bytes {
            let _storage = self.storage.serial().await;
            cleanup_staging(&part_path, &metadata_path).await;
            return Err(DownloadFailure::terminal(format!(
                "Staged download exceeds the configured {} byte limit.",
                self.max_bytes
            )));
        }

        self.update_progress(ModelDownloadPhase::Downloading, offset, None, offset > 0)
            .await;
        let mut request = self.client.get(prepared.url.clone());
        if offset > 0 {
            request = request.header(RANGE, format!("bytes={offset}-"));
        }
        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err(DownloadFailure::resumable("model download cancelled")),
            response = request.send() => response.map_err(|error| DownloadFailure::resumable(format!("model download request failed: {}", error.without_url())))?,
        };

        let response_status = response.status();
        let already_complete = offset > 0
            && response_status == StatusCode::RANGE_NOT_SATISFIABLE
            && parse_unsatisfied_content_range(response.headers().get(CONTENT_RANGE))
                .is_ok_and(|total| total == offset);
        let total_bytes = if already_complete {
            Some(offset)
        } else if offset > 0 && response_status == StatusCode::PARTIAL_CONTENT {
            parse_content_range(response.headers().get(CONTENT_RANGE), offset)
                .map_err(DownloadFailure::resumable)?
        } else if response_status == StatusCode::OK {
            if offset > 0 {
                offset = 0;
            }
            response.content_length()
        } else {
            return Err(DownloadFailure::resumable(format!(
                "model download returned HTTP {response_status}"
            )));
        };
        if let (Some(expected), Some(total)) = (prepared.expected_size_bytes, total_bytes) {
            if total != expected {
                let _storage = self.storage.serial().await;
                cleanup_staging(&part_path, &metadata_path).await;
                return Err(DownloadFailure::terminal(format!(
                    "Model download declared {total} bytes; the signed index requires {expected} bytes."
                )));
            }
        }
        if total_bytes.is_some_and(|total| total > self.max_bytes) {
            let _storage = self.storage.serial().await;
            cleanup_staging(&part_path, &metadata_path).await;
            return Err(DownloadFailure::terminal(format!(
                "Model download exceeds the configured {} byte limit.",
                self.max_bytes
            )));
        }

        let mut options = OpenOptions::new();
        options.create(true).write(true);
        if offset == 0 {
            options.truncate(true);
        } else {
            options.append(true);
        }
        let mut storage = self.storage.serial().await;
        storage
            .reserve_download(&prepared.filename, total_bytes, replaced_bytes, offset)
            .await
            .map_err(DownloadFailure::resumable)?;
        let mut file = options.open(&part_path).await.map_err(|error| {
            DownloadFailure::resumable(format!(
                "failed to open model download staging file: {error}"
            ))
        })?;
        drop(storage);
        let mut downloaded = offset;
        let mut last_reported = offset;
        let mut last_update = Instant::now();
        if !already_complete {
            let mut stream = response.bytes_stream();
            loop {
                let next = tokio::select! {
                    _ = cancellation.cancelled() => return Err(DownloadFailure::resumable("model download cancelled")),
                    next = stream.next() => next,
                };
                let Some(chunk) = next else {
                    break;
                };
                let chunk = chunk.map_err(|error| {
                    DownloadFailure::resumable(format!(
                        "model download stream failed: {}",
                        error.without_url()
                    ))
                })?;
                downloaded = downloaded.saturating_add(chunk.len() as u64);
                if downloaded > self.max_bytes
                    || total_bytes.is_some_and(|total| downloaded > total)
                {
                    drop(file);
                    let _storage = self.storage.serial().await;
                    cleanup_staging(&part_path, &metadata_path).await;
                    return Err(DownloadFailure::terminal(
                        "Model download exceeded its declared or configured size limit.",
                    ));
                }
                let mut storage = self.storage.serial().await;
                if total_bytes.is_none() {
                    storage
                        .ensure_capacity(chunk.len() as u64, 0)
                        .await
                        .map_err(DownloadFailure::resumable)?;
                }
                file.write_all(&chunk).await.map_err(|error| {
                    DownloadFailure::resumable(format!("failed to write model download: {error}"))
                })?;
                storage.consume_download_reservation(&prepared.filename, chunk.len() as u64);
                drop(storage);
                if downloaded.saturating_sub(last_reported) >= STATUS_UPDATE_BYTES
                    || last_update.elapsed() >= STATUS_UPDATE_INTERVAL
                {
                    self.update_progress(
                        ModelDownloadPhase::Downloading,
                        downloaded,
                        total_bytes,
                        true,
                    )
                    .await;
                    last_reported = downloaded;
                    last_update = Instant::now();
                }
            }
        }
        file.sync_all().await.map_err(|error| {
            DownloadFailure::resumable(format!("failed to sync model download: {error}"))
        })?;
        drop(file);

        if let Some(total) = total_bytes {
            if downloaded != total {
                return Err(DownloadFailure::resumable(format!(
                    "Model download ended at {downloaded} bytes; expected {total} bytes."
                )));
            }
        }
        if let Some(expected) = prepared.expected_size_bytes {
            if downloaded != expected {
                let _storage = self.storage.serial().await;
                cleanup_staging(&part_path, &metadata_path).await;
                return Err(DownloadFailure::terminal(format!(
                    "Model download ended at {downloaded} bytes; the signed index requires {expected} bytes."
                )));
            }
        }
        self.update_progress(
            ModelDownloadPhase::Verifying,
            downloaded,
            Some(downloaded),
            false,
        )
        .await;
        let actual_hash = sha256_file(&part_path, cancellation)
            .await
            .map_err(DownloadFailure::resumable)?;
        if actual_hash != prepared.sha256 {
            let _storage = self.storage.serial().await;
            cleanup_staging(&part_path, &metadata_path).await;
            return Err(DownloadFailure::terminal(format!(
                "SHA-256 verification failed for '{}'.",
                prepared.filename
            )));
        }
        if cancellation.is_cancelled() {
            return Err(DownloadFailure::resumable("model download cancelled"));
        }

        let destination = self.models_root.join(&prepared.filename);
        let _storage = self.storage.serial().await;
        if let Some(replacement) = prepared.replacement.clone() {
            commit_model_upgrade(
                &self.models_root,
                &part_path,
                replacement,
                ModelUpgradeTarget {
                    catalog_id: prepared.filename.clone(),
                    size_bytes: downloaded,
                    sha256: prepared.sha256_hex.clone(),
                    source_url: prepared.source_url.clone(),
                    source_host: Some(prepared.source_host.clone()),
                    license: prepared.license.clone(),
                    model_index_id: prepared.model_index_id.clone().ok_or_else(|| {
                        DownloadFailure::terminal(
                            "signed-model upgrade has no persistent index identity",
                        )
                    })?,
                    files: Vec::new(),
                },
            )
            .await
            .map_err(|error| {
                DownloadFailure::resumable(format!(
                    "failed to commit verified model upgrade: {error}"
                ))
            })?;
            cleanup_staging(&part_path, &metadata_path).await;
            return Ok(());
        }
        fs::hard_link(&part_path, &destination)
            .await
            .map_err(|error| {
                DownloadFailure::terminal(format!(
                "failed to install verified model without overwriting an existing entry: {error}"
                ))
            })?;
        if let Err(error) = write_provenance(
            &self.models_root,
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: prepared.model_index_id.clone(),
                filename: prepared.filename.clone(),
                size_bytes: downloaded,
                source_url: prepared.source_url.clone(),
                source_host: Some(prepared.source_host.clone()),
                sha256: prepared.sha256_hex.clone(),
                license: prepared.license.clone(),
            },
        )
        .await
        {
            if let Err(remove_error) = fs::remove_file(&destination).await {
                tracing::error!(%remove_error, "Failed to roll back model after provenance error");
            }
            return Err(DownloadFailure::terminal(format!(
                "failed to persist verified model provenance: {error}"
            )));
        }
        cleanup_staging(&part_path, &metadata_path).await;
        Ok(())
    }

    async fn prepare_staging(
        &self,
        part_path: &Path,
        metadata_path: &Path,
        expected: &StagedDownload,
    ) -> Result<()> {
        reject_symlink(part_path).await?;
        reject_symlink(metadata_path).await?;
        reject_symlink(&metadata_path.with_extension("json.tmp")).await?;
        let existing = match fs::read(metadata_path).await {
            Ok(bytes) => serde_json::from_slice::<StagedDownload>(&bytes).ok(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("failed to read model download metadata"),
        };
        if existing
            .as_ref()
            .is_none_or(|value| !staged_content_identity_matches(value, expected))
        {
            cleanup_staging(part_path, metadata_path).await;
            let metadata = serde_json::to_vec(expected)?;
            let temporary_path = metadata_path.with_extension("json.tmp");
            fs::write(&temporary_path, metadata)
                .await
                .context("failed to write model download metadata")?;
            fs::rename(&temporary_path, metadata_path)
                .await
                .context("failed to commit model download metadata")?;
        }
        Ok(())
    }

    async fn update_progress(
        &self,
        phase: ModelDownloadPhase,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
        resumable: bool,
    ) {
        let mut status = self.status.write().await;
        status.phase = phase;
        status.downloaded_bytes = downloaded_bytes;
        status.total_bytes = total_bytes;
        status.resumable = resumable;
        status.error = None;
    }
}

fn staged_content_identity_matches(left: &StagedDownload, right: &StagedDownload) -> bool {
    left.version == right.version
        && left.url == right.url
        && left.filename == right.filename
        && left.sha256 == right.sha256
        && left.expected_size_bytes == right.expected_size_bytes
        && left.model_index_id == right.model_index_id
        && left.replacement == right.replacement
}

fn validate_replacement_identity(
    replacement: Option<&ModelUpgradeSource>,
    target_id: &str,
    model_index_id: Option<&str>,
) -> Result<()> {
    let Some(replacement) = replacement else {
        return Ok(());
    };
    validate_catalog_id_for_upgrade(&replacement.catalog_id)?;
    validate_catalog_id_for_upgrade(target_id)?;
    if replacement.size_bytes == 0
        || replacement.sha256.len() != 64
        || !replacement
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || Some(replacement.model_index_id.as_str()) != model_index_id
    {
        return Err(anyhow!(
            "signed-model upgrade replacement identity is invalid"
        ));
    }
    Ok(())
}

fn validate_catalog_id_for_upgrade(value: &str) -> Result<()> {
    super::model_manager::validate_catalog_id(value)?;
    if value.starts_with('.') {
        Err(anyhow!(
            "signed-model upgrade catalog IDs must not be hidden"
        ))
    } else {
        Ok(())
    }
}

fn prepare_source_url(input: &str, policy: &DownloadUrlPolicy) -> Result<(Url, String)> {
    let input = input.trim();
    if input.is_empty() || input.len() > MAX_SOURCE_URL_BYTES {
        return Err(anyhow!(
            "model source URL must be between 1 and {MAX_SOURCE_URL_BYTES} bytes"
        ));
    }
    let mut url = Url::parse(input).context("model source URL is invalid")?;
    validate_download_url(&url, policy)?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("model source URL must include a host"))?
        .to_ascii_lowercase();
    if !is_hugging_face_repository_host(&host)
        && !(policy.allow_loopback && is_loopback_host(&host))
    {
        return Err(anyhow!(
            "model source inspection requires a huggingface.co repository URL"
        ));
    }

    let segments = url
        .path_segments()
        .ok_or_else(|| anyhow!("model source URL must include a repository file path"))?
        .collect::<Vec<_>>();
    if segments.len() < 5
        || segments[0].is_empty()
        || segments[1].is_empty()
        || !matches!(segments[2], "blob" | "resolve")
        || segments[3].is_empty()
    {
        return Err(anyhow!(
            "model source URL must use /owner/repository/resolve/revision/file or /blob/"
        ));
    }
    let filename = segments
        .last()
        .copied()
        .filter(|value| !value.is_empty() && !value.contains('%'))
        .ok_or_else(|| anyhow!("model source URL must end with an unambiguous model filename"))?
        .to_string();
    validate_model_filename(&filename)?;

    if segments[2] == "blob" {
        let path = url.path().replacen("/blob/", "/resolve/", 1);
        url.set_path(&path);
    }
    // Download URLs are public. Dropping query and fragment data keeps tokens and
    // other incidental browser state out of requests, responses, and provenance.
    url.set_query(None);
    url.set_fragment(None);
    Ok((url, filename))
}

fn is_hugging_face_repository_host(host: &str) -> bool {
    matches!(host, "huggingface.co" | "www.huggingface.co")
}

fn normalized_metadata_tag(value: &reqwest::header::HeaderValue) -> Option<String> {
    let value = value.to_str().ok()?.trim();
    let value = value.strip_prefix("W/").unwrap_or(value).trim();
    Some(value.trim_matches('"').to_ascii_lowercase())
}

fn metadata_sha256(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(LINKED_ETAG_HEADER)
        .or_else(|| headers.get(ETAG))
        .and_then(normalized_metadata_tag)
        .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn metadata_commit(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(REPO_COMMIT_HEADER)
        .and_then(normalized_metadata_tag)
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn metadata_size(
    headers: &reqwest::header::HeaderMap,
    status: StatusCode,
) -> std::result::Result<Option<u64>, &'static str> {
    let value = headers.get(LINKED_SIZE_HEADER).or_else(|| {
        status
            .is_success()
            .then(|| headers.get(CONTENT_LENGTH))
            .flatten()
    });
    let Some(value) = value else {
        return Ok(None);
    };
    let size = value
        .to_str()
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or("The source returned an invalid file size.")?;
    if size == 0 {
        return Err("The source returned an empty file size.");
    }
    Ok(Some(size))
}

fn pin_source_revision(url: &Url, commit: &str) -> Result<Url> {
    let mut pinned = url.clone();
    let segments = pinned
        .path_segments()
        .ok_or_else(|| anyhow!("model source URL must include a repository file path"))?
        .collect::<Vec<_>>();
    if segments.len() < 5 || segments[2] != "resolve" {
        return Err(anyhow!("model source URL cannot be pinned to a commit"));
    }
    let path = format!(
        "/{}/{}/resolve/{}/{}",
        segments[0],
        segments[1],
        commit,
        segments[4..].join("/")
    );
    pinned.set_path(&path);
    pinned.set_query(None);
    pinned.set_fragment(None);
    Ok(pinned)
}

fn prepare_download(
    request: ModelDownloadRequest,
    policy: &DownloadUrlPolicy,
    license_policy: &ModelLicensePolicy,
) -> Result<PreparedDownload> {
    validate_model_filename(&request.filename)?;
    let url = Url::parse(request.url.trim()).context("model download URL is invalid")?;
    validate_download_url(&url, policy)?;
    let sha256_hex = request.sha256.trim().to_ascii_lowercase();
    let sha256 = parse_sha256(&sha256_hex)?;
    let license = license_policy.enforce(request.license)?;
    let source_host = url
        .host_str()
        .ok_or_else(|| anyhow!("model download URL must include a host"))?
        .to_string();
    let source_url = (url.scheme() == "https").then(|| sanitized_download_source(&url));
    Ok(PreparedDownload {
        url,
        filename: request.filename.trim().to_string(),
        sha256,
        sha256_hex,
        source_host,
        source_url,
        license,
        expected_size_bytes: request.expected_size_bytes,
        model_index_id: super::model_provenance::normalize_model_index_id(request.model_index_id)?,
        replacement: None,
    })
}

fn prepare_package_download(
    request: ModelPackageDownloadRequest,
    policy: &DownloadUrlPolicy,
    license_policy: &ModelLicensePolicy,
) -> Result<PreparedPackageDownload> {
    validate_package_id(&request.directory)?;
    let model_index_id =
        super::model_provenance::normalize_model_index_id(Some(request.model_index_id))?
            .ok_or_else(|| anyhow!("model package downloads require a signed model index ID"))?;
    let license = license_policy.enforce(request.license)?;
    let descriptors = normalize_package_files(
        request
            .files
            .iter()
            .map(|file| ModelPackageFile {
                filename: file.filename.clone(),
                size_bytes: file.size_bytes,
                sha256: file.sha256.clone(),
            })
            .collect(),
        request.size_bytes,
    )?;
    let digest = package_digest(&descriptors)?;
    if !digest.eq_ignore_ascii_case(request.sha256.trim()) {
        return Err(anyhow!(
            "model package digest does not match its signed file manifest"
        ));
    }

    let mut request_files = BTreeMap::new();
    for file in request.files {
        if request_files.insert(file.filename.clone(), file).is_some() {
            return Err(anyhow!("model package filenames must be unique"));
        }
    }
    let mut prepared_files = Vec::with_capacity(descriptors.len());
    let mut source_host = None;
    let mut source_identity = None;
    let mut source_url = None;
    for descriptor in descriptors {
        let source = request_files
            .remove(&descriptor.filename)
            .ok_or_else(|| anyhow!("model package file metadata is incomplete"))?;
        let url = Url::parse(source.url.trim()).context("model package download URL is invalid")?;
        validate_download_url(&url, policy)?;
        let host = url
            .host_str()
            .ok_or_else(|| anyhow!("model package download URL must include a host"))?
            .to_ascii_lowercase();
        if source_host
            .as_ref()
            .is_some_and(|expected| expected != &host)
        {
            return Err(anyhow!(
                "all model package files must use the same source host"
            ));
        }
        source_host.get_or_insert(host);
        let identity = package_source_identity(&url, policy)?;
        if source_identity
            .as_ref()
            .is_some_and(|expected| expected != &identity)
        {
            return Err(anyhow!(
                "all model package files must use the same repository commit"
            ));
        }
        source_identity.get_or_insert(identity);
        source_url.get_or_insert_with(|| {
            (url.scheme() == "https").then(|| sanitized_download_source(&url))
        });
        prepared_files.push(PreparedPackageFile {
            url,
            filename: descriptor.filename,
            size_bytes: descriptor.size_bytes,
            sha256: decode_package_sha256(&descriptor.sha256)?,
            sha256_hex: descriptor.sha256,
        });
    }
    if !request_files.is_empty() {
        return Err(anyhow!("model package contains unexpected file metadata"));
    }
    Ok(PreparedPackageDownload {
        directory: request.directory,
        size_bytes: request.size_bytes,
        sha256: digest,
        files: prepared_files,
        source_host: source_host.ok_or_else(|| anyhow!("model package has no source host"))?,
        source_url: source_url.flatten(),
        license,
        model_index_id,
        replacement: None,
    })
}

fn package_source_identity(url: &Url, policy: &DownloadUrlPolicy) -> Result<String> {
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("model package download URL must include a host"))?
        .to_ascii_lowercase();
    if policy.allow_loopback && is_loopback_host(&host) {
        return Ok(format!("{}://{}", url.scheme(), host));
    }
    if !is_hugging_face_repository_host(&host) || url.query().is_some() || url.fragment().is_some()
    {
        return Err(anyhow!(
            "model package sources require immutable public Hugging Face URLs"
        ));
    }
    let segments = url
        .path_segments()
        .ok_or_else(|| anyhow!("model package URL has no file path"))?
        .collect::<Vec<_>>();
    if segments.len() < 5
        || segments[2] != "resolve"
        || !matches!(segments[3].len(), 40 | 64)
        || !segments[3].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(anyhow!(
            "model package sources must identify one immutable repository commit"
        ));
    }
    Ok(format!(
        "{host}/{}/{}/{}",
        segments[0], segments[1], segments[3]
    ))
}

async fn inspect_staged_package(
    staging_root: &Path,
    metadata_path: &Path,
    metadata: StagedPackageDownload,
    policy: &DownloadUrlPolicy,
    max_bytes: u64,
) -> Result<StagedModelDownload> {
    validate_package_id(&metadata.directory)?;
    validate_replacement_identity(
        metadata.replacement.as_ref(),
        &metadata.directory,
        Some(&metadata.model_index_id),
    )?;
    if metadata.version != 1
        || metadata.size_bytes == 0
        || metadata.size_bytes > max_bytes
        || metadata_path.file_name().and_then(|value| value.to_str())
            != Some(format!("{}.package.json", metadata.directory).as_str())
    {
        return Err(anyhow!("staged model package metadata is invalid"));
    }
    let descriptors = normalize_package_files(
        metadata
            .files
            .iter()
            .map(|file| ModelPackageFile {
                filename: file.filename.clone(),
                size_bytes: file.size_bytes,
                sha256: file.sha256.clone(),
            })
            .collect(),
        metadata.size_bytes,
    )?;
    if package_digest(&descriptors)? != metadata.sha256 {
        return Err(anyhow!("staged model package digest is invalid"));
    }
    let by_name = metadata
        .files
        .iter()
        .map(|file| (file.filename.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    let mut source_host = None;
    let mut identity = None;
    let mut prepared = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let staged = by_name
            .get(descriptor.filename.as_str())
            .ok_or_else(|| anyhow!("staged model package file metadata is incomplete"))?;
        let url = Url::parse(&staged.url)?;
        validate_download_url(&url, policy)?;
        let host = url
            .host_str()
            .ok_or_else(|| anyhow!("staged model package URL has no host"))?
            .to_ascii_lowercase();
        if source_host
            .as_ref()
            .is_some_and(|expected| expected != &host)
        {
            return Err(anyhow!("staged model package source hosts differ"));
        }
        source_host.get_or_insert(host);
        let current_identity = package_source_identity(&url, policy)?;
        if identity
            .as_ref()
            .is_some_and(|expected| expected != &current_identity)
        {
            return Err(anyhow!("staged model package repository commits differ"));
        }
        identity.get_or_insert(current_identity);
        prepared.push(PreparedPackageFile {
            url,
            filename: descriptor.filename,
            size_bytes: descriptor.size_bytes,
            sha256: decode_package_sha256(&descriptor.sha256)?,
            sha256_hex: descriptor.sha256,
        });
    }
    let package_root = staging_root.join(format!("{}.package", metadata.directory));
    let downloaded_bytes = package_staged_bytes(&package_root, &prepared).await?;
    let modified_at = fs::metadata(&package_root)
        .await
        .ok()
        .and_then(|value| value.modified().ok())
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs());
    Ok(StagedModelDownload {
        filename: metadata.directory,
        source_host: source_host.ok_or_else(|| anyhow!("staged model package has no host"))?,
        downloaded_bytes,
        modified_at,
    })
}

fn validate_download_url(url: &Url, policy: &DownloadUrlPolicy) -> Result<()> {
    let valid_scheme = url.scheme() == "https" || (policy.allow_http && url.scheme() == "http");
    if !valid_scheme {
        return Err(anyhow!("model downloads require HTTPS"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(anyhow!("model download URLs must not contain credentials"));
    }
    if url.scheme() == "https" && url.port().is_some_and(|port| port != 443) {
        return Err(anyhow!(
            "model download URLs must use the default HTTPS port"
        ));
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("model download URL must include a host"))?
        .to_ascii_lowercase();
    if !(is_trusted_model_host(&host) || (policy.allow_loopback && is_loopback_host(&host))) {
        return Err(anyhow!(
            "model download host is not trusted; use huggingface.co or an hf.co CDN host"
        ));
    }
    Ok(())
}

fn is_trusted_model_host(host: &str) -> bool {
    host == "huggingface.co"
        || host.ends_with(".huggingface.co")
        || host == "hf.co"
        || host.ends_with(".hf.co")
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "::1" | "localhost")
}

fn parse_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(anyhow!(
            "SHA-256 must contain exactly 64 hexadecimal characters"
        ));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let pair = std::str::from_utf8(pair)?;
        output[index] = u8::from_str_radix(pair, 16)?;
    }
    Ok(output)
}

fn parse_content_range(
    value: Option<&reqwest::header::HeaderValue>,
    expected_start: u64,
) -> Result<Option<u64>> {
    let value = value
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| anyhow!("resumed download did not include a valid Content-Range header"))?;
    let value = value
        .strip_prefix("bytes ")
        .ok_or_else(|| anyhow!("resumed download returned an invalid Content-Range unit"))?;
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| anyhow!("resumed download returned an invalid Content-Range"))?;
    let (start, _) = range
        .split_once('-')
        .ok_or_else(|| anyhow!("resumed download returned an invalid byte range"))?;
    if start.parse::<u64>()? != expected_start {
        return Err(anyhow!(
            "resumed download returned an unexpected byte offset"
        ));
    }
    if total == "*" {
        Ok(None)
    } else {
        Ok(Some(total.parse::<u64>()?))
    }
}

fn parse_unsatisfied_content_range(value: Option<&reqwest::header::HeaderValue>) -> Result<u64> {
    let value = value
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes */"))
        .ok_or_else(|| anyhow!("download returned an invalid unsatisfied Content-Range"))?;
    Ok(value.parse::<u64>()?)
}

async fn file_size(path: &Path) -> Result<u64> {
    match fs::metadata(path).await {
        Ok(metadata) => Ok(metadata.len()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(error).context("failed to inspect model download staging file"),
    }
}

async fn sha256_file(path: &Path, cancellation: &CancellationToken) -> Result<[u8; 32]> {
    let mut file = fs::File::open(path)
        .await
        .context("failed to open model download for verification")?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = tokio::select! {
            _ = cancellation.cancelled() => return Err(anyhow!("model download cancelled")),
            result = file.read(&mut buffer) => result.context("failed to read model download for verification")?,
        };
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

async fn cleanup_staging(part_path: &Path, metadata_path: &Path) {
    for path in [
        part_path,
        metadata_path,
        &metadata_path.with_extension("json.tmp"),
    ] {
        if let Err(error) = fs::remove_file(path).await {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %path.display(), %error, "Failed to clean model download staging file");
            }
        }
    }
}

async fn prepare_package_staging(
    package_root: &Path,
    metadata_path: &Path,
    expected: &StagedPackageDownload,
) -> Result<()> {
    reject_symlink(metadata_path).await?;
    let temporary_path = metadata_path.with_extension("json.tmp");
    reject_symlink(&temporary_path).await?;
    let existing = match fs::read(metadata_path).await {
        Ok(bytes) => serde_json::from_slice::<StagedPackageDownload>(&bytes).ok(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("failed to read model package metadata"),
    };
    if existing.as_ref() != Some(expected) {
        cleanup_package_staging(package_root, metadata_path).await;
        fs::create_dir_all(package_root)
            .await
            .context("failed to create model package staging directory")?;
        secure_package_directory(package_root).await?;
        let bytes = serde_json::to_vec(expected)?;
        fs::write(&temporary_path, bytes)
            .await
            .context("failed to write model package metadata")?;
        fs::rename(&temporary_path, metadata_path)
            .await
            .context("failed to commit model package metadata")?;
    } else {
        match fs::symlink_metadata(package_root).await {
            Ok(_) => secure_package_directory(package_root).await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(package_root)
                    .await
                    .context("failed to recreate model package staging directory")?;
                secure_package_directory(package_root).await?;
            }
            Err(error) => {
                return Err(error).context("failed to inspect model package staging directory")
            }
        }
    }
    Ok(())
}

async fn safe_package_file_path(package_root: &Path, relative: &str) -> Result<PathBuf> {
    validate_package_path(relative)?;
    secure_package_directory(package_root).await?;
    let relative_path = Path::new(relative);
    let mut current = package_root.to_path_buf();
    if let Some(parent) = relative_path.parent() {
        for component in parent.components() {
            let std::path::Component::Normal(name) = component else {
                return Err(anyhow!("model package path is not relative"));
            };
            current.push(name);
            match fs::symlink_metadata(&current).await {
                Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                    return Err(anyhow!(
                        "model package staging parent must be a real directory"
                    ))
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    fs::create_dir(&current)
                        .await
                        .context("failed to create model package staging parent")?;
                }
                Err(error) => {
                    return Err(error).context("failed to inspect model package staging parent")
                }
            }
        }
    }
    Ok(package_root.join(relative_path))
}

async fn package_staged_bytes(package_root: &Path, files: &[PreparedPackageFile]) -> Result<u64> {
    validate_package_tree(package_root, files, true).await?;
    let mut total = 0_u64;
    for file in files {
        let final_path = package_root.join(&file.filename);
        let extension = final_path
            .extension()
            .and_then(|value| value.to_str())
            .ok_or_else(|| anyhow!("model package file has no extension"))?;
        let part_path = final_path.with_extension(format!("{extension}.part"));
        let final_bytes = file_size(&final_path).await?;
        let part_bytes = file_size(&part_path).await?;
        if final_bytes > 0 && part_bytes > 0 {
            return Err(anyhow!(
                "model package staging contains both partial and verified copies"
            ));
        }
        let retained = final_bytes.saturating_add(part_bytes);
        if retained > file.size_bytes {
            return Err(anyhow!(
                "model package staging file exceeds its signed size"
            ));
        }
        total = total
            .checked_add(retained)
            .ok_or_else(|| anyhow!("model package staged size overflowed"))?;
    }
    Ok(total)
}

async fn validate_complete_package_tree(
    package_root: &Path,
    files: &[PreparedPackageFile],
) -> Result<()> {
    validate_package_tree(package_root, files, false).await
}

async fn validate_package_runtime_layout(package_root: &Path) -> Result<()> {
    let package_root = package_root.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let weights = bloomai_engine::resolve_hf_safetensors_files(&package_root)?;
        if weights.is_empty() {
            return Err(anyhow!(
                "model package does not contain a supported Hugging Face Safetensors layout"
            ));
        }
        Ok(())
    })
    .await
    .context("model package runtime-layout validation task failed")?
}

async fn sync_package_tree_for_install(
    package_root: &Path,
    files: &[PreparedPackageFile],
) -> Result<()> {
    let package_root = package_root.to_path_buf();
    let filenames = files
        .iter()
        .map(|file| file.filename.clone())
        .collect::<Vec<_>>();
    tokio::task::spawn_blocking(move || {
        let mut directories = HashSet::new();
        directories.insert(package_root.clone());
        for filename in filenames {
            validate_package_path(&filename)?;
            let path = package_root.join(&filename);
            let metadata = std::fs::symlink_metadata(&path)
                .with_context(|| format!("failed to inspect package file {filename:?}"))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(anyhow!("package file {filename:?} is not a regular file"));
            }
            std::fs::File::open(&path)
                .and_then(|file| file.sync_all())
                .with_context(|| format!("failed to sync package file {filename:?}"))?;
            let mut parent = path.parent();
            while let Some(directory) = parent {
                if !directory.starts_with(&package_root) {
                    break;
                }
                directories.insert(directory.to_path_buf());
                parent = directory.parent();
            }
        }
        let mut directories = directories.into_iter().collect::<Vec<_>>();
        directories.sort_by_key(|directory| std::cmp::Reverse(directory.components().count()));
        for directory in directories {
            sync_directory_blocking(&directory)?;
        }
        if let Some(staging_root) = package_root.parent() {
            sync_directory_blocking(staging_root)?;
        }
        Ok(())
    })
    .await
    .context("model package durability task failed")?
}

async fn validate_package_tree(
    package_root: &Path,
    files: &[PreparedPackageFile],
    allow_parts: bool,
) -> Result<()> {
    let root = package_root.to_path_buf();
    let expected = files
        .iter()
        .map(|file| file.filename.clone())
        .collect::<Vec<_>>();
    tokio::task::spawn_blocking(move || {
        validate_package_tree_blocking(&root, &expected, allow_parts)
    })
    .await
    .context("model package staging inspection task failed")?
}

fn validate_package_tree_blocking(
    package_root: &Path,
    expected_files: &[String],
    allow_parts: bool,
) -> Result<()> {
    let root_metadata = std::fs::symlink_metadata(package_root)
        .context("failed to inspect model package staging directory")?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(anyhow!(
            "model package staging path must be a real directory"
        ));
    }
    let mut allowed_files = HashSet::new();
    let mut required_files = HashSet::new();
    let mut allowed_directories = HashSet::new();
    for filename in expected_files {
        validate_package_path(filename)?;
        allowed_files.insert(filename.clone());
        required_files.insert(filename.clone());
        if allow_parts {
            allowed_files.insert(format!("{filename}.part"));
        }
        let mut parent = Path::new(filename).parent();
        while let Some(path) = parent {
            if path.as_os_str().is_empty() {
                break;
            }
            allowed_directories.insert(path.to_string_lossy().replace('\\', "/"));
            parent = path.parent();
        }
    }

    let mut discovered_files = HashSet::new();
    let mut stack = vec![package_root.to_path_buf()];
    let mut inspected = 0_usize;
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory).with_context(|| {
            format!("failed to read package directory '{}'", directory.display())
        })? {
            let entry = entry.context("failed to inspect model package staging entry")?;
            inspected = inspected.saturating_add(1);
            if inspected > expected_files.len().saturating_mul(4).saturating_add(16) {
                return Err(anyhow!("model package staging contains too many entries"));
            }
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)
                .context("failed to inspect model package staging entry")?;
            if metadata.file_type().is_symlink() {
                return Err(anyhow!("model package staging must not contain symlinks"));
            }
            let relative = path
                .strip_prefix(package_root)
                .context("model package staging entry escaped its root")?
                .to_str()
                .ok_or_else(|| anyhow!("model package staging filename is not UTF-8"))?
                .replace('\\', "/");
            if metadata.is_dir() {
                if !allowed_directories.contains(&relative) {
                    return Err(anyhow!(
                        "model package staging contains an unexpected directory"
                    ));
                }
                stack.push(path);
            } else if metadata.is_file() {
                if !allowed_files.contains(&relative) {
                    return Err(anyhow!("model package staging contains an unexpected file"));
                }
                discovered_files.insert(relative);
            } else {
                return Err(anyhow!(
                    "model package staging contains an unsupported entry"
                ));
            }
        }
    }
    if !allow_parts && discovered_files != required_files {
        return Err(anyhow!("model package staging is incomplete"));
    }
    Ok(())
}

async fn secure_package_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .await
        .context("failed to inspect model package staging directory")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!(
            "model package staging path must be a real directory"
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .await
            .context("failed to secure model package staging directory")?;
    }
    Ok(())
}

async fn cleanup_package_staging(package_root: &Path, metadata_path: &Path) {
    match fs::symlink_metadata(package_root).await {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            let _ = fs::remove_file(package_root).await;
        }
        Ok(metadata) if metadata.is_dir() => {
            let _ = fs::remove_dir_all(package_root).await;
        }
        Ok(_) => {
            let _ = fs::remove_file(package_root).await;
        }
        Err(_) => {}
    }
    cleanup_package_metadata(metadata_path).await;
}

async fn cleanup_package_metadata(metadata_path: &Path) {
    for path in [metadata_path, &metadata_path.with_extension("json.tmp")] {
        if let Err(error) = fs::remove_file(path).await {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %path.display(), %error, "Failed to clean model package metadata");
            }
        }
    }
}

async fn atomic_install_directory(source: &Path, destination: &Path) -> Result<()> {
    let source = source.to_path_buf();
    let destination = destination.to_path_buf();
    let durability_warnings = tokio::task::spawn_blocking(move || {
        let source_parent = source
            .parent()
            .ok_or_else(|| anyhow!("model package staging path has no parent"))?;
        let destination_parent = destination
            .parent()
            .ok_or_else(|| anyhow!("model package destination has no parent"))?;
        sync_directory_blocking(source_parent)?;
        sync_directory_blocking(destination_parent)?;
        atomic_install_directory_blocking(&source, &destination)?;
        let mut warnings = Vec::new();
        for directory in [source_parent, destination_parent] {
            if let Err(error) = sync_directory_blocking(directory) {
                warnings.push(format!(
                    "failed to sync model package rename parent '{}': {error}",
                    directory.display()
                ));
            }
        }
        Ok::<_, anyhow::Error>(warnings)
    })
    .await
    .context("model package installation task failed")??;
    for warning in durability_warnings {
        tracing::warn!(%warning, "Installed model package has a filesystem durability warning");
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory_blocking(path: &Path) -> Result<()> {
    std::fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("failed to sync directory '{}'", path.display()))
}

#[cfg(not(unix))]
fn sync_directory_blocking(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn atomic_install_directory_blocking(source: &Path, destination: &Path) -> Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt as _;

    let source = CString::new(source.as_os_str().as_bytes())?;
    let destination = CString::new(destination.as_os_str().as_bytes())?;
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).context("atomic package rename failed")
    }
}

#[cfg(not(target_os = "linux"))]
fn atomic_install_directory_blocking(source: &Path, destination: &Path) -> Result<()> {
    if std::fs::symlink_metadata(destination).is_ok() {
        return Err(anyhow!("model package destination already exists"));
    }
    std::fs::rename(source, destination).context("atomic package rename failed")
}

async fn secure_staging_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .await
        .context("failed to inspect staging directory")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!(
            "model download staging path must be a real directory"
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .await
            .context("failed to set staging directory permissions")?;
    }
    Ok(())
}

async fn require_safe_staging_directory(
    path: &Path,
    filename: &str,
) -> std::result::Result<(), ModelDownloadStartError> {
    match fs::symlink_metadata(path).await {
        Ok(_) => secure_staging_directory(path).await.map_err(|error| {
            ModelDownloadStartError::Invalid(format!(
                "model download staging directory is unsafe: {error}"
            ))
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Err(ModelDownloadStartError::NotFound(format!(
                "No staged download exists for '{filename}'."
            )))
        }
        Err(error) => Err(ModelDownloadStartError::Internal(format!(
            "failed to inspect model download staging directory: {error}"
        ))),
    }
}

async fn reject_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(anyhow!(
            "model download staging entries must not be symlinks"
        )),
        Ok(metadata) if !metadata.is_file() => Err(anyhow!(
            "model download staging entry is not a regular file"
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("failed to inspect model download staging entry"),
    }
}

async fn remove_staged_file(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.is_file() || metadata.file_type().is_symlink() => {
            fs::remove_file(path)
                .await
                .context("failed to remove staged download file")?;
            Ok(true)
        }
        Ok(_) => Err(anyhow!("staged download entry is not a file")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("failed to inspect staged download file"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::{Path as AxumPath, State};
    use axum::http::{HeaderMap, Response};
    use axum::routing::get;
    use axum::Router;

    struct PackageServerFixture {
        files: BTreeMap<String, Vec<u8>>,
        delay: Duration,
    }

    fn test_policy() -> DownloadUrlPolicy {
        DownloadUrlPolicy {
            allow_http: true,
            allow_loopback: true,
        }
    }

    fn request(url: String, filename: &str, bytes: &[u8]) -> ModelDownloadRequest {
        ModelDownloadRequest {
            url,
            filename: filename.to_string(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            license: None,
            expected_size_bytes: None,
            model_index_id: None,
        }
    }

    fn safetensors_shard(tensor_name: &str) -> Vec<u8> {
        let mut header = serde_json::to_vec(&serde_json::json!({
            (tensor_name): {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        }))
        .unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 4]);
        bytes
    }

    fn sharded_package_files() -> BTreeMap<String, Vec<u8>> {
        let shard_one = "model-00001-of-00002.safetensors";
        let shard_two = "model-00002-of-00002.safetensors";
        BTreeMap::from([
            (
                "config.json".to_string(),
                br#"{"model_type":"qwen2"}"#.to_vec(),
            ),
            (
                "model.safetensors.index.json".to_string(),
                serde_json::to_vec(&serde_json::json!({
                    "metadata": {"total_size": 8},
                    "weight_map": {
                        "model.embed_tokens.weight": shard_one,
                        "model.norm.weight": shard_two
                    }
                }))
                .unwrap(),
            ),
            (
                shard_one.to_string(),
                safetensors_shard("model.embed_tokens.weight"),
            ),
            (
                shard_two.to_string(),
                safetensors_shard("model.norm.weight"),
            ),
            (
                "tokenizer.json".to_string(),
                br#"{"version":"1.0"}"#.to_vec(),
            ),
        ])
    }

    async fn test_server(bytes: Vec<u8>) -> (String, tokio::task::JoinHandle<()>) {
        async fn download(State(bytes): State<Arc<Vec<u8>>>, headers: HeaderMap) -> Response<Body> {
            let offset = headers
                .get(RANGE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("bytes="))
                .and_then(|value| value.strip_suffix('-'))
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let status = if offset == 0 {
                StatusCode::OK
            } else if offset >= bytes.len() {
                StatusCode::RANGE_NOT_SATISFIABLE
            } else {
                StatusCode::PARTIAL_CONTENT
            };
            let mut response = Response::builder()
                .status(status)
                .header("content-length", bytes.len().saturating_sub(offset));
            if offset >= bytes.len() {
                response = response.header(CONTENT_RANGE, format!("bytes */{}", bytes.len()));
            } else if offset > 0 {
                response = response.header(
                    CONTENT_RANGE,
                    format!("bytes {offset}-{}/{}", bytes.len() - 1, bytes.len()),
                );
            }
            response
                .body(Body::from(bytes[offset.min(bytes.len())..].to_vec()))
                .expect("test response should be valid")
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/model.gguf", get(download))
            .with_state(Arc::new(bytes));
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}/model.gguf"), handle)
    }

    async fn package_server(
        files: BTreeMap<String, Vec<u8>>,
        delay: Duration,
    ) -> (String, tokio::task::JoinHandle<()>) {
        async fn download(
            State(fixture): State<Arc<PackageServerFixture>>,
            AxumPath(path): AxumPath<String>,
            headers: HeaderMap,
        ) -> Response<Body> {
            tokio::time::sleep(fixture.delay).await;
            let Some(bytes) = fixture.files.get(&path) else {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::empty())
                    .unwrap();
            };
            let offset = headers
                .get(RANGE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("bytes="))
                .and_then(|value| value.strip_suffix('-'))
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let status = if offset == 0 {
                StatusCode::OK
            } else if offset >= bytes.len() {
                StatusCode::RANGE_NOT_SATISFIABLE
            } else {
                StatusCode::PARTIAL_CONTENT
            };
            let mut response = Response::builder()
                .status(status)
                .header("content-length", bytes.len().saturating_sub(offset));
            if offset >= bytes.len() {
                response = response.header(CONTENT_RANGE, format!("bytes */{}", bytes.len()));
            } else if offset > 0 {
                response = response.header(
                    CONTENT_RANGE,
                    format!("bytes {offset}-{}/{}", bytes.len() - 1, bytes.len()),
                );
            }
            response
                .body(Body::from(bytes[offset.min(bytes.len())..].to_vec()))
                .unwrap()
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/{*path}", get(download))
            .with_state(Arc::new(PackageServerFixture { files, delay }));
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), handle)
    }

    fn package_request(
        base_url: &str,
        directory: &str,
        files: &BTreeMap<String, Vec<u8>>,
    ) -> ModelPackageDownloadRequest {
        let descriptors = files
            .iter()
            .map(|(filename, bytes)| ModelPackageFile {
                filename: filename.clone(),
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            })
            .collect::<Vec<_>>();
        let total_bytes = descriptors.iter().map(|file| file.size_bytes).sum();
        ModelPackageDownloadRequest {
            directory: directory.to_string(),
            size_bytes: total_bytes,
            sha256: package_digest(&descriptors).unwrap(),
            files: descriptors
                .into_iter()
                .map(|file| ModelPackageDownloadFile {
                    url: format!("{base_url}/{}", file.filename),
                    filename: file.filename,
                    size_bytes: file.size_bytes,
                    sha256: file.sha256,
                })
                .collect(),
            license: Some("Apache-2.0".to_string()),
            model_index_id: "tiny-package".to_string(),
        }
    }

    async fn metadata_server(
        status: StatusCode,
        headers: HeaderMap,
    ) -> (String, tokio::task::JoinHandle<()>) {
        async fn metadata(State(fixture): State<Arc<(StatusCode, HeaderMap)>>) -> Response<Body> {
            let mut response = Response::builder()
                .status(fixture.0)
                .body(Body::empty())
                .expect("test response should be valid");
            *response.headers_mut() = fixture.1.clone();
            response
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/acme/repo/resolve/main/model.gguf", get(metadata))
            .with_state(Arc::new((status, headers)));
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (
            format!("http://{address}/acme/repo/resolve/main/model.gguf"),
            handle,
        )
    }

    async fn wait_for_terminal(manager: &ModelDownloadManager) -> ModelDownloadStatus {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = manager.status().await;
                if matches!(
                    status.phase,
                    ModelDownloadPhase::Complete
                        | ModelDownloadPhase::Cancelled
                        | ModelDownloadPhase::Error
                ) {
                    return status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap()
    }

    #[test]
    fn accepts_only_trusted_https_download_urls() {
        let policy = DownloadUrlPolicy::default();
        assert!(validate_download_url(
            &Url::parse("https://huggingface.co/acme/model/resolve/main/model.gguf").unwrap(),
            &policy
        )
        .is_ok());
        assert!(validate_download_url(
            &Url::parse("https://cdn-lfs.hf.co/model.gguf").unwrap(),
            &policy
        )
        .is_ok());
        assert!(validate_download_url(
            &Url::parse("http://huggingface.co/model.gguf").unwrap(),
            &policy
        )
        .is_err());
        assert!(validate_download_url(
            &Url::parse("https://huggingface.co.evil.example/model.gguf").unwrap(),
            &policy
        )
        .is_err());
        assert!(validate_download_url(
            &Url::parse("https://user:secret@huggingface.co/model.gguf").unwrap(),
            &policy
        )
        .is_err());
    }

    #[test]
    fn normalizes_browser_source_urls_and_pins_commits() {
        let (url, filename) = prepare_source_url(
            "https://huggingface.co/acme/repo/blob/main/nested/model.gguf?download=true#readme",
            &DownloadUrlPolicy::default(),
        )
        .unwrap();

        assert_eq!(filename, "model.gguf");
        assert_eq!(
            url.as_str(),
            "https://huggingface.co/acme/repo/resolve/main/nested/model.gguf"
        );
        let commit = "a".repeat(40);
        assert_eq!(
            pin_source_revision(&url, &commit).unwrap().as_str(),
            format!("https://huggingface.co/acme/repo/resolve/{commit}/nested/model.gguf")
        );
    }

    #[tokio::test]
    async fn inspects_hugging_face_metadata_without_downloading_model_bytes() {
        let sha256 = "ab".repeat(32);
        let commit = "cd".repeat(20);
        let mut headers = HeaderMap::new();
        headers.insert(LINKED_ETAG_HEADER, format!("\"{sha256}\"").parse().unwrap());
        headers.insert(LINKED_SIZE_HEADER, "483116416".parse().unwrap());
        headers.insert(REPO_COMMIT_HEADER, commit.parse().unwrap());
        headers.insert(LOCATION, "/objects/model.gguf".parse().unwrap());
        let (url, server) = metadata_server(StatusCode::FOUND, headers).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024 * 1024,
            test_policy(),
        )
        .unwrap();

        let source = manager
            .inspect_source(ModelDownloadSourceRequest { url })
            .await
            .unwrap();

        assert_eq!(source.filename, "model.gguf");
        assert_eq!(source.size_bytes, Some(483116416));
        assert_eq!(source.sha256.as_deref(), Some(sha256.as_str()));
        assert_eq!(source.commit_hash.as_deref(), Some(commit.as_str()));
        assert!(source.verification_ready);
        assert!(source.warning.is_none());
        assert!(source
            .download_url
            .contains(&format!("/resolve/{commit}/model.gguf")));
        server.abort();
    }

    #[tokio::test]
    async fn inspection_keeps_manual_hash_as_a_fail_closed_fallback() {
        let commit = "12".repeat(20);
        let mut headers = HeaderMap::new();
        headers.insert(ETAG, format!("\"{}\"", "ef".repeat(20)).parse().unwrap());
        headers.insert(CONTENT_LENGTH, "4096".parse().unwrap());
        headers.insert(REPO_COMMIT_HEADER, commit.parse().unwrap());
        let (url, server) = metadata_server(StatusCode::OK, headers).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();

        let source = manager
            .inspect_source(ModelDownloadSourceRequest { url })
            .await
            .unwrap();

        assert_eq!(source.sha256, None);
        assert!(!source.verification_ready);
        assert!(source.warning.unwrap().contains("independently obtained"));
        assert_eq!(source.size_bytes, Some(4096));
        assert!(source.download_url.contains(&format!("/resolve/{commit}/")));
        server.abort();
    }

    #[tokio::test]
    async fn inspection_rejects_sources_over_the_download_limit() {
        let mut headers = HeaderMap::new();
        headers.insert(LINKED_SIZE_HEADER, "4097".parse().unwrap());
        let (url, server) = metadata_server(StatusCode::OK, headers).await;
        let temp = tempfile::tempdir().unwrap();
        let manager =
            ModelDownloadManager::with_policy(temp.path().to_path_buf(), 4096, test_policy())
                .unwrap();

        let error = manager
            .inspect_source(ModelDownloadSourceRequest { url })
            .await
            .unwrap_err();

        assert!(matches!(error, ModelDownloadInspectError::TooLarge(_)));
        server.abort();
    }

    #[test]
    fn download_preparation_enforces_the_configured_license_allowlist() {
        let policy = ModelLicensePolicy::new(vec!["Apache-2.0".to_string()]).unwrap();
        let mut download = request(
            "https://huggingface.co/acme/repo/resolve/main/model.gguf".to_string(),
            "model.gguf",
            b"model",
        );

        assert!(
            prepare_download(download.clone(), &DownloadUrlPolicy::default(), &policy)
                .unwrap_err()
                .to_string()
                .contains("required")
        );
        download.license = Some("MIT".to_string());
        assert!(
            prepare_download(download.clone(), &DownloadUrlPolicy::default(), &policy)
                .unwrap_err()
                .to_string()
                .contains("not allowed")
        );
        download.license = Some("apache-2.0".to_string());
        assert_eq!(
            prepare_download(download, &DownloadUrlPolicy::default(), &policy)
                .unwrap()
                .license
                .as_deref(),
            Some("Apache-2.0")
        );
    }

    #[test]
    fn license_policy_updates_do_not_invalidate_matching_partial_bytes() {
        let original = StagedDownload {
            version: 1,
            url: "https://huggingface.co/acme/repo/resolve/main/model.gguf".to_string(),
            filename: "model.gguf".to_string(),
            sha256: "ab".repeat(32),
            license: None,
            expected_size_bytes: None,
            model_index_id: None,
            replacement: None,
        };
        let approved = StagedDownload {
            license: Some("Apache-2.0".to_string()),
            ..original.clone()
        };
        let different_source = StagedDownload {
            url: "https://huggingface.co/acme/repo/resolve/main/other.gguf".to_string(),
            ..approved.clone()
        };

        assert!(staged_content_identity_matches(&original, &approved));
        assert!(!staged_content_identity_matches(
            &original,
            &different_source
        ));
    }

    #[tokio::test]
    async fn downloads_verifies_and_atomically_installs_a_model() {
        let bytes = b"small deterministic GGUF fixture".repeat(2048);
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        let mut download = request(url, "fixture.gguf", &bytes);
        download.license = Some("MIT".to_string());

        manager.start(download).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Complete);
        assert_eq!(
            fs::read(temp.path().join("fixture.gguf")).await.unwrap(),
            bytes
        );
        assert_eq!(manager.catalog_revision(), 1);
        let provenance = super::super::model_provenance::read_provenance(
            temp.path(),
            "fixture.gguf",
            bytes.len() as u64,
        )
        .unwrap()
        .unwrap();
        assert_eq!(provenance.acquisition, ModelAcquisitionKind::Download);
        assert_eq!(provenance.license.as_deref(), Some("MIT"));
        server.abort();
    }

    #[tokio::test]
    async fn downloads_verifies_and_atomically_installs_a_model_package() {
        let files = sharded_package_files();
        let (base_url, server) = package_server(files.clone(), Duration::from_millis(25)).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        let request = package_request(&base_url, "tiny-package", &files);
        let expected_digest = request.sha256.clone();
        let expected_size = request.size_bytes;

        manager.start_package(request).await.unwrap();
        assert!(!temp.path().join("tiny-package").exists());
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Complete);
        for (filename, bytes) in &files {
            assert_eq!(
                fs::read(temp.path().join("tiny-package").join(filename))
                    .await
                    .unwrap(),
                *bytes
            );
        }
        let provenance = super::super::model_provenance::read_provenance(
            temp.path(),
            "tiny-package",
            expected_size,
        )
        .unwrap()
        .unwrap();
        assert_eq!(provenance.sha256, expected_digest);
        assert_eq!(provenance.file_count, Some(files.len()));
        assert_eq!(provenance.model_index_id.as_deref(), Some("tiny-package"));
        let catalog = super::super::model_manager::ModelCatalog::scan(temp.path(), None).unwrap();
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].kind, "directory");
        assert!(catalog.models[0].provenance.is_some());
        server.abort();
    }

    #[tokio::test]
    async fn logically_incomplete_sharded_package_is_rejected_before_publication() {
        let shard = "model-00001-of-00001.safetensors";
        let files = BTreeMap::from([
            (
                "config.json".to_string(),
                br#"{"model_type":"qwen2"}"#.to_vec(),
            ),
            (
                "model.safetensors.index.json".to_string(),
                serde_json::to_vec(&serde_json::json!({
                    "metadata": {"total_size": 4},
                    "weight_map": {"different.weight": shard}
                }))
                .unwrap(),
            ),
            (shard.to_string(), safetensors_shard("weight")),
        ]);
        let (base_url, server) = package_server(files.clone(), Duration::ZERO).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        let request = package_request(&base_url, "incomplete-shards", &files);
        let expected_size = request.size_bytes;

        manager.start_package(request).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(!status.resumable);
        assert!(status
            .error
            .unwrap()
            .contains("weight_map does not match tensor headers"));
        assert!(!temp.path().join("incomplete-shards").exists());
        assert!(!temp
            .path()
            .join(STAGING_DIRECTORY)
            .join("incomplete-shards.package")
            .exists());
        assert!(super::super::model_provenance::read_provenance(
            temp.path(),
            "incomplete-shards",
            expected_size,
        )
        .unwrap()
        .is_none());
        assert_eq!(manager.catalog_revision(), 0);
        server.abort();
    }

    #[tokio::test]
    async fn package_checksum_failure_never_publishes_partial_contents() {
        let files = BTreeMap::from([
            (
                "config.json".to_string(),
                br#"{"model_type":"qwen2"}"#.to_vec(),
            ),
            (
                "model.safetensors".to_string(),
                b"tamper-evident package weights".repeat(1024),
            ),
        ]);
        let (base_url, server) = package_server(files.clone(), Duration::ZERO).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        let mut request = package_request(&base_url, "rejected-package", &files);
        request.files[1].sha256 = "00".repeat(32);
        let signed_files = request
            .files
            .iter()
            .map(|file| ModelPackageFile {
                filename: file.filename.clone(),
                size_bytes: file.size_bytes,
                sha256: file.sha256.clone(),
            })
            .collect::<Vec<_>>();
        request.sha256 = package_digest(&signed_files).unwrap();
        let expected_size = request.size_bytes;

        manager.start_package(request).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(!status.resumable);
        assert!(status
            .error
            .unwrap()
            .contains("SHA-256 verification failed"));
        assert!(!temp.path().join("rejected-package").exists());
        assert!(!temp
            .path()
            .join(STAGING_DIRECTORY)
            .join("rejected-package.package")
            .exists());
        assert!(!temp
            .path()
            .join(STAGING_DIRECTORY)
            .join("rejected-package.package.json")
            .exists());
        assert!(super::super::model_provenance::read_provenance(
            temp.path(),
            "rejected-package",
            expected_size,
        )
        .unwrap()
        .is_none());
        assert_eq!(manager.catalog_revision(), 0);
        server.abort();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn staged_package_symlink_is_removed_without_following_it() {
        use std::os::unix::fs::symlink;

        let files = BTreeMap::from([
            (
                "config.json".to_string(),
                br#"{"model_type":"qwen2"}"#.to_vec(),
            ),
            (
                "model.safetensors".to_string(),
                b"symlink-safe package weights".repeat(64),
            ),
        ]);
        let (base_url, server) = package_server(files.clone(), Duration::ZERO).await;
        let temp = tempfile::tempdir().unwrap();
        let request = package_request(&base_url, "unsafe-package", &files);
        let prepared = prepare_package_download(
            request.clone(),
            &test_policy(),
            &ModelLicensePolicy::default(),
        )
        .unwrap();
        let staging = temp.path().join(STAGING_DIRECTORY);
        fs::create_dir_all(&staging).await.unwrap();
        secure_staging_directory(&staging).await.unwrap();
        let package_root = staging.join("unsafe-package.package");
        let metadata_path = staging.join("unsafe-package.package.json");
        let metadata = StagedPackageDownload {
            version: 1,
            directory: prepared.directory.clone(),
            size_bytes: prepared.size_bytes,
            sha256: prepared.sha256.clone(),
            files: prepared
                .files
                .iter()
                .map(|file| StagedPackageFile {
                    url: file.url.to_string(),
                    filename: file.filename.clone(),
                    size_bytes: file.size_bytes,
                    sha256: file.sha256_hex.clone(),
                })
                .collect(),
            license: prepared.license.clone(),
            model_index_id: prepared.model_index_id.clone(),
            replacement: None,
        };
        prepare_package_staging(&package_root, &metadata_path, &metadata)
            .await
            .unwrap();
        let outside = temp.path().join("outside-data");
        fs::write(&outside, b"must remain untouched").await.unwrap();
        symlink(&outside, package_root.join("config.json.part")).unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();

        manager.start_package(request).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(!status.resumable);
        assert!(status.error.unwrap().contains("must not contain symlinks"));
        assert_eq!(fs::read(&outside).await.unwrap(), b"must remain untouched");
        assert!(!temp.path().join("unsafe-package").exists());
        assert!(!package_root.exists());
        assert!(!metadata_path.exists());
        assert_eq!(manager.catalog_revision(), 0);
        server.abort();
    }

    #[tokio::test]
    async fn resumes_a_matching_staged_model_package_after_restart() {
        let files = sharded_package_files();
        let (base_url, server) = package_server(files.clone(), Duration::ZERO).await;
        let temp = tempfile::tempdir().unwrap();
        let request = package_request(&base_url, "resumable-package", &files);
        let prepared = prepare_package_download(
            request.clone(),
            &test_policy(),
            &ModelLicensePolicy::default(),
        )
        .unwrap();
        let staging = temp.path().join(STAGING_DIRECTORY);
        fs::create_dir_all(&staging).await.unwrap();
        secure_staging_directory(&staging).await.unwrap();
        let package_root = staging.join("resumable-package.package");
        let metadata_path = staging.join("resumable-package.package.json");
        let metadata = StagedPackageDownload {
            version: 1,
            directory: prepared.directory.clone(),
            size_bytes: prepared.size_bytes,
            sha256: prepared.sha256.clone(),
            files: prepared
                .files
                .iter()
                .map(|file| StagedPackageFile {
                    url: file.url.to_string(),
                    filename: file.filename.clone(),
                    size_bytes: file.size_bytes,
                    sha256: file.sha256_hex.clone(),
                })
                .collect(),
            license: prepared.license.clone(),
            model_index_id: prepared.model_index_id.clone(),
            replacement: None,
        };
        prepare_package_staging(&package_root, &metadata_path, &metadata)
            .await
            .unwrap();
        let weights = files.get("model-00001-of-00002.safetensors").unwrap();
        fs::write(
            package_root.join("model-00001-of-00002.safetensors.part"),
            &weights[..weights.len() / 2],
        )
        .await
        .unwrap();

        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        let staged = manager.staged().await;
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].filename, "resumable-package");
        manager
            .resume("resumable-package", Some("Apache-2.0".to_string()))
            .await
            .unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Complete);
        assert_eq!(
            fs::read(
                temp.path()
                    .join("resumable-package/model-00001-of-00002.safetensors")
            )
            .await
            .unwrap(),
            *weights
        );
        server.abort();
    }

    #[tokio::test]
    async fn signed_size_mismatch_is_rejected_before_installation() {
        let bytes = b"signed size mismatch fixture".repeat(256);
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let manager =
            ModelDownloadManager::with_loopback_test_policy(temp.path().to_path_buf(), 1024 * 1024)
                .unwrap();
        let mut download = request(url, "mismatch.gguf", &bytes);
        download.expected_size_bytes = Some(bytes.len() as u64 + 1);

        manager.start(download).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(!status.resumable);
        assert!(status.error.unwrap().contains("signed index requires"));
        assert!(!temp.path().join("mismatch.gguf").exists());
        assert!(manager.staged().await.is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn active_download_identity_requires_filename_hash_and_signed_size() {
        let bytes = b"active download identity fixture".repeat(256);
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let manager =
            ModelDownloadManager::with_loopback_test_policy(temp.path().to_path_buf(), 1024 * 1024)
                .unwrap();
        let mut download = request(url, "identity.gguf", &bytes);
        download.expected_size_bytes = Some(bytes.len() as u64);
        let sha256 = download.sha256.clone();

        manager.start(download).await.unwrap();

        assert!(
            manager
                .active_matches("identity.gguf", &sha256, Some(bytes.len() as u64), None,)
                .await
        );
        assert!(
            !manager
                .active_matches("other.gguf", &sha256, Some(bytes.len() as u64), None,)
                .await
        );
        assert!(
            !manager
                .active_matches(
                    "identity.gguf",
                    &"00".repeat(32),
                    Some(bytes.len() as u64),
                    None,
                )
                .await
        );
        assert!(
            !manager
                .active_matches("identity.gguf", &sha256, None, None)
                .await
        );
        manager.cancel().await;
        let _ = wait_for_terminal(&manager).await;
        server.abort();
    }

    #[tokio::test]
    async fn shared_storage_quota_rejects_a_download_before_writing_its_body() {
        let bytes = b"quota download fixture".repeat(8);
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("installed.gguf"), vec![0_u8; 80])
            .await
            .unwrap();
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 100, 0);
        let manager = ModelDownloadManager::with_policy_and_storage(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
            Arc::clone(&storage),
        )
        .unwrap();

        manager
            .start(request(url, "quota.gguf", &bytes))
            .await
            .unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(status.resumable);
        assert!(status.error.unwrap().contains("quota exceeded"));
        assert!(!manager.staging_root().join("quota.gguf.part").exists());
        assert_eq!(storage.snapshot().await.unwrap().committed_bytes, 80);
        server.abort();
    }

    #[tokio::test]
    async fn upgrades_reserve_peak_space_without_treating_the_old_model_as_reclaimable() {
        let old = b"installed upgrade quota fixture".repeat(4);
        let new = b"replacement upgrade quota fixture".repeat(6);
        let (url, server) = test_server(new.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let filename = "upgrade-quota.gguf";
        let index_id = "upgrade-quota";
        fs::write(temp.path().join(filename), &old).await.unwrap();
        write_provenance(
            temp.path(),
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(index_id.to_string()),
                filename: filename.to_string(),
                size_bytes: old.len() as u64,
                source_url: Some(format!(
                    "https://huggingface.co/acme/old/resolve/{}/{}",
                    "11".repeat(20),
                    filename
                )),
                source_host: Some("huggingface.co".to_string()),
                sha256: format!("{:x}", Sha256::digest(&old)),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let max_storage = (old.len() + new.len() - 1) as u64;
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), max_storage, 0);
        let manager = ModelDownloadManager::with_policy_and_storage(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
            Arc::clone(&storage),
        )
        .unwrap();
        let mut download = request(url, filename, &new);
        download.expected_size_bytes = Some(new.len() as u64);
        download.model_index_id = Some(index_id.to_string());

        manager
            .start_upgrade(
                download,
                ModelUpgradeSource {
                    catalog_id: filename.to_string(),
                    kind: super::super::model_upgrade::ModelUpgradeEntryKind::File,
                    size_bytes: old.len() as u64,
                    sha256: format!("{:x}", Sha256::digest(&old)),
                    model_index_id: index_id.to_string(),
                },
            )
            .await
            .unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(status.resumable);
        assert!(status.error.unwrap().contains("quota exceeded"));
        assert_eq!(fs::read(temp.path().join(filename)).await.unwrap(), old);
        let provenance = super::super::model_provenance::read_provenance(
            temp.path(),
            filename,
            old.len() as u64,
        )
        .unwrap()
        .unwrap();
        assert_eq!(provenance.model_index_id.as_deref(), Some(index_id));
        assert!(!temp
            .path()
            .join(super::super::model_upgrade::UPGRADE_DIRECTORY)
            .exists());
        server.abort();
    }

    #[tokio::test]
    async fn failed_upgrade_verification_leaves_the_installed_model_usable() {
        let old = b"installed checksum rollback fixture";
        let new = b"downloaded but incorrectly signed replacement".repeat(128);
        let (url, server) = test_server(new.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let filename = "upgrade-checksum.gguf";
        let index_id = "upgrade-checksum";
        fs::write(temp.path().join(filename), old).await.unwrap();
        let old_sha256 = format!("{:x}", Sha256::digest(old));
        write_provenance(
            temp.path(),
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(index_id.to_string()),
                filename: filename.to_string(),
                size_bytes: old.len() as u64,
                source_url: Some(format!(
                    "https://huggingface.co/acme/old/resolve/{}/{}",
                    "11".repeat(20),
                    filename
                )),
                source_host: Some("huggingface.co".to_string()),
                sha256: old_sha256.clone(),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        let mut download = request(url, filename, &new);
        download.sha256 = "ff".repeat(32);
        download.expected_size_bytes = Some(new.len() as u64);
        download.model_index_id = Some(index_id.to_string());

        manager
            .start_upgrade(
                download,
                ModelUpgradeSource {
                    catalog_id: filename.to_string(),
                    kind: super::super::model_upgrade::ModelUpgradeEntryKind::File,
                    size_bytes: old.len() as u64,
                    sha256: old_sha256.clone(),
                    model_index_id: index_id.to_string(),
                },
            )
            .await
            .unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(!status.resumable);
        assert!(status
            .error
            .unwrap()
            .contains("SHA-256 verification failed"));
        assert_eq!(fs::read(temp.path().join(filename)).await.unwrap(), old);
        let provenance = super::super::model_provenance::read_provenance(
            temp.path(),
            filename,
            old.len() as u64,
        )
        .unwrap()
        .unwrap();
        assert_eq!(provenance.sha256, old_sha256);
        assert!(!temp
            .path()
            .join(super::super::model_upgrade::UPGRADE_DIRECTORY)
            .exists());
        server.abort();
    }

    #[tokio::test]
    async fn resumes_a_matching_staged_download() {
        let bytes = b"resumable GGUF fixture".repeat(2048);
        let split = bytes.len() / 3;
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        fs::create_dir_all(manager.staging_root()).await.unwrap();
        let download_request = request(url.clone(), "resume.gguf", &bytes);
        let prepared = prepare_download(
            download_request.clone(),
            &test_policy(),
            &ModelLicensePolicy::default(),
        )
        .unwrap();
        let metadata = StagedDownload {
            version: 1,
            url,
            filename: "resume.gguf".to_string(),
            sha256: prepared.sha256_hex,
            license: None,
            expected_size_bytes: None,
            model_index_id: None,
            replacement: None,
        };
        fs::write(
            manager.staging_root().join("resume.gguf.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .await
        .unwrap();
        fs::write(
            manager.staging_root().join("resume.gguf.part"),
            &bytes[..split],
        )
        .await
        .unwrap();

        let staged = manager.staged().await;
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].filename, "resume.gguf");
        assert_eq!(staged[0].downloaded_bytes, split as u64);

        manager.resume("resume.gguf", None).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Complete);
        assert_eq!(
            fs::read(temp.path().join("resume.gguf")).await.unwrap(),
            bytes
        );
        server.abort();
    }

    #[tokio::test]
    async fn restart_resume_retains_the_exact_signed_upgrade_identity() {
        let old = b"previous resumable signed model";
        let new = b"resumable signed upgrade payload".repeat(1024);
        let split = new.len() / 3;
        let (url, server) = test_server(new.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let filename = "resume-upgrade.gguf";
        let index_id = "resume-upgrade";
        fs::write(temp.path().join(filename), old).await.unwrap();
        write_provenance(
            temp.path(),
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(index_id.to_string()),
                filename: filename.to_string(),
                size_bytes: old.len() as u64,
                source_url: Some(format!(
                    "https://huggingface.co/acme/old/resolve/{}/{}",
                    "11".repeat(20),
                    filename
                )),
                source_host: Some("huggingface.co".to_string()),
                sha256: format!("{:x}", Sha256::digest(old)),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        fs::create_dir_all(manager.staging_root()).await.unwrap();
        let metadata = StagedDownload {
            version: 1,
            url,
            filename: filename.to_string(),
            sha256: format!("{:x}", Sha256::digest(&new)),
            license: Some("Apache-2.0".to_string()),
            expected_size_bytes: Some(new.len() as u64),
            model_index_id: Some(index_id.to_string()),
            replacement: Some(ModelUpgradeSource {
                catalog_id: filename.to_string(),
                kind: super::super::model_upgrade::ModelUpgradeEntryKind::File,
                size_bytes: old.len() as u64,
                sha256: format!("{:x}", Sha256::digest(old)),
                model_index_id: index_id.to_string(),
            }),
        };
        fs::write(
            manager.staging_root().join(format!("{filename}.json")),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .await
        .unwrap();
        fs::write(
            manager.staging_root().join(format!("{filename}.part")),
            &new[..split],
        )
        .await
        .unwrap();

        manager.resume(filename, None).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Complete, "{status:?}");
        assert_eq!(fs::read(temp.path().join(filename)).await.unwrap(), new);
        let provenance = super::super::model_provenance::read_provenance(
            temp.path(),
            filename,
            metadata.expected_size_bytes.unwrap(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(provenance.model_index_id.as_deref(), Some(index_id));
        assert_eq!(provenance.sha256, metadata.sha256);
        server.abort();
    }

    #[tokio::test]
    async fn verifies_a_fully_received_staged_download_after_restart() {
        let bytes = b"fully received GGUF fixture".repeat(1024);
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        fs::create_dir_all(manager.staging_root()).await.unwrap();
        let download_request = request(url.clone(), "received.gguf", &bytes);
        let prepared = prepare_download(
            download_request.clone(),
            &test_policy(),
            &ModelLicensePolicy::default(),
        )
        .unwrap();
        let metadata = StagedDownload {
            version: 1,
            url,
            filename: "received.gguf".to_string(),
            sha256: prepared.sha256_hex,
            license: None,
            expected_size_bytes: None,
            model_index_id: None,
            replacement: None,
        };
        fs::write(
            manager.staging_root().join("received.gguf.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .await
        .unwrap();
        fs::write(manager.staging_root().join("received.gguf.part"), &bytes)
            .await
            .unwrap();

        manager.resume("received.gguf", None).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Complete);
        assert_eq!(
            fs::read(temp.path().join("received.gguf")).await.unwrap(),
            bytes
        );
        server.abort();
    }

    #[tokio::test]
    async fn rejects_and_removes_a_download_with_the_wrong_checksum() {
        let bytes = b"bad checksum fixture".repeat(1024);
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        let mut download_request = request(url, "wrong.gguf", &bytes);
        download_request.sha256 = "00".repeat(32);

        manager.start(download_request).await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(!status.resumable);
        assert!(!temp.path().join("wrong.gguf").exists());
        assert!(!manager.staging_root().join("wrong.gguf.part").exists());
        server.abort();
    }

    #[tokio::test]
    async fn discards_staged_download_data_without_exposing_paths() {
        let bytes = b"discard fixture".repeat(128);
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        fs::create_dir_all(manager.staging_root()).await.unwrap();
        let metadata = StagedDownload {
            version: 1,
            url: "http://127.0.0.1:32123/model.gguf".to_string(),
            filename: "discard.gguf".to_string(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            license: None,
            expected_size_bytes: None,
            model_index_id: None,
            replacement: None,
        };
        fs::write(
            manager.staging_root().join("discard.gguf.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .await
        .unwrap();
        fs::write(manager.staging_root().join("discard.gguf.part"), bytes)
            .await
            .unwrap();

        assert_eq!(manager.staged().await.len(), 1);
        manager.discard("discard.gguf").await.unwrap();

        assert!(manager.staged().await.is_empty());
        assert!(matches!(
            manager.discard("discard.gguf").await,
            Err(ModelDownloadStartError::NotFound(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_symlinked_staging_files() {
        use std::os::unix::fs::symlink;

        let bytes = b"symlink fixture".repeat(256);
        let (url, server) = test_server(bytes.clone()).await;
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let manager = ModelDownloadManager::with_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            test_policy(),
        )
        .unwrap();
        fs::create_dir_all(manager.staging_root()).await.unwrap();
        symlink(
            outside.path(),
            manager.staging_root().join("linked.gguf.part"),
        )
        .unwrap();

        manager
            .start(request(url, "linked.gguf", &bytes))
            .await
            .unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelDownloadPhase::Error);
        assert!(!temp.path().join("linked.gguf").exists());
        server.abort();
    }
}
