//! Bounded, resumable local-file imports into the model catalog.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs::{self, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, RwLock};

use super::model_license::{ModelLicensePolicy, ModelLicensePolicyStatus};
use super::model_manager::validate_model_filename;
use super::model_provenance::{
    ModelAcquisitionKind, ModelProvenanceDraft, normalize_source_url, write_provenance,
};
use super::model_storage::{ModelStorageError, ModelStorageManager};

const STAGING_DIRECTORY: &str = ".bloom-imports";
const MAX_STAGED_IMPORTS: usize = 1_000;
const MAX_STAGING_ENTRIES_TO_INSPECT: usize = MAX_STAGED_IMPORTS * 3;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct ModelImportRequest {
    pub filename: String,
    pub total_bytes: u64,
    pub sha256: String,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelImportPhase {
    Idle,
    Ready,
    Uploading,
    Verifying,
    Complete,
    Error,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelImportStatus {
    pub phase: ModelImportPhase,
    pub filename: Option<String>,
    pub uploaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub error: Option<String>,
}

impl Default for ModelImportStatus {
    fn default() -> Self {
        Self {
            phase: ModelImportPhase::Idle,
            filename: None,
            uploaded_bytes: 0,
            total_bytes: None,
            error: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct StagedModelImport {
    pub filename: String,
    pub uploaded_bytes: u64,
    pub total_bytes: u64,
    pub modified_at: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelImportError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    NotFound(String),
    #[error("upload offset does not match the staged file; expected {expected}")]
    OffsetMismatch { expected: u64 },
    #[error("{0}")]
    Internal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct StagedImportMetadata {
    version: u8,
    filename: String,
    total_bytes: u64,
    sha256: String,
    #[serde(default)]
    source_url: Option<String>,
    #[serde(default)]
    license: Option<String>,
}

pub(crate) struct ModelImportManager {
    models_root: PathBuf,
    max_bytes: u64,
    max_chunk_bytes: usize,
    status: RwLock<ModelImportStatus>,
    operation: Mutex<()>,
    catalog_revision: AtomicU64,
    storage: Arc<ModelStorageManager>,
    license_policy: Arc<ModelLicensePolicy>,
}

impl ModelImportManager {
    #[cfg(test)]
    pub(crate) fn new(
        models_root: PathBuf,
        max_bytes: u64,
        max_chunk_bytes: usize,
    ) -> Result<Arc<Self>> {
        let storage = ModelStorageManager::new(models_root.clone(), 0, 0);
        Self::with_storage(models_root, max_bytes, max_chunk_bytes, storage)
    }

    #[cfg(test)]
    pub(crate) fn with_storage(
        models_root: PathBuf,
        max_bytes: u64,
        max_chunk_bytes: usize,
        storage: Arc<ModelStorageManager>,
    ) -> Result<Arc<Self>> {
        Self::with_storage_and_license_policy(
            models_root,
            max_bytes,
            max_chunk_bytes,
            storage,
            Arc::new(ModelLicensePolicy::default()),
        )
    }

    pub(crate) fn with_storage_and_license_policy(
        models_root: PathBuf,
        max_bytes: u64,
        max_chunk_bytes: usize,
        storage: Arc<ModelStorageManager>,
        license_policy: Arc<ModelLicensePolicy>,
    ) -> Result<Arc<Self>> {
        if max_bytes == 0 {
            return Err(anyhow!(
                "maximum model import size must be greater than zero"
            ));
        }
        if max_chunk_bytes == 0 || max_chunk_bytes as u64 > max_bytes {
            return Err(anyhow!(
                "model import chunk size must be greater than zero and no larger than the import limit"
            ));
        }
        Ok(Arc::new(Self {
            models_root,
            max_bytes,
            max_chunk_bytes,
            status: RwLock::new(ModelImportStatus::default()),
            operation: Mutex::new(()),
            catalog_revision: AtomicU64::new(0),
            storage,
            license_policy,
        }))
    }

    pub(crate) async fn status(&self) -> ModelImportStatus {
        self.status.read().await.clone()
    }

    pub(crate) fn catalog_revision(&self) -> u64 {
        self.catalog_revision.load(Ordering::Acquire)
    }

    pub(crate) fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub(crate) fn max_chunk_bytes(&self) -> usize {
        self.max_chunk_bytes
    }

    pub(crate) fn license_policy(&self) -> ModelLicensePolicyStatus {
        self.license_policy.status()
    }

    pub(crate) async fn begin(
        &self,
        request: ModelImportRequest,
    ) -> std::result::Result<ModelImportStatus, ModelImportError> {
        let request = validate_import_request(request, self.max_bytes, &self.license_policy)?;
        let _guard = self.operation.lock().await;
        let mut storage = self.storage.serial().await;
        if self.models_root.join(&request.filename).exists() {
            return Err(ModelImportError::Conflict(format!(
                "A catalog entry named '{}' already exists.",
                request.filename
            )));
        }
        let staging_root = self.ensure_staging_root().await?;
        let metadata_path = staging_root.join(format!("{}.json", request.filename));
        let part_path = staging_root.join(format!("{}.part", request.filename));
        reject_symlink(&metadata_path)
            .await
            .map_err(invalid_staging)?;
        reject_symlink(&part_path).await.map_err(invalid_staging)?;

        let existing = match fs::read(&metadata_path).await {
            Ok(bytes) => Some(
                serde_json::from_slice::<StagedImportMetadata>(&bytes).map_err(|error| {
                    ModelImportError::Invalid(format!(
                        "staged model import metadata is invalid: {error}"
                    ))
                })?,
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(ModelImportError::Internal(format!(
                    "failed to read staged model import metadata: {error}"
                )));
            }
        };
        if let Some(existing) = existing.as_ref()
            && (existing.filename != request.filename
                || existing.total_bytes != request.total_bytes)
        {
            return Err(ModelImportError::Conflict(
                    "A staged import with the same filename has different metadata. Discard it before starting a different file."
                        .to_string(),
                ));
        }

        let uploaded_bytes = if existing.is_none() {
            if staged_metadata_count(&staging_root)
                .await
                .map_err(internal_error)?
                >= MAX_STAGED_IMPORTS
            {
                return Err(ModelImportError::Conflict(format!(
                    "The model import staging area already contains the maximum of {MAX_STAGED_IMPORTS} sessions. Discard an existing session before starting another."
                )));
            }
            let orphan_bytes = file_size(&part_path).await.map_err(internal_error)?;
            storage
                .ensure_capacity(request.total_bytes, orphan_bytes)
                .await
                .map_err(model_storage_error)?;
            remove_file_if_present(&part_path)
                .await
                .map_err(internal_error)?;
            0
        } else {
            file_size(&part_path).await.map_err(internal_error)?
        };
        if uploaded_bytes > request.total_bytes {
            return Err(ModelImportError::Invalid(
                "staged model import is larger than its declared file size".to_string(),
            ));
        }
        if existing
            .as_ref()
            .is_some_and(|metadata| metadata.sha256 != request.sha256)
            && uploaded_bytes != request.total_bytes
        {
            return Err(ModelImportError::Conflict(
                "The staged import has a different checksum. Use its original checksum while uploading, or discard it before starting a different file."
                    .to_string(),
            ));
        }

        let metadata = StagedImportMetadata {
            version: 1,
            filename: request.filename.clone(),
            total_bytes: request.total_bytes,
            sha256: request.sha256,
            source_url: request.source_url,
            license: request.license,
        };
        if existing.as_ref() != Some(&metadata) {
            write_metadata_atomic(&metadata_path, &metadata)
                .await
                .map_err(internal_error)?;
        }
        let status = ModelImportStatus {
            phase: ModelImportPhase::Ready,
            filename: Some(request.filename),
            uploaded_bytes,
            total_bytes: Some(request.total_bytes),
            error: None,
        };
        *self.status.write().await = status.clone();
        Ok(status)
    }

    pub(crate) async fn append_chunk(
        &self,
        filename: &str,
        offset: u64,
        bytes: &[u8],
    ) -> std::result::Result<ModelImportStatus, ModelImportError> {
        validate_model_filename(filename)
            .map_err(|error| ModelImportError::Invalid(error.to_string()))?;
        if bytes.is_empty() {
            return Err(ModelImportError::Invalid(
                "model import chunk must not be empty".to_string(),
            ));
        }
        if bytes.len() > self.max_chunk_bytes {
            return Err(ModelImportError::Invalid(format!(
                "model import chunk exceeds the configured {} byte limit",
                self.max_chunk_bytes
            )));
        }

        let _guard = self.operation.lock().await;
        let _storage = self.storage.serial().await;
        let staging_root = self.require_staging_root(filename).await?;
        let metadata = read_metadata(&staging_root, filename).await?;
        let part_path = staging_root.join(format!("{filename}.part"));
        reject_symlink(&part_path).await.map_err(invalid_staging)?;
        let current = file_size(&part_path).await.map_err(internal_error)?;
        if current != offset {
            return Err(ModelImportError::OffsetMismatch { expected: current });
        }
        let new_size = current.saturating_add(bytes.len() as u64);
        if new_size > metadata.total_bytes || new_size > self.max_bytes {
            return Err(ModelImportError::Invalid(
                "model import chunk exceeds the declared or configured file size".to_string(),
            ));
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .await
            .map_err(|error| {
                ModelImportError::Internal(format!(
                    "failed to open model import staging file: {error}"
                ))
            })?;
        file.write_all(bytes).await.map_err(|error| {
            ModelImportError::Internal(format!("failed to write model import chunk: {error}"))
        })?;
        file.sync_data().await.map_err(|error| {
            ModelImportError::Internal(format!("failed to sync model import chunk: {error}"))
        })?;
        drop(file);

        let status = ModelImportStatus {
            phase: ModelImportPhase::Uploading,
            filename: Some(filename.to_string()),
            uploaded_bytes: new_size,
            total_bytes: Some(metadata.total_bytes),
            error: None,
        };
        *self.status.write().await = status.clone();
        Ok(status)
    }

    pub(crate) async fn complete(
        &self,
        filename: &str,
    ) -> std::result::Result<ModelImportStatus, ModelImportError> {
        validate_model_filename(filename)
            .map_err(|error| ModelImportError::Invalid(error.to_string()))?;
        let _guard = self.operation.lock().await;
        let (staging_root, metadata, part_path, uploaded_bytes) = {
            let mut storage = self.storage.serial().await;
            let staging_root = self.require_staging_root(filename).await?;
            let mut metadata = read_metadata(&staging_root, filename).await?;
            metadata.license = self
                .license_policy
                .enforce(metadata.license)
                .map_err(|error| ModelImportError::Invalid(error.to_string()))?;
            let part_path = staging_root.join(format!("{filename}.part"));
            reject_symlink(&part_path).await.map_err(invalid_staging)?;
            let uploaded_bytes = file_size(&part_path).await.map_err(internal_error)?;
            if uploaded_bytes != metadata.total_bytes {
                return Err(ModelImportError::Conflict(format!(
                    "Model import has {uploaded_bytes} of {} bytes; upload the remaining bytes before verification.",
                    metadata.total_bytes
                )));
            }
            storage.mark_import_active(filename);
            (staging_root, metadata, part_path, uploaded_bytes)
        };
        *self.status.write().await = ModelImportStatus {
            phase: ModelImportPhase::Verifying,
            filename: Some(filename.to_string()),
            uploaded_bytes,
            total_bytes: Some(metadata.total_bytes),
            error: None,
        };

        let result = async {
            let expected = parse_sha256(&metadata.sha256).map_err(|error| {
                ModelImportError::Invalid(format!("staged import checksum is invalid: {error}"))
            })?;
            let actual = sha256_file(&part_path).await.map_err(internal_error)?;
            if actual != expected {
                return Err(ModelImportError::Invalid(format!(
                    "SHA-256 verification failed for '{}'. Correct the checksum or discard the staged import.",
                    filename
                )));
            }

            let _storage = self.storage.serial().await;
            let destination = self.models_root.join(filename);
            fs::hard_link(&part_path, &destination)
                .await
                .map_err(|error| {
                    ModelImportError::Conflict(format!(
                        "failed to install verified model without overwriting an existing entry: {error}"
                    ))
                })?;
            let source_host = metadata
                .source_url
                .as_deref()
                .and_then(|value| reqwest::Url::parse(value).ok())
                .and_then(|url| url.host_str().map(str::to_string));
            if let Err(error) = write_provenance(
                &self.models_root,
                ModelProvenanceDraft {
                    acquisition: ModelAcquisitionKind::Import,
                    model_index_id: None,
                    filename: filename.to_string(),
                    size_bytes: uploaded_bytes,
                    source_url: metadata.source_url.clone(),
                    source_host,
                    sha256: metadata.sha256.clone(),
                    license: metadata.license.clone(),
                },
            )
            .await
            {
                if let Err(remove_error) = fs::remove_file(&destination).await {
                    tracing::error!(
                        %remove_error,
                        "Failed to roll back imported model after provenance error"
                    );
                }
                return Err(ModelImportError::Internal(format!(
                    "failed to persist verified model provenance: {error}"
                )));
            }
            cleanup_staging(&part_path, &staging_root.join(format!("{filename}.json"))).await;
            self.catalog_revision.fetch_add(1, Ordering::AcqRel);
            Ok(ModelImportStatus {
                phase: ModelImportPhase::Complete,
                filename: Some(filename.to_string()),
                uploaded_bytes,
                total_bytes: Some(metadata.total_bytes),
                error: None,
            })
        }
        .await;
        self.storage.release_import(filename).await;
        match result {
            Ok(status) => {
                *self.status.write().await = status.clone();
                Ok(status)
            }
            Err(error) => {
                *self.status.write().await = ModelImportStatus {
                    phase: ModelImportPhase::Error,
                    filename: Some(filename.to_string()),
                    uploaded_bytes,
                    total_bytes: Some(metadata.total_bytes),
                    error: Some(error.to_string()),
                };
                Err(error)
            }
        }
    }

    pub(crate) async fn discard(
        &self,
        filename: &str,
    ) -> std::result::Result<(), ModelImportError> {
        validate_model_filename(filename)
            .map_err(|error| ModelImportError::Invalid(error.to_string()))?;
        let _guard = self.operation.lock().await;
        let _storage = self.storage.serial().await;
        let staging_root = self.require_staging_root(filename).await?;
        let part_path = staging_root.join(format!("{filename}.part"));
        let metadata_path = staging_root.join(format!("{filename}.json"));
        let mut removed = false;
        for path in [
            part_path,
            metadata_path.clone(),
            metadata_path.with_extension("json.tmp"),
        ] {
            removed |= remove_staged_file(&path).await.map_err(internal_error)?;
        }
        if !removed {
            return Err(ModelImportError::NotFound(format!(
                "No staged import exists for '{filename}'."
            )));
        }
        if self.status.read().await.filename.as_deref() == Some(filename) {
            *self.status.write().await = ModelImportStatus::default();
        }
        Ok(())
    }

    pub(crate) async fn staged(&self) -> Vec<StagedModelImport> {
        match self.scan_staged().await {
            Ok(staged) => staged,
            Err(error) => {
                tracing::warn!(%error, "Failed to scan staged model imports");
                Vec::new()
            }
        }
    }

    async fn ensure_staging_root(&self) -> std::result::Result<PathBuf, ModelImportError> {
        fs::create_dir_all(&self.models_root)
            .await
            .map_err(|error| {
                ModelImportError::Internal(format!(
                    "failed to create the model catalog directory: {error}"
                ))
            })?;
        let staging_root = self.models_root.join(STAGING_DIRECTORY);
        fs::create_dir_all(&staging_root).await.map_err(|error| {
            ModelImportError::Internal(format!(
                "failed to create the model import staging directory: {error}"
            ))
        })?;
        secure_staging_directory(&staging_root)
            .await
            .map_err(invalid_staging)?;
        Ok(staging_root)
    }

    async fn require_staging_root(
        &self,
        filename: &str,
    ) -> std::result::Result<PathBuf, ModelImportError> {
        let staging_root = self.models_root.join(STAGING_DIRECTORY);
        match fs::symlink_metadata(&staging_root).await {
            Ok(_) => secure_staging_directory(&staging_root)
                .await
                .map_err(invalid_staging)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ModelImportError::NotFound(format!(
                    "No staged import exists for '{filename}'."
                )));
            }
            Err(error) => {
                return Err(ModelImportError::Internal(format!(
                    "failed to inspect model import staging directory: {error}"
                )));
            }
        }
        Ok(staging_root)
    }

    async fn scan_staged(&self) -> Result<Vec<StagedModelImport>> {
        let staging_root = self.models_root.join(STAGING_DIRECTORY);
        match fs::symlink_metadata(&staging_root).await {
            Ok(_) => secure_staging_directory(&staging_root).await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error).context("failed to inspect import staging directory"),
        }
        let mut entries = fs::read_dir(&staging_root)
            .await
            .context("failed to open import staging directory")?;
        let mut staged = Vec::new();
        let mut inspected = 0_usize;
        while inspected < MAX_STAGING_ENTRIES_TO_INSPECT && staged.len() < MAX_STAGED_IMPORTS {
            let Some(entry) = entries.next_entry().await? else {
                break;
            };
            inspected += 1;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json")
                || reject_symlink(&path).await.is_err()
            {
                continue;
            }
            let Ok(bytes) = fs::read(&path).await else {
                continue;
            };
            let Ok(metadata) = serde_json::from_slice::<StagedImportMetadata>(&bytes) else {
                continue;
            };
            if validate_model_filename(&metadata.filename).is_err()
                || metadata.total_bytes == 0
                || metadata.total_bytes > self.max_bytes
                || path.file_stem().and_then(|value| value.to_str())
                    != Some(metadata.filename.as_str())
            {
                continue;
            }
            let part_path = staging_root.join(format!("{}.part", metadata.filename));
            if reject_symlink(&part_path).await.is_err() {
                continue;
            }
            let (uploaded_bytes, modified_at) = match fs::metadata(&part_path).await {
                Ok(part_metadata)
                    if part_metadata.is_file() && part_metadata.len() <= metadata.total_bytes =>
                {
                    (part_metadata.len(), modified_seconds(&part_metadata))
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let modified_at = fs::metadata(&path)
                        .await
                        .ok()
                        .and_then(|metadata| modified_seconds(&metadata));
                    (0, modified_at)
                }
                _ => continue,
            };
            staged.push(StagedModelImport {
                filename: metadata.filename,
                uploaded_bytes,
                total_bytes: metadata.total_bytes,
                modified_at,
            });
        }
        staged.sort_by(|left, right| left.filename.cmp(&right.filename));
        Ok(staged)
    }
}

async fn staged_metadata_count(staging_root: &Path) -> Result<usize> {
    let mut entries = fs::read_dir(staging_root)
        .await
        .context("failed to count staged model imports")?;
    let mut count = 0;
    let mut inspected = 0;
    while inspected < MAX_STAGING_ENTRIES_TO_INSPECT {
        let Some(entry) = entries.next_entry().await? else {
            return Ok(count);
        };
        inspected += 1;
        if entry.path().extension().and_then(|value| value.to_str()) == Some("json") {
            count += 1;
            if count == MAX_STAGED_IMPORTS {
                return Ok(count);
            }
        }
    }
    Ok(MAX_STAGED_IMPORTS)
}

fn modified_seconds(metadata: &std::fs::Metadata) -> Option<u64> {
    metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
}

fn validate_import_request(
    request: ModelImportRequest,
    max_bytes: u64,
    license_policy: &ModelLicensePolicy,
) -> std::result::Result<ModelImportRequest, ModelImportError> {
    let filename = request.filename.trim().to_string();
    validate_model_filename(&filename)
        .map_err(|error| ModelImportError::Invalid(error.to_string()))?;
    if request.total_bytes == 0 || request.total_bytes > max_bytes {
        return Err(ModelImportError::Invalid(format!(
            "model import size must be between 1 and {max_bytes} bytes"
        )));
    }
    let sha256 = request.sha256.trim().to_ascii_lowercase();
    parse_sha256(&sha256).map_err(|error| ModelImportError::Invalid(error.to_string()))?;
    let source_url = normalize_source_url(request.source_url)
        .map_err(|error| ModelImportError::Invalid(error.to_string()))?;
    let license = license_policy
        .enforce(request.license)
        .map_err(|error| ModelImportError::Invalid(error.to_string()))?;
    Ok(ModelImportRequest {
        filename,
        total_bytes: request.total_bytes,
        sha256,
        source_url,
        license,
    })
}

async fn read_metadata(
    staging_root: &Path,
    filename: &str,
) -> std::result::Result<StagedImportMetadata, ModelImportError> {
    let metadata_path = staging_root.join(format!("{filename}.json"));
    reject_symlink(&metadata_path)
        .await
        .map_err(invalid_staging)?;
    let bytes = fs::read(&metadata_path).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ModelImportError::NotFound(format!("No staged import exists for '{filename}'."))
        } else {
            ModelImportError::Internal(format!(
                "failed to read staged model import metadata: {error}"
            ))
        }
    })?;
    let metadata: StagedImportMetadata = serde_json::from_slice(&bytes).map_err(|error| {
        ModelImportError::Invalid(format!("staged model import metadata is invalid: {error}"))
    })?;
    if metadata.filename != filename {
        return Err(ModelImportError::Invalid(
            "staged model import metadata does not match its filename".to_string(),
        ));
    }
    Ok(metadata)
}

fn parse_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(anyhow!(
            "SHA-256 must contain exactly 64 hexadecimal characters"
        ));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    Ok(output)
}

async fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    let mut file = fs::File::open(path)
        .await
        .context("failed to open staged model import for verification")?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .context("failed to read staged model import for verification")?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

async fn write_metadata_atomic(path: &Path, metadata: &StagedImportMetadata) -> Result<()> {
    reject_symlink(path).await?;
    let temporary = path.with_extension("json.tmp");
    reject_symlink(&temporary).await?;
    remove_file_if_present(&temporary).await?;
    fs::write(&temporary, serde_json::to_vec(metadata)?)
        .await
        .context("failed to write model import metadata")?;
    fs::rename(&temporary, path)
        .await
        .context("failed to commit model import metadata")
}

async fn file_size(path: &Path) -> Result<u64> {
    match fs::metadata(path).await {
        Ok(metadata) if metadata.is_file() => Ok(metadata.len()),
        Ok(_) => Err(anyhow!("model import staging entry is not a file")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(error).context("failed to inspect model import staging file"),
    }
}

async fn secure_staging_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .await
        .context("failed to inspect model import staging directory")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!(
            "model import staging path must be a real directory"
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .await
            .context("failed to set model import staging permissions")?;
    }
    Ok(())
}

async fn reject_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(anyhow!("model import staging entries must not be symlinks"))
        }
        Ok(metadata) if !metadata.is_file() => {
            Err(anyhow!("model import staging entry is not a regular file"))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("failed to inspect model import staging entry"),
    }
}

async fn remove_file_if_present(path: &Path) -> Result<bool> {
    match fs::remove_file(path).await {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("failed to remove model import staging file"),
    }
}

async fn remove_staged_file(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.is_file() || metadata.file_type().is_symlink() => {
            fs::remove_file(path)
                .await
                .context("failed to remove model import staging file")?;
            Ok(true)
        }
        Ok(_) => Err(anyhow!("model import staging entry is not a file")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("failed to inspect model import staging file"),
    }
}

async fn cleanup_staging(part_path: &Path, metadata_path: &Path) {
    for path in [
        part_path,
        metadata_path,
        &metadata_path.with_extension("json.tmp"),
    ] {
        if let Err(error) = fs::remove_file(path).await
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %path.display(), %error, "Failed to clean model import staging file");
        }
    }
}

fn invalid_staging(error: impl std::fmt::Display) -> ModelImportError {
    ModelImportError::Invalid(format!("model import staging is unsafe: {error}"))
}

fn internal_error(error: impl std::fmt::Display) -> ModelImportError {
    ModelImportError::Internal(error.to_string())
}

fn model_storage_error(error: ModelStorageError) -> ModelImportError {
    match error {
        quota @ ModelStorageError::Quota { .. } => ModelImportError::Conflict(quota.to_string()),
        ModelStorageError::Internal(message) => ModelImportError::Internal(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(filename: &str, bytes: &[u8]) -> ModelImportRequest {
        ModelImportRequest {
            filename: filename.to_string(),
            total_bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
            source_url: None,
            license: None,
        }
    }

    fn manager(root: &Path) -> Arc<ModelImportManager> {
        ModelImportManager::new(root.to_path_buf(), 1024 * 1024, 16 * 1024).unwrap()
    }

    #[tokio::test]
    async fn import_begin_enforces_and_canonicalizes_the_license_allowlist() {
        let bytes = b"licensed model";
        let temp = tempfile::tempdir().unwrap();
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);
        let policy = Arc::new(
            ModelLicensePolicy::new(vec!["Apache-2.0".to_string(), "MIT".to_string()]).unwrap(),
        );
        let manager = ModelImportManager::with_storage_and_license_policy(
            temp.path().to_path_buf(),
            1024,
            128,
            storage,
            policy,
        )
        .unwrap();

        assert!(matches!(
            manager.begin(request("missing.gguf", bytes)).await,
            Err(ModelImportError::Invalid(message)) if message.contains("required")
        ));
        let mut disallowed = request("disallowed.gguf", bytes);
        disallowed.license = Some("GPL-3.0-only".to_string());
        assert!(matches!(
            manager.begin(disallowed).await,
            Err(ModelImportError::Invalid(message)) if message.contains("not allowed")
        ));
        let mut allowed = request("allowed.gguf", bytes);
        allowed.license = Some("apache-2.0".to_string());
        manager.begin(allowed).await.unwrap();
        let metadata = read_metadata(&temp.path().join(STAGING_DIRECTORY), "allowed.gguf")
            .await
            .unwrap();
        assert_eq!(metadata.license.as_deref(), Some("Apache-2.0"));
    }

    #[tokio::test]
    async fn import_completion_rechecks_policy_after_a_restart() {
        let bytes = b"staged before policy change";
        let temp = tempfile::tempdir().unwrap();
        let first = manager(temp.path());
        first
            .begin(request("policy-change.gguf", bytes))
            .await
            .unwrap();
        first
            .append_chunk("policy-change.gguf", 0, bytes)
            .await
            .unwrap();
        drop(first);

        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);
        let policy = Arc::new(ModelLicensePolicy::new(vec!["MIT".to_string()]).unwrap());
        let restarted = ModelImportManager::with_storage_and_license_policy(
            temp.path().to_path_buf(),
            1024 * 1024,
            16 * 1024,
            storage,
            policy,
        )
        .unwrap();

        let error = restarted.complete("policy-change.gguf").await.unwrap_err();

        assert!(matches!(error, ModelImportError::Invalid(_)));
        assert!(error.to_string().contains("required"));
        assert!(!temp.path().join("policy-change.gguf").exists());
    }

    #[tokio::test]
    async fn imports_chunks_and_installs_only_after_verification() {
        let bytes = b"GGUF deterministic local import".repeat(1024);
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let midpoint = bytes.len() / 2;
        let mut import = request("local.gguf", &bytes);
        import.source_url =
            Some("https://huggingface.co/acme/model?token=secret#weights".to_string());
        import.license = Some("Apache-2.0".to_string());

        let started = manager.begin(import).await.unwrap();
        assert_eq!(started.uploaded_bytes, 0);
        manager
            .append_chunk("local.gguf", 0, &bytes[..midpoint])
            .await
            .unwrap();
        assert!(!temp.path().join("local.gguf").exists());
        manager
            .append_chunk("local.gguf", midpoint as u64, &bytes[midpoint..])
            .await
            .unwrap();
        let complete = manager.complete("local.gguf").await.unwrap();

        assert_eq!(complete.phase, ModelImportPhase::Complete);
        assert_eq!(
            fs::read(temp.path().join("local.gguf")).await.unwrap(),
            bytes
        );
        assert!(manager.staged().await.is_empty());
        assert_eq!(manager.catalog_revision(), 1);
        let provenance = super::super::model_provenance::read_provenance(
            temp.path(),
            "local.gguf",
            bytes.len() as u64,
        )
        .unwrap()
        .unwrap();
        assert_eq!(provenance.acquisition, ModelAcquisitionKind::Import);
        assert_eq!(provenance.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(
            provenance.source_url.as_deref(),
            Some("https://huggingface.co/acme/model")
        );
    }

    #[tokio::test]
    async fn shared_storage_quota_reserves_the_declared_import_size() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("installed.gguf"), vec![0_u8; 60])
            .await
            .unwrap();
        let storage = ModelStorageManager::new(temp.path().to_path_buf(), 100, 0);
        let manager = ModelImportManager::with_storage(
            temp.path().to_path_buf(),
            1024,
            128,
            Arc::clone(&storage),
        )
        .unwrap();

        assert!(matches!(
            manager.begin(request("too-large.gguf", &[0_u8; 41])).await,
            Err(ModelImportError::Conflict(_))
        ));
        assert!(
            !temp
                .path()
                .join(STAGING_DIRECTORY)
                .join("too-large.gguf.json")
                .exists()
        );
        manager
            .begin(request("fits.gguf", &[0_u8; 40]))
            .await
            .unwrap();
        let status = storage.snapshot().await.unwrap();
        assert_eq!(status.used_bytes, 60);
        assert_eq!(status.reserved_bytes, 40);
        assert_eq!(status.committed_bytes, 100);
    }

    #[tokio::test]
    async fn reports_the_authoritative_offset_without_mutating_data() {
        let bytes = b"GGUF offset fixture".repeat(128);
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        manager.begin(request("offset.gguf", &bytes)).await.unwrap();
        manager
            .append_chunk("offset.gguf", 0, &bytes[..100])
            .await
            .unwrap();

        let error = manager
            .append_chunk("offset.gguf", 50, &bytes[100..200])
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            ModelImportError::OffsetMismatch { expected: 100 }
        ));
        assert_eq!(manager.staged().await[0].uploaded_bytes, 100);
    }

    #[tokio::test]
    async fn resumes_after_manager_restart_and_allows_checksum_correction() {
        let bytes = b"GGUF restart fixture".repeat(256);
        let temp = tempfile::tempdir().unwrap();
        let first = manager(temp.path());
        let mut wrong = request("restart.gguf", &bytes);
        wrong.sha256 = "00".repeat(32);
        first.begin(wrong).await.unwrap();
        first.append_chunk("restart.gguf", 0, &bytes).await.unwrap();
        assert!(matches!(
            first.complete("restart.gguf").await,
            Err(ModelImportError::Invalid(_))
        ));

        let restarted = manager(temp.path());
        let resumed = restarted
            .begin(request("restart.gguf", &bytes))
            .await
            .unwrap();
        assert_eq!(resumed.uploaded_bytes, bytes.len() as u64);
        restarted.complete("restart.gguf").await.unwrap();

        assert_eq!(
            fs::read(temp.path().join("restart.gguf")).await.unwrap(),
            bytes
        );
    }

    #[tokio::test]
    async fn rejects_checksum_changes_while_a_file_is_partial() {
        let bytes = b"GGUF checksum identity fixture";
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        manager
            .begin(request("identity.gguf", bytes))
            .await
            .unwrap();
        manager
            .append_chunk("identity.gguf", 0, &bytes[..8])
            .await
            .unwrap();
        let mut changed = request("identity.gguf", bytes);
        changed.sha256 = "00".repeat(32);

        assert!(matches!(
            manager.begin(changed).await,
            Err(ModelImportError::Conflict(_))
        ));
        assert_eq!(manager.staged().await[0].uploaded_bytes, 8);
    }

    #[tokio::test]
    async fn inventories_and_discards_a_session_before_its_first_chunk() {
        let bytes = b"GGUF pending fixture";
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        manager.begin(request("pending.gguf", bytes)).await.unwrap();

        let staged = manager.staged().await;
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].filename, "pending.gguf");
        assert_eq!(staged[0].uploaded_bytes, 0);
        manager.discard("pending.gguf").await.unwrap();
        assert!(manager.staged().await.is_empty());
    }

    #[tokio::test]
    async fn bounds_the_number_of_staged_sessions() {
        let temp = tempfile::tempdir().unwrap();
        let staging = temp.path().join(STAGING_DIRECTORY);
        std::fs::create_dir(&staging).unwrap();
        for index in 0..MAX_STAGED_IMPORTS {
            std::fs::write(staging.join(format!("session-{index}.json")), b"{}").unwrap();
        }
        let manager = manager(temp.path());

        assert!(matches!(
            manager.begin(request("one-more.gguf", b"GGUF")).await,
            Err(ModelImportError::Conflict(_))
        ));
    }

    #[tokio::test]
    async fn discards_partial_imports() {
        let bytes = b"GGUF discard fixture".repeat(64);
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        manager
            .begin(request("discard.gguf", &bytes))
            .await
            .unwrap();
        manager
            .append_chunk("discard.gguf", 0, &bytes[..128])
            .await
            .unwrap();

        assert_eq!(manager.staged().await.len(), 1);
        manager.discard("discard.gguf").await.unwrap();

        assert!(manager.staged().await.is_empty());
        assert!(matches!(
            manager.discard("discard.gguf").await,
            Err(ModelImportError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn rejects_oversized_files_chunks_and_unsafe_names() {
        let temp = tempfile::tempdir().unwrap();
        let manager = ModelImportManager::new(temp.path().to_path_buf(), 100, 16).unwrap();

        assert!(matches!(
            manager
                .begin(ModelImportRequest {
                    filename: "../bad.gguf".to_string(),
                    total_bytes: 10,
                    sha256: "00".repeat(32),
                    source_url: None,
                    license: None,
                })
                .await,
            Err(ModelImportError::Invalid(_))
        ));
        assert!(matches!(
            manager
                .begin(ModelImportRequest {
                    filename: "large.gguf".to_string(),
                    total_bytes: 101,
                    sha256: "00".repeat(32),
                    source_url: None,
                    license: None,
                })
                .await,
            Err(ModelImportError::Invalid(_))
        ));
        manager
            .begin(ModelImportRequest {
                filename: "chunk.gguf".to_string(),
                total_bytes: 32,
                sha256: "00".repeat(32),
                source_url: None,
                license: None,
            })
            .await
            .unwrap();
        assert!(matches!(
            manager.append_chunk("chunk.gguf", 0, &[0_u8; 17]).await,
            Err(ModelImportError::Invalid(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_a_symlinked_import_staging_directory() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), temp.path().join(STAGING_DIRECTORY)).unwrap();
        let manager = manager(temp.path());
        let bytes = b"GGUF";

        assert!(matches!(
            manager.begin(request("safe.gguf", bytes)).await,
            Err(ModelImportError::Invalid(_))
        ));
        assert!(outside.path().read_dir().unwrap().next().is_none());
    }
}
