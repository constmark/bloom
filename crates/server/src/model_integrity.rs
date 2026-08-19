//! Cancellable, on-demand integrity verification for acquired model files.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt as _;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use super::model_manager::ModelCatalog;
use super::model_package::{ModelPackageFile, package_digest, validate_package_path};
#[cfg(test)]
use super::model_provenance::read_provenance;
use super::model_provenance::{read_provenance_details, record_integrity_result};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;
const STATUS_UPDATE_BYTES: u64 = 8 * 1024 * 1024;
const STATUS_UPDATE_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelIntegrityPhase {
    Idle,
    Queued,
    Verifying,
    Complete,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelIntegrityStatus {
    pub phase: ModelIntegrityPhase,
    pub model_id: Option<String>,
    pub checked_bytes: u64,
    pub total_bytes: Option<u64>,
    pub expected_sha256: Option<String>,
    pub actual_sha256: Option<String>,
    pub matches_expected: Option<bool>,
    pub verified_at: Option<u64>,
    pub error: Option<String>,
}

impl Default for ModelIntegrityStatus {
    fn default() -> Self {
        Self {
            phase: ModelIntegrityPhase::Idle,
            model_id: None,
            checked_bytes: 0,
            total_bytes: None,
            expected_sha256: None,
            actual_sha256: None,
            matches_expected: None,
            verified_at: None,
            error: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelIntegrityError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Internal(String),
}

struct ActiveVerification {
    model_id: String,
    cancellation: CancellationToken,
}

struct PreparedVerification {
    model_id: String,
    target: VerificationTarget,
    total_bytes: u64,
    expected_sha256: String,
}

enum VerificationTarget {
    File(PathBuf),
    Package {
        root: PathBuf,
        files: Vec<ModelPackageFile>,
    },
}

pub(crate) struct ModelIntegrityManager {
    models_root: PathBuf,
    status: RwLock<ModelIntegrityStatus>,
    active: Mutex<Option<ActiveVerification>>,
    catalog_revision: AtomicU64,
}

impl ModelIntegrityManager {
    pub(crate) fn new(models_root: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            models_root,
            status: RwLock::new(ModelIntegrityStatus::default()),
            active: Mutex::new(None),
            catalog_revision: AtomicU64::new(0),
        })
    }

    pub(crate) fn catalog_revision(&self) -> u64 {
        self.catalog_revision.load(Ordering::Acquire)
    }

    pub(crate) async fn status(&self) -> ModelIntegrityStatus {
        self.status.read().await.clone()
    }

    pub(crate) async fn is_active(&self, model_id: &str) -> bool {
        self.active
            .lock()
            .await
            .as_ref()
            .is_some_and(|active| active.model_id == model_id)
    }

    pub(crate) async fn start(
        self: &Arc<Self>,
        model_id: &str,
    ) -> Result<ModelIntegrityStatus, ModelIntegrityError> {
        let model_id = model_id.trim().to_string();
        let root = self.models_root.clone();
        let resolve_id = model_id.clone();
        let prepared =
            tokio::task::spawn_blocking(move || prepare_verification(&root, &resolve_id))
                .await
                .map_err(|error| {
                    ModelIntegrityError::Internal(format!(
                        "model integrity preparation task failed: {error}"
                    ))
                })??;

        let mut active = self.active.lock().await;
        if active.is_some() {
            return Err(ModelIntegrityError::Conflict(
                "Another model integrity verification is already in progress.".to_string(),
            ));
        }
        let cancellation = CancellationToken::new();
        *active = Some(ActiveVerification {
            model_id: prepared.model_id.clone(),
            cancellation: cancellation.clone(),
        });
        let queued = ModelIntegrityStatus {
            phase: ModelIntegrityPhase::Queued,
            model_id: Some(prepared.model_id.clone()),
            checked_bytes: 0,
            total_bytes: Some(prepared.total_bytes),
            expected_sha256: Some(prepared.expected_sha256.clone()),
            actual_sha256: None,
            matches_expected: None,
            verified_at: None,
            error: None,
        };
        *self.status.write().await = queued.clone();
        drop(active);

        let manager = Arc::clone(self);
        tokio::spawn(async move {
            manager.run(prepared, cancellation).await;
        });
        Ok(queued)
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

    async fn run(&self, prepared: PreparedVerification, cancellation: CancellationToken) {
        *self.status.write().await = ModelIntegrityStatus {
            phase: ModelIntegrityPhase::Verifying,
            model_id: Some(prepared.model_id.clone()),
            checked_bytes: 0,
            total_bytes: Some(prepared.total_bytes),
            expected_sha256: Some(prepared.expected_sha256.clone()),
            actual_sha256: None,
            matches_expected: None,
            verified_at: None,
            error: None,
        };

        let outcome = match self.verify(&prepared, &cancellation).await {
            Ok(_) if cancellation.is_cancelled() => Err(VerificationFailure::Cancelled),
            outcome => outcome,
        };
        let status = match outcome {
            Ok(actual_sha256) => {
                let matches_expected =
                    actual_sha256.eq_ignore_ascii_case(&prepared.expected_sha256);
                match record_integrity_result(
                    &self.models_root,
                    &prepared.model_id,
                    prepared.total_bytes,
                    &prepared.expected_sha256,
                    &actual_sha256,
                )
                .await
                {
                    Ok(provenance) => {
                        self.catalog_revision.fetch_add(1, Ordering::AcqRel);
                        ModelIntegrityStatus {
                            phase: ModelIntegrityPhase::Complete,
                            model_id: Some(prepared.model_id.clone()),
                            checked_bytes: prepared.total_bytes,
                            total_bytes: Some(prepared.total_bytes),
                            expected_sha256: Some(prepared.expected_sha256),
                            actual_sha256: Some(actual_sha256),
                            matches_expected: Some(matches_expected),
                            verified_at: if matches_expected {
                                provenance.last_verified_at
                            } else {
                                provenance.integrity_mismatch_at
                            },
                            error: None,
                        }
                    }
                    Err(error) => ModelIntegrityStatus {
                        phase: ModelIntegrityPhase::Error,
                        model_id: Some(prepared.model_id),
                        checked_bytes: prepared.total_bytes,
                        total_bytes: Some(prepared.total_bytes),
                        expected_sha256: Some(prepared.expected_sha256),
                        actual_sha256: Some(actual_sha256),
                        matches_expected: Some(matches_expected),
                        verified_at: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .ok()
                            .map(|duration| duration.as_secs()),
                        error: Some(format!(
                            "Integrity was computed but its durable result could not be recorded: {error}"
                        )),
                    },
                }
            }
            Err(VerificationFailure::Cancelled) => ModelIntegrityStatus {
                phase: ModelIntegrityPhase::Cancelled,
                model_id: Some(prepared.model_id),
                checked_bytes: self.status.read().await.checked_bytes,
                total_bytes: Some(prepared.total_bytes),
                expected_sha256: Some(prepared.expected_sha256),
                actual_sha256: None,
                matches_expected: None,
                verified_at: None,
                error: None,
            },
            Err(VerificationFailure::Error(message)) => ModelIntegrityStatus {
                phase: ModelIntegrityPhase::Error,
                model_id: Some(prepared.model_id),
                checked_bytes: self.status.read().await.checked_bytes,
                total_bytes: Some(prepared.total_bytes),
                expected_sha256: Some(prepared.expected_sha256),
                actual_sha256: None,
                matches_expected: None,
                verified_at: None,
                error: Some(message),
            },
        };
        *self.status.write().await = status;
        self.active.lock().await.take();
    }

    async fn verify(
        &self,
        prepared: &PreparedVerification,
        cancellation: &CancellationToken,
    ) -> Result<String, VerificationFailure> {
        match &prepared.target {
            VerificationTarget::File(path) => {
                let (actual, _) = self
                    .verify_one_file(path, prepared.total_bytes, 0, cancellation)
                    .await?;
                Ok(actual)
            }
            VerificationTarget::Package { root, files } => {
                validate_installed_package_tree(root, files).map_err(VerificationFailure::error)?;
                let mut actual_files = Vec::with_capacity(files.len());
                let mut checked_bytes = 0_u64;
                for expected in files {
                    let path = root.join(&expected.filename);
                    let (sha256, checked) = self
                        .verify_one_file(&path, expected.size_bytes, checked_bytes, cancellation)
                        .await?;
                    checked_bytes = checked;
                    actual_files.push(ModelPackageFile {
                        filename: expected.filename.clone(),
                        size_bytes: expected.size_bytes,
                        sha256,
                    });
                }
                if checked_bytes != prepared.total_bytes {
                    return Err(VerificationFailure::error(
                        "Model package size changed during integrity verification.",
                    ));
                }
                validate_installed_package_tree(root, files).map_err(VerificationFailure::error)?;
                package_digest(&actual_files).map_err(VerificationFailure::error)
            }
        }
    }

    async fn verify_one_file(
        &self,
        path: &Path,
        expected_bytes: u64,
        checked_before: u64,
        cancellation: &CancellationToken,
    ) -> Result<(String, u64), VerificationFailure> {
        let mut file = open_model_file(path)
            .await
            .map_err(VerificationFailure::error)?;
        let initial_metadata = file.metadata().await.map_err(|error| {
            VerificationFailure::error(format!("failed to inspect catalog model: {error}"))
        })?;
        if !initial_metadata.is_file() || initial_metadata.len() != expected_bytes {
            return Err(VerificationFailure::error(
                "Catalog model changed before integrity verification started.",
            ));
        }

        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
        let mut file_bytes = 0_u64;
        let mut last_reported = checked_before;
        let mut last_update = Instant::now();
        loop {
            let read = tokio::select! {
                _ = cancellation.cancelled() => return Err(VerificationFailure::Cancelled),
                result = file.read(&mut buffer) => result,
            }
            .map_err(|error| {
                VerificationFailure::error(format!("failed to read catalog model: {error}"))
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            file_bytes = file_bytes.saturating_add(read as u64);
            let checked_bytes = checked_before.saturating_add(file_bytes);
            if checked_bytes.saturating_sub(last_reported) >= STATUS_UPDATE_BYTES
                || last_update.elapsed() >= STATUS_UPDATE_INTERVAL
            {
                self.status.write().await.checked_bytes = checked_bytes;
                last_reported = checked_bytes;
                last_update = Instant::now();
            }
            #[cfg(test)]
            tokio::task::yield_now().await;
        }
        if cancellation.is_cancelled() {
            return Err(VerificationFailure::Cancelled);
        }
        if file_bytes != expected_bytes {
            return Err(VerificationFailure::error(
                "Catalog model size changed during integrity verification.",
            ));
        }
        ensure_file_unchanged(path, &initial_metadata, &file).await?;
        Ok((
            format!("{:x}", hasher.finalize()),
            checked_before.saturating_add(file_bytes),
        ))
    }
}

fn prepare_verification(
    models_root: &Path,
    model_id: &str,
) -> Result<PreparedVerification, ModelIntegrityError> {
    let catalog = ModelCatalog::scan(models_root, None)
        .map_err(|error| ModelIntegrityError::Internal(error.to_string()))?;
    let catalog_entry = catalog
        .models
        .iter()
        .find(|entry| entry.id == model_id)
        .ok_or_else(|| {
            ModelIntegrityError::Invalid(format!("model '{model_id}' was not found in the catalog"))
        })?;
    if !catalog_entry.size_complete || catalog_entry.size_bytes == 0 {
        return Err(ModelIntegrityError::Invalid(
            "The model directory is too large or deep for complete integrity verification."
                .to_string(),
        ));
    }
    let path = ModelCatalog::resolve(models_root, model_id)
        .map_err(|error| ModelIntegrityError::Invalid(error.to_string()))?;
    let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
        ModelIntegrityError::Internal(format!("failed to inspect catalog model: {error}"))
    })?;
    let details = read_provenance_details(models_root, model_id, catalog_entry.size_bytes)
        .map_err(|error| {
            ModelIntegrityError::Invalid(format!(
                "Stored provenance metadata cannot be verified: {error}"
            ))
        })?
        .ok_or_else(|| {
            ModelIntegrityError::Invalid(
                "No verified acquisition checksum is recorded for this model.".to_string(),
            )
        })?;
    let target = if metadata.is_file() && details.files.is_empty() {
        VerificationTarget::File(path)
    } else if metadata.is_dir() && !details.files.is_empty() {
        validate_installed_package_tree(&path, &details.files)
            .map_err(|error| ModelIntegrityError::Invalid(error.to_string()))?;
        VerificationTarget::Package {
            root: path,
            files: details.files,
        }
    } else {
        return Err(ModelIntegrityError::Invalid(
            "Stored provenance does not match the catalog entry kind.".to_string(),
        ));
    };
    Ok(PreparedVerification {
        model_id: model_id.to_string(),
        target,
        total_bytes: catalog_entry.size_bytes,
        expected_sha256: details.provenance.sha256,
    })
}

fn validate_installed_package_tree(root: &Path, files: &[ModelPackageFile]) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(root)
        .map_err(|error| format!("failed to inspect model package: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("Model package path must be a real directory.".to_string());
    }
    let mut expected_files = std::collections::HashSet::new();
    let mut expected_directories = std::collections::HashSet::new();
    for file in files {
        validate_package_path(&file.filename).map_err(|error| error.to_string())?;
        expected_files.insert(file.filename.clone());
        let mut parent = Path::new(&file.filename).parent();
        while let Some(path) = parent {
            if path.as_os_str().is_empty() {
                break;
            }
            expected_directories.insert(path.to_string_lossy().replace('\\', "/"));
            parent = path.parent();
        }
    }
    let mut discovered = std::collections::HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    let mut inspected = 0_usize;
    while let Some(directory) = stack.pop() {
        let entries = std::fs::read_dir(&directory)
            .map_err(|error| format!("failed to read model package: {error}"))?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("failed to read model package: {error}"))?;
            inspected = inspected.saturating_add(1);
            if inspected > files.len().saturating_mul(3).saturating_add(16) {
                return Err("Model package contains too many filesystem entries.".to_string());
            }
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)
                .map_err(|error| format!("failed to inspect model package: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err("Model package must not contain symlinks.".to_string());
            }
            let relative = path
                .strip_prefix(root)
                .map_err(|_| "Model package entry escaped its root.".to_string())?
                .to_str()
                .ok_or_else(|| "Model package filename is not UTF-8.".to_string())?
                .replace('\\', "/");
            if metadata.is_dir() {
                if !expected_directories.contains(&relative) {
                    return Err("Model package contains an unexpected directory.".to_string());
                }
                stack.push(path);
            } else if metadata.is_file() {
                if !expected_files.contains(&relative) {
                    return Err("Model package contains an unexpected file.".to_string());
                }
                discovered.insert(relative);
            } else {
                return Err("Model package contains an unsupported entry.".to_string());
            }
        }
    }
    if discovered != expected_files {
        return Err("Model package is incomplete.".to_string());
    }
    Ok(())
}

async fn open_model_file(path: &Path) -> Result<tokio::fs::File, String> {
    let path = path.to_path_buf();
    let file = tokio::task::spawn_blocking(move || {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;

            options.custom_flags(libc::O_NOFOLLOW);
        }
        options.open(path)
    })
    .await
    .map_err(|error| format!("model integrity open task failed: {error}"))?
    .map_err(|error| format!("failed to open catalog model: {error}"))?;
    Ok(tokio::fs::File::from_std(file))
}

async fn ensure_file_unchanged(
    path: &Path,
    initial: &std::fs::Metadata,
    file: &tokio::fs::File,
) -> Result<(), VerificationFailure> {
    let handle_metadata = file.metadata().await.map_err(|error| {
        VerificationFailure::error(format!(
            "failed to re-inspect the verified model handle: {error}"
        ))
    })?;
    let path_metadata = tokio::fs::symlink_metadata(path).await.map_err(|error| {
        VerificationFailure::error(format!(
            "failed to re-inspect the verified catalog entry: {error}"
        ))
    })?;
    if path_metadata.file_type().is_symlink()
        || !path_metadata.is_file()
        || !same_file_identity(initial, &handle_metadata)
        || !same_file_identity(initial, &path_metadata)
        || initial.len() != handle_metadata.len()
        || initial.len() != path_metadata.len()
        || initial.modified().ok() != handle_metadata.modified().ok()
        || initial.modified().ok() != path_metadata.modified().ok()
    {
        return Err(VerificationFailure::error(
            "Catalog model changed during integrity verification.",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn same_file_identity(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;

    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(_left: &std::fs::Metadata, _right: &std::fs::Metadata) -> bool {
    true
}

enum VerificationFailure {
    Cancelled,
    Error(String),
}

impl VerificationFailure {
    fn error(message: impl std::fmt::Display) -> Self {
        Self::Error(message.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::super::model_provenance::{
        ModelAcquisitionKind, ModelPackageProvenanceDraft, ModelProvenanceDraft,
        write_package_provenance, write_provenance,
    };
    use super::*;

    async fn recorded_model(root: &Path, filename: &str, bytes: &[u8], expected_sha256: String) {
        tokio::fs::write(root.join(filename), bytes).await.unwrap();
        write_provenance(
            root,
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Import,
                model_index_id: None,
                filename: filename.to_string(),
                size_bytes: bytes.len() as u64,
                source_url: None,
                source_host: None,
                sha256: expected_sha256,
                license: None,
            },
        )
        .await
        .unwrap();
    }

    async fn wait_for_terminal(manager: &ModelIntegrityManager) -> ModelIntegrityStatus {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = manager.status().await;
                if matches!(
                    status.phase,
                    ModelIntegrityPhase::Complete
                        | ModelIntegrityPhase::Cancelled
                        | ModelIntegrityPhase::Error
                ) {
                    return status;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn verifies_a_recorded_model_and_reports_both_hashes() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = b"verified model bytes".repeat(1024);
        let expected = format!("{:x}", Sha256::digest(&bytes));
        recorded_model(temp.path(), "model.gguf", &bytes, expected.clone()).await;
        let manager = ModelIntegrityManager::new(temp.path().to_path_buf());

        let queued = manager.start("model.gguf").await.unwrap();
        assert_eq!(queued.phase, ModelIntegrityPhase::Queued);
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelIntegrityPhase::Complete);
        assert_eq!(status.expected_sha256.as_deref(), Some(expected.as_str()));
        assert_eq!(status.actual_sha256.as_deref(), Some(expected.as_str()));
        assert_eq!(status.matches_expected, Some(true));
        assert!(status.verified_at.is_some());
        assert!(!manager.is_active("model.gguf").await);
        let provenance = read_provenance(temp.path(), "model.gguf", bytes.len() as u64)
            .unwrap()
            .unwrap();
        assert!(provenance.last_verified_at.is_some());
        assert!(provenance.integrity_mismatch_at.is_none());
    }

    #[tokio::test]
    async fn reports_a_checksum_mismatch_as_a_completed_check() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = b"modified model bytes";
        recorded_model(temp.path(), "model.gguf", bytes, "00".repeat(32)).await;
        let manager = ModelIntegrityManager::new(temp.path().to_path_buf());

        manager.start("model.gguf").await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelIntegrityPhase::Complete);
        assert_eq!(status.matches_expected, Some(false));
        assert_ne!(status.actual_sha256, status.expected_sha256);
        let provenance = read_provenance(temp.path(), "model.gguf", bytes.len() as u64)
            .unwrap()
            .unwrap();
        assert!(provenance.integrity_mismatch_at.is_some());
    }

    #[tokio::test]
    async fn verifies_every_file_in_an_acquired_model_package() {
        let temp = tempfile::tempdir().unwrap();
        let package_root = temp.path().join("package-model");
        tokio::fs::create_dir(&package_root).await.unwrap();
        let config = br#"{"model_type":"qwen2"}"#;
        let weights = b"verified package weights".repeat(1024);
        tokio::fs::write(package_root.join("config.json"), config)
            .await
            .unwrap();
        tokio::fs::write(package_root.join("model.safetensors"), &weights)
            .await
            .unwrap();
        let files = vec![
            ModelPackageFile {
                filename: "config.json".to_string(),
                size_bytes: config.len() as u64,
                sha256: format!("{:x}", Sha256::digest(config)),
            },
            ModelPackageFile {
                filename: "model.safetensors".to_string(),
                size_bytes: weights.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&weights)),
            },
        ];
        let total_bytes = files.iter().map(|file| file.size_bytes).sum();
        let digest = package_digest(&files).unwrap();
        write_package_provenance(
            temp.path(),
            ModelPackageProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some("package-model".to_string()),
                directory: "package-model".to_string(),
                size_bytes: total_bytes,
                source_url: Some(
                    "https://huggingface.co/acme/model/resolve/0123456789abcdef0123456789abcdef01234567/config.json"
                        .to_string(),
                ),
                source_host: Some("huggingface.co".to_string()),
                sha256: digest.clone(),
                license: Some("Apache-2.0".to_string()),
                files,
            },
        )
        .await
        .unwrap();
        let manager = ModelIntegrityManager::new(temp.path().to_path_buf());

        manager.start("package-model").await.unwrap();
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelIntegrityPhase::Complete);
        assert_eq!(status.expected_sha256.as_deref(), Some(digest.as_str()));
        assert_eq!(status.actual_sha256.as_deref(), Some(digest.as_str()));
        assert_eq!(status.matches_expected, Some(true));
        assert_eq!(status.checked_bytes, total_bytes);

        let mut changed = weights;
        changed[0] ^= 0xff;
        tokio::fs::write(package_root.join("model.safetensors"), changed)
            .await
            .unwrap();
        manager.start("package-model").await.unwrap();
        let mismatch = wait_for_terminal(&manager).await;
        assert_eq!(mismatch.phase, ModelIntegrityPhase::Complete);
        assert_eq!(mismatch.matches_expected, Some(false));
    }

    #[tokio::test]
    async fn rejects_models_without_acquisition_provenance() {
        let temp = tempfile::tempdir().unwrap();
        tokio::fs::write(temp.path().join("manual.gguf"), b"manual")
            .await
            .unwrap();
        let manager = ModelIntegrityManager::new(temp.path().to_path_buf());

        assert!(matches!(
            manager.start("manual.gguf").await,
            Err(ModelIntegrityError::Invalid(message))
                if message.contains("No verified acquisition checksum")
        ));
    }

    #[tokio::test]
    async fn cancels_an_active_verification() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = vec![42_u8; 16 * 1024 * 1024];
        let expected = format!("{:x}", Sha256::digest(&bytes));
        recorded_model(temp.path(), "large.gguf", &bytes, expected).await;
        let manager = ModelIntegrityManager::new(temp.path().to_path_buf());

        manager.start("large.gguf").await.unwrap();
        assert!(manager.cancel().await);
        let status = wait_for_terminal(&manager).await;

        assert_eq!(status.phase, ModelIntegrityPhase::Cancelled);
        assert_eq!(status.matches_expected, None);
        assert!(!manager.cancel().await);
    }
}
