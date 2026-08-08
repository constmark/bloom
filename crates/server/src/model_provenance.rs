//! Safe, persistent provenance records for verified catalog acquisitions.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::AsyncWriteExt as _;

use super::model_manager::{validate_catalog_id, validate_model_filename};
use super::model_package::{
    normalize_package_files, package_digest, validate_package_id, ModelPackageFile,
};

pub(crate) const METADATA_DIRECTORY: &str = ".bloom-metadata";
const MAX_PROVENANCE_BYTES: u64 = 128 * 1024;
const MAX_LICENSE_LENGTH: usize = 128;
const MAX_SOURCE_URL_LENGTH: usize = 2_048;
const MAX_MODEL_INDEX_ID_LENGTH: usize = 64;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelAcquisitionKind {
    Download,
    Import,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelProvenance {
    pub acquisition: ModelAcquisitionKind,
    pub model_index_id: Option<String>,
    pub source_url: Option<String>,
    pub source_host: Option<String>,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_count: Option<usize>,
    pub license: Option<String>,
    pub installed_at: u64,
    pub last_verified_at: Option<u64>,
    pub integrity_mismatch_at: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelProvenanceDetails {
    pub provenance: ModelProvenance,
    pub files: Vec<ModelPackageFile>,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelProvenanceDraft {
    pub acquisition: ModelAcquisitionKind,
    pub model_index_id: Option<String>,
    pub filename: String,
    pub size_bytes: u64,
    pub source_url: Option<String>,
    pub source_host: Option<String>,
    pub sha256: String,
    pub license: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelPackageProvenanceDraft {
    pub acquisition: ModelAcquisitionKind,
    pub model_index_id: Option<String>,
    pub directory: String,
    pub size_bytes: u64,
    pub source_url: Option<String>,
    pub source_host: Option<String>,
    pub sha256: String,
    pub license: Option<String>,
    pub files: Vec<ModelPackageFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ModelProvenanceRecord {
    version: u8,
    filename: String,
    size_bytes: u64,
    acquisition: ModelAcquisitionKind,
    #[serde(default)]
    model_index_id: Option<String>,
    source_url: Option<String>,
    source_host: Option<String>,
    sha256: String,
    #[serde(default)]
    files: Vec<ModelPackageFile>,
    license: Option<String>,
    installed_at: u64,
    #[serde(default)]
    last_verified_at: Option<u64>,
    #[serde(default)]
    integrity_mismatch_at: Option<u64>,
}

impl ModelProvenanceRecord {
    fn public(&self) -> ModelProvenance {
        ModelProvenance {
            acquisition: self.acquisition,
            model_index_id: self.model_index_id.clone(),
            source_url: self.source_url.clone(),
            source_host: self.source_host.clone(),
            sha256: self.sha256.clone(),
            file_count: (self.version == 2).then_some(self.files.len()),
            license: self.license.clone(),
            installed_at: self.installed_at,
            last_verified_at: self.last_verified_at,
            integrity_mismatch_at: self.integrity_mismatch_at,
        }
    }
}

pub(crate) fn normalize_license(value: Option<String>) -> Result<Option<String>> {
    normalize_optional_text(value, "model license", MAX_LICENSE_LENGTH)
}

pub(crate) fn normalize_model_index_id(value: Option<String>) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty()
        || value.len() > MAX_MODEL_INDEX_ID_LENGTH
        || !value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (index > 0 && matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err(anyhow!(
            "model index ID must use 1 to {MAX_MODEL_INDEX_ID_LENGTH} lowercase ASCII letters, digits, dots, dashes, or underscores"
        ));
    }
    Ok(Some(value))
}

pub(crate) fn normalize_source_url(value: Option<String>) -> Result<Option<String>> {
    let Some(value) = normalize_optional_text(value, "model source URL", MAX_SOURCE_URL_LENGTH)?
    else {
        return Ok(None);
    };
    let mut url = Url::parse(&value).context("model source URL is invalid")?;
    if url.scheme() != "https" {
        return Err(anyhow!("model source URL must use HTTPS"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(anyhow!("model source URL must not contain credentials"));
    }
    if url.host_str().is_none() {
        return Err(anyhow!("model source URL must include a host"));
    }
    url.set_query(None);
    url.set_fragment(None);
    let normalized = url.to_string();
    if normalized.len() > MAX_SOURCE_URL_LENGTH {
        return Err(anyhow!(
            "model source URL must not exceed {MAX_SOURCE_URL_LENGTH} characters"
        ));
    }
    Ok(Some(normalized))
}

pub(crate) fn sanitized_download_source(url: &Url) -> String {
    let mut source = url.clone();
    source.set_query(None);
    source.set_fragment(None);
    source.to_string()
}

pub(crate) async fn write_provenance(
    models_root: &Path,
    draft: ModelProvenanceDraft,
) -> Result<()> {
    validate_model_filename(&draft.filename)?;
    if draft.size_bytes == 0 {
        return Err(anyhow!("model provenance size must be greater than zero"));
    }
    validate_sha256(&draft.sha256)?;
    let license = normalize_license(draft.license)?;
    let model_index_id = normalize_model_index_id(draft.model_index_id)?;
    let (source_url, source_host) = normalize_source_identity(draft.source_url, draft.source_host)?;
    let installed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs();
    let record = ModelProvenanceRecord {
        version: 1,
        filename: draft.filename.clone(),
        size_bytes: draft.size_bytes,
        acquisition: draft.acquisition,
        model_index_id,
        source_url,
        source_host,
        sha256: draft.sha256.to_ascii_lowercase(),
        files: Vec::new(),
        license,
        installed_at,
        last_verified_at: None,
        integrity_mismatch_at: None,
    };
    publish_record(models_root, &draft.filename, &record).await
}

pub(crate) async fn write_package_provenance(
    models_root: &Path,
    draft: ModelPackageProvenanceDraft,
) -> Result<()> {
    validate_package_id(&draft.directory)?;
    let files = normalize_package_files(draft.files, draft.size_bytes)?;
    let sha256 = package_digest(&files)?;
    if !sha256.eq_ignore_ascii_case(&draft.sha256) {
        return Err(anyhow!(
            "model package provenance digest does not match its file manifest"
        ));
    }
    let license = normalize_license(draft.license)?;
    let model_index_id = normalize_model_index_id(draft.model_index_id)?;
    let (source_url, source_host) = normalize_source_identity(draft.source_url, draft.source_host)?;
    let installed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs();
    let record = ModelProvenanceRecord {
        version: 2,
        filename: draft.directory.clone(),
        size_bytes: draft.size_bytes,
        acquisition: draft.acquisition,
        model_index_id,
        source_url,
        source_host,
        sha256,
        files,
        license,
        installed_at,
        last_verified_at: None,
        integrity_mismatch_at: None,
    };
    publish_record(models_root, &draft.directory, &record).await
}

pub(crate) async fn backup_provenance(
    models_root: &Path,
    filename: &str,
    expected_size: u64,
    backup_path: &Path,
) -> Result<()> {
    validate_catalog_id(filename)?;
    read_record(models_root, filename, expected_size)?
        .ok_or_else(|| anyhow!("model provenance record was not found"))?;
    let metadata_root = models_root.join(METADATA_DIRECTORY);
    let source = metadata_path(&metadata_root, filename);
    let backup_parent = backup_path
        .parent()
        .ok_or_else(|| anyhow!("model provenance backup has no parent"))?;
    let parent_metadata = fs::symlink_metadata(backup_parent)
        .await
        .context("failed to inspect model provenance backup directory")?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(anyhow!(
            "model provenance backup parent must be a real directory"
        ));
    }
    reject_symlink_or_non_file(backup_path).await?;
    if fs::symlink_metadata(backup_path).await.is_ok() {
        return Err(anyhow!("model provenance backup already exists"));
    }
    fs::hard_link(&source, backup_path)
        .await
        .context("failed to create model provenance backup")?;
    sync_directory(backup_parent).await?;
    Ok(())
}

pub(crate) async fn restore_provenance_backup(
    models_root: &Path,
    filename: &str,
    expected_size: u64,
    backup_path: &Path,
) -> Result<()> {
    validate_catalog_id(filename)?;
    let metadata = fs::symlink_metadata(backup_path)
        .await
        .context("failed to inspect model provenance backup")?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_PROVENANCE_BYTES
    {
        return Err(anyhow!(
            "model provenance backup is not a bounded regular file"
        ));
    }
    let bytes = fs::read(backup_path)
        .await
        .context("failed to read model provenance backup")?;
    let record = serde_json::from_slice::<ModelProvenanceRecord>(&bytes)
        .context("model provenance backup is invalid")?;
    let record = validate_record(record, filename, expected_size)?;
    publish_record(models_root, filename, &record).await
}

pub(crate) async fn record_integrity_result(
    models_root: &Path,
    filename: &str,
    expected_size: u64,
    expected_sha256: &str,
    actual_sha256: &str,
) -> Result<ModelProvenance> {
    validate_sha256(expected_sha256)?;
    validate_sha256(actual_sha256)?;
    let mut record = read_record(models_root, filename, expected_size)?
        .ok_or_else(|| anyhow!("model provenance record was not found"))?;
    if !record.sha256.eq_ignore_ascii_case(expected_sha256) {
        return Err(anyhow!(
            "model provenance checksum changed during integrity verification"
        ));
    }
    let checked_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs();
    if actual_sha256.eq_ignore_ascii_case(&record.sha256) {
        record.last_verified_at = Some(checked_at);
        record.integrity_mismatch_at = None;
    } else {
        record.integrity_mismatch_at = Some(checked_at);
    }
    publish_record(models_root, filename, &record).await?;
    Ok(record.public())
}

async fn publish_record(
    models_root: &Path,
    filename: &str,
    record: &ModelProvenanceRecord,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(&record)?;
    if bytes.len() as u64 > MAX_PROVENANCE_BYTES {
        return Err(anyhow!("model provenance record is too large"));
    }

    let metadata_root = ensure_metadata_root(models_root).await?;
    let final_path = metadata_path(&metadata_root, filename);
    let temporary_path = final_path.with_extension("json.tmp");
    reject_symlink_or_non_file(&final_path).await?;
    reject_symlink_or_non_file(&temporary_path).await?;
    remove_file_if_present(&temporary_path).await?;
    let mut temporary = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary_path)
        .await
        .context("failed to create temporary model provenance record")?;
    temporary
        .write_all(&bytes)
        .await
        .context("failed to write model provenance record")?;
    temporary
        .sync_all()
        .await
        .context("failed to sync model provenance record")?;
    drop(temporary);

    remove_file_if_present(&final_path).await?;
    if let Err(error) = fs::hard_link(&temporary_path, &final_path).await {
        let _ = remove_file_if_present(&temporary_path).await;
        return Err(error).context("failed to publish model provenance record");
    }
    remove_file_if_present(&temporary_path).await?;
    sync_directory(&metadata_root).await?;
    sync_directory(models_root).await?;
    Ok(())
}

#[cfg(unix)]
async fn sync_directory(path: &Path) -> Result<()> {
    fs::File::open(path)
        .await
        .with_context(|| format!("failed to open directory '{}' for sync", path.display()))?
        .sync_all()
        .await
        .with_context(|| format!("failed to sync directory '{}'", path.display()))
}

#[cfg(not(unix))]
async fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

pub(crate) fn read_provenance(
    models_root: &Path,
    filename: &str,
    expected_size: u64,
) -> Result<Option<ModelProvenance>> {
    Ok(read_record(models_root, filename, expected_size)?.map(|record| record.public()))
}

pub(crate) fn read_provenance_details(
    models_root: &Path,
    catalog_id: &str,
    expected_size: u64,
) -> Result<Option<ModelProvenanceDetails>> {
    Ok(
        read_record(models_root, catalog_id, expected_size)?.map(|record| ModelProvenanceDetails {
            provenance: record.public(),
            files: record.files,
        }),
    )
}

fn read_record(
    models_root: &Path,
    filename: &str,
    expected_size: u64,
) -> Result<Option<ModelProvenanceRecord>> {
    validate_catalog_id(filename)?;
    let metadata_root = models_root.join(METADATA_DIRECTORY);
    let root_metadata = match std::fs::symlink_metadata(&metadata_root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("failed to inspect model provenance directory"),
    };
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(anyhow!("model provenance path must be a real directory"));
    }
    let path = metadata_path(&metadata_root, filename);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("failed to inspect model provenance record"),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(anyhow!("model provenance record must be a regular file"));
    }
    if metadata.len() > MAX_PROVENANCE_BYTES {
        return Err(anyhow!("model provenance record is too large"));
    }
    let bytes = std::fs::read(&path).context("failed to read model provenance record")?;
    let record: ModelProvenanceRecord =
        serde_json::from_slice(&bytes).context("model provenance record is invalid")?;
    Ok(Some(validate_record(record, filename, expected_size)?))
}

fn validate_record(
    record: ModelProvenanceRecord,
    filename: &str,
    expected_size: u64,
) -> Result<ModelProvenanceRecord> {
    if !matches!(record.version, 1 | 2)
        || record.filename != filename
        || record.size_bytes != expected_size
        || record.installed_at == 0
    {
        return Err(anyhow!(
            "model provenance record does not match the catalog entry"
        ));
    }
    validate_sha256(&record.sha256)?;
    match record.version {
        1 => {
            validate_model_filename(filename)?;
            if !record.files.is_empty() {
                return Err(anyhow!(
                    "single-file model provenance must not contain package files"
                ));
            }
        }
        2 => {
            validate_package_id(filename)?;
            let normalized = normalize_package_files(record.files.clone(), record.size_bytes)?;
            if normalized != record.files || package_digest(&record.files)? != record.sha256 {
                return Err(anyhow!(
                    "model package provenance manifest is not canonical"
                ));
            }
        }
        _ => unreachable!(),
    }
    normalize_license(record.license.clone())?;
    normalize_model_index_id(record.model_index_id.clone())?;
    normalize_source_identity(record.source_url.clone(), record.source_host.clone())?;
    if record.last_verified_at == Some(0) || record.integrity_mismatch_at == Some(0) {
        return Err(anyhow!(
            "model provenance verification timestamp is invalid"
        ));
    }
    Ok(record)
}

pub(crate) fn remove_provenance(models_root: &Path, filename: &str) -> Result<()> {
    validate_catalog_id(filename)?;
    let metadata_root = models_root.join(METADATA_DIRECTORY);
    let root_metadata = match std::fs::symlink_metadata(&metadata_root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).context("failed to inspect model provenance directory"),
    };
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(anyhow!("model provenance path must be a real directory"));
    }
    let path = metadata_path(&metadata_root, filename);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() || metadata.file_type().is_symlink() => {
            std::fs::remove_file(path).context("failed to remove model provenance record")
        }
        Ok(_) => Err(anyhow!("model provenance record is not a file")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("failed to inspect model provenance record"),
    }
}

async fn ensure_metadata_root(models_root: &Path) -> Result<PathBuf> {
    fs::create_dir_all(models_root)
        .await
        .context("failed to create model catalog directory")?;
    let metadata_root = models_root.join(METADATA_DIRECTORY);
    fs::create_dir_all(&metadata_root)
        .await
        .context("failed to create model provenance directory")?;
    let metadata = fs::symlink_metadata(&metadata_root)
        .await
        .context("failed to inspect model provenance directory")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!("model provenance path must be a real directory"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        fs::set_permissions(&metadata_root, std::fs::Permissions::from_mode(0o700))
            .await
            .context("failed to secure model provenance directory")?;
    }
    Ok(metadata_root)
}

fn metadata_path(metadata_root: &Path, filename: &str) -> PathBuf {
    metadata_root.join(format!("{filename}.json"))
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(anyhow!(
            "model provenance SHA-256 must contain exactly 64 hexadecimal characters"
        ))
    }
}

fn normalize_optional_text(
    value: Option<String>,
    field: &str,
    max_length: usize,
) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    if value.len() > max_length {
        return Err(anyhow!("{field} must not exceed {max_length} characters"));
    }
    if value.chars().any(char::is_control) {
        return Err(anyhow!("{field} must not contain control characters"));
    }
    Ok(Some(value.to_string()))
}

fn normalize_source_identity(
    source_url: Option<String>,
    source_host: Option<String>,
) -> Result<(Option<String>, Option<String>)> {
    let source_url = normalize_source_url(source_url)?;
    let source_host = normalize_optional_text(source_host, "model source host", 253)?;
    let Some(source_url_value) = source_url.as_deref() else {
        return Ok((None, source_host));
    };
    let derived_host = Url::parse(source_url_value)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .ok_or_else(|| anyhow!("model source URL must include a host"))?;
    if source_host
        .as_deref()
        .is_some_and(|value| !value.eq_ignore_ascii_case(&derived_host))
    {
        return Err(anyhow!("model source host does not match its URL"));
    }
    Ok((source_url, Some(derived_host)))
}

async fn reject_symlink_or_non_file(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(anyhow!("model provenance records must not be symlinks"))
        }
        Ok(metadata) if !metadata.is_file() => {
            Err(anyhow!("model provenance record is not a regular file"))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("failed to inspect model provenance record"),
    }
}

async fn remove_file_if_present(path: &Path) -> Result<bool> {
    match fs::remove_file(path).await {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("failed to remove model provenance file"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(filename: &str, size_bytes: u64) -> ModelProvenanceDraft {
        ModelProvenanceDraft {
            acquisition: ModelAcquisitionKind::Import,
            model_index_id: None,
            filename: filename.to_string(),
            size_bytes,
            source_url: Some("https://huggingface.co/acme/model?token=secret#weights".to_string()),
            source_host: Some("huggingface.co".to_string()),
            sha256: "ab".repeat(32),
            license: Some(" Apache-2.0 ".to_string()),
        }
    }

    #[tokio::test]
    async fn round_trips_a_private_atomic_record_without_url_secrets() {
        let temp = tempfile::tempdir().unwrap();
        let mut record = draft("model.gguf", 42);
        record.model_index_id = Some("tiny-q4".to_string());
        write_provenance(temp.path(), record).await.unwrap();

        let provenance = read_provenance(temp.path(), "model.gguf", 42)
            .unwrap()
            .unwrap();

        assert_eq!(provenance.acquisition, ModelAcquisitionKind::Import);
        assert_eq!(provenance.model_index_id.as_deref(), Some("tiny-q4"));
        assert_eq!(provenance.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(
            provenance.source_url.as_deref(),
            Some("https://huggingface.co/acme/model")
        );
        let stored =
            std::fs::read_to_string(temp.path().join(METADATA_DIRECTORY).join("model.gguf.json"))
                .unwrap();
        assert!(!stored.contains("secret"));
        assert!(!stored.contains("#weights"));
    }

    #[tokio::test]
    async fn rejects_mismatched_or_unsafe_records_and_removes_links_without_following() {
        let temp = tempfile::tempdir().unwrap();
        write_provenance(temp.path(), draft("model.gguf", 42))
            .await
            .unwrap();
        assert!(read_provenance(temp.path(), "model.gguf", 43).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let outside = tempfile::NamedTempFile::new().unwrap();
            let link = temp
                .path()
                .join(METADATA_DIRECTORY)
                .join("linked.gguf.json");
            symlink(outside.path(), &link).unwrap();
            assert!(read_provenance(temp.path(), "linked.gguf", 1).is_err());
            remove_provenance(temp.path(), "linked.gguf").unwrap();
            assert!(outside.path().exists());
        }
    }

    #[tokio::test]
    async fn persists_mismatches_and_clears_them_only_after_a_matching_check() {
        let temp = tempfile::tempdir().unwrap();
        write_provenance(temp.path(), draft("model.gguf", 42))
            .await
            .unwrap();

        let mismatch = record_integrity_result(
            temp.path(),
            "model.gguf",
            42,
            &"ab".repeat(32),
            &"cd".repeat(32),
        )
        .await
        .unwrap();
        assert!(mismatch.integrity_mismatch_at.is_some());
        assert!(mismatch.last_verified_at.is_none());
        let persisted = read_provenance(temp.path(), "model.gguf", 42)
            .unwrap()
            .unwrap();
        assert_eq!(
            persisted.integrity_mismatch_at,
            mismatch.integrity_mismatch_at
        );

        let verified = record_integrity_result(
            temp.path(),
            "model.gguf",
            42,
            &"ab".repeat(32),
            &"ab".repeat(32),
        )
        .await
        .unwrap();
        assert!(verified.integrity_mismatch_at.is_none());
        assert!(verified.last_verified_at.is_some());
    }

    #[test]
    fn validates_optional_governance_fields() {
        assert_eq!(
            normalize_license(Some(" MIT ".into())).unwrap(),
            Some("MIT".into())
        );
        assert!(normalize_license(Some("x".repeat(129))).is_err());
        assert!(normalize_license(Some("bad\nvalue".into())).is_err());
        assert_eq!(
            normalize_model_index_id(Some("tiny.q4_k-m".into())).unwrap(),
            Some("tiny.q4_k-m".into())
        );
        for invalid in ["", "Uppercase", ".leading", "two words", "bad/slash"] {
            assert!(normalize_model_index_id(Some(invalid.into())).is_err());
        }
        assert!(normalize_source_url(Some("http://example.com/model".into())).is_err());
        assert!(
            normalize_source_url(Some("https://user:secret@example.com/model".into())).is_err()
        );
        assert!(normalize_source_identity(
            Some("https://example.com/model".into()),
            Some("elsewhere.example".into())
        )
        .is_err());
    }
}
