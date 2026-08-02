//! Crash-safe publication for verified signed-model upgrades.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncWriteExt as _;

use super::model_manager::{validate_catalog_id, validate_model_filename, ModelCatalog};
use super::model_package::{
    normalize_package_files, package_digest, validate_package_id, validate_package_path,
    ModelPackageFile,
};
use super::model_provenance::{
    backup_provenance, normalize_license, normalize_model_index_id, normalize_source_url,
    read_provenance_details, remove_provenance, restore_provenance_backup,
    write_package_provenance, write_provenance, ModelAcquisitionKind, ModelPackageProvenanceDraft,
    ModelProvenanceDraft,
};

pub(crate) const UPGRADE_DIRECTORY: &str = ".bloom-upgrade";
const MARKER_FILENAME: &str = "transaction.json";
const MARKER_TEMP_FILENAME: &str = "transaction.json.tmp";
const BACKUP_ENTRY_FILENAME: &str = "previous-entry";
const BACKUP_PROVENANCE_FILENAME: &str = "previous-provenance.json";
const MAX_MARKER_BYTES: u64 = 512 * 1024;
const MAX_UPGRADE_DIRECTORY_ENTRIES: usize = 4;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelUpgradeEntryKind {
    File,
    Directory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelUpgradeSource {
    pub catalog_id: String,
    pub kind: ModelUpgradeEntryKind,
    pub size_bytes: u64,
    pub sha256: String,
    pub model_index_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelUpgradeTarget {
    pub catalog_id: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub source_url: Option<String>,
    pub source_host: Option<String>,
    pub license: Option<String>,
    pub model_index_id: String,
    #[serde(default)]
    pub files: Vec<ModelPackageFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ModelUpgradeTransaction {
    version: u8,
    source: ModelUpgradeSource,
    target: ModelUpgradeTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelUpgradeRecovery {
    Completed,
    RolledBack,
    DiscardedPrepared,
}

pub(crate) async fn commit_model_upgrade(
    models_root: &Path,
    staged_entry: &Path,
    source: ModelUpgradeSource,
    target: ModelUpgradeTarget,
) -> Result<()> {
    validate_transaction(models_root, staged_entry, &source, &target)?;
    verify_upgrade_source(models_root, &source).await?;
    verify_target_entry(staged_entry, &target).await?;

    let transaction = ModelUpgradeTransaction {
        version: 1,
        source,
        target,
    };
    let paths = UpgradePaths::new(models_root, &transaction);
    ensure_destination_state(&paths, true).await?;
    create_transaction_marker(models_root, &paths, &transaction).await?;
    if let Err(error) = backup_provenance(
        models_root,
        &transaction.source.catalog_id,
        transaction.source.size_bytes,
        &paths.backup_provenance,
    )
    .await
    {
        let _ = cleanup_transaction_files(models_root, &paths).await;
        return Err(error).context("failed to prepare signed-model upgrade provenance rollback");
    }

    if let Err(error) = move_entry(&paths.source_live, &paths.backup_entry).await {
        let _ = cleanup_transaction_files(models_root, &paths).await;
        return Err(error).context("failed to preserve the installed model before upgrade");
    }
    if let Err(error) = move_entry(staged_entry, &paths.target_live).await {
        return rollback_transaction(models_root, &paths, &transaction)
            .await
            .and(Err(
                error.context("failed to publish the verified upgrade payload")
            ));
    }

    if let Err(error) = publish_target_provenance(models_root, &transaction.target).await {
        return rollback_transaction(models_root, &paths, &transaction)
            .await
            .and(Err(
                error.context("failed to publish upgraded model provenance")
            ));
    }
    if transaction.source.catalog_id != transaction.target.catalog_id {
        if let Err(error) = remove_provenance(models_root, &transaction.source.catalog_id) {
            return rollback_transaction(models_root, &paths, &transaction)
                .await
                .and(Err(
                    error.context("failed to retire previous model provenance")
                ));
        }
    }

    finish_transaction(models_root, &paths).await
}

pub(crate) async fn recover_model_upgrade(
    models_root: &Path,
) -> Result<Option<ModelUpgradeRecovery>> {
    let upgrade_root = models_root.join(UPGRADE_DIRECTORY);
    let root_metadata = match fs::symlink_metadata(&upgrade_root).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("failed to inspect model upgrade transaction"),
    };
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(anyhow!(
            "model upgrade transaction path must be a real directory"
        ));
    }
    secure_private_directory(&upgrade_root).await?;
    validate_upgrade_directory_entries(&upgrade_root).await?;

    let marker_path = upgrade_root.join(MARKER_FILENAME);
    let marker_metadata = match fs::symlink_metadata(&marker_path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let temporary = upgrade_root.join(MARKER_TEMP_FILENAME);
            remove_regular_file_if_present(&temporary).await?;
            fs::remove_dir(&upgrade_root)
                .await
                .context("failed to remove an empty model upgrade transaction")?;
            sync_directory(models_root).await?;
            return Ok(Some(ModelUpgradeRecovery::DiscardedPrepared));
        }
        Err(error) => return Err(error).context("failed to inspect model upgrade marker"),
    };
    if marker_metadata.file_type().is_symlink()
        || !marker_metadata.is_file()
        || marker_metadata.len() == 0
        || marker_metadata.len() > MAX_MARKER_BYTES
    {
        return Err(anyhow!(
            "model upgrade marker is not a bounded regular file"
        ));
    }
    let bytes = fs::read(&marker_path)
        .await
        .context("failed to read model upgrade marker")?;
    let transaction = serde_json::from_slice::<ModelUpgradeTransaction>(&bytes)
        .context("model upgrade marker is invalid")?;
    let staged_entry = staged_entry_path(models_root, &transaction.target);
    validate_transaction(
        models_root,
        &staged_entry,
        &transaction.source,
        &transaction.target,
    )?;
    let paths = UpgradePaths::new(models_root, &transaction);
    let backup_exists = safe_entry_exists(&paths.backup_entry).await?;
    let staged_exists = safe_entry_exists(&staged_entry).await?;
    let target_exists = safe_entry_exists(&paths.target_live).await?;
    let source_exists = if paths.source_live == paths.target_live {
        target_exists
    } else {
        safe_entry_exists(&paths.source_live).await?
    };

    if backup_exists {
        return if staged_exists && !target_exists {
            rollback_transaction(models_root, &paths, &transaction).await?;
            Ok(Some(ModelUpgradeRecovery::RolledBack))
        } else if !staged_exists && target_exists {
            match verify_target_entry(&paths.target_live, &transaction.target).await {
                Ok(()) => {
                    complete_recovered_transaction(models_root, &paths, &transaction).await?;
                    Ok(Some(ModelUpgradeRecovery::Completed))
                }
                Err(error) => {
                    rollback_transaction(models_root, &paths, &transaction)
                        .await
                        .with_context(|| {
                            format!(
                                "upgraded model validation failed ({error}); rollback also failed"
                            )
                        })?;
                    Ok(Some(ModelUpgradeRecovery::RolledBack))
                }
            }
        } else {
            Err(anyhow!(
                "model upgrade transaction has an ambiguous payload state"
            ))
        };
    }

    if staged_exists && source_exists && (paths.source_live == paths.target_live || !target_exists)
    {
        cleanup_transaction_files(models_root, &paths).await?;
        return Ok(Some(ModelUpgradeRecovery::DiscardedPrepared));
    }
    if !staged_exists && target_exists && (paths.source_live == paths.target_live || !source_exists)
    {
        verify_target_entry(&paths.target_live, &transaction.target).await?;
        complete_recovered_transaction(models_root, &paths, &transaction).await?;
        return Ok(Some(ModelUpgradeRecovery::Completed));
    }
    Err(anyhow!(
        "model upgrade transaction cannot be recovered from its filesystem state"
    ))
}

fn validate_transaction(
    models_root: &Path,
    staged_entry: &Path,
    source: &ModelUpgradeSource,
    target: &ModelUpgradeTarget,
) -> Result<()> {
    validate_catalog_id(&source.catalog_id)?;
    validate_catalog_id(&target.catalog_id)?;
    if source.catalog_id.starts_with('.') || target.catalog_id.starts_with('.') {
        return Err(anyhow!("model upgrade catalog IDs must not be hidden"));
    }
    if source.size_bytes == 0 || target.size_bytes == 0 {
        return Err(anyhow!("model upgrade sizes must be greater than zero"));
    }
    validate_sha256(&source.sha256)?;
    validate_sha256(&target.sha256)?;
    if normalize_model_index_id(Some(source.model_index_id.clone()))?.as_deref()
        != Some(source.model_index_id.as_str())
        || normalize_model_index_id(Some(target.model_index_id.clone()))?.as_deref()
            != Some(target.model_index_id.as_str())
        || source.model_index_id != target.model_index_id
    {
        return Err(anyhow!(
            "model upgrade source and target must use one valid signed-index ID"
        ));
    }
    normalize_license(target.license.clone())?;
    if let Some(url) = target.source_url.clone() {
        normalize_source_url(Some(url))?;
    }
    if target.source_host.as_ref().is_some_and(|host| {
        host.is_empty()
            || host.len() > 253
            || host.trim() != host
            || host.chars().any(char::is_control)
    }) {
        return Err(anyhow!("model upgrade source host is invalid"));
    }
    if target.files.is_empty() {
        validate_model_filename(&target.catalog_id)?;
    } else {
        validate_package_id(&target.catalog_id)?;
        let normalized = normalize_package_files(target.files.clone(), target.size_bytes)?;
        if normalized != target.files || package_digest(&normalized)? != target.sha256 {
            return Err(anyhow!("model upgrade package manifest is not canonical"));
        }
    }
    let expected_staging = staged_entry_path(models_root, target);
    if staged_entry != expected_staging {
        return Err(anyhow!("model upgrade staging path is not authoritative"));
    }
    Ok(())
}

async fn verify_upgrade_source(models_root: &Path, source: &ModelUpgradeSource) -> Result<()> {
    let path = ModelCatalog::resolve(models_root, &source.catalog_id)?;
    let details = read_provenance_details(models_root, &source.catalog_id, source.size_bytes)?
        .ok_or_else(|| anyhow!("installed model has no verified provenance"))?;
    if details.provenance.acquisition != ModelAcquisitionKind::Download
        || details.provenance.model_index_id.as_deref() != Some(source.model_index_id.as_str())
        || !details
            .provenance
            .sha256
            .eq_ignore_ascii_case(&source.sha256)
        || details.provenance.integrity_mismatch_at.is_some()
    {
        return Err(anyhow!(
            "installed model no longer matches the upgrade source identity"
        ));
    }
    let files = match source.kind {
        ModelUpgradeEntryKind::File if details.files.is_empty() => Vec::new(),
        ModelUpgradeEntryKind::Directory if !details.files.is_empty() => details.files,
        _ => {
            return Err(anyhow!(
                "installed model kind no longer matches the upgrade source"
            ))
        }
    };
    verify_entry(
        &path,
        source.size_bytes,
        &source.sha256,
        &files,
        "installed upgrade source",
    )
    .await
}

async fn verify_target_entry(path: &Path, target: &ModelUpgradeTarget) -> Result<()> {
    verify_entry(
        path,
        target.size_bytes,
        &target.sha256,
        &target.files,
        "verified upgrade target",
    )
    .await
}

async fn verify_entry(
    path: &Path,
    size_bytes: u64,
    sha256: &str,
    files: &[ModelPackageFile],
    label: &'static str,
) -> Result<()> {
    let path = path.to_path_buf();
    let sha256 = sha256.to_ascii_lowercase();
    let files = files.to_vec();
    tokio::task::spawn_blocking(move || {
        if files.is_empty() {
            verify_regular_file(&path, size_bytes, &sha256, label)
        } else {
            verify_package(&path, size_bytes, &sha256, &files, label)
        }
    })
    .await
    .with_context(|| format!("{label} verification task failed"))?
}

fn verify_regular_file(path: &Path, size_bytes: u64, sha256: &str, label: &str) -> Result<()> {
    let metadata =
        std::fs::symlink_metadata(path).with_context(|| format!("failed to inspect {label}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != size_bytes {
        return Err(anyhow!("{label} is not the expected regular file"));
    }
    if hash_file(path)? != sha256 {
        return Err(anyhow!("{label} checksum does not match its provenance"));
    }
    Ok(())
}

fn verify_package(
    root: &Path,
    size_bytes: u64,
    sha256: &str,
    files: &[ModelPackageFile],
    label: &str,
) -> Result<()> {
    let normalized = normalize_package_files(files.to_vec(), size_bytes)?;
    if normalized != files || package_digest(&normalized)? != sha256 {
        return Err(anyhow!("{label} manifest identity is invalid"));
    }
    let root_metadata =
        std::fs::symlink_metadata(root).with_context(|| format!("failed to inspect {label}"))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(anyhow!("{label} is not a real directory"));
    }
    let mut expected_files = HashSet::new();
    let mut expected_directories = HashSet::new();
    for file in files {
        validate_package_path(&file.filename)?;
        expected_files.insert(file.filename.clone());
        let mut parent = Path::new(&file.filename).parent();
        while let Some(directory) = parent {
            if directory.as_os_str().is_empty() {
                break;
            }
            expected_directories.insert(directory.to_string_lossy().replace('\\', "/"));
            parent = directory.parent();
        }
    }
    let mut discovered = HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    let mut inspected = 0_usize;
    while let Some(directory) = stack.pop() {
        for entry in
            std::fs::read_dir(&directory).with_context(|| format!("failed to read {label}"))?
        {
            let entry = entry.with_context(|| format!("failed to inspect {label}"))?;
            inspected = inspected.saturating_add(1);
            if inspected > files.len().saturating_mul(3).saturating_add(16) {
                return Err(anyhow!("{label} contains too many entries"));
            }
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)
                .with_context(|| format!("failed to inspect {label}"))?;
            if metadata.file_type().is_symlink() {
                return Err(anyhow!("{label} must not contain symlinks"));
            }
            let relative = path
                .strip_prefix(root)
                .with_context(|| format!("{label} entry escaped its root"))?
                .to_str()
                .ok_or_else(|| anyhow!("{label} filename is not UTF-8"))?
                .replace('\\', "/");
            if metadata.is_dir() {
                if !expected_directories.contains(&relative) {
                    return Err(anyhow!("{label} contains an unexpected directory"));
                }
                stack.push(path);
            } else if metadata.is_file() {
                if !expected_files.contains(&relative) {
                    return Err(anyhow!("{label} contains an unexpected file"));
                }
                discovered.insert(relative);
            } else {
                return Err(anyhow!("{label} contains an unsupported entry"));
            }
        }
    }
    if discovered != expected_files {
        return Err(anyhow!("{label} is incomplete"));
    }
    for file in files {
        let path = root.join(&file.filename);
        verify_regular_file(&path, file.size_bytes, &file.sha256, "model package file")?;
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read as _;

    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("failed to open '{}' for hashing", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash '{}'", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

async fn publish_target_provenance(models_root: &Path, target: &ModelUpgradeTarget) -> Result<()> {
    if target.files.is_empty() {
        write_provenance(
            models_root,
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(target.model_index_id.clone()),
                filename: target.catalog_id.clone(),
                size_bytes: target.size_bytes,
                source_url: target.source_url.clone(),
                source_host: target.source_host.clone(),
                sha256: target.sha256.clone(),
                license: target.license.clone(),
            },
        )
        .await
    } else {
        write_package_provenance(
            models_root,
            ModelPackageProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(target.model_index_id.clone()),
                directory: target.catalog_id.clone(),
                size_bytes: target.size_bytes,
                source_url: target.source_url.clone(),
                source_host: target.source_host.clone(),
                sha256: target.sha256.clone(),
                license: target.license.clone(),
                files: target.files.clone(),
            },
        )
        .await
    }
}

async fn complete_recovered_transaction(
    models_root: &Path,
    paths: &UpgradePaths,
    transaction: &ModelUpgradeTransaction,
) -> Result<()> {
    publish_target_provenance(models_root, &transaction.target).await?;
    if transaction.source.catalog_id != transaction.target.catalog_id {
        remove_provenance(models_root, &transaction.source.catalog_id)?;
    }
    finish_transaction(models_root, paths).await
}

async fn rollback_transaction(
    models_root: &Path,
    paths: &UpgradePaths,
    transaction: &ModelUpgradeTransaction,
) -> Result<()> {
    let staged_entry = staged_entry_path(models_root, &transaction.target);
    if safe_entry_exists(&paths.target_live).await? && !safe_entry_exists(&staged_entry).await? {
        move_entry(&paths.target_live, &staged_entry).await?;
    }
    if !safe_entry_exists(&paths.source_live).await?
        && safe_entry_exists(&paths.backup_entry).await?
    {
        move_entry(&paths.backup_entry, &paths.source_live).await?;
    }
    if safe_entry_exists(&paths.backup_provenance).await? {
        restore_provenance_backup(
            models_root,
            &transaction.source.catalog_id,
            transaction.source.size_bytes,
            &paths.backup_provenance,
        )
        .await?;
    }
    if transaction.source.catalog_id != transaction.target.catalog_id {
        remove_provenance(models_root, &transaction.target.catalog_id)?;
    }
    verify_upgrade_source(models_root, &transaction.source).await?;
    cleanup_transaction_files(models_root, paths).await
}

async fn finish_transaction(models_root: &Path, paths: &UpgradePaths) -> Result<()> {
    if safe_entry_exists(&paths.backup_entry).await? {
        remove_entry(&paths.backup_entry).await?;
        sync_directory(&paths.upgrade_root).await?;
    }
    cleanup_transaction_files(models_root, paths).await
}

async fn cleanup_transaction_files(models_root: &Path, paths: &UpgradePaths) -> Result<()> {
    remove_regular_file_if_present(&paths.backup_provenance).await?;
    remove_regular_file_if_present(&paths.marker_temporary).await?;
    remove_regular_file_if_present(&paths.marker).await?;
    match fs::remove_dir(&paths.upgrade_root).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("failed to remove model upgrade transaction"),
    }
    sync_directory(models_root).await
}

async fn create_transaction_marker(
    models_root: &Path,
    paths: &UpgradePaths,
    transaction: &ModelUpgradeTransaction,
) -> Result<()> {
    match fs::create_dir(&paths.upgrade_root).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(anyhow!("another model upgrade transaction already exists"))
        }
        Err(error) => return Err(error).context("failed to create model upgrade transaction"),
    }
    secure_private_directory(&paths.upgrade_root).await?;
    let bytes = serde_json::to_vec(transaction)?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_MARKER_BYTES {
        return Err(anyhow!("model upgrade marker exceeds its size limit"));
    }
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&paths.marker_temporary)
        .await
        .context("failed to create model upgrade marker")?;
    file.write_all(&bytes)
        .await
        .context("failed to write model upgrade marker")?;
    file.sync_all()
        .await
        .context("failed to sync model upgrade marker")?;
    drop(file);
    fs::rename(&paths.marker_temporary, &paths.marker)
        .await
        .context("failed to publish model upgrade marker")?;
    sync_directory(&paths.upgrade_root).await?;
    sync_directory(models_root).await
}

async fn ensure_destination_state(paths: &UpgradePaths, staging_required: bool) -> Result<()> {
    if staging_required && !safe_entry_exists(&paths.staged_entry).await? {
        return Err(anyhow!("verified model upgrade staging entry is missing"));
    }
    if !safe_entry_exists(&paths.source_live).await? {
        return Err(anyhow!("installed model upgrade source is missing"));
    }
    if paths.source_live != paths.target_live && safe_entry_exists(&paths.target_live).await? {
        return Err(anyhow!("model upgrade destination is already occupied"));
    }
    Ok(())
}

async fn validate_upgrade_directory_entries(root: &Path) -> Result<()> {
    let allowed = [
        MARKER_FILENAME,
        MARKER_TEMP_FILENAME,
        BACKUP_ENTRY_FILENAME,
        BACKUP_PROVENANCE_FILENAME,
    ]
    .into_iter()
    .collect::<HashSet<_>>();
    let mut entries = fs::read_dir(root)
        .await
        .context("failed to read model upgrade transaction")?;
    let mut inspected = 0_usize;
    while let Some(entry) = entries
        .next_entry()
        .await
        .context("failed to inspect model upgrade transaction")?
    {
        inspected = inspected.saturating_add(1);
        if inspected > MAX_UPGRADE_DIRECTORY_ENTRIES {
            return Err(anyhow!("model upgrade transaction has too many entries"));
        }
        let name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| anyhow!("model upgrade transaction filename is not UTF-8"))?
            .to_string();
        if !allowed.contains(name.as_str()) {
            return Err(anyhow!(
                "model upgrade transaction contains an unexpected entry"
            ));
        }
        let metadata = fs::symlink_metadata(entry.path())
            .await
            .context("failed to inspect model upgrade transaction entry")?;
        if metadata.file_type().is_symlink() {
            return Err(anyhow!(
                "model upgrade transaction must not contain symlinks"
            ));
        }
    }
    Ok(())
}

async fn safe_entry_exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(anyhow!("model upgrade path must not be a symlink"))
        }
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("failed to inspect model upgrade path"),
    }
}

async fn move_entry(source: &Path, destination: &Path) -> Result<()> {
    if !safe_entry_exists(source).await? {
        return Err(anyhow!("model upgrade rename source is missing"));
    }
    if safe_entry_exists(destination).await? {
        return Err(anyhow!("model upgrade rename destination already exists"));
    }
    fs::rename(source, destination).await.with_context(|| {
        format!(
            "failed to rename '{}' to '{}'",
            source.display(),
            destination.display()
        )
    })?;
    if let Some(parent) = source.parent() {
        sync_directory(parent).await?;
    }
    if let Some(parent) = destination.parent() {
        if source.parent() != Some(parent) {
            sync_directory(parent).await?;
        }
    }
    Ok(())
}

async fn remove_entry(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .await
        .context("failed to inspect retired model upgrade entry")?;
    if metadata.file_type().is_symlink() {
        return Err(anyhow!("retired model upgrade entry must not be a symlink"));
    }
    if metadata.is_file() {
        fs::remove_file(path)
            .await
            .context("failed to remove retired model upgrade file")
    } else if metadata.is_dir() {
        fs::remove_dir_all(path)
            .await
            .context("failed to remove retired model upgrade directory")
    } else {
        Err(anyhow!(
            "retired model upgrade entry has an unsupported type"
        ))
    }
}

async fn remove_regular_file_if_present(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(anyhow!("model upgrade metadata must be a regular file"))
        }
        Ok(_) => fs::remove_file(path)
            .await
            .context("failed to remove model upgrade metadata"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("failed to inspect model upgrade metadata"),
    }
}

async fn secure_private_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .await
        .context("failed to inspect model upgrade directory")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!("model upgrade path must be a real directory"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .await
            .context("failed to secure model upgrade directory")?;
    }
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

fn staged_entry_path(models_root: &Path, target: &ModelUpgradeTarget) -> PathBuf {
    let staging_root = models_root.join(".bloom-downloads");
    if target.files.is_empty() {
        staging_root.join(format!("{}.part", target.catalog_id))
    } else {
        staging_root.join(format!("{}.package", target.catalog_id))
    }
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(anyhow!("model upgrade SHA-256 identity is invalid"))
    }
}

struct UpgradePaths {
    upgrade_root: PathBuf,
    marker: PathBuf,
    marker_temporary: PathBuf,
    backup_entry: PathBuf,
    backup_provenance: PathBuf,
    source_live: PathBuf,
    target_live: PathBuf,
    staged_entry: PathBuf,
}

impl UpgradePaths {
    fn new(models_root: &Path, transaction: &ModelUpgradeTransaction) -> Self {
        let upgrade_root = models_root.join(UPGRADE_DIRECTORY);
        Self {
            marker: upgrade_root.join(MARKER_FILENAME),
            marker_temporary: upgrade_root.join(MARKER_TEMP_FILENAME),
            backup_entry: upgrade_root.join(BACKUP_ENTRY_FILENAME),
            backup_provenance: upgrade_root.join(BACKUP_PROVENANCE_FILENAME),
            source_live: models_root.join(&transaction.source.catalog_id),
            target_live: models_root.join(&transaction.target.catalog_id),
            staged_entry: staged_entry_path(models_root, &transaction.target),
            upgrade_root,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::model_provenance::{read_provenance, ModelProvenanceDraft};
    use super::*;

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    async fn install_old_file(root: &Path, catalog_id: &str, bytes: &[u8], index_id: &str) {
        fs::write(root.join(catalog_id), bytes).await.unwrap();
        write_provenance(
            root,
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some(index_id.to_string()),
                filename: catalog_id.to_string(),
                size_bytes: bytes.len() as u64,
                source_url: Some(format!(
                    "https://huggingface.co/acme/old/resolve/{}/{}",
                    "11".repeat(20),
                    catalog_id
                )),
                source_host: Some("huggingface.co".to_string()),
                sha256: sha256(bytes),
                license: Some("Apache-2.0".to_string()),
            },
        )
        .await
        .unwrap();
    }

    fn source(catalog_id: &str, bytes: &[u8], index_id: &str) -> ModelUpgradeSource {
        ModelUpgradeSource {
            catalog_id: catalog_id.to_string(),
            kind: ModelUpgradeEntryKind::File,
            size_bytes: bytes.len() as u64,
            sha256: sha256(bytes),
            model_index_id: index_id.to_string(),
        }
    }

    fn file_target(catalog_id: &str, bytes: &[u8], index_id: &str) -> ModelUpgradeTarget {
        ModelUpgradeTarget {
            catalog_id: catalog_id.to_string(),
            size_bytes: bytes.len() as u64,
            sha256: sha256(bytes),
            source_url: Some(format!(
                "https://huggingface.co/acme/new/resolve/{}/{}",
                "22".repeat(20),
                catalog_id
            )),
            source_host: Some("huggingface.co".to_string()),
            license: Some("Apache-2.0".to_string()),
            model_index_id: index_id.to_string(),
            files: Vec::new(),
        }
    }

    async fn stage_file(root: &Path, target: &ModelUpgradeTarget, bytes: &[u8]) -> PathBuf {
        let staged = staged_entry_path(root, target);
        fs::create_dir_all(staged.parent().unwrap()).await.unwrap();
        fs::write(&staged, bytes).await.unwrap();
        staged
    }

    async fn prepare_interrupted_transaction(
        root: &Path,
        source: ModelUpgradeSource,
        target: ModelUpgradeTarget,
        install_replacement: bool,
    ) -> (ModelUpgradeTransaction, UpgradePaths) {
        let transaction = ModelUpgradeTransaction {
            version: 1,
            source,
            target,
        };
        let staged = staged_entry_path(root, &transaction.target);
        validate_transaction(root, &staged, &transaction.source, &transaction.target).unwrap();
        verify_upgrade_source(root, &transaction.source)
            .await
            .unwrap();
        verify_target_entry(&staged, &transaction.target)
            .await
            .unwrap();
        let paths = UpgradePaths::new(root, &transaction);
        create_transaction_marker(root, &paths, &transaction)
            .await
            .unwrap();
        backup_provenance(
            root,
            &transaction.source.catalog_id,
            transaction.source.size_bytes,
            &paths.backup_provenance,
        )
        .await
        .unwrap();
        move_entry(&paths.source_live, &paths.backup_entry)
            .await
            .unwrap();
        if install_replacement {
            move_entry(&staged, &paths.target_live).await.unwrap();
        }
        (transaction, paths)
    }

    #[tokio::test]
    async fn commits_a_same_path_file_upgrade_and_replaces_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let old = b"old signed model";
        let new = b"new signed model revision";
        let index_id = "stable-model";
        install_old_file(temp.path(), "model.gguf", old, index_id).await;
        let source = source("model.gguf", old, index_id);
        let target = file_target("model.gguf", new, index_id);
        let staged = stage_file(temp.path(), &target, new).await;

        commit_model_upgrade(temp.path(), &staged, source, target.clone())
            .await
            .unwrap();

        assert_eq!(fs::read(temp.path().join("model.gguf")).await.unwrap(), new);
        assert!(!staged.exists());
        assert!(!temp.path().join(UPGRADE_DIRECTORY).exists());
        let provenance = read_provenance(temp.path(), "model.gguf", new.len() as u64)
            .unwrap()
            .unwrap();
        assert_eq!(provenance.sha256, target.sha256);
        assert_eq!(provenance.model_index_id.as_deref(), Some(index_id));
    }

    #[tokio::test]
    async fn upgrades_across_catalog_paths_and_file_directory_shapes() {
        let temp = tempfile::tempdir().unwrap();
        let old = b"old single file";
        let index_id = "stable-package";
        install_old_file(temp.path(), "old.gguf", old, index_id).await;
        let source = source("old.gguf", old, index_id);
        let config = br#"{"model_type":"qwen2"}"#;
        let weights = b"small package weights";
        let files = normalize_package_files(
            vec![
                ModelPackageFile {
                    filename: "config.json".to_string(),
                    size_bytes: config.len() as u64,
                    sha256: sha256(config),
                },
                ModelPackageFile {
                    filename: "model.safetensors".to_string(),
                    size_bytes: weights.len() as u64,
                    sha256: sha256(weights),
                },
            ],
            (config.len() + weights.len()) as u64,
        )
        .unwrap();
        let target = ModelUpgradeTarget {
            catalog_id: "new-package".to_string(),
            size_bytes: (config.len() + weights.len()) as u64,
            sha256: package_digest(&files).unwrap(),
            source_url: Some(format!(
                "https://huggingface.co/acme/new/resolve/{}/config.json",
                "22".repeat(20)
            )),
            source_host: Some("huggingface.co".to_string()),
            license: Some("Apache-2.0".to_string()),
            model_index_id: index_id.to_string(),
            files,
        };
        let staged = staged_entry_path(temp.path(), &target);
        fs::create_dir_all(&staged).await.unwrap();
        fs::write(staged.join("config.json"), config).await.unwrap();
        fs::write(staged.join("model.safetensors"), weights)
            .await
            .unwrap();

        commit_model_upgrade(temp.path(), &staged, source, target.clone())
            .await
            .unwrap();

        assert!(!temp.path().join("old.gguf").exists());
        assert_eq!(
            fs::read(temp.path().join("new-package/config.json"))
                .await
                .unwrap(),
            config
        );
        assert!(read_provenance(temp.path(), "old.gguf", old.len() as u64)
            .unwrap()
            .is_none());
        let provenance = read_provenance(temp.path(), "new-package", target.size_bytes)
            .unwrap()
            .unwrap();
        assert_eq!(provenance.file_count, Some(2));
        assert_eq!(provenance.sha256, target.sha256);
    }

    #[tokio::test]
    async fn restart_rolls_back_when_only_the_previous_entry_was_preserved() {
        let temp = tempfile::tempdir().unwrap();
        let old = b"old crash-safe model";
        let new = b"new crash-safe model";
        let index_id = "crash-safe";
        install_old_file(temp.path(), "model.gguf", old, index_id).await;
        let source = source("model.gguf", old, index_id);
        let target = file_target("model.gguf", new, index_id);
        let staged = stage_file(temp.path(), &target, new).await;
        prepare_interrupted_transaction(temp.path(), source, target, false).await;

        let recovery = recover_model_upgrade(temp.path()).await.unwrap();

        assert_eq!(recovery, Some(ModelUpgradeRecovery::RolledBack));
        assert_eq!(fs::read(temp.path().join("model.gguf")).await.unwrap(), old);
        assert_eq!(fs::read(staged).await.unwrap(), new);
        assert!(!temp.path().join(UPGRADE_DIRECTORY).exists());
    }

    #[tokio::test]
    async fn restart_completes_a_verified_replacement_missing_only_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let old = b"old committed model";
        let new = b"new committed model";
        let index_id = "complete-after-crash";
        install_old_file(temp.path(), "model.gguf", old, index_id).await;
        let source = source("model.gguf", old, index_id);
        let target = file_target("model.gguf", new, index_id);
        stage_file(temp.path(), &target, new).await;
        prepare_interrupted_transaction(temp.path(), source, target.clone(), true).await;

        let recovery = recover_model_upgrade(temp.path()).await.unwrap();

        assert_eq!(recovery, Some(ModelUpgradeRecovery::Completed));
        assert_eq!(fs::read(temp.path().join("model.gguf")).await.unwrap(), new);
        let provenance = read_provenance(temp.path(), "model.gguf", new.len() as u64)
            .unwrap()
            .unwrap();
        assert_eq!(provenance.sha256, target.sha256);
        assert!(!temp.path().join(UPGRADE_DIRECTORY).exists());
    }

    #[tokio::test]
    async fn restart_rejects_a_corrupt_replacement_and_restores_the_previous_model() {
        let temp = tempfile::tempdir().unwrap();
        let old = b"old rollback model";
        let new = b"new rollback model";
        let index_id = "rollback-corrupt";
        install_old_file(temp.path(), "model.gguf", old, index_id).await;
        let source = source("model.gguf", old, index_id);
        let target = file_target("model.gguf", new, index_id);
        let staged = stage_file(temp.path(), &target, new).await;
        prepare_interrupted_transaction(temp.path(), source, target, true).await;
        fs::write(temp.path().join("model.gguf"), b"tampered replacement")
            .await
            .unwrap();

        let recovery = recover_model_upgrade(temp.path()).await.unwrap();

        assert_eq!(recovery, Some(ModelUpgradeRecovery::RolledBack));
        assert_eq!(fs::read(temp.path().join("model.gguf")).await.unwrap(), old);
        assert_eq!(fs::read(staged).await.unwrap(), b"tampered replacement");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn recovery_never_follows_a_transaction_symlink() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("keep"), b"keep")
            .await
            .unwrap();
        symlink(outside.path(), temp.path().join(UPGRADE_DIRECTORY)).unwrap();

        let error = recover_model_upgrade(temp.path()).await.unwrap_err();

        assert!(error.to_string().contains("must be a real directory"));
        assert_eq!(
            fs::read(outside.path().join("keep")).await.unwrap(),
            b"keep"
        );
    }
}
