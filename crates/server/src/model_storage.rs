//! Shared model-catalog storage accounting, quota admission, and stale cleanup.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use tokio::sync::{Mutex, OwnedMutexGuard, RwLock};

use super::model_provenance::METADATA_DIRECTORY;

const DOWNLOAD_STAGING_DIRECTORY: &str = ".bloom-downloads";
const IMPORT_STAGING_DIRECTORY: &str = ".bloom-imports";
const SNAPSHOT_TTL: Duration = Duration::from_secs(10);
const MAX_STORAGE_SCAN_DEPTH: usize = 16;
const MAX_STORAGE_SCAN_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelStorageStatus {
    pub quota_enabled: bool,
    pub max_bytes: Option<u64>,
    pub used_bytes: u64,
    pub committed_bytes: u64,
    pub available_bytes: Option<u64>,
    pub installed_bytes: u64,
    pub staged_download_bytes: u64,
    pub staged_import_bytes: u64,
    pub reserved_bytes: u64,
    pub staged_retention_seconds: u64,
    pub last_cleanup_at: Option<u64>,
    pub last_cleanup_removed_sessions: u64,
}

#[derive(Debug, Default)]
struct StorageState {
    download_reservations: HashMap<String, u64>,
    active_imports: HashSet<String>,
    last_cleanup_at: Option<u64>,
    last_cleanup_removed_sessions: u64,
}

#[derive(Debug, Clone, Copy)]
struct StorageScan {
    installed_bytes: u64,
    staged_download_bytes: u64,
    staged_import_bytes: u64,
    import_remaining_bytes: u64,
}

impl StorageScan {
    fn used_bytes(self) -> Result<u64> {
        checked_sum([
            self.installed_bytes,
            self.staged_download_bytes,
            self.staged_import_bytes,
        ])
    }

    fn committed_bytes(self) -> Result<u64> {
        self.used_bytes()?
            .checked_add(self.import_remaining_bytes)
            .ok_or_else(|| anyhow!("model storage accounting overflowed"))
    }
}

#[derive(Debug, Clone)]
struct CachedStorageStatus {
    refreshed_at: Instant,
    revision: u64,
    status: ModelStorageStatus,
}

pub(crate) struct ModelStorageManager {
    models_root: PathBuf,
    max_bytes: u64,
    staged_retention: Duration,
    state: Arc<Mutex<StorageState>>,
    revision: AtomicU64,
    cache: RwLock<Option<CachedStorageStatus>>,
}

pub(crate) struct ModelStoragePermit {
    manager: Arc<ModelStorageManager>,
    state: OwnedMutexGuard<StorageState>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelStorageError {
    #[error(
        "Model storage quota exceeded: the operation would commit {prospective_bytes} bytes, but the configured limit is {max_bytes} bytes."
    )]
    Quota {
        prospective_bytes: u64,
        max_bytes: u64,
    },
    #[error("{0}")]
    Internal(String),
}

impl From<anyhow::Error> for ModelStorageError {
    fn from(error: anyhow::Error) -> Self {
        Self::Internal(error.to_string())
    }
}

impl Drop for ModelStoragePermit {
    fn drop(&mut self) {
        self.manager.revision.fetch_add(1, Ordering::AcqRel);
    }
}

impl ModelStorageManager {
    pub(crate) fn new(
        models_root: PathBuf,
        max_bytes: u64,
        staged_retention_seconds: u64,
    ) -> Arc<Self> {
        Arc::new(Self {
            models_root,
            max_bytes,
            staged_retention: Duration::from_secs(staged_retention_seconds),
            state: Arc::new(Mutex::new(StorageState::default())),
            revision: AtomicU64::new(0),
            cache: RwLock::new(None),
        })
    }

    pub(crate) async fn serial(self: &Arc<Self>) -> ModelStoragePermit {
        ModelStoragePermit {
            manager: Arc::clone(self),
            state: Arc::clone(&self.state).lock_owned().await,
        }
    }

    pub(crate) async fn snapshot(&self) -> Result<ModelStorageStatus> {
        let revision = self.revision.load(Ordering::Acquire);
        if let Some(cached) = self.cache.read().await.as_ref()
            && cached.revision == revision
            && cached.refreshed_at.elapsed() < SNAPSHOT_TTL
        {
            return Ok(cached.status.clone());
        }

        let state = self.state.lock().await;
        let scan = scan_storage_async(self.models_root.clone()).await?;
        let runtime_reserved = checked_sum(state.download_reservations.values().copied())?;
        let used_bytes = scan.used_bytes()?;
        let reserved_bytes = scan
            .import_remaining_bytes
            .checked_add(runtime_reserved)
            .ok_or_else(|| anyhow!("model storage reservation accounting overflowed"))?;
        let committed_bytes = used_bytes
            .checked_add(reserved_bytes)
            .ok_or_else(|| anyhow!("model storage commitment accounting overflowed"))?;
        let status = ModelStorageStatus {
            quota_enabled: self.max_bytes > 0,
            max_bytes: (self.max_bytes > 0).then_some(self.max_bytes),
            used_bytes,
            committed_bytes,
            available_bytes: (self.max_bytes > 0)
                .then_some(self.max_bytes.saturating_sub(committed_bytes)),
            installed_bytes: scan.installed_bytes,
            staged_download_bytes: scan.staged_download_bytes,
            staged_import_bytes: scan.staged_import_bytes,
            reserved_bytes,
            staged_retention_seconds: self.staged_retention.as_secs(),
            last_cleanup_at: state.last_cleanup_at,
            last_cleanup_removed_sessions: state.last_cleanup_removed_sessions,
        };
        drop(state);
        *self.cache.write().await = Some(CachedStorageStatus {
            refreshed_at: Instant::now(),
            revision,
            status: status.clone(),
        });
        Ok(status)
    }

    pub(crate) async fn release_download(&self, filename: &str) {
        if self
            .state
            .lock()
            .await
            .download_reservations
            .remove(filename)
            .is_some()
        {
            self.revision.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(crate) async fn mark_download_active(&self, filename: &str) {
        self.state
            .lock()
            .await
            .download_reservations
            .entry(filename.to_string())
            .or_insert(0);
        self.revision.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) async fn cleanup_stale(self: &Arc<Self>) -> Result<u64> {
        self.cleanup_stale_at(SystemTime::now()).await
    }

    async fn cleanup_stale_at(self: &Arc<Self>, now: SystemTime) -> Result<u64> {
        if self.staged_retention.is_zero() {
            return Ok(0);
        }
        let mut permit = self.serial().await;
        let active_downloads = permit
            .state
            .download_reservations
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        let active_imports = permit.state.active_imports.clone();
        let root = self.models_root.clone();
        let retention = self.staged_retention;
        let removed = tokio::task::spawn_blocking(move || {
            cleanup_stale_sessions(&root, retention, now, &active_downloads, &active_imports)
        })
        .await
        .context("model storage cleanup task failed")??;
        permit.state.last_cleanup_at = now
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_secs());
        permit.state.last_cleanup_removed_sessions = removed;
        Ok(removed)
    }
}

impl ModelStoragePermit {
    pub(crate) async fn ensure_capacity(
        &mut self,
        additional_bytes: u64,
        reclaimable_bytes: u64,
    ) -> std::result::Result<(), ModelStorageError> {
        if self.manager.max_bytes == 0 {
            return Ok(());
        }
        let scan = scan_storage_async(self.manager.models_root.clone()).await?;
        let runtime_reserved = checked_sum(self.state.download_reservations.values().copied())?;
        let committed = scan
            .committed_bytes()?
            .checked_add(runtime_reserved)
            .ok_or_else(|| anyhow!("model storage commitment accounting overflowed"))?;
        let prospective = committed
            .saturating_sub(reclaimable_bytes)
            .checked_add(additional_bytes)
            .ok_or_else(|| anyhow!("model storage quota calculation overflowed"))?;
        if prospective > self.manager.max_bytes {
            return Err(ModelStorageError::Quota {
                prospective_bytes: prospective,
                max_bytes: self.manager.max_bytes,
            });
        }
        Ok(())
    }

    pub(crate) async fn reserve_download(
        &mut self,
        filename: &str,
        target_bytes: Option<u64>,
        replaced_bytes: u64,
        retained_bytes: u64,
    ) -> std::result::Result<(), ModelStorageError> {
        let previous = self.state.download_reservations.remove(filename);
        let additional = target_bytes.unwrap_or(retained_bytes);
        if let Err(error) = self.ensure_capacity(additional, replaced_bytes).await {
            if let Some(previous) = previous {
                self.state
                    .download_reservations
                    .insert(filename.to_string(), previous);
            }
            return Err(error);
        }
        let remaining = target_bytes
            .map(|target| target.saturating_sub(retained_bytes))
            .unwrap_or(0);
        self.state
            .download_reservations
            .insert(filename.to_string(), remaining);
        Ok(())
    }

    pub(crate) fn consume_download_reservation(&mut self, filename: &str, bytes: u64) {
        if let Some(remaining) = self.state.download_reservations.get_mut(filename) {
            *remaining = remaining.saturating_sub(bytes);
        }
    }

    pub(crate) fn mark_import_active(&mut self, filename: &str) {
        self.state.active_imports.insert(filename.to_string());
    }
}

impl ModelStorageManager {
    pub(crate) async fn release_import(&self, filename: &str) {
        if self.state.lock().await.active_imports.remove(filename) {
            self.revision.fetch_add(1, Ordering::AcqRel);
        }
    }
}

async fn scan_storage_async(root: PathBuf) -> Result<StorageScan> {
    tokio::task::spawn_blocking(move || scan_storage(&root))
        .await
        .context("model storage scan task failed")?
}

fn scan_storage(root: &Path) -> Result<StorageScan> {
    match std::fs::symlink_metadata(root) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(anyhow!("model catalog root must be a real directory"));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(StorageScan {
                installed_bytes: 0,
                staged_download_bytes: 0,
                staged_import_bytes: 0,
                import_remaining_bytes: 0,
            });
        }
        Err(error) => return Err(error).context("failed to inspect model catalog storage"),
    }

    let mut installed_bytes = 0_u64;
    let mut staged_download_bytes = 0_u64;
    let mut staged_import_bytes = 0_u64;
    let mut import_remaining_bytes = 0_u64;
    let mut inspected = 0_usize;
    for entry in std::fs::read_dir(root).context("failed to read model catalog storage")? {
        let entry = entry.context("failed to read a model catalog storage entry")?;
        inspected += 1;
        ensure_scan_bound(inspected)?;
        let name = entry.file_name();
        if name == METADATA_DIRECTORY {
            continue;
        } else if name == DOWNLOAD_STAGING_DIRECTORY {
            staged_download_bytes = scan_download_staging(&entry.path(), &mut inspected)?;
        } else if name == IMPORT_STAGING_DIRECTORY {
            let import = scan_import_staging(&entry.path(), &mut inspected)?;
            staged_import_bytes = import.0;
            import_remaining_bytes = import.1;
        } else {
            installed_bytes = installed_bytes
                .checked_add(scan_installed_entry(&entry.path(), &mut inspected)?)
                .ok_or_else(|| anyhow!("model storage accounting overflowed"))?;
        }
    }
    Ok(StorageScan {
        installed_bytes,
        staged_download_bytes,
        staged_import_bytes,
        import_remaining_bytes,
    })
}

fn scan_installed_entry(path: &Path, inspected: &mut usize) -> Result<u64> {
    let mut total = 0_u64;
    let mut stack = vec![(path.to_path_buf(), 0_usize)];
    while let Some((path, depth)) = stack.pop() {
        let metadata = std::fs::symlink_metadata(&path).with_context(|| {
            format!("failed to inspect model storage entry '{}'", path.display())
        })?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_file() {
            total = total
                .checked_add(metadata.len())
                .ok_or_else(|| anyhow!("model storage accounting overflowed"))?;
            continue;
        }
        if !metadata.is_dir() {
            continue;
        }
        if depth >= MAX_STORAGE_SCAN_DEPTH {
            return Err(anyhow!(
                "model storage scan exceeded the maximum directory depth of {MAX_STORAGE_SCAN_DEPTH}"
            ));
        }
        for entry in std::fs::read_dir(&path).with_context(|| {
            format!(
                "failed to read model storage directory '{}'",
                path.display()
            )
        })? {
            let entry = entry.context("failed to read a nested model storage entry")?;
            *inspected += 1;
            ensure_scan_bound(*inspected)?;
            stack.push((entry.path(), depth + 1));
        }
    }
    Ok(total)
}

fn scan_download_staging(path: &Path, inspected: &mut usize) -> Result<u64> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect staging directory '{}'", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!("model staging path must be a real directory"));
    }
    let mut actual_bytes = 0_u64;
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("failed to read staging directory '{}'", path.display()))?
    {
        let entry = entry.context("failed to read a model staging entry")?;
        *inspected += 1;
        ensure_scan_bound(*inspected)?;
        let entry_path = entry.path();
        let metadata = std::fs::symlink_metadata(&entry_path)
            .context("failed to inspect a model staging entry")?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            actual_bytes = actual_bytes
                .checked_add(scan_installed_entry(&entry_path, inspected)?)
                .ok_or_else(|| anyhow!("model staging accounting overflowed"))?;
            continue;
        }
        if !metadata.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if name.ends_with(".json") || name.ends_with(".json.tmp") {
            continue;
        }
        actual_bytes = actual_bytes
            .checked_add(metadata.len())
            .ok_or_else(|| anyhow!("model staging accounting overflowed"))?;
    }
    Ok(actual_bytes)
}

fn scan_import_staging(path: &Path, inspected: &mut usize) -> Result<(u64, u64)> {
    let (actual_bytes, parts, metadata_paths) = scan_staging_payload(path, inspected)?;
    let mut remaining = 0_u64;
    for metadata_path in metadata_paths {
        let Some(filename) = metadata_path
            .file_name()
            .and_then(|value| value.to_str())
            .and_then(|value| value.strip_suffix(".json"))
        else {
            continue;
        };
        let Ok(bytes) = std::fs::read(&metadata_path) else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        if value.get("filename").and_then(|value| value.as_str()) != Some(filename) {
            continue;
        }
        let Some(total_bytes) = value.get("total_bytes").and_then(|value| value.as_u64()) else {
            continue;
        };
        let uploaded = parts.get(filename).copied().unwrap_or(0);
        remaining = remaining
            .checked_add(total_bytes.saturating_sub(uploaded))
            .ok_or_else(|| anyhow!("model import commitment accounting overflowed"))?;
    }
    Ok((actual_bytes, remaining))
}

fn scan_staging_payload(
    path: &Path,
    inspected: &mut usize,
) -> Result<(u64, HashMap<String, u64>, Vec<PathBuf>)> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect staging directory '{}'", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!("model staging path must be a real directory"));
    }
    let mut actual_bytes = 0_u64;
    let mut parts = HashMap::new();
    let mut metadata_paths = Vec::new();
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("failed to read staging directory '{}'", path.display()))?
    {
        let entry = entry.context("failed to read a model staging entry")?;
        *inspected += 1;
        ensure_scan_bound(*inspected)?;
        let entry_path = entry.path();
        let metadata = std::fs::symlink_metadata(&entry_path)
            .context("failed to inspect a model staging entry")?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if let Some(filename) = name.strip_suffix(".part") {
            actual_bytes = actual_bytes
                .checked_add(metadata.len())
                .ok_or_else(|| anyhow!("model staging accounting overflowed"))?;
            parts.insert(filename.to_string(), metadata.len());
        } else if name.ends_with(".json") {
            metadata_paths.push(entry_path);
        } else if !name.ends_with(".json.tmp") {
            actual_bytes = actual_bytes
                .checked_add(metadata.len())
                .ok_or_else(|| anyhow!("model staging accounting overflowed"))?;
        }
    }
    Ok((actual_bytes, parts, metadata_paths))
}

fn ensure_scan_bound(inspected: usize) -> Result<()> {
    if inspected > MAX_STORAGE_SCAN_ENTRIES {
        Err(anyhow!(
            "model storage scan exceeded the maximum of {MAX_STORAGE_SCAN_ENTRIES} entries"
        ))
    } else {
        Ok(())
    }
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| anyhow!("model storage accounting overflowed"))
    })
}

#[derive(Default)]
struct StaleSession {
    paths: Vec<PathBuf>,
    newest_modified: Option<SystemTime>,
}

fn staged_session_key(
    staging_kind: &str,
    name: &str,
    metadata: &std::fs::Metadata,
) -> Option<String> {
    let key = if staging_kind == DOWNLOAD_STAGING_DIRECTORY {
        name.strip_suffix(".package.json.tmp")
            .or_else(|| name.strip_suffix(".package.json"))
            .or_else(|| {
                if metadata.is_dir() || metadata.file_type().is_symlink() {
                    name.strip_suffix(".package")
                } else {
                    None
                }
            })
            .or_else(|| name.strip_suffix(".json.tmp"))
            .or_else(|| name.strip_suffix(".json"))
            .or_else(|| name.strip_suffix(".part"))
    } else {
        name.strip_suffix(".json.tmp")
            .or_else(|| name.strip_suffix(".json"))
            .or_else(|| name.strip_suffix(".part"))
    }?;
    (!key.is_empty()).then(|| key.to_string())
}

fn newest_staged_path_modified(root: &Path, inspected: &mut usize) -> Result<Option<SystemTime>> {
    let root_metadata = std::fs::symlink_metadata(root)
        .with_context(|| format!("failed to inspect staged path '{}'", root.display()))?;
    let mut newest = root_metadata.modified().ok();
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Ok(newest);
    }
    let mut pending = vec![(root.to_path_buf(), 0_usize)];
    while let Some((directory, depth)) = pending.pop() {
        if depth >= MAX_STORAGE_SCAN_DEPTH {
            return Err(anyhow!(
                "model staging cleanup exceeded the maximum directory depth of {MAX_STORAGE_SCAN_DEPTH}"
            ));
        }
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("failed to read staged directory '{}'", directory.display()))?
        {
            let entry = entry.context("failed to read a staged model entry")?;
            *inspected += 1;
            ensure_scan_bound(*inspected)?;
            let metadata = std::fs::symlink_metadata(entry.path())
                .context("failed to inspect a staged model entry")?;
            if let Ok(modified) = metadata.modified() {
                newest = Some(newest.map_or(modified, |current| current.max(modified)));
            }
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                pending.push((entry.path(), depth + 1));
            }
        }
    }
    Ok(newest)
}

fn remove_staged_path(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || metadata.is_file() => {
            std::fs::remove_file(path)
                .with_context(|| format!("failed to remove staged path '{}'", path.display()))
        }
        Ok(metadata) if metadata.is_dir() => std::fs::remove_dir_all(path)
            .with_context(|| format!("failed to remove staged directory '{}'", path.display())),
        Ok(_) => Err(anyhow!(
            "staged path '{}' is not a file, directory, or symlink",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect staged path '{}'", path.display())),
    }
}

fn cleanup_stale_sessions(
    root: &Path,
    retention: Duration,
    now: SystemTime,
    active_downloads: &HashSet<String>,
    active_imports: &HashSet<String>,
) -> Result<u64> {
    let mut removed = 0_u64;
    for (directory, active) in [
        (DOWNLOAD_STAGING_DIRECTORY, active_downloads),
        (IMPORT_STAGING_DIRECTORY, active_imports),
    ] {
        let staging = root.join(directory);
        let metadata = match std::fs::symlink_metadata(&staging) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).context("failed to inspect model staging cleanup root");
            }
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(anyhow!(
                "model staging cleanup root must be a real directory"
            ));
        }
        let mut sessions = HashMap::<String, StaleSession>::new();
        let mut inspected = 0_usize;
        for entry in std::fs::read_dir(&staging)
            .with_context(|| format!("failed to read staging directory '{}'", staging.display()))?
        {
            let entry = entry.context("failed to read a staged model session")?;
            inspected += 1;
            ensure_scan_bound(inspected)?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let metadata = std::fs::symlink_metadata(entry.path())
                .context("failed to inspect a staged model session file")?;
            let Some(key) = staged_session_key(directory, name, &metadata) else {
                continue;
            };
            let modified = newest_staged_path_modified(&entry.path(), &mut inspected)?;
            let session = sessions.entry(key).or_default();
            session.paths.push(entry.path());
            if let Some(modified) = modified {
                session.newest_modified = Some(
                    session
                        .newest_modified
                        .map_or(modified, |current| current.max(modified)),
                );
            }
        }
        for (key, session) in sessions {
            if active.contains(&key) {
                continue;
            }
            let Some(modified) = session.newest_modified else {
                continue;
            };
            if !now
                .duration_since(modified)
                .is_ok_and(|age| age >= retention)
            {
                continue;
            }
            for path in session.paths {
                remove_staged_path(&path)?;
            }
            removed = removed.saturating_add(1);
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn accounts_for_installed_staged_and_declared_import_bytes() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("installed.gguf"), vec![0_u8; 10]).unwrap();
        let downloads = temp.path().join(DOWNLOAD_STAGING_DIRECTORY);
        std::fs::create_dir(&downloads).unwrap();
        std::fs::write(downloads.join("download.gguf.part"), vec![0_u8; 5]).unwrap();
        let package = downloads.join("downloaded-package.package");
        std::fs::create_dir(&package).unwrap();
        std::fs::write(package.join("config.json"), vec![0_u8; 11]).unwrap();
        std::fs::create_dir(package.join("tokenizer")).unwrap();
        std::fs::write(
            package.join("tokenizer/tokenizer.json.part"),
            vec![0_u8; 13],
        )
        .unwrap();
        std::fs::write(
            downloads.join("downloaded-package.package.json"),
            br#"{"directory":"downloaded-package"}"#,
        )
        .unwrap();
        let imports = temp.path().join(IMPORT_STAGING_DIRECTORY);
        std::fs::create_dir(&imports).unwrap();
        std::fs::write(imports.join("import.gguf.part"), vec![0_u8; 7]).unwrap();
        std::fs::write(
            imports.join("import.gguf.json"),
            br#"{"filename":"import.gguf","total_bytes":20}"#,
        )
        .unwrap();
        let manager = ModelStorageManager::new(temp.path().to_path_buf(), 100, 0);

        let status = manager.snapshot().await.unwrap();

        assert_eq!(status.installed_bytes, 10);
        assert_eq!(status.staged_download_bytes, 29);
        assert_eq!(status.staged_import_bytes, 7);
        assert_eq!(status.used_bytes, 46);
        assert_eq!(status.reserved_bytes, 13);
        assert_eq!(status.committed_bytes, 59);
        assert_eq!(status.available_bytes, Some(41));
    }

    #[tokio::test]
    async fn shared_reservations_prevent_competing_acquisitions_from_exceeding_quota() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("installed.gguf"), vec![0_u8; 40]).unwrap();
        let manager = ModelStorageManager::new(temp.path().to_path_buf(), 100, 0);
        let mut first = manager.serial().await;
        first
            .reserve_download("first.gguf", Some(50), 0, 0)
            .await
            .unwrap();
        drop(first);
        let mut second = manager.serial().await;

        let error = second.ensure_capacity(20, 0).await.unwrap_err();

        assert!(error.to_string().contains("quota exceeded"));
        drop(second);
        manager.release_download("first.gguf").await;
        let mut retried = manager.serial().await;
        retried.ensure_capacity(20, 0).await.unwrap();
    }

    #[tokio::test]
    async fn cleanup_removes_only_expired_inactive_sessions() {
        let temp = tempfile::tempdir().unwrap();
        let downloads = temp.path().join(DOWNLOAD_STAGING_DIRECTORY);
        let imports = temp.path().join(IMPORT_STAGING_DIRECTORY);
        std::fs::create_dir(&downloads).unwrap();
        std::fs::create_dir(&imports).unwrap();
        std::fs::write(downloads.join("active.gguf.part"), b"active").unwrap();
        std::fs::write(downloads.join("stale.gguf.part"), b"stale").unwrap();
        let active_package = downloads.join("active-package.package");
        std::fs::create_dir(&active_package).unwrap();
        std::fs::write(active_package.join("config.json.part"), b"active package").unwrap();
        std::fs::write(
            downloads.join("active-package.package.json"),
            b"active package metadata",
        )
        .unwrap();
        let stale_package = downloads.join("stale-package.package");
        std::fs::create_dir(&stale_package).unwrap();
        std::fs::create_dir(stale_package.join("tokenizer")).unwrap();
        std::fs::write(
            stale_package.join("tokenizer/tokenizer.json.part"),
            b"stale package",
        )
        .unwrap();
        std::fs::write(
            downloads.join("stale-package.package.json"),
            b"stale package metadata",
        )
        .unwrap();
        std::fs::write(imports.join("active-import.gguf.json"), b"{}").unwrap();
        std::fs::write(imports.join("stale.gguf.json"), b"{}").unwrap();
        let manager = ModelStorageManager::new(temp.path().to_path_buf(), 0, 60);
        manager.mark_download_active("active.gguf").await;
        manager.mark_download_active("active-package").await;
        {
            let mut permit = manager.serial().await;
            permit.mark_import_active("active-import.gguf");
        }

        let removed = manager
            .cleanup_stale_at(SystemTime::now() + Duration::from_secs(61))
            .await
            .unwrap();

        assert_eq!(removed, 3);
        assert!(downloads.join("active.gguf.part").exists());
        assert!(active_package.exists());
        assert!(downloads.join("active-package.package.json").exists());
        assert!(imports.join("active-import.gguf.json").exists());
        assert!(!downloads.join("stale.gguf.part").exists());
        assert!(!stale_package.exists());
        assert!(!downloads.join("stale-package.package.json").exists());
        assert!(!imports.join("stale.gguf.json").exists());
    }

    #[tokio::test]
    async fn restarted_package_staging_remains_part_of_quota_admission() {
        let temp = tempfile::tempdir().unwrap();
        let downloads = temp.path().join(DOWNLOAD_STAGING_DIRECTORY);
        let package = downloads.join("resume-package.package");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(package.join("model.safetensors.part"), vec![0_u8; 70]).unwrap();
        let manager = ModelStorageManager::new(temp.path().to_path_buf(), 100, 0);

        let status = manager.snapshot().await.unwrap();
        assert_eq!(status.staged_download_bytes, 70);
        assert_eq!(status.available_bytes, Some(30));

        let mut permit = manager.serial().await;
        let error = permit.ensure_capacity(31, 0).await.unwrap_err();
        assert!(error.to_string().contains("quota exceeded"));
    }

    #[tokio::test]
    async fn package_staging_accounting_fails_closed_at_the_depth_bound() {
        let temp = tempfile::tempdir().unwrap();
        let downloads = temp.path().join(DOWNLOAD_STAGING_DIRECTORY);
        let mut nested = downloads.join("too-deep.package");
        std::fs::create_dir_all(&nested).unwrap();
        for level in 0..MAX_STORAGE_SCAN_DEPTH {
            nested = nested.join(format!("level-{level}"));
            std::fs::create_dir(&nested).unwrap();
        }
        let manager = ModelStorageManager::new(temp.path().to_path_buf(), 0, 0);

        let error = manager.snapshot().await.unwrap_err();

        assert!(
            error
                .to_string()
                .contains("exceeded the maximum directory depth")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stale_package_cleanup_never_follows_a_staging_symlink() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("keep.bin"), b"keep").unwrap();
        let downloads = temp.path().join(DOWNLOAD_STAGING_DIRECTORY);
        std::fs::create_dir(&downloads).unwrap();
        symlink(outside.path(), downloads.join("orphan.package")).unwrap();
        std::fs::write(downloads.join("orphan.package.json"), b"orphan metadata").unwrap();
        let manager = ModelStorageManager::new(temp.path().to_path_buf(), 0, 60);

        let removed = manager
            .cleanup_stale_at(SystemTime::now() + Duration::from_secs(61))
            .await
            .unwrap();

        assert_eq!(removed, 1);
        assert!(!downloads.join("orphan.package").exists());
        assert!(!downloads.join("orphan.package.json").exists());
        assert_eq!(
            std::fs::read(outside.path().join("keep.bin")).unwrap(),
            b"keep"
        );
    }
}
