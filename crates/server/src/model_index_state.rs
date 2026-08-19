//! Bounded, crash-safe rollback watermarks for signed model indexes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

const WATERMARK_SCHEMA_VERSION: u8 = 1;
const WATERMARK_OBJECT: &str = "bloom.model_index_watermark";
const WATERMARK_PREFIX: &str = "watermark-";
const WATERMARK_SUFFIX: &str = ".json";
const MAX_WATERMARK_BYTES: u64 = 1_024;
const MAX_WATERMARK_RECORDS: usize = 64;
const MAX_TEMPORARY_RECORDS: usize = 16;
const RETAINED_GENERATIONS_PER_SOURCE: usize = 2;
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WatermarkRecord {
    schema_version: u8,
    object: String,
    source_id: String,
    generated_at: u64,
    generation_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelIndexWatermark {
    pub generated_at: u64,
    pub generation_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelIndexWatermarkAdmission {
    Accepted,
    Rollback(ModelIndexWatermark),
    Conflict(ModelIndexWatermark),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelIndexWatermarkDirectoryStatus {
    pub exists: bool,
    pub record_count: usize,
    pub source_count: usize,
}

pub(crate) struct ModelIndexWatermarkStore {
    directory: PathBuf,
    source_id: String,
}

enum PublishOutcome {
    Published,
    Conflict(WatermarkRecord),
}

impl ModelIndexWatermarkStore {
    pub(crate) fn new(directory: PathBuf, source_id: String) -> Result<Self> {
        validate_directory_path(&directory)?;
        validate_digest(&source_id, "model index source ID")?;
        inspect_model_index_watermark_directory(&directory)?;
        Ok(Self {
            directory,
            source_id,
        })
    }

    pub(crate) async fn admit(
        &self,
        candidate: ModelIndexWatermark,
    ) -> Result<ModelIndexWatermarkAdmission> {
        validate_candidate(&candidate)?;
        let directory = self.directory.clone();
        let source_id = self.source_id.clone();
        tokio::task::spawn_blocking(move || admit_sync(&directory, &source_id, candidate))
            .await
            .context("model index watermark task failed")?
    }
}

pub(crate) fn inspect_model_index_watermark_directory(
    directory: &Path,
) -> Result<ModelIndexWatermarkDirectoryStatus> {
    validate_directory_path(directory)?;
    let records = read_records(directory)?;
    let source_count = records
        .iter()
        .map(|record| record.source_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    Ok(ModelIndexWatermarkDirectoryStatus {
        exists: directory.exists(),
        record_count: records.len(),
        source_count,
    })
}

fn admit_sync(
    directory: &Path,
    source_id: &str,
    candidate: ModelIndexWatermark,
) -> Result<ModelIndexWatermarkAdmission> {
    ensure_directory(directory)?;
    let mut records = read_records(directory)?;
    if let Some(current) = latest_watermark(&records, source_id)? {
        match candidate.generated_at.cmp(&current.generated_at) {
            std::cmp::Ordering::Less => return Ok(ModelIndexWatermarkAdmission::Rollback(current)),
            std::cmp::Ordering::Equal if candidate.generation_id == current.generation_id => {
                return Ok(ModelIndexWatermarkAdmission::Accepted);
            }
            std::cmp::Ordering::Equal => {
                return Ok(ModelIndexWatermarkAdmission::Conflict(current));
            }
            std::cmp::Ordering::Greater => {}
        }
    }

    let source_record_count = records
        .iter()
        .filter(|record| record.source_id == source_id)
        .count();
    if records.len() >= MAX_WATERMARK_RECORDS
        && source_record_count < RETAINED_GENERATIONS_PER_SOURCE
    {
        return Err(anyhow!(
            "model index watermark directory has no capacity for another source history"
        ));
    }

    let candidate_record = WatermarkRecord {
        schema_version: WATERMARK_SCHEMA_VERSION,
        object: WATERMARK_OBJECT.to_string(),
        source_id: source_id.to_string(),
        generated_at: candidate.generated_at,
        generation_id: candidate.generation_id.clone(),
    };
    let publication = publish_record(directory, &candidate_record)?;
    if let PublishOutcome::Conflict(existing) = publication {
        return Ok(ModelIndexWatermarkAdmission::Conflict(
            ModelIndexWatermark {
                generated_at: existing.generated_at,
                generation_id: existing.generation_id,
            },
        ));
    }
    records.push(candidate_record);
    prune_old_generations(directory, &records, source_id)?;

    // Re-read after publication. Immutable per-generation filenames make this
    // safe across processes: a concurrent higher generation wins, while an
    // equal-time conflict cannot overwrite the record that was linked first.
    records = read_records(directory)?;
    let current = latest_watermark(&records, source_id)?
        .ok_or_else(|| anyhow!("published model index watermark could not be read back"))?;
    let admission = match candidate.generated_at.cmp(&current.generated_at) {
        std::cmp::Ordering::Less => ModelIndexWatermarkAdmission::Rollback(current),
        std::cmp::Ordering::Equal if candidate.generation_id == current.generation_id => {
            ModelIndexWatermarkAdmission::Accepted
        }
        std::cmp::Ordering::Equal => ModelIndexWatermarkAdmission::Conflict(current),
        std::cmp::Ordering::Greater => {
            return Err(anyhow!(
                "persisted model index watermark moved backwards during publication"
            ));
        }
    };
    Ok(admission)
}

fn latest_watermark(
    records: &[WatermarkRecord],
    source_id: &str,
) -> Result<Option<ModelIndexWatermark>> {
    let Some(latest_generation) = records
        .iter()
        .filter(|record| record.source_id == source_id)
        .map(|record| record.generated_at)
        .max()
    else {
        return Ok(None);
    };
    let generation_ids = records
        .iter()
        .filter(|record| record.source_id == source_id && record.generated_at == latest_generation)
        .map(|record| record.generation_id.as_str())
        .collect::<BTreeSet<_>>();
    if generation_ids.len() != 1 {
        return Err(anyhow!(
            "persisted model index watermarks conflict at the latest generation"
        ));
    }
    Ok(Some(ModelIndexWatermark {
        generated_at: latest_generation,
        generation_id: generation_ids
            .into_iter()
            .next()
            .unwrap_or_default()
            .to_string(),
    }))
}

fn publish_record(directory: &Path, record: &WatermarkRecord) -> Result<PublishOutcome> {
    let bytes = format!("{}\n", serde_json::to_string_pretty(record)?).into_bytes();
    if bytes.is_empty() || bytes.len() as u64 > MAX_WATERMARK_BYTES {
        return Err(anyhow!("model index watermark record is too large"));
    }
    let final_path = directory.join(record_filename(record));
    if final_path.exists() {
        let existing = read_record(&final_path)?;
        if existing == *record {
            return Ok(PublishOutcome::Published);
        }
        return Ok(PublishOutcome::Conflict(existing));
    }

    let mut temporary = None;
    for _ in 0..=MAX_TEMPORARY_RECORDS {
        let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = directory.join(format!(
            ".{}.tmp-{}-{sequence}",
            record_filename(record),
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).context("failed to create model index watermark temporary file");
            }
        }
    }
    let (temporary_path, mut temporary_file) = temporary
        .ok_or_else(|| anyhow!("could not allocate a model index watermark temporary file"))?;
    let publication = (|| -> Result<PublishOutcome> {
        temporary_file
            .write_all(&bytes)
            .context("failed to write model index watermark")?;
        temporary_file
            .sync_all()
            .context("failed to sync model index watermark")?;
        drop(temporary_file);
        match fs::hard_link(&temporary_path, &final_path) {
            Ok(()) => sync_directory(directory)?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let existing = read_record(&final_path)?;
                if existing != *record {
                    return Ok(PublishOutcome::Conflict(existing));
                }
            }
            Err(error) => {
                return Err(error).context("failed to publish model index watermark atomically");
            }
        }
        Ok(PublishOutcome::Published)
    })();
    let cleanup = fs::remove_file(&temporary_path);
    match publication {
        Ok(outcome) => {
            cleanup.context("failed to remove model index watermark temporary file")?;
            Ok(outcome)
        }
        Err(error) => {
            let _ = cleanup;
            Err(error)
        }
    }
}

fn prune_old_generations(
    directory: &Path,
    records: &[WatermarkRecord],
    source_id: &str,
) -> Result<()> {
    let by_generation = records
        .iter()
        .filter(|record| record.source_id == source_id)
        .map(|record| (record.generated_at, record))
        .collect::<BTreeMap<_, _>>();
    let removable = by_generation
        .values()
        .rev()
        .skip(RETAINED_GENERATIONS_PER_SOURCE)
        .copied()
        .collect::<Vec<_>>();
    if removable.is_empty() {
        return Ok(());
    }
    for record in removable {
        let path = directory.join(record_filename(record));
        let metadata =
            fs::symlink_metadata(&path).context("failed to inspect old model index watermark")?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(anyhow!(
                "model index watermark records must be regular, non-symlink files"
            ));
        }
        fs::remove_file(path).context("failed to prune old model index watermark")?;
    }
    sync_directory(directory)
}

fn read_records(directory: &Path) -> Result<Vec<WatermarkRecord>> {
    let metadata = match fs::symlink_metadata(directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).context("failed to inspect model index watermark directory");
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!(
            "model index watermark path must be a real directory"
        ));
    }
    validate_directory_permissions(&metadata)?;
    let mut records = Vec::new();
    let mut temporary_count = 0;
    for entry in
        fs::read_dir(directory).context("failed to read model index watermark directory")?
    {
        let entry = entry.context("failed to read model index watermark entry")?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow!("model index watermark filenames must be UTF-8"))?;
        let file_type = entry
            .file_type()
            .context("failed to inspect model index watermark entry")?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(anyhow!(
                "model index watermark directory contains a non-file entry"
            ));
        }
        if temporary_filename(&name) {
            temporary_count += 1;
            if temporary_count > MAX_TEMPORARY_RECORDS {
                return Err(anyhow!(
                    "model index watermark directory contains too many temporary records"
                ));
            }
            continue;
        }
        if !name.starts_with(WATERMARK_PREFIX) || !name.ends_with(WATERMARK_SUFFIX) {
            return Err(anyhow!(
                "model index watermark directory contains an unexpected file"
            ));
        }
        records.push(read_record(&entry.path())?);
        if records.len() > MAX_WATERMARK_RECORDS {
            return Err(anyhow!(
                "model index watermark directory contains too many records"
            ));
        }
    }
    Ok(records)
}

fn read_record(path: &Path) -> Result<WatermarkRecord> {
    let metadata =
        fs::symlink_metadata(path).context("failed to inspect model index watermark record")?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAX_WATERMARK_BYTES
    {
        return Err(anyhow!("model index watermark record is invalid"));
    }
    let bytes = fs::read(path).context("failed to read model index watermark record")?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_WATERMARK_BYTES {
        return Err(anyhow!(
            "model index watermark record changed while reading"
        ));
    }
    let record = serde_json::from_slice::<WatermarkRecord>(&bytes)
        .context("model index watermark record contains invalid JSON")?;
    validate_record(&record)?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("model index watermark filename is invalid"))?;
    if filename != record_filename(&record) {
        return Err(anyhow!(
            "model index watermark filename does not match its content"
        ));
    }
    Ok(record)
}

fn validate_record(record: &WatermarkRecord) -> Result<()> {
    if record.schema_version != WATERMARK_SCHEMA_VERSION || record.object != WATERMARK_OBJECT {
        return Err(anyhow!("unsupported model index watermark identity"));
    }
    validate_digest(&record.source_id, "model index watermark source ID")?;
    validate_digest(&record.generation_id, "model index watermark generation ID")?;
    if record.generated_at == 0 {
        return Err(anyhow!(
            "model index watermark generation must be greater than zero"
        ));
    }
    Ok(())
}

fn validate_candidate(candidate: &ModelIndexWatermark) -> Result<()> {
    if candidate.generated_at == 0 {
        return Err(anyhow!(
            "model index watermark generation must be greater than zero"
        ));
    }
    validate_digest(
        &candidate.generation_id,
        "model index watermark generation ID",
    )
}

fn validate_digest(value: &str, name: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(anyhow!(
            "{name} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_directory_path(directory: &Path) -> Result<()> {
    if directory.as_os_str().is_empty() {
        return Err(anyhow!(
            "model index watermark directory path must not be empty"
        ));
    }
    Ok(())
}

fn ensure_directory(directory: &Path) -> Result<()> {
    #[cfg(unix)]
    let existed = fs::symlink_metadata(directory).is_ok();
    create_private_directory(directory)?;
    let metadata = fs::symlink_metadata(directory)
        .context("failed to inspect model index watermark directory")?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(anyhow!(
            "model index watermark path must be a real directory"
        ));
    }
    #[cfg(unix)]
    if !existed {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
            .context("failed to protect model index watermark directory")?;
    }
    let metadata = fs::symlink_metadata(directory)
        .context("failed to inspect model index watermark directory permissions")?;
    validate_directory_permissions(&metadata)?;
    Ok(())
}

#[cfg(unix)]
fn create_private_directory(directory: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt as _;

    // Apply the restrictive mode in the directory-creation syscall. A later
    // chmod alone leaves a window where another process can observe or use a
    // group/other-writable directory when the deployment umask permits it.
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true).mode(0o700);
    builder
        .create(directory)
        .context("failed to create model index watermark directory")
}

#[cfg(not(unix))]
fn create_private_directory(directory: &Path) -> Result<()> {
    fs::create_dir_all(directory).context("failed to create model index watermark directory")
}

#[cfg(unix)]
fn validate_directory_permissions(metadata: &fs::Metadata) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    if metadata.permissions().mode() & 0o022 != 0 {
        return Err(anyhow!(
            "model index watermark directory must not be writable by group or other users"
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_directory_permissions(_metadata: &fs::Metadata) -> Result<()> {
    Ok(())
}

fn record_filename(record: &WatermarkRecord) -> String {
    format!(
        "{WATERMARK_PREFIX}{}-{}{WATERMARK_SUFFIX}",
        record.source_id, record.generated_at
    )
}

fn temporary_filename(name: &str) -> bool {
    name.starts_with(&format!(".{WATERMARK_PREFIX}")) && name.contains(".json.tmp-")
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<()> {
    fs::File::open(directory)
        .context("failed to open model index watermark directory for sync")?
        .sync_all()
        .context("failed to sync model index watermark directory")
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(generated_at: u64, byte: &str) -> ModelIndexWatermark {
        ModelIndexWatermark {
            generated_at,
            generation_id: byte.repeat(64),
        }
    }

    #[tokio::test]
    async fn persists_advances_and_rejects_rollbacks_across_store_instances() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let source_id = "ab".repeat(32);
        let store = ModelIndexWatermarkStore::new(state.clone(), source_id.clone()).unwrap();
        assert_eq!(
            store.admit(candidate(10, "a")).await.unwrap(),
            ModelIndexWatermarkAdmission::Accepted
        );
        assert_eq!(
            store.admit(candidate(20, "b")).await.unwrap(),
            ModelIndexWatermarkAdmission::Accepted
        );
        assert_eq!(
            store.admit(candidate(30, "c")).await.unwrap(),
            ModelIndexWatermarkAdmission::Accepted
        );
        let restarted = ModelIndexWatermarkStore::new(state.clone(), source_id).unwrap();
        assert_eq!(
            restarted.admit(candidate(10, "a")).await.unwrap(),
            ModelIndexWatermarkAdmission::Rollback(candidate(30, "c"))
        );
        let status = inspect_model_index_watermark_directory(&state).unwrap();
        assert_eq!(status.record_count, 2);
        assert_eq!(status.source_count, 1);
    }

    #[tokio::test]
    async fn same_generation_conflicts_fail_closed_and_source_ids_are_isolated() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let first = ModelIndexWatermarkStore::new(state.clone(), "ab".repeat(32)).unwrap();
        let second = ModelIndexWatermarkStore::new(state.clone(), "cd".repeat(32)).unwrap();
        first.admit(candidate(10, "a")).await.unwrap();
        assert_eq!(
            first.admit(candidate(10, "b")).await.unwrap(),
            ModelIndexWatermarkAdmission::Conflict(candidate(10, "a"))
        );
        assert_eq!(
            second.admit(candidate(5, "c")).await.unwrap(),
            ModelIndexWatermarkAdmission::Accepted
        );
        let status = inspect_model_index_watermark_directory(&state).unwrap();
        assert_eq!(status.source_count, 2);
    }

    #[tokio::test]
    async fn concurrent_equal_time_generations_cannot_overwrite_each_other() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        let source_id = "ab".repeat(32);
        let first = ModelIndexWatermarkStore::new(state.clone(), source_id.clone()).unwrap();
        let second = ModelIndexWatermarkStore::new(state, source_id).unwrap();
        let (left, right) = tokio::join!(
            first.admit(candidate(10, "a")),
            second.admit(candidate(10, "b"))
        );
        let left = left.unwrap();
        let right = right.unwrap();
        assert!(matches!(
            (&left, &right),
            (
                ModelIndexWatermarkAdmission::Accepted,
                ModelIndexWatermarkAdmission::Conflict(_)
            ) | (
                ModelIndexWatermarkAdmission::Conflict(_),
                ModelIndexWatermarkAdmission::Accepted
            )
        ));
    }

    #[tokio::test]
    async fn global_record_bound_fails_before_mutating_another_source_history() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        for source in 1_u64..=32 {
            let store =
                ModelIndexWatermarkStore::new(state.clone(), format!("{source:064x}")).unwrap();
            store.admit(candidate(10, "a")).await.unwrap();
            store.admit(candidate(20, "b")).await.unwrap();
        }
        let full = inspect_model_index_watermark_directory(&state).unwrap();
        assert_eq!(full.record_count, MAX_WATERMARK_RECORDS);

        let additional =
            ModelIndexWatermarkStore::new(state.clone(), format!("{:064x}", 33_u64)).unwrap();
        assert!(additional.admit(candidate(10, "c")).await.is_err());
        assert_eq!(
            inspect_model_index_watermark_directory(&state)
                .unwrap()
                .record_count,
            MAX_WATERMARK_RECORDS
        );
    }

    #[test]
    fn rejects_corrupt_unknown_and_symlink_state() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("unexpected.txt"), b"unexpected").unwrap();
        assert!(inspect_model_index_watermark_directory(temp.path()).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;

            let outside = tempfile::tempdir().unwrap();
            let link = temp.path().join("linked-state");
            std::os::unix::fs::symlink(outside.path(), &link).unwrap();
            assert!(ModelIndexWatermarkStore::new(link, "ab".repeat(32)).is_err());

            let writable = temp.path().join("writable-state");
            fs::create_dir(&writable).unwrap();
            fs::set_permissions(&writable, fs::Permissions::from_mode(0o777)).unwrap();
            assert!(ModelIndexWatermarkStore::new(writable, "ab".repeat(32)).is_err());
        }
    }
}
