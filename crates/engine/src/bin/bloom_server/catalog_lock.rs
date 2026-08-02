//! Cross-process ownership for one mutable model catalog.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

pub(crate) const CATALOG_LOCK_FILE: &str = ".bloom-catalog.lock";

/// An operating-system-backed exclusive lease held for the server lifetime.
///
/// The lock file is intentionally persistent and empty. Process exit releases
/// the kernel lock, so a crash cannot leave a stale logical owner behind.
pub(crate) struct ModelCatalogLease {
    file: File,
    canonical_root: PathBuf,
}

impl ModelCatalogLease {
    pub(crate) fn acquire(models_root: &Path) -> Result<Self> {
        ensure_real_catalog_root(models_root)?;
        let canonical_root = models_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve model catalog root '{}' for ownership",
                models_root.display()
            )
        })?;
        let lock_path = canonical_root.join(CATALOG_LOCK_FILE);
        reject_existing_symlink(&lock_path)?;
        let file = open_lock_file(&lock_path).with_context(|| {
            format!(
                "failed to open model catalog ownership file '{}'",
                lock_path.display()
            )
        })?;
        let metadata = file.metadata().with_context(|| {
            format!(
                "failed to inspect model catalog ownership file '{}'",
                lock_path.display()
            )
        })?;
        if !metadata.is_file() {
            return Err(anyhow!(
                "model catalog ownership path must be a regular file: {}",
                lock_path.display()
            ));
        }
        if metadata.len() != 0 {
            return Err(anyhow!(
                "model catalog ownership file must be empty: {}",
                lock_path.display()
            ));
        }
        validate_lock_permissions(&metadata, &lock_path)?;

        match file.try_lock() {
            Ok(()) => Ok(Self {
                file,
                canonical_root,
            }),
            Err(TryLockError::WouldBlock) => Err(anyhow!(
                "model catalog is already owned by another Bloom server: {}",
                canonical_root.display()
            )),
            Err(TryLockError::Error(error)) => Err(error).with_context(|| {
                format!(
                    "failed to acquire exclusive ownership of model catalog '{}'",
                    canonical_root.display()
                )
            }),
        }
    }

    #[cfg(test)]
    fn canonical_root(&self) -> &Path {
        &self.canonical_root
    }
}

impl Drop for ModelCatalogLease {
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            tracing::warn!(
                %error,
                models_root = %self.canonical_root.display(),
                "Failed to release the model catalog ownership lease"
            );
        }
    }
}

fn ensure_real_catalog_root(models_root: &Path) -> Result<()> {
    match fs::symlink_metadata(models_root) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => Err(anyhow!(
            "model catalog root must be a real directory: {}",
            models_root.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(models_root).with_context(|| {
                format!(
                    "failed to create model catalog root '{}'",
                    models_root.display()
                )
            })?;
            let metadata = fs::symlink_metadata(models_root).with_context(|| {
                format!(
                    "failed to inspect model catalog root '{}' after creation",
                    models_root.display()
                )
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(anyhow!(
                    "model catalog root must be a real directory: {}",
                    models_root.display()
                ));
            }
            Ok(())
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to inspect model catalog root '{}'",
                models_root.display()
            )
        }),
    }
}

fn reject_existing_symlink(lock_path: &Path) -> Result<()> {
    match fs::symlink_metadata(lock_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(anyhow!(
            "model catalog ownership path must not be a symbolic link: {}",
            lock_path.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to inspect model catalog ownership path '{}'",
                lock_path.display()
            )
        }),
    }
}

#[cfg(unix)]
fn open_lock_file(lock_path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt as _;

    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(lock_path)
}

#[cfg(not(unix))]
fn open_lock_file(lock_path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(lock_path)
}

#[cfg(unix)]
fn validate_lock_permissions(metadata: &fs::Metadata, lock_path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    if metadata.permissions().mode() & 0o022 != 0 {
        return Err(anyhow!(
            "model catalog ownership file is writable by group or other users: {}",
            lock_path.display()
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_lock_permissions(_metadata: &fs::Metadata, _lock_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    const CHILD_ROOT_ENV: &str = "BLOOM_TEST_CATALOG_LEASE_CHILD_ROOT";
    const CHILD_READY_FILE: &str = ".catalog-lease-child-ready";

    #[test]
    fn catalog_lease_child_process() {
        let Some(root) = std::env::var_os(CHILD_ROOT_ENV).map(PathBuf::from) else {
            return;
        };
        let _lease = ModelCatalogLease::acquire(&root).unwrap();
        fs::write(root.join(CHILD_READY_FILE), b"ready").unwrap();

        let mut byte = [0_u8; 1];
        use std::io::Read as _;
        let _ = std::io::stdin().read(&mut byte);
    }

    #[test]
    fn exclusive_lease_is_released_when_the_owner_process_exits() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        let ready = root.join(CHILD_READY_FILE);
        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("catalog_lease_child_process")
            .arg("--nocapture")
            .env(CHILD_ROOT_ENV, &root)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !ready.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        if !ready.exists() {
            let _ = child.kill();
            let _ = child.wait();
            panic!("catalog lease child did not become ready");
        }

        let error = ModelCatalogLease::acquire(&root)
            .err()
            .expect("the child process must retain exclusive ownership");
        assert!(error
            .to_string()
            .contains("already owned by another Bloom server"));

        drop(child.stdin.take());
        assert!(child.wait().unwrap().success());
        let lease = ModelCatalogLease::acquire(&root).unwrap();
        assert_eq!(lease.canonical_root(), root.canonicalize().unwrap());
    }

    #[test]
    fn stale_unlocked_file_is_reusable() {
        let temp = tempfile::tempdir().unwrap();
        drop(open_lock_file(&temp.path().join(CATALOG_LOCK_FILE)).unwrap());

        let first = ModelCatalogLease::acquire(temp.path()).unwrap();
        drop(first);
        ModelCatalogLease::acquire(temp.path()).unwrap();
    }

    #[test]
    fn nonempty_lock_file_is_rejected_without_modification() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(CATALOG_LOCK_FILE);
        drop(open_lock_file(&path).unwrap());
        fs::write(&path, b"untrusted owner metadata").unwrap();

        let error = ModelCatalogLease::acquire(temp.path())
            .err()
            .expect("owner metadata must not be trusted or rewritten");
        assert!(error.to_string().contains("must be empty"));
        assert_eq!(fs::read(path).unwrap(), b"untrusted owner metadata");
    }

    #[cfg(unix)]
    #[test]
    fn group_writable_lock_file_is_rejected() {
        use std::os::unix::fs::PermissionsExt as _;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(CATALOG_LOCK_FILE);
        drop(open_lock_file(&path).unwrap());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o620)).unwrap();

        let error = ModelCatalogLease::acquire(temp.path())
            .err()
            .expect("a group-writable ownership file must fail closed");
        assert!(error.to_string().contains("writable by group or other"));
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_link_lock_is_rejected_without_following_it() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        fs::write(outside.path(), b"outside").unwrap();
        symlink(outside.path(), temp.path().join(CATALOG_LOCK_FILE)).unwrap();

        let error = ModelCatalogLease::acquire(temp.path())
            .err()
            .expect("a symbolic-link lock must fail closed");
        assert!(error.to_string().contains("must not be a symbolic link"));
        assert_eq!(fs::read(outside.path()).unwrap(), b"outside");
    }

    #[test]
    fn non_directory_catalog_root_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        fs::write(&root, b"not a directory").unwrap();

        let error = ModelCatalogLease::acquire(&root)
            .err()
            .expect("a non-directory catalog root must fail closed");
        assert!(error.to_string().contains("must be a real directory"));
    }
}
