use anyhow::{Context, Result, anyhow, ensure};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use tempfile::TempDir;

const GENERATE_KERNEL_SCRIPT: &str = include_str!("../scripts/generate_kernel.py");
// Paths returned by the public API remain usable after a compiler is dropped.
// Never reuse executable artifacts from another process or a shared /tmp path.
static PROCESS_CACHE: OnceLock<TempDir> = OnceLock::new();
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

pub struct TileLangCompiler {
    cache_dir: PathBuf,
    python: PathBuf,
    backend: String,
}

impl TileLangCompiler {
    pub fn new() -> Result<Self> {
        Self::with_backend(&std::env::var("TILELANG_BACKEND").unwrap_or_else(|_| "cpu".to_string()))
    }

    /// Choose a backend without mutating process-wide environment variables.
    pub fn with_backend(backend: &str) -> Result<Self> {
        ensure!(
            matches!(backend, "cpu" | "cuda" | "mlx"),
            "unsupported TileLang backend: {backend}"
        );
        if PROCESS_CACHE.get().is_none() {
            let mut builder = tempfile::Builder::new();
            builder.prefix("bloom-tilelang-");
            // TempDir inherits the process umask unless permissions are
            // explicit. Create executable-cache directories privately from
            // the outset, without a create-then-chmod exposure window.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                builder.permissions(fs::Permissions::from_mode(0o700));
            }
            let directory = builder.tempdir()?;
            let _ = PROCESS_CACHE.set(directory);
        }
        let cache_dir = PROCESS_CACHE
            .get()
            .ok_or_else(|| anyhow!("TileLang cache initialization failed"))?
            .path()
            .to_path_buf();
        let python = std::env::var_os("BLOOM_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(if cfg!(target_os = "windows") {
                    "python"
                } else {
                    "python3"
                })
            });
        Ok(Self {
            cache_dir,
            python,
            backend: backend.to_string(),
        })
    }

    fn ext() -> &'static str {
        if cfg!(target_os = "windows") {
            "dll"
        } else {
            "so"
        }
    }

    fn compile(&self, operation: &str, dimensions: &[usize], name: String) -> Result<PathBuf> {
        let _guard = COMPILE_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let filename = format!("{name}.{}", Self::ext());
        let destination = self.cache_dir.join(&filename);
        match fs::symlink_metadata(&destination) {
            Ok(metadata) => {
                ensure!(
                    metadata.is_file() && !metadata.file_type().is_symlink(),
                    "cached kernel is not a regular file"
                );
                return Ok(destination);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        // A failed compiler can neither publish a partial library nor poison a
        // later retry. TempDir also removes scripts, object files and failures.
        let staging = tempfile::Builder::new()
            .prefix("compile-")
            .tempdir_in(&self.cache_dir)?;
        let script = staging.path().join("generate_kernel.py");
        fs::write(&script, GENERATE_KERNEL_SCRIPT)?;
        let output = Command::new(&self.python)
            .arg(&script)
            .arg(operation)
            .args(dimensions.iter().map(usize::to_string))
            .env("TILELANG_CACHE_DIR", staging.path())
            .env("TILELANG_BACKEND", &self.backend)
            .output()
            .context("failed to launch TileLang compiler")?;
        ensure!(
            output.status.success(),
            "TileLang {operation} compilation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = staging.path().join(&filename);
        let metadata = fs::symlink_metadata(&expected).context("compiled library is missing")?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "compiled library is not a regular file"
        );
        let stdout = String::from_utf8(output.stdout).context("compiler output is not UTF-8")?;
        let reported = PathBuf::from(stdout.lines().last().unwrap_or("").trim());
        ensure!(
            reported.canonicalize()? == expected.canonicalize()?,
            "compiler returned an unexpected library path"
        );
        fs::rename(&expected, &destination).context("failed to publish compiled kernel")?;
        Ok(destination)
    }

    pub fn compile_vector_add(&self, n: usize) -> Result<PathBuf> {
        crate::kernel::checked_elements(&[n])?;
        self.compile(
            "vector_add",
            &[n],
            format!("vector_add_{}_{n}", self.backend),
        )
    }

    pub fn compile_matmul(&self, m: usize, n: usize, k: usize) -> Result<PathBuf> {
        for shape in [[m, k], [k, n], [m, n]] {
            crate::kernel::checked_elements(&shape)?;
        }
        self.compile(
            "matmul",
            &[m, n, k],
            format!("matmul_{}_{m}x{n}x{k}", self.backend),
        )
    }

    pub fn compile_softmax(&self, n: usize) -> Result<PathBuf> {
        crate::kernel::checked_elements(&[n])?;
        self.compile("softmax", &[n], format!("softmax_{}_{n}", self.backend))
    }

    pub fn compile_attention(&self, seq_len: usize, head_dim: usize) -> Result<PathBuf> {
        crate::kernel::checked_elements(&[seq_len, head_dim])?;
        self.compile(
            "attention",
            &[seq_len, head_dim],
            format!("attention_{}_{seq_len}x{head_dim}", self.backend),
        )
    }

    pub fn compile_mrope(&self) -> Result<PathBuf> {
        self.compile("mrope", &[], format!("mrope_{}", self.backend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_uses_a_private_process_cache() {
        let compiler = TileLangCompiler::with_backend("cpu").unwrap();
        assert!(compiler.cache_dir.is_dir());
        assert_ne!(compiler.cache_dir, std::env::temp_dir().join("tilelang"));
        assert_eq!(
            compiler.cache_dir,
            TileLangCompiler::with_backend("cpu").unwrap().cache_dir
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&compiler.cache_dir)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o077,
                0
            );
        }
    }

    #[test]
    fn rejects_invalid_backend_and_shapes_without_spawning() {
        for backend in ["../evil", "", "cpu/../../evil", "metal"] {
            assert!(TileLangCompiler::with_backend(backend).is_err());
        }
        let compiler = TileLangCompiler::with_backend("cpu").unwrap();
        assert!(compiler.compile_softmax(0).is_err());
        assert!(compiler.compile_matmul(usize::MAX, 2, 2).is_err());
        assert!(compiler.compile_attention(65536, 65536).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_cached_symlinks() {
        let compiler = TileLangCompiler::with_backend("cpu").unwrap();
        let path = compiler
            .cache_dir
            .join(format!("vector_add_cpu_17.{}", TileLangCompiler::ext()));
        std::os::unix::fs::symlink("/dev/null", &path).unwrap();
        assert!(compiler.compile_vector_add(17).is_err());
        fs::remove_file(path).unwrap();
    }
}
