use anyhow::{Result, anyhow};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const GENERATE_KERNEL_SCRIPT: &str = include_str!("../scripts/generate_kernel.py");

pub struct TileLangCompiler {
    cache_dir: PathBuf,
    python: PathBuf,
    backend: String,
}

impl TileLangCompiler {
    pub fn new() -> Result<Self> {
        let cache_dir = std::env::temp_dir().join("tilelang");
        fs::create_dir_all(&cache_dir)?;
        let python = if std::path::Path::new("/root/miniconda/envs/vllm/bin/python3").exists() {
            PathBuf::from("/root/miniconda/envs/vllm/bin/python3")
        } else if cfg!(target_os = "windows") {
            PathBuf::from("python")
        } else {
            PathBuf::from("python3")
        };
        let backend = std::env::var("TILELANG_BACKEND").unwrap_or_else(|_| "cpu".to_string());
        Ok(Self {
            cache_dir,
            python,
            backend,
        })
    }

    fn ext() -> &'static str {
        if cfg!(target_os = "windows") {
            "dll"
        } else {
            "so"
        }
    }

    pub fn compile_vector_add(&self, n: usize) -> Result<PathBuf> {
        let name = format!("vector_add_{}_{}", self.backend, n);
        let so_path = self.cache_dir.join(&name).with_extension(Self::ext());

        if so_path.exists() {
            return Ok(so_path);
        }

        let script_path = self.cache_dir.join("generate_kernel.py");
        fs::write(&script_path, GENERATE_KERNEL_SCRIPT)?;

        let output = Command::new(&self.python)
            .arg(&script_path)
            .arg("vector_add")
            .arg(n.to_string())
            .env("TILELANG_CACHE_DIR", &self.cache_dir)
            .env("TILELANG_BACKEND", &self.backend)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("TileLang compilation failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let so_path_str = stdout.lines().last().unwrap_or("").trim();
        let compiled_path = PathBuf::from(so_path_str);
        if !compiled_path.exists() {
            return Err(anyhow!(
                "Compiled library not found at {}",
                compiled_path.display()
            ));
        }

        Ok(compiled_path)
    }

    pub fn compile_matmul(&self, m: usize, n: usize, k: usize) -> Result<PathBuf> {
        let name = format!("matmul_{}_{}x{}x{}", self.backend, m, n, k);
        let so_path = self.cache_dir.join(&name).with_extension(Self::ext());

        if so_path.exists() {
            return Ok(so_path);
        }

        let script_path = self.cache_dir.join("generate_kernel.py");
        fs::write(&script_path, GENERATE_KERNEL_SCRIPT)?;

        let output = Command::new(&self.python)
            .arg(&script_path)
            .arg("matmul")
            .arg(m.to_string())
            .arg(n.to_string())
            .arg(k.to_string())
            .env("TILELANG_CACHE_DIR", &self.cache_dir)
            .env("TILELANG_BACKEND", &self.backend)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("TileLang matmul compilation failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let so_path_str = stdout.lines().last().unwrap_or("").trim();
        let compiled_path = PathBuf::from(so_path_str);
        if !compiled_path.exists() {
            return Err(anyhow!(
                "Compiled library not found at {}",
                compiled_path.display()
            ));
        }

        Ok(compiled_path)
    }

    pub fn compile_softmax(&self, n: usize) -> Result<PathBuf> {
        let name = format!("softmax_{}_{}", self.backend, n);
        let so_path = self.cache_dir.join(&name).with_extension(Self::ext());

        if so_path.exists() {
            return Ok(so_path);
        }

        let script_path = self.cache_dir.join("generate_kernel.py");
        fs::write(&script_path, GENERATE_KERNEL_SCRIPT)?;

        let output = Command::new(&self.python)
            .arg(&script_path)
            .arg("softmax")
            .arg(n.to_string())
            .env("TILELANG_CACHE_DIR", &self.cache_dir)
            .env("TILELANG_BACKEND", &self.backend)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("TileLang softmax compilation failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let so_path_str = stdout.lines().last().unwrap_or("").trim();
        let compiled_path = PathBuf::from(so_path_str);
        if !compiled_path.exists() {
            return Err(anyhow!(
                "Compiled library not found at {}",
                compiled_path.display()
            ));
        }

        Ok(compiled_path)
    }

    pub fn compile_attention(&self, seq_len: usize, head_dim: usize) -> Result<PathBuf> {
        let name = format!("attention_{}_{}x{}", self.backend, seq_len, head_dim);
        let so_path = self.cache_dir.join(&name).with_extension(Self::ext());

        if so_path.exists() {
            return Ok(so_path);
        }

        let script_path = self.cache_dir.join("generate_kernel.py");
        fs::write(&script_path, GENERATE_KERNEL_SCRIPT)?;

        let output = Command::new(&self.python)
            .arg(&script_path)
            .arg("attention")
            .arg(seq_len.to_string())
            .arg(head_dim.to_string())
            .env("TILELANG_CACHE_DIR", &self.cache_dir)
            .env("TILELANG_BACKEND", &self.backend)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("TileLang attention compilation failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let so_path_str = stdout.lines().last().unwrap_or("").trim();
        let compiled_path = PathBuf::from(so_path_str);
        if !compiled_path.exists() {
            return Err(anyhow!(
                "Compiled library not found at {}",
                compiled_path.display()
            ));
        }

        Ok(compiled_path)
    }

    pub fn compile_mrope(&self) -> Result<PathBuf> {
        let name = format!("mrope_{}", self.backend);
        let so_path = self.cache_dir.join(&name).with_extension(Self::ext());

        if so_path.exists() {
            return Ok(so_path);
        }

        let script_path = self.cache_dir.join("generate_kernel.py");
        fs::write(&script_path, GENERATE_KERNEL_SCRIPT)?;

        let output = Command::new(&self.python)
            .arg(&script_path)
            .arg("mrope")
            .env("TILELANG_CACHE_DIR", &self.cache_dir)
            .env("TILELANG_BACKEND", &self.backend)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("TileLang mrope compilation failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let so_path_str = stdout.lines().last().unwrap_or("").trim();
        let compiled_path = PathBuf::from(so_path_str);
        if !compiled_path.exists() {
            return Err(anyhow!(
                "Compiled library not found at {}",
                compiled_path.display()
            ));
        }

        Ok(compiled_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_new_and_cache() {
        let compiler = TileLangCompiler::new().unwrap();
        assert!(compiler.cache_dir.exists());
        assert!(compiler.cache_dir.is_dir());
    }

    #[test]
    fn test_compiler_extension() {
        let ext = TileLangCompiler::ext();
        #[cfg(target_os = "windows")]
        assert_eq!(ext, "dll");
        #[cfg(not(target_os = "windows"))]
        assert_eq!(ext, "so");
    }
}
