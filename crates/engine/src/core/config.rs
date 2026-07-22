use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use bloomai_core::constants::GIB;
use serde::{Deserialize, Serialize};

pub const DEFAULT_CONFIG_FILE: &str = "config.json";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BloomConfig {
    pub infer: InferConfig,
    pub server: ServerConfig,
    pub bench: BenchConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InferConfig {
    pub model: Option<PathBuf>,
    pub prompt: Option<String>,
    pub prompt_file: Option<PathBuf>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
    pub device: Option<String>,
    pub backend: Option<String>,
    pub engine: Option<String>,
    pub dtype: Option<String>,
    pub context_size: Option<usize>,
    pub memory_utilization: Option<f64>,
    pub reserve_memory_bytes: Option<usize>,
    pub disable_memory_prealloc: Option<bool>,
    pub threads: Option<usize>,
    pub batch_size: Option<usize>,
    pub gpu_layers: Option<usize>,
    pub speculative: Option<String>,
    pub draft_model: Option<PathBuf>,
    pub num_speculative_tokens: Option<usize>,
    pub speculative_ngram_order: Option<usize>,
    pub image: Option<PathBuf>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_frames: Option<u32>,
    pub fps: Option<f32>,
    pub guidance_scale: Option<f64>,
    pub num_steps: Option<u32>,
    pub negative_prompt: Option<String>,
    pub output: Option<PathBuf>,
    pub interactive: Option<bool>,
    pub list_engines: Option<bool>,
    pub bench: Option<bool>,
    pub inspect: Option<bool>,
    pub repetitions: Option<usize>,
    pub warmup: Option<usize>,
    pub bench_json: Option<bool>,
    pub quiet: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub model: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub backend: Option<String>,
    pub device: Option<String>,
    pub dtype: Option<String>,
    pub max_concurrent: Option<usize>,
    pub context_size: Option<usize>,
    pub memory_utilization: Option<f64>,
    pub reserve_memory_bytes: Option<usize>,
    pub disable_memory_prealloc: Option<bool>,
    pub max_num_tokens: Option<usize>,
    pub timeout: Option<u64>,
    pub speculative: Option<String>,
    pub draft_model: Option<PathBuf>,
    pub num_speculative_tokens: Option<usize>,
    pub speculative_ngram_order: Option<usize>,
    pub enable_ifb: Option<bool>,
    pub enable_chunked_prefill: Option<bool>,
    pub prefill_chunk_size: Option<usize>,
    pub enable_cachemesh: Option<bool>,
    pub cachemesh_l2_capacity_bytes: Option<usize>,
    pub enable_cachemesh_l3: Option<bool>,
    pub cachemesh_l3_path: Option<PathBuf>,
    pub cachemesh_write_through_l3: Option<bool>,
    pub long_context_policy: Option<String>,
    pub sliding_window_tokens: Option<usize>,
    pub context_shift_max_tokens: Option<usize>,
    pub context_shift_tokens: Option<usize>,
    pub compact_free_blocks: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchConfig {
    pub model: Option<PathBuf>,
    pub prompt: Option<String>,
    pub backend: Option<String>,
    pub device: Option<String>,
    pub dtype: Option<String>,
    pub max_tokens: Option<usize>,
    pub repetitions: Option<usize>,
    pub warmup: Option<usize>,
    pub context_size: Option<usize>,
    pub prompts_file: Option<PathBuf>,
    pub speculative: Option<String>,
    pub draft_model: Option<PathBuf>,
    pub num_speculative_tokens: Option<usize>,
    pub speculative_ngram_order: Option<usize>,
}

pub fn default_config_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("BLOOM_CONFIG_HOME") {
        return Ok(expand_tilde(PathBuf::from(path))?);
    }

    home_dir()
        .map(|home| home.join(".bloom"))
        .ok_or_else(|| anyhow!("could not determine home directory for ~/.bloom"))
}

pub fn default_config_path() -> Result<PathBuf> {
    Ok(default_config_dir()?.join(DEFAULT_CONFIG_FILE))
}

pub fn resolve_config_path(path: Option<&Path>) -> Result<PathBuf> {
    match path {
        Some(path) => expand_tilde(path.to_path_buf()),
        None => {
            if let Some(path) = std::env::var_os("BLOOM_CONFIG") {
                expand_tilde(PathBuf::from(path))
            } else {
                default_config_path()
            }
        }
    }
}

pub fn load_config(path: &Path) -> Result<BloomConfig> {
    if !path.exists() {
        return Ok(BloomConfig::default());
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Bloom config '{}'", path.display()))?;
    let mut config: BloomConfig = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse Bloom config '{}'", path.display()))?;
    config.expand_paths()?;
    Ok(config)
}

pub fn write_default_config(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory '{}'", parent.display()))?;
    }
    if path.exists() {
        return Err(anyhow!("Bloom config already exists: {}", path.display()));
    }

    let mut value = serde_json::to_value(example_config())?;
    remove_nulls(&mut value);
    let content = serde_json::to_string_pretty(&value)?;
    std::fs::write(path, format!("{content}\n"))
        .with_context(|| format!("failed to write Bloom config '{}'", path.display()))
}

fn example_config() -> BloomConfig {
    BloomConfig {
        infer: InferConfig {
            device: Some("cpu".to_string()),
            backend: Some("candle".to_string()),
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: Some(0.9),
            context_size: Some(2048),
            memory_utilization: Some(crate::core::memory::default_memory_utilization()),
            disable_memory_prealloc: Some(false),
            batch_size: Some(1),
            speculative: Some("none".to_string()),
            num_speculative_tokens: Some(5),
            speculative_ngram_order: Some(4),
            output: Some(PathBuf::from("output.gif")),
            interactive: Some(false),
            ..InferConfig::default()
        },
        server: ServerConfig {
            host: Some("127.0.0.1".to_string()),
            port: Some(3000),
            backend: Some("candle".to_string()),
            device: Some("cpu".to_string()),
            max_concurrent: Some(4),
            context_size: Some(2048),
            memory_utilization: Some(crate::core::memory::default_memory_utilization()),
            disable_memory_prealloc: Some(false),
            max_num_tokens: Some(4096),
            timeout: Some(300),
            speculative: Some("none".to_string()),
            num_speculative_tokens: Some(5),
            speculative_ngram_order: Some(4),
            enable_ifb: Some(false),
            enable_chunked_prefill: Some(false),
            prefill_chunk_size: Some(512),
            enable_cachemesh: Some(false),
            cachemesh_l2_capacity_bytes: Some(2 * GIB as usize),
            enable_cachemesh_l3: Some(false),
            cachemesh_write_through_l3: Some(false),
            long_context_policy: Some("full".to_string()),
            sliding_window_tokens: Some(4096),
            context_shift_tokens: Some(1024),
            compact_free_blocks: Some(128),
            ..ServerConfig::default()
        },
        bench: BenchConfig {
            prompt: Some("Explain quantum computing in simple terms".to_string()),
            backend: Some("candle".to_string()),
            device: Some("cpu".to_string()),
            max_tokens: Some(128),
            repetitions: Some(3),
            warmup: Some(1),
            context_size: Some(2048),
            ..BenchConfig::default()
        },
    }
}

impl BloomConfig {
    fn expand_paths(&mut self) -> Result<()> {
        expand_option_path(&mut self.infer.model)?;
        expand_option_path(&mut self.infer.prompt_file)?;
        expand_option_path(&mut self.infer.image)?;
        expand_option_path(&mut self.infer.draft_model)?;
        expand_option_path(&mut self.infer.output)?;
        expand_option_path(&mut self.server.model)?;
        expand_option_path(&mut self.server.draft_model)?;
        expand_option_path(&mut self.server.cachemesh_l3_path)?;
        expand_option_path(&mut self.bench.model)?;
        expand_option_path(&mut self.bench.prompts_file)?;
        Ok(())
    }
}

fn expand_option_path(path: &mut Option<PathBuf>) -> Result<()> {
    if let Some(value) = path.take() {
        *path = Some(expand_tilde(value)?);
    }
    Ok(())
}

fn remove_nulls(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for value in map.values_mut() {
                remove_nulls(value);
            }
            map.retain(|_, value| !value.is_null());
        }
        serde_json::Value::Array(values) => {
            for value in values {
                remove_nulls(value);
            }
        }
        _ => {}
    }
}

fn expand_tilde(path: PathBuf) -> Result<PathBuf> {
    let Some(path_str) = path.to_str() else {
        return Ok(path);
    };
    if path_str == "~" {
        return home_dir().ok_or_else(|| anyhow!("could not determine home directory"));
    }
    if let Some(rest) = path_str.strip_prefix("~/") {
        return home_dir()
            .map(|home| home.join(rest))
            .ok_or_else(|| anyhow!("could not determine home directory"));
    }
    Ok(path)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_missing_config_as_default() {
        let path = PathBuf::from("/tmp/bloom-config-that-should-not-exist.json");
        let config = load_config(&path).unwrap();
        assert!(config.infer.model.is_none());
        assert!(config.server.model.is_none());
    }

    #[test]
    fn example_config_includes_long_context_server_defaults() {
        let config = example_config();
        assert_eq!(config.server.long_context_policy.as_deref(), Some("full"));
        assert_eq!(config.server.sliding_window_tokens, Some(4096));
        assert_eq!(config.server.context_shift_tokens, Some(1024));
        assert_eq!(config.server.compact_free_blocks, Some(128));
    }
}
