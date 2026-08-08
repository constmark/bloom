//! Canonical identities and path rules for verified multi-file model packages.

use std::collections::HashSet;
use std::path::{Component, Path};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::model_manager::validate_catalog_id;

pub(crate) const MAX_MODEL_PACKAGE_FILES: usize = 256;
const MAX_MODEL_PACKAGE_PATH_BYTES: usize = 512;
const MAX_MODEL_PACKAGE_PATH_DEPTH: usize = 8;
const PACKAGE_DIGEST_DOMAIN: &[u8] = b"bloom.model_package.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelPackageFile {
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

pub(crate) fn validate_package_id(id: &str) -> Result<()> {
    validate_catalog_id(id)?;
    let lower = id.to_ascii_lowercase();
    if id.starts_with('.')
        || [".gguf", ".onnx", ".mlmodel", ".mlpackage", ".mlmodelc"]
            .iter()
            .any(|suffix| lower.ends_with(suffix))
    {
        return Err(anyhow!(
            "model package directory must be non-hidden and distinct from single-file model names"
        ));
    }
    Ok(())
}

pub(crate) fn normalize_package_files(
    mut files: Vec<ModelPackageFile>,
    expected_total_bytes: u64,
) -> Result<Vec<ModelPackageFile>> {
    if !(2..=MAX_MODEL_PACKAGE_FILES).contains(&files.len()) {
        return Err(anyhow!(
            "model packages must contain between 2 and {MAX_MODEL_PACKAGE_FILES} files"
        ));
    }

    let mut identities = HashSet::with_capacity(files.len());
    let mut total_bytes = 0_u64;
    let mut has_config = false;
    for file in &mut files {
        validate_package_path(&file.filename)?;
        if !identities.insert(file.filename.to_ascii_lowercase()) {
            return Err(anyhow!("model package filenames must be unique"));
        }
        if file.size_bytes == 0 {
            return Err(anyhow!(
                "model package file sizes must be greater than zero"
            ));
        }
        file.sha256 = normalize_sha256(&file.sha256)?;
        total_bytes = total_bytes
            .checked_add(file.size_bytes)
            .ok_or_else(|| anyhow!("model package size overflowed"))?;
        has_config |= file.filename == "config.json";
    }
    if total_bytes != expected_total_bytes || total_bytes == 0 {
        return Err(anyhow!(
            "model package file sizes do not match the declared total size"
        ));
    }
    if !has_config {
        return Err(anyhow!("model packages must contain config.json"));
    }
    validate_safetensors_layout_names(&files)?;

    files.sort_by(|left, right| left.filename.cmp(&right.filename));
    Ok(files)
}

pub(crate) fn validate_safetensors_layout_names(files: &[ModelPackageFile]) -> Result<()> {
    let has_single = files
        .iter()
        .any(|file| file.filename == "model.safetensors");
    let has_index = files
        .iter()
        .any(|file| file.filename == "model.safetensors.index.json");
    let mut shards = files
        .iter()
        .filter_map(|file| {
            (file.filename.starts_with("model-") && file.filename.ends_with(".safetensors"))
                .then_some(file.filename.as_str())
        })
        .map(|filename| parse_safetensors_shard_name(filename).map(|position| (filename, position)))
        .collect::<Result<Vec<_>>>()?;
    if has_single {
        if has_index || !shards.is_empty() {
            return Err(anyhow!(
                "model.safetensors cannot be combined with a sharded Safetensors layout"
            ));
        }
        return Ok(());
    }
    if !has_index || shards.is_empty() {
        return Err(anyhow!(
            "model packages require model.safetensors or a complete indexed Safetensors shard set"
        ));
    }
    shards.sort_by_key(|(_, (index, _))| *index);
    let expected_total = shards[0].1 .1;
    if shards.len() != expected_total {
        return Err(anyhow!(
            "Safetensors package declares {expected_total} shards but contains {}",
            shards.len()
        ));
    }
    for (position, (filename, (index, total))) in shards.into_iter().enumerate() {
        if index != position + 1 || total != expected_total {
            return Err(anyhow!(
                "Safetensors package shard sequence is incomplete or inconsistent at {filename}"
            ));
        }
    }
    Ok(())
}

fn parse_safetensors_shard_name(filename: &str) -> Result<(usize, usize)> {
    let body = filename
        .strip_prefix("model-")
        .and_then(|value| value.strip_suffix(".safetensors"))
        .ok_or_else(|| anyhow!("invalid Safetensors shard filename"))?;
    let (index, total) = body
        .split_once("-of-")
        .ok_or_else(|| anyhow!("invalid Safetensors shard filename: {filename}"))?;
    if index.len() != 5
        || total.len() != 5
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(anyhow!(
            "Safetensors shards must use model-00001-of-00002.safetensors naming: {filename}"
        ));
    }
    let index = index.parse::<usize>()?;
    let total = total.parse::<usize>()?;
    if index == 0 || total == 0 || index > total || total > MAX_MODEL_PACKAGE_FILES {
        return Err(anyhow!("invalid Safetensors shard position in {filename}"));
    }
    Ok((index, total))
}

pub(crate) fn package_digest(files: &[ModelPackageFile]) -> Result<String> {
    if files.is_empty() || files.len() > MAX_MODEL_PACKAGE_FILES {
        return Err(anyhow!("model package file count is invalid"));
    }
    let mut ordered = files.to_vec();
    ordered.sort_by(|left, right| left.filename.cmp(&right.filename));
    let mut digest = Sha256::new();
    digest.update(PACKAGE_DIGEST_DOMAIN);
    digest.update((ordered.len() as u32).to_be_bytes());
    for file in ordered {
        validate_package_path(&file.filename)?;
        let sha256 = decode_sha256(&file.sha256)?;
        let name = file.filename.as_bytes();
        let name_length =
            u32::try_from(name.len()).map_err(|_| anyhow!("model package filename is too long"))?;
        digest.update(name_length.to_be_bytes());
        digest.update(name);
        digest.update(file.size_bytes.to_be_bytes());
        digest.update(sha256);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn validate_package_path(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > MAX_MODEL_PACKAGE_PATH_BYTES
        || value.trim() != value
        || value.contains(['\\', '%'])
        || value.chars().any(char::is_control)
    {
        return Err(anyhow!("model package filename is invalid"));
    }
    let components = Path::new(value).components().collect::<Vec<_>>();
    if components.is_empty()
        || components.len() > MAX_MODEL_PACKAGE_PATH_DEPTH
        || components.iter().any(|component| {
            !matches!(component, Component::Normal(name) if !name.is_empty() && !name.to_string_lossy().starts_with('.'))
        })
    {
        return Err(anyhow!("model package filename must be a safe relative path"));
    }
    let allowed = [".json", ".safetensors", ".txt", ".model", ".tiktoken"];
    if !allowed.iter().any(|suffix| value.ends_with(suffix)) {
        return Err(anyhow!(
            "model package files must use a supported data or metadata extension"
        ));
    }
    Ok(())
}

pub(crate) fn normalize_sha256(value: &str) -> Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    decode_sha256(&normalized)?;
    Ok(normalized)
}

pub(crate) fn decode_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(anyhow!(
            "model package SHA-256 must contain exactly 64 hexadecimal characters"
        ));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package_files() -> Vec<ModelPackageFile> {
        vec![
            ModelPackageFile {
                filename: "model.safetensors".to_string(),
                size_bytes: 7,
                sha256: "ab".repeat(32),
            },
            ModelPackageFile {
                filename: "config.json".to_string(),
                size_bytes: 3,
                sha256: "cd".repeat(32),
            },
        ]
    }

    #[test]
    fn package_identity_is_order_independent() {
        let files = normalize_package_files(package_files(), 10).unwrap();
        let mut reversed = files.clone();
        reversed.reverse();
        assert_eq!(
            package_digest(&files).unwrap(),
            package_digest(&reversed).unwrap()
        );
    }

    #[test]
    fn package_paths_are_relative_bounded_and_non_executable() {
        for valid in [
            "config.json",
            "1_Pooling/config.json",
            "model-00001-of-00002.safetensors",
            "tokenizer.model",
        ] {
            validate_package_path(valid).unwrap();
        }
        for invalid in [
            "../config.json",
            "/config.json",
            ".hidden.json",
            "nested/.hidden.json",
            "model.py",
            "weights.bin",
            "model%2Fconfig.json",
        ] {
            assert!(validate_package_path(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn package_manifest_requires_exact_total_and_native_inputs() {
        assert!(normalize_package_files(package_files(), 9).is_err());
        let mut missing_config = package_files();
        missing_config[1].filename = "tokenizer.json".to_string();
        assert!(normalize_package_files(missing_config, 10).is_err());
        let mut duplicate = package_files();
        duplicate[1].filename = duplicate[0].filename.to_ascii_uppercase();
        assert!(normalize_package_files(duplicate, 10).is_err());
    }

    #[test]
    fn package_manifest_requires_a_complete_supported_weight_layout() {
        let incomplete = vec![
            ModelPackageFile {
                filename: "config.json".to_string(),
                size_bytes: 3,
                sha256: "cd".repeat(32),
            },
            ModelPackageFile {
                filename: "model-00001-of-00001.safetensors".to_string(),
                size_bytes: 7,
                sha256: "ab".repeat(32),
            },
        ];
        assert!(normalize_package_files(incomplete, 10)
            .unwrap_err()
            .to_string()
            .contains("complete indexed"));

        let mut conflicting = package_files();
        conflicting.push(ModelPackageFile {
            filename: "model.safetensors.index.json".to_string(),
            size_bytes: 1,
            sha256: "ef".repeat(32),
        });
        assert!(normalize_package_files(conflicting, 11)
            .unwrap_err()
            .to_string()
            .contains("cannot be combined"));
    }
}
