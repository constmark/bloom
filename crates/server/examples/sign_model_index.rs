//! Offline helper for signing a Bloom model index payload.

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use clap::Parser;
use ed25519_dalek::{Signer as _, SigningKey};
use reqwest::Url;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const MAX_PAYLOAD_BYTES: usize = 384 * 1024;
const SIGNATURE_DOMAIN_V1: &[u8] = b"bloom.model_index.v1\0";
const SIGNATURE_DOMAIN_V2: &[u8] = b"bloom.model_index.v2\0";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelIndexPayload {
    schema_version: u8,
    object: String,
    name: String,
    generated_at: u64,
    expires_at: u64,
    models: Vec<ModelIndexEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelIndexEntry {
    id: String,
    name: String,
    description: String,
    #[serde(default)]
    download_url: Option<String>,
    filename: String,
    size_bytes: u64,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    files: Vec<ModelIndexFile>,
    license: String,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    parameter_count: Option<u64>,
    #[serde(default)]
    quantization: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelIndexFile {
    download_url: String,
    filename: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Parser)]
#[command(
    name = "sign_model_index",
    about = "Sign one Bloom model index payload with an offline Ed25519 seed"
)]
struct Args {
    /// JSON payload containing a bloom.model_index version 1 or 2 object.
    #[arg(long)]
    payload: PathBuf,

    /// Regular, non-symlink file containing a 32-byte seed as hex or unpadded base64url.
    #[arg(long)]
    private_key: PathBuf,

    /// New file to create for the signed envelope; existing files are never overwritten.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let payload = read_bounded_regular_file(&args.payload, MAX_PAYLOAD_BYTES, "payload")?;
    let schema_version = validate_payload_identity(&payload)?;
    validate_private_key_permissions(&args.private_key)?;
    let key_text = read_bounded_regular_file(&args.private_key, 256, "private key")?;
    let signing_key = parse_signing_key(&key_text)?;
    let verifying_key = signing_key.verifying_key();
    if verifying_key.is_weak() {
        return Err(anyhow!("the derived Ed25519 public key is weak"));
    }
    let key_id = format!("{:x}", Sha256::digest(verifying_key.as_bytes()));
    let signature_domain = if schema_version == 1 {
        SIGNATURE_DOMAIN_V1
    } else {
        SIGNATURE_DOMAIN_V2
    };
    let mut message = Vec::with_capacity(signature_domain.len() + payload.len());
    message.extend_from_slice(signature_domain);
    message.extend_from_slice(&payload);
    let signature = signing_key.sign(&message);
    let envelope = serde_json::to_vec_pretty(&json!({
        "schema_version": schema_version,
        "object": "bloom.signed_model_index",
        "algorithm": "ed25519",
        "key_id": key_id,
        "payload": URL_SAFE_NO_PAD.encode(payload),
        "signature": URL_SAFE_NO_PAD.encode(signature.to_bytes()),
    }))?;

    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        && !parent.is_dir()
    {
        return Err(anyhow!("output parent directory does not exist"));
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .with_context(|| {
            format!(
                "failed to create output '{}'; choose a path that does not exist",
                args.output.display()
            )
        })?;
    output
        .write_all(&envelope)
        .and_then(|()| output.write_all(b"\n"))
        .and_then(|()| output.sync_all())
        .with_context(|| format!("failed to write signed index '{}'", args.output.display()))?;

    println!("Signed model index: {}", args.output.display());
    println!("Public key (hex): {}", encode_hex(verifying_key.as_bytes()));
    println!("Key ID (SHA-256): {key_id}");
    Ok(())
}

fn read_bounded_regular_file(path: &Path, maximum: usize, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} file '{}'", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(anyhow!("{label} must be a regular, non-symlink file"));
    }
    if metadata.len() == 0 || metadata.len() > maximum as u64 {
        return Err(anyhow!(
            "{label} file must be between 1 and {maximum} bytes"
        ));
    }
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read {label} file '{}'", path.display()))?;
    if bytes.is_empty() || bytes.len() > maximum {
        return Err(anyhow!("{label} file changed size while it was read"));
    }
    Ok(bytes)
}

#[cfg(unix)]
fn validate_private_key_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect private key file '{}'", path.display()))?;
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(anyhow!(
            "private key file must not grant group or other permissions; run chmod 600"
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_private_key_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

fn validate_payload_identity(payload: &[u8]) -> Result<u8> {
    let payload = serde_json::from_slice::<ModelIndexPayload>(payload)
        .context("payload is not a valid model index JSON object")?;
    if !matches!(payload.schema_version, 1 | 2) || payload.object != "bloom.model_index" {
        return Err(anyhow!(
            "payload must identify bloom.model_index schema version 1 or 2"
        ));
    }
    validate_text(&payload.name, "index name", 1, 80)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs();
    if payload.generated_at == 0
        || payload.generated_at > now.saturating_add(3_600)
        || payload.expires_at <= payload.generated_at
        || payload.expires_at <= now
        || payload.expires_at - payload.generated_at > 366 * 24 * 60 * 60
    {
        return Err(anyhow!("payload validity times are invalid"));
    }
    if payload.models.len() > 200 {
        return Err(anyhow!("payload contains more than 200 model entries"));
    }
    let mut ids = std::collections::HashSet::with_capacity(payload.models.len());
    let mut filenames = std::collections::HashSet::with_capacity(payload.models.len());
    for model in payload.models {
        validate_entry(&model, payload.schema_version)?;
        if !ids.insert(model.id.to_ascii_lowercase())
            || !filenames.insert(model.filename.to_ascii_lowercase())
        {
            return Err(anyhow!("payload contains duplicate IDs or filenames"));
        }
    }
    Ok(payload.schema_version)
}

fn validate_entry(model: &ModelIndexEntry, schema_version: u8) -> Result<()> {
    validate_text(&model.id, "entry ID", 1, 64)?;
    if !model.id.bytes().enumerate().all(|(index, byte)| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || (index > 0 && matches!(byte, b'-' | b'_' | b'.'))
    }) {
        return Err(anyhow!("entry ID contains unsupported characters"));
    }
    validate_text(&model.name, "entry name", 1, 80)?;
    validate_text(&model.description, "entry description", 1, 400)?;
    if model.size_bytes == 0 {
        return Err(anyhow!("entry size is invalid"));
    }
    if model.files.is_empty() {
        validate_filename(&model.filename)?;
        let download_url = model
            .download_url
            .as_deref()
            .ok_or_else(|| anyhow!("single-file entry download URL is missing"))?;
        validate_download_url(download_url, &model.filename)?;
        let sha256 = model
            .sha256
            .as_deref()
            .ok_or_else(|| anyhow!("single-file entry SHA-256 is missing"))?;
        validate_sha256(sha256)?;
    } else {
        if schema_version != 2 || model.download_url.is_some() || model.sha256.is_some() {
            return Err(anyhow!(
                "multi-file entries require schema version 2 and per-file verification metadata"
            ));
        }
        validate_package_directory(&model.filename)?;
        validate_package_files(&model.files, model.size_bytes)?;
    }
    validate_text(&model.license, "entry license", 1, 128)?;
    if let Some(family) = model.family.as_ref() {
        validate_text(family, "entry family", 1, 64)?;
    }
    if model.parameter_count == Some(0) {
        return Err(anyhow!("entry parameter count must be greater than zero"));
    }
    if let Some(quantization) = model.quantization.as_ref() {
        validate_text(quantization, "entry quantization", 1, 32)?;
    }
    if model.tags.len() > 12 {
        return Err(anyhow!("entry contains more than 12 tags"));
    }
    let mut tags = std::collections::HashSet::with_capacity(model.tags.len());
    for tag in &model.tags {
        validate_text(tag, "entry tag", 1, 32)?;
        if !tag
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            || !tags.insert(tag.to_ascii_lowercase())
        {
            return Err(anyhow!("entry tags are invalid or duplicated"));
        }
    }
    Ok(())
}

fn validate_package_directory(value: &str) -> Result<()> {
    let lower = value.to_ascii_lowercase();
    if value.is_empty()
        || value.len() > 255
        || value.trim() != value
        || value.starts_with('.')
        || value.contains(['/', '\\', '%'])
        || value.chars().any(char::is_control)
        || [".gguf", ".onnx", ".mlmodel", ".mlpackage", ".mlmodelc"]
            .iter()
            .any(|suffix| lower.ends_with(suffix))
    {
        return Err(anyhow!("model package directory is invalid"));
    }
    Ok(())
}

fn validate_package_files(files: &[ModelIndexFile], expected_size: u64) -> Result<()> {
    if !(2..=256).contains(&files.len()) {
        return Err(anyhow!(
            "model package must contain between 2 and 256 files"
        ));
    }
    let mut filenames = std::collections::HashSet::with_capacity(files.len());
    let mut total_size = 0_u64;
    let mut source_identity = None;
    let mut has_config = false;
    for file in files {
        validate_package_path(&file.filename)?;
        if !filenames.insert(file.filename.to_ascii_lowercase()) || file.size_bytes == 0 {
            return Err(anyhow!("model package file identity is invalid"));
        }
        validate_sha256(&file.sha256)?;
        total_size = total_size
            .checked_add(file.size_bytes)
            .ok_or_else(|| anyhow!("model package size overflowed"))?;
        has_config |= file.filename == "config.json";
        let identity = validate_download_url(&file.download_url, &file.filename)?;
        if source_identity
            .as_ref()
            .is_some_and(|expected| expected != &identity)
        {
            return Err(anyhow!(
                "all model package files must use the same repository commit"
            ));
        }
        source_identity.get_or_insert(identity);
    }
    if total_size != expected_size || !has_config {
        return Err(anyhow!("model package manifest is incomplete"));
    }
    validate_package_safetensors_layout(files)?;
    Ok(())
}

fn validate_package_safetensors_layout(files: &[ModelIndexFile]) -> Result<()> {
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
        .map(|filename| parse_package_shard_name(filename).map(|position| (filename, position)))
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
            "model package requires model.safetensors or a complete indexed Safetensors shard set"
        ));
    }
    shards.sort_by_key(|(_, (index, _))| *index);
    let expected_total = shards[0].1.1;
    if shards.len() != expected_total
        || shards
            .iter()
            .enumerate()
            .any(|(position, (_, (index, total)))| {
                *index != position + 1 || *total != expected_total
            })
    {
        return Err(anyhow!(
            "model package Safetensors shard sequence is incomplete or inconsistent"
        ));
    }
    Ok(())
}

fn parse_package_shard_name(filename: &str) -> Result<(usize, usize)> {
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
    if index == 0 || total == 0 || index > total || total > 256 {
        return Err(anyhow!("invalid Safetensors shard position in {filename}"));
    }
    Ok((index, total))
}

fn validate_package_path(value: &str) -> Result<()> {
    let components = value.split('/').collect::<Vec<_>>();
    let supported = [".json", ".safetensors", ".txt", ".model", ".tiktoken"];
    if value.is_empty()
        || value.len() > 512
        || value.trim() != value
        || value.contains(['\\', '%'])
        || value.chars().any(char::is_control)
        || components.is_empty()
        || components.len() > 8
        || components
            .iter()
            .any(|component| component.is_empty() || component.starts_with('.'))
        || !supported.iter().any(|suffix| value.ends_with(suffix))
    {
        return Err(anyhow!("model package filename is invalid"));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(anyhow!("SHA-256 must contain 64 hexadecimal characters"))
    }
}

fn validate_filename(filename: &str) -> Result<()> {
    if filename.is_empty()
        || filename.len() > 255
        || filename.starts_with('.')
        || filename.contains(['/', '\\', '%'])
        || !matches!(
            filename
                .rsplit_once('.')
                .map(|(_, extension)| extension.to_ascii_lowercase())
                .as_deref(),
            Some("gguf" | "onnx" | "mlmodel")
        )
    {
        return Err(anyhow!("entry filename is not a supported direct child"));
    }
    Ok(())
}

fn validate_download_url(value: &str, filename: &str) -> Result<String> {
    if value.is_empty() || value.len() > 2_048 {
        return Err(anyhow!("entry download URL is invalid"));
    }
    let url = Url::parse(value).context("entry download URL is invalid")?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some_and(|port| port != 443)
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(
            url.host_str(),
            Some("huggingface.co" | "www.huggingface.co")
        )
    {
        return Err(anyhow!(
            "entry download URL must use public Hugging Face HTTPS"
        ));
    }
    let segments = url
        .path_segments()
        .ok_or_else(|| anyhow!("entry download URL has no path"))?
        .collect::<Vec<_>>();
    if segments.len() < 5
        || segments[0].is_empty()
        || segments[1].is_empty()
        || segments[2] != "resolve"
        || !matches!(segments[3].len(), 40 | 64)
        || !segments[3].bytes().all(|byte| byte.is_ascii_hexdigit())
        || segments[4..]
            .iter()
            .any(|segment| segment.is_empty() || segment.contains('%'))
        || segments[4..].join("/") != filename
    {
        return Err(anyhow!(
            "entry download URL must use an immutable commit and matching filename"
        ));
    }
    Ok(format!("{}/{}/{}", segments[0], segments[1], segments[3]))
}

fn validate_text(value: &str, field: &str, min: usize, max: usize) -> Result<()> {
    let length = value.chars().count();
    if length < min || length > max || value.trim() != value || value.chars().any(char::is_control)
    {
        return Err(anyhow!("{field} is invalid"));
    }
    Ok(())
}

fn parse_signing_key(value: &[u8]) -> Result<SigningKey> {
    let value = std::str::from_utf8(value)
        .context("private key file must contain UTF-8 text")?
        .trim();
    let bytes = if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        decode_hex(value)?
    } else {
        URL_SAFE_NO_PAD
            .decode(value)
            .context("private key must be 64 hex characters or unpadded base64url")?
            .try_into()
            .map_err(|_| anyhow!("private key must decode to exactly 32 bytes"))?
    };
    Ok(SigningKey::from_bytes(&bytes))
}

fn decode_hex(value: &str) -> Result<[u8; 32]> {
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    Ok(output)
}

fn encode_hex(value: &[u8]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_payload() -> Vec<u8> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut payload = serde_json::from_slice::<serde_json::Value>(include_bytes!(
            "fixtures/model-index-payload.json"
        ))
        .unwrap();
        payload["generated_at"] = serde_json::Value::from(now.saturating_sub(1));
        payload["expires_at"] = serde_json::Value::from(now.saturating_add(3600));
        serde_json::to_vec(&payload).unwrap()
    }

    #[test]
    fn bundled_payload_passes_the_offline_signer_admission_checks() {
        assert_eq!(validate_payload_identity(&valid_payload()).unwrap(), 1);
        let seed = format!("{}\n", "07".repeat(32));
        let key = parse_signing_key(seed.as_bytes()).unwrap();
        assert_eq!(
            encode_hex(key.verifying_key().as_bytes()),
            "ea4a6c63e29c520abef5507b132ec5f9954776aebebe7b92421eea691446d22c"
        );
    }

    #[test]
    fn version_two_package_payload_uses_the_strict_package_admission_path() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut payload = serde_json::from_slice::<serde_json::Value>(include_bytes!(
            "../../../examples/model-index-payload-v2.json"
        ))
        .unwrap();
        payload["generated_at"] = serde_json::Value::from(now.saturating_sub(1));
        payload["expires_at"] = serde_json::Value::from(now.saturating_add(3600));
        assert_eq!(
            validate_payload_identity(&serde_json::to_vec(&payload).unwrap()).unwrap(),
            2
        );

        payload["models"][0]["files"][1]["filename"] =
            serde_json::Value::from("../model.safetensors");
        assert!(validate_payload_identity(&serde_json::to_vec(&payload).unwrap()).is_err());
    }

    #[test]
    fn offline_signer_rejects_mutable_model_sources() {
        let payload = String::from_utf8(valid_payload())
            .unwrap()
            .replace("0123456789abcdef0123456789abcdef01234567", "main");
        assert!(validate_payload_identity(payload.as_bytes()).is_err());
    }
}
