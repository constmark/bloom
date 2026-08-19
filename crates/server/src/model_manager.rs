//! Safe discovery and resolution of locally installed models.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

use super::model_provenance::{ModelProvenance, read_provenance, remove_provenance};

const MAX_DIRECTORY_DEPTH: usize = 4;
const MAX_SCANNED_FILES: usize = 20_000;
const MAX_CATALOG_DIRECTORY_ENTRIES: usize = 65_536;
pub(crate) const MAX_MODEL_CATALOG_ENTRIES: usize = 4_096;
pub(crate) const MODEL_CATALOG_SCHEMA_VERSION: u32 = 1;
pub(crate) const MODEL_CATALOG_OBJECT: &str = "bloom.model_catalog";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelCatalogEntry {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub format: String,
    pub size_bytes: u64,
    pub size_complete: bool,
    pub modified_at: Option<u64>,
    pub active: bool,
    pub provenance: Option<ModelProvenance>,
    pub provenance_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ModelCatalog {
    pub root: String,
    pub root_exists: bool,
    pub models: Vec<ModelCatalogEntry>,
}

impl ModelCatalog {
    pub(crate) fn scan(root: &Path, active_path: Option<&Path>) -> Result<Self> {
        if !root.exists() {
            return Ok(Self {
                root: root.display().to_string(),
                root_exists: false,
                models: Vec::new(),
            });
        }
        if !root.is_dir() {
            return Err(anyhow!(
                "model catalog root is not a directory: {}",
                root.display()
            ));
        }

        let canonical_root = root.canonicalize().with_context(|| {
            format!("failed to resolve model catalog root '{}'", root.display())
        })?;
        let canonical_active = active_path.and_then(|path| path.canonicalize().ok());
        let mut models = Vec::new();

        let mut inspected_entries = 0_usize;
        for item in fs::read_dir(&canonical_root)
            .with_context(|| format!("failed to read model catalog root '{}'", root.display()))?
        {
            inspected_entries = inspected_entries.saturating_add(1);
            if inspected_entries > MAX_CATALOG_DIRECTORY_ENTRIES {
                return Err(anyhow!(
                    "model catalog contains more than {MAX_CATALOG_DIRECTORY_ENTRIES} direct children"
                ));
            }
            let item = item
                .with_context(|| format!("failed to read an entry under '{}'", root.display()))?;
            let file_type = item.file_type().with_context(|| {
                format!(
                    "failed to inspect model candidate '{}'",
                    item.path().display()
                )
            })?;
            if file_type.is_symlink() {
                continue;
            }

            let path = item.path();
            let Some(id) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if validate_catalog_id(id).is_err() {
                continue;
            }
            let candidate = if file_type.is_file() {
                model_file_format(&path).map(|format| ("file", format))
            } else if file_type.is_dir() {
                directory_format(&path)?.map(|format| ("directory", format))
            } else {
                None
            };
            let Some((kind, format)) = candidate else {
                continue;
            };

            let canonical_path = path.canonicalize().with_context(|| {
                format!("failed to resolve model candidate '{}'", path.display())
            })?;
            if !canonical_path.starts_with(&canonical_root) {
                continue;
            }
            let metadata = fs::metadata(&canonical_path).with_context(|| {
                format!("failed to inspect model candidate '{}'", path.display())
            })?;
            let (size_bytes, size_complete) = if metadata.is_file() {
                (metadata.len(), true)
            } else {
                directory_size(&canonical_path)?
            };
            let modified_at = metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs());
            let (provenance, provenance_error) = if size_complete {
                match read_provenance(&canonical_root, id, size_bytes) {
                    Ok(provenance) => (provenance, None),
                    Err(error) => {
                        tracing::warn!(model = id, %error, "Ignored invalid model provenance");
                        (
                            None,
                            Some("Stored provenance metadata is invalid or unsafe.".to_string()),
                        )
                    }
                }
            } else {
                (None, None)
            };

            if models.len() >= MAX_MODEL_CATALOG_ENTRIES {
                return Err(anyhow!(
                    "model catalog contains more than {MAX_MODEL_CATALOG_ENTRIES} recognized models"
                ));
            }
            models.push(ModelCatalogEntry {
                id: id.to_string(),
                name: display_name(id),
                kind: kind.to_string(),
                format: format.to_string(),
                size_bytes,
                size_complete,
                modified_at,
                active: canonical_active.as_ref() == Some(&canonical_path),
                provenance,
                provenance_error,
            });
        }

        models.sort_by(|left, right| {
            left.name
                .to_lowercase()
                .cmp(&right.name.to_lowercase())
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(Self {
            root: canonical_root.display().to_string(),
            root_exists: true,
            models,
        })
    }

    pub(crate) fn resolve(root: &Path, id: &str) -> Result<PathBuf> {
        validate_catalog_id(id)?;
        let canonical_root = root
            .canonicalize()
            .with_context(|| format!("model catalog root is unavailable: {}", root.display()))?;
        let candidate = canonical_root.join(id);
        if fs::symlink_metadata(&candidate)
            .with_context(|| format!("model '{}' was not found in the catalog", id))?
            .file_type()
            .is_symlink()
        {
            return Err(anyhow!("model catalog symlinks are not loadable"));
        }
        let canonical_candidate = candidate
            .canonicalize()
            .with_context(|| format!("model '{}' was not found in the catalog", id))?;
        if !canonical_candidate.starts_with(&canonical_root) {
            return Err(anyhow!("model '{}' resolves outside the catalog root", id));
        }
        if model_file_format(&canonical_candidate).is_none()
            && directory_format(&canonical_candidate)?.is_none()
        {
            return Err(anyhow!("catalog entry '{}' is not a recognized model", id));
        }
        Ok(canonical_candidate)
    }

    pub(crate) fn remove(root: &Path, id: &str, expected_path: &Path) -> Result<()> {
        let candidate = Self::resolve(root, id)?;
        if candidate != expected_path {
            return Err(anyhow!(
                "catalog entry '{}' changed while removal was being prepared",
                id
            ));
        }
        let metadata = fs::symlink_metadata(&candidate)
            .with_context(|| format!("failed to inspect catalog entry '{}'", id))?;
        if metadata.file_type().is_symlink() {
            return Err(anyhow!("model catalog symlinks are not removable"));
        }
        if metadata.is_file() {
            fs::remove_file(&candidate)
                .with_context(|| format!("failed to remove catalog model '{}'", id))?;
        } else if metadata.is_dir() {
            fs::remove_dir_all(&candidate)
                .with_context(|| format!("failed to remove catalog model directory '{}'", id))?;
        } else {
            return Err(anyhow!("catalog entry '{}' is not a file or directory", id));
        }
        if let Err(error) = remove_provenance(root, id) {
            tracing::warn!(model = id, %error, "Failed to remove model provenance");
        }
        Ok(())
    }
}

pub(crate) fn validate_catalog_id(id: &str) -> Result<()> {
    let path = Path::new(id);
    let mut components = path.components();
    let valid = matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && !id.is_empty()
        && id.len() <= 255
        && id.trim() == id
        && !id.contains(['/', '\\'])
        && !id.chars().any(char::is_control);
    if valid {
        Ok(())
    } else {
        Err(anyhow!("invalid model catalog id"))
    }
}

pub(crate) fn validate_model_filename(id: &str) -> Result<()> {
    validate_catalog_id(id)?;
    if id.starts_with('.') || model_file_format(Path::new(id)).is_none() {
        return Err(anyhow!(
            "model filename must be a non-hidden GGUF, ONNX, or Core ML filename"
        ));
    }
    Ok(())
}

fn model_file_format(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("gguf") => Some("gguf"),
        Some("onnx") => Some("onnx"),
        Some("mlmodel") => Some("coreml"),
        _ => None,
    }
}

fn directory_format(path: &Path) -> Result<Option<&'static str>> {
    if matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("mlmodelc") | Some("mlpackage")
    ) {
        return Ok(Some("coreml"));
    }
    if path.join("bloom.json").is_file() {
        return Ok(Some("bloom"));
    }
    if path.join("config.json").is_file() {
        return Ok(Some("transformers"));
    }

    let mut has_openvino_xml = false;
    let mut has_openvino_weights = false;
    let mut inferred_format = None;
    for item in fs::read_dir(path)
        .with_context(|| format!("failed to inspect model directory '{}'", path.display()))?
    {
        let item =
            item.with_context(|| format!("failed to inspect an entry under '{}'", path.display()))?;
        if !item.file_type()?.is_file() {
            continue;
        }
        let candidate = item.path();
        let extension = candidate
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some("xml") => has_openvino_xml = true,
            Some("bin") => has_openvino_weights = true,
            Some("safetensors") => {
                if inferred_format
                    .map(|current| {
                        model_format_priority("transformers") > model_format_priority(current)
                    })
                    .unwrap_or(true)
                {
                    inferred_format = Some("transformers");
                }
            }
            _ => {
                if let Some(format) = model_file_format(&candidate)
                    && inferred_format
                        .map(|current| {
                            model_format_priority(format) > model_format_priority(current)
                        })
                        .unwrap_or(true)
                {
                    inferred_format = Some(format);
                }
            }
        }
    }
    if has_openvino_xml && has_openvino_weights {
        Ok(Some("openvino"))
    } else {
        Ok(inferred_format)
    }
}

fn model_format_priority(format: &str) -> u8 {
    match format {
        "gguf" => 4,
        "onnx" | "coreml" => 3,
        "transformers" => 2,
        _ => 1,
    }
}

fn directory_size(root: &Path) -> Result<(u64, bool)> {
    let mut total = 0_u64;
    let mut scanned = 0_usize;
    let mut complete = true;
    let mut pending = vec![(root.to_path_buf(), 0_usize)];
    while let Some((directory, depth)) = pending.pop() {
        if depth > MAX_DIRECTORY_DEPTH || scanned >= MAX_SCANNED_FILES {
            complete = false;
            continue;
        }
        for item in fs::read_dir(&directory)
            .with_context(|| format!("failed to measure model directory '{}'", root.display()))?
        {
            if scanned >= MAX_SCANNED_FILES {
                complete = false;
                break;
            }
            scanned += 1;
            let item = item?;
            let file_type = item.file_type()?;
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_file() {
                total = total.saturating_add(item.metadata()?.len());
            } else if file_type.is_dir() {
                if depth == MAX_DIRECTORY_DEPTH {
                    complete = false;
                } else {
                    pending.push((item.path(), depth + 1));
                }
            }
        }
    }
    Ok((total, complete))
}

fn display_name(id: &str) -> String {
    let stem = Path::new(id)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(id);
    stem.replace(['-', '_'], " ")
}

#[cfg(test)]
mod tests {
    use super::super::model_provenance::{
        METADATA_DIRECTORY, ModelAcquisitionKind, ModelPackageProvenanceDraft,
        ModelProvenanceDraft, read_provenance, write_package_provenance, write_provenance,
    };
    use super::*;
    use sha2::{Digest as _, Sha256};

    #[test]
    fn missing_catalog_is_an_empty_valid_catalog() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        let catalog = ModelCatalog::scan(&root, None).unwrap();

        assert!(!catalog.root_exists);
        assert!(catalog.models.is_empty());
    }

    #[test]
    fn discovers_supported_files_and_model_directories() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::write(root.join("tiny.gguf"), b"gguf").unwrap();
        fs::write(root.join("notes.txt"), b"ignore").unwrap();
        let model_dir = root.join("sample-model");
        fs::create_dir(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), b"{}").unwrap();
        fs::write(model_dir.join("weights.safetensors"), b"weights").unwrap();

        let catalog = ModelCatalog::scan(root, Some(&model_dir)).unwrap();

        assert_eq!(catalog.models.len(), 2);
        assert_eq!(catalog.models[0].id, "sample-model");
        assert!(catalog.models[0].active);
        assert_eq!(catalog.models[0].format, "transformers");
        assert_eq!(catalog.models[1].id, "tiny.gguf");
        assert_eq!(catalog.models[1].size_bytes, 4);
        assert!(catalog.models[1].size_complete);
    }

    #[test]
    fn rejects_more_than_the_public_catalog_entry_limit() {
        let temp = tempfile::tempdir().unwrap();
        for index in 0..=MAX_MODEL_CATALOG_ENTRIES {
            fs::write(temp.path().join(format!("model-{index}.gguf")), []).unwrap();
        }

        let error = ModelCatalog::scan(temp.path(), None)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("recognized models") && error.contains("4096"),
            "{error}"
        );
    }

    #[test]
    fn resolves_only_single_safe_catalog_ids() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("model.gguf"), b"gguf").unwrap();

        assert_eq!(
            ModelCatalog::resolve(temp.path(), "model.gguf").unwrap(),
            temp.path().join("model.gguf").canonicalize().unwrap()
        );
        assert!(ModelCatalog::resolve(temp.path(), "../model.gguf").is_err());
        assert!(ModelCatalog::resolve(temp.path(), "/tmp/model.gguf").is_err());
        assert!(ModelCatalog::resolve(temp.path(), " model.gguf").is_err());
        assert!(ModelCatalog::resolve(temp.path(), "model.gguf ").is_err());
        assert!(ModelCatalog::resolve(temp.path(), "model\ngguf").is_err());
        assert!(validate_catalog_id("directory\\model.gguf").is_err());
        assert!(validate_catalog_id(&"x".repeat(256)).is_err());
        assert!(ModelCatalog::resolve(temp.path(), "notes.txt").is_err());
    }

    #[test]
    fn validates_downloadable_single_file_names() {
        assert!(validate_model_filename("model.gguf").is_ok());
        assert!(validate_model_filename("model.ONNX").is_ok());
        assert!(validate_model_filename("../model.gguf").is_err());
        assert!(validate_model_filename(".hidden.gguf").is_err());
        assert!(validate_model_filename("model.safetensors").is_err());
    }

    #[test]
    fn removes_only_the_resolved_catalog_entry() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("remove.gguf");
        fs::write(&model, b"gguf").unwrap();
        let resolved = ModelCatalog::resolve(temp.path(), "remove.gguf").unwrap();

        ModelCatalog::remove(temp.path(), "remove.gguf", &resolved).unwrap();

        assert!(!model.exists());
        assert!(ModelCatalog::remove(temp.path(), "../outside.gguf", &resolved).is_err());
    }

    #[tokio::test]
    async fn catalog_exposes_valid_provenance_and_removal_cleans_its_record() {
        let temp = tempfile::tempdir().unwrap();
        let bytes = b"gguf provenance";
        let model = temp.path().join("recorded.gguf");
        fs::write(&model, bytes).unwrap();
        write_provenance(
            temp.path(),
            ModelProvenanceDraft {
                acquisition: ModelAcquisitionKind::Import,
                model_index_id: None,
                filename: "recorded.gguf".to_string(),
                size_bytes: bytes.len() as u64,
                source_url: Some("https://huggingface.co/acme/model".to_string()),
                source_host: Some("huggingface.co".to_string()),
                sha256: "ab".repeat(32),
                license: Some("MIT".to_string()),
            },
        )
        .await
        .unwrap();

        let catalog = ModelCatalog::scan(temp.path(), None).unwrap();
        let provenance = catalog.models[0].provenance.as_ref().unwrap();
        assert_eq!(provenance.acquisition, ModelAcquisitionKind::Import);
        assert_eq!(provenance.license.as_deref(), Some("MIT"));
        let resolved = ModelCatalog::resolve(temp.path(), "recorded.gguf").unwrap();
        ModelCatalog::remove(temp.path(), "recorded.gguf", &resolved).unwrap();

        assert!(!model.exists());
        assert!(
            read_provenance(temp.path(), "recorded.gguf", bytes.len() as u64)
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn package_removal_cleans_version_two_provenance() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("recorded-package");
        fs::create_dir(&model).unwrap();
        let config = b"{}";
        let weights = b"package weights";
        fs::write(model.join("config.json"), config).unwrap();
        fs::write(model.join("model.safetensors"), weights).unwrap();
        let files = vec![
            super::super::model_package::ModelPackageFile {
                filename: "config.json".to_string(),
                size_bytes: config.len() as u64,
                sha256: format!("{:x}", Sha256::digest(config)),
            },
            super::super::model_package::ModelPackageFile {
                filename: "model.safetensors".to_string(),
                size_bytes: weights.len() as u64,
                sha256: format!("{:x}", Sha256::digest(weights)),
            },
        ];
        let size_bytes = files.iter().map(|file| file.size_bytes).sum();
        let sha256 = super::super::model_package::package_digest(&files).unwrap();
        write_package_provenance(
            temp.path(),
            ModelPackageProvenanceDraft {
                acquisition: ModelAcquisitionKind::Download,
                model_index_id: Some("recorded-package".to_string()),
                directory: "recorded-package".to_string(),
                size_bytes,
                source_url: None,
                source_host: Some("huggingface.co".to_string()),
                sha256,
                license: Some("Apache-2.0".to_string()),
                files,
            },
        )
        .await
        .unwrap();

        let catalog = ModelCatalog::scan(temp.path(), None).unwrap();
        assert_eq!(
            catalog.models[0].provenance.as_ref().unwrap().file_count,
            Some(2)
        );
        let resolved = ModelCatalog::resolve(temp.path(), "recorded-package").unwrap();
        ModelCatalog::remove(temp.path(), "recorded-package", &resolved).unwrap();

        assert!(!model.exists());
        assert!(
            read_provenance(temp.path(), "recorded-package", size_bytes)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn catalog_reports_invalid_provenance_without_exposing_details() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("unsafe.gguf"), b"gguf").unwrap();
        let metadata_root = temp.path().join(METADATA_DIRECTORY);
        fs::create_dir(&metadata_root).unwrap();
        fs::write(metadata_root.join("unsafe.gguf.json"), b"not JSON").unwrap();

        let catalog = ModelCatalog::scan(temp.path(), None).unwrap();

        assert!(catalog.models[0].provenance.is_none());
        assert_eq!(
            catalog.models[0].provenance_error.as_deref(),
            Some("Stored provenance metadata is invalid or unsafe.")
        );
    }

    #[cfg(unix)]
    #[test]
    fn directory_removal_does_not_follow_nested_symlinks() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let protected = outside.path().join("protected.bin");
        fs::write(&protected, b"keep").unwrap();
        let model = temp.path().join("directory-model");
        fs::create_dir(&model).unwrap();
        fs::write(model.join("config.json"), b"{}").unwrap();
        symlink(&protected, model.join("linked.bin")).unwrap();
        let resolved = ModelCatalog::resolve(temp.path(), "directory-model").unwrap();

        ModelCatalog::remove(temp.path(), "directory-model", &resolved).unwrap();

        assert!(!model.exists());
        assert_eq!(fs::read(protected).unwrap(), b"keep");
    }

    #[cfg(unix)]
    #[test]
    fn catalog_symlinks_are_not_switchable_even_inside_the_root() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("model.gguf"), b"gguf").unwrap();
        symlink("model.gguf", temp.path().join("alias.gguf")).unwrap();

        assert!(ModelCatalog::resolve(temp.path(), "alias.gguf").is_err());
    }

    #[test]
    fn recognizes_openvino_and_coreml_directories() {
        let temp = tempfile::tempdir().unwrap();
        let openvino = temp.path().join("encoder");
        fs::create_dir(&openvino).unwrap();
        fs::write(openvino.join("model.xml"), b"xml").unwrap();
        fs::write(openvino.join("model.bin"), b"weights").unwrap();
        let coreml = temp.path().join("vision.mlmodelc");
        fs::create_dir(&coreml).unwrap();

        let catalog = ModelCatalog::scan(temp.path(), None).unwrap();

        assert_eq!(catalog.models.len(), 2);
        assert_eq!(catalog.models[0].format, "openvino");
        assert_eq!(catalog.models[1].format, "coreml");
    }

    #[test]
    fn marks_directory_sizes_as_partial_when_the_depth_limit_is_reached() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("deep-model");
        fs::create_dir(&model).unwrap();
        fs::write(model.join("config.json"), b"{}").unwrap();
        let mut nested = model;
        for level in 0..=MAX_DIRECTORY_DEPTH {
            nested = nested.join(format!("level-{level}"));
            fs::create_dir(&nested).unwrap();
        }
        fs::write(nested.join("weights.safetensors"), b"weights").unwrap();

        let catalog = ModelCatalog::scan(temp.path(), None).unwrap();

        assert_eq!(catalog.models.len(), 1);
        assert!(!catalog.models[0].size_complete);
    }
}
