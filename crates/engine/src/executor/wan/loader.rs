//! GGUF and Safetensors weight loading for Wan2.1 DiT model.
//!
//! Handles loading DiT weights from GGUF (quantized) or Safetensors (FP16/FP32)
//! files into Candle VarBuilder for model construction.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// Locate the primary weight file(s) in a model directory.
///
/// Search order:
/// 1. Single `.gguf` file
/// 2. Single `.safetensors` file
/// 3. Sharded `model-*-of-*.safetensors` files
pub fn find_weight_files(model_path: &Path) -> Result<WeightFiles> {
    if model_path.is_file() {
        let ext = model_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        return match ext {
            "gguf" => Ok(WeightFiles::Gguf(model_path.to_path_buf())),
            "safetensors" => Ok(WeightFiles::Safetensors(vec![model_path.to_path_buf()])),
            _ => Err(anyhow!("unsupported weight file extension: {}", ext)),
        };
    }

    // Search for GGUF
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                return Ok(WeightFiles::Gguf(path));
            }
        }
    }

    // Search for safetensors
    let single_st = model_path.join("model.safetensors");
    if single_st.exists() {
        return Ok(WeightFiles::Safetensors(vec![single_st]));
    }

    // Sharded safetensors
    if let Ok(entries) = std::fs::read_dir(model_path) {
        let mut shards: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let n = name.to_string_lossy();
                n.starts_with("model-") && n.ends_with(".safetensors")
            })
            .map(|e| e.path())
            .collect();
        if !shards.is_empty() {
            shards.sort();
            return Ok(WeightFiles::Safetensors(shards));
        }
    }

    // Also check for wan-specific naming patterns
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let Some(name) = path.file_name() else {
                    continue;
                };
                let name = name.to_string_lossy().to_lowercase();
                if name.contains("wan") && name.ends_with(".safetensors") {
                    return Ok(WeightFiles::Safetensors(vec![path]));
                }
            }
        }
    }

    Err(anyhow!(
        "no GGUF or Safetensors weight files found in {}",
        model_path.display()
    ))
}

/// Discovered weight file type.
#[derive(Debug, Clone)]
pub enum WeightFiles {
    Gguf(PathBuf),
    Safetensors(Vec<PathBuf>),
}

/// Build a Candle VarBuilder from discovered weight files.
///
/// For GGUF files, reads the GGUF header and creates a mmaped VarBuilder.
/// For Safetensors, creates a standard mmaped VarBuilder.
pub fn build_var_builder<'a>(
    weights: &WeightFiles,
    dtype: DType,
    device: &'a Device,
    skip_blocks: bool,
) -> Result<VarBuilder<'a>> {
    match weights {
        WeightFiles::Gguf(path) => build_gguf_var_builder(path, dtype, device, skip_blocks),
        WeightFiles::Safetensors(paths) => {
            let vb =
                unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
            Ok(vb)
        }
    }
}

/// Build VarBuilder from a GGUF file.
///
/// Reads the GGUF header, enumerates all tensors, and constructs a VarBuilder
/// from a HashMap of dequantized tensors.
fn build_gguf_var_builder<'a>(
    path: &Path,
    dtype: DType,
    device: &'a Device,
    skip_blocks: bool,
) -> Result<VarBuilder<'a>> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow!("failed to open GGUF file {}: {}", path.display(), e))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("failed to read GGUF header: {}", e))?;

    let tensor_names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    tracing::info!("GGUF file contains {} tensors", tensor_names.len());

    let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
    let mut q_map: HashMap<String, std::sync::Arc<candle_core::quantized::QTensor>> =
        HashMap::new();

    for name in &tensor_names {
        // If we are building the GPU VarBuilder, we can skip block tensors to save VRAM.
        // They will be loaded via the CPU VarBuilder instead.
        if skip_blocks && name.starts_with("blocks.") {
            continue;
        }

        match content.tensor(&mut file, name, device) {
            Ok(qt) => {
                // To avoid massive F32 memory allocations and PCIe bandwidth usage,
                // we keep block weights as quantized QTensor and store them in a thread-local map.
                // The dit::linear layer will lazily instantiate QMatMul directly from these.
                if name.starts_with("blocks.")
                    && name.ends_with(".weight")
                    && !name.contains("norm")
                {
                    q_map.insert(name.clone(), std::sync::Arc::new(qt));
                    continue;
                }

                // content.tensor() returns QTensor; dequantize to get a regular Tensor
                match qt.dequantize(device) {
                    Ok(t) => {
                        let t = t.to_dtype(dtype).unwrap_or(t);
                        tensors.insert(name.clone(), t);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to dequantize tensor '{}': {}", name, e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to load tensor '{}': {}", name, e);
            }
        }
    }

    // Set the global QTENSOR_MAP for dit.rs to consume during initialization
    if !q_map.is_empty() {
        crate::executor::wan::dit::QTENSOR_MAP.with(|m| {
            let mut map_ref = m.borrow_mut();
            if let Some(existing) = map_ref.as_mut() {
                existing.extend(q_map);
            } else {
                *map_ref = Some(q_map);
            }
        });
    }

    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    Ok(vb)
}

/// Find a specific companion file in the model directory.
///
/// Looks for T5 encoder, VAE, and tokenizer files that are needed
/// alongside the main DiT GGUF weights.
pub fn find_companion_file(model_path: &Path, patterns: &[&str]) -> Option<PathBuf> {
    let search_dir = if model_path.is_file() {
        model_path.parent().unwrap_or(Path::new("."))
    } else {
        model_path
    };

    // Direct name matches
    for pattern in patterns {
        let candidate = search_dir.join(pattern);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Fuzzy search
    if let Ok(entries) = std::fs::read_dir(search_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            for pattern in patterns {
                if name.contains(&pattern.to_lowercase()) {
                    return Some(entry.path());
                }
            }
        }
    }

    None
}

/// Find T5 encoder weights in the model directory.
pub fn find_t5_weights(model_path: &Path) -> Option<PathBuf> {
    find_companion_file(
        model_path,
        &[
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "umt5_xxl_fp16.safetensors",
            "umt5_xxl_fp32.safetensors",
            "umt5-xxl-enc-fp32.safetensors",
            "t5_encoder.safetensors",
            "text_encoder.safetensors",
        ],
    )
}

/// Find VAE weights in the model directory.
pub fn find_vae_weights(model_path: &Path) -> Option<PathBuf> {
    find_companion_file(
        model_path,
        &[
            "Wan2.1_VAE_fp32.safetensors",
            "Wan2.1_VAE_fp16.safetensors",
            "vae.safetensors",
            "decoder.safetensors",
        ],
    )
}

/// Find tokenizer config in the model directory.
pub fn find_tokenizer(model_path: &Path) -> Option<PathBuf> {
    find_companion_file(model_path, &["tokenizer.json", "google/umt5-xxl"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_files_enum() {
        let wf = WeightFiles::Gguf(PathBuf::from("model.gguf"));
        match wf {
            WeightFiles::Gguf(p) => assert_eq!(p.to_str().unwrap(), "model.gguf"),
            _ => panic!("expected Gguf variant"),
        }
    }
}
