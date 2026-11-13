//! Bloom RTN v1 portable packed weights. This is a storage format: native
//! models consume validated, materialized floating-point tensors at load time.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

pub fn is_rtn(config: &serde_json::Value) -> bool {
    config["quantization_config"]["quant_method"] == "bloom_rtn"
}

pub fn validate_config(config: &serde_json::Value) -> Result<u32> {
    let quant = &config["quantization_config"];
    ensure!(
        is_rtn(config) && quant["version"] == 1,
        "unsupported Bloom RTN format version"
    );
    let bits = quant["bits"].as_u64().context("missing Bloom RTN bits")?;
    ensure!(matches!(bits, 4 | 8), "Bloom RTN supports only 4 or 8 bits");
    ensure!(
        matches!(
            config["model_type"].as_str(),
            Some("qwen2" | "qwen3" | "llama")
        ),
        "Bloom RTN packages currently support dense Qwen2, Qwen3 and Llama only"
    );
    Ok(bits as u32)
}

pub fn load(
    files: &[PathBuf],
    config: &serde_json::Value,
    dtype: DType,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let bits = validate_config(config)?;
    let mut result = HashMap::new();
    let mut has_versioned_matrix = false;
    for file in files {
        let tensors = candle_core::safetensors::load(file, &Device::Cpu)?;
        has_versioned_matrix |= tensors.keys().any(|name| name.ends_with(".rtn_meta"));
        for (name, tensor) in dequantize_tensors(tensors, Some(bits))? {
            ensure!(
                !result.contains_key(&name),
                "duplicate restored RTN tensor: {name}"
            );
            result.insert(name, tensor.to_dtype(dtype)?.to_device(device)?);
        }
    }
    ensure!(
        has_versioned_matrix,
        "RTN package contains no versioned quantized matrix"
    );
    Ok(result)
}

/// Restore a self-describing RTN tensor bundle; no model/config is required.
pub fn dequantize_tensors(
    mut tensors: HashMap<String, Tensor>,
    expected_bits: Option<u32>,
) -> Result<HashMap<String, Tensor>> {
    let names: Vec<String> = tensors
        .keys()
        .filter_map(|name| name.strip_suffix(".rtn_meta").map(str::to_owned))
        .collect();
    // A model shard may contain only norms/biases, but a standalone bundle
    // needs its own format identity. The package loader checks this globally.
    ensure!(
        expected_bits.is_some() || !names.is_empty(),
        "standalone RTN bundle contains no versioned quantization metadata"
    );
    for name in names {
        ensure!(
            !tensors.contains_key(&name),
            "RTN restored tensor name collision: {name}"
        );
        let meta = tensors
            .remove(&format!("{name}.rtn_meta"))
            .context("missing RTN metadata")?;
        ensure!(
            meta.dtype() == DType::U32 && meta.dims() == [5],
            "invalid RTN metadata shape or dtype: {name}"
        );
        let meta = meta.to_vec1::<u32>()?;
        let (version, bits, group, rows, cols) = (
            meta[0],
            meta[1],
            meta[2] as usize,
            meta[3] as usize,
            meta[4] as usize,
        );
        ensure!(
            version == 1
                && matches!(bits, 4 | 8)
                && expected_bits.is_none_or(|expected| expected == bits),
            "unsupported or inconsistent RTN format: {name}"
        );
        ensure!(
            rows > 0 && cols > 0 && group > 0 && cols.is_multiple_of(group),
            "invalid RTN matrix/group dimensions: {name}"
        );
        ensure!(
            bits != 8 || group == cols,
            "RTN INT8 uses row scales: {name}"
        );
        let q = tensors
            .remove(&format!("{name}.qweight"))
            .context("missing RTN qweight")?;
        let width = if bits == 4 { cols.div_ceil(2) } else { cols };
        ensure!(
            q.dtype() == DType::U8 && q.dims() == [rows, width],
            "invalid RTN packed shape or dtype: {name}"
        );
        let scales = tensors
            .remove(&format!("{name}.scales"))
            .context("missing RTN scales")?;
        ensure!(
            scales.dtype() == DType::F32 && scales.dims() == [rows, cols / group],
            "invalid RTN scale shape or dtype: {name}"
        );
        let scales = scales.flatten_all()?.to_vec1::<f32>()?;
        ensure!(
            scales.iter().all(|x| x.is_finite() && *x > 0.0),
            "invalid RTN scale values: {name}"
        );
        let zeros = tensors.remove(&format!("{name}.zeros"));
        let offsets = if bits == 4 {
            let zeros = zeros.context("missing RTN offsets")?;
            ensure!(
                zeros.dtype() == DType::F32 && zeros.dims() == [rows, cols / group],
                "invalid RTN offset shape or dtype: {name}"
            );
            let values = zeros.flatten_all()?.to_vec1::<f32>()?;
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "nonfinite RTN offset: {name}"
            );
            Some(values)
        } else {
            ensure!(zeros.is_none(), "unexpected RTN INT8 offsets: {name}");
            None
        };
        let packed = q.flatten_all()?.to_vec1::<u8>()?;
        let count = rows
            .checked_mul(cols)
            .context("RTN element count overflow")?;
        let mut values = Vec::with_capacity(count);
        for row in 0..rows {
            if bits == 4 && cols % 2 != 0 {
                ensure!(
                    packed[row * width + width - 1] >> 4 == 0,
                    "nonzero RTN padding nibble: {name}"
                );
            }
            for col in 0..cols {
                let index = row * (cols / group) + col / group;
                let value = if let Some(offsets) = &offsets {
                    let byte = packed[row * width + col / 2];
                    let q = (byte >> (4 * (col % 2))) & 15;
                    f32::from(q) * scales[index] + offsets[index]
                } else {
                    ensure!(
                        packed[row * width + col] != 0,
                        "invalid RTN INT8 code: {name}"
                    );
                    (f32::from(packed[row * width + col]) - 128.0) * scales[index]
                };
                ensure!(value.is_finite(), "nonfinite RTN reconstruction: {name}");
                values.push(value);
            }
        }
        tensors.insert(name, Tensor::from_vec(values, (rows, cols), &Device::Cpu)?);
    }
    ensure!(
        !tensors
            .keys()
            .any(|name| name.ends_with(".qweight") || name.ends_with(".rtn_meta")),
        "orphaned RTN quantization tensors"
    );
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_requires_versioned_weights_but_accepts_norm_only_shards() {
        let directory = tempfile::tempdir().unwrap();
        let norms = directory.path().join("norms.safetensors");
        let weights = directory.path().join("weights.safetensors");
        let plain = HashMap::from([(
            "model.norm.weight".to_string(),
            Tensor::ones(2, DType::F32, &Device::Cpu).unwrap(),
        )]);
        candle_core::safetensors::save(&plain, &norms).unwrap();
        let config = serde_json::json!({"model_type":"qwen2", "quantization_config":{"quant_method":"bloom_rtn", "version":1, "bits":4}});
        let error = load(
            std::slice::from_ref(&norms),
            &config,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap_err();
        assert!(error.to_string().contains("no versioned quantized matrix"));
        assert!(dequantize_tensors(plain, None).is_err());

        let packed = HashMap::from([
            (
                "linear.weight.rtn_meta".into(),
                Tensor::new(&[1u32, 4, 2, 1, 2], &Device::Cpu).unwrap(),
            ),
            (
                "linear.weight.qweight".into(),
                Tensor::new(&[[0xf0u8]], &Device::Cpu).unwrap(),
            ),
            (
                "linear.weight.scales".into(),
                Tensor::new(&[[0.1f32]], &Device::Cpu).unwrap(),
            ),
            (
                "linear.weight.zeros".into(),
                Tensor::new(&[[0.0f32]], &Device::Cpu).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&packed, &weights).unwrap();
        let decoded = load(&[norms, weights], &config, DType::F32, &Device::Cpu).unwrap();
        assert!(decoded.contains_key("model.norm.weight"));
        assert_eq!(
            decoded["linear.weight"].to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 1.5]]
        );

        let mut legacy = packed;
        legacy.remove("linear.weight.rtn_meta");
        assert!(dequantize_tensors(legacy.clone(), None).is_err());
        assert!(dequantize_tensors(legacy, Some(4)).is_err());
    }
}
