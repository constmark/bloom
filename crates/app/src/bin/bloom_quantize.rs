#![cfg_attr(not(test), warn(clippy::unwrap_used))]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Method {
    #[value(name = "rtn_int8")]
    Int8,
    #[value(name = "rtn_int4")]
    Int4,
}

#[derive(Parser, Debug)]
#[command(
    name = "bloom-quantize",
    version,
    about = "Packed RTN INT4/INT8 tensor and native model package export"
)]
struct Args {
    /// Input Safetensors file or validated Hugging Face model directory.
    #[arg(short, long)]
    input: PathBuf,
    /// New output file (single input) or new directory (directory input).
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short, long, value_enum, default_value = "rtn_int8")]
    method: Method,
    /// INT4 group size; must divide every quantized matrix's column count.
    #[arg(short, long, default_value_t = 128)]
    group_size: usize,
    /// Export a loadable dense Qwen2/Qwen3/Llama package from a model directory.
    #[arg(long)]
    model_package: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    export(&args)?;
    println!(
        "Exported packed RTN v1 to {} in {:.2}s. See docs/quantization-export.md for the storage and runtime contract.",
        args.output.display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn export(args: &Args) -> Result<()> {
    ensure!(args.group_size > 0, "group_size must be greater than zero");
    let metadata = std::fs::symlink_metadata(&args.input)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "input must not be a symlink"
    );
    let directory = metadata.is_dir();
    ensure!(
        !args.model_package || directory,
        "--model-package requires a model directory"
    );
    let package_config = if args.model_package {
        let mut config: serde_json::Value =
            serde_json::from_slice(&read_package_file(&args.input.join("config.json"))?)?;
        ensure!(config.is_object(), "config.json must contain an object");
        ensure!(
            config
                .get("quantization_config")
                .is_none_or(serde_json::Value::is_null),
            "input is already quantized"
        );
        config["quantization_config"] = serde_json::json!({"quant_method":"bloom_rtn", "version":1, "bits":match args.method {Method::Int4 => 4, Method::Int8 => 8}});
        bloomai_engine::executor::rtn::validate_config(&config)?;
        Some(config)
    } else {
        None
    };
    let files = if directory {
        bloomai_engine::resolve_hf_safetensors_files(&args.input)?
    } else {
        ensure!(
            metadata.is_file(),
            "input must be a regular Safetensors file"
        );
        ensure!(
            args.input
                .extension()
                .is_some_and(|ext| ext == "safetensors"),
            "input must be a .safetensors file"
        );
        vec![args.input.clone()]
    };
    ensure!(!files.is_empty(), "input contains no Safetensors files");
    match std::fs::symlink_metadata(&args.output) {
        Ok(_) => bail!("output already exists: {}", args.output.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let parent = args
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    // All shards are transformed before publishing anything. The private
    // staging directory and persisted files are on the destination filesystem.
    let staging = tempfile::tempdir_in(parent)?;
    let mut staged = Vec::new();
    let mut original_names = HashSet::new();
    let mut output_names = HashSet::new();
    let mut weight_map = serde_json::Map::new();
    let mut total_size = 0usize;
    for source in &files {
        let tensors = candle_core::safetensors::load(source, &Device::Cpu)?;
        let names = tensors.keys().cloned().collect::<HashSet<_>>();
        ensure!(
            names.is_disjoint(&output_names),
            "cross-shard tensor name collision"
        );
        let output = quantize_tensors(tensors, args.method, args.group_size)
            .with_context(|| format!("quantizing {}", source.display()))?;
        for name in output.keys() {
            ensure!(
                !original_names.contains(name) && output_names.insert(name.clone()),
                "cross-shard tensor name collision: {name}"
            );
        }
        original_names.extend(names);
        let filename = source
            .file_name()
            .context("input has no filename")?
            .to_owned();
        for (name, tensor) in &output {
            weight_map.insert(
                name.clone(),
                serde_json::Value::String(filename.to_string_lossy().into_owned()),
            );
            total_size = total_size
                .checked_add(
                    tensor
                        .elem_count()
                        .checked_mul(tensor.dtype().size_in_bytes())
                        .context("tensor size overflow")?,
                )
                .context("checkpoint size overflow")?;
        }
        let temp = tempfile::NamedTempFile::new_in(staging.path())?;
        candle_core::safetensors::save(&output, temp.path())?;
        temp.as_file().sync_all()?;
        staged.push((filename, temp));
    }
    if let Some(config) = package_config {
        let mut metadata = vec![("config.json", serde_json::to_vec_pretty(&config)?)];
        // Copy only data files consumed by the native text/tokenizer route.
        // Python code, stale bloom manifests, and old shard indexes are excluded.
        metadata.push((
            "tokenizer.json",
            read_package_file(&args.input.join("tokenizer.json"))?,
        ));
        for name in [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
        ] {
            if args.input.join(name).try_exists()? {
                metadata.push((name, read_package_file(&args.input.join(name))?));
            }
        }
        if files.len() > 1 {
            metadata.push(("model.safetensors.index.json", serde_json::to_vec_pretty(&serde_json::json!({"metadata":{"total_size": total_size}, "weight_map": weight_map}))?));
        }
        for (name, bytes) in metadata {
            use std::io::Write;
            let mut file = tempfile::NamedTempFile::new_in(staging.path())?;
            file.write_all(&bytes)?;
            file.as_file().sync_all()?;
            staged.push((name.into(), file));
        }
    }
    if directory {
        // create_dir is an atomic no-overwrite reservation, including symlinks.
        std::fs::create_dir(&args.output)?;
        let mut published = Vec::new();
        let result = (|| -> Result<()> {
            for (name, file) in staged {
                let destination = args.output.join(name);
                file.persist_noclobber(&destination)?;
                published.push(destination);
            }
            Ok(())
        })();
        if result.is_err() {
            // Remove only files created by this operation, never arbitrary
            // directory contents introduced by another process.
            for path in published {
                let _ = std::fs::remove_file(path);
            }
            let _ = std::fs::remove_dir(&args.output);
        }
        result?;
    } else {
        let (_, file) = staged.pop().context("missing staged output")?;
        file.persist_noclobber(&args.output)?;
    }
    Ok(())
}

fn read_package_file(path: &Path) -> Result<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path)?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink(),
        "package metadata must be a regular file: {}",
        path.display()
    );
    ensure!(
        meta.len() <= 64 * 1024 * 1024,
        "package metadata exceeds 64 MiB: {}",
        path.display()
    );
    Ok(std::fs::read(path)?)
}

fn quantize_tensors(
    tensors: HashMap<String, Tensor>,
    method: Method,
    group: usize,
) -> Result<HashMap<String, Tensor>> {
    let mut output = HashMap::new();
    for (name, tensor) in &tensors {
        if tensor.rank() != 2 || !(name.contains("weight") || name.contains("proj")) {
            ensure!(
                output.insert(name.clone(), tensor.clone()).is_none(),
                "output tensor collision: {name}"
            );
            continue;
        }
        for suffix in ["qweight", "scales", "zeros", "rtn_meta"] {
            ensure!(
                !tensors.contains_key(&format!("{name}.{suffix}")),
                "input already contains quantization tensor {name}.{suffix}"
            );
        }
        let (weights, scales, zeros) = quantize_matrix(tensor, method, group)?;
        let (rows, cols) = tensor.dims2()?;
        let bits = match method {
            Method::Int4 => 4u32,
            Method::Int8 => 8,
        };
        let width = if bits == 8 { cols } else { group };
        let metadata = [
            1u32,
            bits,
            u32::try_from(width)?,
            u32::try_from(rows)?,
            u32::try_from(cols)?,
        ];
        output.insert(
            format!("{name}.rtn_meta"),
            Tensor::new(&metadata, &Device::Cpu)?,
        );
        output.insert(format!("{name}.qweight"), weights);
        output.insert(format!("{name}.scales"), scales);
        if let Some(zeros) = zeros {
            output.insert(format!("{name}.zeros"), zeros);
        }
    }
    Ok(output)
}

fn quantize_matrix(
    tensor: &Tensor,
    method: Method,
    group: usize,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    let (rows, cols) = tensor.dims2()?;
    ensure!(rows > 0 && cols > 0, "cannot quantize an empty matrix");
    ensure!(
        matches!(
            tensor.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ),
        "quantization requires floating-point weights"
    );
    ensure!(group > 0, "group_size must be greater than zero");
    let values = tensor
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "weights must be finite"
    );
    ensure!(
        values.iter().all(|v| f64::from(*v as f32) == *v),
        "weights must be exactly representable as F32 before RTN export"
    );
    let width = match method {
        Method::Int8 => cols,
        Method::Int4 => {
            ensure!(
                cols.is_multiple_of(group),
                "group_size {group} must divide matrix column count {cols}"
            );
            group
        }
    };
    let mut quantized = Vec::with_capacity(values.len());
    let mut scales = Vec::new();
    let mut offsets = Vec::new();
    for values in values.chunks_exact(width) {
        let (scale, offset) = match method {
            Method::Int8 => {
                let max = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
                (if max == 0.0 { 1.0 } else { max / 127.0 }, 0.0)
            }
            Method::Int4 => {
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (if min == max { 1.0 } else { (max - min) / 15.0 }, min)
            }
        };
        // Store F32 scales/offsets consistently for F16/BF16/F32/F64 inputs.
        // Reject values that cannot be represented rather than emitting NaN/0.
        let scale = scale as f32;
        let offset = offset as f32;
        ensure!(
            scale.is_finite() && scale > 0.0 && offset.is_finite(),
            "quantization parameters cannot be represented as finite F32"
        );
        scales.push(scale);
        offsets.push(offset);
        for value in values {
            let q = match method {
                Method::Int8 => (value / f64::from(scale)).round().clamp(-127.0, 127.0) + 128.0,
                Method::Int4 => ((value - f64::from(offset)) / f64::from(scale))
                    .round()
                    .clamp(0.0, 15.0),
            };
            quantized.push(q as u8);
        }
    }
    let weights = match method {
        Method::Int8 => Tensor::from_vec(quantized, (rows, cols), &Device::Cpu)?,
        Method::Int4 => {
            let mut packed = Vec::with_capacity(rows * cols.div_ceil(2));
            for row in quantized.chunks_exact(cols) {
                for pair in row.chunks(2) {
                    packed.push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
                }
            }
            Tensor::from_vec(packed, (rows, cols.div_ceil(2)), &Device::Cpu)?
        }
    };
    match method {
        Method::Int8 => Ok((
            weights,
            Tensor::from_vec(scales, (rows, 1), &Device::Cpu)?,
            None,
        )),
        Method::Int4 => Ok((
            weights,
            Tensor::from_vec(scales, (rows, cols / group), &Device::Cpu)?,
            Some(Tensor::from_vec(
                offsets,
                (rows, cols / group),
                &Device::Cpu,
            )?),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_native_fixture(path: &Path, sharded: bool) {
        std::fs::create_dir(path).unwrap();
        let config = serde_json::json!({"architectures":["Qwen2ForCausalLM"], "model_type":"qwen2", "vocab_size":4, "hidden_size":4, "intermediate_size":8, "num_hidden_layers":1, "num_attention_heads":2, "num_key_value_heads":1, "max_position_embeddings":64, "sliding_window":64, "max_window_layers":1, "tie_word_embeddings":false, "rope_theta":10000.0, "rms_norm_eps":1e-6, "use_sliding_window":false, "hidden_act":"silu", "eos_token_id":3, "torch_dtype":"float32"});
        std::fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let tokenizer = serde_json::json!({"version":"1.0", "truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"fixture":1,"hello":2,"<eos>":3},"unk_token":"<unk>"}});
        std::fs::write(
            path.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::ones((4, 4), DType::F32, &Device::Cpu).unwrap(),
        );
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ] {
            tensors.insert(
                name.to_string(),
                Tensor::ones(4, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        for (name, rows, cols) in [
            ("self_attn.q_proj", 4, 4),
            ("self_attn.k_proj", 2, 4),
            ("self_attn.v_proj", 2, 4),
            ("self_attn.o_proj", 4, 4),
            ("mlp.gate_proj", 8, 4),
            ("mlp.up_proj", 8, 4),
            ("mlp.down_proj", 4, 8),
        ] {
            tensors.insert(
                format!("model.layers.0.{name}.weight"),
                Tensor::zeros((rows, cols), DType::F32, &Device::Cpu).unwrap(),
            );
            if matches!(
                name,
                "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj"
            ) {
                tensors.insert(
                    format!("model.layers.0.{name}.bias"),
                    Tensor::zeros(rows, DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        let mut head = vec![0f32; 16];
        head[4..8].fill(1.0);
        tensors.insert(
            "lm_head.weight".into(),
            Tensor::from_vec(head, (4, 4), &Device::Cpu).unwrap(),
        );
        if sharded {
            let mut shards = [HashMap::new(), HashMap::new()];
            let mut index = serde_json::Map::new();
            let filenames = [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ];
            for (i, (name, tensor)) in tensors.into_iter().enumerate() {
                index.insert(name.clone(), serde_json::json!(filenames[i % 2]));
                shards[i % 2].insert(name, tensor);
            }
            for (file, tensors) in filenames.iter().zip(shards) {
                candle_core::safetensors::save(&tensors, path.join(file)).unwrap();
            }
            std::fs::write(
                path.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map":index})).unwrap(),
            )
            .unwrap();
        } else {
            candle_core::safetensors::save(&tensors, path.join("model.safetensors")).unwrap();
        }
    }

    #[test]
    fn exported_single_and_sharded_packages_execute_native_qwen2() {
        use bloomai_core::{DeviceKind, GenerationParams};
        use bloomai_engine::{Engine, ModelInput};
        for sharded in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let source = dir.path().join("source");
            write_native_fixture(&source, sharded);
            for (method, name) in [(Method::Int4, "int4"), (Method::Int8, "int8")] {
                let output = dir.path().join(name);
                export(&Args {
                    input: source.clone(),
                    output: output.clone(),
                    method,
                    group_size: 4,
                    model_package: true,
                })
                .unwrap();
                let manifest = bloomai_engine::load_manifest(&output).unwrap();
                assert!(!manifest.runtime_hints.supports_mmap);
                let estimate =
                    bloomai_engine::estimate_memory_for_device(&manifest, 16, DeviceKind::Cpu);
                assert!(!estimate.mmap_residency_applied);
                assert!(
                    estimate.weight_bytes
                        > manifest.files.iter().map(|f| f.size_bytes).sum::<usize>()
                );
                let model = bloomai_engine::executor::candle::CandleEngine
                    .load(&output, DeviceKind::Cpu)
                    .unwrap();
                let result = model
                    .infer(
                        ModelInput::Text {
                            prompt: "hello".into(),
                        },
                        &GenerationParams {
                            temperature: 0.0,
                            max_tokens: 3,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(result.text.as_deref(), Some("fixture fixture fixture"));
                assert_eq!(
                    bloomai_engine::resolve_hf_safetensors_files(&output)
                        .unwrap()
                        .len(),
                    if sharded { 2 } else { 1 }
                );
                let config_path = output.join("config.json");
                let original_config: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
                for (field, value) in [
                    ("bits", serde_json::json!(3)),
                    ("version", serde_json::json!(2)),
                ] {
                    let mut config = original_config.clone();
                    config["quantization_config"][field] = value;
                    std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
                    assert!(bloomai_engine::load_manifest(&output).is_err());
                }
            }
        }
    }

    #[test]
    fn packed_bundles_restore_odd_rows_and_reject_corrupt_metadata() {
        use bloomai_engine::executor::rtn::dequantize_tensors;
        let original = Tensor::new(&[[0.25f32, -1.0, 3.0], [2.0, 2.0, 2.0]], &Device::Cpu).unwrap();
        for method in [Method::Int4, Method::Int8] {
            let source = HashMap::from([("linear.weight".into(), original.clone())]);
            let packed = quantize_tensors(source, method, 3).unwrap();
            let bits = match method {
                Method::Int4 => 4,
                Method::Int8 => 8,
            };
            assert_eq!(
                packed["linear.weight.qweight"].dims(),
                &[2, if bits == 4 { 2 } else { 3 }]
            );
            let restored = dequantize_tensors(packed.clone(), Some(bits)).unwrap();
            let values = restored["linear.weight"].to_vec2::<f32>().unwrap();
            let expected = original.to_vec2::<f32>().unwrap();
            for (row, target) in values.iter().zip(expected) {
                for (value, target) in row.iter().zip(target) {
                    assert!((value - target).abs() < if bits == 4 { 0.14 } else { 0.013 });
                }
            }
            for metadata in [
                [2, bits, 3, 2, 3],
                [1, bits, 0, 2, 3],
                [1, bits, 3, u32::MAX, 3],
            ] {
                let mut corrupt = packed.clone();
                corrupt.insert(
                    "linear.weight.rtn_meta".into(),
                    Tensor::new(&metadata, &Device::Cpu).unwrap(),
                );
                assert!(dequantize_tensors(corrupt, Some(bits)).is_err());
            }
            let mut corrupt = packed.clone();
            corrupt.remove("linear.weight.scales");
            assert!(dequantize_tensors(corrupt, Some(bits)).is_err());
            assert!(dequantize_tensors(packed, Some(if bits == 4 { 8 } else { 4 })).is_err());
        }
    }

    #[test]
    fn rejects_f64_precision_loss_before_quantization() {
        let weights = Tensor::new(&[[1.00000001f64, 1.00000002]], &Device::Cpu).unwrap();
        assert!(
            quantize_matrix(&weights, Method::Int4, 2)
                .unwrap_err()
                .to_string()
                .contains("exactly representable")
        );
    }

    #[test]
    fn cross_shard_collisions_fail_before_publishing() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("source");
        std::fs::create_dir(&input).unwrap();
        let names = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        let weights = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
        candle_core::safetensors::save(
            &HashMap::from([("linear.weight".to_string(), weights)]),
            input.join(names[0]),
        )
        .unwrap();
        let scale = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        candle_core::safetensors::save(
            &HashMap::from([("linear.weight.scales".to_string(), scale)]),
            input.join(names[1]),
        )
        .unwrap();
        let index = serde_json::json!({"weight_map":{"linear.weight":names[0], "linear.weight.scales": names[1]}});
        std::fs::write(
            input.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let args = Args {
            input,
            output: dir.path().join("export"),
            method: Method::Int4,
            group_size: 4,
            model_package: false,
        };
        assert!(
            export(&args)
                .unwrap_err()
                .to_string()
                .contains("cross-shard")
        );
        assert!(!args.output.exists());
    }

    #[test]
    fn rejects_unknown_and_fake_awq_methods() {
        for method in ["rtn_int9", "awq_int4"] {
            assert!(
                Args::try_parse_from(["quantize", "-i", "in", "-o", "out", "-m", method]).is_err()
            );
        }
    }

    #[test]
    fn reconstructs_fractional_constant_and_mixed_precision_weights() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
            let original = Tensor::new(
                &[
                    [0.2f32, 0.3, 0.4, 0.5],
                    [-0.25, -0.25, -0.25, -0.25],
                    [0., 0., 0., 0.],
                ],
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            for method in [Method::Int4, Method::Int8] {
                let (q, s, z) = quantize_matrix(&original, method, 4).unwrap();
                let q = q.flatten_all().unwrap().to_vec1::<u8>().unwrap();
                let s = s.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let z = z.map(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap());
                let values = original
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (i, value) in values.iter().enumerate() {
                    let decoded = match &z {
                        Some(z) => {
                            f32::from((q[i / 2] >> (4 * (i % 2))) & 15) * s[i / 4] + z[i / 4]
                        }
                        None => (f32::from(q[i]) - 128.0) * s[i / 4],
                    };
                    assert!(
                        (decoded - value).abs() <= s[i / 4] * 0.501 + 1e-6,
                        "{method:?} {dtype:?}: {decoded} != {value}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_groups_and_nonfinite_weights() {
        let weights = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        for group in [0, 3, 8] {
            assert!(quantize_matrix(&weights, Method::Int4, group).is_err());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let weights = Tensor::new(&[[invalid, 0.0]], &Device::Cpu).unwrap();
            assert!(quantize_matrix(&weights, Method::Int8, 2).is_err());
        }
    }

    #[test]
    fn export_preserves_source_and_existing_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model.safetensors");
        let tensors = HashMap::from([(
            "linear.weight".to_string(),
            Tensor::new(&[[0.2f32, 0.4]], &Device::Cpu).unwrap(),
        )]);
        candle_core::safetensors::save(&tensors, &input).unwrap();
        let original = std::fs::read(&input).unwrap();
        let mut args = Args {
            input: input.clone(),
            output: input.clone(),
            method: Method::Int4,
            group_size: 2,
            model_package: false,
        };
        assert!(export(&args).is_err());
        assert_eq!(std::fs::read(&input).unwrap(), original);
        args.output = dir.path().join("new.safetensors");
        export(&args).unwrap();
        let saved = std::fs::read(&args.output).unwrap();
        assert!(export(&args).is_err());
        assert_eq!(std::fs::read(&args.output).unwrap(), saved);
        assert_eq!(std::fs::read(&input).unwrap(), original);
        let reloaded = candle_core::safetensors::load(&args.output, &Device::Cpu).unwrap();
        assert!(reloaded.contains_key("linear.weight.zeros"));
    }

    #[test]
    fn rejects_output_tensor_collisions() {
        let tensor = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        let tensors = HashMap::from([
            ("linear.weight".to_string(), tensor.clone()),
            ("linear.weight.scales".to_string(), tensor),
        ]);
        assert!(quantize_tensors(tensors, Method::Int4, 2).is_err());
    }

    #[test]
    fn directory_export_is_new_and_does_not_copy_stale_package_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("source");
        std::fs::create_dir(&input).unwrap();
        let tensors = HashMap::from([(
            "linear.weight".to_string(),
            Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap(),
        )]);
        candle_core::safetensors::save(&tensors, input.join("model.safetensors")).unwrap();
        let mut args = Args {
            input,
            output: dir.path().join("export"),
            method: Method::Int4,
            group_size: 3,
            model_package: false,
        };
        assert!(export(&args).is_err());
        assert!(!args.output.exists());
        args.group_size = 4;
        export(&args).unwrap();
        assert!(args.output.join("model.safetensors").exists());
        assert!(!args.output.join("config.json").exists());
    }
}
