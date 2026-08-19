#![cfg_attr(not(test), warn(clippy::unwrap_used))]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "bloom-quantize",
    version,
    about = "Bloom offline model compression and quantization CLI tool"
)]
struct Args {
    /// Path to input safetensors model file or directory
    #[arg(short, long)]
    input: PathBuf,

    /// Path to save the output quantized model
    #[arg(short, long)]
    output: PathBuf,

    /// Quantization method: rtn_int8, rtn_int4, awq_int4
    #[arg(short, long, default_value = "rtn_int8")]
    method: String,

    /// Group size for group-wise quantization (e.g. 128, 64)
    #[arg(short, long, default_value_t = 128)]
    group_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();

    println!("============================================================");
    println!("Bloom Offline Model Quantizer CLI");
    println!("- Input path: {:?}", args.input);
    println!("- Output path: {:?}", args.output);
    println!("- Method: {}", args.method);
    println!("- Group size: {}", args.group_size);
    println!("============================================================");

    // 1. Locate Safetensors file(s)
    let files = if args.input.is_file() {
        vec![args.input.clone()]
    } else if args.input.is_dir() {
        let mut sf_files = Vec::new();
        for entry in std::fs::read_dir(&args.input)?.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                sf_files.push(path);
            }
        }
        sf_files.sort();
        if sf_files.is_empty() {
            return Err(anyhow!(
                "No .safetensors files found in directory {:?}",
                args.input
            ));
        }
        sf_files
    } else {
        return Err(anyhow!("Input path {:?} does not exist", args.input));
    };

    let device = Device::Cpu;

    for sf_path in files {
        println!(
            "Processing file: {:?}",
            sf_path.file_name().unwrap_or_default()
        );

        let safetensors = candle_core::safetensors::load(&sf_path, &device)?;
        let mut output_tensors: HashMap<String, Tensor> = HashMap::new();

        for (name, original_tensor) in safetensors {
            let shape = original_tensor.shape().dims().to_vec();
            let _dtype = original_tensor.dtype();

            // Decide whether to quantize this tensor
            // Only quantize 2D weight matrices (e.g. projection, MLP weights)
            if shape.len() == 2 && (name.contains("weight") || name.contains("proj")) {
                println!("  Quantizing tensor: {} (shape: {:?})", name, shape);
                match args.method.as_str() {
                    "rtn_int8" => {
                        // Per-row symmetric INT8 quantization
                        let (quantized, scale) = quantize_rtn_int8(&original_tensor)?;
                        output_tensors.insert(format!("{}.qweight", name), quantized);
                        output_tensors.insert(format!("{}.scales", name), scale);
                    }
                    "rtn_int4" => {
                        // Per-group asymmetric INT4 quantization
                        let (qweight, scales, zeros) =
                            quantize_rtn_int4(&original_tensor, args.group_size)?;
                        output_tensors.insert(format!("{}.qweight", name), qweight);
                        output_tensors.insert(format!("{}.scales", name), scales);
                        output_tensors.insert(format!("{}.zeros", name), zeros);
                    }
                    "awq_int4" => {
                        // Simulated AWQ activation-aware scale scaling, then RTN INT4
                        let (qweight, scales, zeros) =
                            quantize_awq_int4(&original_tensor, args.group_size)?;
                        output_tensors.insert(format!("{}.qweight", name), qweight);
                        output_tensors.insert(format!("{}.scales", name), scales);
                        output_tensors.insert(format!("{}.zeros", name), zeros);
                    }
                    _ => {
                        // Fallback to RTN INT8
                        let (quantized, scale) = quantize_rtn_int8(&original_tensor)?;
                        output_tensors.insert(format!("{}.qweight", name), quantized);
                        output_tensors.insert(format!("{}.scales", name), scale);
                    }
                }
            } else {
                // Pass-through
                output_tensors.insert(name.to_string(), original_tensor);
            }
        }

        // Save output safetensors
        let out_filename = if args.output.is_dir() {
            let filename = sf_path
                .file_name()
                .ok_or_else(|| anyhow!("input path has no file name: {}", sf_path.display()))?;
            args.output.join(filename)
        } else {
            args.output.clone()
        };

        if let Some(parent) = out_filename.parent() {
            std::fs::create_dir_all(parent)?;
        }

        println!("Saving quantized tensors to: {:?}", out_filename);
        candle_core::safetensors::save(&output_tensors, &out_filename)?;
    }

    println!(
        "Quantization finished successfully in {:.2} seconds!",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn quantize_rtn_int8(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_rows, _cols) = w.dims2()?;
    let dev = w.device();

    // Find absolute maximum per row
    let max_abs = w.abs()?.max_keepdim(1)?;
    let eps = Tensor::new(1e-5f32, dev)?.broadcast_as(max_abs.shape())?;
    let scale = (max_abs / 127.0)?.maximum(&eps)?;

    // Quantize using symmetric quantization
    let quantized = (w.broadcast_div(&scale)? + 128.0)?;
    let quantized = quantized.round()?.to_dtype(DType::U8)?;

    let scale = scale.to_dtype(w.dtype())?;
    Ok((quantized, scale))
}

fn quantize_rtn_int4(w: &Tensor, group_size: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    let dev = w.device();

    // Group-wise asymmetric INT4 quantization
    // Reshape to (rows, cols / group_size, group_size)
    let w_reshaped = w.reshape((rows, cols / group_size, group_size))?;

    let max_val = w_reshaped.max_keepdim(2)?;
    let min_val = w_reshaped.min_keepdim(2)?;

    let range = (max_val - &min_val)?;
    let eps = Tensor::new(1e-5f32, dev)?.broadcast_as(range.shape())?;
    let scales = (range / 15.0)?.maximum(&eps)?;
    let zeros = min_val.round()?;

    // quantized = ((w_reshaped - zeros) / scales).round().clamp(0, 15)
    let quantized = w_reshaped
        .broadcast_sub(&zeros)?
        .broadcast_div(&scales)?
        .round()?;

    // Clamp to 0..15
    let zero_tensor = Tensor::zeros_like(&quantized)?;
    let fifteen_tensor = (Tensor::zeros_like(&quantized)? + 15.0)?;
    let quantized = quantized.maximum(&zero_tensor)?.minimum(&fifteen_tensor)?;

    // Repack shape back
    let q_flat = quantized.reshape((rows, cols))?;
    let q_u8 = q_flat.to_dtype(DType::U8)?;

    Ok((
        q_u8,
        scales.to_dtype(w.dtype())?,
        zeros.to_dtype(w.dtype())?,
    ))
}

fn quantize_awq_int4(w: &Tensor, group_size: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (_rows, _cols) = w.dims2()?;
    let dev = w.device();

    // AWQ uses activation-aware scales to protect salient activations.
    // Calculate the row-wise salience using the weight magnitude, and scale the weights accordingly.
    let salience = w.abs()?.mean_keepdim(0)?;
    let max_salience = salience.max_keepdim(1)?;
    let eps = Tensor::new(1e-5f32, dev)?.broadcast_as(salience.shape())?;
    let max_eps = max_salience.broadcast_add(&eps)?;
    let salience_norm = salience.broadcast_div(&max_eps)?;

    // Compute scaling factor s = salience_norm ^ 0.5
    let scale_factor = salience_norm.powf(0.5)?;

    // Scale weights: scaled_w = w * scale_factor
    let scaled_w = w.broadcast_mul(&scale_factor)?;

    // Perform standard RTN quantization on the scaled weights
    let (qweight, scales, zeros) = quantize_rtn_int4(&scaled_w, group_size)?;

    Ok((qweight, scales, zeros))
}
