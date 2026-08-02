#![cfg_attr(not(test), warn(clippy::unwrap_used))]

use std::fs::File;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut json = false;
    let mut limit = 12usize;
    let mut path_arg = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--limit" => {
                let value = args.next().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "--limit requires a numeric value",
                    )
                })?;
                limit = value.parse()?;
            }
            "-h" | "--help" => {
                println!("usage: inspect_gguf [--json] [--limit N] <file.gguf>");
                return Ok(());
            }
            _ if path_arg.is_none() => path_arg = Some(arg),
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unexpected argument: {arg}"),
                )
                .into());
            }
        }
    }

    let path_arg = path_arg.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: inspect_gguf [--json] [--limit N] <file.gguf>",
        )
    })?;
    let path = Path::new(&path_arg);
    let mut file = File::open(path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|value| value.to_string().ok().map(|s| s.to_string()))
        .unwrap_or_else(|| "<unknown>".to_string());
    let name = content
        .metadata
        .get("general.name")
        .and_then(|value| value.to_string().ok().map(|s| s.to_string()))
        .unwrap_or_else(|| "<unknown>".to_string());

    if json {
        let context_length = if arch == "<unknown>" {
            None
        } else {
            content
                .metadata
                .get(&format!("{}.context_length", arch.to_lowercase()))
                .or_else(|| content.metadata.get("general.context_length"))
                .and_then(|value| value.to_u64().ok())
        };
        let tensors: Vec<_> = content
            .tensor_infos
            .iter()
            .take(limit)
            .map(|(name, info)| {
                serde_json::json!({
                    "name": name,
                    "shape": format!("{:?}", info.shape),
                    "dtype": format!("{:?}", info.ggml_dtype),
                })
            })
            .collect();
        let first_weight_dtype = content
            .tensor_infos
            .iter()
            .find(|(name, _)| {
                name.contains("weight") && (name.contains("blk") || name.contains("layers"))
            })
            .or_else(|| content.tensor_infos.iter().next())
            .map(|(_, info)| format!("{:?}", info.ggml_dtype));

        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": path.display().to_string(),
                "name": name,
                "architecture": arch,
                "context_length": context_length,
                "metadata_count": content.metadata.len(),
                "tensor_count": content.tensor_infos.len(),
                "first_weight_dtype": first_weight_dtype,
                "tensors": tensors,
            }))?
        );
        return Ok(());
    }

    println!("GGUF name: {name}");
    println!("GGUF architecture: {arch}");
    println!(
        "GGUF contains {} tensors. First tensors:",
        content.tensor_infos.len()
    );
    for (name, info) in content.tensor_infos.iter().take(limit) {
        println!(
            "Tensor: {} -> shape: {:?}, dtype: {:?}",
            name, info.shape, info.ggml_dtype
        );
    }
    Ok(())
}
