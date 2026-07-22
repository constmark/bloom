use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use bloomai_core::{
    DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat,
    ResidencyStrategy,
};
use candle_core::{DType, Device, Tensor};

use crate::core::parallelism::ParallelStrategy;
use crate::core::quantization::{QuantMethod, QuantizationConfig};
use crate::engine::BackendMaturity;
use crate::{
    engine::{Engine, EngineCapability},
    io::{ModelInput, ModelOutput},
    model::{LoadedModel, ModelMetadata},
};

use super::speculative::{
    config_supports_mtp, verify_greedy_tokens, verify_with_rejection_sampling, NGramStrategy,
    SpeculativeMode, SpeculativeStrategy,
};

/// Model type enum for supported variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    Qwen2,
    Qwen3,
    Gemma4,
    Llama,
}

impl ModelType {
    fn from_config(config: &serde_json::Value) -> Option<Self> {
        config
            .get("model_type")
            .and_then(|t| t.as_str())
            .and_then(|t| match t {
                "qwen2" => Some(ModelType::Qwen2),
                "qwen3" => Some(ModelType::Qwen3),
                "gemma4_unified" | "gemma4" => Some(ModelType::Gemma4),
                "llama" => Some(ModelType::Llama),
                _ => None,
            })
    }
}

/// Find all safetensors files in model directory
fn find_safetensors_files(model_path: &Path) -> Vec<PathBuf> {
    // Check for single model.safetensors first
    let single = model_path.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }

    // Check for sharded model-*-of-*.safetensors files
    if let Ok(entries) = std::fs::read_dir(model_path) {
        let mut shard_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("model-") && name_str.ends_with(".safetensors")
            })
            .map(|e| e.path())
            .collect();

        // Sort by shard number
        shard_files.sort_by(|a, b| {
            let name_a = a.file_name().unwrap().to_string_lossy();
            let name_b = b.file_name().unwrap().to_string_lossy();
            name_a.cmp(&name_b)
        });

        if !shard_files.is_empty() {
            return shard_files;
        }
    }

    Vec::new()
}

/// Find GGUF file in model directory
fn find_gguf_file(model_path: &Path) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                return Some(path);
            }
        }
    }
    None
}

pub struct CandleEngine;

impl Engine for CandleEngine {
    fn name(&self) -> &'static str {
        "candle"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            vec![DeviceKind::Cpu, DeviceKind::Gpu]
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            vec![DeviceKind::Cpu]
        }
    }

    fn default_device(&self) -> DeviceKind {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            DeviceKind::Gpu
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            DeviceKind::Cpu
        }
    }

    fn capability(&self) -> EngineCapability {
        #[allow(unused_mut)]
        let mut devices = vec![DeviceClass::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(DeviceClass::DiscreteGpu);
        #[cfg(feature = "metal")]
        devices.push(DeviceClass::IntegratedGpu);

        EngineCapability {
            engine_name: "candle",
            supported_families: vec![ModelFamily::Qwen, ModelFamily::Gemma, ModelFamily::Llama],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::BF16,
                bloomai_core::DType::I8,
                bloomai_core::DType::U8,
                bloomai_core::DType::I4,
                bloomai_core::DType::NF4,
                bloomai_core::DType::Q8,
                bloomai_core::DType::Q4,
            ],
            supported_formats: vec![ModelFormat::Safetensors, ModelFormat::Gguf],
            supported_devices: devices,
            supported_modalities: vec![Modality::Text],
            supports_streaming: true,
            supports_quantized_models: true,
            supports_embeddings: true,
            supports_rerank: true,
            supports_structured_output: true,
            max_context_tokens: None,
            supported_quant_methods: vec![QuantMethod::Gguf, QuantMethod::Int8, QuantMethod::Nf4],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Ensure config.json or a .gguf weight file is present in the model directory."
                    .to_string(),
                "For GPU: rebuild with --features metal (macOS) or --features cuda (NVIDIA)."
                    .to_string(),
                "If OOM occurs, try --dtype q4 or a smaller GGUF quantization.".to_string(),
            ],
            construction_guide: "Built-in candle backend. No extra build flags needed for CPU. \
                For GPU: cargo build --features metal (macOS) or --features cuda (NVIDIA)."
                .to_string(),
        }
    }

    fn load(&self, model_path: &Path, device_kind: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        // If model_path is a single GGUF file, use its parent directory.
        let mut gguf_file_path = None;
        let model_path_buf;
        let model_path = if model_path.is_file() {
            if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                gguf_file_path = Some(model_path.to_path_buf());
                model_path_buf = model_path.parent().unwrap_or(Path::new(".")).to_path_buf();
                &model_path_buf
            } else {
                return Err(anyhow!(
                    "model_path is a non-GGUF file: {}.\n\
                     [Diagnostic Tip] Make sure you specify either a GGUF file or a directory containing safetensors.",
                    model_path.display()
                ));
            }
        } else {
            model_path
        };

        let device = match device_kind {
            DeviceKind::Cpu => Device::Cpu,
            DeviceKind::Gpu => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0)
                        .map_err(|e| anyhow!(
                            "failed to initialize CUDA device: {}.\n\
                             [Diagnostic Tip] Verify nvidia-smi works and CUDA drivers are properly installed.",
                            e
                        ))?
                }
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0)
                        .map_err(|e| anyhow!(
                            "failed to initialize Metal device: {}.\n\
                             [Diagnostic Tip] Verify macOS version supports Metal on this hardware.",
                            e
                        ))?
                }
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                {
                    return Err(anyhow!(
                        "GPU (CUDA/Metal) is not supported because bloom-engine was compiled without cuda or metal features.\n\
                         [Diagnostic Tip] Rebuild with --features metal or --features cuda to run on GPU."
                    ));
                }
            }
            other => {
                return Err(anyhow!(
                    "candle engine currently only supports CPU and GPU (CUDA/Metal), got {:?}.\n\
                     [Diagnostic Tip] Use --device cpu or --device gpu.",
                    other
                ))
            }
        };

        // If config.json is missing but a GGUF file exists, synthesise a
        // minimal config from GGUF metadata so that the model can still load.
        let config_path = model_path.join("config.json");
        let (config, gguf_model_type) = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| anyhow!("failed to read config.json: {}", e))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| anyhow!("failed to parse config.json: {}", e))?;
            (config, None)
        } else if let Some(path) = gguf_file_path
            .clone()
            .or_else(|| find_gguf_file(model_path))
        {
            // GGUF-only directory: infer config from GGUF header.
            let (cfg, arch) = synthesize_config_from_gguf(&path)?;
            (cfg, Some(arch))
        } else {
            return Err(anyhow!(
                "no config.json or GGUF file found in {}.\n\
                 [Diagnostic Tip] Make sure the model directory has config.json or a .gguf weight file.",
                model_path.display()
            ));
        };

        if std::env::var("BLOOM_SPECULATIVE")
            .map(|mode| matches!(mode.as_str(), "mtp" | "native-mtp" | "draft-mtp"))
            .unwrap_or(false)
            && config_supports_mtp(&config)
        {
            return Err(anyhow!(
                "speculative=mtp was requested and this model advertises MTP/next-n heads, \
                 but the current Candle backend cannot load native MTP auxiliary heads yet"
            ));
        }

        // Determine model type
        let model_type = ModelType::from_config(&config)
            .or_else(|| gguf_model_type.as_deref().and_then(|arch| match arch {
                "qwen2" | "qwen3" => Some(ModelType::Qwen3),
                "gemma4" | "gemma" => Some(ModelType::Gemma4),
                "llama" => Some(ModelType::Llama),
                _ => None,
            }))
            .ok_or_else(|| {
                anyhow!(
                    "unsupported model type, expected qwen2, qwen3, gemma4_unified or llama.\n\
                     [Diagnostic Tip] Check model_type in config.json or verify the GGUF file architecture."
                )
            })?;

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                anyhow!(
                    "failed to load tokenizer: {}.\n\
                         [Diagnostic Tip] tokenizer.json not found in model directory {}.\n\
                         Please download and place tokenizer.json in this directory.",
                    e,
                    model_path.display()
                )
            })?
        } else if let Some(ref path) = gguf_file_path
            .clone()
            .or_else(|| find_gguf_file(model_path))
        {
            tracing::info!("tokenizer.json not found. Extracting tokenizer from GGUF metadata...");
            synthesize_tokenizer_from_gguf(path)?
        } else {
            return Err(anyhow!(
                "tokenizer.json not found and no GGUF file exists in {} to extract tokenizer",
                model_path.display()
            ));
        };

        let is_quantized = config.get("quantization_config").is_some()
            || model_path
                .to_string_lossy()
                .to_lowercase()
                .contains("quant")
            || model_path.to_string_lossy().to_lowercase().contains("awq")
            || model_path.to_string_lossy().to_lowercase().contains("gptq")
            || model_path.to_string_lossy().to_lowercase().contains("gguf");

        // --- Parse tokenizer_config.json if available ---
        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        if tokenizer_config_path.exists() {
            if let Ok(tc_str) = std::fs::read_to_string(&tokenizer_config_path) {
                if let Ok(tc) = serde_json::from_str::<serde_json::Value>(&tc_str) {
                    // Extract chat_template into config if not already present
                    if let Some(ct) = tc.get("chat_template").and_then(|v| v.as_str()) {
                        tracing::info!(
                            "Found chat_template in tokenizer_config.json ({} chars)",
                            ct.len()
                        );
                    }
                    // Extract bos_token / eos_token overrides
                    if let Some(bos) = tc.get("bos_token") {
                        let bos_str = match bos {
                            serde_json::Value::String(s) => Some(s.clone()),
                            serde_json::Value::Object(o) => {
                                o.get("content").and_then(|c| c.as_str()).map(String::from)
                            }
                            _ => None,
                        };
                        if let Some(ref b) = bos_str {
                            tracing::info!("tokenizer_config.json bos_token: {}", b);
                        }
                    }
                    if let Some(eos) = tc.get("eos_token") {
                        let eos_str = match eos {
                            serde_json::Value::String(s) => Some(s.clone()),
                            serde_json::Value::Object(o) => {
                                o.get("content").and_then(|c| c.as_str()).map(String::from)
                            }
                            _ => None,
                        };
                        if let Some(ref e) = eos_str {
                            tracing::info!("tokenizer_config.json eos_token: {}", e);
                        }
                    }
                }
            }
        }

        // Build unified quantization config
        let _quant_config = QuantizationConfig::from_model_config(&config);

        let mut manifest = crate::manifest_adapter::load_manifest(model_path)?;
        let model_id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let processor_name = format!("{}.tokenizer", model_id);
        if !manifest.processors.iter().any(|p| p.name == processor_name) {
            manifest.processors.push(bloomai_core::ProcessorSpec {
                name: processor_name.clone(),
                kind: bloomai_core::ProcessorKind::TextTokenizer,
                version: "1".to_string(),
                inputs: vec![bloomai_core::Modality::Text],
                outputs: vec![bloomai_core::Modality::Text],
                parameters: std::collections::HashMap::new(),
            });
        }

        let metadata = ModelMetadata {
            id: model_id,
            modality: Modality::Text,
            quantized: is_quantized,
            manifest,
        };

        let model_size = estimate_model_size(model_path);
        let is_uma = cfg!(target_os = "macos");

        let model_holder = Arc::new(Mutex::new(None));
        let model_ptr = Arc::clone(&model_holder);
        let offload_cb = Arc::new(move || {
            let mut guard = model_ptr.lock().unwrap_or_else(|e| e.into_inner());
            *guard = None;
            tracing::info!("Model weights successfully offloaded from GPU via dynamic VRAM Coordinator trigger.");
            Ok(())
        });

        // Register memory footprint with the global VRAM coordinator
        let coordinator = bloomai_core::global_vram_coordinator();
        if let Err(e) = coordinator.request_load(&metadata.id, model_size, is_uma, offload_cb) {
            tracing::error!("VRAM registration failed: {}", e);
        }

        let mut eos_token_ids = Vec::new();
        if let Some(eos_id) = config.get("eos_token_id") {
            if let Some(arr) = eos_id.as_array() {
                for v in arr {
                    if let Some(id) = v.as_u64() {
                        eos_token_ids.push(id as u32);
                    }
                }
            } else if let Some(id) = eos_id.as_u64() {
                eos_token_ids.push(id as u32);
            }
        }
        if eos_token_ids.is_empty() {
            match model_type {
                ModelType::Gemma4 => eos_token_ids = vec![1, 106],
                ModelType::Llama => eos_token_ids = vec![128009, 128001, 2],
                _ => eos_token_ids = vec![151645],
            }
        }

        let mut processors = crate::processor::ProcessorRegistry::default();
        processors.register(Box::new(crate::processor::TokenizerProcessor::new(
            processor_name,
            tokenizer.clone(),
        )));

        let kv_cache_pool = Arc::new(crate::scheduler::BloomKvCachePool::new(16, 512));
        let current_prefix = Mutex::new(Vec::new());

        let vocab_size = tokenizer.get_vocab_size(true);
        let mut vocab_strings = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size {
            let s = tokenizer.decode(&[id as u32], true).unwrap_or_default();
            vocab_strings.push(s);
        }

        let model = CandleTextModel {
            model_path: model_path.to_path_buf(),
            gguf_file_path,
            config: config.clone(),
            model_type,
            model: model_holder, // Keeps weights offloaded initially to preserve budget
            tokenizer,
            vocab_strings,
            device,
            device_kind,
            metadata,
            eos_token_ids,
            model_size,
            processors,
            current_prefix,
            kv_cache_pool,
        };

        // --- Verify loading: execute a small dummy forward pass ---
        match model.verify_loading() {
            Ok(()) => tracing::info!("Model loading verification passed (dummy forward pass OK)"),
            Err(e) => tracing::warn!("Model loading verification failed: {}. Model may still work but results could be incorrect.", e),
        }

        Ok(Box::new(model))
    }
}

fn estimate_model_size(model_path: &Path) -> usize {
    let mut total_size = 0;
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            if let Ok(meta) = std::fs::metadata(entry.path()) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".safetensors")
                    || name_str.ends_with(".bin")
                    || name_str.ends_with(".gguf")
                {
                    total_size += meta.len() as usize;
                }
            }
        }
    }
    if total_size == 0 {
        total_size = 1_200_000_000;
    }
    total_size
}

/// Read a GGUF file's header and synthesise a minimal config.json that can be
/// used by candle-transformers model constructors.
///
/// Returns `(config_json, architecture_string)`.
fn synthesize_config_from_gguf(gguf_path: &Path) -> Result<(serde_json::Value, String)> {
    use candle_core::quantized::gguf_file::Content;

    let mut file = std::fs::File::open(gguf_path)
        .map_err(|e| anyhow!("failed to open GGUF {:?}: {}", gguf_path, e))?;
    let content = Content::read(&mut file).map_err(|e| {
        let err_msg = e.to_string();
        if err_msg.contains("unknown dtype for tensor") {
            let dtype_id = err_msg
                .split_whitespace()
                .last()
                .unwrap_or("");
            let filename = gguf_path.file_name().unwrap_or_default().to_string_lossy();
            let tip = format!(
                "unsupported GGUF quantization type (dtype ID: {}). The Candle engine does not support IQ (importance) quantization or legacy Q4_1/Q5_1 formats. [Diagnostic Tip] Re-quantize to Q4_K_M or Q8_0 using llama-quantize: `llama-quantize {} {} Q4_K_M`",
                dtype_id,
                filename,
                filename.replace("iq", "q4_k_m")
            );
            anyhow!(crate::core::quantization::GgufError::UnsupportedQuantType(tip))
        } else {
            anyhow!("failed to read GGUF header: {}", e)
        }
    })?;

    let md = &content.metadata;

    // Helper: read a metadata value as u64.
    let meta_u64 = |key: &str| -> Option<u64> {
        md.get(key).and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::U8(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::U16(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::U32(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::U64(n) => Some(*n),
            candle_core::quantized::gguf_file::Value::I8(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::I16(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::I32(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::I64(n) => Some(*n as u64),
            _ => None,
        })
    };

    let meta_str = |key: &str| -> Option<String> {
        md.get(key).and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
    };

    let arch = meta_str("general.architecture")
        .unwrap_or_default()
        .to_lowercase();

    // Validate architecture compatibility early
    let arch_supported = matches!(
        arch.as_str(),
        "llama"
            | "qwen"
            | "qwen2"
            | "qwen3"
            | "gemma"
            | "gemma2"
            | "gemma4"
            | "wan"
            | "whisper"
            | "funasr"
            | "tts"
            | "mistral"
            | "deepseek"
    );
    if !arch_supported && !arch.is_empty() {
        return Err(anyhow!(
            crate::core::quantization::GgufError::UnsupportedArch(format!(
                "unsupported GGUF architecture: '{}'. Supported architectures are: llama, qwen, qwen2, qwen3, gemma, mistral, deepseek. [Diagnostic Tip] Verify the GGUF file architecture.",
                arch
            ))
        ));
    }

    let gguf_filename = gguf_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    let mut gguf_type_str = "F32".to_string();
    for (name, info) in &content.tensor_infos {
        if name.contains("weight") && (name.contains("blk") || name.contains("layers")) {
            gguf_type_str = format!("{:?}", info.ggml_dtype);
            break;
        }
    }
    if gguf_type_str == "F32" {
        gguf_type_str = content
            .tensor_infos
            .values()
            .next()
            .map(|info| format!("{:?}", info.ggml_dtype))
            .unwrap_or_else(|| "F32".to_string());
    }

    let quant_lower = gguf_type_str.to_lowercase();
    if quant_lower.starts_with("iq") || quant_lower == "q4_1" || quant_lower == "q5_1" {
        return Err(anyhow!(
            crate::core::quantization::GgufError::UnsupportedQuantType(format!(
                "unsupported GGUF quantization type: '{}'. The Candle engine does not support IQ (importance) quantization or legacy Q4_1/Q5_1 formats. [Diagnostic Tip] Re-quantize to Q4_K_M or Q8_0 using llama-quantize: `llama-quantize {} {} Q4_K_M`",
                gguf_type_str,
                gguf_filename,
                gguf_filename.replace(&gguf_type_str, "Q4_K_M")
            ))
        ));
    }

    let prefix = if arch.is_empty() {
        "general".to_string()
    } else {
        arch.clone()
    };

    let block_count = meta_u64(&format!("{}.block_count", prefix)).unwrap_or(28);
    let context_length = meta_u64(&format!("{}.context_length", prefix))
        .or_else(|| meta_u64("general.context_length"))
        .unwrap_or(2048);
    let embedding_length = meta_u64(&format!("{}.embedding_length", prefix)).unwrap_or(4096);
    let head_count = meta_u64(&format!("{}.attention.head_count", prefix)).unwrap_or(32);
    let head_count_kv =
        meta_u64(&format!("{}.attention.head_count_kv", prefix)).unwrap_or(head_count);
    let feed_forward_length =
        meta_u64(&format!("{}.feed_forward_length", prefix)).unwrap_or(embedding_length * 4);
    let max_position_embeddings =
        meta_u64(&format!("{}.context_length", prefix)).unwrap_or(context_length);
    let rms_norm_eps = md
        .get(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::F32(f) => Some(*f as f64),
            candle_core::quantized::gguf_file::Value::F64(f) => Some(*f),
            _ => None,
        })
        .unwrap_or(1e-5);
    let rope_freq_base = md
        .get(&format!("{}.rope.freq_base", prefix))
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::F32(f) => Some(*f as f64),
            candle_core::quantized::gguf_file::Value::F64(f) => Some(*f),
            _ => None,
        })
        .unwrap_or(10000.0);

    let head_dim = embedding_length / head_count;

    // Map GGUF architecture to HF model_type used by candle-transformers.
    let model_type = match arch.as_str() {
        a if a.contains("qwen") => {
            if meta_u64(&format!("{}.attention.sliding_window", prefix)).unwrap_or(0) > 0 {
                "qwen3"
            } else {
                "qwen2"
            }
        }
        a if a.contains("gemma") => "gemma4_unified",
        a if a.contains("llama") || a.contains("mistral") || a.contains("deepseek") => "llama",
        _ => "unknown",
    };

    let nextn_predict_layers = meta_u64(&format!("{}.nextn_predict_layers", prefix))
        .or_else(|| meta_u64(&format!("{}.next_n_predict_layers", prefix)))
        .or_else(|| meta_u64("general.nextn_predict_layers"))
        .or_else(|| meta_u64("general.next_n_predict_layers"));

    let mut config = serde_json::json!({
        "model_type": model_type,
        "hidden_size": embedding_length,
        "intermediate_size": feed_forward_length,
        "num_hidden_layers": block_count,
        "num_attention_heads": head_count,
        "num_key_value_heads": head_count_kv,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_freq_base,
        "vocab_size": md.get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u64),
                _ => None,
            })
            .unwrap_or(151936),
        "attention_bias": true,
        "max_window_layers": block_count,
        "use_sliding_window": false,
        "tie_word_embeddings": false,
        "torch_dtype": "float16",
    });
    if let Some(nextn) = nextn_predict_layers {
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "num_nextn_predict_layers".to_string(),
                serde_json::Value::from(nextn),
            );
        }
    }

    Ok((config, arch))
}

fn synthesize_tokenizer_from_gguf(gguf_path: &Path) -> Result<tokenizers::Tokenizer> {
    use candle_core::quantized::gguf_file::Content;
    use std::str::FromStr;

    let mut file = std::fs::File::open(gguf_path)
        .map_err(|e| anyhow!("failed to open GGUF {:?}: {}", gguf_path, e))?;
    let content = Content::read(&mut file)
        .map_err(|e| anyhow!("failed to read GGUF header for tokenizer: {}", e))?;

    let md = &content.metadata;

    let tokens = md
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens in GGUF"))?;

    let model_type = md
        .get("tokenizer.ggml.model")
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "gpt2".to_string());

    let tokens_list = match tokens {
        candle_core::quantized::gguf_file::Value::Array(arr) => arr,
        _ => return Err(anyhow!("tokenizer.ggml.tokens is not an array")),
    };

    let mut vocab = serde_json::Map::new();
    for (i, val) in tokens_list.iter().enumerate() {
        if let candle_core::quantized::gguf_file::Value::String(s) = val {
            vocab.insert(s.clone(), serde_json::json!(i));
        }
    }

    // Extract merges if BPE
    let mut merges = Vec::new();
    if let Some(candle_core::quantized::gguf_file::Value::Array(arr)) =
        md.get("tokenizer.ggml.merges")
    {
        for val in arr {
            if let candle_core::quantized::gguf_file::Value::String(s) = val {
                merges.push(s.clone());
            }
        }
    }

    // Extract bos/eos special token IDs
    let bos_id = md
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::U32(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::U64(n) => Some(*n),
            candle_core::quantized::gguf_file::Value::I32(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::I64(n) => Some(*n as u64),
            _ => None,
        })
        .unwrap_or(1);
    let eos_id = md
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::U32(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::U64(n) => Some(*n),
            candle_core::quantized::gguf_file::Value::I32(n) => Some(*n as u64),
            candle_core::quantized::gguf_file::Value::I64(n) => Some(*n as u64),
            _ => None,
        })
        .unwrap_or(2);

    let get_token_str = |idx: usize| -> Option<String> {
        tokens_list.get(idx).and_then(|val| match val {
            candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
    };
    let bos_token = get_token_str(bos_id as usize).unwrap_or_else(|| "<s>".to_string());
    let eos_token = get_token_str(eos_id as usize).unwrap_or_else(|| "</s>".to_string());

    let tokenizer_json = if !merges.is_empty() || model_type == "gpt2" || model_type == "bpe" {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                { "id": bos_id, "special": true, "content": bos_token },
                { "id": eos_id, "special": true, "content": eos_token }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": false
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": vocab,
                "merges": merges
            }
        })
    } else {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                { "id": bos_id, "special": true, "content": bos_token },
                { "id": eos_id, "special": true, "content": eos_token }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "WhitespaceSplit"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": []
            }
        })
    };

    let tokenizer_str = serde_json::to_string(&tokenizer_json)?;
    let tokenizer = tokenizers::Tokenizer::from_str(&tokenizer_str)
        .map_err(|e| anyhow!("failed to parse synthesized tokenizer: {}", e))?;
    Ok(tokenizer)
}

/// Wrapper enum for different model variants
/// Wrapper enum for different model variants
pub enum QwenModelWrapper {
    Qwen2(candle_transformers::models::qwen2::ModelForCausalLM),
    Qwen3(candle_transformers::models::qwen3::ModelForCausalLM),
    Gemma4(crate::executor::gemma4::Model),
    Llama(
        candle_transformers::models::llama::Llama,
        candle_transformers::models::llama::Cache,
    ),
    Streaming(crate::executor::qwen_streaming::QwenStreamingModelForCausalLM),
    Gemma4Streaming(crate::executor::gemma4_streaming::Gemma4StreamingModel),
    QuantizedLlama(candle_transformers::models::quantized_llama::ModelWeights),
    QuantizedQwen2(candle_transformers::models::quantized_qwen2::ModelWeights),
}

impl QwenModelWrapper {
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        match self {
            QwenModelWrapper::Qwen2(m) => m.forward(input_ids, start_pos).map_err(Into::into),
            QwenModelWrapper::Qwen3(m) => m.forward(input_ids, start_pos).map_err(Into::into),
            QwenModelWrapper::Gemma4(m) => m.forward(input_ids, start_pos).map_err(Into::into),
            QwenModelWrapper::Llama(m, cache) => {
                m.forward(input_ids, start_pos, cache).map_err(Into::into)
            }
            QwenModelWrapper::Streaming(m) => m.forward(input_ids, start_pos).map_err(Into::into),
            QwenModelWrapper::Gemma4Streaming(m) => {
                m.forward(input_ids, start_pos).map_err(Into::into)
            }
            QwenModelWrapper::QuantizedLlama(m) => {
                m.forward(input_ids, start_pos).map_err(Into::into)
            }
            QwenModelWrapper::QuantizedQwen2(m) => {
                m.forward(input_ids, start_pos).map_err(Into::into)
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            QwenModelWrapper::Qwen2(_)
            | QwenModelWrapper::Qwen3(_)
            | QwenModelWrapper::Llama(_, _)
            | QwenModelWrapper::QuantizedLlama(_)
            | QwenModelWrapper::QuantizedQwen2(_) => {
                // Recreated via reload if start_pos == 0
            }
            QwenModelWrapper::Gemma4(m) => m.clear_kv_cache(),
            QwenModelWrapper::Streaming(m) => m.clear_kv_cache(),
            QwenModelWrapper::Gemma4Streaming(_) => {}
        }
    }

    /// Return the variant name for diagnostic error messages.
    pub fn variant_name(&self) -> &'static str {
        match self {
            QwenModelWrapper::Qwen2(_) => "Qwen2",
            QwenModelWrapper::Qwen3(_) => "Qwen3",
            QwenModelWrapper::Gemma4(_) => "Gemma4",
            QwenModelWrapper::Llama(_, _) => "Llama",
            QwenModelWrapper::Streaming(_) => "Streaming",
            QwenModelWrapper::Gemma4Streaming(_) => "Gemma4Streaming",
            QwenModelWrapper::QuantizedLlama(_) => "QuantizedLlama",
            QwenModelWrapper::QuantizedQwen2(_) => "QuantizedQwen2",
        }
    }
}

/// Handle-keyed [`KvHook`] for the production IFB path.
///
/// Unlike [`crate::executor::qwen_streaming::QwenKvHook`], which is bound to
/// a single shared model instance, this hook dispatches by `handle` into the
/// per-request model map maintained by `bloom_server`'s `forward_fn`. Each
/// handle owns its own `QwenModelWrapper`, and this hook reads/writes the KV
/// cache of whichever wrapper the handle resolves to.
///
/// Currently supports only the `QwenModelWrapper::Streaming` variant, whose
/// `StreamingQwenAttention` uses a `ConcatKvCache` that can be sliced and
/// overwritten. Other variants return an error — the server should only
/// attach this hook when the loaded model is a streaming variant.
pub struct PerRequestKvHook {
    /// Shared per-handle model map (`handle -> Box<QwenModelWrapper>`).
    request_models:
        Arc<Mutex<std::collections::HashMap<usize, Box<dyn std::any::Any + Send + Sync>>>>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
}

impl PerRequestKvHook {
    /// Create a hook that dispatches into the per-request model map.
    ///
    /// `num_layers`, `num_kv_heads` and `head_dim` must match the model's
    /// config; `kv_dim = num_kv_heads * head_dim` is derived and asserted
    /// against the paged cache's config at extract time.
    pub fn new(
        request_models: Arc<
            Mutex<std::collections::HashMap<usize, Box<dyn std::any::Any + Send + Sync>>>,
        >,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            request_models,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim: num_kv_heads * head_dim,
        }
    }
}

impl crate::scheduler::kv_hook::KvHook for PerRequestKvHook {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn extract_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
        let Some(wrapper) = models.get(&handle) else {
            return Err(anyhow!(
                "PerRequestKvHook: handle {} not found in request_models map",
                handle
            ));
        };
        let Some(qw) = wrapper.downcast_ref::<QwenModelWrapper>() else {
            return Err(anyhow!(
                "PerRequestKvHook: model for handle {} is not a QwenModelWrapper",
                handle
            ));
        };
        match qw {
            QwenModelWrapper::Streaming(m) => {
                let (k, v) = m.extract_kv_from_layer(layer_idx, start_pos, seq_len)?;
                if k.len() != seq_len * self.kv_dim {
                    return Err(anyhow!(
                        "PerRequestKvHook: extracted k len {} != seq_len {} * kv_dim {}",
                        k.len(),
                        seq_len,
                        self.kv_dim
                    ));
                }
                Ok((k, v))
            }
            _ => Err(anyhow!(
                "PerRequestKvHook: model variant {:?} does not support KV hook extraction (only Streaming is supported)",
                qw.variant_name()
            )),
        }
    }

    fn inject_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> Result<()> {
        let mut models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
        let Some(wrapper) = models.get_mut(&handle) else {
            return Err(anyhow!(
                "PerRequestKvHook: handle {} not found in request_models map",
                handle
            ));
        };
        let Some(qw) = wrapper.downcast_mut::<QwenModelWrapper>() else {
            return Err(anyhow!(
                "PerRequestKvHook: model for handle {} is not a QwenModelWrapper",
                handle
            ));
        };
        match qw {
            QwenModelWrapper::Streaming(m) => m.inject_kv_to_layer(
                layer_idx,
                start_pos,
                keys,
                values,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
            ),
            _ => Err(anyhow!(
                "PerRequestKvHook: model variant {:?} does not support KV hook injection (only Streaming is supported)",
                qw.variant_name()
            )),
        }
    }
}

struct CandleTextModel {
    model_path: PathBuf,
    gguf_file_path: Option<PathBuf>,
    config: serde_json::Value,
    model_type: ModelType,
    model: Arc<Mutex<Option<QwenModelWrapper>>>,
    tokenizer: tokenizers::Tokenizer,
    vocab_strings: Vec<String>,
    device: Device,
    device_kind: DeviceKind,
    metadata: ModelMetadata,
    eos_token_ids: Vec<u32>,
    model_size: usize,
    processors: crate::processor::ProcessorRegistry,
    current_prefix: Mutex<Vec<u32>>,
    kv_cache_pool: Arc<crate::scheduler::BloomKvCachePool>,
}

impl CandleTextModel {
    fn reload_for_generation(&self) -> Result<QwenModelWrapper> {
        match self.reload() {
            Ok(m) => Ok(m),
            Err(e) => {
                let err_str = e.to_string().to_lowercase();
                if err_str.contains("oom")
                    || err_str.contains("out of memory")
                    || err_str.contains("metal")
                    || err_str.contains("cuda")
                {
                    tracing::warn!(
                        "GPU allocation failed (possibly OOM): {}. Falling back to CPU...",
                        e
                    );
                    self.reload_cpu()
                } else {
                    Err(e)
                }
            }
        }
    }

    fn reload(&self) -> Result<QwenModelWrapper> {
        self.reload_on_device(&self.device)
    }

    fn reload_cpu(&self) -> Result<QwenModelWrapper> {
        self.reload_on_device(&Device::Cpu)
    }

    /// Whether `reload_on_device` would pick the `QwenModelWrapper::Streaming`
    /// variant for the current model + device combination.
    ///
    /// Mirrors the streaming-variant selection logic in `reload_on_device`:
    /// the streaming path is only chosen when the model is large (>3.5 GB)
    /// AND the device is CUDA AND the model type is Qwen2/Qwen3 (Gemma4 on
    /// CUDA takes the `Gemma4Streaming` path instead, which does not yet
    /// expose a `KvHook`).
    fn uses_streaming_variant(&self) -> bool {
        let is_streaming =
            self.model_size > 3_500_000_000 && matches!(self.device, Device::Cuda(_));
        is_streaming && matches!(self.model_type, ModelType::Qwen2 | ModelType::Qwen3)
    }

    fn reload_on_device(&self, device: &Device) -> Result<QwenModelWrapper> {
        let is_streaming = self.model_size > 3_500_000_000 && matches!(device, Device::Cuda(_));
        let offloaded_layers = std::env::var("BLOOM_GPU_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let quantizer = if let Device::Metal(_) = device {
            crate::executor::metal_quant::MetalQuantizer::new(device)
                .ok()
                .map(Arc::new)
        } else {
            None
        };

        let gguf_path = self
            .gguf_file_path
            .clone()
            .or_else(|| find_gguf_file(&self.model_path));

        if let Some(ref path) = gguf_path {
            crate::core::memory::prefetch_file_madvise(path);
        }

        if is_streaming && self.model_type == ModelType::Gemma4 {
            tracing::info!("Initializing memory-efficient layer-wise weight streaming on CUDA for Gemma-4 GGUF...");
            let gguf_path = gguf_path.ok_or_else(|| {
                anyhow!(
                    "no GGUF file found for streaming Gemma-4 in {:?}",
                    self.model_path
                )
            })?;

            let text_config_val = self
                .config
                .get("text_config")
                .ok_or_else(|| anyhow!("missing text_config in gemma4_unified config"))?;
            let gemma4_cfg = crate::executor::gemma4::Config::from_value(text_config_val)?;

            let m = crate::executor::gemma4_streaming::Gemma4StreamingModel::new_with_offload(
                &gguf_path,
                &gemma4_cfg,
                device,
                offloaded_layers,
            )?;
            return Ok(QwenModelWrapper::Gemma4Streaming(m));
        }

        let safetensors_files = find_safetensors_files(&self.model_path);
        if safetensors_files.is_empty() {
            if let Some(path) = gguf_path {
                match self.model_type {
                    ModelType::Gemma4 => {
                        tracing::info!("GGUF-only Gemma-4 directory, using streaming path...");
                        let text_config_val = self
                            .config
                            .get("text_config")
                            .ok_or_else(|| anyhow!("missing text_config in gemma4 config"))?;
                        let gemma4_cfg =
                            crate::executor::gemma4::Config::from_value(text_config_val)?;
                        let m = crate::executor::gemma4_streaming::Gemma4StreamingModel::new_with_offload(
                            &path,
                            &gemma4_cfg,
                            device,
                            offloaded_layers,
                        )?;
                        return Ok(QwenModelWrapper::Gemma4Streaming(m));
                    }
                    ModelType::Llama => {
                        tracing::info!("GGUF Llama model loading quantized weights...");
                        let mut file = std::fs::File::open(&path)
                            .map_err(|e| anyhow!("failed to open GGUF file: {}", e))?;
                        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                            .map_err(|e| anyhow!("failed to read GGUF content: {}", e))?;
                        let m =
                            candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                                content, &mut file, device,
                            )?;
                        return Ok(QwenModelWrapper::QuantizedLlama(m));
                    }
                    ModelType::Qwen2 | ModelType::Qwen3 => {
                        tracing::info!("GGUF Qwen model loading quantized weights...");
                        let mut file = std::fs::File::open(&path)
                            .map_err(|e| anyhow!("failed to open GGUF file: {}", e))?;
                        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                            .map_err(|e| anyhow!("failed to read GGUF content: {}", e))?;
                        let m =
                            candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
                                content, &mut file, device,
                            )?;
                        return Ok(QwenModelWrapper::QuantizedQwen2(m));
                    }
                }
            }
            return Err(anyhow!(
                "no safetensors or GGUF files found in {}",
                self.model_path.display()
            ));
        }

        let mut dtype = match device {
            Device::Cpu => DType::F32,
            Device::Cuda(_) => DType::F16,
            Device::Metal(_) => DType::F16,
        };

        if let Ok(dt_str) = std::env::var("BLOOM_DTYPE") {
            dtype = match dt_str.to_lowercase().as_str() {
                "f32" | "float32" => DType::F32,
                "f16" | "float16" => DType::F16,
                "bf16" | "bfloat16" => DType::BF16,
                _ => dtype,
            };
        }

        let vb_device = if is_streaming { &Device::Cpu } else { device };

        let dtype_str = match dtype {
            DType::F16 => "float16 (16-bit GPU mode)",
            DType::BF16 => "bfloat16 (16-bit CPU mode)",
            DType::F32 => "float32 (32-bit mode)",
            _ => "other",
        };
        let estimated_size_gb = self.model_size as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!("============================================================");
        eprintln!("Memory/GPU Pre-load Checks:");
        eprintln!("- Target Backend Device: {:?}", device);
        eprintln!("- Loading Precision: {}", dtype_str);
        eprintln!(
            "- Estimated Model Memory Occupancy: {:.2} GB",
            estimated_size_gb
        );
        eprintln!("============================================================");
        let _ = std::io::Write::flush(&mut std::io::stderr());

        for f in &safetensors_files {
            crate::core::memory::prefetch_file_madvise(f);
        }

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, vb_device)
                .map_err(|e| anyhow!("failed to load safetensors: {}", e))?
        };

        let model = if is_streaming {
            tracing::info!("Initializing memory-efficient layer-wise weight streaming on CUDA...");
            let mut cfg_val = self.config.clone();
            let obj = cfg_val
                .as_object_mut()
                .ok_or_else(|| anyhow!("config is not a JSON object"))?;

            let hidden_size = obj
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024);
            let num_heads = obj
                .get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(16);
            let num_layers = obj
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(28);

            if !obj.contains_key("head_dim") {
                obj.insert(
                    "head_dim".to_string(),
                    serde_json::Value::from(hidden_size / num_heads),
                );
            }
            if !obj.contains_key("attention_bias") {
                obj.insert("attention_bias".to_string(), serde_json::Value::from(true));
            }
            if !obj.contains_key("max_window_layers") {
                obj.insert(
                    "max_window_layers".to_string(),
                    serde_json::Value::from(num_layers),
                );
            }
            if !obj.contains_key("use_sliding_window") {
                obj.insert(
                    "use_sliding_window".to_string(),
                    serde_json::Value::from(false),
                );
            }
            if !obj.contains_key("tie_word_embeddings") {
                obj.insert(
                    "tie_word_embeddings".to_string(),
                    serde_json::Value::from(false),
                );
            }

            let cfg: candle_transformers::models::qwen3::Config =
                serde_json::from_value(cfg_val)
                    .map_err(|e| anyhow!("failed to parse qwen config: {}", e))?;
            let m =
                crate::executor::qwen_streaming::QwenStreamingModelForCausalLM::new_with_offload(
                    &cfg,
                    vb,
                    device,
                    quantizer.clone(),
                    offloaded_layers,
                )
                .map_err(|e| anyhow!("failed to build streaming model: {}", e))?;
            QwenModelWrapper::Streaming(m)
        } else {
            match self.model_type {
                ModelType::Qwen2 => {
                    let cfg: candle_transformers::models::qwen2::Config =
                        serde_json::from_value(self.config.clone())
                            .map_err(|e| anyhow!("failed to parse qwen2 config: {}", e))?;
                    let m = candle_transformers::models::qwen2::ModelForCausalLM::new(&cfg, vb)
                        .map_err(|e| anyhow!("failed to build qwen2 model: {}", e))?;
                    QwenModelWrapper::Qwen2(m)
                }
                ModelType::Qwen3 => {
                    let cfg: candle_transformers::models::qwen3::Config =
                        serde_json::from_value(self.config.clone())
                            .map_err(|e| anyhow!("failed to parse qwen3 config: {}", e))?;
                    let m = candle_transformers::models::qwen3::ModelForCausalLM::new(&cfg, vb)
                        .map_err(|e| anyhow!("failed to build qwen3 model: {}", e))?;
                    QwenModelWrapper::Qwen3(m)
                }
                ModelType::Gemma4 => {
                    let text_config_val = self
                        .config
                        .get("text_config")
                        .ok_or_else(|| anyhow!("missing text_config in gemma4_unified config"))?;
                    let gemma4_cfg = crate::executor::gemma4::Config::from_value(text_config_val)?;

                    let m = crate::executor::gemma4::Model::new_with_quantizer(
                        false,
                        &gemma4_cfg,
                        vb,
                        quantizer.clone(),
                    )
                    .map_err(|e| anyhow!("failed to build gemma4 model: {}", e))?;

                    QwenModelWrapper::Gemma4(m)
                }
                ModelType::Llama => {
                    let hidden_size = self
                        .config
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4096) as usize;
                    let intermediate_size = self
                        .config
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(11008) as usize;
                    let vocab_size = self
                        .config
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32000) as usize;
                    let num_hidden_layers = self
                        .config
                        .get("num_hidden_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize;
                    let num_attention_heads = self
                        .config
                        .get("num_attention_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize;
                    let num_key_value_heads = self
                        .config
                        .get("num_key_value_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32) as usize;
                    let rms_norm_eps = self
                        .config
                        .get("rms_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-5);
                    let rope_theta = self
                        .config
                        .get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(10000.0) as f32;
                    let max_position_embeddings = self
                        .config
                        .get("max_position_embeddings")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(2048) as usize;
                    let tie_word_embeddings = self
                        .config
                        .get("tie_word_embeddings")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    let cfg = candle_transformers::models::llama::Config {
                        hidden_size,
                        intermediate_size,
                        vocab_size,
                        num_hidden_layers,
                        num_attention_heads,
                        num_key_value_heads,
                        use_flash_attn: false,
                        rms_norm_eps,
                        rope_theta,
                        bos_token_id: None,
                        eos_token_id: None,
                        rope_scaling: None,
                        max_position_embeddings,
                        tie_word_embeddings,
                    };
                    let m = candle_transformers::models::llama::Llama::load(vb, &cfg)
                        .map_err(|e| anyhow!("failed to build llama model: {}", e))?;
                    let cache =
                        candle_transformers::models::llama::Cache::new(true, dtype, &cfg, device)
                            .map_err(|e| anyhow!("failed to create llama cache: {}", e))?;
                    QwenModelWrapper::Llama(m, cache)
                }
            }
        };

        Ok(model)
    }

    /// Verify that the loaded model can execute a dummy forward pass.
    /// This catches issues like corrupted weights, incompatible shapes, etc.
    /// The method loads the model if not already loaded and runs a small forward pass
    /// with input `[1, 2, 3]`.
    fn verify_loading(&self) -> Result<()> {
        let mut model_guard = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if model_guard.is_none() {
            let reloaded = match self.reload() {
                Ok(m) => m,
                Err(e) => {
                    let err_str = e.to_string().to_lowercase();
                    if err_str.contains("oom") || err_str.contains("out of memory") {
                        tracing::warn!("verify_loading: GPU load failed with OOM, trying CPU...");
                        self.reload_cpu()?
                    } else {
                        return Err(e);
                    }
                }
            };
            *model_guard = Some(reloaded);
        }
        let model = model_guard.as_mut().unwrap();

        // Dummy forward pass with tiny input
        let dummy_ids = vec![1u32, 2, 3];
        let input_ids = Tensor::new(dummy_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| anyhow!("verify_loading: failed to create dummy tensor: {}", e))?;
        let _output = model
            .forward(&input_ids, 0)
            .map_err(|e| anyhow!("verify_loading: dummy forward pass failed: {}", e))?;

        tracing::debug!("verify_loading: dummy forward pass succeeded");
        Ok(())
    }

    fn logits_for_last_position(logits: Tensor) -> Result<Tensor> {
        let logits = logits.squeeze(0)?;
        if logits.rank() >= 2 {
            Ok(logits.get(logits.dim(0)? - 1)?)
        } else {
            Ok(logits)
        }
    }

    fn logits_sequence(logits: Tensor) -> Result<Tensor> {
        let logits = logits.squeeze(0)?;
        if logits.rank() == 1 {
            Ok(logits.unsqueeze(0)?)
        } else {
            Ok(logits)
        }
    }

    fn greedy_token(logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let logits_vec = logits.to_vec1::<f32>()?;
        logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| anyhow!("empty logits tensor"))
    }

    /// Convert a `[seq_len, vocab_size]` (or `[vocab_size]`) logits tensor into
    /// a `Vec<Vec<f32>>` for rejection sampling.
    ///
    /// `logits_sequence` already squeezes the batch dim, so the input here is
    /// either rank-2 (`[seq, vocab]`) or rank-1 (`[vocab]` for a single position).
    fn verifier_logits_to_vecs(logits: &Tensor) -> Result<Vec<Vec<f32>>> {
        let logits = logits.to_dtype(DType::F32)?;
        if logits.rank() <= 1 {
            Ok(vec![logits.to_vec1::<f32>()?])
        } else {
            Ok(logits.to_vec2::<f32>()?)
        }
    }

    fn decode_and_emit_delta(
        &self,
        all_tokens: &[u32],
        prompt_len: usize,
        prev_text: &mut String,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        let text = self
            .tokenizer
            .decode(&all_tokens[prompt_len..], true)
            .map_err(|e| anyhow!("tokenizer decode error: {}", e))?;
        if text.len() > prev_text.len() {
            let new_text = &text[prev_text.len()..];
            sink.on_chunk(crate::io::OutputChunk::TextDelta(new_text.to_string()))?;
            *prev_text = text;
        }
        Ok(())
    }

    fn replay_generation_prefix(
        &self,
        model_guard: &mut Option<QwenModelWrapper>,
        prefix_tokens: &[u32],
    ) -> Result<usize> {
        *model_guard = Some(self.reload_for_generation()?);
        if prefix_tokens.is_empty() {
            return Ok(0);
        }

        let input_ids = Tensor::new(prefix_tokens, &self.device)?.unsqueeze(0)?;
        let _ = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("model failed to reload"))?
            .forward(&input_ids, 0)?;
        Ok(prefix_tokens.len())
    }
}

impl LoadedModel for CandleTextModel {
    fn forward(
        &self,
        input_ids: &candle_core::Tensor,
        start_pos: usize,
    ) -> Result<candle_core::Tensor> {
        let mut model_guard = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if model_guard.is_none() {
            let reloaded = self.reload()?;
            *model_guard = Some(reloaded);
        }
        let model = model_guard.as_mut().unwrap();
        model.forward(input_ids, start_pos)
    }

    fn create_wrapper(&self) -> Result<Box<dyn std::any::Any + Send + Sync>> {
        let wrapper = self.reload()?;
        Ok(Box::new(wrapper))
    }

    fn clear_kv_cache(&self) {
        let mut model_guard = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let should_drop = if let Some(wrapper) = model_guard.as_mut() {
            match wrapper {
                QwenModelWrapper::Streaming(m) => {
                    m.clear_kv_cache();
                    false
                }
                QwenModelWrapper::Gemma4Streaming(m) => {
                    m.clear_kv_cache();
                    false
                }
                _ => true,
            }
        } else {
            false
        };
        if should_drop {
            *model_guard = None;
        }
    }

    fn supports_paged_kv(&self) -> bool {
        self.uses_streaming_variant()
    }

    fn vocab_strings(&self) -> &[String] {
        &self.vocab_strings
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn processors(&self) -> Option<&crate::processor::ProcessorRegistry> {
        Some(&self.processors)
    }

    #[cfg(feature = "candle-engine")]
    fn tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        Some(&self.tokenizer)
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let mut text_parts = Vec::new();
        self.infer_stream(input, params, &mut |chunk: crate::io::OutputChunk| {
            if let crate::io::OutputChunk::TextDelta(delta) = chunk {
                text_parts.push(delta);
            }
            Ok(())
        })?;
        let text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };
        Ok(ModelOutput {
            text,
            logits: None,
            image: None,
            audio: None,
            video: None,
        })
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        let prompt = match input {
            ModelInput::Text { prompt } => prompt,
            _ => return Err(anyhow!("CandleTextModel only supports text input")),
        };

        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("tokenizer encode error: {}", e))?;
        let token_ids = encoding.get_ids().to_vec();
        self.infer_tokens_stream(token_ids, params, sink)
    }

    fn infer_request(
        &self,
        request: crate::io::InferenceRequest,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        let mut tokens_opt = None;
        for block in &request.blocks {
            if let crate::io::DataBlock::Tokens(ids) = block {
                tokens_opt = Some(ids.clone());
                break;
            }
        }

        let token_ids = if let Some(ids) = tokens_opt {
            ids
        } else {
            let mut blocks = request.blocks;
            for spec in &self.metadata().manifest.processors {
                if let Ok(proc) = self.processors.get(&spec.name) {
                    blocks = proc.process(blocks)?;
                }
            }
            let mut resolved = None;
            for block in &blocks {
                if let crate::io::DataBlock::Tokens(ids) = block {
                    resolved = Some(ids.clone());
                    break;
                }
            }
            resolved
                .ok_or_else(|| anyhow!("Failed to process input into Tokens for CandleTextModel"))?
        };

        let gen_params = GenerationParams {
            max_tokens: request.params.max_tokens,
            temperature: request.params.temperature,
            top_p: request.params.top_p,
            seed: request.params.seed,
            response_format: request.params.response_format.clone(),
        };

        self.infer_tokens_stream(token_ids, &gen_params, sink)
    }
}

impl CandleTextModel {
    fn infer_tokens_stream(
        &self,
        token_ids: Vec<u32>,
        params: &GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        let mut total_draft_tokens = 0usize;
        let mut total_accepted_tokens = 0usize;

        tracing::debug!("Prompt token IDs: {:?}", token_ids);

        // Check if this is an embedding model
        let is_embedding = self.metadata.id.to_lowercase().contains("bert")
            || self.metadata.id.to_lowercase().contains("embed")
            || self.metadata.id.to_lowercase().contains("bge")
            || self.metadata.id.to_lowercase().contains("rerank");

        let mut start_pos = 0;
        let mut current_prefix_guard = self.current_prefix.lock().unwrap_or_else(|e| e.into_inner());
        if !is_embedding {
            let mut matched_tokens = 0;
            for (a, b) in token_ids.iter().zip(current_prefix_guard.iter()) {
                if a == b {
                    matched_tokens += 1;
                } else {
                    break;
                }
            }

            if matched_tokens > 0 && matched_tokens < token_ids.len() {
                start_pos = matched_tokens;
                tracing::info!(
                    "Prompt cache hit! Reusing {} prefix tokens. start_pos = {}",
                    matched_tokens,
                    start_pos
                );
                let mut pool_state = self.kv_cache_pool.state.lock().unwrap_or_else(|e| e.into_inner());
                pool_state.metrics.hits += 1;
                pool_state.metrics.reuses += matched_tokens;
            } else {
                start_pos = 0;
                let mut pool_state = self.kv_cache_pool.state.lock().unwrap_or_else(|e| e.into_inner());
                pool_state.metrics.misses += 1;
            }
        }

        let mut model_guard = self.model.lock().unwrap_or_else(|e| e.into_inner());
        if start_pos == 0 && !is_embedding {
            let needs_reload = match model_guard.as_ref() {
                None => true,
                Some(wrapper) => matches!(
                    wrapper,
                    QwenModelWrapper::Qwen2(_)
                        | QwenModelWrapper::Qwen3(_)
                        | QwenModelWrapper::Llama(_, _)
                        | QwenModelWrapper::QuantizedLlama(_)
                        | QwenModelWrapper::QuantizedQwen2(_)
                ),
            };

            if needs_reload {
                tracing::info!("Resetting KV cache: recreating model wrapper...");
                let reloaded = match self.reload() {
                    Ok(m) => m,
                    Err(e) => {
                        let err_str = e.to_string().to_lowercase();
                        if err_str.contains("oom")
                            || err_str.contains("out of memory")
                            || err_str.contains("metal")
                            || err_str.contains("cuda")
                        {
                            let _span =
                                tracing::info_span!("model.fallback", reason = %e).entered();
                            tracing::warn!(
                                "GPU allocation failed (possibly OOM): {}. Falling back to CPU...",
                                e
                            );
                            self.reload_cpu()?
                        } else {
                            return Err(e);
                        }
                    }
                };
                *model_guard = Some(reloaded);
            } else if let Some(wrapper) = model_guard.as_mut() {
                tracing::info!("Resetting KV cache in-place...");
                wrapper.clear_kv_cache();
            }
        } else if model_guard.is_none() {
            tracing::info!("Reloading model weights onto GPU/CPU via fast UMA/mmap path...");
            let reloaded = match self.reload() {
                Ok(m) => m,
                Err(e) => {
                    let err_str = e.to_string().to_lowercase();
                    if err_str.contains("oom")
                        || err_str.contains("out of memory")
                        || err_str.contains("metal")
                        || err_str.contains("cuda")
                    {
                        let _span = tracing::info_span!("model.fallback", reason = %e).entered();
                        tracing::warn!(
                            "GPU allocation failed (possibly OOM): {}. Falling back to CPU...",
                            e
                        );
                        self.reload_cpu()?
                    } else {
                        return Err(e);
                    }
                }
            };
            *model_guard = Some(reloaded);
        }
        if is_embedding {
            let input_ids = Tensor::new(token_ids.clone(), &self.device)?.unsqueeze(0)?;
            let logits = model_guard
                .as_mut()
                .ok_or_else(|| anyhow!("model is not loaded"))?
                .forward(&input_ids, 0)?;
            let mean = logits.mean(1)?;
            let mean_vec = mean.squeeze(0)?.to_vec1::<f32>()?;
            sink.on_chunk(crate::io::OutputChunk::Embedding(mean_vec))?;
            sink.on_chunk(crate::io::OutputChunk::End)?;
            return Ok(());
        }

        let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(
            params.seed.unwrap_or(42),
            Some(params.temperature),
            Some(params.top_p),
        );

        *current_prefix_guard = token_ids.clone();

        let mut all_tokens = token_ids.clone();
        let mut prev_text = String::new();
        let mut next_token = 0;
        let speculative_mode = SpeculativeMode::from_env()?;
        if matches!(speculative_mode, SpeculativeMode::Mtp { .. })
            && !config_supports_mtp(&self.config)
        {
            return Err(anyhow!(
                "speculative=mtp was requested, but model config does not advertise MTP/next-n heads"
            ));
        }
        if matches!(speculative_mode, SpeculativeMode::Mtp { .. }) {
            return Err(anyhow!(
                "speculative=mtp was requested and the model advertises MTP metadata, \
                 but the current Candle backend cannot read native MTP auxiliary logits yet"
            ));
        }
        let speculative_strategy: Option<Box<dyn SpeculativeStrategy>> = match speculative_mode {
            SpeculativeMode::NGram { ngram_order, .. } => {
                let strategy = NGramStrategy::new(ngram_order);
                strategy.set_context(&token_ids);
                Some(Box::new(strategy))
            }
            SpeculativeMode::DraftModel {
                ref model_path,
                num_speculative,
            } => {
                let strategy = super::speculative::DraftModelStrategy::new(
                    model_path.clone(),
                    num_speculative,
                    self.device_kind,
                );
                Some(Box::new(strategy))
            }
            _ => None,
        };

        // Trace logits helper
        let trace_logits = |logits: &Tensor| -> Result<()> {
            if tracing::enabled!(tracing::Level::TRACE) {
                let logits_f32 = logits.to_dtype(DType::F32)?;
                let logits_vec = logits_f32.to_vec1::<f32>()?;
                let min_val = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                tracing::trace!(
                    "Logits: min={:.4}, max={:.4}, preview={:?}",
                    min_val,
                    max_val,
                    &logits_vec[..5.min(logits_vec.len())]
                );
            }
            Ok(())
        };

        let _span_prefill =
            tracing::info_span!("prefill", num_tokens = token_ids.len(), start_pos).entered();
        if start_pos > 0 {
            let suffix = &token_ids[start_pos..];
            for (i, &tok) in suffix.iter().enumerate() {
                let input_ids = Tensor::new(&[[tok]], &self.device)?;
                let logits = model_guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("model is not loaded"))?
                    .forward(&input_ids, start_pos + i)?;
                let logits = Self::logits_for_last_position(logits)?;
                trace_logits(&logits)?;
                let filtered = if let Some(ref fmt) = params.response_format {
                    filter_logits_by_grammar(
                        &logits,
                        &prev_text,
                        fmt,
                        &self.vocab_strings,
                        &self.eos_token_ids,
                        &self.device,
                    )?
                } else {
                    logits
                };
                next_token = logits_processor.sample(&filtered)?;
            }
            start_pos += suffix.len();
        } else {
            let input_ids = Tensor::new(token_ids.clone(), &self.device)?.unsqueeze(0)?;
            let logits = model_guard
                .as_mut()
                .ok_or_else(|| anyhow!("model is not loaded"))?
                .forward(&input_ids, 0)?;
            let logits = Self::logits_for_last_position(logits)?;
            trace_logits(&logits)?;
            let filtered = if let Some(ref fmt) = params.response_format {
                filter_logits_by_grammar(
                    &logits,
                    &prev_text,
                    fmt,
                    &self.vocab_strings,
                    &self.eos_token_ids,
                    &self.device,
                )?
            } else {
                logits
            };
            next_token = logits_processor.sample(&filtered)?;
            start_pos += token_ids.len();
        }
        drop(_span_prefill);

        if !self.eos_token_ids.contains(&next_token) {
            all_tokens.push(next_token);
            current_prefix_guard.push(next_token);
            if let Some(strategy) = &speculative_strategy {
                strategy.update_context(&[next_token]);
            }

            self.decode_and_emit_delta(&all_tokens, token_ids.len(), &mut prev_text, sink)?;

            let mut generated = 1usize;
            let _span_decode =
                tracing::info_span!("decode", max_tokens = params.max_tokens).entered();
            while generated < params.max_tokens {
                if let Some(strategy) = &speculative_strategy {
                    if params.response_format.is_none() {
                        let remaining = params.max_tokens - generated;
                        let draft_budget = match speculative_mode {
                            SpeculativeMode::NGram {
                                num_speculative, ..
                            } => num_speculative.min(remaining.saturating_sub(1)),
                            SpeculativeMode::DraftModel {
                                num_speculative, ..
                            } => num_speculative.min(remaining.saturating_sub(1)),
                            _ => 0,
                        };
                        let proposed_with_logits: Vec<(u32, Vec<f32>)> = if draft_budget > 0 {
                            if params.temperature > 1e-6 {
                                strategy.propose_with_logits(
                                    &all_tokens,
                                    draft_budget,
                                    params.temperature,
                                )?
                            } else {
                                strategy
                                    .propose(&all_tokens, draft_budget)?
                                    .into_iter()
                                    .map(|t| (t, Vec::new()))
                                    .collect()
                            }
                        } else {
                            Vec::new()
                        };
                        if !proposed_with_logits.is_empty() {
                            let proposed: Vec<u32> =
                                proposed_with_logits.iter().map(|(t, _)| *t).collect();
                            let has_draft_logits =
                                proposed_with_logits.iter().any(|(_, l)| !l.is_empty());

                            let last = *all_tokens.last().unwrap();
                            let mut verify_tokens = Vec::with_capacity(proposed.len() + 1);
                            verify_tokens.push(last);
                            verify_tokens.extend_from_slice(&proposed);

                            let input_ids = Tensor::new(verify_tokens.as_slice(), &self.device)?
                                .unsqueeze(0)?;
                            let verifier_logits = model_guard
                                .as_mut()
                                .ok_or_else(|| anyhow!("model is not loaded"))?
                                .forward(&input_ids, start_pos)?;
                            let verifier_logits = Self::logits_sequence(verifier_logits)?;

                            // Decide acceptance and the correction/bonus token.
                            //
                            // Use standard rejection sampling (`min(1, p/q)` +
                            // residual correction) when the draft strategy
                            // exposes per-position logits and the sampling
                            // temperature is non-zero. Otherwise fall back to
                            // greedy exact-match verification, which only
                            // accepts draft tokens that equal the target argmax.
                            let (accepted, correction_token) =
                                if params.temperature > 1e-6 && has_draft_logits {
                                    let verifier_vecs =
                                        Self::verifier_logits_to_vecs(&verifier_logits)?;
                                    // Derive an independent PRNG seed from the
                                    // request seed so rejection sampling
                                    // doesn't share state with `logits_processor`.
                                    let mut rng_state =
                                        params.seed.unwrap_or(42).wrapping_add(0x9E3779B97F4A7C15);
                                    verify_with_rejection_sampling(
                                        &verifier_vecs,
                                        &proposed_with_logits,
                                        params.temperature,
                                        &mut rng_state,
                                    )
                                } else {
                                    let mut verifier_greedy = Vec::with_capacity(proposed.len());
                                    for idx in 0..proposed.len() {
                                        verifier_greedy
                                            .push(Self::greedy_token(&verifier_logits.get(idx)?)?);
                                    }
                                    let accepted =
                                        verify_greedy_tokens(proposed.len(), &proposed, |idx| {
                                            verifier_greedy.get(idx).copied()
                                        });
                                    (accepted, None)
                                };

                            total_draft_tokens += proposed.len();
                            total_accepted_tokens += accepted;

                            start_pos += verify_tokens.len();
                            if accepted > 0 {
                                let accepted_tokens = &proposed[..accepted];
                                all_tokens.extend_from_slice(accepted_tokens);
                                current_prefix_guard.extend_from_slice(accepted_tokens);
                                strategy.update_context(accepted_tokens);
                                generated += accepted;
                                self.decode_and_emit_delta(
                                    &all_tokens,
                                    token_ids.len(),
                                    &mut prev_text,
                                    sink,
                                )?;
                            }

                            if generated >= params.max_tokens {
                                break;
                            }

                            // Determine the next token to emit.
                            //
                            // When rejection sampling produced a correction or
                            // bonus token, use it directly — the token is already
                            // temperature-scaled and (for corrections) drawn
                            // from the residual `norm(max(0, p - q))`. When some
                            // draft tokens were rejected, the model's KV cache
                            // contains stale entries for the rejected tokens,
                            // so we replay the accepted prefix to restore a
                            // consistent cache state before emitting the token.
                            //
                            // When no correction/bonus token is available
                            // (greedy path, or rejection sampling with no bonus
                            // slot), fall back to sampling from the verifier
                            // logits at the appropriate position — matching the
                            // pre-existing behaviour.
                            let next_tok = if let Some(tok) = correction_token {
                                if accepted < proposed.len() {
                                    let replay_len = all_tokens.len();
                                    start_pos = self.replay_generation_prefix(
                                        &mut model_guard,
                                        &all_tokens[..replay_len],
                                    )?;
                                }
                                tok
                            } else {
                                let correction_logits = if accepted == proposed.len() {
                                    verifier_logits.get(proposed.len())?
                                } else {
                                    let replay_len = all_tokens.len();
                                    start_pos = self.replay_generation_prefix(
                                        &mut model_guard,
                                        &all_tokens[..replay_len],
                                    )?;
                                    verifier_logits.get(accepted)?
                                };

                                let filtered = if let Some(ref fmt) = params.response_format {
                                    filter_logits_by_grammar(
                                        &correction_logits,
                                        &prev_text,
                                        fmt,
                                        &self.vocab_strings,
                                        &self.eos_token_ids,
                                        &self.device,
                                    )?
                                } else {
                                    correction_logits.clone()
                                };
                                logits_processor.sample(&filtered)?
                            };

                            if self.eos_token_ids.contains(&next_tok) {
                                break;
                            }

                            all_tokens.push(next_tok);
                            current_prefix_guard.push(next_tok);
                            strategy.update_context(&[next_tok]);
                            generated += 1;
                            self.decode_and_emit_delta(
                                &all_tokens,
                                token_ids.len(),
                                &mut prev_text,
                                sink,
                            )?;
                            continue;
                        }
                    }
                }

                let last = *all_tokens.last().unwrap();
                let input_ids = Tensor::new(&[[last]], &self.device)?;
                let logits = model_guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("model is not loaded"))?
                    .forward(&input_ids, start_pos)?;
                let logits = Self::logits_for_last_position(logits)?;
                trace_logits(&logits)?;

                let filtered = if let Some(ref fmt) = params.response_format {
                    filter_logits_by_grammar(
                        &logits,
                        &prev_text,
                        fmt,
                        &self.vocab_strings,
                        &self.eos_token_ids,
                        &self.device,
                    )?
                } else {
                    logits
                };
                let next_tok = logits_processor.sample(&filtered)?;
                start_pos += 1;

                if self.eos_token_ids.contains(&next_tok) {
                    break;
                }

                all_tokens.push(next_tok);
                current_prefix_guard.push(next_tok);
                if let Some(strategy) = &speculative_strategy {
                    strategy.update_context(&[next_tok]);
                }
                generated += 1;

                self.decode_and_emit_delta(&all_tokens, token_ids.len(), &mut prev_text, sink)?;
            }
        }

        let compute_ms = start_time.elapsed().as_millis() as u64;
        let spec_mode = SpeculativeMode::from_env().unwrap_or(SpeculativeMode::None);
        let spec_acceptance_rate = if total_draft_tokens > 0 {
            Some(total_accepted_tokens as f64 / total_draft_tokens as f64)
        } else {
            None
        };
        let (spec_draft, spec_accepted) = if spec_mode.is_enabled() {
            (Some(total_draft_tokens), Some(total_accepted_tokens))
        } else {
            (None, None)
        };

        sink.on_chunk(crate::io::OutputChunk::Metrics {
            compute_ms,
            speculative_acceptance_rate: spec_acceptance_rate,
            speculative_draft_tokens: spec_draft,
            speculative_accepted_tokens: spec_accepted,
        })?;

        sink.on_chunk(crate::io::OutputChunk::End)?;

        let coordinator = bloomai_core::global_vram_coordinator();
        let strategy = coordinator
            .residency_strategy_for_model(&self.metadata.id)
            .unwrap_or(ResidencyStrategy::OnDemand);

        if strategy == ResidencyStrategy::OnDemand {
            *model_guard = None;
            tracing::info!("Model weights successfully offloaded from GPU (OnDemand).");
        } else {
            tracing::debug!("Keeping model weights resident (strategy: {:?}).", strategy);
        }
        Ok(())
    }
}

fn autocomplete_json(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return "{}".to_string();
    }

    let mut completed = text.to_string();
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
        } else if c == '{' {
            stack.push('}');
        } else if c == '[' {
            stack.push(']');
        } else if c == '}' || c == ']' {
            if let Some(&last) = stack.last() {
                if last == c {
                    stack.pop();
                }
            }
        }
        i += 1;
    }

    if in_string {
        completed.push('"');
    }
    while let Some(close_char) = stack.pop() {
        let trimmed = completed.trim_end();
        if close_char == '}' {
            if trimmed.ends_with(':') {
                completed.push_str("null");
            } else if trimmed.ends_with(',') {
                completed.push_str("\"_dummy\":null");
            }
        } else if close_char == ']' {
            if trimmed.ends_with(',') {
                completed.push_str("null");
            }
        }
        completed.push(close_char);
    }

    completed
}

fn validate_partial_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> std::result::Result<(), String> {
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        if value.is_string() {
            let s = value.as_str().unwrap();
            if !enum_values.iter().any(|candidate| {
                if let Some(cand_str) = candidate.as_str() {
                    cand_str.starts_with(s)
                } else {
                    candidate == value
                }
            }) {
                return Err(format!(
                    "{} does not match prefix of any allowed enum value",
                    path
                ));
            }
        } else {
            if !enum_values.iter().any(|candidate| candidate == value) {
                if value != &serde_json::Value::Null {
                    return Err(format!("{} does not match any allowed enum value", path));
                }
            }
        }
    }

    if let Some(schema_type) = schema.get("type").and_then(|v| v.as_str()) {
        if value != &serde_json::Value::Null && !json_value_matches_type(value, schema_type) {
            return Err(format!("{} expected type {}", path, schema_type));
        }
    }

    if value.is_object() {
        let object = value.as_object().unwrap();
        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            for (field, property_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_partial_json_schema(
                        field_value,
                        property_schema,
                        &format!("{}.{}", path, field),
                    )?;
                }
            }
            if schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false) {
                for field in object.keys() {
                    if field != "_dummy" && !properties.contains_key(field) {
                        return Err(format!(
                            "{} contains unsupported property '{}'",
                            path, field
                        ));
                    }
                }
            }
        }
    }

    if let Some(item_schema) = schema.get("items") {
        if let Some(items) = value.as_array() {
            for (idx, item) in items.iter().enumerate() {
                validate_partial_json_schema(item, item_schema, &format!("{}[{}]", path, idx))?;
            }
        }
    }

    Ok(())
}

fn validate_complete_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> std::result::Result<(), String> {
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        if !enum_values.iter().any(|candidate| candidate == value) {
            return Err(format!("{} does not match any allowed enum value", path));
        }
    }

    if let Some(schema_type) = schema.get("type").and_then(|v| v.as_str()) {
        if !json_value_matches_type(value, schema_type) {
            return Err(format!("{} expected type {}", path, schema_type));
        }
    }

    if value.is_object() {
        let object = value.as_object().unwrap();
        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            for field in required.iter().filter_map(|v| v.as_str()) {
                if !object.contains_key(field) {
                    return Err(format!("{} missing required property '{}'", path, field));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            for (field, property_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_complete_json_schema(
                        field_value,
                        property_schema,
                        &format!("{}.{}", path, field),
                    )?;
                }
            }
            if schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false) {
                for field in object.keys() {
                    if !properties.contains_key(field) {
                        return Err(format!(
                            "{} contains unsupported property '{}'",
                            path, field
                        ));
                    }
                }
            }
        }
    }

    if let Some(item_schema) = schema.get("items") {
        if let Some(items) = value.as_array() {
            for (idx, item) in items.iter().enumerate() {
                validate_complete_json_schema(item, item_schema, &format!("{}[{}]", path, idx))?;
            }
        }
    }

    Ok(())
}

fn json_value_matches_type(value: &serde_json::Value, schema_type: &str) -> bool {
    match schema_type {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || (value.is_f64() && value.as_f64().unwrap().fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn is_currently_in_string(text: &str) -> bool {
    let mut in_string = false;
    let mut escaped = false;
    for c in text.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
        }
    }
    in_string
}

fn starts_with_valid_outside_string_char(s: &str) -> bool {
    let s_trimmed = s.trim_start();
    if s_trimmed.is_empty() {
        return true;
    }
    let first_char = s_trimmed.chars().next().unwrap();
    matches!(
        first_char,
        '{' | '}' | '[' | ']' | ':' | ',' | '"' | '-' | '+' | '.' | '0'
            ..='9' | 'e' | 'E' | 't' | 'r' | 'u' | 'f' | 'a' | 'l' | 's' | 'n'
    )
}

fn is_valid_partial_response(text: &str, format: &bloomai_core::ResponseFormat) -> bool {
    match format {
        bloomai_core::ResponseFormat::Text => true,
        bloomai_core::ResponseFormat::JsonObject => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return true;
            }
            if !trimmed.starts_with('{') {
                return false;
            }
            let completed = autocomplete_json(text);
            match serde_json::from_str::<serde_json::Value>(&completed) {
                Ok(v) => v.is_object(),
                Err(_) => false,
            }
        }
        bloomai_core::ResponseFormat::JsonSchema(schema) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return true;
            }
            if !trimmed.starts_with('{') {
                return false;
            }
            let completed = autocomplete_json(text);
            match serde_json::from_str::<serde_json::Value>(&completed) {
                Ok(v) => {
                    if !v.is_object() {
                        false
                    } else {
                        validate_partial_json_schema(&v, schema, "$").is_ok()
                    }
                }
                Err(_) => false,
            }
        }
    }
}

fn is_complete_and_valid_response(text: &str, format: &bloomai_core::ResponseFormat) -> bool {
    match format {
        bloomai_core::ResponseFormat::Text => true,
        bloomai_core::ResponseFormat::JsonObject => {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
                v.is_object()
            } else {
                false
            }
        }
        bloomai_core::ResponseFormat::JsonSchema(schema) => {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
                if !v.is_object() {
                    false
                } else {
                    validate_complete_json_schema(&v, schema, "$").is_ok()
                }
            } else {
                false
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DfaState {
    Start,
    ObjectStart,
    KeyChar,
    KeyCloseQuote,
    Colon,
    ValueStart,
    InStringValue,
    InNumberValue,
    InBoolNullValue,
    ValueEnd,
    Comma,
    ObjectEnd,
    Invalid,
}

#[derive(Debug, Clone)]
pub(crate) struct JsonDfa {
    pub(crate) state: DfaState,
    pub(crate) stack: Vec<char>,
    pub(crate) escaped: bool,
}

impl JsonDfa {
    pub(crate) fn new() -> Self {
        Self {
            state: DfaState::Start,
            stack: Vec::new(),
            escaped: false,
        }
    }

    pub(crate) fn feed(&mut self, text: &str) {
        for c in text.chars() {
            if c.is_whitespace()
                && !matches!(self.state, DfaState::KeyChar | DfaState::InStringValue)
            {
                continue;
            }

            if self.escaped {
                self.escaped = false;
                continue;
            }

            if (matches!(self.state, DfaState::KeyChar | DfaState::InStringValue)) && c == '\\' {
                self.escaped = true;
                continue;
            }

            match self.state {
                DfaState::Start => {
                    if c == '{' {
                        self.state = DfaState::ObjectStart;
                    } else if c == '[' {
                        self.stack.push(']');
                        self.state = DfaState::ValueStart;
                    } else {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::ObjectStart => {
                    if c == '"' {
                        self.state = DfaState::KeyChar;
                    } else if c == '}' {
                        if self.stack.pop() == Some('}') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::ObjectEnd;
                        }
                    } else {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::KeyChar => {
                    if c == '"' {
                        self.state = DfaState::KeyCloseQuote;
                    }
                }
                DfaState::KeyCloseQuote => {
                    if c == ':' {
                        self.state = DfaState::Colon;
                    } else {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::Colon => {
                    if c == '"' {
                        self.state = DfaState::InStringValue;
                    } else if c.is_ascii_digit() || c == '-' {
                        self.state = DfaState::InNumberValue;
                    } else if c == 't' || c == 'f' || c == 'n' {
                        self.state = DfaState::InBoolNullValue;
                    } else if c == '{' {
                        self.stack.push('}');
                        self.state = DfaState::ObjectStart;
                    } else if c == '[' {
                        self.stack.push(']');
                        self.state = DfaState::ValueStart;
                    } else {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::ValueStart => {
                    if c == '"' {
                        self.state = DfaState::InStringValue;
                    } else if c == '{' {
                        self.stack.push('}');
                        self.state = DfaState::ObjectStart;
                    } else if c == '[' {
                        self.stack.push(']');
                        self.state = DfaState::ValueStart;
                    } else if c == 't' || c == 'f' || c == 'n' {
                        self.state = DfaState::InBoolNullValue;
                    } else if c.is_ascii_digit() || c == '-' {
                        self.state = DfaState::InNumberValue;
                    } else if c == ']' {
                        if self.stack.pop() == Some(']') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::Invalid;
                        }
                    } else {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::InStringValue => {
                    if c == '"' {
                        self.state = DfaState::ValueEnd;
                    }
                }
                DfaState::InNumberValue => {
                    if c == ',' {
                        self.state = DfaState::Comma;
                    } else if c == '}' {
                        if self.stack.pop() == Some('}') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::ObjectEnd;
                        }
                    } else if c == ']' {
                        if self.stack.pop() == Some(']') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::Invalid;
                        }
                    } else if !c.is_ascii_digit()
                        && c != '.'
                        && c != 'e'
                        && c != 'E'
                        && c != '-'
                        && c != '+'
                    {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::InBoolNullValue => {
                    if c == ',' {
                        self.state = DfaState::Comma;
                    } else if c == '}' {
                        if self.stack.pop() == Some('}') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::ObjectEnd;
                        }
                    } else if c == ']' {
                        if self.stack.pop() == Some(']') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::Invalid;
                        }
                    } else if !c.is_alphabetic() {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::ValueEnd => {
                    if c == ',' {
                        self.state = DfaState::Comma;
                    } else if c == '}' {
                        if self.stack.pop() == Some('}') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::ObjectEnd;
                        }
                    } else if c == ']' {
                        if self.stack.pop() == Some(']') {
                            self.state = DfaState::ValueEnd;
                        } else {
                            self.state = DfaState::Invalid;
                        }
                    } else {
                        self.state = DfaState::Invalid;
                    }
                }
                DfaState::Comma => {
                    if self.stack.last() == Some(&']') {
                        // Inside an array, a comma means we start a new element
                        if c == '"' {
                            self.state = DfaState::InStringValue;
                        } else if c == '{' {
                            self.stack.push('}');
                            self.state = DfaState::ObjectStart;
                        } else if c == '[' {
                            self.stack.push(']');
                            self.state = DfaState::ValueStart;
                        } else if c == 't' || c == 'f' || c == 'n' {
                            self.state = DfaState::InBoolNullValue;
                        } else if c.is_ascii_digit() || c == '-' {
                            self.state = DfaState::InNumberValue;
                        } else {
                            self.state = DfaState::Invalid;
                        }
                    } else {
                        // Inside an object, a comma must be followed by a key
                        if c == '"' {
                            self.state = DfaState::KeyChar;
                        } else {
                            self.state = DfaState::Invalid;
                        }
                    }
                }
                DfaState::ObjectEnd => {
                    self.state = DfaState::Invalid;
                }
                DfaState::Invalid => {}
            }
        }
    }
}

pub(crate) fn filter_logits_by_grammar(
    logits: &Tensor,
    generated_text: &str,
    format: &bloomai_core::ResponseFormat,
    vocab_strings: &[String],
    eos_token_ids: &[u32],
    device: &Device,
) -> Result<Tensor> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let mut logits_vec = logits_f32.to_vec1::<f32>()?;
    let vocab_size = logits_vec.len();

    // Pre-filtering optimization:
    // Only perform the full grammar constraint check on:
    // 1. Special/EOS tokens (always check them to see if completion is valid)
    // 2. The top N tokens with the highest logit values.
    // This reduces the checks from V (e.g. 150k) to K (at most 256), avoiding CPU starvation.
    let top_n = 256.min(vocab_size);
    let mut top_indices = std::collections::HashSet::with_capacity(top_n);
    if top_n > 0 {
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.select_nth_unstable_by(top_n - 1, |&a, &b| {
            logits_vec[b]
                .partial_cmp(&logits_vec[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &idx in &indices[0..top_n] {
            top_indices.insert(idx);
        }
    }

    let is_json = matches!(
        format,
        bloomai_core::ResponseFormat::JsonObject | bloomai_core::ResponseFormat::JsonSchema(_)
    );

    let dfa = if is_json {
        let mut d = JsonDfa::new();
        d.feed(generated_text);
        Some(d)
    } else {
        None
    };

    let in_string = is_currently_in_string(generated_text);
    let has_enum = match format {
        bloomai_core::ResponseFormat::JsonSchema(schema) => schema.to_string().contains("\"enum\""),
        _ => false,
    };

    let is_complete = is_complete_and_valid_response(generated_text, format);

    for id in 0..vocab_size {
        let is_eos = eos_token_ids.contains(&(id as u32));

        // If not EOS and not a top candidate, prune immediately.
        if !is_eos && !top_indices.contains(&id) {
            logits_vec[id] = -f32::INFINITY;
            continue;
        }

        if is_eos {
            if !is_complete {
                logits_vec[id] = -f32::INFINITY;
            }
            continue;
        }

        let tok_str = if id < vocab_strings.len() {
            &vocab_strings[id]
        } else {
            ""
        };

        if tok_str.is_empty() {
            logits_vec[id] = -f32::INFINITY;
            continue;
        }

        if let Some(ref d) = dfa {
            let mut test_dfa = d.clone();
            test_dfa.feed(tok_str);
            if matches!(test_dfa.state, DfaState::Invalid) {
                logits_vec[id] = -f32::INFINITY;
                continue;
            }
        }

        if in_string {
            if !has_enum && !tok_str.contains('"') && !tok_str.contains('\\') {
                continue;
            }
        } else {
            if !starts_with_valid_outside_string_char(tok_str) {
                logits_vec[id] = -f32::INFINITY;
                continue;
            }
        }

        let next_text = format!("{}{}", generated_text, tok_str);
        if !is_valid_partial_response(&next_text, format) {
            logits_vec[id] = -f32::INFINITY;
        }
    }

    let mut all_neg_inf = true;
    for &val in &logits_vec {
        if val > -f32::INFINITY {
            all_neg_inf = false;
            break;
        }
    }
    if all_neg_inf {
        tracing::warn!(
            "Grammar filtering eliminated all tokens. Falling back to unconstrained logits."
        );
        return Ok(logits.clone());
    }

    Tensor::new(logits_vec, device)?
        .to_dtype(logits.dtype())
        .map_err(Into::into)
}

pub struct ServerKvHook {
    request_models: Arc<Mutex<std::collections::HashMap<usize, Arc<Mutex<QwenModelWrapper>>>>>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
}

impl ServerKvHook {
    pub fn new(
        request_models: Arc<Mutex<std::collections::HashMap<usize, Arc<Mutex<QwenModelWrapper>>>>>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            request_models,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim: num_kv_heads * head_dim,
        }
    }
}

impl crate::scheduler::kv_hook::KvHook for ServerKvHook {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn extract_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let model_arc = {
            let models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
            models.get(&handle).cloned()
        };

        let model_arc = match model_arc {
            Some(m) => m,
            None => return Err(anyhow!("No model wrapper found for handle {}", handle)),
        };

        let mut model_guard = model_arc.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *model_guard {
            QwenModelWrapper::Streaming(model) => model
                .extract_kv(layer_idx, start_pos, seq_len, self.kv_dim)
                .map_err(Into::into),
            QwenModelWrapper::Gemma4Streaming(model) => model
                .extract_kv(layer_idx, start_pos, seq_len, self.kv_dim)
                .map_err(Into::into),
            _ => Ok((
                vec![0.0; seq_len * self.kv_dim],
                vec![0.0; seq_len * self.kv_dim],
            )),
        }
    }

    fn extract_kv_tensor(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> anyhow::Result<Option<(Tensor, Tensor)>> {
        let model_arc = {
            let models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
            models.get(&handle).cloned()
        };

        let model_arc = match model_arc {
            Some(m) => m,
            None => return Err(anyhow!("No model wrapper found for handle {}", handle)),
        };

        let mut model_guard = model_arc.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *model_guard {
            QwenModelWrapper::Streaming(model) => model
                .extract_kv_tensor(layer_idx, start_pos, seq_len)
                .map(Some)
                .map_err(Into::into),
            QwenModelWrapper::Gemma4Streaming(model) => model
                .extract_kv_tensor(layer_idx, start_pos, seq_len)
                .map(Some)
                .map_err(Into::into),
            _ => Ok(None),
        }
    }

    fn inject_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> anyhow::Result<()> {
        let model_arc = {
            let models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
            models.get(&handle).cloned()
        };

        let model_arc = match model_arc {
            Some(m) => m,
            None => return Err(anyhow!("No model wrapper found for handle {}", handle)),
        };

        let mut model_guard = model_arc.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *model_guard {
            QwenModelWrapper::Streaming(model) => model
                .inject_kv(
                    layer_idx,
                    start_pos,
                    keys,
                    values,
                    seq_len,
                    self.num_kv_heads,
                    self.head_dim,
                    self.kv_dim,
                )
                .map_err(Into::into),
            QwenModelWrapper::Gemma4Streaming(model) => model
                .inject_kv(
                    layer_idx,
                    start_pos,
                    keys,
                    values,
                    seq_len,
                    self.num_kv_heads,
                    self.head_dim,
                    self.kv_dim,
                )
                .map_err(Into::into),
            _ => Ok(()),
        }
    }

    fn inject_kv_tensor(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> anyhow::Result<()> {
        let model_arc = {
            let models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
            models.get(&handle).cloned()
        };

        let model_arc = match model_arc {
            Some(m) => m,
            None => return Err(anyhow!("No model wrapper found for handle {}", handle)),
        };

        let mut model_guard = model_arc.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *model_guard {
            QwenModelWrapper::Streaming(model) => model
                .inject_kv_tensor(layer_idx, start_pos, keys, values)
                .map_err(Into::into),
            QwenModelWrapper::Gemma4Streaming(model) => model
                .inject_kv_tensor(layer_idx, start_pos, keys, values)
                .map_err(Into::into),
            _ => Ok(()),
        }
    }

    fn clear_kv_cache(&self, handle: usize) -> anyhow::Result<()> {
        let model_arc = {
            let models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
            models.get(&handle).cloned()
        };
        if let Some(model_arc) = model_arc {
            let mut model_guard = model_arc.lock().unwrap_or_else(|e| e.into_inner());
            match &mut *model_guard {
                QwenModelWrapper::Streaming(model) => {
                    model.clear_kv_cache();
                }
                QwenModelWrapper::Gemma4Streaming(model) => {
                    model.clear_kv_cache();
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn rollback_kv_cache(&self, handle: usize, length: usize) -> anyhow::Result<()> {
        let model_arc = {
            let models = self.request_models.lock().unwrap_or_else(|e| e.into_inner());
            models.get(&handle).cloned()
        };
        if let Some(model_arc) = model_arc {
            let mut model_guard = model_arc.lock().unwrap_or_else(|e| e.into_inner());
            match &mut *model_guard {
                QwenModelWrapper::Streaming(model) => {
                    model.truncate_kv_cache(length)?;
                }
                QwenModelWrapper::Gemma4Streaming(model) => {
                    model.truncate_kv_cache(length)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::SystemTime;

    fn create_temp_dir() -> PathBuf {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("bloom_candle_test_{}", now));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_qwen_model_type_from_config() {
        let v2 = serde_json::json!({ "model_type": "qwen2" });
        assert_eq!(ModelType::from_config(&v2), Some(ModelType::Qwen2));

        let v3 = serde_json::json!({ "model_type": "qwen3" });
        assert_eq!(ModelType::from_config(&v3), Some(ModelType::Qwen3));

        let llama = serde_json::json!({ "model_type": "llama" });
        assert_eq!(ModelType::from_config(&llama), Some(ModelType::Llama));

        let invalid = serde_json::json!({ "model_type": "invalid_model" });
        assert_eq!(ModelType::from_config(&invalid), None);
    }

    #[test]
    fn test_find_safetensors_files() {
        let dir = create_temp_dir();

        // Initially empty
        assert!(find_safetensors_files(&dir).is_empty());

        // Single file
        let single = dir.join("model.safetensors");
        fs::write(&single, "").unwrap();
        let found = find_safetensors_files(&dir);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], single);

        fs::remove_file(single).unwrap();

        // Sharded files
        let shard2 = dir.join("model-00002-of-00002.safetensors");
        let shard1 = dir.join("model-00001-of-00002.safetensors");
        fs::write(&shard2, "").unwrap();
        fs::write(&shard1, "").unwrap();

        let found_shards = find_safetensors_files(&dir);
        assert_eq!(found_shards.len(), 2);
        // Should be sorted
        assert_eq!(found_shards[0], shard1);
        assert_eq!(found_shards[1], shard2);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_candle_engine_load_errors() {
        let engine = CandleEngine;
        let non_existent = Path::new("non_existent_candle_path_12345");

        // Non-existent path
        let res = engine.load(non_existent, DeviceKind::Cpu);
        assert!(res.is_err());
        let err_msg = res.err().unwrap().to_string();
        assert!(
            err_msg.contains("no config.json or GGUF file found")
                || err_msg.contains("failed to read config.json"),
            "unexpected error: {}",
            err_msg
        );

        // Unsupported device
        let dir = create_temp_dir();
        let res_device = engine.load(&dir, DeviceKind::Npu);
        assert!(res_device.is_err());
        assert!(res_device
            .err()
            .unwrap()
            .to_string()
            .contains("only supports CPU"));
        let _ = fs::remove_dir_all(dir);
    }

    fn create_mock_llama_gguf_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV count: 9
        buf.extend_from_slice(&9u64.to_le_bytes());

        // Helper closures to write KV pairs
        fn write_string(buf: &mut Vec<u8>, s: &str) {
            let len = s.len() as u64;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        fn write_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
            write_string(buf, key);
            buf.extend_from_slice(&8u32.to_le_bytes()); // Type String
            write_string(buf, val);
        }
        fn write_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
            write_string(buf, key);
            buf.extend_from_slice(&4u32.to_le_bytes()); // Type U32
            buf.extend_from_slice(&val.to_le_bytes());
        }
        fn write_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
            write_string(buf, key);
            buf.extend_from_slice(&6u32.to_le_bytes()); // Type F32
            buf.extend_from_slice(&val.to_le_bytes());
        }

        // KV 1: general.architecture
        write_kv_string(&mut buf, "general.architecture", "llama");
        // KV 2: general.name
        write_kv_string(&mut buf, "general.name", "mock-llama");
        // KV 3: llama.block_count
        write_kv_u32(&mut buf, "llama.block_count", 32);
        // KV 4: llama.context_length
        write_kv_u32(&mut buf, "llama.context_length", 2048);
        // KV 5: llama.embedding_length
        write_kv_u32(&mut buf, "llama.embedding_length", 4096);
        // KV 6: llama.attention.head_count
        write_kv_u32(&mut buf, "llama.attention.head_count", 32);
        // KV 7: llama.attention.head_count_kv
        write_kv_u32(&mut buf, "llama.attention.head_count_kv", 8);
        // KV 8: llama.feed_forward_length
        write_kv_u32(&mut buf, "llama.feed_forward_length", 11008);
        // KV 9: llama.attention.layer_norm_rms_epsilon
        write_kv_f32(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);

        // Tensor info record (required to avoid Content::read error if tensor count is non-zero)
        write_string(&mut buf, "blk.0.attn.weight");
        // Num dimensions
        buf.extend_from_slice(&2u32.to_le_bytes());
        // Dimension 1
        buf.extend_from_slice(&128u64.to_le_bytes());
        // Dimension 2
        buf.extend_from_slice(&128u64.to_le_bytes());
        // ggml_dtype (2 = Q4_0)
        buf.extend_from_slice(&2u32.to_le_bytes());
        // Offset
        buf.extend_from_slice(&0u64.to_le_bytes());

        buf
    }

    #[test]
    fn test_synthesize_config_from_gguf_llama() {
        let bytes = create_mock_llama_gguf_bytes();
        let dir = create_temp_dir();
        let path = dir.join("model.gguf");
        fs::write(&path, bytes).unwrap();

        let (config, arch) = synthesize_config_from_gguf(&path).unwrap();
        assert_eq!(arch, "llama");

        let config_obj = config.as_object().unwrap();
        assert_eq!(
            config_obj.get("model_type"),
            Some(&serde_json::json!("llama"))
        );
        assert_eq!(
            config_obj.get("hidden_size"),
            Some(&serde_json::json!(4096))
        );
        assert_eq!(
            config_obj.get("num_hidden_layers"),
            Some(&serde_json::json!(32))
        );
        assert_eq!(
            config_obj.get("num_attention_heads"),
            Some(&serde_json::json!(32))
        );
        assert_eq!(
            config_obj.get("num_key_value_heads"),
            Some(&serde_json::json!(8))
        );
        assert_eq!(
            config_obj.get("intermediate_size"),
            Some(&serde_json::json!(11008))
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_synthesize_config_from_gguf_mistral() {
        fn create_mock_gguf_bytes_for_arch(arch: &str) -> Vec<u8> {
            let mut buf = Vec::new();
            buf.extend_from_slice(b"GGUF");
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&9u64.to_le_bytes());

            fn write_string(buf: &mut Vec<u8>, s: &str) {
                let len = s.len() as u64;
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            fn write_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
                write_string(buf, key);
                buf.extend_from_slice(&8u32.to_le_bytes());
                write_string(buf, val);
            }
            fn write_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
                write_string(buf, key);
                buf.extend_from_slice(&4u32.to_le_bytes());
                buf.extend_from_slice(&val.to_le_bytes());
            }
            fn write_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
                write_string(buf, key);
                buf.extend_from_slice(&6u32.to_le_bytes());
                buf.extend_from_slice(&val.to_le_bytes());
            }

            write_kv_string(&mut buf, "general.architecture", arch);
            write_kv_string(&mut buf, "general.name", "mock-model");
            write_kv_u32(&mut buf, &format!("{}.block_count", arch), 32);
            write_kv_u32(&mut buf, &format!("{}.context_length", arch), 2048);
            write_kv_u32(&mut buf, &format!("{}.embedding_length", arch), 4096);
            write_kv_u32(&mut buf, &format!("{}.attention.head_count", arch), 32);
            write_kv_u32(&mut buf, &format!("{}.attention.head_count_kv", arch), 8);
            write_kv_u32(&mut buf, &format!("{}.feed_forward_length", arch), 11008);
            write_kv_f32(
                &mut buf,
                &format!("{}.attention.layer_norm_rms_epsilon", arch),
                1e-5,
            );

            write_string(&mut buf, "blk.0.attn.weight");
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&128u64.to_le_bytes());
            buf.extend_from_slice(&128u64.to_le_bytes());
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());

            buf
        }

        let bytes = create_mock_gguf_bytes_for_arch("mistral");
        let dir = create_temp_dir();
        let path = dir.join("model.gguf");
        fs::write(&path, bytes).unwrap();

        let (config, arch_str) = synthesize_config_from_gguf(&path).unwrap();
        assert_eq!(arch_str, "mistral");

        let config_obj = config.as_object().unwrap();
        // Verification that "mistral" GGUF architecture is successfully mapped to Llama model type
        assert_eq!(
            config_obj.get("model_type"),
            Some(&serde_json::json!("llama"))
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_autocomplete_json() {
        assert_eq!(autocomplete_json(""), "{}");
        assert_eq!(autocomplete_json("{"), "{}");
        assert_eq!(autocomplete_json("{\"a\""), "{\"a\"}");
        assert_eq!(autocomplete_json("{\"a\":"), "{\"a\":null}");
        assert_eq!(autocomplete_json("{\"a\": ["), "{\"a\": []}");
        assert_eq!(autocomplete_json("{\"a\": [1,"), "{\"a\": [1,null]}");
    }

    #[test]
    fn test_is_valid_partial_response() {
        let fmt_obj = bloomai_core::ResponseFormat::JsonObject;
        assert!(is_valid_partial_response("", &fmt_obj));
        assert!(is_valid_partial_response("{", &fmt_obj));
        assert!(is_valid_partial_response(
            "{\"hello\": \"world\"}",
            &fmt_obj
        ));

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name"]
        });
        let fmt_schema = bloomai_core::ResponseFormat::JsonSchema(schema);
        assert!(is_valid_partial_response("{", &fmt_schema));
        assert!(is_valid_partial_response(
            "{\"name\": \"Alice\"",
            &fmt_schema
        ));
        assert!(is_valid_partial_response("{\"age\": 30", &fmt_schema));
        assert!(!is_valid_partial_response("{\"name\": 123", &fmt_schema));
    }

    #[test]
    fn test_filter_logits_by_grammar() {
        let device = Device::Cpu;
        let fmt_obj = bloomai_core::ResponseFormat::JsonObject;
        // Vocab: 0=" ", 1="{", 2="}", 3="abc", 4="\"hello\":", 5=":"
        let vocab = vec![
            " ".to_string(),
            "{".to_string(),
            "}".to_string(),
            "abc".to_string(),
            "\"hello\":".to_string(),
            ":".to_string(),
        ];
        let eos_token_ids = vec![99u32];

        // 1. Initial state (empty string). Valid next token must start with '{'.
        let logits = Tensor::new(&[0.0f32, 10.0, 5.0, 1.0, 2.0, 3.0], &device).unwrap();
        let filtered =
            filter_logits_by_grammar(&logits, "", &fmt_obj, &vocab, &eos_token_ids, &device)
                .unwrap();
        let filtered_vec = filtered.to_vec1::<f32>().unwrap();
        // Since we check the top tokens: '{' has logit 10.0, it is valid and kept.
        // '}' has logit 5.0, but starting with '}' is invalid for an object, so it should be -inf.
        assert!(filtered_vec[1] > -f32::INFINITY); // '{' should be kept
        assert_eq!(filtered_vec[2], -f32::INFINITY); // '}' is invalid starting char

        // 2. We are in a JSON object after '{'. Valid next token should be string key or closing '}'.
        let logits = Tensor::new(&[0.0f32, 1.0, 10.0, 2.0, 15.0, 1.0], &device).unwrap();
        let filtered =
            filter_logits_by_grammar(&logits, "{", &fmt_obj, &vocab, &eos_token_ids, &device)
                .unwrap();
        let filtered_vec = filtered.to_vec1::<f32>().unwrap();
        assert!(filtered_vec[2] > -f32::INFINITY); // '}' closing object is valid
        assert!(filtered_vec[4] > -f32::INFINITY); // '"hello":' string key is valid
        assert_eq!(filtered_vec[3], -f32::INFINITY); // 'abc' is invalid outside string
    }

    #[test]
    fn test_grammar_keyword_completion() {
        let device = Device::Cpu;
        let fmt_obj = bloomai_core::ResponseFormat::JsonObject;
        // Vocab: 0=" ", 1="true", 2="rue", 3="xyz"
        let vocab = vec![
            " ".to_string(),
            "true".to_string(),
            "rue".to_string(),
            "xyz".to_string(),
        ];
        let eos_token_ids = vec![99u32];

        // We are at `{"hello": t`. The next token is "rue".
        // It must NOT be rejected by the whitelist outside of string since it completes `true`.
        let logits = Tensor::new(&[0.0f32, 10.0, 15.0, 1.0], &device).unwrap();
        let filtered = filter_logits_by_grammar(
            &logits,
            "{\"hello\": t",
            &fmt_obj,
            &vocab,
            &eos_token_ids,
            &device,
        )
        .unwrap();
        let filtered_vec = filtered.to_vec1::<f32>().unwrap();
        assert!(
            filtered_vec[2] > -f32::INFINITY,
            "Expected 'rue' to be allowed as it completes the boolean keyword"
        );
        assert_eq!(
            filtered_vec[3],
            -f32::INFINITY,
            "Expected 'xyz' to be filtered out"
        );
    }

    #[test]
    fn test_json_dfa_states() {
        // 1. Basic Object
        let mut dfa = JsonDfa::new();
        assert_eq!(dfa.state, DfaState::Start);

        dfa.feed("{");
        assert_eq!(dfa.state, DfaState::ObjectStart);

        dfa.feed("\"key\"");
        assert_eq!(dfa.state, DfaState::KeyCloseQuote);

        dfa.feed(":");
        assert_eq!(dfa.state, DfaState::Colon);

        dfa.feed("123");
        assert_eq!(dfa.state, DfaState::InNumberValue);

        dfa.feed("}");
        assert_eq!(dfa.state, DfaState::ObjectEnd);

        // 2. Escaped quotes and backslashes inside a string key/value
        let mut dfa_esc = JsonDfa::new();
        dfa_esc.feed("{\"escaped_\\\"key\\\"\": \"val\\\\ue\"}");
        assert_ne!(dfa_esc.state, DfaState::Invalid);
        assert_eq!(dfa_esc.state, DfaState::ObjectEnd);

        // 3. Nested arrays and objects
        let mut dfa_nested = JsonDfa::new();
        dfa_nested.feed("[1, [2, \"three\"], {\"nested\": true}]");
        assert_ne!(dfa_nested.state, DfaState::Invalid);
        assert_eq!(dfa_nested.state, DfaState::ValueEnd);
    }
}
