use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::num::NonZero;
use std::path::Path;

use anyhow::{Context, Result};
use bloomai_core::{
    constants::GIB, BloomError, DType, DeviceKind, Modality, ModelFamily, ModelFile, ModelFormat,
    ModelIoSchema, ModelManifest, QuantScheme, QuantizationInfo,
};

const GENERATION_MODEL_TASKS: &[&str] = &["generation"];
const ENCODER_MODEL_TASKS: &[&str] = &["embedding", "rerank"];
const MAX_MODEL_DIRECTORY_ENTRIES: usize = 65_536;
const MAX_SAFETENSORS_SHARDS: usize = 1_024;
const MAX_SAFETENSORS_INDEX_BYTES: u64 = 8 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SAFETENSORS_TENSOR_NAME_BYTES: usize = 4_096;

/// Return the inference tasks advertised by Bloom for a trusted model manifest.
///
/// BERT packages are encoders by definition. Other families may opt into the
/// encoder runtime through Bloom's trusted `bloom_task` manifest metadata.
/// Encoder runtimes expose both embeddings and reranking because reranking is
/// implemented from the same normalized embedding primitive.
pub fn model_manifest_tasks(manifest: &ModelManifest) -> &'static [&'static str] {
    if model_manifest_supports_embeddings(manifest) {
        ENCODER_MODEL_TASKS
    } else {
        GENERATION_MODEL_TASKS
    }
}

/// Whether a trusted model manifest selects Bloom's embedding runtime.
pub fn model_manifest_supports_embeddings(manifest: &ModelManifest) -> bool {
    manifest.family == ModelFamily::Bert
        || manifest
            .parameters
            .get("bloom_task")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|task| {
                task.eq_ignore_ascii_case("embedding") || task.eq_ignore_ascii_case("rerank")
            })
}

/// Pre-load memory estimation for a model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryEstimate {
    /// Estimated weight bytes (on disk / in memory depending on quant).
    pub weight_bytes: usize,
    /// Estimated host-resident weight bytes after mmap/offload planning.
    pub host_weight_bytes: usize,
    /// Estimated accelerator-resident weight bytes when layer offload is enabled.
    pub device_weight_bytes: usize,
    /// Estimated KV cache bytes for the given context size.
    pub kv_cache_bytes: usize,
    /// Estimated KV cache bytes per token.
    pub kv_cache_bytes_per_token: usize,
    /// Estimated temporary tensor workspace bytes.
    pub temp_tensor_bytes: usize,
    /// Total estimated memory footprint.
    pub total_bytes: usize,
    /// Dtype used for weight estimate.
    pub weight_dtype: DType,
    /// Quantization metadata used for weight estimate, if available.
    pub quantization: Option<QuantizationInfo>,
    /// Dtype used for KV cache estimate.
    pub kv_cache_dtype: DType,
    /// Number of transformer layers used in the estimate, if known.
    pub num_layers: Option<usize>,
    /// Number of layers planned for accelerator offload, if configured.
    pub offloaded_layers: Option<usize>,
    /// Whether mmap residency discount was applied to host weights.
    pub mmap_residency_applied: bool,
    /// Short explanation of what `total_bytes` represents.
    pub memory_scope: String,
}

impl MemoryEstimate {
    /// Format as a human-readable summary.
    pub fn display_summary(&self) -> String {
        format!(
            "weights={}, kv_cache={}, temp={}, total={} ({})",
            format_bytes(self.weight_bytes),
            format_bytes(self.kv_cache_bytes),
            format_bytes(self.temp_tensor_bytes),
            format_bytes(self.total_bytes),
            self.memory_scope,
        )
    }
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1}GB", b / GB)
    } else if b >= MB {
        format!("{:.0}MB", b / MB)
    } else if b >= KB {
        format!("{:.0}KB", b / KB)
    } else {
        format!("{}B", bytes)
    }
}

/// Estimate the memory footprint of a model before loading.
///
/// Uses manifest file sizes (if available), or falls back to
/// `memory_profile.min_ram_bytes`.  KV cache is estimated as
/// `context_size * kv_bytes_per_token` where the per-token cost
/// is read from `manifest.parameters["kv_cache_bytes_per_token"]`
/// or defaults to a conservative 512 KB.
pub fn estimate_memory(manifest: &ModelManifest, context_size: usize) -> MemoryEstimate {
    estimate_memory_with_weight_dtype(manifest, context_size, manifest.primary_dtype)
}

/// Estimate memory using Bloom's effective weight dtype for a target device.
///
/// Native unquantized Safetensors weights are converted to F32 by the current
/// Candle CPU kernels and to F16 on GPU unless `BLOOM_DTYPE` explicitly selects
/// another supported precision. The on-disk dtype remains available through
/// `ModelManifest::primary_dtype`.
pub fn estimate_memory_for_device(
    manifest: &ModelManifest,
    context_size: usize,
    device: DeviceKind,
) -> MemoryEstimate {
    estimate_memory_with_weight_dtype(
        manifest,
        context_size,
        effective_weight_dtype(manifest, device),
    )
}

fn estimate_memory_with_weight_dtype(
    manifest: &ModelManifest,
    context_size: usize,
    weight_dtype: DType,
) -> MemoryEstimate {
    // 1. Weight estimation
    let stored_weight_bytes = if !manifest.files.is_empty() {
        manifest.files.iter().map(|f| f.size_bytes).sum::<usize>()
    } else if manifest.memory_profile.min_ram_bytes > 0 {
        manifest.memory_profile.min_ram_bytes
    } else {
        // Estimate from model parameters if available
        let num_layers = manifest
            .parameters
            .get("num_hidden_layers")
            .or_else(|| manifest.parameters.get("num_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let hidden_size = manifest
            .parameters
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let estimated = if num_layers > 0 && hidden_size > 0 {
            let intermediate_size = manifest
                .parameters
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or((hidden_size * 4) as u64) as usize;
            let vocab_size = manifest
                .parameters
                .get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(32000) as usize;
            let params_per_layer =
                4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size;
            let total_params = num_layers * params_per_layer + 2 * vocab_size * hidden_size;
            let bytes_per_weight = bytes_per_weight(manifest);
            (total_params as f64 * bytes_per_weight) as usize
        } else {
            0
        };

        if estimated > 0 {
            estimated
        } else {
            // Unknown — use a conservative 1 GB placeholder.
            GIB as usize
        }
    };
    let weight_bytes = scale_safetensors_weight_bytes(manifest, stored_weight_bytes, weight_dtype);

    // 2. KV cache estimation: calculate from model parameters if available
    let num_layers = manifest
        .parameters
        .get("num_hidden_layers")
        .or_else(|| manifest.parameters.get("num_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    let num_kv_heads = manifest
        .parameters
        .get("num_key_value_heads")
        .or_else(|| manifest.parameters.get("num_kv_heads"))
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as usize; // default to 8 for GQA models like Llama3
    let head_dim = manifest
        .parameters
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;

    // Check if there is an explicit KV cache dtype in the environment or manifest quantization
    let kv_cache_dtype = std::env::var("BLOOM_KV_CACHE_DTYPE")
        .ok()
        .and_then(|s| match s.to_lowercase().as_str() {
            "f32" => Some(DType::F32),
            "f16" => Some(DType::F16),
            "bf16" => Some(DType::BF16),
            "q8" | "i8" => Some(DType::Q8),
            "q4" | "i4" => Some(DType::Q4),
            _ => None,
        })
        .or_else(|| {
            manifest
                .quantization
                .as_ref()
                .and_then(|q| q.kv_cache_dtype)
        });

    let kv_cache_dtype = kv_cache_dtype.unwrap_or(DType::F16);
    let bytes_per_element = kv_cache_element_bytes(kv_cache_dtype);

    // KV Cache: 2 (key & value) * num_layers * num_kv_heads * head_dim * bytes_per_element
    let computed_kv_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element;

    let kv_per_token: usize = if manifest.family == ModelFamily::Bert {
        0
    } else if manifest.parameters.contains_key("num_hidden_layers")
        || manifest.parameters.contains_key("num_layers")
    {
        computed_kv_per_token
    } else {
        manifest
            .parameters
            .get("kv_cache_bytes_per_token")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(512 * 1024)
    };
    let kv_cache_bytes = context_size * kv_per_token;

    // 3. Temp tensor workspace — ~10% of weights.
    let hidden_size = manifest
        .parameters
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;
    let temp_tensor_bytes = (weight_bytes / 10).max(context_size * hidden_size * 2);

    // Calculate VRAM offload if --gpu-layers is set
    let gpu_layers = std::env::var("BLOOM_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    let mmap_residency_applied = manifest.runtime_hints.supports_mmap && !manifest.files.is_empty();
    let resident_weight_bytes = if mmap_residency_applied {
        // Startup budget cares about resident pages, not the whole mmapped file.
        weight_bytes.saturating_mul(30) / 100
    } else {
        weight_bytes
    };

    let (host_weight_bytes, device_weight_bytes, offloaded_layers, total_bytes, memory_scope) =
        if let Some(layers) = gpu_layers {
            let offloaded_layers = layers.min(num_layers);
            if let Some(nl) = NonZero::new(num_layers) {
                let device_weight = weight_bytes.saturating_mul(offloaded_layers) / nl.get();
                let non_offloaded_weight = weight_bytes.saturating_sub(device_weight);
                let host_weight = if mmap_residency_applied {
                    non_offloaded_weight.saturating_mul(30) / 100
                } else {
                    non_offloaded_weight
                };
                let device_kv = kv_cache_bytes.saturating_mul(offloaded_layers) / nl.get();
                (
                    host_weight,
                    device_weight,
                    Some(offloaded_layers),
                    device_weight
                        .saturating_add(device_kv)
                        .saturating_add(temp_tensor_bytes),
                    "accelerator-resident estimate with layer offload".to_string(),
                )
            } else {
                (
                    resident_weight_bytes,
                    0,
                    Some(0),
                    resident_weight_bytes
                        .saturating_add(kv_cache_bytes)
                        .saturating_add(temp_tensor_bytes),
                    "host-resident estimate".to_string(),
                )
            }
        } else {
            (
                resident_weight_bytes,
                0,
                None,
                resident_weight_bytes
                    .saturating_add(kv_cache_bytes)
                    .saturating_add(temp_tensor_bytes),
                "host-resident estimate".to_string(),
            )
        };

    MemoryEstimate {
        weight_bytes,
        host_weight_bytes,
        device_weight_bytes,
        kv_cache_bytes,
        kv_cache_bytes_per_token: kv_per_token,
        temp_tensor_bytes,
        total_bytes,
        weight_dtype,
        quantization: manifest.quantization.clone(),
        kv_cache_dtype,
        num_layers: if num_layers > 0 {
            Some(num_layers)
        } else {
            None
        },
        offloaded_layers,
        mmap_residency_applied,
        memory_scope,
    }
}

fn effective_weight_dtype(manifest: &ModelManifest, device: DeviceKind) -> DType {
    let requested = std::env::var("BLOOM_DTYPE").ok().and_then(|value| {
        match value.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Some(DType::F32),
            "f16" | "float16" => Some(DType::F16),
            "bf16" | "bfloat16" => Some(DType::BF16),
            _ => None,
        }
    });
    if let Some(dtype) = requested {
        return dtype;
    }
    let native_safetensors = manifest.quantization.is_none()
        && manifest
            .files
            .iter()
            .any(|file| file.format == ModelFormat::Safetensors);
    if !native_safetensors {
        return manifest.primary_dtype;
    }
    match device {
        DeviceKind::Cpu => DType::F32,
        DeviceKind::Gpu => DType::F16,
        DeviceKind::Npu => manifest.primary_dtype,
    }
}

fn scale_safetensors_weight_bytes(
    manifest: &ModelManifest,
    stored_weight_bytes: usize,
    runtime_dtype: DType,
) -> usize {
    let native_safetensors = manifest.quantization.is_none()
        && manifest
            .files
            .iter()
            .any(|file| file.format == ModelFormat::Safetensors);
    if !native_safetensors {
        return stored_weight_bytes;
    }
    let stored_bytes = dtype_weight_bytes(manifest.primary_dtype);
    let runtime_bytes = dtype_weight_bytes(runtime_dtype);
    if stored_bytes <= 0.0 || runtime_bytes <= stored_bytes {
        stored_weight_bytes
    } else {
        (stored_weight_bytes as f64 * runtime_bytes / stored_bytes).ceil() as usize
    }
}

fn bytes_per_weight(manifest: &ModelManifest) -> f64 {
    manifest
        .quantization
        .as_ref()
        .filter(|q| q.bits > 0)
        .map(|q| q.bytes_per_element())
        .unwrap_or_else(|| dtype_weight_bytes(manifest.primary_dtype))
}

fn dtype_weight_bytes(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => 4.0,
        DType::F16 | DType::BF16 => 2.0,
        DType::Q8 | DType::I8 | DType::U8 => 1.0,
        DType::Q4 | DType::I4 | DType::NF4 => 0.5,
        _ => 2.0,
    }
}

fn kv_cache_element_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::Q8 | DType::I8 | DType::U8 => 1,
        // Current engine-side Q4 KV path is still a planning target; use
        // byte-addressable storage until packed KV kernels land.
        DType::Q4 | DType::I4 | DType::NF4 => 1,
        _ => 2,
    }
}

/// Generate concrete downgrade advices when memory allocation budget is exceeded.
pub fn suggest_memory_downgrade(est: &MemoryEstimate, avail: usize) -> Vec<String> {
    let mut suggestions = Vec::new();
    if est.total_bytes > avail {
        suggestions.push(
            "Reduce context size (e.g. set --context-size 1024 or 512 to lower KV Cache footprint)"
                .to_string(),
        );
        suggestions.push(
            "Use quantized weights (e.g. download a Q4 or Q8 quantized GGUF model)".to_string(),
        );
        suggestions.push("Enable CPU offloading or run on CPU backend (--device cpu)".to_string());
        suggestions.push(
            "Select a smaller model architecture (e.g. 1.5B or 2B parameter class instead of 7B+)"
                .to_string(),
        );
    }
    suggestions
}

/// Precision downgrade path for OOM auto-recovery.
///
/// Returns an ordered list of progressively lower-precision dtype labels
/// that the engine can try when the current precision causes OOM.
pub fn precision_downgrade_path(current_dtype: &str) -> Vec<&'static str> {
    let lower = current_dtype.to_lowercase();
    match lower.as_str() {
        "f32" | "float32" => vec!["f16", "bf16", "q8_0", "q4_0", "q2_k"],
        "f16" | "float16" => vec!["bf16", "q8_0", "q4_0", "q2_k"],
        "bf16" | "bfloat16" => vec!["q8_0", "q4_0", "q2_k"],
        "q8_0" | "q8" => vec!["q4_0", "q2_k"],
        "q4_0" | "q4_k_m" | "q4" => vec!["q2_k"],
        _ => vec![],
    }
}

/// Result of an OOM auto-downgrade attempt.
#[derive(Debug, Clone)]
pub struct DowngradeResult {
    /// The dtype that was attempted after downgrade.
    pub attempted_dtype: String,
    /// Whether the downgrade succeeded (model loaded without OOM).
    pub succeeded: bool,
    /// Log of the downgrade path taken.
    pub path_taken: Vec<String>,
}

/// Plan an automatic OOM downgrade strategy.
///
/// Given the current dtype and available memory, returns a list of dtype
/// candidates to try in order, along with their estimated memory ratio.
pub fn plan_oom_downgrade(
    current_dtype: &str,
    current_est: &MemoryEstimate,
    available_bytes: usize,
) -> Vec<(String, f64)> {
    let path = precision_downgrade_path(current_dtype);
    let mut candidates = Vec::new();

    for &dtype in &path {
        let ratio = match dtype {
            "f32" | "float32" => 1.0,
            "f16" | "float16" => 0.5,
            "bf16" | "bfloat16" => 0.5,
            "q8_0" | "q8" => 0.25,
            "q4_0" | "q4_k_m" | "q4" => 0.125,
            "q2_k" | "q2" => 0.0625,
            _ => 0.5,
        };
        let est_bytes = (current_est.weight_bytes as f64 * ratio
            + current_est.kv_cache_bytes as f64
            + current_est.temp_tensor_bytes as f64) as usize;

        candidates.push((
            dtype.to_string(),
            if available_bytes > 0 {
                est_bytes as f64 / available_bytes as f64
            } else {
                f64::MAX
            },
        ));

        if est_bytes <= available_bytes {
            // This candidate should fit
            break;
        }
    }

    candidates
}

/// Infer `QuantizationInfo` from an HF `config.json` value.
pub fn infer_quantization(config: &serde_json::Value) -> Option<QuantizationInfo> {
    let quant_config = config.get("quantization_config")?;
    let method = quant_config.get("quant_method")?.as_str()?;
    match method {
        "awq" => {
            let group_size = quant_config
                .get("group_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            Some(QuantizationInfo {
                scheme: QuantScheme::AWQ,
                bits: quant_config
                    .get("bits")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as u8,
                group_size,
                act_order: false,
                kv_cache_dtype: None,
                imatrix: false,
            })
        }
        "gptq" => {
            let group_size = quant_config
                .get("group_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let act_order = quant_config
                .get("desc_act")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            Some(QuantizationInfo {
                scheme: QuantScheme::GPTQ,
                bits: quant_config
                    .get("bits")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as u8,
                group_size,
                act_order,
                kv_cache_dtype: None,
                imatrix: false,
            })
        }
        _ => None,
    }
}

/// Metadata extracted from a GGUF header and tensor table.
///
/// This summary is intentionally independent from Candle's GGUF reader so that
/// manifest inference can be tested without carrying a large binary fixture.
#[derive(Debug, Clone, Default)]
pub struct GgufMetadataSummary {
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub attention_head_count: Option<u64>,
    pub attention_head_count_kv: Option<u64>,
    pub head_dim: Option<u64>,
    pub rope_freq_base: Option<f64>,
    pub rope_scaling_type: Option<String>,
    pub tokenizer_model: Option<String>,
    /// Bounded, recognized prompt format derived from inert GGUF template text.
    pub chat_template_kind: Option<String>,
    pub tokenizer_vocab_size: Option<u64>,
    pub bos_token_id: Option<u64>,
    pub eos_token_id: Option<u64>,
    pub quantization_type: Option<String>,
    pub file_name: Option<String>,
    pub file_size: usize,
}

const MAX_CHAT_TEMPLATE_BYTES: usize = 256 * 1024;
const MAX_TOKENIZER_CONFIG_BYTES: u64 = 512 * 1024;
const MAX_SENTENCE_TRANSFORMER_CONFIG_BYTES: u64 = 64 * 1024;

/// Classify known GGUF chat-template contracts without evaluating template code.
///
/// GGUF template source is untrusted model metadata. Bloom only scans a bounded
/// string for known token contracts and maps it to hard-coded formatters.
pub fn classify_gguf_chat_template(template: &str) -> Option<&'static str> {
    classify_chat_template(template)
}

fn classify_chat_template(template: &str) -> Option<&'static str> {
    if template.len() > MAX_CHAT_TEMPLATE_BYTES {
        return None;
    }
    if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
        if template.contains("helpful AI assistant named SmolLM") {
            Some("smollm2")
        } else {
            Some("chatml")
        }
    } else if template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
    {
        Some("llama3")
    } else if template.contains("[INST]") && template.contains("[/INST]") {
        Some("llama2")
    } else if template.contains("<start_of_turn>") && template.contains("<end_of_turn>") {
        Some("gemma")
    } else {
        None
    }
}

/// Select a deterministic primary dtype from mixed GGUF tensor metadata.
///
/// GGUF files commonly store normalization weights in F32, embeddings in Q8,
/// and transformer matrices in one or more lower-bit formats. Hash-map order is
/// not a model-level quantization signal, so use the dtype covering the most
/// transformer weight elements and keep deterministic fallbacks for unusual
/// layouts.
pub(crate) fn select_primary_gguf_dtype<'a>(
    tensors: impl IntoIterator<Item = (&'a str, String, usize)>,
) -> String {
    let mut transformer_weights = BTreeMap::<String, u128>::new();
    let mut all_weights = BTreeMap::<String, u128>::new();
    let mut all_tensors = BTreeMap::<String, u128>::new();

    for (name, dtype, element_count) in tensors {
        let weight = element_count.max(1) as u128;
        *all_tensors.entry(dtype.clone()).or_default() += weight;
        if name.contains("weight") {
            *all_weights.entry(dtype.clone()).or_default() += weight;
            if name.contains("blk") || name.contains("layers") {
                *transformer_weights.entry(dtype).or_default() += weight;
            }
        }
    }

    fn largest_dtype(counts: &BTreeMap<String, u128>) -> Option<String> {
        counts
            .iter()
            .max_by(|(left_dtype, left_count), (right_dtype, right_count)| {
                left_count
                    .cmp(right_count)
                    .then_with(|| right_dtype.cmp(left_dtype))
            })
            .map(|(dtype, _)| dtype.clone())
    }

    largest_dtype(&transformer_weights)
        .or_else(|| largest_dtype(&all_weights))
        .or_else(|| largest_dtype(&all_tensors))
        .unwrap_or_else(|| "F32".to_string())
}

/// Infer a `ModelManifest` from normalized GGUF metadata.
pub fn infer_manifest_from_gguf_summary(summary: &GgufMetadataSummary) -> ModelManifest {
    let mut manifest = ModelManifest::default();
    let arch = summary
        .architecture
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();

    manifest.id = summary
        .name
        .clone()
        .or_else(|| summary.file_name.clone())
        .unwrap_or_else(|| "gguf-model".to_string());

    manifest.family = if arch.contains("qwen") {
        ModelFamily::Qwen
    } else if arch.contains("llama") || arch.contains("mistral") || arch.contains("deepseek") {
        ModelFamily::Llama
    } else if arch.contains("gemma") {
        ModelFamily::Gemma
    } else if arch.contains("wan") {
        ModelFamily::Custom("wan".to_string())
    } else if !arch.is_empty() {
        ModelFamily::Custom(arch.clone())
    } else {
        manifest.family
    };

    let quant_type = summary
        .quantization_type
        .clone()
        .unwrap_or_else(|| "F32".to_string());
    let mut quant_info = QuantizationInfo::from_gguf_type(&quant_type);
    quant_info.imatrix = quant_type.to_lowercase().starts_with("iq")
        || summary
            .file_name
            .as_deref()
            .unwrap_or("")
            .to_lowercase()
            .contains("imatrix");
    manifest.primary_dtype = if quant_info.bits <= 4 && quant_info.bits > 0 {
        DType::Q4
    } else if quant_info.bits <= 8 && quant_info.bits > 4 {
        DType::Q8
    } else {
        DType::F32
    };
    manifest.quantization = Some(quant_info);

    let mut insert_u64 = |key: &str, value: Option<u64>| {
        if let Some(value) = value {
            manifest
                .parameters
                .insert(key.to_string(), serde_json::json!(value));
        }
    };
    insert_u64("context_length", summary.context_length);
    insert_u64("num_hidden_layers", summary.block_count);
    insert_u64("num_layers", summary.block_count);
    insert_u64("hidden_size", summary.embedding_length);
    insert_u64("num_attention_heads", summary.attention_head_count);
    insert_u64("num_key_value_heads", summary.attention_head_count_kv);
    let inferred_head_dim = summary.head_dim.or_else(|| {
        let hidden = summary.embedding_length?;
        let heads = summary.attention_head_count?;
        hidden.checked_div(heads)
    });
    insert_u64("head_dim", inferred_head_dim);
    insert_u64("vocab_size", summary.tokenizer_vocab_size);
    insert_u64("bos_token_id", summary.bos_token_id);
    insert_u64("eos_token_id", summary.eos_token_id);

    if let Some(arch) = summary.architecture.as_deref() {
        manifest
            .parameters
            .insert("gguf_architecture".to_string(), serde_json::json!(arch));
    }
    if let Some(base) = summary.rope_freq_base {
        manifest
            .parameters
            .insert("rope_theta".to_string(), serde_json::json!(base));
    }
    if let Some(scaling) = summary.rope_scaling_type.as_deref() {
        manifest
            .parameters
            .insert("rope_scaling_type".to_string(), serde_json::json!(scaling));
    }
    if let Some(tokenizer_model) = summary.tokenizer_model.as_deref() {
        manifest.parameters.insert(
            "tokenizer_model".to_string(),
            serde_json::json!(tokenizer_model),
        );
    }
    if let Some(chat_template_kind) = summary.chat_template_kind.as_deref() {
        manifest.parameters.insert(
            "chat_template_kind".to_string(),
            serde_json::json!(chat_template_kind),
        );
    }
    manifest.parameters.insert(
        "gguf_quantization_type".to_string(),
        serde_json::json!(quant_type),
    );

    if let Some(file_name) = summary.file_name.as_deref() {
        manifest.files = vec![ModelFile {
            name: file_name.to_string(),
            format: ModelFormat::Gguf,
            size_bytes: summary.file_size,
            hash_sha256: None,
            required: true,
        }];
    }

    manifest.io_schema = ModelIoSchema {
        inputs: vec![Modality::Text],
        outputs: vec![Modality::Text],
    };
    manifest.license = Some("unknown".to_string());
    manifest
}

pub fn load_manifest(model_path: &Path) -> Result<ModelManifest> {
    // Single-file GGUF path: --model points to a .gguf file directly.
    if model_path.is_file() {
        if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            #[cfg(feature = "candle-engine")]
            {
                let parent = model_path.parent().unwrap_or(Path::new("."));
                let mut manifest = infer_from_gguf(parent, model_path)?;
                // Skip strict validation for single-file GGUF (no bloom.json license).
                if manifest.license.is_none() {
                    manifest.license = Some("unknown".to_string());
                }
                return Ok(manifest);
            }
            #[cfg(not(feature = "candle-engine"))]
            {
                return Err(BloomError::UnsupportedFormat(
                    "GGUF single-file loading requires the `candle-engine` feature".into(),
                )
                .into());
            }
        }
        if model_path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
        {
            let parent = model_path.parent().unwrap_or(Path::new("."));
            return infer_from_onnx(parent, model_path);
        }
        if model_path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("spv") || ext.eq_ignore_ascii_case("spirv"))
        {
            let parent = model_path.parent().unwrap_or(Path::new("."));
            return infer_from_vulkan(parent, model_path);
        }
        if is_tensorrt_engine_file(model_path) {
            let parent = model_path.parent().unwrap_or(Path::new("."));
            return infer_from_tensorrt(parent, model_path);
        }
        return Err(BloomError::InvalidInput(format!(
            "model_path must be a directory, got file: {}",
            model_path.display()
        ))
        .into());
    }

    let manifest = if model_path.join("bloom.json").exists() {
        let content = fs::read_to_string(model_path.join("bloom.json"))?;
        serde_json::from_str(&content)?
    } else if model_path.join("config.json").exists() {
        let content = fs::read_to_string(model_path.join("config.json"))?;
        let config: serde_json::Value = serde_json::from_str(&content)?;
        infer_from_hf_config(model_path, config)?
    } else if let Some(gguf) = find_gguf_in_dir(model_path) {
        // GGUF-only directory (no config.json / bloom.json).
        #[cfg(feature = "candle-engine")]
        {
            infer_from_gguf(model_path, &gguf)?
        }
        #[cfg(not(feature = "candle-engine"))]
        {
            return Err(BloomError::UnsupportedFormat(
                "GGUF-only directory detected but `candle-engine` feature is not enabled".into(),
            )
            .into());
        }
    } else if let Some(onnx) = find_onnx_in_dir(model_path) {
        infer_from_onnx(model_path, &onnx)?
    } else if let Some(engine) = find_tensorrt_engine_in_dir(model_path) {
        infer_from_tensorrt(model_path, &engine)?
    } else if let Some(spv) = find_vulkan_spv_in_dir(model_path) {
        infer_from_vulkan(model_path, &spv)?
    } else {
        infer_from_path(model_path)?
    };

    validate_manifest(&manifest, model_path)?;
    Ok(manifest)
}

fn validate_manifest(
    manifest: &ModelManifest,
    model_path: &Path,
) -> std::result::Result<(), BloomError> {
    // Verify that required files exist.
    for file in &manifest.files {
        let path = model_path.join(&file.name);
        if file.required && !path.exists() {
            return Err(BloomError::MissingRequiredFile(file.name.clone()));
        }
        if path.exists() && file.size_bytes > 0 {
            let actual = std::fs::metadata(&path)
                .map_err(|e| BloomError::ModelLoad(e.to_string()))?
                .len() as usize;
            if actual < file.size_bytes {
                return Err(BloomError::ModelLoad(format!(
                    "file '{}' is smaller than declared size: actual={}, declared={}",
                    file.name, actual, file.size_bytes
                )));
            }
        }
    }

    // Perform a basic format-support check.
    if manifest.primary_dtype == DType::Unknown {
        return Err(BloomError::UnsupportedFormat("unknown data type".into()));
    }

    if !manifest.files.is_empty() && manifest.license.is_none() {
        return Err(BloomError::MissingLicense(manifest.id.clone()));
    }

    if manifest.id.trim().is_empty() {
        return Err(BloomError::InvalidInput(
            "manifest id cannot be empty".into(),
        ));
    }

    if manifest.io_schema.inputs.is_empty() || manifest.io_schema.outputs.is_empty() {
        return Err(BloomError::InvalidInput(
            "manifest io_schema must declare at least one input and output".into(),
        ));
    }

    Ok(())
}

fn infer_from_hf_config(model_path: &Path, config: serde_json::Value) -> Result<ModelManifest> {
    let mut manifest = ModelManifest::default();
    let arch = config
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_lowercase();
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_lowercase();

    manifest.id = model_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    if arch.contains("qwen") {
        manifest.family = ModelFamily::Qwen;
    } else if arch.contains("llama") {
        manifest.family = ModelFamily::Llama;
    } else if arch.contains("gemma") {
        manifest.family = ModelFamily::Gemma;
    } else if arch.contains("bert") || model_type == "bert" {
        manifest.family = ModelFamily::Bert;
    } else if arch.contains("wan") {
        manifest.family = ModelFamily::Custom("wan".to_string());
    } else {
        manifest.family = ModelFamily::Custom(arch.clone());
    }

    let model_name = config
        .get("model_name")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_lowercase();
    if model_name.contains("longcat-image-edit") || model_name.contains("longcat_image_edit") {
        manifest.family = ModelFamily::Custom("longcat-image-edit".to_string());
    }

    // Infer dtype
    let torch_dtype = config
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("float32");
    manifest.primary_dtype = match torch_dtype {
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        _ => DType::F32,
    };

    // Check quantization and populate QuantizationInfo
    if let Some(quant_info) = infer_quantization(&config) {
        manifest.primary_dtype = if quant_info.bits <= 4 {
            DType::Q4
        } else {
            DType::Q8
        };
        manifest.quantization = Some(quant_info);
    }

    // Populate all configuration keys into manifest parameters
    if let serde_json::Value::Object(map) = &config {
        for (k, v) in map {
            manifest.parameters.insert(k.clone(), v.clone());
        }
    }
    if manifest.family == ModelFamily::Bert {
        manifest.parameters.insert(
            "bloom_task".to_string(),
            serde_json::Value::String("embedding".to_string()),
        );
        infer_sentence_transformer_config(model_path, &mut manifest)?;
    }
    if !manifest.parameters.contains_key("head_dim") {
        let hidden_size = manifest
            .parameters
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64);
        let attention_heads = manifest
            .parameters
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64);
        if let (Some(hidden_size), Some(attention_heads)) = (hidden_size, attention_heads) {
            if attention_heads > 0 && hidden_size % attention_heads == 0 {
                manifest.parameters.insert(
                    "head_dim".to_string(),
                    serde_json::json!(hidden_size / attention_heads),
                );
            }
        }
    }

    infer_hf_safetensors_files(model_path, &mut manifest)?;
    if let Some(kind) = infer_hf_chat_template_kind(model_path)? {
        manifest.parameters.insert(
            "chat_template_kind".to_string(),
            serde_json::Value::String(kind.to_string()),
        );
    }

    let is_longcat_edit =
        matches!(&manifest.family, ModelFamily::Custom(c) if c == "longcat-image-edit");
    let is_vl = model_type.contains("vl") || arch.contains("vl");

    manifest.io_schema = ModelIoSchema {
        inputs: if is_longcat_edit || is_vl {
            vec![Modality::Text, Modality::Vision]
        } else {
            vec![Modality::Text]
        },
        outputs: if is_longcat_edit {
            vec![Modality::Vision]
        } else {
            vec![Modality::Text]
        },
    };

    if is_longcat_edit {
        manifest.license = Some("Apache-2.0".to_string());
        manifest.primary_dtype = DType::BF16;
        manifest.runtime_hints.preferred_backends = vec!["longcat".to_string()];
        manifest.runtime_hints.supports_mmap = true;
        infer_longcat_files(model_path, &mut manifest)?;
    }

    Ok(manifest)
}

fn infer_sentence_transformer_config(
    model_path: &Path,
    manifest: &mut ModelManifest,
) -> Result<()> {
    manifest.parameters.insert(
        "embedding_pooling".to_string(),
        serde_json::Value::String("mean".to_string()),
    );
    if let Some(hidden_size) = manifest
        .parameters
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
    {
        manifest.parameters.insert(
            "embedding_dimensions".to_string(),
            serde_json::Value::from(hidden_size),
        );
    }

    if let Some(value) = read_optional_sentence_transformer_json(
        &model_path.join("sentence_bert_config.json"),
        "sentence_bert_config.json",
    )? {
        if let Some(max_seq_length) = value
            .get("max_seq_length")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
        {
            manifest.parameters.insert(
                "max_seq_length".to_string(),
                serde_json::Value::from(max_seq_length),
            );
        }
    }

    let modules =
        read_optional_sentence_transformer_json(&model_path.join("modules.json"), "modules.json")?;
    let (pooling_path, normalizes) = match modules {
        Some(value) => validate_sentence_transformer_modules(&value)?,
        None => (None, false),
    };
    manifest.parameters.insert(
        "embedding_normalization".to_string(),
        serde_json::Value::String(if normalizes { "l2" } else { "none" }.to_string()),
    );

    let pooling_path = pooling_path
        .map(|path| model_path.join(path).join("config.json"))
        .or_else(|| {
            let conventional = model_path.join("1_Pooling/config.json");
            conventional.exists().then_some(conventional)
        });
    if let Some(pooling_path) = pooling_path {
        let label = pooling_path
            .strip_prefix(model_path)
            .unwrap_or(&pooling_path)
            .to_string_lossy()
            .into_owned();
        let value = read_optional_sentence_transformer_json(&pooling_path, &label)?
            .ok_or_else(|| anyhow::anyhow!("Sentence Transformers pooling config is missing"))?;
        validate_sentence_transformer_pooling(&value, manifest)?;
    }
    Ok(())
}

fn read_optional_sentence_transformer_json(
    path: &Path,
    label: &str,
) -> Result<Option<serde_json::Value>> {
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() {
        return Err(anyhow::anyhow!("{label} must be a regular file"));
    }
    if metadata.len() > MAX_SENTENCE_TRANSFORMER_CONFIG_BYTES {
        return Err(anyhow::anyhow!(
            "{label} exceeds the {} byte metadata limit",
            MAX_SENTENCE_TRANSFORMER_CONFIG_BYTES
        ));
    }
    Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
}

fn validate_sentence_transformer_modules(
    value: &serde_json::Value,
) -> Result<(Option<std::path::PathBuf>, bool)> {
    let modules = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("modules.json must contain an array"))?;
    if modules.is_empty() || modules.len() > 8 {
        return Err(anyhow::anyhow!(
            "modules.json must contain between 1 and 8 modules"
        ));
    }

    let mut transformer_seen = false;
    let mut pooling_path = None;
    let mut normalization_seen = false;
    for (position, module) in modules.iter().enumerate() {
        let module_type = module
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("modules.json module {position} has no string type"))?;
        let raw_path = module
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("modules.json module {position} has no string path"))?;
        if module_type.ends_with(".Transformer") {
            if position != 0 || transformer_seen || pooling_path.is_some() || normalization_seen {
                return Err(anyhow::anyhow!(
                    "Sentence Transformers Transformer must be the first and only transformer module"
                ));
            }
            if !raw_path.is_empty() {
                return Err(anyhow::anyhow!(
                    "Sentence Transformers Transformer must load from the model root"
                ));
            }
            transformer_seen = true;
        } else if module_type.ends_with(".Pooling") {
            if !transformer_seen || pooling_path.is_some() || normalization_seen {
                return Err(anyhow::anyhow!(
                    "Sentence Transformers Pooling must follow Transformer exactly once"
                ));
            }
            pooling_path = Some(validate_sentence_transformer_module_path(raw_path)?);
        } else if module_type.ends_with(".Normalize") {
            if pooling_path.is_none() || normalization_seen {
                return Err(anyhow::anyhow!(
                    "Sentence Transformers Normalize must follow Pooling at most once"
                ));
            }
            validate_sentence_transformer_module_path(raw_path)?;
            normalization_seen = true;
        } else {
            return Err(anyhow::anyhow!(
                "unsupported Sentence Transformers module type '{module_type}'; Bloom supports Transformer, mean Pooling, and optional Normalize"
            ));
        }
    }
    if !transformer_seen || pooling_path.is_none() {
        return Err(anyhow::anyhow!(
            "modules.json must declare Transformer followed by Pooling"
        ));
    }
    Ok((pooling_path, normalization_seen))
}

fn validate_sentence_transformer_module_path(raw_path: &str) -> Result<std::path::PathBuf> {
    let path = Path::new(raw_path);
    let safe = !raw_path.is_empty()
        && raw_path.len() <= 256
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)));
    if !safe {
        return Err(anyhow::anyhow!(
            "Sentence Transformers module path must be a safe relative path"
        ));
    }
    Ok(path.to_path_buf())
}

fn validate_sentence_transformer_pooling(
    value: &serde_json::Value,
    manifest: &mut ModelManifest,
) -> Result<()> {
    let config = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Sentence Transformers pooling config must be an object"))?;
    for (name, value) in config {
        if name.starts_with("pooling_mode_") && !value.is_boolean() {
            return Err(anyhow::anyhow!(
                "Sentence Transformers pooling option '{name}' must be a boolean"
            ));
        }
    }
    if config
        .get("pooling_mode_mean_tokens")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Err(anyhow::anyhow!(
            "unsupported Sentence Transformers pooling configuration: mean-token pooling must be enabled"
        ));
    }
    if config.iter().any(|(name, value)| {
        name.starts_with("pooling_mode_")
            && name != "pooling_mode_mean_tokens"
            && value.as_bool() == Some(true)
    }) {
        return Err(anyhow::anyhow!(
            "unsupported Sentence Transformers pooling configuration: only mean-token pooling may be enabled"
        ));
    }

    let dimensions = config
        .get("word_embedding_dimension")
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Sentence Transformers pooling config must declare a positive word_embedding_dimension"
            )
        })?;
    if manifest
        .parameters
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|hidden_size| hidden_size != dimensions)
    {
        return Err(anyhow::anyhow!(
            "Sentence Transformers pooling dimension {dimensions} does not match the encoder hidden size"
        ));
    }
    manifest.parameters.insert(
        "embedding_dimensions".to_string(),
        serde_json::Value::from(dimensions),
    );
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct HfSafetensorsIndex {
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    weight_map: BTreeMap<String, String>,
}

fn regular_file_metadata(path: &Path, description: &str) -> Result<Option<fs::Metadata>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(anyhow::anyhow!(
            "{description} must be a regular file and must not be a symbolic link: {}",
            path.display()
        ));
    }
    Ok(Some(metadata))
}

fn parse_safetensors_shard_name(name: &str) -> Result<(usize, usize)> {
    let body = name
        .strip_prefix("model-")
        .and_then(|value| value.strip_suffix(".safetensors"))
        .ok_or_else(|| anyhow::anyhow!("invalid Hugging Face Safetensors shard name: {name}"))?;
    let (index, total) = body
        .split_once("-of-")
        .ok_or_else(|| anyhow::anyhow!("invalid Hugging Face Safetensors shard name: {name}"))?;
    if index.len() != 5
        || total.len() != 5
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(anyhow::anyhow!(
            "Safetensors shards must use model-00001-of-00002.safetensors naming: {name}"
        ));
    }
    let index = index.parse::<usize>()?;
    let total = total.parse::<usize>()?;
    if index == 0 || total == 0 || index > total || total > MAX_SAFETENSORS_SHARDS {
        return Err(anyhow::anyhow!(
            "invalid Safetensors shard position in {name}; at most {MAX_SAFETENSORS_SHARDS} shards are allowed"
        ));
    }
    Ok((index, total))
}

fn read_safetensors_header(path: &Path) -> Result<(BTreeSet<String>, u64)> {
    let metadata = regular_file_metadata(path, "Safetensors shard")?
        .ok_or_else(|| anyhow::anyhow!("Safetensors shard is missing: {}", path.display()))?;
    if metadata.len() < 8 {
        return Err(anyhow::anyhow!(
            "Safetensors shard is too short to contain a header: {}",
            path.display()
        ));
    }

    let mut file = File::open(path)?;
    let mut header_length_bytes = [0_u8; 8];
    file.read_exact(&mut header_length_bytes)?;
    let header_length = u64::from_le_bytes(header_length_bytes);
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(anyhow::anyhow!(
            "Safetensors header length in {} must be between 1 and {} bytes",
            path.display(),
            MAX_SAFETENSORS_HEADER_BYTES
        ));
    }
    let data_start = 8_u64
        .checked_add(header_length)
        .ok_or_else(|| anyhow::anyhow!("Safetensors header length overflow"))?;
    if data_start > metadata.len() {
        return Err(anyhow::anyhow!(
            "Safetensors header exceeds the shard length: {}",
            path.display()
        ));
    }
    let header_length = usize::try_from(header_length)
        .map_err(|_| anyhow::anyhow!("Safetensors header does not fit in memory"))?;
    let mut header = vec![0_u8; header_length];
    file.read_exact(&mut header)?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("invalid Safetensors header in {}", path.display()))?;
    let tensors = value.as_object().ok_or_else(|| {
        anyhow::anyhow!(
            "Safetensors header must be a JSON object: {}",
            path.display()
        )
    })?;

    let data_length = metadata.len() - data_start;
    let mut names = BTreeSet::new();
    let mut ranges = Vec::new();
    let mut tensor_bytes = 0_u64;
    for (name, tensor) in tensors {
        if name == "__metadata__" {
            continue;
        }
        if name.is_empty() || name.len() > MAX_SAFETENSORS_TENSOR_NAME_BYTES {
            return Err(anyhow::anyhow!(
                "Safetensors tensor names must contain between 1 and {MAX_SAFETENSORS_TENSOR_NAME_BYTES} bytes"
            ));
        }
        let offsets = tensor
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "tensor {name:?} in {} has invalid data_offsets",
                    path.display()
                )
            })?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("tensor {name:?} has a non-integer start offset"))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("tensor {name:?} has a non-integer end offset"))?;
        if start > end || end > data_length {
            return Err(anyhow::anyhow!(
                "tensor {name:?} has out-of-bounds data offsets [{start}, {end}] in {}",
                path.display()
            ));
        }
        tensor_bytes = tensor_bytes
            .checked_add(end - start)
            .ok_or_else(|| anyhow::anyhow!("Safetensors tensor byte count overflow"))?;
        ranges.push((start, end, name));
        names.insert(name.clone());
    }
    ranges.sort_by_key(|(start, end, _)| (*start, *end));
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(anyhow::anyhow!(
                "Safetensors tensors {:?} and {:?} overlap in {}",
                pair[0].2,
                pair[1].2,
                path.display()
            ));
        }
    }
    Ok((names, tensor_bytes))
}

/// Resolve a Hugging Face Safetensors checkpoint with fail-closed shard checks.
///
/// A consolidated `model.safetensors` remains compatible with existing model
/// packages. A sharded checkpoint must include the standard index, a complete
/// consecutively numbered shard set, and an exact tensor-to-shard mapping.
pub fn resolve_hf_safetensors_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let single = model_path.join("model.safetensors");
    let index_path = model_path.join("model.safetensors.index.json");
    let single_metadata = regular_file_metadata(&single, "model.safetensors")?;
    let index_metadata = regular_file_metadata(&index_path, "Safetensors index")?;

    let mut shard_names = BTreeSet::new();
    for (entry_index, entry) in fs::read_dir(model_path)?.enumerate() {
        if entry_index >= MAX_MODEL_DIRECTORY_ENTRIES {
            return Err(anyhow::anyhow!(
                "model directory exceeds the {MAX_MODEL_DIRECTORY_ENTRIES} entry safety limit"
            ));
        }
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("model directory filenames must be valid UTF-8"))?;
        if name.starts_with("model-") && name.ends_with(".safetensors") {
            parse_safetensors_shard_name(&name)?;
            if !shard_names.insert(name.clone()) {
                return Err(anyhow::anyhow!("duplicate Safetensors shard name: {name}"));
            }
            regular_file_metadata(&entry.path(), "Safetensors shard")?
                .ok_or_else(|| anyhow::anyhow!("Safetensors shard disappeared: {name}"))?;
        }
    }

    if single_metadata.is_some() {
        if index_metadata.is_some() || !shard_names.is_empty() {
            return Err(anyhow::anyhow!(
                "model.safetensors cannot be combined with a sharded Safetensors checkpoint"
            ));
        }
        return Ok(vec![single]);
    }
    if shard_names.is_empty() && index_metadata.is_none() {
        return Ok(Vec::new());
    }
    let index_metadata = index_metadata.ok_or_else(|| {
        anyhow::anyhow!("sharded Safetensors checkpoints require model.safetensors.index.json")
    })?;
    if index_metadata.len() == 0 || index_metadata.len() > MAX_SAFETENSORS_INDEX_BYTES {
        return Err(anyhow::anyhow!(
            "Safetensors index must contain between 1 and {MAX_SAFETENSORS_INDEX_BYTES} bytes"
        ));
    }
    let index: HfSafetensorsIndex = serde_json::from_slice(&fs::read(&index_path)?)
        .context("invalid model.safetensors.index.json")?;
    if index.weight_map.is_empty() {
        return Err(anyhow::anyhow!(
            "Safetensors index weight_map must not be empty"
        ));
    }

    let mut referenced_shards = BTreeSet::new();
    for (tensor_name, shard_name) in &index.weight_map {
        if tensor_name.is_empty() || tensor_name.len() > MAX_SAFETENSORS_TENSOR_NAME_BYTES {
            return Err(anyhow::anyhow!(
                "Safetensors index tensor names must contain between 1 and {MAX_SAFETENSORS_TENSOR_NAME_BYTES} bytes"
            ));
        }
        parse_safetensors_shard_name(shard_name)?;
        referenced_shards.insert(shard_name.clone());
    }
    if referenced_shards != shard_names {
        let missing = referenced_shards
            .difference(&shard_names)
            .cloned()
            .collect::<Vec<_>>();
        let extra = shard_names
            .difference(&referenced_shards)
            .cloned()
            .collect::<Vec<_>>();
        return Err(anyhow::anyhow!(
            "Safetensors shard set does not match the index (missing: {missing:?}, unreferenced: {extra:?})"
        ));
    }

    let expected_total = shard_names
        .iter()
        .next()
        .map(|name| parse_safetensors_shard_name(name).map(|(_, total)| total))
        .transpose()?
        .unwrap_or(0);
    if shard_names.len() != expected_total {
        return Err(anyhow::anyhow!(
            "Safetensors checkpoint declares {expected_total} shards but contains {}",
            shard_names.len()
        ));
    }
    for (position, name) in shard_names.iter().enumerate() {
        let (index, total) = parse_safetensors_shard_name(name)?;
        if index != position + 1 || total != expected_total {
            return Err(anyhow::anyhow!(
                "Safetensors shard sequence is incomplete or inconsistent at {name}"
            ));
        }
    }

    let mut actual_weight_map = BTreeMap::new();
    let mut total_tensor_bytes = 0_u64;
    let mut paths = Vec::with_capacity(shard_names.len());
    for shard_name in &shard_names {
        let path = model_path.join(shard_name);
        let (tensor_names, tensor_bytes) = read_safetensors_header(&path)?;
        total_tensor_bytes = total_tensor_bytes
            .checked_add(tensor_bytes)
            .ok_or_else(|| anyhow::anyhow!("Safetensors checkpoint byte count overflow"))?;
        for tensor_name in tensor_names {
            if let Some(previous) =
                actual_weight_map.insert(tensor_name.clone(), shard_name.clone())
            {
                return Err(anyhow::anyhow!(
                    "tensor {tensor_name:?} appears in both {previous} and {shard_name}"
                ));
            }
        }
        paths.push(path);
    }
    if actual_weight_map != index.weight_map {
        return Err(anyhow::anyhow!(
            "Safetensors index weight_map does not match tensor headers"
        ));
    }
    if let Some(metadata) = index.metadata {
        let metadata = metadata
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Safetensors index metadata must be an object"))?;
        if let Some(total_size) = metadata.get("total_size") {
            let total_size = total_size.as_u64().ok_or_else(|| {
                anyhow::anyhow!("Safetensors index metadata.total_size must be an integer")
            })?;
            if total_size != total_tensor_bytes {
                return Err(anyhow::anyhow!(
                    "Safetensors index metadata.total_size is {total_size}, but tensor headers describe {total_tensor_bytes} bytes"
                ));
            }
        }
    }
    Ok(paths)
}

fn infer_hf_safetensors_files(model_path: &Path, manifest: &mut ModelManifest) -> Result<()> {
    let paths = resolve_hf_safetensors_files(model_path)?;
    if paths.is_empty() {
        return Ok(());
    }
    manifest.files = paths
        .into_iter()
        .map(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| anyhow::anyhow!("Safetensors filename must be valid UTF-8"))?;
            let size_bytes = usize::try_from(fs::metadata(&path)?.len())
                .map_err(|_| anyhow::anyhow!("Safetensors file is too large for this platform"))?;
            Ok(ModelFile {
                name: name.to_string(),
                format: ModelFormat::Safetensors,
                size_bytes,
                hash_sha256: None,
                required: true,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // A raw Hugging Face directory has no Bloom manifest from which to infer a
    // license. Keep the value explicit without inventing a license claim.
    manifest.license = Some("unknown".to_string());
    Ok(())
}

fn infer_hf_chat_template_kind(model_path: &Path) -> Result<Option<&'static str>> {
    let path = model_path.join("tokenizer_config.json");
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() {
        return Err(anyhow::anyhow!(
            "tokenizer_config.json must be a regular file"
        ));
    }
    if metadata.len() > MAX_TOKENIZER_CONFIG_BYTES {
        return Err(anyhow::anyhow!(
            "tokenizer_config.json exceeds the {} byte metadata limit",
            MAX_TOKENIZER_CONFIG_BYTES
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let template = match value.get("chat_template") {
        Some(serde_json::Value::String(template)) => Some(template.as_str()),
        Some(serde_json::Value::Array(templates)) => templates
            .iter()
            .find(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some("default"))
            .or_else(|| templates.first())
            .and_then(|entry| entry.get("template"))
            .and_then(serde_json::Value::as_str),
        Some(serde_json::Value::Object(templates)) => templates
            .get("default")
            .or_else(|| templates.values().next())
            .and_then(serde_json::Value::as_str),
        _ => None,
    };
    Ok(template.and_then(classify_chat_template))
}

fn infer_longcat_files(model_path: &Path, manifest: &mut ModelManifest) -> Result<()> {
    use bloomai_core::{ModelFile, ModelFormat};

    let candidates = [
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model-00001-of-00005.safetensors",
        "text_encoder/model-00002-of-00005.safetensors",
        "text_encoder/model-00003-of-00005.safetensors",
        "text_encoder/model-00004-of-00005.safetensors",
        "text_encoder/model-00005-of-00005.safetensors",
        "text_encoder/model.safetensors.index.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
        "text_processor/tokenizer.json",
    ];

    manifest.files.clear();
    for name in candidates {
        let size_bytes = std::fs::metadata(model_path.join(name))
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        manifest.files.push(ModelFile {
            name: name.to_string(),
            format: if name.ends_with(".safetensors") {
                ModelFormat::Safetensors
            } else {
                ModelFormat::Unknown
            },
            size_bytes,
            hash_sha256: None,
            required: true,
        });
    }

    Ok(())
}

/// Find the first `.gguf` file in a directory (non-recursive).
fn find_gguf_in_dir(model_path: &Path) -> Option<std::path::PathBuf> {
    if !model_path.is_dir() {
        return None;
    }
    std::fs::read_dir(model_path)
        .ok()?
        .flatten()
        .find(|e| {
            let p = e.path();
            p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("gguf")
        })
        .map(|e| e.path())
}

/// Find the preferred `.onnx` file in a directory (non-recursive).
fn find_onnx_in_dir(model_path: &Path) -> Option<std::path::PathBuf> {
    if !model_path.is_dir() {
        return None;
    }

    for name in ["model.onnx", "encoder.onnx", "decoder.onnx"] {
        let candidate = model_path.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let mut candidates = std::fs::read_dir(model_path)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

/// Find the preferred TensorRT engine file in a directory (non-recursive).
fn find_tensorrt_engine_in_dir(model_path: &Path) -> Option<std::path::PathBuf> {
    if !model_path.is_dir() {
        return None;
    }

    for name in ["model.engine", "model.plan", "engine.plan"] {
        let candidate = model_path.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let mut candidates = std::fs::read_dir(model_path)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_tensorrt_engine_file(path))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

fn is_tensorrt_engine_file(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "engine" | "plan"))
}

fn infer_from_onnx(model_path: &Path, onnx_path: &Path) -> Result<ModelManifest> {
    let mut manifest = ModelManifest {
        id: onnx_path
            .file_stem()
            .or_else(|| model_path.file_name())
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        family: ModelFamily::Custom("onnx".to_string()),
        primary_dtype: DType::F32,
        io_schema: ModelIoSchema {
            inputs: vec![Modality::Multi],
            outputs: vec![Modality::Multi],
        },
        license: Some("unknown".to_string()),
        ..ModelManifest::default()
    };

    let name = onnx_path
        .strip_prefix(model_path)
        .unwrap_or(onnx_path)
        .to_string_lossy()
        .into_owned();
    let size_bytes = std::fs::metadata(onnx_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    manifest.files = vec![ModelFile {
        name,
        format: ModelFormat::Onnx,
        size_bytes,
        hash_sha256: None,
        required: true,
    }];
    manifest
        .runtime_hints
        .preferred_backends
        .push("onnxruntime".to_string());
    Ok(manifest)
}

fn infer_from_tensorrt(model_path: &Path, engine_path: &Path) -> Result<ModelManifest> {
    let mut manifest = ModelManifest {
        id: engine_path
            .file_stem()
            .or_else(|| model_path.file_name())
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        family: ModelFamily::Custom("tensorrt".to_string()),
        primary_dtype: DType::F16,
        io_schema: ModelIoSchema {
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Text],
        },
        license: Some("unknown".to_string()),
        ..ModelManifest::default()
    };

    let name = engine_path
        .strip_prefix(model_path)
        .unwrap_or(engine_path)
        .to_string_lossy()
        .into_owned();
    let size_bytes = std::fs::metadata(engine_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    manifest.files = vec![ModelFile {
        name,
        format: ModelFormat::TensorRtEngine,
        size_bytes,
        hash_sha256: None,
        required: true,
    }];
    manifest
        .runtime_hints
        .preferred_backends
        .push("tensorrt".to_string());
    Ok(manifest)
}

fn find_vulkan_spv_in_dir(model_path: &Path) -> Option<std::path::PathBuf> {
    if !model_path.is_dir() {
        return None;
    }
    std::fs::read_dir(model_path)
        .ok()?
        .flatten()
        .find(|e| {
            let p = e.path();
            if p.is_file() {
                if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                    return ext.eq_ignore_ascii_case("spv") || ext.eq_ignore_ascii_case("spirv");
                }
            }
            false
        })
        .map(|e| e.path())
}

fn infer_from_vulkan(model_path: &Path, spv_path: &Path) -> Result<ModelManifest> {
    let mut manifest = ModelManifest {
        id: spv_path
            .file_stem()
            .or_else(|| model_path.file_name())
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        family: ModelFamily::Custom("vulkan".to_string()),
        primary_dtype: DType::F16,
        io_schema: ModelIoSchema {
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Text],
        },
        license: Some("unknown".to_string()),
        ..ModelManifest::default()
    };

    let name = spv_path
        .strip_prefix(model_path)
        .unwrap_or(spv_path)
        .to_string_lossy()
        .into_owned();
    let size_bytes = std::fs::metadata(spv_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    manifest.files = vec![ModelFile {
        name,
        format: ModelFormat::VulkanSpirv,
        size_bytes,
        hash_sha256: None,
        required: true,
    }];
    manifest
        .runtime_hints
        .preferred_backends
        .push("vulkan".to_string());
    Ok(manifest)
}

/// Infer a `ModelManifest` from a GGUF file's header metadata.
///
/// Requires the `candle-engine` feature (for `candle_core::quantized::gguf_file`).
#[cfg(feature = "candle-engine")]
fn infer_from_gguf(_model_path: &Path, gguf_path: &Path) -> Result<ModelManifest> {
    use candle_core::quantized::gguf_file::Content;

    let mut file = std::fs::File::open(gguf_path).map_err(|e| {
        BloomError::Engine(format!("failed to open GGUF file {:?}: {}", gguf_path, e))
    })?;
    let content = Content::read(&mut file).map_err(|e| {
        let err_msg = e.to_string();
        if err_msg.contains("unknown dtype for tensor") {
            let dtype_id = err_msg
                .split_whitespace()
                .last()
                .unwrap_or("");
            let filename = gguf_path.file_name().unwrap_or_default().to_string_lossy();
            let tip = format!(
                "unsupported GGUF quantization type (dtype ID: {}) in this Candle build. [Diagnostic Tip] Re-quantize to Q4_K_M or Q8_0 using llama-quantize: `llama-quantize {} {} Q4_K_M`",
                dtype_id,
                filename,
                filename.replace("iq", "q4_k_m")
            );
            BloomError::UnsupportedFormat(tip)
        } else {
            BloomError::Engine(format!("failed to read GGUF header: {}", e))
        }
    })?;

    let name = content
        .metadata
        .get("general.name")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
        .or_else(|| {
            gguf_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        });

    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
        .unwrap_or_default();
    let arch_lower = arch.to_lowercase();

    // Early architecture validation
    let arch_supported = matches!(
        arch_lower.as_str(),
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
    if !arch_supported && !arch_lower.is_empty() {
        return Err(BloomError::UnsupportedFamily(format!(
            "unsupported GGUF architecture: '{}'. Supported architectures are: llama, qwen, qwen2, qwen3, gemma, mistral, deepseek. [Diagnostic Tip] Run `inspect_gguf` to verify model structure.",
            arch
        )).into());
    }

    let ctx_key = if arch_lower.is_empty() {
        "general.context_length".to_string()
    } else {
        format!("{}.context_length", arch_lower)
    };
    let context_length = content
        .metadata
        .get(&ctx_key)
        .or_else(|| content.metadata.get("general.context_length"))
        .and_then(|ctx_val| ctx_val.to_u64().ok());

    let meta_u64 = |suffix: &str| -> Option<u64> {
        if arch_lower.is_empty() {
            return None;
        }
        content
            .metadata
            .get(&format!("{}.{}", arch_lower, suffix))
            .and_then(|v| v.to_u64().ok())
    };
    let meta_f64 = |suffix: &str| -> Option<f64> {
        if arch_lower.is_empty() {
            return None;
        }
        content
            .metadata
            .get(&format!("{}.{}", arch_lower, suffix))
            .and_then(|v| v.to_f64().ok())
    };
    let meta_str = |suffix: &str| -> Option<String> {
        if arch_lower.is_empty() {
            return None;
        }
        content
            .metadata
            .get(&format!("{}.{}", arch_lower, suffix))
            .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
    };

    let gguf_type_str =
        select_primary_gguf_dtype(content.tensor_infos.iter().map(|(name, info)| {
            (
                name.as_str(),
                format!("{:?}", info.ggml_dtype),
                info.shape.elem_count(),
            )
        }));

    let file_size = std::fs::metadata(gguf_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let gguf_filename = gguf_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    let tokenizer_vocab_size =
        content
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u64),
                _ => None,
            });

    let summary = GgufMetadataSummary {
        name,
        architecture: if arch.is_empty() { None } else { Some(arch) },
        context_length,
        block_count: meta_u64("block_count"),
        embedding_length: meta_u64("embedding_length"),
        attention_head_count: meta_u64("attention.head_count"),
        attention_head_count_kv: meta_u64("attention.head_count_kv"),
        head_dim: None,
        rope_freq_base: meta_f64("rope.freq_base"),
        rope_scaling_type: meta_str("rope.scaling.type"),
        tokenizer_model: content
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok().map(|s| s.to_string())),
        chat_template_kind: content
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|value| match value {
                candle_core::quantized::gguf_file::Value::String(template) => {
                    classify_gguf_chat_template(template).map(str::to_string)
                }
                _ => None,
            }),
        tokenizer_vocab_size,
        bos_token_id: content
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u64().ok()),
        eos_token_id: content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u64().ok()),
        quantization_type: Some(gguf_type_str),
        file_name: Some(gguf_filename),
        file_size,
    };

    Ok(infer_manifest_from_gguf_summary(&summary))
}

fn infer_from_path(model_path: &Path) -> Result<ModelManifest> {
    let mut manifest = ModelManifest::default();
    let path_str = model_path.to_string_lossy().to_lowercase();
    manifest.id = model_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    if path_str.contains("qwen") {
        manifest.family = ModelFamily::Qwen;
    } else if path_str.contains("llama") {
        manifest.family = ModelFamily::Llama;
    } else if path_str.contains("gemma") {
        manifest.family = ModelFamily::Gemma;
    } else if path_str.contains("funasr") || path_str.contains("whisper") {
        manifest.family = ModelFamily::FunAsr;
        manifest.io_schema = ModelIoSchema {
            inputs: vec![Modality::Audio],
            outputs: vec![Modality::Text],
        };
    } else if path_str.contains("wan") {
        manifest.family = ModelFamily::Custom("wan".to_string());
        manifest.io_schema = ModelIoSchema {
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Vision],
        };
    } else if path_str.contains("cosyvoice")
        || path_str.contains("tts")
        || path_str.contains("chattts")
    {
        manifest.family = ModelFamily::Custom("tts".to_string());
        manifest.io_schema = ModelIoSchema {
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Audio],
        };
    }

    // GGUF quantization suffix patterns (llama.cpp naming convention)
    let gguf_quants = [
        "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1",
        "q5_k_s", "q5_k_m", "q6_k", "q8_0", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq3_s", "iq4_xs",
        "iq4_nl",
    ];
    let matched_gguf = gguf_quants.iter().find(|q| path_str.contains(*q));

    if let Some(qstr) = matched_gguf {
        let upper = qstr.to_uppercase();
        let quant_info = QuantizationInfo::from_gguf_type(&upper);
        manifest.primary_dtype = if quant_info.bits <= 4 {
            DType::Q4
        } else {
            DType::Q8
        };
        manifest.quantization = Some(quant_info);
    } else if path_str.contains("q4") || path_str.contains("int4") || path_str.contains("awq") {
        manifest.primary_dtype = DType::Q4;
        manifest.quantization = Some(QuantizationInfo {
            scheme: if path_str.contains("awq") {
                QuantScheme::AWQ
            } else {
                QuantScheme::INT4
            },
            bits: 4,
            group_size: None,
            act_order: false,
            kv_cache_dtype: None,
            imatrix: false,
        });
    } else if path_str.contains("q8") || path_str.contains("int8") {
        manifest.primary_dtype = DType::Q8;
        manifest.quantization = Some(QuantizationInfo {
            scheme: QuantScheme::INT8,
            bits: 8,
            group_size: None,
            act_order: false,
            kv_cache_dtype: None,
            imatrix: false,
        });
    } else if path_str.contains("fp16") {
        manifest.primary_dtype = DType::F16;
    }

    Ok(manifest)
}

/// Verify SHA-256 hashes of model files against the manifest.
///
/// Returns `Ok(())` if all files with a declared hash match, or an error
/// describing the first mismatch.  Files without a declared hash are skipped.
pub fn verify_model_hashes(model_dir: &Path, manifest: &ModelManifest) -> Result<()> {
    use sha2::{Digest, Sha256};

    for file in &manifest.files {
        let expected = match &file.hash_sha256 {
            Some(h) if !h.is_empty() => h.to_lowercase(),
            _ => continue,
        };
        let path = model_dir.join(&file.name);
        if !path.exists() {
            if file.required {
                return Err(BloomError::MissingRequiredFile(file.name.clone()).into());
            }
            continue;
        }
        let bytes = fs::read(&path).map_err(|e| {
            BloomError::Engine(format!("failed to read model file '{}': {}", file.name, e))
        })?;
        let computed = format!("{:x}", Sha256::digest(&bytes));
        if computed != expected {
            return Err(BloomError::HashMismatch(format!(
                "hash mismatch for '{}': expected {}, got {}",
                file.name, expected, computed
            ))
            .into());
        }
    }
    Ok(())
}

/// Validate that the model path is within an allowed base directory (no path traversal).
///
/// Returns the canonicalized model path on success.
pub fn validate_model_path(model_path: &Path) -> Result<std::path::PathBuf> {
    let canonical = model_path.canonicalize().map_err(|e| {
        BloomError::Engine(format!(
            "failed to canonicalize model path '{}': {}",
            model_path.display(),
            e
        ))
    })?;
    // Reject paths that contain suspicious components
    for component in canonical.components() {
        if let std::path::Component::Normal(c) = component {
            let s = c.to_string_lossy();
            if s.contains("..") {
                return Err(BloomError::InvalidInput(format!(
                    "model path contains path traversal component: {}",
                    canonical.display()
                ))
                .into());
            }
        }
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloomai_core::{ModelFile, ModelFormat, ModelMemoryProfile};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn model_tasks_have_one_trusted_manifest_classifier() {
        let generation = ModelManifest {
            family: ModelFamily::Llama,
            ..ModelManifest::default()
        };
        assert_eq!(model_manifest_tasks(&generation), ["generation"]);
        assert!(!model_manifest_supports_embeddings(&generation));

        let bert = ModelManifest {
            family: ModelFamily::Bert,
            ..ModelManifest::default()
        };
        assert_eq!(model_manifest_tasks(&bert), ["embedding", "rerank"]);
        assert!(model_manifest_supports_embeddings(&bert));

        let mut declared_encoder = generation;
        declared_encoder
            .parameters
            .insert("bloom_task".to_string(), serde_json::json!("ReRaNk"));
        assert_eq!(
            model_manifest_tasks(&declared_encoder),
            ["embedding", "rerank"]
        );
    }

    #[test]
    fn primary_gguf_dtype_is_element_weighted_and_order_independent() {
        let tensors = || {
            vec![
                ("blk.0.ffn_down.weight", "Q4_1".to_string(), 4_000_000),
                ("blk.0.ffn_gate.weight", "Q4_0".to_string(), 4_000_000),
                ("blk.0.ffn_up.weight", "Q4_0".to_string(), 4_000_000),
                ("blk.0.attn_norm.weight", "F32".to_string(), 896),
                ("token_embd.weight", "Q8_0".to_string(), 136_000_000),
            ]
        };

        assert_eq!(select_primary_gguf_dtype(tensors()), "Q4_0");
        let mut reversed = tensors();
        reversed.reverse();
        assert_eq!(select_primary_gguf_dtype(reversed), "Q4_0");
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn base_manifest_with_files(sizes: &[usize]) -> ModelManifest {
        ModelManifest {
            files: sizes
                .iter()
                .enumerate()
                .map(|(i, &sz)| ModelFile {
                    name: format!("model-{}.safetensors", i),
                    format: ModelFormat::Safetensors,
                    size_bytes: sz,
                    hash_sha256: None,
                    required: false,
                })
                .collect(),
            ..ModelManifest::default()
        }
    }

    fn write_test_safetensors(path: &Path, tensor_names: &[&str]) {
        let mut header = BTreeMap::new();
        for (position, tensor_name) in tensor_names.iter().enumerate() {
            let start = position * 4;
            header.insert(
                *tensor_name,
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [start, start + 4],
                }),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + tensor_names.len() * 4, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_test_safetensors_index(
        directory: &Path,
        weight_map: &[(&str, &str)],
        total_size: u64,
    ) {
        let weight_map = weight_map
            .iter()
            .map(|(tensor, shard)| ((*tensor).to_string(), (*shard).to_string()))
            .collect::<BTreeMap<_, _>>();
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn resolves_complete_indexed_safetensors_shards_in_order() {
        let directory = tempfile::tempdir().unwrap();
        let shard_one = "model-00001-of-00002.safetensors";
        let shard_two = "model-00002-of-00002.safetensors";
        write_test_safetensors(&directory.path().join(shard_two), &["model.norm.weight"]);
        write_test_safetensors(
            &directory.path().join(shard_one),
            &["model.embed_tokens.weight"],
        );
        write_test_safetensors_index(
            directory.path(),
            &[
                ("model.embed_tokens.weight", shard_one),
                ("model.norm.weight", shard_two),
            ],
            8,
        );

        let resolved = resolve_hf_safetensors_files(directory.path()).unwrap();
        assert_eq!(
            resolved,
            [
                directory.path().join(shard_one),
                directory.path().join(shard_two)
            ]
        );
    }

    #[test]
    fn rejects_sharded_safetensors_without_an_index() {
        let directory = tempfile::tempdir().unwrap();
        write_test_safetensors(
            &directory.path().join("model-00001-of-00001.safetensors"),
            &["weight"],
        );

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error
            .to_string()
            .contains("require model.safetensors.index.json"));
    }

    #[test]
    fn rejects_ambiguous_single_file_and_sharded_safetensors_layouts() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("model.safetensors"), b"weights").unwrap();
        let shard = "model-00001-of-00001.safetensors";
        write_test_safetensors(&directory.path().join(shard), &["weight"]);
        write_test_safetensors_index(directory.path(), &[("weight", shard)], 4);

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error
            .to_string()
            .contains("cannot be combined with a sharded"));
    }

    #[test]
    fn rejects_noncanonical_safetensors_shard_names() {
        let directory = tempfile::tempdir().unwrap();
        write_test_safetensors(
            &directory.path().join("model-1-of-2.safetensors"),
            &["weight"],
        );

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error.to_string().contains("model-00001-of-00002"));
    }

    #[test]
    fn rejects_incomplete_safetensors_shard_sets() {
        let directory = tempfile::tempdir().unwrap();
        let shard_one = "model-00001-of-00002.safetensors";
        let shard_two = "model-00002-of-00002.safetensors";
        write_test_safetensors(&directory.path().join(shard_one), &["weight.one"]);
        write_test_safetensors_index(
            directory.path(),
            &[("weight.one", shard_one), ("weight.two", shard_two)],
            8,
        );

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error.to_string().contains("missing"));
        assert!(error.to_string().contains(shard_two));
    }

    #[test]
    fn rejects_safetensors_index_tensor_mapping_mismatches() {
        let directory = tempfile::tempdir().unwrap();
        let shard_one = "model-00001-of-00002.safetensors";
        let shard_two = "model-00002-of-00002.safetensors";
        write_test_safetensors(&directory.path().join(shard_one), &["weight.one"]);
        write_test_safetensors(&directory.path().join(shard_two), &["weight.two"]);
        write_test_safetensors_index(
            directory.path(),
            &[("weight.one", shard_two), ("weight.two", shard_one)],
            8,
        );

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error.to_string().contains("does not match tensor headers"));
    }

    #[test]
    fn rejects_duplicate_tensors_across_safetensors_shards() {
        let directory = tempfile::tempdir().unwrap();
        let shard_one = "model-00001-of-00002.safetensors";
        let shard_two = "model-00002-of-00002.safetensors";
        write_test_safetensors(&directory.path().join(shard_one), &["weight"]);
        write_test_safetensors(
            &directory.path().join(shard_two),
            &["other_weight", "weight"],
        );
        write_test_safetensors_index(
            directory.path(),
            &[("weight", shard_one), ("other_weight", shard_two)],
            12,
        );

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error.to_string().contains("appears in both"));
    }

    #[test]
    fn rejects_safetensors_index_total_size_mismatches() {
        let directory = tempfile::tempdir().unwrap();
        let shard = "model-00001-of-00001.safetensors";
        write_test_safetensors(&directory.path().join(shard), &["weight"]);
        write_test_safetensors_index(directory.path(), &[("weight", shard)], 5);

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error.to_string().contains("describe 4 bytes"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symbolic_link_safetensors_shards() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let shard = "model-00001-of-00001.safetensors";
        write_test_safetensors(&directory.path().join("weights.safetensors"), &["weight"]);
        symlink("weights.safetensors", directory.path().join(shard)).unwrap();
        write_test_safetensors_index(directory.path(), &[("weight", shard)], 4);

        let error = resolve_hf_safetensors_files(directory.path()).unwrap_err();
        assert!(error.to_string().contains("symbolic link"));
    }

    #[test]
    fn test_estimate_memory_from_files() {
        let m = base_manifest_with_files(&[1_000_000_000, 500_000_000]);
        let est = estimate_memory(&m, 2048);
        // weights = 1.5 GB
        assert_eq!(est.weight_bytes, 1_500_000_000);
        // kv = 2048 * 512KB = 1 GB
        assert_eq!(est.kv_cache_bytes, 2048 * 512 * 1024);
        // temp = 10% of weights
        assert_eq!(est.temp_tensor_bytes, 150_000_000);
        assert_eq!(
            est.total_bytes,
            est.weight_bytes + est.kv_cache_bytes + est.temp_tensor_bytes
        );
    }

    #[test]
    fn test_estimate_memory_fallback_to_min_ram() {
        let m = ModelManifest {
            memory_profile: ModelMemoryProfile {
                min_ram_bytes: 4_000_000_000,
                min_vram_bytes: 0,
                recommended_ram_bytes: 0,
                recommended_vram_bytes: 0,
            },
            ..ModelManifest::default()
        };
        let est = estimate_memory(&m, 1024);
        assert_eq!(est.weight_bytes, 4_000_000_000);
        assert_eq!(est.kv_cache_bytes, 1024 * 512 * 1024);
    }

    #[test]
    fn test_estimate_memory_unknown_fallback() {
        let m = ModelManifest::default();
        let est = estimate_memory(&m, 512);
        // Unknown: 1 GB placeholder
        assert_eq!(est.weight_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_infer_from_path_gguf_q4_k_m() {
        let p = Path::new("/models/qwen2-7b-instruct-q4_k_m.gguf");
        let m = infer_from_path(p).unwrap();
        assert_eq!(m.family, ModelFamily::Qwen);
        assert_eq!(m.primary_dtype, DType::Q4);
        let q = m.quantization.as_ref().unwrap();
        assert!(matches!(q.scheme, QuantScheme::GGUF(ref s) if s == "Q4_K_M"));
        assert_eq!(q.bits, 4);
    }

    #[test]
    fn test_infer_from_path_gguf_q8_0() {
        let p = Path::new("/models/llama-3-8b-q8_0.gguf");
        let m = infer_from_path(p).unwrap();
        assert_eq!(m.family, ModelFamily::Llama);
        assert_eq!(m.primary_dtype, DType::Q8);
        let q = m.quantization.as_ref().unwrap();
        assert!(matches!(q.scheme, QuantScheme::GGUF(ref s) if s == "Q8_0"));
        assert_eq!(q.bits, 8);
    }

    #[test]
    fn test_infer_from_path_gemma_q5_k_m() {
        let p = Path::new("/models/gemma-2b-it-q5_k_m.gguf");
        let m = infer_from_path(p).unwrap();
        assert_eq!(m.family, ModelFamily::Gemma);
        assert_eq!(m.primary_dtype, DType::Q8); // 5 bits → Q8 bucket
        let q = m.quantization.as_ref().unwrap();
        assert!(matches!(q.scheme, QuantScheme::GGUF(ref s) if s == "Q5_K_M"));
    }

    #[test]
    fn test_load_manifest_single_onnx_file() {
        let dir = tempfile::tempdir().unwrap();
        let onnx = dir.path().join("classifier.onnx");
        std::fs::write(&onnx, b"placeholder").unwrap();

        let manifest = load_manifest(&onnx).unwrap();
        assert_eq!(manifest.id, "classifier");
        assert_eq!(manifest.family, ModelFamily::Custom("onnx".to_string()));
        assert_eq!(manifest.files[0].format, ModelFormat::Onnx);
        assert_eq!(manifest.files[0].name, "classifier.onnx");
        assert_eq!(manifest.io_schema.inputs, vec![Modality::Multi]);
        assert!(manifest
            .runtime_hints
            .preferred_backends
            .contains(&"onnxruntime".to_string()));
    }

    #[test]
    fn test_load_manifest_onnx_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"placeholder").unwrap();

        let manifest = load_manifest(dir.path()).unwrap();
        assert_eq!(manifest.files[0].format, ModelFormat::Onnx);
        assert_eq!(manifest.files[0].name, "model.onnx");
    }

    #[test]
    fn test_load_manifest_single_tensorrt_engine_file() {
        let dir = tempfile::tempdir().unwrap();
        let engine = dir.path().join("llama.plan");
        std::fs::write(&engine, b"placeholder").unwrap();

        let manifest = load_manifest(&engine).unwrap();
        assert_eq!(manifest.id, "llama");
        assert_eq!(manifest.family, ModelFamily::Custom("tensorrt".to_string()));
        assert_eq!(manifest.files[0].format, ModelFormat::TensorRtEngine);
        assert_eq!(manifest.files[0].name, "llama.plan");
        assert_eq!(manifest.io_schema.inputs, vec![Modality::Text]);
        assert!(manifest
            .runtime_hints
            .preferred_backends
            .contains(&"tensorrt".to_string()));
    }

    #[test]
    fn test_load_manifest_tensorrt_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.engine"), b"placeholder").unwrap();

        let manifest = load_manifest(dir.path()).unwrap();
        assert_eq!(manifest.files[0].format, ModelFormat::TensorRtEngine);
        assert_eq!(manifest.files[0].name, "model.engine");
    }

    #[test]
    fn test_infer_manifest_from_gguf_summary_llama_metadata() {
        let summary = GgufMetadataSummary {
            name: Some("tiny-llama-q4".to_string()),
            architecture: Some("llama".to_string()),
            context_length: Some(4096),
            block_count: Some(32),
            embedding_length: Some(4096),
            attention_head_count: Some(32),
            attention_head_count_kv: Some(8),
            rope_freq_base: Some(500000.0),
            tokenizer_model: Some("llama".to_string()),
            chat_template_kind: Some("llama3".to_string()),
            tokenizer_vocab_size: Some(32000),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            quantization_type: Some("Q4_K_M".to_string()),
            file_name: Some("tiny-llama-q4.gguf".to_string()),
            file_size: 42,
            ..Default::default()
        };

        let manifest = infer_manifest_from_gguf_summary(&summary);
        assert_eq!(manifest.id, "tiny-llama-q4");
        assert_eq!(manifest.family, ModelFamily::Llama);
        assert_eq!(manifest.primary_dtype, DType::Q4);
        assert_eq!(manifest.files[0].format, ModelFormat::Gguf);
        assert_eq!(
            manifest.parameters.get("context_length"),
            Some(&serde_json::json!(4096))
        );
        assert_eq!(
            manifest.parameters.get("head_dim"),
            Some(&serde_json::json!(128))
        );
        assert_eq!(
            manifest.parameters.get("rope_theta"),
            Some(&serde_json::json!(500000.0))
        );
        assert_eq!(
            manifest.parameters.get("tokenizer_model"),
            Some(&serde_json::json!("llama"))
        );
        assert_eq!(
            manifest.parameters.get("chat_template_kind"),
            Some(&serde_json::json!("llama3"))
        );
        assert_eq!(
            manifest.parameters.get("vocab_size"),
            Some(&serde_json::json!(32000))
        );
        let quant = manifest.quantization.as_ref().unwrap();
        assert!(matches!(quant.scheme, QuantScheme::GGUF(ref s) if s == "Q4_K_M"));
    }

    #[test]
    fn test_classify_gguf_chat_template_is_bounded_and_non_executing() {
        assert_eq!(
            classify_gguf_chat_template(
                "{% for message in messages %}<|im_start|>{{ message.role }}<|im_end|>{% endfor %}"
            ),
            Some("chatml")
        );
        assert_eq!(
            classify_gguf_chat_template("<|im_start|>helpful AI assistant named SmolLM<|im_end|>"),
            Some("smollm2")
        );
        assert_eq!(
            classify_gguf_chat_template(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|><|eot_id|>"
            ),
            Some("llama3")
        );
        assert_eq!(
            classify_gguf_chat_template("[INST] x [/INST]"),
            Some("llama2")
        );
        assert_eq!(classify_gguf_chat_template("{{ dangerous() }}"), None);
        assert_eq!(
            classify_gguf_chat_template(&"<|im_start|><|im_end|>".repeat(20_000)),
            None
        );
    }

    #[test]
    fn hf_manifest_records_safetensors_and_classifies_bounded_template_metadata() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 16
            }"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        fs::write(
            dir.path().join("tokenizer_config.json"),
            br#"{"chat_template":"<|im_start|>You are a helpful AI assistant named SmolLM<|im_end|>"}"#,
        )
        .unwrap();

        let manifest = load_manifest(dir.path()).unwrap();
        assert_eq!(manifest.family, ModelFamily::Llama);
        assert_eq!(manifest.primary_dtype, DType::BF16);
        assert_eq!(manifest.license.as_deref(), Some("unknown"));
        assert_eq!(manifest.files.len(), 1);
        assert_eq!(manifest.files[0].name, "model.safetensors");
        assert_eq!(manifest.files[0].format, ModelFormat::Safetensors);
        assert_eq!(manifest.files[0].size_bytes, 7);
        assert!(manifest.files[0].required);
        assert_eq!(
            manifest.parameters.get("head_dim"),
            Some(&serde_json::json!(4))
        );
        assert_eq!(
            manifest.parameters.get("chat_template_kind"),
            Some(&serde_json::json!("smollm2"))
        );
    }

    #[test]
    fn hf_manifest_records_every_validated_safetensors_shard() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("config.json"),
            br#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "torch_dtype": "float32",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "num_attention_heads": 2
            }"#,
        )
        .unwrap();
        let shard_one = "model-00001-of-00002.safetensors";
        let shard_two = "model-00002-of-00002.safetensors";
        write_test_safetensors(&directory.path().join(shard_one), &["weight.one"]);
        write_test_safetensors(&directory.path().join(shard_two), &["weight.two"]);
        write_test_safetensors_index(
            directory.path(),
            &[("weight.one", shard_one), ("weight.two", shard_two)],
            8,
        );

        let manifest = load_manifest(directory.path()).unwrap();
        assert_eq!(manifest.files.len(), 2);
        assert_eq!(manifest.files[0].name, shard_one);
        assert_eq!(manifest.files[1].name, shard_two);
        assert_eq!(
            manifest
                .files
                .iter()
                .map(|file| file.size_bytes)
                .sum::<usize>(),
            fs::metadata(directory.path().join(shard_one))
                .unwrap()
                .len() as usize
                + fs::metadata(directory.path().join(shard_two))
                    .unwrap()
                    .len() as usize
        );
        assert!(manifest.files.iter().all(|file| file.required));
    }

    #[test]
    fn hf_bert_manifest_declares_embedding_task_and_sentence_limit() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{
                "model_type": "bert",
                "hidden_size": 384,
                "num_hidden_layers": 6,
                "num_attention_heads": 12,
                "max_position_embeddings": 512
            }"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        fs::write(
            dir.path().join("sentence_bert_config.json"),
            br#"{"max_seq_length":256}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("modules.json"),
            br#"[
                {"idx":0,"path":"","type":"sentence_transformers.models.Transformer"},
                {"idx":1,"path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
                {"idx":2,"path":"2_Normalize","type":"sentence_transformers.models.Normalize"}
            ]"#,
        )
        .unwrap();
        fs::create_dir(dir.path().join("1_Pooling")).unwrap();
        fs::write(
            dir.path().join("1_Pooling/config.json"),
            br#"{
                "word_embedding_dimension":384,
                "pooling_mode_cls_token":false,
                "pooling_mode_mean_tokens":true,
                "pooling_mode_max_tokens":false,
                "pooling_mode_mean_sqrt_len_tokens":false
            }"#,
        )
        .unwrap();

        let manifest = load_manifest(dir.path()).unwrap();
        assert_eq!(manifest.family, ModelFamily::Bert);
        assert_eq!(
            manifest.parameters.get("bloom_task"),
            Some(&serde_json::json!("embedding"))
        );
        assert_eq!(
            manifest.parameters.get("max_seq_length"),
            Some(&serde_json::json!(256))
        );
        assert_eq!(
            manifest.parameters.get("head_dim"),
            Some(&serde_json::json!(32))
        );
        assert_eq!(
            manifest.parameters.get("embedding_pooling"),
            Some(&serde_json::json!("mean"))
        );
        assert_eq!(
            manifest.parameters.get("embedding_dimensions"),
            Some(&serde_json::json!(384))
        );
        assert_eq!(
            manifest.parameters.get("embedding_normalization"),
            Some(&serde_json::json!("l2"))
        );

        let estimate = estimate_memory(&manifest, 256);
        assert_eq!(estimate.kv_cache_bytes_per_token, 0);
        assert_eq!(estimate.kv_cache_bytes, 0);
    }

    #[test]
    fn hf_bert_manifest_rejects_oversized_sentence_configuration() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"architectures":["BertModel"],"model_type":"bert"}"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        let oversized = fs::File::create(dir.path().join("sentence_bert_config.json")).unwrap();
        oversized
            .set_len(MAX_SENTENCE_TRANSFORMER_CONFIG_BYTES.saturating_add(1))
            .unwrap();

        let error = load_manifest(dir.path()).unwrap_err().to_string();
        assert!(error.contains("sentence_bert_config.json exceeds"));
    }

    #[test]
    fn hf_bert_manifest_rejects_unimplemented_sentence_transformer_modules() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"bert","hidden_size":384}"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        fs::write(
            dir.path().join("modules.json"),
            br#"[
                {"idx":0,"path":"","type":"sentence_transformers.models.Transformer"},
                {"idx":1,"path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
                {"idx":2,"path":"2_Dense","type":"sentence_transformers.models.Dense"}
            ]"#,
        )
        .unwrap();

        let error = load_manifest(dir.path()).unwrap_err().to_string();
        assert!(error.contains("unsupported Sentence Transformers module type"));
    }

    #[test]
    fn hf_bert_manifest_rejects_incompatible_pooling_contracts() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"bert","hidden_size":384}"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        fs::create_dir(dir.path().join("1_Pooling")).unwrap();
        fs::write(
            dir.path().join("1_Pooling/config.json"),
            br#"{
                "word_embedding_dimension":384,
                "pooling_mode_cls_token":true,
                "pooling_mode_mean_tokens":true
            }"#,
        )
        .unwrap();

        let error = load_manifest(dir.path()).unwrap_err().to_string();
        assert!(error.contains("only mean-token pooling may be enabled"));
    }

    #[test]
    fn hf_bert_manifest_rejects_pooling_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"bert","hidden_size":384}"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        fs::create_dir(dir.path().join("1_Pooling")).unwrap();
        fs::write(
            dir.path().join("1_Pooling/config.json"),
            br#"{
                "word_embedding_dimension":768,
                "pooling_mode_mean_tokens":true
            }"#,
        )
        .unwrap();

        let error = load_manifest(dir.path()).unwrap_err().to_string();
        assert!(error.contains("does not match the encoder hidden size"));
    }

    #[test]
    fn hf_manifest_rejects_oversized_tokenizer_configuration_before_reading_it() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"architectures":["LlamaForCausalLM"],"torch_dtype":"bfloat16"}"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        let oversized = fs::File::create(dir.path().join("tokenizer_config.json")).unwrap();
        oversized
            .set_len(MAX_TOKENIZER_CONFIG_BYTES.saturating_add(1))
            .unwrap();

        let error = load_manifest(dir.path()).unwrap_err().to_string();
        assert!(error.contains("tokenizer_config.json exceeds"));
    }

    #[test]
    fn cpu_safetensors_memory_estimate_accounts_for_f32_materialization() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _dtype = EnvVarGuard::set("BLOOM_DTYPE", "auto");
        let manifest = ModelManifest {
            primary_dtype: DType::BF16,
            files: vec![ModelFile {
                name: "model.safetensors".to_string(),
                format: ModelFormat::Safetensors,
                size_bytes: 1_000,
                hash_sha256: None,
                required: true,
            }],
            ..ModelManifest::default()
        };

        let cpu = estimate_memory_for_device(&manifest, 0, DeviceKind::Cpu);
        let gpu = estimate_memory_for_device(&manifest, 0, DeviceKind::Gpu);
        assert_eq!(cpu.weight_dtype, DType::F32);
        assert_eq!(cpu.weight_bytes, 2_000);
        assert_eq!(gpu.weight_dtype, DType::F16);
        assert_eq!(gpu.weight_bytes, 1_000);
    }

    #[test]
    fn test_infer_manifest_from_gguf_summary_qwen_iq_quant() {
        let summary = GgufMetadataSummary {
            architecture: Some("qwen2".to_string()),
            quantization_type: Some("IQ4_XS".to_string()),
            file_name: Some("qwen-iq4.gguf".to_string()),
            ..Default::default()
        };

        let manifest = infer_manifest_from_gguf_summary(&summary);
        assert_eq!(manifest.family, ModelFamily::Qwen);
        assert_eq!(manifest.primary_dtype, DType::Q4);
        let quant = manifest.quantization.as_ref().unwrap();
        assert!(matches!(quant.scheme, QuantScheme::GGUF(ref s) if s == "IQ4_XS"));
    }

    #[test]
    fn test_infer_manifest_from_gguf_summary_custom_arch() {
        let summary = GgufMetadataSummary {
            architecture: Some("falcon".to_string()),
            quantization_type: Some("Q8_0".to_string()),
            ..Default::default()
        };

        let manifest = infer_manifest_from_gguf_summary(&summary);
        assert_eq!(manifest.family, ModelFamily::Custom("falcon".to_string()));
        assert_eq!(manifest.primary_dtype, DType::Q8);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(2048), "2KB");
        assert_eq!(format_bytes(128 * 1024 * 1024), "128MB");
        assert_eq!(format_bytes(4 * 1024 * 1024 * 1024), "4.0GB");
    }

    #[test]
    fn test_precise_kv_cache_estimation() {
        let mut m = ModelManifest::default();
        m.parameters
            .insert("num_hidden_layers".to_string(), serde_json::json!(32));
        m.parameters
            .insert("num_key_value_heads".to_string(), serde_json::json!(8));
        m.parameters
            .insert("head_dim".to_string(), serde_json::json!(128));

        let est = estimate_memory(&m, 1024);
        // Computed KV per token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes (128 KB)
        // For 1024 tokens: 1024 * 131,072 = 134,217,728 bytes (128 MB)
        assert_eq!(est.kv_cache_bytes, 134_217_728);
    }

    #[test]
    fn test_memory_estimate_uses_quantization_bits() {
        let mut m = ModelManifest::default();
        m.parameters
            .insert("num_hidden_layers".to_string(), serde_json::json!(1));
        m.parameters
            .insert("hidden_size".to_string(), serde_json::json!(16));
        m.parameters
            .insert("intermediate_size".to_string(), serde_json::json!(64));
        m.parameters
            .insert("vocab_size".to_string(), serde_json::json!(32));
        m.primary_dtype = DType::F16;
        m.quantization = Some(QuantizationInfo::from_gguf_type("Q4_K_M"));

        let est = estimate_memory(&m, 1);

        assert_eq!(est.weight_dtype, DType::F16);
        assert_eq!(est.quantization.as_ref().unwrap().bits, 4);
        assert_eq!(est.weight_bytes, 2_560);
    }

    #[test]
    fn test_memory_estimate_uses_manifest_kv_cache_dtype() {
        let mut m = ModelManifest::default();
        m.parameters
            .insert("num_hidden_layers".to_string(), serde_json::json!(2));
        m.parameters
            .insert("num_key_value_heads".to_string(), serde_json::json!(4));
        m.parameters
            .insert("head_dim".to_string(), serde_json::json!(8));
        m.quantization = Some(QuantizationInfo {
            kv_cache_dtype: Some(DType::Q8),
            ..QuantizationInfo::default()
        });

        let est = estimate_memory(&m, 10);

        assert_eq!(est.kv_cache_dtype, DType::Q8);
        assert_eq!(est.kv_cache_bytes_per_token, 128);
        assert_eq!(est.kv_cache_bytes, 1_280);
    }

    #[test]
    fn test_memory_estimate_applies_mmap_residency() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous = std::env::var("BLOOM_GPU_LAYERS").ok();
        std::env::remove_var("BLOOM_GPU_LAYERS");

        let mut m = ModelManifest::default();
        m.runtime_hints.supports_mmap = true;
        m.files.push(ModelFile {
            name: "model.safetensors".to_string(),
            format: ModelFormat::Safetensors,
            size_bytes: 1_000,
            hash_sha256: None,
            required: true,
        });

        let est = estimate_memory(&m, 0);

        let result = std::panic::catch_unwind(|| {
            assert!(est.mmap_residency_applied);
            assert_eq!(est.weight_bytes, 1_000);
            assert_eq!(est.host_weight_bytes, 300);
            assert!(
                est.total_bytes < est.weight_bytes + est.kv_cache_bytes + est.temp_tensor_bytes
            );
        });

        if let Some(val) = previous {
            std::env::set_var("BLOOM_GPU_LAYERS", val);
        } else {
            std::env::remove_var("BLOOM_GPU_LAYERS");
        }

        if let Err(err) = result {
            std::panic::resume_unwind(err);
        }
    }

    #[test]
    fn test_memory_estimate_offload_applies_mmap_only_to_host_weights() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set("BLOOM_GPU_LAYERS", "2");

        let mut m = ModelManifest::default();
        m.runtime_hints.supports_mmap = true;
        m.files.push(ModelFile {
            name: "model.safetensors".to_string(),
            format: ModelFormat::Safetensors,
            size_bytes: 1_000,
            hash_sha256: None,
            required: true,
        });
        m.parameters
            .insert("num_hidden_layers".to_string(), serde_json::json!(10));
        m.parameters
            .insert("num_key_value_heads".to_string(), serde_json::json!(1));
        m.parameters
            .insert("head_dim".to_string(), serde_json::json!(1));

        let est = estimate_memory(&m, 0);

        assert_eq!(est.offloaded_layers, Some(2));
        assert_eq!(est.device_weight_bytes, 200);
        assert_eq!(est.host_weight_bytes, 240);
        assert_eq!(
            est.total_bytes,
            est.device_weight_bytes + est.kv_cache_bytes / 5 + est.temp_tensor_bytes
        );
    }

    #[test]
    fn test_suggest_memory_downgrade() {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut m = ModelManifest::default();
        m.memory_profile.min_ram_bytes = 4_000_000_000;
        let est = estimate_memory(&m, 2048);

        let suggestions = suggest_memory_downgrade(&est, 1_000_000_000); // 1 GB < estimated 4 GB weights + KV
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("Reduce context size"));
    }

    #[test]
    fn test_estimate_memory_from_parameters() {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut m = ModelManifest::default();
        m.parameters
            .insert("num_hidden_layers".to_string(), serde_json::json!(12));
        m.parameters
            .insert("hidden_size".to_string(), serde_json::json!(768));
        m.parameters
            .insert("intermediate_size".to_string(), serde_json::json!(3072));
        m.parameters
            .insert("vocab_size".to_string(), serde_json::json!(30522));
        m.primary_dtype = DType::F16;

        let est = estimate_memory(&m, 512);
        assert_eq!(est.weight_bytes, 320_256_000);
    }

    #[test]
    fn test_infer_from_gguf_unsupported_arch() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("unsupported_arch.gguf");
        let bytes = create_mock_gguf_bytes("falcon", "Q4_K_M", 2); // ggml_dtype_id 2 = Q4_0
        std::fs::write(&gguf_path, bytes).unwrap();

        let res = infer_from_gguf(dir.path(), &gguf_path);
        assert!(res.is_err(), "Expected error, got {:?}", res);
        let err_msg = res.err().unwrap().to_string();
        assert!(
            err_msg.contains("unsupported GGUF architecture: 'falcon'"),
            "Got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("[Diagnostic Tip] Run `inspect_gguf`"),
            "Got: {}",
            err_msg
        );
    }

    #[test]
    fn test_infer_from_gguf_unsupported_quant() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("unsupported_quant.gguf");
        let bytes = create_mock_gguf_bytes("llama", "IQ2_XXS", 16); // ggml_dtype_id 16 = IQ2_XXS
        std::fs::write(&gguf_path, bytes).unwrap();

        let res = infer_from_gguf(dir.path(), &gguf_path);
        assert!(res.is_err(), "Expected error, got {:?}", res);
        let err_msg = res.err().unwrap().to_string();
        assert!(
            err_msg.contains("unsupported GGUF quantization type"),
            "Got: {}",
            err_msg
        );
        assert!(err_msg.contains("16"), "Got: {}", err_msg);
        assert!(err_msg.contains("llama-quantize"), "Got: {}", err_msg);
    }

    #[test]
    fn test_infer_from_gguf_accepts_candle_supported_q4_1() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("supported-q4_1.gguf");
        let bytes = create_mock_gguf_bytes("qwen2", "Q4_1", 3);
        std::fs::write(&gguf_path, bytes).unwrap();

        let manifest = infer_from_gguf(dir.path(), &gguf_path).unwrap();
        assert_eq!(manifest.primary_dtype, DType::Q4);
        assert_eq!(
            manifest.parameters.get("gguf_quantization_type"),
            Some(&serde_json::json!("Q4_1"))
        );
    }

    fn create_mock_gguf_bytes(arch: &str, _quant_name: &str, ggml_dtype_id: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata KV count: 4
        buf.extend_from_slice(&4u64.to_le_bytes());

        // KV 1: general.architecture
        write_kv_string(&mut buf, "general.architecture", arch);

        // KV 2: general.name
        write_kv_string(&mut buf, "general.name", "mock-model");

        // KV 3: tokenizer.ggml.tokens
        write_string(&mut buf, "tokenizer.ggml.tokens");
        buf.extend_from_slice(&9u32.to_le_bytes()); // Array
        buf.extend_from_slice(&8u32.to_le_bytes()); // Array element type String
        buf.extend_from_slice(&0u64.to_le_bytes()); // Length 0

        // KV 4: context_length
        write_kv_u64(&mut buf, "general.context_length", 2048);

        // Tensor info record
        write_string(&mut buf, "blk.0.attn.weight");
        // Num dimensions
        buf.extend_from_slice(&2u32.to_le_bytes());
        // Dimension 1
        buf.extend_from_slice(&128u64.to_le_bytes());
        // Dimension 2
        buf.extend_from_slice(&128u64.to_le_bytes());
        // ggml_dtype
        buf.extend_from_slice(&ggml_dtype_id.to_le_bytes());
        // Offset
        buf.extend_from_slice(&0u64.to_le_bytes());

        buf
    }

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

    fn write_kv_u64(buf: &mut Vec<u8>, key: &str, val: u64) {
        write_string(buf, key);
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }
}
