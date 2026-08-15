//! Unified quantization schema for Bloom engine.
//!
//! Provides a consistent representation of quantization methods,
//! configurations, and KV cache quantization across all engines
//! (Candle, OpenVINO, future TensorRT, etc.).

use serde::{Deserialize, Serialize};

#[cfg(feature = "candle-engine")]
use candle_core::{DType, Tensor};

/// Structured error type for GGUF file parsing and validation.
#[derive(Debug, Clone)]
pub enum GgufError {
    /// GGUF magic bytes are missing or incorrect.
    InvalidHeader(String),
    /// Model architecture is not supported.
    UnsupportedArch(String),
    /// Required metadata key is missing.
    MissingMetadata(String),
    /// File is corrupted or truncated.
    CorruptedFile(String),
    /// Quantization type is not recognized.
    UnsupportedQuantType(String),
    /// Generic I/O error description.
    Io(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader(msg) => write!(f, "invalid GGUF header: {}", msg),
            Self::UnsupportedArch(msg) => write!(f, "unsupported GGUF architecture: {}", msg),
            Self::MissingMetadata(msg) => write!(f, "missing GGUF metadata: {}", msg),
            Self::CorruptedFile(msg) => write!(f, "corrupted GGUF file: {}", msg),
            Self::UnsupportedQuantType(msg) => write!(f, "unsupported quantization type: {}", msg),
            Self::Io(msg) => write!(f, "GGUF I/O error: {}", msg),
        }
    }
}

impl std::error::Error for GgufError {}

/// Quantization method used for model weights and/or KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantMethod {
    /// No quantization — full precision (F16, BF16, F32).
    None,
    /// GGUF quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.).
    Gguf,
    /// AWQ (Activation-aware Weight Quantization) — W4A16.
    Awq,
    /// GPTQ — W4A16 or W4A8 group-wise quantization.
    Gptq,
    /// FP8 (E4M3/E5M2) — used by NVIDIA H100/Ada.
    Fp8,
    /// NVFP4 — NVIDIA FP4 format for Blackwell.
    NvFp4,
    /// INT8 symmetric quantization.
    Int8,
    /// BitsAndBytes NF4 (NormalFloat4).
    Nf4,
    /// BitsAndBytes FP4 (Float4).
    Fp4,
    /// HQQ (Half-Quadratic Quantization).
    Hqq,
    /// EETQ (Easy and Efficient Quantization) — W8A16.
    Eetq,
    /// AQLM (Additive Quantization for Language Models).
    Aqlm,
    /// EXL2 (ExLlamaV2) quantization.
    Exl2,
    /// Quanto (PyTorch quantization toolkit).
    Quanto,
    /// Torchao (PyTorch Architecture Optimization).
    Torchao,
}

impl QuantMethod {
    /// Whether this method uses sub-byte precision for weights.
    pub fn is_sub_byte(&self) -> bool {
        matches!(
            self,
            Self::Gguf
                | Self::Awq
                | Self::Gptq
                | Self::NvFp4
                | Self::Nf4
                | Self::Fp4
                | Self::Hqq
                | Self::Aqlm
                | Self::Exl2
                | Self::Quanto
                | Self::Torchao
        )
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Gguf => "gguf",
            Self::Awq => "awq",
            Self::Gptq => "gptq",
            Self::Fp8 => "fp8",
            Self::NvFp4 => "nvfp4",
            Self::Int8 => "int8",
            Self::Nf4 => "nf4",
            Self::Fp4 => "fp4",
            Self::Hqq => "hqq",
            Self::Eetq => "eetq",
            Self::Aqlm => "aqlm",
            Self::Exl2 => "exl2",
            Self::Quanto => "quanto",
            Self::Torchao => "torchao",
        }
    }

    /// Infer quantization method from a config.json `quantization_config` object.
    pub fn from_hf_quant_config(config: &serde_json::Value) -> Option<Self> {
        let quant_method = config.get("quant_method")?.as_str()?;
        match quant_method.to_lowercase().as_str() {
            "awq" | "awq_marlin" => Some(Self::Awq),
            "gptq" | "marlin" | "gptq_marlin" => Some(Self::Gptq),
            "fp8" | "fbgemm_fp8" | "fp8_marlin" => Some(Self::Fp8),
            "hqq" => Some(Self::Hqq),
            "eetq" => Some(Self::Eetq),
            "aqlm" => Some(Self::Aqlm),
            "exl2" => Some(Self::Exl2),
            "quanto" => Some(Self::Quanto),
            "torchao" => Some(Self::Torchao),
            "bitsandbytes" | "bnb" => {
                let is_4bit = config
                    .get("load_in_4bit")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                    || config.get("bnb_4bit_compute_dtype").is_some();
                if is_4bit {
                    let quant_type = config
                        .get("bnb_4bit_quant_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("fp4");
                    if quant_type.to_lowercase() == "nf4" {
                        Some(Self::Nf4)
                    } else {
                        Some(Self::Fp4)
                    }
                } else {
                    Some(Self::Int8)
                }
            }
            _ => None,
        }
    }

    /// Infer quantization method from GGUF quantization type string.
    pub fn from_gguf_type(type_name: &str) -> Self {
        let lower = type_name.to_lowercase();
        if lower.contains("q4")
            || lower.contains("q5")
            || lower.contains("q8")
            || lower.contains("q2")
            || lower.contains("q3")
            || lower.contains("q6")
            || lower.contains("iq2")
            || lower.contains("iq3")
            || lower.contains("iq4")
        {
            Self::Gguf
        } else {
            Self::None
        }
    }
}

impl std::fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Bits precision for KV cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum KvCacheDtype {
    /// Full precision F16.
    #[default]
    F16,
    /// Brain float 16.
    BF16,
    /// Full precision F32.
    F32,
    /// INT8 symmetric per-token quantization.
    Int8,
    /// FP8 (E4M3) quantization — requires hardware support.
    Fp8,
}

impl KvCacheDtype {
    /// Size in bytes of a single element.
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Int8 | Self::Fp8 => 1,
        }
    }

    /// Whether this dtype requires dequantization before attention computation.
    pub fn needs_dequant(&self) -> bool {
        matches!(self, Self::Int8 | Self::Fp8)
    }
}

/// Unified quantization configuration for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Weight quantization method.
    pub weight_method: QuantMethod,
    /// Effective weight bits (e.g. 4 for AWQ/GPTQ Q4, 8 for INT8).
    pub weight_bits: u8,
    /// Group size for group-wise quantization (0 = per-tensor).
    pub group_size: usize,
    /// KV cache storage dtype.
    pub kv_cache_dtype: KvCacheDtype,
    /// Whether activation quantization is used (e.g. W8A8).
    pub activation_quant: bool,
    /// Original source format identifier (e.g. "awq", "gptq", "gguf_q4_0").
    pub source_format: String,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            weight_method: QuantMethod::None,
            weight_bits: 16,
            group_size: 0,
            kv_cache_dtype: KvCacheDtype::F16,
            activation_quant: false,
            source_format: String::new(),
        }
    }
}

impl QuantizationConfig {
    /// Detect quantization config from a model directory's config.json.
    pub fn from_model_config(config: &serde_json::Value) -> Self {
        if let Some(quant_cfg) = config.get("quantization_config") {
            if let Some(method) = QuantMethod::from_hf_quant_config(quant_cfg) {
                let weight_bits = quant_cfg
                    .get("bits")
                    .and_then(|v| v.as_u64())
                    .map(|b| b as u8)
                    .unwrap_or(match method {
                        QuantMethod::Awq | QuantMethod::Gptq => 4,
                        QuantMethod::Fp8 | QuantMethod::Int8 | QuantMethod::Eetq => 8,
                        QuantMethod::NvFp4 | QuantMethod::Nf4 | QuantMethod::Fp4 => 4,
                        QuantMethod::Aqlm => 2,
                        QuantMethod::Hqq
                        | QuantMethod::Exl2
                        | QuantMethod::Quanto
                        | QuantMethod::Torchao => 4,
                        _ => 16,
                    });

                let group_size = quant_cfg
                    .get("group_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;

                let kv_cache_dtype = quant_cfg
                    .get("kv_cache_dtype")
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "int8" => KvCacheDtype::Int8,
                        "fp8" => KvCacheDtype::Fp8,
                        "bf16" => KvCacheDtype::BF16,
                        _ => KvCacheDtype::F16,
                    })
                    .unwrap_or_default();

                return Self {
                    weight_method: method,
                    weight_bits,
                    group_size,
                    kv_cache_dtype,
                    activation_quant: quant_cfg
                        .get("activation_quant")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    source_format: method.label().to_string(),
                };
            }
        }

        // Check torch_dtype as fallback
        if let Some(dtype_str) = config.get("torch_dtype").and_then(|v| v.as_str()) {
            let kv_cache_dtype = match dtype_str {
                "bfloat16" => KvCacheDtype::BF16,
                "float32" => KvCacheDtype::F32,
                _ => KvCacheDtype::F16,
            };
            return Self {
                kv_cache_dtype,
                ..Self::default()
            };
        }

        Self::default()
    }

    /// Detect quantization from GGUF file metadata.
    pub fn from_gguf_metadata(general_type: Option<&str>) -> Self {
        let (method, bits, source) = match general_type {
            Some(t) => {
                let lower = t.to_lowercase();
                // IQ (importance) quantization variants
                if lower.contains("iq2_xxs") {
                    (QuantMethod::Gguf, 2, "gguf_iq2_xxs")
                } else if lower.contains("iq2_xs") {
                    (QuantMethod::Gguf, 2, "gguf_iq2_xs")
                } else if lower.contains("iq3_xxs") {
                    (QuantMethod::Gguf, 3, "gguf_iq3_xxs")
                } else if lower.contains("iq3_s") {
                    (QuantMethod::Gguf, 3, "gguf_iq3_s")
                } else if lower.contains("iq4_xs") {
                    (QuantMethod::Gguf, 4, "gguf_iq4_xs")
                } else if lower.contains("iq4_nl") {
                    (QuantMethod::Gguf, 4, "gguf_iq4_nl")
                // Standard Q/K quantization variants
                } else if lower.contains("q2_k") {
                    (QuantMethod::Gguf, 2, "gguf_q2_k")
                } else if lower.contains("q3_k_s") {
                    (QuantMethod::Gguf, 3, "gguf_q3_k_s")
                } else if lower.contains("q3_k_m") {
                    (QuantMethod::Gguf, 3, "gguf_q3_k_m")
                } else if lower.contains("q3_k_l") {
                    (QuantMethod::Gguf, 3, "gguf_q3_k_l")
                } else if lower.contains("q4_k_s") {
                    (QuantMethod::Gguf, 4, "gguf_q4_k_s")
                } else if lower.contains("q4_k_m") {
                    (QuantMethod::Gguf, 4, "gguf_q4_k_m")
                } else if lower.contains("q4_0") {
                    (QuantMethod::Gguf, 4, "gguf_q4_0")
                } else if lower.contains("q4_1") {
                    (QuantMethod::Gguf, 4, "gguf_q4_1")
                } else if lower.contains("q5_k_s") {
                    (QuantMethod::Gguf, 5, "gguf_q5_k_s")
                } else if lower.contains("q5_k_m") {
                    (QuantMethod::Gguf, 5, "gguf_q5_k_m")
                } else if lower.contains("q5_0") {
                    (QuantMethod::Gguf, 5, "gguf_q5_0")
                } else if lower.contains("q5_1") {
                    (QuantMethod::Gguf, 5, "gguf_q5_1")
                } else if lower.contains("q6_k") {
                    (QuantMethod::Gguf, 6, "gguf_q6_k")
                } else if lower.contains("q8_0") {
                    (QuantMethod::Gguf, 8, "gguf_q8_0")
                } else if lower.contains("q2") || lower.contains("q3") {
                    (QuantMethod::Gguf, 3, "gguf_q3")
                } else {
                    (QuantMethod::None, 16, "")
                }
            }
            None => (QuantMethod::None, 16, ""),
        };

        Self {
            weight_method: method,
            weight_bits: bits,
            source_format: source.to_string(),
            ..Self::default()
        }
    }

    /// Whether the model weights are quantized (any method).
    pub fn is_quantized(&self) -> bool {
        self.weight_method != QuantMethod::None
    }

    /// Whether KV cache quantization is enabled.
    pub fn has_kv_cache_quant(&self) -> bool {
        self.kv_cache_dtype.needs_dequant()
    }

    /// Memory savings factor compared to F16 (approximate).
    pub fn memory_ratio(&self) -> f64 {
        self.weight_bits as f64 / 16.0
    }
}

/// INT8 symmetric per-token quantization for KV cache tensors.
///
/// Stores quantized values as i8 with a per-row f32 scale factor.
/// Dequantization: `value = quantized_value * scale`
#[derive(Debug, Clone)]
pub struct Int8QuantizedKv {
    /// Quantized data (i8 packed as u8).
    pub data: Vec<u8>,
    /// Per-row scale factors (one per token position).
    pub scales: Vec<f32>,
    /// Number of rows (token positions).
    pub num_rows: usize,
    /// Number of columns (head_dim * num_heads).
    pub num_cols: usize,
}

impl Int8QuantizedKv {
    /// Quantize an F16/F32 KV tensor row-by-row with symmetric INT8 quantization.
    pub fn quantize_f32(data: &[f32], num_rows: usize, num_cols: usize) -> Self {
        assert_eq!(data.len(), num_rows * num_cols);
        let mut quantized = vec![0u8; num_rows * num_cols];
        let mut scales = vec![0.0f32; num_rows];

        for row in 0..num_rows {
            let row_data = &data[row * num_cols..(row + 1) * num_cols];
            let max_abs = row_data
                .iter()
                .map(|v| v.abs())
                .fold(f32::NEG_INFINITY, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[row] = scale;
            let inv_scale = 1.0 / scale;

            for (i, &val) in row_data.iter().enumerate() {
                let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                quantized[row * num_cols + i] = q as u8;
            }
        }

        Self {
            data: quantized,
            scales,
            num_rows,
            num_cols,
        }
    }

    /// Dequantize back to F32.
    pub fn dequantize_f32(&self) -> Vec<f32> {
        let mut result = vec![0.0f32; self.num_rows * self.num_cols];
        for row in 0..self.num_rows {
            let scale = self.scales[row];
            for col in 0..self.num_cols {
                let idx = row * self.num_cols + col;
                result[idx] = (self.data[idx] as i8) as f32 * scale;
            }
        }
        result
    }

    /// Total memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Memory footprint of equivalent F16 storage.
    pub fn f16_memory_bytes(&self) -> usize {
        self.num_rows * self.num_cols * 2
    }

    /// Compression ratio compared to F16.
    pub fn compression_ratio(&self) -> f64 {
        let f16_bytes = self.f16_memory_bytes() as f64;
        if f16_bytes > 0.0 {
            self.memory_bytes() as f64 / f16_bytes
        } else {
            1.0
        }
    }
}

#[cfg(feature = "candle-engine")]
/// Dequantize NF4 (NormalFloat 4) quantized weights using the BitsAndBytes NF4 codebook.
pub fn dequantize_nf4(quantized: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
    let dev = quantized.device();
    let dtype = scale.dtype();

    // 1. Unpack 4-bit indices from U8 tensor.
    let packed_f = quantized.to_dtype(DType::F32)?;
    let q1 = (packed_f.clone() / 16.0)?.floor()?;
    let q2 = (packed_f - (q1.clone() * 16.0)?)?;

    // Stack along a new dimension to preserve order: [..., 2]
    let unpacked = Tensor::stack(&[&q1, &q2], quantized.rank())?;
    let unpacked_flat = unpacked.flatten_all()?;

    // 2. Map indices to NF4 codebook values.
    let nf4_values = [
        -1.0f32,
        -0.6961917,
        -0.52507305,
        -0.39491746,
        -0.28444138,
        -0.18477343,
        -0.09105026,
        0.0,
        0.07958029,
        0.1609302,
        0.2461123,
        0.33791524,
        0.44070983,
        0.562617,
        0.72295684,
        1.0,
    ];
    let codebook = Tensor::new(&nf4_values, dev)?.to_dtype(dtype)?;

    // We can index into codebook using unpacked indices (converted to U32)
    let indices = unpacked_flat.to_dtype(DType::U32)?;
    let mapped = codebook.index_select(&indices, 0)?;

    // 3. Reshape and multiply by scale
    let mut out_shape = quantized.dims().to_vec();
    if let Some(last) = out_shape.last_mut() {
        *last *= 2;
    }
    let reshaped = mapped.reshape(out_shape)?;
    reshaped.broadcast_mul(scale)
}

#[cfg(feature = "candle-engine")]
/// Dequantize FP4 (Float 4) quantized weights using the BitsAndBytes FP4 codebook.
pub fn dequantize_fp4(quantized: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
    let dev = quantized.device();
    let dtype = scale.dtype();

    // 1. Unpack 4-bit indices from U8 tensor.
    let packed_f = quantized.to_dtype(DType::F32)?;
    let q1 = (packed_f.clone() / 16.0)?.floor()?;
    let q2 = (packed_f - (q1.clone() * 16.0)?)?;

    // Stack along a new dimension: [..., 2]
    let unpacked = Tensor::stack(&[&q1, &q2], quantized.rank())?;
    let unpacked_flat = unpacked.flatten_all()?;

    // 2. Map indices to FP4 codebook values.
    let fp4_values = [
        0.0f32, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0,
        -2.0, -3.0,
    ];
    let codebook = Tensor::new(&fp4_values, dev)?.to_dtype(dtype)?;

    // We can index into codebook using unpacked indices (converted to U32)
    let indices = unpacked_flat.to_dtype(DType::U32)?;
    let mapped = codebook.index_select(&indices, 0)?;

    // 3. Reshape and multiply by scale
    let mut out_shape = quantized.dims().to_vec();
    if let Some(last) = out_shape.last_mut() {
        *last *= 2;
    }
    let reshaped = mapped.reshape(out_shape)?;
    reshaped.broadcast_mul(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_method_properties() {
        assert!(QuantMethod::Awq.is_sub_byte());
        assert!(QuantMethod::Gguf.is_sub_byte());
        assert!(!QuantMethod::None.is_sub_byte());
        assert!(!QuantMethod::Fp8.is_sub_byte());
        assert!(QuantMethod::Hqq.is_sub_byte());
        assert!(QuantMethod::Fp4.is_sub_byte());
        assert!(QuantMethod::Aqlm.is_sub_byte());
        assert!(QuantMethod::Exl2.is_sub_byte());
        assert_eq!(QuantMethod::Awq.label(), "awq");
        assert_eq!(QuantMethod::Hqq.label(), "hqq");
    }

    #[test]
    fn test_quant_method_from_hf_config() {
        let awq_config = serde_json::json!({ "quant_method": "awq", "bits": 4 });
        assert_eq!(
            QuantMethod::from_hf_quant_config(&awq_config),
            Some(QuantMethod::Awq)
        );

        let gptq_config = serde_json::json!({ "quant_method": "gptq" });
        assert_eq!(
            QuantMethod::from_hf_quant_config(&gptq_config),
            Some(QuantMethod::Gptq)
        );

        let fp8_config = serde_json::json!({ "quant_method": "fp8" });
        assert_eq!(
            QuantMethod::from_hf_quant_config(&fp8_config),
            Some(QuantMethod::Fp8)
        );

        let hqq_config = serde_json::json!({ "quant_method": "hqq" });
        assert_eq!(
            QuantMethod::from_hf_quant_config(&hqq_config),
            Some(QuantMethod::Hqq)
        );

        let marlin_config = serde_json::json!({ "quant_method": "gptq_marlin" });
        assert_eq!(
            QuantMethod::from_hf_quant_config(&marlin_config),
            Some(QuantMethod::Gptq)
        );

        let bnb_fp4 = serde_json::json!({
            "quant_method": "bitsandbytes",
            "load_in_4bit": true,
            "bnb_4bit_quant_type": "fp4"
        });
        assert_eq!(
            QuantMethod::from_hf_quant_config(&bnb_fp4),
            Some(QuantMethod::Fp4)
        );

        let bnb_nf4 = serde_json::json!({
            "quant_method": "bitsandbytes",
            "load_in_4bit": true,
            "bnb_4bit_quant_type": "nf4"
        });
        assert_eq!(
            QuantMethod::from_hf_quant_config(&bnb_nf4),
            Some(QuantMethod::Nf4)
        );

        let empty = serde_json::json!({});
        assert_eq!(QuantMethod::from_hf_quant_config(&empty), None);
    }

    #[test]
    fn test_quantization_config_from_model_config() {
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128
            }
        });
        let qc = QuantizationConfig::from_model_config(&config);
        assert_eq!(qc.weight_method, QuantMethod::Awq);
        assert_eq!(qc.weight_bits, 4);
        assert_eq!(qc.group_size, 128);
        assert!(qc.is_quantized());
    }

    #[test]
    fn test_quantization_config_default() {
        let config = serde_json::json!({});
        let qc = QuantizationConfig::from_model_config(&config);
        assert_eq!(qc.weight_method, QuantMethod::None);
        assert!(!qc.is_quantized());
        assert!(!qc.has_kv_cache_quant());
    }

    #[test]
    fn test_gguf_quantization_detection() {
        let qc = QuantizationConfig::from_gguf_metadata(Some("Q4_0"));
        assert_eq!(qc.weight_method, QuantMethod::Gguf);
        assert_eq!(qc.weight_bits, 4);
        assert!(qc.is_quantized());

        let qc2 = QuantizationConfig::from_gguf_metadata(Some("Q8_0"));
        assert_eq!(qc2.weight_bits, 8);

        let qc3 = QuantizationConfig::from_gguf_metadata(None);
        assert!(!qc3.is_quantized());

        // Extended GGUF types
        let qc_iq2 = QuantizationConfig::from_gguf_metadata(Some("IQ2_XXS"));
        assert_eq!(qc_iq2.weight_bits, 2);
        assert_eq!(qc_iq2.source_format, "gguf_iq2_xxs");

        let qc_q4km = QuantizationConfig::from_gguf_metadata(Some("Q4_K_M"));
        assert_eq!(qc_q4km.weight_bits, 4);
        assert_eq!(qc_q4km.source_format, "gguf_q4_k_m");

        let qc_q6k = QuantizationConfig::from_gguf_metadata(Some("Q6_K"));
        assert_eq!(qc_q6k.weight_bits, 6);
        assert_eq!(qc_q6k.source_format, "gguf_q6_k");

        let qc_q2k = QuantizationConfig::from_gguf_metadata(Some("Q2_K"));
        assert_eq!(qc_q2k.weight_bits, 2);

        let qc_q5km = QuantizationConfig::from_gguf_metadata(Some("Q5_K_M"));
        assert_eq!(qc_q5km.weight_bits, 5);
        assert_eq!(qc_q5km.source_format, "gguf_q5_k_m");
    }

    #[test]
    fn test_kv_cache_dtype() {
        assert_eq!(KvCacheDtype::F16.element_size(), 2);
        assert_eq!(KvCacheDtype::Int8.element_size(), 1);
        assert_eq!(KvCacheDtype::Fp8.element_size(), 1);
        assert!(!KvCacheDtype::F16.needs_dequant());
        assert!(KvCacheDtype::Int8.needs_dequant());
        assert!(KvCacheDtype::Fp8.needs_dequant());
    }

    #[test]
    fn test_int8_quantize_dequantize_roundtrip() {
        let original = vec![0.5, -0.3, 1.0, -1.0, 0.0, 0.1, -0.7, 0.9];
        let quantized = Int8QuantizedKv::quantize_f32(&original, 2, 4);

        assert_eq!(quantized.num_rows, 2);
        assert_eq!(quantized.num_cols, 4);
        // Small arrays: INT8 (8 bytes) + scales (8 bytes) = F16 (16 bytes)
        // Compression benefits appear at larger sizes.

        let recovered = quantized.dequantize_f32();
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            // INT8 quantization has ~1/127 precision loss per row
            assert!(
                (orig - rec).abs() < 0.02,
                "orig={}, rec={}, diff={}",
                orig,
                rec,
                (orig - rec).abs()
            );
        }
    }

    #[test]
    fn test_int8_quantization_memory() {
        let data = vec![0.0f32; 128 * 64]; // 128 rows x 64 cols
        let quantized = Int8QuantizedKv::quantize_f32(&data, 128, 64);
        // INT8: 128*64 bytes data + 128*4 bytes scales = 8704
        // F16:  128*64*2 bytes = 16384
        assert_eq!(quantized.memory_bytes(), 8704);
        assert_eq!(quantized.f16_memory_bytes(), 16384);
        assert!(quantized.compression_ratio() < 0.6);
    }

    #[test]
    fn test_memory_ratio() {
        let mut qc = QuantizationConfig::default();
        assert!((qc.memory_ratio() - 1.0).abs() < f64::EPSILON);

        qc.weight_bits = 4;
        assert!((qc.memory_ratio() - 0.25).abs() < f64::EPSILON);

        qc.weight_bits = 8;
        assert!((qc.memory_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    #[cfg(feature = "candle-engine")]
    fn test_dequantize_nf4_and_fp4() {
        use candle_core::Device;

        let device = Device::Cpu;
        let packed = Tensor::new(&[23u8, 240u8], &device).unwrap();
        let scale = Tensor::new(2.0f32, &device).unwrap();

        let nf4_dequant = dequantize_nf4(&packed, &scale).unwrap();
        let nf4_vals = nf4_dequant.to_vec1::<f32>().unwrap();
        assert_eq!(nf4_vals.len(), 4);
        assert!((nf4_vals[0] - (-1.3923834)).abs() < 1e-5);
        assert!((nf4_vals[1] - 0.0).abs() < 1e-5);
        assert!((nf4_vals[2] - 2.0).abs() < 1e-5);
        assert!((nf4_vals[3] - (-2.0)).abs() < 1e-5);

        let fp4_dequant = dequantize_fp4(&packed, &scale).unwrap();
        let fp4_vals = fp4_dequant.to_vec1::<f32>().unwrap();
        assert_eq!(fp4_vals.len(), 4);
        assert!((fp4_vals[0] - 0.125).abs() < 1e-5);
        assert!((fp4_vals[1] - 6.0).abs() < 1e-5);
        assert!((fp4_vals[2] - (-6.0)).abs() < 1e-5);
        assert!((fp4_vals[3] - 0.0).abs() < 1e-5);
    }
}
