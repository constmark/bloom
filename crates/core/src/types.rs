use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceClass {
    Cpu,
    IntegratedGpu,
    DiscreteGpu,
    Npu,
    Dsp,
    Remote,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryTopology {
    /// Apple Silicon UMA — GPU and CPU share the same physical memory pool.
    Unified,
    /// Discrete GPU with dedicated VRAM (e.g. PCIe NVIDIA/AMD card).
    Discrete,
    /// Integrated GPU or APU sharing system RAM but not identical to Apple UMA
    /// (e.g. Intel iGPU, AMD APU). Budget should use system RAM minus reserved.
    SharedSystemMemory,
    /// Remote / network-attached accelerator (e.g. cloud NPU, DSP offload).
    /// Memory is not locally addressable; transfers go through a transport layer.
    RemoteMemory,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// Signed 8-bit integer (e.g. INT8 quantized weights).
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 4-bit integer (e.g. INT4 quantized weights).
    I4,
    /// NormalFloat4 — 4-bit normalised float quantization.
    NF4,
    /// Legacy alias for I8 — quantized 8-bit.
    Q8,
    /// Legacy alias for I4 — quantized 4-bit.
    Q4,
    Unknown,
}

/// Quantization scheme identifier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantScheme {
    /// No quantization (native dtype).
    None,
    /// GGUF quantization variant, e.g. "Q4_K_M", "Q8_0", "IQ4_XS".
    GGUF(String),
    /// AWQ (Activation-aware Weight Quantization).
    AWQ,
    /// GPTQ (Optimal Brain Quantization variant).
    GPTQ,
    /// Generic INT4 quantization.
    INT4,
    /// Generic INT8 quantization.
    INT8,
    /// NormalFloat4 quantization.
    NF4,
}

/// Describes the quantization configuration of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// The quantization scheme used.
    pub scheme: QuantScheme,
    /// Number of bits per weight (e.g. 4, 8).
    pub bits: u8,
    /// Group size for grouped quantization (if applicable).
    pub group_size: Option<usize>,
    /// Whether activation order reordering is applied (GPTQ-style).
    pub act_order: bool,
    /// Dtype used for KV cache storage (may differ from weight dtype).
    pub kv_cache_dtype: Option<DType>,
    /// Whether an importance matrix was used for quantization.
    #[serde(default)]
    pub imatrix: bool,
}

impl Default for QuantizationInfo {
    fn default() -> Self {
        Self {
            scheme: QuantScheme::None,
            bits: 16,
            group_size: None,
            act_order: false,
            kv_cache_dtype: None,
            imatrix: false,
        }
    }
}

impl QuantizationInfo {
    /// Infer QuantizationInfo from a DType alone (no extra metadata).
    pub fn from_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::Q4 | DType::I4 => Some(Self {
                scheme: QuantScheme::INT4,
                bits: 4,
                ..Self::default()
            }),
            DType::Q8 | DType::I8 => Some(Self {
                scheme: QuantScheme::INT8,
                bits: 8,
                ..Self::default()
            }),
            DType::NF4 => Some(Self {
                scheme: QuantScheme::NF4,
                bits: 4,
                ..Self::default()
            }),
            _ => None,
        }
    }

    /// Parse a GGUF quantization type string (e.g. "Q4_K_M") into QuantizationInfo.
    pub fn from_gguf_type(type_str: &str) -> Self {
        let upper = type_str.to_uppercase();
        let bits = if upper.starts_with("Q2") || upper.starts_with("IQ2") {
            2
        } else if upper.starts_with("Q3") || upper.starts_with("IQ3") {
            3
        } else if upper.starts_with("Q4") || upper.starts_with("IQ4") {
            4
        } else if upper.starts_with("Q5") || upper.starts_with("IQ5") {
            5
        } else if upper.starts_with("Q6") {
            6
        } else if upper.starts_with("Q8") {
            8
        } else {
            0
        };
        Self {
            scheme: QuantScheme::GGUF(type_str.to_string()),
            bits,
            group_size: None,
            act_order: false,
            kv_cache_dtype: None,
            imatrix: false,
        }
    }

    /// Approximate bytes-per-element for this quantization.
    pub fn bytes_per_element(&self) -> f64 {
        self.bits as f64 / 8.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    OpenVinoIr,
    /// NVIDIA TensorRT serialized engine / plan file.
    TensorRtEngine,
    Onnx,
    /// Apple Core ML model package (.mlpackage / .mlmodel).
    CoreMl,
    /// PyTorch TorchScript serialized model.
    TorchScript,
    /// Apple MLX weight format.
    Mlx,
    /// Vulkan SPIR-V shader/kernel execution format.
    VulkanSpirv,
    /// Vendor-specific opaque bundle (e.g. Qualcomm QNN, MediaTek NeuroPilot).
    VendorBundle,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PowerState {
    Battery,
    PluggedIn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapability {
    pub backend_name: String,
    /// Vendor identifier, e.g. "Intel", "NVIDIA", "Apple".
    pub vendor: Option<String>,
    pub device_class: DeviceClass,
    pub memory_topology: MemoryTopology,
    pub max_memory: usize,       // bytes
    pub available_memory: usize, // bytes
    pub supported_dtypes: Vec<DType>,
    pub supported_formats: Vec<ModelFormat>,
    pub supports_mmap: bool,
    pub has_quantization_kernels: bool,
    pub supports_streaming: bool,
    pub thermal_state: ThermalState,
    pub power_state: PowerState,
    /// Maximum tokens per batch this device can handle.
    pub max_batch_tokens: Option<usize>,
    /// Number of parallel threads / compute units available.
    pub available_parallelism: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Audio,
    Vision,
    Multi,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorShape(pub Vec<usize>);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "json_schema")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            seed: None,
            response_format: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResourcePriority {
    Speculative = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResidencyStrategy {
    Resident,
    OnDemand,
    Mmap,
    Offload,
    Prefetch,
    FallbackBackend,
}

impl ResidencyStrategy {
    /// Infer recommended residency strategy from manifest and device capability.
    pub fn from_manifest_and_capability(
        manifest: &crate::ModelManifest,
        capability: &DeviceCapability,
    ) -> Self {
        match capability.memory_topology {
            MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
                if manifest.runtime_hints.supports_mmap {
                    ResidencyStrategy::Mmap
                } else {
                    ResidencyStrategy::OnDemand
                }
            }
            MemoryTopology::Discrete => {
                if manifest.memory_profile.min_vram_bytes > capability.available_memory {
                    ResidencyStrategy::FallbackBackend
                } else {
                    ResidencyStrategy::Offload
                }
            }
            MemoryTopology::RemoteMemory => {
                // Remote accelerators: prefer on-demand loading to minimise
                // network transfers; fall back if the model is too large.
                if manifest.memory_profile.min_vram_bytes > capability.available_memory {
                    ResidencyStrategy::FallbackBackend
                } else {
                    ResidencyStrategy::OnDemand
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CacheKind {
    KvCache,
    StateCache,
    PrefillCache,
}

/// Result of a benchmark run, capturing performance metrics for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Backend used for this benchmark.
    pub backend: String,
    /// Model identifier.
    pub model_id: String,
    /// Primary dtype used.
    pub dtype: DType,
    /// Quantization level, if any.
    pub quantization: Option<String>,
    /// Tokens per second throughput.
    pub tokens_per_second: f64,
    /// Time-to-first-token in milliseconds.
    pub ttft_ms: Option<f64>,
    /// Average latency per token in milliseconds.
    pub avg_latency_ms: f64,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: usize,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Wall-clock duration in seconds.
    pub duration_secs: f64,
    /// ISO 8601 timestamp of when the benchmark ran.
    pub timestamp: String,
    /// Optional notes or environment info.
    pub notes: Option<String>,
}

impl BenchmarkResult {
    /// Compute efficiency: tokens/sec per GB of peak memory.
    pub fn efficiency_tokens_per_gb(&self) -> f64 {
        let gb = self.peak_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if gb > 0.0 {
            self.tokens_per_second / gb
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_kind_serde() {
        let cpu = DeviceKind::Cpu;
        let serialized = serde_json::to_string(&cpu).unwrap();
        assert_eq!(serialized, "\"Cpu\"");

        let deserialized: DeviceKind = serde_json::from_str("\"Npu\"").unwrap();
        assert_eq!(deserialized, DeviceKind::Npu);
    }

    #[test]
    fn test_modality_serde() {
        let modality = Modality::Multi;
        let serialized = serde_json::to_string(&modality).unwrap();
        assert_eq!(serialized, "\"Multi\"");

        let deserialized: Modality = serde_json::from_str("\"Text\"").unwrap();
        assert_eq!(deserialized, Modality::Text);
    }

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape(vec![1, 3, 224, 224]);
        assert_eq!(shape.0.len(), 4);
        assert_eq!(shape.0[2], 224);
    }

    #[test]
    fn test_generation_params_defaults() {
        let params = GenerationParams::default();
        assert_eq!(params.max_tokens, 128);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.seed, None);
    }

    #[test]
    fn test_benchmark_result_serde() {
        let result = BenchmarkResult {
            backend: "cpu".into(),
            model_id: "qwen-7b".into(),
            dtype: DType::F16,
            quantization: Some("Q4".into()),
            tokens_per_second: 42.5,
            ttft_ms: Some(120.0),
            avg_latency_ms: 23.5,
            peak_memory_bytes: 4 * 1024 * 1024 * 1024,
            tokens_generated: 256,
            duration_secs: 6.02,
            timestamp: "2025-01-01T00:00:00Z".into(),
            notes: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("tokens_per_second"));
        let deser: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.backend, "cpu");
        assert!((deser.tokens_per_second - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_result_efficiency() {
        let result = BenchmarkResult {
            backend: "cuda".into(),
            model_id: "llama-13b".into(),
            dtype: DType::F16,
            quantization: None,
            tokens_per_second: 100.0,
            ttft_ms: None,
            avg_latency_ms: 10.0,
            peak_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            tokens_generated: 500,
            duration_secs: 5.0,
            timestamp: "2025-01-01T00:00:00Z".into(),
            notes: None,
        };
        let eff = result.efficiency_tokens_per_gb();
        // 100 tok/s / 8 GB = 12.5 tok/s/GB
        assert!((eff - 12.5).abs() < 0.01);
    }

    #[test]
    fn test_quantization_info_serde() {
        let info = QuantizationInfo {
            scheme: QuantScheme::GGUF("Q4_K_M".to_string()),
            bits: 4,
            group_size: Some(32),
            act_order: false,
            kv_cache_dtype: Some(DType::F16),
            imatrix: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("Q4_K_M"));
        let deser: QuantizationInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.bits, 4);
        assert!(matches!(deser.scheme, QuantScheme::GGUF(ref s) if s == "Q4_K_M"));
        assert_eq!(deser.kv_cache_dtype, Some(DType::F16));
    }

    #[test]
    fn test_quantization_info_from_dtype() {
        let q4 = QuantizationInfo::from_dtype(DType::Q4).unwrap();
        assert_eq!(q4.bits, 4);
        assert!(matches!(q4.scheme, QuantScheme::INT4));

        let q8 = QuantizationInfo::from_dtype(DType::Q8).unwrap();
        assert_eq!(q8.bits, 8);
        assert!(matches!(q8.scheme, QuantScheme::INT8));

        assert!(QuantizationInfo::from_dtype(DType::F16).is_none());
    }

    #[test]
    fn test_quantization_info_from_gguf_type() {
        let q4km = QuantizationInfo::from_gguf_type("Q4_K_M");
        assert_eq!(q4km.bits, 4);
        assert!(matches!(q4km.scheme, QuantScheme::GGUF(ref s) if s == "Q4_K_M"));

        let q80 = QuantizationInfo::from_gguf_type("Q8_0");
        assert_eq!(q80.bits, 8);

        let iq4xs = QuantizationInfo::from_gguf_type("IQ4_XS");
        assert_eq!(iq4xs.bits, 4);

        let q2k = QuantizationInfo::from_gguf_type("Q2_K");
        assert_eq!(q2k.bits, 2);
    }

    #[test]
    fn test_quantization_info_bytes_per_element() {
        let q4 = QuantizationInfo::from_gguf_type("Q4_K_M");
        assert!((q4.bytes_per_element() - 0.5).abs() < 0.001);

        let q8 = QuantizationInfo::from_gguf_type("Q8_0");
        assert!((q8.bytes_per_element() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quant_scheme_serde() {
        let schemes = vec![
            QuantScheme::None,
            QuantScheme::GGUF("Q4_K_M".to_string()),
            QuantScheme::AWQ,
            QuantScheme::GPTQ,
            QuantScheme::INT4,
            QuantScheme::INT8,
            QuantScheme::NF4,
        ];
        for s in schemes {
            let json = serde_json::to_string(&s).unwrap();
            let deser: QuantScheme = serde_json::from_str(&json).unwrap();
            assert_eq!(deser, s);
        }
    }
}
