//! Plugin manifest types for the Bloom ecosystem.
//!
//! These types define the schema for community-contributed backends, engines,
//! processors, operators, and model packages. Each plugin is described by a
//! JSON manifest that the runtime can load at startup.

use serde::{Deserialize, Serialize};

use crate::{DType, DeviceClass, Modality, ModelFamily, ModelFormat};

// ---------------------------------------------------------------------------
// Common metadata
// ---------------------------------------------------------------------------

/// Common metadata shared by all plugin manifests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Unique plugin name (reverse-DNS recommended, e.g. "com.acme.npu-backend").
    pub name: String,
    /// Semantic version (e.g. "1.2.0").
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// Author or organization.
    pub author: String,
    /// License identifier (SPDX, e.g. "MIT", "Apache-2.0").
    pub license: String,
    /// Homepage or repository URL.
    pub homepage: Option<String>,
    /// Target platforms (e.g. "linux-x86_64", "macos-aarch64").
    pub platforms: Vec<String>,
    /// Minimum Bloom runtime version required.
    pub min_runtime_version: Option<String>,
}

// ---------------------------------------------------------------------------
// Backend plugin
// ---------------------------------------------------------------------------

/// Manifest for a backend plugin (new chip, vendor SDK, remote device).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendPluginManifest {
    pub metadata: PluginMetadata,
    /// Entry point: shared library path or script path.
    pub entry_point: PluginEntryPoint,
    /// Declared device class.
    pub device_class: DeviceClass,
    /// Supported dtypes.
    pub supported_dtypes: Vec<DType>,
    /// Supported model formats.
    pub supported_formats: Vec<ModelFormat>,
    /// Whether the backend supports mmap.
    pub supports_mmap: bool,
    /// Whether the backend has quantization kernels.
    pub has_quantization_kernels: bool,
    /// Whether the backend supports streaming inference.
    pub supports_streaming: bool,
    /// Estimated memory overhead in bytes.
    pub memory_overhead_bytes: Option<usize>,
    /// Capability probe script (optional, for runtime detection).
    pub probe_script: Option<String>,
}

// ---------------------------------------------------------------------------
// Engine plugin
// ---------------------------------------------------------------------------

/// Manifest for an engine plugin (new model family or execution framework).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnginePluginManifest {
    pub metadata: PluginMetadata,
    /// Entry point: shared library path or script path.
    pub entry_point: PluginEntryPoint,
    /// Supported model families.
    pub supported_families: Vec<ModelFamily>,
    /// Supported dtypes.
    pub supported_dtypes: Vec<DType>,
    /// Supported model formats.
    pub supported_formats: Vec<ModelFormat>,
    /// Supported device classes.
    pub supported_devices: Vec<DeviceClass>,
    /// Supported input modalities.
    pub supported_modalities: Vec<Modality>,
    /// Whether the engine supports streaming output.
    pub supports_streaming: bool,
    /// Whether the engine supports quantized models.
    pub supports_quantized_models: bool,
    /// Maximum context length in tokens (if applicable).
    pub max_context_tokens: Option<usize>,
    /// Required backend names (the engine depends on these backends).
    pub required_backends: Vec<String>,
    /// Example model IDs this engine can run.
    pub example_models: Vec<String>,
}

// ---------------------------------------------------------------------------
// Processor plugin
// ---------------------------------------------------------------------------

/// Manifest for a processor plugin (tokenizer, image preprocessor, audio codec).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorPluginManifest {
    pub metadata: PluginMetadata,
    /// Entry point: shared library path or script path.
    pub entry_point: PluginEntryPoint,
    /// Input modalities this processor handles.
    pub input_modalities: Vec<Modality>,
    /// Output modalities this processor produces.
    pub output_modalities: Vec<Modality>,
    /// Input schema description (e.g. "raw audio PCM 16kHz mono").
    pub input_schema: Option<String>,
    /// Output schema description (e.g. "tokenized ids, max_len=512").
    pub output_schema: Option<String>,
    /// External dependencies (e.g. Python packages).
    pub dependencies: Vec<PluginDependency>,
    /// Whether the processor is deterministic (same input → same output).
    pub deterministic: bool,
}

// ---------------------------------------------------------------------------
// Operator plugin
// ---------------------------------------------------------------------------

/// Manifest for an operator plugin (custom kernel, optimized matmul, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorPluginManifest {
    pub metadata: PluginMetadata,
    /// Entry point: shared library path or script path.
    pub entry_point: PluginEntryPoint,
    /// Operator name (e.g. "flash_attention", "quantized_matmul").
    pub operator_name: String,
    /// Supported tensor shapes (empty = any shape).
    pub supported_shapes: Vec<Vec<usize>>,
    /// Supported dtypes.
    pub supported_dtypes: Vec<DType>,
    /// Target backends.
    pub target_backends: Vec<String>,
    /// Benchmark results (tokens/s, latency, etc.).
    pub benchmarks: Vec<OperatorBenchmark>,
    /// Fallback operator name if this one is unavailable.
    pub fallback: Option<String>,
}

/// Benchmark result for an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorBenchmark {
    /// Backend used.
    pub backend: String,
    /// Input shape.
    pub shape: Vec<usize>,
    /// Dtype used.
    pub dtype: DType,
    /// Latency in microseconds.
    pub latency_us: f64,
    /// Throughput (operations per second).
    pub throughput_ops_per_sec: f64,
}

// ---------------------------------------------------------------------------
// Model package manifest
// ---------------------------------------------------------------------------

/// Manifest for publishing a model package to the Bloom ecosystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackageManifest {
    pub metadata: PluginMetadata,
    /// Model family.
    pub family: ModelFamily,
    /// Model version.
    pub model_version: String,
    /// Primary dtype of the weights.
    pub primary_dtype: DType,
    /// Available quantization variants.
    pub quantizations: Vec<QuantizationVariant>,
    /// Files included in the package.
    pub files: Vec<ModelPackageFile>,
    /// Recommended backends (in priority order).
    pub recommended_backends: Vec<String>,
    /// Required processor names.
    pub required_processors: Vec<String>,
    /// Total model size in bytes.
    pub total_size_bytes: usize,
    /// Source URL for downloading the package.
    pub source_url: Option<String>,
    /// SHA-256 hash of the package archive.
    pub package_hash: Option<String>,
    /// License file path within the package.
    pub license_file: Option<String>,
    /// Input/output schema description.
    pub io_description: Option<String>,
}

/// A file within a model package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackageFile {
    /// Relative path within the package.
    pub path: String,
    /// File format.
    pub format: ModelFormat,
    /// File size in bytes.
    pub size_bytes: usize,
    /// SHA-256 hash.
    pub sha256: Option<String>,
}

/// Quantization variant available for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationVariant {
    /// Quantization name (e.g. "Q4_K_M", "INT8", "AWQ-4bit").
    pub name: String,
    /// Dtype after quantization.
    pub dtype: DType,
    /// Model size after quantization in bytes.
    pub size_bytes: usize,
    /// Quality score [0, 1] relative to the original.
    pub quality_score: Option<f32>,
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Entry point for a plugin (shared library or script).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PluginEntryPoint {
    /// Native shared library (.so, .dylib, .dll).
    #[serde(rename = "native")]
    NativeLibrary { path: String },
    /// Python script or module.
    #[serde(rename = "python")]
    PythonScript {
        module: String,
        function: Option<String>,
    },
    /// WebAssembly module.
    #[serde(rename = "wasm")]
    Wasm { path: String },
    /// External process (communicated via stdin/stdout).
    #[serde(rename = "process")]
    Process { command: String, args: Vec<String> },
}

/// External dependency required by a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    /// Dependency name (e.g. "numpy", "torch", "openvino").
    pub name: String,
    /// Version constraint (e.g. ">=1.24", "==2.1.0").
    pub version: Option<String>,
    /// Package manager (e.g. "pip", "apt", "brew").
    pub package_manager: Option<String>,
    /// Whether this dependency is optional.
    pub optional: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_plugin_manifest_serde() {
        let manifest = BackendPluginManifest {
            metadata: PluginMetadata {
                name: "com.acme.npu-backend".into(),
                version: "1.0.0".into(),
                description: "Custom NPU backend for Acme chips".into(),
                author: "Acme Corp".into(),
                license: "MIT".into(),
                homepage: Some("https://acme.com".into()),
                platforms: vec!["linux-x86_64".into()],
                min_runtime_version: Some("0.2.0".into()),
            },
            entry_point: PluginEntryPoint::NativeLibrary {
                path: "libacme_npu.so".into(),
            },
            device_class: DeviceClass::Npu,
            supported_dtypes: vec![DType::F16, DType::Q8],
            supported_formats: vec![ModelFormat::OpenVinoIr],
            supports_mmap: true,
            has_quantization_kernels: true,
            supports_streaming: false,
            memory_overhead_bytes: Some(64 * 1024 * 1024),
            probe_script: Some("python3 -c 'import acme_npu'".into()),
        };
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        assert!(json.contains("com.acme.npu-backend"));
        let deser: BackendPluginManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.metadata.name, "com.acme.npu-backend");
        assert_eq!(deser.device_class, DeviceClass::Npu);
    }

    #[test]
    fn test_engine_plugin_manifest_serde() {
        let manifest = EnginePluginManifest {
            metadata: PluginMetadata {
                name: "org.community.llama-engine".into(),
                version: "0.1.0".into(),
                description: "llama.cpp-based engine".into(),
                author: "Community".into(),
                license: "Apache-2.0".into(),
                homepage: None,
                platforms: vec!["linux-x86_64".into(), "macos-aarch64".into()],
                min_runtime_version: None,
            },
            entry_point: PluginEntryPoint::NativeLibrary {
                path: "libllama_engine.so".into(),
            },
            supported_families: vec![ModelFamily::Llama],
            supported_dtypes: vec![DType::F16, DType::Q8, DType::Q4],
            supported_formats: vec![ModelFormat::Gguf],
            supported_devices: vec![DeviceClass::Cpu, DeviceClass::DiscreteGpu],
            supported_modalities: vec![Modality::Text],
            supports_streaming: true,
            supports_quantized_models: true,
            max_context_tokens: Some(8192),
            required_backends: vec!["cpu".into()],
            example_models: vec!["llama-3-8b".into()],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let deser: EnginePluginManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.supported_families, vec![ModelFamily::Llama]);
        assert!(deser.supports_streaming);
    }

    #[test]
    fn test_processor_plugin_manifest_serde() {
        let manifest = ProcessorPluginManifest {
            metadata: PluginMetadata {
                name: "io.whisper-audio".into(),
                version: "0.1.0".into(),
                description: "Whisper audio preprocessor".into(),
                author: "Community".into(),
                license: "MIT".into(),
                homepage: None,
                platforms: vec![],
                min_runtime_version: None,
            },
            entry_point: PluginEntryPoint::PythonScript {
                module: "whisper_preprocess".into(),
                function: Some("process".into()),
            },
            input_modalities: vec![Modality::Audio],
            output_modalities: vec![Modality::Audio],
            input_schema: Some("raw PCM 16kHz mono".into()),
            output_schema: Some("mel spectrogram 80x3000".into()),
            dependencies: vec![PluginDependency {
                name: "librosa".into(),
                version: Some(">=0.10".into()),
                package_manager: Some("pip".into()),
                optional: false,
            }],
            deterministic: true,
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let deser: ProcessorPluginManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.dependencies.len(), 1);
        assert!(deser.deterministic);
    }

    #[test]
    fn test_operator_plugin_manifest_serde() {
        let manifest = OperatorPluginManifest {
            metadata: PluginMetadata {
                name: "ops.flash-attn".into(),
                version: "2.0.0".into(),
                description: "Flash attention operator".into(),
                author: "Dao-AILab".into(),
                license: "BSD-3-Clause".into(),
                homepage: None,
                platforms: vec!["linux-x86_64".into()],
                min_runtime_version: None,
            },
            entry_point: PluginEntryPoint::NativeLibrary {
                path: "libflash_attn.so".into(),
            },
            operator_name: "flash_attention".into(),
            supported_shapes: vec![],
            supported_dtypes: vec![DType::F16, DType::BF16],
            target_backends: vec!["cuda".into()],
            benchmarks: vec![OperatorBenchmark {
                backend: "cuda".into(),
                shape: vec![1, 32, 2048, 128],
                dtype: DType::F16,
                latency_us: 250.0,
                throughput_ops_per_sec: 4000.0,
            }],
            fallback: Some("standard_attention".into()),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let deser: OperatorPluginManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.operator_name, "flash_attention");
        assert_eq!(deser.benchmarks.len(), 1);
    }

    #[test]
    fn test_model_package_manifest_serde() {
        let manifest = ModelPackageManifest {
            metadata: PluginMetadata {
                name: "models.qwen2-7b".into(),
                version: "1.0.0".into(),
                description: "Qwen2 7B instruct model".into(),
                author: "Alibaba Cloud".into(),
                license: "Apache-2.0".into(),
                homepage: Some("https://huggingface.co/Qwen/Qwen2-7B-Instruct".into()),
                platforms: vec![],
                min_runtime_version: None,
            },
            family: ModelFamily::Qwen,
            model_version: "7b-instruct".into(),
            primary_dtype: DType::F16,
            quantizations: vec![QuantizationVariant {
                name: "Q4_K_M".into(),
                dtype: DType::Q4,
                size_bytes: 4 * 1024 * 1024 * 1024,
                quality_score: Some(0.95),
            }],
            files: vec![ModelPackageFile {
                path: "model.gguf".into(),
                format: ModelFormat::Gguf,
                size_bytes: 4 * 1024 * 1024 * 1024,
                sha256: Some("abc123".into()),
            }],
            recommended_backends: vec!["cpu".into(), "cuda".into()],
            required_processors: vec!["qwen-tokenizer".into()],
            total_size_bytes: 4 * 1024 * 1024 * 1024,
            source_url: Some("https://hf.co/Qwen/Qwen2-7B-Instruct-GGUF".into()),
            package_hash: Some("sha256:abc123def456".into()),
            license_file: Some("LICENSE".into()),
            io_description: Some("Text input, text output".into()),
        };
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        assert!(json.contains("qwen2-7b"));
        let deser: ModelPackageManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.family, ModelFamily::Qwen);
        assert_eq!(deser.quantizations.len(), 1);
        assert_eq!(deser.files.len(), 1);
    }

    #[test]
    fn test_plugin_entry_point_variants() {
        let native = PluginEntryPoint::NativeLibrary {
            path: "lib.so".into(),
        };
        let python = PluginEntryPoint::PythonScript {
            module: "my_mod".into(),
            function: Some("run".into()),
        };
        let wasm = PluginEntryPoint::Wasm {
            path: "op.wasm".into(),
        };
        let process = PluginEntryPoint::Process {
            command: "python3".into(),
            args: vec!["script.py".into()],
        };
        for ep in [native, python, wasm, process] {
            let json = serde_json::to_string(&ep).unwrap();
            let _: PluginEntryPoint = serde_json::from_str(&json).unwrap();
        }
    }
}
