//! ONNX Runtime adapter skeleton.
//!
//! The default Bloom build deliberately does not link ONNX Runtime.  This
//! module provides a stable `Engine` boundary, capability declaration, and
//! diagnostics so ONNX model packages can be routed and inspected without
//! making the core crate depend on a native runtime.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use bloomai_core::{
    DType, DeviceClass, DeviceKind, Modality, ModelFamily, ModelFormat, ModelManifest,
};

use crate::{
    core::parallelism::ParallelStrategy,
    engine::{Engine, EngineCapability, SupportLevel},
    model::LoadedModel,
};

pub struct OnnxRuntimeEngine;

impl OnnxRuntimeEngine {
    pub fn discover_model_file(model_path: &Path) -> Result<PathBuf> {
        if model_path.is_file() {
            if model_path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
            {
                return Ok(model_path.to_path_buf());
            }
            return Err(anyhow!(
                "ONNX Runtime engine expects a .onnx file or a directory containing one: {}",
                model_path.display()
            ));
        }

        if !model_path.is_dir() {
            return Err(anyhow!(
                "model path does not exist or is not readable: {}",
                model_path.display()
            ));
        }

        let preferred = ["model.onnx", "encoder.onnx", "decoder.onnx"];
        for name in preferred {
            let candidate = model_path.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }

        let mut onnx_files = std::fs::read_dir(model_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
            })
            .collect::<Vec<_>>();
        onnx_files.sort();

        onnx_files.into_iter().next().ok_or_else(|| {
            anyhow!(
                "no .onnx file found in {}; add model.onnx or point --model at a .onnx file",
                model_path.display()
            )
        })
    }
}

impl Engine for OnnxRuntimeEngine {
    fn name(&self) -> &'static str {
        "onnxruntime"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![
            Modality::Text,
            Modality::Vision,
            Modality::Audio,
            Modality::Multi,
        ]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Npu]
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![ModelFamily::Custom("*".to_string())],
            supported_dtypes: vec![
                DType::F32,
                DType::F16,
                DType::BF16,
                DType::I8,
                DType::U8,
                DType::Q8,
            ],
            supported_formats: vec![ModelFormat::Onnx],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
                DeviceClass::DiscreteGpu,
                DeviceClass::Npu,
            ],
            supported_modalities: self.supported_modalities(),
            supports_streaming: false,
            supports_quantized_models: true,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![crate::core::quantization::QuantMethod::Int8],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Skeleton,
            diagnostic_tips: vec![
                "Default Bloom builds only validate ONNX packages; they do not link ONNX Runtime yet."
                    .to_string(),
                "Use a model.onnx file or point --model directly at a .onnx file.".to_string(),
                "A future optional feature/plugin should bind ort and execution providers explicitly."
                    .to_string(),
            ],
            construction_guide:
                "Planned optional adapter: build/plugin with ONNX Runtime and selected execution providers."
                    .to_string(),
        }
    }

    fn supports(
        &self,
        manifest: &ModelManifest,
        device_cap: &bloomai_core::DeviceCapability,
    ) -> SupportLevel {
        crate::engine::default_engine_supports(&self.capability(), manifest, device_cap)
    }

    fn load(&self, model_path: &Path, _device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let onnx_file = Self::discover_model_file(model_path)?;
        Err(anyhow!(
            "ONNX Runtime adapter is a skeleton in the default build; found {}, but no runtime is linked. \
             Implement an optional ONNX Runtime plugin/feature before using --backend onnxruntime for real inference.",
            onnx_file.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloomai_core::{
        constants::GIB, DeviceCapability, MemoryTopology, ModelFile, ModelIoSchema,
        ModelMemoryProfile, PowerState, RuntimeHints, ThermalState,
    };

    fn cpu_capability() -> DeviceCapability {
        DeviceCapability {
            backend_name: "cpu".to_string(),
            vendor: None,
            device_class: DeviceClass::Cpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: 8 * GIB as usize,
            available_memory: 6 * GIB as usize,
            supported_dtypes: vec![DType::F32, DType::F16, DType::I8, DType::U8],
            supported_formats: vec![ModelFormat::Onnx],
            supports_mmap: true,
            has_quantization_kernels: true,
            supports_streaming: false,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: Some(4),
        }
    }

    fn onnx_manifest() -> ModelManifest {
        ModelManifest {
            id: "onnx-test".to_string(),
            family: ModelFamily::Custom("classifier".to_string()),
            version: "1.0".to_string(),
            io_schema: ModelIoSchema {
                inputs: vec![Modality::Vision],
                outputs: vec![Modality::Text],
            },
            primary_dtype: DType::F32,
            files: vec![ModelFile {
                name: "model.onnx".to_string(),
                format: ModelFormat::Onnx,
                size_bytes: 1024,
                hash_sha256: None,
                required: true,
            }],
            memory_profile: ModelMemoryProfile {
                min_ram_bytes: 0,
                min_vram_bytes: 0,
                recommended_ram_bytes: 0,
                recommended_vram_bytes: 0,
            },
            runtime_hints: RuntimeHints::default(),
            ..ModelManifest::default()
        }
    }

    #[test]
    fn capability_declares_onnx_contract() {
        let cap = OnnxRuntimeEngine.capability();
        assert_eq!(cap.engine_name, "onnxruntime");
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Skeleton);
        assert!(cap.supported_formats.contains(&ModelFormat::Onnx));
        assert!(cap.supported_modalities.contains(&Modality::Vision));
        assert!(cap.supports_quantized_models);
    }

    #[test]
    fn supports_onnx_manifest_as_skeleton_not_executable() {
        let level = OnnxRuntimeEngine.supports(&onnx_manifest(), &cpu_capability());
        assert!(matches!(level, SupportLevel::Unsupported(_)));
        assert!(level.reason().unwrap().contains("skeleton"));
    }

    #[test]
    fn discovers_single_onnx_file_and_directory_model() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.onnx");
        std::fs::write(&file, b"placeholder").unwrap();

        assert_eq!(OnnxRuntimeEngine::discover_model_file(&file).unwrap(), file);
        assert_eq!(
            OnnxRuntimeEngine::discover_model_file(dir.path()).unwrap(),
            file
        );
    }

    #[test]
    fn load_returns_actionable_runtime_missing_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"placeholder").unwrap();

        let err = match OnnxRuntimeEngine.load(dir.path(), DeviceKind::Cpu) {
            Ok(_) => panic!("ONNX skeleton should not load without a runtime"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("skeleton"));
        assert!(err.contains("no runtime is linked"));
    }
}
