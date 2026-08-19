//! CoreML adapter skeleton.
//!
//! The default Bloom build deliberately does not link Apple CoreML runtime APIs.
//! This module provides a stable `Engine` boundary, capability declaration, and
//! diagnostics so CoreML model packages can be routed and inspected without
//! making the core crate depend on native framework linking.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use bloomai_core::{
    DType, DeviceClass, DeviceKind, Modality, ModelFamily, ModelFormat, ModelManifest,
};

use crate::{
    core::parallelism::ParallelStrategy,
    engine::{Engine, EngineCapability, SupportLevel, default_engine_supports},
    model::LoadedModel,
};

pub struct CoreMlEngine;

impl CoreMlEngine {
    pub fn discover_model_file(model_path: &Path) -> Result<PathBuf> {
        if model_path.is_file() {
            if let Some(ext) = model_path.extension().and_then(|e| e.to_str())
                && (ext.eq_ignore_ascii_case("mlmodel") || ext.eq_ignore_ascii_case("mlpackage"))
            {
                return Ok(model_path.to_path_buf());
            }
            return Err(anyhow!(
                "CoreML engine expects a .mlmodel/.mlpackage file or a directory containing one: {}",
                model_path.display()
            ));
        }

        if !model_path.is_dir() {
            return Err(anyhow!(
                "model path does not exist or is not readable: {}",
                model_path.display()
            ));
        }

        // Search for directories or files ending in .mlpackage or .mlmodel
        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(model_path)?.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && (ext.eq_ignore_ascii_case("mlpackage") || ext.eq_ignore_ascii_case("mlmodel"))
            {
                candidates.push(path);
            }
        }
        candidates.sort();

        candidates.into_iter().next().ok_or_else(|| {
            anyhow!(
                "no CoreML (.mlpackage/.mlmodel) found in {}; point --model at a CoreML model package",
                model_path.display()
            )
        })
    }
}

impl Engine for CoreMlEngine {
    fn name(&self) -> &'static str {
        "coreml"
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

    fn default_device(&self) -> DeviceKind {
        DeviceKind::Gpu
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![ModelFamily::Custom("*".to_string())],
            supported_dtypes: vec![
                DType::F32,
                DType::F16,
            ],
            supported_formats: vec![ModelFormat::CoreMl],
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
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Skeleton,
            diagnostic_tips: vec![
                "Default Bloom builds only validate CoreML packages; they do not link CoreML API yet.".to_string(),
                "Use a .mlmodel or .mlpackage model format on Apple Silicon platforms.".to_string(),
                "A future plugin or native feature should enable native CoreML model compilation and execution.".to_string(),
            ],
            construction_guide:
                "Planned optional adapter: build/plugin compiling CoreML models using Apple's CoreML APIs."
                    .to_string(),
        }
    }

    fn supports(
        &self,
        manifest: &ModelManifest,
        device_cap: &bloomai_core::DeviceCapability,
    ) -> SupportLevel {
        default_engine_supports(&self.capability(), manifest, device_cap)
    }

    fn load(&self, model_path: &Path, _device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let coreml_file = Self::discover_model_file(model_path)?;
        Err(anyhow!(
            "CoreML adapter is a skeleton in the default build; found {}, but no CoreML API runtime is linked. \
             Implement an optional CoreML plugin/feature before using --backend coreml for real inference.",
            coreml_file.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_declares_coreml_contract() {
        let cap = CoreMlEngine.capability();
        assert_eq!(cap.engine_name, "coreml");
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Skeleton);
        assert!(cap.supported_formats.contains(&ModelFormat::CoreMl));
        assert!(cap.supported_devices.contains(&DeviceClass::Npu));
    }

    #[test]
    fn discovers_mlpackage_and_mlmodel() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.mlpackage");
        std::fs::write(&file, b"placeholder").unwrap();

        assert_eq!(CoreMlEngine::discover_model_file(&file).unwrap(), file);
        assert_eq!(CoreMlEngine::discover_model_file(dir.path()).unwrap(), file);
    }

    #[test]
    fn load_returns_actionable_runtime_missing_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.mlmodel"), b"placeholder").unwrap();

        let err = match CoreMlEngine.load(dir.path(), DeviceKind::Cpu) {
            Ok(_) => panic!("CoreML skeleton should not load without a runtime"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("skeleton"));
        assert!(err.contains("no CoreML API runtime is linked"));
    }
}
