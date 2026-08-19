//! Vulkan adapter skeleton.
//!
//! The default Bloom build deliberately does not compile Vulkan compute pipelines.
//! This module provides a stable `Engine` boundary, capability declaration, and
//! diagnostics so Vulkan SPIR-V packages can be routed and inspected without
//! introducing external Vulkan SDK linking dependencies by default.

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

use crate::{ModelInput, ModelMetadata, ModelOutput};
use bloomai_core::GenerationParams;
use std::sync::Arc;

pub struct VulkanPluginModel {
    #[allow(dead_code)]
    lib: Arc<libloading::Library>,
    #[allow(dead_code)]
    handle: *mut std::ffi::c_void,
    metadata: ModelMetadata,
}

unsafe impl Send for VulkanPluginModel {}
unsafe impl Sync for VulkanPluginModel {}

impl LoadedModel for VulkanPluginModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, _input: ModelInput, _params: &GenerationParams) -> Result<ModelOutput> {
        Err(anyhow!(
            "Inference not fully implemented in Vulkan plugin skeleton"
        ))
    }
}

pub struct VulkanEngine;

impl VulkanEngine {
    pub fn discover_model_file(model_path: &Path) -> Result<PathBuf> {
        if model_path.is_file() {
            if let Some(ext) = model_path.extension().and_then(|e| e.to_str())
                && (ext.eq_ignore_ascii_case("spv") || ext.eq_ignore_ascii_case("spirv"))
            {
                return Ok(model_path.to_path_buf());
            }
            return Err(anyhow!(
                "Vulkan engine expects a .spv/.spirv file or a directory containing one: {}",
                model_path.display()
            ));
        }

        if !model_path.is_dir() {
            return Err(anyhow!(
                "model path does not exist or is not readable: {}",
                model_path.display()
            ));
        }

        // Search for directories or files ending in .spv or .spirv
        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(model_path)?.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && (ext.eq_ignore_ascii_case("spv") || ext.eq_ignore_ascii_case("spirv"))
            {
                candidates.push(path);
            }
        }
        candidates.sort();

        candidates.into_iter().next().ok_or_else(|| {
            anyhow!(
                "no Vulkan SPIR-V (.spv/.spirv) found in {}; point --model at a Vulkan SPIR-V model file or directory",
                model_path.display()
            )
        })
    }
}

impl Engine for VulkanEngine {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text, Modality::Vision]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu]
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
            supported_formats: vec![ModelFormat::VulkanSpirv],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
                DeviceClass::DiscreteGpu,
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
                "Default Bloom builds only validate Vulkan SPIR-V packages; they do not compile compute pipelines yet.".to_string(),
                "Use a .spv or .spirv model format on Vulkan compatible platforms.".to_string(),
                "A future plugin or wgpu/kompute feature should enable native Vulkan shader compilation and execution.".to_string(),
            ],
            construction_guide:
                "Planned optional adapter: build/plugin compiling SPIR-V models using wgpu/kompute APIs."
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
        let vulkan_file = Self::discover_model_file(model_path)?;

        let lib_filename = if cfg!(target_os = "windows") {
            "bloom_vulkan.dll"
        } else if cfg!(target_os = "macos") {
            "libbloom_vulkan.dylib"
        } else {
            "libbloom_vulkan.so"
        };

        let plugin_paths = vec![
            PathBuf::from("plugins").join(lib_filename),
            PathBuf::from(lib_filename),
            PathBuf::from("/usr/local/lib").join(lib_filename),
        ];

        for path in plugin_paths {
            if path.exists()
                && let Ok(lib) = unsafe { libloading::Library::new(&path) }
            {
                let lib = Arc::new(lib);
                unsafe {
                    if let Ok(load_fn) =
                        lib.get::<unsafe extern "C" fn(*const u8, usize) -> *mut std::ffi::c_void>(
                            b"bloom_vulkan_load\0",
                        )
                    {
                        let path_str = vulkan_file.to_string_lossy();
                        let handle = load_fn(path_str.as_ptr(), path_str.len());
                        if !handle.is_null() {
                            let manifest = crate::manifest_adapter::load_manifest(model_path)?;
                            return Ok(Box::new(VulkanPluginModel {
                                lib: lib.clone(),
                                handle,
                                metadata: ModelMetadata {
                                    id: vulkan_file
                                        .file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("unknown")
                                        .to_string(),
                                    modality: Modality::Text,
                                    quantized: true,
                                    manifest,
                                },
                            }));
                        }
                    }
                }
            }
        }

        Err(anyhow!(
            "Vulkan adapter is a skeleton in the default build; found {}, but no Vulkan execution runtime is linked. \
             Implement an optional Vulkan plugin/feature before using --backend vulkan for real inference.",
            vulkan_file.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_declares_vulkan_contract() {
        let cap = VulkanEngine.capability();
        assert_eq!(cap.engine_name, "vulkan");
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Skeleton);
        assert!(cap.supported_formats.contains(&ModelFormat::VulkanSpirv));
        assert!(cap.supported_devices.contains(&DeviceClass::DiscreteGpu));
    }

    #[test]
    fn discovers_spv_and_spirv() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.spv");
        std::fs::write(&file, b"placeholder").unwrap();

        assert_eq!(VulkanEngine::discover_model_file(&file).unwrap(), file);
        assert_eq!(VulkanEngine::discover_model_file(dir.path()).unwrap(), file);
    }

    #[test]
    fn load_returns_actionable_runtime_missing_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.spirv"), b"placeholder").unwrap();

        let err = match VulkanEngine.load(dir.path(), DeviceKind::Cpu) {
            Ok(_) => panic!("Vulkan skeleton should not load without a runtime"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("skeleton"));
        assert!(err.contains("no Vulkan execution runtime is linked"));
    }
}
