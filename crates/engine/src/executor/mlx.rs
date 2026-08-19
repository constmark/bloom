//! MLX adapter skeleton.
//!
//! The default Bloom build deliberately does not link the Apple MLX framework or
//! its Python runtime bindings. This module provides a stable `Engine` boundary,
//! capability declaration, and diagnostics so MLX weight packages can be
//! routed and inspected without introducing native framework linking dependencies.

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

pub struct MlxEngine;

impl MlxEngine {
    pub fn discover_model_file(model_path: &Path) -> Result<PathBuf> {
        if model_path.is_file() {
            if let Some(ext) = model_path.extension().and_then(|e| e.to_str())
                && (ext.eq_ignore_ascii_case("mlx")
                    || ext.eq_ignore_ascii_case("npz")
                    || ext.eq_ignore_ascii_case("safetensors"))
            {
                return Ok(model_path.to_path_buf());
            }
            return Err(anyhow!(
                "MLX engine expects a .mlx/.npz/.safetensors file or a directory containing them: {}",
                model_path.display()
            ));
        }

        if !model_path.is_dir() {
            return Err(anyhow!(
                "model path does not exist or is not readable: {}",
                model_path.display()
            ));
        }

        // MLX directories usually contain weights.safetensors or weights.npz or config.json
        for name in ["weights.safetensors", "weights.npz", "config.json"] {
            let candidate = model_path.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }

        // Fallback to searching for any .mlx, .npz, or .safetensors files
        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(model_path)?.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && (ext.eq_ignore_ascii_case("mlx")
                    || ext.eq_ignore_ascii_case("npz")
                    || ext.eq_ignore_ascii_case("safetensors"))
            {
                candidates.push(path);
            }
        }
        candidates.sort();

        candidates.into_iter().next().ok_or_else(|| {
            anyhow!(
                "no MLX weights (.safetensors/.npz/.mlx) found in {}; point --model at an MLX directory or file",
                model_path.display()
            )
        })
    }
}

impl Engine for MlxEngine {
    fn name(&self) -> &'static str {
        "mlx"
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
        vec![DeviceKind::Cpu, DeviceKind::Gpu]
    }

    fn default_device(&self) -> DeviceKind {
        DeviceKind::Gpu
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![
                ModelFamily::Qwen,
                ModelFamily::Gemma,
                ModelFamily::Llama,
                ModelFamily::Custom("*".to_string()),
            ],
            supported_dtypes: vec![
                DType::F32,
                DType::F16,
                DType::BF16,
            ],
            supported_formats: vec![ModelFormat::Mlx],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
            ],
            supported_modalities: self.supported_modalities(),
            supports_streaming: true,
            supports_quantized_models: true,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Skeleton,
            diagnostic_tips: vec![
                "Default Bloom builds only validate MLX weight packages; they do not link MLX framework yet.".to_string(),
                "MLX runs natively on Apple Silicon GPU and CPU using Unified Memory Architecture.".to_string(),
                "A future plugin or native bridge (Python/C++) should enable actual MLX model execution.".to_string(),
            ],
            construction_guide:
                "Planned optional adapter: build/plugin executing MLX models via Python/Rust bindings."
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
        let mlx_file = Self::discover_model_file(model_path)?;
        Err(anyhow!(
            "MLX adapter is a skeleton in the default build; found {}, but no MLX runtime is linked. \
             Implement an optional MLX plugin/feature before using --backend mlx for real inference.",
            mlx_file.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_declares_mlx_contract() {
        let cap = MlxEngine.capability();
        assert_eq!(cap.engine_name, "mlx");
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Skeleton);
        assert!(cap.supported_formats.contains(&ModelFormat::Mlx));
        assert!(cap.supported_devices.contains(&DeviceClass::IntegratedGpu));
    }

    #[test]
    fn discovers_weights_safetensors_and_npz() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.safetensors");
        std::fs::write(&file, b"placeholder").unwrap();

        assert_eq!(MlxEngine::discover_model_file(&file).unwrap(), file);
        assert_eq!(MlxEngine::discover_model_file(dir.path()).unwrap(), file);
    }

    #[test]
    fn load_returns_actionable_runtime_missing_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("weights.npz"), b"placeholder").unwrap();

        let err = match MlxEngine.load(dir.path(), DeviceKind::Cpu) {
            Ok(_) => panic!("MLX skeleton should not load without a runtime"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("skeleton"));
        assert!(err.contains("no MLX runtime is linked"));
    }
}
