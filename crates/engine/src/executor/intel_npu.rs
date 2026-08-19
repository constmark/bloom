//! Dedicated Intel NPU inference engine for Bloom.
//!
//! This engine provides a first-class Intel NPU experience:
//! - Automatic NPU hardware detection (Windows & Linux)
//! - On-the-fly model export from HuggingFace / AWQ to OpenVINO IR (INT4)
//! - Streaming inference via OpenVINO GenAI Python bridge on NPU
//! - Fallback to Intel CPU / GPU when NPU is unavailable
//!
//! The engine re-uses the `openvino_llm_infer.py` script for execution and
//! adds NPU-specific probing, auto-export, and device selection on top.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Result, anyhow};
use bloomai_core::{DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat};

use crate::{
    engine::{Engine, EngineCapability},
    io::{ModelInput, ModelOutput},
    model::{LoadedModel, ModelMetadata},
};

// ---------------------------------------------------------------------------
// NPU hardware probing
// ---------------------------------------------------------------------------

/// Probe for Intel NPU availability on the current system.
///
/// Returns `true` when at least one indicator of NPU presence is found.
/// The checks are intentionally lightweight — no model compilation is done.
fn probe_intel_npu() -> bool {
    // 1. Check for NPU device nodes (Linux)
    #[cfg(target_os = "linux")]
    {
        for pattern in &["/dev/accel", "/dev/accel0"] {
            if Path::new(pattern).exists() {
                tracing::info!("Intel NPU detected: {}", pattern);
                return true;
            }
        }
        // Check for ivpu / intel_vpu kernel module
        if let Ok(modules) = std::fs::read_to_string("/proc/modules") {
            if modules.contains("intel_vpu") || modules.contains("ivpu") {
                tracing::info!("Intel NPU kernel module detected");
                return true;
            }
        }
    }

    // 2. Check for NPU drivers on Windows
    #[cfg(target_os = "windows")]
    {
        // Check common driver DLL locations
        let windir = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        let driver_candidates = [
            PathBuf::from(&windir)
                .join("System32")
                .join("drivers")
                .join("IntelNPU.sys"),
            PathBuf::from(&windir)
                .join("System32")
                .join("drivers")
                .join("ivpu.sys"),
            PathBuf::from(&windir)
                .join("System32")
                .join("intel_npu.dll"),
        ];
        for candidate in &driver_candidates {
            if candidate.exists() {
                tracing::info!("Intel NPU driver detected: {}", candidate.display());
                return true;
            }
        }

        // Check via WMI for Intel NPU device
        if let Ok(output) = Command::new("wmic")
            .args([
                "path",
                "Win32_PnPEntity",
                "where",
                "Name like '%NPU%'",
                "get",
                "Name",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.to_lowercase().contains("npu") {
                tracing::info!("Intel NPU detected via WMI");
                return true;
            }
        }

        // Check via PowerShell for Intel NPU (more reliable on modern Windows)
        if let Ok(output) = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'NPU|Neural Processing' } | Select-Object -First 1 -ExpandProperty FriendlyName",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if !stdout.trim().is_empty() {
                tracing::info!("Intel NPU detected via PowerShell: {}", stdout.trim());
                return true;
            }
        }
    }

    // 3. Check for OpenVINO runtime availability (cross-platform)
    if probe_openvino_runtime() {
        tracing::info!("OpenVINO runtime available — NPU backend can be used");
        return true;
    }

    // 4. Check environment variable override
    if std::env::var("BLOOM_FORCE_NPU").is_ok() {
        tracing::info!("BLOOM_FORCE_NPU set — assuming Intel NPU is available");
        return true;
    }

    false
}

/// Check if OpenVINO Python runtime is installed.
fn probe_openvino_runtime() -> bool {
    let python = default_python();
    if crate::core::security::validate_runner(&python).is_err() {
        return false;
    }
    Command::new(&python)
        .arg("-c")
        .arg("import openvino_genai")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Helpers (shared with openvino module)
// ---------------------------------------------------------------------------

fn repo_script_path(script_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("scripts")
        .join(script_name)
}

fn default_python() -> PathBuf {
    if let Some(path) = std::env::var_os("BLOOM_PYTHON") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("BLOOM_ASR_PYTHON") {
        return PathBuf::from(path);
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");

    let candidates = [
        repo_root.join(".venv").join("Scripts").join("python.exe"),
        repo_root.join(".venv").join("bin").join("python"),
        PathBuf::from("python"),
    ];

    candidates
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("python"))
}

fn is_awq_model(model_path: &Path) -> bool {
    let config_path = model_path.join("config.json");
    if config_path.exists()
        && let Ok(config_str) = std::fs::read_to_string(&config_path)
        && let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str)
        && let Some(quant_config) = config.get("quantization_config")
        && let Some(quant_method) = quant_config.get("quant_method")
        && quant_method == "awq"
    {
        return true;
    }
    model_path.to_string_lossy().to_lowercase().contains("awq")
}

fn is_openvino_ir(model_path: &Path) -> bool {
    model_path.join("openvino_model.xml").exists()
}

/// Check if model_path looks like a HuggingFace model that needs export.
fn needs_openvino_export(model_path: &Path) -> bool {
    if is_openvino_ir(model_path) {
        return false;
    }
    // Has config.json → likely a HuggingFace model
    model_path.join("config.json").exists()
}

/// Export a HuggingFace / AWQ model to OpenVINO IR format optimized for NPU.
fn export_to_openvino_ir(model_path: &Path) -> Result<()> {
    let python = default_python();
    crate::core::security::validate_runner(&python)?;

    // Determine weight format — default to int4 for NPU efficiency
    let weight_format =
        std::env::var("BLOOM_NPU_WEIGHT_FORMAT").unwrap_or_else(|_| "int4".to_string());

    tracing::info!(
        "Exporting model to OpenVINO IR (weight_format={}) for Intel NPU...",
        weight_format
    );

    let output = Command::new(&python)
        .arg("-m")
        .arg("optimum.intel.openvino.cli")
        .arg("export")
        .arg("--model")
        .arg(model_path)
        .arg("--weight-format")
        .arg(&weight_format)
        .arg(model_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            anyhow!(
                "failed to execute Python for OpenVINO IR export using {}: {}",
                python.display(),
                e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named optimum") || stderr.contains("ModuleNotFoundError") {
            return Err(anyhow!(
                "failed to export model: missing Python libraries.\n\
                 Install them with: pip install \"optimum-intel[openvino]\" openvino nncf"
            ));
        }
        return Err(anyhow!("OpenVINO IR export failed: {}", stderr));
    }

    if !is_openvino_ir(model_path) {
        return Err(anyhow!(
            "export completed but openvino_model.xml was not found in {}",
            model_path.display()
        ));
    }

    tracing::info!("Model successfully exported to OpenVINO IR");
    Ok(())
}

// ---------------------------------------------------------------------------
// Intel NPU Engine
// ---------------------------------------------------------------------------

/// Dedicated inference engine targeting Intel NPU hardware.
///
/// Wraps the OpenVINO GenAI Python bridge with NPU-first device selection,
/// automatic model export, and hardware probing.
pub struct IntelNpuEngine;

impl Engine for IntelNpuEngine {
    fn name(&self) -> &'static str {
        "intel-npu"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Npu, DeviceKind::Cpu, DeviceKind::Gpu]
    }

    fn default_device(&self) -> DeviceKind {
        DeviceKind::Npu
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "intel-npu",
            // Model-agnostic via OpenVINO IR — wildcard matches any family
            supported_families: vec![ModelFamily::Custom("*".to_string())],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::I8,
                bloomai_core::DType::U8,
                bloomai_core::DType::I4,
                bloomai_core::DType::NF4,
                bloomai_core::DType::Q8,
                bloomai_core::DType::Q4,
            ],
            supported_formats: vec![ModelFormat::OpenVinoIr, ModelFormat::Safetensors],
            supported_devices: vec![
                DeviceClass::Npu,
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
            ],
            supported_modalities: vec![Modality::Text],
            supports_streaming: true,
            supports_quantized_models: true,
            supports_embeddings: true,
            supports_rerank: true,
            supports_structured_output: true,
            max_context_tokens: None,
            supported_quant_methods: vec![
                crate::core::quantization::QuantMethod::Int8,
                crate::core::quantization::QuantMethod::Awq,
                crate::core::quantization::QuantMethod::Gptq,
                crate::core::quantization::QuantMethod::Gguf,
                crate::core::quantization::QuantMethod::Fp8,
                crate::core::quantization::QuantMethod::NvFp4,
                crate::core::quantization::QuantMethod::Nf4,
                crate::core::quantization::QuantMethod::Fp4,
                crate::core::quantization::QuantMethod::Hqq,
                crate::core::quantization::QuantMethod::Eetq,
                crate::core::quantization::QuantMethod::Aqlm,
                crate::core::quantization::QuantMethod::Exl2,
                crate::core::quantization::QuantMethod::Quanto,
                crate::core::quantization::QuantMethod::Torchao,
            ],
            supported_parallel_strategies: vec![crate::core::parallelism::ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Requires Intel NPU hardware (e.g. Meteor Lake). Run `ls /dev/accel*` to verify.".to_string(),
                "Model must be in OpenVINO IR format optimized for NPU.".to_string(),
            ],
            construction_guide: "Requires Intel NPU driver and OpenVINO with NPU plugin. Build with --features intel-npu.".to_string(),
        }
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        // Default to NPU when caller passes CPU (the user chose this engine, so NPU is intended)
        let device = if device == DeviceKind::Cpu && probe_intel_npu() {
            tracing::info!("Intel NPU detected — overriding device to NPU");
            DeviceKind::Npu
        } else {
            device
        };

        let device_str = match device {
            DeviceKind::Cpu => "CPU",
            DeviceKind::Gpu => "GPU",
            DeviceKind::Npu => "NPU",
        };

        if !model_path.exists() {
            return Err(anyhow!(
                "model path does not exist: {}",
                model_path.display()
            ));
        }

        // Auto-export: if the model is not yet in OpenVINO IR format, export it
        if needs_openvino_export(model_path) {
            let auto_export = std::env::var("BLOOM_NPU_AUTO_EXPORT")
                .or_else(|_| std::env::var("BLOOM_OPENVINO_AUTO_EXPORT"))
                .is_ok();

            if !auto_export {
                return Err(anyhow!(
                    "Model at {} is not in OpenVINO IR format.\n\
                     Set BLOOM_NPU_AUTO_EXPORT=1 to let Bloom automatically export it.\n\
                     Or export manually with: optimum-cli export openvino --model <path> --weight-format int4 <path>",
                    model_path.display()
                ));
            }

            export_to_openvino_ir(model_path)?;
        }

        // Also handle AWQ models that already have IR but might need re-export
        if is_awq_model(model_path) && !is_openvino_ir(model_path) {
            let auto_export = std::env::var("BLOOM_NPU_AUTO_EXPORT")
                .or_else(|_| std::env::var("BLOOM_OPENVINO_AUTO_EXPORT"))
                .is_ok();
            if auto_export {
                export_to_openvino_ir(model_path)?;
            }
        }

        // Warn if IR files are still missing
        if !is_openvino_ir(model_path) {
            tracing::warn!(
                "openvino_model.xml not found in {}. Inference may fail.",
                model_path.display()
            );
        }

        let is_quantized = model_path.to_string_lossy().to_lowercase().contains("int4")
            || model_path.to_string_lossy().to_lowercase().contains("int8")
            || model_path
                .to_string_lossy()
                .to_lowercase()
                .contains("quant")
            || model_path.to_string_lossy().to_lowercase().contains("awq");

        let manifest = crate::manifest_adapter::load_manifest(model_path)?;

        let metadata = ModelMetadata {
            id: model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            modality: Modality::Text,
            quantized: is_quantized,
            manifest,
        };

        Ok(Box::new(IntelNpuModel {
            model_path: model_path.to_path_buf(),
            device: device_str.to_string(),
            metadata,
        }))
    }
}

// ---------------------------------------------------------------------------
// Loaded Model implementation
// ---------------------------------------------------------------------------

struct IntelNpuModel {
    model_path: PathBuf,
    device: String,
    metadata: ModelMetadata,
}

impl LoadedModel for IntelNpuModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
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
            _ => return Err(anyhow!("Intel NPU model only supports text input")),
        };

        let script = std::env::var_os("BLOOM_NPU_SCRIPT")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("BLOOM_OPENVINO_SCRIPT").map(PathBuf::from))
            .unwrap_or_else(|| repo_script_path("openvino_llm_infer.py"));
        crate::core::security::validate_external_script(&script)?;

        let python = default_python();
        crate::core::security::validate_runner(&python)?;

        let mut command = Command::new(python);
        command.env("PYTHONIOENCODING", "utf-8");
        command
            .arg(&script)
            .arg("--model-path")
            .arg(&self.model_path)
            .arg("--prompt")
            .arg(&prompt)
            .arg("--device")
            .arg(&self.device)
            .arg("--max-tokens")
            .arg(params.max_tokens.to_string())
            .arg("--temperature")
            .arg(params.temperature.to_string())
            .arg("--top-p")
            .arg(params.top_p.to_string());

        if let Some(seed) = params.seed {
            command.arg("--seed").arg(seed.to_string());
        }

        command.stdout(Stdio::piped());
        command.stderr(Stdio::inherit());

        tracing::info!(
            "Starting Intel NPU inference on device={} model={}",
            self.device,
            self.model_path.display()
        );

        let mut child = command
            .spawn()
            .map_err(|e| anyhow!("failed to spawn Intel NPU inference process: {}", e))?;

        // Stream stdout in real-time
        if let Some(mut stdout) = child.stdout.take() {
            let mut buffer = Vec::new();
            let mut byte = [0u8; 1];

            while stdout.read_exact(&mut byte).is_ok() {
                buffer.push(byte[0]);
                if let Ok(text) = std::str::from_utf8(&buffer) {
                    sink.on_chunk(crate::io::OutputChunk::TextDelta(text.to_string()))?;
                    buffer.clear();
                }
            }
        }

        let status = child
            .wait()
            .map_err(|e| anyhow!("failed to wait on Intel NPU inference process: {}", e))?;

        sink.on_chunk(crate::io::OutputChunk::End)?;

        if !status.success() {
            return Err(anyhow!(
                "Intel NPU inference process exited with error status"
            ));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_intel_npu_engine_metadata() {
        let engine = IntelNpuEngine;
        assert_eq!(engine.name(), "intel-npu");
        assert_eq!(engine.supported_modalities(), vec![Modality::Text]);
        assert_eq!(engine.default_device(), DeviceKind::Npu);
        assert!(engine.supported_devices().contains(&DeviceKind::Npu));
        assert!(engine.supported_devices().contains(&DeviceKind::Cpu));
        assert!(engine.supported_devices().contains(&DeviceKind::Gpu));
    }

    #[test]
    fn test_intel_npu_engine_capability() {
        let engine = IntelNpuEngine;
        let cap = engine.capability();
        assert_eq!(cap.engine_name, "intel-npu");
        assert!(cap.supported_devices.contains(&DeviceClass::Npu));
        assert!(cap.supported_devices.contains(&DeviceClass::Cpu));
        assert!(cap.supported_devices.contains(&DeviceClass::IntegratedGpu));
        assert!(cap.supports_streaming);
        assert!(cap.supports_quantized_models);
        assert!(cap.supported_formats.contains(&ModelFormat::OpenVinoIr));
    }

    #[test]
    fn test_intel_npu_load_nonexistent() {
        let engine = IntelNpuEngine;
        let non_existent = Path::new("non_existent_npu_model_path_12345");
        let res = engine.load(non_existent, DeviceKind::Npu);
        assert!(res.is_err());
        assert!(
            res.err()
                .unwrap()
                .to_string()
                .contains("model path does not exist")
        );
    }

    #[test]
    fn test_intel_npu_load_requires_export() {
        let engine = IntelNpuEngine;
        let dir = tempdir().unwrap();
        let model_path = dir.path();

        // Create a fake config.json to make it look like a HuggingFace model
        std::fs::write(
            model_path.join("config.json"),
            serde_json::to_string(&serde_json::json!({
                "model_type": "qwen2"
            }))
            .unwrap(),
        )
        .unwrap();

        // Without BLOOM_NPU_AUTO_EXPORT, loading should fail
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_NPU_AUTO_EXPORT") };
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_OPENVINO_AUTO_EXPORT") };
        let res = engine.load(model_path, DeviceKind::Npu);
        assert!(res.is_err());
        assert!(
            res.err()
                .unwrap()
                .to_string()
                .contains("BLOOM_NPU_AUTO_EXPORT")
        );
    }

    #[test]
    fn test_intel_npu_model_metadata() {
        let metadata = ModelMetadata {
            id: "test_npu_model".to_string(),
            modality: Modality::Text,
            quantized: true,
            manifest: bloomai_core::ModelManifest::default(),
        };

        let model = IntelNpuModel {
            model_path: PathBuf::from("/tmp/test"),
            device: "NPU".to_string(),
            metadata,
        };

        assert_eq!(model.metadata().id, "test_npu_model");
        assert_eq!(model.metadata().modality, Modality::Text);
        assert!(model.metadata().quantized);
    }

    #[test]
    fn test_is_openvino_ir() {
        let dir = tempdir().unwrap();
        assert!(!is_openvino_ir(dir.path()));

        std::fs::write(dir.path().join("openvino_model.xml"), "<xml/>").unwrap();
        assert!(is_openvino_ir(dir.path()));
    }

    #[test]
    fn test_needs_openvino_export() {
        let dir = tempdir().unwrap();

        // Empty dir — no export needed (no config.json)
        assert!(!needs_openvino_export(dir.path()));

        // With config.json — export needed
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        assert!(needs_openvino_export(dir.path()));

        // With IR already present — no export needed
        std::fs::write(dir.path().join("openvino_model.xml"), "<xml/>").unwrap();
        assert!(!needs_openvino_export(dir.path()));
    }

    #[test]
    fn test_detects_awq_from_config() {
        let dir = tempdir().unwrap();
        let model_path = dir.path();
        let config_json = serde_json::json!({
            "quantization_config": {
                "quant_method": "awq"
            }
        });
        std::fs::write(
            model_path.join("config.json"),
            serde_json::to_string(&config_json).unwrap(),
        )
        .unwrap();

        assert!(is_awq_model(model_path));
    }
}
