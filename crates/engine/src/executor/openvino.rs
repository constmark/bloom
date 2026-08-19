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

pub struct OpenVINOEngine;

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

fn maybe_export_awq(model_path: &Path) -> Result<()> {
    if model_path.join("openvino_model.xml").exists() || !is_awq_model(model_path) {
        return Ok(());
    }

    if std::env::var_os("BLOOM_OPENVINO_AUTO_EXPORT").is_none() {
        return Err(anyhow!(
            "AWQ model detected, but OpenVINO IR files were not found. \
             Export it first or set BLOOM_OPENVINO_AUTO_EXPORT=1 to let Bloom run optimum-intel export."
        ));
    }

    let python = default_python();
    crate::core::security::validate_runner(&python)?;
    let output = Command::new(&python)
        .arg("-m")
        .arg("optimum.intel.openvino.cli")
        .arg("export")
        .arg("--model")
        .arg(model_path)
        .arg("--weight-format")
        .arg("int4")
        .arg(model_path)
        .output()
        .map_err(|e| {
            anyhow!(
                "failed to execute Python for AWQ export using {}: {}",
                python.display(),
                e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named optimum") || stderr.contains("ModuleNotFoundError") {
            return Err(anyhow!(
                "failed to export AWQ model: missing Python libraries. \
                 Install them with: pip install \"optimum-intel[openvino]\" openvino nncf"
            ));
        }
        return Err(anyhow!("optimum-intel OpenVINO export failed: {}", stderr));
    }

    Ok(())
}

impl Engine for OpenVINOEngine {
    fn name(&self) -> &'static str {
        "openvino"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Npu]
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "openvino",
            // OpenVINO IR is model-agnostic; wildcard "*" matches any family
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
            supported_formats: vec![ModelFormat::OpenVinoIr],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
                DeviceClass::Npu,
            ],
            supported_modalities: vec![Modality::Text],
            supports_streaming: true,
            supports_quantized_models: true, // INT4/INT8 IR
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
            maturity: crate::engine::BackendMaturity::Beta,
            diagnostic_tips: vec![
                "Ensure OpenVINO runtime is installed and LD_LIBRARY_PATH includes OpenVINO libs."
                    .to_string(),
                "Model must be in OpenVINO IR format (.xml + .bin).".to_string(),
            ],
            construction_guide: "Requires OpenVINO runtime. Build with --features openvino."
                .to_string(),
        }
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
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

        maybe_export_awq(model_path)?;

        // Verify it contains either configuration or model representation
        let is_ir = model_path.join("openvino_model.xml").exists();
        if !is_ir {
            tracing::warn!(
                "openvino_model.xml not found in {}. Model might need to be exported.",
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

        Ok(Box::new(OpenVINOModel {
            model_path: model_path.to_path_buf(),
            device: device_str.to_string(),
            metadata,
        }))
    }
}

struct OpenVINOModel {
    model_path: PathBuf,
    device: String,
    metadata: ModelMetadata,
}

impl LoadedModel for OpenVINOModel {
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
            _ => return Err(anyhow!("OpenVINO model only supports text input")),
        };

        let script = std::env::var_os("BLOOM_OPENVINO_SCRIPT")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo_script_path("openvino_llm_infer.py"));
        crate::core::security::validate_external_script(&script)?;

        let python = default_python();
        crate::core::security::validate_runner(&python)?;

        let mut command = Command::new(python);
        command.env("PYTHONIOENCODING", "utf-8");
        command
            .arg(script)
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

        let mut child = command
            .spawn()
            .map_err(|e| anyhow!("failed to spawn OpenVINO inference process: {}", e))?;

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
            .map_err(|e| anyhow!("failed to wait on OpenVINO inference process: {}", e))?;

        sink.on_chunk(crate::io::OutputChunk::End)?;

        if !status.success() {
            return Err(anyhow!(
                "OpenVINO inference process exited with error status"
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn detects_awq_from_config() {
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

        assert!(super::is_awq_model(model_path));
    }

    #[test]
    fn detects_awq_from_path_name() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("my-qwen-awq-model");
        std::fs::create_dir_all(&model_path).unwrap();
        assert!(super::is_awq_model(&model_path));
    }

    #[test]
    fn test_openvino_engine_metadata() {
        let engine = OpenVINOEngine;
        assert_eq!(engine.name(), "openvino");
        assert_eq!(engine.supported_modalities(), vec![Modality::Text]);
        assert_eq!(
            engine.supported_devices(),
            vec![DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Npu]
        );
    }

    #[test]
    fn test_openvino_engine_load_errors() {
        let engine = OpenVINOEngine;
        let non_existent = Path::new("non_existent_openvino_model_path_12345");
        let res = engine.load(non_existent, DeviceKind::Cpu);
        assert!(res.is_err());
        assert!(
            res.err()
                .unwrap()
                .to_string()
                .contains("model path does not exist")
        );
    }

    #[test]
    fn test_openvino_model_metadata() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model_int4");
        std::fs::create_dir_all(&model_path).unwrap();

        let model = OpenVINOModel {
            model_path: model_path.clone(),
            device: "CPU".to_string(),
            metadata: ModelMetadata {
                id: "test_model_int4".to_string(),
                modality: Modality::Text,
                quantized: true,
                manifest: bloomai_core::ModelManifest::default(),
            },
        };

        assert_eq!(model.metadata().id, "test_model_int4");
        assert_eq!(model.metadata().modality, Modality::Text);
        assert!(model.metadata().quantized);
    }
}
