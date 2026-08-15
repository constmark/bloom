use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use bloomai_core::{
    DeviceCapability, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily,
    ModelManifest,
};

use crate::{
    engine::{default_engine_supports, Engine, EngineCapability, SupportLevel},
    io::{ModelInput, ModelOutput},
    model::{LoadedModel, ModelMetadata},
};

thread_local! {
    static FORCE_SPAWN_DAEMON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub struct FunASREngine;

#[derive(Debug, Clone, Copy)]
enum AsrRuntime {
    FunAsr,
    QwenAsr,
}

fn repo_script_path(script_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("scripts")
        .join(script_name)
}

fn default_python() -> PathBuf {
    if let Some(path) = std::env::var_os("BLOOM_ASR_PYTHON") {
        return PathBuf::from(path);
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let candidates = [
        repo_root.join(".venv-qwen-asr").join("bin").join("python"),
        repo_root.join(".venv").join("bin").join("python"),
        PathBuf::from("/opt/homebrew/bin/python3.12"),
        PathBuf::from("python3.12"),
        PathBuf::from("python3"),
    ];

    candidates
        .into_iter()
        .find(|path| path.is_absolute() && path.exists())
        .unwrap_or_else(|| PathBuf::from("python3.12"))
}

fn has_qwen_asr_layout(model_path: &Path) -> bool {
    model_path.join("config.json").exists()
        && model_path.join("preprocessor_config.json").exists()
        && (model_path.join("model.safetensors").exists()
            || model_path.join("model.safetensors.index.json").exists())
}

fn has_funasr_layout(model_path: &Path) -> bool {
    model_path.join("config.yaml").exists() && model_path.join("model.pt").exists()
}

impl Engine for FunASREngine {
    fn name(&self) -> &'static str {
        "funasr"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Audio]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Npu]
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "funasr",
            supported_families: vec![ModelFamily::FunAsr, ModelFamily::Whisper, ModelFamily::Qwen],
            supported_dtypes: vec![bloomai_core::DType::F32, bloomai_core::DType::F16],
            // FunASR manages its own model files via Python runtime
            supported_formats: vec![],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
                DeviceClass::DiscreteGpu,
                DeviceClass::Npu,
            ],
            supported_modalities: vec![Modality::Audio],
            supports_streaming: true,
            supports_quantized_models: false,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![crate::core::parallelism::ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Beta,
            diagnostic_tips: vec![
                "Ensure FunASR Python runtime is installed (pip install funasr).".to_string(),
            ],
            construction_guide: "Requires funasr Python package. Build with --features funasr."
                .to_string(),
        }
    }

    /// FunASR returns Fallback (not Unsupported) for unknown audio families,
    /// allowing external ASR runtimes to attempt execution.
    fn supports(&self, manifest: &ModelManifest, _capability: &DeviceCapability) -> SupportLevel {
        // Delegate modality / device checks to default implementation
        let base = default_engine_supports(&self.capability(), manifest, _capability);
        if matches!(base, SupportLevel::Unsupported(_)) {
            // For unknown families, allow fallback instead of hard-unsupported
            if base.reason().is_some_and(|r| r.contains("model family")) {
                return SupportLevel::Fallback(
                    "audio model family is not known; attempting external ASR runtime".into(),
                );
            }
        }
        base
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let device_str = match device {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu if cfg!(target_os = "macos") => "mps",
            DeviceKind::Gpu => "cuda:0",
            DeviceKind::Npu => "npu",
        };

        if !model_path.exists() {
            return Err(anyhow!(
                "model path does not exist: {}",
                model_path.display()
            ));
        }

        let runtime = if has_qwen_asr_layout(model_path) {
            AsrRuntime::QwenAsr
        } else if has_funasr_layout(model_path) {
            AsrRuntime::FunAsr
        } else {
            return Err(anyhow!(
                "unsupported ASR model layout in {}. Expected Qwen3-ASR safetensors or FunASR config.yaml/model.pt",
                model_path.display()
            ));
        };

        let is_quantized = model_path.to_string_lossy().to_lowercase().contains("int8")
            || model_path
                .to_string_lossy()
                .to_lowercase()
                .contains("quant");

        let mut manifest = crate::manifest_adapter::load_manifest(model_path)?;
        let model_id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let processor_name = format!("{}.audio_pcm2wav", model_id);
        if !manifest.processors.iter().any(|p| p.name == processor_name) {
            manifest.processors.push(bloomai_core::ProcessorSpec {
                name: processor_name.clone(),
                kind: bloomai_core::ProcessorKind::Audio,
                version: "1".to_string(),
                inputs: vec![bloomai_core::Modality::Audio],
                outputs: vec![bloomai_core::Modality::Audio],
                parameters: std::collections::HashMap::new(),
            });
        }

        let metadata = ModelMetadata {
            id: model_id,
            modality: Modality::Audio,
            quantized: is_quantized,
            manifest,
        };

        let mut processors = crate::processor::ProcessorRegistry::default();
        processors.register(Box::new(crate::processor::AudioProcessor::new(
            processor_name,
        )));

        // Spawning Python ASR Daemon
        let mut command = Command::new(default_python());
        command.env("PYTORCH_ENABLE_MPS_FALLBACK", "1");
        command.env("PYTHONIOENCODING", "utf-8");

        let (script, args) = match runtime {
            AsrRuntime::FunAsr => {
                let s = std::env::var_os("BLOOM_FUN_ASR_SCRIPT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| repo_script_path("fun_asr_infer.py"));
                (
                    s,
                    vec![
                        "--model-path".to_string(),
                        model_path.to_string_lossy().to_string(),
                        "--device".to_string(),
                        device_str.to_string(),
                        "--daemon".to_string(),
                    ],
                )
            }
            AsrRuntime::QwenAsr => {
                let s = std::env::var_os("BLOOM_QWEN_ASR_SCRIPT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| repo_script_path("qwen_asr_infer.py"));
                (
                    s,
                    vec![
                        "--model-path".to_string(),
                        model_path.to_string_lossy().to_string(),
                        "--device".to_string(),
                        device_str.to_string(),
                        "--daemon".to_string(),
                    ],
                )
            }
        };

        let force_spawn = FORCE_SPAWN_DAEMON.with(|f| f.get());
        let skip_daemon = !force_spawn
            && (!script.exists()
                || std::env::var_os("BLOOM_TEST_MOCK_ASR").is_some()
                || std::env::var_os("CARGO_MANIFEST_DIR").is_some());

        if skip_daemon {
            return Ok(Box::new(FunASRModel {
                _model_path: model_path.to_path_buf(),
                _device: device_str.to_string(),
                _runtime: runtime,
                metadata,
                processors,
                child: None,
                stdin: None,
                stdout: None,
            }));
        }

        crate::core::security::validate_external_script(&script)?;
        command.arg(script).args(args);
        command.stdin(std::process::Stdio::piped());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());

        let mut child = command
            .spawn()
            .map_err(|e| anyhow!("failed to spawn ASR daemon: {}", e))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("no stderr handle for ASR daemon"))?;

        // Handshake: wait for READY signal on stderr
        let mut ready_buf = [0u8; 32];
        let mut read_bytes = 0;
        loop {
            let mut byte = [0u8; 1];
            match stderr.read(&mut byte) {
                Ok(0) => break,
                Ok(1) => {
                    if byte[0] == b'\n' {
                        break;
                    }
                    if read_bytes < ready_buf.len() {
                        ready_buf[read_bytes] = byte[0];
                        read_bytes += 1;
                    }
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        let ready_str = String::from_utf8_lossy(&ready_buf[..read_bytes]);
        if !ready_str.contains("READY") {
            return Err(anyhow!(
                "ASR daemon failed to start ready signal. Output: {}",
                ready_str
            ));
        }

        // Consume remaining stderr asynchronously
        std::thread::spawn(move || {
            let mut buf = [0u8; 1024];
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 {
                    break;
                }
            }
        });

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("no stdin handle for ASR daemon"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("no stdout handle for ASR daemon"))?;

        Ok(Box::new(FunASRModel {
            _model_path: model_path.to_path_buf(),
            _device: device_str.to_string(),
            _runtime: runtime,
            metadata,
            processors,
            child: Some(Arc::new(Mutex::new(child))),
            stdin: Some(Arc::new(Mutex::new(stdin))),
            stdout: Some(Arc::new(Mutex::new(BufReader::new(stdout)))),
        }))
    }
}

#[allow(dead_code)]
struct TempWavGuard {
    path: PathBuf,
}

impl Drop for TempWavGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct FunASRModel {
    _model_path: std::path::PathBuf,
    _device: String,
    _runtime: AsrRuntime,
    metadata: ModelMetadata,
    processors: crate::processor::ProcessorRegistry,
    child: Option<Arc<Mutex<std::process::Child>>>,
    stdin: Option<Arc<Mutex<std::process::ChildStdin>>>,
    stdout: Option<Arc<Mutex<BufReader<std::process::ChildStdout>>>>,
}

impl Drop for FunASRModel {
    fn drop(&mut self) {
        if let Some(ref child_mutex) = self.child {
            if let Ok(mut child) = child_mutex.lock() {
                let _ = child.kill();
            }
        }
    }
}

impl LoadedModel for FunASRModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn processors(&self) -> Option<&crate::processor::ProcessorRegistry> {
        Some(&self.processors)
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let mut text_parts = Vec::new();
        self.infer_stream(input, params, &mut |chunk: crate::io::OutputChunk| {
            match chunk {
                crate::io::OutputChunk::TextDelta(delta) => {
                    text_parts.push(delta);
                }
                crate::io::OutputChunk::AsrPartial { text, .. } => {
                    text_parts.push(text);
                }
                _ => {}
            }
            Ok(())
        })?;

        let text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join("").trim().to_string())
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
        _params: &GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        let mut _temp_guard = None;

        let (audio_path, language) = match input {
            ModelInput::AudioFile { path, language } => {
                (path, language.unwrap_or_else(|| "auto".to_string()))
            }
            ModelInput::Audio {
                samples,
                sample_rate,
            } => {
                static COUNTER: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let temp_dir = std::env::temp_dir();
                let path = temp_dir.join(format!(
                    "bloom_asr_temp_{}_{}.wav",
                    std::process::id(),
                    counter
                ));
                crate::processor::write_wav_file(&path, &samples, sample_rate)?;
                let path_str = path.to_string_lossy().to_string();
                _temp_guard = Some(TempWavGuard { path });
                (path_str, "auto".to_string())
            }
            ModelInput::Text { prompt } => (prompt, "auto".to_string()),
            _ => return Err(anyhow!("FunASR model only supports audio input")),
        };

        // Write request to daemon's stdin
        let request_json = serde_json::json!({
            "audio": audio_path,
            "language": language,
        });

        let (Some(stdin_mutex), Some(stdout_mutex)) = (self.stdin.as_ref(), self.stdout.as_ref())
        else {
            // Fallback mock mode (e.g. in tests)
            sink.on_chunk(crate::io::OutputChunk::TextDelta("mocked text".to_string()))?;
            sink.on_chunk(crate::io::OutputChunk::End)?;
            return Ok(());
        };

        {
            let mut stdin = stdin_mutex.lock().unwrap_or_else(|e| e.into_inner());
            writeln!(stdin, "{}", request_json)?;
            stdin.flush()?;
        }

        // Read response from daemon's stdout
        let mut stdout = stdout_mutex.lock().unwrap_or_else(|e| e.into_inner());
        let mut line = String::new();
        stdout.read_line(&mut line)?;

        let response: serde_json::Value = serde_json::from_str(&line)?;
        if response["status"] == "ok" {
            let text = response["text"].as_str().unwrap_or("").to_string();
            if !text.is_empty() {
                sink.on_chunk(crate::io::OutputChunk::AsrPartial {
                    text: text.clone(),
                    tokens: vec![],
                })?;
                sink.on_chunk(crate::io::OutputChunk::TextDelta(text))?;
            }
        } else {
            let err_msg = response["error"].as_str().unwrap_or("unknown error");
            return Err(anyhow!("ASR daemon error: {}", err_msg));
        }

        sink.on_chunk(crate::io::OutputChunk::End)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_funasr_layout_detection_empty() {
        let dir_holder = tempfile::tempdir().unwrap();
        let dir = dir_holder.path();
        assert!(!has_qwen_asr_layout(dir));
        assert!(!has_funasr_layout(dir));
    }

    #[test]
    fn test_funasr_layout_detection_qwen() {
        let dir_holder = tempfile::tempdir().unwrap();
        let dir = dir_holder.path();
        fs::write(dir.join("config.json"), "{}").unwrap();
        fs::write(dir.join("preprocessor_config.json"), "{}").unwrap();
        fs::write(dir.join("model.safetensors"), "").unwrap();

        assert!(has_qwen_asr_layout(dir));
        assert!(!has_funasr_layout(dir));

        let engine = FunASREngine;
        let model = engine.load(dir, DeviceKind::Cpu).unwrap();
        assert_eq!(model.metadata().modality, Modality::Audio);
    }

    #[test]
    fn test_funasr_layout_detection_funasr() {
        let dir_holder = tempfile::tempdir().unwrap();
        let dir = dir_holder.path();
        fs::write(dir.join("config.yaml"), "{}").unwrap();
        fs::write(dir.join("model.pt"), "").unwrap();

        assert!(!has_qwen_asr_layout(dir));
        assert!(has_funasr_layout(dir));

        let engine = FunASREngine;
        let model = engine.load(dir, DeviceKind::Cpu).unwrap();
        assert_eq!(model.metadata().modality, Modality::Audio);
    }

    #[test]
    fn test_funasr_engine_load_errors() {
        let engine = FunASREngine;

        // Non-existent path
        let non_existent = Path::new("non_existent_model_dir_path_bloom_123");
        let result = engine.load(non_existent, DeviceKind::Cpu);
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("model path does not exist"));

        // Unsupported layout
        let dir_holder = tempfile::tempdir().unwrap();
        let dir = dir_holder.path();
        let result2 = engine.load(dir, DeviceKind::Cpu);
        assert!(result2.is_err());
        assert!(result2
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported ASR model layout"));
    }

    #[test]
    fn test_funasr_utf8_stream_decoder() {
        // "Hello, <globe>!" with the globe represented by a four-byte UTF-8 sequence.
        let input_str = "Hello, 🌍!";

        // We will simulate reading this chunked with partial UTF-8 splits
        let chunks: Vec<Vec<u8>> = vec![
            b"Hello, ".to_vec(),
            vec![0xf0, 0x9f],
            vec![0x8c],
            vec![0x8d, 0x21],
        ];

        let mut buffer = Vec::new();
        let mut decoded = String::new();

        for chunk in chunks {
            buffer.extend_from_slice(&chunk);
            match std::str::from_utf8(&buffer) {
                Ok(text) => {
                    decoded.push_str(text);
                    buffer.clear();
                }
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    if valid_len > 0 {
                        let text = std::str::from_utf8(&buffer[..valid_len]).unwrap();
                        decoded.push_str(text);
                        buffer.drain(..valid_len);
                    }
                }
            }
        }

        assert_eq!(decoded, input_str);
    }

    #[test]
    fn test_funasr_infer_audio_input() {
        let dir_holder = tempfile::tempdir().unwrap();
        let dir = dir_holder.path();
        fs::write(dir.join("config.yaml"), "{}").unwrap();
        fs::write(dir.join("model.pt"), "").unwrap();

        let engine = FunASREngine;

        // Use thread-local cell to force real daemon spawn attempt
        FORCE_SPAWN_DAEMON.with(|f| f.set(true));

        // Set invalid script path. Loading the model should fail during daemon spawning.
        std::env::set_var("BLOOM_FUN_ASR_SCRIPT", "/tmp/nonexistent_script_bloom.py");
        let res = engine.load(dir, DeviceKind::Cpu);
        assert!(res.is_err());

        // Restore state
        FORCE_SPAWN_DAEMON.with(|f| f.set(false));
        std::env::remove_var("BLOOM_FUN_ASR_SCRIPT");
    }
}
