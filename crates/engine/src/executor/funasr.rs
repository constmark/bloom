use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use bloomai_core::{
    DeviceCapability, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily,
    ModelManifest,
};

use crate::{
    engine::{Engine, EngineCapability, SupportLevel, default_engine_supports},
    io::{ModelInput, ModelOutput},
    model::{LoadedModel, ModelMetadata},
};

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
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    resolve_python(std::env::var_os("BLOOM_ASR_PYTHON"), &repo_root)
}

fn resolve_python(explicit: Option<std::ffi::OsString>, repo_root: &Path) -> PathBuf {
    if let Some(path) = explicit {
        return PathBuf::from(path);
    }

    let mut candidates = Vec::with_capacity(7);
    // Keep the Qwen-specific environment first, but use the native virtualenv
    // layout on each platform. Windows virtualenvs expose Scripts/python.exe;
    // Unix virtualenvs expose bin/python.
    #[cfg(target_os = "windows")]
    {
        candidates.push(
            repo_root
                .join(".venv-qwen-asr")
                .join("Scripts")
                .join("python.exe"),
        );
        candidates.push(repo_root.join(".venv").join("Scripts").join("python.exe"));
    }
    #[cfg(not(target_os = "windows"))]
    {
        candidates.push(repo_root.join(".venv-qwen-asr").join("bin").join("python"));
        candidates.push(repo_root.join(".venv").join("bin").join("python"));
    }

    #[cfg(not(target_os = "windows"))]
    candidates.extend([
        PathBuf::from("/opt/homebrew/bin/python3.12"),
        PathBuf::from("python3.12"),
    ]);

    for candidate in candidates {
        if candidate.is_absolute() {
            if candidate.is_file() {
                return candidate;
            }
        } else if command_on_path(&candidate) {
            return candidate;
        }
    }

    platform_python_launchers()
        .iter()
        .find(|launcher| command_on_path(Path::new(launcher)))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(platform_python_launchers()[0]))
}

fn platform_python_launchers() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["python", "py"]
    }
    #[cfg(not(target_os = "windows"))]
    {
        &["python3", "python"]
    }
}

fn command_on_path(command: &Path) -> bool {
    if command.is_absolute() {
        return command.is_file();
    }
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|directory| {
        let candidate = directory.join(command);
        if candidate.is_file() {
            return true;
        }
        #[cfg(target_os = "windows")]
        {
            [".exe", ".cmd", ".bat"]
                .iter()
                .map(|extension| directory.join(format!("{}{}", command.display(), extension)))
                .any(|candidate| candidate.is_file())
        }
        #[cfg(not(target_os = "windows"))]
        {
            false
        }
    })
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
        let python = default_python();
        let mut command = Command::new(&python);
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

        // Only the unit-test build may construct a simulated ASR runtime.
        let skip_daemon = cfg!(test);

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

        crate::core::security::validate_runner(&python)?;
        crate::core::security::validate_external_script(&script)?;
        command.arg(script).args(args);
        command.stdin(std::process::Stdio::piped());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());

        let (child, stdin, stdout) =
            spawn_asr_daemon(&mut command, std::time::Duration::from_secs(120))?;

        Ok(Box::new(FunASRModel {
            _model_path: model_path.to_path_buf(),
            _device: device_str.to_string(),
            _runtime: runtime,
            metadata,
            processors,
            child: Some(Arc::new(Mutex::new(child))),
            stdin: Some(Arc::new(Mutex::new(stdin))),
            stdout: Some(Arc::new(Mutex::new(stdout))),
        }))
    }
}

type AsrDaemon = (
    crate::core::process::ChildGuard,
    std::process::ChildStdin,
    BufReader<std::process::ChildStdout>,
);

fn spawn_asr_daemon(command: &mut Command, timeout: std::time::Duration) -> Result<AsrDaemon> {
    let mut child = crate::core::process::ChildGuard::spawn(command)?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("no ASR stderr"))?;
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut stderr = BufReader::new(stderr);
        loop {
            let mut line = String::new();
            match Read::take(&mut stderr, 4097).read_line(&mut line) {
                Ok(0) | Err(_) => return,
                Ok(_) if line.trim() == "READY" => {
                    let _ = sender.send(());
                    // Drain diagnostics even after readiness so the daemon can
                    // never block on its stderr pipe while serving a request.
                    let _ = std::io::copy(&mut stderr, &mut std::io::sink());
                    return;
                }
                Ok(_) if line.len() > 4096 => return,
                Ok(_) => {}
            }
        }
    });
    receiver
        .recv_timeout(timeout)
        .map_err(|error| anyhow!("ASR daemon failed to become READY: {error}"))?;
    let stdin = child.stdin.take().ok_or_else(|| anyhow!("no ASR stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("no ASR stdout"))?;
    Ok((child, stdin, BufReader::new(stdout)))
}

struct FunASRModel {
    _model_path: std::path::PathBuf,
    _device: String,
    _runtime: AsrRuntime,
    metadata: ModelMetadata,
    processors: crate::processor::ProcessorRegistry,
    child: Option<Arc<Mutex<crate::core::process::ChildGuard>>>,
    stdin: Option<Arc<Mutex<std::process::ChildStdin>>>,
    stdout: Option<Arc<Mutex<BufReader<std::process::ChildStdout>>>>,
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
                let directory = tempfile::Builder::new().prefix("bloom-asr-").tempdir()?;
                let path = directory.path().join("input.wav");
                crate::processor::write_wav_file(&path, &samples, sample_rate)?;
                let path_str = path.to_string_lossy().to_string();
                _temp_guard = Some(directory);
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
            anyhow::ensure!(cfg!(test), "ASR daemon is unavailable");
            sink.on_chunk(crate::io::OutputChunk::TextDelta("mocked text".to_string()))?;
            sink.on_chunk(crate::io::OutputChunk::End)?;
            return Ok(());
        };

        // Hold stdin until the corresponding response has been read. Separate
        // write/read locks permit concurrent callers to consume each other's text.
        let mut stdin = stdin_mutex.lock().unwrap_or_else(|e| e.into_inner());
        let mut stdout = stdout_mutex.lock().unwrap_or_else(|e| e.into_inner());
        let response = (|| -> Result<String> {
            writeln!(stdin, "{}", request_json)?;
            stdin.flush()?;
            let mut line = String::new();
            Read::take(&mut *stdout, 1024 * 1024 + 1).read_line(&mut line)?;
            anyhow::ensure!(
                line.len() <= 1024 * 1024 && line.ends_with('\n'),
                "ASR response is oversized or incomplete"
            );
            Ok(line)
        })();
        if response.is_err()
            && let Some(child) = &self.child
        {
            let mut child = child.lock().unwrap_or_else(|error| error.into_inner());
            let _ = child.kill();
            let _ = child.wait();
        }
        let line = response?;
        drop(stdout);
        drop(stdin);

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
    fn python_override_wins_over_repository_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let explicit = dir.path().join("custom-python");
        assert_eq!(
            resolve_python(Some(explicit.clone().into_os_string()), dir.path()),
            explicit
        );
    }

    #[test]
    fn repository_virtualenv_uses_platform_layout() {
        let dir = tempfile::tempdir().unwrap();
        #[cfg(target_os = "windows")]
        let python = dir.path().join(".venv/Scripts/python.exe");
        #[cfg(not(target_os = "windows"))]
        let python = dir.path().join(".venv/bin/python");
        fs::create_dir_all(python.parent().unwrap()).unwrap();
        fs::write(&python, b"").unwrap();
        assert_eq!(resolve_python(None, dir.path()), python);
    }

    #[test]
    fn python_fallback_order_matches_documentation() {
        #[cfg(target_os = "windows")]
        assert_eq!(platform_python_launchers(), &["python", "py"]);
        #[cfg(not(target_os = "windows"))]
        assert_eq!(platform_python_launchers(), &["python3", "python"]);
    }

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
        let mut header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut safetensors = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        safetensors.extend(header);
        safetensors.extend([0_u8; 4]);
        fs::write(dir.join("model.safetensors"), safetensors).unwrap();

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
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("model path does not exist")
        );

        // Unsupported layout
        let dir_holder = tempfile::tempdir().unwrap();
        let dir = dir_holder.path();
        let result2 = engine.load(dir, DeviceKind::Cpu);
        assert!(result2.is_err());
        assert!(
            result2
                .err()
                .unwrap()
                .to_string()
                .contains("unsupported ASR model layout")
        );
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

    #[cfg(unix)]
    #[test]
    fn daemon_startup_requires_exact_ready_and_has_a_deadline() {
        for script in ["printf NOT_READY >&2", "exec sleep 30"] {
            let mut command = Command::new("sh");
            command
                .args(["-c", script])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let start = std::time::Instant::now();
            assert!(spawn_asr_daemon(&mut command, std::time::Duration::from_millis(100)).is_err());
            assert!(start.elapsed() < std::time::Duration::from_secs(5));
        }
    }
}
