//! ONNX Runtime adapter through a guarded Python bridge.
//!
//! Bloom keeps the native ONNX Runtime dependency optional. When an operator
//! installs `onnxruntime` (or `onnxruntime-gpu`) in the selected Python
//! environment, this adapter executes the model through the bundled bridge and
//! returns text or numeric output using a small JSON protocol.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Result, anyhow};
use bloomai_core::{
    DType, DeviceClass, DeviceKind, Modality, ModelFamily, ModelFormat, ModelManifest,
};

use crate::{
    core::parallelism::ParallelStrategy,
    engine::{Engine, EngineCapability, SupportLevel},
    io::{ModelInput, ModelOutput, OutputChunk},
    model::{LoadedModel, ModelMetadata, OutputSink},
};

pub struct OnnxRuntimeEngine;

const DEFAULT_BRIDGE_TIMEOUT: Duration = Duration::from_secs(600);
const MAX_BRIDGE_TIMEOUT_SECS: u64 = 3_600;
const MAX_BRIDGE_STDOUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_BRIDGE_STDERR_BYTES: usize = 64 * 1024;

#[derive(Debug)]
struct BridgeOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_exceeded: bool,
    stderr_exceeded: bool,
}

#[derive(Debug, serde::Deserialize)]
struct OnnxProbe {
    ok: bool,
    device: String,
    provider: String,
}

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
        if cfg!(target_os = "macos") {
            // ORT CoreML's CPUAndGPU/ALL policies may still execute portions
            // of a graph on CPU; they do not provide a GPU-only contract.
            vec![DeviceKind::Cpu]
        } else {
            vec![DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Npu]
        }
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
            maturity: crate::engine::BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Install the Python `onnxruntime` package in the interpreter selected by BLOOM_PYTHON (or use BLOOM_ONNX_PYTHON).".to_string(),
                "Use a model.onnx file or point --model directly at a .onnx file.".to_string(),
                "Set BLOOM_ONNX_EXECUTION_PROVIDER to CUDA, OpenVINO, or CPU when the default provider is not suitable.".to_string(),
            ],
            construction_guide: "Uses the bundled Python ONNX Runtime bridge. The bridge accepts text, audio, image bytes, and numeric tensors and returns JSON text/logits.".to_string(),
        }
    }

    fn supports(
        &self,
        manifest: &ModelManifest,
        device_cap: &bloomai_core::DeviceCapability,
    ) -> SupportLevel {
        crate::engine::default_engine_supports(&self.capability(), manifest, device_cap)
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let onnx_file = Self::discover_model_file(model_path)?;
        let python = std::env::var_os("BLOOM_ONNX_PYTHON")
            .or_else(|| std::env::var_os("BLOOM_PYTHON"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python"));
        crate::core::security::validate_runner(&python)?;
        let script = script_path();
        crate::core::security::validate_external_script(&script)?;
        let expected_provider = provider_name(device)?;
        let probe = run_probe(&python, &script, &onnx_file, device)?;
        if probe.device != device_label(device) || probe.provider != expected_provider {
            return Err(anyhow!(
                "ONNX Runtime probe selected device/provider {}/{} but Bloom requested {}/{}",
                probe.device,
                probe.provider,
                device_label(device),
                expected_provider
            ));
        }
        let manifest = crate::manifest_adapter::load_manifest(model_path).unwrap_or_default();
        let metadata = ModelMetadata {
            id: onnx_file
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("onnx-model")
                .to_string(),
            modality: Modality::Multi,
            quantized: false,
            manifest,
        };
        Ok(Box::new(OnnxRuntimeModel {
            model_path: onnx_file,
            python,
            device,
            provider: probe.provider,
            metadata,
        }))
    }
}

struct OnnxRuntimeModel {
    model_path: PathBuf,
    python: PathBuf,
    device: DeviceKind,
    provider: String,
    metadata: ModelMetadata,
}

impl LoadedModel for OnnxRuntimeModel {
    fn actual_device(&self) -> Option<DeviceKind> {
        Some(self.device)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(
        &self,
        input: ModelInput,
        _params: &bloomai_core::GenerationParams,
    ) -> Result<ModelOutput> {
        let request = serde_json::to_vec(&input)?;
        let mut command = Command::new(&self.python);
        command
            .env("PYTHONIOENCODING", "utf-8")
            .arg(script_path())
            .arg("--model-path")
            .arg(&self.model_path)
            .arg("--device")
            .arg(device_label(self.device))
            .arg("--provider")
            .arg(&self.provider)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let script = script_path();
        crate::core::security::validate_external_script(&script)?;
        let output = run_bridge(&mut command, &request)?;
        if output.stdout_exceeded {
            return Err(anyhow!(
                "ONNX Runtime bridge stdout exceeded the {} MiB limit",
                MAX_BRIDGE_STDOUT_BYTES / (1024 * 1024)
            ));
        }
        if output.stderr_exceeded {
            return Err(anyhow!(
                "ONNX Runtime bridge stderr exceeded the {} KiB limit",
                MAX_BRIDGE_STDERR_BYTES / 1024
            ));
        }
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "ONNX Runtime bridge failed ({}): {}",
                output.status,
                stderr.trim()
            ));
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|error| anyhow!("ONNX Runtime bridge emitted invalid UTF-8: {error}"))?;
        serde_json::from_str(stdout.trim()).map_err(|error| {
            anyhow!("ONNX Runtime bridge returned invalid JSON: {error}; output={stdout:?}")
        })
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &bloomai_core::GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let output = self.infer(input, params)?;
        if let Some(text) = output.text {
            sink.on_chunk(OutputChunk::TextDelta(text))?;
        }
        if let Some(logits) = output.logits {
            sink.on_chunk(OutputChunk::Embedding(logits))?;
        }
        sink.on_chunk(OutputChunk::End)
    }
}

fn bundled_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/onnxruntime_infer.py")
}

fn script_path() -> PathBuf {
    if let Some(path) = std::env::var_os("BLOOM_ONNX_SCRIPT") {
        return PathBuf::from(path);
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        for candidate in [
            parent.join("onnxruntime_infer.py"),
            parent.join("scripts").join("onnxruntime_infer.py"),
        ] {
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    bundled_script()
}

fn device_label(device: DeviceKind) -> &'static str {
    match device {
        DeviceKind::Cpu => "CPU",
        DeviceKind::Gpu => "CUDA",
        DeviceKind::Npu => "OpenVINO",
    }
}

fn provider_name(device: DeviceKind) -> Result<&'static str> {
    match device {
        DeviceKind::Cpu => Ok("CPUExecutionProvider"),
        DeviceKind::Gpu if cfg!(target_os = "macos") => Err(anyhow!(
            "ONNX Runtime CoreML GPU execution cannot prove a GPU-only placement on macOS; use --device cpu"
        )),
        DeviceKind::Gpu => Ok("CUDAExecutionProvider"),
        DeviceKind::Npu => Ok("OpenVINOExecutionProvider"),
    }
}

fn run_probe(python: &Path, script: &Path, model: &Path, device: DeviceKind) -> Result<OnnxProbe> {
    let mut command = Command::new(python);
    command
        .env("PYTHONIOENCODING", "utf-8")
        .arg(script)
        .arg("--probe")
        .arg("--model-path")
        .arg(model)
        .arg("--device")
        .arg(device_label(device))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_bridge(&mut command, &[])?;
    if output.stdout_exceeded || output.stderr_exceeded {
        return Err(anyhow!(
            "ONNX Runtime probe output exceeded the 64 KiB limit"
        ));
    }
    if !output.status.success() {
        return Err(anyhow!(
            "ONNX Runtime probe failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let probe: OnnxProbe = serde_json::from_slice(&output.stdout).map_err(|error| {
        anyhow!(
            "ONNX Runtime probe returned invalid JSON: {error}; output={:?}",
            String::from_utf8_lossy(&output.stdout)
        )
    })?;
    if !probe.ok {
        return Err(anyhow!("ONNX Runtime probe reported failure"));
    }
    Ok(probe)
}

fn bridge_timeout() -> Result<Duration> {
    let Some(value) = std::env::var_os("BLOOM_ONNX_TIMEOUT_SECS") else {
        return Ok(DEFAULT_BRIDGE_TIMEOUT);
    };
    let value = value
        .to_str()
        .ok_or_else(|| anyhow!("BLOOM_ONNX_TIMEOUT_SECS must be valid UTF-8"))?;
    let seconds = value.parse::<u64>().map_err(|_| {
        anyhow!(
            "BLOOM_ONNX_TIMEOUT_SECS must be an integer between 1 and {MAX_BRIDGE_TIMEOUT_SECS}"
        )
    })?;
    if !(1..=MAX_BRIDGE_TIMEOUT_SECS).contains(&seconds) {
        return Err(anyhow!(
            "BLOOM_ONNX_TIMEOUT_SECS must be between 1 and {MAX_BRIDGE_TIMEOUT_SECS}"
        ));
    }
    Ok(Duration::from_secs(seconds))
}

/// Execute a Python bridge with bounded output buffers and a hard wall-clock
/// deadline. The readers drain beyond the retained limit so a noisy bridge
/// cannot deadlock while the parent waits for it to exit.
fn run_bridge(command: &mut Command, request: &[u8]) -> Result<BridgeOutput> {
    let timeout = bridge_timeout()?;
    let output = crate::core::process::run_bounded_output(
        command,
        request,
        MAX_BRIDGE_STDOUT_BYTES,
        MAX_BRIDGE_STDERR_BYTES,
        timeout,
    )
    .map_err(|error| anyhow!("ONNX Runtime bridge process failed: {error}"))?;
    Ok(BridgeOutput {
        status: output.status,
        stdout: output.stdout,
        stderr: output.stderr,
        stdout_exceeded: output.stdout_exceeded,
        stderr_exceeded: output.stderr_exceeded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloomai_core::{
        DeviceCapability, MemoryTopology, ModelFile, ModelIoSchema, ModelMemoryProfile, PowerState,
        RuntimeHints, ThermalState, constants::GIB,
    };
    use std::time::Instant;

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
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Experimental);
        assert!(cap.supported_formats.contains(&ModelFormat::Onnx));
        assert!(cap.supported_modalities.contains(&Modality::Vision));
        assert!(cap.supports_quantized_models);
    }

    #[test]
    fn supports_onnx_manifest_for_external_runtime() {
        let level = OnnxRuntimeEngine.supports(&onnx_manifest(), &cpu_capability());
        assert!(matches!(level, SupportLevel::Native));
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
    fn load_validates_external_runtime_probe() {
        use crate::test_env::{ENV_LOCK, EnvVarGuard};
        let _lock = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"placeholder").unwrap();
        let script = dir.path().join("fake-onnx.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\nif [ \"$1\" = \"--probe\" ]; then printf '%s\\n' '{\"ok\":true,\"device\":\"CPU\",\"provider\":\"CPUExecutionProvider\"}'; else cat >/dev/null; printf '%s\\n' '{\"text\":\"fake output\",\"logits\":null}'; fi\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _script = EnvVarGuard::set("BLOOM_ONNX_SCRIPT", script.to_str().unwrap());
        let _python = EnvVarGuard::set("BLOOM_ONNX_PYTHON", "sh");

        let model = OnnxRuntimeEngine
            .load(dir.path(), DeviceKind::Cpu)
            .expect("the bridge probe should validate the runtime boundary");
        assert_eq!(model.metadata().id, "model");
    }

    #[cfg(unix)]
    #[test]
    fn external_bridge_round_trip_uses_json_protocol() {
        use crate::test_env::{ENV_LOCK, EnvVarGuard};
        let _lock = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.onnx");
        std::fs::write(&model_file, b"placeholder").unwrap();
        let script = dir.path().join("fake-onnx.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\nif [ \"$1\" = \"--probe\" ]; then printf '%s\\n' '{\"ok\":true,\"device\":\"CPU\",\"provider\":\"CPUExecutionProvider\"}'; else cat >/dev/null; printf '%s\\n' '{\"text\":\"fake output\",\"logits\":null}'; fi\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _script = EnvVarGuard::set("BLOOM_ONNX_SCRIPT", script.to_str().unwrap());
        let _python = EnvVarGuard::set("BLOOM_ONNX_PYTHON", "sh");
        let model = OnnxRuntimeEngine
            .load(&model_file, DeviceKind::Cpu)
            .unwrap();
        let output = model
            .infer(
                ModelInput::Text {
                    prompt: "hello".to_string(),
                },
                &bloomai_core::GenerationParams::default(),
            )
            .unwrap();
        assert_eq!(output.text.as_deref(), Some("fake output"));
    }

    #[test]
    fn bridge_reader_retains_only_the_configured_limit() {
        let (bytes, exceeded) = crate::core::process::read_bounded("abcdef".as_bytes(), 3).unwrap();
        assert_eq!(bytes, b"abc");
        assert!(exceeded);

        let (bytes, exceeded) = crate::core::process::read_bounded("abc".as_bytes(), 3).unwrap();
        assert_eq!(bytes, b"abc");
        assert!(!exceeded);
    }

    #[cfg(unix)]
    #[test]
    fn bridge_timeout_kills_a_hung_worker() {
        use crate::test_env::{ENV_LOCK, EnvVarGuard};
        let _lock = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let _timeout = EnvVarGuard::set("BLOOM_ONNX_TIMEOUT_SECS", "1");
        let mut command = Command::new("sh");
        command
            .args(["-c", "sleep 30"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let started = Instant::now();
        let error = run_bridge(&mut command, &[]).unwrap_err().to_string();
        assert!(error.contains("timed out after 1 seconds"));
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
