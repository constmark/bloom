//! MLX-LM adapter through a guarded Python bridge.
//!
//! MLX is optional at build time. When an operator installs the pinned
//! `mlx-lm` package in the selected Python environment, this adapter executes
//! local MLX model packages on Apple Silicon CPU or GPU without linking the
//! vendor framework into Bloom.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Result, anyhow};
use bloomai_core::{
    DType, DeviceClass, DeviceKind, Modality, ModelFamily, ModelFormat, ModelManifest,
};

use crate::{
    core::parallelism::ParallelStrategy,
    core::process::ChildGuard,
    engine::{Engine, EngineCapability, SupportLevel, default_engine_supports},
    io::{ModelInput, ModelOutput, OutputChunk},
    model::{LoadedModel, ModelMetadata, OutputSink},
};

pub struct MlxEngine;

const DEFAULT_MLX_TIMEOUT: Duration = Duration::from_secs(600);
const MAX_MLX_TIMEOUT_SECS: u64 = 3_600;
const MAX_MLX_STDOUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_MLX_STDERR_BYTES: usize = 64 * 1024;

#[derive(Debug, serde::Deserialize)]
struct MlxProbe {
    ok: bool,
    device: String,
    runtime: String,
}

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

        for name in [
            "weights.safetensors",
            "model.safetensors",
            "weights.npz",
            "weights.mlx",
        ] {
            let candidate = model_path.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }

        let mut candidates = std::fs::read_dir(model_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "mlx" | "npz" | "safetensors"
                        )
                    })
            })
            .collect::<Vec<_>>();
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
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        if cfg!(target_os = "macos") {
            vec![DeviceKind::Cpu, DeviceKind::Gpu]
        } else {
            // MLX is an Apple Silicon runtime. Keep the CPU bridge available
            // for diagnostics and test doubles on other hosts, but never let
            // device selection or preflight advertise a non-existent GPU.
            vec![DeviceKind::Cpu]
        }
    }

    fn default_device(&self) -> DeviceKind {
        if cfg!(target_os = "macos") {
            DeviceKind::Gpu
        } else {
            DeviceKind::Cpu
        }
    }

    fn capability(&self) -> EngineCapability {
        let supported_devices = self
            .supported_devices()
            .into_iter()
            .flat_map(|device| match device {
                DeviceKind::Cpu => vec![DeviceClass::Cpu],
                DeviceKind::Gpu => vec![DeviceClass::IntegratedGpu],
                DeviceKind::Npu => vec![DeviceClass::Npu],
            })
            .collect();
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![
                ModelFamily::Qwen,
                ModelFamily::Gemma,
                ModelFamily::Llama,
                ModelFamily::Custom("*".to_string()),
            ],
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16],
            supported_formats: vec![ModelFormat::Mlx],
            supported_devices,
            supported_modalities: self.supported_modalities(),
            supports_streaming: true,
            supports_quantized_models: true,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Install the pinned `mlx-lm` package in BLOOM_MLX_PYTHON or BLOOM_PYTHON."
                    .to_string(),
                "Use an MLX-LM model directory containing config.json, tokenizer files, and MLX weights.".to_string(),
                "MLX GPU execution requires Apple Silicon; CPU execution remains available for diagnostics.".to_string(),
            ],
            construction_guide:
                "Uses the bundled mlx_lm Python bridge and preserves the selected CPU/GPU device."
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

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let mlx_file = Self::discover_model_file(model_path)?;
        let package_path = if model_path.is_dir() {
            model_path.to_path_buf()
        } else {
            mlx_file
                .parent()
                .ok_or_else(|| anyhow!("MLX model file has no parent directory"))?
                .to_path_buf()
        };
        let python = python_path();
        crate::core::security::validate_runner(&python)?;
        let script = script_path();
        crate::core::security::validate_external_script(&script)?;
        let probe = run_probe(&python, &script, &package_path, device)?;
        if probe.device != device_label(device) || probe.runtime != "mlx-lm" {
            return Err(anyhow!(
                "MLX probe selected {}/{} but Bloom requested {}/mlx-lm",
                probe.device,
                probe.runtime,
                device_label(device)
            ));
        }
        let manifest = crate::manifest_adapter::load_manifest(model_path).unwrap_or_default();
        let metadata = ModelMetadata {
            id: mlx_file
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("mlx-model")
                .to_string(),
            modality: Modality::Text,
            quantized: true,
            manifest,
        };
        Ok(Box::new(MlxRuntimeModel {
            model_path: package_path,
            python,
            device,
            metadata,
        }))
    }
}

struct MlxRuntimeModel {
    model_path: PathBuf,
    python: PathBuf,
    device: DeviceKind,
    metadata: ModelMetadata,
}

impl MlxRuntimeModel {
    fn command(&self, stream: bool) -> Command {
        let script = script_path();
        let mut command = Command::new(&self.python);
        command
            .env("PYTHONIOENCODING", "utf-8")
            .arg(script)
            .arg("--model-path")
            .arg(&self.model_path)
            .arg("--device")
            .arg(device_label(self.device));
        if stream {
            command.arg("--stream");
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }
}

impl LoadedModel for MlxRuntimeModel {
    fn actual_device(&self) -> Option<DeviceKind> {
        Some(self.device)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(
        &self,
        input: ModelInput,
        params: &bloomai_core::GenerationParams,
    ) -> Result<ModelOutput> {
        let request = serde_json::to_vec(&serde_json::json!({
            "input": input,
            "params": params,
        }))?;
        let script = script_path();
        crate::core::security::validate_external_script(&script)?;
        let mut command = self.command(false);
        let output = crate::core::process::run_bounded_output(
            &mut command,
            &request,
            MAX_MLX_STDOUT_BYTES,
            MAX_MLX_STDERR_BYTES,
            mlx_bridge_timeout()?,
        )
        .map_err(|error| anyhow!("MLX-LM bridge failed: {error}"))?;
        if output.stdout_exceeded {
            return Err(anyhow!(
                "MLX-LM bridge stdout exceeded the {} MiB limit",
                MAX_MLX_STDOUT_BYTES / (1024 * 1024)
            ));
        }
        if output.stderr_exceeded {
            return Err(anyhow!(
                "MLX-LM bridge stderr exceeded the {} KiB limit",
                MAX_MLX_STDERR_BYTES / 1024
            ));
        }
        if !output.status.success() {
            return Err(anyhow!(
                "MLX-LM bridge failed ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        serde_json::from_slice(&output.stdout).map_err(|error| {
            anyhow!(
                "MLX-LM bridge returned invalid JSON: {error}; output={:?}",
                String::from_utf8_lossy(&output.stdout)
            )
        })
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &bloomai_core::GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let request = serde_json::to_vec(&serde_json::json!({
            "input": input,
            "params": params,
        }))?;
        let script = script_path();
        crate::core::security::validate_external_script(&script)?;
        // Keep the bridge under the process guard for the full stream. A sink
        // can reject a chunk (for example when the client disconnects), and
        // dropping a raw Child in that path would leave the Python worker
        // running after the request has already failed.
        let mut command = self.command(true);
        // The stream is consumed synchronously below, so inherit diagnostics
        // instead of leaving a stderr pipe that could fill and block the
        // worker before it closes stdout.
        command.stderr(Stdio::inherit());
        let mut child = ChildGuard::spawn(&mut command)
            .map_err(|error| anyhow!("failed to launch MLX-LM streaming bridge: {error}"))?;
        child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MLX-LM streaming bridge stdin is unavailable"))?
            .write_all(&request)?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MLX-LM streaming bridge stdout is unavailable"))?;
        let mut saw_done = false;
        for line in BufReader::new(stdout).lines() {
            let line = line?;
            if line.len() > 256 * 1024 {
                return Err(anyhow!("MLX-LM streaming event exceeds 256 KiB"));
            }
            let event: serde_json::Value = serde_json::from_str(&line)
                .map_err(|error| anyhow!("MLX-LM bridge returned invalid stream JSON: {error}"))?;
            if let Some(text) = event.get("text").and_then(|value| value.as_str())
                && !text.is_empty()
            {
                sink.on_chunk(OutputChunk::TextDelta(text.to_string()))?;
            }
            if event.get("done").and_then(|value| value.as_bool()) == Some(true) {
                saw_done = true;
            }
        }
        let status = child
            .wait()
            .map_err(|error| anyhow!("failed waiting for MLX-LM streaming bridge: {error}"))?;
        if !status.success() {
            return Err(anyhow!("MLX-LM streaming bridge failed ({status})"));
        }
        if !saw_done {
            return Err(anyhow!(
                "MLX-LM streaming bridge ended without a terminal event"
            ));
        }
        sink.on_chunk(OutputChunk::End)
    }
}

fn bundled_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/mlx_lm_infer.py")
}

fn script_path() -> PathBuf {
    if let Some(path) = std::env::var_os("BLOOM_MLX_SCRIPT") {
        return PathBuf::from(path);
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        for candidate in [
            parent.join("mlx_lm_infer.py"),
            parent.join("scripts").join("mlx_lm_infer.py"),
        ] {
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    bundled_script()
}

fn python_path() -> PathBuf {
    std::env::var_os("BLOOM_MLX_PYTHON")
        .or_else(|| std::env::var_os("BLOOM_PYTHON"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"))
}

fn device_label(device: DeviceKind) -> &'static str {
    match device {
        DeviceKind::Cpu => "CPU",
        DeviceKind::Gpu => "GPU",
        DeviceKind::Npu => "NPU",
    }
}

fn run_probe(python: &Path, script: &Path, model: &Path, device: DeviceKind) -> Result<MlxProbe> {
    let mut command = Command::new(python);
    command
        .env("PYTHONIOENCODING", "utf-8")
        .arg(script)
        .arg("--probe")
        .arg("--model-path")
        .arg(model)
        .arg("--device")
        .arg(device_label(device))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = crate::core::process::run_bounded_output(
        &mut command,
        &[],
        MAX_MLX_STDERR_BYTES,
        MAX_MLX_STDERR_BYTES,
        mlx_bridge_timeout()?,
    )
    .map_err(|error| anyhow!("failed to launch MLX-LM probe: {error}"))?;
    if output.stdout_exceeded || output.stderr_exceeded {
        return Err(anyhow!("MLX-LM probe output exceeded 64 KiB"));
    }
    if !output.status.success() {
        return Err(anyhow!(
            "MLX-LM probe failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let probe: MlxProbe = serde_json::from_slice(&output.stdout)
        .map_err(|error| anyhow!("MLX-LM probe returned invalid JSON: {error}"))?;
    if !probe.ok {
        return Err(anyhow!("MLX-LM probe reported failure"));
    }
    Ok(probe)
}

fn mlx_bridge_timeout() -> Result<Duration> {
    let Some(value) = std::env::var_os("BLOOM_MLX_TIMEOUT_SECS") else {
        return Ok(DEFAULT_MLX_TIMEOUT);
    };
    let value = value
        .to_str()
        .ok_or_else(|| anyhow!("BLOOM_MLX_TIMEOUT_SECS must be valid UTF-8"))?;
    let seconds = value.parse::<u64>().map_err(|_| {
        anyhow!("BLOOM_MLX_TIMEOUT_SECS must be an integer between 1 and {MAX_MLX_TIMEOUT_SECS}")
    })?;
    if !(1..=MAX_MLX_TIMEOUT_SECS).contains(&seconds) {
        return Err(anyhow!(
            "BLOOM_MLX_TIMEOUT_SECS must be between 1 and {MAX_MLX_TIMEOUT_SECS}"
        ));
    }
    Ok(Duration::from_secs(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_declares_experimental_mlx_text_contract() {
        let cap = MlxEngine.capability();
        assert_eq!(cap.engine_name, "mlx");
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Experimental);
        assert_eq!(cap.supported_modalities, vec![Modality::Text]);
        assert!(cap.supported_formats.contains(&ModelFormat::Mlx));
        assert!(cap.supported_devices.contains(&DeviceClass::Cpu));
        #[cfg(target_os = "macos")]
        assert!(cap.supported_devices.contains(&DeviceClass::IntegratedGpu));
        #[cfg(not(target_os = "macos"))]
        assert!(!cap.supported_devices.contains(&DeviceClass::IntegratedGpu));
    }

    #[test]
    fn discovers_weights_safetensors_and_npz() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.safetensors");
        std::fs::write(&file, b"placeholder").unwrap();

        assert_eq!(MlxEngine::discover_model_file(&file).unwrap(), file);
        assert_eq!(MlxEngine::discover_model_file(dir.path()).unwrap(), file);
    }

    #[cfg(unix)]
    #[test]
    fn external_bridge_probe_and_inference_round_trip() {
        use crate::test_env::{ENV_LOCK, EnvVarGuard};
        use std::os::unix::fs::PermissionsExt;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("weights.safetensors");
        std::fs::write(&model_file, b"placeholder").unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let script = dir.path().join("fake-mlx.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\nif [ \"$1\" = \"--probe\" ]; then printf '%s\\n' '{\"ok\":true,\"device\":\"CPU\",\"runtime\":\"mlx-lm\"}'; elif [ \"$1\" = \"--stream\" ]; then cat >/dev/null; printf '%s\\n' '{\"text\":\"hello\"}'; printf '%s\\n' '{\"done\":true}'; else cat >/dev/null; printf '%s\\n' '{\"text\":\"fake output\"}'; fi\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _script = EnvVarGuard::set("BLOOM_MLX_SCRIPT", script.to_str().unwrap());
        let _python = EnvVarGuard::set("BLOOM_MLX_PYTHON", "sh");

        let model = MlxEngine.load(dir.path(), DeviceKind::Cpu).unwrap();
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

    #[cfg(unix)]
    #[test]
    fn streaming_bridge_is_reaped_when_sink_rejects_output() {
        use crate::test_env::{ENV_LOCK, EnvVarGuard};
        use std::os::unix::fs::PermissionsExt;

        let _lock = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("weights.safetensors");
        std::fs::write(&model_file, b"placeholder").unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let pid_path = dir.path().join("stream.pid");
        let script = dir.path().join("fake-mlx-stream.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\ncase \" $* \" in\n  *\" --probe \"*) printf '%s\\n' '{\"ok\":true,\"device\":\"CPU\",\"runtime\":\"mlx-lm\"}' ;;\n  *\" --stream \"*) printf '%s\\n' \"$$\" > \"$BLOOM_MLX_PID_FILE\"; cat >/dev/null; printf '%s\\n' '{\"text\":\"hello\"}'; exec sleep 30 ;;\n  *) cat >/dev/null; printf '%s\\n' '{\"text\":\"fake output\"}' ;;\nesac\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _script = EnvVarGuard::set("BLOOM_MLX_SCRIPT", script.to_str().unwrap());
        let _python = EnvVarGuard::set("BLOOM_MLX_PYTHON", "sh");
        let _pid = EnvVarGuard::set("BLOOM_MLX_PID_FILE", pid_path.to_str().unwrap());

        let model = MlxEngine.load(dir.path(), DeviceKind::Cpu).unwrap();
        let error = model
            .infer_stream(
                ModelInput::Text {
                    prompt: "hello".to_string(),
                },
                &bloomai_core::GenerationParams::default(),
                &mut |_| anyhow::bail!("cancelled"),
            )
            .unwrap_err();
        assert!(error.to_string().contains("cancelled"));

        let pid: libc::pid_t = std::fs::read_to_string(pid_path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // ChildGuard must kill and reap the bridge on the sink's early error;
        // otherwise this worker would keep running after the request failed.
        assert_eq!(
            unsafe { libc::waitpid(pid, std::ptr::null_mut(), libc::WNOHANG) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
    }
}
