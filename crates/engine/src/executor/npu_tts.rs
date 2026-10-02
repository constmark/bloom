//! Intel NPU TTS (Text-to-Speech) inference engine for Bloom.
//!
//! Supports CosyVoice, CosyVoice2, and ChatTTS models via:
//! - OpenVINO GenAI on Intel NPU (primary)
//! - CosyVoice PyTorch backend (fallback)
//! - ChatTTS backend (final fallback)
//!
//! Models can be downloaded automatically from ModelScope to the configured
//! `BLOOM_MODEL_ROOT` (or the user's `models` directory) when not found locally.

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
// Helpers
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
    if let Some(path) = std::env::var_os("BLOOM_TTS_PYTHON") {
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

/// Default model root directory: the user's `models` directory on every
/// platform. `BLOOM_MODEL_ROOT` remains the explicit override for managed or
/// shared deployments.
fn default_model_root() -> PathBuf {
    if let Some(root) = std::env::var_os("BLOOM_MODEL_ROOT") {
        return PathBuf::from(root);
    }

    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| {
            let drive = std::env::var_os("HOMEDRIVE").filter(|value| !value.is_empty())?;
            let path = std::env::var_os("HOMEPATH").filter(|value| !value.is_empty())?;
            let mut home = PathBuf::from(drive);
            home.push(path);
            Some(home)
        })
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("models")
}

/// Check if a TTS model directory has the expected layout.
fn has_tts_model(model_path: &Path) -> bool {
    // CosyVoice layout
    if model_path.join("cosyvoice.yaml").exists() || model_path.join("config.yaml").exists() {
        return true;
    }
    // ChatTTS layout (just needs to be importable)
    if model_path.join("config.json").exists() {
        let path_str = model_path.to_string_lossy().to_lowercase();
        if path_str.contains("chattts") || path_str.contains("chat_tts") {
            return true;
        }
    }
    // OpenVINO IR layout
    if model_path.join("openvino_model.xml").exists() {
        return true;
    }
    false
}

/// Detect which TTS backend the model uses.
#[derive(Debug, Clone, Copy, PartialEq)]
enum TtsBackend {
    CosyVoice,
    ChatTts,
    OpenVinoGenAi,
}

fn detect_tts_backend(model_path: &Path) -> TtsBackend {
    // OpenVINO IR takes priority
    if model_path.join("openvino_model.xml").exists() {
        return TtsBackend::OpenVinoGenAi;
    }

    let path_str = model_path.to_string_lossy().to_lowercase();

    // ChatTTS detection
    if (path_str.contains("chattts") || path_str.contains("chat_tts"))
        && model_path.join("config.json").exists()
    {
        return TtsBackend::ChatTts;
    }

    // Default to CosyVoice
    TtsBackend::CosyVoice
}

// ---------------------------------------------------------------------------
// ModelScope download helper
// ---------------------------------------------------------------------------

/// Known TTS model repositories on ModelScope.
struct ModelScopeRepo {
    /// ModelScope repo id, e.g. "iic/CosyVoice2-0.5B"
    repo_id: &'static str,
    /// Local directory name under the model root
    local_name: &'static str,
}

const KNOWN_TTS_MODELS: &[ModelScopeRepo] = &[
    ModelScopeRepo {
        repo_id: "iic/CosyVoice2-0.5B",
        local_name: "CosyVoice2-0.5B",
    },
    ModelScopeRepo {
        repo_id: "iic/CosyVoice-300M",
        local_name: "CosyVoice-300M",
    },
    ModelScopeRepo {
        repo_id: "AI-ModelScope/ChatTTS",
        local_name: "ChatTTS",
    },
];

/// Download a model from ModelScope if it doesn't exist locally.
///
/// Uses the `modelscope` Python SDK to download to the configured model root.
fn ensure_model_available(model_name: &str) -> Result<PathBuf> {
    let model_root = default_model_root();
    let model_path = model_root.join(model_name);

    // Already downloaded?
    if model_path.exists() && has_tts_model(&model_path) {
        tracing::info!("TTS model found at: {}", model_path.display());
        return Ok(model_path);
    }

    // Find matching ModelScope repo
    let repo = KNOWN_TTS_MODELS.iter().find(|r| {
        model_name.eq_ignore_ascii_case(r.local_name)
            || model_name.contains(&r.local_name.to_lowercase())
    });

    let repo_id = match repo {
        Some(r) => r.repo_id,
        None => {
            // Try using the model_name directly as a ModelScope repo id
            // (e.g., "iic/CosyVoice2-0.5B")
            if model_name.contains('/') {
                tracing::info!(
                    "Model '{}' not found locally, attempting download from ModelScope...",
                    model_name
                );
                return download_from_modelscope(model_name, &model_root);
            }
            return Err(anyhow!(
                "TTS model '{}' not found at {} and no known ModelScope repo matches. \
                 Known models: {:?}. \
                 Set BLOOM_MODEL_ROOT to override the model directory.",
                model_name,
                model_path.display(),
                KNOWN_TTS_MODELS
                    .iter()
                    .map(|r| r.local_name)
                    .collect::<Vec<_>>()
            ));
        }
    };

    tracing::info!(
        "TTS model '{}' not found locally. Downloading from ModelScope: {}...",
        model_name,
        repo_id
    );

    download_from_modelscope(repo_id, &model_root)
}

/// Download a model from ModelScope using the Python SDK.
fn download_from_modelscope(repo_id: &str, model_root: &Path) -> Result<PathBuf> {
    let python = default_python();
    crate::core::security::validate_runner(&python)?;

    // Ensure modelscope is available
    let check = Command::new(&python)
        .arg("-c")
        .arg("import modelscope")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if !check.map(|s| s.success()).unwrap_or(false) {
        return Err(anyhow!(
            "modelscope Python package not found. Install with: pip install modelscope"
        ));
    }

    // Create model root directory
    std::fs::create_dir_all(model_root).map_err(|e| {
        anyhow!(
            "failed to create model directory {}: {}",
            model_root.display(),
            e
        )
    })?;

    // Download via modelscope snapshot_download
    let download_script = format!(
        "from modelscope import snapshot_download; \
         path = snapshot_download('{}', cache_dir='{}'); \
         print(path)",
        repo_id,
        model_root.display()
    );

    let output = Command::new(&python)
        .arg("-c")
        .arg(&download_script)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| anyhow!("failed to run modelscope download: {}", e))?;

    if !output.status.success() {
        return Err(anyhow!(
            "ModelScope download failed for '{}'. \
             You can download manually and place in {}.",
            repo_id,
            model_root.display()
        ));
    }

    let downloaded_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let result_path = PathBuf::from(&downloaded_path);

    if result_path.exists() {
        tracing::info!("Model downloaded to: {}", result_path.display());
        Ok(result_path)
    } else {
        // Fallback: look in model_root for the downloaded directory
        let repo_name = repo_id.split('/').next_back().unwrap_or(repo_id);
        let fallback_path = model_root.join(repo_name);
        if fallback_path.exists() {
            Ok(fallback_path)
        } else {
            Err(anyhow!(
                "download completed but model directory not found at expected location"
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// NPU TTS Engine
// ---------------------------------------------------------------------------

/// Text-to-Speech engine targeting Intel NPU via OpenVINO GenAI,
/// with fallback to CosyVoice/ChatTTS on CPU/GPU.
pub struct NpuTtsEngine;

impl Engine for NpuTtsEngine {
    fn name(&self) -> &'static str {
        "npu-tts"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text, Modality::Audio]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Npu, DeviceKind::Cpu, DeviceKind::Gpu]
    }

    fn default_device(&self) -> DeviceKind {
        DeviceKind::Npu
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "npu-tts",
            supported_families: vec![ModelFamily::Custom("tts".to_string())],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::I8,
                bloomai_core::DType::U8,
                bloomai_core::DType::I4,
            ],
            supported_formats: vec![ModelFormat::OpenVinoIr, ModelFormat::Safetensors],
            supported_devices: vec![
                DeviceClass::Npu,
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
            ],
            supported_modalities: vec![Modality::Text, Modality::Audio],
            supports_streaming: false,
            supports_quantized_models: true,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![crate::core::parallelism::ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Requires Intel NPU with TTS model in OpenVINO IR format.".to_string(),
            ],
            construction_guide: "Requires Intel NPU driver. Build with --features npu-tts."
                .to_string(),
        }
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let device_str = match device {
            DeviceKind::Cpu => "CPU",
            DeviceKind::Gpu => "GPU",
            DeviceKind::Npu => "NPU",
        };

        // Auto-download from ModelScope if model not found
        let resolved_path = if !model_path.exists() {
            let model_name = model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("CosyVoice2-0.5B");

            tracing::info!(
                "Model path does not exist: {}. Attempting ModelScope download...",
                model_path.display()
            );
            ensure_model_available(model_name)?
        } else {
            model_path.to_path_buf()
        };

        if !resolved_path.exists() {
            return Err(anyhow!(
                "model path does not exist: {}",
                resolved_path.display()
            ));
        }

        let backend = detect_tts_backend(&resolved_path);
        tracing::info!(
            "TTS model loaded: {} (backend={:?}, device={})",
            resolved_path.display(),
            backend,
            device_str
        );

        let is_quantized = resolved_path
            .to_string_lossy()
            .to_lowercase()
            .contains("int4")
            || resolved_path
                .to_string_lossy()
                .to_lowercase()
                .contains("int8")
            || resolved_path
                .to_string_lossy()
                .to_lowercase()
                .contains("quant");

        let manifest = crate::manifest_adapter::load_manifest(&resolved_path)?;

        let metadata = ModelMetadata {
            id: resolved_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            modality: Modality::Audio,
            quantized: is_quantized,
            manifest,
        };

        Ok(Box::new(NpuTtsModel {
            model_path: resolved_path,
            device: device_str.to_string(),
            backend,
            metadata,
        }))
    }
}

// ---------------------------------------------------------------------------
// Loaded Model implementation
// ---------------------------------------------------------------------------

struct NpuTtsModel {
    model_path: PathBuf,
    device: String,
    backend: TtsBackend,
    metadata: ModelMetadata,
}

impl LoadedModel for NpuTtsModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let text = match input {
            ModelInput::Text { prompt } => prompt,
            ModelInput::Multi { text: Some(t), .. } => t,
            _ => return Err(anyhow!("TTS model requires text input")),
        };

        let script = std::env::var_os("BLOOM_TTS_SCRIPT")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo_script_path("npu_tts_infer.py"));
        crate::core::security::validate_external_script(&script)?;

        let python = default_python();
        crate::core::security::validate_runner(&python)?;

        let mut command = Command::new(python);
        command.env("PYTHONIOENCODING", "utf-8");
        command
            .arg(&script)
            .arg("--model-path")
            .arg(&self.model_path)
            .arg("--text")
            .arg(&text)
            .arg("--device")
            .arg(&self.device)
            .arg("--sample-rate")
            .arg("22050");

        // Speed from generation params (reuse temperature as speed proxy)
        let speed = if params.temperature > 0.1 && params.temperature < 5.0 {
            params.temperature
        } else {
            1.0
        };
        command.arg("--speed").arg(speed.to_string());

        // Use a temporary output file to get PCM samples back
        let work_dir = tempfile::Builder::new().prefix("bloom-tts-").tempdir()?;
        let temp_wav = work_dir.path().join("output.wav");
        command.arg("--output").arg(&temp_wav);

        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        tracing::info!(
            "Starting TTS inference: device={} model={} backend={:?}",
            self.device,
            self.model_path.display(),
            self.backend
        );

        let output = command
            .output()
            .map_err(|e| anyhow!("failed to spawn TTS inference process: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("TTS inference failed: {}", stderr));
        }

        // Read the WAV output and convert to PCM float samples
        let (samples, sample_rate) = read_wav_to_pcm(&temp_wav)?;

        if samples.is_empty() {
            return Err(anyhow!("TTS inference produced no audio output"));
        }

        let stdout_text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let result_text = if stdout_text.is_empty() {
            Some(text)
        } else {
            Some(stdout_text)
        };

        Ok(ModelOutput {
            text: result_text,
            logits: None,
            image: None,
            audio: Some((samples, sample_rate)),
            video: None,
        })
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        // TTS is inherently non-streaming — run full inference and emit result
        let output = self.infer(input, params)?;

        if let Some(text) = output.text {
            sink.on_chunk(crate::io::OutputChunk::TextDelta(text))?;
        }

        if let Some((samples, sample_rate)) = output.audio {
            let mut chunks = samples.chunks(1000).peekable();
            while let Some(chunk) = chunks.next() {
                sink.on_chunk(crate::io::OutputChunk::TtsAudioChunk {
                    samples: chunk.to_vec(),
                    sample_rate,
                    is_final: chunks.peek().is_none(),
                })?;
            }
        }

        sink.on_chunk(crate::io::OutputChunk::End)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WAV file reader (minimal, for temp output)
// ---------------------------------------------------------------------------

/// Read a WAV file and return PCM float samples and sample rate.
fn read_wav_to_pcm(path: &Path) -> Result<(Vec<f32>, u32)> {
    use std::io::Read;
    const MAX_WAV_BYTES: u64 = 64 * 1024 * 1024;
    let mut data = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_WAV_BYTES + 1)
        .read_to_end(&mut data)?;
    anyhow::ensure!(
        data.len() as u64 <= MAX_WAV_BYTES,
        "TTS WAV output exceeds 64 MiB"
    );
    decode_wav(&data)
}

fn decode_wav(data: &[u8]) -> Result<(Vec<f32>, u32)> {
    anyhow::ensure!(
        data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE",
        "invalid RIFF/WAVE header"
    );
    let u16_at =
        |bytes: &[u8], offset: usize| u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    let u32_at = |bytes: &[u8], offset: usize| {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    let end = (u32_at(data, 4) as usize)
        .checked_add(8)
        .ok_or_else(|| anyhow!("WAV RIFF size overflow"))?;
    anyhow::ensure!(
        end >= 12 && end <= data.len(),
        "truncated WAV RIFF container"
    );
    let mut format = None;
    let mut pcm = None;
    let mut pos = 12usize;
    while pos < end {
        anyhow::ensure!(end - pos >= 8, "truncated WAV chunk header");
        let size = u32_at(data, pos + 4) as usize;
        let start = pos + 8;
        let chunk_end = start
            .checked_add(size)
            .ok_or_else(|| anyhow!("WAV chunk size overflow"))?;
        anyhow::ensure!(chunk_end <= end, "truncated WAV chunk");
        let chunk = &data[start..chunk_end];
        match &data[pos..pos + 4] {
            b"fmt " => {
                anyhow::ensure!(
                    format.is_none() && size >= 16,
                    "invalid or duplicate WAV format"
                );
                format = Some((
                    u16_at(chunk, 0),
                    u16_at(chunk, 2) as usize,
                    u32_at(chunk, 4),
                    u32_at(chunk, 8),
                    u16_at(chunk, 12) as usize,
                    u16_at(chunk, 14),
                ));
            }
            b"data" => {
                anyhow::ensure!(pcm.is_none(), "duplicate WAV data chunk");
                pcm = Some(chunk);
            }
            _ => {}
        }
        pos = chunk_end
            .checked_add(size % 2)
            .ok_or_else(|| anyhow!("WAV padding overflow"))?;
        anyhow::ensure!(pos <= end, "missing WAV chunk padding");
    }
    let (encoding, channels, sample_rate, byte_rate, block_align, bits) =
        format.ok_or_else(|| anyhow!("WAV format chunk is missing"))?;
    anyhow::ensure!(
        channels > 0 && sample_rate > 0,
        "WAV channels and sample rate must be positive"
    );
    anyhow::ensure!(
        matches!((encoding, bits), (1, 8 | 16 | 24 | 32) | (3, 32)),
        "unsupported WAV encoding {encoding}/{bits}"
    );
    let bytes_per_sample = bits as usize / 8;
    let frame_bytes = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| anyhow!("WAV frame size overflow"))?;
    anyhow::ensure!(
        block_align == frame_bytes
            && u64::from(byte_rate) == u64::from(sample_rate) * frame_bytes as u64,
        "inconsistent WAV frame or byte rate"
    );
    let pcm = pcm.ok_or_else(|| anyhow!("WAV data chunk is missing"))?;
    anyhow::ensure!(
        pcm.len().is_multiple_of(frame_bytes),
        "incomplete WAV sample frame"
    );
    let mut samples = Vec::with_capacity(pcm.len() / frame_bytes);
    for frame in pcm.chunks_exact(frame_bytes) {
        let mut sum = 0.0f64;
        for sample in frame.chunks_exact(bytes_per_sample) {
            let value = match (encoding, bits) {
                (1, 8) => (sample[0] as f64 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([sample[0], sample[1]]) as f64 / 32768.0,
                (1, 24) => {
                    ((i32::from_le_bytes([sample[0], sample[1], sample[2], 0]) << 8) >> 8) as f64
                        / 8388608.0
                }
                (1, 32) => {
                    i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]) as f64
                        / 2147483648.0
                }
                (3, 32) => f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]) as f64,
                _ => unreachable!("encoding validated above"),
            };
            anyhow::ensure!(value.is_finite(), "WAV contains non-finite samples");
            sum += value;
        }
        samples.push((sum / channels as f64) as f32);
    }
    Ok((samples, sample_rate))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_npu_tts_engine_metadata() {
        let engine = NpuTtsEngine;
        assert_eq!(engine.name(), "npu-tts");
        assert!(engine.supported_modalities().contains(&Modality::Text));
        assert!(engine.supported_modalities().contains(&Modality::Audio));
        assert_eq!(engine.default_device(), DeviceKind::Npu);
        assert!(engine.supported_devices().contains(&DeviceKind::Npu));
        assert!(engine.supported_devices().contains(&DeviceKind::Cpu));
    }

    #[test]
    fn test_npu_tts_engine_capability() {
        let engine = NpuTtsEngine;
        let cap = engine.capability();
        assert_eq!(cap.engine_name, "npu-tts");
        assert!(cap.supported_devices.contains(&DeviceClass::Npu));
        assert!(cap.supported_formats.contains(&ModelFormat::OpenVinoIr));
        assert!(!cap.supports_streaming);
    }

    #[test]
    fn test_detect_tts_backend_openvino() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("openvino_model.xml"), "<xml/>").unwrap();
        assert_eq!(detect_tts_backend(dir.path()), TtsBackend::OpenVinoGenAi);
    }

    #[test]
    fn test_detect_tts_backend_cosyvoice() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("cosyvoice.yaml"), "").unwrap();
        assert_eq!(detect_tts_backend(dir.path()), TtsBackend::CosyVoice);
    }

    #[test]
    fn test_detect_tts_backend_chattts() {
        let dir = tempdir().unwrap();
        // Need "chattts" in path for detection
        let chattts_dir = dir.path().join("ChatTTS");
        std::fs::create_dir_all(&chattts_dir).unwrap();
        std::fs::write(chattts_dir.join("config.json"), "{}").unwrap();
        assert_eq!(detect_tts_backend(&chattts_dir), TtsBackend::ChatTts);
    }

    #[test]
    fn test_has_tts_model_empty() {
        let dir = tempdir().unwrap();
        assert!(!has_tts_model(dir.path()));
    }

    #[test]
    fn test_has_tts_model_with_yaml() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("config.yaml"), "").unwrap();
        assert!(has_tts_model(dir.path()));
    }

    #[test]
    fn test_npu_tts_load_nonexistent() {
        let engine = NpuTtsEngine;
        let non_existent = Path::new("nonexistent_tts_model_xyz_12345");
        // Should attempt ModelScope download and fail (no modelscope package)
        let res = engine.load(non_existent, DeviceKind::Cpu);
        assert!(res.is_err());
    }

    #[test]
    fn test_read_wav_to_pcm_nonexistent() {
        let dir = tempdir().unwrap();
        assert!(read_wav_to_pcm(&dir.path().join("missing.wav")).is_err());
    }

    #[test]
    fn test_read_wav_to_pcm_valid() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("test.wav");

        // Create a minimal valid 16-bit mono WAV
        let sample_rate: u32 = 22050;
        let num_channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768];
        let data_size = samples.len() * 2;
        let file_size = 36 + data_size;

        let mut wav_data = Vec::new();
        // RIFF header
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&(file_size as u32).to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");
        // fmt chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav_data.extend_from_slice(&num_channels.to_le_bytes());
        wav_data.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
        wav_data.extend_from_slice(&byte_rate.to_le_bytes());
        let block_align = num_channels * bits_per_sample / 8;
        wav_data.extend_from_slice(&block_align.to_le_bytes());
        wav_data.extend_from_slice(&bits_per_sample.to_le_bytes());
        // data chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&(data_size as u32).to_le_bytes());
        for s in &samples {
            wav_data.extend_from_slice(&s.to_le_bytes());
        }

        std::fs::write(&wav_path, &wav_data).unwrap();

        let (pcm, sr) = read_wav_to_pcm(&wav_path).unwrap();
        assert_eq!(sr, 22050);
        assert_eq!(pcm.len(), 5);
        assert!((pcm[1] - 1000.0 / 32768.0).abs() < 0.001);
        assert!((pcm[3] - 1.0).abs() < 0.001);
        for (offset, value) in [
            (0, 0),
            (20, 0),
            (22, 0),
            (24, 0),
            (32, 0),
            (34, 0),
            (34, 7),
            (40, 255),
        ] {
            let mut malformed = wav_data.clone();
            malformed[offset] = value;
            assert!(
                decode_wav(&malformed).is_err(),
                "accepted malformed header at {offset}"
            );
        }
        for length in 0..wav_data.len() {
            assert!(decode_wav(&wav_data[..length]).is_err());
        }
        // A padded ancillary chunk before fmt must not change header offsets.
        let mut extended = wav_data.clone();
        extended.splice(12..12, *b"JUNK\x01\x00\x00\x00x\x00");
        let len = (extended.len() - 8) as u32;
        extended[4..8].copy_from_slice(&len.to_le_bytes());
        assert_eq!(decode_wav(&extended).unwrap(), (pcm, sr));
    }

    #[test]
    fn test_default_model_root() {
        let root = default_model_root();
        #[cfg(target_os = "windows")]
        assert_eq!(root, PathBuf::from("D:\\models"));
        #[cfg(not(target_os = "windows"))]
        assert!(root.to_string_lossy().contains("models"));
    }
}
