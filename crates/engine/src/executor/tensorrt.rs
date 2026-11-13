//! TensorRT numeric execution through a bounded external NVIDIA Python runtime.
//!
//! This adapter loads real serialized plans; tokenizer/LLM generation, explicit
//! graph capture, engine compilation and parallel placement are not emulated.

use anyhow::{Result, anyhow, bail};
use bloomai_core::{
    DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat, ModelManifest,
    constants::GIB,
};
use std::path::{Path, PathBuf};

use crate::{
    core::parallelism::ParallelStrategy,
    core::quantization::KvCacheDtype,
    engine::{Engine, EngineCapability, SupportLevel, default_engine_supports},
    io::{ModelInput, ModelOutput},
    model::{LoadedModel, ModelMetadata},
};

/// Reserved builder/LLM descriptor retained for source compatibility.
///
/// The numeric bridge does not apply these options. Engine building and custom
/// attention methods fail explicitly; this descriptor is not an execution or
/// memory-budget guarantee.
#[derive(Debug, Clone)]
pub struct TensorRtEngineConfig {
    /// Maximum execution batch size.
    pub max_batch_size: usize,
    /// Maximum input/output sequence length.
    pub max_seq_len: usize,
    /// Maximum number of tokens allowed per step (budget constraint).
    pub max_num_tokens: usize,
    /// Max workspace VRAM allocation in bytes (default: 4GB).
    pub max_workspace_size: usize,
    /// Whether to compile and use CUDA graph for decode steps.
    pub enable_cuda_graph: bool,
    /// KV Cache storage data type (FP16, FP8, NVFP4, INT8).
    pub kv_dtype: KvCacheDtype,
    /// Tensor Parallelism size (multi-GPU).
    pub tp_size: usize,
    /// Pipeline Parallelism size (multi-node).
    pub pp_size: usize,
    /// Enable fused Multi-Head/Group Query Attention kernels.
    pub enable_fused_attention: bool,
}

impl Default for TensorRtEngineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 128,
            max_seq_len: 4096,
            max_num_tokens: 8192,
            max_workspace_size: 4 * GIB as usize, // 4 GB
            enable_cuda_graph: true,
            kv_dtype: KvCacheDtype::F16,
            tp_size: 1,
            pp_size: 1,
            enable_fused_attention: true,
        }
    }
}

/// Compatibility API for graph capture, which is not implemented by this adapter.
#[derive(Debug, Default)]
pub struct CudaGraphManager {
    captured: bool,
    num_replays: usize,
}

impl CudaGraphManager {
    pub fn new() -> Self {
        Self {
            captured: false,
            num_replays: 0,
        }
    }

    /// CUDA Graph capture is unavailable until the optional TensorRT runtime
    /// is linked.  The Python tensor bridge
    /// must fail closed instead of reporting a simulated capture as usable.
    pub fn capture_pass(&mut self) -> Result<()> {
        Err(anyhow!(
            "Explicit CUDA Graph capture is unavailable in the TensorRT tensor bridge"
        ))
    }

    /// Replays the captured CUDA graph.
    pub fn replay_pass(&mut self) -> Result<()> {
        if !self.captured {
            return Err(anyhow!("CUDA Graph has not been captured yet."));
        }
        self.num_replays += 1;
        Ok(())
    }

    pub fn is_captured(&self) -> bool {
        self.captured
    }

    pub fn replay_count(&self) -> usize {
        self.num_replays
    }
}

pub struct TensorRtEngine;

impl TensorRtEngine {
    pub fn discover_engine_file(model_path: &Path) -> Result<PathBuf> {
        if model_path.is_file() {
            if is_tensorrt_engine_file(model_path) {
                return Ok(model_path.to_path_buf());
            }
            bail!(
                "TensorRT adapter expects a .engine/.plan file or a directory containing one: {}",
                model_path.display()
            );
        }

        if !model_path.is_dir() {
            bail!(
                "model path does not exist or is not readable: {}",
                model_path.display()
            );
        }

        for name in ["model.engine", "model.plan", "engine.plan"] {
            let candidate = model_path.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }

        let mut candidates = std::fs::read_dir(model_path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && is_tensorrt_engine_file(path))
            .collect::<Vec<_>>();
        candidates.sort();

        candidates.into_iter().next().ok_or_else(|| {
            anyhow!(
                "no TensorRT .engine/.plan file found in {}; add model.engine or point --model at a serialized TensorRT engine",
                model_path.display()
            )
        })
    }
}

impl Engine for TensorRtEngine {
    fn name(&self) -> &'static str {
        "tensorrt"
    }
    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Multi]
    }
    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Gpu]
    }
    fn default_device(&self) -> DeviceKind {
        DeviceKind::Gpu
    }
    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "tensorrt",
            supported_families: vec![ModelFamily::Custom("*".into())],
            supported_dtypes: vec![bloomai_core::DType::F32, bloomai_core::DType::F16,
                bloomai_core::DType::I8, bloomai_core::DType::U8],
            supported_formats: vec![ModelFormat::TensorRtEngine],
            supported_devices: vec![DeviceClass::DiscreteGpu],
            supported_modalities: self.supported_modalities(),
            supports_streaming: false,
            supports_quantized_models: true,
            supports_embeddings: false, supports_rerank: false,
            supports_structured_output: false, max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Requires NVIDIA TensorRT 10.x, CUDA Python bindings and NumPy in BLOOM_TENSORRT_PYTHON (or BLOOM_PYTHON).".into(),
                "Executes named numeric tensors from a serialized .engine/.plan on CUDA device 0; no text decoder or TensorRT-LLM is emulated.".into(),
            ],
            construction_guide: "Use the bundled bounded TensorRT Python bridge and a compatible serialized plan; send named shape/data JSON as Text input.".into(),
        }
    }
    fn supports(
        &self,
        manifest: &ModelManifest,
        cap: &bloomai_core::DeviceCapability,
    ) -> SupportLevel {
        default_engine_supports(&self.capability(), manifest, cap)
    }
    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        anyhow::ensure!(
            device == DeviceKind::Gpu,
            "TensorRT requires --device gpu; CPU/NPU fallback is unsupported"
        );
        let engine_file = Self::discover_engine_file(model_path)?;
        let metadata = std::fs::metadata(&engine_file)?;
        anyhow::ensure!(
            metadata.is_file() && metadata.len() > 0 && metadata.len() <= 8 * 1024 * 1024 * 1024,
            "TensorRT plan must be a nonempty regular file of at most 8 GiB"
        );
        let python = std::env::var_os("BLOOM_TENSORRT_PYTHON")
            .or_else(|| std::env::var_os("BLOOM_PYTHON"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python3"));
        let script = bridge_script();
        let mut manifest = crate::manifest_adapter::load_manifest(model_path)?;
        manifest
            .parameters
            .insert("bloom_task".into(), serde_json::json!("tensor"));
        let plan_digest = plan_sha256(&engine_file)?;
        let quantized = manifest.quantization.is_some();
        let model = TensorRtModel {
            model_path: engine_file,
            device,
            python,
            script,
            plan_digest,
            prediction: std::sync::Mutex::new(()),
            metadata: ModelMetadata {
                id: manifest.id.clone(),
                modality: Modality::Multi,
                quantized,
                manifest,
            },
            config: TensorRtEngineConfig::default(),
        };
        let probe = model.call_bridge(&[], true)?;
        let probe: TensorRtProbe = serde_json::from_slice(&probe)?;
        anyhow::ensure!(
            probe.ok
                && probe.device == "CUDA"
                && probe.runtime_version.starts_with("10.")
                && valid_probe_tensors(&probe.tensors),
            "TensorRT worker returned incompatible probe identity"
        );
        Ok(Box::new(model))
    }
}

fn is_tensorrt_engine_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "engine" | "plan"))
}

const MAX_BRIDGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_TENSOR_VALUES: usize = 1_048_576;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRtProbe {
    ok: bool,
    device: String,
    runtime_version: String,
    tensors: Vec<TensorRtTensor>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRtTensor {
    name: String,
    shape: Vec<i64>,
    dtype: String,
    input: bool,
}

fn valid_probe_tensors(tensors: &[TensorRtTensor]) -> bool {
    let mut names = std::collections::HashSet::new();
    (2..=32).contains(&tensors.len())
        && tensors.iter().any(|t| t.input)
        && tensors.iter().any(|t| !t.input)
        && tensors.iter().all(|t| {
            !t.name.is_empty()
                && t.name.len() <= 256
                && !t.name.chars().any(char::is_control)
                && names.insert(&t.name)
                && t.shape.len() <= 8
                && matches!(
                    t.dtype.as_str(),
                    "float32" | "float16" | "int8" | "uint8" | "int32" | "int64" | "bool"
                )
                && t.shape
                    .iter()
                    .all(|&d| d == -1 || (1..=MAX_TENSOR_VALUES as i64).contains(&d))
                && (t.shape.contains(&-1)
                    || t.shape
                        .iter()
                        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                        .is_some_and(|n| n <= MAX_TENSOR_VALUES))
        })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRtResult {
    text: Option<String>,
    logits: Vec<f32>,
    image: Option<Vec<u8>>,
    audio: Option<(Vec<f32>, u32)>,
    video: Option<crate::io::VideoOutput>,
}

pub struct TensorRtModel {
    model_path: PathBuf,
    device: DeviceKind,
    python: PathBuf,
    script: PathBuf,
    plan_digest: String,
    prediction: std::sync::Mutex<()>,
    metadata: ModelMetadata,
    config: TensorRtEngineConfig,
}

fn plan_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let len = file.read(&mut chunk)?;
        if len == 0 {
            break;
        }
        total = total
            .checked_add(len as u64)
            .ok_or_else(|| anyhow!("TensorRT plan size overflow"))?;
        anyhow::ensure!(
            total <= 8 * 1024 * 1024 * 1024,
            "TensorRT plan exceeded 8 GiB while hashing"
        );
        digest.update(&chunk[..len]);
    }
    anyhow::ensure!(total > 0, "TensorRT plan is empty");
    Ok(format!("{:x}", digest.finalize()))
}

fn bridge_script() -> PathBuf {
    if let Some(path) = std::env::var_os("BLOOM_TENSORRT_SCRIPT") {
        return path.into();
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
        && parent.join("tensorrt_infer.py").is_file()
    {
        return parent.join("tensorrt_infer.py");
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/tensorrt_infer.py")
}

impl TensorRtModel {
    pub fn config(&self) -> &TensorRtEngineConfig {
        &self.config
    }
    /// Building a plan requires a separate reviewed TensorRT compiler.
    pub fn build_engine_profile(&self, _target_path: &Path) -> Result<()> {
        bail!(
            "TensorRT engine compilation is not implemented; compile a real plan with NVIDIA TensorRT tools"
        )
    }
    pub fn execute_fused_attention(
        &self,
        _num_heads: usize,
        _head_dim: usize,
        _is_prefill: bool,
    ) -> Result<()> {
        bail!(
            "Explicit fused attention execution is not implemented; kernels belong to the serialized TensorRT engine"
        )
    }
    fn call_bridge(&self, request: &[u8], probe: bool) -> Result<Vec<u8>> {
        crate::core::security::validate_runner(&self.python)?;
        crate::core::security::validate_external_script(&self.script)?;
        anyhow::ensure!(
            request.len() <= 8 * 1024 * 1024,
            "TensorRT input exceeded 8 MiB"
        );
        let seconds = std::env::var("BLOOM_TENSORRT_TIMEOUT_SECS")
            .unwrap_or_else(|_| "600".into())
            .parse::<u64>()
            .map_err(|_| anyhow!("BLOOM_TENSORRT_TIMEOUT_SECS must be 1..3600"))?;
        anyhow::ensure!(
            (1..=3600).contains(&seconds),
            "BLOOM_TENSORRT_TIMEOUT_SECS must be 1..3600"
        );
        let mut command = std::process::Command::new(&self.python);
        command
            .arg(&self.script)
            .arg("--model-path")
            .arg(&self.model_path)
            .arg("--sha256")
            .arg(&self.plan_digest)
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if probe {
            command.arg("--probe");
        }
        let output = crate::core::process::run_bounded_output(
            &mut command,
            request,
            if probe { 64 * 1024 } else { MAX_BRIDGE_BYTES },
            64 * 1024,
            std::time::Duration::from_secs(seconds),
        )?;
        anyhow::ensure!(
            !output.stdout_exceeded && !output.stderr_exceeded,
            "TensorRT worker output exceeded its byte limit"
        );
        anyhow::ensure!(
            output.status.success(),
            "TensorRT runtime unavailable or inference failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        Ok(output.stdout)
    }
}

impl LoadedModel for TensorRtModel {
    fn actual_device(&self) -> Option<DeviceKind> {
        Some(self.device)
    }
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        anyhow::ensure!(
            params.logprobs.is_none(),
            "this backend does not support token logprobs"
        );
        input.validate().map_err(anyhow::Error::msg)?;
        params.validate().map_err(anyhow::Error::msg)?;
        // Numeric engines have no sampler; accepting active controls would be misleading.
        let mut neutral = params.clone();
        neutral.response_format = None;
        anyhow::ensure!(
            neutral == GenerationParams::default()
                && matches!(
                    params.response_format,
                    None | Some(bloomai_core::ResponseFormat::Text)
                ),
            "TensorRT numeric inference requires default generation parameters; sampling and structured generation are unsupported"
        );
        let ModelInput::Text { prompt } = input else {
            bail!("TensorRT accepts Text containing named shape/data tensor JSON only");
        };
        let _prediction = self.prediction.try_lock()
            .map_err(|_| anyhow!("TensorRT numeric runtime is busy; only one prediction worker is admitted per loaded model"))?;
        let raw = self.call_bridge(prompt.as_bytes(), false)?;
        let decoded: TensorRtResult = serde_json::from_slice(&raw)?;
        let output = ModelOutput {
            logprobs: None,
            text: decoded.text,
            logits: Some(decoded.logits),
            image: decoded.image,
            audio: decoded.audio,
            video: decoded.video,
        };
        output.validate().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            output.text.is_none()
                && output.image.is_none()
                && output.audio.is_none()
                && output.video.is_none(),
            "TensorRT worker returned unexpected nonnumeric output"
        );
        anyhow::ensure!(
            output
                .logits
                .as_ref()
                .is_some_and(|values| !values.is_empty() && values.len() <= MAX_TENSOR_VALUES),
            "TensorRT worker returned missing or oversized numeric output"
        );
        Ok(output)
    }
    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        anyhow::ensure!(
            params.logprobs.is_none(),
            "this backend does not support token logprobs"
        );
        let output = self.infer(input, params)?;
        sink.on_chunk(crate::OutputChunk::Embedding(
            output
                .logits
                .ok_or_else(|| anyhow!("missing numeric output"))?,
        ))?;
        sink.on_chunk(crate::OutputChunk::End)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capabilities_describe_numeric_external_execution() {
        let cap = TensorRtEngine.capability();
        assert_eq!(cap.maturity, crate::engine::BackendMaturity::Experimental);
        assert_eq!(cap.supported_formats, vec![ModelFormat::TensorRtEngine]);
        assert_eq!(cap.supported_modalities, vec![Modality::Multi]);
        assert!(!cap.supports_streaming && !cap.supports_embeddings);
        assert_eq!(
            cap.supported_parallel_strategies,
            vec![ParallelStrategy::None]
        );
        assert!(cap.supported_quant_methods.is_empty());
    }
    #[test]
    fn discovers_plan_and_rejects_non_gpu_before_runtime_execution() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.plan");
        std::fs::write(&file, b"real-runtime-required").unwrap();
        assert_eq!(
            TensorRtEngine::discover_engine_file(dir.path()).unwrap(),
            file
        );
        let err = match TensorRtEngine.load(&file, DeviceKind::Cpu) {
            Ok(_) => panic!("CPU must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("--device gpu"));
    }
    #[test]
    fn no_simulated_cuda_graph_success() {
        let mut graph = CudaGraphManager::new();
        assert!(graph.capture_pass().is_err());
        assert!(graph.replay_pass().is_err());
        assert!(!graph.is_captured());
        assert_eq!(graph.replay_count(), 0);
    }
    #[test]
    fn validates_probe_tensor_shapes_types_and_identity() {
        let probe: TensorRtProbe = serde_json::from_value(serde_json::json!({
            "ok":true,"device":"CUDA","runtime_version":"10.8.0",
            "tensors":[{"name":"x","shape":[1,-1],"dtype":"float32","input":true},
                       {"name":"y","shape":[1,2],"dtype":"bool","input":false}]
        }))
        .unwrap();
        assert!(valid_probe_tensors(&probe.tensors));
        for shape in [vec![0], vec![-2], vec![1; 9], vec![1048576, 2]] {
            let mut tensors = vec![
                TensorRtTensor {
                    name: "x".into(),
                    shape,
                    dtype: "float32".into(),
                    input: true,
                },
                TensorRtTensor {
                    name: "y".into(),
                    shape: vec![1],
                    dtype: "float32".into(),
                    input: false,
                },
            ];
            assert!(!valid_probe_tensors(&tensors));
            tensors[0].shape = vec![1];
            tensors[1].name = "x".into();
            assert!(!valid_probe_tensors(&tensors));
        }
    }

    #[test]
    fn fake_build_and_attention_never_claim_success_or_write_a_plan() {
        let dir = tempfile::tempdir().unwrap();
        let model = TensorRtModel {
            model_path: dir.path().join("model.plan"),
            device: DeviceKind::Gpu,
            python: "python3".into(),
            script: bridge_script(),
            plan_digest: "0".repeat(64),
            prediction: std::sync::Mutex::new(()),
            metadata: ModelMetadata {
                id: "numeric".into(),
                modality: Modality::Multi,
                quantized: false,
                manifest: ModelManifest::default(),
            },
            config: TensorRtEngineConfig::default(),
        };
        let target = dir.path().join("existing.plan");
        std::fs::write(&target, b"real engine").unwrap();
        assert!(model.build_engine_profile(&target).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"real engine");
        assert!(model.execute_fused_attention(8, 128, true).is_err());
        let params = GenerationParams {
            top_k: Some(10),
            ..GenerationParams::default()
        };
        assert!(
            model
                .infer(
                    ModelInput::Text {
                        prompt: "{}".into()
                    },
                    &params
                )
                .unwrap_err()
                .to_string()
                .contains("default generation")
        );
    }
    #[test]
    fn bounded_worker_round_trip_preserves_numeric_output_and_plan_identity() {
        let python = ["python3", "python"]
            .iter()
            .find(|name| {
                std::process::Command::new(name)
                    .args(["-c", "pass"])
                    .status()
                    .is_ok_and(|s| s.success())
            })
            .expect("TensorRT bridge tests require Python");
        let dir = tempfile::tempdir().unwrap();
        let plan = dir.path().join("model.plan");
        std::fs::write(&plan, b"fixture plan identity").unwrap();
        let script = dir.path().join("fixture.py");
        std::fs::write(
            &script,
            r#"
import sys,json,hashlib
path=sys.argv[sys.argv.index('--model-path')+1]
expected=sys.argv[sys.argv.index('--sha256')+1]
if hashlib.sha256(open(path,'rb').read()).hexdigest()!=expected:
    sys.stderr.write('plan identity changed')
    raise SystemExit(1)
if '--probe' in sys.argv:
    print(json.dumps({'ok':True,'device':'CUDA','runtime_version':'10.8.0','tensors':[
      {'name':'x','shape':[2],'dtype':'float32','input':True},
      {'name':'y','shape':[1],'dtype':'float32','input':False}]}))
else:
    request=json.load(sys.stdin)
    print(json.dumps({'text':None,'logits':[sum(request['inputs']['x']['data'])],
                     'image':None,'audio':None,'video':None}))
"#,
        )
        .unwrap();
        let model = TensorRtModel {
            plan_digest: plan_sha256(&plan).unwrap(),
            model_path: plan.clone(),
            device: DeviceKind::Gpu,
            python: (*python).into(),
            script,
            prediction: std::sync::Mutex::new(()),
            metadata: ModelMetadata {
                id: "numeric".into(),
                modality: Modality::Multi,
                quantized: false,
                manifest: ModelManifest::default(),
            },
            config: TensorRtEngineConfig::default(),
        };
        let probe: TensorRtProbe =
            serde_json::from_slice(&model.call_bridge(&[], true).unwrap()).unwrap();
        assert!(valid_probe_tensors(&probe.tensors));
        let input = ModelInput::Text {
            prompt: serde_json::json!({"inputs":{"x":{"shape":[2],"data":[1.25,2.5]}}}).to_string(),
        };
        assert_eq!(
            model
                .infer(input.clone(), &GenerationParams::default())
                .unwrap()
                .logits
                .unwrap(),
            [3.75]
        );
        {
            let _busy = model.prediction.lock().unwrap();
            assert!(
                model
                    .infer(input.clone(), &GenerationParams::default())
                    .unwrap_err()
                    .to_string()
                    .contains("runtime is busy")
            );
        }
        let mut events = Vec::new();
        model
            .infer_stream(input.clone(), &GenerationParams::default(), &mut |chunk| {
                events.push(chunk);
                Ok(())
            })
            .unwrap();
        assert!(
            matches!(&events[..], [crate::OutputChunk::Embedding(values), crate::OutputChunk::End] if values == &[3.75])
        );
        std::fs::write(&plan, b"replacement plan").unwrap();
        assert!(
            model
                .infer(input, &GenerationParams::default())
                .unwrap_err()
                .to_string()
                .contains("identity changed")
        );
    }

    #[test]
    fn numeric_worker_results_reject_unknown_fields_and_non_numeric_values() {
        assert!(serde_json::from_str::<TensorRtResult>(r#"{"logits":[true]}"#).is_err());
        assert!(
            serde_json::from_str::<TensorRtResult>(r#"{"logits":[1],"unknown":true}"#).is_err()
        );
        assert!(serde_json::from_str::<TensorRtResult>(r#"{"logits":[NaN]}"#).is_err());
    }
}
