//! Bounded numeric Core ML execution through Apple's system Swift framework.
//!
//! The worker source is embedded in Bloom and compiled in a private workspace.
//! No Python packages or third-party native libraries are required. This adapter
//! executes one fixed-shape floating-point tensor, not language generation.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, TryLockError};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use bloomai_core::{
    DType, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat,
    ModelManifest, ResponseFormat,
};

use crate::{
    core::parallelism::ParallelStrategy,
    engine::{Engine, EngineCapability, SupportLevel, default_engine_supports},
    io::{DataBlock, InferenceRequest, ModelInput, ModelOutput, OutputChunk},
    model::{LoadedModel, ModelMetadata, OutputSink},
};

const BRIDGE_SOURCE: &str = include_str!("coreml_bridge.swift");
const MAX_COREML_VALUES: usize = 1_048_576;
const MAX_BRIDGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_DIAGNOSTIC_BYTES: usize = 64 * 1024;
const COMPILE_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_INFERENCE_TIMEOUT: Duration = Duration::from_secs(60);

pub struct CoreMlEngine;

impl CoreMlEngine {
    pub fn discover_model_file(model_path: &Path) -> Result<PathBuf> {
        if is_coreml_artifact(model_path) {
            return Ok(model_path.to_path_buf());
        }
        ensure!(
            model_path.is_dir(),
            "CoreML expects a .mlmodel file or .mlpackage directory: {}",
            model_path.display()
        );
        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(model_path)? {
            let path = entry?.path();
            if is_coreml_artifact(&path) {
                candidates.push(path);
            }
        }
        ensure!(
            candidates.len() == 1,
            "CoreML model discovery requires exactly one .mlmodel/.mlpackage artifact in {}; found {}",
            model_path.display(),
            candidates.len()
        );
        Ok(candidates.remove(0))
    }
}

fn is_coreml_artifact(path: &Path) -> bool {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("mlmodel") => path.is_file(),
        Some(extension) if extension.eq_ignore_ascii_case("mlpackage") => path.is_dir(),
        _ => false,
    }
}

impl Engine for CoreMlEngine {
    fn name(&self) -> &'static str {
        "coreml"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Multi]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        if cfg!(target_os = "macos") {
            vec![DeviceKind::Cpu]
        } else {
            vec![]
        }
    }

    fn default_device(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![ModelFamily::Custom("*".to_string())],
            supported_dtypes: vec![DType::F32, DType::F16],
            supported_formats: vec![ModelFormat::CoreMl],
            supported_devices: self.supported_devices().iter().map(|_| DeviceClass::Cpu).collect(),
            supported_modalities: self.supported_modalities(),
            supports_streaming: false,
            supports_quantized_models: false,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Requires macOS and Apple's Command Line Tools (xcode-select --install).".into(),
                "Accepts one fixed-shape floating-point tensor and one numeric output; no text, image, audio, stateful, or LLM preprocessing is supplied.".into(),
                "Use --device cpu. CoreML's accelerator compute-unit policies permit CPU fallback and do not prove GPU/NPU-only placement.".into(),
            ],
            construction_guide: "Uses embedded Swift source, /usr/bin/xcrun swiftc and Apple's CoreML framework with computeUnits=cpuOnly. No Python or additional Rust dependency is required.".into(),
        }
    }

    fn supports(
        &self,
        manifest: &ModelManifest,
        device: &bloomai_core::DeviceCapability,
    ) -> SupportLevel {
        if !cfg!(target_os = "macos") {
            return SupportLevel::Unsupported(
                "CoreML execution requires macOS and Apple's Swift/CoreML toolchain".into(),
            );
        }
        default_engine_supports(&self.capability(), manifest, device)
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        ensure!(
            cfg!(target_os = "macos"),
            "CoreML execution requires macOS and Apple's Command Line Tools"
        );
        ensure!(
            device == DeviceKind::Cpu,
            "CoreML tensor execution requires --device cpu; cpuAndGPU/cpuAndNeuralEngine permit CPU fallback and cannot prove exclusive GPU/NPU placement"
        );
        let source = Self::discover_model_file(model_path)?.canonicalize()?;
        let timeout = bridge_timeout()?;
        let workspace = tempfile::tempdir().context("failed to create private CoreML workspace")?;
        let swift_source = workspace.path().join("bridge.swift");
        let executable = workspace.path().join("coreml-bridge");
        let compiled_model = workspace.path().join("model.mlmodelc");
        std::fs::write(&swift_source, BRIDGE_SOURCE)?;
        // This is trusted embedded code compiled by an immutable system runner,
        // not an operator-supplied external script or executable override.
        let mut compiler = Command::new("/usr/bin/xcrun");
        compiler
            .env_remove("DEVELOPER_DIR")
            .args(["--sdk", "macosx", "swiftc", "-O"])
            .arg(&swift_source)
            .arg("-o")
            .arg(&executable);
        run_worker(&mut compiler, &[], COMPILE_TIMEOUT, MAX_DIAGNOSTIC_BYTES).context(
            "failed to compile embedded CoreML worker; install Apple's Command Line Tools",
        )?;
        let mut prepare = Command::new(&executable);
        prepare.arg("prepare").arg(&source).arg(&compiled_model);
        let probe: CoreMlProbe = serde_json::from_slice(&run_worker(
            &mut prepare,
            &[],
            timeout,
            MAX_DIAGNOSTIC_BYTES,
        )?)
        .context("CoreML worker returned an invalid model contract")?;
        let (input_elements, output_elements) = probe.validate()?;
        let mut manifest = crate::manifest_adapter::load_manifest(model_path).unwrap_or_default();
        manifest
            .parameters
            .insert("bloom_task".into(), serde_json::json!("tensor"));
        manifest.parameters.insert(
            "coreml_input_shape".into(),
            serde_json::json!(probe.input_shape),
        );
        manifest.parameters.insert(
            "coreml_output_shape".into(),
            serde_json::json!(probe.output_shape),
        );
        manifest.parameters.insert(
            "coreml_input_dtype".into(),
            serde_json::json!(probe.input_dtype),
        );
        let metadata = ModelMetadata {
            id: source
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("coreml-tensor")
                .into(),
            modality: Modality::Multi,
            quantized: false,
            manifest,
        };
        Ok(Box::new(CoreMlModel {
            _workspace: workspace,
            executable,
            compiled_model,
            metadata,
            input_elements,
            output_elements,
            timeout,
            prediction_worker: Mutex::new(()),
        }))
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CoreMlProbe {
    schema_version: u32,
    compute_units: String,
    input_name: String,
    input_shape: Vec<usize>,
    input_dtype: String,
    output_name: String,
    output_kind: String,
    output_shape: Option<Vec<usize>>,
}

impl CoreMlProbe {
    fn validate(&self) -> Result<(usize, usize)> {
        ensure!(
            self.schema_version == 1 && self.compute_units == "cpu_only",
            "CoreML worker contract identity or device policy is invalid"
        );
        for name in [&self.input_name, &self.output_name] {
            ensure!(
                !name.is_empty() && name.len() <= 512 && !name.chars().any(char::is_control),
                "invalid CoreML feature name"
            );
        }
        ensure!(
            matches!(self.input_dtype.as_str(), "float16" | "float32" | "float64"),
            "unsupported CoreML input dtype"
        );
        let input_elements = shape_elements(&self.input_shape)?;
        let output_elements = match (self.output_kind.as_str(), &self.output_shape) {
            ("double", None) => 1,
            ("multiarray", Some(shape)) => shape_elements(shape)?,
            _ => bail!("invalid CoreML output contract"),
        };
        Ok((input_elements, output_elements))
    }
}

fn shape_elements(shape: &[usize]) -> Result<usize> {
    ensure!(
        !shape.is_empty() && shape.len() <= 16,
        "CoreML tensor shape must contain 1..=16 dimensions"
    );
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .filter(|count| *count > 0 && *count <= MAX_COREML_VALUES)
            .ok_or_else(|| anyhow!("CoreML tensor shape is empty, overflowing, or oversized"))
    })
}

struct CoreMlModel {
    _workspace: tempfile::TempDir,
    executable: PathBuf,
    compiled_model: PathBuf,
    metadata: ModelMetadata,
    input_elements: usize,
    output_elements: usize,
    timeout: Duration,
    prediction_worker: Mutex<()>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericResponse {
    schema_version: u32,
    compute_units: String,
    values: Vec<f32>,
}

fn validate_generation_params(params: &GenerationParams) -> Result<()> {
    params.validate().map_err(anyhow::Error::msg)?;
    let defaults = GenerationParams::default();
    ensure!(
        params.max_tokens == defaults.max_tokens
            && params.temperature == defaults.temperature
            && params.top_p == defaults.top_p
            && params.top_k.is_none()
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0
            && params
                .logit_bias
                .as_ref()
                .is_none_or(|bias| bias.is_empty())
            && params.seed.is_none()
            && matches!(params.response_format, None | Some(ResponseFormat::Text)),
        "CoreML tensor inference does not support generation controls; use default GenerationParams"
    );
    Ok(())
}

impl CoreMlModel {
    fn predict(&self, values: &[f32]) -> Result<ModelOutput> {
        ensure!(
            values.len() == self.input_elements && values.len() <= MAX_COREML_VALUES,
            "CoreML input tensor requires exactly {} values",
            self.input_elements
        );
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "CoreML input tensor contains non-finite values"
        );
        // A worker loads the complete model. One active worker per loaded model
        // keeps concurrent callers from duplicating the admitted weight budget.
        // Reject pressure immediately instead of waiting beyond cancellation.
        let _worker = self
            .prediction_worker
            .try_lock()
            .map_err(|error| match error {
                TryLockError::WouldBlock => anyhow!(
                    "CoreML prediction worker is busy; only one prediction may run per loaded model"
                ),
                TryLockError::Poisoned(_) => {
                    anyhow!("CoreML prediction worker admission lock is poisoned")
                }
            })?;
        let request = serde_json::to_vec(&serde_json::json!({"values": values}))?;
        ensure!(
            request.len() <= MAX_BRIDGE_BYTES,
            "CoreML tensor JSON exceeds the byte limit"
        );
        let mut command = Command::new(&self.executable);
        command.arg("infer").arg(&self.compiled_model);
        let response: NumericResponse = serde_json::from_slice(&run_worker(
            &mut command,
            &request,
            self.timeout,
            MAX_BRIDGE_BYTES,
        )?)
        .context("CoreML worker returned malformed numeric output")?;
        ensure!(
            response.schema_version == 1 && response.compute_units == "cpu_only",
            "CoreML output contract identity or device policy changed"
        );
        ensure!(
            response.values.len() == self.output_elements
                && response.values.iter().all(|value| value.is_finite()),
            "CoreML output dimensions or finite-value contract failed"
        );
        let output = ModelOutput {
            logprobs: None,
            text: None,
            logits: Some(response.values),
            image: None,
            audio: None,
            video: None,
        };
        output.validate().map_err(anyhow::Error::msg)?;
        Ok(output)
    }

    fn emit(&self, values: &[f32], sink: &mut dyn OutputSink) -> Result<()> {
        let output = self.predict(values)?;
        // This is the shared numeric-vector event; it does not declare a
        // sentence-embedding operation or semantic embedding quality.
        sink.on_chunk(OutputChunk::Embedding(
            output
                .logits
                .ok_or_else(|| anyhow!("CoreML numeric output missing"))?,
        ))?;
        sink.on_chunk(OutputChunk::End)
    }
}

impl LoadedModel for CoreMlModel {
    fn actual_device(&self) -> Option<DeviceKind> {
        Some(DeviceKind::Cpu)
    }
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        anyhow::ensure!(
            params.logprobs.is_none(),
            "this backend does not support token logprobs"
        );
        validate_generation_params(params)?;
        let ModelInput::Text { prompt } = input else {
            bail!("CoreML tensor inference accepts a JSON number array or one DataBlock::Tensor");
        };
        ModelInput::Text {
            prompt: prompt.clone(),
        }
        .validate()
        .map_err(anyhow::Error::msg)?;
        let values: Vec<f32> = serde_json::from_str(&prompt).context(
            "CoreML text input must be a JSON number array, not a natural-language prompt",
        )?;
        self.predict(&values)
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        anyhow::ensure!(
            params.logprobs.is_none(),
            "this backend does not support token logprobs"
        );
        let output = self.infer(input, params)?;
        sink.on_chunk(OutputChunk::Embedding(
            output
                .logits
                .ok_or_else(|| anyhow!("CoreML numeric output missing"))?,
        ))?;
        sink.on_chunk(OutputChunk::End)
    }

    fn infer_request(&self, request: InferenceRequest, sink: &mut dyn OutputSink) -> Result<()> {
        anyhow::ensure!(
            request.params.logprobs.is_none(),
            "this backend does not support token logprobs"
        );
        request.validate().map_err(anyhow::Error::msg)?;
        let params = GenerationParams {
            logprobs: request.params.logprobs,
            max_tokens: request.params.max_tokens,
            temperature: request.params.temperature,
            top_p: request.params.top_p,
            top_k: request.params.top_k,
            presence_penalty: request.params.presence_penalty,
            frequency_penalty: request.params.frequency_penalty,
            logit_bias: request.params.logit_bias,
            seed: request.params.seed,
            response_format: request.params.response_format,
        };
        validate_generation_params(&params)?;
        let [DataBlock::Tensor(values)] = request.blocks.as_slice() else {
            bail!("CoreML inference requests require exactly one Tensor block");
        };
        self.emit(values, sink)
    }
}

fn bridge_timeout() -> Result<Duration> {
    match std::env::var("BLOOM_COREML_TIMEOUT_SECS") {
        Ok(value) => {
            let seconds: u64 = value
                .parse()
                .context("BLOOM_COREML_TIMEOUT_SECS must be an integer")?;
            ensure!(
                (1..=3600).contains(&seconds),
                "BLOOM_COREML_TIMEOUT_SECS must be between 1 and 3600"
            );
            Ok(Duration::from_secs(seconds))
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_INFERENCE_TIMEOUT),
        Err(error) => Err(error.into()),
    }
}

fn run_worker(
    command: &mut Command,
    request: &[u8],
    timeout: Duration,
    stdout_limit: usize,
) -> Result<Vec<u8>> {
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = crate::core::process::run_bounded_output(
        command,
        request,
        stdout_limit,
        MAX_DIAGNOSTIC_BYTES,
        timeout,
    )?;
    ensure!(
        !output.stdout_exceeded && !output.stderr_exceeded,
        "CoreML worker exceeded its bounded output budget"
    );
    ensure!(
        output.status.success(),
        "CoreML worker failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(output.stdout)
}

#[cfg(test)]
mod tests;
