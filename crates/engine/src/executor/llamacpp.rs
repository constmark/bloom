use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use bloomai_core::{
    DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat, ResponseFormat,
};
use serde_json::json;

use crate::core::parallelism::ParallelStrategy;
use crate::core::process::ChildGuard;
use crate::core::quantization::QuantMethod;
use crate::engine::BackendMaturity;
use crate::executor::speculative::speculative_mode_is_mtp;
use crate::{
    engine::{Engine, EngineCapability},
    io::{ModelInput, ModelOutput, OutputChunk},
    model::{LoadedModel, ModelMetadata, OutputSink},
};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_CONTEXT_SIZE: usize = 2048;
const DEFAULT_READY_TIMEOUT: Duration = Duration::from_secs(180);
const DEFAULT_COMPLETION_TIMEOUT: Duration = Duration::from_secs(600);
const LLAMA_HELP_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_LLAMA_HELP_BYTES: usize = 64 * 1024;

pub struct LlamaCppEngine;

impl Engine for LlamaCppEngine {
    fn name(&self) -> &'static str {
        "llamacpp"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu]
    }

    fn default_device(&self) -> DeviceKind {
        DeviceKind::Gpu
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "llamacpp",
            supported_families: vec![ModelFamily::Custom("*".to_string())],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::BF16,
                bloomai_core::DType::Q8,
                bloomai_core::DType::Q4,
                bloomai_core::DType::I4,
                bloomai_core::DType::NF4,
            ],
            supported_formats: vec![ModelFormat::Gguf],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
                DeviceClass::DiscreteGpu,
            ],
            supported_modalities: vec![Modality::Text],
            supports_streaming: true,
            supports_quantized_models: true,
            // Operations remain model-specific: trusted manifest task metadata
            // selects a dedicated generation, embedding, or rank process.
            supports_embeddings: true,
            supports_rerank: true,
            supports_structured_output: true,
            max_context_tokens: None,
            supported_quant_methods: vec![QuantMethod::Gguf],
            supported_parallel_strategies: vec![ParallelStrategy::None],
            maturity: BackendMaturity::Beta,
            diagnostic_tips: vec![
                "Set BLOOM_LLAMA_CPP_SERVER to the llama-server binary if it is not on PATH."
                    .to_string(),
                "Use --speculative mtp with a GGUF that contains native MTP/next-n heads."
                    .to_string(),
            ],
            construction_guide:
                "Uses an external llama.cpp llama-server process. Install a recent llama.cpp build \
                 that supports --spec-type draft-mtp for native MTP models."
                    .to_string(),
        }
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let gguf_path = resolve_gguf_path(model_path)?;
        let manifest = crate::manifest_adapter::load_manifest(model_path).unwrap_or_default();
        let binary = resolve_llama_server_binary()?;
        crate::core::security::validate_runner(&binary)?;
        let port = reserve_port()?;
        let host = DEFAULT_HOST.to_string();
        let context_size =
            env_usize("BLOOM_LLAMA_CPP_CONTEXT_SIZE").unwrap_or(DEFAULT_CONTEXT_SIZE);
        let speculative = std::env::var("BLOOM_SPECULATIVE").unwrap_or_else(|_| "none".to_string());
        let spec_type = llama_spec_type(&speculative);
        let spec_n_max = env_usize("BLOOM_NUM_SPECULATIVE_TOKENS")
            .unwrap_or(5)
            .max(1);
        validate_llama_server_support(&binary, spec_type)?;
        let threads = env_usize("BLOOM_LLAMA_CPP_THREADS").or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
        });

        let mut command = Command::new(&binary);
        command
            .arg("-m")
            .arg(&gguf_path)
            .arg("--host")
            .arg(&host)
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(context_size.to_string())
            .arg("--no-webui");

        configure_model_task(&mut command, &manifest, spec_type)?;
        if let Some(threads) = threads {
            command.arg("--threads").arg(threads.to_string());
        }
        match device {
            DeviceKind::Gpu => {
                command.arg("--gpu-layers").arg("-1");
            }
            DeviceKind::Cpu => {
                command.arg("--gpu-layers").arg("0");
            }
            DeviceKind::Npu => bail!("llamacpp backend does not support NPU devices"),
        }
        if spec_type != "none" {
            command.arg("--spec-type").arg(spec_type);
            command
                .arg("--spec-draft-n-max")
                .arg(spec_n_max.to_string());
        }

        let log_server = std::env::var("BLOOM_LLAMA_CPP_LOG")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);
        if log_server {
            command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        } else {
            command.stdout(Stdio::null()).stderr(Stdio::null());
        }

        let mut child = ChildGuard::spawn(&mut command)
            .with_context(|| format!("failed to start llama-server '{}'", binary.display()))?;

        let addr: SocketAddr = format!("{host}:{port}").parse()?;
        let ready_timeout = env_usize("BLOOM_LLAMA_CPP_READY_TIMEOUT_SECS")
            .map(|secs| Duration::from_secs(secs as u64))
            .unwrap_or(DEFAULT_READY_TIMEOUT);
        wait_until_ready(addr, &mut child, ready_timeout)?;

        let model_id = gguf_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("llamacpp-gguf")
            .to_string();
        let metadata = ModelMetadata {
            id: model_id,
            modality: Modality::Text,
            quantized: true,
            manifest,
        };

        Ok(Box::new(LlamaCppModel {
            child: Mutex::new(Some(child)),
            addr,
            metadata,
            spec_type: spec_type.to_string(),
        }))
    }
}

pub struct LlamaCppModel {
    child: Mutex<Option<ChildGuard>>,
    addr: SocketAddr,
    metadata: ModelMetadata,
    spec_type: String,
}

impl LoadedModel for LlamaCppModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let prompt = match input {
            ModelInput::Text { prompt } => prompt,
            _ => bail!("llamacpp backend only supports text input"),
        };
        anyhow::ensure!(
            !self.is_encoder(),
            "llamacpp encoder models do not generate text; use embed_batch or rerank_batch"
        );
        let text = self.complete(&prompt, params)?;
        Ok(ModelOutput {
            text: Some(text),
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
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        params.validate().map_err(anyhow::Error::msg)?;
        let prompt = match input {
            ModelInput::Text { prompt } => prompt,
            _ => bail!("llamacpp backend only supports text input"),
        };
        if self.supports_native_embedding_batch() {
            anyhow::ensure!(
                matches!(params.response_format, None | Some(ResponseFormat::Text)),
                "encoder models do not support response_format"
            );
            let mut embeddings = self.embed_batch(&[prompt])?;
            sink.on_chunk(OutputChunk::Embedding(embeddings.remove(0)))?;
            return sink.on_chunk(OutputChunk::End);
        }
        anyhow::ensure!(
            !self.is_encoder(),
            "llamacpp cross encoders require rerank_batch"
        );
        self.complete_stream(&prompt, params, sink)
    }

    fn tokenize(&self, text: &str) -> Option<Result<Vec<u32>>> {
        Some((|| {
            ModelInput::Text {
                prompt: text.to_string(),
            }
            .validate()
            .map_err(anyhow::Error::msg)?;
            let response = post_json(
                self.addr,
                "/tokenize",
                &json!({
                    "content": text,
                    "add_special": true,
                    "parse_special": true,
                    "with_pieces": false
                }),
                DEFAULT_COMPLETION_TIMEOUT,
            )?;
            let tokens = response
                .get("tokens")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow!("llama-server tokenize response is missing tokens"))?;
            anyhow::ensure!(
                tokens.len() <= crate::MAX_TOKEN_COUNT,
                "llama-server tokenizer exceeded the token limit"
            );
            tokens
                .iter()
                .map(|token| {
                    token
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| anyhow!("llama-server returned an invalid token id"))
                })
                .collect()
        })())
    }

    fn supports_native_embedding_batch(&self) -> bool {
        crate::model_manifest_supports_embeddings(&self.metadata.manifest)
    }

    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            self.supports_native_embedding_batch(),
            "the loaded llama.cpp model is not an embedding model"
        );
        validate_text_batch(inputs)?;
        let response = post_json(
            self.addr,
            "/v1/embeddings",
            &json!({
                "input": inputs,
                "encoding_format": "float"
            }),
            DEFAULT_COMPLETION_TIMEOUT,
        )?;
        parse_embedding_response(&response, inputs.len())
    }

    fn supports_native_rerank(&self) -> bool {
        crate::model_manifest_is_cross_encoder(&self.metadata.manifest)
    }

    fn rerank_batch(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.supports_native_rerank(),
            "the loaded llama.cpp model is not a cross encoder"
        );
        ModelInput::Text {
            prompt: query.to_string(),
        }
        .validate()
        .map_err(anyhow::Error::msg)?;
        validate_text_batch(documents)?;
        let response = post_json(
            self.addr,
            "/reranking",
            &json!({
                "query": query,
                "documents": documents,
                "top_n": documents.len()
            }),
            DEFAULT_COMPLETION_TIMEOUT,
        )?;
        parse_rerank_response(&response, documents.len())
    }
}

impl LlamaCppModel {
    fn is_encoder(&self) -> bool {
        crate::model_manifest_tasks(&self.metadata.manifest) != ["generation"]
    }

    fn complete(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let body = completion_body(prompt, params, false)?;
        let response = post_json(self.addr, "/completion", &body, DEFAULT_COMPLETION_TIMEOUT)?;
        anyhow::ensure!(
            response.get("error").is_none(),
            "llama-server completion returned an error: {response}"
        );
        self.validate_speculative_response(&response)?;
        let text = response
            .get("content")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| {
                anyhow!("llama-server response missing string field 'content': {response}")
            })?;
        validate_structured_completion(&text, params)?;
        Ok(text)
    }

    fn complete_stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let body = completion_body(prompt, params, true)?;
        let mut saw_terminal = false;
        let mut structured_text = String::new();
        let mut saw_speculative_confirmation = self.spec_type == "none";
        post_json_stream(
            self.addr,
            "/completion",
            &body,
            DEFAULT_COMPLETION_TIMEOUT,
            |event: &serde_json::Value| {
                anyhow::ensure!(!saw_terminal, "llama-server emitted data after completion");
                anyhow::ensure!(
                    event.get("error").is_none(),
                    "llama-server stream returned an error: {event}"
                );
                anyhow::ensure!(
                    event.is_object(),
                    "llama-server stream event must be an object"
                );
                let content = event
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        anyhow!("llama-server stream event is missing string content")
                    })?;
                if matches!(
                    params.response_format,
                    Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema(_))
                ) {
                    anyhow::ensure!(
                        structured_text.len().saturating_add(content.len())
                            <= crate::MAX_TEXT_BYTES,
                        "structured output exceeds the text byte limit"
                    );
                    structured_text.push_str(content);
                }
                if !content.is_empty() {
                    sink.on_chunk(OutputChunk::TextDelta(content.to_string()))?;
                }
                if event
                    .pointer("/generation_settings/speculative.types")
                    .is_some()
                {
                    self.validate_speculative_response(event)?;
                    saw_speculative_confirmation = true;
                }
                if let Some(stop) = event.get("stop") {
                    saw_terminal = stop
                        .as_bool()
                        .ok_or_else(|| anyhow!("llama-server stream stop must be boolean"))?;
                }
                Ok(())
            },
        )?;
        anyhow::ensure!(
            saw_terminal,
            "llama-server stream ended before its terminal completion event"
        );

        if !saw_speculative_confirmation {
            bail!(
                "llama-server stream did not confirm speculative mode '{}'",
                self.spec_type
            );
        }
        validate_structured_completion(&structured_text, params)?;
        // The llama-server wire stream has no Bloom-native terminal chunk.
        // Emit one after the process stream has been fully consumed so the
        // pipeline can uphold the same completion contract as native models.
        sink.on_chunk(OutputChunk::End)
    }

    fn validate_speculative_response(&self, response: &serde_json::Value) -> Result<()> {
        if self.spec_type == "none" {
            return Ok(());
        }

        let enabled = response
            .pointer("/generation_settings/speculative.types")
            .map(|value| speculative_types_include(value, &self.spec_type))
            .unwrap_or(false);
        if !enabled {
            bail!(
                "llama-server response did not confirm speculative mode '{}': {}",
                self.spec_type,
                response
            );
        }
        Ok(())
    }
}

fn validate_structured_completion(text: &str, params: &GenerationParams) -> Result<()> {
    match &params.response_format {
        None | Some(ResponseFormat::Text) => Ok(()),
        Some(format) => {
            anyhow::ensure!(
                text.len() <= crate::MAX_TEXT_BYTES,
                "structured output exceeds the text byte limit"
            );
            let value: serde_json::Value = serde_json::from_str(text)
                .context("llama-server returned invalid structured JSON")?;
            anyhow::ensure!(
                value.is_object(),
                "llama-server structured output must be a JSON object"
            );
            if let ResponseFormat::JsonSchema(schema) = format {
                crate::core::structured::validate_complete_json_schema(&value, schema, "$")
                    .map_err(anyhow::Error::msg)?;
            }
            Ok(())
        }
    }
}

// Use the same endpoint contract as the pinned llama.cpp reference. Keeping
// this conversion shared prevents buffered and streamed sampling from drifting.
fn completion_body(
    prompt: &str,
    params: &GenerationParams,
    stream: bool,
) -> Result<serde_json::Value> {
    params.validate().map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        params.seed.is_none_or(|seed| seed <= u32::MAX as u64),
        "llama.cpp seeds must fit in an unsigned 32-bit integer"
    );
    ModelInput::Text {
        prompt: prompt.to_string(),
    }
    .validate()
    .map_err(anyhow::Error::msg)?;
    let mut body = json!({
        "prompt": prompt,
        "n_predict": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stream": stream,
        // Validate the external runtime's uint32 seed instead of wrapping it.
        "seed": params.seed.map(serde_json::Value::from).unwrap_or(json!(-1)),
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
    });
    if let Some(top_k) = params.top_k {
        body["top_k"] = json!(top_k);
    }
    if let Some(logit_bias) = &params.logit_bias {
        body["logit_bias"] = json!(
            logit_bias
                .iter()
                .map(|(token, bias)| json!([token, bias]))
                .collect::<Vec<_>>()
        );
    }
    match &params.response_format {
        None | Some(ResponseFormat::Text) => {}
        Some(ResponseFormat::JsonObject) => body["json_schema"] = json!({"type": "object"}),
        Some(ResponseFormat::JsonSchema(schema)) => body["json_schema"] = schema.clone(),
    }
    Ok(body)
}

fn configure_model_task(
    command: &mut Command,
    manifest: &bloomai_core::ModelManifest,
    spec_type: &str,
) -> Result<()> {
    let cross_encoder = crate::model_manifest_is_cross_encoder(manifest);
    let embedding = crate::model_manifest_supports_embeddings(manifest);
    if !cross_encoder && !embedding {
        return Ok(());
    }
    anyhow::ensure!(
        spec_type == "none",
        "llama.cpp encoder models do not support speculative generation"
    );
    command.arg("--embedding");
    if cross_encoder {
        command.arg("--reranking").arg("--pooling").arg("rank");
    } else {
        let pooling = manifest
            .parameters
            .get("embedding_pooling")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("mean");
        anyhow::ensure!(
            matches!(pooling, "mean" | "cls" | "last"),
            "llama.cpp pooled embeddings require mean, cls, or last pooling"
        );
        command.arg("--pooling").arg(pooling);
    }
    Ok(())
}

const MAX_ENCODER_BATCH_ITEMS: usize = 256;

fn validate_text_batch(inputs: &[String]) -> Result<()> {
    anyhow::ensure!(
        !inputs.is_empty() && inputs.len() <= MAX_ENCODER_BATCH_ITEMS,
        "encoder batches must contain 1..={MAX_ENCODER_BATCH_ITEMS} inputs"
    );
    let mut bytes = 0usize;
    for prompt in inputs {
        ModelInput::Text {
            prompt: prompt.clone(),
        }
        .validate()
        .map_err(anyhow::Error::msg)?;
        bytes = bytes.saturating_add(prompt.len());
    }
    anyhow::ensure!(
        bytes <= crate::MAX_TEXT_BYTES,
        "encoder batch exceeds the text byte limit"
    );
    Ok(())
}

fn response_index(value: &serde_json::Value, count: usize) -> Result<usize> {
    value
        .get("index")
        .and_then(serde_json::Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .filter(|index| *index < count)
        .ok_or_else(|| anyhow!("llama-server returned an invalid result index"))
}

fn finite_f32(value: &serde_json::Value) -> Result<f32> {
    let number = value
        .as_f64()
        .ok_or_else(|| anyhow!("llama-server returned a non-numeric value"))?
        as f32;
    anyhow::ensure!(
        number.is_finite(),
        "llama-server returned a non-finite float32 value"
    );
    Ok(number)
}

fn parse_embedding_response(response: &serde_json::Value, count: usize) -> Result<Vec<Vec<f32>>> {
    let data = response
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("llama-server embedding response is missing data"))?;
    anyhow::ensure!(
        data.len() == count,
        "llama-server returned the wrong embedding count"
    );
    let mut output = vec![None; count];
    let mut dimensions = None;
    let mut values = 0usize;
    for item in data {
        let index = response_index(item, count)?;
        anyhow::ensure!(
            output[index].is_none(),
            "llama-server returned a duplicate embedding index"
        );
        let embedding = item
            .get("embedding")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow!("llama-server returned an invalid embedding"))?;
        anyhow::ensure!(
            !embedding.is_empty()
                && dimensions.is_none_or(|dimensions| dimensions == embedding.len()),
            "llama-server embedding dimensions are empty or inconsistent"
        );
        values = values.saturating_add(embedding.len());
        anyhow::ensure!(
            values <= crate::MAX_TENSOR_VALUES,
            "llama-server embedding batch exceeds the value limit"
        );
        dimensions = Some(embedding.len());
        output[index] = Some(
            embedding
                .iter()
                .map(finite_f32)
                .collect::<Result<Vec<_>>>()?,
        );
    }
    output
        .into_iter()
        .map(|embedding| {
            embedding.ok_or_else(|| anyhow!("llama-server embedding index is missing"))
        })
        .collect()
}

fn parse_rerank_response(response: &serde_json::Value, count: usize) -> Result<Vec<f32>> {
    let data = response
        .get("results")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("llama-server rerank response is missing results"))?;
    anyhow::ensure!(
        data.len() == count,
        "llama-server returned the wrong rerank count"
    );
    let mut output = vec![None; count];
    for item in data {
        let index = response_index(item, count)?;
        anyhow::ensure!(
            output[index].is_none(),
            "llama-server returned a duplicate rerank index"
        );
        output[index] = Some(finite_f32(item.get("relevance_score").ok_or_else(
            || anyhow!("llama-server rerank response is missing relevance_score"),
        )?)?);
    }
    output
        .into_iter()
        .map(|score| score.ok_or_else(|| anyhow!("llama-server rerank index is missing")))
        .collect()
}

impl Drop for LlamaCppModel {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock()
            && let Some(mut child) = child.take()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn resolve_gguf_path(model_path: &Path) -> Result<PathBuf> {
    if model_path.is_file() {
        if model_path.extension().and_then(|v| v.to_str()) == Some("gguf") {
            return Ok(model_path.to_path_buf());
        }
        bail!(
            "llamacpp backend expects a GGUF file or directory, got {}",
            model_path.display()
        );
    }

    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(model_path)
        .with_context(|| format!("failed to read model directory {}", model_path.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|v| v.to_str()) == Some("gguf")
            && path
                .file_name()
                .and_then(|v| v.to_str())
                .map(|name| !name.starts_with("mmproj-"))
                .unwrap_or(true)
        {
            candidates.push(path);
        }
    }
    candidates.sort();
    candidates.into_iter().next().ok_or_else(|| {
        anyhow!(
            "no GGUF model file found in {} for llamacpp backend",
            model_path.display()
        )
    })
}

fn resolve_llama_server_binary() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("BLOOM_LLAMA_CPP_SERVER").map(PathBuf::from) {
        if path.exists() {
            return Ok(path);
        }
        bail!("BLOOM_LLAMA_CPP_SERVER does not exist: {}", path.display());
    }

    let mut candidates = Vec::new();
    if let Some(home) = user_home_dir() {
        candidates.push(home.join(".docker/bin/inference/llama-server"));
    }
    candidates.push(PathBuf::from("llama-server"));
    for candidate in candidates {
        if candidate.is_absolute() {
            if candidate.exists() {
                return Ok(candidate);
            }
        } else if command_exists(&candidate) {
            return Ok(candidate);
        }
    }
    bail!(
        "llama-server not found. Set BLOOM_LLAMA_CPP_SERVER to a recent llama.cpp llama-server binary"
    )
}

fn user_home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
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
}

fn command_exists(binary: &Path) -> bool {
    if crate::core::security::validate_runner(binary).is_err() {
        return false;
    }
    let mut command = Command::new(binary);
    command.arg("--version");
    crate::core::process::run_bounded_output(
        &mut command,
        &[],
        MAX_LLAMA_HELP_BYTES,
        MAX_LLAMA_HELP_BYTES,
        LLAMA_HELP_TIMEOUT,
    )
    .is_ok_and(|output| {
        output.status.success() && !output.stdout_exceeded && !output.stderr_exceeded
    })
}

fn validate_llama_server_support(binary: &Path, spec_type: &str) -> Result<()> {
    crate::core::security::validate_runner(binary)?;
    if spec_type == "none" {
        return Ok(());
    }

    let mut command = Command::new(binary);
    command
        .arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = crate::core::process::run_bounded_output(
        &mut command,
        &[],
        MAX_LLAMA_HELP_BYTES,
        MAX_LLAMA_HELP_BYTES,
        LLAMA_HELP_TIMEOUT,
    )
    .with_context(|| format!("failed to inspect llama-server '{}'", binary.display()))?;
    if output.stdout_exceeded || output.stderr_exceeded {
        bail!(
            "llama-server '{}' emitted more than 64 KiB of --help output",
            binary.display()
        );
    }
    if !output.status.success() {
        bail!(
            "failed to inspect llama-server '{}': --help exited with {}",
            binary.display(),
            output.status
        );
    }
    let help = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !help.contains(spec_type) {
        bail!(
            "llama-server '{}' does not advertise speculative mode '{}'. \
             Install a recent llama.cpp build or set BLOOM_LLAMA_CPP_SERVER to one that supports it.",
            binary.display(),
            spec_type
        );
    }
    Ok(())
}

fn reserve_port() -> Result<u16> {
    let listener = TcpListener::bind((DEFAULT_HOST, 0))?;
    Ok(listener.local_addr()?.port())
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn llama_spec_type(mode: &str) -> &'static str {
    let mode = mode.trim().to_ascii_lowercase();
    if speculative_mode_is_mtp(&mode) {
        "draft-mtp"
    } else {
        match mode.as_str() {
            "ngram" | "n-gram" => "ngram-simple",
            _ => "none",
        }
    }
}

fn wait_until_ready(addr: SocketAddr, child: &mut Child, timeout: Duration) -> Result<()> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            bail!("llama-server exited before becoming ready: {status}");
        }

        match get(addr, "/health", Duration::from_secs(2)) {
            Ok(_) => return Ok(()),
            Err(err) if started.elapsed() < timeout => {
                tracing::debug!("waiting for llama-server readiness: {err}");
                thread::sleep(Duration::from_millis(250));
            }
            Err(err) => {
                return Err(anyhow!(
                    "llama-server did not become ready after {:.1}s: {}",
                    timeout.as_secs_f64(),
                    err
                ));
            }
        }
    }
}

fn get(addr: SocketAddr, path: &str, timeout: Duration) -> Result<serde_json::Value> {
    request(addr, "GET", path, None, timeout)
}

fn post_json(
    addr: SocketAddr,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<serde_json::Value> {
    request(addr, "POST", path, Some(body), timeout)
}

fn post_json_stream<F>(
    addr: SocketAddr,
    path: &str,
    body: &serde_json::Value,
    timeout: Duration,
    on_event: F,
) -> Result<()>
where
    F: FnMut(&serde_json::Value) -> Result<()>,
{
    let body_text = serde_json::to_string(body)?;
    let mut stream = DeadlineStream::connect(addr, timeout)
        .with_context(|| format!("failed to connect to llama-server at {addr}"))?;

    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
        body_text.len(),
        body_text
    );
    stream.write_all(request.as_bytes())?;

    read_streaming_http_json_events(stream, on_event)
}

const MAX_HTTP_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_HTTP_HEADER_BYTES: usize = 16 * 1024;
const MAX_STREAM_EVENT_BYTES: usize = 1024 * 1024;

// A socket idle timeout alone can be reset indefinitely by a noisy or hung
// worker. Share one absolute budget across connection, writes, and all reads.
struct DeadlineStream {
    stream: TcpStream,
    deadline: Instant,
}

impl DeadlineStream {
    fn connect(addr: SocketAddr, timeout: Duration) -> std::io::Result<Self> {
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid llama-server deadline",
            )
        })?;
        let stream = TcpStream::connect_timeout(&addr, timeout)?;
        Ok(Self { stream, deadline })
    }

    fn remaining(&self) -> std::io::Result<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "llama-server request deadline exceeded",
                )
            })
    }
}

impl Read for DeadlineStream {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.stream.set_read_timeout(Some(self.remaining()?))?;
        self.stream.read(bytes)
    }
}

impl Write for DeadlineStream {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.stream.set_write_timeout(Some(self.remaining()?))?;
        self.stream.write(bytes)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.set_write_timeout(Some(self.remaining()?))?;
        self.stream.flush()
    }
}

#[derive(Default)]
struct HttpResponseFraming {
    content_length: Option<usize>,
    chunked: bool,
}

impl HttpResponseFraming {
    fn add_header(&mut self, header: &str) -> Result<()> {
        let (name, value) = header
            .trim_end_matches(['\r', '\n'])
            .split_once(':')
            .ok_or_else(|| anyhow!("malformed llama-server HTTP header"))?;
        if name.eq_ignore_ascii_case("content-length") {
            anyhow::ensure!(
                self.content_length.is_none() && !self.chunked,
                "ambiguous llama-server response framing"
            );
            let value = value.trim();
            anyhow::ensure!(
                !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
                "invalid llama-server content length"
            );
            let length = value.parse::<usize>()?;
            anyhow::ensure!(
                length <= MAX_HTTP_RESPONSE_BYTES,
                "llama-server content length exceeds the byte limit"
            );
            self.content_length = Some(length);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            anyhow::ensure!(
                !self.chunked && self.content_length.is_none(),
                "ambiguous llama-server response framing"
            );
            anyhow::ensure!(
                value.trim().eq_ignore_ascii_case("chunked"),
                "unsupported llama-server transfer encoding"
            );
            self.chunked = true;
        }
        Ok(())
    }
}

fn consume_chunk_trailers(reader: &mut impl BufRead) -> Result<()> {
    let mut bytes = 0usize;
    loop {
        let trailer = bounded_line(reader, MAX_HTTP_HEADER_BYTES.saturating_sub(bytes))?;
        bytes += trailer.len();
        if trailer == "\r\n" || trailer == "\n" {
            return Ok(());
        }
        let (name, _) = trailer
            .split_once(':')
            .ok_or_else(|| anyhow!("malformed llama-server chunk trailer"))?;
        anyhow::ensure!(
            !name.eq_ignore_ascii_case("content-length")
                && !name.eq_ignore_ascii_case("transfer-encoding"),
            "llama-server chunk trailer contains response framing"
        );
    }
}

fn bounded_line(reader: &mut impl BufRead, limit: usize) -> Result<String> {
    let mut line = String::new();
    Read::take(reader, limit as u64 + 1).read_line(&mut line)?;
    anyhow::ensure!(
        line.len() <= limit && line.ends_with('\n'),
        "oversized or truncated llama-server protocol line"
    );
    Ok(line)
}

fn request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let body_text = match body {
        Some(body) => serde_json::to_string(body)?,
        None => String::new(),
    };
    let mut stream = DeadlineStream::connect(addr, timeout)
        .with_context(|| format!("failed to connect to llama-server at {addr}"))?;

    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body_text.len(),
        body_text
    );
    stream.write_all(request.as_bytes())?;

    let mut raw = Vec::new();
    stream
        .take(MAX_HTTP_RESPONSE_BYTES as u64 + 1)
        .read_to_end(&mut raw)?;
    anyhow::ensure!(
        raw.len() <= MAX_HTTP_RESPONSE_BYTES,
        "llama-server response exceeds the byte limit"
    );
    parse_http_json(&raw)
}

fn read_streaming_http_json_events<F>(stream: impl Read, mut on_event: F) -> Result<()>
where
    F: FnMut(&serde_json::Value) -> Result<()>,
{
    let mut reader = BufReader::new(stream);
    let status = bounded_line(&mut reader, MAX_HTTP_HEADER_BYTES)?;
    if status.split_whitespace().nth(1) != Some("200") {
        bail!("llama-server returned {}", status.trim());
    }
    let mut framing = HttpResponseFraming::default();
    let mut header_bytes = status.len();
    loop {
        let header = bounded_line(
            &mut reader,
            MAX_HTTP_HEADER_BYTES.saturating_sub(header_bytes),
        )?;
        header_bytes += header.len();
        if header == "\r\n" || header == "\n" {
            break;
        }
        framing.add_header(&header)?;
    }
    let mut parser = StreamJsonEventParser::default();
    let mut total_bytes = 0usize;
    let mut buffer = [0u8; 8192];
    if framing.chunked {
        loop {
            let size_line = bounded_line(&mut reader, MAX_HTTP_HEADER_BYTES)?;
            let size_hex = size_line.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_hex, 16)?;
            anyhow::ensure!(
                size <= MAX_HTTP_RESPONSE_BYTES.saturating_sub(total_bytes),
                "llama-server stream exceeds the byte limit"
            );
            if size == 0 {
                consume_chunk_trailers(&mut reader)?;
                break;
            }
            total_bytes += size;
            let mut remaining = size;
            while remaining > 0 {
                let length = remaining.min(buffer.len());
                reader.read_exact(&mut buffer[..length])?;
                parser.push_bytes(&buffer[..length], &mut on_event)?;
                remaining -= length;
            }
            let mut crlf = [0u8; 2];
            reader.read_exact(&mut crlf)?;
            anyhow::ensure!(crlf == *b"\r\n", "invalid llama-server chunk terminator");
        }
    } else {
        // Connection-close streams must deliver deltas before EOF too.
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            total_bytes += read;
            anyhow::ensure!(
                total_bytes <= framing.content_length.unwrap_or(MAX_HTTP_RESPONSE_BYTES),
                "llama-server stream exceeds its declared length or byte limit"
            );
            parser.push_bytes(&buffer[..read], &mut on_event)?;
        }
    }
    anyhow::ensure!(
        framing
            .content_length
            .is_none_or(|length| length == total_bytes),
        "truncated llama-server content-length stream"
    );
    parser.finish(&mut on_event)
}

fn parse_http_json(raw: &[u8]) -> Result<serde_json::Value> {
    anyhow::ensure!(
        raw.len() <= MAX_HTTP_RESPONSE_BYTES,
        "llama-server response exceeds the byte limit"
    );
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| anyhow!("invalid HTTP response from llama-server"))?;
    anyhow::ensure!(
        header_end <= MAX_HTTP_HEADER_BYTES,
        "llama-server headers exceed the byte limit"
    );
    let headers = std::str::from_utf8(&raw[..header_end])?;
    let body_bytes = &raw[header_end + 4..];
    let status = headers.lines().next().unwrap_or_default();
    if status.split_whitespace().nth(1) != Some("200") {
        bail!(
            "llama-server returned {status}: {}",
            String::from_utf8_lossy(body_bytes)
        );
    }

    let mut framing = HttpResponseFraming::default();
    for header in headers.lines().skip(1) {
        framing.add_header(header)?;
    }
    let body = if framing.chunked {
        decode_chunked_body(body_bytes)?
    } else {
        anyhow::ensure!(
            framing
                .content_length
                .is_none_or(|length| length == body_bytes.len()),
            "llama-server response does not match its content length"
        );
        body_bytes.to_vec()
    };
    serde_json::from_slice(&body).with_context(|| {
        format!(
            "failed to parse llama-server JSON response: {}",
            String::from_utf8_lossy(&body)
        )
    })
}

fn decode_chunked_body(mut body: &[u8]) -> Result<Vec<u8>> {
    let mut decoded = Vec::new();
    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| anyhow!("invalid chunked response from llama-server"))?;
        let size_hex = std::str::from_utf8(&body[..line_end])?;
        let size_hex = size_hex.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)?;
        if size == 0 {
            let remaining = &body[line_end + 2..];
            let mut trailers = std::io::Cursor::new(remaining);
            consume_chunk_trailers(&mut trailers)?;
            anyhow::ensure!(
                trailers.position() == remaining.len() as u64,
                "llama-server response contains data after chunk termination"
            );
            break;
        }
        let chunk_start = line_end + 2;
        let chunk_end = chunk_start
            .checked_add(size)
            .ok_or_else(|| anyhow!("llama-server chunk size overflow"))?;
        if chunk_end > body.len() || body.len() - chunk_end < 2 {
            bail!("truncated chunked response from llama-server");
        }
        anyhow::ensure!(
            decoded.len().saturating_add(size) <= MAX_HTTP_RESPONSE_BYTES,
            "llama-server response exceeds the byte limit"
        );
        anyhow::ensure!(
            &body[chunk_end..chunk_end + 2] == b"\r\n",
            "invalid llama-server chunk terminator"
        );
        decoded.extend_from_slice(&body[chunk_start..chunk_end]);
        body = &body[chunk_end + 2..];
    }
    Ok(decoded)
}

fn speculative_types_include(value: &serde_json::Value, expected: &str) -> bool {
    match value {
        serde_json::Value::String(types) => types.split(',').any(|ty| ty.trim() == expected),
        serde_json::Value::Array(types) => types
            .iter()
            .filter_map(|value| value.as_str())
            .any(|ty| ty == expected),
        _ => false,
    }
}

#[derive(Default)]
struct StreamJsonEventParser {
    pending: Vec<u8>,
    sse_data_lines: Vec<String>,
    sse_data_bytes: usize,
}

impl StreamJsonEventParser {
    fn push_bytes<F>(&mut self, bytes: &[u8], on_event: &mut F) -> Result<()>
    where
        F: FnMut(&serde_json::Value) -> Result<()>,
    {
        anyhow::ensure!(
            self.pending.len().saturating_add(bytes.len()) <= MAX_STREAM_EVENT_BYTES,
            "llama-server stream line exceeds the byte limit"
        );
        let mut pending = std::mem::take(&mut self.pending);
        pending.extend_from_slice(bytes);
        let mut consumed = 0;
        while let Some(relative_line_end) = pending[consumed..].iter().position(|&b| b == b'\n') {
            let line_end = consumed + relative_line_end;
            let line_result = {
                let mut line = &pending[consumed..line_end];
                if line.ends_with(b"\r") {
                    line = &line[..line.len() - 1];
                }
                self.handle_line(line, on_event)
            };
            consumed = line_end + 1;
            if let Err(err) = line_result {
                pending.drain(..consumed);
                self.pending = pending;
                return Err(err);
            }
        }
        // Defer compaction until all complete lines in this chunk are handled.
        // Draining once avoids repeatedly moving the unconsumed tail for streams
        // containing many short lines.
        pending.drain(..consumed);
        self.pending = pending;
        Ok(())
    }

    fn finish<F>(&mut self, on_event: &mut F) -> Result<()>
    where
        F: FnMut(&serde_json::Value) -> Result<()>,
    {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            self.handle_line(&line, on_event)?;
        }
        self.flush_sse_event(on_event)
    }

    fn handle_line<F>(&mut self, line: &[u8], on_event: &mut F) -> Result<()>
    where
        F: FnMut(&serde_json::Value) -> Result<()>,
    {
        let trimmed = trim_ascii(line);
        if trimmed.is_empty() {
            return self.flush_sse_event(on_event);
        }
        if trimmed.starts_with(b":") {
            return Ok(());
        }
        if let Some(data) = trimmed.strip_prefix(b"data:") {
            let data = trim_ascii(data);
            self.sse_data_bytes = self
                .sse_data_bytes
                .saturating_add(data.len())
                .saturating_add(1);
            anyhow::ensure!(
                self.sse_data_bytes <= MAX_STREAM_EVENT_BYTES,
                "llama-server SSE event exceeds the byte limit"
            );
            self.sse_data_lines
                .push(std::str::from_utf8(data)?.to_string());
            return Ok(());
        }
        if trimmed.starts_with(b"{") {
            let value = serde_json::from_slice(trimmed)?;
            on_event(&value)?;
        }
        Ok(())
    }

    fn flush_sse_event<F>(&mut self, on_event: &mut F) -> Result<()>
    where
        F: FnMut(&serde_json::Value) -> Result<()>,
    {
        if self.sse_data_lines.is_empty() {
            return Ok(());
        }
        let payload = self.sse_data_lines.join("\n");
        self.sse_data_lines.clear();
        self.sse_data_bytes = 0;
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            return Ok(());
        }
        let value: serde_json::Value = serde_json::from_str(payload)?;
        on_event(&value)
    }
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map(|idx| idx + 1)
        .unwrap_or(start);
    &bytes[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_model(addr: SocketAddr, task: &str) -> LlamaCppModel {
        let mut manifest = bloomai_core::ModelManifest::default();
        match task {
            "embedding" => {
                manifest
                    .parameters
                    .insert("bloom_task".into(), json!("embedding"));
            }
            "rerank" => {
                manifest
                    .parameters
                    .insert("bloom_cross_encoder".into(), json!(true));
            }
            _ => {}
        }
        LlamaCppModel {
            child: Mutex::new(None),
            addr,
            spec_type: "none".into(),
            metadata: ModelMetadata {
                id: "fixture".into(),
                modality: Modality::Text,
                quantized: true,
                manifest,
            },
        }
    }

    fn mock_endpoint(
        path: &'static str,
        response: &str,
        sse: bool,
    ) -> (SocketAddr, thread::JoinHandle<serde_json::Value>) {
        let listener = TcpListener::bind((DEFAULT_HOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let response = response.to_string();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut reader = BufReader::new(stream);
            let line = bounded_line(&mut reader, MAX_HTTP_HEADER_BYTES).unwrap();
            assert_eq!(line.trim(), format!("POST {path} HTTP/1.1"));
            let mut length = None;
            loop {
                let line = bounded_line(&mut reader, MAX_HTTP_HEADER_BYTES).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
            }
            let mut body = vec![0; length.unwrap()];
            reader.read_exact(&mut body).unwrap();
            let content_type = if sse {
                "text/event-stream"
            } else {
                "application/json"
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
                response.len()
            );
            reader.get_mut().write_all(header.as_bytes()).unwrap();
            reader.get_mut().write_all(response.as_bytes()).unwrap();
            serde_json::from_slice(&body).unwrap()
        });
        (addr, worker)
    }

    #[test]
    fn completion_controls_are_identical_across_transport_modes() {
        let schema = json!({"type":"object", "properties":{"answer":{"type":"string"}}, "required":["answer"]});
        let params = GenerationParams {
            seed: Some(u32::MAX as u64),
            top_k: Some(12),
            presence_penalty: 0.5,
            frequency_penalty: -1.0,
            logit_bias: Some([(42, 2.5)].into_iter().collect()),
            response_format: Some(ResponseFormat::JsonSchema(schema.clone())),
            ..Default::default()
        };
        let mut buffered = completion_body("你好", &params, false).unwrap();
        let mut streamed = completion_body("你好", &params, true).unwrap();
        assert_eq!(buffered["json_schema"], schema);
        assert_eq!(buffered["seed"], u32::MAX);
        assert_eq!(buffered["logit_bias"], json!([[42, 2.5]]));
        assert_eq!(buffered["top_k"], 12);
        buffered.as_object_mut().unwrap().remove("stream");
        streamed.as_object_mut().unwrap().remove("stream");
        assert_eq!(buffered, streamed);
        assert!(
            completion_body(
                "x",
                &GenerationParams {
                    seed: Some(u64::MAX),
                    ..params.clone()
                },
                false
            )
            .is_err()
        );
        assert!(
            completion_body(
                "x",
                &GenerationParams {
                    max_tokens: 0,
                    ..params
                },
                false
            )
            .is_err()
        );
    }

    #[test]
    fn streams_exactly_one_terminal_and_forwards_json_grammar() {
        let (addr, worker) = mock_endpoint(
            "/completion",
            "data: {\"content\":\"{\\\"ok\\\":true}\",\"stop\":false}\n\ndata: {\"content\":\"\",\"stop\":true}\n\ndata: [DONE]\n\n",
            true,
        );
        let model = fixture_model(addr, "generation");
        let mut text = String::new();
        let mut ends = 0;
        model
            .infer_stream(
                ModelInput::Text {
                    prompt: "你好".into(),
                },
                &GenerationParams {
                    response_format: Some(ResponseFormat::JsonObject),
                    ..Default::default()
                },
                &mut |chunk| {
                    assert_eq!(ends, 0, "output after terminal event");
                    match chunk {
                        OutputChunk::TextDelta(delta) => text.push_str(&delta),
                        OutputChunk::End => ends += 1,
                        _ => panic!("unexpected chunk"),
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(text, "{\"ok\":true}");
        assert_eq!(ends, 1);
        let request = worker.join().unwrap();
        assert_eq!(request["json_schema"], json!({"type":"object"}));
        assert_eq!(request["prompt"], "你好");
        assert_eq!(request["stream"], true);
    }

    #[test]
    fn cancelled_stream_does_not_emit_a_successful_terminal() {
        let (addr, worker) = mock_endpoint(
            "/completion",
            "data: {\"content\":\"first\",\"stop\":false}\n\ndata: {\"content\":\"second\",\"stop\":true}\n\n",
            true,
        );
        let mut chunks = 0;
        let error = fixture_model(addr, "generation")
            .infer_stream(
                ModelInput::Text { prompt: "x".into() },
                &GenerationParams::default(),
                &mut |_| {
                    chunks += 1;
                    bail!("cancelled by caller")
                },
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "cancelled by caller");
        assert_eq!(chunks, 1);
        worker.join().unwrap();
    }

    #[test]
    fn malformed_failed_truncated_and_post_terminal_streams_never_succeed() {
        for body in [
            "data: {\"content\":\"partial\",\"stop\":false}\n\n",
            "data: {\"error\":{\"message\":\"worker failed\"}}\n\n",
            "data: {\"content\":42,\"stop\":true}\n\n",
            "data: {\"content\":\"\",\"stop\":\"true\"}\n\n",
            "data: {\"content\":\"\",\"stop\":true}\n\ndata: {\"content\":\"late\",\"stop\":true}\n\n",
        ] {
            let (addr, worker) = mock_endpoint("/completion", body, true);
            let mut ends = 0;
            assert!(
                fixture_model(addr, "generation")
                    .infer_stream(
                        ModelInput::Text { prompt: "x".into() },
                        &GenerationParams::default(),
                        &mut |chunk| {
                            if matches!(chunk, OutputChunk::End) {
                                ends += 1;
                            }
                            Ok(())
                        }
                    )
                    .is_err(),
                "accepted {body}"
            );
            assert_eq!(ends, 0);
            worker.join().unwrap();
        }
    }

    #[test]
    fn rejects_structured_results_before_a_successful_terminal() {
        let schema = json!({"type":"object", "properties":{"ok":{"type":"boolean"}}, "required":["ok"], "additionalProperties":false});
        let params = GenerationParams {
            response_format: Some(ResponseFormat::JsonSchema(schema)),
            ..Default::default()
        };
        for content in [
            "not JSON",
            "[]",
            "{}",
            "{\"ok\":1}",
            "{\"ok\":true,\"extra\":1}",
        ] {
            let body = format!("data: {}\n\n", json!({"content":content,"stop":true}));
            let (addr, worker) = mock_endpoint("/completion", &body, true);
            let mut ended = false;
            assert!(
                fixture_model(addr, "generation")
                    .infer_stream(
                        ModelInput::Text { prompt: "x".into() },
                        &params,
                        &mut |chunk| {
                            ended |= matches!(chunk, OutputChunk::End);
                            Ok(())
                        }
                    )
                    .is_err()
            );
            assert!(!ended);
            worker.join().unwrap();
        }
        let (addr, worker) = mock_endpoint(
            "/completion",
            &json!({"content":"{\"ok\":true}"}).to_string(),
            false,
        );
        assert_eq!(
            fixture_model(addr, "generation")
                .complete("x", &params)
                .unwrap(),
            "{\"ok\":true}"
        );
        worker.join().unwrap();
    }

    #[test]
    fn external_embedding_batch_restores_order_and_preserves_task_isolation() {
        let (addr, worker) = mock_endpoint(
            "/v1/embeddings",
            &json!({"data":[{"index":1,"embedding":[0.0,1.0]},{"index":0,"embedding":[1.0,0.0]}]})
                .to_string(),
            false,
        );
        let model = fixture_model(addr, "embedding");
        assert!(model.supports_native_embedding_batch());
        assert!(!model.supports_native_rerank());
        assert_eq!(
            model
                .embed_batch(&["first".into(), "second".into()])
                .unwrap(),
            [vec![1.0, 0.0], vec![0.0, 1.0]]
        );
        assert!(
            model
                .infer(
                    ModelInput::Text { prompt: "x".into() },
                    &GenerationParams::default()
                )
                .is_err()
        );
        let request = worker.join().unwrap();
        assert_eq!(request["input"], json!(["first", "second"]));
        assert_eq!(request["encoding_format"], "float");
    }

    #[test]
    fn scalar_embedding_stream_emits_vector_and_one_terminal() {
        let (addr, worker) = mock_endpoint(
            "/v1/embeddings",
            &json!({"data":[{"index":0,"embedding":[0.5,-0.5]}]}).to_string(),
            false,
        );
        let mut chunks = Vec::new();
        fixture_model(addr, "embedding")
            .infer_stream(
                ModelInput::Text { prompt: "x".into() },
                &GenerationParams::default(),
                &mut |chunk| {
                    chunks.push(chunk);
                    Ok(())
                },
            )
            .unwrap();
        assert!(matches!(&chunks[0], OutputChunk::Embedding(values) if values == &[0.5,-0.5]));
        assert!(matches!(&chunks[1], OutputChunk::End));
        assert_eq!(chunks.len(), 2);
        worker.join().unwrap();
    }

    #[test]
    fn cross_encoder_restores_document_order_without_embedding_fallback() {
        let (addr, worker) = mock_endpoint("/reranking", &json!({"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}).to_string(), false);
        let model = fixture_model(addr, "rerank");
        assert!(model.supports_native_rerank());
        assert!(!model.supports_native_embedding_batch());
        assert!(model.embed_batch(&["x".into()]).is_err());
        assert_eq!(
            model
                .rerank_batch("query", &["low".into(), "high".into()])
                .unwrap(),
            [0.1, 0.9]
        );
        let request = worker.join().unwrap();
        assert_eq!(request["query"], "query");
        assert_eq!(request["top_n"], 2);
        assert_eq!(request["documents"], json!(["low", "high"]));
    }

    #[test]
    fn encoder_result_parsers_fail_closed_for_corrupted_batches() {
        for response in [
            json!({"data":[]}),
            json!({"data":[{"index":0,"embedding":[1.0]},{"index":0,"embedding":[1.0]}]}),
            json!({"data":[{"index":2,"embedding":[1.0]},{"index":0,"embedding":[1.0]}]}),
            json!({"data":[{"index":1,"embedding":[1.0]},{"index":0,"embedding":[]}]}),
            json!({"data":[{"index":1,"embedding":[1.0]},{"index":0,"embedding":[1.0,2.0]}]}),
            json!({"data":[{"index":1,"embedding":[1e100]},{"index":0,"embedding":[1.0]}]}),
        ] {
            assert!(
                parse_embedding_response(&response, 2).is_err(),
                "accepted {response}"
            );
        }
        for response in [
            json!({"results":[]}),
            json!({"results":[{"index":0,"relevance_score":1.0},{"index":0,"relevance_score":0.0}]}),
            json!({"results":[{"index":1,"relevance_score":1e100},{"index":0,"relevance_score":1.0}]}),
            json!({"results":[{"index":1,"score":1.0},{"index":0,"relevance_score":1.0}]}),
        ] {
            assert!(
                parse_rerank_response(&response, 2).is_err(),
                "accepted {response}"
            );
        }
        assert!(validate_text_batch(&[]).is_err());
        assert!(validate_text_batch(&vec!["x".into(); MAX_ENCODER_BATCH_ITEMS + 1]).is_err());
    }

    #[test]
    fn trusted_encoder_metadata_configures_exclusive_worker_modes() {
        let model = fixture_model("127.0.0.1:1".parse().unwrap(), "embedding");
        let mut command = Command::new("llama-server");
        configure_model_task(&mut command, &model.metadata.manifest, "none").unwrap();
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            ["--embedding", "--pooling", "mean"]
        );
        assert!(configure_model_task(&mut command, &model.metadata.manifest, "draft-mtp").is_err());
        let model = fixture_model("127.0.0.1:1".parse().unwrap(), "rerank");
        let mut command = Command::new("llama-server");
        configure_model_task(&mut command, &model.metadata.manifest, "none").unwrap();
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            ["--embedding", "--reranking", "--pooling", "rank"]
        );
    }

    #[test]
    fn external_tokenizer_uses_model_special_tokens_and_exact_ids() {
        let (addr, worker) = mock_endpoint(
            "/tokenize",
            &json!({"tokens":[1,42,9000]}).to_string(),
            false,
        );
        assert_eq!(
            fixture_model(addr, "generation")
                .tokenize("你好 Bloom")
                .unwrap()
                .unwrap(),
            [1, 42, 9000]
        );
        let request = worker.join().unwrap();
        assert_eq!(request["content"], "你好 Bloom");
        assert_eq!(request["add_special"], true);
        assert_eq!(request["parse_special"], true);
        for tokens in [json!([-1]), json!([u64::MAX]), json!([1.5]), json!(["1"])] {
            let (addr, worker) =
                mock_endpoint("/tokenize", &json!({"tokens":tokens}).to_string(), false);
            assert!(
                fixture_model(addr, "generation")
                    .tokenize("x")
                    .unwrap()
                    .is_err()
            );
            worker.join().unwrap();
        }
    }

    #[test]
    fn capability_matches_the_external_operations() {
        let capability = LlamaCppEngine.capability();
        assert!(capability.supports_embeddings);
        assert!(capability.supports_rerank);
        assert!(capability.supports_structured_output);
        assert!(capability.supports_streaming);
    }

    #[test]
    fn rejects_overflowing_chunks_bad_terminators_and_oversized_events() {
        let body = format!("{:x}\r\n", usize::MAX);
        assert!(decode_chunked_body(body.as_bytes()).is_err());
        assert!(decode_chunked_body(b"1\r\nxXX0\r\n\r\n").is_err());
        let mut parser = StreamJsonEventParser::default();
        assert!(
            parser
                .push_bytes(&vec![b'x'; MAX_STREAM_EVENT_BYTES + 1], &mut |_| Ok(()))
                .is_err()
        );
        let mut parser = StreamJsonEventParser::default();
        for _ in 0..MAX_STREAM_EVENT_BYTES / 2 {
            if parser.push_bytes(b"data: x\n", &mut |_| Ok(())).is_err() {
                return;
            }
        }
        assert!(parser.push_bytes(b"data: x\n", &mut |_| Ok(())).is_err());
    }

    #[test]
    fn connection_close_stream_delivers_before_the_peer_closes() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        server.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"content\":\"hello\"}\n\n").unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        let mut received = false;
        assert!(
            read_streaming_http_json_events(client, |_| {
                received = true;
                bail!("stop after first event")
            })
            .is_err()
        );
        assert!(received, "event was buffered until connection close");
    }

    #[test]
    fn maps_speculative_modes() {
        assert_eq!(llama_spec_type("mtp"), "draft-mtp");
        assert_eq!(llama_spec_type(" DRAFT-MTP "), "draft-mtp");
        assert_eq!(llama_spec_type("ngram"), "ngram-simple");
        assert_eq!(llama_spec_type("none"), "none");
    }

    #[cfg(unix)]
    #[test]
    fn readiness_timeout_reaps_server_process() {
        let mut command = Command::new("sleep");
        command.arg("30");

        let mut child = ChildGuard::spawn(&mut command).unwrap();
        let pid = child.id() as libc::pid_t;
        let listener = TcpListener::bind((DEFAULT_HOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        assert!(wait_until_ready(addr, &mut child, Duration::ZERO).is_err());
        drop(child);

        // A successful waitpid with WNOHANG would mean the guard left a
        // zombie behind. ChildGuard must reap the process on drop.
        assert_eq!(
            unsafe { libc::waitpid(pid, std::ptr::null_mut(), libc::WNOHANG) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
    }

    #[test]
    fn response_framing_rejects_truncation_and_ambiguous_headers() {
        for raw in [
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n{\"a\":1}",
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n{\"a\":1}",
            "HTTP/1.1 200 OK\r\nContent-Length: 7\r\nContent-Length: 7\r\n\r\n{\"a\":1}",
            "HTTP/1.1 200 OK\r\nContent-Length: 7\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n{\"a\":1}\r\n0\r\n\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 7\r\n\r\n7\r\n{\"a\":1}\r\n0\r\n\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked, gzip\r\n\r\n{\"a\":1}",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n{\"a\":1}\r\n0\r\n",
        ] {
            assert!(parse_http_json(raw.as_bytes()).is_err(), "accepted {raw:?}");
        }
        for raw in [
            "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\ndata: {\"content\":\"\",\"stop\":true}\n\n",
            "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n",
        ] {
            assert!(
                read_streaming_http_json_events(raw.as_bytes(), |_| Ok(())).is_err(),
                "accepted {raw:?}"
            );
        }
        let raw = b"HTTP/1.1 200 OK\r\ntRaNsFeR-EnCoDiNg: CHUNKED\r\n\r\n7\r\n{\"a\":1}\r\n0\r\nX-Trailer: yes\r\n\r\n";
        assert_eq!(parse_http_json(raw).unwrap()["a"], 1);
    }

    #[test]
    fn slow_drip_worker_cannot_reset_the_absolute_request_deadline() {
        let listener = TcpListener::bind((DEFAULT_HOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            loop {
                if bounded_line(&mut reader, MAX_HTTP_HEADER_BYTES).unwrap() == "\r\n" {
                    break;
                }
            }
            reader
                .get_mut()
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 22\r\n\r\n")
                .unwrap();
            for byte in b"{\"content\":\"dripping\"}" {
                if reader.get_mut().write_all(&[*byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(30));
            }
        });
        let started = Instant::now();
        let result = get(addr, "/health", Duration::from_millis(100));
        assert!(result.is_err());
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "slow drip reset the total request deadline"
        );
        worker.join().unwrap();
    }

    #[test]
    fn parses_plain_http_json_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\n\r\n{\"content\":\"hi\"}";
        let parsed = parse_http_json(raw).unwrap();
        assert_eq!(parsed["content"], "hi");
    }

    #[test]
    fn parses_chunked_http_json_response() {
        let raw =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n{\"a\":1}\r\n0\r\n\r\n";
        let parsed = parse_http_json(raw).unwrap();
        assert_eq!(parsed["a"], 1);
    }

    #[test]
    fn parses_chunked_http_json_response_with_extension() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7;foo=bar\r\n{\"a\":1}\r\n0\r\n\r\n";
        let parsed = parse_http_json(raw).unwrap();
        assert_eq!(parsed["a"], 1);
    }

    #[test]
    fn rejects_non_success_http_response() {
        let raw =
            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 15\r\n\r\n{\"error\":\"x\"}";
        assert!(parse_http_json(raw).is_err());
    }

    #[test]
    fn validates_speculative_response() {
        let model = LlamaCppModel {
            child: Mutex::new(None),
            addr: "127.0.0.1:1".parse().unwrap(),
            metadata: ModelMetadata {
                id: "test".to_string(),
                modality: Modality::Text,
                quantized: true,
                manifest: Default::default(),
            },
            spec_type: "draft-mtp".to_string(),
        };

        let response = serde_json::json!({
            "content": "ok",
            "generation_settings": {
                "speculative.types": "none,draft-mtp"
            }
        });
        model.validate_speculative_response(&response).unwrap();

        let response = serde_json::json!({
            "content": "ok",
            "generation_settings": {
                "speculative.types": ["none", "draft-mtp"]
            }
        });
        model.validate_speculative_response(&response).unwrap();

        let response = serde_json::json!({
            "content": "ok",
            "generation_settings": {
                "speculative.types": "none"
            }
        });
        assert!(model.validate_speculative_response(&response).is_err());
    }

    #[test]
    fn parses_sse_stream_events_across_chunks() {
        let mut parser = StreamJsonEventParser::default();
        let mut content = String::new();
        let mut confirmed = false;

        parser
            .push_bytes(
                b"data: {\"content\":\"he\"}\n\ndata: {\"content\":\"ll",
                &mut |event| {
                    if let Some(delta) = event.get("content").and_then(|v| v.as_str()) {
                        content.push_str(delta);
                    }
                    Ok(())
                },
            )
            .unwrap();
        parser
            .push_bytes(
                b"o\",\"generation_settings\":{\"speculative.types\":[\"draft-mtp\"]}}\n\ndata: [DONE]\n\n",
                &mut |event| {
                    if let Some(delta) = event.get("content").and_then(|v| v.as_str()) {
                        content.push_str(delta);
                    }
                    if event
                        .pointer("/generation_settings/speculative.types")
                        .map(|value| speculative_types_include(value, "draft-mtp"))
                        .unwrap_or(false)
                    {
                        confirmed = true;
                    }
                    Ok(())
                },
            )
            .unwrap();
        parser.finish(&mut |_| Ok(())).unwrap();

        assert_eq!(content, "hello");
        assert!(confirmed);
    }

    #[test]
    fn parses_ndjson_stream_events() {
        let mut parser = StreamJsonEventParser::default();
        let mut content = String::new();
        parser
            .push_bytes(
                b"{\"content\":\"a\"}\n{\"content\":\"b\"}\n",
                &mut |event| {
                    content.push_str(event.get("content").and_then(|v| v.as_str()).unwrap());
                    Ok(())
                },
            )
            .unwrap();
        parser.finish(&mut |_| Ok(())).unwrap();

        assert_eq!(content, "ab");
    }

    #[test]
    fn parses_many_short_ndjson_lines() {
        const EVENT_COUNT: usize = 30_000;
        let line = b"{\"content\":\"x\"}\n";
        let mut stream = Vec::with_capacity(EVENT_COUNT * line.len());
        for _ in 0..EVENT_COUNT {
            stream.extend_from_slice(line);
        }

        let mut parser = StreamJsonEventParser::default();
        let mut count = 0;
        let mut content = String::with_capacity(EVENT_COUNT);
        parser
            .push_bytes(&stream, &mut |event| {
                count += 1;
                content.push_str(event["content"].as_str().unwrap());
                Ok(())
            })
            .unwrap();
        parser.finish(&mut |_| Ok(())).unwrap();

        assert_eq!(count, EVENT_COUNT);
        assert_eq!(content, "x".repeat(EVENT_COUNT));
    }

    #[test]
    fn streaming_requires_speculative_confirmation_when_enabled() {
        let model = LlamaCppModel {
            child: Mutex::new(None),
            addr: "127.0.0.1:1".parse().unwrap(),
            metadata: ModelMetadata {
                id: "test".to_string(),
                modality: Modality::Text,
                quantized: true,
                manifest: Default::default(),
            },
            spec_type: "draft-mtp".to_string(),
        };
        assert!(
            model
                .validate_speculative_response(&serde_json::json!({
                    "generation_settings": { "speculative.types": ["draft-mtp"] }
                }))
                .is_ok()
        );
    }
}
