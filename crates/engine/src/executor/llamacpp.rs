use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use bloomai_core::{DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat};
use serde_json::json;

use crate::core::parallelism::ParallelStrategy;
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

        let mut child = command
            .spawn()
            .with_context(|| format!("failed to start llama-server '{}'", binary.display()))?;

        let addr: SocketAddr = format!("{host}:{port}").parse()?;
        let ready_timeout = env_usize("BLOOM_LLAMA_CPP_READY_TIMEOUT_SECS")
            .map(|secs| Duration::from_secs(secs as u64))
            .unwrap_or(DEFAULT_READY_TIMEOUT);
        wait_until_ready(addr, &mut child, ready_timeout)?;

        let manifest = crate::manifest_adapter::load_manifest(model_path).unwrap_or_default();
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
    child: Mutex<Option<Child>>,
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
        let prompt = match input {
            ModelInput::Text { prompt } => prompt,
            _ => bail!("llamacpp backend only supports text input"),
        };
        self.complete_stream(&prompt, params, sink)?;
        sink.on_chunk(OutputChunk::End)?;
        Ok(())
    }
}

impl LlamaCppModel {
    fn complete(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let body = json!({
            "prompt": prompt,
            "n_predict": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": false,
            "seed": params.seed.map(|v| v as i64).unwrap_or(-1),
        });
        let response = post_json(self.addr, "/completion", &body, DEFAULT_COMPLETION_TIMEOUT)?;
        self.validate_speculative_response(&response)?;
        response
            .get("content")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| {
                anyhow!("llama-server response missing string field 'content': {response}")
            })
    }

    fn complete_stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let body = json!({
            "prompt": prompt,
            "n_predict": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": true,
            "seed": params.seed.map(|v| v as i64).unwrap_or(-1),
        });
        let mut saw_speculative_confirmation = self.spec_type == "none";
        post_json_stream(
            self.addr,
            "/completion",
            &body,
            DEFAULT_COMPLETION_TIMEOUT,
            |event: &serde_json::Value| {
                if let Some(content) = event.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() {
                        sink.on_chunk(OutputChunk::TextDelta(content.to_string()))?;
                    }
                }
                if event
                    .pointer("/generation_settings/speculative.types")
                    .is_some()
                {
                    self.validate_speculative_response(event)?;
                    saw_speculative_confirmation = true;
                }
                Ok(())
            },
        )?;

        if !saw_speculative_confirmation {
            bail!(
                "llama-server stream did not confirm speculative mode '{}'",
                self.spec_type
            );
        }
        Ok(())
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

impl Drop for LlamaCppModel {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            if let Some(mut child) = child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
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
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
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

fn command_exists(binary: &Path) -> bool {
    if crate::core::security::validate_runner(binary).is_err() {
        return false;
    }
    Command::new(binary)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn validate_llama_server_support(binary: &Path, spec_type: &str) -> Result<()> {
    crate::core::security::validate_runner(binary)?;
    if spec_type == "none" {
        return Ok(());
    }

    let output = Command::new(binary)
        .arg("--help")
        .output()
        .with_context(|| format!("failed to inspect llama-server '{}'", binary.display()))?;
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
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .with_context(|| format!("failed to connect to llama-server at {addr}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
        body_text.len(),
        body_text
    );
    stream.write_all(request.as_bytes())?;

    read_streaming_http_json_events(stream, on_event)
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
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .with_context(|| format!("failed to connect to llama-server at {addr}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body_text.len(),
        body_text
    );
    stream.write_all(request.as_bytes())?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    parse_http_json(&raw)
}

fn read_streaming_http_json_events<F>(stream: TcpStream, mut on_event: F) -> Result<()>
where
    F: FnMut(&serde_json::Value) -> Result<()>,
{
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.read_line(&mut status)?;
    if !status.contains(" 200 ") {
        let mut body = String::new();
        let _ = reader.read_to_string(&mut body);
        bail!("llama-server returned {}: {}", status.trim(), body);
    }

    let mut transfer_encoding = String::new();
    loop {
        let mut header = String::new();
        let bytes = reader.read_line(&mut header)?;
        if bytes == 0 || header == "\r\n" || header == "\n" {
            break;
        }
        if header
            .to_ascii_lowercase()
            .starts_with("transfer-encoding:")
        {
            transfer_encoding = header;
        }
    }

    let mut parser = StreamJsonEventParser::default();
    if transfer_encoding.to_ascii_lowercase().contains("chunked") {
        loop {
            let mut size_line = String::new();
            if reader.read_line(&mut size_line)? == 0 {
                break;
            }
            let size_hex = size_line.split(';').next().unwrap_or("").trim();
            if size_hex.is_empty() {
                continue;
            }
            let size = usize::from_str_radix(size_hex, 16)?;
            if size == 0 {
                break;
            }
            let mut chunk = vec![0u8; size];
            reader.read_exact(&mut chunk)?;
            let mut crlf = [0u8; 2];
            reader.read_exact(&mut crlf)?;
            parser.push_bytes(&chunk, &mut on_event)?;
        }
    } else {
        let mut body = Vec::new();
        reader.read_to_end(&mut body)?;
        parser.push_bytes(&body, &mut on_event)?;
    }
    parser.finish(&mut on_event)
}

fn parse_http_json(raw: &[u8]) -> Result<serde_json::Value> {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| anyhow!("invalid HTTP response from llama-server"))?;
    let headers = String::from_utf8_lossy(&raw[..header_end]);
    let body_bytes = &raw[header_end + 4..];
    let status = headers.lines().next().unwrap_or_default();
    if !status.contains(" 200 ") {
        bail!(
            "llama-server returned {status}: {}",
            String::from_utf8_lossy(body_bytes)
        );
    }

    let body = if headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        decode_chunked_body(body_bytes)?
    } else {
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
            break;
        }
        let chunk_start = line_end + 2;
        let chunk_end = chunk_start + size;
        if body.len() < chunk_end + 2 {
            bail!("truncated chunked response from llama-server");
        }
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
}

impl StreamJsonEventParser {
    fn push_bytes<F>(&mut self, bytes: &[u8], on_event: &mut F) -> Result<()>
    where
        F: FnMut(&serde_json::Value) -> Result<()>,
    {
        self.pending.extend_from_slice(bytes);
        while let Some(line_end) = self.pending.iter().position(|&b| b == b'\n') {
            let mut line = self.pending.drain(..=line_end).collect::<Vec<_>>();
            if line.ends_with(b"\n") {
                line.pop();
            }
            if line.ends_with(b"\r") {
                line.pop();
            }
            self.handle_line(&line, on_event)?;
        }
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

    #[test]
    fn maps_speculative_modes() {
        assert_eq!(llama_spec_type("mtp"), "draft-mtp");
        assert_eq!(llama_spec_type(" DRAFT-MTP "), "draft-mtp");
        assert_eq!(llama_spec_type("ngram"), "ngram-simple");
        assert_eq!(llama_spec_type("none"), "none");
    }

    #[test]
    fn parses_plain_http_json_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 17\r\n\r\n{\"content\":\"hi\"}";
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
        assert!(model
            .validate_speculative_response(&serde_json::json!({
                "generation_settings": { "speculative.types": ["draft-mtp"] }
            }))
            .is_ok());
    }
}
