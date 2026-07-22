//! OpenAI-compatible API client for the Bloom server, plus SSE streaming.

use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

/// A single chat message (OpenAI wire format).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

/// Server health snapshot from GET /health.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Health {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub in_flight_requests: u64,
    #[serde(default)]
    pub requests_total: u64,
}

/// One model entry from GET /v1/models.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelList {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    #[serde(default)]
    pub id: String,
}

/// Connection settings, persisted to localStorage.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnConfig {
    pub base_url: String,
    pub api_key: String,
}

impl Default for ConnConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            api_key: String::new(),
        }
    }
}

/// Default base URL: same origin the page was served from.
pub fn default_base_url() -> String {
    web_sys::window()
        .and_then(|w| w.location().origin().ok())
        .unwrap_or_else(|| "http://127.0.0.1:8080".to_string())
}

fn auth_headers(cfg: &ConnConfig, headers: &web_sys::Headers) {
    headers.set("Content-Type", "application/json").ok();
    if !cfg.api_key.is_empty() {
        headers
            .set("Authorization", &format!("Bearer {}", cfg.api_key))
            .ok();
    }
}

/// GET /health — server liveness + model id.
pub async fn fetch_health(cfg: &ConnConfig) -> Result<Health, String> {
    let url = format!("{}/health", cfg.base_url.trim_end_matches('/'));
    let headers = web_sys::Headers::new().map_err(|e| format!("{e:?}"))?;
    auth_headers(cfg, &headers);
    let opts = web_sys::RequestInit::new();
    opts.set_method("GET");
    opts.set_headers(&headers);
    let resp = http_request(&url, &opts).await?;
    let text = resp_text(&resp).await?;
    serde_json::from_str(&text).map_err(|e| format!("invalid health json: {e}"))
}

/// GET /v1/models — first model id (Bloom serves a single active model).
pub async fn fetch_first_model(cfg: &ConnConfig) -> Result<String, String> {
    let url = format!("{}/v1/models", cfg.base_url.trim_end_matches('/'));
    let headers = web_sys::Headers::new().map_err(|e| format!("{e:?}"))?;
    auth_headers(cfg, &headers);
    let opts = web_sys::RequestInit::new();
    opts.set_method("GET");
    opts.set_headers(&headers);
    let resp = http_request(&url, &opts).await?;
    let text = resp_text(&resp).await?;
    let list: ModelList =
        serde_json::from_str(&text).map_err(|e| format!("invalid models json: {e}"))?;
    Ok(list
        .data
        .first()
        .map(|m| m.id.clone())
        .unwrap_or_else(|| "loading...".into()))
}

/// Streaming chat completion. Invokes `on_token` for each decoded text delta.
/// Returns the fully accumulated assistant text.
pub async fn chat_stream<F>(
    cfg: &ConnConfig,
    messages: &[ChatMessage],
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    mut on_token: F,
) -> Result<String, String>
where
    F: FnMut(String),
{
    let url = format!("{}/v1/chat/completions", cfg.base_url.trim_end_matches('/'));
    let body = ChatRequest {
        messages,
        stream: true,
        max_tokens,
        temperature,
    };
    let body_text = serde_json::to_string(&body).map_err(|e| e.to_string())?;

    let headers = web_sys::Headers::new().map_err(|e| format!("{e:?}"))?;
    auth_headers(cfg, &headers);
    let opts = web_sys::RequestInit::new();
    opts.set_method("POST");
    opts.set_headers(&headers);
    opts.set_body(&wasm_bindgen::JsValue::from_str(&body_text));

    let resp = http_request(&url, &opts).await?;
    if !resp.ok() {
        let status = resp.status();
        let text = resp_text(&resp).await.unwrap_or_default();
        return Err(format!("HTTP {status}: {text}"));
    }

    let raw_body = resp.body().ok_or("response has no body")?;
    let reader = raw_body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| "failed to get stream reader")?;

    let mut buffer = String::new();
    let mut accumulated = String::new();
    let decoder = web_sys::TextDecoder::new().map_err(|e| format!("{e:?}"))?;

    loop {
        let chunk = JsFuture::from(reader.read())
            .await
            .map_err(|e| format!("{e:?}"))?;
        let done = js_sys::Reflect::get(&chunk, &"done".into())
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if done {
            break;
        }
        let value = js_sys::Reflect::get(&chunk, &"value".into()).map_err(|e| format!("{e:?}"))?;
        if value.is_undefined() {
            continue;
        }
        let arr = js_sys::Uint8Array::new(&value);
        let text = decoder.decode_with_js_u8_array(&arr).unwrap_or_default();
        buffer.push_str(&text);

        // SSE frames are separated by a blank line; each data line is `data: <json>`.
        while let Some(pos) = buffer.find("\n\n") {
            let frame = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();
            for line in frame.lines() {
                let line = line.trim();
                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        return Ok(accumulated);
                    }
                    if let Some(delta) = parse_chunk_delta(data) {
                        accumulated.push_str(&delta);
                        on_token(delta);
                    }
                }
            }
        }
    }
    Ok(accumulated)
}

/// Extract `choices[0].delta.content` from a chat.completion.chunk JSON string.
fn parse_chunk_delta(data: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(data).ok()?;
    v.get("choices")?
        .get(0)?
        .get("delta")?
        .get("content")?
        .as_str()
        .map(|s| s.to_string())
}

async fn http_request(url: &str, opts: &web_sys::RequestInit) -> Result<web_sys::Response, String> {
    let window = web_sys::window().ok_or("no window")?;
    let request =
        web_sys::Request::new_with_str_and_init(url, opts).map_err(|e| format!("{e:?}"))?;
    let promise = window.fetch_with_request(&request);
    let resp = JsFuture::from(promise)
        .await
        .map_err(|e| format!("{e:?}"))?;
    resp.dyn_into::<web_sys::Response>()
        .map_err(|_| "not a Response".to_string())
}

async fn resp_text(resp: &web_sys::Response) -> Result<String, String> {
    let promise = resp.text().map_err(|e| format!("{e:?}"))?;
    let v = JsFuture::from(promise)
        .await
        .map_err(|e| format!("{e:?}"))?;
    v.as_string().ok_or_else(|| "not text".to_string())
}
