//! OpenAI-compatible API client for the Bloom server, plus SSE streaming.

use std::collections::HashSet;
use std::fmt;

use futures_util::FutureExt;
use gloo_timers::future::TimeoutFuture;
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::browser;

pub const MAX_RESPONSE_JSON_SCHEMA_BYTES: usize = 64 * 1024;
pub const MAX_OBSERVABILITY_RESPONSE_BYTES: usize = 256 * 1024;
pub const MAX_MODEL_DOWNLOAD_SOURCE_RESPONSE_BYTES: usize = 16 * 1024;
pub const MAX_MODEL_INDEX_RESPONSE_BYTES: usize = 512 * 1024;
pub const MAX_CHAT_INPUT_CHARS: usize = 262_144;
pub const MAX_EMBEDDING_INPUTS: usize = 256;
pub const MAX_EMBEDDING_INPUT_CHARS: usize = 262_144;
pub const MAX_EMBEDDING_CONTENT_BYTES: usize = 768 * 1024;
pub const MAX_EMBEDDING_DIMENSIONS: usize = 16_384;
pub const MAX_RERANK_DOCUMENTS: usize = 256;
pub const MAX_RERANK_QUERY_CHARS: usize = 65_536;
pub const MAX_RERANK_DOCUMENT_CHARS: usize = 262_144;
pub const MAX_RERANK_CONTENT_BYTES: usize = 768 * 1024;
pub const EMBEDDING_EXPORT_FILENAME: &str = "bloom-embeddings.json";
pub const RERANK_EXPORT_FILENAME: &str = "bloom-rerank.json";
pub const MAX_SYSTEM_PROMPT_CHARS: usize = 65_536;
pub const MAX_STOP_SEQUENCES: usize = 4;
pub const MAX_STOP_SEQUENCE_CHARS: usize = 1_024;
pub const MAX_STOP_SEQUENCES_BYTES: usize = 16 * 1_024;
pub const MAX_BASE_URL_CHARS: usize = 2_048;
pub const MAX_API_KEY_CHARS: usize = 4_096;
pub const MAX_IMAGE_ATTACHMENT_BYTES: u64 = 10 * 1024 * 1024;
pub const SPEECH_SAMPLE_RATE: u32 = 16_000;
pub const MAX_SPEECH_SEGMENT_SAMPLES: usize = SPEECH_SAMPLE_RATE as usize * 3;
const MAX_HTTP_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_HTTP_ERROR_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_HTTP_ERROR_MESSAGE_BYTES: usize = 4 * 1024;
const HTTP_REQUEST_ID_HEADER: &str = "x-request-id";
const MAX_HTTP_REQUEST_ID_CHARS: usize = 128;
const HTTP_RETRY_AFTER_HEADER: &str = "retry-after";
const MAX_HTTP_RETRY_AFTER_SECONDS: u32 = 300;
const MAX_READINESS_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_AUTH_PROBE_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_EMBEDDING_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const MAX_EMBEDDING_VALUES: usize = 1_048_576;
const MAX_RERANK_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_EMBEDDING_EXPORT_BYTES: usize = 40 * 1024 * 1024;
const MAX_RERANK_EXPORT_BYTES: usize = 4 * 1024 * 1024;
const MAX_ENCODER_CLIPBOARD_BYTES: usize = 1024 * 1024;
const READINESS_SCHEMA_VERSION: u32 = 3;
const READINESS_PROTOCOL_VERSION: u32 = 3;
const READINESS_OBJECT: &str = "bloom.readiness";
const MAX_READINESS_SERVER_VERSION_CHARS: usize = 64;
const MAX_READINESS_MODEL_CHARS: usize = 256;
const MAX_READINESS_ERROR_CHARS: usize = 512;
const MAX_READINESS_MODALITIES: usize = 16;
const MAX_READINESS_MODALITY_CHARS: usize = 64;
const MAX_READINESS_MODEL_TASKS: usize = 3;
const MAX_MODEL_IMPORT_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_MODEL_PREFLIGHT_RESPONSE_BYTES: usize = 256 * 1024;
const MODEL_PREFLIGHT_SCHEMA_VERSION: u32 = 1;
const MODEL_PREFLIGHT_OBJECT: &str = "bloom.model_preflight";
const MAX_MODEL_PREFLIGHT_ID_CHARS: usize = 256;
const MAX_MODEL_PREFLIGHT_METADATA_CHARS: usize = 4_096;
const MAX_MODEL_PREFLIGHT_LIST_ITEMS: usize = 32;
const MAX_MODEL_PREFLIGHT_LIST_ITEM_CHARS: usize = 2_048;
const MAX_MODEL_CATALOG_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MODEL_CATALOG_SCHEMA_VERSION: u32 = 1;
const MODEL_CATALOG_OBJECT: &str = "bloom.model_catalog";
const MAX_MODEL_CATALOG_ENTRIES: usize = 4_096;
const MAX_MODEL_CATALOG_STAGED_ENTRIES: usize = 1_000;
const MAX_MODEL_CATALOG_ID_CHARS: usize = 255;
const MAX_MODEL_CATALOG_PATH_CHARS: usize = 8_192;
const MAX_MODEL_CATALOG_TEXT_CHARS: usize = 4_096;
const MAX_MODEL_CATALOG_LICENSES: usize = 64;
const MAX_MODEL_RESTORE_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_HTTP_REQUEST_WAIT_MS: u32 = 120_000;
const MAX_HTTP_RESPONSE_IDLE_WAIT_MS: u32 = 30_000;
const MAX_HTTP_RESPONSE_TOTAL_WAIT_MS: u32 = 300_000;
const MAX_HTTP_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_CHAT_REQUEST_MESSAGES: usize = 2_048;
const MAX_CHAT_CONTENT_BYTES: usize = 768 * 1024;
const MAX_CHAT_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_MULTIMODAL_PROMPT_BYTES: usize = 1024 * 1024;
const MAX_SPEECH_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_ATTACHMENT_NAME_CHARS: usize = 255;
const MAX_MODEL_IMPORT_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const _: () = {
    assert!(MAX_HTTP_REQUEST_WAIT_MS > 0);
    assert!(MAX_HTTP_RESPONSE_IDLE_WAIT_MS > 0);
    assert!(MAX_HTTP_RESPONSE_IDLE_WAIT_MS < MAX_HTTP_RESPONSE_TOTAL_WAIT_MS);
    assert!(MAX_CHAT_CONTENT_BYTES < MAX_CHAT_REQUEST_BYTES);
    assert!(MAX_CHAT_REQUEST_BYTES <= MAX_HTTP_REQUEST_BODY_BYTES);
};
const MAX_MODEL_DOWNLOAD_SOURCE_URL_BYTES: usize = 2_048;
const MAX_MODEL_DOWNLOAD_SOURCE_WARNING_CHARS: usize = 512;
const MAX_MODEL_INDEX_ENTRIES: usize = 200;
const MAX_MODEL_PACKAGE_FILES: usize = 256;
const MAX_STREAM_MODEL_ID_CHARS: usize = 256;
const MAX_STREAM_REQUEST_ID_CHARS: usize = 128;
const MAX_SSE_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;
const MAX_STREAM_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_STREAM_ERROR_MESSAGE_BYTES: usize = 4 * 1024;
const MAX_SSE_SEPARATOR_BYTES: usize = 4;
const MAX_RESPONSE_JSON_SCHEMA_DEPTH: usize = 16;
const MAX_RESPONSE_JSON_SCHEMA_NODES: usize = 1_024;
const MAX_RESPONSE_JSON_SCHEMA_PROPERTIES: usize = 256;
const MAX_RESPONSE_JSON_SCHEMA_ENUM_VALUES: usize = 256;
const MAX_RESPONSE_JSON_SCHEMA_ANNOTATION_CHARS: usize = 1_024;

/// A single chat message in OpenAI wire format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// In-memory image selected for one multimodal request.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageAttachment {
    pub name: String,
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// Generation settings exposed by the UI and persisted locally.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ChatOptions {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    pub system_prompt: String,
    pub response_format: ResponseFormatMode,
    pub json_schema: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatMode {
    #[default]
    Text,
    JsonObject,
    JsonSchema,
}

impl ResponseFormatMode {
    pub fn form_value(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::JsonObject => "json_object",
            Self::JsonSchema => "json_schema",
        }
    }

    pub fn from_form_value(value: &str) -> Option<Self> {
        match value {
            "text" => Some(Self::Text),
            "json_object" => Some(Self::JsonObject),
            "json_schema" => Some(Self::JsonSchema),
            _ => None,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::JsonObject => "JSON object",
            Self::JsonSchema => "JSON Schema",
        }
    }
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            seed: None,
            stop_sequences: Vec::new(),
            system_prompt: String::new(),
            response_format: ResponseFormatMode::Text,
            json_schema: String::new(),
        }
    }
}

impl ChatOptions {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=32_768).contains(&self.max_tokens) {
            return Err("Maximum generated tokens must be between 1 and 32768.".into());
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err("Temperature must be between 0 and 2.".into());
        }
        if !self.top_p.is_finite() || !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err("Top P must be greater than 0 and at most 1.".into());
        }
        validate_stop_sequences(&self.stop_sequences)?;
        if self.system_prompt.chars().count() > MAX_SYSTEM_PROMPT_CHARS {
            return Err(format!(
                "System prompt cannot exceed {MAX_SYSTEM_PROMPT_CHARS} characters."
            ));
        }
        if self.response_format == ResponseFormatMode::JsonSchema {
            parse_supported_response_schema(&self.json_schema)?;
        }
        Ok(())
    }

    fn response_format_payload(&self) -> Result<Option<serde_json::Value>, String> {
        match self.response_format {
            ResponseFormatMode::Text => Ok(None),
            ResponseFormatMode::JsonObject => Ok(Some(serde_json::json!({
                "type": "json_object"
            }))),
            ResponseFormatMode::JsonSchema => {
                let schema = parse_supported_response_schema(&self.json_schema)?;
                Ok(Some(serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bloom_response",
                        "strict": true,
                        "schema": schema
                    }
                })))
            }
        }
    }
}

fn validate_stop_sequences(sequences: &[String]) -> Result<(), String> {
    if sequences.len() > MAX_STOP_SEQUENCES {
        return Err(format!(
            "Stop sequences cannot contain more than {MAX_STOP_SEQUENCES} entries."
        ));
    }
    let mut total_bytes = 0_usize;
    for (index, sequence) in sequences.iter().enumerate() {
        let characters = sequence.chars().count();
        if characters == 0 {
            return Err(format!("Stop sequence {} cannot be empty.", index + 1));
        }
        if characters > MAX_STOP_SEQUENCE_CHARS {
            return Err(format!(
                "Stop sequence {} cannot exceed {MAX_STOP_SEQUENCE_CHARS} characters.",
                index + 1
            ));
        }
        total_bytes = total_bytes
            .checked_add(sequence.len())
            .ok_or_else(|| "Combined stop sequence size overflowed.".to_string())?;
        if total_bytes > MAX_STOP_SEQUENCES_BYTES {
            return Err(format!(
                "Combined stop sequences cannot exceed {MAX_STOP_SEQUENCES_BYTES} bytes."
            ));
        }
    }
    Ok(())
}

pub fn parse_stop_sequences_setting(text: &str) -> Result<Vec<String>, String> {
    let sequences = if text.trim().is_empty() {
        Vec::new()
    } else {
        serde_json::from_str::<Vec<String>>(text)
            .map_err(|error| format!("Stop sequences must be a JSON array of strings: {error}"))?
    };
    validate_stop_sequences(&sequences)?;
    Ok(sequences)
}

fn parse_supported_response_schema(input: &str) -> Result<serde_json::Value, String> {
    let input = input.trim();
    if input.is_empty() || input.len() > MAX_RESPONSE_JSON_SCHEMA_BYTES {
        return Err(format!(
            "JSON Schema must be between 1 and {MAX_RESPONSE_JSON_SCHEMA_BYTES} bytes."
        ));
    }
    let schema = serde_json::from_str::<serde_json::Value>(input)
        .map_err(|error| format!("JSON Schema is not valid JSON: {error}"))?;
    if schema.get("type").and_then(serde_json::Value::as_str) != Some("object") {
        return Err("JSON Schema root type must be object.".to_string());
    }
    let mut nodes = 0_usize;
    validate_response_schema_node(&schema, "$", 0, &mut nodes)?;
    Ok(schema)
}

fn validate_response_schema_node(
    schema: &serde_json::Value,
    path: &str,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), String> {
    if depth > MAX_RESPONSE_JSON_SCHEMA_DEPTH {
        return Err(format!(
            "JSON Schema exceeds the maximum depth of {MAX_RESPONSE_JSON_SCHEMA_DEPTH}."
        ));
    }
    *nodes = nodes
        .checked_add(1)
        .ok_or_else(|| "JSON Schema node count overflowed.".to_string())?;
    if *nodes > MAX_RESPONSE_JSON_SCHEMA_NODES {
        return Err(format!(
            "JSON Schema contains more than {MAX_RESPONSE_JSON_SCHEMA_NODES} schema nodes."
        ));
    }
    let object = schema
        .as_object()
        .ok_or_else(|| format!("JSON Schema at {path} must be an object."))?;
    for field in object.keys() {
        if !matches!(
            field.as_str(),
            "$schema"
                | "title"
                | "description"
                | "type"
                | "enum"
                | "required"
                | "properties"
                | "additionalProperties"
                | "items"
        ) {
            return Err(format!(
                "JSON Schema at {path} uses unsupported keyword {field:?}."
            ));
        }
    }
    for field in ["$schema", "title", "description"] {
        if object
            .get(field)
            .is_some_and(|value| !valid_response_schema_annotation(value))
        {
            return Err(format!(
                "JSON Schema {path}.{field} must be a bounded string."
            ));
        }
    }
    let schema_type = object
        .get("type")
        .map(|value| {
            value
                .as_str()
                .filter(|schema_type| {
                    matches!(
                        *schema_type,
                        "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
                    )
                })
                .ok_or_else(|| format!("JSON Schema at {path} has an unsupported type."))
        })
        .transpose()?;
    if let Some(enum_values) = object.get("enum") {
        let enum_values = enum_values
            .as_array()
            .ok_or_else(|| format!("JSON Schema at {path}.enum must be an array."))?;
        if enum_values.is_empty() || enum_values.len() > MAX_RESPONSE_JSON_SCHEMA_ENUM_VALUES {
            return Err(format!(
                "JSON Schema at {path}.enum must contain 1 to {MAX_RESPONSE_JSON_SCHEMA_ENUM_VALUES} values."
            ));
        }
    }
    let properties = object
        .get("properties")
        .map(|value| {
            value
                .as_object()
                .ok_or_else(|| format!("JSON Schema at {path}.properties must be an object."))
        })
        .transpose()?;
    if let Some(properties) = properties {
        if schema_type != Some("object") {
            return Err(format!(
                "JSON Schema at {path} can use properties only with type object."
            ));
        }
        if properties.len() > MAX_RESPONSE_JSON_SCHEMA_PROPERTIES {
            return Err(format!(
                "JSON Schema at {path} contains more than {MAX_RESPONSE_JSON_SCHEMA_PROPERTIES} properties."
            ));
        }
        for (field, property_schema) in properties {
            if field.is_empty()
                || field.chars().count() > 128
                || field.chars().any(char::is_control)
            {
                return Err(format!(
                    "JSON Schema at {path} contains an invalid property name."
                ));
            }
            validate_response_schema_node(
                property_schema,
                &format!("{path}.{field}"),
                depth + 1,
                nodes,
            )?;
        }
    }
    if let Some(required) = object.get("required") {
        if schema_type != Some("object") {
            return Err(format!(
                "JSON Schema at {path} can use required only with type object."
            ));
        }
        let required = required
            .as_array()
            .ok_or_else(|| format!("JSON Schema at {path}.required must be an array."))?;
        if required.len() > MAX_RESPONSE_JSON_SCHEMA_PROPERTIES {
            return Err(format!(
                "JSON Schema at {path}.required contains too many fields."
            ));
        }
        let mut names = HashSet::with_capacity(required.len());
        for field in required {
            let field = field
                .as_str()
                .ok_or_else(|| format!("JSON Schema at {path}.required must contain strings."))?;
            if !names.insert(field)
                || properties.is_none_or(|properties| !properties.contains_key(field))
            {
                return Err(format!(
                    "JSON Schema at {path}.required contains a duplicate or unknown property {field:?}."
                ));
            }
        }
    }
    if let Some(additional) = object.get("additionalProperties") {
        if schema_type != Some("object") || !additional.is_boolean() {
            return Err(format!(
                "JSON Schema at {path}.additionalProperties requires type object and a boolean value."
            ));
        }
    }
    if let Some(items) = object.get("items") {
        if schema_type != Some("array") {
            return Err(format!(
                "JSON Schema at {path} can use items only with type array."
            ));
        }
        validate_response_schema_node(items, &format!("{path}[]"), depth + 1, nodes)?;
    }
    Ok(())
}

fn valid_response_schema_annotation(value: &serde_json::Value) -> bool {
    value
        .as_str()
        .is_some_and(|text| text.chars().count() <= MAX_RESPONSE_JSON_SCHEMA_ANNOTATION_CHARS)
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    stream_options: ChatStreamOptions,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    stop: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ChatStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct SpeechInferenceRequest<'a> {
    blocks: [SpeechDataBlock<'a>; 1],
    params: SpeechInferenceParams,
}

#[derive(Debug, Serialize)]
enum SpeechDataBlock<'a> {
    AudioPcm {
        samples: &'a [f32],
        sample_rate: u32,
    },
}

#[derive(Debug, Serialize)]
struct SpeechInferenceParams {
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    seed: Option<u64>,
}

/// Server readiness snapshot from `GET /ready`.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct Readiness {
    pub schema_version: u32,
    pub object: String,
    pub protocol_version: u32,
    pub minimum_ui_protocol_version: u32,
    pub maximum_ui_protocol_version: u32,
    pub server_version: String,
    pub status: String,
    pub progress: u8,
    pub model: String,
    pub in_flight_requests: u64,
    pub available_permits: u64,
    pub memory_pressure_high: bool,
    pub ram_utilization: f64,
    pub loading: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub load_error: Option<String>,
    pub input_modalities: Vec<String>,
    pub model_tasks: Vec<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub context_window: Option<u64>,
}

/// One validated L2-normalized vector returned by `/v1/embeddings`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVector {
    pub index: usize,
    pub input: String,
    pub values: Vec<f32>,
}

/// A bounded embedding result safe for the browser to summarize.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingBatch {
    pub model: String,
    pub vectors: Vec<EmbeddingVector>,
    pub prompt_tokens: u64,
}

/// One validated reranking result with its exact submitted document.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
    pub document: String,
}

/// A bounded rerank response safe for browser rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankBatch {
    pub id: String,
    pub model: String,
    pub query: String,
    pub results: Vec<RerankResult>,
    pub prompt_tokens: u64,
}

#[derive(Serialize)]
struct EmbeddingExport<'a> {
    schema_version: u32,
    object: &'static str,
    model: &'a str,
    prompt_tokens: u64,
    vectors: Vec<EmbeddingExportVector<'a>>,
}

#[derive(Serialize)]
struct EmbeddingExportVector<'a> {
    index: usize,
    input: &'a str,
    embedding: &'a [f32],
}

#[derive(Serialize)]
struct RerankExport<'a> {
    schema_version: u32,
    object: &'static str,
    id: &'a str,
    model: &'a str,
    query: &'a str,
    prompt_tokens: u64,
    results: &'a [RerankResult],
}

#[derive(Debug, Deserialize)]
struct EmbeddingWireResponse {
    object: String,
    data: Vec<EmbeddingWireItem>,
    model: String,
    usage: EmbeddingWireUsage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingWireItem {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct EmbeddingWireUsage {
    prompt_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct RerankWireResponse {
    id: String,
    object: String,
    model: String,
    results: Vec<RerankWireItem>,
    usage: EmbeddingWireUsage,
}

#[derive(Debug, Deserialize)]
struct RerankWireItem {
    index: usize,
    relevance_score: f64,
    document: Option<RerankWireDocument>,
}

#[derive(Debug, Deserialize)]
struct RerankWireDocument {
    text: String,
}

impl Readiness {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != READINESS_SCHEMA_VERSION || self.object != READINESS_OBJECT {
            return Err(format!(
                "unsupported readiness contract: expected {READINESS_OBJECT} schema {READINESS_SCHEMA_VERSION}"
            ));
        }
        if self.minimum_ui_protocol_version == 0
            || self.maximum_ui_protocol_version == 0
            || self.minimum_ui_protocol_version > self.maximum_ui_protocol_version
            || !(self.minimum_ui_protocol_version..=self.maximum_ui_protocol_version)
                .contains(&self.protocol_version)
        {
            return Err("readiness protocol compatibility range is invalid".to_string());
        }
        if !(self.minimum_ui_protocol_version..=self.maximum_ui_protocol_version)
            .contains(&READINESS_PROTOCOL_VERSION)
        {
            return Err(format!(
                "unsupported readiness protocol: this UI implements protocol {READINESS_PROTOCOL_VERSION}, but the server supports UI protocols {} through {}",
                self.minimum_ui_protocol_version, self.maximum_ui_protocol_version
            ));
        }
        if self.server_version.is_empty()
            || self.server_version.trim() != self.server_version
            || self.server_version.chars().count() > MAX_READINESS_SERVER_VERSION_CHARS
            || self
                .server_version
                .chars()
                .any(|character| character.is_control() || character.is_whitespace())
        {
            return Err("readiness server version is missing or invalid".to_string());
        }
        if !matches!(self.status.as_str(), "ready" | "not_ready") {
            return Err("readiness status is invalid".to_string());
        }
        if self.progress > 100 {
            return Err("readiness progress is invalid".to_string());
        }
        if self.model.is_empty()
            || self.model.trim() != self.model
            || self.model.chars().count() > MAX_READINESS_MODEL_CHARS
            || self.model.chars().any(char::is_control)
        {
            return Err("readiness model identity is missing or invalid".to_string());
        }
        if self.load_error.as_ref().is_some_and(|message| {
            message.is_empty()
                || message.chars().count() > MAX_READINESS_ERROR_CHARS
                || message.chars().any(char::is_control)
        }) {
            return Err("readiness load error is invalid".to_string());
        }
        let mut unique_modalities = HashSet::new();
        if self.input_modalities.len() > MAX_READINESS_MODALITIES
            || self.input_modalities.iter().any(|modality| {
                modality.is_empty()
                    || modality.chars().count() > MAX_READINESS_MODALITY_CHARS
                    || modality.chars().any(char::is_control)
                    || !unique_modalities.insert(modality)
            })
        {
            return Err("readiness input modalities are invalid".to_string());
        }
        let mut unique_tasks = HashSet::new();
        if self.model_tasks.len() > MAX_READINESS_MODEL_TASKS
            || self.model_tasks.iter().any(|task| {
                !matches!(task.as_str(), "generation" | "embedding" | "rerank")
                    || !unique_tasks.insert(task.as_str())
            })
            || (unique_tasks.contains("rerank") && !unique_tasks.contains("embedding"))
        {
            return Err("readiness model tasks are invalid".to_string());
        }
        if self.context_window == Some(0) {
            return Err("readiness context window must be positive when present".to_string());
        }
        if !self.ram_utilization.is_finite() || !(0.0..=1.0).contains(&self.ram_utilization) {
            return Err("readiness RAM utilization is invalid".to_string());
        }
        if self.status == "ready"
            && (self.model == "not loaded"
                || self.progress != 100
                || self.loading
                || self.load_error.is_some()
                || self.model_tasks.is_empty()
                || self.context_window.is_none()
                || self.available_permits == 0
                || self.memory_pressure_high)
        {
            return Err("readiness ready state is internally inconsistent".to_string());
        }
        Ok(())
    }
}

/// Separates an unavailable endpoint from a reachable but incompatible server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadinessError {
    Authentication(String),
    Unavailable(String),
    Incompatible(String),
}

impl ReadinessError {
    pub fn is_authentication(&self) -> bool {
        matches!(self, Self::Authentication(_))
    }

    pub fn is_incompatible(&self) -> bool {
        matches!(self, Self::Incompatible(_))
    }
}

impl fmt::Display for ReadinessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Authentication(message)
            | Self::Unavailable(message)
            | Self::Incompatible(message) => formatter.write_str(message),
        }
    }
}

/// Versioned, credential-free runtime snapshot from `GET /v1/observability`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilitySnapshot {
    pub schema_version: u32,
    pub object: String,
    pub created: u64,
    pub server: ObservabilityServer,
    pub model: String,
    pub ready: bool,
    pub load: ObservabilityLoad,
    pub speculative_mode: String,
    pub requests: ObservabilityRequests,
    pub tokens: ObservabilityTokens,
    pub scheduler: ObservabilityScheduler,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub startup_memory_estimate: Option<ObservabilityMemoryEstimate>,
    pub kv_cache: ObservabilityKvCache,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub cachemesh: Option<ObservabilityCacheMesh>,
    pub memory: ObservabilityMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityServer {
    pub version: String,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityLoad {
    pub phase: String,
    pub progress: u8,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub requested_model: Option<String>,
    pub failure_present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityRequests {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
    pub in_flight: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityTokens {
    pub prompt_total: u64,
    pub generated_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityScheduler {
    pub ifb_enabled: bool,
    pub prefill_queue: u64,
    pub decoding_queue: u64,
    pub active_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityMemoryEstimate {
    pub weight_bytes: u64,
    pub host_weight_bytes: u64,
    pub device_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub kv_cache_bytes_per_token: u64,
    pub temp_tensor_bytes: u64,
    pub total_bytes: u64,
    pub weight_dtype: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub quantization: Option<serde_json::Value>,
    pub kv_cache_dtype: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub num_layers: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub offloaded_layers: Option<u64>,
    pub mmap_residency_applied: bool,
    pub memory_scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityKvCache {
    pub total_blocks: u64,
    pub free_blocks: u64,
    pub active_blocks: u64,
    pub cached_blocks: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub reuses: u64,
    pub utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityCacheMesh {
    pub enabled: bool,
    pub l1: ObservabilityCacheTier,
    pub l2: ObservabilityCacheTier,
    pub l3: ObservabilityCacheTier,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityCacheTier {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub offloads: u64,
    pub restores: u64,
    pub failed_offloads: u64,
    pub dropped: u64,
    pub bytes: u64,
    pub items: u64,
    pub hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityMemory {
    pub total_vram: u64,
    pub used_vram: u64,
    pub total_ram: u64,
    pub used_ram: u64,
    pub peak_vram: u64,
    pub peak_ram: u64,
    pub device_name: String,
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

impl ObservabilitySnapshot {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.object != "bloom.observability_snapshot" {
            return Err("unsupported observability snapshot version".to_string());
        }
        validate_diagnostic_label("server version", &self.server.version, 64, false)?;
        validate_diagnostic_label("model", &self.model, 256, false)?;
        validate_diagnostic_label("speculative mode", &self.speculative_mode, 64, false)?;
        if !matches!(
            self.load.phase.as_str(),
            "idle" | "loading" | "ready" | "failed"
        ) {
            return Err("observability snapshot contains an invalid load phase".to_string());
        }
        if self.load.progress > 100 {
            return Err("observability snapshot contains invalid load progress".to_string());
        }
        if self.load.failure_present != (self.load.phase == "failed") {
            return Err("observability snapshot contains inconsistent failure state".to_string());
        }
        if let Some(requested_model) = &self.load.requested_model {
            validate_diagnostic_label("requested model", requested_model, 256, false)?;
        }
        if self.kv_cache.free_blocks > self.kv_cache.total_blocks
            || !self.kv_cache.utilization.is_finite()
            || !(0.0..=1.0).contains(&self.kv_cache.utilization)
        {
            return Err("observability snapshot contains invalid KV cache metrics".to_string());
        }
        if let Some(cachemesh) = &self.cachemesh {
            for tier in [&cachemesh.l1, &cachemesh.l2, &cachemesh.l3] {
                if !tier.hit_rate.is_finite() || !(0.0..=1.0).contains(&tier.hit_rate) {
                    return Err(
                        "observability snapshot contains invalid CacheMesh metrics".to_string()
                    );
                }
            }
        }
        if !self.memory.device_name.is_empty() {
            validate_diagnostic_label("device name", &self.memory.device_name, 256, true)?;
        }
        if let Some(estimate) = &self.startup_memory_estimate {
            validate_diagnostic_label("weight dtype", &estimate.weight_dtype, 64, false)?;
            validate_diagnostic_label("KV cache dtype", &estimate.kv_cache_dtype, 64, false)?;
            validate_diagnostic_label("memory scope", &estimate.memory_scope, 512, true)?;
        }
        Ok(())
    }

    pub fn to_pretty_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map(|json| format!("{json}\n"))
            .map_err(|error| format!("failed to encode diagnostics snapshot: {error}"))
    }
}

fn validate_diagnostic_label(
    label: &str,
    value: &str,
    max_chars: usize,
    allow_empty: bool,
) -> Result<(), String> {
    if (!allow_empty && value.trim().is_empty())
        || value.chars().count() > max_chars
        || value.chars().any(char::is_control)
    {
        return Err(format!(
            "observability snapshot contains an invalid {label}"
        ));
    }
    Ok(())
}

/// A model discovered under the server's configured catalog root.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelCatalogEntry {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub format: String,
    pub size_bytes: u64,
    pub size_complete: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub modified_at: Option<u64>,
    pub active: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub provenance: Option<ModelProvenance>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub provenance_error: Option<String>,
}

/// Verified acquisition metadata recorded beside an installed model.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelProvenance {
    pub acquisition: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub model_index_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub source_url: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub source_host: Option<String>,
    pub sha256: String,
    #[serde(default)]
    pub file_count: Option<usize>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub license: Option<String>,
    pub installed_at: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub last_verified_at: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub integrity_mismatch_at: Option<u64>,
}

pub const MODEL_INVENTORY_FILENAME: &str = "bloom-model-inventory.json";
pub const MAX_MODEL_INVENTORY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MODEL_INVENTORY_DRIFT_ENTRIES: usize = 200;
const MODEL_INVENTORY_SCHEMA_VERSION: u8 = 2;
const MIN_MODEL_INVENTORY_SCHEMA_VERSION: u8 = 1;
const MODEL_INVENTORY_RECONCILIATION_SCHEMA_VERSION: u8 = 1;
const MODEL_INVENTORY_OBJECT: &str = "bloom.model_inventory";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInventory {
    pub schema_version: u8,
    pub object: String,
    pub summary: ModelInventorySummary,
    pub models: Vec<ModelInventoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInventorySummary {
    pub model_count: usize,
    pub provenance_count: usize,
    pub source_locked_count: usize,
    pub quarantined_count: usize,
    pub invalid_provenance_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInventoryEntry {
    pub id: String,
    pub kind: String,
    pub format: String,
    pub size_bytes: u64,
    pub size_complete: bool,
    pub provenance_status: String,
    pub acquisition: Option<String>,
    #[serde(default)]
    pub model_index_id: Option<String>,
    pub source: Option<ModelInventorySource>,
    pub sha256: Option<String>,
    pub license: Option<String>,
    pub installed_at: Option<u64>,
    pub last_verified_at: Option<u64>,
    pub integrity: String,
    pub source_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInventorySource {
    pub url: Option<String>,
    pub host: Option<String>,
    pub revision: Option<String>,
    pub immutable_revision: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ModelInventoryReconciliation {
    pub schema_version: u8,
    pub object: String,
    pub in_sync: bool,
    pub truncated: bool,
    pub summary: ModelInventoryReconciliationSummary,
    pub drift: Vec<ModelInventoryDrift>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ModelInventoryReconciliationSummary {
    pub expected_model_count: usize,
    pub current_model_count: usize,
    pub matching_count: usize,
    pub missing_count: usize,
    pub unexpected_count: usize,
    pub changed_count: usize,
    pub blocking_count: usize,
    pub restorable_count: usize,
    pub drift_count: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ModelInventoryDrift {
    pub id: String,
    pub status: String,
    pub severity: String,
    pub changes: Vec<String>,
    pub restore_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInventoryComparison {
    pub expected: ModelInventory,
    pub report: ModelInventoryReconciliation,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActiveModel {
    pub id: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub catalog_id: Option<String>,
    pub source: String,
    pub input_modalities: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelLoadStatus {
    pub phase: String,
    pub progress: u8,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub requested_model: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelDownloadStatus {
    pub phase: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub filename: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub source_host: Option<String>,
    pub downloaded_bytes: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub total_bytes: Option<u64>,
    pub resumable: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelDownloadCapability {
    pub enabled: bool,
    pub license_policy: ModelLicensePolicy,
    pub status: ModelDownloadStatus,
    pub staged: Vec<StagedModelDownload>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelIndexCapability {
    pub enabled: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub key_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub trust_id: Option<String>,
    pub trusted_key_count: usize,
    pub refresh_seconds: u64,
    pub persistent_rollback_protection: bool,
}

/// A publisher-signed, server-verified model discovery snapshot.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelIndexSnapshot {
    pub schema_version: u8,
    pub object: String,
    pub key_id: String,
    pub name: String,
    pub generated_at: u64,
    pub expires_at: u64,
    pub source_kind: String,
    pub cache_status: String,
    pub warning: Option<String>,
    pub data: Vec<ModelIndexEntry>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelIndexEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub download_url: Option<String>,
    pub filename: String,
    pub format: String,
    pub size_bytes: u64,
    pub sha256: String,
    #[serde(default)]
    pub files: Vec<ModelIndexFile>,
    pub license: String,
    pub family: Option<String>,
    pub parameter_count: Option<u64>,
    pub quantization: Option<String>,
    pub tags: Vec<String>,
    pub downloadable: bool,
    pub blocking_reasons: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelIndexFile {
    pub download_url: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelLicensePolicy {
    pub enforced: bool,
    pub allowed: Vec<String>,
}

/// Trusted metadata discovered for one public Hugging Face model file.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelDownloadSource {
    pub object: String,
    pub download_url: String,
    pub filename: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub commit_hash: Option<String>,
    pub verification_ready: bool,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StagedModelDownload {
    pub filename: String,
    pub source_host: String,
    pub downloaded_bytes: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub modified_at: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelImportStatus {
    pub phase: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub filename: Option<String>,
    pub uploaded_bytes: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub total_bytes: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StagedModelImport {
    pub filename: String,
    pub uploaded_bytes: u64,
    pub total_bytes: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub modified_at: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelImportCapability {
    pub enabled: bool,
    pub max_bytes: u64,
    pub max_chunk_bytes: usize,
    pub license_policy: ModelLicensePolicy,
    pub status: ModelImportStatus,
    pub staged: Vec<StagedModelImport>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelStorageStatus {
    pub quota_enabled: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub max_bytes: Option<u64>,
    pub used_bytes: u64,
    pub committed_bytes: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub available_bytes: Option<u64>,
    pub installed_bytes: u64,
    pub staged_download_bytes: u64,
    pub staged_import_bytes: u64,
    pub reserved_bytes: u64,
    pub staged_retention_seconds: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub last_cleanup_at: Option<u64>,
    pub last_cleanup_removed_sessions: u64,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelIntegrityStatus {
    pub phase: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub model_id: Option<String>,
    pub checked_bytes: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub total_bytes: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub expected_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub actual_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub matches_expected: Option<bool>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub verified_at: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub error: Option<String>,
}

/// Bounded manifest metadata returned by a model preflight inspection.
#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelManifestSummary {
    pub id: String,
    pub family: String,
    pub version: String,
    pub model_tasks: Vec<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub description: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub license: Option<String>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub formats: Vec<String>,
    pub primary_dtype: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub quantization: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub quantization_bits: Option<u8>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub parameter_count: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub context_length: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub num_layers: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub hidden_size: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub vocab_size: Option<u64>,
    pub supports_mmap: bool,
    pub requires_streaming: bool,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelRuntimeCompatibility {
    pub configured_engine: String,
    pub selected_engine: String,
    pub engine_maturity: String,
    pub device: String,
    pub device_backend: String,
    pub backend_available: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub backend_reason: Option<String>,
    pub support: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub support_reason: Option<String>,
    pub diagnostic_tips: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelMemoryPreflight {
    pub per_request_context_tokens: u64,
    pub max_concurrent: u64,
    pub planned_context_tokens: u64,
    pub weight_bytes: u64,
    pub host_weight_bytes: u64,
    pub device_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub kv_cache_bytes_per_token: u64,
    pub temp_tensor_bytes: u64,
    pub total_bytes: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub available_bytes: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub budget_bytes: Option<u64>,
    pub reserve_bytes: u64,
    pub memory_utilization: f64,
    pub preallocation_enabled: bool,
    pub fits_budget: bool,
    pub scope: String,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelPreflightReport {
    pub model_id: String,
    pub inspected_at: u64,
    pub loadable: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub load_blocker: Option<String>,
    pub manifest: ModelManifestSummary,
    pub runtime: ModelRuntimeCompatibility,
    pub memory: ModelMemoryPreflight,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelPreflightResponse {
    schema_version: u32,
    object: String,
    data: ModelPreflightReport,
}

impl ModelPreflightResponse {
    fn validate(self, requested_model_id: &str) -> Result<ModelPreflightReport, String> {
        if self.schema_version != MODEL_PREFLIGHT_SCHEMA_VERSION {
            return Err(format!(
                "model preflight response uses unsupported schema version {}; expected {}",
                self.schema_version, MODEL_PREFLIGHT_SCHEMA_VERSION
            ));
        }
        if self.object != MODEL_PREFLIGHT_OBJECT {
            return Err("model preflight response has an invalid object type".to_string());
        }
        self.data.validate(requested_model_id)?;
        Ok(self.data)
    }
}

impl ModelPreflightReport {
    fn validate(&self, requested_model_id: &str) -> Result<(), String> {
        validate_preflight_text(
            "requested model ID",
            requested_model_id,
            1,
            MAX_MODEL_PREFLIGHT_ID_CHARS,
        )?;
        validate_preflight_text("model ID", &self.model_id, 1, MAX_MODEL_PREFLIGHT_ID_CHARS)?;
        if self.model_id != requested_model_id {
            return Err("model preflight response does not match the requested model".to_string());
        }
        if self.inspected_at == 0 {
            return Err("model preflight response has an invalid inspection time".to_string());
        }
        validate_preflight_text(
            "manifest ID",
            &self.manifest.id,
            1,
            MAX_MODEL_PREFLIGHT_ID_CHARS,
        )?;
        validate_preflight_text("model family", &self.manifest.family, 1, 256)?;
        validate_preflight_text("model version", &self.manifest.version, 1, 256)?;
        validate_model_tasks(&self.manifest.model_tasks)?;
        validate_optional_preflight_text(
            "model description",
            self.manifest.description.as_deref(),
        )?;
        validate_optional_preflight_text("model license", self.manifest.license.as_deref())?;
        validate_preflight_list("input modalities", &self.manifest.input_modalities, 16, 64)?;
        if self.manifest.input_modalities.is_empty()
            || self
                .manifest
                .input_modalities
                .iter()
                .any(|value| !matches!(value.as_str(), "text" | "audio" | "vision" | "multi"))
        {
            return Err("model preflight response has invalid input modalities".to_string());
        }
        validate_preflight_list(
            "output modalities",
            &self.manifest.output_modalities,
            16,
            64,
        )?;
        if self.manifest.output_modalities.is_empty()
            || self
                .manifest
                .output_modalities
                .iter()
                .any(|value| !matches!(value.as_str(), "text" | "audio" | "vision" | "multi"))
        {
            return Err("model preflight response has invalid output modalities".to_string());
        }
        validate_preflight_list("model formats", &self.manifest.formats, 32, 64)?;
        if self.manifest.formats.iter().any(|value| {
            !matches!(
                value.as_str(),
                "gguf"
                    | "safetensors"
                    | "openvino_ir"
                    | "tensorrt_engine"
                    | "onnx"
                    | "coreml"
                    | "torchscript"
                    | "mlx"
                    | "vulkan_spirv"
                    | "vendor_bundle"
                    | "unknown"
            )
        }) {
            return Err("model preflight response has invalid model formats".to_string());
        }
        validate_preflight_text("primary dtype", &self.manifest.primary_dtype, 1, 64)?;
        if !matches!(
            self.manifest.primary_dtype.as_str(),
            "f32" | "f16" | "bf16" | "i8" | "u8" | "i4" | "nf4" | "q8" | "q4" | "unknown"
        ) {
            return Err("model preflight response has an invalid primary dtype".to_string());
        }
        if let Some(quantization) = self.manifest.quantization.as_deref() {
            validate_preflight_text("quantization", quantization, 1, 128)?;
        }
        if self
            .manifest
            .quantization_bits
            .is_some_and(|bits| !(1..=16).contains(&bits))
        {
            return Err("model preflight response has invalid quantization bits".to_string());
        }

        for (label, value) in [
            ("configured engine", self.runtime.configured_engine.as_str()),
            ("selected engine", self.runtime.selected_engine.as_str()),
            ("engine maturity", self.runtime.engine_maturity.as_str()),
            ("device", self.runtime.device.as_str()),
            ("device backend", self.runtime.device_backend.as_str()),
            ("runtime support", self.runtime.support.as_str()),
            ("memory scope", self.memory.scope.as_str()),
        ] {
            validate_preflight_text(label, value, 1, 256)?;
        }
        if !matches!(
            self.runtime.support.as_str(),
            "native" | "fallback" | "unsupported"
        ) {
            return Err(
                "model preflight response has an invalid runtime support level".to_string(),
            );
        }
        if !matches!(
            self.runtime.engine_maturity.as_str(),
            "production" | "beta" | "experimental" | "skeleton"
        ) {
            return Err("model preflight response has an invalid engine maturity".to_string());
        }
        if !matches!(self.runtime.device.as_str(), "cpu" | "gpu" | "npu") {
            return Err("model preflight response has an invalid device".to_string());
        }
        validate_optional_preflight_text("backend reason", self.runtime.backend_reason.as_deref())?;
        validate_optional_preflight_text("support reason", self.runtime.support_reason.as_deref())?;
        validate_preflight_list(
            "runtime diagnostic tips",
            &self.runtime.diagnostic_tips,
            MAX_MODEL_PREFLIGHT_LIST_ITEMS,
            MAX_MODEL_PREFLIGHT_LIST_ITEM_CHARS,
        )?;
        validate_preflight_list(
            "warnings",
            &self.warnings,
            MAX_MODEL_PREFLIGHT_LIST_ITEMS,
            MAX_MODEL_PREFLIGHT_LIST_ITEM_CHARS,
        )?;
        if self.memory.per_request_context_tokens == 0
            || self.memory.max_concurrent == 0
            || self.memory.planned_context_tokens
                != self
                    .memory
                    .per_request_context_tokens
                    .saturating_mul(self.memory.max_concurrent)
            || !self.memory.memory_utilization.is_finite()
            || !(0.0..=1.0).contains(&self.memory.memory_utilization)
        {
            return Err("model preflight response has an invalid memory plan".to_string());
        }
        if self.loadable != self.load_blocker.is_none() {
            return Err("model preflight response has an inconsistent load decision".to_string());
        }
        validate_optional_preflight_text("load blocker", self.load_blocker.as_deref())?;
        if self.loadable
            && (!self.runtime.backend_available
                || self.runtime.support == "unsupported"
                || !self.memory.fits_budget)
        {
            return Err("model preflight response has an unsafe load decision".to_string());
        }
        Ok(())
    }
}

fn validate_model_tasks(tasks: &[String]) -> Result<(), String> {
    let valid = matches!(tasks, [generation] if generation == "generation")
        || matches!(tasks, [embedding, rerank] if embedding == "embedding" && rerank == "rerank");
    if valid {
        Ok(())
    } else {
        Err("model preflight response has invalid model tasks".to_string())
    }
}

fn validate_optional_preflight_text(label: &str, value: Option<&str>) -> Result<(), String> {
    value.map_or(Ok(()), |value| {
        validate_preflight_text(label, value, 1, MAX_MODEL_PREFLIGHT_METADATA_CHARS)
    })
}

fn validate_preflight_list(
    label: &str,
    values: &[String],
    max_items: usize,
    max_chars: usize,
) -> Result<(), String> {
    let mut unique = HashSet::with_capacity(values.len());
    if values.len() > max_items {
        return Err(format!(
            "model preflight response contains too many {label}"
        ));
    }
    for value in values {
        validate_preflight_text(label, value, 1, max_chars)?;
        if !unique.insert(value) {
            return Err(format!(
                "model preflight response contains duplicate {label}"
            ));
        }
    }
    Ok(())
}

fn validate_preflight_text(
    label: &str,
    value: &str,
    min_chars: usize,
    max_chars: usize,
) -> Result<(), String> {
    let length = value.chars().count();
    if length < min_chars
        || length > max_chars
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(format!(
            "model preflight response contains an invalid {label}"
        ))
    } else {
        Ok(())
    }
}

/// Model lifecycle snapshot from the Bloom server.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelCatalog {
    pub schema_version: u32,
    pub object: String,
    pub root: String,
    pub root_exists: bool,
    pub data: Vec<ModelCatalogEntry>,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub active_model: Option<ActiveModel>,
    pub load: ModelLoadStatus,
    pub download: ModelDownloadCapability,
    pub import: ModelImportCapability,
    pub index: ModelIndexCapability,
    pub storage: ModelStorageStatus,
    pub integrity: ModelIntegrityStatus,
}

impl ModelCatalog {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != MODEL_CATALOG_SCHEMA_VERSION {
            return Err(format!(
                "model catalog response uses unsupported schema version {}; expected {}",
                self.schema_version, MODEL_CATALOG_SCHEMA_VERSION
            ));
        }
        if self.object != MODEL_CATALOG_OBJECT {
            return Err("model catalog response has an invalid object type".to_string());
        }
        if self.root.is_empty() || self.root.chars().count() > MAX_MODEL_CATALOG_PATH_CHARS {
            return Err("model catalog response contains an invalid catalog root".to_string());
        }
        if self.data.len() > MAX_MODEL_CATALOG_ENTRIES {
            return Err("model catalog response contains too many models".to_string());
        }
        if !self.root_exists && !self.data.is_empty() {
            return Err("model catalog response contains models for a missing root".to_string());
        }

        let mut model_ids = HashSet::with_capacity(self.data.len());
        let mut active_catalog_ids = Vec::new();
        for model in &self.data {
            validate_catalog_id("model ID", &model.id)?;
            validate_catalog_text("model name", &model.name, MAX_MODEL_CATALOG_ID_CHARS, true)?;
            if !model_ids.insert(model.id.as_str()) {
                return Err("model catalog response contains duplicate model IDs".to_string());
            }
            if !matches!(model.kind.as_str(), "file" | "directory")
                || !matches!(
                    model.format.as_str(),
                    "gguf" | "onnx" | "coreml" | "bloom" | "transformers" | "openvino"
                )
            {
                return Err(
                    "model catalog response contains an invalid model kind or format".to_string(),
                );
            }
            if model.active {
                active_catalog_ids.push(model.id.as_str());
            }
            if model.provenance.is_some() && model.provenance_error.is_some() {
                return Err(
                    "model catalog response contains conflicting provenance state".to_string(),
                );
            }
            if let Some(provenance) = &model.provenance {
                validate_model_provenance(provenance)?;
                match (model.kind.as_str(), provenance.file_count) {
                    ("file", None) => {}
                    ("directory", Some(count))
                        if (2..=MAX_MODEL_PACKAGE_FILES).contains(&count) => {}
                    _ => {
                        return Err(
                            "model catalog response has inconsistent provenance kind".to_string()
                        )
                    }
                }
            }
            validate_catalog_optional_text(
                "provenance error",
                model.provenance_error.as_deref(),
                MAX_MODEL_CATALOG_TEXT_CHARS,
            )?;
        }
        if active_catalog_ids.len() > 1 {
            return Err("model catalog response contains multiple active entries".to_string());
        }

        validate_active_model(self.active_model.as_ref())?;
        validate_model_load_status(&self.load, self.active_model.as_ref())?;
        match self.active_model.as_ref() {
            Some(active) if active.source == "catalog" => {
                let catalog_id = active.catalog_id.as_deref().ok_or_else(|| {
                    "model catalog response omits the active catalog ID".to_string()
                })?;
                if active_catalog_ids.as_slice() != [catalog_id] {
                    return Err(
                        "model catalog response has inconsistent active catalog state".to_string(),
                    );
                }
            }
            Some(active) if active.source == "external" => {
                if active.catalog_id.is_some() || !active_catalog_ids.is_empty() {
                    return Err(
                        "model catalog response has inconsistent external model state".to_string(),
                    );
                }
            }
            None if !active_catalog_ids.is_empty() => {
                return Err("model catalog response omits active model metadata".to_string());
            }
            _ => {}
        }

        validate_download_capability(&self.download)?;
        validate_import_capability(&self.import)?;
        validate_index_capability(&self.index)?;
        validate_storage_status(&self.storage)?;
        validate_integrity_status(&self.integrity)?;
        Ok(())
    }
}

fn validate_model_provenance(provenance: &ModelProvenance) -> Result<(), String> {
    if !matches!(provenance.acquisition.as_str(), "download" | "import")
        || !valid_sha256(&provenance.sha256)
        || provenance.installed_at == 0
    {
        return Err("model catalog response contains invalid provenance".to_string());
    }
    if let Some(model_index_id) = provenance.model_index_id.as_deref() {
        validate_catalog_text("model index ID", model_index_id, 64, true)?;
    }
    validate_catalog_optional_text("model source URL", provenance.source_url.as_deref(), 2_048)?;
    validate_catalog_optional_text("model source host", provenance.source_host.as_deref(), 253)?;
    validate_catalog_optional_text("model license", provenance.license.as_deref(), 128)?;
    for timestamp in [
        provenance.last_verified_at,
        provenance.integrity_mismatch_at,
    ]
    .into_iter()
    .flatten()
    {
        if timestamp == 0 || timestamp < provenance.installed_at {
            return Err(
                "model catalog response contains invalid provenance timestamps".to_string(),
            );
        }
    }
    Ok(())
}

fn validate_active_model(active: Option<&ActiveModel>) -> Result<(), String> {
    let Some(active) = active else {
        return Ok(());
    };
    validate_catalog_text("active model ID", &active.id, 256, true)?;
    if let Some(catalog_id) = active.catalog_id.as_deref() {
        validate_catalog_id("active catalog ID", catalog_id)?;
    }
    if !matches!(active.source.as_str(), "catalog" | "external")
        || active.input_modalities.is_empty()
        || active.input_modalities.len() > 16
    {
        return Err("model catalog response contains invalid active model metadata".to_string());
    }
    let mut modalities = HashSet::with_capacity(active.input_modalities.len());
    for modality in &active.input_modalities {
        if !matches!(modality.as_str(), "text" | "audio" | "vision" | "multi")
            || !modalities.insert(modality.as_str())
        {
            return Err("model catalog response contains invalid active modalities".to_string());
        }
    }
    Ok(())
}

fn validate_model_load_status(
    load: &ModelLoadStatus,
    active: Option<&ActiveModel>,
) -> Result<(), String> {
    if !matches!(load.phase.as_str(), "idle" | "loading" | "ready" | "error") || load.progress > 100
    {
        return Err("model catalog response contains an invalid load status".to_string());
    }
    if let Some(requested_model) = load.requested_model.as_deref() {
        validate_catalog_text("requested model", requested_model, 256, true)?;
    }
    validate_catalog_optional_text("load error", load.error.as_deref(), 16_384)?;
    if (load.phase == "loading" && (load.requested_model.is_none() || load.error.is_some()))
        || (load.phase == "ready" && active.is_none())
        || (load.phase == "idle" && active.is_some())
        || (load.phase == "error" && (active.is_some() || load.error.is_none()))
    {
        return Err("model catalog response contains inconsistent load state".to_string());
    }
    Ok(())
}

fn validate_download_capability(download: &ModelDownloadCapability) -> Result<(), String> {
    validate_license_policy(&download.license_policy)?;
    if !matches!(
        download.status.phase.as_str(),
        "idle" | "queued" | "downloading" | "verifying" | "complete" | "cancelled" | "error"
    ) {
        return Err("model catalog response contains an invalid download phase".to_string());
    }
    validate_catalog_optional_id("download filename", download.status.filename.as_deref())?;
    validate_catalog_optional_text(
        "download source host",
        download.status.source_host.as_deref(),
        253,
    )?;
    validate_catalog_optional_text("download error", download.status.error.as_deref(), 16_384)?;
    if download
        .status
        .total_bytes
        .is_some_and(|total| download.status.downloaded_bytes > total)
        || (!download.enabled
            && (download.status.phase != "idle"
                || download.status.filename.is_some()
                || download.status.source_host.is_some()
                || download.status.downloaded_bytes != 0
                || download.status.total_bytes.is_some()
                || download.status.resumable
                || download.status.error.is_some()
                || !download.staged.is_empty()))
        || download.staged.len() > MAX_MODEL_CATALOG_STAGED_ENTRIES
    {
        return Err("model catalog response contains inconsistent download state".to_string());
    }
    let mut filenames = HashSet::with_capacity(download.staged.len());
    for staged in &download.staged {
        validate_catalog_id("staged download filename", &staged.filename)?;
        validate_catalog_text(
            "staged download source host",
            &staged.source_host,
            253,
            true,
        )?;
        if !filenames.insert(staged.filename.as_str()) {
            return Err("model catalog response contains duplicate staged downloads".to_string());
        }
    }
    Ok(())
}

fn validate_import_capability(import: &ModelImportCapability) -> Result<(), String> {
    validate_license_policy(&import.license_policy)?;
    if !matches!(
        import.status.phase.as_str(),
        "idle" | "ready" | "uploading" | "verifying" | "complete" | "error"
    ) {
        return Err("model catalog response contains an invalid import phase".to_string());
    }
    validate_catalog_optional_id("import filename", import.status.filename.as_deref())?;
    validate_catalog_optional_text("import error", import.status.error.as_deref(), 16_384)?;
    if import
        .status
        .total_bytes
        .is_some_and(|total| import.status.uploaded_bytes > total)
        || (import.enabled
            && (import.max_bytes == 0
                || import.max_chunk_bytes == 0
                || import.max_chunk_bytes as u64 > import.max_bytes))
        || (!import.enabled
            && (import.max_bytes != 0
                || import.max_chunk_bytes != 0
                || import.status.phase != "idle"
                || import.status.filename.is_some()
                || import.status.uploaded_bytes != 0
                || import.status.total_bytes.is_some()
                || import.status.error.is_some()
                || !import.staged.is_empty()))
        || import.staged.len() > MAX_MODEL_CATALOG_STAGED_ENTRIES
    {
        return Err("model catalog response contains inconsistent import state".to_string());
    }
    let mut filenames = HashSet::with_capacity(import.staged.len());
    for staged in &import.staged {
        validate_catalog_id("staged import filename", &staged.filename)?;
        if staged.total_bytes == 0
            || staged.uploaded_bytes > staged.total_bytes
            || !filenames.insert(staged.filename.as_str())
        {
            return Err("model catalog response contains invalid staged imports".to_string());
        }
    }
    Ok(())
}

fn validate_license_policy(policy: &ModelLicensePolicy) -> Result<(), String> {
    if policy.allowed.len() > MAX_MODEL_CATALOG_LICENSES
        || policy.enforced != !policy.allowed.is_empty()
    {
        return Err("model catalog response contains an invalid license policy".to_string());
    }
    let mut allowed = HashSet::with_capacity(policy.allowed.len());
    for license in &policy.allowed {
        validate_catalog_text("allowed model license", license, 128, true)?;
        if !allowed.insert(license.to_ascii_lowercase()) {
            return Err("model catalog response contains duplicate model licenses".to_string());
        }
    }
    Ok(())
}

fn validate_index_capability(index: &ModelIndexCapability) -> Result<(), String> {
    for value in [index.key_id.as_deref(), index.trust_id.as_deref()]
        .into_iter()
        .flatten()
    {
        if !valid_sha256(value) {
            return Err(
                "model catalog response contains an invalid model index identity".to_string(),
            );
        }
    }
    if (index.enabled
        && (index.trust_id.is_none()
            || !(1..=8).contains(&index.trusted_key_count)
            || !(1..=86_400).contains(&index.refresh_seconds)))
        || (!index.enabled
            && (index.key_id.is_some()
                || index.trust_id.is_some()
                || index.trusted_key_count != 0
                || index.refresh_seconds != 0
                || index.persistent_rollback_protection))
    {
        return Err("model catalog response contains inconsistent model index state".to_string());
    }
    Ok(())
}

fn validate_storage_status(storage: &ModelStorageStatus) -> Result<(), String> {
    let used_bytes = storage
        .installed_bytes
        .checked_add(storage.staged_download_bytes)
        .and_then(|value| value.checked_add(storage.staged_import_bytes));
    let committed_bytes = used_bytes.and_then(|value| value.checked_add(storage.reserved_bytes));
    if used_bytes != Some(storage.used_bytes)
        || committed_bytes != Some(storage.committed_bytes)
        || (storage.quota_enabled
            && (storage.max_bytes.is_none()
                || storage.available_bytes
                    != storage
                        .max_bytes
                        .map(|maximum| maximum.saturating_sub(storage.committed_bytes))))
        || (!storage.quota_enabled
            && (storage.max_bytes.is_some() || storage.available_bytes.is_some()))
        || storage.last_cleanup_at == Some(0)
    {
        return Err("model catalog response contains inconsistent storage accounting".to_string());
    }
    Ok(())
}

fn validate_integrity_status(integrity: &ModelIntegrityStatus) -> Result<(), String> {
    if !matches!(
        integrity.phase.as_str(),
        "idle" | "queued" | "verifying" | "complete" | "cancelled" | "error"
    ) {
        return Err("model catalog response contains an invalid integrity phase".to_string());
    }
    validate_catalog_optional_id("integrity model ID", integrity.model_id.as_deref())?;
    validate_catalog_optional_text("integrity error", integrity.error.as_deref(), 16_384)?;
    for hash in [
        integrity.expected_sha256.as_deref(),
        integrity.actual_sha256.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if !valid_sha256(hash) {
            return Err(
                "model catalog response contains an invalid integrity checksum".to_string(),
            );
        }
    }
    if integrity
        .total_bytes
        .is_some_and(|total| integrity.checked_bytes > total)
        || integrity.verified_at == Some(0)
        || (integrity.matches_expected.is_some()
            && (integrity.expected_sha256.is_none() || integrity.actual_sha256.is_none()))
        || (integrity.phase == "idle"
            && (integrity.model_id.is_some()
                || integrity.checked_bytes != 0
                || integrity.total_bytes.is_some()
                || integrity.expected_sha256.is_some()
                || integrity.actual_sha256.is_some()
                || integrity.matches_expected.is_some()
                || integrity.verified_at.is_some()
                || integrity.error.is_some()))
        || (integrity.phase == "complete"
            && (integrity.model_id.is_none()
                || integrity.total_bytes.is_none()
                || integrity.expected_sha256.is_none()
                || integrity.actual_sha256.is_none()
                || integrity.matches_expected.is_none()
                || integrity.verified_at.is_none()
                || integrity.error.is_some()))
        || (integrity.phase == "error" && integrity.error.is_none())
    {
        return Err("model catalog response contains inconsistent integrity state".to_string());
    }
    Ok(())
}

fn validate_catalog_optional_id(label: &str, value: Option<&str>) -> Result<(), String> {
    value.map_or(Ok(()), |value| validate_catalog_id(label, value))
}

fn validate_catalog_id(label: &str, value: &str) -> Result<(), String> {
    validate_catalog_text(label, value, MAX_MODEL_CATALOG_ID_CHARS, true)?;
    if value.contains('/') || value.contains('\\') || matches!(value, "." | "..") {
        return Err(format!(
            "model catalog response contains an invalid {label}"
        ));
    }
    Ok(())
}

fn validate_catalog_optional_text(
    label: &str,
    value: Option<&str>,
    max_chars: usize,
) -> Result<(), String> {
    value.map_or(Ok(()), |value| {
        validate_catalog_text(label, value, max_chars, true)
    })
}

fn validate_catalog_text(
    label: &str,
    value: &str,
    max_chars: usize,
    require_trimmed: bool,
) -> Result<(), String> {
    if value.is_empty()
        || value.chars().count() > max_chars
        || (require_trimmed && value.trim() != value)
        || value.chars().any(char::is_control)
    {
        Err(format!(
            "model catalog response contains an invalid {label}"
        ))
    } else {
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Serialize)]
struct ModelDownloadRequest<'a> {
    url: &'a str,
    filename: &'a str,
    sha256: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    license: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct ModelDownloadSourceRequest<'a> {
    url: &'a str,
}

#[derive(Debug, Serialize)]
struct ModelImportRequest<'a> {
    filename: &'a str,
    total_bytes: u64,
    sha256: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    license: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct ModelImportResponse {
    status: ModelImportStatus,
}

/// Optional governance metadata attached to one verified browser import.
pub struct ModelImportMetadata<'a> {
    pub source_url: &'a str,
    pub license: &'a str,
}

/// Connection settings and the user's credential-persistence choice.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnConfig {
    pub base_url: String,
    pub api_key: String,
    pub remember_api_key: bool,
}

impl Default for ConnConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            api_key: String::new(),
            remember_api_key: false,
        }
    }
}

impl ConnConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.base_url.is_empty()
            || self.base_url.trim() != self.base_url
            || self.base_url.chars().count() > MAX_BASE_URL_CHARS
            || self.base_url.chars().any(char::is_whitespace)
            || self.base_url.chars().any(char::is_control)
            || self.base_url.contains(['?', '#'])
        {
            return Err(format!(
                "Server URL must be an HTTP or HTTPS base URL of at most {MAX_BASE_URL_CHARS} characters without whitespace, a query, or a fragment."
            ));
        }
        let Some((scheme, authority_and_path)) = self.base_url.split_once("://") else {
            return Err("Server URL must use HTTP or HTTPS.".to_string());
        };
        let authority = authority_and_path.split('/').next().unwrap_or_default();
        if !(scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"))
            || authority.is_empty()
            || authority.contains('@')
            || !valid_http_authority(authority)
        {
            return Err(
                "Server URL must use HTTP or HTTPS with a non-empty host and no embedded credentials."
                    .to_string(),
            );
        }
        if self.api_key.chars().count() > MAX_API_KEY_CHARS
            || !self
                .api_key
                .bytes()
                .all(|byte| (0x21..=0x7e).contains(&byte))
        {
            return Err(format!(
                "API key must contain at most {MAX_API_KEY_CHARS} printable ASCII characters without spaces."
            ));
        }
        Ok(())
    }
}

fn valid_http_authority(authority: &str) -> bool {
    if let Some(ipv6) = authority.strip_prefix('[') {
        let Some((host, suffix)) = ipv6.split_once(']') else {
            return false;
        };
        return !host.is_empty()
            && host.contains(':')
            && (suffix.is_empty() || valid_http_port(suffix.strip_prefix(':')));
    }
    if authority.contains(['[', ']']) || authority.matches(':').count() > 1 {
        return false;
    }
    let (host, port) = authority
        .rsplit_once(':')
        .map_or((authority, None), |(host, port)| (host, Some(port)));
    !host.is_empty() && port.is_none_or(|port| valid_http_port(Some(port)))
}

fn valid_http_port(port: Option<&str>) -> bool {
    port.is_some_and(|port| {
        !port.is_empty()
            && port.bytes().all(|byte| byte.is_ascii_digit())
            && port.parse::<u16>().is_ok_and(|port| port > 0)
    })
}

/// A streaming update emitted while a chat request is active.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamUpdate {
    RequestId(String),
    Model(String),
    TextDelta(String),
    Usage(ChatUsage),
}

/// Exact token counts returned by a compatible streaming chat endpoint.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// A browser cancellation handle for one streaming request.
#[derive(Clone)]
pub struct ChatCancellation {
    controller: web_sys::AbortController,
}

impl ChatCancellation {
    pub fn new() -> Result<Self, String> {
        web_sys::AbortController::new()
            .map(|controller| Self { controller })
            .map_err(|error| format!("failed to create cancellation controller: {error:?}"))
    }

    pub fn cancel(&self) {
        self.controller.abort();
    }

    fn is_cancelled(&self) -> bool {
        self.controller.signal().aborted()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatStreamError {
    Cancelled,
    Request(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelImportClientError {
    Cancelled,
    Request(String),
}

impl fmt::Display for ModelImportClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("model import cancelled"),
            Self::Request(message) => formatter.write_str(message),
        }
    }
}

#[derive(Clone)]
pub struct ModelImportCancellation {
    controller: web_sys::AbortController,
}

impl ModelImportCancellation {
    pub fn new() -> Result<Self, String> {
        web_sys::AbortController::new()
            .map(|controller| Self { controller })
            .map_err(|error| format!("failed to create import cancellation controller: {error:?}"))
    }

    pub fn cancel(&self) {
        self.controller.abort();
    }

    fn is_cancelled(&self) -> bool {
        self.controller.signal().aborted()
    }
}

impl fmt::Display for ChatStreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("generation stopped"),
            Self::Request(message) => formatter.write_str(message),
        }
    }
}

/// Default base URL: the same origin that served the page.
pub fn default_base_url() -> String {
    web_sys::window()
        .and_then(|window| window.location().origin().ok())
        .unwrap_or_else(|| "http://127.0.0.1:3000".to_string())
}

fn auth_headers(cfg: &ConnConfig, headers: &web_sys::Headers) {
    if !cfg.api_key.is_empty() {
        headers
            .set("Authorization", &format!("Bearer {}", cfg.api_key))
            .ok();
    }
}

/// Fetch liveness and model-loading state.
///
/// A loading server returns HTTP 503 with a useful JSON body, so callers receive
/// a readiness snapshot for both ready and loading states.
pub async fn fetch_readiness(cfg: &ConnConfig) -> Result<Readiness, ReadinessError> {
    let url = format!("{}/ready", cfg.base_url.trim_end_matches('/'));
    let response = request(cfg, &url, "GET", None)
        .await
        .map_err(ReadinessError::Unavailable)?;
    let status = response.status();
    if !response.ok() && status != 503 {
        return Err(ReadinessError::Unavailable(
            read_response_error(&response).await,
        ));
    }
    let text = read_json_response_text(&response, MAX_READINESS_RESPONSE_BYTES)
        .await
        .map_err(ReadinessError::Incompatible)?;
    decode_readiness(&text).map_err(ReadinessError::Incompatible)
}

/// Fetch readiness and verify access to Bloom's protected API namespace.
pub async fn fetch_connection_readiness(cfg: &ConnConfig) -> Result<Readiness, ReadinessError> {
    let readiness = fetch_readiness(cfg).await?;
    verify_api_access(cfg).await?;
    Ok(readiness)
}

async fn verify_api_access(cfg: &ConnConfig) -> Result<(), ReadinessError> {
    let url = format!("{}/v1/models", cfg.base_url.trim_end_matches('/'));
    let response = request(cfg, &url, "GET", None)
        .await
        .map_err(ReadinessError::Unavailable)?;
    let status = response.status();
    if !response.ok() {
        let message = read_response_error(&response).await;
        return if status == 401 {
            Err(ReadinessError::Authentication(message))
        } else {
            Err(ReadinessError::Unavailable(message))
        };
    }
    let text = read_json_response_text(&response, MAX_AUTH_PROBE_RESPONSE_BYTES)
        .await
        .map_err(ReadinessError::Incompatible)?;
    decode_api_access_probe(&text).map_err(ReadinessError::Incompatible)
}

fn decode_api_access_probe(text: &str) -> Result<(), String> {
    if text.is_empty() || text.len() > MAX_AUTH_PROBE_RESPONSE_BYTES {
        return Err(format!(
            "API access probe response must be between 1 and {MAX_AUTH_PROBE_RESPONSE_BYTES} bytes"
        ));
    }
    let value = serde_json::from_str::<serde_json::Value>(text)
        .map_err(|error| format!("invalid API access probe response: {error}"))?;
    let data = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .filter(|data| data.len() <= 1);
    if value.get("object").and_then(serde_json::Value::as_str) != Some("list") || data.is_none() {
        return Err("API access probe did not return Bloom's bounded Models list".to_string());
    }
    if let Some(model) = data.and_then(|data| data.first()) {
        let id = model.get("id").and_then(serde_json::Value::as_str);
        if id.is_none_or(|id| {
            id.is_empty()
                || id.trim() != id
                || id.chars().count() > MAX_READINESS_MODEL_CHARS
                || id.chars().any(char::is_control)
        }) || model.get("object").and_then(serde_json::Value::as_str) != Some("model")
            || !model
                .get("created")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|created| created > 0)
            || model.get("owned_by").and_then(serde_json::Value::as_str) != Some("bloom")
        {
            return Err("API access probe returned an invalid Model resource".to_string());
        }
    }
    Ok(())
}

fn decode_readiness(text: &str) -> Result<Readiness, String> {
    if text.is_empty() || text.len() > MAX_READINESS_RESPONSE_BYTES {
        return Err(format!(
            "readiness response must be between 1 and {MAX_READINESS_RESPONSE_BYTES} bytes"
        ));
    }
    let readiness = serde_json::from_str::<Readiness>(text)
        .map_err(|error| format!("invalid readiness response: {error}"))?;
    readiness.validate()?;
    Ok(readiness)
}

/// Parse the embedding playground's one-nonblank-line-per-vector input.
pub fn parse_embedding_lines(text: &str) -> Result<Vec<String>, String> {
    let inputs = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if inputs.is_empty() || inputs.len() > MAX_EMBEDDING_INPUTS {
        return Err(format!(
            "Provide between 1 and {MAX_EMBEDDING_INPUTS} non-empty embedding lines."
        ));
    }
    let mut content_bytes = 0_usize;
    for (index, input) in inputs.iter().enumerate() {
        if input.chars().count() > MAX_EMBEDDING_INPUT_CHARS {
            return Err(format!(
                "Embedding line {} cannot exceed {MAX_EMBEDDING_INPUT_CHARS} characters.",
                index + 1
            ));
        }
        content_bytes = content_bytes
            .checked_add(input.len())
            .ok_or_else(|| "Embedding input size overflowed.".to_string())?;
        if content_bytes > MAX_EMBEDDING_CONTENT_BYTES {
            return Err(format!(
                "Combined embedding input cannot exceed {MAX_EMBEDDING_CONTENT_BYTES} bytes."
            ));
        }
    }
    Ok(inputs)
}

/// Validate the rerank playground's query, one-line documents, and result count.
pub fn prepare_rerank_input(
    query: &str,
    document_lines: &str,
    top_n: usize,
) -> Result<(String, Vec<String>, usize), String> {
    let query = query.trim().to_string();
    if query.is_empty() || query.chars().count() > MAX_RERANK_QUERY_CHARS {
        return Err(format!(
            "Rerank query must contain between 1 and {MAX_RERANK_QUERY_CHARS} characters."
        ));
    }
    let documents = document_lines
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if documents.is_empty() || documents.len() > MAX_RERANK_DOCUMENTS {
        return Err(format!(
            "Provide between 1 and {MAX_RERANK_DOCUMENTS} non-empty document lines."
        ));
    }
    if top_n == 0 || top_n > documents.len() {
        return Err("Top results must be between 1 and the document count.".to_string());
    }
    let mut content_bytes = query.len();
    for (index, document) in documents.iter().enumerate() {
        if document.chars().count() > MAX_RERANK_DOCUMENT_CHARS {
            return Err(format!(
                "Rerank document {} cannot exceed {MAX_RERANK_DOCUMENT_CHARS} characters.",
                index + 1
            ));
        }
        content_bytes = content_bytes
            .checked_add(document.len())
            .ok_or_else(|| "Rerank input size overflowed.".to_string())?;
        if content_bytes > MAX_RERANK_CONTENT_BYTES {
            return Err(format!(
                "Combined rerank input cannot exceed {MAX_RERANK_CONTENT_BYTES} bytes."
            ));
        }
    }
    Ok((query, documents, top_n))
}

/// Request a bounded normalized embedding batch from the active model.
pub async fn create_embeddings(
    cfg: &ConnConfig,
    model: &str,
    inputs: Vec<String>,
    dimensions: Option<usize>,
) -> Result<EmbeddingBatch, String> {
    validate_model_id(model)?;
    let joined = inputs.join("\n");
    let validated_inputs = parse_embedding_lines(&joined)?;
    if validated_inputs != inputs {
        return Err("Embedding inputs must already be trimmed non-empty lines.".to_string());
    }
    if dimensions.is_some_and(|value| !(1..=MAX_EMBEDDING_DIMENSIONS).contains(&value)) {
        return Err(format!(
            "Embedding dimensions must be between 1 and {MAX_EMBEDDING_DIMENSIONS}."
        ));
    }
    let body = serde_json::to_string(&serde_json::json!({
        "model": model,
        "input": inputs,
        "encoding_format": "float",
        "dimensions": dimensions,
    }))
    .map_err(|error| format!("failed to encode embedding request: {error}"))?;
    let url = format!("{}/v1/embeddings", cfg.base_url.trim_end_matches('/'));
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_EMBEDDING_RESPONSE_BYTES).await?;
    decode_embedding_response(&text, model, &validated_inputs, dimensions)
}

fn decode_embedding_response(
    text: &str,
    expected_model: &str,
    expected_inputs: &[String],
    requested_dimensions: Option<usize>,
) -> Result<EmbeddingBatch, String> {
    if text.is_empty() || text.len() > MAX_EMBEDDING_RESPONSE_BYTES {
        return Err(format!(
            "Embedding response must be between 1 and {MAX_EMBEDDING_RESPONSE_BYTES} bytes."
        ));
    }
    let response = serde_json::from_str::<EmbeddingWireResponse>(text)
        .map_err(|error| format!("invalid embedding response: {error}"))?;
    if response.object != "list"
        || response.model != expected_model
        || response.data.len() != expected_inputs.len()
        || response.usage.total_tokens != response.usage.prompt_tokens
    {
        return Err("Embedding response identity, count, model, or usage is inconsistent.".into());
    }
    let mut dimensions = None;
    let mut total_values = 0_usize;
    let mut vectors = Vec::with_capacity(response.data.len());
    for (expected_index, item) in response.data.into_iter().enumerate() {
        if item.object != "embedding" || item.index != expected_index {
            return Err("Embedding response indices or item identities are invalid.".into());
        }
        let width = item.embedding.len();
        if width == 0
            || width > MAX_EMBEDDING_DIMENSIONS
            || requested_dimensions.is_some_and(|requested| requested != width)
            || dimensions.is_some_and(|expected| expected != width)
        {
            return Err("Embedding response dimensions are invalid or inconsistent.".into());
        }
        total_values = total_values
            .checked_add(width)
            .ok_or_else(|| "Embedding response value count overflowed.".to_string())?;
        if total_values > MAX_EMBEDDING_VALUES {
            return Err("Embedding response exceeds the aggregate vector limit.".into());
        }
        let norm_squared = item.embedding.iter().try_fold(0.0_f64, |sum, value| {
            value
                .is_finite()
                .then_some(sum + f64::from(*value) * f64::from(*value))
                .ok_or_else(|| "Embedding response contains a non-finite value.".to_string())
        })?;
        let norm = norm_squared.sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 0.001 {
            return Err("Embedding response vector is not L2-normalized.".into());
        }
        dimensions = Some(width);
        vectors.push(EmbeddingVector {
            index: item.index,
            input: expected_inputs[item.index].clone(),
            values: item.embedding,
        });
    }
    Ok(EmbeddingBatch {
        model: response.model,
        vectors,
        prompt_tokens: response.usage.prompt_tokens,
    })
}

/// Request bounded bi-encoder reranking from the active embedding model.
pub async fn rerank_documents(
    cfg: &ConnConfig,
    model: &str,
    query: String,
    documents: Vec<String>,
    top_n: usize,
) -> Result<RerankBatch, String> {
    validate_model_id(model)?;
    let joined = documents.join("\n");
    let (validated_query, validated_documents, validated_top_n) =
        prepare_rerank_input(&query, &joined, top_n)?;
    if validated_query != query || validated_documents != documents {
        return Err("Rerank inputs must already be trimmed non-empty lines.".to_string());
    }
    let body = serde_json::to_string(&serde_json::json!({
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": validated_top_n,
        "return_documents": true,
    }))
    .map_err(|error| format!("failed to encode rerank request: {error}"))?;
    let url = format!("{}/v1/rerank", cfg.base_url.trim_end_matches('/'));
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_RERANK_RESPONSE_BYTES).await?;
    decode_rerank_response(
        &text,
        model,
        &validated_query,
        &validated_documents,
        validated_top_n,
    )
}

fn decode_rerank_response(
    text: &str,
    expected_model: &str,
    expected_query: &str,
    documents: &[String],
    expected_count: usize,
) -> Result<RerankBatch, String> {
    if text.is_empty() || text.len() > MAX_RERANK_RESPONSE_BYTES {
        return Err(format!(
            "Rerank response must be between 1 and {MAX_RERANK_RESPONSE_BYTES} bytes."
        ));
    }
    let response = serde_json::from_str::<RerankWireResponse>(text)
        .map_err(|error| format!("invalid rerank response: {error}"))?;
    if response.object != "rerank"
        || !valid_http_request_id(&response.id)
        || response.model != expected_model
        || response.results.len() != expected_count
        || response.usage.total_tokens != response.usage.prompt_tokens
    {
        return Err("Rerank response identity, count, model, or usage is inconsistent.".into());
    }
    let mut seen = HashSet::new();
    let mut previous = None::<(f64, usize)>;
    let mut results = Vec::with_capacity(response.results.len());
    for item in response.results {
        if item.index >= documents.len()
            || !seen.insert(item.index)
            || !item.relevance_score.is_finite()
            || !(-1.0..=1.0).contains(&item.relevance_score)
            || previous.is_some_and(|(score, index)| {
                item.relevance_score > score
                    || (item.relevance_score == score && item.index <= index)
            })
        {
            return Err("Rerank response scores or indices are invalid.".into());
        }
        let document = item
            .document
            .map(|document| document.text)
            .ok_or_else(|| "Rerank response omitted a requested document.".to_string())?;
        if document != documents[item.index] {
            return Err("Rerank response document does not match the submitted input.".into());
        }
        previous = Some((item.relevance_score, item.index));
        results.push(RerankResult {
            index: item.index,
            relevance_score: item.relevance_score,
            document,
        });
    }
    Ok(RerankBatch {
        id: response.id,
        model: response.model,
        query: expected_query.to_string(),
        results,
        prompt_tokens: response.usage.prompt_tokens,
    })
}

fn validate_embedding_vector_for_export(
    vector: &EmbeddingVector,
    expected_index: usize,
    expected_dimensions: Option<usize>,
) -> Result<usize, String> {
    if vector.index != expected_index || vector.index >= MAX_EMBEDDING_INPUTS {
        return Err("Embedding export indices are invalid or non-contiguous.".to_string());
    }
    if vector.input.is_empty()
        || vector.input.trim() != vector.input
        || vector.input.chars().count() > MAX_EMBEDDING_INPUT_CHARS
    {
        return Err("Embedding export contains an invalid input.".to_string());
    }
    let dimensions = vector.values.len();
    if dimensions == 0
        || dimensions > MAX_EMBEDDING_DIMENSIONS
        || expected_dimensions.is_some_and(|expected| expected != dimensions)
    {
        return Err("Embedding export dimensions are invalid or inconsistent.".to_string());
    }
    let norm_squared = vector.values.iter().try_fold(0.0_f64, |sum, value| {
        value
            .is_finite()
            .then_some(sum + f64::from(*value) * f64::from(*value))
            .ok_or_else(|| "Embedding export contains a non-finite value.".to_string())
    })?;
    let norm = norm_squared.sqrt();
    if !norm.is_finite() || (norm - 1.0).abs() > 0.001 {
        return Err("Embedding export vector is not L2-normalized.".to_string());
    }
    Ok(dimensions)
}

fn validate_embedding_batch_for_export(batch: &EmbeddingBatch) -> Result<(), String> {
    validate_model_id(&batch.model)?;
    if batch.vectors.is_empty() || batch.vectors.len() > MAX_EMBEDDING_INPUTS {
        return Err(format!(
            "Embedding export must contain between 1 and {MAX_EMBEDDING_INPUTS} vectors."
        ));
    }
    let mut dimensions = None;
    let mut content_bytes = 0_usize;
    let mut total_values = 0_usize;
    for (expected_index, vector) in batch.vectors.iter().enumerate() {
        let width = validate_embedding_vector_for_export(vector, expected_index, dimensions)?;
        dimensions = Some(width);
        content_bytes = content_bytes
            .checked_add(vector.input.len())
            .ok_or_else(|| "Embedding export input size overflowed.".to_string())?;
        total_values = total_values
            .checked_add(width)
            .ok_or_else(|| "Embedding export value count overflowed.".to_string())?;
        if content_bytes > MAX_EMBEDDING_CONTENT_BYTES || total_values > MAX_EMBEDDING_VALUES {
            return Err("Embedding export exceeds its aggregate content limit.".to_string());
        }
    }
    Ok(())
}

fn validate_rerank_result_for_export(result: &RerankResult) -> Result<(), String> {
    if result.index >= MAX_RERANK_DOCUMENTS
        || !result.relevance_score.is_finite()
        || !(-1.0..=1.0).contains(&result.relevance_score)
        || result.document.is_empty()
        || result.document.trim() != result.document
        || result.document.chars().count() > MAX_RERANK_DOCUMENT_CHARS
    {
        return Err("Rerank export contains an invalid result.".to_string());
    }
    Ok(())
}

fn validate_rerank_batch_for_export(batch: &RerankBatch) -> Result<(), String> {
    validate_model_id(&batch.model)?;
    if !valid_http_request_id(&batch.id)
        || batch.query.is_empty()
        || batch.query.trim() != batch.query
        || batch.query.chars().count() > MAX_RERANK_QUERY_CHARS
        || batch.results.is_empty()
        || batch.results.len() > MAX_RERANK_DOCUMENTS
    {
        return Err("Rerank export identity, query, or result count is invalid.".to_string());
    }
    let mut seen = HashSet::new();
    let mut previous = None::<(f64, usize)>;
    let mut content_bytes = batch.query.len();
    for result in &batch.results {
        validate_rerank_result_for_export(result)?;
        if !seen.insert(result.index)
            || previous.is_some_and(|(score, index)| {
                result.relevance_score > score
                    || (result.relevance_score == score && result.index <= index)
            })
        {
            return Err("Rerank export indices or stable score order are invalid.".to_string());
        }
        content_bytes = content_bytes
            .checked_add(result.document.len())
            .ok_or_else(|| "Rerank export content size overflowed.".to_string())?;
        if content_bytes > MAX_RERANK_CONTENT_BYTES {
            return Err("Rerank export exceeds its aggregate content limit.".to_string());
        }
        previous = Some((result.relevance_score, result.index));
    }
    Ok(())
}

/// Encode one complete normalized vector for an explicit clipboard action.
pub fn embedding_vector_clipboard_text(vector: &EmbeddingVector) -> Result<String, String> {
    validate_embedding_vector_for_export(vector, vector.index, None)?;
    let json = serde_json::to_string(&vector.values)
        .map_err(|error| format!("failed to encode embedding vector: {error}"))?;
    if json.len() > MAX_ENCODER_CLIPBOARD_BYTES {
        return Err("Embedding vector exceeds the clipboard size limit.".to_string());
    }
    Ok(json)
}

/// Encode one ranked document and score for an explicit clipboard action.
pub fn rerank_result_clipboard_text(result: &RerankResult) -> Result<String, String> {
    validate_rerank_result_for_export(result)?;
    let json = serde_json::to_string(result)
        .map_err(|error| format!("failed to encode rerank result: {error}"))?;
    if json.len() > MAX_ENCODER_CLIPBOARD_BYTES {
        return Err("Rerank result exceeds the clipboard size limit.".to_string());
    }
    Ok(json)
}

/// Encode a bounded, versioned embedding result artifact.
pub fn encode_embedding_export(batch: &EmbeddingBatch) -> Result<String, String> {
    validate_embedding_batch_for_export(batch)?;
    let vectors = batch
        .vectors
        .iter()
        .map(|vector| EmbeddingExportVector {
            index: vector.index,
            input: &vector.input,
            embedding: &vector.values,
        })
        .collect();
    let export = EmbeddingExport {
        schema_version: 1,
        object: "bloom.embedding_result",
        model: &batch.model,
        prompt_tokens: batch.prompt_tokens,
        vectors,
    };
    let mut json = serde_json::to_string(&export)
        .map_err(|error| format!("failed to encode embedding export: {error}"))?;
    json.push('\n');
    if json.len() > MAX_EMBEDDING_EXPORT_BYTES {
        return Err("Embedding export exceeds the 40 MiB download limit.".to_string());
    }
    Ok(json)
}

/// Encode a bounded, versioned rerank result artifact.
pub fn encode_rerank_export(batch: &RerankBatch) -> Result<String, String> {
    validate_rerank_batch_for_export(batch)?;
    let export = RerankExport {
        schema_version: 1,
        object: "bloom.rerank_result",
        id: &batch.id,
        model: &batch.model,
        query: &batch.query,
        prompt_tokens: batch.prompt_tokens,
        results: &batch.results,
    };
    let mut json = serde_json::to_string(&export)
        .map_err(|error| format!("failed to encode rerank export: {error}"))?;
    json.push('\n');
    if json.len() > MAX_RERANK_EXPORT_BYTES {
        return Err("Rerank export exceeds the 4 MiB download limit.".to_string());
    }
    Ok(json)
}

/// Download one validated embedding result artifact.
pub fn download_embedding_export(batch: &EmbeddingBatch) -> Result<(), String> {
    let json = encode_embedding_export(batch)?;
    browser::download_text_file(
        EMBEDDING_EXPORT_FILENAME,
        "application/json;charset=utf-8",
        &json,
    )
}

/// Download one validated rerank result artifact.
pub fn download_rerank_export(batch: &RerankBatch) -> Result<(), String> {
    let json = encode_rerank_export(batch)?;
    browser::download_text_file(
        RERANK_EXPORT_FILENAME,
        "application/json;charset=utf-8",
        &json,
    )
}

/// Fetch a bounded, versioned server diagnostics snapshot.
pub async fn fetch_observability(cfg: &ConnConfig) -> Result<ObservabilitySnapshot, String> {
    let url = format!("{}/v1/observability", cfg.base_url.trim_end_matches('/'));
    let response = request(cfg, &url, "GET", None).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_OBSERVABILITY_RESPONSE_BYTES).await?;
    decode_observability_snapshot(&text)
}

fn decode_observability_snapshot(text: &str) -> Result<ObservabilitySnapshot, String> {
    if text.is_empty() || text.len() > MAX_OBSERVABILITY_RESPONSE_BYTES {
        return Err(format!(
            "observability response must be between 1 and {MAX_OBSERVABILITY_RESPONSE_BYTES} bytes"
        ));
    }
    let value = serde_json::from_str::<serde_json::Value>(text)
        .map_err(|error| format!("invalid observability response: {error}"))?;
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
        || value.get("object").and_then(serde_json::Value::as_str)
            != Some("bloom.observability_snapshot")
    {
        return Err("unsupported observability snapshot version".to_string());
    }
    let snapshot = serde_json::from_value::<ObservabilitySnapshot>(value)
        .map_err(|error| format!("invalid observability response: {error}"))?;
    snapshot.validate()?;
    Ok(snapshot)
}

/// Fetch the safe server-side model catalog and lifecycle state.
pub async fn fetch_model_catalog(cfg: &ConnConfig) -> Result<ModelCatalog, String> {
    let url = format!(
        "{}/v1/model-management/models",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &url, "GET", None).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_MODEL_CATALOG_RESPONSE_BYTES).await?;
    decode_model_catalog(&text)
}

fn decode_model_catalog(text: &str) -> Result<ModelCatalog, String> {
    if text.is_empty() || text.len() > MAX_MODEL_CATALOG_RESPONSE_BYTES {
        return Err(format!(
            "model catalog response must be between 1 and {MAX_MODEL_CATALOG_RESPONSE_BYTES} bytes"
        ));
    }
    let catalog = serde_json::from_str::<ModelCatalog>(text)
        .map_err(|error| format!("invalid model catalog response: {error}"))?;
    catalog
        .validate()
        .map_err(|error| format!("invalid model catalog response: {error}"))?;
    Ok(catalog)
}

/// Fetch or explicitly refresh the configured publisher-signed model index.
pub async fn fetch_model_index(
    cfg: &ConnConfig,
    force_refresh: bool,
) -> Result<ModelIndexSnapshot, String> {
    let url = format!(
        "{}/v1/model-management/index",
        cfg.base_url.trim_end_matches('/')
    );
    let method = if force_refresh { "POST" } else { "GET" };
    let response = request(cfg, &url, method, None).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_MODEL_INDEX_RESPONSE_BYTES).await?;
    decode_model_index(&text)
}

fn decode_model_index(text: &str) -> Result<ModelIndexSnapshot, String> {
    if text.is_empty() || text.len() > MAX_MODEL_INDEX_RESPONSE_BYTES {
        return Err(format!(
            "model index response must be between 1 and {MAX_MODEL_INDEX_RESPONSE_BYTES} bytes"
        ));
    }
    let snapshot = serde_json::from_str::<ModelIndexSnapshot>(text)
        .map_err(|error| format!("invalid model index response: {error}"))?;
    snapshot.validate()?;
    Ok(snapshot)
}

impl ModelIndexSnapshot {
    fn validate(&self) -> Result<(), String> {
        if !matches!(self.schema_version, 1 | 2) || self.object != "bloom.model_index" {
            return Err("unsupported model index response".to_string());
        }
        if !is_lower_hex(&self.key_id, 64) {
            return Err("model index response contains an invalid key ID".to_string());
        }
        validate_index_text("name", &self.name, 1, 80)?;
        if self.generated_at == 0 || self.expires_at <= self.generated_at {
            return Err("model index response contains invalid validity times".to_string());
        }
        if !matches!(self.source_kind.as_str(), "file" | "https")
            || !matches!(self.cache_status.as_str(), "fresh" | "cached" | "stale")
        {
            return Err("model index response contains invalid source metadata".to_string());
        }
        if self
            .warning
            .as_ref()
            .is_some_and(|warning| validate_index_text("warning", warning, 1, 512).is_err())
            || (self.cache_status == "stale" && self.warning.is_none())
        {
            return Err("model index response contains an invalid cache warning".to_string());
        }
        if self.data.len() > MAX_MODEL_INDEX_ENTRIES {
            return Err("model index response contains too many entries".to_string());
        }
        let mut ids = HashSet::with_capacity(self.data.len());
        let mut filenames = HashSet::with_capacity(self.data.len());
        for entry in &self.data {
            entry.validate(self.schema_version)?;
            if !ids.insert(entry.id.to_ascii_lowercase())
                || !filenames.insert(entry.filename.to_ascii_lowercase())
            {
                return Err("model index response contains duplicate entries".to_string());
            }
        }
        Ok(())
    }
}

impl ModelIndexEntry {
    fn validate(&self, schema_version: u8) -> Result<(), String> {
        validate_index_text("entry ID", &self.id, 1, 64)?;
        if !self.id.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (index > 0 && matches!(byte, b'-' | b'_' | b'.'))
        }) {
            return Err("model index response contains an invalid entry ID".to_string());
        }
        validate_index_text("entry name", &self.name, 1, 80)?;
        validate_index_text("entry description", &self.description, 1, 400)?;
        if self.size_bytes == 0 || !is_lower_hex(&self.sha256, 64) {
            return Err("model index response contains invalid verification metadata".to_string());
        }
        if self.files.is_empty() {
            validate_index_filename(&self.filename, &self.format)?;
            let url = self.download_url.as_deref().ok_or_else(|| {
                "model index response omits a single-file download URL".to_string()
            })?;
            validate_index_download_url(url, &self.filename)?;
        } else {
            if schema_version != 2
                || self.download_url.is_some()
                || self.format != "transformers"
                || !valid_package_directory(&self.filename)
                || !(2..=MAX_MODEL_PACKAGE_FILES).contains(&self.files.len())
            {
                return Err("model index response contains an invalid model package".to_string());
            }
            let mut names = HashSet::with_capacity(self.files.len());
            let mut source_identity = None;
            let mut total_bytes = 0_u64;
            let mut has_config = false;
            for file in &self.files {
                validate_package_filename(&file.filename)?;
                if file.size_bytes == 0
                    || !is_lower_hex(&file.sha256, 64)
                    || !names.insert(file.filename.to_ascii_lowercase())
                {
                    return Err(
                        "model index response contains invalid package verification metadata"
                            .to_string(),
                    );
                }
                total_bytes = total_bytes
                    .checked_add(file.size_bytes)
                    .ok_or_else(|| "model index response package size overflowed".to_string())?;
                has_config |= file.filename == "config.json";
                let identity = validate_index_download_url(&file.download_url, &file.filename)?;
                if source_identity
                    .as_ref()
                    .is_some_and(|expected| expected != &identity)
                {
                    return Err(
                        "model index response package files use different commits".to_string()
                    );
                }
                source_identity.get_or_insert(identity);
            }
            if total_bytes != self.size_bytes || !has_config {
                return Err("model index response package manifest is incomplete".to_string());
            }
            validate_package_safetensors_layout(&self.files)?;
        }
        validate_index_text("entry license", &self.license, 1, 128)?;
        if let Some(family) = self.family.as_ref() {
            validate_index_text("model family", family, 1, 64)?;
        }
        if self.parameter_count == Some(0) {
            return Err("model index response contains an invalid parameter count".to_string());
        }
        if let Some(quantization) = self.quantization.as_ref() {
            validate_index_text("model quantization", quantization, 1, 32)?;
        }
        if self.tags.len() > 12 {
            return Err("model index response contains too many tags".to_string());
        }
        let mut tags = HashSet::with_capacity(self.tags.len());
        for tag in &self.tags {
            validate_index_text("model tag", tag, 1, 32)?;
            if !tag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
                || !tags.insert(tag)
            {
                return Err("model index response contains invalid tags".to_string());
            }
        }
        let mut reasons = HashSet::with_capacity(self.blocking_reasons.len());
        if self.blocking_reasons.len() > 2
            || self.blocking_reasons.iter().any(|reason| {
                !matches!(reason.as_str(), "size_limit" | "license_policy")
                    || !reasons.insert(reason)
            })
            || self.downloadable != self.blocking_reasons.is_empty()
        {
            return Err("model index response contains invalid download policy state".to_string());
        }
        Ok(())
    }
}

fn validate_index_text(label: &str, value: &str, min: usize, max: usize) -> Result<(), String> {
    let length = value.chars().count();
    if length < min || length > max || value.trim() != value || value.chars().any(char::is_control)
    {
        return Err(format!("model index response contains an invalid {label}"));
    }
    Ok(())
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_index_filename(filename: &str, format: &str) -> Result<(), String> {
    let expected_format = filename
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase());
    if filename.is_empty()
        || filename.len() > 255
        || filename.starts_with('.')
        || filename.contains(['/', '\\', '%'])
        || !matches!(
            expected_format.as_deref(),
            Some("gguf" | "onnx" | "mlmodel")
        )
        || expected_format.as_deref() != Some(format)
    {
        return Err("model index response contains an invalid filename".to_string());
    }
    Ok(())
}

fn validate_index_download_url(url: &str, filename: &str) -> Result<String, String> {
    if url.is_empty() || url.len() > MAX_MODEL_DOWNLOAD_SOURCE_URL_BYTES || url.contains(['?', '#'])
    {
        return Err("model index response contains an invalid download URL".to_string());
    }
    let Some(authority_and_path) = url.strip_prefix("https://") else {
        return Err("model index response contains a non-HTTPS download URL".to_string());
    };
    let mut split = authority_and_path.split('/');
    if !matches!(split.next(), Some("huggingface.co" | "www.huggingface.co")) {
        return Err("model index response contains an untrusted download host".to_string());
    }
    let segments = split.collect::<Vec<_>>();
    let valid_revision = segments.get(3).is_some_and(|revision| {
        matches!(revision.len(), 40 | 64) && is_lower_hex(revision, revision.len())
    });
    if segments.len() < 5
        || segments[0].is_empty()
        || segments[1].is_empty()
        || segments[2] != "resolve"
        || !valid_revision
        || segments[4..]
            .iter()
            .any(|segment| segment.is_empty() || segment.contains('%'))
        || segments[4..].join("/") != filename
    {
        return Err(
            "model index response download URL is not pinned to an immutable commit".to_string(),
        );
    }
    Ok(format!("{}/{}/{}", segments[0], segments[1], segments[3]))
}

fn valid_package_directory(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    !value.is_empty()
        && value.len() <= 255
        && value.trim() == value
        && !value.starts_with('.')
        && !value.contains(['/', '\\', '%'])
        && !value.chars().any(char::is_control)
        && ![".gguf", ".onnx", ".mlmodel", ".mlpackage", ".mlmodelc"]
            .iter()
            .any(|suffix| lower.ends_with(suffix))
}

fn validate_package_filename(value: &str) -> Result<(), String> {
    let components = value.split('/').collect::<Vec<_>>();
    let supported = [".json", ".safetensors", ".txt", ".model", ".tiktoken"];
    if value.is_empty()
        || value.len() > 512
        || value.trim() != value
        || value.contains(['\\', '%'])
        || value.chars().any(char::is_control)
        || components.is_empty()
        || components.len() > 8
        || components
            .iter()
            .any(|component| component.is_empty() || component.starts_with('.'))
        || !supported.iter().any(|suffix| value.ends_with(suffix))
    {
        return Err("model index response contains an invalid package filename".to_string());
    }
    Ok(())
}

fn validate_package_safetensors_layout(files: &[ModelIndexFile]) -> Result<(), String> {
    let has_single = files
        .iter()
        .any(|file| file.filename == "model.safetensors");
    let has_index = files
        .iter()
        .any(|file| file.filename == "model.safetensors.index.json");
    let mut shards = files
        .iter()
        .filter_map(|file| {
            (file.filename.starts_with("model-") && file.filename.ends_with(".safetensors"))
                .then_some(file.filename.as_str())
        })
        .map(|filename| parse_package_shard_name(filename).map(|position| (filename, position)))
        .collect::<Result<Vec<_>, _>>()?;
    if has_single {
        if has_index || !shards.is_empty() {
            return Err(
                "model index response mixes consolidated and sharded Safetensors layouts"
                    .to_string(),
            );
        }
        return Ok(());
    }
    if !has_index || shards.is_empty() {
        return Err(
            "model index response package has an incomplete Safetensors layout".to_string(),
        );
    }
    shards.sort_by_key(|(_, (index, _))| *index);
    let expected_total = shards[0].1 .1;
    if shards.len() != expected_total
        || shards
            .iter()
            .enumerate()
            .any(|(position, (_, (index, total)))| {
                *index != position + 1 || *total != expected_total
            })
    {
        return Err(
            "model index response package has an incomplete Safetensors shard sequence".to_string(),
        );
    }
    Ok(())
}

fn parse_package_shard_name(filename: &str) -> Result<(usize, usize), String> {
    let body = filename
        .strip_prefix("model-")
        .and_then(|value| value.strip_suffix(".safetensors"))
        .ok_or_else(|| "model index response has an invalid Safetensors shard name".to_string())?;
    let (index, total) = body
        .split_once("-of-")
        .ok_or_else(|| "model index response has an invalid Safetensors shard name".to_string())?;
    if index.len() != 5
        || total.len() != 5
        || !index.bytes().all(|byte| byte.is_ascii_digit())
        || !total.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err("model index response has an invalid Safetensors shard name".to_string());
    }
    let index = index
        .parse::<usize>()
        .map_err(|_| "model index response has an invalid Safetensors shard name".to_string())?;
    let total = total
        .parse::<usize>()
        .map_err(|_| "model index response has an invalid Safetensors shard name".to_string())?;
    if index == 0 || total == 0 || index > total || total > MAX_MODEL_PACKAGE_FILES {
        return Err("model index response has an invalid Safetensors shard position".to_string());
    }
    Ok((index, total))
}

/// Fetch the stable, path-free model inventory used for audit and backup.
pub async fn fetch_model_inventory(cfg: &ConnConfig) -> Result<ModelInventory, String> {
    let url = format!(
        "{}/v1/model-management/inventory",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &url, "GET", None).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_MODEL_INVENTORY_BYTES as usize).await?;
    decode_model_inventory(&text)
}

/// Save a validated model inventory as a pretty-printed JSON download.
pub fn download_model_inventory(inventory: &ModelInventory) -> Result<(), String> {
    validate_model_inventory(inventory)?;
    let mut json = serde_json::to_string_pretty(inventory)
        .map_err(|error| format!("failed to encode model inventory: {error}"))?;
    json.push('\n');
    browser::download_text_file(MODEL_INVENTORY_FILENAME, "application/json", &json)
}

/// Compare a bounded versioned inventory file with the current server catalog.
pub async fn reconcile_model_inventory_file(
    cfg: &ConnConfig,
    file: &web_sys::File,
) -> Result<ModelInventoryComparison, String> {
    let size = file.size();
    if !size.is_finite() || size <= 0.0 || size > MAX_MODEL_INVENTORY_BYTES as f64 {
        return Err(format!(
            "Model inventory must be between 1 byte and {} bytes.",
            MAX_MODEL_INVENTORY_BYTES
        ));
    }
    let text = JsFuture::from(file.text())
        .await
        .map_err(|error| format!("failed to read model inventory: {error:?}"))?
        .as_string()
        .ok_or_else(|| "model inventory is not valid text".to_string())?;
    if text.len() as u64 > MAX_MODEL_INVENTORY_BYTES {
        return Err("model inventory text exceeds the supported size limit".to_string());
    }
    let expected = serde_json::from_str::<ModelInventory>(&text)
        .map_err(|error| format!("invalid model inventory file: {error}"))?;
    let url = format!(
        "{}/v1/model-management/inventory/reconcile",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &url, "POST", Some(&text)).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let response_text = read_json_response_text(&response, MAX_HTTP_RESPONSE_BYTES).await?;
    let report = decode_model_inventory_reconciliation(&response_text)?;
    Ok(ModelInventoryComparison { expected, report })
}

/// Queue one explicit, exact-commit verified restore from a validated inventory.
pub async fn restore_model_from_inventory(
    cfg: &ConnConfig,
    inventory: &ModelInventory,
    model_id: &str,
) -> Result<(), String> {
    validate_model_inventory(inventory)?;
    let encoded_id = js_sys::encode_uri_component(model_id)
        .as_string()
        .ok_or_else(|| "failed to encode inventory model ID".to_string())?;
    let url = format!(
        "{}/v1/model-management/inventory/restore/{encoded_id}",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(inventory)
        .map_err(|error| format!("failed to encode model inventory: {error}"))?;
    if body.len() as u64 > MAX_MODEL_INVENTORY_BYTES {
        return Err("model inventory exceeds the supported restore size limit".to_string());
    }
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_MODEL_RESTORE_RESPONSE_BYTES).await?;
    let response: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("invalid model inventory restore response: {error}"))?;
    if response.get("object").and_then(serde_json::Value::as_str)
        != Some("bloom.model_inventory_restore")
        || response
            .get("accepted")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || response.get("model").and_then(serde_json::Value::as_str) != Some(model_id)
    {
        return Err("server returned an inconsistent model inventory restore response".to_string());
    }
    Ok(())
}

fn decode_model_inventory(text: &str) -> Result<ModelInventory, String> {
    let inventory = serde_json::from_str::<ModelInventory>(text)
        .map_err(|error| format!("invalid model inventory response: {error}"))?;
    validate_model_inventory(&inventory)?;
    Ok(inventory)
}

fn validate_model_inventory(inventory: &ModelInventory) -> Result<(), String> {
    if !(MIN_MODEL_INVENTORY_SCHEMA_VERSION..=MODEL_INVENTORY_SCHEMA_VERSION)
        .contains(&inventory.schema_version)
    {
        return Err(format!(
            "unsupported model inventory schema version: {}",
            inventory.schema_version
        ));
    }
    if inventory.object != MODEL_INVENTORY_OBJECT {
        return Err("server returned an unexpected model inventory object".to_string());
    }
    if inventory.summary.model_count != inventory.models.len() {
        return Err("server returned an inconsistent model inventory count".to_string());
    }
    let provenance_count = inventory
        .models
        .iter()
        .filter(|model| model.provenance_status == "recorded")
        .count();
    let source_locked_count = inventory
        .models
        .iter()
        .filter(|model| model.source_locked)
        .count();
    let quarantined_count = inventory
        .models
        .iter()
        .filter(|model| model.integrity == "quarantined")
        .count();
    let invalid_provenance_count = inventory
        .models
        .iter()
        .filter(|model| model.provenance_status == "invalid")
        .count();
    if inventory.summary.provenance_count != provenance_count
        || inventory.summary.source_locked_count != source_locked_count
        || inventory.summary.quarantined_count != quarantined_count
        || inventory.summary.invalid_provenance_count != invalid_provenance_count
    {
        return Err("server returned inconsistent model inventory summary counts".to_string());
    }
    if inventory
        .models
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
    {
        return Err("server returned an unsorted or duplicate model inventory".to_string());
    }
    for model in &inventory.models {
        if inventory.schema_version == 1 && model.model_index_id.is_some() {
            return Err("model inventory version 1 cannot contain a signed-index ID".to_string());
        }
        if let Some(model_index_id) = model.model_index_id.as_deref() {
            let valid = !model_index_id.is_empty()
                && model_index_id.len() <= 64
                && model_index_id.bytes().enumerate().all(|(index, byte)| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || (index > 0 && matches!(byte, b'-' | b'_' | b'.'))
                });
            if !valid
                || model.provenance_status != "recorded"
                || model.acquisition.as_deref() != Some("download")
            {
                return Err("server returned an invalid inventory signed-index ID".to_string());
            }
        }
    }
    Ok(())
}

fn decode_model_inventory_reconciliation(
    text: &str,
) -> Result<ModelInventoryReconciliation, String> {
    let report = serde_json::from_str::<ModelInventoryReconciliation>(text)
        .map_err(|error| format!("invalid model inventory reconciliation response: {error}"))?;
    validate_model_inventory_reconciliation(&report)?;
    Ok(report)
}

fn validate_model_inventory_reconciliation(
    report: &ModelInventoryReconciliation,
) -> Result<(), String> {
    if report.schema_version != MODEL_INVENTORY_RECONCILIATION_SCHEMA_VERSION
        || report.object != "bloom.model_inventory_reconciliation"
    {
        return Err("server returned an unsupported inventory reconciliation object".to_string());
    }
    let summary = &report.summary;
    if summary.expected_model_count
        != summary.matching_count + summary.missing_count + summary.changed_count
        || summary.current_model_count
            != summary.matching_count + summary.unexpected_count + summary.changed_count
        || summary.drift_count
            != summary.missing_count + summary.unexpected_count + summary.changed_count
        || summary.blocking_count > summary.drift_count
        || summary.restorable_count > summary.missing_count
        || report.in_sync != (summary.drift_count == 0)
        || report.truncated != (report.drift.len() < summary.drift_count)
        || report.drift.len() > summary.drift_count
        || report.drift.len() > MAX_MODEL_INVENTORY_DRIFT_ENTRIES
        || (report.truncated
            && (report.drift.len() != MAX_MODEL_INVENTORY_DRIFT_ENTRIES
                || summary.drift_count <= MAX_MODEL_INVENTORY_DRIFT_ENTRIES))
    {
        return Err("server returned inconsistent inventory reconciliation counts".to_string());
    }
    if report.drift.windows(2).any(|pair| pair[0].id >= pair[1].id) {
        return Err("server returned unsorted or duplicate inventory drift".to_string());
    }
    let mut detailed_blocking_count = 0;
    let mut detailed_restorable_count = 0;
    for drift in &report.drift {
        if drift.id.is_empty()
            || !matches!(drift.status.as_str(), "missing" | "unexpected" | "changed")
            || !matches!(drift.severity.as_str(), "blocking" | "warning")
            || drift.changes.is_empty()
            || drift
                .changes
                .iter()
                .any(|change| inventory_change_rank(change).is_none())
            || drift.changes.windows(2).any(|pair| {
                inventory_change_rank(&pair[0]).unwrap_or(usize::MAX)
                    >= inventory_change_rank(&pair[1]).unwrap_or(usize::MAX)
            })
        {
            return Err("server returned an invalid inventory drift entry".to_string());
        }
        let status_matches_changes = match drift.status.as_str() {
            "missing" => {
                drift.severity == "blocking" && drift.changes.as_slice() == ["model_missing"]
            }
            "unexpected" => drift.changes.as_slice() == ["model_unexpected"],
            "changed" => !drift
                .changes
                .iter()
                .any(|change| matches!(change.as_str(), "model_missing" | "model_unexpected")),
            _ => false,
        };
        if !status_matches_changes {
            return Err("server returned inconsistent inventory drift fields".to_string());
        }
        if drift.restore_available && drift.status != "missing" {
            return Err("server returned an invalid inventory restore capability".to_string());
        }
        if drift.restore_available {
            detailed_restorable_count += 1;
        }
        if drift.severity == "blocking" {
            detailed_blocking_count += 1;
        }
    }
    if detailed_blocking_count > summary.blocking_count
        || (!report.truncated && detailed_blocking_count != summary.blocking_count)
    {
        return Err("server returned an inconsistent inventory blocking count".to_string());
    }
    if detailed_restorable_count > summary.restorable_count
        || (!report.truncated && detailed_restorable_count != summary.restorable_count)
    {
        return Err("server returned an inconsistent inventory restorable count".to_string());
    }
    Ok(())
}

fn inventory_change_rank(change: &str) -> Option<usize> {
    [
        "model_missing",
        "model_unexpected",
        "kind",
        "format",
        "size_bytes",
        "size_complete",
        "provenance_status",
        "acquisition",
        "model_index_id",
        "source_url",
        "source_host",
        "source_revision",
        "source_lock",
        "sha256",
        "license",
        "installed_at",
        "last_verified_at",
        "integrity",
    ]
    .iter()
    .position(|candidate| *candidate == change)
}

/// Inspect one catalog model against the server's configured engine, device, and memory budget.
pub async fn preflight_model(
    cfg: &ConnConfig,
    model_id: &str,
) -> Result<ModelPreflightReport, String> {
    let url = format!(
        "{}/v1/model-management/preflight",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&serde_json::json!({ "id": model_id }))
        .map_err(|error| format!("failed to encode model preflight request: {error}"))?;
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_MODEL_PREFLIGHT_RESPONSE_BYTES).await?;
    serde_json::from_str::<ModelPreflightResponse>(&text)
        .map_err(|error| format!("invalid model preflight response: {error}"))?
        .validate(model_id)
        .map_err(|error| format!("invalid model preflight response: {error}"))
}

/// Queue a transactional switch to a model returned by the catalog.
pub async fn switch_model(cfg: &ConnConfig, model_id: &str) -> Result<(), String> {
    let url = format!(
        "{}/v1/model-management/switch",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&serde_json::json!({ "id": model_id }))
        .map_err(|error| format!("failed to encode model switch request: {error}"))?;
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Unload the active model and release its runtime resources.
pub async fn unload_model(cfg: &ConnConfig) -> Result<(), String> {
    let url = format!(
        "{}/v1/model-management/unload",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &url, "POST", Some("{}")).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Permanently remove one inactive entry returned by the safe model catalog.
pub async fn remove_model(cfg: &ConnConfig, model_id: &str) -> Result<(), String> {
    let url = format!(
        "{}/v1/model-management/remove",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&serde_json::json!({ "id": model_id }))
        .map_err(|error| format!("failed to encode model removal request: {error}"))?;
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Queue an on-demand SHA-256 check against a model's acquisition record.
pub async fn verify_model_integrity(cfg: &ConnConfig, model_id: &str) -> Result<(), String> {
    let url = format!(
        "{}/v1/model-management/integrity",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&serde_json::json!({ "id": model_id }))
        .map_err(|error| format!("failed to encode model integrity request: {error}"))?;
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Cancel the active on-demand model integrity check.
pub async fn cancel_model_integrity(cfg: &ConnConfig) -> Result<(), String> {
    let url = format!(
        "{}/v1/model-management/integrity",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &url, "DELETE", None).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Inspect a trusted source and discover immutable, verification-ready metadata.
pub async fn inspect_model_download_source(
    cfg: &ConnConfig,
    source_url: &str,
) -> Result<ModelDownloadSource, String> {
    let endpoint = format!(
        "{}/v1/model-management/downloads/inspect",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&ModelDownloadSourceRequest { url: source_url })
        .map_err(|error| format!("failed to encode model source inspection request: {error}"))?;
    let response = request(cfg, &endpoint, "POST", Some(&body)).await?;
    if !response.ok() {
        return Err(read_response_error(&response).await);
    }
    let text = read_json_response_text(&response, MAX_MODEL_DOWNLOAD_SOURCE_RESPONSE_BYTES).await?;
    decode_model_download_source(&text)
}

fn decode_model_download_source(text: &str) -> Result<ModelDownloadSource, String> {
    if text.is_empty() || text.len() > MAX_MODEL_DOWNLOAD_SOURCE_RESPONSE_BYTES {
        return Err(format!(
            "model download source response must be between 1 and {MAX_MODEL_DOWNLOAD_SOURCE_RESPONSE_BYTES} bytes"
        ));
    }
    let source = serde_json::from_str::<ModelDownloadSource>(text)
        .map_err(|error| format!("invalid model download source response: {error}"))?;
    source.validate()?;
    Ok(source)
}

impl ModelDownloadSource {
    fn validate(&self) -> Result<(), String> {
        if self.object != "bloom.model_download_source" {
            return Err("unsupported model download source response".to_string());
        }
        if self.download_url.is_empty()
            || self.download_url.len() > MAX_MODEL_DOWNLOAD_SOURCE_URL_BYTES
            || self.download_url.contains(['?', '#'])
        {
            return Err("model download source returned an invalid download URL".to_string());
        }
        let Some(authority_and_path) = self.download_url.strip_prefix("https://") else {
            return Err("model download source returned a non-HTTPS URL".to_string());
        };
        let host = authority_and_path.split('/').next().unwrap_or_default();
        if !matches!(host, "huggingface.co" | "www.huggingface.co") {
            return Err("model download source returned an untrusted host".to_string());
        }
        if self.filename.is_empty()
            || self.filename.len() > 255
            || self.filename.starts_with('.')
            || self.filename.contains(['/', '\\', '%'])
            || !matches!(
                self.filename
                    .rsplit_once('.')
                    .map(|(_, extension)| extension.to_ascii_lowercase())
                    .as_deref(),
                Some("gguf" | "onnx" | "mlmodel")
            )
        {
            return Err("model download source returned an invalid filename".to_string());
        }
        if self.size_bytes == Some(0) {
            return Err("model download source returned an empty file size".to_string());
        }
        if self.sha256.as_ref().is_some_and(|value| {
            value.len() != 64
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }) {
            return Err("model download source returned an invalid SHA-256".to_string());
        }
        if self.commit_hash.as_ref().is_some_and(|value| {
            value.len() != 40
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }) {
            return Err("model download source returned an invalid commit hash".to_string());
        }
        if self.verification_ready != self.sha256.is_some() {
            return Err(
                "model download source returned inconsistent verification metadata".to_string(),
            );
        }
        if let Some(commit) = self.commit_hash.as_deref() {
            if !self.download_url.contains(&format!("/resolve/{commit}/")) {
                return Err(
                    "model download source URL is not pinned to its declared commit".to_string(),
                );
            }
        }
        match self.warning.as_deref() {
            Some(warning)
                if warning.trim().is_empty()
                    || warning.chars().count() > MAX_MODEL_DOWNLOAD_SOURCE_WARNING_CHARS =>
            {
                return Err("model download source returned an invalid warning".to_string())
            }
            None if !self.verification_ready => {
                return Err("unverified model download source did not include a warning".to_string())
            }
            _ => {}
        }
        Ok(())
    }
}

/// Start or resume a verified download into the server's model catalog.
pub async fn download_model(
    cfg: &ConnConfig,
    url: &str,
    filename: &str,
    sha256: &str,
    license: &str,
) -> Result<(), String> {
    let url_endpoint = format!(
        "{}/v1/model-management/downloads",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&ModelDownloadRequest {
        url,
        filename,
        sha256,
        license: non_empty(license),
    })
    .map_err(|error| format!("failed to encode model download request: {error}"))?;
    let response = request(cfg, &url_endpoint, "POST", Some(&body)).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Start a server-authoritative acquisition from a verified signed-index entry.
pub async fn download_index_model(cfg: &ConnConfig, model_index_id: &str) -> Result<(), String> {
    if model_index_id.is_empty()
        || model_index_id.len() > 64
        || !model_index_id.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (index > 0 && matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err("model index ID is invalid".to_string());
    }
    let endpoint = format!(
        "{}/v1/model-management/index/{model_index_id}/download",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &endpoint, "POST", None).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Cancel the active download while retaining its verified resume metadata.
pub async fn cancel_model_download(cfg: &ConnConfig) -> Result<(), String> {
    let url = format!(
        "{}/v1/model-management/downloads",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &url, "DELETE", None).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Resume a previously staged download without exposing its source URL.
pub async fn resume_model_download(
    cfg: &ConnConfig,
    filename: &str,
    license: &str,
) -> Result<(), String> {
    model_download_control(cfg, "resume", filename, license).await
}

/// Permanently discard the partial bytes and metadata for a staged download.
pub async fn discard_model_download(cfg: &ConnConfig, filename: &str) -> Result<(), String> {
    model_download_control(cfg, "discard", filename, "").await
}

async fn model_download_control(
    cfg: &ConnConfig,
    action: &str,
    filename: &str,
    license: &str,
) -> Result<(), String> {
    let url = format!(
        "{}/v1/model-management/downloads/{action}",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&serde_json::json!({
        "filename": filename,
        "license": non_empty(license)
    }))
    .map_err(|error| format!("failed to encode model download action: {error}"))?;
    let response = request(cfg, &url, "POST", Some(&body)).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

/// Upload a browser file in bounded Blob slices, then ask the server to verify and install it.
pub async fn import_model_file<F>(
    cfg: &ConnConfig,
    file: &web_sys::File,
    sha256: &str,
    metadata: ModelImportMetadata<'_>,
    chunk_bytes: usize,
    cancellation: &ModelImportCancellation,
    mut on_progress: F,
) -> Result<(), ModelImportClientError>
where
    F: FnMut(u64, u64),
{
    if chunk_bytes == 0 || chunk_bytes > MAX_MODEL_IMPORT_CHUNK_BYTES {
        return Err(ModelImportClientError::Request(format!(
            "Model import chunk size must be between 1 and {MAX_MODEL_IMPORT_CHUNK_BYTES} bytes."
        )));
    }
    let filename = file.name();
    let total_bytes = file.size() as u64;
    let begin_url = format!(
        "{}/v1/model-management/imports",
        cfg.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_string(&ModelImportRequest {
        filename: &filename,
        total_bytes,
        sha256,
        source_url: non_empty(metadata.source_url),
        license: non_empty(metadata.license),
    })
    .map_err(|error| {
        ModelImportClientError::Request(format!("failed to encode model import request: {error}"))
    })?;
    let begin =
        cancellable_request(cfg, &begin_url, "POST", Some(&body), None, cancellation).await?;
    let mut status = decode_model_import_response(begin).await?;
    if status.total_bytes != Some(total_bytes) || status.uploaded_bytes > total_bytes {
        return Err(ModelImportClientError::Request(
            "Server returned inconsistent model import progress.".to_string(),
        ));
    }
    let mut offset = status.uploaded_bytes;
    on_progress(offset, total_bytes);

    let encoded_filename = js_sys::encode_uri_component(&filename)
        .as_string()
        .ok_or_else(|| {
            ModelImportClientError::Request("failed to encode model filename".to_string())
        })?;
    let chunk_url = format!(
        "{}/v1/model-management/imports/{}",
        cfg.base_url.trim_end_matches('/'),
        encoded_filename
    );
    let blob = file.dyn_ref::<web_sys::Blob>().ok_or_else(|| {
        ModelImportClientError::Request("selected file is unavailable".to_string())
    })?;
    while offset < total_bytes {
        if cancellation.is_cancelled() {
            return Err(ModelImportClientError::Cancelled);
        }
        let end = offset.saturating_add(chunk_bytes as u64).min(total_bytes);
        let chunk = blob
            .slice_with_f64_and_f64(offset as f64, end as f64)
            .map_err(|error| {
                ModelImportClientError::Request(format!(
                    "failed to read model file slice: {error:?}"
                ))
            })?;
        let response = cancellable_request(
            cfg,
            &chunk_url,
            "PUT",
            None,
            Some((&chunk, offset)),
            cancellation,
        )
        .await?;
        status = decode_model_import_response(response).await?;
        if status.uploaded_bytes != end || status.total_bytes != Some(total_bytes) {
            return Err(ModelImportClientError::Request(
                "Server returned an unexpected model import offset.".to_string(),
            ));
        }
        offset = status.uploaded_bytes;
        on_progress(offset, total_bytes);
    }

    let complete_url = format!("{chunk_url}/complete");
    let response =
        cancellable_request(cfg, &complete_url, "POST", None, None, cancellation).await?;
    let completed = decode_model_import_response(response).await?;
    if completed.phase != "complete" {
        return Err(ModelImportClientError::Request(
            "Server did not complete model import installation.".to_string(),
        ));
    }
    Ok(())
}

/// Permanently discard a staged local-file import.
pub async fn discard_model_import(cfg: &ConnConfig, filename: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(filename)
        .as_string()
        .ok_or_else(|| "failed to encode model filename".to_string())?;
    let url = format!(
        "{}/v1/model-management/imports/{encoded}",
        cfg.base_url.trim_end_matches('/')
    );
    let response = request(cfg, &url, "DELETE", None).await?;
    if response.ok() {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

async fn cancellable_request(
    cfg: &ConnConfig,
    url: &str,
    method: &str,
    json_body: Option<&str>,
    blob_body: Option<(&web_sys::Blob, u64)>,
    cancellation: &ModelImportCancellation,
) -> Result<web_sys::Response, ModelImportClientError> {
    cfg.validate().map_err(ModelImportClientError::Request)?;
    if let Some(body) = json_body {
        validate_request_body_size(
            "Model import request",
            body.len(),
            MAX_HTTP_REQUEST_BODY_BYTES,
        )
        .map_err(ModelImportClientError::Request)?;
    }
    let headers = web_sys::Headers::new()
        .map_err(|error| ModelImportClientError::Request(format!("{error:?}")))?;
    auth_headers(cfg, &headers);
    let opts = web_sys::RequestInit::new();
    opts.set_method(method);
    opts.set_headers(&headers);
    opts.set_signal(Some(&cancellation.controller.signal()));
    if let Some(body) = json_body {
        headers.set("Content-Type", "application/json").ok();
        opts.set_body(&wasm_bindgen::JsValue::from_str(body));
    }
    if let Some((blob, offset)) = blob_body {
        headers.set("Content-Type", "application/octet-stream").ok();
        headers.set("Upload-Offset", &offset.to_string()).ok();
        opts.set_body(blob.as_ref());
    }
    match http_request(url, &opts).await {
        Ok(response) if response.ok() => Ok(response),
        Ok(response) => Err(ModelImportClientError::Request(
            read_response_error(&response).await,
        )),
        Err(_) if cancellation.is_cancelled() => Err(ModelImportClientError::Cancelled),
        Err(error) => Err(ModelImportClientError::Request(error)),
    }
}

async fn decode_model_import_response(
    response: web_sys::Response,
) -> Result<ModelImportStatus, ModelImportClientError> {
    let text = read_json_response_text(&response, MAX_MODEL_IMPORT_RESPONSE_BYTES)
        .await
        .map_err(ModelImportClientError::Request)?;
    serde_json::from_str::<ModelImportResponse>(&text)
        .map(|response| response.status)
        .map_err(|error| {
            ModelImportClientError::Request(format!("invalid model import response: {error}"))
        })
}

/// Stream one bounded mono PCM window through the active speech-to-text model.
pub async fn speech_to_text_stream<F>(
    cfg: &ConnConfig,
    model: &str,
    samples: &[f32],
    sample_rate: u32,
    cancellation: &ChatCancellation,
    mut on_update: F,
) -> Result<String, ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    cfg.validate().map_err(ChatStreamError::Request)?;
    validate_stream_model_id(model)?;
    if sample_rate != SPEECH_SAMPLE_RATE {
        return Err(ChatStreamError::Request(format!(
            "Speech audio must use a {SPEECH_SAMPLE_RATE} Hz sample rate."
        )));
    }
    if samples.is_empty() || samples.len() > MAX_SPEECH_SEGMENT_SAMPLES {
        return Err(ChatStreamError::Request(format!(
            "Speech audio must contain between 1 and {MAX_SPEECH_SEGMENT_SAMPLES} samples."
        )));
    }
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
    {
        return Err(ChatStreamError::Request(
            "Speech audio contains an invalid PCM sample.".to_string(),
        ));
    }

    let body = SpeechInferenceRequest {
        blocks: [SpeechDataBlock::AudioPcm {
            samples,
            sample_rate,
        }],
        params: SpeechInferenceParams {
            max_tokens: 1,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
        },
    };
    let body_text = serde_json::to_string(&body).map_err(|error| {
        ChatStreamError::Request(format!("failed to encode speech audio: {error}"))
    })?;
    validate_request_body_size("Speech request", body_text.len(), MAX_SPEECH_REQUEST_BYTES)
        .map_err(ChatStreamError::Request)?;

    let url = format!(
        "{}/v1/multimodal/stream",
        cfg.base_url.trim_end_matches('/')
    );
    let headers =
        web_sys::Headers::new().map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    auth_headers(cfg, &headers);
    headers.set("Content-Type", "application/json").ok();
    let opts = web_sys::RequestInit::new();
    opts.set_method("POST");
    opts.set_headers(&headers);
    opts.set_body(&wasm_bindgen::JsValue::from_str(&body_text));
    opts.set_signal(Some(&cancellation.controller.signal()));

    let response = match http_request(&url, &opts).await {
        Ok(response) => response,
        Err(_) if cancellation.is_cancelled() => return Err(ChatStreamError::Cancelled),
        Err(error) => return Err(ChatStreamError::Request(error)),
    };
    if !response.ok() {
        return Err(ChatStreamError::Request(
            read_response_error(&response).await,
        ));
    }

    let mut accumulated = String::new();
    let mut observed_model = None;
    let mut observed_request_id = None;
    let mut observed_asr_partial = false;
    consume_sse(&response, cancellation, |frame| {
        process_speech_frame(
            frame,
            model,
            &mut observed_model,
            &mut observed_request_id,
            &mut observed_asr_partial,
            &mut accumulated,
            &mut on_update,
        )
    })
    .await?;
    ensure_stream_model_observed(&observed_model)?;
    ensure_stream_request_id_observed(&observed_request_id)?;
    Ok(accumulated)
}

/// Upload one image and stream multimodal text output.
pub async fn multimodal_stream<F>(
    cfg: &ConnConfig,
    model: &str,
    prompt: &str,
    attachment: &ImageAttachment,
    options: ChatOptions,
    cancellation: &ChatCancellation,
    mut on_update: F,
) -> Result<String, ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    cfg.validate().map_err(ChatStreamError::Request)?;
    options.validate().map_err(ChatStreamError::Request)?;
    validate_stream_model_id(model)?;
    validate_multimodal_request(prompt, attachment).map_err(ChatStreamError::Request)?;
    if options.response_format != ResponseFormatMode::Text {
        return Err(ChatStreamError::Request(
            "Structured output is currently available for text chat only.".to_string(),
        ));
    }
    if !options.stop_sequences.is_empty() {
        return Err(ChatStreamError::Request(
            "Stop sequences are currently available for text chat only.".to_string(),
        ));
    }
    let url = format!(
        "{}/v1/multimodal/upload",
        cfg.base_url.trim_end_matches('/')
    );
    let form =
        web_sys::FormData::new().map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    form.append_with_str("model", model)
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    form.append_with_str("prompt", prompt)
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    form.append_with_str("max_tokens", &options.max_tokens.to_string())
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    form.append_with_str("temperature", &options.temperature.to_string())
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    form.append_with_str("top_p", &options.top_p.to_string())
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    if let Some(seed) = options.seed {
        form.append_with_str("seed", &seed.to_string())
            .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    }

    let bytes = js_sys::Uint8Array::from(attachment.bytes.as_slice());
    let parts = js_sys::Array::new();
    parts.push(&bytes);
    let blob_options = web_sys::BlobPropertyBag::new();
    blob_options.set_type(&attachment.mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(parts.as_ref(), &blob_options)
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    form.append_with_blob_and_filename("image", &blob, &attachment.name)
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;

    let headers =
        web_sys::Headers::new().map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    auth_headers(cfg, &headers);
    let opts = web_sys::RequestInit::new();
    opts.set_method("POST");
    opts.set_headers(&headers);
    opts.set_body(form.as_ref());
    opts.set_signal(Some(&cancellation.controller.signal()));
    let response = match http_request(&url, &opts).await {
        Ok(response) => response,
        Err(_) if cancellation.is_cancelled() => return Err(ChatStreamError::Cancelled),
        Err(error) => return Err(ChatStreamError::Request(error)),
    };
    if !response.ok() {
        return Err(ChatStreamError::Request(
            read_response_error(&response).await,
        ));
    }

    let mut accumulated = String::new();
    let mut observed_model = None;
    let mut observed_request_id = None;
    consume_sse(&response, cancellation, |frame| {
        process_multimodal_frame(
            frame,
            model,
            &mut observed_model,
            &mut observed_request_id,
            &mut accumulated,
            &mut on_update,
        )
    })
    .await?;
    ensure_stream_model_observed(&observed_model)?;
    ensure_stream_request_id_observed(&observed_request_id)?;
    Ok(accumulated)
}

/// Ask the server to cancel an in-flight request by ID.
pub async fn cancel_request(cfg: &ConnConfig, request_id: &str) -> Result<(), String> {
    validate_stream_request_id(request_id).map_err(str::to_string)?;
    let encoded_request_id = js_sys::encode_uri_component(request_id)
        .as_string()
        .ok_or_else(|| "failed to encode request ID".to_string())?;
    let url = format!(
        "{}/v1/cancel/{}",
        cfg.base_url.trim_end_matches('/'),
        encoded_request_id
    );
    let response = request(cfg, &url, "POST", None).await?;
    if response.ok() || response.status() == 404 {
        Ok(())
    } else {
        Err(read_response_error(&response).await)
    }
}

fn validate_chat_messages(messages: &[ChatMessage]) -> Result<(), String> {
    if messages.is_empty() {
        return Err("Chat request must contain at least one message.".to_string());
    }
    if messages.len() > MAX_CHAT_REQUEST_MESSAGES {
        return Err(format!(
            "Chat request contains more than {MAX_CHAT_REQUEST_MESSAGES} messages. Start a new conversation or remove older turns."
        ));
    }
    let mut content_bytes = 0_usize;
    for message in messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err("Chat request contains an unsupported message role.".to_string());
        }
        if message.role == "user" && message.content.chars().count() > MAX_CHAT_INPUT_CHARS {
            return Err(format!(
                "User message cannot exceed {MAX_CHAT_INPUT_CHARS} characters."
            ));
        }
        if message.role == "system" && message.content.chars().count() > MAX_SYSTEM_PROMPT_CHARS {
            return Err(format!(
                "System message cannot exceed {MAX_SYSTEM_PROMPT_CHARS} characters."
            ));
        }
        content_bytes = content_bytes
            .checked_add(message.content.len())
            .ok_or_else(|| "Chat history byte count overflowed.".to_string())?;
        if content_bytes > MAX_CHAT_CONTENT_BYTES {
            return Err(format!(
                "Chat history exceeds the {MAX_CHAT_CONTENT_BYTES}-byte content budget. Start a new conversation or shorten earlier messages."
            ));
        }
    }
    Ok(())
}

fn encode_chat_request(
    model: &str,
    messages: &[ChatMessage],
    options: &ChatOptions,
) -> Result<String, String> {
    validate_chat_messages(messages)?;
    let body = ChatRequest {
        model,
        messages,
        stream: true,
        stream_options: ChatStreamOptions {
            include_usage: true,
        },
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        seed: options.seed,
        stop: &options.stop_sequences,
        response_format: options.response_format_payload()?,
    };
    let body_text = serde_json::to_string(&body)
        .map_err(|error| format!("failed to encode chat request: {error}"))?;
    validate_request_body_size("Chat request", body_text.len(), MAX_CHAT_REQUEST_BYTES).map_err(
        |error| format!("{error} Start a new conversation or shorten earlier messages."),
    )?;
    Ok(body_text)
}

/// Validate a complete browser chat submission before changing conversation state.
///
/// Text requests return `None`. Multimodal requests return the exact bounded
/// prompt that should be passed to [`multimodal_stream`].
pub fn preflight_chat_submission(
    cfg: &ConnConfig,
    model: &str,
    messages: &[ChatMessage],
    options: &ChatOptions,
    attachment: Option<&ImageAttachment>,
) -> Result<Option<String>, String> {
    cfg.validate()?;
    options.validate()?;
    validate_model_id(model)?;
    if let Some(attachment) = attachment {
        if options.response_format != ResponseFormatMode::Text {
            return Err("Structured output is currently available for text chat only.".to_string());
        }
        if !options.stop_sequences.is_empty() {
            return Err("Stop sequences are currently available for text chat only.".to_string());
        }
        let prompt = format_multimodal_prompt(messages)?;
        validate_multimodal_request(&prompt, attachment)?;
        Ok(Some(prompt))
    } else {
        encode_chat_request(model, messages, options)?;
        Ok(None)
    }
}

/// Format a validated bounded text history for one multimodal request.
pub fn format_multimodal_prompt(messages: &[ChatMessage]) -> Result<String, String> {
    validate_chat_messages(messages)?;
    let mut prompt = String::new();
    for (index, message) in messages.iter().enumerate() {
        if index > 0 {
            append_request_text(&mut prompt, "\n", MAX_MULTIMODAL_PROMPT_BYTES)?;
        }
        append_request_text(&mut prompt, &message.role, MAX_MULTIMODAL_PROMPT_BYTES)?;
        append_request_text(&mut prompt, ": ", MAX_MULTIMODAL_PROMPT_BYTES)?;
        append_request_text(&mut prompt, &message.content, MAX_MULTIMODAL_PROMPT_BYTES)?;
    }
    append_request_text(&mut prompt, "\nassistant:", MAX_MULTIMODAL_PROMPT_BYTES)?;
    Ok(prompt)
}

fn append_request_text(text: &mut String, chunk: &str, max_bytes: usize) -> Result<(), String> {
    let next = text
        .len()
        .checked_add(chunk.len())
        .ok_or_else(|| "Request text byte count overflowed.".to_string())?;
    if next > max_bytes {
        return Err(format!(
            "Multimodal prompt exceeds the {max_bytes}-byte browser limit. Start a new conversation or shorten earlier messages."
        ));
    }
    text.push_str(chunk);
    Ok(())
}

fn validate_multimodal_request(prompt: &str, attachment: &ImageAttachment) -> Result<(), String> {
    validate_request_body_size(
        "Multimodal prompt",
        prompt.len(),
        MAX_MULTIMODAL_PROMPT_BYTES,
    )?;
    if attachment.bytes.is_empty() || attachment.bytes.len() as u64 > MAX_IMAGE_ATTACHMENT_BYTES {
        return Err(format!(
            "Image attachment must be between 1 byte and {MAX_IMAGE_ATTACHMENT_BYTES} bytes."
        ));
    }
    if !matches!(attachment.mime.as_str(), "image/jpeg" | "image/png") {
        return Err("Image attachment must be JPEG or PNG.".to_string());
    }
    if attachment.name.is_empty()
        || attachment.name.chars().count() > MAX_ATTACHMENT_NAME_CHARS
        || attachment.name.contains(['/', '\\'])
        || attachment.name.chars().any(char::is_control)
    {
        return Err("Image attachment has an invalid filename.".to_string());
    }
    Ok(())
}

/// Stream a chat completion and emit request metadata and text deltas.
pub async fn chat_stream<F>(
    cfg: &ConnConfig,
    model: &str,
    messages: &[ChatMessage],
    options: ChatOptions,
    cancellation: &ChatCancellation,
    mut on_update: F,
) -> Result<String, ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    cfg.validate().map_err(ChatStreamError::Request)?;
    options.validate().map_err(ChatStreamError::Request)?;
    validate_stream_model_id(model)?;
    validate_chat_messages(messages).map_err(ChatStreamError::Request)?;
    let url = format!("{}/v1/chat/completions", cfg.base_url.trim_end_matches('/'));
    let body_text =
        encode_chat_request(model, messages, &options).map_err(ChatStreamError::Request)?;

    let headers =
        web_sys::Headers::new().map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    auth_headers(cfg, &headers);
    headers.set("Content-Type", "application/json").ok();
    let opts = web_sys::RequestInit::new();
    opts.set_method("POST");
    opts.set_headers(&headers);
    opts.set_body(&wasm_bindgen::JsValue::from_str(&body_text));
    opts.set_signal(Some(&cancellation.controller.signal()));

    let response = match http_request(&url, &opts).await {
        Ok(response) => response,
        Err(_) if cancellation.is_cancelled() => return Err(ChatStreamError::Cancelled),
        Err(error) => return Err(ChatStreamError::Request(error)),
    };
    if !response.ok() {
        return Err(ChatStreamError::Request(
            read_response_error(&response).await,
        ));
    }

    let mut accumulated = String::new();
    let mut observed_model = None;
    let mut observed_request_id = None;
    consume_sse(&response, cancellation, |frame| {
        process_sse_frame(
            frame,
            model,
            &mut observed_model,
            &mut observed_request_id,
            &mut accumulated,
            &mut on_update,
        )
    })
    .await?;
    ensure_stream_model_observed(&observed_model)?;
    ensure_stream_request_id_observed(&observed_request_id)?;
    Ok(accumulated)
}

async fn consume_sse<F>(
    response: &web_sys::Response,
    cancellation: &ChatCancellation,
    mut on_frame: F,
) -> Result<(), ChatStreamError>
where
    F: FnMut(&str) -> Result<bool, ChatStreamError>,
{
    validate_sse_response_headers(response)?;
    let raw_body = response
        .body()
        .ok_or_else(|| ChatStreamError::Request("response has no body".into()))?;
    let reader = raw_body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| ChatStreamError::Request("failed to read response stream".into()))?;
    let mut buffer = String::new();
    let mut response_bytes = 0_usize;
    let decoder = web_sys::TextDecoder::new()
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    let decode_options = web_sys::TextDecodeOptions::new();
    decode_options.set_stream(true);

    loop {
        let chunk = match JsFuture::from(reader.read()).await {
            Ok(chunk) => chunk,
            Err(_) if cancellation.is_cancelled() => return Err(ChatStreamError::Cancelled),
            Err(error) => return Err(ChatStreamError::Request(format!("{error:?}"))),
        };
        let done = js_sys::Reflect::get(&chunk, &"done".into())
            .ok()
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        if done {
            break;
        }
        let value = js_sys::Reflect::get(&chunk, &"value".into())
            .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
        if value.is_undefined() {
            continue;
        }
        let bytes = js_sys::Uint8Array::new(&value);
        if let Err(error) = observe_stream_response_bytes(
            &mut response_bytes,
            bytes.length() as usize,
            MAX_SSE_RESPONSE_BYTES,
        ) {
            let _ = reader.cancel();
            return Err(error);
        }
        let text = decoder
            .decode_with_js_u8_array_and_options(&bytes, &decode_options)
            .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
        buffer.push_str(&text);

        loop {
            let frame = match take_sse_frame(&mut buffer) {
                Ok(Some(frame)) => frame,
                Ok(None) => break,
                Err(error) => {
                    let _ = reader.cancel();
                    return Err(error);
                }
            };
            match on_frame(&frame) {
                Ok(true) => {
                    let _ = reader.cancel();
                    return Ok(());
                }
                Ok(false) => {}
                Err(error) => {
                    let _ = reader.cancel();
                    return Err(error);
                }
            }
        }
    }

    let decoder_tail = decoder
        .decode()
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    buffer.push_str(&decoder_tail);
    if let Err(error) = validate_sse_frame_size(buffer.len(), MAX_SSE_FRAME_BYTES) {
        let _ = reader.cancel();
        return Err(error);
    }
    if !buffer.trim().is_empty() {
        match on_frame(&buffer) {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(error) => return Err(error),
        }
    }
    if cancellation.is_cancelled() {
        stream_eof_result(true)
    } else {
        stream_eof_result(false)
    }
}

fn stream_eof_result(cancelled: bool) -> Result<(), ChatStreamError> {
    if cancelled {
        Err(ChatStreamError::Cancelled)
    } else {
        Err(ChatStreamError::Request(
            "stream ended before its terminal [DONE] event".to_string(),
        ))
    }
}

fn validate_sse_response_headers(response: &web_sys::Response) -> Result<(), ChatStreamError> {
    let content_type = response
        .headers()
        .get("Content-Type")
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    validate_sse_content_type(content_type.as_deref())?;
    let content_length = response
        .headers()
        .get("Content-Length")
        .map_err(|error| ChatStreamError::Request(format!("{error:?}")))?;
    validate_sse_content_length(content_length.as_deref(), MAX_SSE_RESPONSE_BYTES)
}

fn validate_sse_content_type(content_type: Option<&str>) -> Result<(), ChatStreamError> {
    let is_event_stream = content_type
        .and_then(|value| value.split(';').next())
        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"));
    if is_event_stream {
        Ok(())
    } else {
        Err(ChatStreamError::Request(
            "stream response Content-Type must be text/event-stream".to_string(),
        ))
    }
}

fn validate_sse_content_length(
    content_length: Option<&str>,
    max_bytes: usize,
) -> Result<(), ChatStreamError> {
    let Some(content_length) = content_length else {
        return Ok(());
    };
    let length = content_length.trim().parse::<u64>().map_err(|_| {
        ChatStreamError::Request("stream response has an invalid Content-Length".to_string())
    })?;
    if length > max_bytes as u64 {
        Err(ChatStreamError::Request(format!(
            "stream response exceeds the {max_bytes}-byte limit"
        )))
    } else {
        Ok(())
    }
}

fn observe_stream_response_bytes(
    observed: &mut usize,
    chunk_bytes: usize,
    max_bytes: usize,
) -> Result<(), ChatStreamError> {
    let next = observed.checked_add(chunk_bytes).ok_or_else(|| {
        ChatStreamError::Request("stream response byte count overflowed".to_string())
    })?;
    if next > max_bytes {
        return Err(ChatStreamError::Request(format!(
            "stream response exceeds the {max_bytes}-byte limit"
        )));
    }
    *observed = next;
    Ok(())
}

fn process_sse_frame<F>(
    frame: &str,
    expected_model: &str,
    observed_model: &mut Option<String>,
    observed_request_id: &mut Option<String>,
    accumulated: &mut String,
    on_update: &mut F,
) -> Result<bool, ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    let Some(data) = sse_data(frame) else {
        return Ok(false);
    };
    if data == "[DONE]" {
        return Ok(true);
    }
    let chunk: serde_json::Value = serde_json::from_str(&data)
        .map_err(|error| ChatStreamError::Request(format!("invalid stream event: {error}")))?;
    if let Some(message) = chunk
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        validate_stream_error_message(message)?;
        return Err(ChatStreamError::Request(message.to_string()));
    }
    observe_stream_model(&chunk, expected_model, observed_model, on_update)?;
    ensure_stream_model_observed(observed_model)?;
    observe_stream_request_id(&chunk, observed_request_id, on_update)?;
    ensure_stream_request_id_observed(observed_request_id)?;
    if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
        let usage = serde_json::from_value::<ChatUsage>(usage.clone()).map_err(|error| {
            ChatStreamError::Request(format!("invalid stream usage metadata: {error}"))
        })?;
        if usage.prompt_tokens.checked_add(usage.completion_tokens) != Some(usage.total_tokens) {
            return Err(ChatStreamError::Request(
                "invalid stream usage metadata: total token count does not match".to_string(),
            ));
        }
        on_update(StreamUpdate::Usage(usage));
    }
    if let Some(delta) = chunk
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(serde_json::Value::as_str)
    {
        append_stream_delta(accumulated, delta, MAX_STREAM_OUTPUT_BYTES)?;
        on_update(StreamUpdate::TextDelta(delta.to_string()));
    }
    Ok(false)
}

fn process_multimodal_frame<F>(
    frame: &str,
    expected_model: &str,
    observed_model: &mut Option<String>,
    observed_request_id: &mut Option<String>,
    accumulated: &mut String,
    on_update: &mut F,
) -> Result<bool, ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    let Some(data) = sse_data(frame) else {
        return Ok(false);
    };
    if data == "[DONE]" {
        return Ok(true);
    }
    let event: serde_json::Value = serde_json::from_str(&data)
        .map_err(|error| ChatStreamError::Request(format!("invalid stream event: {error}")))?;
    if let Some(message) = event
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        validate_stream_error_message(message)?;
        return Err(ChatStreamError::Request(message.to_string()));
    }
    observe_stream_model(&event, expected_model, observed_model, on_update)?;
    ensure_stream_model_observed(observed_model)?;
    observe_stream_request_id(&event, observed_request_id, on_update)?;
    ensure_stream_request_id_observed(observed_request_id)?;
    let chunk = event.get("chunk");
    let delta = chunk
        .and_then(|chunk| chunk.get("TextDelta"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            chunk
                .and_then(|chunk| chunk.get("VlmToken"))
                .and_then(|token| token.get("text"))
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            chunk
                .and_then(|chunk| chunk.get("AsrPartial"))
                .and_then(|partial| partial.get("text"))
                .and_then(serde_json::Value::as_str)
        });
    if let Some(delta) = delta {
        append_stream_delta(accumulated, delta, MAX_STREAM_OUTPUT_BYTES)?;
        on_update(StreamUpdate::TextDelta(delta.to_string()));
    }
    Ok(false)
}

fn process_speech_frame<F>(
    frame: &str,
    expected_model: &str,
    observed_model: &mut Option<String>,
    observed_request_id: &mut Option<String>,
    observed_asr_partial: &mut bool,
    accumulated: &mut String,
    on_update: &mut F,
) -> Result<bool, ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    let Some(data) = sse_data(frame) else {
        return Ok(false);
    };
    if data == "[DONE]" {
        return Ok(true);
    }
    let event: serde_json::Value = serde_json::from_str(&data)
        .map_err(|error| ChatStreamError::Request(format!("invalid stream event: {error}")))?;
    if let Some(message) = event
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        validate_stream_error_message(message)?;
        return Err(ChatStreamError::Request(message.to_string()));
    }
    observe_stream_model(&event, expected_model, observed_model, on_update)?;
    ensure_stream_model_observed(observed_model)?;
    observe_stream_request_id(&event, observed_request_id, on_update)?;
    ensure_stream_request_id_observed(observed_request_id)?;

    let chunk = event.get("chunk");
    if let Some(partial) = chunk
        .and_then(|chunk| chunk.get("AsrPartial"))
        .and_then(|partial| partial.get("text"))
        .and_then(serde_json::Value::as_str)
    {
        *observed_asr_partial = true;
        append_speech_text(accumulated, partial, on_update)?;
    } else if !*observed_asr_partial {
        if let Some(delta) = chunk
            .and_then(|chunk| chunk.get("TextDelta"))
            .and_then(serde_json::Value::as_str)
        {
            append_speech_text(accumulated, delta, on_update)?;
        }
    }
    Ok(false)
}

fn append_speech_text<F>(
    accumulated: &mut String,
    text: &str,
    on_update: &mut F,
) -> Result<(), ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    // Some ASR runtimes emit cumulative hypotheses, while others emit deltas.
    // Only forward the new suffix for cumulative updates.
    let delta = text.strip_prefix(accumulated.as_str()).unwrap_or(text);
    if delta.is_empty() {
        return Ok(());
    }
    append_stream_delta(accumulated, delta, MAX_STREAM_OUTPUT_BYTES)?;
    on_update(StreamUpdate::TextDelta(delta.to_string()));
    Ok(())
}

fn append_stream_delta(
    accumulated: &mut String,
    delta: &str,
    max_bytes: usize,
) -> Result<(), ChatStreamError> {
    let next = accumulated.len().checked_add(delta.len()).ok_or_else(|| {
        ChatStreamError::Request("generated stream output byte count overflowed".to_string())
    })?;
    if next > max_bytes {
        return Err(ChatStreamError::Request(format!(
            "generated stream output exceeds the {max_bytes}-byte limit"
        )));
    }
    accumulated.push_str(delta);
    Ok(())
}

fn validate_stream_error_message(message: &str) -> Result<(), ChatStreamError> {
    if message.is_empty() || message.len() > MAX_STREAM_ERROR_MESSAGE_BYTES {
        Err(ChatStreamError::Request(
            "stream contains an invalid or oversized error message".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn validate_model_id(model: &str) -> Result<(), String> {
    if model.is_empty()
        || model.trim() != model
        || model.chars().take(MAX_STREAM_MODEL_ID_CHARS + 1).count() > MAX_STREAM_MODEL_ID_CHARS
        || model.chars().any(char::is_control)
    {
        return Err(
            "model identifier must contain 1 to 256 characters without surrounding whitespace or control characters"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_stream_model_id(model: &str) -> Result<(), ChatStreamError> {
    validate_model_id(model).map_err(ChatStreamError::Request)
}

fn ensure_stream_model_observed(observed_model: &Option<String>) -> Result<(), ChatStreamError> {
    observed_model.as_ref().map(|_| ()).ok_or_else(|| {
        ChatStreamError::Request("stream did not identify its execution model".to_string())
    })
}

fn validate_stream_request_id(request_id: &str) -> Result<(), &'static str> {
    if request_id.is_empty()
        || request_id.len() > MAX_STREAM_REQUEST_ID_CHARS
        || !request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(
            "request ID must contain 1 to 128 ASCII letters, digits, hyphens, or underscores",
        );
    }
    Ok(())
}

fn ensure_stream_request_id_observed(
    observed_request_id: &Option<String>,
) -> Result<(), ChatStreamError> {
    observed_request_id.as_ref().map(|_| ()).ok_or_else(|| {
        ChatStreamError::Request("stream did not identify its request ID".to_string())
    })
}

fn observe_stream_request_id<F>(
    event: &serde_json::Value,
    observed_request_id: &mut Option<String>,
    on_update: &mut F,
) -> Result<(), ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    let Some(value) = event.get("id") else {
        return Ok(());
    };
    let request_id = value.as_str().ok_or_else(|| {
        ChatStreamError::Request("stream contains invalid request ID metadata".to_string())
    })?;
    validate_stream_request_id(request_id)
        .map_err(|message| ChatStreamError::Request(message.to_string()))?;
    match observed_request_id {
        Some(previous) if previous != request_id => Err(ChatStreamError::Request(
            "stream changed its request ID".to_string(),
        )),
        Some(_) => Ok(()),
        None => {
            *observed_request_id = Some(request_id.to_string());
            on_update(StreamUpdate::RequestId(request_id.to_string()));
            Ok(())
        }
    }
}

fn observe_stream_model<F>(
    event: &serde_json::Value,
    expected_model: &str,
    observed_model: &mut Option<String>,
    on_update: &mut F,
) -> Result<(), ChatStreamError>
where
    F: FnMut(StreamUpdate),
{
    let Some(value) = event.get("model") else {
        return Ok(());
    };
    let model = value.as_str().ok_or_else(|| {
        ChatStreamError::Request("stream contains invalid model metadata".to_string())
    })?;
    validate_stream_model_id(model)?;
    if expected_model != "default" && model != expected_model {
        return Err(ChatStreamError::Request(
            "stream model does not match the requested model".to_string(),
        ));
    }
    match observed_model {
        Some(previous) if previous != model => Err(ChatStreamError::Request(
            "stream changed its execution model".to_string(),
        )),
        Some(_) => Ok(()),
        None => {
            *observed_model = Some(model.to_string());
            on_update(StreamUpdate::Model(model.to_string()));
            Ok(())
        }
    }
}

fn take_sse_frame(buffer: &mut String) -> Result<Option<String>, ChatStreamError> {
    let lf = buffer.find("\n\n").map(|position| (position, 2));
    let crlf = buffer.find("\r\n\r\n").map(|position| (position, 4));
    let (position, separator_len) = match (lf, crlf) {
        (Some(left), Some(right)) => left.min(right),
        (Some(found), None) | (None, Some(found)) => found,
        (None, None) => {
            let pending_limit = MAX_SSE_FRAME_BYTES.saturating_add(MAX_SSE_SEPARATOR_BYTES - 1);
            if buffer.len() > pending_limit {
                return Err(ChatStreamError::Request(format!(
                    "stream event exceeds the {MAX_SSE_FRAME_BYTES}-byte frame limit"
                )));
            }
            return Ok(None);
        }
    };
    validate_sse_frame_size(position, MAX_SSE_FRAME_BYTES)?;
    let frame = buffer[..position].to_string();
    buffer.drain(..position + separator_len);
    Ok(Some(frame))
}

fn validate_sse_frame_size(frame_bytes: usize, max_bytes: usize) -> Result<(), ChatStreamError> {
    if frame_bytes > max_bytes {
        Err(ChatStreamError::Request(format!(
            "stream event exceeds the {max_bytes}-byte frame limit"
        )))
    } else {
        Ok(())
    }
}

fn sse_data(frame: &str) -> Option<String> {
    let data = frame
        .lines()
        .filter_map(|line| line.trim_end_matches('\r').strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>();
    (!data.is_empty()).then(|| data.join("\n"))
}

async fn http_request(url: &str, opts: &web_sys::RequestInit) -> Result<web_sys::Response, String> {
    let window = web_sys::window().ok_or("browser window is unavailable")?;
    let request =
        web_sys::Request::new_with_str_and_init(url, opts).map_err(|error| format!("{error:?}"))?;
    let response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|error| format!("{error:?}"))?;
    response
        .dyn_into::<web_sys::Response>()
        .map_err(|_| "fetch returned an invalid response".to_string())
}

trait CancelNetworkWork {
    fn cancel_network_work(&self);
}

impl CancelNetworkWork for web_sys::AbortController {
    fn cancel_network_work(&self) {
        self.abort();
    }
}

impl CancelNetworkWork for web_sys::ReadableStreamDefaultReader {
    fn cancel_network_work(&self) {
        let _ = self.cancel();
    }
}

struct NetworkCancellationGuard<T: CancelNetworkWork> {
    inner: T,
    armed: bool,
}

impl<T: CancelNetworkWork> NetworkCancellationGuard<T> {
    fn new(inner: T) -> Self {
        Self { inner, armed: true }
    }

    fn inner(&self) -> &T {
        &self.inner
    }

    fn cancel(&mut self) {
        if self.armed {
            self.inner.cancel_network_work();
            self.armed = false;
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl<T: CancelNetworkWork> Drop for NetworkCancellationGuard<T> {
    fn drop(&mut self) {
        self.cancel();
    }
}

async fn timed_http_request(
    url: &str,
    opts: &web_sys::RequestInit,
    abort_guard: &mut NetworkCancellationGuard<web_sys::AbortController>,
) -> Result<web_sys::Response, String> {
    let fetch = http_request(url, opts).fuse();
    let timeout = TimeoutFuture::new(MAX_HTTP_REQUEST_WAIT_MS).fuse();
    futures_util::pin_mut!(fetch, timeout);
    futures_util::select_biased! {
        response = fetch => response,
        _ = timeout => {
            abort_guard.cancel();
            let timeout_seconds = MAX_HTTP_REQUEST_WAIT_MS / 1_000;
            Err(format!(
                "HTTP request timed out after {timeout_seconds} seconds before response headers were received"
            ))
        }
    }
}

async fn request(
    cfg: &ConnConfig,
    url: &str,
    method: &str,
    body: Option<&str>,
) -> Result<web_sys::Response, String> {
    cfg.validate()?;
    if let Some(body) = body {
        validate_request_body_size("HTTP request", body.len(), MAX_HTTP_REQUEST_BODY_BYTES)?;
    }
    let headers = web_sys::Headers::new().map_err(|error| format!("{error:?}"))?;
    auth_headers(cfg, &headers);
    let opts = web_sys::RequestInit::new();
    opts.set_method(method);
    opts.set_headers(&headers);
    let controller = web_sys::AbortController::new()
        .map_err(|error| format!("failed to create HTTP request cancellation: {error:?}"))?;
    let mut abort_guard = NetworkCancellationGuard::new(controller);
    opts.set_signal(Some(&abort_guard.inner().signal()));
    if let Some(body) = body {
        headers.set("Content-Type", "application/json").ok();
        opts.set_body(&wasm_bindgen::JsValue::from_str(body));
    }
    let response = timed_http_request(url, &opts, &mut abort_guard).await;
    if response.is_ok() {
        abort_guard.disarm();
    }
    response
}

fn validate_request_body_size(
    description: &str,
    body_bytes: usize,
    max_bytes: usize,
) -> Result<(), String> {
    if body_bytes > max_bytes {
        Err(format!(
            "{description} exceeds the {max_bytes}-byte browser limit."
        ))
    } else {
        Ok(())
    }
}

fn non_empty(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

fn response_error(status: u16, text: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(message) = value.get("error").and_then(|error| {
            error
                .get("message")
                .and_then(serde_json::Value::as_str)
                .or_else(|| error.as_str())
        }) {
            if !message.trim().is_empty() {
                return format_response_error_detail(status, message);
            }
        }
    }
    format_response_error_detail(status, text)
}

fn format_response_error_detail(status: u16, detail: &str) -> String {
    let detail = detail.trim();
    if detail.is_empty() {
        return format!("HTTP {status}");
    }
    if detail.len() > MAX_HTTP_ERROR_MESSAGE_BYTES {
        return format!(
            "HTTP {status}: response error detail exceeds the {MAX_HTTP_ERROR_MESSAGE_BYTES}-byte display limit"
        );
    }
    let detail = normalize_response_error_detail(detail);
    if detail.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status}: {detail}")
    }
}

fn normalize_response_error_detail(detail: &str) -> String {
    let mut normalized = String::with_capacity(detail.len());
    let mut needs_separator = false;
    for character in detail.chars() {
        if character.is_whitespace() || character.is_control() {
            needs_separator = !normalized.is_empty();
            continue;
        }
        if needs_separator {
            normalized.push(' ');
            needs_separator = false;
        }
        normalized.push(character);
    }
    normalized
}

fn valid_http_request_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_HTTP_REQUEST_ID_CHARS
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

fn retry_after_seconds(status: u16, value: Option<&str>) -> Option<u32> {
    if status != 429 {
        return None;
    }
    let value = value?;
    if value.is_empty() || value.len() > 3 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value
        .parse::<u32>()
        .ok()
        .filter(|seconds| (1..=MAX_HTTP_RETRY_AFTER_SECONDS).contains(seconds))
}

fn append_http_error_context(
    message: String,
    status: u16,
    request_id: Option<&str>,
    retry_after: Option<&str>,
) -> String {
    let mut context = Vec::with_capacity(2);
    if let Some(seconds) = retry_after_seconds(status, retry_after) {
        let unit = if seconds == 1 { "second" } else { "seconds" };
        context.push(format!("Retry after {seconds} {unit}"));
    }
    if let Some(request_id) = request_id.filter(|value| valid_http_request_id(value)) {
        context.push(format!("Request ID: {request_id}"));
    }
    if context.is_empty() {
        message
    } else {
        format!("{message} ({})", context.join("; "))
    }
}

async fn read_json_response_text(
    response: &web_sys::Response,
    max_bytes: usize,
) -> Result<String, String> {
    let content_type = match response.headers().get("Content-Type") {
        Ok(content_type) => content_type,
        Err(error) => {
            if let Some(body) = response.body().as_ref() {
                cancel_readable_stream(body);
            }
            return Err(format!("failed to read response Content-Type: {error:?}"));
        }
    };
    if let Err(error) = validate_json_content_type(content_type.as_deref()) {
        if let Some(body) = response.body().as_ref() {
            cancel_readable_stream(body);
        }
        return Err(error);
    }
    read_response_text(response, max_bytes).await
}

fn validate_json_content_type(content_type: Option<&str>) -> Result<(), String> {
    let Some(media_type) = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
    else {
        return Err(
            "response Content-Type must be application/json or application/*+json".to_string(),
        );
    };
    let is_json = media_type.eq_ignore_ascii_case("application/json")
        || media_type.split_once('/').is_some_and(|(kind, subtype)| {
            kind.eq_ignore_ascii_case("application")
                && subtype.rsplit_once('+').is_some_and(|(prefix, suffix)| {
                    !prefix.is_empty() && suffix.eq_ignore_ascii_case("json")
                })
        });
    if is_json {
        Ok(())
    } else {
        Err("response Content-Type must be application/json or application/*+json".to_string())
    }
}

async fn read_response_error(response: &web_sys::Response) -> String {
    let status = response.status();
    let request_id = response
        .headers()
        .get(HTTP_REQUEST_ID_HEADER)
        .ok()
        .flatten();
    let retry_after = response
        .headers()
        .get(HTTP_RETRY_AFTER_HEADER)
        .ok()
        .flatten();
    let message = match read_response_text(response, MAX_HTTP_ERROR_RESPONSE_BYTES).await {
        Ok(text) => response_error(status, &text),
        Err(error) => format!("HTTP {status}: {error}"),
    };
    append_http_error_context(
        message,
        status,
        request_id.as_deref(),
        retry_after.as_deref(),
    )
}

async fn read_response_text(
    response: &web_sys::Response,
    max_bytes: usize,
) -> Result<String, String> {
    let body = response.body();
    let content_length = match response.headers().get("Content-Length") {
        Ok(content_length) => content_length,
        Err(error) => {
            if let Some(body) = body.as_ref() {
                cancel_readable_stream(body);
            }
            return Err(format!("failed to read response Content-Length: {error:?}"));
        }
    };
    if let Err(error) = validate_response_content_length(content_length.as_deref(), max_bytes) {
        if let Some(body) = body.as_ref() {
            cancel_readable_stream(body);
        }
        return Err(error);
    }
    let Some(body) = body else {
        return Ok(String::new());
    };
    let decoder = web_sys::TextDecoder::new()
        .map_err(|error| format!("failed to create response text decoder: {error:?}"))?;
    let decode_options = web_sys::TextDecodeOptions::new();
    decode_options.set_stream(true);
    let mut reader = NetworkCancellationGuard::new(
        body.get_reader()
            .dyn_into::<web_sys::ReadableStreamDefaultReader>()
            .map_err(|_| "failed to read response body stream".to_string())?,
    );
    let mut observed_bytes = 0_usize;
    let mut text = String::new();
    let total_timeout = TimeoutFuture::new(MAX_HTTP_RESPONSE_TOTAL_WAIT_MS).fuse();
    futures_util::pin_mut!(total_timeout);

    loop {
        let read = JsFuture::from(reader.inner().read()).fuse();
        let idle_timeout = TimeoutFuture::new(MAX_HTTP_RESPONSE_IDLE_WAIT_MS).fuse();
        futures_util::pin_mut!(read, idle_timeout);
        let chunk = futures_util::select_biased! {
            chunk = read => match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    reader.cancel();
                    return Err(format!("failed to read response body: {error:?}"));
                }
            },
            _ = total_timeout => {
                reader.cancel();
                let timeout_seconds = MAX_HTTP_RESPONSE_TOTAL_WAIT_MS / 1_000;
                return Err(format!(
                    "response body timed out after {timeout_seconds} seconds"
                ));
            },
            _ = idle_timeout => {
                reader.cancel();
                let timeout_seconds = MAX_HTTP_RESPONSE_IDLE_WAIT_MS / 1_000;
                return Err(format!(
                    "response body was idle for {timeout_seconds} seconds"
                ));
            }
        };
        let done = match js_sys::Reflect::get(&chunk, &"done".into()) {
            Ok(value) => match value.as_bool() {
                Some(done) => done,
                None => {
                    reader.cancel();
                    return Err("response body chunk has an invalid completion flag".to_string());
                }
            },
            Err(error) => {
                reader.cancel();
                return Err(format!("failed to inspect response body chunk: {error:?}"));
            }
        };
        if done {
            break;
        }
        let value = match js_sys::Reflect::get(&chunk, &"value".into()) {
            Ok(value) => value,
            Err(error) => {
                reader.cancel();
                return Err(format!("failed to inspect response body chunk: {error:?}"));
            }
        };
        if value.is_undefined() {
            continue;
        }
        let bytes = js_sys::Uint8Array::new(&value);
        if let Err(error) =
            observe_response_body_bytes(&mut observed_bytes, bytes.length() as usize, max_bytes)
        {
            reader.cancel();
            return Err(error);
        }
        let decoded = match decoder.decode_with_js_u8_array_and_options(&bytes, &decode_options) {
            Ok(decoded) => decoded,
            Err(error) => {
                reader.cancel();
                return Err(format!("failed to decode response body: {error:?}"));
            }
        };
        if let Err(error) = append_response_text(&mut text, &decoded, max_bytes) {
            reader.cancel();
            return Err(error);
        }
    }

    let decoder_tail = decoder
        .decode()
        .map_err(|error| format!("failed to finish decoding response body: {error:?}"))?;
    append_response_text(&mut text, &decoder_tail, max_bytes)?;
    reader.disarm();
    Ok(text)
}

fn cancel_readable_stream(body: &web_sys::ReadableStream) {
    if let Ok(reader) = body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
    {
        let _ = reader.cancel();
    }
}

fn validate_response_content_length(
    content_length: Option<&str>,
    max_bytes: usize,
) -> Result<(), String> {
    let Some(content_length) = content_length else {
        return Ok(());
    };
    let length = content_length
        .trim()
        .parse::<u64>()
        .map_err(|_| "response has an invalid Content-Length".to_string())?;
    if length > max_bytes as u64 {
        Err(format!("response body exceeds the {max_bytes}-byte limit"))
    } else {
        Ok(())
    }
}

fn observe_response_body_bytes(
    observed: &mut usize,
    chunk_bytes: usize,
    max_bytes: usize,
) -> Result<(), String> {
    let next = observed
        .checked_add(chunk_bytes)
        .ok_or_else(|| "response body byte count overflowed".to_string())?;
    if next > max_bytes {
        return Err(format!("response body exceeds the {max_bytes}-byte limit"));
    }
    *observed = next;
    Ok(())
}

fn append_response_text(text: &mut String, chunk: &str, max_bytes: usize) -> Result<(), String> {
    let next = text
        .len()
        .checked_add(chunk.len())
        .ok_or_else(|| "decoded response text byte count overflowed".to_string())?;
    if next > max_bytes {
        return Err(format!(
            "decoded response text exceeds the {max_bytes}-byte limit"
        ));
    }
    text.push_str(chunk);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_connection(base_url: &str, api_key: &str) -> ConnConfig {
        ConnConfig {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            remember_api_key: false,
        }
    }

    fn test_attachment(name: &str, mime: &str, size: usize) -> ImageAttachment {
        ImageAttachment {
            name: name.to_string(),
            mime: mime.to_string(),
            bytes: vec![0; size],
        }
    }

    fn valid_readiness_value() -> serde_json::Value {
        serde_json::json!({
            "schema_version": READINESS_SCHEMA_VERSION,
            "object": READINESS_OBJECT,
            "protocol_version": READINESS_PROTOCOL_VERSION,
            "minimum_ui_protocol_version": READINESS_PROTOCOL_VERSION,
            "maximum_ui_protocol_version": READINESS_PROTOCOL_VERSION,
            "server_version": "0.1.0",
            "status": "ready",
            "progress": 100,
            "model": "tiny.gguf",
            "loading": false,
            "load_error": null,
            "input_modalities": ["text"],
            "model_tasks": ["generation"],
            "context_window": 4096,
            "in_flight_requests": 0,
            "available_permits": 4,
            "memory_pressure_high": false,
            "ram_utilization": 0.25,
            "future_additive_field": true
        })
    }

    #[test]
    fn readiness_decoder_requires_a_compatible_bounded_contract() {
        let readiness = decode_readiness(&valid_readiness_value().to_string()).unwrap();
        assert_eq!(readiness.server_version, "0.1.0");
        assert_eq!(readiness.model, "tiny.gguf");
        assert_eq!(readiness.model_tasks, vec!["generation"]);
        assert_eq!(readiness.context_window, Some(4096));

        for (field, value) in [
            ("schema_version", serde_json::json!(2)),
            ("object", serde_json::json!("another.server")),
        ] {
            let mut invalid = valid_readiness_value();
            invalid[field] = value;
            assert!(decode_readiness(&invalid.to_string())
                .unwrap_err()
                .contains("unsupported readiness contract"));
        }

        let mut future_compatible = valid_readiness_value();
        future_compatible["protocol_version"] = serde_json::json!(4);
        future_compatible["maximum_ui_protocol_version"] = serde_json::json!(4);
        assert_eq!(
            decode_readiness(&future_compatible.to_string())
                .unwrap()
                .protocol_version,
            4
        );

        let mut incompatible = valid_readiness_value();
        incompatible["protocol_version"] = serde_json::json!(4);
        incompatible["minimum_ui_protocol_version"] = serde_json::json!(4);
        incompatible["maximum_ui_protocol_version"] = serde_json::json!(4);
        assert!(decode_readiness(&incompatible.to_string())
            .unwrap_err()
            .contains("this UI implements protocol 3"));

        for (minimum, maximum, protocol) in [(0, 3, 3), (4, 3, 3), (3, 4, 5)] {
            let mut invalid = valid_readiness_value();
            invalid["minimum_ui_protocol_version"] = serde_json::json!(minimum);
            invalid["maximum_ui_protocol_version"] = serde_json::json!(maximum);
            invalid["protocol_version"] = serde_json::json!(protocol);
            assert!(decode_readiness(&invalid.to_string())
                .unwrap_err()
                .contains("compatibility range is invalid"));
        }

        for field in ["minimum_ui_protocol_version", "maximum_ui_protocol_version"] {
            let mut missing = valid_readiness_value();
            missing.as_object_mut().unwrap().remove(field);
            assert!(decode_readiness(&missing.to_string())
                .unwrap_err()
                .contains("missing field"));
        }

        let mut missing = valid_readiness_value();
        missing.as_object_mut().unwrap().remove("available_permits");
        assert!(decode_readiness(&missing.to_string())
            .unwrap_err()
            .contains("missing field"));

        let mut inconsistent = valid_readiness_value();
        inconsistent["loading"] = serde_json::Value::Bool(true);
        assert!(decode_readiness(&inconsistent.to_string())
            .unwrap_err()
            .contains("internally inconsistent"));

        let mut invalid_version = valid_readiness_value();
        invalid_version["server_version"] = serde_json::json!(" 0.1.0");
        assert!(decode_readiness(&invalid_version.to_string())
            .unwrap_err()
            .contains("server version"));

        assert!(
            decode_readiness(&"x".repeat(MAX_READINESS_RESPONSE_BYTES + 1))
                .unwrap_err()
                .contains("must be between")
        );
    }

    #[test]
    fn readiness_errors_preserve_compatibility_classification() {
        let authentication = ReadinessError::Authentication("invalid API key".to_string());
        let incompatible = ReadinessError::Incompatible("schema mismatch".to_string());
        let unavailable = ReadinessError::Unavailable("connection refused".to_string());
        assert!(authentication.is_authentication());
        assert!(!authentication.is_incompatible());
        assert!(incompatible.is_incompatible());
        assert!(!incompatible.is_authentication());
        assert!(!unavailable.is_incompatible());
        assert!(!unavailable.is_authentication());
        assert_eq!(authentication.to_string(), "invalid API key");
        assert_eq!(incompatible.to_string(), "schema mismatch");
        assert_eq!(unavailable.to_string(), "connection refused");
    }

    #[test]
    fn api_access_probe_requires_a_bounded_openai_models_contract() {
        decode_api_access_probe(r#"{"object":"list","data":[]}"#).unwrap();
        decode_api_access_probe(
            r#"{"object":"list","data":[{"id":"tiny.gguf","object":"model","created":1700000000,"owned_by":"bloom"}]}"#,
        )
        .unwrap();

        for invalid in [
            r#"{}"#,
            r#"{"object":"future","data":[]}"#,
            r#"{"object":"list","data":{}}"#,
            r#"{"object":"list","data":[{},{}]}"#,
            r#"{"object":"list","data":[{"id":" tiny","object":"model","created":1,"owned_by":"bloom"}]}"#,
            r#"{"object":"list","data":[{"id":"tiny","object":"model","created":0,"owned_by":"bloom"}]}"#,
            r#"{"object":"list","data":[{"id":"tiny","object":"model","created":1,"owned_by":"other"}]}"#,
        ] {
            assert!(
                decode_api_access_probe(invalid).is_err(),
                "unexpectedly accepted {invalid}"
            );
        }
        assert!(decode_api_access_probe(&"x".repeat(MAX_AUTH_PROBE_RESPONSE_BYTES + 1)).is_err());
    }

    #[test]
    fn connection_settings_accept_bounded_http_origins_and_paths() {
        for base_url in [
            "http://127.0.0.1:3000",
            "https://example.com/bloom/",
            "HTTP://localhost",
            "http://[::1]:8080",
        ] {
            test_connection(base_url, "token-123").validate().unwrap();
        }
        test_connection("https://example.com", "")
            .validate()
            .unwrap();
    }

    #[test]
    fn connection_settings_reject_ambiguous_or_unbounded_values() {
        for base_url in [
            "",
            " example.com",
            "example.com",
            "ftp://example.com",
            "https://user@example.com",
            "https://example.com?mode=1",
            "https://example.com#fragment",
            "https://:443",
            "http://example.com:",
            "http://example.com:70000",
            "http://::1",
            "http://[::1",
        ] {
            assert!(
                test_connection(base_url, "").validate().is_err(),
                "{base_url}"
            );
        }
        assert!(
            test_connection(&format!("http://{}", "a".repeat(MAX_BASE_URL_CHARS)), "")
                .validate()
                .is_err()
        );
        for api_key in [
            "contains space".to_string(),
            "caf\u{e9}".to_string(),
            "x".repeat(MAX_API_KEY_CHARS + 1),
        ] {
            assert!(test_connection("http://localhost", &api_key)
                .validate()
                .is_err());
        }
    }

    #[test]
    fn generation_settings_reject_oversized_system_prompts() {
        let options = ChatOptions {
            system_prompt: "x".repeat(MAX_SYSTEM_PROMPT_CHARS + 1),
            ..ChatOptions::default()
        };

        assert!(options.validate().is_err());
    }

    #[test]
    fn chat_message_admission_enforces_shape_and_content_budgets() {
        let valid = vec![
            ChatMessage {
                role: "system".into(),
                content: "Be concise.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
        ];
        validate_chat_messages(&valid).unwrap();
        assert!(validate_chat_messages(&[]).is_err());

        let invalid_role = vec![ChatMessage {
            role: "tool".into(),
            content: "output".into(),
        }];
        assert!(validate_chat_messages(&invalid_role).is_err());

        let too_many = vec![
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
            };
            MAX_CHAT_REQUEST_MESSAGES + 1
        ];
        assert!(validate_chat_messages(&too_many).is_err());

        let oversized_user = vec![ChatMessage {
            role: "user".into(),
            content: "x".repeat(MAX_CHAT_INPUT_CHARS + 1),
        }];
        assert!(validate_chat_messages(&oversized_user).is_err());

        let oversized_system = vec![ChatMessage {
            role: "system".into(),
            content: "x".repeat(MAX_SYSTEM_PROMPT_CHARS + 1),
        }];
        assert!(validate_chat_messages(&oversized_system).is_err());

        let oversized_history = vec![
            ChatMessage {
                role: "assistant".into(),
                content: "x".repeat(MAX_CHAT_CONTENT_BYTES / 2 + 1),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "x".repeat(MAX_CHAT_CONTENT_BYTES / 2 + 1),
            },
        ];
        assert!(validate_chat_messages(&oversized_history).is_err());
    }

    #[test]
    fn chat_submission_preflight_covers_text_and_multimodal_transport_limits() {
        let cfg = test_connection("http://localhost", "");
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Describe this.".into(),
        }];
        let options = ChatOptions::default();

        assert_eq!(
            preflight_chat_submission(&cfg, "tiny.gguf", &messages, &options, None).unwrap(),
            None
        );

        let attachment = test_attachment("photo.png", "image/png", 8);
        let prompt =
            preflight_chat_submission(&cfg, "tiny.gguf", &messages, &options, Some(&attachment))
                .unwrap()
                .unwrap();
        assert_eq!(prompt, "user: Describe this.\nassistant:");

        let structured = ChatOptions {
            response_format: ResponseFormatMode::JsonObject,
            ..ChatOptions::default()
        };
        assert!(preflight_chat_submission(
            &cfg,
            "tiny.gguf",
            &messages,
            &structured,
            Some(&attachment)
        )
        .is_err());
        let stopped = ChatOptions {
            stop_sequences: vec!["END".to_string()],
            ..ChatOptions::default()
        };
        assert!(preflight_chat_submission(
            &cfg,
            "tiny.gguf",
            &messages,
            &stopped,
            Some(&attachment)
        )
        .unwrap_err()
        .contains("text chat only"));
        assert!(preflight_chat_submission(&cfg, "tiny.gguf", &messages, &stopped, None).is_ok());
        assert!(preflight_chat_submission(&cfg, " bad ", &messages, &options, None).is_err());

        let too_many = vec![
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
            };
            MAX_CHAT_REQUEST_MESSAGES + 1
        ];
        assert!(preflight_chat_submission(&cfg, "tiny.gguf", &too_many, &options, None).is_err());
    }

    #[test]
    fn multimodal_prompt_is_bounded_and_preserves_roles() {
        let prompt = format_multimodal_prompt(&[
            ChatMessage {
                role: "system".into(),
                content: "Be concise.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Describe this.".into(),
            },
        ])
        .unwrap();

        assert_eq!(
            prompt,
            "system: Be concise.\nuser: Describe this.\nassistant:"
        );
        assert!(format_multimodal_prompt(&[]).is_err());
    }

    #[test]
    fn multimodal_request_admission_enforces_prompt_and_image_metadata() {
        let valid = test_attachment("photo.png", "image/png", 1);
        validate_multimodal_request("Describe this.", &valid).unwrap();
        assert!(
            validate_multimodal_request(&"x".repeat(MAX_MULTIMODAL_PROMPT_BYTES + 1), &valid)
                .is_err()
        );
        assert!(validate_multimodal_request(
            "Describe this.",
            &test_attachment("empty.png", "image/png", 0)
        )
        .is_err());
        assert!(validate_multimodal_request(
            "Describe this.",
            &test_attachment(
                "large.png",
                "image/png",
                MAX_IMAGE_ATTACHMENT_BYTES as usize + 1
            )
        )
        .is_err());
        assert!(validate_multimodal_request(
            "Describe this.",
            &test_attachment("photo.gif", "image/gif", 1)
        )
        .is_err());
        assert!(validate_multimodal_request(
            "Describe this.",
            &test_attachment("../photo.png", "image/png", 1)
        )
        .is_err());
    }

    #[test]
    fn encoded_request_admission_accepts_the_limit_and_rejects_overflow() {
        validate_request_body_size(
            "Chat request",
            MAX_CHAT_REQUEST_BYTES,
            MAX_CHAT_REQUEST_BYTES,
        )
        .unwrap();
        assert!(validate_request_body_size(
            "Chat request",
            MAX_CHAT_REQUEST_BYTES + 1,
            MAX_CHAT_REQUEST_BYTES
        )
        .is_err());
    }

    #[test]
    fn decoder_handles_lf_and_crlf_frames() {
        let mut buffer = "data: one\n\ndata: two\r\n\r\npartial".to_string();

        assert_eq!(
            take_sse_frame(&mut buffer).unwrap().as_deref(),
            Some("data: one")
        );
        assert_eq!(
            take_sse_frame(&mut buffer).unwrap().as_deref(),
            Some("data: two")
        );
        assert_eq!(buffer, "partial");
    }

    #[test]
    fn stream_admission_requires_sse_mime_and_bounded_length() {
        assert!(validate_sse_content_type(Some("text/event-stream")).is_ok());
        assert!(validate_sse_content_type(Some("Text/Event-Stream; charset=utf-8")).is_ok());
        assert!(validate_sse_content_type(None).is_err());
        assert!(validate_sse_content_type(Some("application/json")).is_err());

        assert!(validate_sse_content_length(None, 16).is_ok());
        assert!(validate_sse_content_length(Some("16"), 16).is_ok());
        assert!(validate_sse_content_length(Some("17"), 16).is_err());
        assert!(validate_sse_content_length(Some("invalid"), 16).is_err());
    }

    #[test]
    fn ordinary_json_admission_requires_a_json_media_type_and_bounded_waits() {
        assert!(validate_json_content_type(Some("application/json")).is_ok());
        assert!(validate_json_content_type(Some("Application/JSON; charset=utf-8")).is_ok());
        assert!(validate_json_content_type(Some("application/problem+json")).is_ok());
        assert!(validate_json_content_type(Some("application/vnd.bloom.v1+JSON")).is_ok());

        assert!(validate_json_content_type(None).is_err());
        assert!(validate_json_content_type(Some("")).is_err());
        assert!(validate_json_content_type(Some("text/json")).is_err());
        assert!(validate_json_content_type(Some("application/jsonp")).is_err());
        assert!(validate_json_content_type(Some("application/+json")).is_err());
        assert!(validate_json_content_type(Some("application/json, text/html")).is_err());
    }

    #[test]
    fn network_cancellation_guard_cancels_once_unless_settled() {
        struct MockCancellation(std::rc::Rc<std::cell::Cell<usize>>);

        impl CancelNetworkWork for MockCancellation {
            fn cancel_network_work(&self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let cancellations = std::rc::Rc::new(std::cell::Cell::new(0));
        {
            let _guard =
                NetworkCancellationGuard::new(MockCancellation(std::rc::Rc::clone(&cancellations)));
        }
        assert_eq!(cancellations.get(), 1);

        {
            let mut guard =
                NetworkCancellationGuard::new(MockCancellation(std::rc::Rc::clone(&cancellations)));
            guard.cancel();
            guard.cancel();
        }
        assert_eq!(cancellations.get(), 2);

        {
            let mut guard =
                NetworkCancellationGuard::new(MockCancellation(std::rc::Rc::clone(&cancellations)));
            guard.disarm();
        }
        assert_eq!(cancellations.get(), 2);
    }

    #[test]
    fn ordinary_response_budgets_fail_closed_and_remain_transactional() {
        assert!(validate_response_content_length(None, 16).is_ok());
        assert!(validate_response_content_length(Some(" 16 "), 16).is_ok());
        assert!(validate_response_content_length(Some("17"), 16).is_err());
        assert!(validate_response_content_length(Some("-1"), 16).is_err());
        assert!(validate_response_content_length(Some("1, 2"), 16).is_err());

        let mut observed = 4;
        observe_response_body_bytes(&mut observed, 6, 10).unwrap();
        assert_eq!(observed, 10);
        assert!(observe_response_body_bytes(&mut observed, 1, 10).is_err());
        assert_eq!(observed, 10);

        let mut decoded = "safe".to_string();
        append_response_text(&mut decoded, "!", 5).unwrap();
        assert_eq!(decoded, "safe!");
        assert!(append_response_text(&mut decoded, "extra", 5).is_err());
        assert_eq!(decoded, "safe!");

        let mut replacement = String::new();
        assert!(append_response_text(&mut replacement, "�", 2).is_err());
        assert!(replacement.is_empty());
    }

    #[test]
    fn stream_byte_and_frame_budgets_fail_closed() {
        let mut observed = 0;
        observe_stream_response_bytes(&mut observed, 7, 10).unwrap();
        assert_eq!(observed, 7);
        assert!(observe_stream_response_bytes(&mut observed, 4, 10).is_err());
        assert_eq!(observed, 7);

        let mut exact = format!("{}\n\n", "x".repeat(MAX_SSE_FRAME_BYTES));
        assert_eq!(
            take_sse_frame(&mut exact).unwrap().unwrap().len(),
            MAX_SSE_FRAME_BYTES
        );

        let mut delimited = format!("{}\n\n", "x".repeat(MAX_SSE_FRAME_BYTES + 1));
        assert!(take_sse_frame(&mut delimited).is_err());

        let mut unterminated = "x".repeat(MAX_SSE_FRAME_BYTES + MAX_SSE_SEPARATOR_BYTES);
        assert!(take_sse_frame(&mut unterminated).is_err());
    }

    #[test]
    fn stream_output_budget_is_transactional() {
        let mut accumulated = "hello".to_string();
        append_stream_delta(&mut accumulated, "!", 6).unwrap();
        assert_eq!(accumulated, "hello!");
        assert!(append_stream_delta(&mut accumulated, "extra", 6).is_err());
        assert_eq!(accumulated, "hello!");
    }

    #[test]
    fn stream_requires_explicit_terminal_event() {
        assert_eq!(stream_eof_result(true), Err(ChatStreamError::Cancelled));
        assert_eq!(
            stream_eof_result(false),
            Err(ChatStreamError::Request(
                "stream ended before its terminal [DONE] event".into()
            ))
        );
    }

    #[test]
    fn decoder_joins_multiline_data() {
        assert_eq!(
            sse_data("event: message\ndata: first\ndata: second").as_deref(),
            Some("first\nsecond")
        );
    }

    #[test]
    fn stream_frame_emits_request_id_and_delta() {
        let mut updates = Vec::new();
        let mut accumulated = String::new();
        let mut observed_model = None;
        let mut observed_request_id = None;
        let finished = process_sse_frame(
            r#"data: {"id":"chatcmpl-1","model":"tiny.gguf","choices":[{"delta":{"content":"Hello"}}]}"#,
            "tiny.gguf",
            &mut observed_model,
            &mut observed_request_id,
            &mut accumulated,
            &mut |update| updates.push(update),
        )
        .unwrap();

        assert!(!finished);
        assert_eq!(accumulated, "Hello");
        assert_eq!(
            updates,
            vec![
                StreamUpdate::Model("tiny.gguf".into()),
                StreamUpdate::RequestId("chatcmpl-1".into()),
                StreamUpdate::TextDelta("Hello".into())
            ]
        );
    }

    #[test]
    fn stream_frame_surfaces_server_errors() {
        let error = process_sse_frame(
            r#"data: {"error":{"message":"model failed"}}"#,
            "tiny.gguf",
            &mut None,
            &mut None,
            &mut String::new(),
            &mut |_| {},
        )
        .unwrap_err();

        assert_eq!(error, ChatStreamError::Request("model failed".into()));
    }

    #[test]
    fn stream_frame_emits_final_usage_metadata() {
        let mut updates = Vec::new();
        let mut observed_model = None;
        let mut observed_request_id = None;
        let finished = process_sse_frame(
            r#"data: {"id":"chatcmpl-1","model":"tiny.gguf","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}"#,
            "tiny.gguf",
            &mut observed_model,
            &mut observed_request_id,
            &mut String::new(),
            &mut |update| updates.push(update),
        )
        .unwrap();

        assert!(!finished);
        assert_eq!(
            updates,
            vec![
                StreamUpdate::Model("tiny.gguf".into()),
                StreamUpdate::RequestId("chatcmpl-1".into()),
                StreamUpdate::Usage(ChatUsage {
                    prompt_tokens: 12,
                    completion_tokens: 4,
                    total_tokens: 16,
                })
            ]
        );
    }

    #[test]
    fn stream_frame_rejects_inconsistent_usage_metadata() {
        let error = process_sse_frame(
            r#"data: {"id":"chatcmpl-1","model":"tiny.gguf","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":15}}"#,
            "tiny.gguf",
            &mut None,
            &mut None,
            &mut String::new(),
            &mut |_| {},
        )
        .unwrap_err();

        assert_eq!(
            error,
            ChatStreamError::Request(
                "invalid stream usage metadata: total token count does not match".into()
            )
        );
    }

    #[test]
    fn stream_rejects_content_before_model_identity() {
        let error = process_sse_frame(
            r#"data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Unbound"}}]}"#,
            "tiny.gguf",
            &mut None,
            &mut None,
            &mut String::new(),
            &mut |_| {},
        )
        .unwrap_err();

        assert_eq!(
            error,
            ChatStreamError::Request("stream did not identify its execution model".into())
        );
    }

    #[test]
    fn multimodal_frame_emits_vlm_text_and_request_id() {
        let mut updates = Vec::new();
        let mut accumulated = String::new();
        let mut observed_model = None;
        let mut observed_request_id = None;
        let finished = process_multimodal_frame(
            r#"data: {"id":"mms-1","model":"vision.gguf","chunk":{"VlmToken":{"text":"A cat","bounding_box":null}}}"#,
            "vision.gguf",
            &mut observed_model,
            &mut observed_request_id,
            &mut accumulated,
            &mut |update| updates.push(update),
        )
        .unwrap();

        assert!(!finished);
        assert_eq!(accumulated, "A cat");
        assert_eq!(
            updates,
            vec![
                StreamUpdate::Model("vision.gguf".into()),
                StreamUpdate::RequestId("mms-1".into()),
                StreamUpdate::TextDelta("A cat".into())
            ]
        );
    }

    #[test]
    fn multimodal_end_chunk_waits_for_http_terminal_event() {
        let mut observed_model = None;
        let mut observed_request_id = None;
        let mut accumulated = String::new();
        let finished = process_multimodal_frame(
            r#"data: {"id":"mms-1","model":"vision.gguf","chunk":"End"}"#,
            "vision.gguf",
            &mut observed_model,
            &mut observed_request_id,
            &mut accumulated,
            &mut |_| {},
        )
        .unwrap();
        assert!(!finished);

        assert!(process_multimodal_frame(
            "data: [DONE]",
            "vision.gguf",
            &mut observed_model,
            &mut observed_request_id,
            &mut accumulated,
            &mut |_| {},
        )
        .unwrap());
    }

    #[test]
    fn speech_frame_prefers_asr_partials_and_merges_cumulative_hypotheses() {
        let mut updates = Vec::new();
        let mut accumulated = String::new();
        let mut observed_model = None;
        let mut observed_request_id = None;
        let mut observed_asr_partial = false;

        for frame in [
            r#"data: {"id":"mms-1","model":"qwen-asr","chunk":{"AsrPartial":{"text":"hello","tokens":[]}}}"#,
            r#"data: {"id":"mms-1","model":"qwen-asr","chunk":{"AsrPartial":{"text":"hello world","tokens":[]}}}"#,
            r#"data: {"id":"mms-1","model":"qwen-asr","chunk":{"TextDelta":"hello world"}}"#,
        ] {
            process_speech_frame(
                frame,
                "qwen-asr",
                &mut observed_model,
                &mut observed_request_id,
                &mut observed_asr_partial,
                &mut accumulated,
                &mut |update| updates.push(update),
            )
            .unwrap();
        }

        assert_eq!(accumulated, "hello world");
        assert_eq!(
            updates,
            vec![
                StreamUpdate::Model("qwen-asr".into()),
                StreamUpdate::RequestId("mms-1".into()),
                StreamUpdate::TextDelta("hello".into()),
                StreamUpdate::TextDelta(" world".into()),
            ]
        );
    }

    #[test]
    fn speech_request_uses_bounded_inline_pcm_shape() {
        let samples = [0.0_f32, 0.5, -0.5];
        let body = serde_json::to_value(SpeechInferenceRequest {
            blocks: [SpeechDataBlock::AudioPcm {
                samples: &samples,
                sample_rate: SPEECH_SAMPLE_RATE,
            }],
            params: SpeechInferenceParams {
                max_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                seed: None,
            },
        })
        .unwrap();
        assert_eq!(
            body["blocks"][0]["AudioPcm"]["sample_rate"],
            SPEECH_SAMPLE_RATE
        );
        assert_eq!(body["blocks"][0]["AudioPcm"]["samples"][1], 0.5);
        assert_eq!(body["params"]["max_tokens"], 1);
    }

    #[test]
    fn stream_rejects_oversized_error_messages() {
        let frame = format!(
            "data: {}",
            serde_json::json!({
                "error": {"message": "x".repeat(MAX_STREAM_ERROR_MESSAGE_BYTES + 1)}
            })
        );
        assert_eq!(
            process_sse_frame(
                &frame,
                "tiny.gguf",
                &mut None,
                &mut None,
                &mut String::new(),
                &mut |_| {},
            ),
            Err(ChatStreamError::Request(
                "stream contains an invalid or oversized error message".into()
            ))
        );
    }

    #[test]
    fn stream_model_metadata_is_bounded_bound_and_consistent() {
        let mut observed = None;
        let mut updates = Vec::new();
        observe_stream_model(
            &serde_json::json!({"model": "tiny.gguf"}),
            "tiny.gguf",
            &mut observed,
            &mut |update| updates.push(update),
        )
        .unwrap();
        observe_stream_model(
            &serde_json::json!({"model": "tiny.gguf"}),
            "tiny.gguf",
            &mut observed,
            &mut |update| updates.push(update),
        )
        .unwrap();
        assert_eq!(observed.as_deref(), Some("tiny.gguf"));
        assert_eq!(updates, vec![StreamUpdate::Model("tiny.gguf".into())]);

        let mismatch = observe_stream_model(
            &serde_json::json!({"model": "other.gguf"}),
            "tiny.gguf",
            &mut None,
            &mut |_| {},
        )
        .unwrap_err();
        assert_eq!(
            mismatch,
            ChatStreamError::Request("stream model does not match the requested model".into())
        );

        let mut alias_observed = Some("tiny.gguf".to_string());
        let changed = observe_stream_model(
            &serde_json::json!({"model": "other.gguf"}),
            "default",
            &mut alias_observed,
            &mut |_| {},
        )
        .unwrap_err();
        assert_eq!(
            changed,
            ChatStreamError::Request("stream changed its execution model".into())
        );

        for invalid in ["", " tiny.gguf", "tiny.gguf ", "tiny\ngguf"] {
            assert!(validate_stream_model_id(invalid).is_err());
        }
        assert!(validate_stream_model_id(&"m".repeat(MAX_STREAM_MODEL_ID_CHARS + 1)).is_err());
        assert!(ensure_stream_model_observed(&observed).is_ok());
        assert_eq!(
            ensure_stream_model_observed(&None),
            Err(ChatStreamError::Request(
                "stream did not identify its execution model".into()
            ))
        );
    }

    #[test]
    fn stream_request_id_metadata_is_bounded_safe_and_consistent() {
        let mut observed = None;
        let mut updates = Vec::new();
        observe_stream_request_id(
            &serde_json::json!({"id": "chatcmpl-123_4"}),
            &mut observed,
            &mut |update| updates.push(update),
        )
        .unwrap();
        observe_stream_request_id(
            &serde_json::json!({"id": "chatcmpl-123_4"}),
            &mut observed,
            &mut |update| updates.push(update),
        )
        .unwrap();
        assert_eq!(observed.as_deref(), Some("chatcmpl-123_4"));
        assert_eq!(
            updates,
            vec![StreamUpdate::RequestId("chatcmpl-123_4".into())]
        );

        let changed = observe_stream_request_id(
            &serde_json::json!({"id": "chatcmpl-5"}),
            &mut observed,
            &mut |_| {},
        )
        .unwrap_err();
        assert_eq!(
            changed,
            ChatStreamError::Request("stream changed its request ID".into())
        );

        for invalid in [
            "",
            "../model-management/unload",
            "chatcmpl?admin=true",
            "chatcmpl%2Funload",
            "chatcmpl/other",
        ] {
            assert!(validate_stream_request_id(invalid).is_err());
        }
        assert!(validate_stream_request_id(&"r".repeat(MAX_STREAM_REQUEST_ID_CHARS + 1)).is_err());
        assert!(ensure_stream_request_id_observed(&observed).is_ok());
        assert_eq!(
            ensure_stream_request_id_observed(&None),
            Err(ChatStreamError::Request(
                "stream did not identify its request ID".into()
            ))
        );
    }

    #[test]
    fn stream_rejects_content_before_request_id() {
        let error = process_sse_frame(
            r#"data: {"model":"tiny.gguf","choices":[{"delta":{"content":"Unbound"}}]}"#,
            "tiny.gguf",
            &mut None,
            &mut None,
            &mut String::new(),
            &mut |_| {},
        )
        .unwrap_err();

        assert_eq!(
            error,
            ChatStreamError::Request("stream did not identify its request ID".into())
        );
    }

    #[test]
    fn generation_options_validate_supported_ranges() {
        assert!(ChatOptions::default().validate().is_ok());

        let invalid = ChatOptions {
            top_p: 0.0,
            ..ChatOptions::default()
        };
        assert_eq!(
            invalid.validate(),
            Err("Top P must be greater than 0 and at most 1.".to_string())
        );

        assert_eq!(
            parse_stop_sequences_setting(r#"["END", "\nUser:"]"#).unwrap(),
            vec!["END", "\nUser:"]
        );
        assert_eq!(
            parse_stop_sequences_setting("  ").unwrap(),
            Vec::<String>::new()
        );
        for invalid in [
            r#"[""]"#.to_string(),
            r#"["a", "b", "c", "d", "e"]"#.to_string(),
            r#"["ok", 3]"#.to_string(),
            serde_json::to_string(&vec!["x".repeat(MAX_STOP_SEQUENCE_CHARS + 1)]).unwrap(),
        ] {
            assert!(parse_stop_sequences_setting(&invalid).is_err());
        }
    }

    #[test]
    fn generation_options_keep_legacy_storage_compatible() {
        let legacy = serde_json::from_str::<ChatOptions>(
            r#"{"max_tokens":64,"temperature":0.2,"top_p":0.8,"seed":null,"system_prompt":""}"#,
        )
        .unwrap();

        assert_eq!(legacy.response_format, ResponseFormatMode::Text);
        assert!(legacy.json_schema.is_empty());
        assert!(legacy.stop_sequences.is_empty());
        assert!(legacy.validate().is_ok());
    }

    #[test]
    fn model_download_source_decoder_accepts_pinned_verified_metadata() {
        let commit = "ab".repeat(20);
        let sha256 = "cd".repeat(32);
        let text = serde_json::json!({
            "object": "bloom.model_download_source",
            "download_url": format!("https://huggingface.co/acme/repo/resolve/{commit}/model.gguf"),
            "filename": "model.gguf",
            "size_bytes": 483116416,
            "sha256": sha256,
            "commit_hash": commit,
            "verification_ready": true,
            "warning": null
        })
        .to_string();

        let source = decode_model_download_source(&text).unwrap();

        assert!(source.verification_ready);
        assert_eq!(source.size_bytes, Some(483116416));
        assert!(source.warning.is_none());
    }

    #[test]
    fn model_download_source_decoder_preserves_manual_hash_fallback() {
        let commit = "12".repeat(20);
        let text = serde_json::json!({
            "object": "bloom.model_download_source",
            "download_url": format!("https://huggingface.co/acme/repo/resolve/{commit}/model.onnx"),
            "filename": "model.onnx",
            "size_bytes": 4096,
            "sha256": null,
            "commit_hash": commit,
            "verification_ready": false,
            "warning": "Enter an independently obtained SHA-256 before downloading."
        })
        .to_string();

        let source = decode_model_download_source(&text).unwrap();

        assert!(!source.verification_ready);
        assert!(source.sha256.is_none());
        assert!(source.warning.is_some());
    }

    #[test]
    fn model_download_source_decoder_fails_closed_on_inconsistent_or_large_responses() {
        let inconsistent = r#"{
            "object":"bloom.model_download_source",
            "download_url":"https://huggingface.co/acme/repo/resolve/main/model.gguf",
            "filename":"model.gguf",
            "size_bytes":1,
            "sha256":null,
            "commit_hash":null,
            "verification_ready":true,
            "warning":null
        }"#;
        assert!(decode_model_download_source(inconsistent)
            .unwrap_err()
            .contains("inconsistent"));

        let unexpected = inconsistent.replace(
            "\"warning\":null",
            "\"warning\":null,\"api_key\":\"secret\"",
        );
        assert!(decode_model_download_source(&unexpected)
            .unwrap_err()
            .contains("unknown field"));
        assert!(decode_model_download_source(
            &"x".repeat(MAX_MODEL_DOWNLOAD_SOURCE_RESPONSE_BYTES + 1)
        )
        .unwrap_err()
        .contains("must be between"));
    }

    #[test]
    fn model_index_decoder_accepts_only_bounded_immutable_policy_consistent_entries() {
        let commit = "ab".repeat(20);
        let snapshot = serde_json::json!({
            "schema_version": 1,
            "object": "bloom.model_index",
            "key_id": "cd".repeat(32),
            "name": "Verified CPU Models",
            "generated_at": 1_785_513_600_u64,
            "expires_at": 1_816_963_200_u64,
            "source_kind": "https",
            "cache_status": "fresh",
            "warning": null,
            "data": [{
                "id": "tiny-q4",
                "name": "Tiny Q4",
                "description": "A compact test model.",
                "download_url": format!("https://huggingface.co/acme/tiny/resolve/{commit}/tiny-q4.gguf"),
                "filename": "tiny-q4.gguf",
                "format": "gguf",
                "size_bytes": 4096,
                "sha256": "ef".repeat(32),
                "license": "Apache-2.0",
                "family": "Llama",
                "parameter_count": 1_000_000_u64,
                "quantization": "Q4_K_M",
                "tags": ["chat", "cpu"],
                "downloadable": true,
                "blocking_reasons": []
            }]
        });

        let decoded = decode_model_index(&snapshot.to_string()).unwrap();
        assert_eq!(decoded.data[0].id, "tiny-q4");
        assert!(decoded.data[0].downloadable);

        let mut mutable = snapshot.clone();
        mutable["data"][0]["download_url"] = serde_json::Value::String(
            "https://huggingface.co/acme/tiny/resolve/main/tiny-q4.gguf".to_string(),
        );
        assert!(decode_model_index(&mutable.to_string())
            .unwrap_err()
            .contains("immutable commit"));

        let mut inconsistent = snapshot;
        inconsistent["data"][0]["downloadable"] = serde_json::Value::Bool(false);
        assert!(decode_model_index(&inconsistent.to_string())
            .unwrap_err()
            .contains("policy state"));
        assert!(
            decode_model_index(&"x".repeat(MAX_MODEL_INDEX_RESPONSE_BYTES + 1))
                .unwrap_err()
                .contains("must be between")
        );
    }

    #[test]
    fn model_index_decoder_validates_atomic_v2_package_manifests() {
        let commit = "12".repeat(20);
        let snapshot = serde_json::json!({
            "schema_version": 2,
            "object": "bloom.model_index",
            "key_id": "cd".repeat(32),
            "name": "Verified Model Packages",
            "generated_at": 1_785_513_600_u64,
            "expires_at": 1_816_963_200_u64,
            "source_kind": "https",
            "cache_status": "fresh",
            "warning": null,
            "data": [{
                "id": "tiny-package",
                "name": "Tiny Package",
                "description": "A compact multi-file model package.",
                "filename": "tiny-package",
                "format": "transformers",
                "size_bytes": 10,
                "sha256": "ef".repeat(32),
                "files": [
                    {
                        "download_url": format!("https://huggingface.co/acme/tiny/resolve/{commit}/config.json"),
                        "filename": "config.json",
                        "size_bytes": 3,
                        "sha256": "ab".repeat(32)
                    },
                    {
                        "download_url": format!("https://huggingface.co/acme/tiny/resolve/{commit}/model.safetensors"),
                        "filename": "model.safetensors",
                        "size_bytes": 7,
                        "sha256": "bc".repeat(32)
                    }
                ],
                "license": "Apache-2.0",
                "family": "Qwen2",
                "parameter_count": null,
                "quantization": null,
                "tags": ["chat", "package"],
                "downloadable": true,
                "blocking_reasons": []
            }]
        });

        let decoded = decode_model_index(&snapshot.to_string()).unwrap();
        assert_eq!(decoded.data[0].files.len(), 2);
        assert!(decoded.data[0].download_url.is_none());

        let mut missing_shard_index = snapshot.clone();
        missing_shard_index["data"][0]["files"][1]["filename"] =
            serde_json::Value::from("model-00001-of-00001.safetensors");
        missing_shard_index["data"][0]["files"][1]["download_url"] =
            serde_json::Value::from(format!(
                "https://huggingface.co/acme/tiny/resolve/{commit}/model-00001-of-00001.safetensors"
            ));
        assert!(decode_model_index(&missing_shard_index.to_string())
            .unwrap_err()
            .contains("incomplete Safetensors layout"));

        let mut wrong_total = snapshot.clone();
        wrong_total["data"][0]["size_bytes"] = serde_json::Value::from(11);
        assert!(decode_model_index(&wrong_total.to_string())
            .unwrap_err()
            .contains("incomplete"));

        let mut mixed_commit = snapshot.clone();
        mixed_commit["data"][0]["files"][1]["download_url"] = serde_json::Value::from(format!(
            "https://huggingface.co/acme/tiny/resolve/{}/model.safetensors",
            "34".repeat(20)
        ));
        assert!(decode_model_index(&mixed_commit.to_string())
            .unwrap_err()
            .contains("different commits"));

        let mut unsafe_path = snapshot;
        unsafe_path["data"][0]["files"][1]["filename"] =
            serde_json::Value::from("../model.safetensors");
        assert!(decode_model_index(&unsafe_path.to_string())
            .unwrap_err()
            .contains("package filename"));
    }

    #[test]
    fn structured_output_payloads_are_bounded_and_fail_closed() {
        let json_object = ChatOptions {
            response_format: ResponseFormatMode::JsonObject,
            ..ChatOptions::default()
        };
        assert_eq!(
            json_object.response_format_payload().unwrap(),
            Some(serde_json::json!({"type": "json_object"}))
        );

        let schema_text = r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}"#;
        let json_schema = ChatOptions {
            response_format: ResponseFormatMode::JsonSchema,
            json_schema: schema_text.into(),
            ..ChatOptions::default()
        };
        let payload = json_schema.response_format_payload().unwrap().unwrap();
        assert_eq!(payload["type"], "json_schema");
        assert_eq!(payload["json_schema"]["name"], "bloom_response");
        assert_eq!(payload["json_schema"]["strict"], true);
        assert_eq!(payload["json_schema"]["schema"]["type"], "object");

        let unsupported = ChatOptions {
            response_format: ResponseFormatMode::JsonSchema,
            json_schema: r#"{"type":"object","minProperties":1}"#.into(),
            ..ChatOptions::default()
        };
        assert!(unsupported
            .validate()
            .unwrap_err()
            .contains("unsupported keyword"));

        let oversized = ChatOptions {
            response_format: ResponseFormatMode::JsonSchema,
            json_schema: "x".repeat(MAX_RESPONSE_JSON_SCHEMA_BYTES + 1),
            ..ChatOptions::default()
        };
        assert!(oversized.validate().is_err());
    }

    #[test]
    fn readiness_context_window_is_required_but_nullable() {
        let current = decode_readiness(&valid_readiness_value().to_string()).unwrap();
        assert_eq!(current.context_window, Some(4_096));

        let mut without_model = valid_readiness_value();
        without_model["status"] = serde_json::json!("not_ready");
        without_model["model"] = serde_json::json!("not loaded");
        without_model["model_tasks"] = serde_json::json!([]);
        without_model["context_window"] = serde_json::Value::Null;
        let without_model = decode_readiness(&without_model.to_string()).unwrap();
        assert_eq!(without_model.context_window, None);

        let mut missing = valid_readiness_value();
        missing.as_object_mut().unwrap().remove("context_window");
        assert!(decode_readiness(&missing.to_string())
            .unwrap_err()
            .contains("missing field"));
    }

    #[test]
    fn readiness_model_tasks_are_required_bounded_and_consistent() {
        let mut embedding = valid_readiness_value();
        embedding["model_tasks"] = serde_json::json!(["embedding", "rerank"]);
        assert_eq!(
            decode_readiness(&embedding.to_string())
                .unwrap()
                .model_tasks,
            vec!["embedding", "rerank"]
        );

        for tasks in [
            serde_json::json!([]),
            serde_json::json!(["future"]),
            serde_json::json!(["generation", "generation"]),
            serde_json::json!(["rerank"]),
        ] {
            let mut invalid = valid_readiness_value();
            invalid["model_tasks"] = tasks;
            assert!(decode_readiness(&invalid.to_string()).is_err());
        }

        let mut missing = valid_readiness_value();
        missing.as_object_mut().unwrap().remove("model_tasks");
        assert!(decode_readiness(&missing.to_string())
            .unwrap_err()
            .contains("missing field"));
    }

    #[test]
    fn embedding_playground_inputs_and_responses_are_bounded() {
        assert_eq!(
            parse_embedding_lines(" first input \n\nsecond input\n").unwrap(),
            vec!["first input", "second input"]
        );
        assert!(parse_embedding_lines(" \n ").is_err());
        assert!(parse_embedding_lines(&"x\n".repeat(MAX_EMBEDDING_INPUTS + 1)).is_err());

        let response = serde_json::json!({
            "object": "list",
            "data": [
                {"object":"embedding","embedding":[1.0,0.0],"index":0},
                {"object":"embedding","embedding":[0.0,1.0],"index":1}
            ],
            "model": "encoder",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        });
        let inputs = vec!["first input".to_string(), "second input".to_string()];
        let decoded =
            decode_embedding_response(&response.to_string(), "encoder", &inputs, Some(2)).unwrap();
        assert_eq!(decoded.vectors.len(), 2);
        assert_eq!(decoded.vectors[1].input, "second input");
        assert_eq!(decoded.vectors[1].values, vec![0.0, 1.0]);
        assert_eq!(decoded.prompt_tokens, 5);
        assert_eq!(
            embedding_vector_clipboard_text(&decoded.vectors[1]).unwrap(),
            "[0.0,1.0]"
        );
        let exported = encode_embedding_export(&decoded).unwrap();
        let artifact: serde_json::Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(artifact["schema_version"], 1);
        assert_eq!(artifact["object"], "bloom.embedding_result");
        assert_eq!(artifact["vectors"][0]["input"], "first input");

        let mut wrong_model = response.clone();
        wrong_model["model"] = serde_json::json!("other");
        assert!(
            decode_embedding_response(&wrong_model.to_string(), "encoder", &inputs, Some(2))
                .is_err()
        );
        let mut wrong_norm = response;
        wrong_norm["data"][0]["embedding"] = serde_json::json!([2.0, 0.0]);
        assert!(
            decode_embedding_response(&wrong_norm.to_string(), "encoder", &inputs, Some(2))
                .unwrap_err()
                .contains("L2-normalized")
        );

        let mut invalid_export = decoded;
        invalid_export.vectors[1].index = 0;
        assert!(encode_embedding_export(&invalid_export)
            .unwrap_err()
            .contains("non-contiguous"));
    }

    #[test]
    fn rerank_playground_validates_exact_documents_and_stable_order() {
        let (query, documents, top_n) =
            prepare_rerank_input(" local runtime ", " first \nsecond\nthird", 2).unwrap();
        assert_eq!(query, "local runtime");
        assert_eq!(documents, vec!["first", "second", "third"]);
        assert_eq!(top_n, 2);
        assert!(prepare_rerank_input("", "first", 1).is_err());
        assert!(prepare_rerank_input("query", "first", 2).is_err());

        let response = serde_json::json!({
            "id": "rerank-1",
            "object": "rerank",
            "model": "encoder",
            "results": [
                {"index":1,"relevance_score":0.75,"document":{"text":"second"}},
                {"index":0,"relevance_score":0.25,"document":{"text":"first"}}
            ],
            "usage": {"prompt_tokens": 9, "total_tokens": 9}
        });
        let decoded = decode_rerank_response(
            &response.to_string(),
            "encoder",
            "local runtime",
            &["first".into(), "second".into(), "third".into()],
            2,
        )
        .unwrap();
        assert_eq!(decoded.query, "local runtime");
        assert_eq!(decoded.results[0].index, 1);
        assert_eq!(decoded.results[0].document, "second");
        assert_eq!(
            rerank_result_clipboard_text(&decoded.results[0]).unwrap(),
            r#"{"index":1,"relevance_score":0.75,"document":"second"}"#
        );
        let exported = encode_rerank_export(&decoded).unwrap();
        let artifact: serde_json::Value = serde_json::from_str(&exported).unwrap();
        assert_eq!(artifact["object"], "bloom.rerank_result");
        assert_eq!(artifact["query"], "local runtime");
        let mut invalid_export = decoded.clone();
        invalid_export.results.reverse();
        assert!(encode_rerank_export(&invalid_export)
            .unwrap_err()
            .contains("stable score order"));

        let mut out_of_order = response.clone();
        out_of_order["results"].as_array_mut().unwrap().reverse();
        assert!(decode_rerank_response(
            &out_of_order.to_string(),
            "encoder",
            "local runtime",
            &["first".into(), "second".into(), "third".into()],
            2,
        )
        .is_err());
        let mut substituted = response;
        substituted["results"][0]["document"]["text"] = serde_json::json!("changed");
        assert!(decode_rerank_response(
            &substituted.to_string(),
            "encoder",
            "local runtime",
            &["first".into(), "second".into(), "third".into()],
            2,
        )
        .unwrap_err()
        .contains("does not match"));
    }

    #[test]
    fn observability_snapshot_is_bounded_versioned_and_exportable() {
        let text = r#"{
            "schema_version":1,
            "object":"bloom.observability_snapshot",
            "created":1775000000,
            "server":{"version":"0.1.0","uptime_seconds":42},
            "model":"tiny.gguf",
            "ready":true,
            "load":{"phase":"ready","progress":100,"requested_model":"tiny.gguf","failure_present":false},
            "speculative_mode":"none",
            "requests":{"total":4,"completed":3,"failed":1,"in_flight":0},
            "tokens":{"prompt_total":30,"generated_total":12},
            "scheduler":{"ifb_enabled":false,"prefill_queue":0,"decoding_queue":0,"active_requests":0},
            "startup_memory_estimate":{"weight_bytes":1024,"host_weight_bytes":1024,"device_weight_bytes":0,"kv_cache_bytes":512,"kv_cache_bytes_per_token":64,"temp_tensor_bytes":256,"total_bytes":1792,"weight_dtype":"Q4","quantization":null,"kv_cache_dtype":"F16","num_layers":2,"offloaded_layers":null,"mmap_residency_applied":true,"memory_scope":"estimated runtime allocation"},
            "kv_cache":{"total_blocks":10,"free_blocks":8,"active_blocks":1,"cached_blocks":1,"hits":2,"misses":1,"evictions":0,"reuses":1,"utilization":0.2},
            "cachemesh":null,
            "memory":{"total_vram":0,"used_vram":0,"total_ram":8192,"used_ram":1024,"peak_vram":0,"peak_ram":2048,"device_name":""}
        }"#;

        let snapshot = decode_observability_snapshot(text).unwrap();
        assert_eq!(snapshot.server.uptime_seconds, 42);
        assert_eq!(snapshot.requests.completed, 3);
        assert_eq!(snapshot.startup_memory_estimate.unwrap().total_bytes, 1792);
        let exported = decode_observability_snapshot(text)
            .unwrap()
            .to_pretty_json()
            .unwrap();
        assert!(exported.ends_with('\n'));
        assert!(exported.contains("bloom.observability_snapshot"));
        assert!(exported.contains("mmap_residency_applied"));
        assert!(!exported.contains("api_key"));
    }

    #[test]
    fn observability_snapshot_rejects_unknown_versions_and_invalid_metrics() {
        let unknown = r#"{"schema_version":2,"object":"bloom.observability_snapshot"}"#;
        assert_eq!(
            decode_observability_snapshot(unknown),
            Err("unsupported observability snapshot version".to_string())
        );

        let invalid = r#"{
            "schema_version":1,
            "object":"bloom.observability_snapshot",
            "server":{"version":"0.1.0"},
            "model":"not loaded",
            "load":{"phase":"failed","failure_present":false},
            "speculative_mode":"none"
        }"#;
        let error = decode_observability_snapshot(invalid).unwrap_err();
        assert!(error.starts_with("invalid observability response: missing field"));

        let inconsistent = ObservabilitySnapshot {
            schema_version: 1,
            object: "bloom.observability_snapshot".to_string(),
            server: ObservabilityServer {
                version: "0.1.0".to_string(),
                uptime_seconds: 0,
            },
            model: "not loaded".to_string(),
            load: ObservabilityLoad {
                phase: "failed".to_string(),
                failure_present: false,
                ..ObservabilityLoad::default()
            },
            speculative_mode: "none".to_string(),
            ..ObservabilitySnapshot::default()
        };
        assert_eq!(
            inconsistent.validate(),
            Err("observability snapshot contains inconsistent failure state".to_string())
        );

        let unexpected = r#"{
            "schema_version":1,
            "object":"bloom.observability_snapshot",
            "created":1,
            "server":{"version":"0.1.0","uptime_seconds":1,"api_key":"secret"}
        }"#;
        assert!(decode_observability_snapshot(unexpected)
            .unwrap_err()
            .contains("unknown field `api_key`"));

        assert!(
            decode_observability_snapshot(&"x".repeat(MAX_OBSERVABILITY_RESPONSE_BYTES + 1))
                .unwrap_err()
                .contains("must be between")
        );
    }

    #[test]
    fn chat_request_opts_into_final_stream_usage() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let stop = vec!["END".to_string()];
        let request = ChatRequest {
            model: "tiny.gguf",
            messages: &messages,
            stream: true,
            stream_options: ChatStreamOptions {
                include_usage: true,
            },
            max_tokens: 64,
            temperature: 0.7,
            top_p: 0.9,
            seed: None,
            stop: &stop,
            response_format: None,
        };
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["model"], "tiny.gguf");
        assert_eq!(value["stream_options"]["include_usage"], true);
        assert_eq!(value["stop"], serde_json::json!(["END"]));
    }

    #[test]
    fn response_errors_extract_openai_message() {
        assert_eq!(
            response_error(
                409,
                r#"{"error":{"message":"A model load is already running."}}"#
            ),
            "HTTP 409: A model load is already running."
        );
        assert_eq!(response_error(500, ""), "HTTP 500");
        assert_eq!(
            response_error(502, "  upstream\n\u{1b}[31m\tfailed  "),
            "HTTP 502: upstream [31m failed"
        );

        let oversized = response_error(
            503,
            &serde_json::json!({
                "error": { "message": "x".repeat(MAX_HTTP_ERROR_MESSAGE_BYTES + 1) }
            })
            .to_string(),
        );
        assert_eq!(
            oversized,
            format!(
                "HTTP 503: response error detail exceeds the {MAX_HTTP_ERROR_MESSAGE_BYTES}-byte display limit"
            )
        );
        assert!(!oversized.contains("xxxx"));
    }

    #[test]
    fn response_errors_append_only_safe_bounded_http_request_ids() {
        let message = "HTTP 503: Model is not loaded.".to_string();
        assert_eq!(
            append_http_error_context(
                message.clone(),
                503,
                Some("8f8fefc0-7676-4de0-b18b-68f9748ab06e"),
                None,
            ),
            "HTTP 503: Model is not loaded. (Request ID: 8f8fefc0-7676-4de0-b18b-68f9748ab06e)"
        );
        assert_eq!(
            append_http_error_context(message.clone(), 503, Some("proxy.node_1:request-42"), None,),
            "HTTP 503: Model is not loaded. (Request ID: proxy.node_1:request-42)"
        );
        assert_eq!(
            append_http_error_context(
                message.clone(),
                503,
                Some(&"a".repeat(MAX_HTTP_REQUEST_ID_CHARS)),
                None,
            ),
            format!(
                "HTTP 503: Model is not loaded. (Request ID: {})",
                "a".repeat(MAX_HTTP_REQUEST_ID_CHARS)
            )
        );

        for invalid in [
            "",
            "request id",
            "request/id",
            "request?id",
            "request\nid",
            "réquest-id",
        ] {
            assert_eq!(
                append_http_error_context(message.clone(), 503, Some(invalid), None),
                message,
                "unexpectedly accepted {invalid:?}"
            );
        }
        assert_eq!(
            append_http_error_context(
                message.clone(),
                503,
                Some(&"a".repeat(MAX_HTTP_REQUEST_ID_CHARS + 1)),
                None,
            ),
            message
        );
        assert_eq!(
            append_http_error_context(message.clone(), 503, None, None),
            message
        );
    }

    #[test]
    fn response_errors_surface_only_bounded_delta_second_retry_hints() {
        let message = "HTTP 429: Server is busy.".to_string();
        assert_eq!(
            append_http_error_context(message.clone(), 429, None, Some("1")),
            "HTTP 429: Server is busy. (Retry after 1 second)"
        );
        assert_eq!(
            append_http_error_context(
                message.clone(),
                429,
                Some("support-request_42"),
                Some("300"),
            ),
            "HTTP 429: Server is busy. (Retry after 300 seconds; Request ID: support-request_42)"
        );

        for invalid in ["", "0", "301", "+1", " 1", "1 ", "1.0", "Wed, 21 Oct 2015"] {
            assert_eq!(
                append_http_error_context(message.clone(), 429, None, Some(invalid)),
                message,
                "unexpectedly accepted {invalid:?}"
            );
        }
        assert_eq!(
            append_http_error_context(message.clone(), 503, None, Some("1")),
            message
        );
    }

    #[test]
    fn correlated_response_errors_keep_body_and_request_id_bounds_independent() {
        let oversized = response_error(
            503,
            &serde_json::json!({
                "error": { "message": "x".repeat(MAX_HTTP_ERROR_MESSAGE_BYTES + 1) }
            })
            .to_string(),
        );
        let correlated =
            append_http_error_context(oversized, 503, Some("support-request_42"), None);
        assert_eq!(
            correlated,
            format!(
                "HTTP 503: response error detail exceeds the {MAX_HTTP_ERROR_MESSAGE_BYTES}-byte display limit (Request ID: support-request_42)"
            )
        );
        assert!(!correlated.contains("xxxx"));
    }

    #[test]
    fn model_catalog_decodes_download_capability_and_progress() {
        let example = include_str!("../../examples/model-catalog.json");
        let catalog = decode_model_catalog(example).unwrap();

        assert_eq!(catalog.schema_version, MODEL_CATALOG_SCHEMA_VERSION);
        assert_eq!(catalog.object, MODEL_CATALOG_OBJECT);
        assert!(catalog.download.enabled);
        assert!(catalog.download.license_policy.enforced);
        assert_eq!(catalog.download.license_policy.allowed.len(), 2);
        let provenance = catalog.data[0].provenance.as_ref().unwrap();
        assert_eq!(provenance.acquisition, "download");
        assert_eq!(provenance.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(provenance.source_host.as_deref(), Some("huggingface.co"));
        assert_eq!(provenance.last_verified_at, Some(1700000100));
        assert!(provenance.integrity_mismatch_at.is_none());
        assert_eq!(catalog.download.status.phase, "downloading");
        assert_eq!(catalog.download.status.total_bytes, Some(100));
        assert!(catalog.download.status.resumable);
        assert_eq!(catalog.download.staged.len(), 1);
        assert_eq!(catalog.download.staged[0].filename, "pending.gguf");
        assert!(catalog.import.enabled);
        assert!(catalog.import.license_policy.enforced);
        assert_eq!(catalog.import.max_chunk_bytes, 100);
        assert_eq!(catalog.import.status.uploaded_bytes, 25);
        assert_eq!(catalog.import.staged[0].total_bytes, 100);
        assert!(catalog.index.enabled);
        assert!(catalog.index.key_id.is_none());
        assert_eq!(catalog.index.trusted_key_count, 2);
        assert_eq!(catalog.index.refresh_seconds, 300);
        assert!(catalog.index.persistent_rollback_protection);
        assert_eq!(
            catalog.index.trust_id.as_deref(),
            Some("cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd")
        );
        assert!(catalog.storage.quota_enabled);
        assert_eq!(catalog.storage.committed_bytes, 750);
        assert_eq!(catalog.storage.available_bytes, Some(1250));
        assert_eq!(catalog.storage.last_cleanup_removed_sessions, 2);
        assert_eq!(catalog.integrity.phase, "complete");
        assert_eq!(catalog.integrity.matches_expected, Some(true));
        assert_eq!(catalog.integrity.checked_bytes, 100);

        let empty =
            decode_model_catalog(include_str!("../../examples/model-catalog-empty.json")).unwrap();
        assert!(empty.data.is_empty());
        assert!(empty.active_model.is_none());
        assert_eq!(empty.load.phase, "idle");
        assert!(!empty.download.enabled);
        assert!(!empty.import.enabled);
        assert!(!empty.index.enabled);

        let mut future: serde_json::Value = serde_json::from_str(example).unwrap();
        future["schema_version"] = serde_json::json!(2);
        assert!(decode_model_catalog(&future.to_string())
            .unwrap_err()
            .contains("unsupported schema version"));

        let mut unknown: serde_json::Value = serde_json::from_str(example).unwrap();
        unknown["storage"]["future_field"] = serde_json::json!(true);
        assert!(decode_model_catalog(&unknown.to_string())
            .unwrap_err()
            .contains("unknown field"));

        let mut missing: serde_json::Value = serde_json::from_str(example).unwrap();
        missing.as_object_mut().unwrap().remove("root_exists");
        assert!(decode_model_catalog(&missing.to_string())
            .unwrap_err()
            .contains("missing field"));

        let mut inconsistent: serde_json::Value = serde_json::from_str(example).unwrap();
        inconsistent["storage"]["committed_bytes"] = serde_json::json!(751);
        assert!(decode_model_catalog(&inconsistent.to_string())
            .unwrap_err()
            .contains("storage accounting"));

        let mut wrong_active: serde_json::Value = serde_json::from_str(example).unwrap();
        wrong_active["active_model"]["catalog_id"] = serde_json::json!("other.gguf");
        assert!(decode_model_catalog(&wrong_active.to_string())
            .unwrap_err()
            .contains("active catalog state"));
    }

    #[test]
    fn model_preflight_decodes_compatibility_and_memory_details() {
        let response: ModelPreflightResponse = serde_json::from_str(
            r#"{
                "schema_version":1,
                "object":"bloom.model_preflight",
                "data":{
                    "model_id":"tiny",
                    "inspected_at":1700000000,
                    "loadable":false,
                    "load_blocker":"engine is unavailable",
                    "manifest":{
                        "id":"tiny-llama",
                        "family":"llama",
                        "version":"1",
                        "model_tasks":["generation"],
                        "description":null,
                        "license":"Apache-2.0",
                        "input_modalities":["text"],
                        "output_modalities":["text"],
                        "formats":["gguf"],
                        "primary_dtype":"q4",
                        "quantization":"q4_k_m",
                        "quantization_bits":4,
                        "parameter_count":null,
                        "context_length":4096,
                        "num_layers":null,
                        "hidden_size":null,
                        "vocab_size":null,
                        "supports_mmap":true,
                        "requires_streaming":false
                    },
                    "runtime":{
                        "configured_engine":"candle",
                        "selected_engine":"candle",
                        "engine_maturity":"experimental",
                        "device":"cpu",
                        "device_backend":"cpu",
                        "backend_available":true,
                        "backend_reason":null,
                        "support":"unsupported",
                        "support_reason":"engine is unavailable",
                        "diagnostic_tips":["Use a supported model family."]
                    },
                    "memory":{
                        "per_request_context_tokens":2048,
                        "max_concurrent":2,
                        "planned_context_tokens":4096,
                        "weight_bytes":100,
                        "host_weight_bytes":100,
                        "device_weight_bytes":0,
                        "kv_cache_bytes":50,
                        "kv_cache_bytes_per_token":1,
                        "temp_tensor_bytes":10,
                        "total_bytes":160,
                        "available_bytes":1000,
                        "budget_bytes":750,
                        "reserve_bytes":60,
                        "memory_utilization":0.75,
                        "preallocation_enabled":true,
                        "fits_budget":true,
                        "scope":"host-resident estimate"
                    },
                    "warnings":[]
                }
            }"#,
        )
        .unwrap();

        assert_eq!(response.schema_version, MODEL_PREFLIGHT_SCHEMA_VERSION);
        assert_eq!(response.object, MODEL_PREFLIGHT_OBJECT);
        assert!(!response.data.loadable);
        assert_eq!(response.data.manifest.quantization_bits, Some(4));
        assert_eq!(response.data.manifest.context_length, Some(4096));
        assert_eq!(response.data.manifest.model_tasks, ["generation"]);
        assert_eq!(response.data.runtime.selected_engine, "candle");
        assert_eq!(response.data.memory.planned_context_tokens, 4096);
        let mut unsupported = response.clone();
        unsupported.schema_version = MODEL_PREFLIGHT_SCHEMA_VERSION + 1;
        assert!(unsupported.validate("tiny").is_err());
        assert!(response.validate("tiny").is_ok());

        let public_example = include_str!("../../examples/model-preflight.json");
        let published: ModelPreflightResponse = serde_json::from_str(public_example).unwrap();
        assert!(published.validate("local-model").is_ok());

        let mut missing_version: serde_json::Value = serde_json::from_str(public_example).unwrap();
        missing_version
            .as_object_mut()
            .unwrap()
            .remove("schema_version");
        assert!(serde_json::from_value::<ModelPreflightResponse>(missing_version).is_err());

        let mut unknown_field: serde_json::Value = serde_json::from_str(public_example).unwrap();
        unknown_field["data"]["manifest"]["future_field"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ModelPreflightResponse>(unknown_field).is_err());
    }

    #[test]
    fn model_preflight_rejects_task_identity_and_load_inconsistency() {
        let mut report = ModelPreflightReport {
            model_id: "encoder".to_string(),
            inspected_at: 1_700_000_000,
            loadable: true,
            manifest: ModelManifestSummary {
                id: "encoder".to_string(),
                family: "bert".to_string(),
                version: "1".to_string(),
                model_tasks: vec!["embedding".to_string(), "rerank".to_string()],
                input_modalities: vec!["text".to_string()],
                output_modalities: vec!["text".to_string()],
                formats: vec!["safetensors".to_string()],
                primary_dtype: "f32".to_string(),
                ..ModelManifestSummary::default()
            },
            runtime: ModelRuntimeCompatibility {
                configured_engine: "candle".to_string(),
                selected_engine: "candle".to_string(),
                engine_maturity: "production".to_string(),
                device: "cpu".to_string(),
                device_backend: "cpu".to_string(),
                backend_available: true,
                support: "native".to_string(),
                ..ModelRuntimeCompatibility::default()
            },
            memory: ModelMemoryPreflight {
                per_request_context_tokens: 256,
                max_concurrent: 2,
                planned_context_tokens: 512,
                memory_utilization: 0.75,
                fits_budget: true,
                scope: "host-resident estimate".to_string(),
                ..ModelMemoryPreflight::default()
            },
            ..ModelPreflightReport::default()
        };
        assert!(report.validate("encoder").is_ok());

        report.manifest.model_tasks = vec!["chat".to_string()];
        assert!(report.validate("encoder").is_err());
        report.manifest.model_tasks = vec!["embedding".to_string(), "rerank".to_string()];
        report.load_blocker = Some("blocked".to_string());
        assert!(report.validate("encoder").is_err());
    }

    #[test]
    fn model_inventory_validates_the_versioned_export_contract() {
        let inventory = decode_model_inventory(
            r#"{
                "schema_version":2,
                "object":"bloom.model_inventory",
                "summary":{
                    "model_count":1,
                    "provenance_count":1,
                    "source_locked_count":1,
                    "quarantined_count":0,
                    "invalid_provenance_count":0
                },
                "models":[{
                    "id":"tiny.gguf",
                    "kind":"file",
                    "format":"gguf",
                    "size_bytes":42,
                    "size_complete":true,
                    "provenance_status":"recorded",
                    "acquisition":"download",
                    "model_index_id":"tiny-q4",
                    "source":{
                        "url":"https://huggingface.co/acme/tiny/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tiny.gguf",
                        "host":"huggingface.co",
                        "revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "immutable_revision":true
                    },
                    "sha256":"abababababababababababababababababababababababababababababababab",
                    "license":"Apache-2.0",
                    "installed_at":1700000000,
                    "last_verified_at":1700000100,
                    "integrity":"verified",
                    "source_locked":true
                }]
            }"#,
        )
        .unwrap();

        assert_eq!(inventory.summary.model_count, 1);
        assert_eq!(inventory.summary.source_locked_count, 1);
        assert!(inventory.models[0].source_locked);
        assert_eq!(
            inventory.models[0].model_index_id.as_deref(),
            Some("tiny-q4")
        );

        let mut legacy = inventory.clone();
        legacy.schema_version = 1;
        legacy.models[0].model_index_id = None;
        assert!(validate_model_inventory(&legacy).is_ok());

        let mut wrong_version = inventory.clone();
        wrong_version.schema_version = 3;
        assert_eq!(
            validate_model_inventory(&wrong_version),
            Err("unsupported model inventory schema version: 3".to_string())
        );

        let mut wrong_count = inventory;
        wrong_count.summary.model_count = 2;
        assert_eq!(
            validate_model_inventory(&wrong_count),
            Err("server returned an inconsistent model inventory count".to_string())
        );
    }

    #[test]
    fn inventory_reconciliation_validates_counts_order_and_drift_types() {
        let report = decode_model_inventory_reconciliation(
            r#"{
                "schema_version":1,
                "object":"bloom.model_inventory_reconciliation",
                "in_sync":false,
                "truncated":false,
                "summary":{
                    "expected_model_count":2,
                    "current_model_count":2,
                    "matching_count":1,
                    "missing_count":0,
                    "unexpected_count":0,
                    "changed_count":1,
                    "blocking_count":1,
                    "restorable_count":0,
                    "drift_count":1
                },
                "drift":[{
                    "id":"tiny.gguf",
                    "status":"changed",
                    "severity":"blocking",
                    "changes":["sha256"],
                    "restore_available":false
                }]
            }"#,
        )
        .unwrap();

        assert_eq!(report.summary.changed_count, 1);
        assert_eq!(report.drift[0].changes, vec!["sha256"]);

        let mut invalid = report;
        invalid.summary.current_model_count = 3;
        assert_eq!(
            validate_model_inventory_reconciliation(&invalid),
            Err("server returned inconsistent inventory reconciliation counts".to_string())
        );

        invalid.summary.current_model_count = 2;
        invalid.drift[0].changes = vec!["future_field".to_string()];
        assert_eq!(
            validate_model_inventory_reconciliation(&invalid),
            Err("server returned an invalid inventory drift entry".to_string())
        );

        let restorable = decode_model_inventory_reconciliation(
            r#"{
                "schema_version":1,
                "object":"bloom.model_inventory_reconciliation",
                "in_sync":false,
                "truncated":false,
                "summary":{
                    "expected_model_count":1,
                    "current_model_count":0,
                    "matching_count":0,
                    "missing_count":1,
                    "unexpected_count":0,
                    "changed_count":0,
                    "blocking_count":1,
                    "restorable_count":1,
                    "drift_count":1
                },
                "drift":[{
                    "id":"missing.gguf",
                    "status":"missing",
                    "severity":"blocking",
                    "changes":["model_missing"],
                    "restore_available":true
                }]
            }"#,
        )
        .unwrap();
        assert_eq!(restorable.summary.restorable_count, 1);
        assert!(restorable.drift[0].restore_available);
    }
}
