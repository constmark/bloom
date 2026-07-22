#![allow(unused_imports, dead_code)]
use super::*;

// ─── Helpers ────────────────────────────────────────────────────────────────

pub fn chat_prompt(
    messages: &[ChatCompletionMessage],
    family: &bloomai_core::ModelFamily,
) -> String {
    let chat_msgs: Vec<ChatMessage> = messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let tpl = select_template(family);
    tracing::debug!("Using chat template: {}", tpl.name());
    tpl.format(&chat_msgs)
}

pub(crate) fn chat_stop_chunk(request_id: String, model_id: String) -> Event {
    Event::default()
        .json_data(json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": unix_seconds(),
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }))
        .unwrap()
}

pub(crate) fn response_format_mode(
    response_format: Option<&ResponseFormat>,
) -> std::result::Result<ResponseFormatMode, String> {
    let Some(response_format) = response_format else {
        return Ok(ResponseFormatMode::Text);
    };
    match response_format.format_type.as_str() {
        "text" => Ok(ResponseFormatMode::Text),
        "json_object" => Ok(ResponseFormatMode::JsonObject),
        "json_schema" => {
            let schema = response_format
                .json_schema
                .as_ref()
                .and_then(|value| value.get("schema").or(Some(value)))
                .cloned()
                .ok_or_else(|| {
                    "response_format=json_schema requires a json_schema.schema object.".to_string()
                })?;
            Ok(ResponseFormatMode::JsonSchema(schema))
        }
        other => Err(format!(
            "unsupported response_format type '{}'; expected text, json_object, or json_schema",
            other
        )),
    }
}

pub(crate) fn apply_response_format_instruction(prompt: String, mode: &ResponseFormatMode) -> String {
    match mode {
        ResponseFormatMode::Text => prompt,
        ResponseFormatMode::JsonObject => format!(
            "{}\n\nReturn a valid JSON object only. Do not include Markdown fences or explanatory text.",
            prompt
        ),
        ResponseFormatMode::JsonSchema(schema) => format!(
            "{}\n\nReturn a valid JSON object only. It must satisfy this JSON Schema: {}. Do not include Markdown fences or explanatory text.",
            prompt,
            schema
        ),
    }
}

pub(crate) fn validate_structured_output(
    text: &str,
    mode: &ResponseFormatMode,
) -> std::result::Result<(), String> {
    match mode {
        ResponseFormatMode::Text => Ok(()),
        ResponseFormatMode::JsonObject => {
            let value = parse_json_object_output(text, "response_format=json_object")?;
            if value.is_object() {
                Ok(())
            } else {
                Err("model output was valid JSON but not a JSON object.".to_string())
            }
        }
        ResponseFormatMode::JsonSchema(schema) => {
            let value = parse_json_object_output(text, "response_format=json_schema")?;
            validate_json_schema_subset(&value, schema, "$")
        }
    }
}

pub(crate) fn parse_json_object_output(
    text: &str,
    label: &str,
) -> std::result::Result<serde_json::Value, String> {
    serde_json::from_str(text)
        .map_err(|err| format!("model output was not valid JSON for {}: {}", label, err))
}

pub(crate) fn validate_json_schema_subset(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> std::result::Result<(), String> {
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        if !enum_values.iter().any(|candidate| candidate == value) {
            return Err(format!("{} does not match any allowed enum value", path));
        }
    }

    if let Some(schema_type) = schema.get("type").and_then(|v| v.as_str()) {
        if !json_value_matches_type(value, schema_type) {
            return Err(format!("{} expected type {}", path, schema_type));
        }
    }

    if value.is_object() {
        let object = value.as_object().unwrap();
        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            for field in required.iter().filter_map(|v| v.as_str()) {
                if !object.contains_key(field) {
                    return Err(format!("{} missing required property '{}'", path, field));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            for (field, property_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_json_schema_subset(
                        field_value,
                        property_schema,
                        &format!("{}.{}", path, field),
                    )?;
                }
            }
            if schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false) {
                for field in object.keys() {
                    if !properties.contains_key(field) {
                        return Err(format!(
                            "{} contains unsupported property '{}'",
                            path, field
                        ));
                    }
                }
            }
        }
    }

    if let Some(item_schema) = schema.get("items") {
        if let Some(items) = value.as_array() {
            for (idx, item) in items.iter().enumerate() {
                validate_json_schema_subset(item, item_schema, &format!("{}[{}]", path, idx))?;
            }
        }
    }

    Ok(())
}

pub(crate) fn json_value_matches_type(value: &serde_json::Value, schema_type: &str) -> bool {
    match schema_type {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

pub(crate) fn normalize_embedding_input(
    input: &serde_json::Value,
) -> std::result::Result<Vec<String>, String> {
    match input {
        serde_json::Value::String(text) => {
            if text.is_empty() {
                Err("Embedding input string must not be empty.".to_string())
            } else {
                Ok(vec![text.clone()])
            }
        }
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                return Err("Embedding input array must not be empty.".to_string());
            }
            let mut texts = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    serde_json::Value::String(text) if !text.is_empty() => texts.push(text.clone()),
                    serde_json::Value::String(_) => {
                        return Err("Embedding input strings must not be empty.".to_string());
                    }
                    _ => {
                        return Err(
                            "Embedding input currently supports a string or an array of strings."
                                .to_string(),
                        );
                    }
                }
            }
            Ok(texts)
        }
        _ => Err("Embedding input currently supports a string or an array of strings.".to_string()),
    }
}

pub(crate) fn model_supports_embeddings(pipeline: &InferencePipeline) -> bool {
    let metadata = pipeline.metadata();
    let id = metadata.id.to_lowercase();
    let manifest_id = metadata.manifest.id.to_lowercase();
    ["embed", "embedding", "bge", "bert", "rerank"]
        .iter()
        .any(|needle| id.contains(needle) || manifest_id.contains(needle))
}

pub(crate) fn unsupported_embeddings_response(model_id: &str) -> axum::response::Response {
    error_response(
        axum::http::StatusCode::NOT_IMPLEMENTED,
        "unsupported_operation",
        format!(
            "Model '{model_id}' does not advertise embedding/rerank support. Load an embedding model such as BGE or another model id containing embed/bert/rerank."
        ),
    )
}

pub(crate) fn count_text_tokens(pipeline: &InferencePipeline, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| pipeline.tokenize(text).map(|t| t.len()).unwrap_or(0))
        .sum()
}

pub(crate) async fn run_embedding_task(
    pipeline: Arc<InferencePipeline>,
    text: String,
) -> std::result::Result<Vec<f32>, String> {
    match task::spawn_blocking(move || collect_embedding(pipeline, text)).await {
        Ok(Ok(embedding)) => Ok(embedding),
        Ok(Err(err)) => Err(format!("Embedding inference failed: {}", err)),
        Err(err) => Err(format!("Embedding task join failed: {}", err)),
    }
}

pub(crate) fn collect_embedding(pipeline: Arc<InferencePipeline>, text: String) -> Result<Vec<f32>> {
    let embedding = Arc::new(std::sync::Mutex::new(None::<Vec<f32>>));
    let embedding_sink = Arc::clone(&embedding);
    let params = GenerationParams {
        max_tokens: 1,
        temperature: 0.0,
        top_p: 1.0,
        seed: None,
        response_format: None,
    };
    pipeline.run_stream(
        ModelInput::Text { prompt: text },
        &params,
        &mut move |chunk: OutputChunk| {
            if let OutputChunk::Embedding(values) = chunk {
                let mut slot = embedding_sink.lock().unwrap();
                *slot = Some(values);
            }
            Ok(())
        },
    )?;
    let output = embedding
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow!("model did not produce OutputChunk::Embedding"));
    output
}

pub(crate) fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let len = left.len().min(right.len());
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for idx in 0..len {
        let l = left[idx] as f64;
        let r = right[idx] as f64;
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

pub(crate) fn estimate_delta_tokens(pipeline: &InferencePipeline, text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    pipeline
        .tokenize(text)
        .map(|tokens| tokens.len().max(1) as u64)
        .unwrap_or(1)
}

pub(crate) fn record_stream_tokens(
    state: &ServerState,
    request_start: Instant,
    first_token_seen: &AtomicBool,
    last_token_time: &std::sync::Mutex<Option<Instant>>,
    generated_count: &AtomicU64,
    token_count: u64,
) {
    if token_count == 0 {
        return;
    }
    let now = Instant::now();
    if let Ok(mut last_time) = last_token_time.lock() {
        if !first_token_seen.swap(true, Ordering::Relaxed) {
            state
                .metrics
                .record_first_token_latency(request_start.elapsed().as_secs_f64() * 1000.0);
        } else if let Some(prev) = *last_time {
            let delta = now.duration_since(prev).as_secs_f64();
            state.metrics.record_inter_token_latency(delta);
        }
        *last_time = Some(now);
    }
    generated_count.fetch_add(token_count, Ordering::Relaxed);
}

pub(crate) fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(crate) fn next_request_id(state: &ServerState, prefix: &str) -> String {
    next_request_id_from_counter(&state.request_counter, prefix)
}

pub(crate) fn next_request_id_from_counter(counter: &AtomicU64, prefix: &str) -> String {
    let seq = counter.fetch_add(1, Ordering::Relaxed) + 1;
    format!("{}-{}-{}", prefix, unix_seconds(), seq)
}
