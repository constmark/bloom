//! Bounded OpenAI Chat Completions adapter for the native multimodal stream.

use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::convert::Infallible;

const MAX_OPENAI_VISION_BASE64_BYTES: usize = MAX_MULTIMODAL_IMAGE_BYTES.div_ceil(3) * 4;
const MAX_OPENAI_VISION_STREAM_BYTES: usize = 16 * MIB as usize;
const MAX_OPENAI_VISION_STREAM_EVENTS: usize = 131_072;

pub(crate) fn prepare_openai_vision_request(
    request: &ChatRequest,
) -> std::result::Result<Option<InferenceRequest>, String> {
    let has_image = request.messages.iter().any(|message| {
        message.content.as_array().is_some_and(|parts| {
            parts.iter().any(|part| {
                part.get("type").and_then(serde_json::Value::as_str) == Some("image_url")
            })
        })
    });
    if !has_image {
        return Ok(None);
    }
    if request.messages.len() != 1 || request.messages[0].role != "user" {
        return Err(
            "Image input currently supports exactly one user message and no system, assistant, tool, or history messages."
                .to_string(),
        );
    }
    if chat_tool_config(request)?.is_some() {
        return Err("Function tools cannot be combined with image input.".to_string());
    }
    if !normalize_stop_sequences(request.stop.as_ref())?.is_empty() {
        return Err("Stop sequences cannot be combined with image input.".to_string());
    }
    if !matches!(
        response_format_mode(request.response_format.as_ref())?,
        ResponseFormatMode::Text
    ) {
        return Err("Structured response formats cannot be combined with image input.".to_string());
    }
    if request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage)
    {
        return Err(
            "stream_options.include_usage is not available for image input because the native multimodal stream does not report token usage."
                .to_string(),
        );
    }

    let max_tokens = resolve_chat_max_tokens(request.max_tokens, request.max_completion_tokens)?;
    let temperature = request.temperature.unwrap_or(0.7);
    let top_p = request.top_p.unwrap_or(0.9);
    validate_generation_controls(max_tokens, temperature, top_p)?;

    let content = request.messages[0].content.as_array().ok_or_else(|| {
        "A user message with image input must use an array of text and image_url parts.".to_string()
    })?;
    if content.is_empty() || content.len() > MAX_CHAT_CONTENT_PARTS {
        return Err(format!(
            "Image message content must contain between 1 and {MAX_CHAT_CONTENT_PARTS} parts."
        ));
    }

    let mut text = String::new();
    let mut image = None;
    for (part_index, part) in content.iter().enumerate() {
        let object = part
            .as_object()
            .ok_or_else(|| format!("Image message content part {part_index} must be an object."))?;
        match object.get("type").and_then(serde_json::Value::as_str) {
            Some("text") => {
                reject_part_extensions(object, &["type", "text"], part_index)?;
                let part_text = object
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        format!(
                            "Image message text part {part_index} requires string field `text`."
                        )
                    })?;
                if part_text.len() > MAX_MULTIMODAL_TEXT_BYTES.saturating_sub(text.len()) {
                    return Err(format!(
                        "Combined image prompt text cannot exceed {MAX_MULTIMODAL_TEXT_BYTES} bytes."
                    ));
                }
                text.push_str(part_text);
            }
            Some("image_url") => {
                reject_part_extensions(object, &["type", "image_url"], part_index)?;
                if image.is_some() {
                    return Err(
                        "OpenAI-compatible image requests can contain at most one image."
                            .to_string(),
                    );
                }
                image = Some(parse_image_url_part(object, part_index)?);
            }
            Some(other) => {
                return Err(format!(
                    "Image message content part {part_index} has unsupported type {other:?}; only text and image_url are supported."
                ));
            }
            None => {
                return Err(format!(
                    "Image message content part {part_index} requires string field `type`."
                ));
            }
        }
    }
    if text.chars().count() > MAX_MULTIMODAL_TEXT_CHARS {
        return Err(format!(
            "Image prompt text cannot exceed {MAX_MULTIMODAL_TEXT_CHARS} characters."
        ));
    }
    let image = image.ok_or_else(|| "An image_url content part is required.".to_string())?;
    let mut blocks = Vec::with_capacity(2);
    if !text.trim().is_empty() {
        blocks.push(DataBlock::Text(text));
    }
    blocks.push(image);

    Ok(Some(InferenceRequest {
        blocks,
        params: InferenceParams {
            max_tokens,
            temperature,
            top_p,
            seed: request.seed,
            response_format: None,
        },
    }))
}

fn reject_part_extensions(
    object: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
    part_index: usize,
) -> std::result::Result<(), String> {
    if let Some(field) = object
        .iter()
        .find(|(field, value)| !value.is_null() && !allowed.contains(&field.as_str()))
        .map(|(field, _)| reported_extension_field(field))
    {
        return Err(format!(
            "Image message content part {part_index} contains unsupported field {field:?}."
        ));
    }
    Ok(())
}

fn parse_image_url_part(
    part: &serde_json::Map<String, serde_json::Value>,
    part_index: usize,
) -> std::result::Result<DataBlock, String> {
    let image_url = part
        .get("image_url")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            format!("Image message content part {part_index} requires an image_url object.")
        })?;
    if let Some(field) = image_url
        .iter()
        .find(|(field, value)| !value.is_null() && !matches!(field.as_str(), "url" | "detail"))
        .map(|(field, _)| reported_extension_field(field))
    {
        return Err(format!(
            "Image message image_url at part {part_index} contains unsupported field {field:?}."
        ));
    }
    let detail = image_url.get("detail").filter(|value| !value.is_null());
    if detail.is_some_and(|value| value.as_str() != Some("auto")) {
        return Err(
            "Image detail must be omitted or `auto`; Bloom does not emulate low/high processing modes."
                .to_string(),
        );
    }
    let url = image_url
        .get("url")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            format!("Image message image_url at part {part_index} requires string field `url`.")
        })?;
    let (mime, encoded) = if let Some(encoded) = url.strip_prefix("data:image/png;base64,") {
        ("image/png", encoded)
    } else if let Some(encoded) = url.strip_prefix("data:image/jpeg;base64,") {
        ("image/jpeg", encoded)
    } else {
        return Err(
            "Image URL must be an inline data:image/png;base64 or data:image/jpeg;base64 URL; remote URLs are not fetched."
                .to_string(),
        );
    };
    if encoded.is_empty() || encoded.len() > MAX_OPENAI_VISION_BASE64_BYTES {
        return Err(format!(
            "Image data must decode to between 1 and {MAX_MULTIMODAL_IMAGE_BYTES} bytes."
        ));
    }
    if encoded.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err("Image data must use canonical base64 without whitespace.".to_string());
    }
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|_| "Image data contains invalid base64.".to_string())?;
    if STANDARD.encode(&bytes) != encoded {
        return Err("Image data must use canonical padded base64.".to_string());
    }
    if bytes.is_empty() || bytes.len() > MAX_MULTIMODAL_IMAGE_BYTES {
        return Err(format!(
            "Image data must decode to between 1 and {MAX_MULTIMODAL_IMAGE_BYTES} bytes."
        ));
    }
    validate_uploaded_image(&bytes, mime)
        .map_err(|message| format!("Image data is invalid: {message}"))?;
    Ok(DataBlock::Image {
        bytes,
        mime: mime.to_string(),
    })
}

pub(crate) async fn openai_chat_from_multimodal_response(
    response: axum::response::Response,
    stream: bool,
) -> axum::response::Response {
    if !response.status().is_success() {
        return response;
    }
    if !response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
    {
        return error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "The internal multimodal response did not use SSE.",
        );
    }
    if stream {
        return stream_openai_vision_response(response);
    }
    collect_openai_vision_response(response).await
}

async fn collect_openai_vision_response(
    response: axum::response::Response,
) -> axum::response::Response {
    let mut body = response.into_body().into_data_stream();
    let mut decoder = ChatSseDecoder::default();
    let mut state = OpenAiVisionStreamState::default();
    let mut transport_bytes = 0_usize;
    while let Some(chunk) = body.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(_) => return internal_vision_error("The internal multimodal stream body failed."),
        };
        transport_bytes = match transport_bytes.checked_add(chunk.len()) {
            Some(bytes) if bytes <= MAX_OPENAI_VISION_STREAM_BYTES => bytes,
            _ => {
                return internal_vision_error(
                    "The internal multimodal stream exceeded its byte limit.",
                );
            }
        };
        let frames = match decoder.push(&chunk) {
            Ok(frames) => frames,
            Err(message) => return internal_vision_error(&message),
        };
        let frame_count = frames.len();
        for (index, frame) in frames.into_iter().enumerate() {
            if frame == "[DONE]" {
                if index + 1 != frame_count || decoder.finish().is_err() {
                    return internal_vision_error(
                        "The internal multimodal stream ended with invalid framing.",
                    );
                }
                return match state.finish_buffered() {
                    Ok(payload) => Json(payload).into_response(),
                    Err(message) => internal_vision_error(&message),
                };
            }
            let payload = match serde_json::from_str::<serde_json::Value>(&frame) {
                Ok(payload) => payload,
                Err(_) => {
                    return internal_vision_error(
                        "The internal multimodal stream emitted invalid JSON.",
                    );
                }
            };
            if let Err(message) = state.ingest(payload, true) {
                return internal_vision_error(&message);
            }
        }
    }
    internal_vision_error("The internal multimodal stream ended before its terminal marker.")
}

fn stream_openai_vision_response(response: axum::response::Response) -> axum::response::Response {
    let mut body = response.into_body().into_data_stream();
    let (tx, rx) = mpsc::channel::<std::result::Result<Event, Infallible>>(32);
    task::spawn(async move {
        let mut decoder = ChatSseDecoder::default();
        let mut state = OpenAiVisionStreamState::default();
        let mut transport_bytes = 0_usize;
        loop {
            let chunk = tokio::select! {
                _ = tx.closed() => return,
                chunk = body.next() => chunk,
            };
            let Some(chunk) = chunk else {
                send_openai_vision_stream_error(
                    &tx,
                    "The internal multimodal stream ended before its terminal marker.",
                )
                .await;
                return;
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) => {
                    send_openai_vision_stream_error(
                        &tx,
                        "The internal multimodal stream body failed.",
                    )
                    .await;
                    return;
                }
            };
            transport_bytes = match transport_bytes.checked_add(chunk.len()) {
                Some(bytes) if bytes <= MAX_OPENAI_VISION_STREAM_BYTES => bytes,
                _ => {
                    send_openai_vision_stream_error(
                        &tx,
                        "The internal multimodal stream exceeded its byte limit.",
                    )
                    .await;
                    return;
                }
            };
            let frames = match decoder.push(&chunk) {
                Ok(frames) => frames,
                Err(message) => {
                    send_openai_vision_stream_error(&tx, &message).await;
                    return;
                }
            };
            let frame_count = frames.len();
            for (index, frame) in frames.into_iter().enumerate() {
                if frame == "[DONE]" {
                    if index + 1 != frame_count || decoder.finish().is_err() {
                        send_openai_vision_stream_error(
                            &tx,
                            "The internal multimodal stream ended with invalid framing.",
                        )
                        .await;
                        return;
                    }
                    match state.finish_streaming() {
                        Ok(payload) => {
                            if tx.send(Ok(json_event(payload))).await.is_err() {
                                return;
                            }
                            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                        }
                        Err(message) => {
                            send_openai_vision_stream_error(&tx, &message).await;
                        }
                    }
                    return;
                }
                let payload = match serde_json::from_str::<serde_json::Value>(&frame) {
                    Ok(payload) => payload,
                    Err(_) => {
                        send_openai_vision_stream_error(
                            &tx,
                            "The internal multimodal stream emitted invalid JSON.",
                        )
                        .await;
                        return;
                    }
                };
                match state.ingest(payload, false) {
                    Ok(Some(payload)) => {
                        if tx.send(Ok(json_event(payload))).await.is_err() {
                            return;
                        }
                    }
                    Ok(None) => {}
                    Err(message) => {
                        send_openai_vision_stream_error(&tx, &message).await;
                        return;
                    }
                }
            }
        }
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

async fn send_openai_vision_stream_error(
    tx: &mpsc::Sender<std::result::Result<Event, Infallible>>,
    message: &str,
) {
    let _ = tx
        .send(Ok(json_event(json!({
            "error": {
                "message": bounded_vision_error(message),
                "type": "internal_error"
            }
        }))))
        .await;
    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
}

fn internal_vision_error(message: &str) -> axum::response::Response {
    error_response(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        bounded_vision_error(message),
    )
}

fn bounded_vision_error(message: &str) -> String {
    let mut bounded = String::new();
    for character in message.chars() {
        if bounded.len().saturating_add(character.len_utf8()) > MAX_RESPONSES_STREAM_ERROR_BYTES {
            break;
        }
        bounded.push(character);
    }
    if bounded.trim().is_empty() {
        "The local multimodal generation stream failed.".to_string()
    } else {
        bounded
    }
}

#[derive(Default)]
struct OpenAiVisionStreamState {
    id: Option<String>,
    model: Option<String>,
    created: Option<u64>,
    saw_start: bool,
    saw_end: bool,
    events: usize,
    output_bytes: usize,
    buffered: String,
}

impl OpenAiVisionStreamState {
    fn ingest(
        &mut self,
        payload: serde_json::Value,
        collect: bool,
    ) -> std::result::Result<Option<serde_json::Value>, String> {
        self.events = self.events.saturating_add(1);
        if self.events > MAX_OPENAI_VISION_STREAM_EVENTS {
            return Err("The internal multimodal stream emitted too many events.".to_string());
        }
        if let Some(error) = payload.get("error") {
            return Err(error
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("The local multimodal generation stream failed.")
                .to_string());
        }
        let object = payload
            .as_object()
            .ok_or_else(|| "The internal multimodal stream emitted a non-object.".to_string())?;
        if object.iter().any(|(field, value)| {
            !value.is_null()
                && !matches!(
                    field.as_str(),
                    "id" | "object" | "created" | "model" | "chunk"
                )
        }) {
            return Err(
                "The internal multimodal stream event contained unsupported fields.".to_string(),
            );
        }
        if payload.get("object").and_then(serde_json::Value::as_str) != Some("multimodal.chunk") {
            return Err(
                "The internal multimodal stream used an unexpected object type.".to_string(),
            );
        }
        self.validate_identity(&payload)?;
        let chunk = payload
            .get("chunk")
            .ok_or_else(|| "The internal multimodal stream omitted its chunk.".to_string())?;
        if chunk.is_null() {
            if self.saw_start || self.saw_end || self.output_bytes != 0 {
                return Err(
                    "The internal multimodal stream emitted an invalid start event.".to_string(),
                );
            }
            self.saw_start = true;
            return Ok((!collect).then(|| self.start_chunk()));
        }
        if !self.saw_start {
            return Err(
                "The internal multimodal stream emitted output before its start event.".to_string(),
            );
        }
        if self.saw_end {
            return Err(
                "The internal multimodal stream emitted output after its end event.".to_string(),
            );
        }
        if chunk.as_str() == Some("End") {
            self.saw_end = true;
            return Ok(None);
        }
        let chunk = chunk
            .as_object()
            .filter(|chunk| chunk.len() == 1)
            .ok_or_else(|| {
                "The internal multimodal stream emitted an invalid chunk.".to_string()
            })?;
        if chunk.contains_key("Metrics") {
            if !chunk["Metrics"].is_object() {
                return Err("The internal multimodal stream emitted invalid metrics.".to_string());
            }
            return Ok(None);
        }
        let text = if let Some(text) = chunk.get("TextDelta") {
            text.as_str().ok_or_else(|| {
                "The internal multimodal stream emitted a non-text delta.".to_string()
            })?
        } else if let Some(token) = chunk.get("VlmToken") {
            let token = token.as_object().ok_or_else(|| {
                "The internal multimodal stream emitted an invalid VLM token.".to_string()
            })?;
            if token.iter().any(|(field, value)| {
                !value.is_null() && !matches!(field.as_str(), "text" | "bounding_box")
            }) {
                return Err(
                    "The internal multimodal VLM token contained unsupported fields.".to_string(),
                );
            }
            token
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "The internal multimodal stream omitted VLM token text.".to_string()
                })?
        } else {
            return Err(
                "The internal multimodal stream emitted an unsupported output chunk.".to_string(),
            );
        };
        self.output_bytes = self
            .output_bytes
            .checked_add(text.len())
            .filter(|bytes| *bytes <= MAX_OPENAI_VISION_STREAM_BYTES)
            .ok_or_else(|| "The internal multimodal output exceeded its byte limit.".to_string())?;
        if collect {
            self.buffered.push_str(text);
            Ok(None)
        } else {
            Ok(Some(self.text_chunk(text)))
        }
    }

    fn validate_identity(
        &mut self,
        payload: &serde_json::Value,
    ) -> std::result::Result<(), String> {
        let id = payload
            .get("id")
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| "The internal multimodal stream omitted its id.".to_string())?;
        let model = payload
            .get("model")
            .and_then(serde_json::Value::as_str)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| "The internal multimodal stream omitted its model.".to_string())?;
        let created = payload
            .get("created")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                "The internal multimodal stream omitted its creation time.".to_string()
            })?;
        if self.id.as_deref().is_some_and(|expected| expected != id)
            || self
                .model
                .as_deref()
                .is_some_and(|expected| expected != model)
            || self.created.is_some_and(|expected| expected != created)
        {
            return Err("The internal multimodal stream changed identity.".to_string());
        }
        self.id.get_or_insert_with(|| id.to_string());
        self.model.get_or_insert_with(|| model.to_string());
        self.created.get_or_insert(created);
        Ok(())
    }

    fn start_chunk(&self) -> serde_json::Value {
        self.chunk(json!({"role": "assistant", "content": ""}), None)
    }

    fn text_chunk(&self, text: &str) -> serde_json::Value {
        self.chunk(json!({"content": text}), None)
    }

    fn finish_streaming(&self) -> std::result::Result<serde_json::Value, String> {
        self.validate_complete()?;
        Ok(self.chunk(json!({}), Some("stop")))
    }

    fn finish_buffered(&self) -> std::result::Result<serde_json::Value, String> {
        self.validate_complete()?;
        Ok(json!({
            "id": self.id.as_deref().unwrap_or_default(),
            "object": "chat.completion",
            "created": self.created.unwrap_or_default(),
            "model": self.model.as_deref().unwrap_or_default(),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self.buffered},
                "finish_reason": "stop"
            }]
        }))
    }

    fn validate_complete(&self) -> std::result::Result<(), String> {
        if !self.saw_start || !self.saw_end {
            return Err(
                "The internal multimodal stream omitted its start or end event.".to_string(),
            );
        }
        Ok(())
    }

    fn chunk(&self, delta: serde_json::Value, finish_reason: Option<&str>) -> serde_json::Value {
        json!({
            "id": self.id.as_deref().unwrap_or_default(),
            "object": "chat.completion.chunk",
            "created": self.created.unwrap_or_default(),
            "model": self.model.as_deref().unwrap_or_default(),
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_PIXEL_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

    fn request(content: serde_json::Value) -> ChatRequest {
        serde_json::from_value(json!({
            "model": "vision",
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": 32
        }))
        .unwrap()
    }

    #[test]
    fn openai_image_parts_become_bounded_multimodal_blocks() {
        let vision_request = request(json!([
            {"type": "text", "text": "Describe this."},
            {"type": "image_url", "image_url": {
                "url": format!("data:image/png;base64,{ONE_PIXEL_PNG}"),
                "detail": "auto"
            }}
        ]));
        let prepared = prepare_openai_vision_request(&vision_request)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.params.max_tokens, 32);
        assert!(matches!(&prepared.blocks[0], DataBlock::Text(text) if text == "Describe this."));
        assert!(
            matches!(&prepared.blocks[1], DataBlock::Image { mime, bytes } if mime == "image/png" && !bytes.is_empty())
        );

        let text_only = request(json!([{"type": "text", "text": "Hello"}]));
        assert!(prepare_openai_vision_request(&text_only).unwrap().is_none());
    }

    #[test]
    fn openai_image_parts_fail_closed_for_unsupported_semantics() {
        for content in [
            json!([{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]),
            json!([{"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{ONE_PIXEL_PNG}"), "detail": "high"}}]),
            json!([
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{ONE_PIXEL_PNG}")}},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{ONE_PIXEL_PNG}")}}
            ]),
            json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,not-base64"}}]),
        ] {
            assert!(prepare_openai_vision_request(&request(content)).is_err());
        }

        let mut with_usage = request(json!([{
            "type": "image_url",
            "image_url": {"url": format!("data:image/png;base64,{ONE_PIXEL_PNG}")}
        }]));
        with_usage.stream = true;
        with_usage.stream_options = Some(StreamOptions {
            include_usage: true,
            extensions: BTreeMap::new(),
        });
        assert!(prepare_openai_vision_request(&with_usage).is_err());
    }

    #[test]
    fn multimodal_events_map_to_openai_chat_shapes() {
        let mut buffered = OpenAiVisionStreamState::default();
        for payload in [
            json!({"id":"mms-1","object":"multimodal.chunk","created":7,"model":"vision","chunk":null}),
            json!({"id":"mms-1","object":"multimodal.chunk","created":7,"model":"vision","chunk":{"VlmToken":{"text":"A cat","bounding_box":null}}}),
            json!({"id":"mms-1","object":"multimodal.chunk","created":7,"model":"vision","chunk":{"Metrics":{"compute_ms":2}}}),
            json!({"id":"mms-1","object":"multimodal.chunk","created":7,"model":"vision","chunk":"End"}),
        ] {
            assert!(buffered.ingest(payload, true).unwrap().is_none());
        }
        let response = buffered.finish_buffered().unwrap();
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(response["choices"][0]["message"]["content"], "A cat");

        let mut streaming = OpenAiVisionStreamState::default();
        let start = streaming
            .ingest(
                json!({"id":"mms-2","object":"multimodal.chunk","created":8,"model":"vision","chunk":null}),
                false,
            )
            .unwrap()
            .unwrap();
        assert_eq!(start["choices"][0]["delta"]["role"], "assistant");
        let delta = streaming
            .ingest(
                json!({"id":"mms-2","object":"multimodal.chunk","created":8,"model":"vision","chunk":{"TextDelta":"Hello"}}),
                false,
            )
            .unwrap()
            .unwrap();
        assert_eq!(delta["choices"][0]["delta"]["content"], "Hello");
        streaming
            .ingest(
                json!({"id":"mms-2","object":"multimodal.chunk","created":8,"model":"vision","chunk":"End"}),
                false,
            )
            .unwrap();
        let terminal = streaming.finish_streaming().unwrap();
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    }
}
