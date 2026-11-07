//! Bounded OpenAI-shaped WAV transcription over an audio-capable local runtime.
use super::*;
use axum::extract::RawQuery;
use axum::http::StatusCode;
use std::collections::HashSet;

const MAX_AUDIO_FILE_BYTES: usize = 6 * 1024 * 1024;
const MAX_FIELD_BYTES: usize = 1024;
const MAX_TRANSCRIPT_BYTES: usize = 1024 * 1024;
const MAX_AUDIO_SECONDS: usize = 60;
const MAX_TRANSCRIPT_EVENTS: usize = 16_384;

#[derive(Debug)]
struct TranscriptionInput {
    model: String,
    file: Vec<u8>,
    wav: WavLayout,
    plain_text: bool,
}

fn invalid(message: &'static str) -> Response {
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

fn multipart_failure(error: axum::extract::multipart::MultipartError) -> Response {
    error_response(
        error.status(),
        "invalid_request_error",
        "Invalid audio multipart request.",
    )
}

async fn bounded_field(
    mut field: axum::extract::multipart::Field<'_>,
    limit: usize,
) -> std::result::Result<Vec<u8>, Response> {
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(multipart_failure)? {
        if chunk.len() > limit.saturating_sub(bytes.len()) {
            return Err(error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "Audio attachment or form field exceeds its byte limit.",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn parse_upload(
    mut multipart: Multipart,
) -> std::result::Result<TranscriptionInput, Response> {
    let mut seen = HashSet::new();
    let mut model = None;
    let mut file = None;
    let mut plain_text = false;
    while let Some(field) = multipart.next_field().await.map_err(multipart_failure)? {
        let name = field.name().unwrap_or_default().to_string();
        if !matches!(
            name.as_str(),
            "file" | "model" | "response_format" | "stream"
        ) {
            return Err(invalid(
                "Supported transcription fields are file, model, response_format and stream=false. Other fields are not supported.",
            ));
        }
        if !seen.insert(name.clone()) {
            return Err(invalid(
                "Duplicate transcription form fields are not supported.",
            ));
        }
        if name == "file" {
            // Never use the uploaded filename as a local path. WAV identity is
            // established from the payload, not the supplied name or MIME type.
            file = Some(bounded_field(field, MAX_AUDIO_FILE_BYTES).await?);
            continue;
        }
        let bytes = bounded_field(field, MAX_FIELD_BYTES).await?;
        let value = std::str::from_utf8(&bytes)
            .map_err(|_| invalid("Transcription form fields must be UTF-8 text."))?;
        match name.as_str() {
            "model" => {
                validate_model_selector(value).map_err(|_| invalid("model must contain 1 to 256 characters without surrounding whitespace or control characters."))?;
                model = Some(value.to_string());
            }
            "response_format" => match value {
                "json" => plain_text = false,
                "text" => plain_text = true,
                _ => return Err(invalid("response_format supports only json or text.")),
            },
            "stream" if value == "false" => {}
            "stream" => {
                return Err(invalid(
                    "Streaming transcription is not supported; stream must be false or omitted.",
                ));
            }
            _ => unreachable!(),
        }
    }
    let model = model.ok_or_else(|| invalid("The model field is required."))?;
    let file = file.ok_or_else(|| invalid("The file attachment is required."))?;
    let wav = validate_wav(&file).map_err(invalid)?;
    Ok(TranscriptionInput {
        model,
        file,
        wav,
        plain_text,
    })
}

#[derive(Debug)]
struct WavLayout {
    data: std::ops::Range<usize>,
    sample_rate: u32,
}

/// Validate at most 128 RIFF chunk headers without scanning sample bytes or
/// allocating decoded floats. PCM expansion happens only after admission.
fn validate_wav(bytes: &[u8]) -> std::result::Result<WavLayout, &'static str> {
    const BAD_WAV: &str = "file must be a complete nonempty RIFF/WAVE PCM16 mono recording at 8000 to 48000 Hz and at most 60 seconds.";
    if bytes.len() < 12
        || bytes.len() > MAX_AUDIO_FILE_BYTES
        || &bytes[..4] != b"RIFF"
        || &bytes[8..12] != b"WAVE"
    {
        return Err(BAD_WAV);
    }
    let u16_at = |data: &[u8], i: usize| u16::from_le_bytes([data[i], data[i + 1]]);
    let u32_at = |data: &[u8], i: usize| {
        u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
    };
    if usize::try_from(u32_at(bytes, 4))
        .ok()
        .and_then(|n| n.checked_add(8))
        != Some(bytes.len())
    {
        return Err(BAD_WAV);
    }
    let mut cursor = 12;
    let mut format = None;
    let mut audio = None;
    let mut chunks = 0;
    while cursor < bytes.len() {
        chunks += 1;
        if chunks > 128 || bytes.len() - cursor < 8 {
            return Err(BAD_WAV);
        }
        let kind = &bytes[cursor..cursor + 4];
        let length = usize::try_from(u32_at(bytes, cursor + 4)).map_err(|_| BAD_WAV)?;
        cursor += 8;
        let end = cursor
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or(BAD_WAV)?;
        let payload = &bytes[cursor..end];
        match kind {
            b"fmt " => {
                if format.is_some() || !(length == 16 || (length == 18 && u16_at(payload, 16) == 0))
                {
                    return Err(BAD_WAV);
                }
                let sample_rate = u32_at(payload, 4);
                if u16_at(payload, 0) != 1
                    || u16_at(payload, 2) != 1
                    || !(8_000..=48_000).contains(&sample_rate)
                    || u32_at(payload, 8) != sample_rate * 2
                    || u16_at(payload, 12) != 2
                    || u16_at(payload, 14) != 16
                {
                    return Err(BAD_WAV);
                }
                format = Some(sample_rate);
            }
            b"data" => {
                if audio.is_some() || payload.is_empty() || !length.is_multiple_of(2) {
                    return Err(BAD_WAV);
                }
                audio = Some(cursor..end);
            }
            _ => {}
        }
        cursor = end
            .checked_add(length % 2)
            .filter(|end| *end <= bytes.len())
            .ok_or(BAD_WAV)?;
    }
    let sample_rate = format.ok_or(BAD_WAV)?;
    let audio = audio.ok_or(BAD_WAV)?;
    if audio.len() / 2 > sample_rate as usize * MAX_AUDIO_SECONDS {
        return Err("Audio duration exceeds 60 seconds.");
    }
    Ok(WavLayout {
        data: audio,
        sample_rate,
    })
}

fn decode_samples(file: &[u8], wav: &WavLayout) -> Vec<f32> {
    file[wav.data.clone()]
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
        .collect()
}

#[cfg(test)]
fn decode_wav(bytes: &[u8]) -> std::result::Result<(Vec<f32>, u32), &'static str> {
    let wav = validate_wav(bytes)?;
    Ok((decode_samples(bytes, &wav), wav.sample_rate))
}

pub(crate) async fn handle_transcription(
    State(state): State<Arc<ServerState>>,
    RawQuery(query): RawQuery,
    multipart: Multipart,
) -> Response {
    if query.is_some() {
        return invalid("Transcription requests do not accept query parameters.");
    }
    let input = match parse_upload(multipart).await {
        Ok(input) => input,
        Err(error) => return error,
    };
    if !state.app.ready.load(Ordering::Acquire) {
        let (kind, message) = state.model_unavailable().await;
        return error_response(StatusCode::SERVICE_UNAVAILABLE, kind, message);
    }
    let runtime_lease = match state.app.lease_runtime(Some(&input.model)).await {
        Ok(Some(lease)) => lease,
        Ok(None) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_unavailable",
                "No audio model is currently available.",
            );
        }
        Err(RequestedModelError::Invalid) => return invalid("Invalid model selector."),
        Err(RequestedModelError::NotLoaded) => {
            return error_response(
                StatusCode::NOT_FOUND,
                "model_not_found",
                "The requested model is not loaded.",
            );
        }
        Err(RequestedModelError::Revoked) => {
            return error_response(
                StatusCode::GONE,
                "model_version_revoked",
                "The requested model version has been revoked.",
            );
        }
    };
    let runtime = runtime_lease.runtime();
    if !runtime
        .input_modalities
        .contains(&bloomai_core::Modality::Audio)
    {
        return error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_operation",
            "The selected model does not declare audio input support.",
        );
    }
    let pipeline = Arc::clone(&runtime.pipeline);
    let permit = match Arc::clone(&state.app.semaphore).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };
    let request_id = next_request_id(&state, "transcribe");
    let Some(cancel_guard) = state.app.cancellations.register(request_id, None) else {
        return error_response(
            StatusCode::CONFLICT,
            "request_id_conflict",
            "A request with the same ID is already active.",
        );
    };
    state.app.metrics.record_request_start();
    let request_start = Instant::now();
    let token = cancel_guard.token();
    let lifecycle = InferenceLifecycle::new(
        cancel_guard,
        InferenceLifecycleResources {
            metrics: Arc::clone(&state.app.metrics),
            request_start,
            generated_tokens: Arc::new(AtomicU64::new(0)),
            prompt_tokens: 0,
            permit,
            runtime_lease,
        },
        StreamExecution::Blocking,
    );
    let worker_guard = lifecycle.worker_guard();
    let mut client_guard = lifecycle.client_guard();
    let worker_token = token.clone();
    let mut worker = task::spawn_blocking(move || {
        let _worker_guard = worker_guard;
        let mut collector = TranscriptCollector::default();
        if worker_token.is_cancelled() {
            return Err(anyhow!("cancelled"));
        }
        // Expansion from PCM16 to f32 is CPU work and doubles the sample
        // payload size, so it belongs to this admitted blocking worker.
        let samples = decode_samples(&input.file, &input.wav);
        drop(input.file);
        if worker_token.is_cancelled() {
            return Err(anyhow!("cancelled"));
        }
        pipeline.run_stream(
            ModelInput::Audio {
                samples,
                sample_rate: input.wav.sample_rate,
            },
            &GenerationParams::default(),
            &mut |chunk| {
                if worker_token.is_cancelled() {
                    return Err(anyhow!("cancelled"));
                }
                collector.push(chunk)
            },
        )?;
        if worker_token.is_cancelled() {
            return Err(anyhow!("cancelled"));
        }
        collector.finish()
    });
    let result = tokio::select! {
        result = &mut worker => result,
        _ = token.cancelled() => {
            client_guard.finish(false);
            return error_response(StatusCode::REQUEST_TIMEOUT, "request_cancelled", "The transcription request was cancelled.");
        }
    };
    state
        .app
        .metrics
        .record_inference_latency(request_start.elapsed().as_secs_f64());
    if token.is_cancelled() {
        client_guard.finish(false);
        return error_response(
            StatusCode::REQUEST_TIMEOUT,
            "request_cancelled",
            "The transcription request was cancelled.",
        );
    }
    let text = match result {
        Ok(Ok(text)) => text,
        _ => {
            client_guard.finish(false);
            // Adapter failures can contain private paths or process details.
            return error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "transcription_failed",
                "The audio model could not produce a valid bounded transcription.",
            );
        }
    };
    client_guard.finish(true);
    if input.plain_text {
        ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
    } else {
        Json(json!({"text": text})).into_response()
    }
}

#[derive(Default)]
struct TranscriptCollector {
    text: String,
    partial: String,
    has_text: bool,
    events: usize,
    ended: bool,
}

impl TranscriptCollector {
    fn push(&mut self, chunk: OutputChunk) -> Result<()> {
        anyhow::ensure!(!self.ended, "transcription output after terminal event");
        self.events += 1;
        anyhow::ensure!(
            self.events <= MAX_TRANSCRIPT_EVENTS,
            "transcription event limit"
        );
        match chunk {
            OutputChunk::TextDelta(text) => {
                anyhow::ensure!(
                    text.len() <= MAX_TRANSCRIPT_BYTES.saturating_sub(self.text.len()),
                    "transcription byte limit"
                );
                self.has_text |= !text.is_empty();
                self.text.push_str(&text);
            }
            OutputChunk::AsrPartial { text, .. } => {
                anyhow::ensure!(
                    text.len() <= MAX_TRANSCRIPT_BYTES,
                    "transcription byte limit"
                );
                self.partial = text;
            }
            OutputChunk::End => self.ended = true,
            OutputChunk::Metrics { .. } => {}
            _ => anyhow::bail!("unexpected transcription output"),
        }
        Ok(())
    }

    fn finish(self) -> Result<String> {
        anyhow::ensure!(self.ended, "transcription ended without terminal event");
        // The real ASR bridge emits End alone for successful silence.
        Ok(if self.has_text {
            self.text
        } else {
            self.partial
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::{Condvar, Mutex as StdMutex};
    use tower::ServiceExt;

    #[test]
    fn empty_text_delta_preserves_an_asr_snapshot() {
        let mut collector = TranscriptCollector::default();
        collector
            .push(OutputChunk::AsrPartial {
                text: "hello".into(),
                tokens: vec![],
            })
            .unwrap();
        collector
            .push(OutputChunk::TextDelta(String::new()))
            .unwrap();
        collector.push(OutputChunk::End).unwrap();
        assert_eq!(collector.finish().unwrap(), "hello");
    }

    fn wav(rate: u32, samples: &[i16]) -> Vec<u8> {
        let length = samples.len() * 2;
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(36_u32 + length as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(length as u32).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn request(fields: &[(&str, &[u8])], query: &str) -> Request<Body> {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(
                format!(
                    "--bloom-audio-test\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--bloom-audio-test--\r\n");
        Request::post(format!("/{query}"))
            .header(
                "content-type",
                "multipart/form-data; boundary=bloom-audio-test",
            )
            .body(Body::from(body))
            .unwrap()
    }

    #[derive(Clone)]
    enum Behavior {
        Success,
        Fail,
        TooLarge,
        Blocked(Arc<tokio::sync::Notify>, Arc<(StdMutex<bool>, Condvar)>),
    }
    struct AudioEngine(Behavior);
    impl bloomai_engine::Engine for AudioEngine {
        fn name(&self) -> &'static str {
            "test-audio"
        }
        fn supported_modalities(&self) -> Vec<bloomai_core::Modality> {
            vec![bloomai_core::Modality::Audio]
        }
        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }
        fn load(&self, _: &Path, _: DeviceKind) -> Result<Box<dyn bloomai_engine::LoadedModel>> {
            Ok(Box::new(AudioModel {
                behavior: self.0.clone(),
                metadata: bloomai_engine::ModelMetadata {
                    id: "test-audio".into(),
                    modality: bloomai_core::Modality::Audio,
                    quantized: false,
                    manifest: bloomai_core::ModelManifest {
                        id: "test-audio".into(),
                        ..Default::default()
                    },
                },
            }))
        }
    }
    struct AudioModel {
        metadata: bloomai_engine::ModelMetadata,
        behavior: Behavior,
    }
    impl bloomai_engine::LoadedModel for AudioModel {
        fn metadata(&self) -> &bloomai_engine::ModelMetadata {
            &self.metadata
        }
        fn infer(
            &self,
            _: ModelInput,
            _: &GenerationParams,
        ) -> Result<bloomai_engine::ModelOutput> {
            anyhow::bail!("must use stream for audio")
        }
        fn infer_stream(
            &self,
            input: ModelInput,
            _: &GenerationParams,
            sink: &mut dyn bloomai_engine::model::OutputSink,
        ) -> Result<()> {
            let ModelInput::Audio {
                samples,
                sample_rate,
            } = input
            else {
                anyhow::bail!("wrong input")
            };
            assert_eq!(sample_rate, 16_000);
            assert_eq!(samples, vec![-1.0, 0.0, 0.5]);
            match &self.behavior {
                Behavior::Fail => anyhow::bail!("private /path/model and secret details"),
                Behavior::TooLarge => {
                    sink.on_chunk(OutputChunk::TextDelta("x".repeat(MAX_TRANSCRIPT_BYTES + 1)))?
                }
                Behavior::Blocked(started, release) => {
                    started.notify_one();
                    let (lock, wake) = &**release;
                    let mut released = lock.lock().unwrap();
                    while !*released {
                        released = wake.wait(released).unwrap();
                    }
                }
                Behavior::Success => {}
            }
            sink.on_chunk(OutputChunk::AsrPartial {
                text: "hello".into(),
                tokens: vec![],
            })?;
            sink.on_chunk(OutputChunk::TextDelta("hello".into()))?;
            sink.on_chunk(OutputChunk::End)
        }
    }

    async fn state(root: PathBuf, behavior: Behavior, audio: bool) -> Arc<ServerState> {
        let (state, _) = crate::tests::test_server_state(root.clone());
        let mut runtime = crate::tests::test_text_runtime_with_engine(
            root.join("audio.fixture"),
            &AudioEngine(behavior),
        );
        if audio {
            Arc::get_mut(&mut runtime).unwrap().input_modalities =
                vec![bloomai_core::Modality::Audio];
        }
        state
            .app
            .runtime_pool
            .write()
            .await
            .publish_default(runtime);
        state.app.ready.store(true, Ordering::Release);
        state
    }
    fn app(state: Arc<ServerState>) -> Router {
        Router::new()
            .route("/", post(handle_transcription))
            .with_state(state)
    }

    #[test]
    fn wav_validation_records_only_the_original_pcm_range() {
        let bytes = wav(16_000, &[-32768, 0, 16384]);
        let layout = validate_wav(&bytes).unwrap();
        assert_eq!(layout.data, 44..50);
        assert_eq!(layout.sample_rate, 16_000);
        assert_eq!(decode_samples(&bytes, &layout), vec![-1.0, 0.0, 0.5]);
    }

    #[test]
    fn wav_decoding_validates_samples_and_complete_chunk_layout() {
        let valid = wav(16_000, &[-32768, 0, 16384]);
        assert_eq!(decode_wav(&valid).unwrap(), (vec![-1.0, 0.0, 0.5], 16_000));
        for offset in [0, 4, 8, 16, 20, 22, 24, 28, 32, 34, 40] {
            let mut invalid = valid.clone();
            invalid[offset] ^= 0xFF;
            assert!(decode_wav(&invalid).is_err(), "offset {offset}");
        }
        for length in 0..valid.len() {
            assert!(decode_wav(&valid[..length]).is_err());
        }
        assert!(decode_wav(&wav(16_000, &[])).is_err());
        assert!(decode_wav(&wav(16_000, &vec![0; 16_000 * 60 + 1])).is_err());
        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(decode_wav(&trailing).is_err());
        let mut duplicate = valid.clone();
        duplicate.extend_from_slice(&valid[12..36]);
        let length = (duplicate.len() - 8) as u32;
        duplicate[4..8].copy_from_slice(&length.to_le_bytes());
        assert!(decode_wav(&duplicate).is_err());
    }

    #[test]
    fn transcript_projection_is_bounded_and_does_not_duplicate_asr_partials() {
        let mut collector = TranscriptCollector::default();
        collector
            .push(OutputChunk::AsrPartial {
                text: "old".into(),
                tokens: vec![],
            })
            .unwrap();
        collector
            .push(OutputChunk::AsrPartial {
                text: "hello".into(),
                tokens: vec![],
            })
            .unwrap();
        collector
            .push(OutputChunk::TextDelta("hello".into()))
            .unwrap();
        collector.push(OutputChunk::End).unwrap();
        assert_eq!(collector.finish().unwrap(), "hello");
        let mut collector = TranscriptCollector::default();
        assert!(
            collector
                .push(OutputChunk::TextDelta("x".repeat(MAX_TRANSCRIPT_BYTES + 1)))
                .is_err()
        );
        assert!(
            TranscriptCollector::default()
                .push(OutputChunk::Embedding(vec![0.0]))
                .is_err()
        );
    }

    #[test]
    fn transcription_requires_exactly_one_terminal_but_accepts_silence() {
        assert!(TranscriptCollector::default().finish().is_err());
        let mut collector = TranscriptCollector::default();
        collector
            .push(OutputChunk::TextDelta("partial".into()))
            .unwrap();
        assert!(collector.finish().is_err());
        let mut collector = TranscriptCollector::default();
        collector.push(OutputChunk::End).unwrap();
        assert!(collector.push(OutputChunk::End).is_err());
        assert!(
            collector
                .push(OutputChunk::TextDelta("late".into()))
                .is_err()
        );
        assert_eq!(collector.finish().unwrap(), "");
    }

    #[tokio::test]
    async fn multipart_success_binds_model_and_returns_json_or_text_once() {
        let root = tempfile::tempdir().unwrap();
        let state = state(root.path().into(), Behavior::Success, true).await;
        let bytes = wav(16_000, &[-32768, 0, 16384]);
        for (format, expected) in [("json", "{\"text\":\"hello\"}"), ("text", "hello")] {
            let response = app(state.clone())
                .oneshot(request(
                    &[
                        ("model", b"default"),
                        ("file", &bytes),
                        ("response_format", format.as_bytes()),
                        ("stream", b"false"),
                    ],
                    "",
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(
                response.headers()[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with(if format == "json" {
                        "application/json"
                    } else {
                        "text/plain"
                    })
            );
            assert_eq!(
                axum::body::to_bytes(response.into_body(), MAX_TRANSCRIPT_BYTES)
                    .await
                    .unwrap()
                    .as_ref(),
                expected.as_bytes()
            );
        }
        assert_eq!(
            state.app.metrics.requests_completed.load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            state
                .app
                .metrics
                .tokens_generated_total
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(state.app.semaphore.available_permits(), 1);
        assert!(state.app.cancellations.is_empty());
    }

    #[tokio::test]
    async fn multipart_rejects_duplicate_unknown_unsupported_missing_and_query_fields() {
        let root = tempfile::tempdir().unwrap();
        let state = state(root.path().into(), Behavior::Success, true).await;
        let bytes = wav(16_000, &[-32768, 0, 16384]);
        for field in [
            "file",
            "model",
            "language",
            "prompt",
            "temperature",
            "unknown",
            "timestamp_granularities[]",
        ] {
            let response = app(state.clone())
                .oneshot(request(
                    &[("model", b"default"), ("file", &bytes), (field, b"x")],
                    "",
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
        }
        for extra in [
            [("stream", b"true".as_slice())],
            [("response_format", b"verbose_json".as_slice())],
        ] {
            let mut fields = vec![("model", b"default".as_slice()), ("file", bytes.as_slice())];
            fields.extend(extra);
            assert_eq!(
                app(state.clone())
                    .oneshot(request(&fields, ""))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::BAD_REQUEST
            );
        }
        for fields in [
            vec![("model", b"default".as_slice())],
            vec![("file", bytes.as_slice())],
        ] {
            assert_eq!(
                app(state.clone())
                    .oneshot(request(&fields, ""))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(
            app(state.clone())
                .oneshot(request(
                    &[("model", b"default"), ("file", &bytes)],
                    "?ignored=true"
                ))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        let oversized = vec![b'x'; MAX_FIELD_BYTES + 1];
        assert_eq!(
            app(state.clone())
                .oneshot(request(&[("model", &oversized), ("file", &bytes)], ""))
                .await
                .unwrap()
                .status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(state.app.metrics.requests_total.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn admission_and_backend_failures_are_bounded_and_release_resources() {
        let bytes = wav(16_000, &[-32768, 0, 16384]);
        for (behavior, audio, expected) in [
            (Behavior::Success, false, StatusCode::UNPROCESSABLE_ENTITY),
            (Behavior::Fail, true, StatusCode::UNPROCESSABLE_ENTITY),
            (Behavior::TooLarge, true, StatusCode::UNPROCESSABLE_ENTITY),
        ] {
            let root = tempfile::tempdir().unwrap();
            let state = state(root.path().into(), behavior, audio).await;
            let response = app(state.clone())
                .oneshot(request(&[("model", b"default"), ("file", &bytes)], ""))
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
            let body = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            assert!(!String::from_utf8_lossy(&body).contains("private /path"));
            assert_eq!(state.app.semaphore.available_permits(), 1);
            assert_eq!(
                state.app.metrics.in_flight_requests.load(Ordering::Relaxed),
                0
            );
        }
        let root = tempfile::tempdir().unwrap();
        let state = state(root.path().into(), Behavior::Success, true).await;
        assert_eq!(
            app(state.clone())
                .oneshot(request(&[("model", b"wrong"), ("file", &bytes)], ""))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        let permit = state.app.semaphore.acquire().await.unwrap();
        assert_eq!(
            app(state.clone())
                .oneshot(request(&[("model", b"default"), ("file", &bytes)], ""))
                .await
                .unwrap()
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(permit);
    }

    #[tokio::test]
    async fn disconnect_retains_permit_and_runtime_until_audio_worker_exits() {
        let root = tempfile::tempdir().unwrap();
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        let state = state(
            root.path().into(),
            Behavior::Blocked(started.clone(), release.clone()),
            true,
        )
        .await;
        let runtime = state
            .app
            .runtime_pool
            .read()
            .await
            .default_runtime()
            .unwrap();
        let bytes = wav(16_000, &[-32768, 0, 16384]);
        let handler = tokio::spawn(
            app(state.clone()).oneshot(request(&[("model", b"default"), ("file", &bytes)], "")),
        );
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        handler.abort();
        let _ = handler.await;
        assert_eq!(state.app.semaphore.available_permits(), 0);
        assert_eq!(runtime.active_request_leases.load(Ordering::Acquire), 1);
        assert_eq!(
            state.app.metrics.in_flight_requests.load(Ordering::Relaxed),
            1
        );
        *release.0.lock().unwrap() = true;
        release.1.notify_all();
        tokio::time::timeout(Duration::from_secs(5), async {
            while state.app.semaphore.available_permits() != 1
                || runtime.active_request_leases.load(Ordering::Acquire) != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(runtime.active_request_leases.load(Ordering::Acquire), 0);
        assert_eq!(state.app.metrics.requests_failed.load(Ordering::Relaxed), 1);
        assert!(state.app.cancellations.is_empty());
    }
}
