//! Bounded direct and scheduled execution for independent OpenAI choices.
//!
//! Choices execute sequentially under one request lease and admission permit.
//! A bounded channel propagates backpressure; dropping the response cancels the
//! whole request while the lifecycle retains resources until the worker exits.
use super::*;

pub(crate) struct TextChoiceStream {
    pub(crate) request_id: String,
    pub(crate) model_id: String,
    pub(crate) inputs: Vec<ModelInput>,
    pub(crate) echo_prompts: Option<Vec<String>>,
    pub(crate) choice_count: usize,
    pub(crate) params: GenerationParams,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) response_format: ResponseFormatMode,
    pub(crate) tool_config: Option<ToolConfig>,
    pub(crate) chat: bool,
    pub(crate) include_usage: bool,
    pub(crate) prompt_tokens: u64,
    pub(crate) scheduled: Option<ScheduledChoices>,
}

pub(crate) struct ScheduledChoices {
    pub(crate) scheduler: Arc<InferenceScheduler>,
    pub(crate) execution_guard: Arc<dyn std::any::Any + Send + Sync>,
}

enum StreamItem {
    Data(serde_json::Value),
    Complete,
    Error(&'static str, String),
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct ChoiceValidationError {
    kind: &'static str,
    message: String,
}

#[derive(Debug, thiserror::Error)]
#[error("stop sequence reached")]
struct StopSequenceReached;

fn check_candidate_result(result: Result<()>) -> Result<()> {
    match result {
        Err(error) if error.is::<StopSequenceReached>() => Ok(()),
        other => other,
    }
}

pub(crate) fn stream_text_choices(
    state: Arc<ServerState>,
    pipeline: Arc<InferencePipeline>,
    request: TextChoiceStream,
    lifecycle: Arc<InferenceLifecycle>,
    generated_count: Arc<AtomicU64>,
    cancel: CancellationToken,
    request_start: Instant,
) -> Response {
    let created = unix_seconds();
    let rx = spawn_choices(
        state,
        pipeline,
        request,
        Arc::clone(&lifecycle),
        generated_count,
        cancel.clone(),
        request_start,
        created,
    );

    let completed = Arc::new(AtomicBool::new(false));
    let failed = Arc::new(AtomicBool::new(false));
    let stream_completed = Arc::clone(&completed);
    let stream_failed = Arc::clone(&failed);
    let stream = ReceiverStream::new(rx).filter_map(move |item| {
        let event = match item {
            StreamItem::Data(data) => Some(json_event(data)),
            StreamItem::Complete => {
                stream_completed.store(true, Ordering::Release);
                None
            }
            StreamItem::Error(kind, message) => {
                stream_failed.store(true, Ordering::Release);
                Some(json_event(stream_error_payload(kind, message)))
            }
        };
        futures::future::ready(event.map(Ok::<_, std::convert::Infallible>))
    });
    let mut client_guard = lifecycle.client_guard();
    let terminal = futures::stream::once(async move {
        let success = completed.load(Ordering::Acquire) && !cancel.is_cancelled();
        let mut events = Vec::with_capacity(2);
        if !success && !failed.load(Ordering::Acquire) {
            events.push(Ok(json_event(stream_error_payload(
                "internal_error",
                "The text choices worker did not complete the request.",
            ))));
        }
        client_guard.finish(success);
        events.push(Ok::<_, std::convert::Infallible>(
            Event::default().data("[DONE]"),
        ));
        futures::stream::iter(events)
    })
    .flatten();
    Sse::new(stream.chain(terminal))
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

#[allow(clippy::too_many_arguments)]
fn spawn_choices(
    state: Arc<ServerState>,
    pipeline: Arc<InferencePipeline>,
    request: TextChoiceStream,
    lifecycle: Arc<InferenceLifecycle>,
    generated_count: Arc<AtomicU64>,
    cancel: CancellationToken,
    request_start: Instant,
    created: u64,
) -> mpsc::Receiver<StreamItem> {
    let (tx, rx) = mpsc::channel(32);
    let worker_guard = lifecycle.worker_guard();
    let worker_cancel = cancel;
    task::spawn_blocking(move || {
        let _worker_guard = worker_guard;
        let inference_start = Instant::now();
        let result = run_choices(
            &state,
            &pipeline,
            &request,
            &tx,
            &generated_count,
            &worker_cancel,
            created,
            request_start,
        );
        state
            .app
            .metrics
            .record_inference_latency(inference_start.elapsed().as_secs_f64());
        let item = match result {
            Ok(()) => StreamItem::Complete,
            Err(error) => StreamItem::Error(
                error
                    .downcast_ref::<ChoiceValidationError>()
                    .map_or("internal_error", |error| error.kind),
                error.to_string(),
            ),
        };
        let _ = send_item(&tx, &worker_cancel, item);
    });

    rx
}

/// Collect the same validated candidate events for a buffered response. A
/// failed candidate fails the whole response, with one shared admission lease.
pub(crate) async fn buffered_text_choices(
    state: Arc<ServerState>,
    pipeline: Arc<InferencePipeline>,
    mut request: TextChoiceStream,
    lifecycle: Arc<InferenceLifecycle>,
    generated_count: Arc<AtomicU64>,
    cancel: CancellationToken,
    request_start: Instant,
) -> Response {
    let created = unix_seconds();
    let request_id = request.request_id.clone();
    let model_id = request.model_id.clone();
    let chat = request.chat;
    let include_logprobs = request.params.logprobs.is_some();
    let mut choices = (0..request.inputs.len() * request.choice_count)
        .map(|index| if chat {
            json!({"index": index, "message": {"role": "assistant", "content": ""}, "finish_reason": null})
        } else {
            json!({"index": index, "text": "", "finish_reason": null})
        }).collect::<Vec<_>>();
    if include_logprobs {
        for choice in &mut choices {
            choice["logprobs"] = if chat {
                json!({"content": [], "refusal": null})
            } else {
                json!({"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []})
            };
        }
    }
    request.include_usage = true;
    let mut guard = lifecycle.client_guard();
    let mut rx = spawn_choices(
        state,
        pipeline,
        request,
        lifecycle,
        generated_count,
        cancel.clone(),
        request_start,
        created,
    );
    let mut usage = None;
    let mut content_bytes = 0_usize;
    while let Some(item) = rx.recv().await {
        match item {
            StreamItem::Data(data) => {
                if let Some(value) = data.get("usage") {
                    usage = Some(value.clone());
                }
                for choice in data["choices"].as_array().into_iter().flatten() {
                    let index = choice["index"].as_u64().unwrap_or(u64::MAX) as usize;
                    let Some(target) = choices.get_mut(index) else {
                        return error_response(
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "internal_error",
                            "The choices worker emitted an invalid candidate index.",
                        );
                    };
                    let delta = if chat {
                        &choice["delta"]["content"]
                    } else {
                        &choice["text"]
                    };
                    if let Some(text) = delta.as_str() {
                        content_bytes = content_bytes.saturating_add(text.len());
                        let field = if chat {
                            &mut target["message"]["content"]
                        } else {
                            &mut target["text"]
                        };
                        if let serde_json::Value::String(value) = field {
                            value.push_str(text);
                        } else {
                            *field = json!(text);
                        }
                    }
                    if let Some(calls) = choice["delta"].get("tool_calls") {
                        let mut calls = calls.clone();
                        if let Some(items) = calls.as_array_mut() {
                            for item in items {
                                if let Some(object) = item.as_object_mut() {
                                    object.remove("index");
                                }
                            }
                        }
                        content_bytes = content_bytes.saturating_add(calls.to_string().len());
                        target["message"]["content"] = serde_json::Value::Null;
                        target["message"]["tool_calls"] = calls;
                    }
                    if let Some(call) = choice["delta"].get("function_call") {
                        content_bytes = content_bytes.saturating_add(call.to_string().len());
                        target["message"]["content"] = serde_json::Value::Null;
                        target["message"]["function_call"] = call.clone();
                    }
                    if let Some(probabilities) =
                        choice.get("logprobs").filter(|value| !value.is_null())
                    {
                        content_bytes =
                            content_bytes.saturating_add(probabilities.to_string().len());
                        let fields: &[&str] = if chat {
                            &["content"]
                        } else {
                            &["tokens", "token_logprobs", "top_logprobs", "text_offset"]
                        };
                        for field in fields {
                            if let Some(items) = probabilities[*field].as_array() {
                                target["logprobs"][*field]
                                    .as_array_mut()
                                    .expect("initialized logprob arrays")
                                    .extend(items.iter().cloned());
                            }
                        }
                    }
                    if content_bytes > MAX_RESPONSES_STREAM_OUTPUT_BYTES {
                        return error_response(
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "output_limit_exceeded",
                            "The combined choices exceeded the output byte limit.",
                        );
                    }
                    if !choice["finish_reason"].is_null() {
                        target["finish_reason"] = choice["finish_reason"].clone();
                    }
                }
            }
            StreamItem::Complete
                if !cancel.is_cancelled()
                    && usage.is_some()
                    && choices
                        .iter()
                        .all(|choice| !choice["finish_reason"].is_null()) =>
            {
                guard.finish(true);
                return Json(
                    json!({"id": request_id, "model": model_id, "created": created,
                    "object": if chat { "chat.completion" } else { "text_completion" },
                    "choices": choices, "usage": usage}),
                )
                .into_response();
            }
            StreamItem::Error(kind, message) => {
                guard.finish(false);
                let status = if cancel.is_cancelled() {
                    axum::http::StatusCode::REQUEST_TIMEOUT
                } else if matches!(kind, "invalid_tool_call" | "invalid_response_format") {
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY
                } else {
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR
                };
                return error_response(status, kind, message);
            }
            StreamItem::Complete => break,
        }
    }
    guard.finish(false);
    if cancel.is_cancelled() {
        error_response(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            "request_cancelled",
            "The generation request was cancelled.",
        )
    } else {
        error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "The choices worker did not complete the request.",
        )
    }
}

// This runs only inside the blocking inference worker. Waiting on the channel
// must still observe cancellation when a connected client stops consuming.
fn send_item(
    tx: &mpsc::Sender<StreamItem>,
    cancel: &CancellationToken,
    item: StreamItem,
) -> Result<()> {
    tokio::runtime::Handle::current().block_on(async {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(anyhow!("request cancelled")),
            result = tx.send(item) => result.map_err(|_| anyhow!("client disconnected")),
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn run_choices(
    state: &ServerState,
    pipeline: &InferencePipeline,
    request: &TextChoiceStream,
    tx: &mpsc::Sender<StreamItem>,
    generated_count: &Arc<AtomicU64>,
    cancel: &CancellationToken,
    created: u64,
    request_start: Instant,
) -> Result<()> {
    let first_token = Arc::new(AtomicBool::new(false));
    let last_token = Arc::new(std::sync::Mutex::new(None));
    let mut completion_tokens = 0_u64;
    let requires_generation_metadata =
        request.scheduled.is_none() && pipeline.model().supports_generation_metadata();
    let mut exact_prompt_counts = vec![None; request.inputs.len()];
    // Never publish unvalidated structured content or the private tool control
    // protocol. Only one unvalidated candidate is accumulated at a time.
    let legacy = legacy_function_output(request.tool_config.as_ref());
    let buffer_output = request.tool_config.is_some()
        || !matches!(request.response_format, ResponseFormatMode::Text);
    let output_bytes = std::sync::atomic::AtomicUsize::new(0);
    let send = |data| {
        if cancel.is_cancelled() {
            return Err(anyhow!("request cancelled"));
        }
        charge_output_budget(&output_bytes, &data)?;
        send_item(tx, cancel, StreamItem::Data(data))
    };
    let chunk = |index: usize, content: serde_json::Value, finish: Option<&str>| {
        let choice = if request.chat {
            json!({"index": index, "delta": content, "finish_reason": finish})
        } else {
            json!({"index": index, "text": content, "finish_reason": finish})
        };
        json!({
            "id": request.request_id,
            "object": if request.chat { "chat.completion.chunk" } else { "text_completion" },
            "created": created,
            "model": request.model_id,
            "choices": [choice]
        })
    };
    for (prompt_index, input) in request.inputs.iter().enumerate() {
        for choice_index in 0..request.choice_count {
            if cancel.is_cancelled() {
                return Err(anyhow!("request cancelled"));
            }
            let index = prompt_index * request.choice_count + choice_index;
            let mut params = request.params.clone();
            params.seed = params.seed.map(|seed| seed.wrapping_add(index as u64));
            if request.chat {
                send(chunk(
                    index,
                    json!({"role": "assistant", "content": ""}),
                    None,
                ))?;
            } else if let Some(prompts) = &request.echo_prompts {
                send(chunk(index, json!(prompts[prompt_index]), None))?;
            }
            let mut filter = StopSequenceFilter::new(request.stop_sequences.clone());
            let mut text = String::new();
            let timing_count = AtomicU64::new(0);
            let mut send_delta = |delta: String| -> Result<()> {
                if delta.is_empty() {
                    return Ok(());
                }
                if text.len().saturating_add(delta.len()) > MAX_RESPONSES_STREAM_OUTPUT_BYTES {
                    return Err(anyhow!("text choice exceeded the output byte limit"));
                }
                text.push_str(&delta);
                record_stream_tokens(
                    state,
                    request_start,
                    &first_token,
                    &last_token,
                    if requires_generation_metadata {
                        &timing_count
                    } else {
                        generated_count
                    },
                    estimate_delta_tokens(pipeline, &delta),
                );
                if buffer_output {
                    return Ok(());
                }
                let content = if request.chat {
                    json!({"content": delta})
                } else {
                    json!(delta)
                };
                send(chunk(index, content, None))
            };
            let mut probability_bytes = Vec::new();
            let mut probability_count = 0usize;
            let mut probability_memory = 0usize;
            let mut generation = bloomai_engine::GenerationStreamState::default();
            let mut sink = |output| {
                if cancel.is_cancelled() {
                    return Err(anyhow!("request cancelled"));
                }
                generation
                    .observe(&output, params.max_tokens)
                    .map_err(anyhow::Error::msg)?;
                if let OutputChunk::TokenUsage { output_tokens, .. } = output {
                    generated_count.store(
                        completion_tokens.saturating_add(output_tokens as u64),
                        Ordering::Relaxed,
                    );
                } else if let OutputChunk::TokenLogprobs(probability) = output {
                    let Some(top_count) = params.logprobs else {
                        return Err(anyhow!(
                            "backend emitted unrequested token log probabilities"
                        ));
                    };
                    probability.validate().map_err(anyhow::Error::msg)?;
                    if probability.top_logprobs.len() > top_count
                        || probability_count >= params.max_tokens
                    {
                        return Err(anyhow!(
                            "backend exceeded the requested token logprob limits"
                        ));
                    }
                    probability_memory = probability_memory.saturating_add(probability.byte_size());
                    if probability_memory > 8 * 1024 * 1024 {
                        return Err(anyhow!("token logprob metadata exceeded the byte limit"));
                    }
                    let offset = complete_utf8_char_count(&probability_bytes)?;
                    probability_bytes.extend_from_slice(&probability.token.bytes);
                    complete_utf8_char_count(&probability_bytes)?;
                    probability_count += 1;
                    let mut data = chunk(
                        index,
                        if request.chat { json!({}) } else { json!("") },
                        None,
                    );
                    data["choices"][0]["logprobs"] =
                        public_token_logprobs(&probability, request.chat, offset);
                    send(data)?;
                } else if let OutputChunk::TextDelta(delta) = output {
                    let update = filter.push(&delta);
                    send_delta(update.text)?;
                    if update.stopped {
                        return Err(StopSequenceReached.into());
                    }
                }
                Ok(())
            };
            let mut scheduled_tokens = 0;
            let result = if let Some(scheduled) = &request.scheduled {
                run_scheduled_choice(
                    scheduled,
                    pipeline,
                    request,
                    &next_request_id(state, "ifb-choice"),
                    input,
                    &params,
                    cancel,
                    &mut scheduled_tokens,
                    &mut sink,
                )
            } else {
                pipeline.run_stream(input.clone(), &params, &mut sink)
            };
            if cancel.is_cancelled() {
                return Err(anyhow!("request cancelled"));
            }
            // A chunk can match stop before its visible prefix fails to send.
            // Only our explicit stop sentinel is successful early termination.
            check_candidate_result(result)?;
            send_delta(filter.finish())?;
            if !filter.stopped() {
                generation
                    .finish(requires_generation_metadata || params.logprobs.is_some())
                    .map_err(anyhow::Error::msg)?;
            }
            let metadata = generation.metadata();
            let tokens = if params.logprobs.is_some() {
                if probability_bytes != text.as_bytes() {
                    return Err(anyhow!(
                        "backend token logprob bytes do not match generated text"
                    ));
                }
                validated_logprob_usage(probability_count, metadata, params.max_tokens)?
            } else if request.scheduled.is_some() {
                scheduled_tokens
            } else if let Some(exact) = generation.output_tokens() {
                exact
            } else if text.is_empty() {
                0
            } else {
                pipeline.tokenize(&text)?.len()
            };
            if let Some(input_count) = generation.input_tokens() {
                if exact_prompt_counts[prompt_index].is_some_and(|previous| previous != input_count)
                {
                    return Err(anyhow!(
                        "backend changed prompt token usage between choices"
                    ));
                }
                exact_prompt_counts[prompt_index] = Some(input_count);
            }
            completion_tokens = completion_tokens.saturating_add(tokens as u64);
            generated_count.store(completion_tokens, Ordering::Relaxed);
            let parsed = if buffer_output {
                let output = parse_chat_output(
                    &text,
                    &request.response_format,
                    request.tool_config.as_ref(),
                )
                .map_err(|message| ChoiceValidationError {
                    kind: if request.tool_config.is_some() {
                        "invalid_tool_call"
                    } else {
                        "invalid_response_format"
                    },
                    message,
                })?;
                let content = if request.chat {
                    chat_output_delta_for_choice(&request.request_id, index, &output, legacy)
                } else {
                    json!(text)
                };
                send(chunk(index, content, None))?;
                Some(output)
            } else {
                None
            };
            let reason = if filter.stopped() {
                "stop"
            } else if let Some(metadata) = metadata {
                match metadata.reason {
                    bloomai_engine::GenerationFinishReason::Length => "length",
                    bloomai_engine::GenerationFinishReason::Eos
                    | bloomai_engine::GenerationFinishReason::Stop => parsed
                        .as_ref()
                        .map_or("stop", |output| chat_output_stop_reason(output, legacy)),
                }
            } else if let Some(output) = &parsed {
                chat_output_finish_reason(output, tokens, params.max_tokens, legacy)
            } else {
                generation_finish_reason(tokens, params.max_tokens)
            };
            send(chunk(
                index,
                if request.chat { json!({}) } else { json!("") },
                Some(reason),
            ))?;
        }
    }
    if request.include_usage {
        let prompt_tokens = exact_prompt_counts
            .iter()
            .try_fold(0_u64, |sum, count| {
                count.map(|value| sum.saturating_add(value as u64))
            })
            .unwrap_or(request.prompt_tokens);
        send(json!({
            "id": request.request_id,
            "object": if request.chat { "chat.completion.chunk" } else { "text_completion" },
            "created": created,
            "model": request.model_id,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens.saturating_add(completion_tokens)
            }
        }))?;
    }
    Ok(())
}

// Logprob records cover visible samples; only an explicit EOS finish permits
// one additional accepted token. Counts must never be used to guess EOS.
fn validated_logprob_usage(
    visible: usize,
    metadata: Option<bloomai_engine::GenerationMetadata>,
    maximum: usize,
) -> Result<usize> {
    let metadata =
        metadata.ok_or_else(|| anyhow!("backend omitted completion metadata for logprobs"))?;
    let expected = visible
        .checked_add(usize::from(
            metadata.reason == bloomai_engine::GenerationFinishReason::Eos,
        ))
        .ok_or_else(|| anyhow!("logprob token count overflowed"))?;
    if metadata.output_tokens > maximum || metadata.output_tokens != expected {
        return Err(anyhow!(
            "backend token usage does not match logprob records and termination reason"
        ));
    }
    Ok(metadata.output_tokens)
}

// Count only complete Unicode scalar values before a token. A byte fragment
// inside one character shares that character's offset with its sibling tokens.
fn complete_utf8_char_count(bytes: &[u8]) -> Result<usize> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(text.chars().count()),
        Err(error) if error.error_len().is_none() => {
            Ok(std::str::from_utf8(&bytes[..error.valid_up_to()])?
                .chars()
                .count())
        }
        Err(_) => Err(anyhow!("token logprob bytes contain invalid UTF-8")),
    }
}

fn public_token_logprobs(
    probability: &bloomai_engine::TokenLogprobs,
    chat: bool,
    offset: usize,
) -> serde_json::Value {
    let token = &probability.token;
    if chat {
        let alternatives = probability
            .top_logprobs
            .iter()
            .map(|item| {
                json!({
                    "token": item.token, "bytes": item.bytes, "logprob": item.logprob,
                })
            })
            .collect::<Vec<_>>();
        json!({"content": [{"token": token.token, "bytes": token.bytes,
            "logprob": token.logprob, "top_logprobs": alternatives}], "refusal": null})
    } else {
        // Legacy's string-keyed map cannot distinguish byte fragments sharing
        // a lossy display. Retain the most likely alternative for each display,
        // then ensure the actual chosen token's probability is represented.
        let mut alternatives = serde_json::Map::new();
        for item in &probability.top_logprobs {
            alternatives
                .entry(item.token.clone())
                .or_insert_with(|| json!(item.logprob));
        }
        alternatives.insert(token.token.clone(), json!(token.logprob));
        json!({"tokens": [token.token], "token_logprobs": [token.logprob],
            "top_logprobs": [alternatives], "text_offset": [offset]})
    }
}

// Runs on the blocking choices worker. Drop waits for any scheduler execution
// to finish before releasing this candidate. Internal IDs are never reused.
struct ScheduledCandidateGuard<'a> {
    scheduler: &'a InferenceScheduler,
    request_id: &'a str,
}

impl Drop for ScheduledCandidateGuard<'_> {
    fn drop(&mut self) {
        self.scheduler.cancel_request_and_wait(self.request_id);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_scheduled_choice(
    scheduled: &ScheduledChoices,
    pipeline: &InferencePipeline,
    request: &TextChoiceStream,
    scheduler_request_id: &str,
    input: &ModelInput,
    params: &GenerationParams,
    cancel: &CancellationToken,
    token_count: &mut usize,
    sink: &mut dyn FnMut(OutputChunk) -> Result<()>,
) -> Result<()> {
    let ModelInput::Text { prompt } = input else {
        return Err(anyhow!("scheduled choices require text input"));
    };
    let prompt_tokens = pipeline.tokenize(prompt)?;
    if cancel.is_cancelled() {
        return Err(anyhow!("request cancelled"));
    }
    let (tx, mut rx) = mpsc::unbounded_channel();
    let _candidate = ScheduledCandidateGuard {
        scheduler: &scheduled.scheduler,
        request_id: scheduler_request_id,
    };
    scheduled
        .scheduler
        .token_senders
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(scheduler_request_id.to_string(), tx);
    scheduled.scheduler.submit_with_execution_guard(
        Request {
            id: scheduler_request_id.to_string(),
            model_id: request.model_id.clone(),
            prompt_tokens,
            generated_tokens: Vec::new(),
            params: params.clone(),
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        },
        Arc::clone(&scheduled.execution_guard),
    )?;
    if cancel.is_cancelled() {
        return Err(anyhow!("request cancelled"));
    }
    let mut tokens = Vec::new();
    let mut emitted = String::new();
    let mut saw_eos = false;
    loop {
        let item = tokio::runtime::Handle::current().block_on(async {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => Err(anyhow!("request cancelled")),
                item = rx.recv() => Ok(item),
            }
        })?;
        let Some(item) = item else {
            break;
        };
        let token = item.map_err(anyhow::Error::msg)?;
        if pipeline.model().eos_token_ids().contains(&token) {
            saw_eos = true;
            break;
        }
        *token_count += 1;
        if *token_count > params.max_tokens {
            return Err(anyhow!("scheduler exceeded the token limit"));
        }
        tokens.push(token);
        if let Some(delta) =
            stable_decoded_delta(&pipeline.detokenize(&tokens)?, &mut emitted, false)?
        {
            sink(OutputChunk::TextDelta(delta))?;
        }
    }
    if !saw_eos && *token_count < params.max_tokens {
        return Err(anyhow!(
            "scheduler ended the candidate without EOS or its token budget"
        ));
    }
    if let Some(delta) = stable_decoded_delta(&pipeline.detokenize(&tokens)?, &mut emitted, true)? {
        sink(OutputChunk::TextDelta(delta))?;
    }
    Ok(())
}

// Detokenization can expose U+FFFD for an incomplete UTF-8 byte sequence.
// Delay that tail until subsequent tokens resolve it; terminal decoding still
// matches the tokenizer's final policy, including literal replacement chars.
pub(crate) fn stable_decoded_delta(
    decoded: &str,
    emitted: &mut String,
    terminal: bool,
) -> Result<Option<String>> {
    if decoded.len() > MAX_RESPONSES_STREAM_OUTPUT_BYTES {
        return Err(anyhow!("decoded candidate exceeded the output byte limit"));
    }
    let stable = if terminal {
        decoded
    } else {
        decoded.trim_end_matches('\u{fffd}')
    };
    let delta = stable
        .strip_prefix(emitted.as_str())
        .ok_or_else(|| anyhow!("tokenizer rewrote previously emitted text"))?;
    if delta.is_empty() {
        return Ok(None);
    }
    let delta = delta.to_string();
    emitted.push_str(&delta);
    Ok(Some(delta))
}

// Count serialized event bytes before enqueueing. This bounds all candidates
// collectively, including escaped tool arguments and echoed prompts, without
// allocating a second serialized copy of a potentially large event.
fn charge_output_budget(
    total: &std::sync::atomic::AtomicUsize,
    value: &serde_json::Value,
) -> Result<()> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > MAX_RESPONSES_STREAM_OUTPUT_BYTES.saturating_sub(self.0) {
                return Err(std::io::Error::other("choice output byte limit exceeded"));
            }
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, value).map_err(|_| ChoiceValidationError {
        kind: "output_limit_exceeded",
        message: "The combined choices exceeded the output byte limit.".into(),
    })?;
    total
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
            used.checked_add(counter.0)
                .filter(|next| *next <= MAX_RESPONSES_STREAM_OUTPUT_BYTES)
        })
        .map_err(|_| ChoiceValidationError {
            kind: "output_limit_exceeded",
            message: "The combined choices exceeded the output byte limit.".into(),
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprob_usage_requires_the_explicit_termination_reason() {
        use bloomai_engine::{GenerationFinishReason as Reason, GenerationMetadata};
        let metadata = |output_tokens, reason| {
            Some(GenerationMetadata {
                input_tokens: 1,
                output_tokens,
                reason,
            })
        };
        assert_eq!(
            validated_logprob_usage(0, metadata(1, Reason::Eos), 4).unwrap(),
            1
        );
        assert_eq!(
            validated_logprob_usage(3, metadata(4, Reason::Eos), 4).unwrap(),
            4
        );
        assert_eq!(
            validated_logprob_usage(4, metadata(4, Reason::Length), 4).unwrap(),
            4
        );
        for (visible, reported, maximum) in [
            (1, None, 4),
            (2, metadata(1, Reason::Eos), 4),
            (0, metadata(2, Reason::Eos), 4),
            (4, metadata(5, Reason::Eos), 4),
            (3, metadata(4, Reason::Length), 4),
        ] {
            assert!(validated_logprob_usage(visible, reported, maximum).is_err());
        }
    }

    #[test]
    fn public_logprobs_preserve_partial_utf8_and_legacy_offsets() {
        let first = bloomai_engine::TokenLogprob {
            token_id: 10,
            token: "�".into(),
            bytes: vec![0xe2],
            logprob: -0.5,
        };
        let probability = bloomai_engine::TokenLogprobs {
            token: first.clone(),
            top_logprobs: vec![first],
        };
        let chat = public_token_logprobs(&probability, true, 0);
        assert_eq!(chat["content"][0]["bytes"], json!([226]));
        assert_eq!(chat["content"][0]["logprob"], -0.5);
        let mut bytes = Vec::new();
        for fragment in [vec![0xe2], vec![0x82], vec![0xac]] {
            assert_eq!(complete_utf8_char_count(&bytes).unwrap(), 0);
            bytes.extend(fragment);
        }
        assert_eq!(complete_utf8_char_count(&bytes).unwrap(), 1);
        let legacy = public_token_logprobs(&probability, false, 1);
        assert_eq!(legacy["text_offset"], json!([1]));
        assert_eq!(legacy["top_logprobs"][0]["�"], -0.5);
        assert!(complete_utf8_char_count(&[0xff]).is_err());
    }

    #[test]
    fn scheduled_decode_preserves_split_unicode_and_terminal_replacements() {
        let mut emitted = String::new();
        let mut deltas = String::new();
        for text in ["你好", "你好，Bloom �", "你好，Bloom ��", "你好，Bloom 🌸"]
        {
            if let Some(delta) = stable_decoded_delta(text, &mut emitted, false).unwrap() {
                deltas.push_str(&delta);
            }
        }
        assert_eq!(deltas, "你好，Bloom 🌸");
        assert!(stable_decoded_delta("changed prefix", &mut emitted, false).is_err());
        assert!(
            stable_decoded_delta("你好，Bloom 🌸�", &mut emitted, false)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            stable_decoded_delta("你好，Bloom 🌸�", &mut emitted, true).unwrap(),
            Some("�".into())
        );
    }

    #[test]
    fn output_budget_bounds_all_candidates_including_json_escaping() {
        let total = std::sync::atomic::AtomicUsize::new(MAX_RESPONSES_STREAM_OUTPUT_BYTES - 6);
        charge_output_budget(&total, &json!("hi")).unwrap(); // four JSON bytes
        assert_eq!(
            total.load(Ordering::Relaxed),
            MAX_RESPONSES_STREAM_OUTPUT_BYTES - 2
        );
        assert!(charge_output_budget(&total, &json!("\n")).is_err());
        assert_eq!(
            total.load(Ordering::Relaxed),
            MAX_RESPONSES_STREAM_OUTPUT_BYTES - 2
        );
        charge_output_budget(&total, &json!("")).unwrap();
        assert_eq!(
            total.load(Ordering::Relaxed),
            MAX_RESPONSES_STREAM_OUTPUT_BYTES
        );
        assert!(charge_output_budget(&total, &json!({})).is_err());
    }

    #[test]
    fn stop_match_does_not_hide_an_output_budget_failure() {
        let mut filter = StopSequenceFilter::new(vec!["<stop>".into()]);
        let update = filter.push("a long visible prefix<stop>");
        assert!(update.stopped);
        let total = std::sync::atomic::AtomicUsize::new(MAX_RESPONSES_STREAM_OUTPUT_BYTES - 4);
        let result = charge_output_budget(&total, &json!(update.text));
        let error = check_candidate_result(result).unwrap_err();
        assert_eq!(
            error.downcast_ref::<ChoiceValidationError>().unwrap().kind,
            "output_limit_exceeded"
        );
        // A smaller finish event still fits: the failure must propagate even
        // though neither the stop flag nor the budget alone would stop it.
        charge_output_budget(&total, &json!("")).unwrap();
        assert!(check_candidate_result(Err(StopSequenceReached.into())).is_ok());
        assert!(check_candidate_result(Err(anyhow!("executor failed"))).is_err());
    }

    #[tokio::test]
    async fn cancellation_releases_a_full_queue_without_a_receiver_poll() {
        for item in [
            StreamItem::Data(json!({"delta": "blocked"})),
            StreamItem::Complete,
            StreamItem::Error("internal_error", "blocked error".to_string()),
        ] {
            let (tx, mut rx) = mpsc::channel(1);
            tx.send(StreamItem::Data(json!({"delta": "queued"})))
                .await
                .unwrap();
            let cancel = CancellationToken::new();
            let worker_cancel = cancel.clone();
            let (entered, waiting) = tokio::sync::oneshot::channel();
            let worker = task::spawn_blocking(move || {
                let _ = entered.send(());
                send_item(&tx, &worker_cancel, item)
            });
            waiting.await.unwrap();
            // Leave the queue full: cancellation must unblock both payload and
            // worker-terminal sends without requiring the client to read.
            cancel.cancel();
            let result = tokio::time::timeout(Duration::from_secs(2), worker)
                .await
                .unwrap()
                .unwrap();
            assert!(result.is_err());
            assert!(matches!(rx.recv().await, Some(StreamItem::Data(_))));
            assert!(rx.recv().await.is_none());
        }
    }
}
