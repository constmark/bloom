//! Bounded direct streaming for independent OpenAI text choices.
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
    pub(crate) chat: bool,
    pub(crate) include_usage: bool,
    pub(crate) prompt_tokens: u64,
}

enum StreamItem {
    Data(serde_json::Value),
    Complete,
    Error(String),
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
    let (tx, rx) = mpsc::channel(32);
    let created = unix_seconds();
    let worker_guard = lifecycle.worker_guard();
    let worker_cancel = cancel.clone();
    task::spawn_blocking(move || {
        let _worker_guard = worker_guard;
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
        let item = match result {
            Ok(()) => StreamItem::Complete,
            Err(error) => StreamItem::Error(error.to_string()),
        };
        let _ = send_item(&tx, &worker_cancel, item);
    });

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
            StreamItem::Error(message) => {
                stream_failed.store(true, Ordering::Release);
                Some(json_event(stream_error_payload("internal_error", message)))
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
    let send = |data| {
        if cancel.is_cancelled() {
            return Err(anyhow!("request cancelled"));
        }
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
                    generated_count,
                    estimate_delta_tokens(pipeline, &delta),
                );
                let content = if request.chat {
                    json!({"content": delta})
                } else {
                    json!(delta)
                };
                send(chunk(index, content, None))
            };
            let result = pipeline.run_stream(input.clone(), &params, &mut |output| {
                if cancel.is_cancelled() {
                    return Err(anyhow!("request cancelled"));
                }
                if let OutputChunk::TextDelta(delta) = output {
                    let update = filter.push(&delta);
                    send_delta(update.text)?;
                    if update.stopped {
                        return Err(anyhow!("stop sequence reached"));
                    }
                }
                Ok(())
            });
            if cancel.is_cancelled() {
                return Err(anyhow!("request cancelled"));
            }
            if !filter.stopped() {
                result?;
            }
            send_delta(filter.finish())?;
            let tokens = pipeline.tokenize(&text)?.len();
            completion_tokens = completion_tokens.saturating_add(tokens as u64);
            generated_count.store(completion_tokens, Ordering::Relaxed);
            let reason = if filter.stopped() {
                "stop"
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
        send(json!({
            "id": request.request_id,
            "object": if request.chat { "chat.completion.chunk" } else { "text_completion" },
            "created": created,
            "model": request.model_id,
            "choices": [],
            "usage": {
                "prompt_tokens": request.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": request.prompt_tokens.saturating_add(completion_tokens)
            }
        }))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancellation_releases_a_full_queue_without_a_receiver_poll() {
        for item in [
            StreamItem::Data(json!({"delta": "blocked"})),
            StreamItem::Complete,
            StreamItem::Error("blocked error".to_string()),
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
