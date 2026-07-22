#![allow(unused_imports, dead_code)]
use super::*;

pub(crate) async fn handle_health(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let in_flight = state
        .metrics
        .in_flight_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let requests_total = state
        .metrics
        .requests_total
        .load(std::sync::atomic::Ordering::Relaxed);
    let model_id = state.model_id.read().await.clone();
    Json(json!({
        "status": "ok",
        "model": if model_id.is_empty() { "loading..." } else { &model_id },
        "in_flight_requests": in_flight,
        "requests_total": requests_total,
        "speculative_mode": state.speculative_mode,
    }))
}

pub(crate) async fn handle_ready(State(state): State<Arc<ServerState>>) -> axum::response::Response {
    let in_flight = state.metrics.in_flight_requests.load(Ordering::Relaxed);
    let available_permits = state.semaphore.available_permits();
    let mut memory = bloomai_engine::MemoryTelemetry::new();
    memory.refresh_ram();
    let ready =
        state.ready.load(Ordering::Relaxed) && available_permits > 0 && !memory.is_high_pressure();
    let load_progress = state.load_progress.load(Ordering::Relaxed);
    let model_id = state.model_id.read().await.clone();

    let body = json!({
        "status": if ready { "ready" } else { "not_ready" },
        "progress": load_progress,
        "model": if model_id.is_empty() { "loading..." } else { &model_id },
        "in_flight_requests": in_flight,
        "available_permits": available_permits,
        "memory_pressure_high": memory.is_high_pressure(),
        "ram_utilization": memory.ram_utilization(),
    });

    if ready {
        Json(body).into_response()
    } else {
        (axum::http::StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
    }
}

// ─── /metrics ──────────────────────────────────────────────────────────────

pub(crate) async fn handle_metrics(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let kv_metrics = {
        let guard = state.kv_cache_pool.read().await;
        if let Some(ref pool) = *guard {
            pool.get_metrics()
        } else {
            bloomai_engine::KvCacheMetrics::default()
        }
    };
    let queue_stats = {
        let guard = state.scheduler.read().await;
        if let Some(ref scheduler) = *guard {
            scheduler.queue_stats()
        } else {
            (0, 0, 0)
        }
    };
    let cachemesh_metrics = {
        let guard = state.cachemesh.read().await;
        guard.as_ref().map(|mesh| mesh.metrics())
    };
    let body =
        state
            .metrics
            .render_prometheus(&kv_metrics, cachemesh_metrics.as_ref(), queue_stats);
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

// ─── /v1/observability ─────────────────────────────────────────────────────

pub(crate) async fn handle_observability(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let kv_metrics = {
        let guard = state.kv_cache_pool.read().await;
        if let Some(ref pool) = *guard {
            pool.get_metrics()
        } else {
            bloomai_engine::KvCacheMetrics::default()
        }
    };
    let kv_utilization = if kv_metrics.total_blocks > 0 {
        (kv_metrics.total_blocks - kv_metrics.free_blocks) as f64 / kv_metrics.total_blocks as f64
    } else {
        0.0
    };
    let queue_stats = {
        let guard = state.scheduler.read().await;
        if let Some(ref scheduler) = *guard {
            scheduler.queue_stats()
        } else {
            (0, 0, 0)
        }
    };
    let mut memory = bloomai_engine::MemoryTelemetry::new();
    memory.refresh_ram();

    let model_id = state.model_id.read().await.clone();
    let startup_memory_estimate = state.memory_estimate.read().await.clone();
    let cachemesh = {
        let guard = state.cachemesh.read().await;
        guard.as_ref().map(|mesh| mesh.metrics())
    };

    Json(json!({
        "object": "bloom.observability_snapshot",
        "created": unix_seconds(),
        "model": if model_id.is_empty() { "loading..." } else { &model_id },
        "ready": state.ready.load(Ordering::Relaxed),
        "speculative_mode": state.speculative_mode,
        "requests": {
            "total": state.metrics.requests_total.load(Ordering::Relaxed),
            "completed": state.metrics.requests_completed.load(Ordering::Relaxed),
            "failed": state.metrics.requests_failed.load(Ordering::Relaxed),
            "in_flight": state.metrics.in_flight_requests.load(Ordering::Relaxed)
        },
        "tokens": {
            "prompt_total": state.metrics.prompt_tokens_total.load(Ordering::Relaxed),
            "generated_total": state.metrics.tokens_generated_total.load(Ordering::Relaxed)
        },
        "scheduler": {
            "ifb_enabled": state.enable_ifb,
            "prefill_queue": queue_stats.0,
            "decoding_queue": queue_stats.1,
            "active_requests": queue_stats.2
        },
        "startup_memory_estimate": startup_memory_estimate,
        "kv_cache": {
            "total_blocks": kv_metrics.total_blocks,
            "free_blocks": kv_metrics.free_blocks,
            "active_blocks": kv_metrics.active_blocks,
            "cached_blocks": kv_metrics.cached_blocks,
            "hits": kv_metrics.hits,
            "misses": kv_metrics.misses,
            "evictions": kv_metrics.evictions,
            "reuses": kv_metrics.reuses,
            "utilization": kv_utilization
        },
        "cachemesh": cachemesh,
        "memory": memory,
    }))
}

// ─── /v1/kv-cache-stats ────────────────────────────────────────────────────

pub(crate) async fn handle_kv_cache_stats(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let kv_metrics = {
        let guard = state.kv_cache_pool.read().await;
        if let Some(ref pool) = *guard {
            pool.get_metrics()
        } else {
            bloomai_engine::KvCacheMetrics::default()
        }
    };
    let utilization = if kv_metrics.total_blocks > 0 {
        (kv_metrics.total_blocks - kv_metrics.free_blocks) as f64 / kv_metrics.total_blocks as f64
    } else {
        0.0
    };
    let cachemesh = {
        let guard = state.cachemesh.read().await;
        guard.as_ref().map(|mesh| mesh.metrics())
    };
    Json(json!({
        "total_blocks": kv_metrics.total_blocks,
        "free_blocks": kv_metrics.free_blocks,
        "active_blocks": kv_metrics.active_blocks,
        "cached_blocks": kv_metrics.cached_blocks,
        "hits": kv_metrics.hits,
        "misses": kv_metrics.misses,
        "evictions": kv_metrics.evictions,
        "reuses": kv_metrics.reuses,
        "utilization": utilization,
        "cachemesh": cachemesh,
    }))
}

// ─── /v1/models ─────────────────────────────────────────────────────────────

pub(crate) async fn handle_models(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let model_id = state.model_id.read().await.clone();
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": if model_id.is_empty() { "loading..." } else { &model_id },
                "object": "model",
                "created": 1677610200,
                "owned_by": "bloom"
            }
        ]
    }))
}

// ─── /v1/chat/completions ───────────────────────────────────────────────────

pub(crate) async fn handle_chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<ChatRequest>,
) -> impl IntoResponse {
    if !state.ready.load(Ordering::Relaxed) {
        return error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "model_loading",
            format!(
                "Model is still loading (progress: {}%)",
                state.load_progress.load(Ordering::Relaxed)
            ),
        );
    }

    let pipeline = match state.get_pipeline().await {
        Ok(p) => p,
        Err(e) => {
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model_loading",
                e.to_string(),
            );
        }
    };

    let model_family = {
        let guard = state.model_family.read().await;
        guard.clone()
    };

    let model_id = {
        let guard = state.model_id.read().await;
        guard.clone()
    };

    let scheduler_opt = {
        let guard = state.scheduler.read().await;
        guard.clone()
    };

    let _permit = match state.semaphore.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = std::time::Instant::now();

    let response_format = match response_format_mode(payload.response_format.as_ref()) {
        Ok(mode) => mode,
        Err(message) => {
            state
                .metrics
                .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };

    let core_response_format = match &response_format {
        ResponseFormatMode::Text => bloomai_core::ResponseFormat::Text,
        ResponseFormatMode::JsonObject => bloomai_core::ResponseFormat::JsonObject,
        ResponseFormatMode::JsonSchema(schema) => {
            bloomai_core::ResponseFormat::JsonSchema(schema.clone())
        }
    };

    let params = GenerationParams {
        max_tokens: payload.max_tokens.unwrap_or(128),
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        seed: payload.seed,
        response_format: Some(core_response_format),
    };

    let prompt = apply_response_format_instruction(
        chat_prompt(&payload.messages, &model_family),
        &response_format,
    );
    let prompt_tokens_vec = pipeline.tokenize(&prompt).unwrap_or_default();
    let prompt_tokens = prompt_tokens_vec.len();
    let input = ModelInput::Text { prompt };
    let request_id = next_request_id(&state, "chatcmpl");
    let mut cancel_guard = CancelTokenGuard::register(&state, request_id.clone());
    let cancel_token = cancel_guard.token();

    if state.enable_ifb {
        let scheduler = scheduler_opt.unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<Result<u32, String>>();
        scheduler
            .token_senders
            .lock()
            .unwrap()
            .insert(request_id.clone(), tx);

        let req = Request {
            id: request_id.clone(),
            model_id: model_id.clone(),
            prompt_tokens: prompt_tokens_vec,
            generated_tokens: Vec::new(),
            params: params.clone(),
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        if let Err(e) = scheduler.submit(req) {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("Failed to submit to scheduler: {}", e),
            );
        }

        if !payload.stream {
            let mut generated_tokens = Vec::new();
            while let Some(res) = rx.recv().await {
                match res {
                    Ok(tok) => generated_tokens.push(tok),
                    Err(e) => {
                        state.metrics.record_request_end(
                            false,
                            request_start.elapsed().as_secs_f64(),
                            generated_tokens.len() as u64,
                            prompt_tokens as u64,
                        );
                        return error_response(
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "internal_error",
                            format!("Scheduler execution failed: {}", e),
                        );
                    }
                }
            }
            let generated_text = pipeline.detokenize(&generated_tokens).unwrap_or_default();
            if let Err(message) = validate_structured_output(&generated_text, &response_format) {
                state.metrics.record_request_end(
                    false,
                    request_start.elapsed().as_secs_f64(),
                    generated_tokens.len() as u64,
                    prompt_tokens as u64,
                );
                return error_response(
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    "invalid_response_format",
                    message,
                );
            }
            let duration = request_start.elapsed().as_secs_f64();
            state.metrics.record_request_end(
                true,
                duration,
                generated_tokens.len() as u64,
                prompt_tokens as u64,
            );

            return Json(json!({
                "id": request_id,
                "object": "chat.completion",
                "created": unix_seconds(),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": generated_tokens.len(),
                    "total_tokens": prompt_tokens + generated_tokens.len()
                }
            }))
            .into_response();
        }

        // Streaming under IFB scheduler
        let req_id = request_id.clone();
        let model = model_id.clone();
        let state_clone = Arc::clone(&state);
        let generated_count = Arc::new(AtomicU64::new(0));
        let first_token_seen = Arc::new(AtomicBool::new(false));
        let last_token_time = Arc::new(std::sync::Mutex::new(None));
        let stream_failed = Arc::new(AtomicBool::new(false));
        let generated_count_for_stream = Arc::clone(&generated_count);
        let first_token_for_stream = Arc::clone(&first_token_seen);
        let last_token_for_stream = Arc::clone(&last_token_time);
        let stream_failed_for_stream = Arc::clone(&stream_failed);

        let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
        let accumulated_text_for_stream = Arc::clone(&accumulated_text);

        let pipeline_for_stream = Arc::clone(&pipeline);
        let sse_stream = UnboundedReceiverStream::new(rx).map(move |item| {
            let chunk = match item {
                Ok(tok) => {
                    record_stream_tokens(
                        &state_clone,
                        request_start,
                        &first_token_for_stream,
                        &last_token_for_stream,
                        &generated_count_for_stream,
                        1,
                    );
                    let text = pipeline_for_stream.detokenize(&[tok]).unwrap_or_default();
                    {
                        let mut acc = accumulated_text_for_stream.lock().unwrap();
                        acc.push_str(&text);
                    }
                    json!({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": unix_seconds(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": text },
                            "finish_reason": null
                        }]
                    })
                }
                Err(message) => {
                    stream_failed_for_stream.store(true, Ordering::Relaxed);
                    json!({
                        "error": {
                            "message": message,
                            "type": "internal_error"
                        }
                    })
                }
            };
            Ok::<Event, std::convert::Infallible>(Event::default().json_data(chunk).unwrap())
        });

        let stop_chunk = chat_stop_chunk(request_id.clone(), model_id.clone());
        let state_for_final = Arc::clone(&state);
        let req_id_final = request_id.clone();
        let generated_count_for_final = Arc::clone(&generated_count);
        let stream_failed_for_final = Arc::clone(&stream_failed);
        let stream_failed_for_validation = Arc::clone(&stream_failed);
        let accumulated_text_for_final = Arc::clone(&accumulated_text);
        let response_format_for_final = response_format.clone();

        let final_stream = sse_stream
            .chain(futures::stream::once(async move {
                let text = {
                    let acc = accumulated_text_for_final.lock().unwrap();
                    acc.clone()
                };
                if let Err(message) = validate_structured_output(&text, &response_format_for_final) {
                    stream_failed_for_validation.store(true, Ordering::Relaxed);
                    let err_chunk = json!({
                        "error": {
                            "message": format!("Stream structured output validation failed: {}", message),
                            "type": "invalid_response_format"
                        }
                    });
                    Ok::<Event, std::convert::Infallible>(Event::default().json_data(err_chunk).unwrap())
                } else {
                    Ok::<Event, std::convert::Infallible>(stop_chunk)
                }
            }))
            .chain(futures::stream::once(async move {
                {
                    let mut tokens = state_for_final.cancel_tokens.lock().unwrap();
                    tokens.remove(&req_id_final);
                }
                state_for_final.metrics.record_request_end(
                    !stream_failed_for_final.load(Ordering::Relaxed),
                    request_start.elapsed().as_secs_f64(),
                    generated_count_for_final.load(Ordering::Relaxed),
                    prompt_tokens as u64,
                );
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }));

        cancel_guard.disarm();
        return Sse::new(final_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    if !payload.stream {
        let pipeline_for_run = Arc::clone(&pipeline);
        let inference_start = std::time::Instant::now();
        let res = task::spawn_blocking(move || pipeline_for_run.run(input, &params)).await;
        state
            .metrics
            .record_inference_latency(inference_start.elapsed().as_secs_f64());

        let output = match res {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                state.metrics.record_request_end(
                    false,
                    request_start.elapsed().as_secs_f64(),
                    0,
                    prompt_tokens as u64,
                );
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Inference failed: {}", e),
                );
            }
            Err(e) => {
                state.metrics.record_request_end(
                    false,
                    request_start.elapsed().as_secs_f64(),
                    0,
                    prompt_tokens as u64,
                );
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Task join failed: {}", e),
                );
            }
        };

        let generated_text = output.text.unwrap_or_default();
        if let Err(message) = validate_structured_output(&generated_text, &response_format) {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_response_format",
                message,
            );
        }
        let completion_tokens = pipeline.tokenize(&generated_text).unwrap_or_default().len();

        let duration = request_start.elapsed().as_secs_f64();
        state.metrics.record_request_end(
            true,
            duration,
            completion_tokens as u64,
            prompt_tokens as u64,
        );

        return Json(json!({
            "id": request_id,
            "object": "chat.completion",
            "created": unix_seconds(),
            "model": model_id.clone(),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }))
        .into_response();
    }

    // Streaming
    let (tx, rx) = mpsc::channel::<std::result::Result<String, String>>(100);
    let pipeline_for_stream_run = Arc::clone(&pipeline);
    let cancel_token_clone = cancel_token.clone();
    task::spawn_blocking(move || {
        let tx_clone = tx.clone();
        let run_res = pipeline_for_stream_run.run_stream(
            input,
            &params,
            &mut |chunk: bloomai_engine::io::OutputChunk| {
                // Check for cancellation
                if cancel_token_clone.is_cancelled() {
                    return Err(anyhow!("request cancelled"));
                }
                if let bloomai_engine::io::OutputChunk::TextDelta(text) = chunk {
                    if tx_clone.blocking_send(Ok(text)).is_err() {
                        return Err(anyhow!("client disconnected"));
                    }
                }
                Ok(())
            },
        );
        if let Err(e) = run_res {
            let _ = tx.blocking_send(Err(e.to_string()));
        }
    });

    let req_id = request_id.clone();
    let model = model_id.clone();
    let state_for_stream = Arc::clone(&state);
    let generated_count = Arc::new(AtomicU64::new(0));
    let first_token_seen = Arc::new(AtomicBool::new(false));
    let last_token_time = Arc::new(std::sync::Mutex::new(None));
    let stream_failed = Arc::new(AtomicBool::new(false));
    let generated_count_for_stream = Arc::clone(&generated_count);
    let first_token_for_stream = Arc::clone(&first_token_seen);
    let last_token_for_stream = Arc::clone(&last_token_time);
    let stream_failed_for_stream = Arc::clone(&stream_failed);

    let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
    let accumulated_text_for_stream = Arc::clone(&accumulated_text);

    let pipeline_for_stream = Arc::clone(&pipeline);
    let sse_stream = ReceiverStream::new(rx).map(move |item| {
        let chunk = match item {
            Ok(token) => {
                {
                    let mut acc = accumulated_text_for_stream.lock().unwrap();
                    acc.push_str(&token);
                }
                let token_count = estimate_delta_tokens(&pipeline_for_stream, &token);
                record_stream_tokens(
                    &state_for_stream,
                    request_start,
                    &first_token_for_stream,
                    &last_token_for_stream,
                    &generated_count_for_stream,
                    token_count,
                );
                json!({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": unix_seconds(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": token },
                        "finish_reason": null
                    }]
                })
            }
            Err(message) => {
                stream_failed_for_stream.store(true, Ordering::Relaxed);
                json!({
                    "error": {
                        "message": message,
                        "type": "internal_error"
                    }
                })
            }
        };
        Ok::<Event, std::convert::Infallible>(Event::default().json_data(chunk).unwrap())
    });

    let stop_chunk = chat_stop_chunk(request_id.clone(), model_id.clone());
    let cancel_for_final = cancel_token.clone();
    let state_for_final = Arc::clone(&state);
    let req_id_final = request_id.clone();
    let generated_count_for_final = Arc::clone(&generated_count);
    let stream_failed_for_final = Arc::clone(&stream_failed);
    let stream_failed_for_validation = Arc::clone(&stream_failed);
    let accumulated_text_for_final = Arc::clone(&accumulated_text);
    let response_format_for_final = response_format.clone();

    let final_stream = sse_stream
        .chain(futures::stream::once(async move {
            let text = {
                let acc = accumulated_text_for_final.lock().unwrap();
                acc.clone()
            };
            if let Err(message) = validate_structured_output(&text, &response_format_for_final) {
                stream_failed_for_validation.store(true, Ordering::Relaxed);
                let err_chunk = json!({
                    "error": {
                        "message": format!("Stream structured output validation failed: {}", message),
                        "type": "invalid_response_format"
                    }
                });
                Ok::<Event, std::convert::Infallible>(Event::default().json_data(err_chunk).unwrap())
            } else {
                Ok::<Event, std::convert::Infallible>(stop_chunk)
            }
        }))
        .chain(futures::stream::once(async move {
            // Clean up cancel token on stream completion
            {
                let mut tokens = state_for_final.cancel_tokens.lock().unwrap();
                tokens.remove(&req_id_final);
            }
            state_for_final.metrics.record_request_end(
                !stream_failed_for_final.load(Ordering::Relaxed)
                    && !cancel_for_final.is_cancelled(),
                request_start.elapsed().as_secs_f64(),
                generated_count_for_final.load(Ordering::Relaxed),
                prompt_tokens as u64,
            );
            Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
        }));

    cancel_guard.disarm();
    Sse::new(final_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ─── /v1/completions ────────────────────────────────────────────────────────

pub(crate) async fn handle_completions(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<CompletionRequest>,
) -> impl IntoResponse {
    if !state.ready.load(Ordering::Relaxed) {
        return error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "model_loading",
            format!(
                "Model is still loading (progress: {}%)",
                state.load_progress.load(Ordering::Relaxed)
            ),
        );
    }

    let pipeline = match state.get_pipeline().await {
        Ok(p) => p,
        Err(e) => {
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model_loading",
                e.to_string(),
            );
        }
    };

    let model_id = {
        let guard = state.model_id.read().await;
        guard.clone()
    };

    let scheduler_opt = {
        let guard = state.scheduler.read().await;
        guard.clone()
    };

    let _permit = match state.semaphore.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = std::time::Instant::now();

    let response_format = match response_format_mode(payload.response_format.as_ref()) {
        Ok(mode) => mode,
        Err(message) => {
            state
                .metrics
                .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };

    let core_response_format = match &response_format {
        ResponseFormatMode::Text => bloomai_core::ResponseFormat::Text,
        ResponseFormatMode::JsonObject => bloomai_core::ResponseFormat::JsonObject,
        ResponseFormatMode::JsonSchema(schema) => {
            bloomai_core::ResponseFormat::JsonSchema(schema.clone())
        }
    };

    let params = GenerationParams {
        max_tokens: payload.max_tokens.unwrap_or(128),
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        seed: payload.seed,
        response_format: Some(core_response_format),
    };

    let prompt = match &payload.prompt {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            if let Some(first) = arr.first().and_then(|v| v.as_str()) {
                first.to_string()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    };
    let prompt = apply_response_format_instruction(prompt, &response_format);

    let prompt_tokens_vec = pipeline.tokenize(&prompt).unwrap_or_default();
    let prompt_tokens = prompt_tokens_vec.len();
    let input = ModelInput::Text { prompt };
    let request_id = next_request_id(&state, "cmpl");
    let mut cancel_guard = CancelTokenGuard::register(&state, request_id.clone());
    let cancel_token = cancel_guard.token();

    if state.enable_ifb {
        let scheduler = scheduler_opt.unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel::<Result<u32, String>>();
        scheduler
            .token_senders
            .lock()
            .unwrap()
            .insert(request_id.clone(), tx);

        let req = Request {
            id: request_id.clone(),
            model_id: model_id.clone(),
            prompt_tokens: prompt_tokens_vec,
            generated_tokens: Vec::new(),
            params: params.clone(),
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        if let Err(e) = scheduler.submit(req) {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("Failed to submit to scheduler: {}", e),
            );
        }

        if !payload.stream {
            let mut generated_tokens = Vec::new();
            while let Some(res) = rx.recv().await {
                match res {
                    Ok(tok) => generated_tokens.push(tok),
                    Err(e) => {
                        state.metrics.record_request_end(
                            false,
                            request_start.elapsed().as_secs_f64(),
                            generated_tokens.len() as u64,
                            prompt_tokens as u64,
                        );
                        return error_response(
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "internal_error",
                            format!("Scheduler execution failed: {}", e),
                        );
                    }
                }
            }
            let generated_text = pipeline.detokenize(&generated_tokens).unwrap_or_default();
            if let Err(message) = validate_structured_output(&generated_text, &response_format) {
                state.metrics.record_request_end(
                    false,
                    request_start.elapsed().as_secs_f64(),
                    generated_tokens.len() as u64,
                    prompt_tokens as u64,
                );
                return error_response(
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    "invalid_response_format",
                    message,
                );
            }
            let duration = request_start.elapsed().as_secs_f64();
            state.metrics.record_request_end(
                true,
                duration,
                generated_tokens.len() as u64,
                prompt_tokens as u64,
            );

            return Json(json!({
                "id": request_id,
                "object": "text_completion",
                "created": unix_seconds(),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "text": generated_text,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": generated_tokens.len(),
                    "total_tokens": prompt_tokens + generated_tokens.len()
                }
            }))
            .into_response();
        }

        // Streaming under IFB scheduler
        let req_id = request_id.clone();
        let model = model_id.clone();
        let state_clone = Arc::clone(&state);
        let generated_count = Arc::new(AtomicU64::new(0));
        let first_token_seen = Arc::new(AtomicBool::new(false));
        let last_token_time = Arc::new(std::sync::Mutex::new(None));
        let stream_failed = Arc::new(AtomicBool::new(false));
        let generated_count_for_stream = Arc::clone(&generated_count);
        let first_token_for_stream = Arc::clone(&first_token_seen);
        let last_token_for_stream = Arc::clone(&last_token_time);
        let stream_failed_for_stream = Arc::clone(&stream_failed);

        let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
        let accumulated_text_for_stream = Arc::clone(&accumulated_text);

        let pipeline_for_stream = Arc::clone(&pipeline);
        let sse_stream = UnboundedReceiverStream::new(rx).map(move |item| {
            let chunk = match item {
                Ok(tok) => {
                    record_stream_tokens(
                        &state_clone,
                        request_start,
                        &first_token_for_stream,
                        &last_token_for_stream,
                        &generated_count_for_stream,
                        1,
                    );
                    let text = pipeline_for_stream.detokenize(&[tok]).unwrap_or_default();
                    {
                        let mut acc = accumulated_text_for_stream.lock().unwrap();
                        acc.push_str(&text);
                    }
                    json!({
                        "id": req_id,
                        "object": "text_completion.chunk",
                        "created": unix_seconds(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "text": text,
                            "finish_reason": null
                        }]
                    })
                }
                Err(message) => {
                    stream_failed_for_stream.store(true, Ordering::Relaxed);
                    json!({
                        "error": {
                            "message": message,
                            "type": "internal_error"
                        }
                    })
                }
            };
            Ok::<Event, std::convert::Infallible>(Event::default().json_data(chunk).unwrap())
        });

        let stop_chunk = Event::default()
            .json_data(json!({
                "id": request_id,
                "object": "text_completion.chunk",
                "created": unix_seconds(),
                "model": model_id.clone(),
                "choices": [{
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop"
                }]
            }))
            .unwrap();

        let state_for_final = Arc::clone(&state);
        let req_id_final = request_id.clone();
        let generated_count_for_final = Arc::clone(&generated_count);
        let stream_failed_for_final = Arc::clone(&stream_failed);
        let stream_failed_for_validation = Arc::clone(&stream_failed);
        let accumulated_text_for_final = Arc::clone(&accumulated_text);
        let response_format_for_final = response_format.clone();

        let final_stream = sse_stream
            .chain(futures::stream::once(async move {
                let text = {
                    let acc = accumulated_text_for_final.lock().unwrap();
                    acc.clone()
                };
                if let Err(message) = validate_structured_output(&text, &response_format_for_final) {
                    stream_failed_for_validation.store(true, Ordering::Relaxed);
                    let err_chunk = json!({
                        "error": {
                            "message": format!("Stream structured output validation failed: {}", message),
                            "type": "invalid_response_format"
                        }
                    });
                    Ok::<Event, std::convert::Infallible>(Event::default().json_data(err_chunk).unwrap())
                } else {
                    Ok::<Event, std::convert::Infallible>(stop_chunk)
                }
            }))
            .chain(futures::stream::once(async move {
                {
                    let mut tokens = state_for_final.cancel_tokens.lock().unwrap();
                    tokens.remove(&req_id_final);
                }
                state_for_final.metrics.record_request_end(
                    !stream_failed_for_final.load(Ordering::Relaxed),
                    request_start.elapsed().as_secs_f64(),
                    generated_count_for_final.load(Ordering::Relaxed),
                    prompt_tokens as u64,
                );
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }));

        cancel_guard.disarm();
        return Sse::new(final_stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    if !payload.stream {
        let pipeline_for_run = Arc::clone(&pipeline);
        let inference_start = std::time::Instant::now();
        let res = task::spawn_blocking(move || pipeline_for_run.run(input, &params)).await;
        state
            .metrics
            .record_inference_latency(inference_start.elapsed().as_secs_f64());

        let output = match res {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                state.metrics.record_request_end(
                    false,
                    request_start.elapsed().as_secs_f64(),
                    0,
                    prompt_tokens as u64,
                );
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Inference failed: {}", e),
                );
            }
            Err(e) => {
                state.metrics.record_request_end(
                    false,
                    request_start.elapsed().as_secs_f64(),
                    0,
                    prompt_tokens as u64,
                );
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("Task join failed: {}", e),
                );
            }
        };

        let generated_text = output.text.unwrap_or_default();
        if let Err(message) = validate_structured_output(&generated_text, &response_format) {
            state.metrics.record_request_end(
                false,
                request_start.elapsed().as_secs_f64(),
                0,
                prompt_tokens as u64,
            );
            return error_response(
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_response_format",
                message,
            );
        }
        let completion_tokens = pipeline.tokenize(&generated_text).unwrap_or_default().len();

        let duration = request_start.elapsed().as_secs_f64();
        state.metrics.record_request_end(
            true,
            duration,
            completion_tokens as u64,
            prompt_tokens as u64,
        );

        return Json(json!({
            "id": request_id,
            "object": "text_completion",
            "created": unix_seconds(),
            "model": model_id.clone(),
            "choices": [{
                "index": 0,
                "text": generated_text,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }))
        .into_response();
    }

    // Streaming
    let (tx, rx) = mpsc::channel::<std::result::Result<String, String>>(100);
    let pipeline_for_stream_run = Arc::clone(&pipeline);
    let cancel_token_clone = cancel_token.clone();
    task::spawn_blocking(move || {
        let tx_clone = tx.clone();
        let run_res = pipeline_for_stream_run.run_stream(
            input,
            &params,
            &mut |chunk: bloomai_engine::io::OutputChunk| {
                if cancel_token_clone.is_cancelled() {
                    return Err(anyhow!("request cancelled"));
                }
                if let bloomai_engine::io::OutputChunk::TextDelta(text) = chunk {
                    if tx_clone.blocking_send(Ok(text)).is_err() {
                        return Err(anyhow!("client disconnected"));
                    }
                }
                Ok(())
            },
        );
        if let Err(e) = run_res {
            let _ = tx.blocking_send(Err(e.to_string()));
        }
    });

    let req_id = request_id.clone();
    let model = model_id.clone();
    let state_for_stream = Arc::clone(&state);
    let generated_count = Arc::new(AtomicU64::new(0));
    let first_token_seen = Arc::new(AtomicBool::new(false));
    let last_token_time = Arc::new(std::sync::Mutex::new(None));
    let stream_failed = Arc::new(AtomicBool::new(false));
    let generated_count_for_stream = Arc::clone(&generated_count);
    let first_token_for_stream = Arc::clone(&first_token_seen);
    let last_token_for_stream = Arc::clone(&last_token_time);
    let stream_failed_for_stream = Arc::clone(&stream_failed);

    let accumulated_text = Arc::new(std::sync::Mutex::new(String::new()));
    let accumulated_text_for_stream = Arc::clone(&accumulated_text);

    let pipeline_for_stream = Arc::clone(&pipeline);
    let sse_stream = ReceiverStream::new(rx).map(move |item| {
        let chunk = match item {
            Ok(token) => {
                {
                    let mut acc = accumulated_text_for_stream.lock().unwrap();
                    acc.push_str(&token);
                }
                let token_count = estimate_delta_tokens(&pipeline_for_stream, &token);
                record_stream_tokens(
                    &state_for_stream,
                    request_start,
                    &first_token_for_stream,
                    &last_token_for_stream,
                    &generated_count_for_stream,
                    token_count,
                );
                json!({
                    "id": req_id,
                    "object": "text_completion.chunk",
                    "created": unix_seconds(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "text": token,
                        "finish_reason": null
                    }]
                })
            }
            Err(message) => {
                stream_failed_for_stream.store(true, Ordering::Relaxed);
                json!({
                    "error": {
                        "message": message,
                        "type": "internal_error"
                    }
                })
            }
        };
        Ok::<Event, std::convert::Infallible>(Event::default().json_data(chunk).unwrap())
    });

    let stop_chunk = Event::default()
        .json_data(json!({
            "id": request_id.clone(),
            "object": "text_completion.chunk",
            "created": unix_seconds(),
            "model": model_id.clone(),
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": "stop"
            }]
        }))
        .unwrap();

    let state_for_final = Arc::clone(&state);
    let req_id_final = request_id.clone();
    let generated_count_for_final = Arc::clone(&generated_count);
    let stream_failed_for_final = Arc::clone(&stream_failed);
    let stream_failed_for_validation = Arc::clone(&stream_failed);
    let accumulated_text_for_final = Arc::clone(&accumulated_text);
    let response_format_for_final = response_format.clone();

    let final_stream = sse_stream
        .chain(futures::stream::once(async move {
            let text = {
                let acc = accumulated_text_for_final.lock().unwrap();
                acc.clone()
            };
            if let Err(message) = validate_structured_output(&text, &response_format_for_final) {
                stream_failed_for_validation.store(true, Ordering::Relaxed);
                let err_chunk = json!({
                    "error": {
                        "message": format!("Stream structured output validation failed: {}", message),
                        "type": "invalid_response_format"
                    }
                });
                Ok::<Event, std::convert::Infallible>(Event::default().json_data(err_chunk).unwrap())
            } else {
                Ok::<Event, std::convert::Infallible>(stop_chunk)
            }
        }))
        .chain(futures::stream::once(async move {
            {
                let mut tokens = state_for_final.cancel_tokens.lock().unwrap();
                tokens.remove(&req_id_final);
            }
            state_for_final.metrics.record_request_end(
                !stream_failed_for_final.load(Ordering::Relaxed),
                request_start.elapsed().as_secs_f64(),
                generated_count_for_final.load(Ordering::Relaxed),
                prompt_tokens as u64,
            );
            Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
        }));

    cancel_guard.disarm();
    Sse::new(final_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ─── /v1/embeddings ────────────────────────────────────────────────────────

pub(crate) async fn handle_embeddings(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    if !state.ready.load(Ordering::Relaxed) {
        return error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "model_loading",
            format!(
                "Model is still loading (progress: {}%)",
                state.load_progress.load(Ordering::Relaxed)
            ),
        );
    }

    let pipeline = match state.get_pipeline().await {
        Ok(p) => p,
        Err(e) => {
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model_loading",
                e.to_string(),
            );
        }
    };

    let model_id = {
        let guard = state.model_id.read().await;
        guard.clone()
    };

    let _permit = match state.semaphore.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = Instant::now();

    if let Some(format) = payload.encoding_format.as_deref() {
        if format != "float" {
            state
                .metrics
                .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Only encoding_format='float' is currently supported.",
            );
        }
    }

    if !model_supports_embeddings(&pipeline) {
        state
            .metrics
            .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
        return unsupported_embeddings_response(&model_id);
    }

    let inputs = match normalize_embedding_input(&payload.input) {
        Ok(inputs) => inputs,
        Err(message) => {
            state
                .metrics
                .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            );
        }
    };

    let prompt_tokens = count_text_tokens(&pipeline, &inputs);
    let mut data = Vec::with_capacity(inputs.len());
    for (index, text) in inputs.into_iter().enumerate() {
        let pipeline_clone = Arc::clone(&pipeline);
        let embedding =
            match task::spawn_blocking(move || collect_embedding(pipeline_clone, text)).await {
                Ok(Ok(embedding)) => embedding,
                Ok(Err(err)) => {
                    state.metrics.record_request_end(
                        false,
                        request_start.elapsed().as_secs_f64(),
                        0,
                        prompt_tokens as u64,
                    );
                    return error_response(
                        axum::http::StatusCode::NOT_IMPLEMENTED,
                        "unsupported_operation",
                        format!("Embedding inference failed: {}", err),
                    );
                }
                Err(err) => {
                    state.metrics.record_request_end(
                        false,
                        request_start.elapsed().as_secs_f64(),
                        0,
                        prompt_tokens as u64,
                    );
                    return error_response(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "internal_error",
                        format!("Embedding task join failed: {}", err),
                    );
                }
            };
        data.push(json!({
            "object": "embedding",
            "embedding": embedding,
            "index": index
        }));
    }

    state.metrics.record_request_end(
        true,
        request_start.elapsed().as_secs_f64(),
        0,
        prompt_tokens as u64,
    );

    Json(json!({
        "object": "list",
        "data": data,
        "model": payload.model.unwrap_or(model_id),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    }))
    .into_response()
}

// ─── /v1/rerank ────────────────────────────────────────────────────────────

pub(crate) async fn handle_rerank(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<RerankRequest>,
) -> impl IntoResponse {
    if !state.ready.load(Ordering::Relaxed) {
        return error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "model_loading",
            format!(
                "Model is still loading (progress: {}%)",
                state.load_progress.load(Ordering::Relaxed)
            ),
        );
    }

    let pipeline = match state.get_pipeline().await {
        Ok(p) => p,
        Err(e) => {
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model_loading",
                e.to_string(),
            );
        }
    };

    let model_id = {
        let guard = state.model_id.read().await;
        guard.clone()
    };

    let _permit = match state.semaphore.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = Instant::now();

    if payload.query.trim().is_empty() || payload.documents.is_empty() {
        state
            .metrics
            .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
        return error_response(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Rerank requires a non-empty query and at least one document.",
        );
    }

    if !model_supports_embeddings(&pipeline) {
        state
            .metrics
            .record_request_end(false, request_start.elapsed().as_secs_f64(), 0, 0);
        return unsupported_embeddings_response(&model_id);
    }

    let mut texts = Vec::with_capacity(payload.documents.len() + 1);
    texts.push(payload.query.clone());
    texts.extend(payload.documents.iter().cloned());
    let prompt_tokens = count_text_tokens(&pipeline, &texts);

    let query_embedding =
        match run_embedding_task(Arc::clone(&pipeline), payload.query.clone()).await {
            Ok(embedding) => embedding,
            Err(message) => {
                state.metrics.record_request_end(
                    false,
                    request_start.elapsed().as_secs_f64(),
                    0,
                    prompt_tokens as u64,
                );
                return error_response(
                    axum::http::StatusCode::NOT_IMPLEMENTED,
                    "unsupported_operation",
                    message,
                );
            }
        };

    let mut results = Vec::with_capacity(payload.documents.len());
    for (index, document) in payload.documents.iter().cloned().enumerate() {
        let document_embedding =
            match run_embedding_task(Arc::clone(&pipeline), document.clone()).await {
                Ok(embedding) => embedding,
                Err(message) => {
                    state.metrics.record_request_end(
                        false,
                        request_start.elapsed().as_secs_f64(),
                        0,
                        prompt_tokens as u64,
                    );
                    return error_response(
                        axum::http::StatusCode::NOT_IMPLEMENTED,
                        "unsupported_operation",
                        message,
                    );
                }
            };
        let score = cosine_similarity(&query_embedding, &document_embedding);
        let mut item = json!({
            "index": index,
            "relevance_score": score
        });
        if payload.return_documents.unwrap_or(false) {
            item["document"] = json!({ "text": document });
        }
        results.push(item);
    }

    results.sort_by(|a, b| {
        let a = a
            .get("relevance_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let b = b
            .get("relevance_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(top_n) = payload.top_n {
        results.truncate(top_n);
    }

    state.metrics.record_request_end(
        true,
        request_start.elapsed().as_secs_f64(),
        0,
        prompt_tokens as u64,
    );

    Json(json!({
        "id": format!("rerank-{}", unix_seconds()),
        "object": "rerank",
        "model": payload.model.unwrap_or(model_id),
        "results": results,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    }))
    .into_response()
}

// ─── /v1/multimodal/stream ──────────────────────────────────────────────────

pub(crate) async fn handle_multimodal_stream(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<InferenceRequest>,
) -> impl IntoResponse {
    if !state.ready.load(Ordering::Relaxed) {
        return error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "model_loading",
            format!(
                "Model is still loading (progress: {}%)",
                state.load_progress.load(Ordering::Relaxed)
            ),
        );
    }

    let pipeline = match state.get_pipeline().await {
        Ok(p) => p,
        Err(e) => {
            return error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model_loading",
                e.to_string(),
            );
        }
    };

    let model_id = {
        let guard = state.model_id.read().await;
        guard.clone()
    };

    let _permit = match state.semaphore.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "too_many_requests",
                "Too many concurrent requests. Server is busy.",
            );
        }
    };

    state.metrics.record_request_start();
    let request_start = std::time::Instant::now();
    let request_id = next_request_id(&state, "mms");
    let mut cancel_guard = CancelTokenGuard::register(&state, request_id.clone());
    let cancel_token = cancel_guard.token();

    let (tx, rx) = mpsc::channel::<std::result::Result<OutputChunk, String>>(100);
    let pipeline_clone = Arc::clone(&pipeline);
    let cancel_token_clone = cancel_token.clone();

    task::spawn_blocking(move || {
        let tx_clone = tx.clone();
        let run_res = pipeline_clone.run_request(payload, &mut |chunk: OutputChunk| {
            if cancel_token_clone.is_cancelled() {
                return Err(anyhow!("request cancelled"));
            }
            if tx_clone.blocking_send(Ok(chunk)).is_err() {
                return Err(anyhow!("client disconnected"));
            }
            Ok(())
        });
        if let Err(e) = run_res {
            let _ = tx.blocking_send(Err(e.to_string()));
        }
    });

    let stream_failed = Arc::new(AtomicBool::new(false));
    let stream_failed_for_stream = Arc::clone(&stream_failed);
    let req_id_for_stream = request_id.clone();
    let model_id_clone = model_id.clone();

    let sse_stream = ReceiverStream::new(rx).map(move |item| {
        let chunk = match item {
            Ok(out_chunk) => {
                json!({
                    "id": req_id_for_stream.clone(),
                    "object": "multimodal.chunk",
                    "created": unix_seconds(),
                    "model": model_id_clone.clone(),
                    "chunk": out_chunk,
                })
            }
            Err(message) => {
                stream_failed_for_stream.store(true, Ordering::Relaxed);
                json!({
                    "error": {
                        "message": message,
                        "type": "internal_error"
                    }
                })
            }
        };
        Ok::<Event, std::convert::Infallible>(Event::default().json_data(chunk).unwrap())
    });

    let state_for_final = Arc::clone(&state);
    let req_id_final = request_id.clone();
    let stream_failed_for_final = Arc::clone(&stream_failed);
    let final_stream = sse_stream.chain(futures::stream::once(async move {
        {
            let mut tokens = state_for_final.cancel_tokens.lock().unwrap();
            tokens.remove(&req_id_final);
        }
        state_for_final.metrics.record_request_end(
            !stream_failed_for_final.load(Ordering::Relaxed),
            request_start.elapsed().as_secs_f64(),
            0,
            0,
        );
        Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
    }));

    cancel_guard.disarm();
    Sse::new(final_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorldCacheConfig {
    pub max_bytes: Option<usize>,
    pub max_entries: Option<usize>,
    pub default_ttl_ms: Option<u64>,
    pub auto_compress: Option<bool>,
    pub compress_after_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorldStepRequest {
    pub observations: Vec<bloomai_core::WorldObservation>,
    #[serde(default = "default_horizon")]
    pub horizon: u32,
    pub state_schema: Option<bloomai_engine::WorldStateSchema>,
    pub action_schema: Option<bloomai_engine::ActionSchema>,
    pub cache_config: Option<WorldCacheConfig>,
    pub thermal_state: Option<bloomai_core::ThermalState>,
    pub power_state: Option<bloomai_core::PowerState>,
    #[serde(default)]
    pub stream: bool,
}

pub(crate) fn default_horizon() -> u32 {
    1
}

pub(crate) async fn handle_world_step(
    State(_state): State<Arc<ServerState>>,
    Json(payload): Json<WorldStepRequest>,
) -> impl IntoResponse {
    let wm = Box::new(bloomai_engine::MockWorldModel::new("mock-world-model"));
    let policy = Box::new(bloomai_engine::MockPolicyEngine::new(
        "mock-policy",
        "robot_velocity",
        2,
    ));

    let mut cache_config = bloomai_core::StateCacheConfig::default();
    if let Some(ref c) = payload.cache_config {
        if let Some(mb) = c.max_bytes {
            cache_config.max_bytes = mb;
        }
        if let Some(me) = c.max_entries {
            cache_config.max_entries = me;
        }
        if let Some(ttl) = c.default_ttl_ms {
            cache_config.default_ttl_ms = ttl;
        }
        if let Some(ac) = c.auto_compress {
            cache_config.auto_compress = ac;
        }
        if let Some(cam) = c.compress_after_ms {
            cache_config.compress_after_ms = cam;
        }
    }

    let mut world_loop = bloomai_engine::WorldModelLoop::new(wm, policy, cache_config);
    world_loop.set_schemas(payload.state_schema, payload.action_schema);

    if let (Some(t), Some(p)) = (payload.thermal_state, payload.power_state) {
        world_loop.set_environment(t, p);
    }

    let chunks = match world_loop.step(payload.observations, payload.horizon) {
        Ok(c) => c,
        Err(e) => {
            return error_response(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_observation_or_action",
                e.to_string(),
            );
        }
    };

    if payload.stream {
        // SSE stream response
        let (tx, rx) = mpsc::channel::<
            std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
        >(10);

        tokio::spawn(async move {
            for chunk in chunks {
                let event = axum::response::sse::Event::default()
                    .json_data(&chunk)
                    .unwrap();
                if tx.send(Ok(event)).await.is_err() {
                    break;
                }
            }
        });

        Sse::new(ReceiverStream::new(rx))
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    } else {
        Json(chunks).into_response()
    }
}

// ─── /v1/cancel/:request_id ──────────────────────────────────────────────

pub(crate) async fn handle_cancel(
    State(state): State<Arc<ServerState>>,
    axum::extract::Path(request_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut cancelled = false;

    // Try IFB scheduler cancel first
    let scheduler_guard = state.scheduler.read().await;
    if let Some(ref scheduler) = *scheduler_guard {
        if scheduler.cancel_request(&request_id) {
            cancelled = true;
        }
    }

    // Try cancel token
    {
        let tokens = state.cancel_tokens.lock().unwrap();
        if let Some(token) = tokens.get(&request_id) {
            token.cancel();
            cancelled = true;
        }
    }

    // Clean up cancel token
    {
        let mut tokens = state.cancel_tokens.lock().unwrap();
        tokens.remove(&request_id);
    }

    if cancelled {
        Json(json!({
            "id": request_id,
            "object": "cancellation",
            "cancelled": true
        }))
    } else {
        Json(json!({
            "id": request_id,
            "object": "cancellation",
            "cancelled": false,
            "error": "request not found or already completed"
        }))
    }
}

// ─── /v1/backends ───────────────────────────────────────────────────────────

pub(crate) async fn handle_backends(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let model_id = {
        let guard = state.model_id.read().await;
        guard.clone()
    };
    let registry = {
        let mut reg = EngineRegistry::default();
        reg.register("candle", Box::new(CandleEngine));
        reg.register("openvino", Box::new(OpenVINOEngine));
        reg.register("funasr", Box::new(FunASREngine));
        reg.register("qwen3_vl", Box::new(Qwen3VLEngine));
        reg.register("intel-npu", Box::new(IntelNpuEngine));
        reg.register("npu-tts", Box::new(NpuTtsEngine));
        reg.register("onnxruntime", Box::new(OnnxRuntimeEngine));
        reg.register("coreml", Box::new(CoreMlEngine));
        reg.register("mlx", Box::new(MlxEngine));
        reg.register("vulkan", Box::new(VulkanEngine));
        reg.register("llamacpp", Box::new(LlamaCppEngine));
        reg
    };

    let backends: Vec<serde_json::Value> = registry
        .iter()
        .map(|(name, engine)| {
            let cap = engine.capability();
            json!({
                "name": name,
                "maturity": cap.maturity.to_string(),
                "supported_families": cap.supported_families.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>(),
                "supported_formats": cap.supported_formats.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>(),
                "supported_devices": cap.supported_devices.iter().map(|d| format!("{:?}", d)).collect::<Vec<_>>(),
                "supports_streaming": cap.supports_streaming,
                "supports_quantized_models": cap.supports_quantized_models,
                "supports_embeddings": cap.supports_embeddings,
                "supports_rerank": cap.supports_rerank,
                "supports_structured_output": cap.supports_structured_output,
                "max_context_tokens": cap.max_context_tokens,
                "diagnostic_tips": cap.diagnostic_tips,
                "construction_guide": cap.construction_guide,
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": backends,
        "active_model": model_id,
    }))
}
