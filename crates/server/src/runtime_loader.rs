//! Model loading, backend admission, and runtime publication lifecycle.

use super::*;

pub(crate) fn engine_registry() -> EngineRegistry {
    let mut registry = EngineRegistry::default();
    registry.register("candle", Box::new(CandleEngine));
    registry.register("openvino", Box::new(OpenVINOEngine));
    registry.register("funasr", Box::new(FunASREngine));
    registry.register("qwen3_vl", Box::new(Qwen3VLEngine));
    registry.register("intel-npu", Box::new(IntelNpuEngine));
    registry.register("npu-tts", Box::new(NpuTtsEngine));
    registry.register("onnxruntime", Box::new(OnnxRuntimeEngine));
    registry.register("coreml", Box::new(CoreMlEngine));
    registry.register("mlx", Box::new(MlxEngine));
    registry.register("vulkan", Box::new(VulkanEngine));
    registry.register("llamacpp", Box::new(LlamaCppEngine));
    registry.register("longcat", Box::new(LongCatImageEditEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("wan", Box::new(WanEngine));
    registry
}

pub(super) async fn model_loader_loop(
    state: Arc<ServerState>,
    args: Args,
    device_kind: DeviceKind,
    mut requests: mpsc::Receiver<ModelLoadRequest>,
) {
    while let Some(request) = requests.recv().await {
        let request_label = request
            .catalog_id
            .clone()
            .unwrap_or_else(|| model_path_label(&request.path));
        tracing::info!(model = %request_label, "Starting model load");
        state.load_progress.store(1, Ordering::Release);
        *state.load_error.write().await = None;

        // Keep the managed model source immutable from the final manifest
        // read through publication. Pull/delete/upgrade use the same storage
        // fence, so the engine cannot load one artifact generation while the
        // memory permit and catalog identity describe another.
        let _storage_guard = state.model_storage.serial().await;

        match prepare_loaded_runtime(
            Arc::clone(&state),
            &args,
            device_kind,
            request.path,
            request.catalog_id,
            request.memory_permit,
        )
        .await
        {
            Ok(runtime) => {
                // Runtime preparation is isolated from publication so an
                // existing resident remains available during a long load.
                // Publication under the pool write lock is the admission
                // boundary. Existing request leases keep retired generations
                // alive without blocking unrelated model traffic.
                let runtime = Arc::new(runtime);
                let model_id = runtime.model_id.clone();
                match state.publish_default_runtime(Arc::clone(&runtime)).await {
                    Ok(retired) => {
                        drop(retired);
                        state.load_progress.store(100, Ordering::Release);
                        state
                            .finish_model_load(
                                request.sequence,
                                ModelLoadOutcome::Ready { runtime },
                            )
                            .await;
                        tracing::info!(model = %model_id, "Model load completed");
                    }
                    Err(message) => {
                        tracing::error!(model = %model_id, error = %message, "Model publication failed");
                        state.discard_unpublished_runtime(runtime);
                        *state.load_error.write().await = Some(message.clone());
                        state.load_progress.store(0, Ordering::Release);
                        let has_fallback = !state.runtime_pool.read().await.is_empty();
                        state.ready.store(has_fallback, Ordering::Release);
                        state
                            .finish_model_load(
                                request.sequence,
                                ModelLoadOutcome::Failed { message },
                            )
                            .await;
                    }
                }
            }
            Err(error) => {
                let message = error.to_string();
                let requested_model = state
                    .requested_model
                    .read()
                    .await
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                tracing::error!(path = %requested_model, error = %message, "Model load failed");
                *state.load_error.write().await = Some(message.clone());
                state.load_progress.store(0, Ordering::Release);
                let has_fallback = !state.runtime_pool.read().await.is_empty();
                state.ready.store(has_fallback, Ordering::Release);
                state
                    .finish_model_load(
                        request.sequence,
                        ModelLoadOutcome::Failed {
                            message: message.clone(),
                        },
                    )
                    .await;
            }
        }
    }
    tracing::error!("Model loader stopped because its request channel was closed");
}

pub(super) fn model_path_label(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("external model")
        .to_string()
}

pub(super) fn validate_loaded_runtime_model_id(model_id: &str) -> Result<()> {
    if model_id == "default" {
        return Err(anyhow!(
            "loaded model ID 'default' is reserved for the current default runtime selector"
        ));
    }
    validate_model_selector(model_id).map_err(|_| {
        anyhow!(
            "loaded model ID must contain 1 to 256 characters without surrounding whitespace or control characters"
        )
    })
}

fn signed_model_version_from_catalog(
    catalog: &ModelCatalog,
    model_path: &Path,
    catalog_id: Option<&str>,
) -> Option<SignedModelVersion> {
    let entry_matches_path = |entry: &model_manager::ModelCatalogEntry| {
        Path::new(&catalog.root)
            .join(&entry.id)
            .canonicalize()
            .is_ok_and(|candidate| candidate == model_path)
    };
    let entry = catalog_id
        .and_then(|catalog_id| catalog.models.iter().find(|entry| entry.id == catalog_id))
        .filter(|entry| entry_matches_path(entry))
        .or_else(|| {
            catalog
                .models
                .iter()
                .find(|entry| entry_matches_path(entry))
        })?;
    let provenance = entry.provenance.as_ref()?;
    provenance
        .model_index_id
        .as_ref()
        .map(|model_index_id| SignedModelVersion {
            model_index_id: model_index_id.clone(),
            sha256: provenance.sha256.clone(),
        })
}

async fn prepare_loaded_runtime(
    state: Arc<ServerState>,
    args: &Args,
    device_kind: DeviceKind,
    model_path: PathBuf,
    catalog_id: Option<String>,
    mut memory_permit: Option<RuntimeMemoryPermit>,
) -> Result<LoadedRuntime> {
    let model_path = tokio::fs::canonicalize(&model_path)
        .await
        .map_err(|error| anyhow!("failed to resolve the model source: {error}"))?;
    let signed_model_version = if state.model_index.is_some() {
        let (catalog, _) = state
            .fresh_model_catalog_snapshot()
            .await
            .map_err(|error| {
                anyhow!("failed to inspect signed model provenance before loading: {error}")
            })?;
        signed_model_version_from_catalog(&catalog, &model_path, catalog_id.as_deref())
    } else {
        None
    };
    if signed_model_version.as_ref().is_some_and(|version| {
        state
            .model_index
            .as_ref()
            .is_some_and(|index| index.is_revoked(&version.model_index_id, &version.sha256))
    }) {
        return Err(anyhow!(
            "the verified signed-index model version has been revoked; install a replacement with a different digest"
        ));
    }
    state.load_progress.store(5, Ordering::Release);
    let manifest = bloomai_engine::load_manifest(&model_path)?;
    let backend_name = select_backend_name(&args.backend, &args.speculative, &manifest);
    validate_ifb_backend(args.enable_ifb, &backend_name)?;
    validate_strict_runtime_backend(&backend_name)?;
    engine_registry().get(&backend_name).map_err(|error| {
        anyhow!(
            "{}. Supported engines are: candle, openvino, funasr, qwen3_vl, longcat, intel-npu, npu-tts, onnxruntime, coreml, mlx, vulkan, llamacpp, wan.",
            error
        )
    })?;

    // Re-read and re-account at the loader boundary so a source changed after
    // admission cannot bypass the aggregate budget. Missing/corrupt sources
    // still arrive here through the historical asynchronous error path.
    let planned_memory = state
        .runtime_memory
        .planned_estimate(&manifest)
        .map_err(|error| anyhow!("runtime memory estimation failed before model load: {error}"))?;
    let planned_footprint = state
        .runtime_memory
        .footprint(&manifest, &planned_memory)
        .map_err(|error| anyhow!("runtime memory planning failed before model load: {error}"))?;
    let planned_available = state
        .runtime_memory
        .available()
        .map_err(|error| anyhow!("runtime memory probe failed before model load: {error}"))?;
    match memory_permit.as_mut() {
        Some(permit) => permit
            .resize(planned_footprint, planned_available)
            .map_err(|error| {
                anyhow!("runtime memory admission failed before model load: {error}")
            })?,
        None => {
            memory_permit = Some(
                state
                    .runtime_memory
                    .reserve(planned_footprint, planned_available)
                    .map_err(|error| {
                        anyhow!("runtime memory admission failed before model load: {error}")
                    })?,
            );
        }
    }
    state.load_progress.store(15, Ordering::Release);
    // This short-lived allocation is only a physical page-touch probe. The
    // aggregate runtime ledger above remains authoritative regardless of the
    // `disable_memory_prealloc` optimization switch.
    let page_touch_bytes = if args.disable_memory_prealloc {
        0
    } else if let Some(bytes) = args.reserve_memory_bytes {
        bytes
    } else {
        planned_memory
            .kv_cache_bytes
            .checked_add(planned_memory.temp_tensor_bytes)
            .ok_or_else(|| anyhow!("startup page-touch memory estimate overflow"))?
    };
    if page_touch_bytes > isize::MAX as usize {
        return Err(anyhow!(
            "startup page-touch reservation exceeds the platform allocation limit"
        ));
    }
    state.load_progress.store(25, Ordering::Release);
    drop(
        bloomai_engine::MemoryReservation::reserve(page_touch_bytes)
            .map_err(|error| anyhow!("startup page-touch reservation failed: {error}"))?,
    );

    let load_available = state
        .runtime_memory
        .available()
        .map_err(|error| anyhow!("runtime memory probe failed immediately before load: {error}"))?;
    memory_permit
        .as_ref()
        .ok_or_else(|| anyhow!("runtime memory permit is missing after admission"))?
        .revalidate_available(load_available)
        .map_err(|error| anyhow!("runtime memory headroom changed before model load: {error}"))?;

    state.load_progress.store(35, Ordering::Release);
    let context_size = args.context_size;
    let pipeline_path = model_path.clone();
    let pipeline = task::spawn_blocking(move || {
        let registry = engine_registry();
        let engine = registry
            .get(&backend_name)
            .map_err(|error| anyhow!(error.to_string()))?;
        InferencePipeline::load_standalone_with_context_strict(
            engine,
            device_kind,
            &pipeline_path,
            context_size,
        )
    })
    .await
    .map_err(|error| anyhow!("model loader task failed: {error}"))??;
    let pipeline = Arc::new(pipeline);
    let model_id = pipeline.metadata().id.clone();
    let loaded_manifest = pipeline.metadata().manifest.clone();
    validate_loaded_runtime_model_id(&model_id)?;
    let actual_device = pipeline.device();
    if actual_device != device_kind {
        return Err(anyhow!(
            "loaded pipeline changed device from {device_kind:?} to {actual_device:?}; refusing to publish a runtime whose memory topology cannot be accounted safely"
        ));
    }
    if args.enable_ifb {
        // IFB creates independently accounted execution wrappers. Do not keep
        // the startup verification wrapper as an untracked extra weight copy.
        pipeline.release_idle_weights();
    }
    tracing::info!(model = %model_id, "Model pipeline is loaded; preparing runtime services");

    let actual_context_size = pipeline
        .context_size()
        .checked_mul(args.max_concurrent.max(1))
        .ok_or_else(|| anyhow!("loaded runtime context memory estimate overflow"))?;
    let memory_estimate = state
        .runtime_memory
        .estimate_for_context(&loaded_manifest, actual_context_size, actual_device)
        .map_err(|error| anyhow!("loaded runtime memory estimation failed: {error}"))?;
    let actual_footprint = state
        .runtime_memory
        .footprint(&loaded_manifest, &memory_estimate)
        .map_err(|error| anyhow!("loaded runtime memory planning failed: {error}"))?;
    let actual_available = state
        .runtime_memory
        .available()
        .map_err(|error| anyhow!("loaded runtime memory probe failed: {error}"))?;
    memory_permit
        .as_mut()
        .ok_or_else(|| anyhow!("runtime memory permit is missing after admission"))?
        .resize(actual_footprint, actual_available)
        .map_err(|error| anyhow!("loaded runtime exceeds its memory admission: {error}"))?;
    state.load_progress.store(65, Ordering::Release);
    let active_request_leases = Arc::new(AtomicU64::new(0));
    let scheduling = build_scheduling_runtime(
        &state,
        args,
        &loaded_manifest,
        Arc::clone(&pipeline),
        &model_id,
        SchedulingRuntimeBuildContext {
            memory_context_size: actual_context_size,
            memory_estimate: &memory_estimate,
            runtime_lifetime_marker: Arc::clone(&active_request_leases),
            runtime_memory_permit: memory_permit.clone(),
        },
    )?;

    state.load_progress.store(95, Ordering::Release);
    let model_architecture = loaded_manifest
        .parameters
        .get("gguf_architecture")
        .or_else(|| loaded_manifest.parameters.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let model_chat_template = loaded_manifest
        .parameters
        .get("chat_template_kind")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Ok(LoadedRuntime {
        pipeline,
        model_id,
        model_family: loaded_manifest.family,
        model_architecture,
        model_chat_template,
        input_modalities: manifest.io_schema.inputs,
        memory_estimate,
        kv_cache_pool: scheduling.kv_cache_pool,
        cachemesh: scheduling.cachemesh,
        scheduler: scheduling.scheduler,
        _memory_reservation: scheduling.memory_reservation,
        scheduler_shutdown: scheduling.shutdown,
        published_at: unix_seconds(),
        source_path: model_path,
        catalog_id,
        signed_model_version,
        _runtime_memory_permit: memory_permit,
        active_request_leases,
    })
}

pub(crate) fn validate_ifb_backend(enable_ifb: bool, backend_name: &str) -> Result<()> {
    if enable_ifb && backend_name != "candle" {
        return Err(anyhow!(
            "in-flight batching requires the verified Candle batch backend; selected backend '{backend_name}' cannot be published with IFB enabled"
        ));
    }
    Ok(())
}

pub(crate) fn validate_strict_runtime_backend(backend_name: &str) -> Result<()> {
    if !matches!(backend_name, "candle" | "qwen3_vl") {
        return Err(anyhow!(
            "strict aggregate memory admission requires a backend that reports its verified physical device; selected backend '{backend_name}' does not yet provide that contract"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_runtime_model_ids_exclude_the_reserved_default_alias() {
        assert!(validate_loaded_runtime_model_id("resident-model").is_ok());
        assert!(validate_loaded_runtime_model_id("default").is_err());
        assert!(validate_loaded_runtime_model_id(" invalid ").is_err());
        assert!(validate_loaded_runtime_model_id("").is_err());
    }

    #[test]
    fn signed_provenance_is_bound_by_path_for_startup_model_loading() {
        let temp = tempfile::tempdir().unwrap();
        let model_path = temp.path().join("signed.gguf");
        std::fs::write(&model_path, b"signed model").unwrap();
        let model_path = model_path.canonicalize().unwrap();
        let catalog = ModelCatalog {
            root: temp.path().canonicalize().unwrap().display().to_string(),
            root_exists: true,
            models: vec![model_manager::ModelCatalogEntry {
                id: "signed.gguf".to_string(),
                name: "Signed".to_string(),
                kind: "file".to_string(),
                format: "gguf".to_string(),
                size_bytes: 12,
                size_complete: true,
                modified_at: Some(1),
                active: false,
                provenance: Some(model_provenance::ModelProvenance {
                    acquisition: model_provenance::ModelAcquisitionKind::Download,
                    model_index_id: Some("signed-model".to_string()),
                    source_url: None,
                    source_host: Some("huggingface.co".to_string()),
                    sha256: "ab".repeat(32),
                    file_count: None,
                    license: Some("Apache-2.0".to_string()),
                    installed_at: 1,
                    last_verified_at: None,
                    integrity_mismatch_at: None,
                }),
                provenance_error: None,
            }],
        };

        let identity = signed_model_version_from_catalog(&catalog, &model_path, None).unwrap();
        assert_eq!(identity.model_index_id, "signed-model");
        assert_eq!(identity.sha256, "ab".repeat(32));
    }

    #[test]
    fn ifb_rejects_non_candle_backends_before_runtime_publication() {
        let error = validate_ifb_backend(true, "onnxruntime")
            .unwrap_err()
            .to_string();
        assert!(error.contains("verified Candle batch backend"));
        assert!(validate_ifb_backend(true, "candle").is_ok());
        assert!(validate_ifb_backend(false, "onnxruntime").is_ok());
    }

    #[test]
    fn strict_admission_rejects_backends_without_verified_device_reporting() {
        assert!(validate_strict_runtime_backend("candle").is_ok());
        assert!(validate_strict_runtime_backend("qwen3_vl").is_ok());
        let error = validate_strict_runtime_backend("wan")
            .unwrap_err()
            .to_string();
        assert!(error.contains("verified physical device"));
    }
}
