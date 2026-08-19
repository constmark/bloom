use std::path::Path;

use anyhow::Result;
use bloomai_backend::Backend;
use bloomai_core::{
    BackendLease, BloomError, DeviceKind, GenerationParams, ResourceError, ResourceTicket,
};

use crate::core::memory::{available_system_memory, error_text_indicates_oom};
use crate::{Engine, InferenceRequest, LoadedModel, ModelInput, ModelOutput, model::OutputSink};

/// Default context size used for memory pre-checks when none is specified.
const DEFAULT_CONTEXT_SIZE: usize = 2048;

fn model_context_limit(manifest: &bloomai_core::ModelManifest) -> Option<usize> {
    [
        "max_seq_length",
        "max_position_embeddings",
        "context_length",
    ]
    .into_iter()
    .find_map(|name| {
        manifest
            .parameters
            .get(name)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
    })
}

pub struct InferencePipeline {
    model: Box<dyn LoadedModel>,
    /// Held lease (if any) for the resources used by this pipeline.
    #[allow(dead_code)]
    lease: Option<BackendLease>,
    /// Pre-load memory estimate (computed after manifest is available).
    memory_estimate: Option<crate::manifest_adapter::MemoryEstimate>,
    context_size: usize,
    device: DeviceKind,
}

impl InferencePipeline {
    /// Standalone load: no Backend required — the engine manages its own device.
    ///
    /// This is the primary entry-point for running the engine independently
    /// (like vLLM / SGLang), without the full runtime / backend layer.
    pub fn load_standalone(
        engine: &dyn Engine,
        device: DeviceKind,
        model_path: &Path,
    ) -> Result<Self> {
        Self::load_standalone_with_context(engine, device, model_path, DEFAULT_CONTEXT_SIZE)
    }

    /// Standalone load with an explicit context size for memory pre-check.
    pub fn load_standalone_with_context(
        engine: &dyn Engine,
        device: DeviceKind,
        model_path: &Path,
        context_size: usize,
    ) -> Result<Self> {
        let _span = tracing::info_span!(
            "pipeline.load_standalone",
            engine = engine.name(),
            device = ?device,
            model = %model_path.display()
        )
        .entered();
        tracing::info!("loading model (standalone, device={:?})", device);

        // Path validation: canonicalize and check for traversal
        let _canonical_path = crate::manifest_adapter::validate_model_path(model_path)?;

        // Pre-load memory check (best-effort; non-fatal if manifest can't be loaded).
        let pre_manifest = crate::manifest_adapter::load_manifest(model_path).ok();
        if let Some(ref manifest) = pre_manifest {
            // Hash verification is a hard gate when the manifest declares
            // hashes. Set BLOOM_ALLOW_HASH_MISMATCH=1 only for local
            // recovery/debugging of known-bad model packages.
            if let Err(e) = crate::manifest_adapter::verify_model_hashes(model_path, manifest) {
                if allow_hash_mismatch() {
                    tracing::warn!(
                        "Model file hash verification failed but BLOOM_ALLOW_HASH_MISMATCH=1 is set: {}",
                        e
                    );
                } else {
                    return Err(e);
                }
            }
        }
        let mut current_device = device;
        let mut current_context_size = pre_manifest
            .as_ref()
            .and_then(model_context_limit)
            .map_or(context_size, |limit| context_size.min(limit));
        if current_context_size < context_size {
            tracing::warn!(
                requested_context_size = context_size,
                model_context_size = current_context_size,
                "Clamped the runtime context to the model's advertised limit"
            );
        }
        let mut attempts = 0;
        let mut last_err = None;

        let model = loop {
            attempts += 1;
            if attempts > 5 {
                return Err(last_err
                    .unwrap_or_else(|| {
                        BloomError::Runtime("OOM recovery failed: too many attempts".into())
                    })
                    .into());
            }

            // 1. Pre-load memory check
            let temp_estimate = pre_manifest.as_ref().map(|m| {
                crate::manifest_adapter::estimate_memory_for_device(
                    m,
                    current_context_size,
                    current_device,
                )
            });

            if let Some(ref est) = temp_estimate {
                if let Some(avail) = available_system_memory() {
                    if est.total_bytes > avail {
                        let message = memory_budget_exceeded_message(est, avail);
                        if strict_memory_budget() {
                            return Err(BloomError::Resource(ResourceError::InsufficientRam {
                                requested: est.total_bytes,
                                available: avail,
                            })
                            .into());
                        }
                        tracing::warn!("{}. Attempting OOM degradation step...", message);

                        if current_context_size > 512 {
                            current_context_size = (current_context_size / 2).max(512);
                            tracing::warn!(
                                "Degradation Step: Reducing context size to {}",
                                current_context_size
                            );
                            continue;
                        } else if current_device != DeviceKind::Cpu {
                            current_device = DeviceKind::Cpu;
                            current_context_size = context_size; // Try full context size on CPU first
                            tracing::warn!("Degradation Step: Falling back to CPU device");
                            continue;
                        } else if current_context_size > 128 {
                            current_context_size = 128;
                            tracing::warn!(
                                "Degradation Step: Reducing context size to minimum (128) on CPU"
                            );
                            continue;
                        }
                    }
                } else if strict_memory_budget() {
                    return Err(BloomError::Resource(ResourceError::BackendUnavailable {
                        backend: "system".into(),
                        reason:
                            "Strict memory budget requested with BLOOM_STRICT_MEMORY_BUDGET=1, \
                                 but available system memory could not be detected on this platform"
                                .into(),
                    })
                    .into());
                }
                if attempts == 1 {
                    tracing::info!("Memory estimate: {}", est.display_summary());
                }
            }

            // 2. Load model
            match engine.load(model_path, current_device) {
                Ok(m) => {
                    if attempts > 1 {
                        tracing::info!(
                            "OOM Recovery Succeeded! Loaded model on {:?} with context size {}",
                            current_device,
                            current_context_size
                        );
                    } else {
                        tracing::info!("model loaded successfully (standalone)");
                    }
                    break m;
                }
                Err(e) => {
                    let err_str = e.to_string().to_lowercase();
                    let is_typed_oom = matches!(
                        e.downcast_ref::<BloomError>(),
                        Some(BloomError::Resource(ResourceError::InsufficientRam { .. }))
                            | Some(BloomError::Resource(ResourceError::InsufficientVram { .. }))
                            | Some(BloomError::Resource(
                                ResourceError::InsufficientUnifiedMemory { .. }
                            ))
                            | Some(BloomError::Resource(ResourceError::BudgetExceeded { .. }))
                    );
                    let is_oom = is_typed_oom
                        || error_text_indicates_oom(&err_str)
                        || err_str.contains("metal")
                        || err_str.contains("cuda");

                    if is_oom && !strict_memory_budget() {
                        tracing::warn!(
                            "Model load failed with OOM: {}. Attempting OOM degradation step...",
                            e
                        );
                        last_err = Some(BloomError::Runtime(format!("{}", e)));
                        if current_context_size > 512 {
                            current_context_size = (current_context_size / 2).max(512);
                            tracing::warn!(
                                "Degradation Step: Reducing context size to {}",
                                current_context_size
                            );
                            continue;
                        } else if current_device != DeviceKind::Cpu {
                            current_device = DeviceKind::Cpu;
                            current_context_size = context_size; // reset context size
                            tracing::warn!("Degradation Step: Falling back to CPU device");
                            continue;
                        } else if current_context_size > 128 {
                            current_context_size = 128;
                            tracing::warn!(
                                "Degradation Step: Reducing context size to minimum (128) on CPU"
                            );
                            continue;
                        }
                    }
                    return Err(e);
                }
            }
        };

        // Post-load estimate (manifest is now populated from actual model).
        let post_estimate = {
            let m = &model.metadata().manifest;
            let est = crate::manifest_adapter::estimate_memory_for_device(
                m,
                current_context_size,
                current_device,
            );
            Some(est)
        };

        let final_pre_estimate = pre_manifest.as_ref().map(|m| {
            crate::manifest_adapter::estimate_memory_for_device(
                m,
                current_context_size,
                current_device,
            )
        });

        Ok(Self {
            model,
            lease: None,
            memory_estimate: post_estimate.or(final_pre_estimate),
            context_size: current_context_size,
            device: current_device,
        })
    }

    /// Legacy load: requires a Backend for warmup and device discovery.
    pub fn load(engine: &dyn Engine, backend: &dyn Backend, model_path: &Path) -> Result<Self> {
        let _span = tracing::info_span!(
            "pipeline.load",
            engine = engine.name(),
            backend = backend.info().name,
            model = %model_path.display()
        )
        .entered();
        tracing::info!("loading model");
        let device = backend.info().device;
        let model = engine.load(model_path, device)?;
        let context_size = model_context_limit(&model.metadata().manifest)
            .map_or(DEFAULT_CONTEXT_SIZE, |limit| {
                DEFAULT_CONTEXT_SIZE.min(limit)
            });
        backend.warmup()?;
        tracing::info!("model loaded successfully");
        Ok(Self {
            model,
            lease: None,
            memory_estimate: None,
            context_size,
            device,
        })
    }

    /// Load with a resource ticket: reserves resources before loading.
    pub fn load_with_ticket(
        engine: &dyn Engine,
        backend: &dyn Backend,
        model_path: &Path,
        ticket: ResourceTicket,
    ) -> Result<Self> {
        let _span = tracing::info_span!(
            "pipeline.load_with_ticket",
            engine = engine.name(),
            backend = backend.info().name,
            model = %model_path.display(),
            model_id = %ticket.model_id
        )
        .entered();
        tracing::info!("reserving resources");
        let lease = backend.reserve(&ticket).map_err(|e| {
            BloomError::Resource(ResourceError::BackendUnavailable {
                backend: backend.info().name.to_string(),
                reason: format!("{}", e),
            })
        })?;
        backend.warmup()?;
        tracing::info!("loading model weights");
        let device = backend.info().device;
        let model = engine.load(model_path, device)?;
        let context_size = model_context_limit(&model.metadata().manifest)
            .map_or(DEFAULT_CONTEXT_SIZE, |limit| {
                DEFAULT_CONTEXT_SIZE.min(limit)
            });
        tracing::info!("model loaded with ticket");
        Ok(Self {
            model,
            lease: Some(lease),
            memory_estimate: None,
            context_size,
            device,
        })
    }

    pub fn run(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let _span = tracing::info_span!("pipeline.run").entered();
        self.model.infer(input, params)
    }

    pub fn run_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let _span = tracing::info_span!("pipeline.run_stream").entered();
        self.model.infer_stream(input, params, sink)
    }

    /// Whether the loaded model has a native embedding-batch execution path.
    pub fn supports_native_embedding_batch(&self) -> bool {
        self.model.supports_native_embedding_batch()
    }

    /// Produce one embedding per input while retaining input order.
    pub fn run_embedding_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let _span = tracing::info_span!("pipeline.run_embedding_batch", batch_size = inputs.len())
            .entered();
        self.model.embed_batch(inputs)
    }

    pub fn run_request(&self, request: InferenceRequest, sink: &mut dyn OutputSink) -> Result<()> {
        let _span = tracing::info_span!("pipeline.run_request").entered();
        self.model.infer_request(request, sink)
    }

    pub fn metadata(&self) -> &crate::ModelMetadata {
        self.model.metadata()
    }

    pub fn model(&self) -> &dyn LoadedModel {
        self.model.as_ref()
    }

    pub fn context_size(&self) -> usize {
        self.context_size
    }

    pub fn device(&self) -> DeviceKind {
        self.device
    }

    /// Return the pre-load memory estimate, if available.
    pub fn memory_estimate(&self) -> Option<&crate::manifest_adapter::MemoryEstimate> {
        self.memory_estimate.as_ref()
    }

    /// Tokenize text with the exact tokenizer owned by the loaded model.
    /// Processor-based tokenizers remain supported for non-Candle backends;
    /// the final approximation is only for engines that expose neither.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        #[cfg(feature = "candle-engine")]
        if let Some(tokenizer) = self.model.tokenizer() {
            let add_special_tokens =
                self.model.metadata().manifest.family == bloomai_core::ModelFamily::Bert;
            let encoding = tokenizer
                .encode(text, add_special_tokens)
                .map_err(|error| {
                    BloomError::Engine(format!("failed to tokenize benchmark input: {error}"))
                })?;
            return Ok(encoding.get_ids().to_vec());
        }

        if let Some(registry) = self.model.processors() {
            for name in registry.specs().iter().map(|s| &s.name) {
                if name.contains("tokenizer")
                    && let Ok(proc) = registry.get(name)
                {
                    let blocks =
                        proc.process(vec![crate::io::DataBlock::Text(text.to_string())])?;
                    if let Some(crate::io::DataBlock::Tokens(ids)) = blocks.first() {
                        return Ok(ids.clone());
                    }
                }
            }
        }
        // Fallback to word/character approximation
        Ok(text.split_whitespace().map(|_| 0).collect())
    }

    /// Detokenize token IDs back into a string using model processors
    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        if let Some(registry) = self.model.processors() {
            for name in registry.specs().iter().map(|s| &s.name) {
                if name.contains("tokenizer")
                    && let Ok(proc) = registry.get(name)
                {
                    let blocks =
                        proc.process(vec![crate::io::DataBlock::Tokens(tokens.to_vec())])?;
                    if let Some(crate::io::DataBlock::Text(text)) = blocks.first() {
                        return Ok(text.clone());
                    }
                }
            }
        }
        Err(BloomError::Engine("No tokenizer processor found".into()).into())
    }
}

fn allow_hash_mismatch() -> bool {
    std::env::var("BLOOM_ALLOW_HASH_MISMATCH")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn strict_memory_budget() -> bool {
    std::env::var("BLOOM_STRICT_MEMORY_BUDGET")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn memory_budget_exceeded_message(
    est: &crate::manifest_adapter::MemoryEstimate,
    available_bytes: usize,
) -> String {
    format!(
        "Memory pre-check failed: estimated {} > available {}",
        est.display_summary(),
        crate::manifest_adapter::format_bytes(available_bytes),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Engine, ModelMetadata};
    use bloomai_core::{
        DType, DeviceKind, Modality, ModelFamily, ModelFile, ModelFormat, ModelIoSchema,
        ModelManifest,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[test]
    fn sentence_encoder_context_prefers_task_limit_over_architecture_capacity() {
        let mut manifest = ModelManifest {
            family: ModelFamily::Bert,
            ..ModelManifest::default()
        };
        manifest.parameters.insert(
            "max_position_embeddings".to_string(),
            serde_json::json!(512),
        );
        manifest
            .parameters
            .insert("max_seq_length".to_string(), serde_json::json!(256));

        assert_eq!(model_context_limit(&manifest), Some(256));
    }

    struct DummyEngine {
        loaded: Arc<AtomicBool>,
    }

    impl Engine for DummyEngine {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn supported_modalities(&self) -> Vec<Modality> {
            vec![Modality::Text]
        }

        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }

        fn load(
            &self,
            _model_path: &Path,
            _device: DeviceKind,
        ) -> Result<Box<dyn crate::LoadedModel>> {
            self.loaded.store(true, Ordering::SeqCst);
            let manifest = ModelManifest {
                id: "dummy".to_string(),
                family: ModelFamily::Custom("dummy".to_string()),
                primary_dtype: DType::F32,
                io_schema: ModelIoSchema {
                    inputs: vec![Modality::Text],
                    outputs: vec![Modality::Text],
                },
                license: Some("MIT".to_string()),
                ..ModelManifest::default()
            };
            Ok(Box::new(DummyModel {
                metadata: ModelMetadata {
                    id: "dummy".to_string(),
                    modality: Modality::Text,
                    quantized: false,
                    manifest,
                },
            }))
        }
    }

    struct DummyModel {
        metadata: ModelMetadata,
    }

    impl crate::LoadedModel for DummyModel {
        fn metadata(&self) -> &ModelMetadata {
            &self.metadata
        }

        fn infer(&self, _input: ModelInput, _params: &GenerationParams) -> Result<ModelOutput> {
            Ok(ModelOutput {
                text: Some("ok".to_string()),
                logits: None,
                image: None,
                audio: None,
                video: None,
            })
        }
    }

    #[test]
    fn load_standalone_rejects_declared_hash_mismatch_before_engine_load() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("weights.bin"), b"not-the-declared-content").unwrap();

        let manifest = ModelManifest {
            id: "hash-gated-model".to_string(),
            family: ModelFamily::Custom("hash-gate".to_string()),
            primary_dtype: DType::F32,
            io_schema: ModelIoSchema {
                inputs: vec![Modality::Text],
                outputs: vec![Modality::Text],
            },
            license: Some("MIT".to_string()),
            files: vec![ModelFile {
                name: "weights.bin".to_string(),
                format: ModelFormat::Unknown,
                size_bytes: 1,
                hash_sha256: Some(
                    "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
                ),
                required: true,
            }],
            ..ModelManifest::default()
        };
        std::fs::write(
            dir.path().join("bloom.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let loaded = Arc::new(AtomicBool::new(false));
        let engine = DummyEngine {
            loaded: Arc::clone(&loaded),
        };

        let err = match InferencePipeline::load_standalone_with_context(
            &engine,
            DeviceKind::Cpu,
            dir.path(),
            16,
        ) {
            Ok(_) => panic!("hash mismatch should fail before model load"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("hash mismatch"));
        assert!(!loaded.load(Ordering::SeqCst));
    }

    struct FailOnGpuEngine {
        attempts: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Engine for FailOnGpuEngine {
        fn name(&self) -> &'static str {
            "fail-gpu"
        }
        fn supported_modalities(&self) -> Vec<Modality> {
            vec![Modality::Text]
        }
        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu, DeviceKind::Gpu]
        }

        fn load(
            &self,
            _model_path: &Path,
            device: DeviceKind,
        ) -> Result<Box<dyn crate::LoadedModel>> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            if device == DeviceKind::Gpu {
                return Err(BloomError::Resource(ResourceError::InsufficientVram {
                    requested: 0,
                    available: 0,
                })
                .into());
            }
            let manifest = ModelManifest {
                id: "dummy".to_string(),
                family: ModelFamily::Custom("dummy".to_string()),
                primary_dtype: DType::F32,
                ..ModelManifest::default()
            };
            Ok(Box::new(DummyModel {
                metadata: ModelMetadata {
                    id: "dummy".to_string(),
                    modality: Modality::Text,
                    quantized: false,
                    manifest,
                },
            }))
        }
    }

    #[test]
    fn test_load_standalone_oom_cascade_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = ModelManifest {
            id: "dummy".to_string(),
            family: ModelFamily::Custom("dummy".to_string()),
            primary_dtype: DType::F32,
            ..ModelManifest::default()
        };
        std::fs::write(
            dir.path().join("bloom.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine = FailOnGpuEngine {
            attempts: Arc::clone(&attempts),
        };

        // If strict memory budget is NOT set, it should fallback to CPU and succeed
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_STRICT_MEMORY_BUDGET") };
        let pipeline = InferencePipeline::load_standalone_with_context(
            &engine,
            DeviceKind::Gpu,
            dir.path(),
            2048,
        )
        .unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 4); // GPU 2048 -> GPU 1024 -> GPU 512 -> CPU 2048
        assert_eq!(pipeline.model.metadata().id, "dummy");
        assert_eq!(pipeline.device(), DeviceKind::Cpu);
        assert_eq!(pipeline.context_size(), 2048);
    }
}
