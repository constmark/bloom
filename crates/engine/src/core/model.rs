use anyhow::Result;
use bloomai_core::{BloomError, CacheHandle, GenerationParams, Modality, ModelManifest};

use crate::io::{DataBlock, InferenceRequest, ModelInput, ModelOutput, OutputChunk};

/// Opaque blob representing exported model state (KV cache, latent state,
/// audio window, etc.) that can be migrated across backends.
#[derive(Debug, Clone)]
pub struct StateBlob {
    /// Identifier of the cache handle this blob was exported from.
    pub handle_id: u64,
    /// Model that owns this state.
    pub model_id: String,
    /// Opaque serialised state bytes.
    pub bytes: Vec<u8>,
    /// Human-readable description of the state kind (e.g. "kv_cache",
    /// "world_state", "audio_window").
    pub kind: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub modality: Modality,
    pub quantized: bool,
    pub manifest: ModelManifest,
}

pub trait OutputSink: Send + Sync {
    fn on_chunk(&mut self, chunk: OutputChunk) -> Result<()>;
}

impl<F> OutputSink for F
where
    F: FnMut(OutputChunk) -> Result<()> + Send + Sync,
{
    fn on_chunk(&mut self, chunk: OutputChunk) -> Result<()> {
        self(chunk)
    }
}

pub trait LoadedModel: Send + Sync {
    #[cfg(feature = "candle-engine")]
    fn forward(
        &self,
        _input_ids: &candle_core::Tensor,
        _start_pos: usize,
    ) -> Result<candle_core::Tensor> {
        Err(BloomError::Engine("forward not supported for this model".into()).into())
    }

    #[cfg(feature = "candle-engine")]
    fn create_wrapper(&self) -> Result<Box<dyn std::any::Any + Send + Sync>> {
        Err(BloomError::Engine(
            "create_wrapper not supported for this model".into(),
        )
        .into())
    }

    fn clear_kv_cache(&self) {}

    /// Whether this model's KV cache can be bridged into the paged
    /// attention cache via a `KvHook`.
    ///
    /// Returns `false` by default. Engines that expose a streaming variant
    /// with a hookable KV cache (e.g. Candle's `QwenStreamingModelForCausalLM`)
    /// should override this to return `true` so the server can attach a
    /// `KvHook` to the batch executor, turning the paged KV cache from
    /// metadata-only into real cross-request KV reuse.
    fn supports_paged_kv(&self) -> bool {
        false
    }

    /// Decoded vocabulary strings indexed by token id, used for token-level
    /// grammar constraints (e.g. JSON schema filtering) on the IFB hot path.
    ///
    /// Returns an empty slice by default. Engines that can supply a vocab
    /// (e.g. Candle's `CandleTextModel`) should override this so the batch
    /// executor can apply `filter_logits_by_grammar` before sampling.
    fn vocab_strings(&self) -> &[String] {
        &[]
    }

    /// End-of-sequence token ids for the loaded model, used by grammar
    /// filtering to decide when a structured response is allowed to
    /// terminate. Returns an empty slice by default.
    fn eos_token_ids(&self) -> &[u32] {
        &[]
    }

    fn metadata(&self) -> &ModelMetadata;
    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput>;
    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let output = self.infer(input, params)?;
        if let Some(text) = output.text {
            sink.on_chunk(OutputChunk::TextDelta(text))?;
        }
        sink.on_chunk(OutputChunk::End)?;
        Ok(())
    }

    fn processors(&self) -> Option<&crate::processor::ProcessorRegistry> {
        None
    }

    #[cfg(feature = "candle-engine")]
    fn tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        None
    }

    /// Export the current model state (KV cache, latent state, audio
    /// window, etc.) as an opaque blob for migration across backends.
    ///
    /// The default implementation returns `Unsupported` — engines that
    /// support state migration should override this.
    fn export_state(&self, handle: &CacheHandle) -> Result<StateBlob> {
        Err(BloomError::Engine(
            format!(
                "model '{}' does not support state export (handle_id={}); \
                 engine '{}' has not implemented export_state",
                self.metadata().id,
                handle.handle_id,
                self.metadata().manifest.id,
            )
        )
        .into())
    }

    /// Import a previously exported state blob, returning a new cache
    /// handle that the runtime can track.
    ///
    /// The default implementation returns `Unsupported`.
    fn import_state(&self, _blob: StateBlob) -> Result<CacheHandle> {
        Err(BloomError::Engine(
            format!(
                "model '{}' does not support state import; \
                 engine '{}' has not implemented import_state",
                self.metadata().id, self.metadata().manifest.id,
            )
        )
        .into())
    }

    fn infer_request(&self, request: InferenceRequest, sink: &mut dyn OutputSink) -> Result<()> {
        let mut blocks = request.blocks;
        if let Some(registry) = self.processors() {
            for spec in &self.metadata().manifest.processors {
                if let Ok(proc) = registry.get(&spec.name) {
                    blocks = proc.process(blocks)?;
                }
            }
        }
        let input = convert_blocks_to_model_input(blocks)?;
        let gen_params = GenerationParams {
            max_tokens: request.params.max_tokens,
            temperature: request.params.temperature,
            top_p: request.params.top_p,
            seed: request.params.seed,
            response_format: request.params.response_format.clone(),
        };
        self.infer_stream(input, &gen_params, sink)
    }
}

fn convert_blocks_to_model_input(blocks: Vec<DataBlock>) -> Result<ModelInput> {
    if blocks.is_empty() {
        return Err(BloomError::InvalidInput("Empty blocks in InferenceRequest".into()).into());
    }

    let mut text_opt = None;
    let mut audio_opt = None;
    let mut audio_sample_rate = None;
    let mut audio_file_opt = None;
    let mut image_opt = None;
    let mut image_mime = None;

    for block in &blocks {
        match block {
            DataBlock::Text(text) => text_opt = Some(text.clone()),
            DataBlock::AudioPcm {
                samples,
                sample_rate,
            } => {
                audio_opt = Some(samples.clone());
                audio_sample_rate = Some(*sample_rate);
            }
            DataBlock::AudioFile { path, language } => {
                audio_file_opt = Some((path.clone(), language.clone()));
            }
            DataBlock::Image { bytes, mime } => {
                image_opt = Some(bytes.clone());
                image_mime = Some(mime.clone());
            }
            _ => {}
        }
    }

    if audio_file_opt.is_some()
        && (text_opt.is_some() || audio_opt.is_some() || image_opt.is_some())
    {
        return Err(BloomError::InvalidInput(
            "AudioFile cannot be combined with other blocks in legacy ModelInput; use decoded AudioPcm for multimodal requests".into()
        ).into());
    }

    let typed_modality_count = usize::from(text_opt.is_some())
        + usize::from(audio_opt.is_some())
        + usize::from(image_opt.is_some());

    if typed_modality_count > 1 {
        return Ok(ModelInput::Multi {
            text: text_opt,
            audio: audio_opt,
            image: image_opt,
        });
    }

    if let Some((path, language)) = audio_file_opt {
        return Ok(ModelInput::AudioFile { path, language });
    }

    if let Some(samples) = audio_opt {
        return Ok(ModelInput::Audio {
            samples,
            sample_rate: audio_sample_rate.unwrap_or(16000),
        });
    }

    if let Some(bytes) = image_opt {
        return Ok(ModelInput::Vision {
            bytes,
            mime: image_mime.unwrap_or_else(|| "application/octet-stream".to_string()),
        });
    }

    if let Some(prompt) = text_opt {
        return Ok(ModelInput::Text { prompt });
    }

    Err(BloomError::InvalidInput(
        "Could not convert blocks to legacy ModelInput".into()
    ).into())
}

pub struct EchoTextModel {
    meta: ModelMetadata,
}

impl Default for EchoTextModel {
    fn default() -> Self {
        Self {
            meta: ModelMetadata {
                id: "demo.echo.text".to_string(),
                modality: Modality::Text,
                quantized: false,
                manifest: ModelManifest::default(),
            },
        }
    }
}

impl LoadedModel for EchoTextModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }

    fn infer(&self, input: ModelInput, _params: &GenerationParams) -> Result<ModelOutput> {
        let text = match input {
            ModelInput::Text { prompt } => Some(format!("echo: {}", prompt)),
            _ => Some("model expects text input".to_string()),
        };

        Ok(ModelOutput {
            text,
            logits: None,
            image: None,
            audio: None,
            video: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ModelInput;

    #[test]
    fn test_echo_text_model_metadata() {
        let model = EchoTextModel::default();
        let meta = model.metadata();
        assert_eq!(meta.id, "demo.echo.text");
        assert_eq!(meta.modality, Modality::Text);
        assert!(!meta.quantized);
    }

    #[test]
    fn test_echo_text_model_infer_text() {
        let model = EchoTextModel::default();
        let input = ModelInput::Text {
            prompt: "hello".to_string(),
        };
        let params = GenerationParams::default();
        let output = model.infer(input, &params).unwrap();
        assert_eq!(output.text, Some("echo: hello".to_string()));
        assert!(output.logits.is_none());
    }

    #[test]
    fn test_echo_text_model_infer_invalid() {
        let model = EchoTextModel::default();
        let input = ModelInput::Audio {
            samples: vec![],
            sample_rate: 16000,
        };
        let params = GenerationParams::default();
        let output = model.infer(input, &params).unwrap();
        assert_eq!(output.text, Some("model expects text input".to_string()));
    }

    #[test]
    fn test_echo_text_model_infer_stream() {
        let model = EchoTextModel::default();
        let input = ModelInput::Text {
            prompt: "stream_test".to_string(),
        };
        let params = GenerationParams::default();

        let mut stream_output = String::new();
        model
            .infer_stream(input, &params, &mut |chunk: crate::io::OutputChunk| {
                if let crate::io::OutputChunk::TextDelta(text) = chunk {
                    stream_output.push_str(&text);
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(stream_output, "echo: stream_test");
    }

    #[test]
    fn test_model_infer_request_with_processor() {
        let mut manifest = ModelManifest::default();
        manifest.processors.push(bloomai_core::ProcessorSpec {
            name: "test_processor".to_string(),
            kind: bloomai_core::ProcessorKind::TextTokenizer,
            version: "1".to_string(),
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Text],
            parameters: std::collections::HashMap::new(),
        });

        struct ProcessedEchoModel {
            meta: ModelMetadata,
            processors: crate::processor::ProcessorRegistry,
        }

        impl LoadedModel for ProcessedEchoModel {
            fn metadata(&self) -> &ModelMetadata {
                &self.meta
            }
            fn processors(&self) -> Option<&crate::processor::ProcessorRegistry> {
                Some(&self.processors)
            }
            fn infer(&self, input: ModelInput, _params: &GenerationParams) -> Result<ModelOutput> {
                let text = match input {
                    ModelInput::Text { prompt } => Some(prompt),
                    _ => None,
                };
                Ok(ModelOutput {
                    text,
                    logits: None,
                    image: None,
                    audio: None,
                    video: None,
                })
            }
        }

        struct AppendProcessor;
        impl crate::processor::Processor for AppendProcessor {
            fn name(&self) -> &str {
                "test_processor"
            }
            fn kind(&self) -> bloomai_core::ProcessorKind {
                bloomai_core::ProcessorKind::TextTokenizer
            }
            fn input_modalities(&self) -> Vec<Modality> {
                vec![Modality::Text]
            }
            fn output_modalities(&self) -> Vec<Modality> {
                vec![Modality::Text]
            }
            fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>> {
                let mut out = Vec::new();
                for b in blocks {
                    if let DataBlock::Text(t) = b {
                        out.push(DataBlock::Text(format!("{} processed", t)));
                    } else {
                        out.push(b);
                    }
                }
                Ok(out)
            }
        }

        let mut registry = crate::processor::ProcessorRegistry::default();
        registry.register(Box::new(AppendProcessor));

        let model = ProcessedEchoModel {
            meta: ModelMetadata {
                id: "test.processed.echo".to_string(),
                modality: Modality::Text,
                quantized: false,
                manifest,
            },
            processors: registry,
        };

        let request = InferenceRequest {
            blocks: vec![DataBlock::Text("input".to_string())],
            params: crate::io::InferenceParams::default(),
        };

        let mut output = String::new();
        model
            .infer_request(request, &mut |chunk: crate::io::OutputChunk| {
                if let crate::io::OutputChunk::TextDelta(t) = chunk {
                    output.push_str(&t);
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(output, "input processed");
    }

    #[test]
    fn test_convert_blocks_preserves_multimodal_text_image() {
        let input = convert_blocks_to_model_input(vec![
            DataBlock::Text("describe".to_string()),
            DataBlock::Image {
                bytes: vec![1, 2, 3],
                mime: "image/png".to_string(),
            },
        ])
        .unwrap();

        match input {
            ModelInput::Multi { text, audio, image } => {
                assert_eq!(text, Some("describe".to_string()));
                assert_eq!(audio, None);
                assert_eq!(image, Some(vec![1, 2, 3]));
            }
            _ => panic!("Expected ModelInput::Multi"),
        }
    }

    #[test]
    fn test_convert_blocks_preserves_single_audio_sample_rate() {
        let input = convert_blocks_to_model_input(vec![DataBlock::AudioPcm {
            samples: vec![0.1, 0.2],
            sample_rate: 44100,
        }])
        .unwrap();

        match input {
            ModelInput::Audio {
                samples,
                sample_rate,
            } => {
                assert_eq!(samples, vec![0.1, 0.2]);
                assert_eq!(sample_rate, 44100);
            }
            _ => panic!("Expected ModelInput::Audio"),
        }
    }

    #[test]
    fn test_default_export_import_state_unsupported() {
        // EchoTextModel does not override export_state/import_state,
        // so they should return errors explaining the model doesn't support it.
        let model = EchoTextModel::default();
        let handle = CacheHandle {
            handle_id: 42,
            model_id: "demo.echo.text".to_string(),
            cache_kind: bloomai_core::CacheKind::KvCache,
            bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
        };
        let export_err = model.export_state(&handle).unwrap_err();
        assert!(export_err
            .to_string()
            .contains("does not support state export"));

        let blob = StateBlob {
            handle_id: 1,
            model_id: "demo.echo.text".to_string(),
            bytes: vec![],
            kind: "kv_cache".to_string(),
        };
        let import_err = model.import_state(blob).unwrap_err();
        assert!(import_err
            .to_string()
            .contains("does not support state import"));
    }

    #[test]
    fn test_custom_model_export_import_state() {
        use std::sync::atomic::{AtomicU64, Ordering};

        struct StatefulModel {
            meta: ModelMetadata,
            next_handle: AtomicU64,
        }

        impl LoadedModel for StatefulModel {
            fn metadata(&self) -> &ModelMetadata {
                &self.meta
            }
            fn infer(&self, _input: ModelInput, _params: &GenerationParams) -> Result<ModelOutput> {
                Ok(ModelOutput {
                    text: Some("ok".into()),
                    logits: None,
                    image: None,
                    audio: None,
                    video: None,
                })
            }
            fn export_state(&self, handle: &CacheHandle) -> Result<StateBlob> {
                Ok(StateBlob {
                    handle_id: handle.handle_id,
                    model_id: self.meta.id.clone(),
                    bytes: vec![1, 2, 3, 4], // simulated state
                    kind: "kv_cache".to_string(),
                })
            }
            fn import_state(&self, blob: StateBlob) -> Result<CacheHandle> {
                let id = self.next_handle.fetch_add(1, Ordering::SeqCst);
                Ok(CacheHandle {
                    handle_id: id,
                    model_id: blob.model_id,
                    cache_kind: bloomai_core::CacheKind::KvCache,
                    bytes: blob.bytes.len(),
                    priority: bloomai_core::ResourcePriority::Normal,
                })
            }
        }

        let model = StatefulModel {
            meta: ModelMetadata {
                id: "test.stateful".to_string(),
                modality: Modality::Text,
                quantized: false,
                manifest: ModelManifest::default(),
            },
            next_handle: AtomicU64::new(100),
        };

        // Export
        let handle = CacheHandle {
            handle_id: 42,
            model_id: "test.stateful".to_string(),
            cache_kind: bloomai_core::CacheKind::KvCache,
            bytes: 256,
            priority: bloomai_core::ResourcePriority::Normal,
        };
        let blob = model.export_state(&handle).unwrap();
        assert_eq!(blob.bytes, vec![1, 2, 3, 4]);
        assert_eq!(blob.kind, "kv_cache");

        // Import
        let new_handle = model.import_state(blob).unwrap();
        assert_eq!(new_handle.handle_id, 100);
        assert_eq!(new_handle.bytes, 4);
    }
}
