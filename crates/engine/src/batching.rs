//! Assemble a loaded model's batch executor without exposing tensor or model-wrapper
//! implementation details to application layers. Worker lifetime and process shutdown
//! remain the caller's responsibility.

use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::{
    BloomKvCachePool, CacheMesh, EngineExecutor, InferencePipeline, KvCachePool, LongContextPolicy,
};

/// Model and cache layout admitted by the application's memory planner.
#[derive(Debug, Clone)]
pub struct BatchExecutorConfig {
    pub block_size: usize,
    pub total_blocks: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub long_context_policy: LongContextPolicy,
}

/// Backend-neutral execution capability implemented only by batchable models.
/// Tensor layout, wrapper ownership, tokenization, and KV hooks stay in the adapter.
pub trait BatchModel: Send + Sync {
    fn build_executor(
        self: Arc<Self>,
        kv_pool: Arc<BloomKvCachePool>,
        cachemesh: Option<Arc<CacheMesh>>,
        config: BatchExecutorConfig,
    ) -> Result<Arc<dyn EngineExecutor>>;
}

/// Build a batch executor using the loaded model's capability, after checking
/// cache layout and admitted device. Unsupported models fail before construction;
/// this function never spawns a worker or selects a different backend.
pub fn build_batch_executor(
    pipeline: Arc<InferencePipeline>,
    kv_pool: Arc<BloomKvCachePool>,
    cachemesh: Option<Arc<CacheMesh>>,
    config: BatchExecutorConfig,
) -> Result<Arc<dyn EngineExecutor>> {
    config.validate(&kv_pool)?;
    let model = pipeline.shared_model();
    let actual_device = model.actual_device();
    let batch_model = model
        .batch_model()
        .ok_or_else(|| anyhow!("the loaded model does not support continuous batching"))?;
    if actual_device != Some(pipeline.device()) {
        anyhow::bail!("the batch executor device does not match the admitted pipeline device");
    }
    let executor = batch_model.build_executor(kv_pool, cachemesh, config)?;
    Ok(Arc::new(PipelineBatchExecutor {
        executor,
        _pipeline: pipeline,
    }))
}

// Keep the pipeline's backend resource lease alive until executor teardown.
// Holding just the model would release that lease before its wrappers finish.
struct PipelineBatchExecutor {
    executor: Arc<dyn EngineExecutor>,
    _pipeline: Arc<InferencePipeline>,
}

impl EngineExecutor for PipelineBatchExecutor {
    fn execute(
        &self,
        batch: crate::scheduler::ExecutionBatch,
    ) -> Result<crate::scheduler::BatchResult> {
        self.executor.execute(batch)
    }

    fn max_batch_size(&self, phase: crate::scheduler::ExecutionPhase) -> usize {
        self.executor.max_batch_size(phase)
    }
}

impl BatchExecutorConfig {
    pub(crate) fn validate(&self, kv_pool: &BloomKvCachePool) -> Result<()> {
        self.num_kv_heads
            .checked_mul(self.head_dim)
            .filter(|dim| *dim > 0)
            .ok_or_else(|| {
                anyhow!("batch executor KV dimensions must be positive and not overflow")
            })?;
        if self.block_size == 0 || self.total_blocks == 0 || self.num_layers == 0 {
            anyhow::bail!("batch executor cache dimensions must be positive");
        }
        if self.block_size != kv_pool.block_size()
            || self.total_blocks != kv_pool.get_metrics().total_blocks
        {
            anyhow::bail!("batch executor layout does not match its admitted KV pool");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EchoTextModel, Engine, LoadedModel};
    use bloomai_core::{DeviceKind, Modality};

    struct UnbatchableEngine;

    impl Engine for UnbatchableEngine {
        fn name(&self) -> &'static str {
            "unbatchable-test"
        }
        fn supported_modalities(&self) -> Vec<Modality> {
            vec![Modality::Text]
        }
        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }
        fn load(&self, _: &std::path::Path, _: DeviceKind) -> Result<Box<dyn LoadedModel>> {
            Ok(Box::new(EchoTextModel::default()))
        }
    }

    fn unbatchable_pipeline() -> Arc<InferencePipeline> {
        Arc::new(
            InferencePipeline::load(
                &UnbatchableEngine,
                &bloomai_backend::CpuBackend,
                std::path::Path::new("unused"),
            )
            .unwrap(),
        )
    }

    fn config() -> BatchExecutorConfig {
        BatchExecutorConfig {
            block_size: 16,
            total_blocks: 8,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            long_context_policy: LongContextPolicy::Full,
        }
    }

    #[test]
    fn unavailable_batch_implementation_fails_without_retaining_pipeline() {
        let pipeline = unbatchable_pipeline();
        let weak = Arc::downgrade(&pipeline);
        let error = build_batch_executor(
            pipeline,
            Arc::new(BloomKvCachePool::new(16, 8)),
            None,
            config(),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("does not support continuous batching"));
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn invalid_layout_is_rejected_before_executor_construction() {
        for invalid in [
            BatchExecutorConfig {
                head_dim: 0,
                ..config()
            },
            BatchExecutorConfig {
                head_dim: usize::MAX,
                num_kv_heads: 2,
                ..config()
            },
            BatchExecutorConfig {
                block_size: 0,
                ..config()
            },
            BatchExecutorConfig {
                total_blocks: 9,
                ..config()
            },
        ] {
            let error = build_batch_executor(
                unbatchable_pipeline(),
                Arc::new(BloomKvCachePool::new(16, 8)),
                None,
                invalid,
            )
            .err()
            .unwrap()
            .to_string();
            assert!(error.contains("batch executor"), "{error}");
        }
    }
    struct PortableModel {
        echo: EchoTextModel,
        device: Option<DeviceKind>,
    }

    impl LoadedModel for PortableModel {
        fn actual_device(&self) -> Option<DeviceKind> {
            self.device
        }
        fn batch_model(self: Arc<Self>) -> Option<Arc<dyn BatchModel>> {
            Some(self)
        }
        fn metadata(&self) -> &crate::ModelMetadata {
            self.echo.metadata()
        }
        fn infer(
            &self,
            input: crate::ModelInput,
            params: &bloomai_core::GenerationParams,
        ) -> Result<crate::ModelOutput> {
            self.echo.infer(input, params)
        }
        fn tokenize(&self, text: &str) -> Option<Result<Vec<u32>>> {
            Some(if text == "invalid" {
                Err(anyhow!("encoding failed"))
            } else {
                Ok(vec![42, 43])
            })
        }
    }

    impl BatchModel for PortableModel {
        fn build_executor(
            self: Arc<Self>,
            _: Arc<BloomKvCachePool>,
            _: Option<Arc<CacheMesh>>,
            _: BatchExecutorConfig,
        ) -> Result<Arc<dyn EngineExecutor>> {
            Ok(self)
        }
    }

    impl EngineExecutor for PortableModel {
        fn execute(
            &self,
            batch: crate::scheduler::ExecutionBatch,
        ) -> Result<crate::scheduler::BatchResult> {
            Ok(crate::scheduler::BatchResult {
                next_tokens: batch.tokens,
                speculative_tokens: None,
            })
        }
        fn max_batch_size(&self, _: crate::scheduler::ExecutionPhase) -> usize {
            3
        }
    }

    struct PortableEngine(Option<DeviceKind>);
    impl Engine for PortableEngine {
        fn name(&self) -> &'static str {
            "portable-test"
        }
        fn supported_modalities(&self) -> Vec<Modality> {
            vec![Modality::Text]
        }
        fn supported_devices(&self) -> Vec<DeviceKind> {
            vec![DeviceKind::Cpu]
        }
        fn load(&self, _: &std::path::Path, _: DeviceKind) -> Result<Box<dyn LoadedModel>> {
            Ok(Box::new(PortableModel {
                echo: EchoTextModel::default(),
                device: self.0,
            }))
        }
    }

    fn portable_pipeline(device: Option<DeviceKind>) -> Arc<InferencePipeline> {
        Arc::new(
            InferencePipeline::load(
                &PortableEngine(device),
                &bloomai_backend::CpuBackend,
                std::path::Path::new("unused"),
            )
            .unwrap(),
        )
    }

    #[test]
    fn portable_capabilities_retain_pipeline_until_executor_teardown() {
        use crate::scheduler::{ExecutionBatch, ExecutionPhase};
        let pipeline = portable_pipeline(Some(DeviceKind::Cpu));
        let weak = Arc::downgrade(&pipeline);
        // Exact adapter encoding works even without Candle; failures never fall
        // back to approximate word counts.
        assert_eq!(pipeline.tokenize("two tokens").unwrap(), vec![42, 43]);
        assert!(
            pipeline
                .tokenize("invalid")
                .unwrap_err()
                .to_string()
                .contains("encoding failed")
        );
        let executor = build_batch_executor(
            pipeline,
            Arc::new(BloomKvCachePool::new(16, 8)),
            None,
            config(),
        )
        .unwrap();
        assert!(weak.upgrade().is_some());
        assert_eq!(executor.max_batch_size(ExecutionPhase::Decode), 3);
        let result = executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Decode,
                request_ids: vec!["portable".to_string()],
                tokens: vec![42],
                cu_seqlens: vec![0, 1],
                kv_handles: vec![0],
                start_positions: vec![0],
                params: vec![bloomai_core::GenerationParams::default()],
                generated_tokens: vec![vec![]],
            })
            .unwrap();
        assert_eq!(result.next_tokens, vec![42]);
        drop(executor);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn batch_capability_rejects_unknown_or_mismatched_physical_device() {
        for device in [None, Some(DeviceKind::Gpu)] {
            let pipeline = portable_pipeline(device);
            let weak = Arc::downgrade(&pipeline);
            let error = build_batch_executor(
                pipeline,
                Arc::new(BloomKvCachePool::new(16, 8)),
                None,
                config(),
            )
            .err()
            .unwrap();
            assert!(error.to_string().contains("admitted pipeline device"));
            assert!(weak.upgrade().is_none());
        }
    }
}
