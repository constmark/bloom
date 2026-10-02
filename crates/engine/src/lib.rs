// Surface new `unwrap` usage in production code. Tests may use unwrap to keep
// assertions concise; runtime paths must recover or return a structured error.
#![cfg_attr(not(test), warn(clippy::unwrap_used))]
//! Model abstraction for multimodal inference.

pub mod batching;
pub mod cachemesh;
pub mod core;
pub mod executor;
pub mod plugin;
pub mod processor;
pub mod scheduler;
pub mod world;

/// Shared process-environment isolation for unit tests.
///
/// Production code never compiles this module. Tests that exercise
/// environment-backed configuration must hold `ENV_LOCK` for the complete
/// operation and keep an `EnvVarGuard` alive until all environment reads have
/// finished.
#[cfg(test)]
pub(crate) mod test_env {
    use std::ffi::OsString;
    use std::sync::Mutex;

    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

    pub(crate) struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        pub(crate) fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: callers hold the shared test environment lock.
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }

        pub(crate) fn remove(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: callers hold the shared test environment lock.
            unsafe { std::env::remove_var(key) };
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: the guard is only used while its shared test lock is held.
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }
}

// --- Compatibility & Convenience Crate-Root Module Names ---
// These allow `crate::engine::*` and external `bloomai_engine::engine::*` to work exactly as before.
pub use crate::core::engine;
pub use crate::core::io;
pub use crate::core::manifest as manifest_adapter;
pub use crate::core::model;
pub use crate::core::pipeline;
pub use crate::plugin as plugin_manager;
pub use crate::world as world_model;

#[cfg(feature = "candle-engine")]
pub use crate::executor as engines;

// --- Direct re-exports at the crate root ---
pub use crate::cachemesh::{
    CacheMesh, CacheMeshBlock, CacheMeshConfig, CacheMeshKey, CacheMeshMetrics, CacheMeshSnapshot,
    CacheMeshTier, FileSystemRemoteCache, InMemoryRemoteCache, RemoteCacheBackend, TierMetrics,
};
pub use crate::core::engine::{
    BackendMaturity, Engine, EngineCapability, EngineRegistry, EngineRouter, RoutingDecision,
    SupportLevel, default_engine_supports, device_kind_from_capability,
};
pub use crate::core::io::{
    DataBlock, InferenceParams, InferenceRequest, ModelInput, ModelOutput, OutputChunk,
};
pub use crate::core::manifest::{
    MemoryEstimate, estimate_memory, estimate_memory_for_device, format_bytes, infer_quantization,
    load_manifest, model_manifest_supports_embeddings, model_manifest_tasks,
    resolve_hf_safetensors_files,
};
pub use crate::core::memory::{
    MemoryPreallocationConfig, MemoryPreallocationPlan, MemoryReservation, available_system_memory,
    default_memory_utilization, plan_memory_preallocation, reserve_memory_for_plan,
};
pub use crate::core::model::{EchoTextModel, LoadedModel, ModelMetadata, StateBlob};
pub use crate::core::parallelism::{
    CollectiveOps, MoeParallelConfig, NoOpCollective, ParallelConfig, ParallelStrategy,
};
pub use crate::core::pipeline::InferencePipeline;
pub use crate::core::quantization::{
    GgufError, Int8QuantizedKv, KvCacheDtype, QuantMethod, QuantizationConfig,
};
pub use crate::core::security::{
    is_strict_security, validate_external_script, validate_plugin, validate_runner,
};
pub use crate::core::telemetry::MemoryTelemetry;
#[cfg(feature = "candle-engine")]
pub use crate::executor::batch_executor::{BatchableModel, CandleBatchExecutor, TokenBudget};
pub use crate::executor::coreml::CoreMlEngine;
pub use crate::executor::intel_npu::IntelNpuEngine;
pub use crate::executor::mlx::MlxEngine;
pub use crate::executor::npu_tts::NpuTtsEngine;
pub use crate::executor::onnx::OnnxRuntimeEngine;
pub use crate::executor::vulkan::VulkanEngine;

pub use crate::batching::{BatchExecutorConfig, build_batch_executor};
#[allow(deprecated)]
pub use crate::executor::speculative::{
    DraftModelStrategy, NGramStrategy, SpeculativeMode, SpeculativeResult, SpeculativeStrategy,
    speculative_mode_is_mtp, verify_greedy_tokens, verify_speculative_tokens,
    verify_with_rejection_sampling,
};
pub use crate::plugin::{PluginEntryPoint, PluginManager, PluginManifest, PluginMetadata};
pub use crate::processor::{
    AudioProcessor, AudioProcessorConfig, IdentityProcessor, ImageProcessor, ImageProcessorConfig,
    Processor, ProcessorRegistry, TokenizerProcessor,
};
#[cfg(feature = "candle-engine")]
pub use crate::scheduler::kv_hook::KvHook;
#[cfg(feature = "candle-engine")]
pub use crate::scheduler::paged_cache::{BlockKvData, PagedAttentionCache, PagedCacheConfig};
pub use crate::scheduler::policy::LongContextPolicy;
pub use crate::scheduler::{
    BatchResult, BloomKvCachePool, BloomScheduler, EngineExecutor, EnvironmentConstraints,
    ExecutionBatch, ExecutionGuard, ExecutionPhase, FairnessStrategy, InferenceScheduler,
    KvCacheAllocation, KvCacheMetrics, KvCachePool, ModelRoute, ModelSwitchReason, Request,
    RequestClass, RequestState, ScheduledSegment, Scheduler, SegmentResult,
};
pub use crate::world::{
    ActionSchema, MockPolicyEngine, MockWorldModel, PolicyEngine, StateCacheManager,
    WorldModelConstraints, WorldModelEngine, WorldModelLoop, WorldStateSchema,
};
