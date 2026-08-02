#![allow(
    clippy::collapsible_if,
    clippy::derivable_impls,
    clippy::double_ended_iterator_last,
    clippy::field_reassign_with_default,
    clippy::for_kv_map,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::manual_repeat_n,
    clippy::missing_const_for_thread_local,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_character_iteration,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::needless_update,
    clippy::needless_question_mark,
    clippy::new_without_default,
    clippy::redundant_field_names,
    clippy::repeat_vec_with_capacity,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// Surface new `unwrap` usage in production code. Tests may use unwrap to keep
// assertions concise; runtime paths must recover or return a structured error.
#![cfg_attr(not(test), warn(clippy::unwrap_used))]
//! Model abstraction for multimodal inference.

pub mod cachemesh;
pub mod core;
pub mod executor;
pub mod plugin;
pub mod processor;
pub mod scheduler;
pub mod world;

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
pub use crate::core::config::{
    default_config_dir, default_config_path, load_config, resolve_config_path,
    write_default_config, BenchConfig, BloomConfig, InferConfig, ServerConfig,
};
pub use crate::core::engine::{
    default_engine_supports, device_kind_from_capability, BackendMaturity, Engine,
    EngineCapability, EngineRegistry, EngineRouter, RoutingDecision, SupportLevel,
};
pub use crate::core::io::{
    DataBlock, InferenceParams, InferenceRequest, ModelInput, ModelOutput, OutputChunk,
};
pub use crate::core::manifest::{
    estimate_memory, estimate_memory_for_device, format_bytes, infer_quantization, load_manifest,
    model_manifest_supports_embeddings, model_manifest_tasks, resolve_hf_safetensors_files,
    MemoryEstimate,
};
pub use crate::core::memory::{
    available_system_memory, default_memory_utilization, plan_memory_preallocation,
    reserve_memory_for_plan, MemoryPreallocationConfig, MemoryPreallocationPlan, MemoryReservation,
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

#[allow(deprecated)]
pub use crate::executor::speculative::{
    speculative_mode_is_mtp, verify_greedy_tokens, verify_speculative_tokens,
    verify_with_rejection_sampling, DraftModelStrategy, NGramStrategy, SpeculativeMode,
    SpeculativeResult, SpeculativeStrategy,
};
pub use crate::plugin::{PluginEntryPoint, PluginManager, PluginManifest, PluginMetadata};
pub use crate::processor::{
    AudioProcessor, AudioProcessorConfig, IdentityProcessor, ImageProcessor, ImageProcessorConfig,
    Processor, ProcessorRegistry, TokenizerProcessor,
};
pub use crate::scheduler::kv_hook::KvHook;
pub use crate::scheduler::paged_cache::{
    BlockKvData, LongContextPolicy, PagedAttentionCache, PagedCacheConfig,
};
pub use crate::scheduler::{
    BatchResult, BloomKvCachePool, BloomScheduler, EngineExecutor, EnvironmentConstraints,
    ExecutionBatch, ExecutionPhase, FairnessStrategy, InferenceScheduler, KvCacheAllocation,
    KvCacheMetrics, KvCachePool, ModelRoute, ModelSwitchReason, Request, RequestClass,
    RequestState, ScheduledSegment, Scheduler, SegmentResult,
};
pub use crate::world::{
    ActionSchema, MockPolicyEngine, MockWorldModel, PolicyEngine, StateCacheManager,
    WorldModelConstraints, WorldModelEngine, WorldModelLoop, WorldStateSchema,
};
