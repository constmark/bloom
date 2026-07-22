//! Core abstractions for Bloom runtime.

pub mod config;
pub mod constants;
pub mod error;
pub mod kv_overlay;
pub mod manifest;
pub mod online_switching;
pub mod plugin;
pub mod resource;
pub mod runtime;
pub mod scheduler;
pub mod token_scheduling;
pub mod types;
pub mod unified_memory;
pub mod vram;
pub mod world;

pub use config::BloomConfig;
pub use error::{BloomError, ResourceError};
pub use kv_overlay::{KvOverlayConfig, KvOverlayEntry, KvOverlayManager, KvOverlayMetrics};
pub use manifest::{
    ModelFamily, ModelFile, ModelIoSchema, ModelManifest, ModelMemoryProfile, ProcessorKind,
    ProcessorSpec, RuntimeHints,
};
pub use online_switching::{
    ModelSwitchCandidate, OnlineSwitchingConfig, OnlineSwitchingPolicy, SwitchDecision,
    SwitchPolicy,
};
pub use plugin::{
    BackendPluginManifest, EnginePluginManifest, ModelPackageFile, ModelPackageManifest,
    OperatorBenchmark, OperatorPluginManifest, PluginDependency, PluginEntryPoint, PluginMetadata,
    ProcessorPluginManifest, QuantizationVariant,
};
pub use resource::{
    BackendLease, CacheHandle, ModelResidencyRecord, ModelResourceSnapshot, OffloadCallback,
    ResourceSnapshot, ResourceTicket,
};
pub use runtime::{ExecutionContext, Runtime};
pub use scheduler::{
    CoreScheduler, ExecutionPhase, InvocationRequest, InvocationResult, LanguageModelChunk,
    LanguageModelInvoker, LanguageModelMessage, LanguageModelRequest, LanguageModelResponse,
    RequestClass, ScheduledInvocation, SchedulerConfig, SchedulerSnapshot, SchedulingLevel,
    WorkloadKind,
};
pub use token_scheduling::{
    // Chunked prefill
    chunked_prefill::{
        ChunkedPrefillConfig, ChunkedPrefillQueue, ChunkedPrefillState, PrefillChunk,
    },
    // Preemption
    preemption::{
        PreemptibleRequest, PreemptionConfig, PreemptionDecision, PreemptionManager,
        PreemptionPolicy,
    },
    // Priority KV eviction
    priority_eviction::{
        AdmissionResult, EvictionDecision, KvEvictionConfig, KvEvictionManager, KvEvictionPolicy,
        KvSessionInfo,
    },
    // Rate limiter
    rate_limiter::{
        RateLimitDecision, RateLimiterConfig, TokenBucketConfig, TokenBucketRateLimiter,
    },
    TokenAdmission,
    TokenBudget,
    TokenPhase,
    TokenSchedulingConfig,
};
pub use types::{
    BenchmarkResult, CacheKind, DType, DeviceCapability, DeviceClass, DeviceKind, GenerationParams,
    MemoryTopology, Modality, ModelFormat, PowerState, QuantScheme, QuantizationInfo,
    ResidencyStrategy, ResourcePriority, ResponseFormat, TensorShape, ThermalState,
};
pub use unified_memory::{MemoryReservation, UnifiedMemoryConfig};
pub use vram::{
    global_resource_coordinator, global_vram_coordinator, init_global_resource_coordinator,
    ResourceCoordinator, VRAMCoordinator,
};
pub use world::{
    Action, PredictedFuture, StateCacheConfig, StateCacheEntry, StateCachePriority, StateDelta,
    WorldObservation, WorldState,
};
