use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{
    BloomError, CacheHandle, CacheKind, DeviceCapability, GenerationParams, ResourceCoordinator,
    ResourceError, ResourcePriority, ResourceTicket, TokenAdmission, TokenPhase,
    TokenSchedulingConfig,
};

/// Scheduling granularity used by the core dispatch layer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SchedulingLevel {
    /// Token-level scheduling for LLM prefill/decode loops.
    Token,
    /// Request-level scheduling for whole-call workloads such as ASR, vision, or tools.
    Request,
    /// Segment-level scheduling for long sequence chunks or pipeline-parallel work.
    Segment,
}

/// Workload category for routing, admission, and observability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WorkloadKind {
    LanguageModel,
    AudioTranscription,
    Vision,
    Diffusion,
    Embedding,
    Custom(String),
}

/// Execution phase currently being admitted.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
    Encode,
    Process,
    Segment,
}

impl ExecutionPhase {
    fn token_phase(self) -> TokenPhase {
        match self {
            Self::Decode => TokenPhase::Decode,
            Self::Prefill | Self::Encode | Self::Process | Self::Segment => TokenPhase::Prefill,
        }
    }
}

/// Request class controls queue ordering independent from model residency priority.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RequestClass {
    RealtimeStream,
    ForegroundInteractive,
    BackgroundBatch,
    Speculative,
}

impl RequestClass {
    fn rank(self) -> u8 {
        match self {
            Self::RealtimeStream => 0,
            Self::ForegroundInteractive => 1,
            Self::BackgroundBatch => 2,
            Self::Speculative => 3,
        }
    }
}

/// Runtime scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchedulerConfig {
    pub max_concurrent_requests: usize,
    pub token_scheduling: TokenSchedulingConfig,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let token_scheduling = TokenSchedulingConfig::default();
        Self {
            max_concurrent_requests: token_scheduling.budget().concurrent_segments,
            token_scheduling,
        }
    }
}

/// A generic invocation submitted to Bloom's core scheduler.
#[derive(Debug, Clone)]
pub struct InvocationRequest {
    pub request_id: String,
    pub model_id: String,
    pub workload: WorkloadKind,
    pub level: SchedulingLevel,
    pub phase: ExecutionPhase,
    pub class: RequestClass,
    pub priority: ResourcePriority,
    pub ticket: ResourceTicket,
    /// Estimated input tokens for token-level admission.
    pub input_tokens: usize,
    /// Estimated output tokens for token/cache admission.
    pub output_tokens: usize,
    /// Optional per-request cache budget. This is released on completion.
    pub cache_bytes: usize,
    pub cache_kind: CacheKind,
}

impl InvocationRequest {
    pub fn token_cost(&self) -> usize {
        match self.phase {
            ExecutionPhase::Decode => self.output_tokens.max(1),
            _ => self.input_tokens.max(1),
        }
    }

    pub fn for_language_model(
        request_id: impl Into<String>,
        model_id: impl Into<String>,
        ticket: ResourceTicket,
        input_tokens: usize,
        output_tokens: usize,
    ) -> Self {
        let model_id = model_id.into();
        Self {
            request_id: request_id.into(),
            model_id,
            workload: WorkloadKind::LanguageModel,
            level: SchedulingLevel::Token,
            phase: ExecutionPhase::Prefill,
            class: RequestClass::ForegroundInteractive,
            priority: ResourcePriority::Critical,
            ticket,
            input_tokens,
            output_tokens,
            cache_bytes: 0,
            cache_kind: CacheKind::KvCache,
        }
    }
}

/// A request admitted by the scheduler for execution.
#[derive(Debug, Clone)]
pub struct ScheduledInvocation {
    pub request_id: String,
    pub model_id: String,
    pub workload: WorkloadKind,
    pub level: SchedulingLevel,
    pub phase: ExecutionPhase,
    pub class: RequestClass,
    pub token_cost: usize,
    pub ticket: ResourceTicket,
    pub cache_handle: Option<CacheHandle>,
    pub queued_at: Instant,
    pub started_at: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvocationResult {
    Success,
    Failed { reason: String },
    Cancelled,
}

#[derive(Debug, Clone)]
struct QueuedInvocation {
    seq: u64,
    request: InvocationRequest,
    cache_handle: Option<CacheHandle>,
    queued_at: Instant,
}

#[derive(Debug, Clone)]
struct ActiveInvocation {
    scheduled: ScheduledInvocation,
}

#[derive(Default)]
struct SchedulerState {
    pending: VecDeque<QueuedInvocation>,
    active: HashMap<String, ActiveInvocation>,
    completed: HashMap<String, InvocationResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    pub pending_count: usize,
    pub active_count: usize,
    pub completed_count: usize,
    pub max_concurrent_requests: usize,
}

/// Core dispatch scheduler. It owns queue ordering and request-level resource
/// handles while concrete engines remain behind adapter code.
pub struct CoreScheduler {
    config: SchedulerConfig,
    resources: Arc<ResourceCoordinator>,
    state: Mutex<SchedulerState>,
    next_sequence: AtomicU64,
    next_cache_id: AtomicU64,
}

impl CoreScheduler {
    pub fn new(config: SchedulerConfig, resources: Arc<ResourceCoordinator>) -> Self {
        Self {
            config,
            resources,
            state: Mutex::new(SchedulerState::default()),
            next_sequence: AtomicU64::new(1),
            next_cache_id: AtomicU64::new(1),
        }
    }

    pub fn submit(&self, request: InvocationRequest) -> Result<(), BloomError> {
        let cache_handle = if request.cache_bytes > 0 {
            let handle = CacheHandle {
                handle_id: self.next_cache_id.fetch_add(1, Ordering::SeqCst),
                model_id: request.model_id.clone(),
                cache_kind: request.cache_kind,
                bytes: request.cache_bytes,
                priority: request.priority,
            };
            self.resources
                .register_cache(handle.clone())
                .map_err(BloomError::Resource)?;
            Some(handle)
        } else {
            None
        };

        let mut state = self.state.lock().unwrap();
        if state.active.contains_key(&request.request_id)
            || state
                .pending
                .iter()
                .any(|queued| queued.request.request_id == request.request_id)
        {
            if let Some(handle) = cache_handle {
                self.resources.release_cache(handle.handle_id);
            }
            return Err(BloomError::SchedulingFailed(format!(
                "request '{}' is already scheduled",
                request.request_id
            )));
        }

        state.pending.push_back(QueuedInvocation {
            seq: self.next_sequence.fetch_add(1, Ordering::SeqCst),
            request,
            cache_handle,
            queued_at: Instant::now(),
        });
        Ok(())
    }

    pub fn next_ready(
        &self,
        capabilities: &[DeviceCapability],
    ) -> Result<Vec<ScheduledInvocation>, BloomError> {
        let mut state = self.state.lock().unwrap();
        let capacity = self
            .config
            .max_concurrent_requests
            .saturating_sub(state.active.len());
        if capacity == 0 || state.pending.is_empty() {
            return Ok(Vec::new());
        }

        let mut admission = TokenAdmission::default();
        let mut admitted = Vec::new();

        while admitted.len() < capacity {
            let Some(index) = self.best_pending_index(&state.pending, capabilities, &mut admission)
            else {
                break;
            };

            let queued = state.pending.remove(index).expect("pending index valid");
            let scheduled = ScheduledInvocation {
                request_id: queued.request.request_id.clone(),
                model_id: queued.request.model_id.clone(),
                workload: queued.request.workload.clone(),
                level: queued.request.level,
                phase: queued.request.phase,
                class: queued.request.class,
                token_cost: queued.request.token_cost(),
                ticket: queued.request.ticket.clone(),
                cache_handle: queued.cache_handle,
                queued_at: queued.queued_at,
                started_at: Instant::now(),
            };
            state.active.insert(
                scheduled.request_id.clone(),
                ActiveInvocation {
                    scheduled: scheduled.clone(),
                },
            );
            admitted.push(scheduled);
        }

        Ok(admitted)
    }

    pub fn complete(&self, request_id: &str, result: InvocationResult) -> Result<(), BloomError> {
        let mut state = self.state.lock().unwrap();
        let active = state.active.remove(request_id).ok_or_else(|| {
            BloomError::SchedulingFailed(format!("request '{request_id}' is not active"))
        })?;

        if let Some(handle) = active.scheduled.cache_handle {
            self.resources.release_cache(handle.handle_id);
        }
        state.completed.insert(request_id.to_string(), result);
        Ok(())
    }

    pub fn snapshot(&self) -> SchedulerSnapshot {
        let state = self.state.lock().unwrap();
        SchedulerSnapshot {
            pending_count: state.pending.len(),
            active_count: state.active.len(),
            completed_count: state.completed.len(),
            max_concurrent_requests: self.config.max_concurrent_requests,
        }
    }

    fn best_pending_index(
        &self,
        pending: &VecDeque<QueuedInvocation>,
        capabilities: &[DeviceCapability],
        admission: &mut TokenAdmission,
    ) -> Option<usize> {
        let mut candidates = pending
            .iter()
            .enumerate()
            .filter(|(_, queued)| self.backend_is_eligible(&queued.request, capabilities))
            .collect::<Vec<_>>();

        candidates.sort_by(|(_, a), (_, b)| {
            a.request
                .class
                .rank()
                .cmp(&b.request.class.rank())
                .then_with(|| b.request.priority.cmp(&a.request.priority))
                .then_with(|| a.seq.cmp(&b.seq))
        });

        for (index, queued) in candidates {
            if admission.try_reserve(
                &self.config.token_scheduling,
                queued.request.phase.token_phase(),
                queued.request.token_cost(),
            ) {
                return Some(index);
            }
        }
        None
    }

    fn backend_is_eligible(
        &self,
        request: &InvocationRequest,
        capabilities: &[DeviceCapability],
    ) -> bool {
        let Some(preferred) = request.ticket.preferred_backend.as_deref() else {
            return true;
        };

        capabilities.is_empty()
            || capabilities
                .iter()
                .any(|capability| capability.backend_name == preferred)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelRequest {
    pub request_id: String,
    pub model: Option<String>,
    pub messages: Vec<LanguageModelMessage>,
    pub params: GenerationParams,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelChunk {
    pub content: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanguageModelResponse {
    pub request_id: String,
    pub model_id: String,
    pub content: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Adapter trait for request-level language-model calls. Implementors can use
/// a local pipeline, a remote endpoint, or any other engine behind this stable
/// core interface.
pub trait LanguageModelInvoker: Send + Sync {
    fn invoke_language_model(
        &self,
        request: LanguageModelRequest,
        on_chunk: &mut dyn FnMut(LanguageModelChunk) -> Result<(), BloomError>,
    ) -> Result<LanguageModelResponse, BloomError>;
}

impl From<ResourceError> for BloomError {
    fn from(value: ResourceError) -> Self {
        Self::Resource(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryTopology;

    fn ticket(id: &str) -> ResourceTicket {
        ResourceTicket {
            model_id: id.to_string(),
            ram_bytes: 0,
            vram_bytes: 0,
            cache_bytes: 0,
            priority: ResourcePriority::Normal,
            strategy: crate::ResidencyStrategy::OnDemand,
            preferred_backend: Some("cpu".to_string()),
            fallback_backends: vec![],
        }
    }

    fn capability() -> DeviceCapability {
        DeviceCapability {
            backend_name: "cpu".to_string(),
            vendor: None,
            device_class: crate::DeviceClass::Cpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: 1024,
            available_memory: 1024,
            supported_dtypes: vec![],
            supported_formats: vec![],
            supports_mmap: true,
            has_quantization_kernels: false,
            supports_streaming: true,
            thermal_state: crate::ThermalState::Nominal,
            power_state: crate::PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        }
    }

    #[test]
    fn schedules_foreground_before_background() {
        let scheduler = CoreScheduler::new(
            SchedulerConfig {
                max_concurrent_requests: 1,
                ..Default::default()
            },
            Arc::new(ResourceCoordinator::new(
                1024,
                1024,
                MemoryTopology::Unified,
            )),
        );

        let mut background =
            InvocationRequest::for_language_model("background", "m", ticket("m"), 8, 8);
        background.class = RequestClass::BackgroundBatch;
        scheduler.submit(background).unwrap();

        let mut foreground =
            InvocationRequest::for_language_model("foreground", "m", ticket("m"), 8, 8);
        foreground.class = RequestClass::ForegroundInteractive;
        scheduler.submit(foreground).unwrap();

        let ready = scheduler.next_ready(&[capability()]).unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].request_id, "foreground");
    }

    #[test]
    fn releases_request_cache_on_completion() {
        let resources = Arc::new(ResourceCoordinator::new(
            1024,
            1024,
            MemoryTopology::Unified,
        ));
        let scheduler = CoreScheduler::new(SchedulerConfig::default(), Arc::clone(&resources));

        let mut request = InvocationRequest::for_language_model("r1", "m", ticket("m"), 8, 8);
        request.cache_bytes = 128;
        scheduler.submit(request).unwrap();
        assert_eq!(resources.snapshot().cache_count, 1);

        let ready = scheduler.next_ready(&[capability()]).unwrap();
        scheduler
            .complete(&ready[0].request_id, InvocationResult::Success)
            .unwrap();

        let snapshot = resources.snapshot();
        assert_eq!(snapshot.cache_count, 0);
        assert_eq!(snapshot.ram_allocated, 0);
    }
}
