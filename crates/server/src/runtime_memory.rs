use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
use std::process::Command;

use anyhow::{Result, anyhow, bail};
use bloomai_core::{DeviceKind, MemoryTopology, ModelManifest};
use bloomai_engine::{MemoryEstimate, estimate_memory_for_device};

use crate::cli::Args;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RuntimeMemoryFootprint {
    pub(crate) host_bytes: usize,
    pub(crate) device_bytes: usize,
}

impl RuntimeMemoryFootprint {
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            host_bytes: self.host_bytes.checked_add(other.host_bytes)?,
            device_bytes: self.device_bytes.checked_add(other.device_bytes)?,
        })
    }

    fn checked_sub(self, other: Self) -> Option<Self> {
        Some(Self {
            host_bytes: self.host_bytes.checked_sub(other.host_bytes)?,
            device_bytes: self.device_bytes.checked_sub(other.device_bytes)?,
        })
    }

    fn positive_delta_from(self, previous: Self) -> Self {
        Self {
            host_bytes: self.host_bytes.saturating_sub(previous.host_bytes),
            device_bytes: self.device_bytes.saturating_sub(previous.device_bytes),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuntimeMemoryBudgetSnapshot {
    pub(crate) host_limit_bytes: usize,
    pub(crate) host_used_bytes: usize,
    pub(crate) device_limit_bytes: usize,
    pub(crate) device_used_bytes: usize,
    pub(crate) generations: usize,
}

pub(crate) struct RuntimeMemoryCandidate {
    pub(crate) estimate: MemoryEstimate,
    pub(crate) footprint: RuntimeMemoryFootprint,
    pub(crate) available: RuntimeMemoryFootprint,
}

pub(crate) struct RuntimeMemoryCandidateEvaluation {
    pub(crate) snapshot: RuntimeMemoryBudgetSnapshot,
    pub(crate) verdict: Result<()>,
}

#[derive(Debug, Default)]
struct RuntimeMemoryBudgetState {
    used: RuntimeMemoryFootprint,
    generations: usize,
}

#[derive(Debug)]
struct RuntimeMemoryBudget {
    limit: RuntimeMemoryFootprint,
    state: Mutex<RuntimeMemoryBudgetState>,
}

impl RuntimeMemoryBudget {
    fn new(limit: RuntimeMemoryFootprint) -> Self {
        Self {
            limit,
            state: Mutex::new(RuntimeMemoryBudgetState::default()),
        }
    }

    fn reserve(
        self: &Arc<Self>,
        footprint: RuntimeMemoryFootprint,
        available: RuntimeMemoryFootprint,
    ) -> Result<RuntimeMemoryPermit> {
        ensure_dynamic_headroom(footprint, available)?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let next = state
            .used
            .checked_add(footprint)
            .ok_or_else(|| anyhow!("runtime memory accounting overflow"))?;
        ensure_within_limit(next, self.limit)?;
        let next_generations = state
            .generations
            .checked_add(1)
            .ok_or_else(|| anyhow!("runtime generation accounting overflow"))?;
        state.used = next;
        state.generations = next_generations;
        Ok(RuntimeMemoryPermit {
            inner: Arc::new(RuntimeMemoryPermitInner {
                budget: Arc::clone(self),
                footprint: Mutex::new(footprint),
            }),
        })
    }

    fn evaluate(
        &self,
        footprint: RuntimeMemoryFootprint,
        available: RuntimeMemoryFootprint,
    ) -> RuntimeMemoryCandidateEvaluation {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let snapshot = RuntimeMemoryBudgetSnapshot {
            host_limit_bytes: self.limit.host_bytes,
            host_used_bytes: state.used.host_bytes,
            device_limit_bytes: self.limit.device_bytes,
            device_used_bytes: state.used.device_bytes,
            generations: state.generations,
        };
        let verdict = ensure_dynamic_headroom(footprint, available).and_then(|()| {
            let next = state
                .used
                .checked_add(footprint)
                .ok_or_else(|| anyhow!("runtime memory accounting overflow"))?;
            ensure_within_limit(next, self.limit)
        });
        RuntimeMemoryCandidateEvaluation { snapshot, verdict }
    }

    fn resize(
        &self,
        previous: RuntimeMemoryFootprint,
        next: RuntimeMemoryFootprint,
        available: RuntimeMemoryFootprint,
    ) -> Result<()> {
        ensure_dynamic_headroom(next.positive_delta_from(previous), available)?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let without_previous = state
            .used
            .checked_sub(previous)
            .ok_or_else(|| anyhow!("runtime memory accounting underflow"))?;
        let resized = without_previous
            .checked_add(next)
            .ok_or_else(|| anyhow!("runtime memory accounting overflow"))?;
        ensure_within_limit(resized, self.limit)?;
        state.used = resized;
        Ok(())
    }

    fn release(&self, footprint: RuntimeMemoryFootprint) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let Some(next_used) = state.used.checked_sub(footprint) else {
            tracing::error!(
                ?footprint,
                used = ?state.used,
                "runtime memory accounting invariant violation while releasing a permit"
            );
            debug_assert!(false, "runtime memory accounting underflow");
            return;
        };
        let Some(next_generations) = state.generations.checked_sub(1) else {
            tracing::error!(
                ?footprint,
                "runtime generation accounting invariant violation while releasing a permit"
            );
            debug_assert!(false, "runtime generation accounting underflow");
            return;
        };
        state.used = next_used;
        state.generations = next_generations;
    }

    fn snapshot(&self) -> RuntimeMemoryBudgetSnapshot {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        RuntimeMemoryBudgetSnapshot {
            host_limit_bytes: self.limit.host_bytes,
            host_used_bytes: state.used.host_bytes,
            device_limit_bytes: self.limit.device_bytes,
            device_used_bytes: state.used.device_bytes,
            generations: state.generations,
        }
    }
}

struct RuntimeMemoryPermitInner {
    budget: Arc<RuntimeMemoryBudget>,
    footprint: Mutex<RuntimeMemoryFootprint>,
}

impl Drop for RuntimeMemoryPermitInner {
    fn drop(&mut self) {
        let footprint = *self
            .footprint
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        self.budget.release(footprint);
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeMemoryPermit {
    inner: Arc<RuntimeMemoryPermitInner>,
}

impl std::fmt::Debug for RuntimeMemoryPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeMemoryPermit")
            .field("footprint", &self.footprint())
            .finish_non_exhaustive()
    }
}

impl RuntimeMemoryPermit {
    pub(crate) fn resize(
        &self,
        next: RuntimeMemoryFootprint,
        available: RuntimeMemoryFootprint,
    ) -> Result<()> {
        let mut footprint = self
            .inner
            .footprint
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.inner.budget.resize(*footprint, next, available)?;
        *footprint = next;
        Ok(())
    }

    pub(crate) fn footprint(&self) -> RuntimeMemoryFootprint {
        *self
            .inner
            .footprint
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// Re-check the full physical footprint immediately before a heavyweight
    /// allocation. `resize` only needs dynamic headroom for a positive delta,
    /// but a newly admitted candidate has not materialized any of its charged
    /// bytes yet, so the complete footprint must still fit the latest probe.
    pub(crate) fn revalidate_available(&self, available: RuntimeMemoryFootprint) -> Result<()> {
        ensure_dynamic_headroom(self.footprint(), available)
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeMemoryPlanner {
    budget: Arc<RuntimeMemoryBudget>,
    device: DeviceKind,
    topology: MemoryTopology,
    context_size: usize,
    max_concurrent: usize,
    enable_ifb: bool,
    retain_ifb_kv_reservation: bool,
    cachemesh_l2_bytes: usize,
    memory_utilization: f64,
}

impl RuntimeMemoryPlanner {
    pub(crate) fn new(args: &Args, device: DeviceKind) -> Result<Self> {
        if matches!(
            args.speculative.trim().to_ascii_lowercase().as_str(),
            "draft" | "draft_model" | "draft-model"
        ) {
            bail!(
                "strict runtime memory admission does not yet support per-request draft-model loading; use --speculative=none, ngram, or mtp"
            );
        }
        let host = strict_host_memory_probe()?;
        let device_probe = strict_device_memory_probe(device)?;
        let topology = device_probe.topology;
        let host_limit = utilization_limit(host.total_bytes, args.memory_utilization)?;
        let device_limit = match topology {
            MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => 0,
            MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
                utilization_limit(device_probe.total_bytes, args.memory_utilization)?
            }
        };

        Ok(Self {
            budget: Arc::new(RuntimeMemoryBudget::new(RuntimeMemoryFootprint {
                host_bytes: host_limit,
                device_bytes: device_limit,
            })),
            device,
            topology,
            context_size: args.context_size,
            max_concurrent: args.max_concurrent.max(1),
            enable_ifb: args.enable_ifb,
            retain_ifb_kv_reservation: args.enable_ifb && !args.disable_memory_prealloc,
            cachemesh_l2_bytes: if args.enable_cachemesh {
                args.cachemesh_l2_capacity_bytes
            } else {
                0
            },
            memory_utilization: args.memory_utilization,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        host_limit_bytes: usize,
        device_limit_bytes: usize,
        topology: MemoryTopology,
    ) -> Self {
        let device_limit_bytes = match topology {
            MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => 0,
            MemoryTopology::Discrete | MemoryTopology::RemoteMemory => device_limit_bytes,
        };
        Self {
            budget: Arc::new(RuntimeMemoryBudget::new(RuntimeMemoryFootprint {
                host_bytes: host_limit_bytes,
                device_bytes: device_limit_bytes,
            })),
            device: DeviceKind::Cpu,
            topology,
            context_size: 128,
            max_concurrent: 1,
            enable_ifb: false,
            retain_ifb_kv_reservation: false,
            cachemesh_l2_bytes: 0,
            memory_utilization: 1.0,
        }
    }

    pub(crate) fn planned_estimate(&self, manifest: &ModelManifest) -> Result<MemoryEstimate> {
        let memory_context_size = self
            .context_size
            .checked_mul(self.max_concurrent)
            .ok_or_else(|| anyhow!("runtime context memory estimate overflow"))?;
        self.estimate_for_context(manifest, memory_context_size, self.device)
    }

    pub(crate) fn assess_planned(
        &self,
        manifest: &ModelManifest,
    ) -> Result<RuntimeMemoryCandidate> {
        let estimate = self.planned_estimate(manifest)?;
        let footprint = self.footprint(manifest, &estimate)?;
        let available = self.available()?;
        self.evaluate_candidate(footprint, available).verdict?;
        Ok(RuntimeMemoryCandidate {
            estimate,
            footprint,
            available,
        })
    }

    pub(crate) fn plan_for_context(
        &self,
        manifest: &ModelManifest,
        memory_context_size: usize,
        device: DeviceKind,
    ) -> Result<RuntimeMemoryCandidate> {
        let estimate = self.estimate_for_context(manifest, memory_context_size, device)?;
        let footprint = self.footprint(manifest, &estimate)?;
        let available = self.available()?;
        Ok(RuntimeMemoryCandidate {
            estimate,
            footprint,
            available,
        })
    }

    pub(crate) fn evaluate_candidate(
        &self,
        footprint: RuntimeMemoryFootprint,
        available: RuntimeMemoryFootprint,
    ) -> RuntimeMemoryCandidateEvaluation {
        self.budget.evaluate(footprint, available)
    }

    pub(crate) fn estimate_for_context(
        &self,
        manifest: &ModelManifest,
        memory_context_size: usize,
        device: DeviceKind,
    ) -> Result<MemoryEstimate> {
        if device != self.device {
            bail!(
                "loaded runtime changed device from {:?} to {:?}",
                self.device,
                device
            );
        }
        validate_estimate_inputs(manifest, memory_context_size)?;
        Ok(estimate_memory_for_device(
            manifest,
            memory_context_size,
            device,
        ))
    }

    pub(crate) fn footprint(
        &self,
        manifest: &ModelManifest,
        estimate: &MemoryEstimate,
    ) -> Result<RuntimeMemoryFootprint> {
        let weight_copies = if self.enable_ifb {
            self.max_concurrent
        } else {
            1
        };
        let retained_host_bytes = if self.retain_ifb_kv_reservation {
            estimate.kv_cache_bytes
        } else {
            0
        };
        // Candle's execution wrappers currently materialize their general KV
        // tensors as F32 unless a backend explicitly proves otherwise. The
        // manifest estimate may describe a narrower requested storage dtype,
        // so admission must normalize that copy to its physical F32 size.
        let strict_kv_bytes = strict_runtime_kv_bytes(manifest, estimate, self.device)?;
        let kv_f32_mirror_bytes = if self.enable_ifb {
            // IFB owns a distinct paged-cache mirror in addition to the KV
            // tensors held by the execution wrapper.
            strict_kv_bytes
        } else {
            0
        };
        conservative_footprint(
            manifest,
            estimate,
            ConservativeFootprintPolicy {
                topology: self.topology,
                weight_copies,
                strict_kv_bytes,
                retained_host_bytes,
                kv_f32_mirror_bytes,
                cachemesh_l2_bytes: self.cachemesh_l2_bytes,
            },
        )
    }

    pub(crate) fn available(&self) -> Result<RuntimeMemoryFootprint> {
        let host = strict_host_memory_probe()?;
        let host_bytes = utilization_limit(host.available_bytes, self.memory_utilization)?;
        let device_bytes = match self.topology {
            MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => 0,
            MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
                let probe = strict_device_memory_probe(self.device)?;
                if probe.topology != self.topology {
                    bail!("configured device memory topology changed during admission");
                }
                utilization_limit(probe.available_bytes, self.memory_utilization)?
            }
        };
        Ok(RuntimeMemoryFootprint {
            host_bytes,
            device_bytes,
        })
    }

    pub(crate) fn reserve(
        &self,
        footprint: RuntimeMemoryFootprint,
        available: RuntimeMemoryFootprint,
    ) -> Result<RuntimeMemoryPermit> {
        self.budget.reserve(footprint, available)
    }

    pub(crate) fn snapshot(&self) -> RuntimeMemoryBudgetSnapshot {
        self.budget.snapshot()
    }
}

#[derive(Debug, Clone, Copy)]
struct ConservativeFootprintPolicy {
    topology: MemoryTopology,
    weight_copies: usize,
    strict_kv_bytes: usize,
    retained_host_bytes: usize,
    kv_f32_mirror_bytes: usize,
    cachemesh_l2_bytes: usize,
}

fn conservative_footprint(
    manifest: &ModelManifest,
    estimate: &MemoryEstimate,
    policy: ConservativeFootprintPolicy,
) -> Result<RuntimeMemoryFootprint> {
    let weights = estimate
        .weight_bytes
        .checked_mul(policy.weight_copies)
        .ok_or_else(|| anyhow!("runtime weight memory estimate overflow"))?;
    let workspace = estimate
        .temp_tensor_bytes
        .checked_mul(policy.weight_copies)
        .ok_or_else(|| anyhow!("runtime workspace memory estimate overflow"))?;
    let runtime_bytes = weights
        .checked_add(policy.strict_kv_bytes)
        .and_then(|bytes| bytes.checked_add(workspace))
        .ok_or_else(|| anyhow!("runtime memory estimate overflow"))?;
    let recommended_host = manifest
        .memory_profile
        .recommended_ram_bytes
        .max(manifest.memory_profile.min_ram_bytes);
    let recommended_device = manifest
        .memory_profile
        .recommended_vram_bytes
        .max(manifest.memory_profile.min_vram_bytes);

    match policy.topology {
        MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
            Ok(RuntimeMemoryFootprint {
                host_bytes: runtime_bytes
                    .max(recommended_host)
                    .max(recommended_device)
                    .checked_add(policy.retained_host_bytes)
                    .ok_or_else(|| anyhow!("unified retained-memory estimate overflow"))?
                    .checked_add(policy.kv_f32_mirror_bytes)
                    .ok_or_else(|| anyhow!("unified KV mirror estimate overflow"))?
                    .checked_add(policy.cachemesh_l2_bytes)
                    .ok_or_else(|| anyhow!("unified runtime memory estimate overflow"))?,
                device_bytes: 0,
            })
        }
        MemoryTopology::Discrete | MemoryTopology::RemoteMemory => {
            // Until every backend exposes an exact per-layer placement plan,
            // charge each live execution wrapper in both domains. This is
            // deliberately conservative: partial offload can retain host
            // layers per wrapper, while full-device loading still needs a
            // complete host staging/mmap view during reload.
            let host_bytes = weights
                .max(recommended_host)
                .checked_add(policy.strict_kv_bytes)
                .and_then(|bytes| bytes.checked_add(workspace))
                .and_then(|bytes| bytes.checked_add(policy.retained_host_bytes))
                .and_then(|bytes| bytes.checked_add(policy.kv_f32_mirror_bytes))
                .and_then(|bytes| bytes.checked_add(policy.cachemesh_l2_bytes))
                .ok_or_else(|| anyhow!("host runtime memory estimate overflow"))?;
            Ok(RuntimeMemoryFootprint {
                host_bytes,
                device_bytes: runtime_bytes.max(recommended_device),
            })
        }
    }
}

fn strict_f32_kv_bytes(estimate: &MemoryEstimate) -> Result<usize> {
    if estimate.kv_cache_bytes == 0 {
        return Ok(0);
    }
    let multiplier = match estimate.kv_cache_dtype {
        bloomai_core::DType::F32 => 1,
        bloomai_core::DType::F16 | bloomai_core::DType::BF16 => 2,
        bloomai_core::DType::I8 | bloomai_core::DType::U8 | bloomai_core::DType::Q8 => 4,
        // The engine currently accounts the 4-bit variants as one byte per
        // element until packed KV kernels land, so their F32 expansion is 4x.
        bloomai_core::DType::I4 | bloomai_core::DType::NF4 | bloomai_core::DType::Q4 => 4,
        bloomai_core::DType::Unknown => {
            bail!("KV-cache dtype is unknown; strict F32 memory cannot be planned")
        }
    };
    estimate
        .kv_cache_bytes
        .checked_mul(multiplier)
        .ok_or_else(|| anyhow!("strict F32 KV-cache estimate overflow"))
}

fn strict_runtime_kv_bytes(
    manifest: &ModelManifest,
    estimate: &MemoryEstimate,
    device: DeviceKind,
) -> Result<usize> {
    let is_qwen3_vl = manifest
        .parameters
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type.eq_ignore_ascii_case("qwen3_vl"));
    if device == DeviceKind::Gpu && is_qwen3_vl {
        // Qwen3-VL constructs its K/V tensors on the same verified Candle GPU
        // device and F16 dtype as the loaded model. Do not turn that known
        // physical layout into a false 2x admission rejection.
        return match estimate.kv_cache_dtype {
            bloomai_core::DType::F16 | bloomai_core::DType::BF16 => Ok(estimate.kv_cache_bytes),
            bloomai_core::DType::F32 => Ok(estimate.kv_cache_bytes),
            other => {
                bail!("Qwen3-VL GPU KV-cache dtype {other:?} is not a verified physical layout")
            }
        };
    }
    strict_f32_kv_bytes(estimate)
}

fn ensure_dynamic_headroom(
    requested: RuntimeMemoryFootprint,
    available: RuntimeMemoryFootprint,
) -> Result<()> {
    if requested.host_bytes > available.host_bytes {
        bail!(
            "runtime host memory headroom is insufficient: requires {} bytes but {} bytes are available",
            requested.host_bytes,
            available.host_bytes
        );
    }
    if requested.device_bytes > available.device_bytes {
        bail!(
            "runtime device memory headroom is insufficient: requires {} bytes but {} bytes are available",
            requested.device_bytes,
            available.device_bytes
        );
    }
    Ok(())
}

fn ensure_within_limit(
    requested: RuntimeMemoryFootprint,
    limit: RuntimeMemoryFootprint,
) -> Result<()> {
    if requested.host_bytes > limit.host_bytes {
        bail!(
            "runtime host memory budget is exhausted: requested aggregate {} bytes exceeds {} bytes",
            requested.host_bytes,
            limit.host_bytes
        );
    }
    if requested.device_bytes > limit.device_bytes {
        bail!(
            "runtime device memory budget is exhausted: requested aggregate {} bytes exceeds {} bytes",
            requested.device_bytes,
            limit.device_bytes
        );
    }
    Ok(())
}

fn utilization_limit(bytes: usize, utilization: f64) -> Result<usize> {
    if !utilization.is_finite() || !(0.0..=1.0).contains(&utilization) {
        bail!("memory utilization must be between 0 and 1");
    }
    Ok(((bytes as f64) * utilization).floor().max(0.0) as usize)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictMemoryProbe {
    topology: MemoryTopology,
    total_bytes: usize,
    available_bytes: usize,
}

fn validate_probe(probe: StrictMemoryProbe, domain: &str) -> Result<StrictMemoryProbe> {
    if probe.total_bytes == 0 || probe.available_bytes == 0 {
        bail!("{domain} memory capacity could not be detected");
    }
    if probe.available_bytes > probe.total_bytes {
        bail!("{domain} memory probe is inconsistent: available bytes exceed total bytes");
    }
    Ok(probe)
}

fn strict_device_memory_probe(device: DeviceKind) -> Result<StrictMemoryProbe> {
    match device {
        DeviceKind::Cpu => strict_host_memory_probe(),
        DeviceKind::Npu => {
            bail!(
                "strict NPU memory admission requires a backend-verified physical placement probe"
            )
        }
        DeviceKind::Gpu => strict_gpu_memory_probe(),
    }
}

#[cfg(target_os = "linux")]
fn strict_host_memory_probe() -> Result<StrictMemoryProbe> {
    let content = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| anyhow!("failed to read host memory telemetry: {error}"))?;
    let parse = |key: &str| -> Result<usize> {
        let kib = content
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name == key)
                    .then(|| value.split_whitespace().next())
                    .flatten()
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .ok_or_else(|| anyhow!("host memory telemetry is missing {key}"))?;
        kib.checked_mul(1024)
            .ok_or_else(|| anyhow!("host memory telemetry overflow"))
    };
    let mut probe = StrictMemoryProbe {
        topology: MemoryTopology::Unified,
        total_bytes: parse("MemTotal")?,
        available_bytes: parse("MemAvailable")?,
    };
    if let Some((limit, current)) = strict_linux_cgroup_memory()? {
        let cgroup_available = limit.checked_sub(current).ok_or_else(|| {
            anyhow!("cgroup memory telemetry is inconsistent: usage exceeds the limit")
        })?;
        probe.total_bytes = probe.total_bytes.min(limit);
        probe.available_bytes = probe.available_bytes.min(cgroup_available);
    }
    validate_probe(probe, "host")
}

#[cfg(target_os = "linux")]
fn strict_linux_cgroup_memory() -> Result<Option<(usize, usize)>> {
    let membership = std::fs::read_to_string("/proc/self/cgroup")
        .map_err(|error| anyhow!("failed to read cgroup membership: {error}"))?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo")
        .map_err(|error| anyhow!("failed to read cgroup mount information: {error}"))?;
    strict_linux_cgroup_memory_from(&membership, &mountinfo)
}

#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CgroupKind {
    V1Memory,
    V2,
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug)]
struct CgroupMount {
    kind: CgroupKind,
    root: std::path::PathBuf,
    mount_point: std::path::PathBuf,
}

#[cfg(any(target_os = "linux", test))]
fn strict_linux_cgroup_memory_from(
    membership: &str,
    mountinfo: &str,
) -> Result<Option<(usize, usize)>> {
    let (v2_membership, v1_memory_membership) = parse_cgroup_membership(membership)?;
    let mounts = parse_cgroup_mounts(mountinfo)?;

    if let Some(relative) = v2_membership {
        let roots = mapped_cgroup_roots(&mounts, CgroupKind::V2, &relative)?;
        return read_cgroup_constraints(&roots, "memory.max", "memory.current");
    }
    if let Some(relative) = v1_memory_membership {
        let roots = mapped_cgroup_roots(&mounts, CgroupKind::V1Memory, &relative)?;
        return read_cgroup_constraints(&roots, "memory.limit_in_bytes", "memory.usage_in_bytes");
    }
    Ok(None)
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_membership(
    membership: &str,
) -> Result<(Option<std::path::PathBuf>, Option<std::path::PathBuf>)> {
    let mut v2 = None;
    let mut v1_memory = None;
    for line in membership.lines().filter(|line| !line.trim().is_empty()) {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields
            .next()
            .ok_or_else(|| anyhow!("cgroup membership entry is invalid"))?;
        let controllers = fields
            .next()
            .ok_or_else(|| anyhow!("cgroup membership entry is invalid"))?;
        let path = fields
            .next()
            .ok_or_else(|| anyhow!("cgroup membership entry is invalid"))?;

        if hierarchy == "0" && controllers.is_empty() {
            if v2.is_some() {
                bail!("cgroup membership contains multiple v2 entries");
            }
            v2 = Some(safe_cgroup_path(path)?);
        } else if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            if v1_memory.is_some() {
                bail!("cgroup membership contains multiple v1 memory entries");
            }
            v1_memory = Some(safe_cgroup_path(path)?);
        }
    }
    Ok((v2, v1_memory))
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_mounts(mountinfo: &str) -> Result<Vec<CgroupMount>> {
    let mut mounts = Vec::new();
    for line in mountinfo.lines().filter(|line| !line.trim().is_empty()) {
        let Some((before_separator, after_separator)) = line.split_once(" - ") else {
            continue;
        };
        let after = after_separator.split_whitespace().collect::<Vec<_>>();
        let Some(file_system_type) = after.first().copied() else {
            continue;
        };
        if file_system_type != "cgroup" && file_system_type != "cgroup2" {
            continue;
        }
        if after.len() < 3 {
            bail!("cgroup mount information is incomplete");
        }
        let before = before_separator.split_whitespace().collect::<Vec<_>>();
        if before.len() < 6 {
            bail!("cgroup mount information is incomplete");
        }

        let kind = if file_system_type == "cgroup2" {
            CgroupKind::V2
        } else if after[2].split(',').any(|controller| controller == "memory") {
            CgroupKind::V1Memory
        } else {
            continue;
        };
        let decoded_root = decode_mountinfo_path(before[3])?;
        let decoded_mount_point = decode_mountinfo_path(before[4])?;
        mounts.push(CgroupMount {
            kind,
            root: safe_cgroup_path(
                decoded_root
                    .to_str()
                    .ok_or_else(|| anyhow!("cgroup mount root is not valid UTF-8"))?,
            )?,
            mount_point: safe_mount_point(decoded_mount_point)?,
        });
    }
    Ok(mounts)
}

#[cfg(any(target_os = "linux", test))]
fn decode_mountinfo_path(value: &str) -> Result<std::path::PathBuf> {
    let mut decoded = String::with_capacity(value.len());
    let mut characters = value.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            decoded.push(character);
            continue;
        }
        let escape = [characters.next(), characters.next(), characters.next()];
        match escape {
            [Some('0'), Some('4'), Some('0')] => decoded.push(' '),
            [Some('0'), Some('1'), Some('1')] => decoded.push('\t'),
            [Some('0'), Some('1'), Some('2')] => decoded.push('\n'),
            [Some('1'), Some('3'), Some('4')] => decoded.push('\\'),
            _ => bail!("cgroup mount information contains an invalid path escape"),
        }
    }
    Ok(std::path::PathBuf::from(decoded))
}

#[cfg(any(target_os = "linux", test))]
fn safe_mount_point(path: std::path::PathBuf) -> Result<std::path::PathBuf> {
    use std::path::Component;

    if !path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::RootDir | Component::Normal(_)))
    {
        bail!("cgroup mount point contains an unsafe path");
    }
    Ok(path)
}

#[cfg(any(target_os = "linux", test))]
fn mapped_cgroup_roots(
    mounts: &[CgroupMount],
    kind: CgroupKind,
    membership: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>> {
    let mount = mounts
        .iter()
        .filter(|mount| mount.kind == kind && membership.starts_with(&mount.root))
        .max_by_key(|mount| mount.root.components().count())
        .ok_or_else(|| anyhow!("cgroup membership has no matching mounted hierarchy"))?;
    let visible_membership = membership
        .strip_prefix(&mount.root)
        .map_err(|_| anyhow!("cgroup membership is outside its mounted hierarchy"))?;
    Ok(cgroup_candidate_roots(
        &mount.mount_point,
        Some(visible_membership),
    ))
}

#[cfg(any(target_os = "linux", test))]
fn safe_cgroup_path(path: &str) -> Result<std::path::PathBuf> {
    use std::path::Component;

    let absolute = std::path::Path::new(path);
    if !absolute.is_absolute() {
        bail!("cgroup hierarchy contains a non-absolute path");
    }
    let mut relative = std::path::PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(component) => relative.push(component),
            _ => bail!("cgroup hierarchy contains an unsafe path"),
        }
    }
    Ok(relative)
}

#[cfg(any(target_os = "linux", test))]
fn cgroup_candidate_roots(
    base: &std::path::Path,
    relative: Option<&std::path::Path>,
) -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    let mut current = relative.map_or_else(|| base.to_path_buf(), |path| base.join(path));
    loop {
        roots.push(current.clone());
        if current == base || !current.pop() || !current.starts_with(base) {
            break;
        }
    }
    if roots.last().is_none_or(|root| root != base) {
        roots.push(base.to_path_buf());
    }
    roots
}

#[cfg(any(target_os = "linux", test))]
fn read_cgroup_constraints(
    roots: &[std::path::PathBuf],
    limit_name: &str,
    current_name: &str,
) -> Result<Option<(usize, usize)>> {
    let mut effective_limit: Option<usize> = None;
    let mut effective_available: Option<usize> = None;
    for root in roots {
        let (limit, current) = read_cgroup_pair(root, limit_name, current_name)?;
        let Some(limit) = limit else {
            continue;
        };
        let available = limit.checked_sub(current).ok_or_else(|| {
            anyhow!("cgroup memory telemetry is inconsistent: usage exceeds the limit")
        })?;
        effective_limit = Some(effective_limit.map_or(limit, |value| value.min(limit)));
        effective_available =
            Some(effective_available.map_or(available, |value| value.min(available)));
    }
    match (effective_limit, effective_available) {
        (Some(limit), Some(available)) => Ok(Some((limit, limit - available.min(limit)))),
        (None, None) => Ok(None),
        _ => unreachable!("cgroup constraints are updated together"),
    }
}

#[cfg(any(target_os = "linux", test))]
fn read_cgroup_pair(
    root: &std::path::Path,
    limit_name: &str,
    current_name: &str,
) -> Result<(Option<usize>, usize)> {
    let limit_path = root.join(limit_name);
    let current_path = root.join(current_name);
    let limit_text = std::fs::read_to_string(&limit_path).map_err(|error| {
        anyhow!(
            "failed to read cgroup memory limit telemetry at {}: {error}",
            limit_path.display()
        )
    })?;
    let current_text = std::fs::read_to_string(&current_path).map_err(|error| {
        anyhow!(
            "failed to read cgroup memory usage telemetry at {}: {error}",
            current_path.display()
        )
    })?;
    let current = current_text
        .trim()
        .parse::<u64>()
        .map_err(|_| anyhow!("cgroup memory usage is invalid"))?;
    let current =
        usize::try_from(current).map_err(|_| anyhow!("cgroup memory usage is too large"))?;
    if limit_text.trim() == "max" {
        return Ok((None, current));
    }
    let limit = limit_text
        .trim()
        .parse::<u64>()
        .map_err(|_| anyhow!("cgroup memory limit is invalid"))?;
    Ok((
        Some(usize::try_from(limit).map_err(|_| anyhow!("cgroup memory limit is too large"))?),
        current,
    ))
}

#[cfg(target_os = "macos")]
fn strict_host_memory_probe() -> Result<StrictMemoryProbe> {
    let total_output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .map_err(|error| anyhow!("failed to query total host memory: {error}"))?;
    if !total_output.status.success() {
        bail!("total host memory probe failed");
    }
    let total_bytes = String::from_utf8(total_output.stdout)
        .map_err(|_| anyhow!("total host memory probe returned non-UTF-8 output"))?
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("total host memory probe returned an invalid byte count"))?;

    let available_output = Command::new("vm_stat")
        .output()
        .map_err(|error| anyhow!("failed to query available host memory: {error}"))?;
    if !available_output.status.success() {
        bail!("available host memory probe failed");
    }
    let available_text = String::from_utf8(available_output.stdout)
        .map_err(|_| anyhow!("available host memory probe returned non-UTF-8 output"))?;
    let available_bytes = parse_macos_vm_stat_available(&available_text)
        .ok_or_else(|| anyhow!("available host memory probe returned incomplete statistics"))?;
    validate_probe(
        StrictMemoryProbe {
            topology: MemoryTopology::Unified,
            total_bytes,
            available_bytes,
        },
        "host",
    )
}

#[cfg(target_os = "windows")]
fn strict_host_memory_probe() -> Result<StrictMemoryProbe> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: `status` points to a writable `MEMORYSTATUSEX` with the length
    // field initialized as required by `GlobalMemoryStatusEx`.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        bail!("Windows host memory telemetry probe failed");
    }
    validate_probe(
        StrictMemoryProbe {
            topology: MemoryTopology::Unified,
            total_bytes: usize::try_from(status.ullTotalPhys)
                .map_err(|_| anyhow!("Windows total host memory exceeds the platform size"))?,
            available_bytes: usize::try_from(status.ullAvailPhys)
                .map_err(|_| anyhow!("Windows available host memory exceeds the platform size"))?,
        },
        "host",
    )
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn strict_host_memory_probe() -> Result<StrictMemoryProbe> {
    bail!("strict host memory admission is unsupported on this operating system")
}

#[cfg(target_os = "macos")]
fn strict_gpu_memory_probe() -> Result<StrictMemoryProbe> {
    // Apple GPUs share the same physical memory pool as the CPU. Counting the
    // model in a second device ledger would double-charge UMA.
    strict_host_memory_probe()
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn strict_gpu_memory_probe() -> Result<StrictMemoryProbe> {
    use candle_core::cuda_backend::cudarc::driver::{CudaContext, result};

    let context = CudaContext::new(0)
        .map_err(|error| anyhow!("failed to initialize CUDA logical device 0: {error}"))?;
    context
        .bind_to_thread()
        .map_err(|error| anyhow!("failed to bind CUDA logical device 0: {error}"))?;
    let (available_bytes, total_bytes) = result::mem_get_info()
        .map_err(|error| anyhow!("failed to query CUDA logical device 0 memory: {error}"))?;
    validate_probe(
        StrictMemoryProbe {
            topology: MemoryTopology::Discrete,
            total_bytes,
            available_bytes,
        },
        "device",
    )
}

#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
fn strict_gpu_memory_probe() -> Result<StrictMemoryProbe> {
    bail!("strict CUDA memory admission requires a server built with the cuda feature")
}

#[cfg(any(target_os = "macos", test))]
fn parse_macos_vm_stat_available(text: &str) -> Option<usize> {
    let page_size = text.lines().find_map(|line| {
        line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|value| value.parse::<usize>().ok())
    })?;
    let parse_pages = |key: &str| {
        text.lines().find_map(|line| {
            line.trim()
                .strip_prefix(key)
                .and_then(|rest| rest.strip_prefix(':'))
                .map(|rest| rest.trim().trim_end_matches('.'))
                .and_then(|rest| rest.parse::<usize>().ok())
        })
    };
    let pages = parse_pages("Pages free")?.checked_add(parse_pages("Pages inactive")?)?;
    pages.checked_mul(page_size)
}

fn validate_estimate_inputs(manifest: &ModelManifest, context_size: usize) -> Result<()> {
    let file_bytes = manifest.files.iter().try_fold(0usize, |total, file| {
        if file.required && file.size_bytes == 0 {
            bail!("required model file has no plannable size");
        }
        total
            .checked_add(file.size_bytes)
            .ok_or_else(|| anyhow!("model file-size memory estimate overflow"))
    })?;
    let parameter = |names: &[&str], default: usize| -> Result<usize> {
        let value = names
            .iter()
            .find_map(|name| manifest.parameters.get(*name))
            .and_then(serde_json::Value::as_u64);
        value
            .map(usize::try_from)
            .transpose()
            .map_err(|_| anyhow!("model memory parameter exceeds the platform size limit"))
            .map(|value| value.unwrap_or(default))
    };
    let layers = parameter(&["num_hidden_layers", "num_layers"], 32)?;
    let hidden = parameter(&["hidden_size"], 4096)?;
    let kv_heads = parameter(&["num_key_value_heads", "num_kv_heads"], 8)?;
    let head_dim = parameter(&["head_dim"], 128)?;
    let intermediate = if let Some(value) = manifest
        .parameters
        .get("intermediate_size")
        .and_then(serde_json::Value::as_u64)
    {
        usize::try_from(value)
            .map_err(|_| anyhow!("model intermediate-size estimate exceeds the platform limit"))?
    } else {
        hidden
            .checked_mul(4)
            .ok_or_else(|| anyhow!("model intermediate-size memory estimate overflow"))?
    };
    let vocab = parameter(&["vocab_size"], 32_000)?;

    let has_explicit_shape = manifest
        .parameters
        .get("num_hidden_layers")
        .or_else(|| manifest.parameters.get("num_layers"))
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|value| value > 0)
        && manifest
            .parameters
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|value| value > 0);
    if !manifest.files.is_empty() && file_bytes == 0 {
        bail!("model manifest files do not contain a plannable weight size");
    }
    if manifest.files.is_empty()
        && manifest.memory_profile.min_ram_bytes == 0
        && !has_explicit_shape
    {
        bail!("model manifest does not contain a plannable weight size");
    }

    let attention = hidden
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| anyhow!("model attention memory estimate overflow"))?;
    let feed_forward = hidden
        .checked_mul(intermediate)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| anyhow!("model feed-forward memory estimate overflow"))?;
    layers
        .checked_mul(
            attention
                .checked_add(feed_forward)
                .ok_or_else(|| anyhow!("model layer memory estimate overflow"))?,
        )
        .and_then(|value| {
            vocab
                .checked_mul(hidden)
                .and_then(|embedding| embedding.checked_mul(2))
                .and_then(|embedding| value.checked_add(embedding))
        })
        .ok_or_else(|| anyhow!("model parameter memory estimate overflow"))?;
    let kv_per_token = if manifest.parameters.contains_key("num_hidden_layers")
        || manifest.parameters.contains_key("num_layers")
    {
        layers
            .checked_mul(kv_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| anyhow!("model KV-cache memory estimate overflow"))?
    } else {
        parameter(&["kv_cache_bytes_per_token"], 512 * 1024)?
    };
    context_size
        .checked_mul(kv_per_token)
        .ok_or_else(|| anyhow!("runtime KV-cache memory estimate overflow"))?;
    context_size
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| anyhow!("runtime workspace memory estimate overflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn available() -> RuntimeMemoryFootprint {
        RuntimeMemoryFootprint {
            host_bytes: usize::MAX,
            device_bytes: usize::MAX,
        }
    }

    #[test]
    fn permits_are_domain_aware_transactional_and_raii_released() {
        let planner = RuntimeMemoryPlanner::for_test(100, 80, MemoryTopology::Discrete);
        let first = planner
            .reserve(
                RuntimeMemoryFootprint {
                    host_bytes: 60,
                    device_bytes: 30,
                },
                available(),
            )
            .unwrap();
        assert!(
            planner
                .reserve(
                    RuntimeMemoryFootprint {
                        host_bytes: 41,
                        device_bytes: 1,
                    },
                    available(),
                )
                .is_err()
        );
        assert!(
            planner
                .reserve(
                    RuntimeMemoryFootprint {
                        host_bytes: 1,
                        device_bytes: 51,
                    },
                    available(),
                )
                .is_err()
        );
        assert_eq!(planner.snapshot().generations, 1);
        drop(first);
        assert_eq!(planner.snapshot().host_used_bytes, 0);
        assert_eq!(planner.snapshot().device_used_bytes, 0);
        assert_eq!(planner.snapshot().generations, 0);
    }

    #[test]
    fn failed_resize_preserves_the_original_charge() {
        let planner = RuntimeMemoryPlanner::for_test(100, 100, MemoryTopology::Discrete);
        let permit = planner
            .reserve(
                RuntimeMemoryFootprint {
                    host_bytes: 40,
                    device_bytes: 40,
                },
                available(),
            )
            .unwrap();
        assert!(
            permit
                .resize(
                    RuntimeMemoryFootprint {
                        host_bytes: 101,
                        device_bytes: 40,
                    },
                    available(),
                )
                .is_err()
        );
        let snapshot = planner.snapshot();
        assert_eq!(snapshot.host_used_bytes, 40);
        assert_eq!(snapshot.device_used_bytes, 40);
        drop(permit);
        assert_eq!(planner.snapshot().generations, 0);
    }

    #[test]
    fn candidate_revalidates_its_full_footprint_before_materialization() {
        let planner = RuntimeMemoryPlanner::for_test(100, 0, MemoryTopology::Unified);
        let permit = planner
            .reserve(
                RuntimeMemoryFootprint {
                    host_bytes: 60,
                    device_bytes: 0,
                },
                available(),
            )
            .unwrap();

        let error = permit
            .revalidate_available(RuntimeMemoryFootprint {
                host_bytes: 59,
                device_bytes: 0,
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("headroom is insufficient"));
        assert_eq!(permit.footprint().host_bytes, 60);
        assert_eq!(planner.snapshot().host_used_bytes, 60);
    }

    #[test]
    fn cloned_permit_releases_only_after_the_last_physical_owner() {
        let planner = RuntimeMemoryPlanner::for_test(100, 100, MemoryTopology::Discrete);
        let permit = planner
            .reserve(
                RuntimeMemoryFootprint {
                    host_bytes: 20,
                    device_bytes: 10,
                },
                available(),
            )
            .unwrap();
        let worker_permit = permit.clone();

        drop(permit);
        assert_eq!(planner.snapshot().host_used_bytes, 20);
        assert_eq!(planner.snapshot().generations, 1);
        drop(worker_permit);
        assert_eq!(planner.snapshot().host_used_bytes, 0);
        assert_eq!(planner.snapshot().generations, 0);
    }

    #[test]
    fn candidate_evaluation_snapshot_tracks_a_concurrent_residency_change() {
        let planner = RuntimeMemoryPlanner::for_test(100, 0, MemoryTopology::Unified);
        let worker_planner = planner.clone();
        let (held_tx, held_rx) = std::sync::mpsc::sync_channel(0);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            let permit = worker_planner
                .reserve(
                    RuntimeMemoryFootprint {
                        host_bytes: 80,
                        device_bytes: 0,
                    },
                    available(),
                )
                .unwrap();
            held_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(permit);
        });
        held_rx.recv().unwrap();

        let candidate = RuntimeMemoryFootprint {
            host_bytes: 30,
            device_bytes: 0,
        };
        let blocked = planner.evaluate_candidate(candidate, available());
        assert_eq!(blocked.snapshot.host_used_bytes, 80);
        assert_eq!(blocked.snapshot.generations, 1);
        assert!(blocked.verdict.is_err());

        release_tx.send(()).unwrap();
        worker.join().unwrap();
        let admitted = planner.evaluate_candidate(candidate, available());
        assert_eq!(admitted.snapshot.host_used_bytes, 0);
        assert_eq!(admitted.snapshot.generations, 0);
        assert!(admitted.verdict.is_ok());
    }

    #[test]
    fn ifb_and_cachemesh_are_included_in_the_conservative_charge() {
        let manifest = ModelManifest::default();
        let estimate = MemoryEstimate {
            weight_bytes: 10,
            host_weight_bytes: 10,
            device_weight_bytes: 0,
            kv_cache_bytes: 5,
            kv_cache_bytes_per_token: 0,
            temp_tensor_bytes: 2,
            total_bytes: 17,
            weight_dtype: bloomai_core::DType::F32,
            quantization: None,
            kv_cache_dtype: bloomai_core::DType::F16,
            num_layers: None,
            offloaded_layers: None,
            mmap_residency_applied: false,
            memory_scope: "test".to_string(),
        };
        let mut planner = RuntimeMemoryPlanner::for_test(usize::MAX, 0, MemoryTopology::Unified);
        planner.max_concurrent = 4;
        planner.enable_ifb = true;
        planner.retain_ifb_kv_reservation = true;
        planner.cachemesh_l2_bytes = 7;
        let footprint = planner.footprint(&manifest, &estimate).unwrap();
        assert_eq!(footprint.host_bytes, 80);
        assert_eq!(footprint.device_bytes, 0);

        let mut planner =
            RuntimeMemoryPlanner::for_test(usize::MAX, usize::MAX, MemoryTopology::Discrete);
        planner.max_concurrent = 4;
        planner.enable_ifb = true;
        planner.retain_ifb_kv_reservation = true;
        planner.cachemesh_l2_bytes = 7;
        let discrete = planner.footprint(&manifest, &estimate).unwrap();
        assert_eq!(discrete.host_bytes, 80);
        assert_eq!(discrete.device_bytes, 58);
    }

    #[test]
    fn strict_f32_kv_charge_scales_by_accounted_storage_dtype() {
        let mut estimate = MemoryEstimate {
            weight_bytes: 0,
            host_weight_bytes: 0,
            device_weight_bytes: 0,
            kv_cache_bytes: 10,
            kv_cache_bytes_per_token: 0,
            temp_tensor_bytes: 0,
            total_bytes: 10,
            weight_dtype: bloomai_core::DType::F32,
            quantization: None,
            kv_cache_dtype: bloomai_core::DType::F32,
            num_layers: None,
            offloaded_layers: None,
            mmap_residency_applied: false,
            memory_scope: "test".to_string(),
        };
        for (dtype, expected) in [
            (bloomai_core::DType::F32, 10),
            (bloomai_core::DType::F16, 20),
            (bloomai_core::DType::BF16, 20),
            (bloomai_core::DType::I8, 40),
            (bloomai_core::DType::U8, 40),
            (bloomai_core::DType::Q8, 40),
            (bloomai_core::DType::I4, 40),
            (bloomai_core::DType::NF4, 40),
            (bloomai_core::DType::Q4, 40),
        ] {
            estimate.kv_cache_dtype = dtype;
            assert_eq!(strict_f32_kv_bytes(&estimate).unwrap(), expected);
        }

        estimate.kv_cache_dtype = bloomai_core::DType::Unknown;
        assert!(strict_f32_kv_bytes(&estimate).is_err());
        estimate.kv_cache_bytes = 0;
        assert_eq!(strict_f32_kv_bytes(&estimate).unwrap(), 0);
        estimate.kv_cache_dtype = bloomai_core::DType::F16;
        estimate.kv_cache_bytes = usize::MAX;
        assert!(strict_f32_kv_bytes(&estimate).is_err());
    }

    #[test]
    fn qwen3_vl_gpu_uses_its_verified_half_precision_kv_layout() {
        let mut manifest = ModelManifest::default();
        manifest
            .parameters
            .insert("model_type".to_string(), serde_json::json!("qwen3_vl"));
        let estimate = MemoryEstimate {
            weight_bytes: 0,
            host_weight_bytes: 0,
            device_weight_bytes: 0,
            kv_cache_bytes: 10,
            kv_cache_bytes_per_token: 0,
            temp_tensor_bytes: 0,
            total_bytes: 10,
            weight_dtype: bloomai_core::DType::F16,
            quantization: None,
            kv_cache_dtype: bloomai_core::DType::F16,
            num_layers: None,
            offloaded_layers: None,
            mmap_residency_applied: false,
            memory_scope: "test".to_string(),
        };

        assert_eq!(
            strict_runtime_kv_bytes(&manifest, &estimate, DeviceKind::Gpu).unwrap(),
            10
        );
        assert_eq!(
            strict_runtime_kv_bytes(&manifest, &estimate, DeviceKind::Cpu).unwrap(),
            20
        );

        manifest.parameters.clear();
        assert_eq!(
            strict_runtime_kv_bytes(&manifest, &estimate, DeviceKind::Gpu).unwrap(),
            20
        );
    }

    #[test]
    fn non_ifb_wrapper_still_charges_general_kv_as_f32() {
        let manifest = ModelManifest::default();
        let estimate = MemoryEstimate {
            weight_bytes: 0,
            host_weight_bytes: 0,
            device_weight_bytes: 0,
            kv_cache_bytes: 10,
            kv_cache_bytes_per_token: 0,
            temp_tensor_bytes: 0,
            total_bytes: 10,
            weight_dtype: bloomai_core::DType::F32,
            quantization: None,
            kv_cache_dtype: bloomai_core::DType::F16,
            num_layers: None,
            offloaded_layers: None,
            mmap_residency_applied: false,
            memory_scope: "test".to_string(),
        };
        let planner = RuntimeMemoryPlanner::for_test(usize::MAX, 0, MemoryTopology::Unified);
        let footprint = planner.footprint(&manifest, &estimate).unwrap();
        assert_eq!(footprint.host_bytes, 20);
        assert_eq!(footprint.device_bytes, 0);
    }

    #[test]
    fn macos_available_memory_excludes_speculative_pages() {
        let vm_stat = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n\
                       Pages free:                               10.\n\
                       Pages inactive:                           20.\n\
                       Pages speculative:                         7.\n";
        assert_eq!(parse_macos_vm_stat_available(vm_stat), Some(30 * 4096));
    }

    #[test]
    fn cgroup_v2_probe_applies_the_tightest_ancestor_constraint() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("mounted hierarchy");
        std::fs::create_dir_all(&root).unwrap();
        let leaf = root.join("tenant/service");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(leaf.join("memory.max"), "max\n").unwrap();
        std::fs::write(leaf.join("memory.current"), "100\n").unwrap();

        let parent = root.join("tenant");
        std::fs::write(parent.join("memory.max"), "1000\n").unwrap();
        std::fs::write(parent.join("memory.current"), "400\n").unwrap();
        std::fs::write(root.join("memory.max"), "2000\n").unwrap();
        std::fs::write(root.join("memory.current"), "1500\n").unwrap();

        let mountinfo = format!(
            "29 23 0:26 / {} rw - cgroup2 cgroup rw\n",
            mountinfo_path(&root)
        );
        let constraint = strict_linux_cgroup_memory_from("0::/tenant/service\n", &mountinfo)
            .unwrap()
            .unwrap();
        assert_eq!(constraint, (1000, 500));
    }

    #[test]
    fn cgroup_v1_probe_reads_memory_controller_membership() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("combined-controller-mount");
        let leaf = base.join("abc");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(leaf.join("memory.limit_in_bytes"), "800\n").unwrap();
        std::fs::write(leaf.join("memory.usage_in_bytes"), "100\n").unwrap();
        std::fs::write(base.join("memory.limit_in_bytes"), "1000\n").unwrap();
        std::fs::write(base.join("memory.usage_in_bytes"), "600\n").unwrap();

        let mountinfo = format!(
            "31 23 0:28 /docker {} rw - cgroup cgroup rw,cpu,memory\n",
            mountinfo_path(&base)
        );
        let constraint = strict_linux_cgroup_memory_from("7:cpu,memory:/docker/abc\n", &mountinfo)
            .unwrap()
            .unwrap();
        assert_eq!(constraint, (800, 400));
    }

    #[test]
    fn cgroup_probe_rejects_unsafe_membership_paths() {
        assert!(strict_linux_cgroup_memory_from("0::/tenant/../escape\n", "").is_err());
    }

    #[test]
    fn cgroup_probe_fails_closed_when_control_files_are_missing_or_invalid() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let leaf = root.join("tenant");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(leaf.join("memory.max"), "1000\n").unwrap();
        let mountinfo = format!(
            "29 23 0:26 / {} rw - cgroup2 cgroup rw\n",
            mountinfo_path(root)
        );
        assert!(strict_linux_cgroup_memory_from("0::/tenant\n", &mountinfo).is_err());

        std::fs::write(leaf.join("memory.current"), "not-a-number\n").unwrap();
        assert!(strict_linux_cgroup_memory_from("0::/tenant\n", &mountinfo).is_err());
    }

    #[test]
    fn cgroup_probe_fails_closed_without_a_matching_mount_root() {
        let temp = tempfile::tempdir().unwrap();
        let mountinfo = format!(
            "29 23 0:26 /other {} rw - cgroup2 cgroup rw\n",
            mountinfo_path(temp.path())
        );
        assert!(strict_linux_cgroup_memory_from("0::/tenant\n", &mountinfo).is_err());
    }

    #[test]
    fn cgroup_probe_validates_unlimited_control_pairs() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let leaf = root.join("tenant");
        std::fs::create_dir_all(&leaf).unwrap();
        for directory in [&leaf, root] {
            std::fs::write(directory.join("memory.max"), "max\n").unwrap();
            std::fs::write(directory.join("memory.current"), "100\n").unwrap();
        }
        let mountinfo = format!(
            "29 23 0:26 / {} rw - cgroup2 cgroup rw\n",
            mountinfo_path(root)
        );
        assert_eq!(
            strict_linux_cgroup_memory_from("0::/tenant\n", &mountinfo).unwrap(),
            None
        );
    }

    fn mountinfo_path(path: &std::path::Path) -> String {
        path.to_string_lossy()
            .replace('\\', "\\134")
            .replace(' ', "\\040")
            .replace('\t', "\\011")
            .replace('\n', "\\012")
    }

    #[test]
    fn strict_estimation_rejects_unknown_and_overflowing_manifests() {
        let unknown = ModelManifest::default();
        assert!(validate_estimate_inputs(&unknown, 128).is_err());

        let overflowing = ModelManifest {
            files: vec![
                bloomai_core::ModelFile {
                    name: "a.gguf".to_string(),
                    format: bloomai_core::ModelFormat::Gguf,
                    size_bytes: usize::MAX,
                    hash_sha256: None,
                    required: true,
                },
                bloomai_core::ModelFile {
                    name: "b.gguf".to_string(),
                    format: bloomai_core::ModelFormat::Gguf,
                    size_bytes: 1,
                    hash_sha256: None,
                    required: true,
                },
            ],
            ..ModelManifest::default()
        };
        assert!(validate_estimate_inputs(&overflowing, 128).is_err());
    }
}
