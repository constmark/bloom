use anyhow::{anyhow, bail, Context, Result};

use crate::core::manifest::{format_bytes, suggest_memory_downgrade, MemoryEstimate};

const DEFAULT_MEMORY_UTILIZATION: f64 = 0.75;
const MIN_MEMORY_UTILIZATION: f64 = 0.05;
const MAX_MEMORY_UTILIZATION: f64 = 0.95;
const DEFAULT_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const PAGE_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy)]
pub struct MemoryPreallocationConfig {
    pub enabled: bool,
    pub memory_utilization: f64,
    pub reserve_memory_bytes: Option<usize>,
}

impl Default for MemoryPreallocationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_utilization: DEFAULT_MEMORY_UTILIZATION,
            reserve_memory_bytes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryPreallocationPlan {
    pub estimate: MemoryEstimate,
    pub available_bytes: Option<usize>,
    pub budget_bytes: Option<usize>,
    pub reserve_bytes: usize,
    pub memory_utilization: f64,
}

impl MemoryPreallocationPlan {
    pub fn display_summary(&self) -> String {
        let budget = self
            .budget_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unknown".to_string());
        format!(
            "reserve={}, budget={}, estimate={}",
            format_bytes(self.reserve_bytes),
            budget,
            self.estimate.display_summary()
        )
    }
}

pub struct MemoryReservation {
    bytes: usize,
    _chunks: Vec<Vec<u8>>,
}

impl MemoryReservation {
    pub fn reserve(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Ok(Self {
                bytes,
                _chunks: Vec::new(),
            });
        }

        let mut chunks = Vec::new();
        let mut remaining = bytes;
        while remaining > 0 {
            let chunk_bytes = remaining.min(DEFAULT_CHUNK_BYTES);
            let mut chunk: Vec<u8> = Vec::new();
            chunk
                .try_reserve_exact(chunk_bytes)
                .map_err(|e| anyhow!("failed to reserve {}: {e}", format_bytes(chunk_bytes)))?;
            chunk.resize(chunk_bytes, 0u8);

            for offset in (0..chunk.len()).step_by(PAGE_BYTES) {
                chunk[offset] = chunk[offset].wrapping_add(1);
            }
            if let Some(last) = chunk.last_mut() {
                *last = (*last).wrapping_add(1);
            }

            chunks.push(chunk);
            remaining -= chunk_bytes;
        }

        Ok(Self {
            bytes,
            _chunks: chunks,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

pub fn default_memory_utilization() -> f64 {
    DEFAULT_MEMORY_UTILIZATION
}

pub fn plan_memory_preallocation(
    estimate: MemoryEstimate,
    config: MemoryPreallocationConfig,
) -> Result<MemoryPreallocationPlan> {
    if !config.enabled {
        return Ok(MemoryPreallocationPlan {
            estimate,
            available_bytes: available_system_memory(),
            budget_bytes: None,
            reserve_bytes: 0,
            memory_utilization: clamp_memory_utilization(config.memory_utilization)?,
        });
    }

    let memory_utilization = clamp_memory_utilization(config.memory_utilization)?;
    let available_bytes = available_system_memory();
    let budget_bytes = available_bytes
        .map(|available| ((available as f64) * memory_utilization).floor().max(0.0) as usize);

    if let Some(budget) = budget_bytes {
        if estimate.total_bytes > budget {
            let mut message = format!(
                "memory budget exceeded before model load: estimated {} but conservative budget is {} (utilization {:.0}% of available {})",
                estimate.display_summary(),
                format_bytes(budget),
                memory_utilization * 100.0,
                format_bytes(available_bytes.unwrap_or_default())
            );
            for suggestion in suggest_memory_downgrade(&estimate, budget) {
                message.push_str("\n  - ");
                message.push_str(&suggestion);
            }
            bail!(message);
        }
    }

    let reserve_bytes = config.reserve_memory_bytes.unwrap_or_else(|| {
        estimate
            .kv_cache_bytes
            .saturating_add(estimate.temp_tensor_bytes)
    });

    if let Some(budget) = budget_bytes {
        if reserve_bytes > budget {
            bail!(
                "requested startup memory reservation {} exceeds conservative budget {}",
                format_bytes(reserve_bytes),
                format_bytes(budget)
            );
        }
    }

    Ok(MemoryPreallocationPlan {
        estimate,
        available_bytes,
        budget_bytes,
        reserve_bytes,
        memory_utilization,
    })
}

pub fn reserve_memory_for_plan(plan: &MemoryPreallocationPlan) -> Result<MemoryReservation> {
    MemoryReservation::reserve(plan.reserve_bytes).with_context(|| {
        format!(
            "startup memory preallocation failed ({})",
            plan.display_summary()
        )
    })
}

pub fn available_system_memory() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        return parse_meminfo_kib(&content, "MemAvailable").map(|kib| kib.saturating_mul(1024));
    }

    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("vm_stat").output().ok()?;
        if output.status.success() {
            if let Some(bytes) =
                parse_macos_vm_stat_available(&String::from_utf8_lossy(&output.stdout))
            {
                return Some(bytes);
            }
        }

        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        return String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<usize>()
            .ok();
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Return whether an untyped runtime error explicitly reports memory exhaustion.
///
/// Matching `oom` as an arbitrary substring is unsafe because ordinary Bloom
/// configuration names such as `BLOOM_DTYPE` contain those letters. Keep the
/// fallback narrow: accept established phrases or `oom` as a standalone token.
pub(crate) fn error_text_indicates_oom(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("cannot allocate memory")
        || lower.contains("failed to allocate memory")
        || lower.contains("insufficient ram")
        || lower.contains("insufficient vram")
        || lower
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|token| token == "oom")
}

fn clamp_memory_utilization(value: f64) -> Result<f64> {
    if !value.is_finite() {
        bail!("memory utilization must be finite");
    }
    if !(MIN_MEMORY_UTILIZATION..=MAX_MEMORY_UTILIZATION).contains(&value) {
        bail!(
            "memory utilization must be between {:.2} and {:.2}",
            MIN_MEMORY_UTILIZATION,
            MAX_MEMORY_UTILIZATION
        );
    }
    Ok(value)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(content: &str, key: &str) -> Option<usize> {
    content.lines().find_map(|line| {
        let (line_key, rest) = line.split_once(':')?;
        if line_key != key {
            return None;
        }
        rest.split_whitespace()
            .next()
            .and_then(|value| value.parse::<usize>().ok())
    })
}

#[cfg(target_os = "macos")]
fn parse_macos_vm_stat_available(text: &str) -> Option<usize> {
    let page_size = text
        .lines()
        .find_map(|line| {
            line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|n| n.parse::<usize>().ok())
        })
        .unwrap_or(16_384);

    let mut pages = 0usize;
    for key in [
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    ] {
        if let Some(value) = text.lines().find_map(|line| {
            line.trim()
                .strip_prefix(key)
                .and_then(|rest| rest.strip_prefix(':'))
                .map(|rest| rest.trim().trim_end_matches('.'))
                .and_then(|rest| rest.parse::<usize>().ok())
        }) {
            pages = pages.saturating_add(value);
        }
    }

    pages.checked_mul(page_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloomai_core::DType;

    fn estimate() -> MemoryEstimate {
        MemoryEstimate {
            weight_bytes: 100,
            host_weight_bytes: 100,
            device_weight_bytes: 0,
            kv_cache_bytes: 20,
            kv_cache_bytes_per_token: 20,
            temp_tensor_bytes: 10,
            total_bytes: 130,
            weight_dtype: DType::F16,
            quantization: None,
            kv_cache_dtype: DType::F16,
            num_layers: None,
            offloaded_layers: None,
            mmap_residency_applied: false,
            memory_scope: "host-resident estimate".to_string(),
        }
    }

    #[test]
    fn explicit_reservation_overrides_runtime_bytes() {
        let plan = plan_memory_preallocation(
            estimate(),
            MemoryPreallocationConfig {
                enabled: true,
                memory_utilization: default_memory_utilization(),
                reserve_memory_bytes: Some(7),
            },
        )
        .unwrap();

        assert_eq!(plan.reserve_bytes, 7);
    }

    #[test]
    fn disabled_preallocation_reserves_nothing() {
        let plan = plan_memory_preallocation(
            estimate(),
            MemoryPreallocationConfig {
                enabled: false,
                memory_utilization: default_memory_utilization(),
                reserve_memory_bytes: Some(7),
            },
        )
        .unwrap();

        assert_eq!(plan.reserve_bytes, 0);
    }

    #[test]
    fn rejects_invalid_memory_utilization() {
        let err = plan_memory_preallocation(
            estimate(),
            MemoryPreallocationConfig {
                enabled: true,
                memory_utilization: 1.5,
                reserve_memory_bytes: None,
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("memory utilization"));
    }

    #[test]
    fn reserves_small_memory_block() {
        let reservation = MemoryReservation::reserve(8192).unwrap();
        assert_eq!(reservation.bytes(), 8192);
        assert!(!reservation.is_empty());
    }

    #[test]
    fn oom_detection_requires_an_explicit_memory_error() {
        assert!(error_text_indicates_oom(
            "CUDA OOM while allocating a tensor"
        ));
        assert!(error_text_indicates_oom("allocator: out of memory"));
        assert!(error_text_indicates_oom(
            "cannot allocate memory for weights"
        ));
        assert!(!error_text_indicates_oom(
            "BLOOM_DTYPE=bf16 is unsupported on CPU"
        ));
        assert!(!error_text_indicates_oom(
            "bloom model configuration failed"
        ));
    }

    #[test]
    fn test_prefetch_file_madvise_test() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), b"test data").unwrap();
        prefetch_file_madvise(temp_file.path());
    }

    #[test]
    fn test_layer_placement_strategy() {
        let strategy = LayerPlacementStrategy::new(8, Some(2));
        assert_eq!(strategy.placements.len(), 8);
        assert_eq!(strategy.placements[0], DevicePlacement::Gpu);
        assert_eq!(strategy.placements[1], DevicePlacement::Gpu);
        assert_eq!(strategy.placements[2], DevicePlacement::Cpu);
        assert_eq!(strategy.placements[7], DevicePlacement::Cpu);

        std::env::set_var("BLOOM_GPU_LAYERS", "4");
        std::env::set_var("BLOOM_NPU_LAYERS", "2");
        let strategy_env = LayerPlacementStrategy::new(8, None);
        std::env::remove_var("BLOOM_GPU_LAYERS");
        std::env::remove_var("BLOOM_NPU_LAYERS");

        assert_eq!(strategy_env.placements[0], DevicePlacement::Gpu);
        assert_eq!(strategy_env.placements[3], DevicePlacement::Gpu);
        assert_eq!(strategy_env.placements[4], DevicePlacement::Npu);
        assert_eq!(strategy_env.placements[5], DevicePlacement::Npu);
        assert_eq!(strategy_env.placements[6], DevicePlacement::Cpu);
    }
}

/// Advise the kernel to prefetch a file's pages for sequential access.
///
/// Linux exposes `posix_fadvise`; macOS/BSD lack it and instead use
/// `fcntl(F_RDADVISE)`. Other platforms are a no-op.
#[cfg(target_os = "linux")]
pub fn prefetch_file_madvise<P: AsRef<std::path::Path>>(path: P) {
    if let Ok(file) = std::fs::File::open(path) {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        unsafe {
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL);
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_WILLNEED);
        }
    }
}

#[cfg(target_os = "macos")]
pub fn prefetch_file_madvise<P: AsRef<std::path::Path>>(path: P) {
    if let Ok(file) = std::fs::File::open(path) {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        // radvisory is libc's struct for F_RDADVISE on Apple platforms.
        let mut advisory = libc::radvisory {
            ra_offset: 0,
            ra_count: 0, // 0 = to end of file
        };
        unsafe {
            libc::fcntl(fd, libc::F_RDADVISE, &mut advisory);
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn prefetch_file_madvise<P: AsRef<std::path::Path>>(_path: P) {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevicePlacement {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Debug, Clone)]
pub struct LayerPlacementStrategy {
    pub placements: Vec<DevicePlacement>,
}

impl LayerPlacementStrategy {
    pub fn new(num_layers: usize, offloaded_layers: Option<usize>) -> Self {
        let npu_layers = std::env::var("BLOOM_NPU_LAYERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let mut placements = vec![DevicePlacement::Cpu; num_layers];

        let gpu_count = offloaded_layers
            .or_else(|| {
                std::env::var("BLOOM_GPU_LAYERS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(num_layers)
            .min(num_layers);

        let npu_count = npu_layers.unwrap_or(0).min(num_layers - gpu_count);

        for i in 0..gpu_count {
            placements[i] = DevicePlacement::Gpu;
        }
        for i in gpu_count..(gpu_count + npu_count) {
            placements[i] = DevicePlacement::Npu;
        }

        Self { placements }
    }

    #[cfg(feature = "candle-engine")]
    pub fn device_for_layer(
        &self,
        layer_idx: usize,
        default_device: &candle_core::Device,
    ) -> candle_core::Device {
        match self
            .placements
            .get(layer_idx)
            .unwrap_or(&DevicePlacement::Cpu)
        {
            DevicePlacement::Cpu => candle_core::Device::Cpu,
            DevicePlacement::Gpu => default_device.clone(),
            DevicePlacement::Npu => default_device.clone(),
        }
    }
}
