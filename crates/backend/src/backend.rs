use anyhow::Result;
use bloomai_core::{
    BackendLease, DType, DeviceCapability, DeviceClass, DeviceKind, MemoryTopology, ModelFormat,
    PowerState, ResourceError, ResourceTicket, ThermalState, constants::GIB,
};

fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

pub(crate) fn system_memory_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb = rest.split_whitespace().next()?.parse::<usize>().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
    }
    #[cfg(target_os = "windows")]
    {
        None
    }
}

fn conservative_available_memory(total: usize) -> usize {
    total.saturating_mul(3) / 4
}

/// Read actual free memory from /proc/meminfo (Linux) or fall back to 75% estimate.
pub(crate) fn available_free_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb) = rest.split_whitespace().next() {
                        if let Ok(kb_val) = kb.parse::<usize>() {
                            return kb_val * 1024;
                        }
                    }
                }
            }
        }
        let total = system_memory_bytes().unwrap_or(8 * GIB as usize);
        conservative_available_memory(total)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let total = system_memory_bytes().unwrap_or(8 * GIB as usize);
        conservative_available_memory(total)
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaDeviceSnapshot {
    name: Option<String>,
    total_memory: usize,
    free_memory: usize,
}

#[cfg(feature = "cuda")]
fn cuda_device_zero() -> std::result::Result<CudaDeviceSnapshot, String> {
    use cudarc::driver::CudaContext;

    // cudarc's dynamic loader panics when the CUDA driver shared library is absent.
    // Convert that boundary into a normal unavailable result so probing is safe on
    // CUDA-enabled binaries running on hosts without an NVIDIA driver.
    let context = std::panic::catch_unwind(|| CudaContext::new(0))
        .map_err(|_| "CUDA driver library could not be loaded".to_string())?
        .map_err(|error| format!("could not open CUDA logical device 0: {error}"))?;
    let (free_memory, total_memory) = context
        .mem_get_info()
        .map_err(|error| format!("could not query CUDA logical device 0 memory: {error}"))?;

    Ok(CudaDeviceSnapshot {
        name: context.name().ok(),
        total_memory,
        free_memory,
    })
}

#[cfg(any(feature = "cuda", test))]
fn cuda_availability_from_probe(
    probe: std::result::Result<CudaDeviceSnapshot, String>,
    nvidia_smi_driver_version: Option<String>,
) -> BackendAvailability {
    let mut details = vec!["CUDA backend compiled".to_string()];
    if let Some(driver) = nvidia_smi_driver_version {
        details.push(format!(
            "nvidia-smi driver version (descriptive metadata only): {driver}"
        ));
    }

    match probe {
        Ok(snapshot) => {
            details.push("CUDA driver logical device 0 is available".to_string());
            if let Some(name) = snapshot.name {
                details.push(format!("logical device 0: {name}"));
            }
            details.push(format!(
                "CUDA driver total memory: {} MB",
                snapshot.total_memory / 1024 / 1024
            ));
            details.push(format!(
                "CUDA driver free memory: {} MB",
                snapshot.free_memory / 1024 / 1024
            ));
            BackendAvailability::available(details)
        }
        Err(error) => BackendAvailability::unavailable(
            format!("CUDA driver logical device 0 is unavailable: {error}"),
            details,
        ),
    }
}

#[cfg(any(feature = "cuda", test))]
fn cuda_memory_from_probe(
    probe: std::result::Result<CudaDeviceSnapshot, String>,
) -> (usize, usize) {
    match probe {
        Ok(snapshot) => (snapshot.total_memory, snapshot.free_memory),
        Err(_) => (0, 0),
    }
}

#[cfg(feature = "cuda")]
fn cuda_driver_version() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8(output.stdout)
            .ok()?
            .lines()
            .next()?
            .trim()
            .to_string(),
    )
}

fn cpu_vendor() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if let Some(rest) = line.strip_prefix("vendor_id") {
                    if let Some(id) = rest.split(':').nth(1) {
                        return Some(id.trim().to_string());
                    }
                }
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.vendor"])
            .output()
            .ok()?;
        if output.status.success() {
            let v = String::from_utf8(output.stdout).ok()?.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
        Some("Apple".to_string())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn check_mlx_python_available() -> bool {
    std::process::Command::new("python3")
        .args(["-c", "import mlx.core"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn check_mps_available() -> bool {
    std::process::Command::new("python3")
        .args([
            "-c",
            "import torch; print(torch.backends.mps.is_available())",
        ])
        .output()
        .map(|o| {
            o.status.success()
                && String::from_utf8(o.stdout)
                    .map(|s| s.trim() == "True")
                    .unwrap_or(false)
        })
        .unwrap_or(false)
}

fn check_coreml_available() -> bool {
    std::process::Command::new("python3")
        .args(["-c", "import coremltools"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: &'static str,
    pub device: DeviceKind,
    pub accelerated: bool,
}

#[derive(Debug, Clone)]
pub struct BackendAvailability {
    pub available: bool,
    pub reason: Option<String>,
    pub details: Vec<String>,
}

impl BackendAvailability {
    pub fn available(details: Vec<String>) -> Self {
        Self {
            available: true,
            reason: None,
            details,
        }
    }

    pub fn unavailable(reason: impl Into<String>, details: Vec<String>) -> Self {
        Self {
            available: false,
            reason: Some(reason.into()),
            details,
        }
    }
}

pub trait Backend: Send + Sync {
    fn info(&self) -> BackendInfo;
    fn availability(&self) -> BackendAvailability;
    fn capability(&self) -> DeviceCapability;
    fn warmup(&self) -> Result<()>;

    /// Reserve resources for a model on this backend.
    /// Default implementation uses the global resource coordinator.
    fn reserve(&self, ticket: &ResourceTicket) -> std::result::Result<BackendLease, ResourceError> {
        let _span = tracing::info_span!(
            "backend.reserve",
            backend = self.info().name,
            model_id = %ticket.model_id,
        )
        .entered();
        let avail = self.availability();
        if !avail.available {
            return Err(ResourceError::BackendUnavailable {
                backend: self.info().name.to_string(),
                reason: avail.reason.unwrap_or_else(|| "unknown".to_string()),
            });
        }
        let coordinator = bloomai_core::global_resource_coordinator();
        let cb = std::sync::Arc::new(|| Ok(()));
        let mut ticket = ticket.clone();
        ticket.preferred_backend = Some(self.info().name.to_string());
        coordinator.reserve(ticket, cb)
    }

    /// Release a previously acquired lease.
    fn release_lease(&self, lease_id: u64) {
        let coordinator = bloomai_core::global_resource_coordinator();
        coordinator.release(lease_id);
    }
}

#[derive(Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "cpu",
            device: DeviceKind::Cpu,
            accelerated: false,
        }
    }

    fn availability(&self) -> BackendAvailability {
        BackendAvailability::available(vec![
            "cpu backend is always available".to_string(),
            format!("logical threads: {}", available_parallelism()),
        ])
    }

    fn capability(&self) -> DeviceCapability {
        let total_memory = system_memory_bytes().unwrap_or(8 * GIB as usize);
        let parallelism = available_parallelism();
        DeviceCapability {
            backend_name: "cpu".to_string(),
            vendor: cpu_vendor(),
            device_class: DeviceClass::Cpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: total_memory,
            available_memory: available_free_memory(),
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16, DType::Q8, DType::Q4],
            supported_formats: vec![
                ModelFormat::Gguf,
                ModelFormat::Safetensors,
                ModelFormat::OpenVinoIr,
            ],
            supports_mmap: true,
            has_quantization_kernels: true, // Candle CPU supports quantization
            supports_streaming: true,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: Some(parallelism),
        }
    }

    fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "metal",
            device: DeviceKind::Gpu,
            accelerated: true,
        }
    }

    fn availability(&self) -> BackendAvailability {
        if cfg!(target_os = "macos") {
            let mut details = vec![
                format!("host os: {}", std::env::consts::OS),
                "Metal backend provides GPU acceleration via MPS/CoreML".to_string(),
            ];
            // Check for MPS availability via Python torch
            if check_mps_available() {
                details.push("PyTorch MPS backend is available".to_string());
            } else {
                details
                    .push("PyTorch MPS not detected; Metal ops may use CPU fallback".to_string());
            }
            // Check for CoreML availability
            if check_coreml_available() {
                details.push("CoreML framework is available".to_string());
            }
            BackendAvailability::available(details)
        } else {
            BackendAvailability::unavailable(
                "metal backend supports macOS only",
                vec![format!("host os: {}", std::env::consts::OS)],
            )
        }
    }

    fn capability(&self) -> DeviceCapability {
        let total_memory = system_memory_bytes().unwrap_or(8 * GIB as usize);
        DeviceCapability {
            backend_name: "metal".to_string(),
            vendor: Some("Apple".to_string()),
            device_class: DeviceClass::IntegratedGpu,
            memory_topology: MemoryTopology::Unified, // UMA
            max_memory: total_memory,
            available_memory: available_free_memory(),
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16, DType::Q8, DType::Q4],
            supported_formats: vec![ModelFormat::Gguf, ModelFormat::Safetensors],
            supports_mmap: true,
            has_quantization_kernels: true, // we added metal_quant
            supports_streaming: true,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        }
    }

    fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "cuda",
            device: DeviceKind::Gpu,
            accelerated: true,
        }
    }

    fn availability(&self) -> BackendAvailability {
        #[cfg(feature = "cuda")]
        {
            cuda_availability_from_probe(cuda_device_zero(), cuda_driver_version())
        }
        #[cfg(not(feature = "cuda"))]
        {
            BackendAvailability::unavailable(
                "cuda backend not enabled in compile features",
                vec!["Recompile with --features cuda".to_string()],
            )
        }
    }

    fn capability(&self) -> DeviceCapability {
        #[cfg(feature = "cuda")]
        let (total_memory, avail_memory) = cuda_memory_from_probe(cuda_device_zero());
        #[cfg(not(feature = "cuda"))]
        let (total_memory, avail_memory) = (0, 0);
        DeviceCapability {
            backend_name: "cuda".to_string(),
            vendor: Some("NVIDIA".to_string()),
            device_class: DeviceClass::DiscreteGpu,
            memory_topology: MemoryTopology::Discrete,
            max_memory: total_memory,
            available_memory: avail_memory,
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16, DType::Q8, DType::Q4],
            supported_formats: vec![ModelFormat::Gguf, ModelFormat::Safetensors],
            supports_mmap: false,
            has_quantization_kernels: true,
            supports_streaming: true,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        }
    }

    fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct MlxBackend;

impl Backend for MlxBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "mlx",
            device: DeviceKind::Gpu,
            accelerated: true,
        }
    }

    fn availability(&self) -> BackendAvailability {
        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            let mut details = vec![
                "MLX backend is available on Apple Silicon".to_string(),
                format!("host: {} {}", std::env::consts::OS, std::env::consts::ARCH),
            ];
            // Check if mlx Python package is importable
            if check_mlx_python_available() {
                details.push("mlx Python package is importable".to_string());
                BackendAvailability::available(details)
            } else {
                details.push("mlx Python package NOT found (pip install mlx)".to_string());
                BackendAvailability::unavailable(
                    "mlx Python package is not installed; run `pip install mlx`",
                    details,
                )
            }
        } else {
            BackendAvailability::unavailable(
                "mlx backend supports Apple Silicon (macOS aarch64) only",
                vec![format!(
                    "host os: {}, arch: {}",
                    std::env::consts::OS,
                    std::env::consts::ARCH
                )],
            )
        }
    }

    fn capability(&self) -> DeviceCapability {
        let total_memory = system_memory_bytes().unwrap_or(8 * GIB as usize);
        DeviceCapability {
            backend_name: "mlx".to_string(),
            vendor: Some("Apple".to_string()),
            device_class: DeviceClass::IntegratedGpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: total_memory,
            available_memory: available_free_memory(),
            supported_dtypes: vec![DType::F32, DType::F16, DType::BF16, DType::Q4],
            supported_formats: vec![ModelFormat::Safetensors],
            supports_mmap: true,
            has_quantization_kernels: true,
            supports_streaming: true,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        }
    }

    fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_capability_reports_vendor_and_threads() {
        let backend = CpuBackend;
        let cap = backend.capability();
        assert_eq!(cap.backend_name, "cpu");
        assert_eq!(cap.device_class, DeviceClass::Cpu);
        assert!(
            cap.supports_streaming,
            "CPU backend should support streaming"
        );
        assert!(cap.supports_mmap, "CPU backend should support mmap");
        assert!(cap.has_quantization_kernels);
        assert!(
            cap.available_parallelism.is_some(),
            "CPU backend should report available_parallelism"
        );
        let threads = cap.available_parallelism.unwrap();
        assert!(threads >= 1, "should have at least 1 thread");
        if cfg!(target_os = "linux") {
            assert!(
                cap.vendor.is_some(),
                "CPU vendor should be available on Linux"
            );
        }
    }

    #[test]
    fn cpu_capability_reports_real_available_memory() {
        let backend = CpuBackend;
        let cap = backend.capability();
        assert!(
            cap.available_memory > 0,
            "available_memory should be positive, got {}",
            cap.available_memory
        );
        assert!(
            cap.available_memory <= cap.max_memory,
            "available_memory ({}) should not exceed max_memory ({})",
            cap.available_memory,
            cap.max_memory
        );
    }

    #[test]
    fn cpu_availability_reports_threads() {
        let backend = CpuBackend;
        let avail = backend.availability();
        assert!(avail.available);
        assert!(
            avail.details.iter().any(|d| d.contains("logical threads")),
            "CPU availability should report thread count: {:?}",
            avail.details
        );
    }

    #[test]
    fn metal_capability_reports_apple_vendor() {
        let backend = MetalBackend;
        let cap = backend.capability();
        assert_eq!(cap.backend_name, "metal");
        assert_eq!(cap.vendor.as_deref(), Some("Apple"));
        assert_eq!(cap.device_class, DeviceClass::IntegratedGpu);
        assert!(cap.supports_streaming);
        assert!(cap.supports_mmap);
    }

    #[test]
    fn metal_availability_gives_mps_coreml_info() {
        let backend = MetalBackend;
        let avail = backend.availability();
        if cfg!(target_os = "macos") {
            assert!(avail.available);
            assert!(
                avail
                    .details
                    .iter()
                    .any(|d| d.contains("MPS") || d.contains("CoreML")),
                "Metal availability should mention MPS or CoreML: {:?}",
                avail.details
            );
        } else {
            assert!(!avail.available);
            assert!(avail.reason.is_some());
        }
    }

    #[test]
    fn cuda_capability_reports_nvidia_vendor() {
        let backend = CudaBackend;
        let cap = backend.capability();
        assert_eq!(cap.backend_name, "cuda");
        assert_eq!(cap.vendor.as_deref(), Some("NVIDIA"));
        assert_eq!(cap.device_class, DeviceClass::DiscreteGpu);
        assert_eq!(cap.memory_topology, MemoryTopology::Discrete);
    }

    #[test]
    fn cuda_availability_reports_logical_device_zero() {
        let backend = CudaBackend;
        let avail = backend.availability();
        #[cfg(feature = "cuda")]
        {
            if avail.available {
                assert!(
                    avail.details.iter().any(|d| d.contains("logical device 0")),
                    "CUDA availability should report logical device 0: {:?}",
                    avail.details
                );
                assert!(avail.details.iter().any(|d| d.contains("total memory")));
                assert!(avail.details.iter().any(|d| d.contains("free memory")));
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            assert!(!avail.available);
            assert!(
                avail.reason.as_ref().unwrap().contains("not enabled"),
                "CUDA unavailable reason: {:?}",
                avail.reason
            );
        }
    }

    #[test]
    fn cuda_driver_probe_is_the_only_availability_gate() {
        let snapshot = CudaDeviceSnapshot {
            name: Some("test gpu".to_string()),
            total_memory: 24 * GIB as usize,
            free_memory: 20 * GIB as usize,
        };
        let available = cuda_availability_from_probe(Ok(snapshot), None);
        assert!(available.available);
        assert!(
            available
                .details
                .iter()
                .any(|detail| detail.contains("logical device 0"))
        );

        let unavailable = cuda_availability_from_probe(
            Err("driver probe failed".to_string()),
            Some("999.0".to_string()),
        );
        assert!(!unavailable.available);
        assert!(
            unavailable
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("driver probe failed"))
        );
        assert!(
            unavailable
                .details
                .iter()
                .any(|detail| detail.contains("descriptive metadata only")),
            "nvidia-smi metadata must not make the backend available"
        );
    }

    #[test]
    fn cuda_capability_memory_uses_driver_probe_without_fallback() {
        let expected = (24 * GIB as usize, 20 * GIB as usize);
        let snapshot = CudaDeviceSnapshot {
            name: None,
            total_memory: expected.0,
            free_memory: expected.1,
        };
        assert_eq!(cuda_memory_from_probe(Ok(snapshot)), expected);
        assert_eq!(
            cuda_memory_from_probe(Err("driver probe failed".to_string())),
            (0, 0)
        );
    }

    #[test]
    fn mlx_capability_reports_apple_vendor() {
        let backend = MlxBackend;
        let cap = backend.capability();
        assert_eq!(cap.backend_name, "mlx");
        assert_eq!(cap.vendor.as_deref(), Some("Apple"));
        assert_eq!(cap.device_class, DeviceClass::IntegratedGpu);
        assert!(
            cap.supports_streaming,
            "MLX backend should support streaming"
        );
    }

    #[test]
    fn mlx_availability_checks_python_on_apple_silicon() {
        let backend = MlxBackend;
        let avail = backend.availability();
        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            if avail.available {
                assert!(
                    avail.details.iter().any(|d| d.contains("importable")),
                    "should mention mlx importable: {:?}",
                    avail.details
                );
            } else {
                assert!(
                    avail.details.iter().any(|d| d.contains("NOT found")),
                    "should mention mlx NOT found: {:?}",
                    avail.details
                );
            }
        } else {
            assert!(!avail.available);
        }
    }

    #[test]
    fn all_backends_populate_new_capability_fields() {
        for backend_name in ["cpu", "metal", "cuda", "mlx"] {
            let cap = match backend_name {
                "cpu" => CpuBackend.capability(),
                "metal" => MetalBackend.capability(),
                "cuda" => CudaBackend.capability(),
                "mlx" => MlxBackend.capability(),
                _ => unreachable!(),
            };
            let _ = cap.max_batch_tokens;
            let _ = cap.available_parallelism;
            let _ = cap.vendor;
        }
    }

    #[test]
    fn available_free_memory_returns_positive() {
        let mem = available_free_memory();
        assert!(
            mem > 0,
            "available_free_memory should be positive, got {}",
            mem
        );
    }
}
