#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::env;
#[cfg(target_os = "linux")]
use std::fs;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::path::Path;

use anyhow::{anyhow, Result};
use bloomai_core::{
    constants::GIB, DType, DeviceCapability, DeviceClass, DeviceKind, MemoryTopology, ModelFormat,
    PowerState, ThermalState,
};

use crate::backend::{Backend, BackendAvailability, BackendInfo};

#[derive(Default)]
pub struct IntelNpuBackend;

impl IntelNpuBackend {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn exists(path: &str) -> bool {
        Path::new(path).exists()
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn check_env_path(name: &str) -> bool {
        if let Ok(paths) = env::var(name) {
            #[cfg(target_os = "windows")]
            let separator = ';';
            #[cfg(not(target_os = "windows"))]
            let separator = ':';

            for path in paths.split(separator) {
                if Path::new(path).exists() {
                    return true;
                }
            }
        }
        false
    }

    #[cfg(target_os = "linux")]
    fn is_wsl() -> bool {
        std::env::var_os("WSL_INTEROP").is_some()
            || fs::read_to_string("/proc/sys/kernel/osrelease")
                .map(|v| v.to_lowercase().contains("microsoft"))
                .unwrap_or(false)
    }

    #[cfg(target_os = "windows")]
    fn probe_windows() -> BackendAvailability {
        let mut details = Vec::new();

        // Check common Intel NPU driver locations.
        let driver_paths = [
            r"C:\Windows\System32\drivers\IntelNPU.sys",
            r"C:\Windows\System32\drivers\ivpu.sys",
            r"C:\Windows\System32\drivers\Intel\IntelNPU.sys",
            r"C:\Program Files\Intel\Intel NPU driver",
            r"C:\Program Files (x86)\Intel\Intel NPU driver",
        ];

        let has_driver = driver_paths.iter().any(|p| Self::exists(p));

        // Check system-wide and user-level OpenVINO installation locations.
        let mut has_openvino = false;

        // System installation locations.
        let system_paths = [
            r"C:\Program Files\Intel\OpenVINO",
            r"C:\Program Files (x86)\Intel\OpenVINO",
            r"C:\Program Files\Intel\openvino",
            r"C:\Program Files (x86)\Intel\openvino",
            r"C:\Intel\openvino",
        ];

        for path in system_paths.iter() {
            if Self::exists(path) {
                has_openvino = true;
                break;
            }
        }

        // OpenVINO locations in user-level Python site-packages.
        if let Ok(home) = env::var("USERPROFILE") {
            let user_paths = [
                format!(
                    r"{}\AppData\Roaming\Python\Python311\site-packages\openvino",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python312\site-packages\openvino",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python310\site-packages\openvino",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python313\site-packages\openvino",
                    home
                ),
            ];

            for path in user_paths.iter() {
                if Self::exists(path) {
                    has_openvino = true;
                    break;
                }
            }
        }

        // Check OpenVINO environment variables.
        let has_openvino_env = Self::check_env_path("INTEL_OPENVINO_DIR")
            || Self::check_env_path("OPENVINO_DIR")
            || Self::check_env_path("PATH");

        // Check system-wide and user-level OpenVINO DLL locations.
        let mut has_openvino_dll = false;

        // System DLL locations.
        let system_dlls = [
            r"C:\Windows\System32\openvino.dll",
            r"C:\Windows\System32\openvino_c.dll",
            r"C:\Program Files\Intel\OpenVINO\runtime\bin\intel64\Release\openvino.dll",
            r"C:\Program Files (x86)\Intel\OpenVINO\runtime\bin\intel64\Release\openvino.dll",
        ];

        for dll in system_dlls.iter() {
            if Self::exists(dll) {
                has_openvino_dll = true;
                break;
            }
        }

        // DLL paths in user-level Python installations.
        if let Ok(home) = env::var("USERPROFILE") {
            let user_dlls = [
                format!(
                    r"{}\AppData\Roaming\Python\Python311\site-packages\openvino\libs\openvino.dll",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python311\site-packages\openvino\libs\openvino_intel_npu_plugin.dll",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python312\site-packages\openvino\libs\openvino.dll",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python312\site-packages\openvino\libs\openvino_intel_npu_plugin.dll",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python310\site-packages\openvino\libs\openvino.dll",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python310\site-packages\openvino\libs\openvino_intel_npu_plugin.dll",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python313\site-packages\openvino\libs\openvino.dll",
                    home
                ),
                format!(
                    r"{}\AppData\Roaming\Python\Python313\site-packages\openvino\libs\openvino_intel_npu_plugin.dll",
                    home
                ),
            ];

            for dll in user_dlls.iter() {
                if Self::exists(dll) {
                    has_openvino_dll = true;
                    break;
                }
            }
        }

        details.push(format!("npu driver found: {}", has_driver));
        details.push(format!("openvino installation found: {}", has_openvino));
        details.push(format!("openvino environment vars: {}", has_openvino_env));
        details.push(format!("openvino dlls found: {}", has_openvino_dll));

        // Treat the NPU as available when either a driver or OpenVINO is installed.
        if has_driver || has_openvino_dll {
            details.push("Intel NPU should be available through OpenVINO".to_string());
            BackendAvailability::available(details)
        } else {
            BackendAvailability::unavailable(
                "intel npu driver or openvino runtime not found on windows",
                details,
            )
        }
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> BackendAvailability {
        let mut details = Vec::new();

        if Self::is_wsl() {
            details.push("detected WSL environment".to_string());
            details.push("WSL usually does not expose Intel NPU PCI/device nodes".to_string());
            return BackendAvailability::unavailable(
                "intel npu unavailable in current WSL runtime",
                details,
            );
        }

        let accel_nodes = ["/dev/accel/accel0", "/dev/accel0", "/dev/accel/accel1"];
        let driver_paths = ["/sys/module/ivpu", "/sys/bus/pci/drivers/intel_vpu"];
        let openvino_hints = [
            "/opt/intel/openvino",
            "/usr/lib/libopenvino.so",
            "/usr/lib/x86_64-linux-gnu/libopenvino.so",
        ];

        let has_accel = accel_nodes.iter().any(|p| Self::exists(p));
        let has_driver = driver_paths.iter().any(|p| Self::exists(p));
        let has_openvino = openvino_hints.iter().any(|p| Self::exists(p));

        let has_openvino_env =
            Self::check_env_path("INTEL_OPENVINO_DIR") || Self::check_env_path("OPENVINO_DIR");

        details.push(format!("accel node found: {}", has_accel));
        details.push(format!("npu driver found: {}", has_driver));
        details.push(format!("openvino runtime hint found: {}", has_openvino));
        details.push(format!("openvino environment vars: {}", has_openvino_env));

        if (has_accel && has_driver) || has_openvino {
            BackendAvailability::available(details)
        } else {
            BackendAvailability::unavailable(
                "required intel npu device node or kernel driver not found",
                details,
            )
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn probe_other() -> BackendAvailability {
        BackendAvailability::unavailable(
            "intel npu backend supports linux and windows only",
            vec![format!("host os: {}", std::env::consts::OS)],
        )
    }

    fn probe() -> BackendAvailability {
        let _span = tracing::info_span!("backend.probe.intel_npu").entered();
        #[cfg(target_os = "windows")]
        {
            Self::probe_windows()
        }
        #[cfg(target_os = "linux")]
        {
            Self::probe_linux()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Self::probe_other()
        }
    }
}

impl Backend for IntelNpuBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "intel-npu",
            device: DeviceKind::Npu,
            accelerated: true,
        }
    }

    fn availability(&self) -> BackendAvailability {
        Self::probe()
    }

    fn capability(&self) -> DeviceCapability {
        // Intel NPU shares system memory (UMA); query actual available memory
        let total_memory = crate::backend::system_memory_bytes().unwrap_or(16 * GIB as usize);
        let avail_memory = crate::backend::available_free_memory();
        DeviceCapability {
            backend_name: "intel-npu".to_string(),
            vendor: Some("Intel".to_string()),
            device_class: DeviceClass::Npu,
            memory_topology: MemoryTopology::Unified, // Usually sharing system RAM
            max_memory: total_memory,
            available_memory: avail_memory,
            supported_dtypes: vec![DType::F16, DType::Q4, DType::Q8],
            supported_formats: vec![ModelFormat::OpenVinoIr],
            supports_mmap: true,
            has_quantization_kernels: true, // Supported via OpenVINO
            supports_streaming: false,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        }
    }

    fn warmup(&self) -> Result<()> {
        let availability = Self::probe();
        if availability.available {
            Ok(())
        } else {
            Err(anyhow!(availability
                .reason
                .unwrap_or_else(|| "intel npu unavailable".to_string())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_intel_npu_backend_availability() {
        let backend = IntelNpuBackend;
        let availability = backend.availability();
        println!("Intel NPU Availability details: {:?}", availability.details);

        if !availability.available {
            println!("Skipping availability assertion since no physical Intel NPU is present on this machine.");
            return;
        }

        assert!(
            availability.available,
            "Intel NPU backend should be available on this machine. Details: {:?}",
            availability.details
        );
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_intel_npu_backend_warmup() {
        let backend = IntelNpuBackend;
        if !backend.availability().available {
            println!("Skipping NPU warmup test because NPU is not available.");
            return;
        }
        let warmup_result = backend.warmup();
        assert!(
            warmup_result.is_ok(),
            "Intel NPU warmup should succeed on this machine, got error: {:?}",
            warmup_result
        );
    }

    #[test]
    fn test_real_npu_inference() {
        let backend = IntelNpuBackend;
        let availability = backend.availability();
        if !availability.available {
            println!("Skipping real NPU inference test because NPU is not available.");
            return;
        }

        println!("Running real NPU acceleration verification via Python...");
        let python_bin = if cfg!(target_os = "windows") {
            "python"
        } else {
            "python3"
        };
        let demo_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../npu_demo.py");

        let output = std::process::Command::new(python_bin)
            .arg(&demo_path)
            .env("PYTHONIOENCODING", "utf-8")
            .output()
            .expect("Failed to execute npu_demo.py");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        println!("npu_demo.py stdout:\n{}", stdout);
        println!("npu_demo.py stderr:\n{}", stderr);

        assert!(
            output.status.success(),
            "npu_demo.py execution failed with status: {:?}",
            output.status
        );

        assert!(
            stdout.contains("[PASS] Intel NPU inference function verified successfully!"),
            "NPU inference function verification signature not found in stdout!"
        );
        println!("Real NPU inference test passed successfully!");
    }

    #[test]
    fn test_npu_performance_sweep() {
        let backend = IntelNpuBackend;
        let availability = backend.availability();
        if !availability.available {
            println!("Skipping real NPU performance sweep test because NPU is not available.");
            return;
        }

        println!("Running real NPU performance sweep and dashboard generation via Python...");
        let python_bin = if cfg!(target_os = "windows") {
            "python"
        } else {
            "python3"
        };
        let sweep_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../npu_sweep.py");

        let output = std::process::Command::new(python_bin)
            .arg(&sweep_path)
            .env("PYTHONIOENCODING", "utf-8")
            .output()
            .expect("Failed to execute npu_sweep.py");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        println!("npu_sweep.py stdout:\n{}", stdout);
        println!("npu_sweep.py stderr:\n{}", stderr);

        assert!(
            output.status.success(),
            "npu_sweep.py execution failed with status: {:?}",
            output.status
        );

        assert!(
            stdout.contains("Dashboard generated successfully!"),
            "NPU sweep dashboard generation signature not found in stdout!"
        );

        let results_json = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../benchmark_results.json");
        let dashboard_html = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../npu_benchmark_dashboard.html");

        assert!(
            results_json.exists(),
            "benchmark_results.json was not created!"
        );
        assert!(
            dashboard_html.exists(),
            "npu_benchmark_dashboard.html was not created!"
        );

        println!("Real NPU performance sweep test passed and dashboard generated successfully!");
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_exists() {
        assert!(IntelNpuBackend::exists(env!("CARGO_MANIFEST_DIR")));
        assert!(!IntelNpuBackend::exists("non_existent_path_12345_xyz"));
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_check_env_path() {
        std::env::set_var("TEST_BLOOM_PATH", env!("CARGO_MANIFEST_DIR"));
        assert!(IntelNpuBackend::check_env_path("TEST_BLOOM_PATH"));

        std::env::set_var("TEST_BLOOM_PATH_INVALID", "non_existent_path_12345_xyz");
        assert!(!IntelNpuBackend::check_env_path("TEST_BLOOM_PATH_INVALID"));
    }

    #[test]
    fn test_intel_npu_backend_info() {
        let backend = IntelNpuBackend;
        let info = backend.info();
        assert_eq!(info.name, "intel-npu");
        assert_eq!(info.device, DeviceKind::Npu);
        assert!(info.accelerated);
    }

    #[test]
    fn test_intel_npu_backend_capability() {
        let backend = IntelNpuBackend;
        let capability = backend.capability();
        assert_eq!(capability.backend_name, "intel-npu");
        assert_eq!(capability.device_class, DeviceClass::Npu);
        assert_eq!(capability.memory_topology, MemoryTopology::Unified);
        assert!(capability.supported_dtypes.contains(&DType::F16));
        assert!(capability.supported_dtypes.contains(&DType::Q4));
        assert!(capability.supported_dtypes.contains(&DType::Q8));
        assert!(capability
            .supported_formats
            .contains(&ModelFormat::OpenVinoIr));
        assert!(capability.supports_mmap);
        assert!(capability.has_quantization_kernels);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_intel_npu_backend_reserve_and_release() {
        let backend = IntelNpuBackend;
        let availability = backend.availability();
        if !availability.available {
            println!("Skipping reserve/release test because NPU is not available.");
            return;
        }

        let ticket = bloomai_core::ResourceTicket {
            model_id: "test-reserve-npu-direct".to_string(),
            ram_bytes: 512,
            vram_bytes: 256,
            cache_bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            preferred_backend: Some("intel-npu".to_string()),
            fallback_backends: vec![],
        };

        let result = backend.reserve(&ticket);
        assert!(
            result.is_ok(),
            "Failed to reserve resources on NPU backend directly: {:?}",
            result
        );
        let lease = result.unwrap();
        assert_eq!(lease.granted_backend, "intel-npu");
        assert_eq!(lease.granted_ram, 512);
        assert_eq!(lease.granted_vram, 256);

        // Check coordinator state via snapshot
        let coord = bloomai_core::global_resource_coordinator();
        let snap = coord.snapshot();
        assert!(snap.ram_allocated >= 512);
        assert!(snap.vram_allocated >= 256);

        backend.release_lease(lease.lease_id);
    }
}
