use anyhow::Result;
use bloomai_backend::{Backend, BackendAvailability, BackendInfo};
use bloomai_core::{
    DeviceCapability, DeviceClass, DeviceKind, MemoryTopology, ModelFormat, PowerState,
    ThermalState,
};

/// Skeleton struct for a vendor custom NPU/DSP accelerator backend.
#[derive(Default)]
pub struct VendorSdkBackend;

impl Backend for VendorSdkBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "vendor-npu",
            device: DeviceKind::Npu,
            accelerated: true,
        }
    }

    fn availability(&self) -> BackendAvailability {
        // Here, a vendor would query the driver, sysfs, or PCIe registration
        // to detect if their specific NPU/DSP chip is physically connected.
        let device_detected = true;

        if device_detected {
            BackendAvailability::available(vec![
                "Vendor Custom NPU chip detected on PCIe bus".to_string(),
                "Driver version: v2.4.0".to_string(),
            ])
        } else {
            BackendAvailability::unavailable(
                "No Vendor Custom NPU hardware found".to_string(),
                vec![],
            )
        }
    }

    fn capability(&self) -> DeviceCapability {
        // Query NPU VRAM / Unified memory limits and chip attributes
        DeviceCapability {
            backend_name: "vendor-npu".to_string(),
            vendor: Some("VendorCorp".to_string()),
            device_class: DeviceClass::Npu,
            memory_topology: MemoryTopology::Discrete,
            max_memory: 8 * 1024 * 1024 * 1024,       // 8 GB
            available_memory: 8 * 1024 * 1024 * 1024, // 8 GB
            supported_dtypes: vec![bloomai_core::DType::F16, bloomai_core::DType::I8],
            supported_formats: vec![ModelFormat::Onnx],
            supports_mmap: false,
            has_quantization_kernels: true,
            supports_streaming: false,
            thermal_state: ThermalState::Nominal,
            power_state: PowerState::PluggedIn,
            max_batch_tokens: Some(4096),
            available_parallelism: Some(4),
        }
    }

    fn warmup(&self) -> Result<()> {
        let _span = tracing::info_span!("vendor_npu.warmup").entered();
        tracing::info!("Initializing vendor NPU compiler pipeline and loading libraries...");
        // Vendors perform runtime/driver pre-compilation or graph compilation here
        Ok(())
    }
}

// Expose the dynamic plugin factory entry point for Bloom loader
#[unsafe(no_mangle)]
pub extern "C" fn bloomai_backend_plugin_init() -> *mut fn() -> Box<dyn Backend> {
    let factory: fn() -> Box<dyn Backend> = || Box::new(VendorSdkBackend);
    Box::into_raw(Box::new(factory))
}
