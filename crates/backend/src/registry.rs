use std::collections::HashMap;

use anyhow::{Result, anyhow};
use bloomai_core::{BackendLease, DeviceCapability, ResourceError, ResourceTicket};

use crate::backend::{
    Backend, BackendAvailability, CpuBackend, CudaBackend, MetalBackend, MlxBackend,
};
use crate::intel_npu::IntelNpuBackend;

pub struct BackendStatus {
    pub name: String,
    pub available: bool,
    pub reason: Option<String>,
    pub details: Vec<String>,
    pub capability: Option<DeviceCapability>,
}

pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn Backend>>,
}

impl Default for BackendRegistry {
    fn default() -> Self {
        let mut registry = Self {
            backends: HashMap::new(),
        };
        registry.register("cpu", Box::<CpuBackend>::default());
        registry.register("metal", Box::<MetalBackend>::default());
        registry.register("mlx", Box::<MlxBackend>::default());
        registry.register("cuda", Box::<CudaBackend>::default());
        registry.register("gpu", Box::<CudaBackend>::default());
        registry.register("intel-npu", Box::<IntelNpuBackend>::default());
        registry
    }
}

impl BackendRegistry {
    pub fn register(&mut self, name: impl Into<String>, backend: Box<dyn Backend>) {
        self.backends.insert(name.into(), backend);
    }

    pub fn get(&self, name: &str) -> Result<&dyn Backend> {
        self.backends
            .get(name)
            .map(|b| b.as_ref())
            .ok_or_else(|| anyhow!("backend '{}' not found", name))
    }

    pub fn ensure_available(&self, name: &str) -> Result<&dyn Backend> {
        let _span = tracing::info_span!("registry.ensure_available", backend = name).entered();
        let backend = self.get(name)?;
        let status: BackendAvailability = backend.availability();
        if status.available {
            tracing::info!("backend '{}' is available", name);
            Ok(backend)
        } else {
            let reason = status
                .reason
                .unwrap_or_else(|| format!("backend '{}' unavailable", name));
            tracing::warn!("backend '{}' unavailable: {}", name, reason);
            Err(anyhow!(reason))
        }
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.backends.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    pub fn status(&self) -> Vec<BackendStatus> {
        let mut out = Vec::new();
        for name in self.names() {
            if let Ok(backend) = self.get(name) {
                let s = backend.availability();
                let cap = if s.available {
                    Some(backend.capability())
                } else {
                    None
                };
                out.push(BackendStatus {
                    name: name.to_string(),
                    available: s.available,
                    reason: s.reason,
                    details: s.details,
                    capability: cap,
                });
            }
        }
        out
    }

    /// Reserve resources on a specific backend by name.
    pub fn reserve_on(
        &self,
        backend_name: &str,
        ticket: &ResourceTicket,
    ) -> std::result::Result<BackendLease, ResourceError> {
        let backend = self
            .get(backend_name)
            .map_err(|_| ResourceError::BackendUnavailable {
                backend: backend_name.to_string(),
                reason: format!("backend '{}' not found", backend_name),
            })?;
        let avail = backend.availability();
        if !avail.available {
            return Err(ResourceError::BackendUnavailable {
                backend: backend_name.to_string(),
                reason: avail.reason.unwrap_or_else(|| "unavailable".to_string()),
            });
        }
        backend.reserve(ticket)
    }

    /// Try preferred backend first, then fallback list. Returns a (possibly degraded) lease.
    pub fn reserve_with_fallback(
        &self,
        ticket: &ResourceTicket,
    ) -> std::result::Result<BackendLease, ResourceError> {
        let mut tried: Vec<String> = Vec::new();

        // Try preferred
        if let Some(ref preferred) = ticket.preferred_backend {
            tried.push(preferred.clone());
            match self.reserve_on(preferred, ticket) {
                Ok(lease) => return Ok(lease),
                Err(_) => { /* fall through */ }
            }
        }

        // Try fallbacks
        for fallback in &ticket.fallback_backends {
            tried.push(fallback.clone());
            match self.reserve_on(fallback, ticket) {
                Ok(mut lease) => {
                    lease.degraded = true;
                    lease.degraded_reason = Some(format!(
                        "preferred backend '{}' unavailable, fell back to '{}'",
                        ticket.preferred_backend.as_deref().unwrap_or("none"),
                        fallback
                    ));
                    return Ok(lease);
                }
                Err(_) => { /* try next */ }
            }
        }

        Err(ResourceError::AllBackendsExhausted {
            model_id: ticket.model_id.clone(),
            tried,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::BackendRegistry;
    use bloomai_core::{DeviceCapability, DeviceClass, MemoryTopology, PowerState, ThermalState};

    #[test]
    fn default_registry_has_cpu() {
        let registry = BackendRegistry::default();
        assert!(registry.get("cpu").is_ok());
    }

    #[test]
    fn default_registry_has_intel_npu() {
        let registry = BackendRegistry::default();
        assert!(registry.get("intel-npu").is_ok());
    }

    #[test]
    fn default_registry_has_metal() {
        let registry = BackendRegistry::default();
        assert!(registry.get("metal").is_ok());
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_intel_npu_backend_status() {
        let registry = BackendRegistry::default();
        let status = registry.status();
        let npu_status = status.iter().find(|s| s.name == "intel-npu").unwrap();
        assert_eq!(npu_status.name, "intel-npu");
        if !npu_status.available {
            println!("Skipping NPU backend status check since NPU is not available.");
            return;
        }
        assert!(
            npu_status.available,
            "NPU backend should be available on this machine! Reason: {:?}, Details: {:?}",
            npu_status.reason, npu_status.details
        );
        assert!(npu_status.capability.is_some());
    }

    #[test]
    fn test_invalid_backend_get() {
        let registry = BackendRegistry::default();
        let result = registry.get("non-existent-backend");
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "backend 'non-existent-backend' not found"
        );
    }

    #[test]
    fn test_ensure_available_on_unavailable() {
        let registry = BackendRegistry::default();

        // On Windows, metal is not available. Let's verify ensure_available returns an error.
        #[cfg(target_os = "windows")]
        {
            let result = registry.ensure_available("metal");
            assert!(result.is_err());
            assert!(
                result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("metal backend supports macOS only")
            );
        }

        // On non-macOS, let's verify cpu is always available.
        let cpu_result = registry.ensure_available("cpu");
        assert!(cpu_result.is_ok());
    }

    #[test]
    fn test_registry_names() {
        let registry = BackendRegistry::default();
        let mut names = registry.names();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["cpu", "cuda", "gpu", "intel-npu", "metal", "mlx"]
        );
    }

    struct DummyBackend;
    impl crate::backend::Backend for DummyBackend {
        fn info(&self) -> crate::backend::BackendInfo {
            crate::backend::BackendInfo {
                name: "dummy",
                device: bloomai_core::DeviceKind::Cpu,
                accelerated: false,
            }
        }
        fn availability(&self) -> crate::backend::BackendAvailability {
            crate::backend::BackendAvailability::available(vec![
                "dummy is always ready".to_string(),
            ])
        }
        fn capability(&self) -> DeviceCapability {
            DeviceCapability {
                backend_name: "dummy".to_string(),
                vendor: None,
                device_class: DeviceClass::Cpu,
                memory_topology: MemoryTopology::Unified,
                max_memory: 0,
                available_memory: 0,
                supported_dtypes: vec![],
                supported_formats: vec![],
                supports_mmap: false,
                has_quantization_kernels: false,
                supports_streaming: false,
                thermal_state: ThermalState::Nominal,
                power_state: PowerState::PluggedIn,
                max_batch_tokens: None,
                available_parallelism: None,
            }
        }
        fn warmup(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_custom_backend_registration() {
        let mut registry = BackendRegistry::default();
        registry.register("dummy", Box::new(DummyBackend));
        assert!(registry.get("dummy").is_ok());
        assert!(registry.ensure_available("dummy").is_ok());

        let names = registry.names();
        assert!(names.contains(&"dummy"));

        let status = registry.status();
        let dummy_status = status.iter().find(|s| s.name == "dummy").unwrap();
        assert!(dummy_status.available);
        assert_eq!(
            dummy_status.details,
            vec!["dummy is always ready".to_string()]
        );
        assert!(dummy_status.capability.is_some());
    }

    #[test]
    fn test_registry_reserve_on_available() {
        let registry = BackendRegistry::default();
        let ticket = bloomai_core::ResourceTicket {
            model_id: "test-reserve-cpu".to_string(),
            ram_bytes: 1024,
            vram_bytes: 0,
            cache_bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            preferred_backend: Some("cpu".to_string()),
            fallback_backends: vec![],
        };
        let result = registry.reserve_on("cpu", &ticket);
        assert!(result.is_ok());
        let lease = result.unwrap();
        assert_eq!(lease.granted_backend, "cpu");
    }

    #[test]
    fn test_registry_reserve_on_nonexistent() {
        let registry = BackendRegistry::default();
        let ticket = bloomai_core::ResourceTicket {
            model_id: "test-phantom".to_string(),
            ram_bytes: 100,
            vram_bytes: 0,
            cache_bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            preferred_backend: None,
            fallback_backends: vec![],
        };
        let result = registry.reserve_on("nonexistent-backend", &ticket);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(bloomai_core::ResourceError::BackendUnavailable { .. })
        ));
    }

    #[test]
    fn test_registry_reserve_with_fallback_success() {
        let registry = BackendRegistry::default();
        let ticket = bloomai_core::ResourceTicket {
            model_id: "test-fallback-ok".to_string(),
            ram_bytes: 512,
            vram_bytes: 0,
            cache_bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            // On non-macOS, cuda is not available, so should fallback to cpu
            preferred_backend: Some("cuda".to_string()),
            fallback_backends: vec!["cpu".to_string()],
        };
        let result = registry.reserve_with_fallback(&ticket);
        assert!(result.is_ok());
        let lease = result.unwrap();
        let _ = lease; // silence unused variable warning when cuda feature is enabled
        // On macOS, cuda is unavailable, so it should have fallen back to cpu
        #[cfg(not(feature = "cuda"))]
        {
            assert!(lease.is_degraded());
            assert_eq!(lease.granted_backend, "cpu");
        }
    }

    #[test]
    fn test_registry_reserve_all_exhausted() {
        let registry = BackendRegistry::default();
        let ticket = bloomai_core::ResourceTicket {
            model_id: "test-exhausted".to_string(),
            ram_bytes: 512,
            vram_bytes: 0,
            cache_bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            // Request a non-existent backend with no fallbacks
            preferred_backend: Some("nonexistent".to_string()),
            fallback_backends: vec![],
        };
        let result = registry.reserve_with_fallback(&ticket);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(bloomai_core::ResourceError::AllBackendsExhausted { .. })
        ));
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_registry_reserve_on_intel_npu() {
        let registry = BackendRegistry::default();
        let n = registry.get("intel-npu").unwrap();
        if !n.availability().available {
            println!("Skipping registry reserve on intel-npu test because NPU is not available.");
            return;
        }

        let ticket = bloomai_core::ResourceTicket {
            model_id: "test-reserve-registry-npu".to_string(),
            ram_bytes: 256,
            vram_bytes: 128,
            cache_bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            preferred_backend: Some("intel-npu".to_string()),
            fallback_backends: vec![],
        };

        let result = registry.reserve_on("intel-npu", &ticket);
        assert!(
            result.is_ok(),
            "Failed registry reservation on NPU: {:?}",
            result
        );
        let lease = result.unwrap();
        assert_eq!(lease.granted_backend, "intel-npu");
        assert_eq!(lease.granted_ram, 256);
        assert_eq!(lease.granted_vram, 128);

        registry
            .get("intel-npu")
            .unwrap()
            .release_lease(lease.lease_id);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_registry_reserve_with_fallback_intel_npu() {
        let registry = BackendRegistry::default();
        let n = registry.get("intel-npu").unwrap();
        if !n.availability().available {
            println!(
                "Skipping registry reserve with fallback intel-npu test because NPU is not available."
            );
            return;
        }

        let ticket = bloomai_core::ResourceTicket {
            model_id: "test-fallback-registry-npu".to_string(),
            ram_bytes: 256,
            vram_bytes: 128,
            cache_bytes: 0,
            priority: bloomai_core::ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            preferred_backend: Some("intel-npu".to_string()),
            fallback_backends: vec!["cpu".to_string()],
        };

        let result = registry.reserve_with_fallback(&ticket);
        assert!(
            result.is_ok(),
            "Failed reservation with fallback: {:?}",
            result
        );
        let lease = result.unwrap();
        assert_eq!(lease.granted_backend, "intel-npu");
        assert!(
            !lease.is_degraded(),
            "Should not be degraded since preferred backend NPU is available"
        );

        registry
            .get("intel-npu")
            .unwrap()
            .release_lease(lease.lease_id);
    }
}
