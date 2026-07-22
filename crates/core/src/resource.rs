use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::constants::GIB;
use crate::manifest::ModelManifest;
use crate::types::{
    CacheKind, DeviceCapability, MemoryTopology, ResidencyStrategy, ResourcePriority,
};

/// A resource reservation request describing what a model needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTicket {
    pub model_id: String,
    pub ram_bytes: usize,
    pub vram_bytes: usize,
    pub cache_bytes: usize,
    pub priority: ResourcePriority,
    pub strategy: ResidencyStrategy,
    pub preferred_backend: Option<String>,
    pub fallback_backends: Vec<String>,
}

impl ResourceTicket {
    /// Construct a ResourceTicket from a ModelManifest and desired priority.
    pub fn from_manifest(manifest: &ModelManifest, priority: ResourcePriority) -> Self {
        let strategy = ResidencyStrategy::OnDemand;
        let mut ram_bytes = manifest.memory_profile.recommended_ram_bytes;
        let mut vram_bytes = manifest.memory_profile.recommended_vram_bytes;

        if ram_bytes == 0 && vram_bytes == 0 {
            let file_sum: usize = manifest.files.iter().map(|f| f.size_bytes).sum();
            let mut weight_bytes = if file_sum > 0 {
                file_sum
            } else {
                manifest
                    .memory_profile
                    .min_ram_bytes
                    .max(manifest.memory_profile.min_vram_bytes)
            };

            // Estimate from parameters if available
            if weight_bytes == 0 {
                let num_layers = manifest
                    .parameters
                    .get("num_hidden_layers")
                    .or_else(|| manifest.parameters.get("num_layers"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let hidden_size = manifest
                    .parameters
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if num_layers > 0 && hidden_size > 0 {
                    let intermediate_size = manifest
                        .parameters
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or((hidden_size * 4) as u64)
                        as usize;
                    let vocab_size = manifest
                        .parameters
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32000) as usize;
                    let params_per_layer =
                        4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size;
                    let total_params = num_layers * params_per_layer + 2 * vocab_size * hidden_size;
                    let bytes_per_weight = match manifest.primary_dtype {
                        crate::types::DType::F32 => 4.0,
                        crate::types::DType::F16 | crate::types::DType::BF16 => 2.0,
                        crate::types::DType::Q8 | crate::types::DType::I8 => 1.0,
                        crate::types::DType::Q4
                        | crate::types::DType::I4
                        | crate::types::DType::NF4 => 0.5,
                        _ => 2.0,
                    };
                    weight_bytes = (total_params as f64 * bytes_per_weight) as usize;
                }
            }

            if weight_bytes == 0 {
                weight_bytes = GIB as usize; // 1 GB fallback
            }

            let preferred_gpu = manifest.runtime_hints.preferred_backends.iter().any(|b| {
                let b_lower = b.to_lowercase();
                b_lower.contains("cuda") || b_lower.contains("metal") || b_lower.contains("gpu")
            });
            let temp_bytes = weight_bytes / 10;
            if preferred_gpu {
                ram_bytes = 0;
                vram_bytes = weight_bytes + temp_bytes;
            } else {
                ram_bytes = weight_bytes + temp_bytes;
                vram_bytes = 0;
            }
        }

        Self {
            model_id: manifest.id.clone(),
            ram_bytes,
            vram_bytes,
            cache_bytes: 0,
            priority,
            strategy,
            preferred_backend: manifest.runtime_hints.preferred_backends.first().cloned(),
            fallback_backends: manifest
                .runtime_hints
                .preferred_backends
                .iter()
                .skip(1)
                .cloned()
                .collect(),
        }
    }

    /// Construct with recommended strategy from manifest + capability.
    pub fn from_manifest_and_capability(
        manifest: &ModelManifest,
        priority: ResourcePriority,
        capability: &DeviceCapability,
    ) -> Self {
        let strategy = ResidencyStrategy::from_manifest_and_capability(manifest, capability);
        let mut ram_bytes = manifest.memory_profile.recommended_ram_bytes;
        let mut vram_bytes = manifest.memory_profile.recommended_vram_bytes;

        if ram_bytes == 0 && vram_bytes == 0 {
            let file_sum: usize = manifest.files.iter().map(|f| f.size_bytes).sum();
            let mut weight_bytes = if file_sum > 0 {
                file_sum
            } else {
                manifest
                    .memory_profile
                    .min_ram_bytes
                    .max(manifest.memory_profile.min_vram_bytes)
            };

            // Estimate from parameters if available
            if weight_bytes == 0 {
                let num_layers = manifest
                    .parameters
                    .get("num_hidden_layers")
                    .or_else(|| manifest.parameters.get("num_layers"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let hidden_size = manifest
                    .parameters
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if num_layers > 0 && hidden_size > 0 {
                    let intermediate_size = manifest
                        .parameters
                        .get("intermediate_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or((hidden_size * 4) as u64)
                        as usize;
                    let vocab_size = manifest
                        .parameters
                        .get("vocab_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(32000) as usize;
                    let params_per_layer =
                        4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size;
                    let total_params = num_layers * params_per_layer + 2 * vocab_size * hidden_size;
                    let bytes_per_weight = match manifest.primary_dtype {
                        crate::types::DType::F32 => 4.0,
                        crate::types::DType::F16 | crate::types::DType::BF16 => 2.0,
                        crate::types::DType::Q8 | crate::types::DType::I8 => 1.0,
                        crate::types::DType::Q4
                        | crate::types::DType::I4
                        | crate::types::DType::NF4 => 0.5,
                        _ => 2.0,
                    };
                    weight_bytes = (total_params as f64 * bytes_per_weight) as usize;
                }
            }

            if weight_bytes == 0 {
                weight_bytes = GIB as usize; // 1 GB fallback
            }

            // Apply mmap residency discount (30% physical resident memory multiplier)
            let weight_resident = if strategy == ResidencyStrategy::Mmap {
                weight_bytes * 30 / 100
            } else {
                weight_bytes
            };

            let temp_bytes = weight_bytes / 10;

            match capability.memory_topology {
                MemoryTopology::Unified | MemoryTopology::SharedSystemMemory => {
                    ram_bytes = weight_resident + temp_bytes;
                    vram_bytes = 0;
                }
                MemoryTopology::Discrete => {
                    if strategy == ResidencyStrategy::Offload {
                        ram_bytes = 0;
                        vram_bytes = weight_bytes + temp_bytes;
                    } else {
                        ram_bytes = weight_resident + temp_bytes;
                        vram_bytes = 0;
                    }
                }
                MemoryTopology::RemoteMemory => {
                    ram_bytes = 0;
                    vram_bytes = weight_bytes + temp_bytes;
                }
            }
        }

        Self {
            model_id: manifest.id.clone(),
            ram_bytes,
            vram_bytes,
            cache_bytes: 0,
            priority,
            strategy,
            preferred_backend: manifest.runtime_hints.preferred_backends.first().cloned(),
            fallback_backends: manifest
                .runtime_hints
                .preferred_backends
                .iter()
                .skip(1)
                .cloned()
                .collect(),
        }
    }

    /// Total memory footprint (for UMA unified budget checks).
    pub fn total_bytes(&self) -> usize {
        self.ram_bytes + self.vram_bytes + self.cache_bytes
    }
}

/// A granted resource lease returned by the coordinator after successful reservation.
#[derive(Debug, Clone)]
pub struct BackendLease {
    pub lease_id: u64,
    pub ticket: ResourceTicket,
    pub granted_backend: String,
    pub granted_ram: usize,
    pub granted_vram: usize,
    pub degraded: bool,
    pub degraded_reason: Option<String>,
    pub evicted_models: Vec<String>,
    pub created_at: Instant,
}

impl BackendLease {
    pub fn is_degraded(&self) -> bool {
        self.degraded
    }
}

/// Tracks a model's residency state within the coordinator.
pub struct ModelResidencyRecord {
    pub model_id: String,
    pub lease_id: u64,
    pub ram_bytes: usize,
    pub vram_bytes: usize,
    pub cache_bytes: usize,
    pub priority: ResourcePriority,
    pub strategy: ResidencyStrategy,
    pub backend_name: String,
    pub offload_cb: Arc<dyn Fn() -> Result<(), String> + Send + Sync>,
    pub last_accessed: Instant,
}

/// Lightweight handle for cache (KV cache, state cache, prefill cache) management.
#[derive(Debug, Clone)]
pub struct CacheHandle {
    pub handle_id: u64,
    pub model_id: String,
    pub cache_kind: CacheKind,
    pub bytes: usize,
    pub priority: ResourcePriority,
}

/// Callback type for model offload during eviction.
pub type OffloadCallback = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// Snapshot of current resource allocation state.
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    pub ram_budget: usize,
    pub vram_budget: usize,
    pub ram_allocated: usize,
    pub vram_allocated: usize,
    pub memory_topology: MemoryTopology,
    pub model_count: usize,
    pub cache_count: usize,
    pub lease_count: usize,
}

/// Per-model resource residency snapshot for observability and routing.
#[derive(Debug, Clone)]
pub struct ModelResourceSnapshot {
    pub model_id: String,
    pub lease_id: u64,
    pub ram_bytes: usize,
    pub vram_bytes: usize,
    pub cache_bytes: usize,
    pub priority: ResourcePriority,
    pub strategy: ResidencyStrategy,
    pub backend_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelManifest, ModelMemoryProfile, RuntimeHints};

    fn make_test_manifest(id: &str, ram: usize, vram: usize) -> ModelManifest {
        ModelManifest {
            id: id.to_string(),
            memory_profile: ModelMemoryProfile {
                min_ram_bytes: ram / 2,
                min_vram_bytes: vram / 2,
                recommended_ram_bytes: ram,
                recommended_vram_bytes: vram,
            },
            runtime_hints: RuntimeHints {
                preferred_backends: vec!["metal".to_string(), "cpu".to_string()],
                supports_mmap: true,
                requires_streaming: false,
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_resource_ticket_from_manifest() {
        let manifest = make_test_manifest("test-model", 1024, 2048);
        let ticket = ResourceTicket::from_manifest(&manifest, ResourcePriority::Normal);
        assert_eq!(ticket.model_id, "test-model");
        assert_eq!(ticket.ram_bytes, 1024);
        assert_eq!(ticket.vram_bytes, 2048);
        assert_eq!(ticket.priority, ResourcePriority::Normal);
        assert_eq!(ticket.preferred_backend, Some("metal".to_string()));
        assert_eq!(ticket.fallback_backends, vec!["cpu".to_string()]);
    }

    #[test]
    fn test_resource_priority_ordering() {
        assert!(ResourcePriority::Critical > ResourcePriority::High);
        assert!(ResourcePriority::High > ResourcePriority::Normal);
        assert!(ResourcePriority::Normal > ResourcePriority::Low);
        assert!(ResourcePriority::Low > ResourcePriority::Speculative);
    }

    #[test]
    fn test_resource_ticket_total_bytes() {
        let ticket = ResourceTicket {
            model_id: "m".into(),
            ram_bytes: 100,
            vram_bytes: 200,
            cache_bytes: 50,
            priority: ResourcePriority::Normal,
            strategy: ResidencyStrategy::OnDemand,
            preferred_backend: None,
            fallback_backends: vec![],
        };
        assert_eq!(ticket.total_bytes(), 350);
    }

    #[test]
    fn test_residency_strategy_from_capability() {
        let manifest = make_test_manifest("m1", 1024, 512);
        let cap_uma = DeviceCapability {
            backend_name: "metal".into(),
            vendor: Some("Apple".into()),
            device_class: crate::DeviceClass::IntegratedGpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: 16 * 1024 * 1024 * 1024,
            available_memory: 8 * 1024 * 1024 * 1024,
            supported_dtypes: vec![],
            supported_formats: vec![],
            supports_mmap: true,
            has_quantization_kernels: false,
            supports_streaming: false,
            thermal_state: crate::ThermalState::Nominal,
            power_state: crate::PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        };
        let strategy = ResidencyStrategy::from_manifest_and_capability(&manifest, &cap_uma);
        assert_eq!(strategy, ResidencyStrategy::Mmap);

        let cap_discrete = DeviceCapability {
            backend_name: "cuda".into(),
            vendor: Some("NVIDIA".into()),
            device_class: crate::DeviceClass::DiscreteGpu,
            memory_topology: MemoryTopology::Discrete,
            max_memory: 24 * 1024 * 1024 * 1024,
            available_memory: 24 * 1024 * 1024 * 1024,
            supported_dtypes: vec![],
            supported_formats: vec![],
            supports_mmap: false,
            has_quantization_kernels: false,
            supports_streaming: false,
            thermal_state: crate::ThermalState::Nominal,
            power_state: crate::PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        };
        // min_vram_bytes (256) < available_memory (24GB) -> Offload
        let strategy2 = ResidencyStrategy::from_manifest_and_capability(&manifest, &cap_discrete);
        assert_eq!(strategy2, ResidencyStrategy::Offload);
    }

    #[test]
    fn test_backend_lease_is_degraded() {
        let lease = BackendLease {
            lease_id: 1,
            ticket: ResourceTicket {
                model_id: "m".into(),
                ram_bytes: 0,
                vram_bytes: 0,
                cache_bytes: 0,
                priority: ResourcePriority::Normal,
                strategy: ResidencyStrategy::OnDemand,
                preferred_backend: None,
                fallback_backends: vec![],
            },
            granted_backend: "cpu".into(),
            granted_ram: 0,
            granted_vram: 0,
            degraded: true,
            degraded_reason: Some("fallback".into()),
            evicted_models: vec![],
            created_at: Instant::now(),
        };
        assert!(lease.is_degraded());
    }

    #[test]
    fn test_cache_handle() {
        let handle = CacheHandle {
            handle_id: 42,
            model_id: "llama-7b".into(),
            cache_kind: CacheKind::KvCache,
            bytes: 1024 * 1024,
            priority: ResourcePriority::High,
        };
        assert_eq!(handle.handle_id, 42);
        assert_eq!(handle.cache_kind, CacheKind::KvCache);
    }

    #[test]
    fn test_dynamic_estimation_from_files() {
        use crate::manifest::ModelFile;
        use crate::types::ModelFormat;

        let mut manifest = make_test_manifest("test-dyn-files", 0, 0);
        manifest.runtime_hints.preferred_backends = vec!["cpu".to_string()];
        manifest.files = vec![ModelFile {
            name: "model.safetensors".to_string(),
            format: ModelFormat::Safetensors,
            size_bytes: 100 * 1024 * 1024, // 100 MB
            hash_sha256: None,
            required: true,
        }];

        let ticket = ResourceTicket::from_manifest(&manifest, ResourcePriority::Normal);
        // ram_bytes = 100 MB + 10 MB (temp) = 110 MB
        assert_eq!(ticket.ram_bytes, 110 * 1024 * 1024);
        assert_eq!(ticket.vram_bytes, 0);
    }

    #[test]
    fn test_dynamic_estimation_from_parameters() {
        let mut manifest = make_test_manifest("test-dyn-params", 0, 0);
        manifest.runtime_hints.preferred_backends = vec!["cpu".to_string()];
        manifest
            .parameters
            .insert("num_hidden_layers".to_string(), serde_json::json!(28));
        manifest
            .parameters
            .insert("hidden_size".to_string(), serde_json::json!(4096));
        manifest.primary_dtype = crate::types::DType::Q4; // bits: 4

        let ticket = ResourceTicket::from_manifest(&manifest, ResourcePriority::Normal);
        assert!(ticket.ram_bytes > 0);
        assert_eq!(ticket.vram_bytes, 0);
    }

    #[test]
    fn test_mmap_residency_discount() {
        use crate::manifest::ModelFile;
        use crate::types::ModelFormat;

        let mut manifest = make_test_manifest("test-mmap-discount", 0, 0);
        manifest.files = vec![ModelFile {
            name: "model.safetensors".to_string(),
            format: ModelFormat::Safetensors,
            size_bytes: 100 * 1024 * 1024, // 100 MB
            hash_sha256: None,
            required: true,
        }];
        manifest.runtime_hints.supports_mmap = true;

        let cap_uma = DeviceCapability {
            backend_name: "metal".into(),
            vendor: Some("Apple".into()),
            device_class: crate::DeviceClass::IntegratedGpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: 16 * 1024 * 1024 * 1024,
            available_memory: 8 * 1024 * 1024 * 1024,
            supported_dtypes: vec![],
            supported_formats: vec![],
            supports_mmap: true,
            has_quantization_kernels: false,
            supports_streaming: false,
            thermal_state: crate::ThermalState::Nominal,
            power_state: crate::PowerState::PluggedIn,
            max_batch_tokens: None,
            available_parallelism: None,
        };

        let ticket = ResourceTicket::from_manifest_and_capability(
            &manifest,
            ResourcePriority::Normal,
            &cap_uma,
        );
        // strategy should be Mmap
        assert_eq!(ticket.strategy, ResidencyStrategy::Mmap);
        // ram_bytes = 100 MB * 30% (mmap) + 10 MB (temp) = 30 MB + 10 MB = 40 MB
        assert_eq!(ticket.ram_bytes, 40 * 1024 * 1024);
        assert_eq!(ticket.vram_bytes, 0);
    }
}
