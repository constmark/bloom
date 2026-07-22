//! Device backend abstraction.

pub mod backend;
pub mod intel_npu;
pub mod registry;

pub use backend::{
    Backend, BackendAvailability, BackendInfo, CpuBackend, CudaBackend, MetalBackend, MlxBackend,
};
pub use intel_npu::IntelNpuBackend;
pub use registry::{BackendRegistry, BackendStatus};
