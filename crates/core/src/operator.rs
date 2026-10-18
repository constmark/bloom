//! Runtime operator registration and capability based dispatch.
//!
//! Operators are intentionally represented as byte oriented kernels at this
//! boundary.  Concrete engines can decode the bytes into their native tensor
//! type, while the core runtime still enforces dtype/layout compatibility and
//! provides deterministic fallback selection.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{DType, TensorShape};

/// Memory layout of a tensor passed to an operator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    /// Row-major contiguous storage (the default for flat tensors).
    RowMajor,
    /// Column-major contiguous storage.
    ColMajor,
    /// Image tensors in channels-first order.
    Nchw,
    /// Image tensors in channels-last order.
    Nhwc,
    /// Video tensors in channels-first order.
    Ncdhw,
    /// Video tensors in channels-last order.
    Ndhwc,
    /// Operator accepts any layout and performs its own normalization.
    Any,
}

/// An opaque tensor value crossing the runtime/operator boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OperatorTensor {
    pub data: Vec<u8>,
    pub shape: TensorShape,
    pub dtype: DType,
    pub layout: TensorLayout,
}

impl OperatorTensor {
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType, layout: TensorLayout) -> Self {
        Self {
            data,
            shape: TensorShape(shape),
            dtype,
            layout,
        }
    }
}

/// A registered operator's capability and fallback metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorDescriptor {
    /// Stable operator name, for example `matmul` or `flash_attention`.
    pub name: String,
    /// Plugin/implementation version used for diagnostics and replacement.
    pub version: String,
    /// Backend that owns the implementation, for example `cpu` or `metal`.
    pub backend: String,
    /// Empty means every dtype.  Otherwise only listed dtypes are accepted.
    #[serde(default)]
    pub supported_dtypes: Vec<DType>,
    /// Empty means every layout.  `Any` also accepts every layout.
    #[serde(default)]
    pub supported_layouts: Vec<TensorLayout>,
    /// Higher priority implementations win when several match.
    #[serde(default)]
    pub priority: u32,
    /// Name of the operator to try when this implementation is unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
}

impl OperatorDescriptor {
    fn validate(&self) -> Result<(), OperatorRegistryError> {
        if self.name.trim().is_empty() {
            return Err(OperatorRegistryError::InvalidDescriptor(
                "operator name cannot be empty".into(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(OperatorRegistryError::InvalidDescriptor(format!(
                "operator '{}' version cannot be empty",
                self.name
            )));
        }
        if self.backend.trim().is_empty() {
            return Err(OperatorRegistryError::InvalidDescriptor(format!(
                "operator '{}' backend cannot be empty",
                self.name
            )));
        }
        if self.fallback.as_deref() == Some(self.name.as_str()) {
            return Err(OperatorRegistryError::InvalidDescriptor(format!(
                "operator '{}' cannot fall back to itself",
                self.name
            )));
        }
        Ok(())
    }

    fn supports(&self, dtype: DType, layout: TensorLayout, backend: Option<&str>) -> bool {
        let dtype_ok = self.supported_dtypes.is_empty() || self.supported_dtypes.contains(&dtype);
        let layout_ok = self.supported_layouts.is_empty()
            || self.supported_layouts.contains(&TensorLayout::Any)
            || self.supported_layouts.contains(&layout);
        let backend_ok = backend.is_none_or(|requested| requested == self.backend);
        dtype_ok && layout_ok && backend_ok
    }
}

/// Errors raised by operator registration and dispatch.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum OperatorRegistryError {
    #[error("invalid operator descriptor: {0}")]
    InvalidDescriptor(String),
    #[error("operator '{name}' version '{version}' for backend '{backend}' is already registered")]
    Duplicate {
        name: String,
        version: String,
        backend: String,
    },
    #[error(
        "operator '{name}' has no compatible implementation for dtype {dtype:?}, layout {layout:?}, backend {backend:?}"
    )]
    Unsupported {
        name: String,
        dtype: DType,
        layout: TensorLayout,
        backend: Option<String>,
    },
    #[error("operator fallback cycle detected at '{0}'")]
    FallbackCycle(String),
    #[error("operator '{0}' is not registered")]
    NotFound(String),
    #[error("operator kernel failed: {0}")]
    Kernel(String),
    #[error("operator registry lock poisoned")]
    LockPoisoned,
}

/// Implement an operator without coupling the core crate to a tensor backend.
pub trait OperatorKernel: Send + Sync {
    fn execute(&self, inputs: &[OperatorTensor]) -> Result<OperatorTensor, OperatorRegistryError>;
}

struct OperatorEntry {
    descriptor: OperatorDescriptor,
    kernel: Arc<dyn OperatorKernel>,
}

/// Thread-safe registry for native and plugin-provided operators.
#[derive(Default)]
pub struct OperatorRegistry {
    entries: RwLock<HashMap<String, Vec<OperatorEntry>>>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one implementation. Duplicate (name, version, backend)
    /// entries are rejected so plugin load order cannot silently alter output.
    pub fn register<K>(
        &self,
        descriptor: OperatorDescriptor,
        kernel: K,
    ) -> Result<(), OperatorRegistryError>
    where
        K: OperatorKernel + 'static,
    {
        descriptor.validate()?;
        let mut entries = self
            .entries
            .write()
            .map_err(|_| OperatorRegistryError::LockPoisoned)?;
        let implementations = entries.entry(descriptor.name.clone()).or_default();
        if implementations.iter().any(|entry| {
            entry.descriptor.version == descriptor.version
                && entry.descriptor.backend == descriptor.backend
        }) {
            return Err(OperatorRegistryError::Duplicate {
                name: descriptor.name,
                version: descriptor.version,
                backend: descriptor.backend,
            });
        }
        implementations.push(OperatorEntry {
            descriptor,
            kernel: Arc::new(kernel),
        });
        implementations.sort_by_key(|entry| std::cmp::Reverse(entry.descriptor.priority));
        Ok(())
    }

    /// Remove one exact implementation. Returns whether an entry was removed.
    pub fn unregister(
        &self,
        name: &str,
        backend: &str,
        version: &str,
    ) -> Result<bool, OperatorRegistryError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| OperatorRegistryError::LockPoisoned)?;
        let Some(implementations) = entries.get_mut(name) else {
            return Ok(false);
        };
        let before = implementations.len();
        implementations.retain(|entry| {
            !(entry.descriptor.backend == backend && entry.descriptor.version == version)
        });
        let removed = before != implementations.len();
        if implementations.is_empty() {
            entries.remove(name);
        }
        Ok(removed)
    }

    /// Return stable metadata for all currently registered implementations.
    pub fn descriptors(&self) -> Result<Vec<OperatorDescriptor>, OperatorRegistryError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| OperatorRegistryError::LockPoisoned)?;
        let mut result = entries
            .values()
            .flat_map(|items| items.iter().map(|entry| entry.descriptor.clone()))
            .collect::<Vec<_>>();
        result.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| b.priority.cmp(&a.priority))
                .then_with(|| a.backend.cmp(&b.backend))
                .then_with(|| a.version.cmp(&b.version))
        });
        Ok(result)
    }

    /// Resolve an implementation, following fallback names if necessary.
    pub fn resolve(
        &self,
        name: &str,
        dtype: DType,
        layout: TensorLayout,
        backend: Option<&str>,
    ) -> Result<Arc<dyn OperatorKernel>, OperatorRegistryError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| OperatorRegistryError::LockPoisoned)?;
        let mut current = name.to_string();
        let mut visited = HashSet::new();
        loop {
            if !visited.insert(current.clone()) {
                return Err(OperatorRegistryError::FallbackCycle(current));
            }
            let implementations = entries
                .get(&current)
                .ok_or_else(|| OperatorRegistryError::NotFound(current.clone()))?;
            if let Some(entry) = implementations
                .iter()
                .find(|entry| entry.descriptor.supports(dtype, layout, backend))
            {
                return Ok(entry.kernel.clone());
            }
            let fallback = implementations
                .iter()
                .find_map(|entry| entry.descriptor.fallback.clone());
            let Some(fallback) = fallback else {
                return Err(OperatorRegistryError::Unsupported {
                    name: current,
                    dtype,
                    layout,
                    backend: backend.map(str::to_string),
                });
            };
            current = fallback;
        }
    }

    /// Resolve and execute an operator atomically from the caller's point of view.
    pub fn execute(
        &self,
        name: &str,
        inputs: &[OperatorTensor],
        backend: Option<&str>,
    ) -> Result<OperatorTensor, OperatorRegistryError> {
        let first = inputs.first().ok_or_else(|| {
            OperatorRegistryError::Kernel("operator requires at least one input".into())
        })?;
        let kernel = self.resolve(name, first.dtype, first.layout, backend)?;
        kernel.execute(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    impl OperatorKernel for Echo {
        fn execute(
            &self,
            inputs: &[OperatorTensor],
        ) -> Result<OperatorTensor, OperatorRegistryError> {
            inputs
                .first()
                .cloned()
                .ok_or_else(|| OperatorRegistryError::Kernel("empty".into()))
        }
    }

    fn descriptor(name: &str, backend: &str, priority: u32) -> OperatorDescriptor {
        OperatorDescriptor {
            name: name.into(),
            version: "1.0.0".into(),
            backend: backend.into(),
            supported_dtypes: vec![DType::F16],
            supported_layouts: vec![TensorLayout::RowMajor],
            priority,
            fallback: None,
        }
    }

    #[test]
    fn resolves_highest_priority_compatible_kernel() {
        let registry = OperatorRegistry::new();
        registry
            .register(descriptor("matmul", "cpu", 1), Echo)
            .unwrap();
        registry
            .register(descriptor("matmul", "metal", 10), Echo)
            .unwrap();
        let input = OperatorTensor::new(vec![1], vec![1], DType::F16, TensorLayout::RowMajor);
        let output = registry
            .execute("matmul", std::slice::from_ref(&input), Some("metal"))
            .unwrap();
        assert_eq!(output, input);
        assert_eq!(registry.descriptors().unwrap().len(), 2);
    }

    #[test]
    fn follows_fallback_for_unsupported_dtype() {
        let registry = OperatorRegistry::new();
        let mut optimized = descriptor("attention", "cpu", 20);
        optimized.supported_dtypes = vec![DType::F16];
        optimized.fallback = Some("attention_reference".into());
        registry.register(optimized, Echo).unwrap();
        let mut reference = descriptor("attention_reference", "cpu", 1);
        reference.supported_dtypes = vec![];
        reference.supported_layouts = vec![TensorLayout::Any];
        registry.register(reference, Echo).unwrap();
        let input = OperatorTensor::new(vec![1], vec![1], DType::F32, TensorLayout::Nhwc);
        assert_eq!(
            registry
                .execute("attention", std::slice::from_ref(&input), None)
                .unwrap(),
            input
        );
    }

    #[test]
    fn rejects_duplicate_and_detects_fallback_cycle() {
        let registry = OperatorRegistry::new();
        registry.register(descriptor("a", "cpu", 1), Echo).unwrap();
        assert!(matches!(
            registry.register(descriptor("a", "cpu", 1), Echo),
            Err(OperatorRegistryError::Duplicate { .. })
        ));
        let mut b = descriptor("b", "cpu", 1);
        b.fallback = Some("c".into());
        let mut c = descriptor("c", "cpu", 1);
        c.fallback = Some("b".into());
        registry.register(b, Echo).unwrap();
        registry.register(c, Echo).unwrap();
        assert!(matches!(
            registry.resolve("b", DType::F32, TensorLayout::Nhwc, None),
            Err(OperatorRegistryError::FallbackCycle(_))
        ));
    }

    #[test]
    fn layout_and_descriptor_round_trip() {
        let descriptor = descriptor("conv", "cpu", 1);
        let json = serde_json::to_string(&descriptor).unwrap();
        let restored: OperatorDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, descriptor);
        assert_eq!(
            serde_json::to_string(&TensorLayout::Nchw).unwrap(),
            "\"nchw\""
        );
    }
}
