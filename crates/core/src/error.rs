use thiserror::Error;

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ResourceError {
    #[error("insufficient RAM: requested {requested} bytes, available {available} bytes")]
    InsufficientRam { requested: usize, available: usize },

    #[error("insufficient VRAM: requested {requested} bytes, available {available} bytes")]
    InsufficientVram { requested: usize, available: usize },

    #[error(
        "insufficient unified memory: requested {requested} bytes, available {available} bytes"
    )]
    InsufficientUnifiedMemory { requested: usize, available: usize },

    #[error("resource budget exceeded after eviction: still need {deficit} bytes")]
    BudgetExceeded { deficit: usize },

    #[error("backend '{backend}' unavailable for reservation: {reason}")]
    BackendUnavailable { backend: String, reason: String },

    #[error("all backends exhausted for model '{model_id}'")]
    AllBackendsExhausted {
        model_id: String,
        tried: Vec<String>,
    },

    #[error("lease {lease_id} not found")]
    LeaseNotFound { lease_id: u64 },

    #[error("model '{model_id}' already has active lease {lease_id}")]
    AlreadyLoaded { model_id: String, lease_id: u64 },
}

impl ResourceError {
    /// Return human-readable recovery hints for each error variant.
    pub fn recovery_hints(&self) -> Vec<String> {
        match self {
            Self::InsufficientRam { .. } => vec![
                "Consider evicting lower-priority models".into(),
                "Try a smaller quantization (e.g. Q4 instead of F16)".into(),
                "Enable mmap mode to reduce physical RAM usage".into(),
            ],
            Self::InsufficientVram { .. } => vec![
                "Evict cached models from GPU memory".into(),
                "Fallback to CPU backend for this model".into(),
                "Reduce KV cache size".into(),
            ],
            Self::InsufficientUnifiedMemory { .. } => vec![
                "Close other applications to free system memory".into(),
                "Try a smaller model or quantization".into(),
            ],
            Self::BudgetExceeded { .. } => vec![
                "Unload background models to free resources".into(),
                "Increase memory budget via BLOOM_MEMORY_BUDGET env var".into(),
            ],
            Self::AllBackendsExhausted { tried, .. } => vec![
                format!("Tried backends: {}", tried.join(", ")),
                "No backend has sufficient resources".into(),
                "Consider reducing model size or using streaming/offload mode".into(),
            ],
            _ => vec![],
        }
    }
}

impl BloomError {
    /// Return recovery hints for each BloomError variant.
    pub fn recovery_hints(&self) -> Vec<String> {
        match self {
            Self::RoutingFailed(_) => vec![
                "Check that at least one engine supports the model family and dtype".into(),
                "Verify backend availability with `bloom health`".into(),
            ],
            Self::SchedulingFailed(_) => vec![
                "Check device thermal and power state".into(),
                "Reduce concurrent requests or batch size".into(),
            ],
            Self::Timeout(msg) => vec![
                format!("Timed out: {}", msg),
                "Increase timeout or reduce model size".into(),
            ],
            Self::BackendProbe(_) => vec![
                "Check hardware drivers and runtime dependencies".into(),
                "Ensure correct feature flags are enabled at compile time".into(),
            ],
            Self::UnsupportedFamily(_) => vec![
                "Check available engines for this model family".into(),
                "Consider converting the model to a supported format".into(),
            ],
            Self::Resource(re) => re.recovery_hints(),
            _ => vec![],
        }
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum BloomError {
    #[error("unsupported modality: {0}")]
    UnsupportedModality(String),
    #[error("unsupported device: {0}")]
    UnsupportedDevice(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("engine error: {0}")]
    Engine(String),
    #[error("model load error: {0}")]
    ModelLoad(String),
    #[error("missing required file: {0}")]
    MissingRequiredFile(String),
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("missing license for model: {0}")]
    MissingLicense(String),
    #[error("backend mismatch: {0}")]
    BackendMismatch(String),
    #[error("hash mismatch for file: {0}")]
    HashMismatch(String),
    #[error("resource error: {0}")]
    Resource(ResourceError),
    #[error("routing failed: {0}")]
    RoutingFailed(String),
    #[error("scheduling failed: {0}")]
    SchedulingFailed(String),
    #[error("operation timed out: {0}")]
    Timeout(String),
    #[error("backend probe failed: {0}")]
    BackendProbe(String),
    #[error("unsupported model family: {0}")]
    UnsupportedFamily(String),
    #[error("plugin error: {0}")]
    Plugin(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ErrorCategory {
    Model,
    Backend,
    Resource,
    Format,
    Protocol,
    Plugin,
    Runtime,
}

impl BloomError {
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::UnsupportedModality(_) | Self::UnsupportedFormat(_) => ErrorCategory::Format,
            Self::UnsupportedDevice(_) | Self::BackendMismatch(_) | Self::BackendProbe(_) => {
                ErrorCategory::Backend
            }
            Self::InvalidInput(_) => ErrorCategory::Protocol,
            Self::ModelLoad(_)
            | Self::MissingRequiredFile(_)
            | Self::MissingLicense(_)
            | Self::HashMismatch(_)
            | Self::UnsupportedFamily(_) => ErrorCategory::Model,
            Self::Resource(_) => ErrorCategory::Resource,
            Self::Plugin(_) => ErrorCategory::Plugin,
            Self::RoutingFailed(_)
            | Self::SchedulingFailed(_)
            | Self::Timeout(_)
            | Self::Runtime(_)
            | Self::Engine(_) => ErrorCategory::Runtime,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_error_formatting() {
        let err1 = BloomError::UnsupportedModality("Audio".to_string());
        assert_eq!(err1.to_string(), "unsupported modality: Audio");

        let err2 = BloomError::UnsupportedDevice("GPU".to_string());
        assert_eq!(err2.to_string(), "unsupported device: GPU");

        let err3 = BloomError::InvalidInput("Empty prompt".to_string());
        assert_eq!(err3.to_string(), "invalid input: Empty prompt");

        let err4 = BloomError::Runtime("NPU execution failed".to_string());
        assert_eq!(err4.to_string(), "runtime error: NPU execution failed");

        let err5 = BloomError::Engine("Initialization failed".to_string());
        assert_eq!(err5.to_string(), "engine error: Initialization failed");

        let err6 = BloomError::ModelLoad("File not found".to_string());
        assert_eq!(err6.to_string(), "model load error: File not found");
    }

    #[test]
    fn test_bloom_error_equality() {
        let err1 = BloomError::Runtime("error".to_string());
        let err2 = BloomError::Runtime("error".to_string());
        let err3 = BloomError::Runtime("different error".to_string());

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_resource_error_display() {
        let e1 = ResourceError::InsufficientRam {
            requested: 1024,
            available: 512,
        };
        assert!(e1.to_string().contains("1024"));
        assert!(e1.to_string().contains("512"));

        let e2 = ResourceError::AllBackendsExhausted {
            model_id: "m1".into(),
            tried: vec!["metal".into(), "cpu".into()],
        };
        assert!(e2.to_string().contains("m1"));
    }

    #[test]
    fn test_resource_error_recovery_hints() {
        let e1 = ResourceError::InsufficientRam {
            requested: 1024,
            available: 512,
        };
        let hints = e1.recovery_hints();
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.contains("mmap")));

        let e2 = ResourceError::InsufficientVram {
            requested: 2048,
            available: 1024,
        };
        assert!(e2.recovery_hints().iter().any(|h| h.contains("CPU")));

        let e3 = ResourceError::AllBackendsExhausted {
            model_id: "m".into(),
            tried: vec!["cpu".into()],
        };
        assert!(e3.recovery_hints().iter().any(|h| h.contains("cpu")));
    }

    #[test]
    fn test_resource_error_into_bloom_error() {
        let re = ResourceError::InsufficientRam {
            requested: 100,
            available: 50,
        };
        let be = BloomError::Resource(re.clone());
        assert!(be.to_string().contains("resource error"));
        assert_eq!(be, BloomError::Resource(re));
    }

    #[test]
    fn test_new_bloom_error_variants() {
        let err1 = BloomError::RoutingFailed("no engine for Qwen/F16".into());
        assert_eq!(err1.to_string(), "routing failed: no engine for Qwen/F16");
        assert!(!err1.recovery_hints().is_empty());

        let err2 = BloomError::SchedulingFailed("all devices busy".into());
        assert_eq!(err2.to_string(), "scheduling failed: all devices busy");

        let err3 = BloomError::Timeout("inference exceeded 30s".into());
        assert!(err3.to_string().contains("timed out"));

        let err4 = BloomError::BackendProbe("NPU not detected".into());
        assert_eq!(err4.to_string(), "backend probe failed: NPU not detected");

        let err5 = BloomError::UnsupportedFamily("Falcon".into());
        assert_eq!(err5.to_string(), "unsupported model family: Falcon");
    }

    #[test]
    fn test_bloom_error_recovery_hints_coverage() {
        // Verify all new variants produce hints
        let variants: Vec<BloomError> = vec![
            BloomError::RoutingFailed("test".into()),
            BloomError::SchedulingFailed("test".into()),
            BloomError::Timeout("test".into()),
            BloomError::BackendProbe("test".into()),
            BloomError::UnsupportedFamily("test".into()),
            BloomError::Resource(ResourceError::InsufficientRam {
                requested: 100,
                available: 50,
            }),
        ];
        for variant in &variants {
            let hints = variant.recovery_hints();
            assert!(
                !hints.is_empty(),
                "BloomError::{:?} should have recovery hints",
                variant
            );
        }
    }

    #[test]
    fn test_bloom_error_categories() {
        assert_eq!(
            BloomError::UnsupportedModality("Audio".into()).category(),
            ErrorCategory::Format
        );
        assert_eq!(
            BloomError::UnsupportedDevice("GPU".into()).category(),
            ErrorCategory::Backend
        );
        assert_eq!(
            BloomError::InvalidInput("empty".into()).category(),
            ErrorCategory::Protocol
        );
        assert_eq!(
            BloomError::ModelLoad("fail".into()).category(),
            ErrorCategory::Model
        );
        assert_eq!(
            BloomError::Plugin("err".into()).category(),
            ErrorCategory::Plugin
        );
        assert_eq!(
            BloomError::RoutingFailed("fail".into()).category(),
            ErrorCategory::Runtime
        );
    }
}
