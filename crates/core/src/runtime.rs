use crate::{DeviceKind, Modality};

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub device: DeviceKind,
    pub modality: Modality,
    pub max_tokens: usize,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            device: DeviceKind::Cpu,
            modality: Modality::Multi,
            max_tokens: 128,
        }
    }
}

#[derive(Debug, Default)]
pub struct Runtime {
    pub context: ExecutionContext,
}

impl Runtime {
    pub fn with_context(context: ExecutionContext) -> Self {
        Self { context }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceKind, Modality};

    #[test]
    fn test_execution_context_default() {
        let ctx = ExecutionContext::default();
        assert_eq!(ctx.device, DeviceKind::Cpu);
        assert_eq!(ctx.modality, Modality::Multi);
        assert_eq!(ctx.max_tokens, 128);
    }

    #[test]
    fn test_runtime_creation() {
        let rt = Runtime::default();
        assert_eq!(rt.context.device, DeviceKind::Cpu);

        let custom_ctx = ExecutionContext {
            device: DeviceKind::Npu,
            modality: Modality::Text,
            max_tokens: 256,
        };
        let custom_rt = Runtime::with_context(custom_ctx);
        assert_eq!(custom_rt.context.device, DeviceKind::Npu);
        assert_eq!(custom_rt.context.modality, Modality::Text);
        assert_eq!(custom_rt.context.max_tokens, 256);
    }
}
