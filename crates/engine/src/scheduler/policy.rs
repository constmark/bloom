//! Backend-independent long-context scheduling policy.

/// Policy for keeping long-context decode bounded before model-level kernels
/// wire in full context shifting.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LongContextPolicy {
    /// Attend to every allocated KV block.
    #[default]
    Full,
    /// Attend only to the most recent window of tokens.
    SlidingWindow { window_tokens: usize },
    /// Drop the oldest blocks once the context grows past the configured window.
    ContextShift {
        max_context_tokens: usize,
        shift_tokens: usize,
    },
    /// Keep all active blocks visible, but proactively compact inactive cache.
    CompactInactive { target_free_blocks: usize },
}
