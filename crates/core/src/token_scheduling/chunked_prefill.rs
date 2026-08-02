//! Chunked Prefill — Aegaeon §4.1
//!
//! Splits long prompts into fixed-size chunks so prefill chunks and decode
//! tokens can be interleaved in the same forward pass. This prevents a long
//! prompt prefill from monopolizing the GPU and reduces decode queue latency.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

fn default_chunk_size() -> usize {
    512
}

/// Chunked prefill configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkedPrefillConfig {
    /// Whether chunked prefill is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum number of tokens in each prefill chunk.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Whether prefill chunks may be mixed with decode batches in one step.
    #[serde(default = "default_true")]
    pub interleave_with_decode: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            chunk_size: default_chunk_size(),
            interleave_with_decode: true,
        }
    }
}

/// Chunking state for a request undergoing chunked prefill.
#[derive(Debug, Clone)]
pub struct ChunkedPrefillState {
    /// Request ID.
    pub request_id: String,
    /// Complete prompt token sequence.
    pub prompt_tokens: Vec<u32>,
    /// Number of tokens already prefetched.
    pub filled_tokens: usize,
    /// Chunk size selected at creation time.
    pub chunk_size: usize,
    /// Whether all prefill work is complete.
    pub finished: bool,
}

impl ChunkedPrefillState {
    /// Create chunked prefill state for a new request.
    pub fn new(request_id: String, prompt_tokens: Vec<u32>, chunk_size: usize) -> Self {
        Self {
            request_id,
            prompt_tokens,
            filled_tokens: 0,
            chunk_size: chunk_size.max(1),
            finished: false,
        }
    }

    /// Return the next token chunk and its starting offset in the prompt.
    /// Returns `None` when prefill is complete.
    pub fn next_chunk(&self) -> Option<(usize, &[u32])> {
        if self.finished || self.filled_tokens >= self.prompt_tokens.len() {
            return None;
        }
        let start = self.filled_tokens;
        let end = (start + self.chunk_size).min(self.prompt_tokens.len());
        Some((start, &self.prompt_tokens[start..end]))
    }

    /// Number of tokens in the next chunk.
    pub fn next_chunk_size(&self) -> usize {
        if self.finished || self.filled_tokens >= self.prompt_tokens.len() {
            return 0;
        }
        let remaining = self.prompt_tokens.len() - self.filled_tokens;
        remaining.min(self.chunk_size)
    }

    /// Mark the current chunk complete and advance `filled_tokens`.
    /// Returns `true` when all prefill work is complete.
    pub fn advance_chunk(&mut self) -> bool {
        let chunk_len = self.next_chunk_size();
        self.filled_tokens += chunk_len;
        if self.filled_tokens >= self.prompt_tokens.len() {
            self.finished = true;
            true
        } else {
            false
        }
    }

    /// Number of tokens that remain to be prefetched.
    pub fn remaining_tokens(&self) -> usize {
        self.prompt_tokens.len().saturating_sub(self.filled_tokens)
    }

    /// Prefill progress from 0.0 to 1.0.
    pub fn progress(&self) -> f64 {
        if self.prompt_tokens.is_empty() {
            return 1.0;
        }
        self.filled_tokens as f64 / self.prompt_tokens.len() as f64
    }
}

/// Scheduler that manages chunked prefill queues for multiple requests.
#[derive(Debug)]
pub struct ChunkedPrefillQueue {
    pub config: ChunkedPrefillConfig,
    /// Requests waiting for or currently undergoing chunked prefill.
    pub queue: VecDeque<ChunkedPrefillState>,
}

impl ChunkedPrefillQueue {
    pub fn new(config: ChunkedPrefillConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
        }
    }

    /// Submit a new prefill request. When chunked prefill is disabled, the
    /// entire prompt is treated as one chunk, matching regular prefill.
    pub fn submit(&mut self, request_id: String, prompt_tokens: Vec<u32>) {
        let chunk_size = if self.config.enabled {
            self.config.chunk_size
        } else {
            prompt_tokens.len().max(1)
        };
        self.queue.push_back(ChunkedPrefillState::new(
            request_id,
            prompt_tokens,
            chunk_size,
        ));
    }

    /// Schedule as many prefill chunks as possible within the token budget.
    /// Returns scheduled `(request_id, chunk_start, chunk_tokens)` entries.
    pub fn schedule_chunks(&mut self, token_budget: usize) -> Vec<PrefillChunk> {
        let mut scheduled = Vec::new();
        let mut used = 0usize;

        // Walk the queue and greedily fill chunks.
        let mut i = 0;
        while i < self.queue.len() && used < token_budget {
            let state = &mut self.queue[i];
            let chunk_size = state.next_chunk_size();
            if chunk_size == 0 {
                i += 1;
                continue;
            }
            let can_take = (token_budget - used).min(chunk_size);
            // Schedule complete chunks only; partial chunks are not supported.
            if can_take < chunk_size {
                i += 1;
                continue;
            }

            let Some((start, tokens)) = state.next_chunk() else {
                i += 1;
                continue;
            };
            let chunk = PrefillChunk {
                request_id: state.request_id.clone(),
                chunk_start: start,
                tokens: tokens.to_vec(),
                is_final: state.filled_tokens + tokens.len() >= state.prompt_tokens.len(),
            };
            scheduled.push(chunk);
            used += chunk_size;

            let finished = state.advance_chunk();
            if finished {
                self.queue.remove(i);
                // Do not increment i; removal moves the next element into this slot.
            } else {
                i += 1;
            }
        }

        scheduled
    }

    /// Cancel prefill for a request.
    pub fn cancel(&mut self, request_id: &str) -> bool {
        if let Some(pos) = self.queue.iter().position(|s| s.request_id == request_id) {
            self.queue.remove(pos);
            return true;
        }
        false
    }

    /// Number of requests waiting in the queue.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Total number of remaining prefill tokens across queued requests.
    pub fn total_remaining_tokens(&self) -> usize {
        self.queue.iter().map(|s| s.remaining_tokens()).sum()
    }
}

/// Result of scheduling one prefill chunk.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    pub request_id: String,
    pub chunk_start: usize,
    pub tokens: Vec<u32>,
    pub is_final: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_prefill_state_basic() {
        let mut state =
            ChunkedPrefillState::new("r1".into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4);
        assert_eq!(state.next_chunk_size(), 4);
        assert_eq!(state.remaining_tokens(), 10);

        let (start, chunk) = state.next_chunk().unwrap();
        assert_eq!(start, 0);
        assert_eq!(chunk, &[1, 2, 3, 4]);

        let done = state.advance_chunk();
        assert!(!done);
        assert_eq!(state.filled_tokens, 4);

        let (start2, chunk2) = state.next_chunk().unwrap();
        assert_eq!(start2, 4);
        assert_eq!(chunk2, &[5, 6, 7, 8]);

        state.advance_chunk();
        let (start3, chunk3) = state.next_chunk().unwrap();
        assert_eq!(start3, 8);
        assert_eq!(chunk3, &[9, 10]);

        let done = state.advance_chunk();
        assert!(done);
        assert!(state.finished);
        assert!(state.next_chunk().is_none());
    }

    #[test]
    fn chunked_prefill_queue_scheduling() {
        let config = ChunkedPrefillConfig {
            enabled: true,
            chunk_size: 4,
            interleave_with_decode: true,
        };
        let mut queue = ChunkedPrefillQueue::new(config);

        // A 10-token request produces three chunks (4 + 4 + 2).
        queue.submit("r1".into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        // A 3-token request produces one chunk.
        queue.submit("r2".into(), vec![100, 200, 300]);

        // With a budget of 6, only r1 chunk 1 (4 tokens) fits; adding r2 would require 7.
        let chunks = queue.schedule_chunks(6);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].request_id, "r1");
        assert_eq!(chunks[0].tokens.len(), 4);
        assert!(!chunks[0].is_final);

        // budget = 8: r1 chunk2(4) + r2(3) = 7 <= 8
        let chunks2 = queue.schedule_chunks(8);
        assert_eq!(chunks2.len(), 2);
        assert_eq!(chunks2[0].request_id, "r1");
        assert_eq!(chunks2[0].tokens.len(), 4);
        assert!(!chunks2[0].is_final); // r1 still has two tokens in its third chunk.
        assert_eq!(chunks2[1].request_id, "r2");
        assert!(chunks2[1].is_final);
    }

    #[test]
    fn chunked_prefill_queue_cancel() {
        let config = ChunkedPrefillConfig {
            enabled: true,
            chunk_size: 4,
            ..Default::default()
        };
        let mut queue = ChunkedPrefillQueue::new(config);
        queue.submit("r1".into(), vec![1, 2, 3, 4]);
        queue.submit("r2".into(), vec![5, 6, 7, 8]);

        assert!(queue.cancel("r1"));
        assert_eq!(queue.pending_count(), 1);
        assert!(!queue.cancel("r1")); // already cancelled
    }

    #[test]
    fn disabled_chunked_prefill_uses_full_prompt() {
        let config = ChunkedPrefillConfig {
            enabled: false,
            chunk_size: 4, // should be ignored
            ..Default::default()
        };
        let mut queue = ChunkedPrefillQueue::new(config);
        queue.submit("r1".into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let chunks = queue.schedule_chunks(20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].tokens.len(), 10);
        assert!(chunks[0].is_final);
    }

    #[test]
    fn prefill_chunk_progress_tracking() {
        let mut state = ChunkedPrefillState::new("r1".into(), vec![0; 100], 30);
        assert!((state.progress() - 0.0).abs() < f64::EPSILON);

        state.advance_chunk(); // 30/100
        assert!((state.progress() - 0.3).abs() < 0.01);

        state.advance_chunk(); // 60/100
        assert!((state.progress() - 0.6).abs() < 0.01);

        state.advance_chunk(); // 90/100
        assert!((state.progress() - 0.9).abs() < 0.01);

        let done = state.advance_chunk(); // 100/100
        assert!(done);
        assert!((state.progress() - 1.0).abs() < f64::EPSILON);
    }
}
