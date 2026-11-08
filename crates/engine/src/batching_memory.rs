//! Conservative bounds for the audited CPU/F32 dense Qwen implementation.
//! Counts tensor storage and GEMM packing; it is not a process-RSS quota.
use crate::scheduler::ExecutionBatch;
use anyhow::{Result, anyhow, ensure};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct DenseMemoryPlan {
    pub template_bytes: usize,
    pub template_load_bytes: usize,
    pub gemm_thread_bytes: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    context: usize,
}
fn mul(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| anyhow!("dense memory dimensions overflow"))
    })
}
fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| anyhow!("dense memory sum overflows"))
    })
}
impl DenseMemoryPlan {
    /// Bound a validated dense Qwen CPU/F32 architecture. Runtime callers
    /// should use `dense_memory_plan`, which also verifies device/artifacts.
    pub fn cpu_qwen(config: &Value, file_bytes: usize, vocabulary_bytes: usize) -> Result<Self> {
        let get = |name: &str| {
            config[name]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| anyhow!("missing dense memory dimension {name}"))
        };
        let layers = get("num_hidden_layers")?;
        let heads = get("num_attention_heads")?;
        let kv_heads = get("num_key_value_heads")?;
        let hidden = get("hidden_size")?;
        let head_dim = get("head_dim").unwrap_or(hidden / heads);
        let intermediate = get("intermediate_size")?;
        let vocab = get("vocab_size")?;
        let context = get("max_position_embeddings")?;
        ensure!(
            head_dim > 0 && heads % kv_heads == 0,
            "unsupported dense attention layout"
        );
        // Architecture-derived F32 weights, including all optional biases,
        // both embedding/head matrices even when tied, and Qwen3 Q/K norms.
        // No weight/header allocation is needed to inspect this upper bound.
        let qkv_width = mul(&[sum(&[heads, kv_heads, kv_heads])?, head_dim])?;
        let layer_weights = sum(&[
            mul(&[hidden, qkv_width])?,
            qkv_width,
            mul(&[hidden, heads, head_dim])?,
            hidden,
            mul(&[3, hidden, intermediate])?,
            mul(&[2, intermediate])?,
            hidden,
            mul(&[2, hidden])?,
            mul(&[2, head_dim])?,
        ])?;
        let weight_elements = sum(&[
            mul(&[2, vocab, hidden])?,
            hidden,
            mul(&[layers, layer_weights])?,
        ])?;
        let weights = mul(&[weight_elements, 4])?;
        // Conservatively allow each layer its own cosine/sine tables even
        // though the current implementation shares one rotary embedding Arc.
        let rotary = mul(&[layers, context, head_dim, 4])?;
        let template_bytes = sum(&[weights, rotary, vocabulary_bytes])?;
        // mmap source + converted weights + rotary construction intermediates.
        let template_load_bytes = sum(&[file_bytes, weights, mul(&[rotary, 6])?])?;
        Ok(Self {
            template_bytes,
            template_load_bytes,
            gemm_thread_bytes: gemm_common::cache::CACHE_INFO[1].cache_bytes
                + gemm_common::gemm::CACHELINE_ALIGN,
            layers,
            heads,
            kv_heads,
            head_dim,
            hidden,
            intermediate,
            vocab,
            context,
        })
    }
    /// Native private KV plus compact paged mirror, both F32 key/value pairs.
    pub fn request_kv_bytes(&self, max_tokens: usize, block_size: usize) -> Result<usize> {
        ensure!(
            max_tokens > 0 && max_tokens <= self.context && block_size > 0,
            "dense request exceeds bounded context"
        );
        let rounded = max_tokens
            .checked_add(block_size - 1)
            .ok_or_else(|| anyhow!("KV block rounding overflows"))?
            / block_size
            * block_size;
        mul(&[2, 2, self.layers, self.kv_heads, self.head_dim, rounded, 4])
    }
    /// Bound all allocations that can overlap one native model call. The
    /// formula deliberately sums intermediates rather than relying on eager
    /// drop timing, and checks every multiplication before native allocation.
    pub fn scratch_bytes(&self, batch: &ExecutionBatch) -> Result<usize> {
        let b = batch.request_ids.len();
        ensure!(
            b > 0 && batch.cu_seqlens.len() == b + 1 && batch.start_positions.len() == b,
            "invalid dense batch layout"
        );
        let t = batch.cu_seqlens[1];
        let offset = batch.start_positions[0];
        ensure!(
            t > 0
                && batch
                    .cu_seqlens
                    .iter()
                    .enumerate()
                    .all(|(i, value)| Some(*value) == i.checked_mul(t))
                && batch
                    .start_positions
                    .iter()
                    .all(|position| *position == offset),
            "memory bound requires equal dense shapes"
        );
        let length = offset
            .checked_add(t)
            .ok_or_else(|| anyhow!("dense sequence length overflows"))?;
        ensure!(
            length <= self.context,
            "dense forward exceeds bounded context"
        );
        let bt = mul(&[b, t])?;
        ensure!(
            batch.tokens.len() == bt
                && batch.kv_handles.len() == b
                && batch.params.len() == b
                && batch.generated_tokens.len() == b,
            "dense batch arrays do not match memory dimensions"
        );
        // Packed and copy-back caches, plus a full extra replacement while
        // append/copy and paged mirror replacement overlap existing caches.
        let kv_copies = mul(&[
            3,
            2,
            self.layers,
            b,
            self.kv_heads,
            self.head_dim,
            length,
            4,
        ])?;
        let hidden = mul(&[bt, self.hidden])?;
        let q = mul(&[bt, self.heads, self.head_dim])?;
        let k = mul(&[bt, self.kv_heads, self.head_dim])?;
        // Two block RMS norms + final RMS norm: squares, division, product,
        // and residual/embedding/projection results (reduction vectors <= H).
        let norms_residuals = mul(&[24, hidden, 4])?;
        // Q/K norms, rotary rotations and products, Q/K/V projections.
        let qkv_rope = mul(&[16, sum(&[q, k, k])?, 4])?;
        // Repeated GQA K/V and their contiguous copies.
        let repeated_kv = mul(&[4, b, self.heads, self.head_dim, length, 4])?;
        // Matmul score, scaling, mask addition, softmax, GEMM inputs, and
        // reduction temporaries; mask construction is separately counted.
        let attention = mul(&[8, b, self.heads, t, length, 4])?;
        let mask = mul(&[4, b, t, length, 4])?;
        let mlp = mul(&[6, bt, self.intermediate, 4])?;
        let logits_sampling = mul(&[8, b, self.vocab, 4])?;
        // gemm 0.19's F32 kernels have MR/NR <= 64. Pack complete operands
        // with that padding, exceeding any cache-blocked packing allocation.
        // One owned Rayon thread executes every audited forward, so TLS slab
        // memory is held by the runtime lease, not leaked into a global pool.
        let m = bt
            .max(length)
            .max(self.heads)
            .checked_add(64)
            .ok_or_else(|| anyhow!("GEMM bound overflows"))?;
        let projected = mul(&[self.heads, self.head_dim])?;
        let n = self
            .hidden
            .max(projected)
            .max(self.intermediate)
            .max(self.vocab)
            .max(length)
            .checked_add(64)
            .ok_or_else(|| anyhow!("GEMM bound overflows"))?;
        let inner = self
            .hidden
            .max(projected)
            .max(self.intermediate)
            .max(self.head_dim)
            .max(length);
        let packing = mul(&[sum(&[m, n])?, inner, 4])?;
        sum(&[
            kv_copies,
            norms_residuals,
            qkv_rope,
            repeated_kv,
            attention,
            mask,
            mlp,
            logits_sampling,
            packing,
        ])
    }
}
