//! Paged attention cache layer that bridges `BloomKvCachePool` block allocation
//! with the model's actual attention computation.
//!
//! This module provides `PagedAttentionCache` which manages per-layer KV tensors
//! indexed by block IDs from the pool, supporting:
//! - Block-level gather/scatter for attention computation
//! - Prefix caching with matched token skipping
//! - LRU eviction when pool is full
//! - Optional INT8/FP8 KV cache quantization

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::cachemesh::{CacheMesh, CacheMeshBlock, CacheMeshKey, CacheMeshMetrics};
use crate::core::quantization::{Int8QuantizedKv, KvCacheDtype};
use crate::scheduler::{BloomKvCachePool, KvCacheAllocation, KvCacheMetrics, KvCachePool};

use candle_core::Tensor;

/// Per-layer KV cache state for a single block.
#[derive(Debug, Clone)]
pub struct BlockKvData {
    /// Key tensor data (F16 or quantized).
    pub keys: Vec<f32>,
    /// Value tensor data (F16 or quantized).
    pub values: Vec<f32>,
    /// Optional INT8 quantized keys (when kv_dtype is Int8).
    pub quantized_keys: Option<Int8QuantizedKv>,
    /// Optional INT8 quantized values.
    pub quantized_values: Option<Int8QuantizedKv>,
    /// Number of valid tokens in this block.
    pub num_tokens: usize,
    /// GPU keys Tensor (if supported by backend).
    pub keys_tensor: Option<Tensor>,
    /// GPU values Tensor (if supported by backend).
    pub values_tensor: Option<Tensor>,
}

/// Configuration for the paged attention cache.
#[derive(Debug, Clone)]
pub struct PagedCacheConfig {
    /// Block size (tokens per block).
    pub block_size: usize,
    /// Total number of blocks in the pool.
    pub total_blocks: usize,
    /// Number of attention layers in the model.
    pub num_layers: usize,
    /// Key/value dimension per head (head_dim * num_kv_heads).
    pub kv_dim: usize,
    /// Storage dtype for KV cache.
    pub kv_dtype: KvCacheDtype,
    /// Long-context policy used when selecting blocks for attention.
    pub long_context_policy: LongContextPolicy,
}

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

impl LongContextPolicy {
    fn window_blocks(&self, block_size: usize, total_blocks: usize) -> usize {
        match *self {
            Self::Full | Self::CompactInactive { .. } => total_blocks,
            Self::SlidingWindow { window_tokens } => tokens_to_blocks(window_tokens, block_size),
            Self::ContextShift {
                max_context_tokens,
                shift_tokens,
            } => {
                let retained_tokens = max_context_tokens.saturating_sub(shift_tokens);
                tokens_to_blocks(retained_tokens.max(block_size), block_size)
            }
        }
    }
}

impl Default for PagedCacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            total_blocks: 512,
            num_layers: 28,
            kv_dim: 128,
            kv_dtype: KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::Full,
        }
    }
}

/// Paged attention cache that manages per-layer KV tensors indexed by block IDs.
///
/// This wraps `BloomKvCachePool` for block allocation and adds actual KV tensor
/// storage and access methods for the attention computation.
pub struct PagedAttentionCache {
    /// Underlying block pool for allocation/eviction.
    pool: Arc<BloomKvCachePool>,
    /// Configuration.
    config: PagedCacheConfig,
    /// Per-layer, per-block KV data.
    /// Indexed as: layer_data[layer_idx][block_id] = BlockKvData
    layer_data: Arc<Mutex<Vec<HashMap<usize, BlockKvData>>>>,
    /// Stable CacheMesh keys for each layer/block currently assigned by the pool.
    block_keys: Arc<Mutex<HashMap<(usize, usize), CacheMeshKey>>>,
    /// Optional L2/L3 cachemesh backing store.
    cachemesh: Option<Arc<CacheMesh>>,
}

impl PagedAttentionCache {
    /// Create a new paged attention cache with the given configuration.
    pub fn new(config: PagedCacheConfig) -> Self {
        let pool = Arc::new(BloomKvCachePool::new(
            config.block_size,
            config.total_blocks,
        ));
        Self::from_pool(pool, config)
    }

    /// Create a new paged attention cache with cachemesh enabled.
    pub fn with_cachemesh(config: PagedCacheConfig, cachemesh: Arc<CacheMesh>) -> Self {
        let pool = Arc::new(BloomKvCachePool::new(
            config.block_size,
            config.total_blocks,
        ));
        Self::from_pool_and_cachemesh(pool, config, Some(cachemesh))
    }

    /// Create from an existing pool (shared with other components).
    pub fn from_pool(pool: Arc<BloomKvCachePool>, config: PagedCacheConfig) -> Self {
        Self::from_pool_and_cachemesh(pool, config, None)
    }

    /// Create from an existing pool and optional cachemesh backing store.
    pub fn from_pool_and_cachemesh(
        pool: Arc<BloomKvCachePool>,
        config: PagedCacheConfig,
        cachemesh: Option<Arc<CacheMesh>>,
    ) -> Self {
        let layer_data = Arc::new(Mutex::new(
            (0..config.num_layers).map(|_| HashMap::new()).collect(),
        ));
        let block_keys = Arc::new(Mutex::new(HashMap::new()));
        install_cachemesh_eviction_hook(
            &pool,
            layer_data.clone(),
            block_keys.clone(),
            cachemesh.clone(),
            config.clone(),
        );
        Self {
            pool,
            config,
            layer_data,
            block_keys,
            cachemesh,
        }
    }

    /// Allocate blocks for a new request, returning the allocation with matched prefix info.
    pub fn allocate(
        &self,
        request_id: &str,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<KvCacheAllocation> {
        self.allocate_multimodal(request_id, prompt_tokens, max_new_tokens, None)
    }

    pub fn allocate_multimodal(
        &self,
        request_id: &str,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        multimodal_hash: Option<&str>,
    ) -> Result<KvCacheAllocation> {
        let allocation =
            self.pool
                .allocate_paged(request_id, prompt_tokens, max_new_tokens, multimodal_hash)?;
        self.register_block_keys(
            request_id,
            prompt_tokens,
            &allocation.allocated_blocks,
            multimodal_hash,
        );
        Ok(allocation)
    }

    /// Free a request's blocks, making them available for reuse/eviction.
    pub fn free(&self, request_id: &str) {
        self.pool.free_paged(request_id);
    }

    /// Write KV data into specific blocks for a given layer.
    pub fn write_kv(
        &self,
        layer_idx: usize,
        block_id: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
        num_tokens: usize,
    ) -> Result<()> {
        if layer_idx >= self.config.num_layers {
            return Err(anyhow!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.config.num_layers
            ));
        }
        if num_tokens > self.config.block_size {
            return Err(anyhow!(
                "num_tokens {} > block_size {}",
                num_tokens,
                self.config.block_size
            ));
        }
        let expected = num_tokens.saturating_mul(self.config.kv_dim);
        if keys.len() != expected || values.len() != expected {
            return Err(anyhow!(
                "KV block shape mismatch: expected {} elements per tensor, got keys={}, values={}",
                expected,
                keys.len(),
                values.len()
            ));
        }

        let (stored_keys, stored_values, quant_keys, quant_values) =
            if self.config.kv_dtype.needs_dequant() {
                // Quantize before storing
                let qk = Int8QuantizedKv::quantize_f32(&keys, num_tokens, self.config.kv_dim);
                let qv = Int8QuantizedKv::quantize_f32(&values, num_tokens, self.config.kv_dim);
                (Vec::new(), Vec::new(), Some(qk), Some(qv))
            } else {
                (keys, values, None, None)
            };

        let block_data = BlockKvData {
            keys: stored_keys,
            values: stored_values,
            quantized_keys: quant_keys,
            quantized_values: quant_values,
            num_tokens,
            keys_tensor: None,
            values_tensor: None,
        };

        let mut layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
        layer_data[layer_idx].insert(block_id, block_data);
        Ok(())
    }

    /// Write KV data directly as GPU Tensors.
    pub fn write_kv_tensor(
        &self,
        layer_idx: usize,
        block_id: usize,
        keys: Tensor,
        values: Tensor,
        num_tokens: usize,
    ) -> Result<()> {
        if layer_idx >= self.config.num_layers {
            return Err(anyhow!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.config.num_layers
            ));
        }
        if num_tokens > self.config.block_size {
            return Err(anyhow!(
                "num_tokens {} > block_size {}",
                num_tokens,
                self.config.block_size
            ));
        }

        // Convert to F32 CPU vector for fallback compatibility
        let keys_f32 = keys.to_dtype(candle_core::DType::F32)?.flatten_all()?;
        let values_f32 = values.to_dtype(candle_core::DType::F32)?.flatten_all()?;
        let keys_vec = keys_f32.to_vec1::<f32>()?;
        let values_vec = values_f32.to_vec1::<f32>()?;

        let block_data = BlockKvData {
            keys: keys_vec,
            values: values_vec,
            quantized_keys: None,
            quantized_values: None,
            num_tokens,
            keys_tensor: Some(keys),
            values_tensor: Some(values),
        };

        let mut layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
        layer_data[layer_idx].insert(block_id, block_data);
        Ok(())
    }

    /// Read KV data as GPU Tensors.
    pub fn read_kv_tensor(
        &self,
        layer_idx: usize,
        block_id: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if layer_idx >= self.config.num_layers {
            return Err(anyhow!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.config.num_layers
            ));
        }

        let layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(block) = layer_data[layer_idx].get(&block_id)
            && let (Some(k), Some(v)) = (&block.keys_tensor, &block.values_tensor)
        {
            return Ok(Some((k.clone(), v.clone())));
        }
        Ok(None)
    }

    /// Read KV data from blocks for a given layer, gathering into a contiguous tensor.
    ///
    /// Returns `(keys, values)` as flat F32 vectors.
    pub fn read_kv(&self, layer_idx: usize, block_ids: &[usize]) -> Result<(Vec<f32>, Vec<f32>)> {
        if layer_idx >= self.config.num_layers {
            return Err(anyhow!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.config.num_layers
            ));
        }

        let mut all_keys = Vec::new();
        let mut all_values = Vec::new();

        for &block_id in &self.visible_blocks(block_ids) {
            match self.get_l1_block(layer_idx, block_id) {
                Some(block) => {
                    if let Some(mesh) = &self.cachemesh {
                        mesh.record_l1_lookup(true);
                    }
                    if self.config.kv_dtype.needs_dequant() {
                        // Dequantize on read
                        if let (Some(qk), Some(qv)) =
                            (&block.quantized_keys, &block.quantized_values)
                        {
                            all_keys.extend(qk.dequantize_f32());
                            all_values.extend(qv.dequantize_f32());
                        }
                    } else {
                        all_keys.extend(&block.keys);
                        all_values.extend(&block.values);
                    }
                }
                _ => {
                    if let Some(block) = self.restore_from_cachemesh(layer_idx, block_id)? {
                        all_keys.extend(&block.keys);
                        all_values.extend(&block.values);
                    } else {
                        if let Some(mesh) = &self.cachemesh
                            && self.cachemesh_key(layer_idx, block_id).is_none()
                        {
                            mesh.record_l1_lookup(false);
                        }
                        // Block not yet populated — fill with zeros
                        let block_tokens = self.config.block_size;
                        let block_elements = block_tokens * self.config.kv_dim;
                        all_keys.extend(std::iter::repeat_n(0.0f32, block_elements));
                        all_values.extend(std::iter::repeat_n(0.0f32, block_elements));
                    }
                }
            }
        }

        Ok((all_keys, all_values))
    }

    /// Runs paged attention using TileLang compiler/kernel.
    pub fn paged_attention(
        &self,
        q: &[f32],
        layer_idx: usize,
        block_ids: &[usize],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        use bloomai_tilelang::{TileLangCompiler, TileLangKernel};

        // 1. Read K and V from the paged cache blocks
        let (k, v) = self.read_kv(layer_idx, block_ids)?;

        // Ensure we only use the valid seq_len elements of k and v
        let expected_elements = seq_len * head_dim;
        if k.len() < expected_elements || v.len() < expected_elements {
            anyhow::bail!(
                "Paged cache blocks contain fewer elements than expected for seq_len {} and head_dim {} (K: {}, V: {})",
                seq_len,
                head_dim,
                k.len(),
                v.len()
            );
        }

        // 2. Initialize compiler and compile the kernel for the specific size
        let compiler = TileLangCompiler::new()?;
        let so_path = compiler.compile_attention(seq_len, head_dim)?;
        let kernel = unsafe { TileLangKernel::load(&so_path)? };

        // 3. Execute the kernel
        let ret = kernel.attention(
            q,
            &k[..expected_elements],
            &v[..expected_elements],
            output,
            seq_len,
            head_dim,
        )?;
        if ret != 0 {
            anyhow::bail!(
                "TileLang attention kernel execution failed with error code {}",
                ret
            );
        }

        Ok(())
    }

    /// Return the KV blocks visible to an attention step after applying the
    /// configured long-context policy.
    pub fn visible_blocks(&self, block_ids: &[usize]) -> Vec<usize> {
        let max_blocks = self
            .config
            .long_context_policy
            .window_blocks(self.config.block_size, block_ids.len());
        if max_blocks >= block_ids.len() {
            block_ids.to_vec()
        } else {
            block_ids[block_ids.len() - max_blocks..].to_vec()
        }
    }

    /// Apply policy maintenance that can run outside the model forward path.
    /// Returns the number of cache blocks made free by compaction.
    pub fn maintain_long_context(&self) -> usize {
        match self.config.long_context_policy {
            LongContextPolicy::CompactInactive { target_free_blocks } => {
                self.pool.compact_inactive(target_free_blocks)
            }
            _ => 0,
        }
    }

    /// Get current cache metrics.
    pub fn metrics(&self) -> KvCacheMetrics {
        self.pool.get_metrics()
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &PagedCacheConfig {
        &self.config
    }

    /// Get a reference to the underlying block pool.
    pub fn pool(&self) -> &Arc<BloomKvCachePool> {
        &self.pool
    }

    /// Resolve the block id that owns `token_pos` for the request identified by
    /// `handle`. Returns `None` when the handle is unknown to the pool or the
    /// position is outside the allocated range. Used by the batch executor's
    /// KV bridge to route model KV into the correct paged-cache block without
    /// leaking the pool's internal allocation table.
    pub fn block_for_handle(&self, handle: usize, token_pos: usize) -> Option<usize> {
        self.pool.block_for_handle(handle, token_pos)
    }

    /// Return CacheMesh metrics when an L2/L3 backing store is configured.
    pub fn cachemesh_metrics(&self) -> Option<CacheMeshMetrics> {
        self.cachemesh.as_ref().map(|mesh| mesh.metrics())
    }

    /// Offload a specific L1 block into cachemesh L2.
    pub fn offload_block(&self, layer_idx: usize, block_id: usize) -> Result<bool> {
        let Some(mesh) = &self.cachemesh else {
            return Ok(false);
        };
        let Some(key) = self.cachemesh_key(layer_idx, block_id) else {
            return Ok(false);
        };
        let block = {
            let mut layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
            if layer_idx >= layer_data.len() {
                return Err(anyhow!(
                    "layer_idx {} >= num_layers {}",
                    layer_idx,
                    self.config.num_layers
                ));
            }
            layer_data[layer_idx].remove(&block_id)
        };
        let Some(block) = block else {
            return Ok(false);
        };
        mesh.offload_to_l2(to_cachemesh_block(key, &block, &self.config))?;
        Ok(true)
    }

    /// Number of blocks currently stored for a specific layer.
    pub fn layer_block_count(&self, layer_idx: usize) -> usize {
        let layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
        if layer_idx < layer_data.len() {
            layer_data[layer_idx].len()
        } else {
            0
        }
    }

    /// Total memory footprint of the KV cache in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        let layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
        let element_size = self.config.kv_dtype.element_size();
        let mut total = 0usize;

        for layer_map in layer_data.iter() {
            for block in layer_map.values() {
                if self.config.kv_dtype.needs_dequant() {
                    if let (Some(qk), Some(qv)) = (&block.quantized_keys, &block.quantized_values) {
                        total += qk.memory_bytes() + qv.memory_bytes();
                    }
                } else {
                    total += (block.keys.len() + block.values.len()) * element_size;
                }
            }
        }

        total
    }

    /// Perform a paged attention forward pass.
    ///
    /// Gathers KV data from the specified blocks, dequantizes if needed,
    /// and computes scaled dot-product attention with the given query tensor.
    ///
    /// # Arguments
    /// * `layer_idx` - The attention layer index.
    /// * `query` - Query tensor as a flat f32 vector of shape `[seq_len, kv_dim]`.
    /// * `block_ids` - Block IDs to gather KV from.
    /// * `scale` - Attention scale factor (typically `1.0 / sqrt(head_dim)`).
    ///
    /// # Returns
    /// Attention output as a flat f32 vector of shape `[seq_len, kv_dim]`.
    pub fn paged_attention_forward(
        &self,
        layer_idx: usize,
        query: &[f32],
        block_ids: &[usize],
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Gather KV from blocks (dequantizes automatically if needed)
        let (keys, values) = self.read_kv(layer_idx, block_ids)?;

        let kv_dim = self.config.kv_dim;
        let visible_block_ids = self.visible_blocks(block_ids);
        let num_kv_tokens = visible_block_ids.len() * self.config.block_size;

        if keys.len() != num_kv_tokens * kv_dim {
            return Err(anyhow!(
                "KV size mismatch: expected {}, got {}",
                num_kv_tokens * kv_dim,
                keys.len()
            ));
        }

        let seq_len = query.len() / kv_dim;
        if query.len() != seq_len * kv_dim {
            return Err(anyhow!(
                "query size {} not divisible by kv_dim {}",
                query.len(),
                kv_dim
            ));
        }

        // Scaled dot-product attention: softmax(Q * K^T * scale) * V
        let mut output = vec![0.0f32; seq_len * kv_dim];

        for q_idx in 0..seq_len {
            let q = &query[q_idx * kv_dim..(q_idx + 1) * kv_dim];

            // Compute attention scores: Q * K^T * scale
            let mut scores = vec![0.0f32; num_kv_tokens];
            for k_idx in 0..num_kv_tokens {
                let k = &keys[k_idx * kv_dim..(k_idx + 1) * kv_dim];
                let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                scores[k_idx] = dot * scale;
            }

            // Softmax over scores
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let attn_weights: Vec<f32> = if sum_exp > 0.0 {
                exp_scores.iter().map(|&e| e / sum_exp).collect()
            } else {
                vec![0.0; num_kv_tokens]
            };

            // Weighted sum of values
            for d in 0..kv_dim {
                let mut val = 0.0f32;
                for v_idx in 0..num_kv_tokens {
                    val += attn_weights[v_idx] * values[v_idx * kv_dim + d];
                }
                output[q_idx * kv_dim + d] = val;
            }
        }

        Ok(output)
    }

    /// Runs paged attention using GPU tensors, returning the computed attention output.
    pub fn paged_attention_forward_tensor(
        &self,
        layer_idx: usize,
        query: &Tensor,
        block_ids: &[usize],
        scale: f64,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if layer_idx >= self.config.num_layers {
            return Err(anyhow!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.config.num_layers
            ));
        }

        let device = query.device();
        let dtype = query.dtype();
        let visible_block_ids = self.visible_blocks(block_ids);

        let mut keys_tensors = Vec::with_capacity(visible_block_ids.len());
        let mut values_tensors = Vec::with_capacity(visible_block_ids.len());

        let q_dims = query.dims();
        let batch = q_dims[0];
        let head_dim = q_dims[3];

        let num_kv_heads = self.config.kv_dim / head_dim;

        for &block_id in &visible_block_ids {
            let mut got_tensor = false;
            {
                let layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(block) = layer_data[layer_idx].get(&block_id)
                    && let (Some(k), Some(v)) = (&block.keys_tensor, &block.values_tensor)
                {
                    keys_tensors.push(k.clone());
                    values_tensors.push(v.clone());
                    got_tensor = true;
                }
            }

            if !got_tensor {
                // Construct tensor on device from CPU fallback keys/values
                let (keys_vec, values_vec) = self.read_kv(layer_idx, &[block_id])?;
                let shape = (1usize, num_kv_heads, self.config.block_size, head_dim);
                let k_t = Tensor::from_slice(&keys_vec, shape, device)?.to_dtype(dtype)?;
                let v_t = Tensor::from_slice(&values_vec, shape, device)?.to_dtype(dtype)?;
                keys_tensors.push(k_t);
                values_tensors.push(v_t);
            }
        }

        if keys_tensors.is_empty() {
            // Return zeros of appropriate shape
            let k_shape = (batch, num_kv_heads, 0, head_dim);
            let v_shape = (batch, num_kv_heads, 0, head_dim);
            let k = Tensor::zeros(k_shape, dtype, device)?;
            let v = Tensor::zeros(v_shape, dtype, device)?;
            return self.sdpa(query, &k, &v, scale, attn_mask);
        }

        // Concatenate along the sequence dimension (dim 2)
        let k = Tensor::cat(&keys_tensors, 2)?;
        let v = Tensor::cat(&values_tensors, 2)?;

        // Run SDPA
        self.sdpa(query, &k, &v, scale, attn_mask)
    }

    fn sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q_dims = q.dims();
        let num_heads = q_dims[1];
        let num_kv_heads = k.dim(1)?;
        let num_kv_groups = num_heads / num_kv_heads;

        let k = if num_kv_groups > 1 {
            repeat_kv(k.clone(), num_kv_groups)?
        } else {
            k.clone()
        };
        let v = if num_kv_groups > 1 {
            repeat_kv(v.clone(), num_kv_groups)?
        } else {
            v.clone()
        };

        #[cfg(feature = "flash-attn")]
        {
            let is_cuda = q.device().is_cuda();
            let is_metal = q.device().is_metal();
            if (is_cuda || is_metal) && attn_mask.is_none() {
                let causal = q_dims[2] > 1;
                if let Ok(res) = candle_flash_attn::flash_attn(q, &k, &v, scale as f32, causal) {
                    return Ok(res.transpose(1, 2)?);
                }
            }
        }

        // Standard scaled dot-product attention
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        Ok(ctx)
    }

    fn register_block_keys(
        &self,
        request_id: &str,
        prompt_tokens: &[u32],
        block_ids: &[usize],
        multimodal_hash: Option<&str>,
    ) {
        let namespace = self
            .cachemesh
            .as_ref()
            .map(|mesh| mesh.config().namespace.as_str())
            .unwrap_or("default");
        let mut block_keys = self.block_keys.lock().unwrap_or_else(|e| e.into_inner());
        for (block_idx, &block_id) in block_ids.iter().enumerate() {
            for layer_idx in 0..self.config.num_layers {
                let prefix_len = (block_idx + 1) * self.config.block_size;
                let key = if prefix_len <= prompt_tokens.len() {
                    CacheMeshKey::from_tokens_multimodal(
                        namespace,
                        layer_idx,
                        &prompt_tokens[..prefix_len],
                        multimodal_hash,
                    )
                } else {
                    CacheMeshKey::from_request_block(namespace, layer_idx, request_id, block_idx)
                };
                block_keys.insert((layer_idx, block_id), key);
            }
        }
    }

    fn cachemesh_key(&self, layer_idx: usize, block_id: usize) -> Option<CacheMeshKey> {
        self.block_keys
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(layer_idx, block_id))
            .cloned()
    }

    fn get_l1_block(&self, layer_idx: usize, block_id: usize) -> Option<BlockKvData> {
        let layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
        layer_data
            .get(layer_idx)
            .and_then(|layer| layer.get(&block_id).cloned())
    }

    fn restore_from_cachemesh(
        &self,
        layer_idx: usize,
        block_id: usize,
    ) -> Result<Option<CacheMeshBlock>> {
        let Some(mesh) = &self.cachemesh else {
            return Ok(None);
        };
        let Some(key) = self.cachemesh_key(layer_idx, block_id) else {
            return Ok(None);
        };
        mesh.record_l1_lookup(false);
        let Some((_tier, block)) = mesh.restore(&key)? else {
            return Ok(None);
        };
        block.validate()?;

        let block_data = from_cachemesh_block(&block, &self.config);
        let mut layer_data = self.layer_data.lock().unwrap_or_else(|e| e.into_inner());
        layer_data[layer_idx].insert(block_id, block_data);
        Ok(Some(block))
    }
}

fn install_cachemesh_eviction_hook(
    pool: &Arc<BloomKvCachePool>,
    layer_data: Arc<Mutex<Vec<HashMap<usize, BlockKvData>>>>,
    block_keys: Arc<Mutex<HashMap<(usize, usize), CacheMeshKey>>>,
    cachemesh: Option<Arc<CacheMesh>>,
    config: PagedCacheConfig,
) {
    let Some(mesh) = cachemesh else {
        return;
    };
    pool.set_on_evict(move |block_ids| {
        let mut layer_data = layer_data.lock().unwrap_or_else(|e| e.into_inner());
        let block_keys = block_keys.lock().unwrap_or_else(|e| e.into_inner());
        for &block_id in block_ids {
            for layer_idx in 0..config.num_layers {
                let Some(block) = layer_data[layer_idx].remove(&block_id) else {
                    continue;
                };
                let Some(key) = block_keys.get(&(layer_idx, block_id)).cloned() else {
                    continue;
                };
                let _ = mesh.offload_to_l2(to_cachemesh_block(key, &block, &config));
            }
        }
    });
}

fn to_cachemesh_block(
    key: CacheMeshKey,
    block: &BlockKvData,
    config: &PagedCacheConfig,
) -> CacheMeshBlock {
    let (keys, values) = if config.kv_dtype.needs_dequant() {
        let keys = block
            .quantized_keys
            .as_ref()
            .map(|q| q.dequantize_f32())
            .unwrap_or_default();
        let values = block
            .quantized_values
            .as_ref()
            .map(|q| q.dequantize_f32())
            .unwrap_or_default();
        (keys, values)
    } else {
        (block.keys.clone(), block.values.clone())
    };
    CacheMeshBlock {
        key,
        block_size: config.block_size,
        kv_dim: config.kv_dim,
        keys,
        values,
        num_tokens: block.num_tokens,
    }
}

fn from_cachemesh_block(block: &CacheMeshBlock, config: &PagedCacheConfig) -> BlockKvData {
    if config.kv_dtype.needs_dequant() {
        BlockKvData {
            keys: Vec::new(),
            values: Vec::new(),
            quantized_keys: Some(Int8QuantizedKv::quantize_f32(
                &block.keys,
                block.num_tokens,
                config.kv_dim,
            )),
            quantized_values: Some(Int8QuantizedKv::quantize_f32(
                &block.values,
                block.num_tokens,
                config.kv_dim,
            )),
            num_tokens: block.num_tokens,
            keys_tensor: None,
            values_tensor: None,
        }
    } else {
        BlockKvData {
            keys: block.keys.clone(),
            values: block.values.clone(),
            quantized_keys: None,
            quantized_values: None,
            num_tokens: block.num_tokens,
            keys_tensor: None,
            values_tensor: None,
        }
    }
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
        let res = xs
            .unsqueeze(2)?
            .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(res)
    }
}

fn tokens_to_blocks(tokens: usize, block_size: usize) -> usize {
    if tokens == 0 || block_size == 0 {
        0
    } else {
        tokens.div_ceil(block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cachemesh::{CacheMeshConfig, InMemoryRemoteCache};
    use candle_core::Device;
    use std::sync::Arc;

    fn test_config() -> PagedCacheConfig {
        PagedCacheConfig {
            block_size: 4,
            total_blocks: 32,
            num_layers: 2,
            kv_dim: 8,
            kv_dtype: KvCacheDtype::F16,
            long_context_policy: LongContextPolicy::Full,
        }
    }

    #[test]
    fn test_paged_cache_allocate_and_free() {
        let cache = PagedAttentionCache::new(test_config());
        let alloc = cache
            .allocate("req-1", &[1, 2, 3, 4, 5, 6, 7, 8], 10)
            .unwrap();
        assert!(!alloc.allocated_blocks.is_empty());
        assert_eq!(alloc.matched_tokens, 0);

        cache.free("req-1");
        let metrics = cache.metrics();
        assert!(metrics.active_blocks == 0 || metrics.cached_blocks > 0);
    }

    #[test]
    fn test_paged_cache_write_read() {
        let cache = PagedAttentionCache::new(test_config());
        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();

        let block_id = alloc.allocated_blocks[0];
        let keys = vec![1.0f32; 4 * 8]; // 4 tokens x 8 dim
        let values = vec![2.0f32; 4 * 8];

        cache
            .write_kv(0, block_id, keys.clone(), values.clone(), 4)
            .unwrap();

        let (read_keys, read_values) = cache.read_kv(0, &[block_id]).unwrap();
        assert_eq!(read_keys.len(), 4 * 8);
        assert_eq!(read_keys[..keys.len()], keys[..]);
        assert_eq!(read_values[..values.len()], values[..]);
    }

    #[test]
    fn test_paged_cache_int8_quantization() {
        let mut config = test_config();
        config.kv_dtype = KvCacheDtype::Int8;
        let cache = PagedAttentionCache::new(config);

        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];

        let keys: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.8).collect();

        cache
            .write_kv(0, block_id, keys.clone(), values.clone(), 4)
            .unwrap();

        let (read_keys, read_values) = cache.read_kv(0, &[block_id]).unwrap();
        assert_eq!(read_keys.len(), keys.len());
        assert_eq!(read_values.len(), values.len());

        // Verify approximate roundtrip (INT8 has limited precision)
        for (orig, rec) in keys.iter().zip(read_keys.iter()) {
            assert!((orig - rec).abs() < 0.05, "orig={}, rec={}", orig, rec);
        }
    }

    #[test]
    fn test_paged_cache_prefix_reuse() {
        let cache = PagedAttentionCache::new(test_config());

        // First request: allocate and "fill" blocks
        let alloc1 = cache.allocate("req-1", &[10, 20, 30, 40], 4).unwrap();
        assert_eq!(alloc1.matched_tokens, 0);

        // Write some KV data
        for &block_id in &alloc1.allocated_blocks {
            let keys = vec![1.0f32; 4 * 8];
            let values = vec![2.0f32; 4 * 8];
            let _ = cache.write_kv(0, block_id, keys, values, 4);
        }

        // Free req-1 (blocks become cached, available for LRU)
        cache.free("req-1");

        // Second request with same prefix should get a cache hit
        let alloc2 = cache
            .allocate("req-2", &[10, 20, 30, 40, 50, 60, 70, 80], 4)
            .unwrap();
        assert!(alloc2.matched_tokens > 0, "expected prefix cache hit");
    }

    #[test]
    fn test_paged_cache_memory_bytes() {
        let cache = PagedAttentionCache::new(test_config());
        assert_eq!(cache.memory_bytes(), 0);

        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];

        cache
            .write_kv(0, block_id, vec![0.0; 32], vec![0.0; 32], 4)
            .unwrap();
        assert!(cache.memory_bytes() > 0);
    }

    #[test]
    fn test_paged_cache_empty_read() {
        let cache = PagedAttentionCache::new(test_config());
        // Reading a non-existent block should return zeros
        let (keys, values) = cache.read_kv(0, &[999]).unwrap();
        assert_eq!(keys.len(), 4 * 8); // block_size * kv_dim
        assert!(keys.iter().all(|&v| v == 0.0));
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_paged_cache_layer_isolation() {
        let cache = PagedAttentionCache::new(test_config());
        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];

        // Write to layer 0
        cache
            .write_kv(0, block_id, vec![1.0; 32], vec![2.0; 32], 4)
            .unwrap();
        // Write different data to layer 1
        cache
            .write_kv(1, block_id, vec![3.0; 32], vec![4.0; 32], 4)
            .unwrap();

        let (k0, _) = cache.read_kv(0, &[block_id]).unwrap();
        let (k1, _) = cache.read_kv(1, &[block_id]).unwrap();
        assert!(k0.iter().all(|&v| v == 1.0));
        assert!(k1.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn test_cachemesh_l2_offload_restore_and_metrics() {
        let mesh = Arc::new(CacheMesh::new(CacheMeshConfig {
            enabled: true,
            l2_capacity_bytes: 1024,
            ..Default::default()
        }));
        let cache = PagedAttentionCache::with_cachemesh(test_config(), Arc::clone(&mesh));
        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];
        cache
            .write_kv(0, block_id, vec![1.0; 32], vec![2.0; 32], 4)
            .unwrap();

        assert!(cache.offload_block(0, block_id).unwrap());
        assert_eq!(cache.layer_block_count(0), 0);

        let (keys, values) = cache.read_kv(0, &[block_id]).unwrap();
        assert_eq!(keys, vec![1.0; 32]);
        assert_eq!(values, vec![2.0; 32]);

        let metrics = cache.cachemesh_metrics().unwrap();
        assert_eq!(metrics.l2.hits, 1);
        assert_eq!(metrics.l2.hit_rate, 1.0);
        assert_eq!(metrics.l2.restores, 1);
    }

    #[test]
    fn test_cachemesh_l3_restore_after_l2_spill() {
        let remote = Arc::new(InMemoryRemoteCache::new());
        let mesh = Arc::new(CacheMesh::with_remote(
            CacheMeshConfig {
                enabled: true,
                l2_capacity_bytes: 1,
                l3_enabled: true,
                ..Default::default()
            },
            remote,
        ));
        let cache = PagedAttentionCache::with_cachemesh(test_config(), Arc::clone(&mesh));
        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];
        cache
            .write_kv(0, block_id, vec![3.0; 32], vec![4.0; 32], 4)
            .unwrap();

        assert!(cache.offload_block(0, block_id).unwrap());
        let (keys, values) = cache.read_kv(0, &[block_id]).unwrap();
        assert_eq!(keys, vec![3.0; 32]);
        assert_eq!(values, vec![4.0; 32]);

        let metrics = cache.cachemesh_metrics().unwrap();
        assert_eq!(metrics.l2.misses, 1);
        assert_eq!(metrics.l3.hits, 1);
        assert_eq!(metrics.l3.hit_rate, 1.0);
    }

    #[test]
    fn test_cachemesh_int8_restore_rehydrates_quantized_l1() {
        let mesh = Arc::new(CacheMesh::new(CacheMeshConfig {
            enabled: true,
            l2_capacity_bytes: 1024,
            ..Default::default()
        }));
        let mut config = test_config();
        config.kv_dtype = KvCacheDtype::Int8;
        let cache = PagedAttentionCache::with_cachemesh(config, Arc::clone(&mesh));
        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];
        let keys: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.02).collect();
        cache
            .write_kv(0, block_id, keys.clone(), values.clone(), 4)
            .unwrap();

        assert!(cache.offload_block(0, block_id).unwrap());
        let (read_keys, read_values) = cache.read_kv(0, &[block_id]).unwrap();
        assert_eq!(read_keys.len(), keys.len());
        assert_eq!(read_values.len(), values.len());
        assert!(
            read_keys
                .iter()
                .zip(keys.iter())
                .all(|(a, b)| (a - b).abs() < 0.01)
        );
    }

    #[test]
    fn test_sliding_window_limits_visible_blocks() {
        let mut config = test_config();
        config.long_context_policy = LongContextPolicy::SlidingWindow { window_tokens: 8 };
        let cache = PagedAttentionCache::new(config);

        assert_eq!(cache.visible_blocks(&[1, 2, 3, 4]), vec![3, 4]);
    }

    #[test]
    fn test_context_shift_drops_oldest_blocks() {
        let mut config = test_config();
        config.long_context_policy = LongContextPolicy::ContextShift {
            max_context_tokens: 16,
            shift_tokens: 4,
        };
        let cache = PagedAttentionCache::new(config);

        assert_eq!(
            cache.visible_blocks(&[10, 11, 12, 13, 14]),
            vec![12, 13, 14]
        );
    }

    #[test]
    fn test_compact_inactive_releases_cached_blocks() {
        let mut config = test_config();
        config.total_blocks = 4;
        config.long_context_policy = LongContextPolicy::CompactInactive {
            target_free_blocks: 4,
        };
        let cache = PagedAttentionCache::new(config);
        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        assert_eq!(alloc.allocated_blocks.len(), 2);

        cache.free("req-1");
        let released = cache.maintain_long_context();

        assert_eq!(released, 2);
        let metrics = cache.metrics();
        assert_eq!(metrics.free_blocks, 4);
        assert_eq!(metrics.active_blocks, 0);
    }

    // Requires the TileLang Python toolchain (numpy etc.) to JIT-compile the
    // attention kernel; not available on standard CI runners. Run locally with
    // `cargo test --workspace --features hardware-tests`.
    #[cfg(feature = "hardware-tests")]
    #[test]
    fn test_paged_attention_tilelang() {
        let mut config = test_config();
        config.block_size = 4;
        config.total_blocks = 4;
        config.num_layers = 1;
        config.kv_dim = 4; // head_dim = 4
        let cache = PagedAttentionCache::new(config);

        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];

        let keys = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let values = vec![
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
        ];
        cache.write_kv(0, block_id, keys, values, 4).unwrap();

        let q = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut output = vec![0.0f32; 16];

        cache
            .paged_attention(&q, 0, &[block_id], &mut output, 4, 4)
            .unwrap();

        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_paged_attention_forward_tensor() {
        let mut config = test_config();
        config.block_size = 4;
        config.total_blocks = 4;
        config.num_layers = 1;
        config.kv_dim = 4; // head_dim = 4
        let cache = PagedAttentionCache::new(config);

        let alloc = cache.allocate("req-1", &[1, 2, 3, 4], 4).unwrap();
        let block_id = alloc.allocated_blocks[0];

        // shape: [1, num_kv_heads=1, block_size=4, head_dim=4]
        let device = Device::Cpu;
        let k_t = Tensor::new(
            &[[[
                [1.0f32, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]]],
            &device,
        )
        .unwrap();
        let v_t = Tensor::new(
            &[[[
                [5.0f32, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
            ]]],
            &device,
        )
        .unwrap();

        cache.write_kv_tensor(0, block_id, k_t, v_t, 4).unwrap();

        // query shape: [batch=1, num_heads=1, seq_len=1, head_dim=4]
        let query = Tensor::new(&[[[[1.0f32, 1.0, 1.0, 1.0]]]], &device).unwrap();
        let scale = 1.0 / (4.0f64).sqrt();

        let output = cache
            .paged_attention_forward_tensor(0, &query, &[block_id], scale, None)
            .unwrap();
        assert_eq!(output.dims(), &[1, 1, 1, 4]);

        let out_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &v in &out_vec {
            assert!((v - 5.0).abs() < 1e-4);
        }
    }
}
