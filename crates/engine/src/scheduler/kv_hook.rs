//! Bridge trait connecting a model's internal KV state with `PagedAttentionCache`.
//!
//! Without an implementation of `KvHook`, the paged cache only tracks block
//! allocation metadata — the actual KV tensors stay inside the model's own
//! KV cache (e.g. candle_nn's `ConcatKvCache`). Wiring a `KvHook` lets the
//! scheduler-driven batch path extract KV from the model after prefill and
//! write it into the paged cache, then read it back and inject it before
//! decode. That is what unlocks real cross-request KV reuse, prefix
//! caching hit semantics, and the `paged_attention_forward` kernel path.
//!
//! The default `CandleBatchExecutor` path can run without a hook. In that
//! configuration, the model owns its internal KV state and paged-cache reuse is
//! unavailable. See `docs/scheduler.md` for the scheduling contract.

use anyhow::Result;
#[cfg(test)]
use bloomai_core::BloomError;
use candle_core::Tensor;
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Mutex;

/// Per-layer KV bridge between a model's internal attention state and the
/// scheduler-managed `PagedAttentionCache`.
///
/// Implementations must be `Send + Sync` because the batch executor may be
/// shared across scheduler threads. The hook is consulted inside the
/// executor's prefill/decode hot path, so `extract_kv` / `inject_kv` must
/// stay cheap — the returned vectors are the only allowed allocation.
///
/// All operations are keyed by `handle` (the scheduler-assigned KV cache
/// handle), because production deployments allocate one model instance per
/// request and the hook must read from the model that just produced the
/// forward pass for that handle.
pub trait KvHook: Send + Sync {
    /// Number of attention layers in the model.
    fn num_layers(&self) -> usize;

    /// Key/value dimension per layer (typically `head_dim * num_kv_heads`).
    fn kv_dim(&self) -> usize;

    /// Extract KV state for tokens `[start_pos, start_pos + seq_len)` from
    /// layer `layer_idx` of the model bound to `handle`. Returns flat f32
    /// vectors of length `seq_len * kv_dim()` for keys and values respectively.
    ///
    /// Called by the batch executor after prefill so the paged cache can be
    /// populated with the model's freshly computed KV. Returning an error
    /// stops the batch step with an actionable message rather than silently
    /// degrading into placeholder writes.
    fn extract_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)>;

    /// Extract KV state as GPU Tensors (if supported by the backend).
    fn extract_kv_tensor(
        &self,
        _handle: usize,
        _layer_idx: usize,
        _start_pos: usize,
        _seq_len: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        Ok(None)
    }

    /// Inject KV state into layer `layer_idx` at tokens
    /// `[start_pos, start_pos + seq_len)` of the model bound to `handle`.
    /// `keys` and `values` are flat f32 vectors of length `seq_len * kv_dim()`.
    ///
    /// Called by the batch executor before decode when the paged cache holds
    /// KV that the model's internal cache does not — for example after an
    /// inactive block was restored from CacheMesh L2/L3, or when a request
    /// reuses a prefix that was computed by another request.
    fn inject_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> Result<()>;

    /// Inject KV state as GPU Tensors (if supported by the backend).
    fn inject_kv_tensor(
        &self,
        _handle: usize,
        _layer_idx: usize,
        _start_pos: usize,
        _keys: &Tensor,
        _values: &Tensor,
    ) -> Result<()> {
        Ok(())
    }

    /// Clear the internal KV cache of the model bound to `handle`.
    fn clear_kv_cache(&self, _handle: usize) -> Result<()> {
        Ok(())
    }

    /// Rollback the internal KV cache of the model bound to `handle` to `length` tokens.
    fn rollback_kv_cache(&self, _handle: usize, _length: usize) -> Result<()> {
        Ok(())
    }
}

/// In-memory `KvHook` backed by a per-handle, per-layer `Vec<f32>`.
///
/// Used by the integration tests in `batch_executor.rs` and `paged_cache.rs`
/// to prove the round-trip without depending on a real Candle model. Lives
/// behind `#[cfg(test)]` so it never ships in a release build.
#[cfg(test)]
pub struct InMemoryKvHook {
    num_layers: usize,
    kv_dim: usize,
    // Per-handle, per-layer KV storage as flat f32 vectors. Length grows as
    // tokens are written. Reads outside the populated range return zeros.
    keys: Mutex<HashMap<usize, Vec<Mutex<Vec<f32>>>>>,
    values: Mutex<HashMap<usize, Vec<Mutex<Vec<f32>>>>>,
}

#[cfg(test)]
impl InMemoryKvHook {
    pub fn new(num_layers: usize, kv_dim: usize) -> Self {
        Self {
            num_layers,
            kv_dim,
            keys: Mutex::new(HashMap::new()),
            values: Mutex::new(HashMap::new()),
        }
    }

    /// Ensure storage exists for `handle` (lazily allocating per-layer Vecs).
    /// Returns the handle's per-layer Vec so callers can lock individual layers.
    fn ensure_handle(&self, handle: usize) -> (Vec<Mutex<Vec<f32>>>, Vec<Mutex<Vec<f32>>>) {
        let mut keys = self.keys.lock().unwrap_or_else(|e| e.into_inner());
        let mut values = self.values.lock().unwrap_or_else(|e| e.into_inner());
        if let std::collections::hash_map::Entry::Vacant(entry) = keys.entry(handle) {
            let per_layer_k: Vec<Mutex<Vec<f32>>> = (0..self.num_layers)
                .map(|_| Mutex::new(Vec::new()))
                .collect();
            let per_layer_v: Vec<Mutex<Vec<f32>>> = (0..self.num_layers)
                .map(|_| Mutex::new(Vec::new()))
                .collect();
            entry.insert(per_layer_k);
            values.insert(handle, per_layer_v);
        }
        (
            keys.remove(&handle).unwrap(),
            values.remove(&handle).unwrap(),
        )
    }

    /// Write deterministic KV for `[start_pos, start_pos + seq_len)` so
    /// tests can assert non-zero round-trips through the paged cache.
    pub fn populate_deterministic(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
        key_value: f32,
        value_value: f32,
    ) {
        let (keys, values) = self.ensure_handle(handle);
        {
            let mut k = keys[layer_idx].lock().unwrap_or_else(|e| e.into_inner());
            let mut v = values[layer_idx].lock().unwrap_or_else(|e| e.into_inner());
            let needed = (start_pos + seq_len) * self.kv_dim;
            if k.len() < needed {
                k.resize(needed, 0.0);
                v.resize(needed, 0.0);
            }
            for i in 0..seq_len {
                for d in 0..self.kv_dim {
                    let pos = (start_pos + i) * self.kv_dim + d;
                    k[pos] = key_value + (i as f32) * 0.001 + (d as f32) * 0.0001;
                    v[pos] = value_value + (i as f32) * 0.002 + (d as f32) * 0.0002;
                }
            }
        }
        // Put back
        self.keys
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(handle, keys);
        self.values
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(handle, values);
    }
}

#[cfg(test)]
impl KvHook for InMemoryKvHook {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn extract_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if layer_idx >= self.num_layers {
            return Err(BloomError::Engine(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx, self.num_layers
            ))
            .into());
        }
        // Take ownership of per-handle Vec to avoid double-lock on outer map.
        let (keys, values) = self.ensure_handle(handle);
        let result = {
            let keys_l = keys[layer_idx].lock().unwrap_or_else(|e| e.into_inner());
            let values_l = values[layer_idx].lock().unwrap_or_else(|e| e.into_inner());
            let start = start_pos * self.kv_dim;
            let end = (start_pos + seq_len) * self.kv_dim;
            let keys_slice = if start < keys_l.len() {
                &keys_l[start..keys_l.len().min(end)]
            } else {
                &[]
            };
            let values_slice = if start < values_l.len() {
                &values_l[start..values_l.len().min(end)]
            } else {
                &[]
            };
            let mut keys_out = keys_slice.to_vec();
            let mut values_out = values_slice.to_vec();
            keys_out.resize(seq_len * self.kv_dim, 0.0);
            values_out.resize(seq_len * self.kv_dim, 0.0);
            (keys_out, values_out)
        };
        // Put back
        self.keys
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(handle, keys);
        self.values
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(handle, values);
        Ok(result)
    }

    fn inject_kv(
        &self,
        handle: usize,
        layer_idx: usize,
        start_pos: usize,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> Result<()> {
        if layer_idx >= self.num_layers {
            return Err(BloomError::Engine(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx, self.num_layers
            ))
            .into());
        }
        let expected = seq_len * self.kv_dim;
        if keys.len() != expected || values.len() != expected {
            return Err(BloomError::Engine(format!(
                "kv shape mismatch: expected {}, got keys={}, values={}",
                expected,
                keys.len(),
                values.len()
            ))
            .into());
        }
        let (keys_map, values_map) = self.ensure_handle(handle);
        {
            let mut k = keys_map[layer_idx]
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut v = values_map[layer_idx]
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let start = start_pos * self.kv_dim;
            let end = start + expected;
            if k.len() < end {
                k.resize(end, 0.0);
                v.resize(end, 0.0);
            }
            k[start..end].copy_from_slice(keys);
            v[start..end].copy_from_slice(values);
        }
        self.keys
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(handle, keys_map);
        self.values
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(handle, values_map);
        Ok(())
    }

    fn clear_kv_cache(&self, handle: usize) -> Result<()> {
        let mut keys = self.keys.lock().unwrap_or_else(|e| e.into_inner());
        let mut values = self.values.lock().unwrap_or_else(|e| e.into_inner());
        keys.remove(&handle);
        values.remove(&handle);
        Ok(())
    }

    fn rollback_kv_cache(&self, handle: usize, length: usize) -> Result<()> {
        let keys = self.keys.lock().unwrap_or_else(|e| e.into_inner());
        let values = self.values.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(per_layer_k) = keys.get(&handle) {
            if let Some(per_layer_v) = values.get(&handle) {
                for layer_idx in 0..self.num_layers {
                    let mut k = per_layer_k[layer_idx]
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let mut v = per_layer_v[layer_idx]
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let end = length * self.kv_dim;
                    if k.len() > end {
                        k.truncate(end);
                        v.truncate(end);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_kv_hook_round_trip() {
        let hook = InMemoryKvHook::new(2, 4);
        hook.populate_deterministic(7, 0, 0, 3, 1.0, 2.0);

        let (keys, values) = hook.extract_kv(7, 0, 0, 3).unwrap();
        assert_eq!(keys.len(), 12);
        assert_eq!(values.len(), 12);
        assert!((keys[0] - 1.0).abs() < 1e-6);
        assert!((values[0] - 2.0).abs() < 1e-6);

        // Layer 1 untouched — should return zeros
        let (k1, v1) = hook.extract_kv(7, 1, 0, 3).unwrap();
        assert!(k1.iter().all(|&v| v == 0.0));
        assert!(v1.iter().all(|&v| v == 0.0));

        // Inject then re-extract
        let new_keys = vec![5.0; 12];
        let new_values = vec![6.0; 12];
        hook.inject_kv(7, 1, 0, &new_keys, &new_values, 3).unwrap();
        let (k1b, v1b) = hook.extract_kv(7, 1, 0, 3).unwrap();
        assert!(k1b.iter().all(|&v| (v - 5.0).abs() < 1e-6));
        assert!(v1b.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_in_memory_kv_hook_rejects_bad_layer() {
        let hook = InMemoryKvHook::new(1, 4);
        assert!(hook.extract_kv(1, 5, 0, 1).is_err());
        assert!(hook.inject_kv(1, 5, 0, &[0.0; 4], &[0.0; 4], 1).is_err());
    }

    #[test]
    fn test_in_memory_kv_hook_rejects_shape_mismatch() {
        let hook = InMemoryKvHook::new(1, 4);
        assert!(hook.inject_kv(1, 0, 0, &[0.0; 3], &[0.0; 4], 1).is_err());
    }

    #[test]
    fn test_in_memory_kv_hook_isolates_handles() {
        // KV written under handle 1 must not leak into handle 2.
        let hook = InMemoryKvHook::new(1, 4);
        hook.populate_deterministic(1, 0, 0, 2, 9.0, 11.0);
        let (k_other, v_other) = hook.extract_kv(2, 0, 0, 2).unwrap();
        assert!(k_other.iter().all(|&v| v == 0.0));
        assert!(v_other.iter().all(|&v| v == 0.0));
        let (k_self, v_self) = hook.extract_kv(1, 0, 0, 2).unwrap();
        assert!((k_self[0] - 9.0).abs() < 1e-6);
        assert!((v_self[0] - 11.0).abs() < 1e-6);
    }
}
