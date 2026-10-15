//! Sparse Qwen2-MoE feed-forward blocks for the extractable Qwen KV runtime.
//! Checkpoint semantics follow Qwen2-MoE and Apollo's inference-only upcycling
//! variant. No remote Python model code is executed.

use super::{StreamingLinear, StreamingQwenMLP};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

const EXPERT_CACHE_ENV: &str = "BLOOM_MOE_EXPERT_CACHE_MIB";

// The budget covers persistent expert weights, not activations, KV, or the
// temporary weights used by the original streaming fallback. Divide across all
// decoder layers so sparse/dense mixtures never exceed the model-wide setting.
fn layer_cache_budget(value: Option<&str>, layer_count: usize) -> Option<usize> {
    let mib = value.unwrap_or("0").parse::<usize>().ok()?;
    mib.checked_mul(1024 * 1024)?.checked_div(layer_count)
}

#[derive(Debug, Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
    uploads: u64,
    uploaded_bytes: u64,
    evictions: u64,
    fallbacks: u64,
}

#[derive(Debug)]
struct CacheEntry<K, V> {
    key: K,
    value: Arc<V>,
    bytes: usize,
}

/// Oldest entry first. Arc leases keep weights alive outside the mutex during
/// forward; busy entries cannot be evicted or forgotten from byte accounting.
#[derive(Debug)]
struct ExpertLru<K, V> {
    entries: VecDeque<CacheEntry<K, V>>,
    budget_bytes: usize,
    current_bytes: usize,
    stats: CacheStats,
}

impl<K: PartialEq, V> ExpertLru<K, V> {
    fn new(budget_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            budget_bytes,
            current_bytes: 0,
            stats: CacheStats::default(),
        }
    }

    fn get(&mut self, key: &K) -> Option<Arc<V>> {
        let Some(index) = self.entries.iter().position(|entry| &entry.key == key) else {
            self.stats.misses += 1;
            return None;
        };
        let entry = self.entries.remove(index).expect("LRU entry exists");
        let value = Arc::clone(&entry.value);
        self.entries.push_back(entry);
        self.stats.hits += 1;
        Some(value)
    }

    fn make_room(&mut self, bytes: usize) -> bool {
        if bytes > self.budget_bytes {
            return false;
        }
        // Check feasibility first: a failed acquisition should not discard
        // useful entries when active forwards already consume the budget.
        let reclaimable: usize = self
            .entries
            .iter()
            .filter(|entry| Arc::strong_count(&entry.value) == 1)
            .map(|entry| entry.bytes)
            .sum();
        if self.current_bytes - reclaimable > self.budget_bytes - bytes {
            return false;
        }
        while self.current_bytes > self.budget_bytes - bytes {
            let index = self
                .entries
                .iter()
                .position(|entry| Arc::strong_count(&entry.value) == 1)
                .expect("reclaimable capacity was checked");
            let entry = self.entries.remove(index).expect("LRU entry exists");
            self.current_bytes -= entry.bytes;
            self.stats.evictions += 1;
        }
        true
    }

    fn insert(&mut self, key: K, value: V, bytes: usize) -> Arc<V> {
        debug_assert!(bytes <= self.budget_bytes - self.current_bytes);
        let value = Arc::new(value);
        self.entries.push_back(CacheEntry {
            key,
            value: Arc::clone(&value),
            bytes,
        });
        self.current_bytes += bytes;
        self.stats.uploads += 1;
        self.stats.uploaded_bytes += bytes as u64;
        value
    }
}

#[derive(Debug)]
struct ExpertKey {
    index: usize,
    device: Device,
}

impl PartialEq for ExpertKey {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.device.same_device(&other.device)
    }
}

#[derive(Debug)]
struct ExpertCache {
    layer: usize,
    lru: Mutex<ExpertLru<ExpertKey, StreamingQwenMLP>>,
}

impl ExpertCache {
    fn acquire(
        &self,
        index: usize,
        expert: &StreamingQwenMLP,
        device: &Device,
    ) -> Result<Option<Arc<StreamingQwenMLP>>> {
        if matches!(device, Device::Cpu) {
            return Ok(None);
        }
        let mut cache = self
            .lru
            .lock()
            .map_err(|_| candle_core::Error::Msg("MoE expert cache mutex poisoned".into()))?;
        let key = ExpertKey {
            index,
            device: device.clone(),
        };
        let (value, event) = if let Some(value) = cache.get(&key) {
            (Some(value), "hit")
        } else if let Some(bytes) = expert.standard_weight_bytes() {
            if cache.make_room(bytes) {
                // Only uploads hold the lock; matrix multiplications run after
                // returning the Arc lease. This also prevents duplicate uploads.
                let value = expert.standard_to_device(device)?;
                (Some(cache.insert(key, value, bytes)), "upload")
            } else {
                cache.stats.fallbacks += 1;
                (None, "capacity_fallback")
            }
        } else {
            // Packed Quantized projections retain their existing dequantization
            // behavior; only fully materialized Standard tensors are cached.
            cache.stats.fallbacks += 1;
            (None, "quantized_fallback")
        };
        tracing::debug!(
            target: "bloom_moe_expert_cache",
            layer = self.layer,
            expert = index,
            event,
            hits = cache.stats.hits,
            misses = cache.stats.misses,
            uploads = cache.stats.uploads,
            uploaded_bytes = cache.stats.uploaded_bytes,
            current_bytes = cache.current_bytes,
            budget_bytes = cache.budget_bytes,
            evictions = cache.stats.evictions,
            fallbacks = cache.stats.fallbacks,
            "MoE expert cache"
        );
        Ok(value)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(skip)]
    shared_expert: bool,
}

impl MoeConfig {
    pub(crate) fn from_json(value: &serde_json::Value) -> anyhow::Result<Self> {
        let mut config: Self = serde_json::from_value(value.clone())?;
        anyhow::ensure!(config.num_experts > 0, "MoE num_experts must be positive");
        anyhow::ensure!(
            config.num_experts_per_tok > 0 && config.num_experts_per_tok <= config.num_experts,
            "MoE num_experts_per_tok must be in 1..=num_experts"
        );
        anyhow::ensure!(
            config.decoder_sparse_step > 0,
            "MoE decoder_sparse_step must be positive"
        );
        anyhow::ensure!(
            config.moe_intermediate_size > 0,
            "MoE intermediate size must be positive"
        );
        anyhow::ensure!(
            value.get("use_sliding_window").and_then(|v| v.as_bool()) != Some(true),
            "Qwen2-MoE sliding-window attention is not supported by the extractable KV runtime"
        );
        let upcycling =
            value.get("model_type").and_then(|v| v.as_str()) == Some("upcycling-qwen2-moe");
        // Apollo's shared variant changes the expert counts during Python
        // construction; reject it until that layout has pinned evidence.
        anyhow::ensure!(
            !upcycling || value.get("share_flag").and_then(|v| v.as_bool()) == Some(false),
            "Apollo shared-expert upcycling is not supported; expected share_flag=false"
        );
        config.shared_expert = !upcycling;
        anyhow::ensure!(
            !config.shared_expert || config.shared_expert_intermediate_size > 0,
            "Qwen2-MoE shared expert size must be positive"
        );
        Ok(config)
    }

    pub(super) fn sparse_layer(&self, layer: usize) -> bool {
        !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

#[derive(Debug, Clone)]
pub(super) struct SparseMoe {
    gate: StreamingLinear,
    experts: Vec<StreamingQwenMLP>,
    shared: Option<(StreamingQwenMLP, StreamingLinear)>,
    top_k: usize,
    norm_topk_prob: bool,
    // Model clones share this layer's budget instead of duplicating it. Separate
    // model loads each receive their own model-wide budget.
    expert_cache: Option<Arc<ExpertCache>>,
}

impl SparseMoe {
    pub(super) fn new(
        config: &MoeConfig,
        layer: usize,
        layer_count: usize,
        hidden: usize,
        activation: Activation,
        vb: VarBuilder,
        quantizer: Option<&Arc<crate::executor::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        let gate = StreamingLinear::new(
            vb.pp("gate").get((config.num_experts, hidden), "weight")?,
            None,
        );
        let mut experts = Vec::with_capacity(config.num_experts);
        let expert_vb = vb.pp("experts");
        if expert_vb.contains_tensor("gate_up_proj") {
            let gate_up = expert_vb.get(
                (config.num_experts, 2 * config.moe_intermediate_size, hidden),
                "gate_up_proj",
            )?;
            let down = expert_vb.get(
                (config.num_experts, hidden, config.moe_intermediate_size),
                "down_proj",
            )?;
            for index in 0..config.num_experts {
                let gate_up = gate_up.get(index)?;
                // contiguous() can retain the whole packed layer's storage even
                // for an expert view. to_device would then upload every expert.
                // Materialize compact expert tensors for cached and uncached use.
                experts.push(StreamingQwenMLP {
                    gate_proj: StreamingLinear::new(
                        gate_up
                            .narrow(0, 0, config.moe_intermediate_size)?
                            .force_contiguous()?,
                        None,
                    ),
                    up_proj: StreamingLinear::new(
                        gate_up
                            .narrow(
                                0,
                                config.moe_intermediate_size,
                                config.moe_intermediate_size,
                            )?
                            .force_contiguous()?,
                        None,
                    ),
                    down_proj: StreamingLinear::new(down.get(index)?.force_contiguous()?, None),
                    act_fn: activation,
                });
            }
        } else {
            for index in 0..config.num_experts {
                experts.push(StreamingQwenMLP::new(
                    hidden,
                    config.moe_intermediate_size,
                    activation,
                    expert_vb.pp(index),
                    quantizer,
                )?);
            }
        }
        let shared = if config.shared_expert {
            Some((
                StreamingQwenMLP::new(
                    hidden,
                    config.shared_expert_intermediate_size,
                    activation,
                    vb.pp("shared_expert"),
                    quantizer,
                )?,
                StreamingLinear::new(
                    vb.pp("shared_expert_gate").get((1, hidden), "weight")?,
                    None,
                ),
            ))
        } else {
            None
        };
        let cache_env = std::env::var(EXPERT_CACHE_ENV).ok();
        let budget = layer_cache_budget(cache_env.as_deref(), layer_count);
        if budget.is_none() {
            tracing::warn!(layer, value = ?cache_env, "Invalid {EXPERT_CACHE_ENV}; expert cache disabled");
        }
        let expert_cache = budget.filter(|&bytes| bytes > 0).map(|budget_bytes| {
            tracing::info!(
                target: "bloom_moe_expert_cache",
                layer,
                layer_count,
                budget_bytes,
                expert_bytes = ?experts.first().and_then(StreamingQwenMLP::standard_weight_bytes),
                "MoE expert cache enabled (GPU only; standard weights)"
            );
            Arc::new(ExpertCache {
                layer,
                lru: Mutex::new(ExpertLru::new(budget_bytes)),
            })
        });
        Ok(Self {
            gate,
            experts,
            shared,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            expert_cache,
        })
    }

    pub(super) fn forward(&self, input: &Tensor, device: &Device) -> Result<Tensor> {
        let (batch, sequence, hidden) = input.dims3()?;
        let input = input.reshape((batch * sequence, hidden))?;
        // Routing softmax is F32, matching Transformers even for FP16 weights.
        let logits = self.gate.forward(&input, device)?.to_dtype(DType::F32)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&logits)?;
        let indices = probabilities
            .arg_sort_last_dim(false)?
            .narrow(1, 0, self.top_k)?
            .contiguous()?;
        let selected = probabilities.gather(&indices, 1)?.to_vec2::<f32>()?;
        let indices = indices.to_vec2::<u32>()?;
        let mut rows = vec![Vec::new(); self.experts.len()];
        let mut weights = vec![Vec::new(); self.experts.len()];
        for (row, (experts, probabilities)) in indices.iter().zip(&selected).enumerate() {
            let sum: f32 = probabilities.iter().sum();
            for (&expert, &weight) in experts.iter().zip(probabilities) {
                rows[expert as usize].push(row as u32);
                weights[expert as usize].push(if self.norm_topk_prob {
                    weight / sum
                } else {
                    weight
                });
            }
        }
        let mut output = input.zeros_like()?;
        for (index, expert) in self.experts.iter().enumerate() {
            if rows[index].is_empty() {
                continue;
            }
            let row_ids = Tensor::new(rows[index].as_slice(), device)?;
            let weight = Tensor::new(weights[index].as_slice(), device)?
                .unsqueeze(1)?
                .to_dtype(input.dtype())?;
            let expert_input = input.index_select(&row_ids, 0)?;
            // Acquire one selected expert at a time: prefill may select every
            // expert, but it must not upload all of them before enforcing LRU.
            let cached = self
                .expert_cache
                .as_ref()
                .map(|cache| cache.acquire(index, expert, device))
                .transpose()?
                .flatten();
            let result = cached
                .as_deref()
                .unwrap_or(expert)
                .forward(&expert_input, device)?
                .broadcast_mul(&weight)?;
            output = output.index_add(&row_ids, &result, 0)?;
        }
        if let Some((expert, gate)) = &self.shared {
            let weight = candle_nn::ops::sigmoid(&gate.forward(&input, device)?)?;
            output = (output + expert.forward(&input, device)?.broadcast_mul(&weight)?)?;
        }
        output.reshape((batch, sequence, hidden))
    }
}

#[derive(Debug, Clone)]
pub(super) enum FeedForward {
    Dense(StreamingQwenMLP),
    Sparse(SparseMoe),
}

impl FeedForward {
    pub(super) fn forward(&self, input: &Tensor, device: &Device) -> Result<Tensor> {
        match self {
            Self::Dense(model) => model.forward(input, device),
            Self::Sparse(model) => model.forward(input, device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{CpuStorage, Storage};
    use std::collections::HashMap;

    #[test]
    fn expert_cache_defaults_to_disabled_and_divides_model_budget() {
        assert_eq!(layer_cache_budget(None, 12), Some(0));
        assert_eq!(layer_cache_budget(Some("0"), 12), Some(0));
        assert_eq!(layer_cache_budget(Some("12"), 12), Some(1024 * 1024));
        assert!(layer_cache_budget(Some("1"), 3).unwrap() * 3 <= 1024 * 1024);
        assert_eq!(layer_cache_budget(Some("bad"), 12), None);
        assert_eq!(layer_cache_budget(Some("-1"), 12), None);
        assert_eq!(layer_cache_budget(Some("1"), 0), None);
        assert_eq!(layer_cache_budget(Some(&usize::MAX.to_string()), 1), None);
    }

    #[test]
    fn expert_lru_reuses_hits_and_evicts_oldest_within_budget() {
        let mut cache = ExpertLru::new(10);
        assert!(cache.make_room(4));
        cache.insert(0, "first", 4);
        assert!(cache.make_room(4));
        cache.insert(1, "second", 4);
        assert_eq!(*cache.get(&0).unwrap(), "first");
        assert!(cache.make_room(6));
        cache.insert(2, "third", 6);
        assert_eq!(cache.current_bytes, 10);
        assert_eq!(
            cache
                .entries
                .iter()
                .map(|entry| entry.key)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.stats.hits, 1);
        assert_eq!(cache.stats.misses, 1);
        assert_eq!(cache.stats.uploads, 3);
        assert_eq!(cache.stats.uploaded_bytes, 14);
        assert_eq!(cache.stats.evictions, 1);
        assert!(!cache.make_room(11));
        assert_eq!(cache.current_bytes, 10);
    }

    #[test]
    fn expert_lru_counts_active_leases_and_falls_back_when_busy() {
        let mut cache = ExpertLru::new(8);
        let lease = cache.insert(0, (), 4);
        cache.insert(1, (), 4);
        assert!(!cache.make_room(5));
        assert_eq!(cache.current_bytes, 8);
        assert_eq!(cache.entries.len(), 2);
        // The oldest entry is busy, so only the next idle entry can be evicted.
        assert!(cache.make_room(4));
        cache.insert(2, (), 4);
        assert_eq!(cache.entries.front().unwrap().key, 0);
        assert_eq!(cache.current_bytes, 8);
        drop(lease);
        assert!(cache.make_room(8));
        assert_eq!(cache.current_bytes, 0);
        assert_eq!(cache.stats.evictions, 3);
    }

    fn assert_compact_f32(tensor: &Tensor) {
        let (storage, layout) = tensor.storage_and_layout();
        assert_eq!(layout.start_offset(), 0);
        assert!(layout.is_contiguous());
        match &*storage {
            Storage::Cpu(CpuStorage::F32(values)) => {
                assert_eq!(values.len(), tensor.elem_count());
            }
            _ => panic!("expected F32 CPU storage"),
        }
    }

    fn standard_weight(linear: &StreamingLinear) -> &Tensor {
        match linear {
            StreamingLinear::Standard { weight, .. } => weight,
            _ => panic!("expected Standard projection"),
        }
    }

    #[test]
    fn packed_experts_have_compact_storage_and_preserve_outputs() -> Result<()> {
        let device = &Device::Cpu;
        let gate_up = (Tensor::arange(0f32, 24f32, device)? / 100.0)?.reshape((2, 4, 3))?;
        let down = (Tensor::arange(0f32, 12f32, device)? / 100.0)?.reshape((2, 3, 2))?;
        let tensors = HashMap::from([
            (
                "gate.weight".into(),
                Tensor::zeros((2, 3), DType::F32, device)?,
            ),
            ("experts.gate_up_proj".into(), gate_up.clone()),
            ("experts.down_proj".into(), down.clone()),
        ]);
        let config = MoeConfig {
            num_experts: 2,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            moe_intermediate_size: 2,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: vec![],
            shared_expert: false,
        };
        let compact = SparseMoe::new(
            &config,
            0,
            1,
            3,
            Activation::Silu,
            VarBuilder::from_tensors(tensors, DType::F32, device),
            None,
        )?;
        let mut legacy = compact.clone();
        for index in 0..2 {
            let packed = gate_up.get(index)?;
            legacy.experts[index] = StreamingQwenMLP {
                gate_proj: StreamingLinear::new(packed.narrow(0, 0, 2)?.contiguous()?, None),
                up_proj: StreamingLinear::new(packed.narrow(0, 2, 2)?.contiguous()?, None),
                down_proj: StreamingLinear::new(down.get(index)?.contiguous()?, None),
                act_fn: Activation::Silu,
            };
            let expert = &compact.experts[index];
            assert_compact_f32(standard_weight(&expert.gate_proj));
            assert_compact_f32(standard_weight(&expert.up_proj));
            assert_compact_f32(standard_weight(&expert.down_proj));
            assert_eq!(expert.standard_weight_bytes(), Some(3 * 3 * 2 * 4));
        }
        let input = Tensor::from_vec(vec![0.5f32, 0.25, -0.5, 0.1, -0.2, 0.3], (1, 2, 3), device)?;
        let expected = legacy
            .forward(&input, device)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = compact
            .forward(&input, device)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn cached_standard_upload_compacts_views_and_counts_bias() -> Result<()> {
        let source = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((4, 3))?;
        let bias = Tensor::arange(0f32, 4f32, &Device::Cpu)?.narrow(0, 1, 2)?;
        let linear = StreamingLinear::new(source.narrow(0, 1, 2)?, Some(bias));
        assert_eq!(linear.standard_weight_bytes(), Some((6 + 2) * 4));
        let uploaded = linear.standard_to_device(&Device::Cpu)?;
        let StreamingLinear::Standard { weight, bias } = uploaded else {
            panic!("expected Standard projection")
        };
        assert_compact_f32(&weight);
        assert_compact_f32(bias.as_ref().unwrap());
        assert_eq!(
            weight.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 4., 5., 6., 7., 8.]
        );
        Ok(())
    }

    #[test]
    fn moe_request_forks_share_weights_and_expert_cache_with_private_kv() -> Result<()> {
        use super::super::QwenStreamingModelForCausalLM;
        use candle_transformers::models::qwen3::Config;

        let device = &Device::Cpu;
        let cfg = Config {
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            head_dim: 2,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 0,
            attention_bias: false,
        };
        let moe = MoeConfig::from_json(&serde_json::json!({
            "model_type": "upcycling-qwen2-moe", "share_flag": false,
            "num_experts": 2, "num_experts_per_tok": 2,
            "decoder_sparse_step": 1, "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 0, "norm_topk_prob": true
        }))
        .unwrap();
        let mut tensors = HashMap::new();
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![8, 4]),
            ("lm_head.weight", vec![8, 4]),
            ("model.norm.weight", vec![4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.mlp.gate.weight", vec![2, 4]),
            ("model.layers.0.mlp.experts.gate_up_proj", vec![2, 4, 4]),
            ("model.layers.0.mlp.experts.down_proj", vec![2, 4, 2]),
        ] {
            tensors.insert(name.to_string(), Tensor::ones(shape, DType::F32, device)?);
        }
        let mut source = QwenStreamingModelForCausalLM::new_with_moe(
            &cfg,
            VarBuilder::from_tensors(tensors, DType::F32, device),
            device,
            None,
            None,
            Some(&moe),
        )?;
        source.forward(&Tensor::new(&[[1u32, 2, 3]], device)?, 0)?;

        // Seed the real LRU with CPU tensors to test its shared lifetime without
        // requiring a GPU or changing the process-wide cache environment.
        let FeedForward::Sparse(sparse) = &mut source.base.layers[0].mlp else {
            panic!("expected sparse layer");
        };
        let bytes = sparse.experts[0].standard_weight_bytes().unwrap();
        let cache = Arc::new(ExpertCache {
            layer: 0,
            lru: Mutex::new(ExpertLru::new(bytes)),
        });
        cache.lru.lock().unwrap().insert(
            ExpertKey {
                index: 0,
                device: device.clone(),
            },
            sparse.experts[0].clone(),
            bytes,
        );
        sparse.expert_cache = Some(Arc::clone(&cache));

        let template = source.fork_with_empty_kv();
        let mut first = template.fork_with_empty_kv();
        let mut second = template.fork_with_empty_kv();
        let FeedForward::Sparse(source_sparse) = &source.base.layers[0].mlp else {
            panic!("expected sparse layer");
        };
        let (source_storage, _) = source.base.embed_tokens.storage_and_layout();
        for model in [&template, &first, &second] {
            let (storage, _) = model.base.embed_tokens.storage_and_layout();
            assert!(std::ptr::eq(&*source_storage, &*storage));
            let (head, _) = standard_weight(&model.lm_head).storage_and_layout();
            let (source_head, _) = standard_weight(&source.lm_head).storage_and_layout();
            assert!(std::ptr::eq(&*head, &*source_head));
            assert!(model.base.layers[0].self_attn.kv_cache.k().is_none());
            let FeedForward::Sparse(sparse) = &model.base.layers[0].mlp else {
                panic!("expected sparse layer");
            };
            assert!(Arc::ptr_eq(sparse.expert_cache.as_ref().unwrap(), &cache));
            for (projection, source_projection) in [
                (
                    &sparse.experts[0].gate_proj,
                    &source_sparse.experts[0].gate_proj,
                ),
                (
                    &sparse.experts[0].up_proj,
                    &source_sparse.experts[0].up_proj,
                ),
                (
                    &sparse.experts[0].down_proj,
                    &source_sparse.experts[0].down_proj,
                ),
            ] {
                let (storage, _) = standard_weight(projection).storage_and_layout();
                let (source_storage, _) = standard_weight(source_projection).storage_and_layout();
                assert!(std::ptr::eq(&*storage, &*source_storage));
            }
        }
        drop(source_storage);

        first.forward(&Tensor::new(&[[1u32, 2, 3, 4]], device)?, 0)?;
        second.forward(&Tensor::new(&[[5u32, 6]], device)?, 0)?;
        first.forward(&Tensor::new(&[[7u32]], device)?, 4)?;
        assert_eq!(
            first.base.layers[0]
                .self_attn
                .kv_cache
                .k()
                .unwrap()
                .dim(2)?,
            5
        );
        assert_eq!(
            second.base.layers[0]
                .self_attn
                .kv_cache
                .k()
                .unwrap()
                .dim(2)?,
            2
        );
        assert_eq!(
            source.base.layers[0]
                .self_attn
                .kv_cache
                .k()
                .unwrap()
                .dim(2)?,
            3
        );
        assert!(template.base.layers[0].self_attn.kv_cache.k().is_none());
        first.clear_kv_cache();
        assert_eq!(
            second.base.layers[0]
                .self_attn
                .kv_cache
                .k()
                .unwrap()
                .dim(2)?,
            2
        );
        let mut lru = cache.lru.lock().unwrap();
        assert_eq!(lru.current_bytes, bytes);
        assert_eq!(lru.stats.uploads, 1);
        assert!(
            lru.get(&ExpertKey {
                index: 0,
                device: device.clone()
            })
            .is_some()
        );
        assert_eq!(lru.stats.hits, 1);
        Ok(())
    }
}
