// Streaming attention entry points mirror the model's separate state tensors.
#![allow(clippy::too_many_arguments)]

use crate::executor::gemma4::{repeat_kv, Config};
use candle::{DType, Device, Result, Tensor};
use candle_core as candle;
use candle_core::quantized::QTensor;
use candle_nn::Activation;

use candle_nn::kv_cache::ConcatKvCache;

fn quantize_tensor(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, h, l, d) = x.dims4()?;
    let dev = x.device();
    let x_flat = x.transpose(1, 2)?.reshape((b * l, h * d))?;
    let max_abs = x_flat.abs()?.max_keepdim(1)?;
    let eps = Tensor::new(1e-5f32, dev)?.broadcast_as(max_abs.shape())?;
    let scale = (max_abs / 127.0)?.maximum(&eps)?;
    let quantized = (x_flat.broadcast_div(&scale)? + 128.0)?;
    let quantized = quantized.round()?.to_dtype(candle_core::DType::U8)?;
    let quantized = quantized.reshape((b, l, h, d))?.transpose(1, 2)?;
    let scale = scale.reshape((b, 1, l, 1))?.to_dtype(x.dtype())?;
    Ok((quantized, scale))
}

fn dequantize_tensor(quantized: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let q_float = quantized.to_dtype(scale.dtype())?;
    let q_offset = (q_float - 128.0)?;
    let dequantized = q_offset.broadcast_mul(scale)?;
    Ok(dequantized)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheDType {
    F16,
    Int8,
    Int4,
}

fn quantize_tensor_int4(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, h, l, d) = x.dims4()?;
    let dev = x.device();
    let x_flat = x.transpose(1, 2)?.reshape((b * l, h * d))?;
    let max_abs = x_flat.abs()?.max_keepdim(1)?;
    let eps = Tensor::new(1e-5f32, dev)?.broadcast_as(max_abs.shape())?;
    let scale = (max_abs / 7.5)?.maximum(&eps)?;

    let quantized = x.broadcast_div(&scale)?;
    let quantized = (quantized.clamp(-7.5f32, 7.5f32)? + 7.5)?;
    let quantized = quantized.round()?;

    let q_reshaped = quantized.reshape((b, h, l, d / 2, 2))?;
    let q1 = q_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
    let q2 = q_reshaped.narrow(4, 1, 1)?.squeeze(4)?;
    let packed = ((q1 * 16.0)? + &q2)?;
    let packed_u8 = packed.to_dtype(DType::U8)?;

    let scale = scale.reshape((b, 1, l, 1))?.to_dtype(x.dtype())?;
    Ok((packed_u8, scale))
}

fn dequantize_tensor_int4(quantized: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let (b, h, l, d_half) = quantized.dims4()?;
    let d = d_half * 2;
    let packed_f = quantized.to_dtype(DType::F32)?;
    let q1 = (packed_f.clone() / 16.0)?.floor()?;
    let q2 = (packed_f - (q1.clone() * 16.0)?)?;
    let q_unpacked = Tensor::stack(&[&q1, &q2], 4)?;
    let q_flat = q_unpacked.reshape((b, h, l, d))?.to_dtype(scale.dtype())?;
    let q_centered = (q_flat - 7.5)?;
    let dequantized = q_centered.broadcast_mul(scale)?;
    Ok(dequantized)
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        dim: usize,
        rope_freq: f64,
        max_position_embeddings: usize,
        partial_rotary_factor: Option<f64>,
        dev: &Device,
    ) -> Result<Self> {
        let rotary_dim = if let Some(factor) = partial_rotary_factor {
            (dim as f64 * factor) as usize
        } else {
            dim
        };

        // Note: compute inv_freq in F32 to avoid precision loss/overflow in F16 range
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let mut sin = freqs.sin()?.to_dtype(dtype)?;
        let mut cos = freqs.cos()?.to_dtype(dtype)?;

        if rotary_dim < dim {
            let pad_len = (dim - rotary_dim) / 2;
            let zeros = Tensor::zeros((max_position_embeddings, pad_len), dtype, dev)?;
            let ones = Tensor::ones((max_position_embeddings, pad_len), dtype, dev)?;
            sin = Tensor::cat(&[&sin, &zeros], candle::D::Minus1)?;
            cos = Tensor::cat(&[&cos, &ones], candle::D::Minus1)?;
        }

        Ok(Self { sin, cos })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(
            &q.contiguous()?,
            &cos.to_device(q.device())?,
            &sin.to_device(q.device())?,
        )?;
        let k_embed = candle_nn::rotary_emb::rope(
            &k.contiguous()?,
            &cos.to_device(q.device())?,
            &sin.to_device(q.device())?,
        )?;
        Ok((q_embed, k_embed))
    }
}

fn gemma_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(candle::D::Minus1)?;
    let x_f32 = x.to_dtype(internal_dtype)?;
    let norm_x = (x_f32.sqr()?.sum_keepdim(candle::D::Minus1)? / hidden_size as f64)?;
    let x_normed = x_f32.broadcast_div(&(norm_x.clone() + eps)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)
}

fn matmul_3d_2d(xs: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_size) = xs.dims3()?;
    let xs_2d = xs.reshape((b_sz * seq_len, hidden_size))?;
    let out_2d = xs_2d.matmul(w)?;
    let out_dim = out_2d.dim(candle::D::Minus1)?;
    out_2d.reshape((b_sz, seq_len, out_dim))
}

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], candle::D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

struct LayerWeights {
    q_proj: QTensor,
    k_proj: QTensor,
    v_proj: Option<QTensor>,
    o_proj: QTensor,
    q_norm: QTensor,
    k_norm: QTensor,
    gate_proj: QTensor,
    up_proj: QTensor,
    down_proj: QTensor,
    input_layernorm: QTensor,
    pre_feedforward_layernorm: QTensor,
    post_attention_layernorm: QTensor,
    post_feedforward_layernorm: QTensor,
    layer_output_scale: QTensor,
}

pub struct Gemma4StreamingModel {
    embed_tokens: Tensor, // CPU float tensor [vocab_size, hidden_size]
    layer_weights: Vec<LayerWeights>,
    norm: Tensor,    // CPU float tensor [hidden_size]
    lm_head: Tensor, // CPU float tensor [vocab_size, hidden_size] (linked to embed_tokens)
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    num_hidden_layers: usize,
    final_logit_softcapping: Option<f64>,
    sliding_window_pattern: Vec<bool>,
    head_dim: usize,
    global_head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_global_key_value_heads: usize,
    rms_norm_eps: f64,
    attn_logit_softcapping: Option<f64>,
    kv_caches: Vec<KvCache>,
    rotary_emb_sliding: RotaryEmbedding,
    rotary_emb_global: RotaryEmbedding,
    kv_cache_dtype: KvCacheDType,
    k_scale_caches: Vec<ConcatKvCache>,
    v_scale_caches: Vec<ConcatKvCache>,
    pub offloaded_layers: Option<usize>,
}

impl Gemma4StreamingModel {
    pub fn new(gguf_path: &std::path::Path, cfg: &Config, device: &Device) -> Result<Self> {
        Self::new_with_offload(gguf_path, cfg, device, None)
    }

    pub fn new_with_offload(
        gguf_path: &std::path::Path,
        cfg: &Config,
        device: &Device,
        offloaded_layers: Option<usize>,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(gguf_path).map_err(candle::Error::wrap)?;
        let gguf = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        let dtype = match device {
            Device::Cpu => DType::F32,
            Device::Cuda(_) => DType::F16,
            _ => DType::F32,
        };

        // 1. Load embed_tokens on CPU as float tensor
        let embed_qtensor = gguf.tensor(&mut file, "token_embd.weight", &Device::Cpu)?;
        let embed_tokens = embed_qtensor.dequantize(&Device::Cpu)?.to_dtype(dtype)?;

        // 2. Load norm on CPU as float tensor
        let norm_qtensor = gguf.tensor(&mut file, "output_norm.weight", &Device::Cpu)?;
        let norm = norm_qtensor.dequantize(&Device::Cpu)?.to_dtype(dtype)?;

        // 3. lm_head is tied to embed_tokens
        let lm_head = embed_tokens.clone();

        // 4. Load layers
        let mut layer_weights = Vec::with_capacity(cfg.num_hidden_layers);
        let mut kv_caches = Vec::with_capacity(cfg.num_hidden_layers);
        let mut sliding_window_pattern = Vec::with_capacity(cfg.num_hidden_layers);

        for i in 0..cfg.num_hidden_layers {
            let is_sliding = (i + 1) % cfg.sliding_window_pattern > 0;
            sliding_window_pattern.push(is_sliding);

            let q_proj =
                gguf.tensor(&mut file, &format!("blk.{}.attn_q.weight", i), &Device::Cpu)?;
            let k_proj =
                gguf.tensor(&mut file, &format!("blk.{}.attn_k.weight", i), &Device::Cpu)?;

            let has_v_proj = is_sliding || !cfg.attention_k_eq_v;
            let v_proj = if has_v_proj {
                Some(gguf.tensor(&mut file, &format!("blk.{}.attn_v.weight", i), &Device::Cpu)?)
            } else {
                None
            };

            let o_proj = gguf.tensor(
                &mut file,
                &format!("blk.{}.attn_output.weight", i),
                &Device::Cpu,
            )?;

            let q_norm = gguf.tensor(
                &mut file,
                &format!("blk.{}.attn_q_norm.weight", i),
                &Device::Cpu,
            )?;
            let k_norm = gguf.tensor(
                &mut file,
                &format!("blk.{}.attn_k_norm.weight", i),
                &Device::Cpu,
            )?;

            let gate_proj = gguf.tensor(
                &mut file,
                &format!("blk.{}.ffn_gate.weight", i),
                &Device::Cpu,
            )?;
            let up_proj =
                gguf.tensor(&mut file, &format!("blk.{}.ffn_up.weight", i), &Device::Cpu)?;
            let down_proj = gguf.tensor(
                &mut file,
                &format!("blk.{}.ffn_down.weight", i),
                &Device::Cpu,
            )?;

            let input_layernorm = gguf.tensor(
                &mut file,
                &format!("blk.{}.attn_norm.weight", i),
                &Device::Cpu,
            )?;
            let pre_feedforward_layernorm = gguf.tensor(
                &mut file,
                &format!("blk.{}.ffn_norm.weight", i),
                &Device::Cpu,
            )?;
            let post_attention_layernorm = gguf.tensor(
                &mut file,
                &format!("blk.{}.post_attention_norm.weight", i),
                &Device::Cpu,
            )?;
            let post_feedforward_layernorm = gguf.tensor(
                &mut file,
                &format!("blk.{}.post_ffw_norm.weight", i),
                &Device::Cpu,
            )?;
            let layer_output_scale = gguf.tensor(
                &mut file,
                &format!("blk.{}.layer_output_scale.weight", i),
                &Device::Cpu,
            )?;

            layer_weights.push(LayerWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                gate_proj,
                up_proj,
                down_proj,
                input_layernorm,
                pre_feedforward_layernorm,
                post_attention_layernorm,
                post_feedforward_layernorm,
                layer_output_scale,
            });

            let sliding_window = is_sliding.then_some(cfg.sliding_window);
            let kv_cache = if let Some(sliding_window) = sliding_window {
                KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, sliding_window))
            } else {
                KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                    2,
                    cfg.max_position_embeddings,
                ))
            };
            kv_caches.push(kv_cache);
        }

        let rotary_emb_sliding = RotaryEmbedding::new(
            dtype,
            cfg.head_dim,
            cfg.rope_local_base_freq,
            cfg.max_position_embeddings,
            None,
            device,
        )?;

        let rotary_emb_global = RotaryEmbedding::new(
            dtype,
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.max_position_embeddings,
            Some(cfg.partial_rotary_factor),
            device,
        )?;

        let kv_cache_dtype = std::env::var("BLOOM_KV_CACHE_DTYPE")
            .map(|v| match v.to_lowercase().as_str() {
                "int8" => KvCacheDType::Int8,
                "int4" => KvCacheDType::Int4,
                _ => KvCacheDType::F16,
            })
            .unwrap_or(KvCacheDType::F16);
        let mut k_scale_caches = Vec::with_capacity(cfg.num_hidden_layers);
        let mut v_scale_caches = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            k_scale_caches.push(ConcatKvCache::new(2));
            v_scale_caches.push(ConcatKvCache::new(2));
        }

        Ok(Self {
            embed_tokens,
            layer_weights,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            num_hidden_layers: cfg.num_hidden_layers,
            final_logit_softcapping: cfg.final_logit_softcapping,
            sliding_window_pattern,
            head_dim: cfg.head_dim,
            global_head_dim: cfg.global_head_dim,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            num_global_key_value_heads: cfg.num_global_key_value_heads,
            rms_norm_eps: cfg.rms_norm_eps,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            kv_caches,
            rotary_emb_sliding,
            rotary_emb_global,
            kv_cache_dtype,
            k_scale_caches,
            v_scale_caches,
            offloaded_layers,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }

        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;

        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;

        Ok((Some(mask), Some(sliding_mask)))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        _layer_idx: usize,
        xs: &Tensor,
        weights: &LayerWeights,
        head_dim: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        rotary_emb: &RotaryEmbedding,
        device: &Device,
        rms_norm_eps: f64,
        attn_logit_softcapping: Option<f64>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut KvCache,
        kv_cache_dtype: KvCacheDType,
        k_scale_cache: &mut ConcatKvCache,
        _v_scale_cache: &mut ConcatKvCache,
    ) -> Result<Tensor> {
        let num_kv_groups = num_attention_heads / num_kv_heads;
        let residual = xs;
        let dtype = xs.dtype();

        // 1. input_layernorm
        let input_layernorm_w = weights
            .input_layernorm
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let xs_norm = gemma_rms_norm(xs, &input_layernorm_w, rms_norm_eps)?;

        // 2. Attention
        let q_proj_w = weights
            .q_proj
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let query_states = matmul_3d_2d(&xs_norm, &q_proj_w.transpose(0, 1)?)?;

        let k_proj_w = weights
            .k_proj
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let key_states = matmul_3d_2d(&xs_norm, &k_proj_w.transpose(0, 1)?)?;

        let value_states = if let Some(v_proj) = &weights.v_proj {
            let v_proj_w = v_proj
                .dequantize(&Device::Cpu)?
                .to_device(device)?
                .to_dtype(dtype)?;
            matmul_3d_2d(&xs_norm, &v_proj_w.transpose(0, 1)?)?
        } else {
            key_states.clone()
        };

        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = query_states
            .reshape((b_sz, q_len, num_attention_heads, head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?;

        let q_norm_w = weights
            .q_norm
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let k_norm_w = weights
            .k_norm
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;

        let query_states = gemma_rms_norm(&query_states, &q_norm_w, rms_norm_eps)?;
        let key_states = gemma_rms_norm(&key_states, &k_norm_w, rms_norm_eps)?;

        let (query_states, key_states) =
            rotary_emb.apply(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match kv_cache_dtype {
            KvCacheDType::Int8 => {
                let (q_k, scale_k) = quantize_tensor(&key_states)?;
                let (q_v, scale_v) = quantize_tensor(&value_states)?;

                let (full_q_k, full_q_v) = match kv_cache {
                    KvCache::Normal(cache) => cache.append(&q_k, &q_v)?,
                    KvCache::Rotating(cache) => cache.append(&q_k, &q_v)?,
                };
                let (full_scale_k, full_scale_v) = k_scale_cache.append(&scale_k, &scale_v)?;

                let k = dequantize_tensor(&full_q_k, &full_scale_k)?;
                let v = dequantize_tensor(&full_q_v, &full_scale_v)?;
                (k, v)
            }
            KvCacheDType::Int4 => {
                let (q_k, scale_k) = quantize_tensor_int4(&key_states)?;
                let (q_v, scale_v) = quantize_tensor_int4(&value_states)?;

                let (full_q_k, full_q_v) = match kv_cache {
                    KvCache::Normal(cache) => cache.append(&q_k, &q_v)?,
                    KvCache::Rotating(cache) => cache.append(&q_k, &q_v)?,
                };
                let (full_scale_k, full_scale_v) = k_scale_cache.append(&scale_k, &scale_v)?;

                let k = dequantize_tensor_int4(&full_q_k, &full_scale_k)?;
                let v = dequantize_tensor_int4(&full_q_v, &full_scale_v)?;
                (k, v)
            }
            KvCacheDType::F16 => match kv_cache {
                KvCache::Normal(cache) => cache.append(&key_states, &value_states)?,
                KvCache::Rotating(cache) => cache.append(&key_states, &value_states)?,
            },
        };

        let key_states = repeat_kv(key_states, num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, num_kv_groups)?.contiguous()?;

        // Attention score calculation
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        if let Some(sc) = attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let attn_output = attn_weights.matmul(&value_states)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?;

        // Attention output projection and normalization in F32 to avoid overflow
        let attn_output_f32 = attn_output.to_dtype(DType::F32)?;
        let o_proj_w_f32 = weights
            .o_proj
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let attn_output_f32 = matmul_3d_2d(&attn_output_f32, &o_proj_w_f32.transpose(0, 1)?)?;

        let post_attention_layernorm_w_f32 = weights
            .post_attention_layernorm
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let attn_output_f32 = gemma_rms_norm(
            &attn_output_f32,
            &post_attention_layernorm_w_f32,
            rms_norm_eps,
        )?;
        let attn_output = attn_output_f32.to_dtype(dtype)?;

        let xs = (attn_output + residual)?;
        let residual = &xs;

        // 3. MLP
        let pre_feedforward_layernorm_w = weights
            .pre_feedforward_layernorm
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let xs_norm = gemma_rms_norm(&xs, &pre_feedforward_layernorm_w, rms_norm_eps)?;

        let gate_proj_w = weights
            .gate_proj
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let gate_out = matmul_3d_2d(&xs_norm, &gate_proj_w.transpose(0, 1)?)?;
        let gate_activated = gate_out.apply(&Activation::GeluPytorchTanh)?;

        let up_proj_w = weights
            .up_proj
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let up_out = matmul_3d_2d(&xs_norm, &up_proj_w.transpose(0, 1)?)?;

        // Elementwise multiplication, down projection, and post-normalization in F32 to avoid overflow
        let mlp_out_f32 = (gate_activated.to_dtype(DType::F32)? * up_out.to_dtype(DType::F32)?)?;
        let down_proj_w_f32 = weights
            .down_proj
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let mlp_out_f32 = matmul_3d_2d(&mlp_out_f32, &down_proj_w_f32.transpose(0, 1)?)?;

        let post_feedforward_layernorm_w_f32 = weights
            .post_feedforward_layernorm
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let mlp_out_f32 = gemma_rms_norm(
            &mlp_out_f32,
            &post_feedforward_layernorm_w_f32,
            rms_norm_eps,
        )?;

        // Gemma 4 residual scaling (layer_output_scale)
        let layer_output_scale_w_f32 = weights
            .layer_output_scale
            .dequantize(&Device::Cpu)?
            .to_device(device)?
            .to_dtype(DType::F32)?;
        let output_unscaled_f32 = (mlp_out_f32 + residual.to_dtype(DType::F32)?)?;
        let output_f32 = output_unscaled_f32.broadcast_mul(&layer_output_scale_w_f32)?;
        let output = output_f32.to_dtype(dtype)?;
        Ok(output)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        // 1. Embed tokens on CPU and copy selection to CUDA
        let flat_input = input_ids.flatten_all()?.to_device(&Device::Cpu)?;
        let xs_cpu = self.embed_tokens.index_select(&flat_input, 0)?;
        let xs_cpu = xs_cpu.reshape((b_size, seq_len, self.hidden_size))?;
        let mut xs = xs_cpu.to_device(&self.device)?;

        // Embedding scaling
        xs = (xs * (self.hidden_size as f64).sqrt())?;

        // 2. Prepare masks
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset)?;

        // 3. Forward layers
        let placements = crate::core::memory::LayerPlacementStrategy::new(
            self.num_hidden_layers,
            self.offloaded_layers,
        );

        for layer_idx in 0..self.num_hidden_layers {
            let target_device_owned = placements.device_for_layer(layer_idx, &self.device);
            let target_device = &target_device_owned;

            let mask = if self.sliding_window_pattern[layer_idx] {
                &sliding_attention_mask
            } else {
                &attention_mask
            };

            let xs_layer = xs.to_device(target_device)?;
            let mask_layer = if let Some(m) = mask {
                Some(m.to_device(target_device)?)
            } else {
                None
            };

            let is_sliding = self.sliding_window_pattern[layer_idx];
            let head_dim = if is_sliding {
                self.head_dim
            } else {
                self.global_head_dim
            };
            let num_kv_heads = if is_sliding {
                self.num_key_value_heads
            } else {
                self.num_global_key_value_heads
            };
            let rotary_emb = if is_sliding {
                &self.rotary_emb_sliding
            } else {
                &self.rotary_emb_global
            };

            xs = Self::forward_layer(
                layer_idx,
                &xs_layer,
                &self.layer_weights[layer_idx],
                head_dim,
                self.num_attention_heads,
                num_kv_heads,
                rotary_emb,
                target_device,
                self.rms_norm_eps,
                self.attn_logit_softcapping,
                mask_layer.as_ref(),
                seqlen_offset,
                &mut self.kv_caches[layer_idx],
                self.kv_cache_dtype,
                &mut self.k_scale_caches[layer_idx],
                &mut self.v_scale_caches[layer_idx],
            )?;
        }

        // Move hidden states back to self.device for final norm and head projection
        xs = xs.to_device(&self.device)?;

        // 4. Final norm
        let norm_w = self.norm.to_device(&self.device)?;
        let logits_input =
            gemma_rms_norm(&xs.narrow(1, seq_len - 1, 1)?, &norm_w, self.rms_norm_eps)?;

        // 5. LM head on CPU to save CUDA memory and PCIe bandwidth
        let logits_input_cpu = logits_input.to_device(&Device::Cpu)?;
        let logits = matmul_3d_2d(&logits_input_cpu, &self.lm_head.transpose(0, 1)?)?;

        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };

        // Move logits back to CUDA device (shape: [batch, 1, vocab_size])
        logits.to_device(&self.device)
    }

    pub fn clear_kv_cache(&mut self) {
        for cache in self.kv_caches.iter_mut() {
            match cache {
                KvCache::Normal(c) => c.reset(),
                KvCache::Rotating(c) => c.reset(),
            }
        }
        for cache in self.k_scale_caches.iter_mut() {
            cache.reset();
        }
        for cache in self.v_scale_caches.iter_mut() {
            cache.reset();
        }
    }

    pub fn extract_kv(
        &self,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
        _kv_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if layer_idx >= self.kv_caches.len() {
            return Err(candle::Error::Msg(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.kv_caches.len()
            )));
        }
        let cache = &self.kv_caches[layer_idx];
        let k_opt = match cache {
            KvCache::Normal(c) => c.k()?,
            KvCache::Rotating(c) => c.k()?,
        };
        let k = k_opt.ok_or_else(|| {
            candle::Error::Msg(format!("KV k cache empty for layer {}", layer_idx))
        })?;

        let v_opt = match cache {
            KvCache::Normal(c) => c.v()?,
            KvCache::Rotating(c) => c.v()?,
        };
        let v = v_opt.ok_or_else(|| {
            candle::Error::Msg(format!("KV v cache empty for layer {}", layer_idx))
        })?;

        if seq_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let current_seq_len = k.dim(2)?;
        if start_pos + seq_len > current_seq_len {
            return Err(candle::Error::Msg(format!(
                "extract_kv out of range: start_pos={} seq_len={} but cache holds {} tokens",
                start_pos, seq_len, current_seq_len
            )));
        }
        let k_slice = k.narrow(2, start_pos, seq_len)?;
        let v_slice = v.narrow(2, start_pos, seq_len)?;
        let k_f32 = k_slice.to_dtype(DType::F32)?.flatten_all()?;
        let v_f32 = v_slice.to_dtype(DType::F32)?.flatten_all()?;
        let k_vec = k_f32.to_vec1::<f32>()?;
        let v_vec = v_f32.to_vec1::<f32>()?;
        Ok((k_vec, v_vec))
    }

    pub fn inject_kv(
        &mut self,
        layer_idx: usize,
        start_pos: usize,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        _kv_dim: usize,
    ) -> Result<()> {
        if layer_idx >= self.kv_caches.len() {
            return Err(candle::Error::Msg(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.kv_caches.len()
            )));
        }
        if seq_len == 0 {
            return Ok(());
        }
        let shape = (1usize, num_kv_heads, seq_len, head_dim);
        let k_tensor = Tensor::from_slice(keys, shape, &self.device)?
            .to_dtype(self.lm_head.dtype())?
            .contiguous()?;
        let v_tensor = Tensor::from_slice(values, shape, &self.device)?
            .to_dtype(self.lm_head.dtype())?
            .contiguous()?;

        let kv_cache = &mut self.kv_caches[layer_idx];
        match kv_cache {
            KvCache::Normal(c) => {
                let current_seq_len = c.current_seq_len();
                if start_pos == 0 && current_seq_len == 0 {
                    c.append(&k_tensor, &v_tensor)?;
                } else if start_pos + seq_len <= current_seq_len {
                    if let Some(k_buf) = c.k()? {
                        k_buf.slice_set(&k_tensor, 2, start_pos)?;
                    }
                    if let Some(v_buf) = c.v()? {
                        v_buf.slice_set(&v_tensor, 2, start_pos)?;
                    }
                }
            }
            KvCache::Rotating(_) => {
                // Sliding window attention not mutable via direct slice_set
            }
        }
        Ok(())
    }

    pub fn extract_kv_tensor(
        &self,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self.kv_caches.len() {
            return Err(candle::Error::Msg(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.kv_caches.len()
            )));
        }
        let cache = &self.kv_caches[layer_idx];
        let k_opt = match cache {
            KvCache::Normal(c) => c.k()?,
            KvCache::Rotating(c) => c.k()?,
        };
        let k = k_opt.ok_or_else(|| {
            candle::Error::Msg(format!("KV k cache empty for layer {}", layer_idx))
        })?;

        let v_opt = match cache {
            KvCache::Normal(c) => c.v()?,
            KvCache::Rotating(c) => c.v()?,
        };
        let v = v_opt.ok_or_else(|| {
            candle::Error::Msg(format!("KV v cache empty for layer {}", layer_idx))
        })?;

        if seq_len == 0 {
            return Ok((k.zeros_like()?, v.zeros_like()?));
        }
        let current_seq_len = k.dim(2)?;
        if start_pos + seq_len > current_seq_len {
            return Err(candle::Error::Msg(format!(
                "extract_kv_tensor out of range: start_pos={} seq_len={} but cache holds {} tokens",
                start_pos, seq_len, current_seq_len
            )));
        }
        let k_slice = k.narrow(2, start_pos, seq_len)?;
        let v_slice = v.narrow(2, start_pos, seq_len)?;
        Ok((k_slice, v_slice))
    }

    pub fn inject_kv_tensor(
        &mut self,
        layer_idx: usize,
        start_pos: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<()> {
        if layer_idx >= self.kv_caches.len() {
            return Err(candle::Error::Msg(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.kv_caches.len()
            )));
        }
        let kv_cache = &mut self.kv_caches[layer_idx];
        match kv_cache {
            KvCache::Normal(c) => {
                let current_seq_len = c.current_seq_len();
                if start_pos == 0 && current_seq_len == 0 {
                    c.append(keys, values)?;
                } else if start_pos + keys.dim(2)? <= current_seq_len {
                    if let Some(k_buf) = c.k()? {
                        k_buf.slice_set(keys, 2, start_pos)?;
                    }
                    if let Some(v_buf) = c.v()? {
                        v_buf.slice_set(values, 2, start_pos)?;
                    }
                }
            }
            KvCache::Rotating(_) => {
                // Sliding window attention not mutable via direct slice_set
            }
        }
        Ok(())
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        for cache in self.kv_caches.iter_mut() {
            match cache {
                KvCache::Normal(c) => {
                    if let Some(k) = c.k()? {
                        let current_seq_len = k.dim(2)?;
                        if len < current_seq_len {
                            let k_new = k.narrow(2, 0, len)?;
                            let v_new = c
                                .v()?
                                .ok_or_else(|| {
                                    candle_core::Error::Msg("KV cache value is missing".into())
                                })?
                                .narrow(2, 0, len)?;
                            c.reset();
                            c.append(&k_new, &v_new)?;
                        }
                    }
                }
                KvCache::Rotating(_) => {}
            }
        }
        for cache in self.k_scale_caches.iter_mut() {
            if let Some(k) = cache.k() {
                let current_seq_len = k.dim(2)?;
                if len < current_seq_len {
                    let k_new = k.narrow(2, 0, len)?;
                    let v_new = cache
                        .v()
                        .ok_or_else(|| {
                            candle_core::Error::Msg("K-scale cache value is missing".into())
                        })?
                        .narrow(2, 0, len)?;
                    cache.reset();
                    cache.append(&k_new, &v_new)?;
                }
            }
        }
        for cache in self.v_scale_caches.iter_mut() {
            if let Some(k) = cache.k() {
                let current_seq_len = k.dim(2)?;
                if len < current_seq_len {
                    let k_new = k.narrow(2, 0, len)?;
                    let v_new = cache
                        .v()
                        .ok_or_else(|| {
                            candle_core::Error::Msg("V-scale cache value is missing".into())
                        })?
                        .narrow(2, 0, len)?;
                    cache.reset();
                    cache.append(&k_new, &v_new)?;
                }
            }
        }
        Ok(())
    }
}
