// Streaming attention entry points mirror the model's separate state tensors.
#![allow(clippy::too_many_arguments)]

use bloomai_core::BloomError;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::ConcatKvCache, Activation, VarBuilder};
use candle_transformers::models::qwen3::Config;
use std::sync::{Arc, Mutex};

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
pub struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
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

pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
        xs.unsqueeze(2)?
            .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))
    }
}

#[derive(Debug, Clone)]
pub enum StreamingLinear {
    Standard {
        weight: Tensor,       // held on CPU
        bias: Option<Tensor>, // held on CPU
    },
    Quantized {
        qweight: Tensor,       // held on CPU
        scales: Tensor,        // held on CPU
        qzeros: Tensor,        // held on CPU
        g_idx: Option<Tensor>, // held on CPU
        bias: Option<Tensor>,  // held on CPU
        quantizer: Option<Arc<super::metal_quant::MetalQuantizer>>,
        total_elements: usize,
    },
}

impl StreamingLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        StreamingLinear::Standard { weight, bias }
    }

    pub fn new_with_quantizer(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        quantizer: Option<&Arc<super::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        if vb.get((out_dim, in_dim / 8), "qweight").is_ok() {
            let qweight = vb.get((out_dim, in_dim / 8), "qweight")?;
            let scales = vb.get(out_dim, "scales")?;
            let qzeros = vb.get(out_dim, "qzeros")?;
            let g_idx = vb.get(in_dim, "g_idx").ok();
            let b = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            let total_elements = out_dim * in_dim;

            if let (Some(q), Device::Metal(_)) = (quantizer, vb.device()) {
                Ok(StreamingLinear::Quantized {
                    qweight,
                    scales,
                    qzeros,
                    g_idx,
                    bias: b,
                    quantizer: Some(q.clone()),
                    total_elements,
                })
            } else {
                // Load-time CPU Fallback: dequantize directly to standard weight.
                let decompressed = if let Some(g) = &g_idx {
                    super::metal_quant::MetalQuantizer::dequantize_gptq_cpu(
                        &qweight,
                        &scales,
                        &qzeros,
                        g,
                        total_elements,
                    )?
                } else {
                    super::metal_quant::MetalQuantizer::dequantize_awq_cpu(
                        &qweight,
                        &scales,
                        &qzeros,
                        total_elements,
                    )?
                };
                let weight = decompressed.reshape((out_dim, in_dim))?;
                let weight_cpu = weight.to_device(&Device::Cpu)?;
                Ok(StreamingLinear::Standard {
                    weight: weight_cpu,
                    bias: b,
                })
            }
        } else {
            let weight = vb.get((out_dim, in_dim), "weight")?;
            let bias = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(StreamingLinear::Standard { weight, bias })
        }
    }

    pub fn forward(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        let (w_cuda, bias_cuda) = match self {
            StreamingLinear::Standard { weight, bias } => {
                let w_cuda = weight.to_device(device)?;
                let bias_cuda = if let Some(b) = bias {
                    Some(b.to_device(device)?)
                } else {
                    None
                };
                (w_cuda, bias_cuda)
            }
            StreamingLinear::Quantized {
                qweight,
                scales,
                qzeros,
                g_idx,
                bias,
                quantizer,
                total_elements,
            } => {
                let qw_gpu = qweight.to_device(device)?;
                let sc_gpu = scales.to_device(device)?;
                let qz_gpu = qzeros.to_device(device)?;
                let gi_gpu = if let Some(gi) = g_idx {
                    Some(gi.to_device(device)?)
                } else {
                    None
                };

                let dequant = if let (Some(q), Device::Metal(_)) = (quantizer, device) {
                    if let Some(g) = &gi_gpu {
                        q.dequantize_gptq(&qw_gpu, &sc_gpu, &qz_gpu, g, *total_elements)?
                    } else {
                        q.dequantize_awq(&qw_gpu, &sc_gpu, &qz_gpu, *total_elements)?
                    }
                } else {
                    let dec = if let Some(g) = g_idx {
                        super::metal_quant::MetalQuantizer::dequantize_gptq_cpu(
                            qweight,
                            scales,
                            qzeros,
                            g,
                            *total_elements,
                        )?
                    } else {
                        super::metal_quant::MetalQuantizer::dequantize_awq_cpu(
                            qweight,
                            scales,
                            qzeros,
                            *total_elements,
                        )?
                    };
                    dec.to_device(device)?
                };

                let bias_gpu = if let Some(b) = bias {
                    Some(b.to_device(device)?)
                } else {
                    None
                };
                (dequant, bias_gpu)
            }
        };

        let w_t = w_cuda.transpose(0, 1)?;
        // Flatten batch dims to 2D for candle matmul compatibility, then restore shape
        let (b_shape, last_dim) = {
            let dims = x.dims();
            (dims[..dims.len() - 1].to_vec(), dims[dims.len() - 1])
        };
        let x_2d = x.reshape((b_shape.iter().product::<usize>(), last_dim))?;
        let mut out = x_2d.matmul(&w_t)?;
        // Restore original batch dimensions
        let mut out_shape = b_shape;
        out_shape.push(out.dim(1)?);
        out = out.reshape(out_shape)?;
        if let Some(b) = bias_cuda {
            out = out.broadcast_add(&b)?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingRmsNorm {
    weight: Tensor, // held on CPU
    eps: f64,
}

impl StreamingRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x_f32 = x.to_dtype(internal_dtype)?;
        let norm_x = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        let w_cuda = self.weight.to_device(device)?;
        x_normed.broadcast_mul(&w_cuda)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingQwenAttention {
    q_proj: StreamingLinear,
    k_proj: StreamingLinear,
    v_proj: StreamingLinear,
    o_proj: StreamingLinear,
    q_norm: Option<StreamingRmsNorm>,
    k_norm: Option<StreamingRmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: ConcatKvCache,
    kv_cache_dtype: KvCacheDType,
    k_scale_cache: ConcatKvCache,
    v_scale_cache: ConcatKvCache,
}

impl StreamingQwenAttention {
    pub fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Proj
        let q = self.q_proj.forward(x, device)?;
        let k = self.k_proj.forward(x, device)?;
        let v = self.v_proj.forward(x, device)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per‑head RMSNorm (Qwen3 only)
        let q = if let Some(q_norm) = &self.q_norm {
            let q_flat = q.flatten(0, 2)?;
            let q_flat = q_norm.forward(&q_flat, device)?;
            q_flat.reshape((b, self.num_heads, l, self.head_dim))?
        } else {
            q
        };
        let k = if let Some(k_norm) = &self.k_norm {
            let k_flat = k.flatten(0, 2)?;
            let k_flat = k_norm.forward(&k_flat, device)?;
            k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?
        } else {
            k
        };

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV cache
        let (k, v) = match self.kv_cache_dtype {
            KvCacheDType::Int8 => {
                let (q_k, scale_k) = quantize_tensor(&k)?;
                let (q_v, scale_v) = quantize_tensor(&v)?;
                let (full_q_k, full_q_v) = self.kv_cache.append(&q_k, &q_v)?;
                let (full_scale_k, full_scale_v) = self.k_scale_cache.append(&scale_k, &scale_v)?;
                let k = dequantize_tensor(&full_q_k, &full_scale_k)?;
                let v = dequantize_tensor(&full_q_v, &full_scale_v)?;
                (k, v)
            }
            KvCacheDType::Int4 => {
                let (q_k, scale_k) = quantize_tensor_int4(&k)?;
                let (q_v, scale_v) = quantize_tensor_int4(&v)?;
                let (full_q_k, full_q_v) = self.kv_cache.append(&q_k, &q_v)?;
                let (full_scale_k, full_scale_v) = self.k_scale_cache.append(&scale_k, &scale_v)?;
                let k = dequantize_tensor_int4(&full_q_k, &full_scale_k)?;
                let v = dequantize_tensor_int4(&full_q_v, &full_scale_v)?;
                (k, v)
            }
            KvCacheDType::F16 => self.kv_cache.append(&k, &v)?,
        };

        // 6. GQA repeat_kv
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // 7. Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        // 8. Output proj
        let out = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out, device)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
        self.k_scale_cache.reset();
        self.v_scale_cache.reset();
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        if let Some(k) = self.kv_cache.k() {
            let current_seq_len = k.dim(2)?;
            if len < current_seq_len {
                let k_new = k.narrow(2, 0, len)?;
                let v_new = self
                    .kv_cache
                    .v()
                    .ok_or_else(|| candle_core::Error::Msg("KV cache value is missing".into()))?
                    .narrow(2, 0, len)?;
                self.kv_cache.reset();
                self.kv_cache.append(&k_new, &v_new)?;
            }
        }
        if let Some(k_scale) = self.k_scale_cache.k() {
            let current_seq_len = k_scale.dim(2)?;
            if len < current_seq_len {
                let k_scale_new = k_scale.narrow(2, 0, len)?;
                let v_scale_new = self
                    .k_scale_cache
                    .v()
                    .ok_or_else(|| {
                        candle_core::Error::Msg("KV scale cache value is missing".into())
                    })?
                    .narrow(2, 0, len)?;
                self.k_scale_cache.reset();
                self.k_scale_cache.append(&k_scale_new, &v_scale_new)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StreamingQwenMLP {
    gate_proj: StreamingLinear,
    up_proj: StreamingLinear,
    down_proj: StreamingLinear,
    act_fn: Activation,
}

impl StreamingQwenMLP {
    pub fn forward(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        let gate_out = self.gate_proj.forward(x, device)?;
        let lhs = gate_out.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(x, device)?;
        let prod = (lhs * rhs)?;
        self.down_proj.forward(&prod, device)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingDecoderLayer {
    self_attn: StreamingQwenAttention,
    mlp: StreamingQwenMLP,
    ln1: StreamingRmsNorm,
    ln2: StreamingRmsNorm,
}

impl StreamingDecoderLayer {
    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let h = self.ln1.forward(x, device)?;
        let h = self.self_attn.forward(&h, mask, offset, device)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x, device)?;
        let h2 = self.mlp.forward(&h2, device)?;
        x + h2
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct StreamingQwenModel {
    embed_tokens: Tensor, // held directly on CUDA device
    pub layers: Vec<StreamingDecoderLayer>,
    norm: StreamingRmsNorm,
    device: Device,
    dtype: DType,
}

impl StreamingQwenModel {
    pub fn new(cfg: &Config, vb: VarBuilder, target_device: &Device) -> Result<Self> {
        Self::new_with_quantizer(cfg, vb, target_device, None)
    }

    pub fn new_with_quantizer(
        cfg: &Config,
        vb: VarBuilder,
        target_device: &Device,
        quantizer: Option<Arc<super::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        // Load embed_tokens weight on CPU and move immediately to CUDA device
        let embed_tokens_weight = vb
            .pp("model.embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = embed_tokens_weight.to_device(target_device)?;

        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, target_device)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");

        for i in 0..cfg.num_hidden_layers {
            let vb_layer = vb_l.pp(i);

            let self_attn = {
                let vb_attn = vb_layer.pp("self_attn");
                let q_proj = StreamingLinear::new_with_quantizer(
                    cfg.hidden_size,
                    cfg.num_attention_heads * cfg.head_dim,
                    vb_attn
                        .pp("q_proj")
                        .get(cfg.num_attention_heads * cfg.head_dim, "bias")
                        .is_ok(),
                    vb_attn.pp("q_proj"),
                    quantizer.as_ref(),
                )?;
                let k_proj = StreamingLinear::new_with_quantizer(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * cfg.head_dim,
                    vb_attn
                        .pp("k_proj")
                        .get(cfg.num_key_value_heads * cfg.head_dim, "bias")
                        .is_ok(),
                    vb_attn.pp("k_proj"),
                    quantizer.as_ref(),
                )?;
                let v_proj = StreamingLinear::new_with_quantizer(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * cfg.head_dim,
                    vb_attn
                        .pp("v_proj")
                        .get(cfg.num_key_value_heads * cfg.head_dim, "bias")
                        .is_ok(),
                    vb_attn.pp("v_proj"),
                    quantizer.as_ref(),
                )?;
                let o_proj = StreamingLinear::new_with_quantizer(
                    cfg.num_attention_heads * cfg.head_dim,
                    cfg.hidden_size,
                    vb_attn.pp("o_proj").get(cfg.hidden_size, "bias").is_ok(),
                    vb_attn.pp("o_proj"),
                    quantizer.as_ref(),
                )?;

                let q_norm = if let Ok(w) = vb_attn.pp("q_norm").get(cfg.head_dim, "weight") {
                    Some(StreamingRmsNorm::new(w, cfg.rms_norm_eps))
                } else {
                    None
                };
                let k_norm = if let Ok(w) = vb_attn.pp("k_norm").get(cfg.head_dim, "weight") {
                    Some(StreamingRmsNorm::new(w, cfg.rms_norm_eps))
                } else {
                    None
                };

                let num_heads = cfg.num_attention_heads;
                let num_kv_heads = cfg.num_key_value_heads;
                let num_kv_groups = num_heads / num_kv_heads;
                let head_dim = cfg.head_dim;
                let kv_cache = ConcatKvCache::new(2);
                let kv_cache_dtype = std::env::var("BLOOM_KV_CACHE_DTYPE")
                    .map(|v| match v.to_lowercase().as_str() {
                        "int8" => KvCacheDType::Int8,
                        "int4" => KvCacheDType::Int4,
                        _ => KvCacheDType::F16,
                    })
                    .unwrap_or(KvCacheDType::F16);
                let k_scale_cache = ConcatKvCache::new(2);
                let v_scale_cache = ConcatKvCache::new(2);

                StreamingQwenAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    num_kv_groups,
                    head_dim,
                    rotary_emb: rotary.clone(),
                    kv_cache,
                    kv_cache_dtype,
                    k_scale_cache,
                    v_scale_cache,
                }
            };

            let mlp = {
                let vb_mlp = vb_layer.pp("mlp");
                let gate_proj = StreamingLinear::new_with_quantizer(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    false,
                    vb_mlp.pp("gate_proj"),
                    quantizer.as_ref(),
                )?;
                let up_proj = StreamingLinear::new_with_quantizer(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    false,
                    vb_mlp.pp("up_proj"),
                    quantizer.as_ref(),
                )?;
                let down_proj = StreamingLinear::new_with_quantizer(
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    false,
                    vb_mlp.pp("down_proj"),
                    quantizer.as_ref(),
                )?;
                StreamingQwenMLP {
                    gate_proj,
                    up_proj,
                    down_proj,
                    act_fn: cfg.hidden_act,
                }
            };

            let ln1 = StreamingRmsNorm::new(
                vb_layer
                    .pp("input_layernorm")
                    .get(cfg.hidden_size, "weight")?,
                cfg.rms_norm_eps,
            );
            let ln2 = StreamingRmsNorm::new(
                vb_layer
                    .pp("post_attention_layernorm")
                    .get(cfg.hidden_size, "weight")?,
                cfg.rms_norm_eps,
            );

            layers.push(StreamingDecoderLayer {
                self_attn,
                mlp,
                ln1,
                ln2,
            });
        }

        let norm = StreamingRmsNorm::new(
            vb.pp("model.norm").get(cfg.hidden_size, "weight")?,
            cfg.rms_norm_eps,
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: target_device.clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }
}

#[derive(Debug, Clone)]
pub struct QwenStreamingModelForCausalLM {
    base: StreamingQwenModel,
    lm_head: StreamingLinear,
    pub offloaded_layers: Option<usize>,
}

impl QwenStreamingModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder, target_device: &Device) -> Result<Self> {
        Self::new_with_quantizer(cfg, vb, target_device, None)
    }

    pub fn new_with_quantizer(
        cfg: &Config,
        vb: VarBuilder,
        target_device: &Device,
        quantizer: Option<Arc<super::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        Self::new_with_offload(cfg, vb, target_device, quantizer, None)
    }

    pub fn new_with_offload(
        cfg: &Config,
        vb: VarBuilder,
        target_device: &Device,
        quantizer: Option<Arc<super::metal_quant::MetalQuantizer>>,
        offloaded_layers: Option<usize>,
    ) -> Result<Self> {
        let base = StreamingQwenModel::new_with_quantizer(
            cfg,
            vb.clone(),
            target_device,
            quantizer.clone(),
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            let w = base.embed_tokens.clone();
            StreamingLinear::new(w, None)
        } else {
            let lm_vb = vb.pp("lm_head");
            StreamingLinear::new_with_quantizer(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                lm_vb,
                quantizer.as_ref(),
            )?
        };
        Ok(Self {
            base,
            lm_head,
            offloaded_layers,
        })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let input_device = input.to_device(&self.base.device)?;
        let (b, l) = input_device.dims2()?;
        let flat_input = input_device.flatten_all()?;
        let mut h = self.base.embed_tokens.index_select(&flat_input, 0)?;
        h = h.reshape((b, l, self.base.embed_tokens.dim(1)?))?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.base.causal_mask(input.dims2()?.0, l, offset)?)
        };

        let placements = crate::core::memory::LayerPlacementStrategy::new(
            self.base.layers.len(),
            self.offloaded_layers,
        );

        for (idx, layer) in self.base.layers.iter_mut().enumerate() {
            let target_device_owned = placements.device_for_layer(idx, &self.base.device);
            let target_device = &target_device_owned;

            let h_layer = h.to_device(target_device)?;
            let causal_layer = if let Some(m) = &causal {
                Some(m.to_device(target_device)?)
            } else {
                None
            };

            let out = layer.forward(&h_layer, causal_layer.as_ref(), offset, target_device)?;
            h = out;
        }

        h = h.to_device(&self.base.device)?;

        let h_normed = self.base.norm.forward(&h, &self.base.device)?;
        let h_last = h_normed.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&h_last, &self.base.device)
    }

    pub fn clear_kv_cache(&mut self) {
        for l in &mut self.base.layers {
            l.clear_kv_cache();
        }
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        for l in &mut self.base.layers {
            l.self_attn.truncate_kv_cache(len)?;
        }
        Ok(())
    }

    pub fn extract_kv(
        &mut self,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
        _kv_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.extract_kv_from_layer(layer_idx, start_pos, seq_len)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    /// Number of transformer layers in the model.
    pub fn num_layers(&self) -> usize {
        self.base.layers.len()
    }

    /// Extract KV cache for `[start_pos, start_pos + seq_len)` from a given
    /// layer's `ConcatKvCache`, returning flat f32 vectors of length
    /// `seq_len * kv_dim` each.
    ///
    /// This is the per-layer read path shared by [`QwenKvHook`] and
    /// [`crate::executor::candle::PerRequestKvHook`]. Exposed publicly so
    /// that handle-keyed hooks in other modules can read a streaming
    /// model's KV without duplicating the `ConcatKvCache` slicing logic.
    pub fn extract_kv_from_layer(
        &self,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        if layer_idx >= self.base.layers.len() {
            return Err(BloomError::Engine(format!(
                "layer_idx {} > model layers {}",
                layer_idx,
                self.base.layers.len()
            ))
            .into());
        }
        let kv_cache = &self.base.layers[layer_idx].self_attn.kv_cache;
        let k = kv_cache.k().ok_or_else(|| {
            BloomError::Engine(format!("KV k cache empty for layer {}", layer_idx))
        })?;
        let v = kv_cache.v().ok_or_else(|| {
            BloomError::Engine(format!("KV v cache empty for layer {}", layer_idx))
        })?;

        if seq_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let current_seq_len = k.dim(2)?;
        if start_pos + seq_len > current_seq_len {
            return Err(BloomError::Engine(format!(
                "extract_kv out of range: start_pos={} seq_len={} but cache holds {} tokens",
                start_pos, seq_len, current_seq_len
            ))
            .into());
        }
        let k_slice = k.narrow(2, start_pos, seq_len)?;
        let v_slice = v.narrow(2, start_pos, seq_len)?;
        let k_f32 = k_slice.to_dtype(DType::F32)?.flatten_all()?;
        let v_f32 = v_slice.to_dtype(DType::F32)?.flatten_all()?;
        Ok((k_f32.to_vec1::<f32>()?, v_f32.to_vec1::<f32>()?))
    }

    /// Inject KV for `[start_pos, start_pos + seq_len)` into a given layer's
    /// `ConcatKvCache`.
    ///
    /// Handles two layouts:
    /// 1. **Full restore** (`start_pos == 0` AND cache empty): calls `append`.
    /// 2. **In-place overwrite** (`start_pos + seq_len <= current_seq_len`):
    ///    uses `slice_set` on the underlying K/V tensors.
    ///
    /// Any other layout (notably `start_pos >= current_seq_len`, the decode
    /// hot path where the position doesn't exist yet) is a **no-op**: the
    /// model's own forward pass will compute and append the KV for that
    /// position. See [`QwenKvHook`] docstring for the rationale.
    pub fn inject_kv_to_layer(
        &mut self,
        layer_idx: usize,
        start_pos: usize,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<()> {
        if layer_idx >= self.base.layers.len() {
            return Err(BloomError::Engine(format!(
                "layer_idx {} > model layers {}",
                layer_idx,
                self.base.layers.len()
            ))
            .into());
        }
        if seq_len == 0 {
            return Ok(());
        }
        let kv_dim = num_kv_heads * head_dim;
        let expected = seq_len * kv_dim;
        if keys.len() != expected || values.len() != expected {
            return Err(BloomError::Engine(format!(
                "inject_kv len mismatch: keys={} values={} expected={} (seq_len={} kv_dim={})",
                keys.len(),
                values.len(),
                expected,
                seq_len,
                kv_dim
            ))
            .into());
        }

        let device = self.base.device.clone();
        let dtype = self.base.dtype;
        let shape = (1usize, num_kv_heads, seq_len, head_dim);
        let k_tensor = Tensor::from_slice(keys, shape, &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v_tensor = Tensor::from_slice(values, shape, &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        let kv_cache = &mut self.base.layers[layer_idx].self_attn.kv_cache;
        let current_seq_len = kv_cache.current_seq_len();
        if start_pos == 0 && current_seq_len == 0 {
            kv_cache.append(&k_tensor, &v_tensor)?;
        } else if start_pos + seq_len <= current_seq_len {
            {
                let k_buf_opt = kv_cache.k_mut();
                match k_buf_opt {
                    Some(k_buf) => k_buf.slice_set(&k_tensor, 2, start_pos)?,
                    None => {
                        return Err(BloomError::Engine(
                            "inject_kv case 2 (in-place) but k cache buffer is None".into(),
                        )
                        .into());
                    }
                }
            }
            {
                let v_buf_opt = kv_cache.v_mut();
                match v_buf_opt {
                    Some(v_buf) => v_buf.slice_set(&v_tensor, 2, start_pos)?,
                    None => {
                        return Err(BloomError::Engine(
                            "inject_kv case 2 (in-place) but v cache buffer is None".into(),
                        )
                        .into());
                    }
                }
            }
        }
        // else: no-op (see docstring)
        Ok(())
    }

    /// Wrapper around [`inject_kv_to_layer`] preserving the legacy
    /// `inject_kv` name and candle `Result` return type. The `_kv_dim`
    /// parameter is retained for callers (e.g. `PerRequestKvHook` in
    /// `crate::executor::candle`) that pass the derived
    /// `kv_dim = num_kv_heads * head_dim`.
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
        self.inject_kv_to_layer(
            layer_idx,
            start_pos,
            keys,
            values,
            seq_len,
            num_kv_heads,
            head_dim,
        )
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    pub fn extract_kv_tensor(
        &self,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self.base.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.base.layers.len()
            )));
        }
        let kv_cache = &self.base.layers[layer_idx].self_attn.kv_cache;
        let k = kv_cache.k().ok_or_else(|| {
            candle_core::Error::Msg(format!("KV k cache empty for layer {}", layer_idx))
        })?;
        let v = kv_cache.v().ok_or_else(|| {
            candle_core::Error::Msg(format!("KV v cache empty for layer {}", layer_idx))
        })?;

        if seq_len == 0 {
            return Ok((k.zeros_like()?, v.zeros_like()?));
        }
        let current_seq_len = k.dim(2)?;
        if start_pos + seq_len > current_seq_len {
            return Err(candle_core::Error::Msg(format!(
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
        if layer_idx >= self.base.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx,
                self.base.layers.len()
            )));
        }
        let kv_cache = &mut self.base.layers[layer_idx].self_attn.kv_cache;
        let current_seq_len = kv_cache.current_seq_len();
        if start_pos == 0 && current_seq_len == 0 {
            kv_cache.append(keys, values)?;
        } else if start_pos + keys.dim(2)? <= current_seq_len {
            {
                let k_buf_opt = kv_cache.k_mut();
                if let Some(k_buf) = k_buf_opt {
                    k_buf.slice_set(keys, 2, start_pos)?;
                }
            }
            {
                let v_buf_opt = kv_cache.v_mut();
                if let Some(v_buf) = v_buf_opt {
                    v_buf.slice_set(values, 2, start_pos)?;
                }
            }
        }
        Ok(())
    }
}

/// `KvHook` implementation for `QwenStreamingModelForCausalLM`.
///
/// Reads per-layer KV state from the model's internal `ConcatKvCache`
/// (managed by `StreamingQwenAttention`). The hook shares the model state
/// with the batch executor's forward closure via `Arc<Mutex<...>>`, so
/// `extract_kv` sees the KV that the most recent `forward()` produced.
///
/// `inject_kv` writes KV back into the model's `ConcatKvCache`. It handles
/// two layouts:
/// 1. **Full restore** (`start_pos == 0` AND cache empty): calls `append`,
///    repopulating the cache from scratch. This is the preemption-restore
///    path — the scheduler has reset the model's cache and is replaying the
///    full KV history from the paged cache.
/// 2. **In-place overwrite** (`start_pos + seq_len <= current_seq_len`):
///    uses `slice_set` on the underlying K/V tensors. Used to refresh a
///    slot whose KV was overwritten or never written.
///
/// Any other layout (notably `start_pos >= current_seq_len`, the decode
/// hot path where the position doesn't exist yet) is a **no-op**: the
/// model's own forward pass will compute and append the KV for that
/// position. Trying to "extend" the cache via `inject_kv` would conflict
/// with `ConcatKvCache::append` (which the forward calls) and produce a
/// double-append. The decode-path caller (`restore_request_kv`) invokes
/// `inject_kv` unconditionally before forward, so we must silently ignore
/// positions the model has not yet reached.
///
/// `handle` is ignored because this hook is bound to a single shared model
/// instance. Production deployments that allocate one model per request
/// must wrap each model with its own `QwenKvHook`, or use a handle-keyed
/// variant (TODO: `PerRequestQwenKvHook`).
pub struct QwenKvHook {
    model: Arc<Mutex<QwenStreamingModelForCausalLM>>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
}

impl QwenKvHook {
    /// Build a hook from a shared streaming model.
    ///
    /// `num_layers`, `num_kv_heads`, and `head_dim` are taken from the model
    /// config; `kv_dim = num_kv_heads * head_dim` is derived and asserted
    /// against the paged cache's config at extract time.
    pub fn new(
        model: Arc<Mutex<QwenStreamingModelForCausalLM>>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            model,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim: num_kv_heads * head_dim,
        }
    }
}

impl crate::scheduler::kv_hook::KvHook for QwenKvHook {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn extract_kv(
        &self,
        _handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let (k_vec, v_vec) = model.extract_kv_from_layer(layer_idx, start_pos, seq_len)?;
        if k_vec.len() != seq_len * self.kv_dim {
            return Err(BloomError::Engine(format!(
                "extracted k len {} != seq_len {} * kv_dim {}",
                k_vec.len(),
                seq_len,
                self.kv_dim
            ))
            .into());
        }
        Ok((k_vec, v_vec))
    }

    fn extract_kv_tensor(
        &self,
        _handle: usize,
        layer_idx: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> anyhow::Result<Option<(Tensor, Tensor)>> {
        let model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let (k, v) = model.extract_kv_tensor(layer_idx, start_pos, seq_len)?;
        Ok(Some((k, v)))
    }

    fn inject_kv(
        &self,
        _handle: usize,
        layer_idx: usize,
        start_pos: usize,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> anyhow::Result<()> {
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        model.inject_kv_to_layer(
            layer_idx,
            start_pos,
            keys,
            values,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )
    }

    fn inject_kv_tensor(
        &self,
        _handle: usize,
        layer_idx: usize,
        start_pos: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> anyhow::Result<()> {
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        model.inject_kv_tensor(layer_idx, start_pos, keys, values)?;
        Ok(())
    }

    fn clear_kv_cache(&self, _handle: usize) -> anyhow::Result<()> {
        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        model.clear_kv_cache();
        Ok(())
    }
}

#[cfg(test)]
mod qwen_kv_hook_tests {
    use super::*;
    use crate::scheduler::kv_hook::KvHook;
    use std::collections::HashMap;

    /// Build a `HashMap`-backed `VarBuilder` whose backend only resolves
    /// the standard `weight` tensors (in F32 zeros) and rejects lookups for
    /// `qweight`/`bias`. This is necessary because `VarBuilder::zeros`
    /// (the `Zeros` backend) returns `Ok` for *every* name/shape, which
    /// would make `StreamingLinear::new_with_quantizer` falsely detect a
    /// quantized model on the `qweight` probe — and the quantized fallback
    /// hardcodes F16 output (`dequantize_awq_cpu`), causing an F32-vs-F16
    /// matmul error during the forward pass.
    ///
    /// Using a `HashMap` backend restores checkpoint-like semantics:
    /// `vb.get(..., "qweight")` returns `Err(CannotFindTensor)`, so the
    /// standard F32 weight branch is taken.
    fn build_zero_f32_varbuilder(cfg: &Config, device: &Device) -> VarBuilder<'static> {
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        let dz = |shape: &[usize]| -> Result<Tensor> {
            Tensor::zeros(shape.to_vec(), DType::F32, device)
        };
        let qkv_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;

        ts.insert(
            "model.embed_tokens.weight".to_string(),
            dz(&[cfg.vocab_size, cfg.hidden_size]).unwrap(),
        );
        ts.insert(
            "model.norm.weight".to_string(),
            dz(&[cfg.hidden_size]).unwrap(),
        );
        if !cfg.tie_word_embeddings {
            ts.insert(
                "lm_head.weight".to_string(),
                dz(&[cfg.vocab_size, cfg.hidden_size]).unwrap(),
            );
        }
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{}.{}", i, "");
            let attn = format!("{}self_attn.", p);
            ts.insert(
                format!("{}q_proj.weight", attn),
                dz(&[qkv_out, cfg.hidden_size]).unwrap(),
            );
            ts.insert(
                format!("{}k_proj.weight", attn),
                dz(&[kv_out, cfg.hidden_size]).unwrap(),
            );
            ts.insert(
                format!("{}v_proj.weight", attn),
                dz(&[kv_out, cfg.hidden_size]).unwrap(),
            );
            ts.insert(
                format!("{}o_proj.weight", attn),
                dz(&[cfg.hidden_size, qkv_out]).unwrap(),
            );
            ts.insert(
                format!("{}q_norm.weight", attn),
                dz(&[cfg.head_dim]).unwrap(),
            );
            ts.insert(
                format!("{}k_norm.weight", attn),
                dz(&[cfg.head_dim]).unwrap(),
            );

            let mlp = format!("{}mlp.", p);
            ts.insert(
                format!("{}gate_proj.weight", mlp),
                dz(&[cfg.intermediate_size, cfg.hidden_size]).unwrap(),
            );
            ts.insert(
                format!("{}up_proj.weight", mlp),
                dz(&[cfg.intermediate_size, cfg.hidden_size]).unwrap(),
            );
            ts.insert(
                format!("{}down_proj.weight", mlp),
                dz(&[cfg.hidden_size, cfg.intermediate_size]).unwrap(),
            );

            ts.insert(
                format!("{}input_layernorm.weight", p),
                dz(&[cfg.hidden_size]).unwrap(),
            );
            ts.insert(
                format!("{}post_attention_layernorm.weight", p),
                dz(&[cfg.hidden_size]).unwrap(),
            );
        }
        VarBuilder::from_tensors(ts, DType::F32, device)
    }

    /// Build a minimal `QwenStreamingModelForCausalLM` directly from zero
    /// weights so the test does not depend on downloading a real model. We
    /// only need the attention path to run far enough to populate the
    /// `ConcatKvCache`, so the LM head / MLP weights can be anything.
    #[test]
    fn test_qwen_kv_hook_extracts_real_kv() {
        // Tiny model: 2 layers, 4 heads, head_dim=8, kv_heads=1 (GQA 4:1),
        // hidden=32, vocab=64. This is small enough to run on CPU in tests.
        let device = Device::Cpu;
        let dtype = DType::F32;
        let cfg = Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 0,
            attention_bias: false,
        };

        // Build a HashMap-backed VarBuilder populated with F32 zero weights.
        // `VarBuilder::zeros` would falsely trigger the quantized-weight
        // branch in `StreamingLinear::new_with_quantizer` because the `Zeros`
        // backend reports every probe as `Ok`. See `build_zero_f32_varbuilder`
        // docstring for details.
        let vb = build_zero_f32_varbuilder(&cfg, &device);
        let _ = dtype;

        let model = QwenStreamingModelForCausalLM::new(&cfg, vb, &device).unwrap();
        let model = Arc::new(Mutex::new(model));

        // Run a prefill of 4 tokens to populate the ConcatKvCache.
        let input_ids = Tensor::new(vec![10u32, 20, 30, 40], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        {
            let mut m = model.lock().unwrap_or_else(|e| e.into_inner());
            m.forward(&input_ids, 0).unwrap();
        }

        let hook = QwenKvHook::new(
            Arc::clone(&model),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let (k, v) = hook.extract_kv(0, 0, 0, 4).unwrap();
        assert_eq!(k.len(), 4 * cfg.num_key_value_heads * cfg.head_dim);
        assert_eq!(v.len(), 4 * cfg.num_key_value_heads * cfg.head_dim);

        // Out-of-range extract must error.
        assert!(hook.extract_kv(0, 0, 0, 5).is_err());
        assert!(hook.extract_kv(0, 0, 3, 2).is_err());
        // Bad layer idx must error.
        assert!(hook.extract_kv(0, 99, 0, 1).is_err());
        // Bad layer idx on inject must also error.
        assert!(hook.inject_kv(0, 99, 0, &k, &v, 4).is_err());
        // Length mismatch must error.
        let bad_short = vec![0f32; 3];
        assert!(hook.inject_kv(0, 0, 0, &bad_short, &bad_short, 1).is_err());
    }

    /// Verify `inject_kv` round-trips through `extract_kv` after a full
    /// cache reset: extract KV from a populated model, reset the cache,
    /// inject the previously-extracted KV back, then re-extract and compare.
    ///
    /// This is the preemption-restore flow: scheduler preempts a request
    /// (model's KV cache is reset), then restores it from the paged cache.
    #[test]
    fn test_qwen_kv_hook_inject_full_restore_round_trips() {
        let device = Device::Cpu;
        let cfg = Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 0,
            attention_bias: false,
        };
        let vb = build_zero_f32_varbuilder(&cfg, &device);
        let model = QwenStreamingModelForCausalLM::new(&cfg, vb, &device).unwrap();
        let model = Arc::new(Mutex::new(model));

        let input_ids = Tensor::new(vec![10u32, 20, 30, 40], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        {
            let mut m = model.lock().unwrap_or_else(|e| e.into_inner());
            m.forward(&input_ids, 0).unwrap();
        }

        let hook = QwenKvHook::new(
            Arc::clone(&model),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );

        // Snapshot KV for both layers before reset.
        let (k0_orig, v0_orig) = hook.extract_kv(0, 0, 0, 4).unwrap();
        let (k1_orig, v1_orig) = hook.extract_kv(0, 1, 0, 4).unwrap();

        // Reset the model's KV cache (simulating preemption).
        {
            let mut m = model.lock().unwrap_or_else(|e| e.into_inner());
            m.clear_kv_cache();
        }
        // After reset, extract must fail because the cache is empty.
        assert!(hook.extract_kv(0, 0, 0, 1).is_err());

        // Restore KV at position 0 with seq_len=4 — Case 1 (full restore).
        hook.inject_kv(0, 0, 0, &k0_orig, &v0_orig, 4).unwrap();
        hook.inject_kv(0, 1, 0, &k1_orig, &v1_orig, 4).unwrap();

        // Re-extract and compare — must match the original snapshot.
        let (k0_restored, v0_restored) = hook.extract_kv(0, 0, 0, 4).unwrap();
        let (k1_restored, v1_restored) = hook.extract_kv(0, 1, 0, 4).unwrap();
        assert_eq!(k0_restored, k0_orig);
        assert_eq!(v0_restored, v0_orig);
        assert_eq!(k1_restored, k1_orig);
        assert_eq!(v1_restored, v1_orig);
    }

    /// Verify `inject_kv` Case 3 (in-place overwrite at a single-token
    /// slot). This is the decode hot-path: the executor reads one token's
    /// KV from the paged cache and writes it back into the model at
    /// `start_pos` (where the model would have written during its own
    /// forward).
    #[test]
    fn test_qwen_kv_hook_inject_in_place_overwrite() {
        let device = Device::Cpu;
        let cfg = Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 0,
            attention_bias: false,
        };
        let vb = build_zero_f32_varbuilder(&cfg, &device);
        let model = QwenStreamingModelForCausalLM::new(&cfg, vb, &device).unwrap();
        let model = Arc::new(Mutex::new(model));

        // Prefill 4 tokens to populate the cache.
        let input_ids = Tensor::new(vec![10u32, 20, 30, 40], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        {
            let mut m = model.lock().unwrap_or_else(|e| e.into_inner());
            m.forward(&input_ids, 0).unwrap();
        }

        let hook = QwenKvHook::new(
            Arc::clone(&model),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );

        // Snapshot token at position 2.
        let (k_orig, v_orig) = hook.extract_kv(0, 0, 2, 1).unwrap();
        // Snapshot token at position 0 for sanity.
        let (k0_orig, _) = hook.extract_kv(0, 0, 0, 1).unwrap();

        // Overwrite the slot at position 2 with a sentinel pattern.
        // kv_dim = num_kv_heads * head_dim = 1 * 8 = 8.
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let sentinel_k = vec![0.123f32; kv_dim];
        let sentinel_v = vec![0.456f32; kv_dim];
        hook.inject_kv(0, 0, 2, &sentinel_k, &sentinel_v, 1)
            .unwrap();

        // Re-extract token at position 2 — must equal the sentinel.
        let (k_after, v_after) = hook.extract_kv(0, 0, 2, 1).unwrap();
        assert_eq!(k_after, sentinel_k);
        assert_eq!(v_after, sentinel_v);

        // Tokens at other positions must be unchanged.
        let (k0_after, _) = hook.extract_kv(0, 0, 0, 1).unwrap();
        assert_eq!(k0_after, k0_orig);
        let _ = (k_orig, v_orig); // captured only for shape assertion above

        // Overwriting with a length mismatch must error.
        let bad = vec![0f32; kv_dim + 1];
        assert!(hook.inject_kv(0, 0, 2, &bad, &bad, 1).is_err());

        // Out-of-range inject (start_pos + seq_len > current_seq_len) is a
        // no-op, NOT an error — the model's own forward will compute and
        // append KV for that position. See `QwenKvHook` docstring.
        let oob_k = vec![0f32; kv_dim];
        let oob_v = vec![0f32; kv_dim];
        hook.inject_kv(0, 0, 99, &oob_k, &oob_v, 1).unwrap();
        // Cache state must be unchanged: still 4 tokens, slot 0 unchanged.
        let (k0_after_oob, _) = hook.extract_kv(0, 0, 0, 1).unwrap();
        assert_eq!(k0_after_oob, k0_orig);
    }

    /// End-to-end integration: wire `QwenKvHook` into `CandleBatchExecutor`
    /// with a real (mini) Qwen model, run a prefill, and verify that the
    /// paged cache's `read_kv` matches what the hook's `extract_kv` returns
    /// from the model's internal `ConcatKvCache`.
    ///
    /// This is the smoke test that the full plumbing (model forward →
    /// `write_request_kv` → `extract_kv` → `PagedAttentionCache::write_kv`)
    /// works with real Candle tensors, not just the `InMemoryKvHook` stub.
    #[test]
    fn test_qwen_kv_hook_integration_with_batch_executor() {
        use crate::executor::batch_executor::CandleBatchExecutor;
        use crate::scheduler::paged_cache::{
            LongContextPolicy, PagedAttentionCache, PagedCacheConfig,
        };
        use crate::scheduler::{EngineExecutor, ExecutionBatch, ExecutionPhase};
        use bloomai_core::GenerationParams;

        let device = Device::Cpu;
        let cfg = Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 0,
            attention_bias: false,
        };
        let vb = build_zero_f32_varbuilder(&cfg, &device);
        let model = QwenStreamingModelForCausalLM::new(&cfg, vb, &device).unwrap();
        let model = Arc::new(Mutex::new(model));

        let hook = Arc::new(QwenKvHook::new(
            Arc::clone(&model),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
        ));

        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let block_size = 4;
        let cache_config = PagedCacheConfig {
            block_size,
            total_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            kv_dim,
            kv_dtype: crate::core::quantization::KvCacheDtype::F32,
            long_context_policy: LongContextPolicy::Full,
        };
        let cache = Arc::new(PagedAttentionCache::new(cache_config));

        // Forward closure: lock the shared model and call its real forward.
        // The executor will call this for each request in the batch.
        // We convert `candle_core::Result` to `anyhow::Result` to match
        // the `EngineExecutor` signature expected by `CandleBatchExecutor`.
        let model_for_forward = Arc::clone(&model);
        let forward_fn = Box::new(
            move |input: &Tensor,
                  start_pos: usize,
                  _handle: Option<usize>|
                  -> anyhow::Result<Tensor> {
                let mut m = model_for_forward.lock().unwrap_or_else(|e| e.into_inner());
                Ok(m.forward(input, start_pos)?)
            },
        );

        let executor = CandleBatchExecutor::new(forward_fn, device, 4, 32)
            .with_cache(Arc::clone(&cache))
            .with_kv_hook(Arc::clone(&hook) as Arc<dyn crate::scheduler::kv_hook::KvHook>);

        // Allocate blocks for a 4-token prompt.
        let prompt_tokens = vec![10u32, 20, 30, 40];
        let alloc = cache.allocate("req-1", &prompt_tokens, 0).unwrap();
        let handle = alloc.handle;

        // Run prefill.
        executor
            .execute(ExecutionBatch {
                phase: ExecutionPhase::Prefill,
                request_ids: vec!["req-1".to_string()],
                tokens: prompt_tokens,
                cu_seqlens: vec![0, 4],
                kv_handles: vec![handle],
                start_positions: vec![0],
                params: vec![GenerationParams {
                    temperature: 0.0,
                    ..Default::default()
                }],
                generated_tokens: vec![vec![]],
            })
            .unwrap();

        // Verify the paged cache was populated for both layers.
        assert_eq!(cache.layer_block_count(0), 1);
        assert_eq!(cache.layer_block_count(1), 1);

        // Round-trip: read_kv from paged cache must match extract_kv from
        // the model's ConcatKvCache (via the hook). This proves the full
        // extract → write_kv path works with real Candle tensors.
        let block0 = alloc.allocated_blocks[0];
        for layer in 0..cfg.num_hidden_layers {
            let (paged_k, paged_v) = cache.read_kv(layer, &[block0]).unwrap();
            assert_eq!(paged_k.len(), block_size * kv_dim);
            assert_eq!(paged_v.len(), block_size * kv_dim);
            let (model_k, model_v) = hook.extract_kv(handle, layer, 0, 4).unwrap();
            assert_eq!(model_k.len(), 4 * kv_dim);
            assert_eq!(model_v.len(), 4 * kv_dim);
            // The paged cache stores `block_size * kv_dim` elements per
            // block; only the first `4 * kv_dim` correspond to the 4 tokens
            // we prefilled. Compare those.
            assert_eq!(
                &paged_k[0..4 * kv_dim],
                model_k.as_slice(),
                "layer {} paged cache K mismatch vs model extract",
                layer
            );
            assert_eq!(
                &paged_v[0..4 * kv_dim],
                model_v.as_slice(),
                "layer {} paged cache V mismatch vs model extract",
                layer
            );
        }

        // Now simulate preemption: reset the model's KV cache.
        {
            let mut m = model.lock().unwrap_or_else(|e| e.into_inner());
            m.clear_kv_cache();
        }
        // After reset, extract must fail (cache empty).
        assert!(hook.extract_kv(handle, 0, 0, 1).is_err());

        // Restore full KV history from the paged cache via inject_kv
        // (full-restore path). This is what the scheduler would do on
        // resume after preemption.
        for layer in 0..cfg.num_hidden_layers {
            let (paged_k, paged_v) = cache.read_kv(layer, &[block0]).unwrap();
            let k_slice = &paged_k[0..4 * kv_dim];
            let v_slice = &paged_v[0..4 * kv_dim];
            hook.inject_kv(handle, layer, 0, k_slice, v_slice, 4)
                .unwrap();
        }
        // Re-extract and verify it matches the post-prefill snapshot.
        for layer in 0..cfg.num_hidden_layers {
            let (restored_k, restored_v) = hook.extract_kv(handle, layer, 0, 4).unwrap();
            let (paged_k, paged_v) = cache.read_kv(layer, &[block0]).unwrap();
            assert_eq!(
                &paged_k[0..4 * kv_dim],
                restored_k.as_slice(),
                "layer {} post-restore K mismatch",
                layer
            );
            assert_eq!(
                &paged_v[0..4 * kv_dim],
                restored_v.as_slice(),
                "layer {} post-restore V mismatch",
                layer
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_helpers() {
        let device = Device::Cpu;
        let data = vec![
            -1.0f32, 0.5, 0.25, 0.75, 1.25, -2.5, 0.0, 0.1, 2.0f32, -1.5, 0.8, -0.3, 0.05, 0.45,
            1.5, -0.9, -0.5f32, 0.9, -1.2, 0.35, -0.75, 0.6, -1.0, 1.1, 0.15f32, -0.25, 0.35,
            -0.45, 0.55, -0.65, 0.75, -0.85, 1.0f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, -1.0f32,
            -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, 0.5f32, -0.5, 0.25, -0.25, 0.125, -0.125,
            0.0625, -0.0625, 0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let original = Tensor::new(data.as_slice(), &device)
            .unwrap()
            .reshape((1, 2, 4, 8))
            .unwrap();

        let (quantized, scale) = quantize_tensor(&original).unwrap();
        assert_eq!(quantized.dtype(), DType::U8);
        assert_eq!(quantized.dims(), &[1, 2, 4, 8]);
        assert_eq!(scale.dims(), &[1, 1, 4, 1]);

        let dequantized = dequantize_tensor(&quantized, &scale).unwrap();
        assert_eq!(dequantized.dtype(), DType::F32);
        assert_eq!(dequantized.dims(), &[1, 2, 4, 8]);

        let orig_vec = original.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dequant_vec = dequantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for idx in 0..orig_vec.len() {
            let diff = (orig_vec[idx] - dequant_vec[idx]).abs();
            assert!(
                diff < 0.05,
                "Mismatch at idx {}: original={}, dequantized={}, diff={}",
                idx,
                orig_vec[idx],
                dequant_vec[idx],
                diff
            );
        }
    }

    #[test]
    fn test_npu_gpu_heterogeneous_offload() {
        let device = Device::Cpu;
        let cfg = Config {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            head_dim: 32,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 0,
            attention_bias: false,
        };

        let vb = build_zero_f32_varbuilder(&cfg, &device);
        let mut model =
            QwenStreamingModelForCausalLM::new_with_offload(&cfg, vb, &device, None, Some(2))
                .unwrap();

        let input = Tensor::new(&[1u32, 2u32], &device)
            .unwrap()
            .reshape((1, 2))
            .unwrap();
        let output = model.forward(&input, 0).unwrap();
        assert_eq!(output.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_quantize_dequantize_int4() {
        let device = Device::Cpu;
        let data = vec![
            -1.0f32, 0.5, 0.25, 0.75, 1.25, -2.5, 0.0, 0.1, 2.0f32, -1.5, 0.8, -0.3, 0.05, 0.45,
            1.5, -0.9, -0.5f32, 0.9, -1.2, 0.35, -0.75, 0.6, -1.0, 1.1, 0.15f32, -0.25, 0.35,
            -0.45, 0.55, -0.65, 0.75, -0.85, 1.0f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, -1.0f32,
            -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, 0.5f32, -0.5, 0.25, -0.25, 0.125, -0.125,
            0.0625, -0.0625, 0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let original = Tensor::new(data.as_slice(), &device)
            .unwrap()
            .reshape((1, 2, 4, 8))
            .unwrap();

        let (quantized, scale) = quantize_tensor_int4(&original).unwrap();
        assert_eq!(quantized.dtype(), DType::U8);
        assert_eq!(quantized.dims(), &[1, 2, 4, 4]); // Packed along last dimension
        assert_eq!(scale.dims(), &[1, 1, 4, 1]);

        let dequantized = dequantize_tensor_int4(&quantized, &scale).unwrap();
        assert_eq!(dequantized.dtype(), DType::F32);
        assert_eq!(dequantized.dims(), &[1, 2, 4, 8]);

        let orig_vec = original.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dequant_vec = dequantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for idx in 0..orig_vec.len() {
            let diff = (orig_vec[idx] - dequant_vec[idx]).abs();
            assert!(
                diff < 0.2,
                "Mismatch at idx {}: original={}, dequantized={}, diff={}",
                idx,
                orig_vec[idx],
                dequant_vec[idx],
                diff
            );
        }
    }
}
