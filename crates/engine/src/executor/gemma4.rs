use std::sync::Arc;

use bloomai_core::BloomError;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_core as candle;
use candle_nn::{Activation, VarBuilder};

#[derive(Debug, Clone)]
pub enum Linear {
    Standard {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Quantized {
        qweight: Tensor,
        scales: Tensor,
        qzeros: Tensor,
        g_idx: Option<Tensor>,
        bias: Option<Tensor>,
        quantizer: Option<Arc<super::metal_quant::MetalQuantizer>>,
        total_elements: usize,
    },
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Linear::Standard { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let w = match self {
            Linear::Standard { weight, .. } => weight.t()?,
            Linear::Quantized {
                qweight,
                scales,
                qzeros,
                g_idx,
                quantizer,
                total_elements,
                ..
            } => {
                let dequant = if let (Some(q), Device::Metal(_)) = (quantizer, qweight.device()) {
                    if let Some(g) = g_idx {
                        q.dequantize_gptq(qweight, scales, qzeros, g, *total_elements)?
                    } else {
                        q.dequantize_awq(qweight, scales, qzeros, *total_elements)?
                    }
                } else if let Some(g) = g_idx {
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
                dequant.t()?
            }
        };

        let bias = match self {
            Linear::Standard { bias, .. } => bias.as_ref(),
            Linear::Quantized { bias, .. } => bias.as_ref(),
        };

        let x_dims = x.dims();
        let is_3d = x_dims.len() == 3;

        let (x_processed, b_sz, seq_len, _hidden_size) = if is_3d {
            (
                x.reshape((x_dims[0] * x_dims[1], x_dims[2]))?,
                x_dims[0],
                x_dims[1],
                x_dims[2],
            )
        } else {
            (x.clone(), 0, 0, 0)
        };

        let mut res = if x_dtype == DType::BF16 && x.device().is_cpu() {
            let w_f32 = w.to_dtype(DType::F32)?;
            let x_f32 = x_processed.to_dtype(DType::F32)?;
            let r = x_f32.matmul(&w_f32)?;
            r.to_dtype(DType::BF16)?
        } else {
            x_processed.matmul(&w)?
        };

        if let Some(b) = bias {
            res = res.broadcast_add(b)?;
        }

        if is_3d {
            res = res.reshape((b_sz, seq_len, w.dim(1)?))?;
        }

        Ok(res)
    }
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    quantizer: Option<&Arc<super::metal_quant::MetalQuantizer>>,
) -> Result<Linear> {
    // If the model directory contains qweight, load as Quantized.
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
            Ok(Linear::Quantized {
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
            Ok(Linear::Standard { weight, bias: b })
        }
    } else {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if bias {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        Ok(Linear::Standard { weight, bias })
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub attention_k_eq_v: bool,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_local_base_freq: f64,
    pub partial_rotary_factor: f64,
    pub vocab_size: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
}

impl Config {
    pub fn from_value(text_config: &serde_json::Value) -> anyhow::Result<Self> {
        let attention_bias = text_config
            .get("attention_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let attention_k_eq_v = text_config
            .get("attention_k_eq_v")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let head_dim = text_config
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BloomError::ModelLoad("missing head_dim".into()))? as usize;
        let global_head_dim = text_config
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;

        let hidden_activation = {
            let act_str = text_config
                .get("hidden_activation")
                .and_then(|v| v.as_str())
                .unwrap_or("gelu_pytorch_tanh");
            match act_str {
                "gelu_pytorch_tanh" => Activation::GeluPytorchTanh,
                "gelu" => Activation::Gelu,
                "silu" => Activation::Silu,
                _ => Activation::GeluPytorchTanh,
            }
        };

        let hidden_size = text_config
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BloomError::ModelLoad("missing hidden_size".into()))?
            as usize;
        let intermediate_size = text_config
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BloomError::ModelLoad("missing intermediate_size".into()))?
            as usize;
        let num_attention_heads = text_config
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BloomError::ModelLoad("missing num_attention_heads".into()))?
            as usize;
        let num_hidden_layers = text_config
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BloomError::ModelLoad("missing num_hidden_layers".into()))?
            as usize;
        let num_key_value_heads = text_config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BloomError::ModelLoad("missing num_key_value_heads".into()))?
            as usize;
        let num_global_key_value_heads = text_config
            .get("num_global_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let rms_norm_eps = text_config
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6);

        let rope_theta = text_config
            .get("rope_parameters")
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(|t| t.as_f64())
            .unwrap_or(1000000.0);

        let rope_local_base_freq = text_config
            .get("rope_parameters")
            .and_then(|rp| rp.get("sliding_attention"))
            .and_then(|sa| sa.get("rope_theta"))
            .and_then(|t| t.as_f64())
            .unwrap_or(10000.0);

        let partial_rotary_factor = text_config
            .get("rope_parameters")
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let vocab_size = text_config
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BloomError::ModelLoad("missing vocab_size".into()))?
            as usize;
        let final_logit_softcapping = text_config
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64());
        let attn_logit_softcapping = text_config
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64());

        let query_pre_attn_scalar = head_dim;
        let sliding_window = text_config
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize;
        let sliding_window_pattern = 6;
        let max_position_embeddings = text_config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(131072) as usize;

        Ok(Self {
            attention_bias,
            attention_k_eq_v,
            head_dim,
            global_head_dim,
            hidden_activation,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            num_key_value_heads,
            num_global_key_value_heads,
            rms_norm_eps,
            rope_theta,
            rope_local_base_freq,
            partial_rotary_factor,
            vocab_size,
            final_logit_softcapping,
            attn_logit_softcapping,
            query_pre_attn_scalar,
            sliding_window,
            sliding_window_pattern,
            max_position_embeddings,
        })
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

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, with_scale: bool, vb: VarBuilder) -> Result<Self> {
        let weight = if with_scale {
            Some(vb.get(dim, "weight")?)
        } else {
            None
        };
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        match &self.weight {
            Some(weight) => x_normed.broadcast_mul(weight),
            None => Ok(x_normed),
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        dim: usize,
        denominator_dim: usize,
        rope_freq: f64,
        max_position_embeddings: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / denominator_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        let rotary_dim = self.cos.dim(1)? * 2;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        if rotary_dim < head_dim {
            let q_to_rotate = q.narrow(D::Minus1, 0, rotary_dim)?;
            let q_pass_through = q.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;
            let k_to_rotate = k.narrow(D::Minus1, 0, rotary_dim)?;
            let k_pass_through = k.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;

            let q_rotated = candle_nn::rotary_emb::rope(&q_to_rotate.contiguous()?, &cos, &sin)?;
            let k_rotated = candle_nn::rotary_emb::rope(&k_to_rotate.contiguous()?, &cos, &sin)?;

            let q_embed = Tensor::cat(&[&q_rotated, &q_pass_through], D::Minus1)?;
            let k_embed = Tensor::cat(&[&k_rotated, &k_pass_through], D::Minus1)?;

            Ok((q_embed, k_embed))
        } else {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        }
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl MLP {
    #[allow(dead_code)]
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_quantizer(cfg, vb, None)
    }

    fn new_with_quantizer(
        cfg: &Config,
        vb: VarBuilder,
        quantizer: Option<&Arc<super::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(
            hidden_sz,
            intermediate_sz,
            false,
            vb.pp("gate_proj"),
            quantizer,
        )?;
        let up_proj = linear(
            hidden_sz,
            intermediate_sz,
            false,
            vb.pp("up_proj"),
            quantizer,
        )?;
        let down_proj = linear(
            intermediate_sz,
            hidden_sz,
            false,
            vb.pp("down_proj"),
            quantizer,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    use_flash_attn: bool,
}

impl Attention {
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        use_flash_attn: bool,
        hidden_sz: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        attn_logit_softcapping: Option<f64>,
        max_position_embeddings: usize,
        sliding_window: Option<usize>,
        attention_bias: bool,
        attention_k_eq_v: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_quantizer(
            rotary_emb,
            use_flash_attn,
            hidden_sz,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            attn_logit_softcapping,
            max_position_embeddings,
            sliding_window,
            attention_bias,
            attention_k_eq_v,
            vb,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_quantizer(
        rotary_emb: Arc<RotaryEmbedding>,
        use_flash_attn: bool,
        hidden_sz: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        attn_logit_softcapping: Option<f64>,
        max_position_embeddings: usize,
        sliding_window: Option<usize>,
        attention_bias: bool,
        attention_k_eq_v: bool,
        vb: VarBuilder,
        quantizer: Option<&Arc<super::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = linear(
            hidden_sz,
            num_heads * head_dim,
            attention_bias,
            vb.pp("q_proj"),
            quantizer,
        )?;
        let k_proj = linear(
            hidden_sz,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("k_proj"),
            quantizer,
        )?;
        let v_proj = if attention_k_eq_v {
            None
        } else {
            Some(linear(
                hidden_sz,
                num_kv_heads * head_dim,
                attention_bias,
                vb.pp("v_proj"),
                quantizer,
            )?)
        };
        let o_proj = linear(
            num_heads * head_dim,
            hidden_sz,
            attention_bias,
            vb.pp("o_proj"),
            quantizer,
        )?;
        let q_norm = RmsNorm::new(head_dim, rms_norm_eps, true, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, rms_norm_eps, true, vb.pp("k_norm"))?;
        let v_norm = RmsNorm::new(head_dim, rms_norm_eps, false, vb.pp("v_norm"))?;
        let kv_cache = if let Some(sliding_window) = sliding_window {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, sliding_window))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                2,
                max_position_embeddings,
            ))
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            attn_logit_softcapping,
            rotary_emb,
            kv_cache,
            use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = match &self.v_proj {
            Some(v_proj) => v_proj.forward(xs)?,
            None => key_states.clone(),
        };

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;
        let value_states = self.v_norm.forward(&value_states)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let key_states = key_states.contiguous()?;
        let value_states = value_states.contiguous()?;
        let (key_states, value_states) = match &mut self.kv_cache {
            KvCache::Normal(cache) => cache.append(&key_states, &value_states)?,
            KvCache::Rotating(cache) => cache.append(&key_states, &value_states)?,
        };

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = if self.use_flash_attn {
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let scale = 1.0f32;
            flash_attn(&q, &k, &v, scale, attention_mask.is_some())?.transpose(1, 2)?
        } else {
            let scale = 1.0f64;
            let is_cpu = query_states.device().is_cpu() && query_states.dtype() == DType::BF16;

            let attn_weights = if is_cpu {
                let q_f32 = query_states.to_dtype(DType::F32)?;
                let k_f32 = key_states.transpose(2, 3)?.to_dtype(DType::F32)?;
                (q_f32.matmul(&k_f32)? * scale)?
            } else {
                (query_states
                    .contiguous()?
                    .matmul(&key_states.transpose(2, 3)?.contiguous()?)?
                    * scale)?
            };

            let attn_weights = match self.attn_logit_softcapping {
                None => attn_weights,
                Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
            };

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => {
                    if is_cpu {
                        attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?
                    } else {
                        attn_weights.broadcast_add(mask)?
                    }
                }
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

            if is_cpu {
                let v_f32 = value_states.to_dtype(DType::F32)?;
                attn_weights.matmul(&v_f32)?.to_dtype(DType::BF16)?
            } else {
                attn_weights
                    .contiguous()?
                    .matmul(&value_states.contiguous()?)?
            }
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    layer_scalar: Tensor,
    sliding_window: Option<usize>,
}

impl DecoderLayer {
    #[allow(dead_code)]
    fn new(
        use_flash_attn: bool,
        cfg: &Config,
        vb: VarBuilder,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        Self::new_with_quantizer(use_flash_attn, cfg, vb, sliding_window, None)
    }

    fn new_with_quantizer(
        use_flash_attn: bool,
        cfg: &Config,
        vb: VarBuilder,
        sliding_window: Option<usize>,
        quantizer: Option<&Arc<super::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        let is_sliding = sliding_window.is_some();
        let head_dim = if is_sliding {
            cfg.head_dim
        } else {
            cfg.global_head_dim
        };
        let rotary_dim = if is_sliding {
            cfg.head_dim
        } else {
            (cfg.global_head_dim as f64 * cfg.partial_rotary_factor) as usize
        };
        let num_kv_heads = if is_sliding {
            cfg.num_key_value_heads
        } else {
            cfg.num_global_key_value_heads
        };
        let rope_freq = if is_sliding {
            cfg.rope_local_base_freq
        } else {
            cfg.rope_theta
        };
        let attention_k_eq_v = !is_sliding && cfg.attention_k_eq_v;

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            rotary_dim,
            head_dim,
            rope_freq,
            cfg.max_position_embeddings,
            vb.device(),
        )?);

        let self_attn = Attention::new_with_quantizer(
            rotary_emb,
            use_flash_attn,
            cfg.hidden_size,
            cfg.num_attention_heads,
            num_kv_heads,
            head_dim,
            cfg.rms_norm_eps,
            cfg.attn_logit_softcapping,
            cfg.max_position_embeddings,
            sliding_window,
            cfg.attention_bias,
            attention_k_eq_v,
            vb.pp("self_attn"),
            quantizer,
        )?;
        let mlp = MLP::new_with_quantizer(cfg, vb.pp("mlp"), quantizer)?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("input_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("post_attention_layernorm"),
        )?;
        let layer_scalar = vb.get(1, "layer_scalar")?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            layer_scalar,
            sliding_window,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let xs = (residual + xs)?;
        xs.broadcast_mul(&self.layer_scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
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
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
}

impl Model {
    pub fn new(use_flash_attn: bool, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_quantizer(use_flash_attn, cfg, vb, None)
    }

    pub fn new_with_quantizer(
        use_flash_attn: bool,
        cfg: &Config,
        vb: VarBuilder,
        quantizer: Option<Arc<super::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model.language_model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let sliding_window = (layer_idx + 1) % cfg.sliding_window_pattern > 0;
            let layer = DecoderLayer::new_with_quantizer(
                use_flash_attn,
                cfg,
                vb_l.pp(layer_idx),
                sliding_window.then_some(cfg.sliding_window),
                quantizer.as_ref(),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, true, vb_m.pp("norm"))?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
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

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;

        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset)?;

        for layer in self.layers.iter_mut() {
            let mask = if layer.sliding_window.is_some() {
                &sliding_attention_mask
            } else {
                &attention_mask
            };
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset)?
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };

        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
