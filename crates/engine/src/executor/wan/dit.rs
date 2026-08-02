//! Wan2.1 DiT (Diffusion Transformer) implementation in Candle.
//!
//! Architecture: 3D patch embedding + transformer blocks (self-attn + cross-attn + FFN)
//! with AdaLN modulation, 3D RoPE, and flow-matching conditioning.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, Module};

thread_local! {
    pub static QTENSOR_MAP: std::cell::RefCell<Option<std::collections::HashMap<String, std::sync::Arc<candle_core::quantized::QTensor>>>> = std::cell::RefCell::new(None);
}

#[derive(Clone)]
pub enum Linear {
    F32 {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Quantized {
        qweight: std::sync::Arc<candle_core::quantized::QMatMul>,
        bias: Option<Tensor>,
    },
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Linear")
    }
}

impl Linear {
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        match self {
            Self::F32 { weight, bias } => Ok(Self::F32 {
                weight: weight.to_device(device)?,
                bias: bias.as_ref().map(|t| t.to_device(device)).transpose()?,
            }),
            Self::Quantized { qweight, bias } => Ok(Self::Quantized {
                qweight: qweight.clone(),
                bias: bias.as_ref().map(|t| t.to_device(device)).transpose()?,
            }),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        let is_3d = x_shape.len() == 3;
        let x_flat = if is_3d {
            x.reshape((x_shape[0] * x_shape[1], x_shape[2]))?
        } else {
            x.clone()
        };

        let mut out = match self {
            Self::F32 { weight, .. } => x_flat.matmul(&weight.t()?)?,
            Self::Quantized { qweight, .. } => qweight.forward(&x_flat)?,
        };

        if is_3d {
            out = out.reshape((x_shape[0], x_shape[1], ()))?;
        }

        let bias = match self {
            Self::F32 { bias, .. } => bias,
            Self::Quantized { bias, .. } => bias,
        };
        if let Some(ref b) = bias {
            out = out.broadcast_add(b)?;
        }
        Ok(out)
    }
}

pub fn linear(in_dim: usize, out_dim: usize, vb: nn::VarBuilder) -> Result<Linear> {
    let prefix = vb.prefix();
    let weight_name = if prefix.is_empty() {
        "weight".to_string()
    } else {
        format!("{}.weight", prefix)
    };

    let qmatmul = QTENSOR_MAP
        .with(|map| {
            if let Some(m) = map.borrow().as_ref() {
                if let Some(qt) = m.get(&weight_name) {
                    return Some(
                        candle_core::quantized::QMatMul::from_arc(qt.clone())
                            .map(std::sync::Arc::new),
                    );
                }
            }
            None
        })
        .transpose()?;

    let bias = vb.get(out_dim, "bias").ok();

    if let Some(qm) = qmatmul {
        Ok(Linear::Quantized { qweight: qm, bias })
    } else {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        Ok(Linear::F32 { weight, bias })
    }
}

/// Wan2.1 DiT configuration.
#[derive(Debug, Clone)]
pub struct WanConfig {
    pub model_type: String,
    pub patch_size: (usize, usize, usize),
    pub text_len: usize,
    pub in_dim: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub text_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub qk_norm: bool,
    pub cross_attn_norm: bool,
    pub eps: f64,
}

impl WanConfig {
    /// Default config for Wan2.1-T2V-1.3B.
    pub fn t2v_1_3b() -> Self {
        Self {
            model_type: "t2v".to_string(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 16,
            dim: 1536,
            ffn_dim: 8960,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 12,
            num_layers: 30,
            qk_norm: true,
            cross_attn_norm: true,
            eps: 1e-6,
        }
    }
}

/// 1D sinusoidal positional embedding.
/// dim must be even. position is a 1D tensor of positions.
pub fn sinusoidal_embedding_1d(dim: usize, position: &Tensor) -> Result<Tensor> {
    let half = dim / 2;
    let device = position.device();
    let position = position.to_dtype(DType::F32)?;
    let half_range: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let half_range = Tensor::from_vec(half_range, (half,), device)?;
    let freqs = half_range
        .affine(-10000.0_f64.ln() / half as f64, 0.0)?
        .exp()?;
    // freqs = 10000^(-i/half) for i in 0..half
    let sinusoid = position.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let cos = sinusoid.cos()?;
    let sin = sinusoid.sin()?;
    Tensor::cat(&[&cos, &sin], 1)?.to_dtype(DType::F32)
}

/// Compute 3D RoPE frequencies.
/// Returns complex-like pairs stored as [max_seq_len, dim/2, 2] (real, imag).
pub fn rope_params(max_seq_len: usize, dim: usize, theta: f64, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let half_range: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq_len,), device)?;
    let half_range = Tensor::from_vec(half_range, (half,), device)?;
    let inv_freq = half_range.affine(-theta.ln() / half as f64, 0.0)?.exp()?;
    let freqs = positions
        .unsqueeze(1)?
        .broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    // Return cos/sin pairs: [max_seq_len, half, 2]
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Tensor::stack(&[&cos, &sin], 2)?.to_dtype(DType::F32)
}

/// Apply 3D RoPE to query/key tensors.
///
/// x: [batch, seq_len, num_heads, head_dim]
/// grid_sizes: [batch, 3] containing (F, H, W) per sample
/// freqs_t/h/w: precomputed 3D RoPE frequencies for each axis
///   freqs_t: [max_t, head_dim_t/2, 2] (temporal)
///   freqs_h: [max_h, head_dim_h/2, 2] (spatial height)
///   freqs_w: [max_w, head_dim_w/2, 2] (spatial width)
///
/// Returns: rotated x with same shape [batch, seq_len, num_heads, head_dim].
///
/// Note: This is a pure Candle implementation. For production GPU use,
/// this should be replaced with a TileLang kernel (similar to
/// qwen3_vl::apply_tilelang_mrope).
pub fn rope_apply(
    x: &Tensor,
    grid_sizes: &[(usize, usize, usize)],
    freqs_t: &Tensor,
    freqs_h: &Tensor,
    freqs_w: &Tensor,
) -> Result<Tensor> {
    let batch = x.dim(0)?;
    let seq_len = x.dim(1)?;
    let num_heads = x.dim(2)?;
    let head_dim = x.dim(3)?;

    // Split head_dim into 3 parts: temporal, spatial_h, spatial_w
    let c_t = head_dim - 2 * (head_dim / 3);
    let c_h = head_dim / 3;
    let c_w = head_dim / 3;

    let mut batch_outputs = Vec::new();

    for i in 0..batch {
        let (f, h, w) = grid_sizes[i.min(grid_sizes.len() - 1)];
        let spatial_seq = f * h * w;

        // Get this sample's tokens: [spatial_seq, num_heads, head_dim]
        let x_i = x.narrow(0, i, 1)?.squeeze(0)?; // [seq_len, num_heads, head_dim]
        let x_i = x_i.narrow(0, 0, spatial_seq)?; // [spatial_seq, num_heads, head_dim]

        // Reshape to pairs for complex multiplication: [spatial_seq, num_heads, head_dim/2, 2]
        let x_pairs = x_i.reshape((spatial_seq, num_heads, head_dim / 2, 2))?;
        let x_real = x_pairs.narrow(3, 0, 1)?.squeeze(3)?;
        let x_imag = x_pairs.narrow(3, 1, 1)?.squeeze(3)?;

        // Build 3D frequency grid for this sample
        let freq_t_slice = freqs_t.narrow(0, 0, f)?; // [f, c_t/2, 2]
        let freq_h_slice = freqs_h.narrow(0, 0, h)?; // [h, c_h/2, 2]
        let freq_w_slice = freqs_w.narrow(0, 0, w)?; // [w, c_w/2, 2]

        // Expand to grid: [f*h*w, axis_dim/2, 2] for each axis
        // For temporal: broadcast h and w dims
        let ft_cos = freq_t_slice
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .reshape((f, 1, 1, c_t / 2))?
            .broadcast_as((f, h, w, c_t / 2))?
            .reshape((spatial_seq, c_t / 2))?;
        let ft_sin = freq_t_slice
            .narrow(2, 1, 1)?
            .squeeze(2)?
            .reshape((f, 1, 1, c_t / 2))?
            .broadcast_as((f, h, w, c_t / 2))?
            .reshape((spatial_seq, c_t / 2))?;

        let fh_cos = freq_h_slice
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .reshape((1, h, 1, c_h / 2))?
            .broadcast_as((f, h, w, c_h / 2))?
            .reshape((spatial_seq, c_h / 2))?;
        let fh_sin = freq_h_slice
            .narrow(2, 1, 1)?
            .squeeze(2)?
            .reshape((1, h, 1, c_h / 2))?
            .broadcast_as((f, h, w, c_h / 2))?
            .reshape((spatial_seq, c_h / 2))?;

        let fw_cos = freq_w_slice
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .reshape((1, 1, w, c_w / 2))?
            .broadcast_as((f, h, w, c_w / 2))?
            .reshape((spatial_seq, c_w / 2))?;
        let fw_sin = freq_w_slice
            .narrow(2, 1, 1)?
            .squeeze(2)?
            .reshape((1, 1, w, c_w / 2))?
            .broadcast_as((f, h, w, c_w / 2))?
            .reshape((spatial_seq, c_w / 2))?;

        // Concatenate: [spatial_seq, head_dim/2]
        let cos_full = Tensor::cat(&[&ft_cos, &fh_cos, &fw_cos], 1)?;
        let sin_full = Tensor::cat(&[&ft_sin, &fh_sin, &fw_sin], 1)?;

        // Broadcast over num_heads: [spatial_seq, 1, head_dim/2]
        let cos_full = cos_full.unsqueeze(1)?;
        let sin_full = sin_full.unsqueeze(1)?;

        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let new_real = (x_real.broadcast_mul(&cos_full)? - x_imag.broadcast_mul(&sin_full)?)?;
        let new_imag = (x_real.broadcast_mul(&sin_full)? + x_imag.broadcast_mul(&cos_full)?)?;

        // Interleave back: [spatial_seq, num_heads, head_dim/2, 2]
        let rotated = Tensor::stack(&[&new_real, &new_imag], 3)?.reshape((
            spatial_seq,
            num_heads,
            head_dim,
        ))?;

        // Append padding tokens (if any) unchanged
        if spatial_seq < seq_len {
            let padding =
                x.narrow(0, i, 1)?
                    .squeeze(0)?
                    .narrow(0, spatial_seq, seq_len - spatial_seq)?;
            batch_outputs.push(Tensor::cat(&[&rotated, &padding], 0)?);
        } else {
            batch_outputs.push(rotated);
        }
    }

    // Stack batch dimension: [batch, seq_len, num_heads, head_dim]
    Tensor::stack(&batch_outputs, 0)
}

/// RMS normalization used in Wan model.
#[derive(Debug, Clone)]
pub struct WanRMSNorm {
    weight: Tensor,
    eps: f64,
}

impl WanRMSNorm {
    pub fn new(dim: usize, eps: f64, vb: nn::VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_device(device)?,
            eps: self.eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f = x.to_dtype(DType::F32)?;
        let norm = (x_f
            .sqr()?
            .mean_keepdim(candle_core::D::Minus1)?
            .affine(1.0, self.eps))?
        .recip()?;
        x_f.broadcast_mul(&norm)?
            .to_dtype(x.dtype())?
            .broadcast_mul(&self.weight)
    }
}

/// Layer normalization (optionally without affine).
#[derive(Debug, Clone)]
pub struct WanLayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl WanLayerNorm {
    pub fn new(dim: usize, eps: f64, affine: bool, vb: nn::VarBuilder) -> Result<Self> {
        let (weight, bias) = if affine {
            let weight = vb.get(dim, "weight")?;
            let bias = vb.get(dim, "bias")?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        Ok(Self { weight, bias, eps })
    }

    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: self
                .weight
                .as_ref()
                .map(|t| t.to_device(device))
                .transpose()?,
            bias: self
                .bias
                .as_ref()
                .map(|t| t.to_device(device))
                .transpose()?,
            eps: self.eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
        let var = x_f32
            .broadcast_sub(&mean)?
            .sqr()?
            .mean_keepdim(candle_core::D::Minus1)?;
        let inv_std = (var + self.eps)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_sub(&mean)?.broadcast_mul(&inv_std)?;
        let mut out = normed.to_dtype(x_dtype)?;
        if let (Some(w), Some(b)) = (&self.weight, &self.bias) {
            out = out.broadcast_mul(w)?.broadcast_add(b)?;
        }
        Ok(out)
    }
}

/// Self-attention with 3D RoPE.
#[derive(Debug, Clone)]
pub struct WanSelfAttention {
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    qk_norm: bool,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: Option<WanRMSNorm>,
    norm_k: Option<WanRMSNorm>,
    eps: f64,
}

impl WanSelfAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qk_norm: bool,
        eps: f64,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q = linear(dim, dim, vb.pp("q"))?;
        let k = linear(dim, dim, vb.pp("k"))?;
        let v = linear(dim, dim, vb.pp("v"))?;
        let o = linear(dim, dim, vb.pp("o"))?;

        let norm_q = if qk_norm {
            Some(WanRMSNorm::new(dim, eps, vb.pp("norm_q"))?)
        } else {
            None
        };
        let norm_k = if qk_norm {
            Some(WanRMSNorm::new(dim, eps, vb.pp("norm_k"))?)
        } else {
            None
        };

        Ok(Self {
            dim,
            num_heads,
            head_dim,
            qk_norm,
            q,
            k,
            v,
            o,
            norm_q,
            norm_k,
            eps,
        })
    }

    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            dim: self.dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            qk_norm: self.qk_norm,
            q: self.q.to_device(device)?,
            k: self.k.to_device(device)?,
            v: self.v.to_device(device)?,
            o: self.o.to_device(device)?,
            norm_q: self
                .norm_q
                .as_ref()
                .map(|n| n.to_device(device))
                .transpose()?,
            norm_k: self
                .norm_k
                .as_ref()
                .map(|n| n.to_device(device))
                .transpose()?,
            eps: self.eps,
        })
    }

    /// Forward pass.
    /// x: [batch, seq_len, dim]
    /// grid_sizes: [(F, H, W)] per sample
    /// freqs_t/h/w: precomputed 3D RoPE frequencies
    pub fn forward(
        &self,
        x: &Tensor,
        grid_sizes: &[(usize, usize, usize)],
        freqs_t: &Tensor,
        freqs_h: &Tensor,
        freqs_w: &Tensor,
    ) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;
        let n = self.num_heads;
        let dh = self.head_dim;

        let q = self.q.forward(x)?.reshape((b, s, n, dh))?;
        let k = self.k.forward(x)?.reshape((b, s, n, dh))?;
        let v = self.v.forward(x)?.reshape((b, s, n, dh))?;

        // QK norm
        let q = if let Some(ref nq) = self.norm_q {
            nq.forward(&q.flatten(2, 3)?)?.reshape((b, s, n, dh))?
        } else {
            q
        };
        let k = if let Some(ref nk) = self.norm_k {
            nk.forward(&k.flatten(2, 3)?)?.reshape((b, s, n, dh))?
        } else {
            k
        };

        // Apply 3D RoPE (input: [batch, seq_len, heads, head_dim])
        let q = rope_apply(&q, grid_sizes, freqs_t, freqs_h, freqs_w)?;
        let k = rope_apply(&k, grid_sizes, freqs_t, freqs_h, freqs_w)?;

        // Transpose to [batch, heads, seq_len, head_dim]
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        // Scaled dot-product attention
        let scale = (dh as f64).sqrt().recip();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&(q.matmul(&k_t)?.affine(scale, 0.0)?))?;
        let out = attn.matmul(&v)?;

        // [batch, heads, seq_len, head_dim] -> [batch, seq_len, dim]
        let out = out.permute((0, 2, 1, 3))?.reshape((b, s, self.dim))?;
        self.o.forward(&out)
    }
}

/// Cross-attention for text-to-video (query from video, key/value from text).
#[derive(Debug, Clone)]
pub struct WanT2VCrossAttention {
    dim: usize,
    num_heads: usize,
    head_dim: usize,
    qk_norm: bool,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: Option<WanRMSNorm>,
    norm_k: Option<WanRMSNorm>,
}

impl WanT2VCrossAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qk_norm: bool,
        eps: f64,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q = linear(dim, dim, vb.pp("q"))?;
        let k = linear(dim, dim, vb.pp("k"))?;
        let v = linear(dim, dim, vb.pp("v"))?;
        let o = linear(dim, dim, vb.pp("o"))?;

        let norm_q = if qk_norm {
            Some(WanRMSNorm::new(dim, eps, vb.pp("norm_q"))?)
        } else {
            None
        };
        let norm_k = if qk_norm {
            Some(WanRMSNorm::new(dim, eps, vb.pp("norm_k"))?)
        } else {
            None
        };

        Ok(Self {
            dim,
            num_heads,
            head_dim,
            qk_norm,
            q,
            k,
            v,
            o,
            norm_q,
            norm_k,
        })
    }

    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            dim: self.dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            qk_norm: self.qk_norm,
            q: self.q.to_device(device)?,
            k: self.k.to_device(device)?,
            v: self.v.to_device(device)?,
            o: self.o.to_device(device)?,
            norm_q: self
                .norm_q
                .as_ref()
                .map(|n| n.to_device(device))
                .transpose()?,
            norm_k: self
                .norm_k
                .as_ref()
                .map(|n| n.to_device(device))
                .transpose()?,
        })
    }

    /// x: [batch, seq_len_1, dim] (video tokens)
    /// context: [batch, seq_len_2, dim] (text embeddings)
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, s1, _d) = x.dims3()?;
        let s2 = context.dim(1)?;
        let n = self.num_heads;
        let dh = self.head_dim;

        let q = self.q.forward(x)?.reshape((b, s1, n, dh))?;
        let k = self.k.forward(context)?.reshape((b, s2, n, dh))?;
        let v = self.v.forward(context)?.reshape((b, s2, n, dh))?;

        let q = if let Some(ref nq) = self.norm_q {
            nq.forward(&q.flatten(2, 3)?)?.reshape((b, s1, n, dh))?
        } else {
            q
        };
        let k = if let Some(ref nk) = self.norm_k {
            nk.forward(&k.flatten(2, 3)?)?.reshape((b, s2, n, dh))?
        } else {
            k
        };

        // [batch, heads, seq, head_dim]
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        let scale = (dh as f64).sqrt().recip();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&(q.matmul(&k_t)?.affine(scale, 0.0)?))?;
        let out = attn.matmul(&v)?;

        let out = out.permute((0, 2, 1, 3))?.reshape((b, s1, self.dim))?;
        self.o.forward(&out)
    }
}

/// Transformer block with self-attn + cross-attn + FFN + AdaLN modulation.
#[derive(Debug, Clone)]
pub struct WanAttentionBlock {
    dim: usize,
    ffn_dim: usize,
    norm1: WanLayerNorm,
    self_attn: WanSelfAttention,
    norm3: Option<WanLayerNorm>,
    cross_attn: WanT2VCrossAttention,
    norm2: WanLayerNorm,
    ffn_0: Linear,
    ffn_2: Linear,
    modulation: Tensor,
}

impl WanAttentionBlock {
    pub fn new(cfg: &WanConfig, vb: nn::VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let ffn_dim = cfg.ffn_dim;
        let num_heads = cfg.num_heads;
        let eps = cfg.eps;

        let norm1 = WanLayerNorm::new(dim, eps, false, vb.pp("norm1"))?;
        let self_attn =
            WanSelfAttention::new(dim, num_heads, cfg.qk_norm, eps, vb.pp("self_attn"))?;

        let norm3 = if cfg.cross_attn_norm {
            Some(WanLayerNorm::new(dim, eps, true, vb.pp("norm3"))?)
        } else {
            None
        };

        let cross_attn =
            WanT2VCrossAttention::new(dim, num_heads, cfg.qk_norm, eps, vb.pp("cross_attn"))?;

        let norm2 = WanLayerNorm::new(dim, eps, false, vb.pp("norm2"))?;
        let ffn_0 = linear(dim, ffn_dim, vb.pp("ffn.0"))?;
        let ffn_2 = linear(ffn_dim, dim, vb.pp("ffn.2"))?;

        // modulation: [1, 6, dim]
        let modulation = vb.get((1, 6, dim), "modulation")?;

        Ok(Self {
            dim,
            ffn_dim,
            norm1,
            self_attn,
            norm3,
            cross_attn,
            norm2,
            ffn_0,
            ffn_2,
            modulation,
        })
    }

    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            dim: self.dim,
            ffn_dim: self.ffn_dim,
            norm1: self.norm1.to_device(device)?,
            self_attn: self.self_attn.to_device(device)?,
            norm3: self
                .norm3
                .as_ref()
                .map(|n| n.to_device(device))
                .transpose()?,
            cross_attn: self.cross_attn.to_device(device)?,
            norm2: self.norm2.to_device(device)?,
            ffn_0: self.ffn_0.to_device(device)?,
            ffn_2: self.ffn_2.to_device(device)?,
            modulation: self.modulation.to_device(device)?,
        })
    }

    /// x: [batch, seq_len, dim]
    /// e: [batch, 6, dim] (modulation from timestep)
    /// context: [batch, text_len, dim] (text embeddings)
    pub fn forward(
        &self,
        x: &Tensor,
        e: &Tensor,
        grid_sizes: &[(usize, usize, usize)],
        freqs_t: &Tensor,
        freqs_h: &Tensor,
        freqs_w: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        // e = (modulation + e) split into 6 chunks
        let e_mod = self.modulation.broadcast_add(e)?;
        let e_chunks: Vec<_> = (0..6)
            .map(|index| e_mod.narrow(1, index, 1))
            .collect::<candle_core::Result<_>>()?;

        // Self-attention with AdaLN modulation
        let norm1_out = self.norm1.forward(x)?;
        let norm1_mod = norm1_out
            .broadcast_mul(&(Tensor::ones_like(&e_chunks[1])?.broadcast_add(&e_chunks[1])?))?
            .broadcast_add(&e_chunks[0])?;
        let sa_out = self
            .self_attn
            .forward(&norm1_mod, grid_sizes, freqs_t, freqs_h, freqs_w)?;
        let x = (x + sa_out.broadcast_mul(&e_chunks[2])?)?;

        // Cross-attention
        let x_ca = if let Some(ref n3) = self.norm3 {
            n3.forward(&x)?
        } else {
            x.clone()
        };
        let x = (x + self.cross_attn.forward(&x_ca, context)?)?;

        // FFN with AdaLN modulation
        let norm2_out = self.norm2.forward(&x)?;
        let norm2_mod = norm2_out
            .broadcast_mul(&(Tensor::ones_like(&e_chunks[4])?.broadcast_add(&e_chunks[4])?))?
            .broadcast_add(&e_chunks[3])?;
        let ffn_out = self.ffn_0.forward(&norm2_mod)?.gelu()?;
        let ffn_out = self.ffn_2.forward(&ffn_out)?;
        let x = (x + ffn_out.broadcast_mul(&e_chunks[5])?)?;

        Ok(x)
    }
}

/// Output head with AdaLN modulation.
#[derive(Debug, Clone)]
pub struct Head {
    norm: WanLayerNorm,
    head: Linear,
    modulation: Tensor,
}

impl Head {
    pub fn new(
        dim: usize,
        out_dim: usize,
        patch_size: (usize, usize, usize),
        eps: f64,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let norm = WanLayerNorm::new(dim, eps, false, vb.pp("norm"))?;
        let linear_out = out_dim * patch_size.0 * patch_size.1 * patch_size.2;
        let head = linear(dim, linear_out, vb.pp("head"))?;
        let modulation = vb.get((1, 2, dim), "modulation")?;

        Ok(Self {
            norm,
            head,
            modulation,
        })
    }

    /// x: [batch, seq_len, dim]
    /// e: [batch, dim] (time embedding, not projected)
    pub fn forward(&self, x: &Tensor, e: &Tensor) -> Result<Tensor> {
        // e: [batch, dim] -> [batch, 1, dim] for broadcasting with modulation [1, 2, dim]
        let e_3d = e.unsqueeze(1)?;
        let e_mod = self.modulation.broadcast_add(&e_3d)?; // [batch, 2, dim]
                                                           // Keep dim 1 as size 1 for broadcasting with x [batch, seq_len, dim]
        let e0 = e_mod.narrow(1, 0, 1)?; // [batch, 1, dim]
        let e1 = e_mod.narrow(1, 1, 1)?; // [batch, 1, dim]

        let x = self.norm.forward(x)?;
        let x = x
            .broadcast_mul(&(Tensor::ones_like(&e1)?.broadcast_add(&e1)?))?
            .broadcast_add(&e0)?;
        self.head.forward(&x)
    }
}

/// Full Wan DiT model.
pub struct WanModel {
    pub config: WanConfig,
    /// Patch embedding loaded from Wan's 3D conv weights.
    patch_embedding: PatchEmbedding,
    text_embedding_0: Linear,
    text_embedding_2: Linear,
    time_embedding_0: Linear,
    time_embedding_2: Linear,
    time_projection_1: Linear,
    blocks: Vec<WanAttentionBlock>,
    head: Head,
    freqs_t: Tensor,
    freqs_h: Tensor,
    freqs_w: Tensor,
}

struct PatchEmbedding {
    conv: nn::Conv2d,
}

impl PatchEmbedding {
    fn new(cfg: &WanConfig, vb: nn::VarBuilder) -> Result<Self> {
        let (pt, ph, pw) = cfg.patch_size;
        if pt != 1 || ph != pw {
            candle_core::bail!(
                "unsupported Wan patch size {:?}; only temporal patch=1 and square spatial patches are implemented",
                cfg.patch_size
            );
        }

        let weight = vb.get((cfg.dim, cfg.in_dim, pt, ph, pw), "weight")?;
        let weight = weight.squeeze(2)?;
        let bias = vb.get((cfg.dim,), "bias").ok();
        let conv = nn::Conv2d::new(
            weight,
            bias,
            nn::Conv2dConfig {
                stride: ph,
                padding: 0,
                ..Default::default()
            },
        );
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

impl WanModel {
    pub fn new(
        cfg: &WanConfig,
        vb: nn::VarBuilder,
        vb_cpu: Option<nn::VarBuilder>,
        device: &Device,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let num_heads = cfg.num_heads;
        let head_dim = dim / num_heads;

        let patch_embedding = PatchEmbedding::new(cfg, vb.pp("patch_embedding"))?;

        // Text embedding: Linear(text_dim, dim) -> GELU -> Linear(dim, dim)
        let text_embedding_0 = linear(cfg.text_dim, dim, vb.pp("text_embedding.0"))?;
        let text_embedding_2 = linear(dim, dim, vb.pp("text_embedding.2"))?;

        // Time embedding: Linear(freq_dim, dim) -> SiLU -> Linear(dim, dim)
        let time_embedding_0 = linear(cfg.freq_dim, dim, vb.pp("time_embedding.0"))?;
        let time_embedding_2 = linear(dim, dim, vb.pp("time_embedding.2"))?;

        // Time projection: SiLU -> Linear(dim, dim*6)
        let time_projection_1 = linear(dim, dim * 6, vb.pp("time_projection.1"))?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let block_vb = vb_cpu.as_ref().unwrap_or(&vb);
        for i in 0..cfg.num_layers {
            let block = WanAttentionBlock::new(cfg, block_vb.pp(&format!("blocks.{i}")))?;
            blocks.push(block);
        }

        // Head
        let head = Head::new(dim, cfg.out_dim, cfg.patch_size, cfg.eps, vb.pp("head"))?;

        // Pre-compute 3D RoPE frequencies
        let d = head_dim;
        let d_t = d - 2 * (d / 3);
        let d_h = d / 3;
        let d_w = d / 3;
        let freqs_t = rope_params(1024, d_t, 10000.0, device)?;
        let freqs_h = rope_params(1024, d_h, 10000.0, device)?;
        let freqs_w = rope_params(1024, d_w, 10000.0, device)?;

        Ok(Self {
            config: cfg.clone(),
            patch_embedding,
            text_embedding_0,
            text_embedding_2,
            time_embedding_0,
            time_embedding_2,
            time_projection_1,
            blocks,
            head,
            freqs_t,
            freqs_h,
            freqs_w,
        })
    }

    /// Forward pass through the DiT.
    ///
    /// x: latent tensor [batch, in_dim, F, H, W] (or list of such)
    /// t: timestep tensor [batch]
    /// context: text embeddings [batch, text_len, text_dim]
    ///
    /// Returns: denoised latent [batch, out_dim, F, H, W]
    pub fn forward(&self, x: &Tensor, t: &Tensor, context: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let (b, _c, f, h, w) = x.dims5()?;

        let grid_sizes: Vec<(usize, usize, usize)> = (0..b)
            .map(|_| {
                (
                    f / cfg.patch_size.0,
                    h / cfg.patch_size.1,
                    w / cfg.patch_size.2,
                )
            })
            .collect();

        // Patch embedding using Conv2d (temporal_patch=1, spatial_patch=2)
        // x: [b, in_dim, F, H, W] -> reshape to [b*F, in_dim, H, W] -> Conv2d -> [b*F, dim, H', W'] -> reshape back
        let x_4d = x.reshape((b * f, _c, h, w))?;
        let x = self.patch_embedding.forward(&x_4d)?; // [b*F, dim, H', W']
        let (_bf, _d, hp, wp) = x.dims4()?;
        let x = x.reshape((b, f, cfg.dim, hp, wp))?;
        // Permute to [b, dim, F, H', W']
        let x = x.permute((0, 2, 1, 3, 4))?;
        let (_b, _d, fp, hp, wp) = x.dims5()?;
        let x = x.flatten_from(2)?.t()?; // [b, seq_len, dim]

        // Time embedding
        let t_emb = sinusoidal_embedding_1d(cfg.freq_dim, t)?;
        let e = self.time_embedding_0.forward(&t_emb)?.silu()?;
        let e = self.time_embedding_2.forward(&e)?;

        // Time projection for AdaLN modulation: [b, dim*6] -> [b, 6, dim]
        let e0 = self
            .time_projection_1
            .forward(&e.silu()?)?
            .reshape((b, 6, cfg.dim))?;

        // Text embedding
        let ctx = self.text_embedding_0.forward(context)?.gelu()?;
        let ctx = self.text_embedding_2.forward(&ctx)?;

        // Transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(
                &x,
                &e0,
                &grid_sizes,
                &self.freqs_t,
                &self.freqs_h,
                &self.freqs_w,
                &ctx,
            )?;
        }

        // Head
        let x = self.head.forward(&x, &e)?;

        // Unpatchify: [b, seq_len, out_dim * prod(patch_size)] -> [b, out_dim, F, H, W]
        self.unpatchify(&x, &grid_sizes, (fp, hp, wp))
    }

    /// Reconstruct video latent from patch tokens.
    fn unpatchify(
        &self,
        x: &Tensor,
        grid_sizes: &[(usize, usize, usize)],
        _grid_shape: (usize, usize, usize),
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let c = cfg.out_dim;
        let (pt, ph, pw) = cfg.patch_size;
        let b = x.dim(0)?;

        let mut outputs = Vec::new();
        for i in 0..b {
            let (f, h, w) = grid_sizes[i];
            let seq = f * h * w;
            let x_i = x.narrow(0, i, 1)?.squeeze(0)?.narrow(0, 0, seq)?;
            // [seq, c*pt*ph*pw] -> [f*h*w, c, pt*ph*pw]
            let x_i = x_i.reshape((f * h * w, c, pt * ph * pw))?;
            // Process per-channel to achieve unpatchify (Candle max 5 dims)
            // Target: [c, f*pt, h*ph, w*pw] from [f*h*w, c*pt*ph*pw]
            let mut ch_outputs = Vec::new();
            for ci in 0..c {
                let ch = x_i.narrow(1, ci, 1)?.squeeze(1)?; // [fhw, pt*ph*pw]
                let ch = ch.reshape((f, h, w, pt, ph * pw))?; // 5D
                let ch = ch.permute((0, 3, 1, 4, 2))?; // [f, pt, h, ph*pw, w]
                let ch = ch.reshape((f * pt, h, ph * pw, w))?; // 4D merge f*pt
                let ch = ch.reshape((f * pt, h, ph, pw, w))?; // 5D split ph*pw
                let ch = ch.permute((0, 1, 2, 4, 3))?; // [fpt, h, ph, w, pw]
                let ch = ch.reshape((f * pt, h * ph, w * pw))?; // 3D final
                ch_outputs.push(ch.unsqueeze(0)?); // [1, fpt, hph, wpw]
            }
            let x_i = Tensor::cat(&ch_outputs, 0)?; // [c, f*pt, h*ph, w*pw]
            outputs.push(x_i);
        }
        Tensor::stack(&outputs, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_embedding_1d() {
        let device = Device::Cpu;
        let pos = Tensor::new(&[0.0f32, 1.0, 2.0, 3.0], &device).unwrap();
        let emb = sinusoidal_embedding_1d(64, &pos).unwrap();
        assert_eq!(emb.dims(), &[4, 64]);
    }

    #[test]
    fn test_rope_params_shape() {
        let freqs = rope_params(128, 32, 10000.0, &Device::Cpu).unwrap();
        assert_eq!(freqs.dims(), &[128, 16, 2]);
    }

    #[test]
    fn test_wan_config_1_3b() {
        let cfg = WanConfig::t2v_1_3b();
        assert_eq!(cfg.dim, 1536);
        assert_eq!(cfg.num_heads, 12);
        assert_eq!(cfg.num_layers, 30);
        assert_eq!(cfg.patch_size, (1, 2, 2));
    }
}
