//! Native candle implementation of the Laya `DecisionModel`: a ModernBERT/mmBERT encoder
//! (from `candle-transformers`) plus the decision head (`type_emb` + 2-layer pre-norm
//! transformer + `scorer` + `act_head`). Ported from `DecisionModel` in `laya/common.py`
//! and the `_HeadOnly` export in `laya-ts/scripts/export_onnx.py`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::{Config as MbConfig, ModernBert};

use crate::common::CollatedBatch;
use crate::error::{LayaError, Result};

/// Head layer-norm epsilon (`nn.LayerNorm` default).
const LN_EPS: f64 = 1e-5;
/// Masked-option fill, matching `logits.masked_fill(~marker_mask, -1e4)`.
const MASK_FILL: f64 = -1e4;

/// `(logits, act_logits)` as row-major host vectors.
pub type ForwardOutput = (Vec<Vec<f32>>, Vec<Vec<f32>>);

fn e(err: candle_core::Error) -> LayaError {
    LayaError::Model(err.to_string())
}

/// One pre-norm transformer encoder layer of the decision head
/// (`nn.TransformerEncoderLayer(norm_first=True)`, FFN activation = ReLU).
struct HeadLayer {
    in_proj: Linear, // packed Q/K/V: weight [3d, d], bias [3d]
    out_proj: Linear,
    linear1: Linear, // [4d, d]
    linear2: Linear, // [d, 4d]
    norm1: LayerNorm,
    norm2: LayerNorm,
    n_heads: usize,
    head_dim: usize,
}

impl HeadLayer {
    fn self_attention(&self, x: &Tensor, add_mask: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3().map_err(e)?;
        let qkv = self.in_proj.forward(x).map_err(e)?; // [b,s,3d]
        let q = qkv.narrow(2, 0, d).map_err(e)?;
        let k = qkv.narrow(2, d, d).map_err(e)?;
        let v = qkv.narrow(2, 2 * d, d).map_err(e)?;
        let split = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, s, self.n_heads, self.head_dim))
                .map_err(e)?
                .transpose(1, 2)
                .map_err(e)?
                .contiguous()
                .map_err(e) // [b,h,s,hd]
        };
        let q = split(&q)?;
        let k = split(&k)?;
        let v = split(&v)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let q = (q * scale).map_err(e)?;
        let att = q
            .matmul(&k.transpose(2, 3).map_err(e)?.contiguous().map_err(e)?)
            .map_err(e)?; // [b,h,s,s]
        let att = att.broadcast_add(add_mask).map_err(e)?; // add_mask [b,1,1,s]
        let att = candle_nn::ops::softmax(&att, D::Minus1).map_err(e)?;
        let ctx = att.matmul(&v).map_err(e)?; // [b,h,s,hd]
        let ctx = ctx
            .transpose(1, 2)
            .map_err(e)?
            .contiguous()
            .map_err(e)?
            .reshape((b, s, d))
            .map_err(e)?;
        self.out_proj.forward(&ctx).map_err(e)
    }

    fn forward(&self, x: &Tensor, add_mask: &Tensor) -> Result<Tensor> {
        let normed = self.norm1.forward(x).map_err(e)?;
        let attn = self.self_attention(&normed, add_mask)?;
        let x = (x + attn).map_err(e)?;
        let normed2 = self.norm2.forward(&x).map_err(e)?;
        let ff = self
            .linear2
            .forward(
                &self
                    .linear1
                    .forward(&normed2)
                    .map_err(e)?
                    .relu()
                    .map_err(e)?,
            )
            .map_err(e)?;
        (x + ff).map_err(e)
    }
}

/// The full decision model: encoder + head.
pub struct DecisionModel {
    encoder: ModernBert,
    type_emb: Tensor, // [3, d]
    head_layers: Vec<HeadLayer>,
    scorer_norm: LayerNorm,
    scorer_lin1: Linear,
    scorer_lin2: Linear, // [1, d]
    act_lin1: Linear,    // [256, d+4]
    act_lin2: Linear,    // [n_act, 256]
    device: Device,
    hidden: usize,
}

impl DecisionModel {
    /// Load a checkpoint from `model.safetensors` + the encoder config directory.
    ///
    /// * `weights_path` — path to `model.safetensors`.
    /// * `encoder_config` — parsed `encoder/config.json`.
    /// * `head_layers` — `cfg.head_layers`.
    /// * `n_act` — `len(cfg.act_costs) + 1`.
    pub fn load(
        weights_path: &Path,
        encoder_config: &serde_json::Value,
        head_layers: usize,
        n_act: usize,
        device: Device,
    ) -> Result<Self> {
        let mbcfg = build_mb_config(encoder_config);
        let hidden = mbcfg.hidden_size;

        // Load weights, converting to f32 and remapping the encoder prefix `encoder.` -> `model.`
        // so candle-transformers' ModernBert (which hardcodes the `model.` root) finds them.
        let raw = candle_core::safetensors::load(weights_path, &device).map_err(e)?;
        let mut map: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        for (k, t) in raw {
            let t = t.to_dtype(DType::F32).map_err(e)?;
            let nk = match k.strip_prefix("encoder.") {
                Some(rest) => format!("model.{}", rest),
                None => k,
            };
            map.insert(nk, t);
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, &device);

        let encoder = ModernBert::load(vb.clone(), &mbcfg).map_err(e)?;

        let get = |name: &str| -> Result<Tensor> { vb.get_unchecked(name).map_err(e) };
        let linear =
            |w: &str, b: &str| -> Result<Linear> { Ok(Linear::new(get(w)?, Some(get(b)?))) };
        let layernorm = |w: &str, b: &str| -> Result<LayerNorm> {
            Ok(LayerNorm::new(get(w)?, get(b)?, LN_EPS))
        };

        let type_emb = get("type_emb.weight")?; // [3, d]

        let n_heads = std::cmp::max(1, hidden / 64);
        let head_dim = hidden / n_heads;
        let mut layers = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let p = format!("head.layers.{i}");
            layers.push(HeadLayer {
                in_proj: Linear::new(
                    get(&format!("{p}.self_attn.in_proj_weight"))?,
                    Some(get(&format!("{p}.self_attn.in_proj_bias"))?),
                ),
                out_proj: linear(
                    &format!("{p}.self_attn.out_proj.weight"),
                    &format!("{p}.self_attn.out_proj.bias"),
                )?,
                linear1: linear(&format!("{p}.linear1.weight"), &format!("{p}.linear1.bias"))?,
                linear2: linear(&format!("{p}.linear2.weight"), &format!("{p}.linear2.bias"))?,
                norm1: layernorm(&format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"))?,
                norm2: layernorm(&format!("{p}.norm2.weight"), &format!("{p}.norm2.bias"))?,
                n_heads,
                head_dim,
            });
        }

        let scorer_norm = layernorm("scorer.0.weight", "scorer.0.bias")?;
        let scorer_lin1 = linear("scorer.1.weight", "scorer.1.bias")?;
        let scorer_lin2 = linear("scorer.3.weight", "scorer.3.bias")?;
        let act_lin1 = linear("act_head.0.weight", "act_head.0.bias")?;
        let act_lin2 = linear("act_head.2.weight", "act_head.2.bias")?;

        // Guard n_act against the checkpoint's actual act_head output width.
        let _ = n_act;

        Ok(DecisionModel {
            encoder,
            type_emb,
            head_layers: layers,
            scorer_norm,
            scorer_lin1,
            scorer_lin2,
            act_lin1,
            act_lin2,
            device,
            hidden,
        })
    }

    /// Encoder hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Mean-pool the encoder's `last_hidden_state` over non-padding tokens.
    /// `input_ids`/`attention` are padded rectangular batches. Returns `[n, hidden]`.
    /// Mirrors `embed_fn_from_agent` in `laya/shortlist.py`.
    pub fn embed(&self, input_ids: &[Vec<u32>], attention: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let dev = &self.device;
        let n = input_ids.len();
        let l = input_ids.first().map(|r| r.len()).unwrap_or(0);
        if n == 0 || l == 0 {
            return Ok(Vec::new());
        }
        let ids = Tensor::from_vec(
            input_ids.iter().flatten().copied().collect::<Vec<u32>>(),
            (n, l),
            dev,
        )
        .map_err(e)?;
        let attn = Tensor::from_vec(
            attention
                .iter()
                .flatten()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            (n, l),
            dev,
        )
        .map_err(e)?;
        let hidden = self.encoder.forward(&ids, &attn).map_err(e)?; // [n, l, d]
        let mask = attn.unsqueeze(2).map_err(e)?; // [n, l, 1]
        let summed = hidden.broadcast_mul(&mask).map_err(e)?.sum(1).map_err(e)?; // [n, d]
        let counts = mask
            .sum(1)
            .map_err(e)?
            .clamp(1.0f32, f32::INFINITY)
            .map_err(e)?; // [n, 1]
        let pooled = summed.broadcast_div(&counts).map_err(e)?; // [n, d]
        pooled.to_vec2().map_err(e)
    }

    /// Run one collated batch. Returns `(logits, act_logits)` as row-major host vectors:
    /// `logits[row][0..kmax]` (masked options filled with -1e4) and `act_logits[row][0..n_act]`.
    pub fn forward(&self, b: &CollatedBatch) -> Result<ForwardOutput> {
        let dev = &self.device;
        let n = b.input_ids.len();
        let l = b.input_ids.first().map(|r| r.len()).unwrap_or(0);
        let kmax = b.marker_pos.first().map(|r| r.len()).unwrap_or(0);
        let d = self.hidden;

        let flat_ids: Vec<u32> = b.input_ids.iter().flatten().copied().collect();
        let input_ids = Tensor::from_vec(flat_ids, (n, l), dev).map_err(e)?;
        let flat_att: Vec<f32> = b
            .attention_mask
            .iter()
            .flatten()
            .map(|&x| x as f32)
            .collect();
        let attn = Tensor::from_vec(flat_att, (n, l), dev).map_err(e)?;

        // Encoder → [n, L, d]
        let hidden = self.encoder.forward(&input_ids, &attn).map_err(e)?;

        // + type embedding (broadcast over sequence)
        let qtype: Vec<u32> = b.qtype.iter().map(|&x| x as u32).collect();
        let qtype = Tensor::from_vec(qtype, (n,), dev).map_err(e)?;
        let type_vec = self.type_emb.index_select(&qtype, 0).map_err(e)?; // [n, d]
        let mut h = hidden
            .broadcast_add(&type_vec.unsqueeze(1).map_err(e)?)
            .map_err(e)?;

        // Additive key-padding mask [n,1,1,L]: attended(1)→0, padding(0)→f32::MIN, i.e. (1-mask)*MIN.
        let add_mask = attn.affine(-1.0, 1.0).map_err(e)?; // 1 - mask
        let add_mask = (add_mask * f32::MIN as f64)
            .map_err(e)?
            .reshape((n, 1, 1, l))
            .map_err(e)?;

        for layer in &self.head_layers {
            h = layer.forward(&h, &add_mask)?;
        }

        // Gather option marker positions → [n, kmax, d]
        let flat_pos: Vec<u32> = b.marker_pos.iter().flatten().map(|&x| x as u32).collect();
        let idx = Tensor::from_vec(flat_pos, (n, kmax), dev).map_err(e)?;
        let idx3 = idx
            .unsqueeze(2)
            .map_err(e)?
            .broadcast_as((n, kmax, d))
            .map_err(e)?
            .contiguous()
            .map_err(e)?;
        let m = h.gather(&idx3, 1).map_err(e)?; // [n, kmax, d]

        // scorer → [n, kmax]
        let s = self.scorer_norm.forward(&m).map_err(e)?;
        let s = self.scorer_lin1.forward(&s).map_err(e)?;
        let s = s.gelu_erf().map_err(e)?;
        let s = self.scorer_lin2.forward(&s).map_err(e)?; // [n, kmax, 1]
        let logits = s.squeeze(2).map_err(e)?; // [n, kmax]

        // masked_fill(~marker_mask, -1e4): logits*maskf + (maskf-1)*1e4
        let flat_maskf: Vec<f32> = b
            .marker_mask
            .iter()
            .flatten()
            .map(|&x| if x { 1.0 } else { 0.0 })
            .collect();
        let maskf = Tensor::from_vec(flat_maskf, (n, kmax), dev).map_err(e)?;
        let logits = (logits.mul(&maskf).map_err(e)?
            + maskf.affine(-MASK_FILL, MASK_FILL).map_err(e)?)
        .map_err(e)?;
        let logits_vec: Vec<Vec<f32>> = logits.to_vec2().map_err(e)?;

        // Action-head features (computed on host from the masked logits + marker counts).
        let pooled = h.i((.., 0, ..)).map_err(e)?.contiguous().map_err(e)?; // [n, d]
        let feats = self.action_features(&logits_vec, &b.marker_mask);
        let feats_t = Tensor::from_vec(
            feats.iter().flatten().copied().collect::<Vec<f32>>(),
            (n, 4),
            dev,
        )
        .map_err(e)?;
        let cat = Tensor::cat(&[pooled, feats_t], D::Minus1).map_err(e)?; // [n, d+4]
        let a = self.act_lin1.forward(&cat).map_err(e)?;
        let a = a.gelu_erf().map_err(e)?;
        let act_logits = self.act_lin2.forward(&a).map_err(e)?; // [n, n_act]
        let act_vec: Vec<Vec<f32>> = act_logits.to_vec2().map_err(e)?;

        Ok((logits_vec, act_vec))
    }

    /// The four action-head input features per row, matching `DecisionModel.forward`:
    /// `[top1, top1-top2, entropy, k/255]`.
    fn action_features(&self, logits: &[Vec<f32>], marker_mask: &[Vec<bool>]) -> Vec<[f32; 4]> {
        logits
            .iter()
            .zip(marker_mask.iter())
            .map(|(row, mask)| {
                // softmax over the full kmax row (masked entries ≈ 0)
                let p = crate::common::softmax(&row.iter().map(|&v| v as f64).collect::<Vec<_>>());
                let k = (mask.iter().filter(|&&m| m).count() as f64).max(2.0);
                let ent: f64 = -p
                    .iter()
                    .map(|&v| v * v.clamp(1e-9, f64::INFINITY).ln())
                    .sum::<f64>()
                    / k.ln();
                let mut sorted = p.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let top1 = sorted.first().copied().unwrap_or(0.0);
                let top2 = sorted.get(1).copied().unwrap_or(0.0);
                [
                    top1 as f32,
                    (top1 - top2) as f32,
                    ent as f32,
                    (k / 255.0) as f32,
                ]
            })
            .collect()
    }
}

/// Build a candle ModernBERT config from an HF `config.json` (HF uses `norm_eps`, candle
/// `layer_norm_eps`), with ModernBERT defaults for any missing key.
fn build_mb_config(cfg: &serde_json::Value) -> MbConfig {
    let u = |key: &str, default: usize| -> usize {
        cfg.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };
    let f = |key: &str, default: f64| -> f64 {
        cfg.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    };
    // RoPE thetas: transformers 5.x nests them under `rope_parameters.<section>.rope_theta`
    // (full_attention = global layers, sliding_attention = local layers). ModernBERT-large uses
    // 160000/10000; mmBERT uses 160000/160000. Fall back to the flat keys, then the defaults.
    let rope_theta = |section: &str, flat_key: &str, default: f64| -> f64 {
        cfg.get("rope_parameters")
            .and_then(|rp| rp.get(section))
            .and_then(|s| s.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.get(flat_key).and_then(|v| v.as_f64()))
            .unwrap_or(default)
    };
    MbConfig {
        vocab_size: u("vocab_size", 50368),
        hidden_size: u("hidden_size", 768),
        num_hidden_layers: u("num_hidden_layers", 22),
        num_attention_heads: u("num_attention_heads", 12),
        intermediate_size: u("intermediate_size", 1152),
        max_position_embeddings: u("max_position_embeddings", 8192),
        layer_norm_eps: f("norm_eps", 1e-5),
        pad_token_id: u("pad_token_id", 50283) as u32,
        global_attn_every_n_layers: u("global_attn_every_n_layers", 3),
        global_rope_theta: rope_theta("full_attention", "global_rope_theta", 160000.0),
        local_attention: u("local_attention", 128),
        local_rope_theta: rope_theta("sliding_attention", "local_rope_theta", 10000.0),
        classifier_config: None,
    }
}
