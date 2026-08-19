#![cfg(feature = "candle-engine")]
#![allow(dead_code)]
// Multiaxis position construction intentionally indexes coordinate planes.
#![allow(clippy::needless_range_loop)]

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};

use bloomai_core::{
    DeviceCapability, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat,
};

use crate::{
    engine::{Engine, EngineCapability, SupportLevel, default_engine_supports},
    io::{ModelInput, ModelOutput},
    model::{LoadedModel, ModelMetadata},
};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct RopeScaling {
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
    pub rope_type: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub in_channels: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Qwen3VLConfig {
    pub model_type: String,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub image_token_id: u32,
}

fn find_safetensors_files(model_path: &Path) -> Vec<PathBuf> {
    let single = model_path.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }
    if let Ok(entries) = std::fs::read_dir(model_path) {
        let mut shard_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("model-") && name_str.ends_with(".safetensors")
            })
            .map(|e| e.path())
            .collect();
        shard_files.sort();
        return shard_files;
    }
    vec![]
}

fn linspace(start: f32, end: f32, steps: usize, dev: &Device) -> Result<Tensor> {
    if steps <= 1 {
        Ok(Tensor::new(&[start], dev)?)
    } else {
        let step = (end - start) / (steps - 1) as f32;
        let v: Vec<_> = (0..steps).map(|i| start + i as f32 * step).collect();
        Ok(Tensor::new(v, dev)?)
    }
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let mut h_bar = ((height as f32 / factor as f32).round() as usize) * factor;
    let mut w_bar = ((width as f32 / factor as f32).round() as usize) * factor;

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f32 / max_pixels as f32).sqrt();
        h_bar = factor.max(((height as f32 / beta / factor as f32).floor() as usize) * factor);
        w_bar = factor.max(((width as f32 / beta / factor as f32).floor() as usize) * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (height * width) as f32).sqrt();
        h_bar = ((height as f32 * beta / factor as f32).ceil() as usize) * factor;
        w_bar = ((width as f32 * beta / factor as f32).ceil() as usize) * factor;
    }

    (h_bar, w_bar)
}

fn preprocess_image(
    image_bytes: &[u8],
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(Vec<f32>, usize, usize, usize)> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| anyhow!("Failed to load image from memory: {}", e))?;
    let img = img.to_rgb8();
    let width = img.width() as usize;
    let height = img.height() as usize;

    let (resized_height, resized_width) = smart_resize(height, width, 32, min_pixels, max_pixels);

    let resized_img = image::imageops::resize(
        &img,
        resized_width as u32,
        resized_height as u32,
        image::imageops::FilterType::Triangle,
    );

    let grid_t = 1;
    let grid_h = resized_height / 16;
    let grid_w = resized_width / 16;

    let total_patches = grid_t * grid_h * grid_w;
    let patch_features = 1536;

    let mut flatten_patches = vec![0.0f32; total_patches * patch_features];

    let merge_size = 2;
    let patch_size = 16;
    let temp_patch_size = 2;

    for bh in 0..(grid_h / merge_size) {
        for bw in 0..(grid_w / merge_size) {
            for ih in 0..merge_size {
                for iw in 0..merge_size {
                    let start_y = (bh * merge_size + ih) * patch_size;
                    let start_x = (bw * merge_size + iw) * patch_size;

                    let patch_idx = bh * (grid_w / merge_size) * merge_size * merge_size
                        + bw * merge_size * merge_size
                        + ih * merge_size
                        + iw;

                    for c in 0..3 {
                        for t in 0..temp_patch_size {
                            for ph in 0..patch_size {
                                for pw in 0..patch_size {
                                    let pixel_y = start_y + ph;
                                    let pixel_x = start_x + pw;

                                    let rgb_pixel =
                                        resized_img.get_pixel(pixel_x as u32, pixel_y as u32);
                                    let val = rgb_pixel[c] as f32;
                                    let normalized_val = val / 127.5 - 1.0;

                                    let feature_idx = c
                                        * (temp_patch_size * patch_size * patch_size)
                                        + t * (patch_size * patch_size)
                                        + ph * patch_size
                                        + pw;

                                    flatten_patches[patch_idx * patch_features + feature_idx] =
                                        normalized_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((flatten_patches, grid_t, grid_h, grid_w))
}

fn get_rope_index(
    token_ids: &[u32],
    image_grid_thw: Option<(usize, usize, usize)>,
) -> (Vec<Vec<usize>>, usize) {
    let seq_len = token_ids.len();
    let Some((t, h, w)) = image_grid_thw else {
        let coords: Vec<usize> = (0..seq_len).collect();
        return (
            vec![coords.clone(), coords.clone(), coords.clone()],
            seq_len,
        );
    };
    let llm_grid_t = t;
    let llm_grid_h = h / 2;
    let llm_grid_w = w / 2;
    let image_token_id = 151655;

    let mut t_ids = Vec::with_capacity(seq_len);
    let mut h_ids = Vec::with_capacity(seq_len);
    let mut w_ids = Vec::with_capacity(seq_len);

    let mut st = 0;
    let mut st_idx = 0;

    if let Some(ed) = token_ids.iter().position(|&x| x == image_token_id) {
        let text_len = ed - st;
        for i in 0..text_len {
            let val = st_idx + i;
            t_ids.push(val);
            h_ids.push(val);
            w_ids.push(val);
        }
        st_idx += text_len;

        for _gt in 0..llm_grid_t {
            for gh in 0..llm_grid_h {
                for gw in 0..llm_grid_w {
                    t_ids.push(st_idx);
                    h_ids.push(st_idx + gh);
                    w_ids.push(st_idx + gw);
                }
            }
        }

        let max_coord = (llm_grid_h - 1).max(llm_grid_w - 1);
        st_idx += max_coord + 1;
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
    }

    if st < seq_len {
        let text_len = seq_len - st;
        for i in 0..text_len {
            let val = st_idx + i;
            t_ids.push(val);
            h_ids.push(val);
            w_ids.push(val);
        }
        st_idx += text_len;
    }

    (vec![t_ids, h_ids, w_ids], st_idx)
}

fn get_mrope_cos_sin(position_ids: &[Vec<usize>], inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let seq_len = position_ids[0].len();
    let mut cos_vec = vec![0.0f32; 3 * seq_len * 128];
    let mut sin_vec = vec![0.0f32; 3 * seq_len * 128];

    for comp in 0..3 {
        let pos_comp = &position_ids[comp];
        for s in 0..seq_len {
            let p = pos_comp[s] as f32;
            for i in 0..64 {
                let freq = p * inv_freq[i];
                let c = freq.cos();
                let s_val = freq.sin();

                let idx1 = comp * (seq_len * 128) + s * 128 + i;
                let idx2 = comp * (seq_len * 128) + s * 128 + i + 64;

                cos_vec[idx1] = c;
                cos_vec[idx2] = c;
                sin_vec[idx1] = s_val;
                sin_vec[idx2] = s_val;
            }
        }
    }
    (cos_vec, sin_vec)
}

fn apply_tilelang_mrope(
    _kernel: &bloomai_tilelang::TileLangKernel,
    _q: &Tensor,
    _k: &Tensor,
    _cos: &[f32],
    _sin: &[f32],
    _device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::bail!("mrope is no longer supported by TileLangKernel");
}

struct Qwen3VLVisionMLP {
    linear_fc1: candle_nn::Linear,
    linear_fc2: candle_nn::Linear,
}

impl Qwen3VLVisionMLP {
    fn new(vb: VarBuilder) -> Result<Self> {
        let linear_fc1 = candle_nn::linear(1024, 4096, vb.pp("linear_fc1"))?;
        let linear_fc2 = candle_nn::linear(4096, 1024, vb.pp("linear_fc2"))?;
        Ok(Self {
            linear_fc1,
            linear_fc2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_fc1.forward(x)?;
        let x_cubic = (x.powf(3.0)? * 0.044715)?;
        let inner = (((x.clone() + x_cubic)? * (2.0f64 / std::f64::consts::PI).sqrt())?.tanh())?;
        let x = ((x * 0.5)? * (inner + 1.0)?)?;
        Ok(self.linear_fc2.forward(&x)?)
    }
}

struct Qwen3VLVisionAttention {
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl Qwen3VLVisionAttention {
    fn new(num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = candle_nn::linear(1024, 3072, vb.pp("qkv"))?;
        let proj = candle_nn::linear(1024, 1024, vb.pp("proj"))?;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            scaling,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (seq_len, hidden_size) = x.dims2()?;
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((seq_len, 3, self.num_heads, self.head_dim))?;
        let q = qkv.narrow(1, 0, 1)?.squeeze(1)?;
        let k = qkv.narrow(1, 1, 1)?.squeeze(1)?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;

        let q = apply_rotary_pos_emb_vision(&q, cos, sin)?;
        let k = apply_rotary_pos_emb_vision(&k, cos, sin)?;

        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        let out = sdpa_vision(&q, &k, &v, self.scaling)?;
        let out = out.transpose(0, 1)?.reshape((seq_len, hidden_size))?;
        Ok(self.proj.forward(&out)?)
    }
}

fn sdpa_vision(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    #[cfg(feature = "flash-attn")]
    {
        let is_cuda = q.device().is_cuda();
        let is_metal = q.device().is_metal();
        if is_cuda || is_metal {
            let q4 = q.unsqueeze(0)?;
            let k4 = k.unsqueeze(0)?;
            let v4 = v.unsqueeze(0)?;
            if let Ok(res) = candle_flash_attn::flash_attn(&q4, &k4, &v4, scale as f32, false) {
                return Ok(res.transpose(1, 2)?.squeeze(0)?);
            }
        }
    }

    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    Ok(probs.matmul(v)?)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(candle_core::D::Minus1)?;
    let half = last_dim / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?)
}

fn apply_rotary_pos_emb_vision(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let r_x = rotate_half(x)?;
    Ok((x.broadcast_mul(&cos)? + r_x.broadcast_mul(&sin)?)?)
}

struct Qwen3VLVisionBlock {
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    attn: Qwen3VLVisionAttention,
    mlp: Qwen3VLVisionMLP,
}

impl Qwen3VLVisionBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(1024, 1e-6, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(1024, 1e-6, vb.pp("norm2"))?;
        let attn = Qwen3VLVisionAttention::new(16, 64, vb.pp("attn"))?;
        let mlp = Qwen3VLVisionMLP::new(vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = self.attn.forward(&h, cos, sin)?;
        let x = (x + h)?;
        let h = self.norm2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok((x + h)?)
    }
}

struct Qwen3VLVisionPatchMerger {
    norm: candle_nn::LayerNorm,
    linear_fc1: candle_nn::Linear,
    linear_fc2: candle_nn::Linear,
    use_postshuffle_norm: bool,
    hidden_size: usize,
}

impl Qwen3VLVisionPatchMerger {
    fn new(
        hidden_size: usize,
        out_hidden_size: usize,
        use_postshuffle_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm_size = if use_postshuffle_norm {
            hidden_size
        } else {
            1024
        };
        let norm = candle_nn::layer_norm(norm_size, 1e-6, vb.pp("norm"))?;
        let linear_fc1 = candle_nn::linear(hidden_size, hidden_size, vb.pp("linear_fc1"))?;
        let linear_fc2 = candle_nn::linear(hidden_size, out_hidden_size, vb.pp("linear_fc2"))?;
        Ok(Self {
            norm,
            linear_fc1,
            linear_fc2,
            use_postshuffle_norm,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let x = if self.use_postshuffle_norm {
            let x = x.reshape((n / 4, self.hidden_size))?;
            self.norm.forward(&x)?
        } else {
            let x = self.norm.forward(x)?;
            x.reshape((n / 4, self.hidden_size))?
        };
        let x = self.linear_fc1.forward(&x)?;
        let x = x.apply(&candle_nn::Activation::Gelu)?;
        Ok(self.linear_fc2.forward(&x)?)
    }
}

struct VisionAttentionPool {
    q_proj: Option<candle_nn::Linear>,
    k_proj: Option<candle_nn::Linear>,
    v_proj: Option<candle_nn::Linear>,
    out_proj: Option<candle_nn::Linear>,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl VisionAttentionPool {
    fn new(hidden_size: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = if vb
            .get((num_heads * head_dim, hidden_size), "q_proj.weight")
            .is_ok()
        {
            Some(candle_nn::linear(
                hidden_size,
                num_heads * head_dim,
                vb.pp("q_proj"),
            )?)
        } else {
            None
        };
        let k_proj = if q_proj.is_some() {
            Some(candle_nn::linear(
                hidden_size,
                num_heads * head_dim,
                vb.pp("k_proj"),
            )?)
        } else {
            None
        };
        let v_proj = if q_proj.is_some() {
            Some(candle_nn::linear(
                hidden_size,
                num_heads * head_dim,
                vb.pp("v_proj"),
            )?)
        } else {
            None
        };
        let out_proj = if q_proj.is_some() {
            Some(candle_nn::linear(
                num_heads * head_dim,
                hidden_size,
                vb.pp("out_proj"),
            )?)
        } else {
            None
        };
        let scaling = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scaling,
        })
    }

    fn forward(&self, x: &Tensor, pooled_len: usize) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        if seq_len <= pooled_len {
            return Ok(x.clone());
        }

        if let (Some(q_proj), Some(k_proj), Some(v_proj), Some(out_proj)) =
            (&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj)
        {
            let indices: Vec<u32> = (0..pooled_len)
                .map(|i| ((i * seq_len) / pooled_len) as u32)
                .collect();
            let query_device = x.device();
            let indices_tensor = Tensor::new(indices.as_slice(), query_device)?;
            let q_states = x.index_select(&indices_tensor, 0)?;

            let q = q_proj.forward(&q_states)?;
            let k = k_proj.forward(x)?;
            let v = v_proj.forward(x)?;

            let q = q
                .reshape((pooled_len, self.num_heads, self.head_dim))?
                .transpose(0, 1)?;
            let k = k
                .reshape((seq_len, self.num_heads, self.head_dim))?
                .transpose(0, 1)?;
            let v = v
                .reshape((seq_len, self.num_heads, self.head_dim))?
                .transpose(0, 1)?;

            // Run SDPA
            let scores = (q.matmul(&k.transpose(1, 2)?)? * self.scaling)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let out = probs.matmul(&v)?;

            let out = out
                .transpose(0, 1)?
                .reshape((pooled_len, self.num_heads * self.head_dim))?;
            Ok(out_proj.forward(&out)?)
        } else {
            let factor = (seq_len as f32 / pooled_len as f32).ceil() as usize;
            let mut pooled_vectors = Vec::with_capacity(pooled_len);
            for i in 0..pooled_len {
                let start = i * factor;
                let end = ((i + 1) * factor).min(seq_len);
                if start >= seq_len {
                    break;
                }
                let len = end - start;
                let slice = x.narrow(0, start, len)?;
                let mean = slice.mean(0)?;
                pooled_vectors.push(mean);
            }
            Ok(Tensor::stack(&pooled_vectors, 0)?)
        }
    }
}

struct Qwen3VLVisionModel {
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Qwen3VLVisionBlock>,
    merger: Qwen3VLVisionPatchMerger,
    deepstack_merger_list: Vec<Qwen3VLVisionPatchMerger>,
    deepstack_visual_indexes: Vec<usize>,
    num_grid_per_side: usize,
    inv_freq: Tensor,
    attention_pool: VisionAttentionPool,
    device: Device,
    dtype: DType,
}

impl Qwen3VLVisionModel {
    fn new(cfg: &VisionConfig, vb: VarBuilder, dev: &Device, dtype: DType) -> Result<Self> {
        let patch_embed_weight = vb
            .pp("patch_embed.proj")
            .get((1024, 3, 2, 16, 16), "weight")?;
        let patch_embed_bias = vb.pp("patch_embed.proj").get(1024, "bias")?;
        let pos_embed = vb.pp("pos_embed").get((2304, 1024), "weight")?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        let vb_b = vb.pp("blocks");
        for i in 0..cfg.depth {
            blocks.push(Qwen3VLVisionBlock::new(vb_b.pp(i))?);
        }

        let merger =
            Qwen3VLVisionPatchMerger::new(4096, cfg.out_hidden_size, false, vb.pp("merger"))?;
        let mut deepstack_merger_list = Vec::with_capacity(cfg.deepstack_visual_indexes.len());
        let vb_d = vb.pp("deepstack_merger_list");
        for i in 0..cfg.deepstack_visual_indexes.len() {
            deepstack_merger_list.push(Qwen3VLVisionPatchMerger::new(
                4096,
                cfg.out_hidden_size,
                true,
                vb_d.pp(i),
            )?);
        }

        let inv_freq_vec: Vec<f32> = (0..16)
            .map(|i| 1.0f32 / 10000.0f32.powf((2 * i) as f32 / 32.0f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq_vec, 16, dev)?.to_dtype(dtype)?;

        let attention_pool =
            VisionAttentionPool::new(cfg.out_hidden_size, 16, 64, vb.pp("attention_pool"))?;

        Ok(Self {
            patch_embed_weight,
            patch_embed_bias,
            pos_embed,
            blocks,
            merger,
            deepstack_merger_list,
            deepstack_visual_indexes: cfg.deepstack_visual_indexes.clone(),
            num_grid_per_side: 48,
            inv_freq,
            attention_pool,
            device: dev.clone(),
            dtype,
        })
    }

    fn patch_embed(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.patch_embed_weight.reshape((1024, 1536))?;
        let out = x.matmul(&w.transpose(0, 1)?)?;
        Ok(out.broadcast_add(&self.patch_embed_bias)?)
    }

    fn fast_pos_embed_interpolate(
        &self,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let h_idxs = linspace(
            0.0,
            (self.num_grid_per_side - 1) as f32,
            grid_h,
            &self.device,
        )?;
        let w_idxs = linspace(
            0.0,
            (self.num_grid_per_side - 1) as f32,
            grid_w,
            &self.device,
        )?;

        let mut idx_list = (0..4)
            .map(|_| Vec::with_capacity(grid_h * grid_w))
            .collect::<Vec<_>>();
        let mut weight_list = (0..4)
            .map(|_| Vec::with_capacity(grid_h * grid_w))
            .collect::<Vec<_>>();

        let h_idxs_vec = h_idxs.to_vec1::<f32>()?;
        let w_idxs_vec = w_idxs.to_vec1::<f32>()?;

        for h_val in h_idxs_vec {
            let h_floor = h_val.floor() as i32;
            let h_ceil = (h_floor + 1).min((self.num_grid_per_side - 1) as i32);
            let dh = h_val - h_floor as f32;
            let base_h = h_floor * self.num_grid_per_side as i32;
            let base_h_ceil = h_ceil * self.num_grid_per_side as i32;

            for &w_val in &w_idxs_vec {
                let w_floor = w_val.floor() as i32;
                let w_ceil = (w_floor + 1).min((self.num_grid_per_side - 1) as i32);
                let dw = w_val - w_floor as f32;

                idx_list[0].push((base_h + w_floor) as u32);
                idx_list[1].push((base_h + w_ceil) as u32);
                idx_list[2].push((base_h_ceil + w_floor) as u32);
                idx_list[3].push((base_h_ceil + w_ceil) as u32);

                weight_list[0].push((1.0 - dh) * (1.0 - dw));
                weight_list[1].push((1.0 - dh) * dw);
                weight_list[2].push(dh * (1.0 - dw));
                weight_list[3].push(dh * dw);
            }
        }

        let mut patch_pos_embeds =
            Tensor::zeros((grid_h * grid_w, 1024), self.dtype, &self.device)?;
        for i in 0..4 {
            let idx_tensor = Tensor::from_vec(idx_list[i].clone(), grid_h * grid_w, &self.device)?;
            let weight_tensor =
                Tensor::from_vec(weight_list[i].clone(), (grid_h * grid_w, 1), &self.device)?
                    .to_dtype(self.dtype)?;
            let emb = self.pos_embed.index_select(&idx_tensor, 0)?;
            let weighted_emb = emb.broadcast_mul(&weight_tensor)?;
            patch_pos_embeds = (patch_pos_embeds + weighted_emb)?;
        }

        let mut t_embeds = Vec::with_capacity(grid_t);
        for _ in 0..grid_t {
            let pos_view = patch_pos_embeds
                .reshape((grid_h / 2, 2, grid_w / 2, 2, 1024))?
                .transpose(1, 2)?
                .reshape(((grid_h / 2) * (grid_w / 2) * 4, 1024))?;
            t_embeds.push(pos_view);
        }
        Ok(Tensor::cat(&t_embeds, 0)?)
    }

    fn rot_pos_emb(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> Result<(Tensor, Tensor)> {
        let max_hw = grid_h.max(grid_w);
        let seq: Vec<f32> = (0..max_hw).map(|i| i as f32).collect();
        let seq_tensor = Tensor::new(seq, &self.device)?.unsqueeze(1)?;
        let inv_freq = self.inv_freq.unsqueeze(0)?;
        let freq_table = seq_tensor.matmul(&inv_freq)?;

        let merged_h = grid_h / 2;
        let merged_w = grid_w / 2;

        let mut pos_ids = Vec::with_capacity(grid_t * merged_h * merged_w * 4);
        for _ in 0..grid_t {
            for bh in 0..merged_h {
                for bw in 0..merged_w {
                    for ih in 0..2 {
                        for iw in 0..2 {
                            pos_ids.push((bh * 2 + ih, bw * 2 + iw));
                        }
                    }
                }
            }
        }

        let freq_table_vec = freq_table.to_vec2::<f32>()?;
        let mut cos_vec = Vec::with_capacity(pos_ids.len() * 64);
        let mut sin_vec = Vec::with_capacity(pos_ids.len() * 64);

        for &(r, c) in &pos_ids {
            let row_freqs = &freq_table_vec[r];
            let col_freqs = &freq_table_vec[c];
            let mut emb = Vec::with_capacity(32);
            emb.extend_from_slice(row_freqs);
            emb.extend_from_slice(col_freqs);

            let mut emb64 = Vec::with_capacity(64);
            emb64.extend_from_slice(&emb);
            emb64.extend_from_slice(&emb);

            for &val in &emb64 {
                cos_vec.push(val.cos());
                sin_vec.push(val.sin());
            }
        }

        let total_tokens = pos_ids.len();
        let cos =
            Tensor::from_vec(cos_vec, (total_tokens, 64), &self.device)?.to_dtype(self.dtype)?;
        let sin =
            Tensor::from_vec(sin_vec, (total_tokens, 64), &self.device)?.to_dtype(self.dtype)?;

        Ok((cos, sin))
    }

    fn forward(
        &self,
        x: &Tensor,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut hidden_states = self.patch_embed(x)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_t, grid_h, grid_w)?;
        hidden_states = (hidden_states + pos_embeds)?;

        let (cos, sin) = self.rot_pos_emb(grid_t, grid_h, grid_w)?;
        let mut deepstack_features = Vec::new();
        for (idx, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, &cos, &sin)?;
            if let Some(list_idx) = self
                .deepstack_visual_indexes
                .iter()
                .position(|&candidate| candidate == idx)
            {
                let feat = self.deepstack_merger_list[list_idx].forward(&hidden_states)?;
                deepstack_features.push(feat);
            }
        }
        let output = self.merger.forward(&hidden_states)?;
        let output = self.attention_pool.forward(&output, 512)?;
        Ok((output, deepstack_features))
    }
}

pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
        let r = xs
            .unsqueeze(2)?
            .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(r)
    }
}

struct Qwen3RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen3RmsNorm {
    fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x_f32 = x.to_dtype(internal_dtype)?;
        let norm_x = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        Ok(x_normed.broadcast_mul(&self.weight)?)
    }
}

struct Qwen3VLAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: Option<Qwen3RmsNorm>,
    k_norm: Option<Qwen3RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    kv_cache: candle_nn::kv_cache::ConcatKvCache,
}

impl Qwen3VLAttention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let q_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;

        let q_norm = if vb.pp("q_norm").get(cfg.head_dim, "weight").is_ok() {
            Some(Qwen3RmsNorm::new(
                cfg.head_dim,
                cfg.rms_norm_eps,
                vb.pp("q_norm"),
            )?)
        } else {
            None
        };
        let k_norm = if vb.pp("k_norm").get(cfg.head_dim, "weight").is_ok() {
            Some(Qwen3RmsNorm::new(
                cfg.head_dim,
                cfg.rms_norm_eps,
                vb.pp("k_norm"),
            )?)
        } else {
            None
        };

        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let kv_cache = candle_nn::kv_cache::ConcatKvCache::new(2);

        Ok(Self {
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
            kv_cache,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        kernel: &bloomai_tilelang::TileLangKernel,
        cos: &[f32],
        sin: &[f32],
        attn_mask: Option<&Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = if let Some(q_norm) = &self.q_norm {
            let q_flat = q.flatten(0, 2)?;
            let q_flat = q_norm.forward(&q_flat)?;
            q_flat.reshape((b, self.num_heads, l, self.head_dim))?
        } else {
            q
        };
        let k = if let Some(k_norm) = &self.k_norm {
            let k_flat = k.flatten(0, 2)?;
            let k_flat = k_norm.forward(&k_flat)?;
            k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?
        } else {
            k
        };

        let (q, k) = apply_tilelang_mrope(kernel, &q, &k, cos, sin, device)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let out = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

struct Qwen3VLMLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Qwen3VLMLP {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_out = self.gate_proj.forward(x)?;
        let lhs = gate_out.apply(&candle_nn::Activation::Silu)?;
        let rhs = self.up_proj.forward(x)?;
        let prod = (lhs * rhs)?;
        Ok(self.down_proj.forward(&prod)?)
    }
}

struct Qwen3VLDecoderLayer {
    self_attn: Qwen3VLAttention,
    mlp: Qwen3VLMLP,
    ln1: Qwen3RmsNorm,
    ln2: Qwen3RmsNorm,
}

impl Qwen3VLDecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3VLAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Qwen3VLMLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = Qwen3RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = Qwen3RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        kernel: &bloomai_tilelang::TileLangKernel,
        cos: &[f32],
        sin: &[f32],
        mask: Option<&Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, kernel, cos, sin, mask, device)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        Ok((x + h2)?)
    }
}

fn add_deepstack_features(
    hidden_states: &Tensor,
    visual_indices: &[usize],
    visual_embeds: &Tensor,
) -> Result<Tensor> {
    let (b, seq_len, hs) = hidden_states.dims3()?;
    let dev = hidden_states.device();
    let dtype = hidden_states.dtype();

    let visual_embeds_vec = visual_embeds.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut full_vec = vec![0.0f32; seq_len * hs];
    for (i, &idx) in visual_indices.iter().enumerate() {
        if idx < seq_len {
            full_vec[idx * hs..(idx + 1) * hs].copy_from_slice(&visual_embeds_vec[i]);
        }
    }
    let full_tensor = Tensor::from_vec(full_vec, (b, seq_len, hs), dev)?.to_dtype(dtype)?;
    Ok((hidden_states + full_tensor)?)
}

struct Qwen3VLModelInner {
    layers: Vec<Qwen3VLDecoderLayer>,
    current_prefix: Vec<u32>,
}

pub struct Qwen3VLModel {
    config: Qwen3VLConfig,
    tokenizer: tokenizers::Tokenizer,
    vision_model: Qwen3VLVisionModel,
    embed_tokens: Tensor,
    inner: Mutex<Qwen3VLModelInner>,
    norm: Qwen3RmsNorm,
    lm_head: Option<candle_nn::Linear>,
    tilelang_kernel: bloomai_tilelang::TileLangKernel,
    inv_freq: Vec<f32>,
    device: Device,
    dtype: DType,
    metadata: ModelMetadata,
}

impl Qwen3VLModel {
    fn forward_pass(
        &self,
        inner: &mut Qwen3VLModelInner,
        input_ids: &[u32],
        image_bytes: Option<&[u8]>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let mut vision_features = None;
        let mut image_grid_thw = None;
        if let Some(bytes) = image_bytes {
            let (patches, gt, gh, gw) = preprocess_image(bytes, 65536, 16777216)?;
            let patches_tensor = Tensor::from_vec(patches, (gt * gh * gw, 1536), &self.device)?
                .to_dtype(self.dtype)?;
            let (feat, deepstack_feats) = self.vision_model.forward(&patches_tensor, gt, gh, gw)?;
            vision_features = Some((feat, deepstack_feats));
            image_grid_thw = Some((gt, gh, gw));
        }

        let position_ids = if seq_len == 1 && image_bytes.is_none() {
            vec![vec![start_pos], vec![start_pos], vec![start_pos]]
        } else {
            let (ids, _) = get_rope_index(input_ids, image_grid_thw);
            ids
        };
        let (cos_mrope, sin_mrope) = get_mrope_cos_sin(&position_ids, &self.inv_freq);

        let input_ids_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let mut h = self
            .embed_tokens
            .index_select(&input_ids_tensor.flatten_all()?, 0)?
            .reshape((1, seq_len, self.config.text_config.hidden_size))?;

        let mut visual_indices = Vec::new();
        if let Some((feat, _)) = &vision_features {
            let image_token_id = 151655;
            for (i, &tok) in input_ids.iter().enumerate() {
                if tok == image_token_id {
                    visual_indices.push(i);
                }
            }
            let k = feat.dim(0)?;
            if visual_indices.len() == k {
                let feat_f32 = feat.to_dtype(DType::F32)?;
                let feat_vec = feat_f32.to_vec2::<f32>()?;
                let mut h_f32 = h.to_dtype(DType::F32)?.to_vec3::<f32>()?;

                for (i, &idx) in visual_indices.iter().enumerate() {
                    h_f32[0][idx] = feat_vec[i].clone();
                }
                h = Tensor::new(h_f32, &self.device)?.to_dtype(self.dtype)?;
            }
        }

        let mask = if seq_len == 1 {
            None
        } else {
            let minf = f32::NEG_INFINITY;
            let mask_vec: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0. } else { minf }))
                .collect();
            let m = Tensor::from_slice(&mask_vec, (1, 1, seq_len, seq_len), &self.device)?
                .to_dtype(self.dtype)?;
            Some(m)
        };

        for (layer_idx, layer) in inner.layers.iter_mut().enumerate() {
            h = layer.forward(
                &h,
                &self.tilelang_kernel,
                &cos_mrope,
                &sin_mrope,
                mask.as_ref(),
                &self.device,
            )?;
            if let Some((_, deepstack_feats)) = &vision_features
                && layer_idx < deepstack_feats.len()
            {
                h = add_deepstack_features(&h, &visual_indices, &deepstack_feats[layer_idx])?;
            }
        }

        let h = self.norm.forward(&h)?;
        let h_last = h.narrow(1, seq_len - 1, 1)?;
        let logits = if let Some(lm_head) = &self.lm_head {
            lm_head.forward(&h_last)?
        } else {
            let (b_sz, seq_len, hidden_size) = h_last.dims3()?;
            let h_2d = h_last.reshape((b_sz * seq_len, hidden_size))?;
            let logits_2d = h_2d.matmul(&self.embed_tokens.transpose(0, 1)?)?;
            logits_2d.reshape((b_sz, seq_len, self.config.text_config.vocab_size))?
        };
        Ok(logits.squeeze(0)?.squeeze(0)?)
    }
}

impl LoadedModel for Qwen3VLModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let mut text_parts = Vec::new();
        self.infer_stream(input, params, &mut |chunk: crate::io::OutputChunk| {
            if let crate::io::OutputChunk::TextDelta(delta) = chunk {
                text_parts.push(delta);
            }
            Ok(())
        })?;

        let text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };

        Ok(ModelOutput {
            text,
            logits: None,
            image: None,
            audio: None,
            video: None,
        })
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        let (prompt, image_bytes) = match input {
            ModelInput::Text { prompt } => (prompt, None),
            ModelInput::Vision { bytes, .. } => ("Describe this image.".to_string(), Some(bytes)),
            ModelInput::Multi { text, image, .. } => {
                let p = text.unwrap_or_else(|| "Describe this image.".to_string());
                (p, image)
            }
            _ => {
                return Err(anyhow!(
                    "Qwen3-VL model only supports text or image/text input"
                ));
            }
        };

        let mut formatted_prompt = prompt.clone();
        let mut img_bytes_ref = None;
        let mut grid_dims = None;

        if let Some(ref bytes) = image_bytes {
            let img = image::load_from_memory(bytes)
                .map_err(|e| anyhow!("Failed to load image: {}", e))?;
            let width = img.width() as usize;
            let height = img.height() as usize;
            let (resized_height, resized_width) = smart_resize(height, width, 32, 65536, 16777216);

            let grid_t = 1;
            let grid_h = resized_height / 16;
            let grid_w = resized_width / 16;
            grid_dims = Some((grid_t, grid_h, grid_w));
            let pad_tokens_count = (grid_t * grid_h * grid_w) / 4;

            let pad_str = "<|image_pad|>".repeat(pad_tokens_count);
            formatted_prompt = format!(
                "<|im_start|>user\n<|vision_start|>{}{}<|vision_end|>{}\n<|im_end|>\n<|im_start|>assistant\n",
                pad_str, "", prompt
            );
            img_bytes_ref = Some(bytes.as_slice());
        }

        let encoding = self
            .tokenizer
            .encode(formatted_prompt, false)
            .map_err(|e| anyhow!("Tokenizer encode error: {}", e))?;
        let token_ids = encoding.get_ids().to_vec();

        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        for layer in &mut inner.layers {
            layer.self_attn.clear_kv_cache();
        }
        inner.current_prefix = token_ids.clone();

        let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(
            params.seed.unwrap_or(42),
            Some(params.temperature),
            Some(params.top_p),
        );

        let mut all_tokens = token_ids.clone();
        let mut prev_text = String::new();

        let mut current_ids = token_ids.clone();
        let mut logits = self.forward_pass(&mut inner, &current_ids, img_bytes_ref, 0)?;
        let mut next_token = logits_processor.sample(&logits)?;

        let eos_token_id = 151645;
        let max_tokens = params.max_tokens.min(4096);
        let mut steps = 0;
        let (_, initial_next_pos) = get_rope_index(&token_ids, grid_dims);
        let mut start_pos = initial_next_pos;

        while next_token != eos_token_id && steps < max_tokens {
            all_tokens.push(next_token);
            inner.current_prefix.push(next_token);

            let text = self
                .tokenizer
                .decode(&all_tokens[token_ids.len()..], true)
                .map_err(|e| anyhow!("tokenizer decode error: {}", e))?;
            if text.len() > prev_text.len() {
                let new_text = &text[prev_text.len()..];
                sink.on_chunk(crate::io::OutputChunk::TextDelta(new_text.to_string()))?;
                prev_text = text;
            }

            steps += 1;

            current_ids = vec![next_token];
            logits = self.forward_pass(&mut inner, &current_ids, None, start_pos)?;
            next_token = logits_processor.sample(&logits)?;
            start_pos += 1;
        }

        sink.on_chunk(crate::io::OutputChunk::End)?;
        Ok(())
    }
}

pub struct Qwen3VLEngine;

impl Engine for Qwen3VLEngine {
    fn name(&self) -> &'static str {
        "qwen3_vl"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text, Modality::Vision]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu, DeviceKind::Gpu]
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "qwen3_vl",
            supported_families: vec![ModelFamily::Qwen],
            supported_dtypes: vec![
                bloomai_core::DType::F32,
                bloomai_core::DType::F16,
                bloomai_core::DType::BF16,
            ],
            supported_formats: vec![ModelFormat::Safetensors],
            supported_devices: vec![
                DeviceClass::Cpu,
                DeviceClass::IntegratedGpu,
                DeviceClass::DiscreteGpu,
            ],
            supported_modalities: vec![Modality::Text, Modality::Vision],
            supports_streaming: true,
            supports_quantized_models: false,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: true,
            max_context_tokens: None,
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![crate::core::parallelism::ParallelStrategy::None],
            maturity: crate::engine::BackendMaturity::Beta,
            diagnostic_tips: vec![
                "Qwen3-VL requires safetensors weights with vision encoder.".to_string(),
                "Ensure config.json has model_type=qwen2_vl or qwen3_vl.".to_string(),
            ],
            construction_guide:
                "Built-in candle backend for Qwen3-VL. Build with --features candle-engine."
                    .to_string(),
        }
    }

    fn supports(
        &self,
        manifest: &bloomai_core::ModelManifest,
        capability: &DeviceCapability,
    ) -> SupportLevel {
        let model_type_ok = manifest.id.to_lowercase().contains("vl")
            || manifest
                .parameters
                .get("model_type")
                .and_then(|v| v.as_str())
                .map(|s| s.contains("vl"))
                .unwrap_or(false);

        if !model_type_ok {
            return SupportLevel::Unsupported("model is not a Qwen VL model".to_string());
        }

        default_engine_supports(&self.capability(), manifest, capability)
    }

    fn load(&self, model_path: &Path, device_kind: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let device = match device_kind {
            DeviceKind::Cpu => Device::Cpu,
            DeviceKind::Gpu => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0)?
                }
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0)?
                }
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                {
                    Device::Cpu
                }
            }
            DeviceKind::Npu => Device::Cpu,
        };

        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => DType::F16,
        };

        let config_content = std::fs::read_to_string(model_path.join("config.json"))?;
        let config: Qwen3VLConfig = serde_json::from_str(&config_content)?;

        let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let safetensors_files = find_safetensors_files(model_path);
        if safetensors_files.is_empty() {
            return Err(anyhow!(
                "No safetensors files found in {}",
                model_path.display()
            ));
        }

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)? };

        let compiler = bloomai_tilelang::TileLangCompiler::new()?;
        let so_path = compiler.compile_vector_add(1024)?;
        let tilelang_kernel = unsafe { bloomai_tilelang::TileLangKernel::load(&so_path)? };

        let vision_model =
            Qwen3VLVisionModel::new(&config.vision_config, vb.pp("model.visual"), &device, dtype)?;

        let embed_tokens = vb.pp("model.language_model.embed_tokens").get(
            (
                config.text_config.vocab_size,
                config.text_config.hidden_size,
            ),
            "weight",
        )?;

        let mut layers = Vec::with_capacity(config.text_config.num_hidden_layers);
        let vb_l = vb.pp("model.language_model.layers");
        for i in 0..config.text_config.num_hidden_layers {
            let layer = Qwen3VLDecoderLayer::new(&config.text_config, vb_l.pp(i))?;
            layers.push(layer);
        }

        let norm = Qwen3RmsNorm::new(
            config.text_config.hidden_size,
            config.text_config.rms_norm_eps,
            vb.pp("model.language_model.norm"),
        )?;

        let lm_head = if config.text_config.tie_word_embeddings {
            None
        } else {
            Some(candle_nn::linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        let inv_freq: Vec<f32> = (0..64)
            .map(|i| {
                1.0f32 / (config.text_config.rope_theta as f32).powf((2 * i) as f32 / 128.0f32)
            })
            .collect();

        let is_quantized = model_path.to_string_lossy().to_lowercase().contains("int8")
            || model_path.to_string_lossy().to_lowercase().contains("int4")
            || model_path
                .to_string_lossy()
                .to_lowercase()
                .contains("quant");

        let manifest = crate::manifest_adapter::load_manifest(model_path)?;
        let metadata = ModelMetadata {
            id: model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            modality: Modality::Vision,
            quantized: is_quantized,
            manifest,
        };

        let inner = Mutex::new(Qwen3VLModelInner {
            layers,
            current_prefix: Vec::new(),
        });

        Ok(Box::new(Qwen3VLModel {
            config,
            tokenizer,
            vision_model,
            embed_tokens,
            inner,
            norm,
            lm_head,
            tilelang_kernel,
            inv_freq,
            device,
            dtype,
            metadata,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vlm_minimal_path() {
        use candle_core::{DType, Device};
        use candle_nn::VarMap;

        let dev = Device::Cpu;
        let dtype = DType::F32;
        let vm = VarMap::new();

        // 1. Create a dummy image of 32x32 pixels
        let mut img = image::RgbImage::new(32, 32);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([128, 128, 128]);
        }
        let mut cursor = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        let image_bytes = cursor.into_inner();

        // Run image preprocessor
        let (patches, gt, gh, gw) = preprocess_image(&image_bytes, 65536, 16777216).unwrap();
        assert!(gt > 0);
        assert!(gh > 0);
        assert!(gw > 0);
        assert_eq!(patches.len(), gt * gh * gw * 1536);

        // 2. Setup mock weights for Qwen3VLVisionModel
        let cfg = VisionConfig {
            depth: 1,
            hidden_size: 1024,
            in_channels: 3,
            intermediate_size: 2048,
            num_heads: 16,
            num_position_embeddings: 2304,
            out_hidden_size: 512,
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            deepstack_visual_indexes: vec![0],
        };

        // We need to register variables in VarMap for the visual model
        let _w_patch = vm
            .get(
                (1024, 3, 2, 16, 16),
                "model.visual.patch_embed.proj.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _b_patch = vm
            .get(
                1024,
                "model.visual.patch_embed.proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _pos_emb = vm
            .get(
                (2304, 1024),
                "model.visual.pos_embed.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();

        let _merger_ln = vm
            .get(
                4096,
                "model.visual.merger.ln_q.weight",
                candle_nn::Init::Const(1.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _merger_ln_b = vm
            .get(
                4096,
                "model.visual.merger.ln_q.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _merger_proj = vm
            .get(
                (512, 4096),
                "model.visual.merger.proj.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _merger_bias = vm
            .get(
                512,
                "model.visual.merger.proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();

        let _ds_ln = vm
            .get(
                4096,
                "model.visual.deepstack_merger_list.0.ln_q.weight",
                candle_nn::Init::Const(1.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _ds_ln_b = vm
            .get(
                4096,
                "model.visual.deepstack_merger_list.0.ln_q.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _ds_proj = vm
            .get(
                (512, 4096),
                "model.visual.deepstack_merger_list.0.proj.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _ds_bias = vm
            .get(
                512,
                "model.visual.deepstack_merger_list.0.proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();

        let _ln1 = vm
            .get(
                1024,
                "model.visual.blocks.0.norm1.weight",
                candle_nn::Init::Const(1.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _ln1_b = vm
            .get(
                1024,
                "model.visual.blocks.0.norm1.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _ln2 = vm
            .get(
                1024,
                "model.visual.blocks.0.norm2.weight",
                candle_nn::Init::Const(1.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _ln2_b = vm
            .get(
                1024,
                "model.visual.blocks.0.norm2.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();

        let _qkv = vm
            .get(
                (3072, 1024),
                "model.visual.blocks.0.attn.qkv.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _qkv_b = vm
            .get(
                3072,
                "model.visual.blocks.0.attn.qkv.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _proj = vm
            .get(
                (1024, 1024),
                "model.visual.blocks.0.attn.proj.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _proj_b = vm
            .get(
                1024,
                "model.visual.blocks.0.attn.proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();

        let _fc1 = vm
            .get(
                (2048, 1024),
                "model.visual.blocks.0.mlp.fc1.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _fc1_b = vm
            .get(
                2048,
                "model.visual.blocks.0.mlp.fc1.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _fc2 = vm
            .get(
                (1024, 2048),
                "model.visual.blocks.0.mlp.fc2.weight",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _fc2_b = vm
            .get(
                1024,
                "model.visual.blocks.0.mlp.fc2.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();

        let vb = VarBuilder::from_varmap(&vm, dtype, &dev);

        // Instantiate Vision Model
        let vision_model =
            Qwen3VLVisionModel::new(&cfg, vb.pp("model.visual"), &dev, dtype).unwrap();

        // 3. Process image with vision encoder
        let patches_tensor = Tensor::from_vec(patches, (gt * gh * gw, 1536), &dev).unwrap();
        let (output, deepstack_feats) = vision_model.forward(&patches_tensor, gt, gh, gw).unwrap();

        // Verification of vision encoder outputs
        assert_eq!(output.dims(), &[(gt * gh * gw) / 4, 512]);
        assert_eq!(deepstack_feats.len(), 1);
        assert_eq!(deepstack_feats[0].dims(), &[(gt * gh * gw) / 4, 512]);
    }

    #[test]
    fn test_vision_attention_pool() {
        use candle_core::{DType, Device, Tensor};
        use candle_nn::VarMap;

        let dev = Device::Cpu;
        let dtype = DType::F32;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, dtype, &dev);

        // 1. Without weights (fallback to grid pooling)
        let pool = VisionAttentionPool::new(512, 8, 32, vb.pp("pool")).unwrap();
        // Generate a 16x512 tensor of features
        let x = Tensor::ones((16, 512), dtype, &dev).unwrap();
        // Pool down to 8 tokens
        let out = pool.forward(&x, 8).unwrap();
        assert_eq!(out.dims(), &[8, 512]);

        // 2. With weights (full attention pooling path)
        let _q_w = vm
            .get(
                (256, 512),
                "pool.q_proj.weight",
                candle_nn::Init::Const(0.1),
                dtype,
                &dev,
            )
            .unwrap();
        let _q_b = vm
            .get(
                256,
                "pool.q_proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _k_w = vm
            .get(
                (256, 512),
                "pool.k_proj.weight",
                candle_nn::Init::Const(0.1),
                dtype,
                &dev,
            )
            .unwrap();
        let _k_b = vm
            .get(
                256,
                "pool.k_proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _v_w = vm
            .get(
                (256, 512),
                "pool.v_proj.weight",
                candle_nn::Init::Const(0.1),
                dtype,
                &dev,
            )
            .unwrap();
        let _v_b = vm
            .get(
                256,
                "pool.v_proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();
        let _out_w = vm
            .get(
                (512, 256),
                "pool.out_proj.weight",
                candle_nn::Init::Const(0.1),
                dtype,
                &dev,
            )
            .unwrap();
        let _out_b = vm
            .get(
                512,
                "pool.out_proj.bias",
                candle_nn::Init::Const(0.0),
                dtype,
                &dev,
            )
            .unwrap();

        let vb_w = VarBuilder::from_varmap(&vm, dtype, &dev);
        let pool_w = VisionAttentionPool::new(512, 8, 32, vb_w.pp("pool")).unwrap();
        let out_w = pool_w.forward(&x, 8).unwrap();
        assert_eq!(out_w.dims(), &[8, 512]);
    }
}
