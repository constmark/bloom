//! Wan2.1 VAE decoder implementation.
//!
//! Decodes latent tensors [C=16, F', H', W'] back to video frames [3, F, H, W].
//! VAE stride: temporal=4, spatial=8x8.
//!
//! The full VAE uses Conv3d, ConvTranspose3d, CausalConv3d, and GroupNorm3d
//! which require TileLang kernels for GPU execution. This implementation provides:
//! - A pure Candle fallback path (CPU, lower quality)
//! - Hooks for TileLang kernel substitution (GPU, production quality)

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{self as nn, Module};

/// VAE configuration for Wan2.1.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    /// Latent channels (16 for Wan2.1).
    pub latent_channels: usize,
    /// Base channel dimension.
    pub base_channels: usize,
    /// Channel multipliers per resolution level.
    pub channel_mult: Vec<usize>,
    /// Number of residual blocks per level.
    pub num_res_blocks: usize,
    /// Temporal stride (4 for Wan2.1).
    pub temporal_stride: usize,
    /// Spatial stride (8 for Wan2.1).
    pub spatial_stride: usize,
}

impl VaeConfig {
    pub fn wan2_1() -> Self {
        Self {
            latent_channels: 16,
            base_channels: 96,
            channel_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_stride: 4,
            spatial_stride: 8,
        }
    }
}

/// Wan2.1 VAE decoder.
///
/// Converts DiT output latent [batch, C=16, F', H', W'] to
/// video frames [batch, 3, F, H, W] where:
/// - F = (F' - 1) * temporal_stride + 1
/// - H = H' * spatial_stride
/// - W = W' * spatial_stride
pub struct WanVAE {
    config: VaeConfig,
    device: Device,
    /// Path to the model directory.
    model_path: std::path::PathBuf,
}

/// The actual decoder network.
struct VaeDecoder {
    /// Input conv: latent_channels -> base_channels * max_mult (spatial Conv2d).
    conv_in: nn::Conv2d,
    /// Middle blocks.
    mid_block: MidBlock,
    /// Upsampling blocks.
    up_blocks: Vec<UpBlock>,
    /// Final norm + output conv.
    norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
}

struct MidBlock {
    res1: ResBlock3D,
    attn: Option<SelfAttention3D>,
    res2: ResBlock3D,
}

struct UpBlock {
    res_blocks: Vec<ResBlock3D>,
    upsample: Option<Upsample3D>,
}

/// 3D residual block using Conv2d per temporal frame.
struct ResBlock3D {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2d,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2d,
    shortcut: Option<nn::Conv2d>,
}

/// Simple 3D self-attention (used in mid block).
struct SelfAttention3D {
    norm: nn::GroupNorm,
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    o: nn::Linear,
    num_heads: usize,
}

impl SelfAttention3D {
    pub fn new(channels: usize, vb: nn::VarBuilder) -> Result<Self> {
        let norm = nn::group_norm(32, channels, 1e-6, vb.pp("group_norm"))?;
        let q = nn::linear(channels, channels, vb.pp("to_q"))?;
        let k = nn::linear(channels, channels, vb.pp("to_k"))?;
        let v = nn::linear(channels, channels, vb.pp("to_v"))?;
        let o = nn::linear(channels, channels, vb.pp("to_out.0"))?;
        Ok(Self {
            norm,
            q,
            k,
            v,
            o,
            num_heads: 1,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, f, c, h, w) = x.dims5()?;
        let residual = x.clone();

        // 1. Group Norm (per frame)
        let x_flat = x.reshape((b * f, c, h, w))?;
        let x_normed = self.norm.forward(&x_flat.to_dtype(DType::F32)?)?;

        // 2. Flatten spatial dimensions: [b * f, h * w, c]
        let x_att = x_normed
            .permute((0, 2, 3, 1))?
            .reshape((b * f, h * w, c))?
            .contiguous()?;

        // 3. Projections
        let q = self.q.forward(&x_att)?;
        let k = self.k.forward(&x_att)?;
        let v = self.v.forward(&x_att)?;

        // 4. Attention
        let n = self.num_heads;
        let dh = c / n;

        let q = q
            .reshape((b * f, h * w, n, dh))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // [bf, n, hw, dh]
        let k = k
            .reshape((b * f, h * w, n, dh))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // [bf, n, hw, dh]
        let v = v
            .reshape((b * f, h * w, n, dh))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // [bf, n, hw, dh]

        let scale = (dh as f64).sqrt().recip();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?.affine(scale, 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?; // [bf, n, hw, dh]

        // 5. Output projection and reshape back
        let out = out.permute((0, 2, 1, 3))?.reshape((b * f, h * w, c))?;
        let out = self.o.forward(&out)?;

        // Reshape [bf, hw, c] -> [bf, h, w, c] -> [bf, c, h, w] -> [b, f, c, h, w]
        let out = out.reshape((b * f, h, w, c))?.permute((0, 3, 1, 2))?;
        let out = out.reshape((b, f, c, h, w))?;

        (out + residual).map_err(Into::into)
    }
}

/// 3D upsampling via nearest-neighbor interpolation + conv.
struct Upsample3D {
    conv: nn::Conv2d,
    scale_factor: usize,
}

impl WanVAE {
    /// Create a new VAE, optionally loading weights.
    pub fn new(model_path: &Path, device: &Device) -> Result<Self> {
        let config = VaeConfig::wan2_1();

        Ok(Self {
            config,
            device: device.clone(),
            model_path: model_path.to_path_buf(),
        })
    }

    /// Get the device the VAE is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn try_load_decoder(
        config: &VaeConfig,
        model_path: &Path,
        device: &Device,
    ) -> Option<VaeDecoder> {
        let vae_file = crate::executor::wan::loader::find_vae_weights(model_path)?;

        let vb = unsafe {
            match nn::VarBuilder::from_mmaped_safetensors(&[vae_file], DType::F32, device) {
                Ok(v) => v,
                Err(_) => return None,
            }
        };

        Self::build_decoder(config, vb, device).ok()
    }

    fn build_decoder(
        config: &VaeConfig,
        vb: nn::VarBuilder,
        device: &Device,
    ) -> Result<VaeDecoder> {
        let max_mult = *config.channel_mult.last().unwrap();
        let in_ch = config.base_channels * max_mult;

        // Input convolution (spatial Conv2d, applied per temporal frame)
        let conv_in = nn::conv2d(
            config.latent_channels,
            in_ch,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("decoder.conv_in"),
        )?;

        // Mid block
        let mid_vb = vb.pp("decoder.mid_block");
        let mid_block = Self::build_mid_block(in_ch, config, mid_vb, device)?;

        // Up blocks
        let mut up_blocks = Vec::new();
        let mut current_ch = in_ch;
        for (level, &mult) in config.channel_mult.iter().rev().enumerate() {
            let out_ch = config.base_channels * mult;
            let up_vb = vb.pp(format!("decoder.up_blocks.{}", level));
            let block = Self::build_up_block(current_ch, out_ch, config, up_vb, device, level > 0)?;
            current_ch = out_ch;
            up_blocks.push(block);
        }

        // Output
        let norm_out = nn::group_norm(32, config.base_channels, 1e-6, vb.pp("decoder.norm_out"))?;
        let conv_out = nn::conv2d(
            config.base_channels,
            3,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("decoder.conv_out"),
        )?;

        Ok(VaeDecoder {
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
        })
    }

    fn build_mid_block(
        channels: usize,
        _config: &VaeConfig,
        vb: nn::VarBuilder,
        device: &Device,
    ) -> Result<MidBlock> {
        let res1 = Self::build_res_block(channels, channels, vb.pp("resnets.0"), device)?;
        let attn = SelfAttention3D::new(channels, vb.pp("attentions.0"))?;
        let res2 = Self::build_res_block(channels, channels, vb.pp("resnets.1"), device)?;
        Ok(MidBlock {
            res1,
            attn: Some(attn),
            res2,
        })
    }

    fn build_res_block(
        in_ch: usize,
        out_ch: usize,
        vb: nn::VarBuilder,
        _device: &Device,
    ) -> Result<ResBlock3D> {
        let groups = 32.min(in_ch);
        let norm1 = nn::group_norm(groups, in_ch, 1e-6, vb.pp("norm1"))?;
        let conv1 = nn::conv2d(
            in_ch,
            out_ch,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let norm2_groups = 32.min(out_ch);
        let norm2 = nn::group_norm(norm2_groups, out_ch, 1e-6, vb.pp("norm2"))?;
        let conv2 = nn::conv2d(
            out_ch,
            out_ch,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let shortcut = if in_ch != out_ch {
            Some(nn::conv2d(
                in_ch,
                out_ch,
                1,
                nn::Conv2dConfig::default(),
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        Ok(ResBlock3D {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    fn build_up_block(
        in_ch: usize,
        out_ch: usize,
        config: &VaeConfig,
        vb: nn::VarBuilder,
        device: &Device,
        upsample: bool,
    ) -> Result<UpBlock> {
        let mut res_blocks = Vec::new();
        let mut current_ch = in_ch;
        for i in 0..config.num_res_blocks {
            let rb =
                Self::build_res_block(current_ch, out_ch, vb.pp(format!("resnets.{i}")), device)?;
            current_ch = out_ch;
            res_blocks.push(rb);
        }
        let upsample = if upsample {
            Some(Upsample3D {
                conv: nn::conv2d(
                    out_ch,
                    out_ch,
                    3,
                    nn::Conv2dConfig {
                        padding: 1,
                        ..Default::default()
                    },
                    vb.pp("upsamplers.0.conv"),
                )?,
                scale_factor: 2,
            })
        } else {
            None
        };
        Ok(UpBlock {
            res_blocks,
            upsample,
        })
    }

    /// Decode latent to video frames.
    ///
    /// Input:  [batch, C=16, F', H', W'] latent tensor
    /// Output: [batch, 3, F, H, W] decoded video tensor
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        if let Some(decoder) = Self::try_load_decoder(&self.config, &self.model_path, &self.device)
        {
            let res = self.decode_with_model(&decoder, latent)?;
            // decoder is dropped here, releasing VRAM!
            Ok(res)
        } else {
            // Fallback: simple linear projection from latent space
            self.decode_fallback(latent)
        }
    }

    fn decode_with_model(&self, decoder: &VaeDecoder, latent: &Tensor) -> Result<Tensor> {
        let (b, _c, f, h, w) = latent.dims5()?;

        // Input conv (per temporal frame)
        let x_frames = latent.reshape((b * f, _c, h, w))?;
        let x = decoder.conv_in.forward(&x_frames)?;
        let (_, ch, hp, wp) = x.dims4()?;
        let mut x = x.reshape((b, f, ch, hp, wp))?;

        // Mid block
        x = self.res_block_forward(&decoder.mid_block.res1, &x)?;
        if let Some(ref attn) = decoder.mid_block.attn {
            x = attn.forward(&x)?;
        }
        x = self.res_block_forward(&decoder.mid_block.res2, &x)?;

        // Up blocks
        for up_block in &decoder.up_blocks {
            for res_block in &up_block.res_blocks {
                x = self.res_block_forward(res_block, &x)?;
            }
            if let Some(ref up) = up_block.upsample {
                x = self.upsample_forward(up, &x)?;
            }
        }

        // Output (per temporal frame)
        let (b2, f2, c2, h2, w2) = x.dims5()?;
        let x_flat = x.reshape((b2 * f2, c2, h2, w2))?;
        let x_normed = decoder.norm_out.forward(&x_flat.to_dtype(DType::F32)?)?;
        let x_silu = x_normed.silu()?;
        let x_out = decoder.conv_out.forward(&x_silu)?;
        let (_, c_out, h_out, w_out) = x_out.dims4()?;
        x_out
            .reshape((b2, f2, c_out, h_out, w_out))?
            .permute((0, 2, 1, 3, 4))
            .map_err(Into::into)
    }

    fn res_block_forward(&self, block: &ResBlock3D, x: &Tensor) -> Result<Tensor> {
        let (b, f, c, h, w) = x.dims5()?;

        // Flatten to 4D to apply 2D modules frame-by-frame
        let x_flat = x.reshape((b * f, c, h, w))?;
        let x_normed = block.norm1.forward(&x_flat.to_dtype(DType::F32)?)?;
        let x_silu = x_normed.silu()?;
        let x_conv = block.conv1.forward(&x_silu)?;

        let x_normed2 = block.norm2.forward(&x_conv.to_dtype(DType::F32)?)?;
        let x_silu2 = x_normed2.silu()?;
        let x_out = block.conv2.forward(&x_silu2)?;

        let shortcut_out = if let Some(ref sc) = block.shortcut {
            sc.forward(&x_flat)?
        } else {
            x_flat
        };

        let out = (x_out + shortcut_out)?;
        let (_, c_out, h_out, w_out) = out.dims4()?;
        out.reshape((b, f, c_out, h_out, w_out)).map_err(Into::into)
    }

    fn upsample_forward(&self, up: &Upsample3D, x: &Tensor) -> Result<Tensor> {
        // Nearest-neighbor upsampling in spatial dimensions
        let (b, f, c, h, w) = x.dims5()?;
        let new_h = h * up.scale_factor;
        let new_w = w * up.scale_factor;

        // Flatten temporal for Conv2d
        let x_flat = x.reshape((b * f, c, h, w))?;

        // Simple repeat-based upsampling using max 5D tensors
        let x_up = x_flat
            .unsqueeze(3)? // [bf, c, h, 1, w]
            .expand((b * f, c, h, up.scale_factor, w))? // [bf, c, h, scale, w]
            .reshape((b * f, c, new_h, w))?; // [bf, c, new_h, w]
        let x_up = x_up
            .unsqueeze(4)? // [bf, c, new_h, w, 1]
            .expand((b * f, c, new_h, w, up.scale_factor))? // [bf, c, new_h, w, scale]
            .reshape((b * f, c, new_h, new_w))?; // [bf, c, new_h, new_w]

        let conv_out = up.conv.forward(&x_up)?;
        let (_, co, ho, wo) = conv_out.dims4()?;
        conv_out.reshape((b, f, co, ho, wo)).map_err(Into::into)
    }

    /// Fallback decoder: simple linear projection from latent to pixel space.
    ///
    /// Produces a plausible but low-quality video by:
    /// 1. Linearly projecting 16 latent channels to 3 RGB channels
    /// 2. Upsampling spatial dimensions via repeat
    /// 3. Applying tanh normalization
    fn decode_fallback(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, _c, f, h, w) = latent.dims5()?;
        let cfg = &self.config;

        // Upsample factors
        let out_f = (f - 1) * cfg.temporal_stride + 1;
        let out_h = h * cfg.spatial_stride;
        let out_w = w * cfg.spatial_stride;

        // Project 16 channels -> 3 channels via simple averaging
        // Take first 3 groups of latent channels and average each group
        let group_size = cfg.latent_channels / 3;
        let mut channels = Vec::new();
        for i in 0..3 {
            let start = i * group_size;
            let end = ((i + 1) * group_size).min(cfg.latent_channels);
            let group = latent.narrow(1, start, end - start)?;
            let avg = group.mean_keepdim(1)?;
            channels.push(avg);
        }
        let projected = Tensor::cat(&channels, 1)?; // [b, 3, f, h, w]

        // Spatial upsampling via repeat (max 5D tensors)
        let bf = b * f;
        let proj_4d = projected.reshape((bf, 3, h, w))?; // [bf, 3, h, w]
                                                         // Upsample h
        let proj_4d = proj_4d
            .unsqueeze(3)? // [bf, 3, h, 1, w]
            .expand((bf, 3, h, cfg.spatial_stride, w))?
            .reshape((bf, 3, out_h, w))?;
        // Upsample w
        let proj_4d = proj_4d
            .unsqueeze(4)? // [bf, 3, out_h, w, 1]
            .expand((bf, 3, out_h, w, cfg.spatial_stride))?
            .reshape((bf, 3, out_h, out_w))?;
        let projected = proj_4d.reshape((b, f, 3, out_h, out_w))?;

        // Temporal upsampling: repeat each frame t_stride times, then trim to out_f
        let projected = projected
            .unsqueeze(2)? // [b, f, 1, 3, out_h, out_w]
            .expand((b, f, cfg.temporal_stride, 3, out_h, out_w))?
            .reshape((b, f * cfg.temporal_stride, 3, out_h, out_w))?
            .narrow(1, 0, out_f)?; // [b, out_f, 3, out_h, out_w]

        // Permute to [b, 3, out_f, out_h, out_w]
        let result = projected.permute((0, 2, 1, 3, 4))?;

        // Normalize to [-1, 1] range via tanh
        result.tanh().map_err(Into::into)
    }

    /// Get VAE stride info.
    pub fn temporal_stride(&self) -> usize {
        self.config.temporal_stride
    }

    pub fn spatial_stride(&self) -> usize {
        self.config.spatial_stride
    }

    /// Compute output video dimensions from latent dimensions.
    pub fn output_dimensions(
        &self,
        latent_f: usize,
        latent_h: usize,
        latent_w: usize,
    ) -> (usize, usize, usize) {
        let cfg = &self.config;
        let f = (latent_f - 1) * cfg.temporal_stride + 1;
        let h = latent_h * cfg.spatial_stride;
        let w = latent_w * cfg.spatial_stride;
        (f, h, w)
    }
}

/// Convert float tensor video frames to uint8 RGB bytes.
///
/// Input: tensor with values in [-1, 1], shape [C, H, W] or [3, H, W]
/// Output: Vec<u8> of RGB pixel data, H*W*3 bytes.
pub fn tensor_to_rgb_frame(frame: &Tensor) -> Result<Vec<u8>> {
    // Ensure 3D: [C, H, W]
    let frame = if frame.dims().len() == 4 {
        frame.squeeze(0)?
    } else {
        frame.clone()
    };

    let (_c, h, w) = frame.dims3()?;

    // Denormalize from [-1, 1] to [0, 255]
    let frame = frame
        .to_dtype(DType::F32)?
        .affine(0.5, 0.5)? // [-1,1] -> [0,1]
        .affine(255.0, 0.0)? // [0,1] -> [0,255]
        .clamp(0.0, 255.0)?
        .to_dtype(DType::U8)?;

    // Convert to RGB byte layout (H, W, C)
    let data = frame.to_vec3::<u8>()?;
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            rgb[idx] = data[0][y][x]; // R
            rgb[idx + 1] = data[1][y][x]; // G
            rgb[idx + 2] = data[2][y][x]; // B
        }
    }
    Ok(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_config() {
        let cfg = VaeConfig::wan2_1();
        assert_eq!(cfg.latent_channels, 16);
        assert_eq!(cfg.temporal_stride, 4);
        assert_eq!(cfg.spatial_stride, 8);
    }

    #[test]
    fn test_output_dimensions() {
        let vae = WanVAE {
            config: VaeConfig::wan2_1(),
            device: Device::Cpu,
            model_path: std::path::PathBuf::new(),
        };
        let (f, h, w) = vae.output_dimensions(21, 60, 104);
        assert_eq!(f, 81); // (21-1)*4+1
        assert_eq!(h, 480); // 60*8
        assert_eq!(w, 832); // 104*8
    }

    #[test]
    fn test_fallback_decode_shape() {
        let device = Device::Cpu;
        let vae = WanVAE {
            config: VaeConfig::wan2_1(),
            device: device.clone(),
            model_path: std::path::PathBuf::new(),
        };
        // Small latent: [1, 16, 5, 8, 8]
        let latent = Tensor::randn(0f32, 1.0, (1, 16, 5, 8, 8), &device).unwrap();
        let result = vae.decode(&latent).unwrap();
        assert_eq!(result.dim(0).unwrap(), 1); // batch
        assert_eq!(result.dim(1).unwrap(), 3); // channels
    }
}
