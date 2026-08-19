// Tensor-coordinate loops and graph entry points mirror the published model layout.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! Native LongCat image-edit graph planning and metadata validation.
//!
//! This module intentionally stays off the PyTorch/Diffusers path.  It builds
//! the execution plan from local configs, tokenizer/image metadata, and model
//! weight headers before the heavier Candle graph is instantiated.

use std::collections::HashSet;
use std::fmt;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::z_image::{AutoEncoderKL, VaeConfig as CandleVaeConfig};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct LongCatNativeReport {
    pub target_width: usize,
    pub target_height: usize,
    pub latent_width: usize,
    pub latent_height: usize,
    pub latent_channels: usize,
    pub packed_latent_tokens: usize,
    pub packed_latent_width: usize,
    pub packed_latent_height: usize,
    pub prompt_tokens: usize,
    pub prompt_embed_tokens: usize,
    pub qwen_image_grid: (usize, usize, usize),
    pub qwen_image_tokens: usize,
    pub scheduler_shift: f64,
    pub scheduler_sigmas: Vec<f64>,
    pub transformer_tensors: usize,
    pub vae_tensors: usize,
    pub text_encoder_tensors: usize,
    pub transformer_smoke_forward: String,
    pub validated_components: Vec<&'static str>,
}

impl fmt::Display for LongCatNativeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "target={}x{}, latent={}x{}x{}, packed={} tokens ({}x{}), prompt_tokens={}, prompt_embed_tokens={}, qwen_grid={:?}, qwen_image_tokens={}, scheduler_shift={:.4}, sigmas={}, tensors(transformer/vae/text)={}/{}/{}, transformer_smoke={}, components={}",
            self.target_width,
            self.target_height,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            self.packed_latent_tokens,
            self.packed_latent_height,
            self.packed_latent_width,
            self.prompt_tokens,
            self.prompt_embed_tokens,
            self.qwen_image_grid,
            self.qwen_image_tokens,
            self.scheduler_shift,
            self.scheduler_sigmas.len(),
            self.transformer_tensors,
            self.vae_tensors,
            self.text_encoder_tensors,
            self.transformer_smoke_forward,
            self.validated_components.join(", ")
        )
    }
}

#[derive(Debug, serde::Deserialize)]
struct TransformerConfig {
    attention_head_dim: usize,
    axes_dims_rope: Option<Vec<usize>>,
    in_channels: usize,
    joint_attention_dim: usize,
    num_attention_heads: usize,
    num_layers: usize,
    num_single_layers: usize,
    patch_size: usize,
}

#[derive(Debug, serde::Deserialize)]
struct VaeConfig {
    block_out_channels: Vec<usize>,
    latent_channels: usize,
    scaling_factor: f64,
    shift_factor: f64,
}

#[derive(Debug, serde::Deserialize)]
struct SchedulerConfig {
    base_image_seq_len: usize,
    base_shift: f64,
    max_image_seq_len: usize,
    max_shift: f64,
    num_train_timesteps: u32,
}

#[derive(Debug, serde::Deserialize)]
struct TextEncoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rope_theta: f64,
    rope_scaling: RopeScalingConfig,
    image_token_id: u32,
    vision_config: VisionConfig,
}

#[derive(Debug, serde::Deserialize)]
struct RopeScalingConfig {
    mrope_section: Vec<usize>,
}

#[derive(Debug, serde::Deserialize)]
struct VisionConfig {
    depth: usize,
    #[serde(alias = "in_chans", default = "default_rgb_channels")]
    in_channels: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    out_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
}

#[derive(Debug, serde::Deserialize)]
struct ProcessorConfig {
    min_pixels: usize,
    max_pixels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

#[derive(Debug, Clone)]
struct TensorHeaders {
    names: HashSet<String>,
    count: usize,
}

pub fn preview_graph(
    model_path: &Path,
    prompt: &str,
    image_bytes: &[u8],
    num_steps: u32,
    device: &Device,
) -> Result<LongCatNativeReport> {
    if matches!(device, Device::Cpu) {
        bail!("LongCat native graph requires a GPU Candle device");
    }

    require_full_package(model_path)?;

    let transformer_cfg: TransformerConfig =
        read_json(&model_path.join("transformer/config.json"))?;
    let vae_cfg: VaeConfig = read_json(&model_path.join("vae/config.json"))?;
    let scheduler_cfg: SchedulerConfig =
        read_json(&model_path.join("scheduler/scheduler_config.json"))?;
    let text_cfg: TextEncoderConfig = read_json(&model_path.join("text_encoder/config.json"))?;
    let processor_cfg: ProcessorConfig =
        read_json(&model_path.join("text_processor/preprocessor_config.json"))?;

    validate_config_compatibility(&transformer_cfg, &vae_cfg, &text_cfg, &processor_cfg)?;

    let (source_width, source_height) = image_dimensions(image_bytes)?;
    let (target_width, target_height) =
        calculate_dimensions(1024 * 1024, source_width as f64 / source_height as f64);

    let vae_scale_factor = 2usize.pow(vae_cfg.block_out_channels.len().saturating_sub(1) as u32);
    let latent_height = 2 * (target_height / (vae_scale_factor * 2));
    let latent_width = 2 * (target_width / (vae_scale_factor * 2));
    let packed_latent_height = latent_height / 2;
    let packed_latent_width = latent_width / 2;
    let packed_latent_tokens = packed_latent_height * packed_latent_width;
    let packed_channels = vae_cfg.latent_channels * 4;

    if packed_channels != transformer_cfg.in_channels {
        bail!(
            "LongCat latent pack channels {} do not match transformer in_channels {}",
            packed_channels,
            transformer_cfg.in_channels
        );
    }

    let prompt_tokens = count_prompt_tokens(model_path, prompt)?;
    let (qwen_grid, qwen_image_tokens) =
        qwen_image_grid(source_width, source_height, &processor_cfg);
    let prompt_embed_tokens =
        estimate_prompt_embed_tokens(model_path, qwen_image_tokens, prompt_tokens)?;

    let shift = calculate_shift(
        packed_latent_tokens,
        scheduler_cfg.base_image_seq_len,
        scheduler_cfg.max_image_seq_len,
        scheduler_cfg.base_shift,
        scheduler_cfg.max_shift,
    );
    let scheduler_sigmas = flow_match_sigmas(num_steps, shift);
    if scheduler_cfg.num_train_timesteps == 0 {
        bail!("LongCat scheduler num_train_timesteps must be positive");
    }

    let transformer_headers =
        safetensor_headers(&[model_path.join("transformer/diffusion_pytorch_model.safetensors")])?;
    let vae_headers =
        safetensor_headers(&[model_path.join("vae/diffusion_pytorch_model.safetensors")])?;
    let text_headers = safetensor_headers(&text_encoder_weight_files(model_path))?;

    validate_transformer_headers(&transformer_headers, &transformer_cfg)?;
    validate_vae_headers(&vae_headers)?;
    validate_text_encoder_headers(&text_headers, &text_cfg)?;
    let transformer_smoke_forward =
        smoke_forward_transformer(model_path, &transformer_cfg, device)?;

    Ok(LongCatNativeReport {
        target_width,
        target_height,
        latent_width,
        latent_height,
        latent_channels: vae_cfg.latent_channels,
        packed_latent_tokens,
        packed_latent_width,
        packed_latent_height,
        prompt_tokens,
        prompt_embed_tokens,
        qwen_image_grid: qwen_grid,
        qwen_image_tokens,
        scheduler_shift: shift,
        scheduler_sigmas,
        transformer_tensors: transformer_headers.count,
        vae_tensors: vae_headers.count,
        text_encoder_tensors: text_headers.count,
        transformer_smoke_forward,
        validated_components: vec![
            "tokenizer",
            "qwen2.5-vl config",
            "vae config",
            "transformer config",
            "flow-match scheduler",
            "weight metadata",
            "full transformer forward",
            "latent packing",
            "position id plan",
        ],
    })
}

pub fn run_draft_image_edit(
    model_path: &Path,
    prompt: &str,
    image_bytes: &[u8],
    num_steps: u32,
    device: &Device,
) -> Result<Vec<u8>> {
    if matches!(device, Device::Cpu) {
        bail!("LongCat native image edit requires a GPU Candle device");
    }

    let report = preview_graph(model_path, prompt, image_bytes, num_steps, device)?;
    tracing::info!("LongCat native draft graph: {report}");

    let transformer_cfg: TransformerConfig =
        read_json(&model_path.join("transformer/config.json"))?;
    let vae_cfg: VaeConfig = read_json(&model_path.join("vae/config.json"))?;
    let (source_width, source_height) = image_dimensions(image_bytes)?;
    let (target_width, target_height) =
        draft_dimensions(source_width, source_height, native_draft_max_side());
    let image = preprocess_image_tensor(image_bytes, target_width, target_height, device)?;

    let dtype = DType::F32;
    let vae = load_vae(model_path, &vae_cfg, device, dtype)?;
    let latent = vae.encode(&image)?;
    let (b, c, latent_h, latent_w) = latent.dims4()?;
    if b != 1 || c != vae_cfg.latent_channels {
        bail!("unexpected LongCat VAE latent shape {:?}", latent.dims());
    }

    let packed = pack_latents(&latent)?;
    let text_cfg: TextEncoderConfig = read_json(&model_path.join("text_encoder/config.json"))?;
    let processor_cfg: ProcessorConfig =
        read_json(&model_path.join("text_processor/preprocessor_config.json"))?;
    let context = prompt_context_tensor(
        model_path,
        prompt,
        image_bytes,
        &text_cfg,
        &processor_cfg,
        device,
        dtype,
    )?;
    let edited_packed =
        forward_complete_transformer(model_path, &transformer_cfg, &packed, &context, 0.5, device)?;
    let raw_delta = unpack_latents(&edited_packed, vae_cfg.latent_channels, latent_h, latent_w)?;
    let delta = normalize_draft_delta(&latent, &raw_delta)?;
    let edited_latent = (latent + (delta * native_draft_strength())?)?;
    let decoded = vae.decode(&edited_latent)?;
    encode_png_tensor(&decoded)
}

fn require_full_package(model_path: &Path) -> Result<()> {
    for rel in [
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_processor/tokenizer.json",
        "text_processor/preprocessor_config.json",
        "scheduler/scheduler_config.json",
    ] {
        let path = model_path.join(rel);
        if !path.exists() {
            bail!(
                "LongCat native graph needs the complete HF package; missing {}. GGUF-only packages provide transformer weights but not VAE/text-encoder components",
                rel
            );
        }
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn validate_config_compatibility(
    transformer: &TransformerConfig,
    vae: &VaeConfig,
    text: &TextEncoderConfig,
    processor: &ProcessorConfig,
) -> Result<()> {
    if transformer.patch_size != 1 {
        bail!(
            "LongCat transformer patch_size={} is not supported yet",
            transformer.patch_size
        );
    }
    let inner_dim = transformer.num_attention_heads * transformer.attention_head_dim;
    if inner_dim != 3072 {
        bail!("unexpected LongCat transformer inner_dim {inner_dim}; expected 3072");
    }
    if transformer.joint_attention_dim != text.hidden_size {
        bail!(
            "transformer joint_attention_dim {} does not match text hidden_size {}",
            transformer.joint_attention_dim,
            text.hidden_size
        );
    }
    if !text.hidden_size.is_multiple_of(text.num_attention_heads) {
        bail!("Qwen2.5-VL hidden_size must be divisible by attention heads");
    }
    if !text
        .num_attention_heads
        .is_multiple_of(text.num_key_value_heads)
    {
        bail!("Qwen2.5-VL attention heads must be divisible by KV heads");
    }
    if text.rope_scaling.mrope_section.iter().sum::<usize>() * 2
        != text.hidden_size / text.num_attention_heads
    {
        bail!("Qwen2.5-VL mrope_section does not match attention head dim");
    }
    if !text
        .vision_config
        .hidden_size
        .is_multiple_of(text.vision_config.num_heads)
    {
        bail!("Qwen2.5-VL vision hidden_size must be divisible by vision heads");
    }
    if vae.latent_channels != 16 {
        bail!(
            "LongCat VAE latent_channels={} is not supported yet",
            vae.latent_channels
        );
    }
    if !vae.scaling_factor.is_finite() || !vae.shift_factor.is_finite() {
        bail!("LongCat VAE scaling/shift factors must be finite");
    }
    if processor.patch_size != text.vision_config.patch_size
        || processor.temporal_patch_size != text.vision_config.temporal_patch_size
        || processor.merge_size != text.vision_config.spatial_merge_size
    {
        bail!("Qwen2.5-VL processor config does not match text_encoder vision_config");
    }
    Ok(())
}

fn default_rgb_channels() -> usize {
    3
}

fn image_dimensions(image_bytes: &[u8]) -> Result<(usize, usize)> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| anyhow!("failed to decode LongCat input image: {e}"))?;
    Ok((img.width() as usize, img.height() as usize))
}

fn calculate_dimensions(target_area: usize, ratio: f64) -> (usize, usize) {
    let width = (target_area as f64 * ratio).sqrt();
    let height = width / ratio;
    (
        round_up_to(width as usize, 16),
        round_up_to(height as usize, 16),
    )
}

fn round_up_to(value: usize, factor: usize) -> usize {
    if value.is_multiple_of(factor) {
        value
    } else {
        (value / factor + 1) * factor
    }
}

fn count_prompt_tokens(model_path: &Path, prompt: &str) -> Result<usize> {
    let tokenizer_path = model_path.join("text_processor/tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer {}: {e}", tokenizer_path.display()))?;
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow!("failed to tokenize LongCat prompt: {e}"))?;
    Ok(encoding.get_ids().len().min(512))
}

fn qwen_image_grid(
    source_width: usize,
    source_height: usize,
    cfg: &ProcessorConfig,
) -> ((usize, usize, usize), usize) {
    let factor = cfg.patch_size * cfg.merge_size;
    let (resized_height, resized_width) = smart_resize(
        source_height,
        source_width,
        factor,
        cfg.min_pixels,
        cfg.max_pixels,
    );
    let grid_t = 1usize;
    let grid_h = resized_height / cfg.patch_size;
    let grid_w = resized_width / cfg.patch_size;
    let image_tokens = grid_t * grid_h * grid_w / (cfg.merge_size * cfg.merge_size);
    ((grid_t, grid_h, grid_w), image_tokens)
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let mut h_bar = round_to_factor(height, factor).max(factor);
    let mut w_bar = round_to_factor(width, factor).max(factor);

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = floor_to_factor((height as f64 / beta) as usize, factor).max(factor);
        w_bar = floor_to_factor((width as f64 / beta) as usize, factor).max(factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = round_up_to((height as f64 * beta) as usize, factor).max(factor);
        w_bar = round_up_to((width as f64 * beta) as usize, factor).max(factor);
    }

    (h_bar, w_bar)
}

fn round_to_factor(value: usize, factor: usize) -> usize {
    (((value as f64 / factor as f64).round() as usize) * factor).max(factor)
}

fn floor_to_factor(value: usize, factor: usize) -> usize {
    (value / factor) * factor
}

fn estimate_prompt_embed_tokens(
    model_path: &Path,
    image_tokens: usize,
    prompt_tokens: usize,
) -> Result<usize> {
    let tokenizer_path = model_path.join("text_processor/tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer {}: {e}", tokenizer_path.display()))?;
    let prefix = format!(
        "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|>{}<|vision_end|>",
        "<|image_pad|>".repeat(image_tokens)
    );
    let prefix_ids = tokenizer
        .encode(prefix, false)
        .map_err(|e| anyhow!("failed to tokenize LongCat prompt prefix: {e}"))?;
    let vision_start_id = tokenizer
        .token_to_id("<|vision_start|>")
        .ok_or_else(|| anyhow!("tokenizer does not define <|vision_start|>"))?;
    let prefix_len = prefix_ids
        .get_ids()
        .iter()
        .position(|id| *id == vision_start_id)
        .ok_or_else(|| anyhow!("LongCat prompt prefix did not include <|vision_start|>"))?;
    let padded_text_len = prompt_tokens.max(512);
    Ok(prefix_ids.get_ids().len() - prefix_len + padded_text_len)
}

fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

fn flow_match_sigmas(num_steps: u32, shift: f64) -> Vec<f64> {
    (0..=num_steps)
        .map(|i| {
            let t = 1.0 - (i as f64) / (num_steps as f64);
            shift * t / (1.0 + (shift - 1.0) * t)
        })
        .collect()
}

fn text_encoder_weight_files(model_path: &Path) -> Vec<PathBuf> {
    (1..=5)
        .map(|idx| model_path.join(format!("text_encoder/model-{idx:05}-of-00005.safetensors")))
        .collect()
}

fn safetensor_headers(paths: &[PathBuf]) -> Result<TensorHeaders> {
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };
    let views = tensors.tensors();
    Ok(TensorHeaders {
        names: views.iter().map(|(name, _)| name.clone()).collect(),
        count: views.len(),
    })
}

fn validate_transformer_headers(headers: &TensorHeaders, cfg: &TransformerConfig) -> Result<()> {
    let last_joint = cfg.num_layers.saturating_sub(1);
    let last_single = cfg.num_single_layers.saturating_sub(1);
    require_tensors(
        headers,
        &[
            "x_embedder.weight",
            "x_embedder.bias",
            "context_embedder.weight",
            "context_embedder.bias",
            "norm_out.linear.weight",
            "norm_out.linear.bias",
            "proj_out.weight",
            "proj_out.bias",
            "transformer_blocks.0.norm1.linear.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
            "single_transformer_blocks.0.norm.linear.weight",
            "single_transformer_blocks.0.attn.to_q.weight",
            "single_transformer_blocks.0.proj_out.weight",
        ],
        "LongCat transformer",
    )?;
    require_tensors_owned(
        headers,
        vec![
            format!("transformer_blocks.{last_joint}.attn.to_out.0.weight"),
            format!("transformer_blocks.{last_joint}.ff_context.net.2.weight"),
            format!("single_transformer_blocks.{last_single}.attn.to_v.weight"),
            format!("single_transformer_blocks.{last_single}.proj_mlp.weight"),
        ],
        "LongCat transformer",
    )
}

fn validate_vae_headers(headers: &TensorHeaders) -> Result<()> {
    require_tensors(
        headers,
        &[
            "encoder.conv_in.weight",
            "decoder.conv_in.weight",
            "decoder.mid_block.attentions.0.to_q.weight",
            "decoder.up_blocks.0.resnets.0.conv1.weight",
            "decoder.conv_out.weight",
        ],
        "LongCat VAE",
    )
}

fn validate_text_encoder_headers(headers: &TensorHeaders, cfg: &TextEncoderConfig) -> Result<()> {
    let last_layer = cfg.num_hidden_layers.saturating_sub(1);
    require_tensors(
        headers,
        &[
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "visual.patch_embed.proj.weight",
            "visual.blocks.0.attn.qkv.weight",
            "visual.merger.mlp.0.weight",
        ],
        "LongCat Qwen2.5-VL text encoder",
    )?;
    require_tensors_owned(
        headers,
        vec![
            format!("model.layers.{last_layer}.self_attn.o_proj.weight"),
            format!("model.layers.{last_layer}.mlp.down_proj.weight"),
        ],
        "LongCat Qwen2.5-VL text encoder",
    )
}

fn require_tensors(headers: &TensorHeaders, names: &[&str], component: &str) -> Result<()> {
    for name in names {
        if !headers.names.contains(*name) {
            bail!("{component} is missing tensor {name}");
        }
    }
    Ok(())
}

fn require_tensors_owned(
    headers: &TensorHeaders,
    names: Vec<String>,
    component: &str,
) -> Result<()> {
    for name in names {
        if !headers.names.contains(&name) {
            bail!("{component} is missing tensor {name}");
        }
    }
    Ok(())
}

struct LongCatTimeEmbed {
    linear_1: candle_nn::Linear,
    linear_2: candle_nn::Linear,
}

impl LongCatTimeEmbed {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: candle_nn::linear(256, 3072, vb.pp("timestep_embedder.linear_1"))?,
            linear_2: candle_nn::linear(3072, 3072, vb.pp("timestep_embedder.linear_2"))?,
        })
    }

    fn forward(&self, timestep: f32, device: &Device, dtype: DType) -> Result<Tensor> {
        let emb = timestep_embedding(timestep, 256, device)?.to_dtype(dtype)?;
        let emb = self.linear_1.forward(&emb)?;
        let emb = emb.apply(&candle_nn::Activation::Silu)?;
        Ok(self.linear_2.forward(&emb)?)
    }
}

struct LongCatAdaLayerNormZero {
    linear: candle_nn::Linear,
}

impl LongCatAdaLayerNormZero {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(3072, 3072 * 6, vb.pp("linear"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let emb = self
            .linear
            .forward(&emb.apply(&candle_nn::Activation::Silu)?)?;
        let dim = emb.dims2()?.1 / 6;
        let shift_msa = emb.narrow(1, 0, dim)?;
        let scale_msa = emb.narrow(1, dim, dim)?;
        let gate_msa = emb.narrow(1, dim * 2, dim)?;
        let shift_mlp = emb.narrow(1, dim * 3, dim)?;
        let scale_mlp = emb.narrow(1, dim * 4, dim)?;
        let gate_mlp = emb.narrow(1, dim * 5, dim)?;
        let hidden = modulate(&layer_norm_no_affine(hidden, 1e-6)?, &shift_msa, &scale_msa)?;
        Ok((hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

struct LongCatAdaLayerNormZeroSingle {
    linear: candle_nn::Linear,
}

impl LongCatAdaLayerNormZeroSingle {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(3072, 3072 * 3, vb.pp("linear"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, emb: &Tensor) -> Result<(Tensor, Tensor)> {
        let emb = self
            .linear
            .forward(&emb.apply(&candle_nn::Activation::Silu)?)?;
        let dim = emb.dims2()?.1 / 3;
        let shift = emb.narrow(1, 0, dim)?;
        let scale = emb.narrow(1, dim, dim)?;
        let gate = emb.narrow(1, dim * 2, dim)?;
        let hidden = modulate(&layer_norm_no_affine(hidden, 1e-6)?, &shift, &scale)?;
        Ok((hidden, gate))
    }
}

struct LongCatAdaLayerNormContinuous {
    linear: candle_nn::Linear,
}

impl LongCatAdaLayerNormContinuous {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(3072, 3072 * 2, vb.pp("linear"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let emb = self
            .linear
            .forward(&emb.apply(&candle_nn::Activation::Silu)?)?;
        let dim = emb.dims2()?.1 / 2;
        let scale = emb.narrow(1, 0, dim)?;
        let shift = emb.narrow(1, dim, dim)?;
        modulate(&layer_norm_no_affine(hidden, 1e-6)?, &shift, &scale)
    }
}

struct LongCatRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl LongCatRmsNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let inv = (x_f32.sqr()?.mean_keepdim(D::Minus1)?.affine(1.0, self.eps))?
            .sqrt()?
            .recip()?;
        x_f32
            .broadcast_mul(&inv)?
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)
            .map_err(Into::into)
    }
}

struct LongCatFeedForward {
    proj_in: candle_nn::Linear,
    proj_out: candle_nn::Linear,
}

impl LongCatFeedForward {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_in: candle_nn::linear(3072, 12288, vb.pp("net.0.proj"))?,
            proj_out: candle_nn::linear(12288, 3072, vb.pp("net.2"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.proj_in.forward(hidden)?;
        let hidden = hidden.apply(&candle_nn::Activation::GeluPytorchTanh)?;
        Ok(self.proj_out.forward(&hidden)?)
    }
}

struct LongCatAttention {
    heads: usize,
    head_dim: usize,
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    add_q: Option<candle_nn::Linear>,
    add_k: Option<candle_nn::Linear>,
    add_v: Option<candle_nn::Linear>,
    to_out: Option<candle_nn::Linear>,
    to_add_out: Option<candle_nn::Linear>,
    norm_q: LongCatRmsNorm,
    norm_k: LongCatRmsNorm,
    norm_added_q: Option<LongCatRmsNorm>,
    norm_added_k: Option<LongCatRmsNorm>,
}

impl LongCatAttention {
    fn new_joint(heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            heads,
            head_dim,
            to_q: candle_nn::linear(3072, 3072, vb.pp("to_q"))?,
            to_k: candle_nn::linear(3072, 3072, vb.pp("to_k"))?,
            to_v: candle_nn::linear(3072, 3072, vb.pp("to_v"))?,
            add_q: Some(candle_nn::linear(3072, 3072, vb.pp("add_q_proj"))?),
            add_k: Some(candle_nn::linear(3072, 3072, vb.pp("add_k_proj"))?),
            add_v: Some(candle_nn::linear(3072, 3072, vb.pp("add_v_proj"))?),
            to_out: Some(candle_nn::linear(3072, 3072, vb.pp("to_out.0"))?),
            to_add_out: Some(candle_nn::linear(3072, 3072, vb.pp("to_add_out"))?),
            norm_q: LongCatRmsNorm::new(head_dim, vb.pp("norm_q"))?,
            norm_k: LongCatRmsNorm::new(head_dim, vb.pp("norm_k"))?,
            norm_added_q: Some(LongCatRmsNorm::new(head_dim, vb.pp("norm_added_q"))?),
            norm_added_k: Some(LongCatRmsNorm::new(head_dim, vb.pp("norm_added_k"))?),
        })
    }

    fn new_single(heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            heads,
            head_dim,
            to_q: candle_nn::linear(3072, 3072, vb.pp("to_q"))?,
            to_k: candle_nn::linear(3072, 3072, vb.pp("to_k"))?,
            to_v: candle_nn::linear(3072, 3072, vb.pp("to_v"))?,
            add_q: None,
            add_k: None,
            add_v: None,
            to_out: None,
            to_add_out: None,
            norm_q: LongCatRmsNorm::new(head_dim, vb.pp("norm_q"))?,
            norm_k: LongCatRmsNorm::new(head_dim, vb.pp("norm_k"))?,
            norm_added_q: None,
            norm_added_k: None,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (b, image_tokens, _) = hidden.dims3()?;
        let q = project_heads(
            &self.to_q,
            hidden,
            b,
            image_tokens,
            self.heads,
            self.head_dim,
        )?;
        let k = project_heads(
            &self.to_k,
            hidden,
            b,
            image_tokens,
            self.heads,
            self.head_dim,
        )?;
        let v = project_heads(
            &self.to_v,
            hidden,
            b,
            image_tokens,
            self.heads,
            self.head_dim,
        )?;
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        let (q, k, v, text_tokens) = if let Some(encoder) = encoder {
            let (b2, text_tokens, _) = encoder.dims3()?;
            if b2 != b {
                bail!("LongCat attention batch mismatch");
            }
            let add_q = project_heads(
                self.add_q.as_ref().context("missing joint add_q")?,
                encoder,
                b,
                text_tokens,
                self.heads,
                self.head_dim,
            )?;
            let add_k = project_heads(
                self.add_k.as_ref().context("missing joint add_k")?,
                encoder,
                b,
                text_tokens,
                self.heads,
                self.head_dim,
            )?;
            let add_v = project_heads(
                self.add_v.as_ref().context("missing joint add_v")?,
                encoder,
                b,
                text_tokens,
                self.heads,
                self.head_dim,
            )?;
            let add_q = self
                .norm_added_q
                .as_ref()
                .context("missing joint norm_added_q")?
                .forward(&add_q)?;
            let add_k = self
                .norm_added_k
                .as_ref()
                .context("missing joint norm_added_k")?
                .forward(&add_k)?;
            (
                Tensor::cat(&[&add_q, &q], 1)?,
                Tensor::cat(&[&add_k, &k], 1)?,
                Tensor::cat(&[&add_v, &v], 1)?,
                text_tokens,
            )
        } else {
            (q, k, v, 0)
        };

        let q = apply_rotary(&q, cos, sin)?;
        let k = apply_rotary(&k, cos, sin)?;
        let q = q.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;
        let scale = (self.head_dim as f64).sqrt().recip();
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?;
        let (_, _, total_tokens, _) = out.dims4()?;
        let out =
            out.permute((0, 2, 1, 3))?
                .reshape((b, total_tokens, self.heads * self.head_dim))?;

        if encoder.is_some() {
            let enc = out.narrow(1, 0, text_tokens)?;
            let img = out.narrow(1, text_tokens, image_tokens)?;
            let img = self
                .to_out
                .as_ref()
                .context("missing joint to_out")?
                .forward(&img)?;
            let enc = self
                .to_add_out
                .as_ref()
                .context("missing joint to_add_out")?
                .forward(&enc)?;
            Ok((img, Some(enc)))
        } else {
            Ok((out, None))
        }
    }
}

struct LongCatJointBlock {
    norm1: LongCatAdaLayerNormZero,
    norm1_context: LongCatAdaLayerNormZero,
    attn: LongCatAttention,
    ff: LongCatFeedForward,
    ff_context: LongCatFeedForward,
}

impl LongCatJointBlock {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: LongCatAdaLayerNormZero::new(vb.pp("norm1"))?,
            norm1_context: LongCatAdaLayerNormZero::new(vb.pp("norm1_context"))?,
            attn: LongCatAttention::new_joint(
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                vb.pp("attn"),
            )?,
            ff: LongCatFeedForward::new(vb.pp("ff"))?,
            ff_context: LongCatFeedForward::new(vb.pp("ff_context"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        emb: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.norm1.forward(hidden, emb)?;
        let (norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
            self.norm1_context.forward(encoder, emb)?;
        let (attn_hidden, attn_encoder) =
            self.attn
                .forward(&norm_hidden, Some(&norm_encoder), cos, sin)?;
        let hidden = hidden.broadcast_add(&gate(&attn_hidden, &gate_msa)?)?;
        let encoder = encoder.broadcast_add(&gate(
            &attn_encoder.context("joint attention did not return encoder states")?,
            &c_gate_msa,
        )?)?;
        let ff_hidden = modulate(
            &layer_norm_no_affine(&hidden, 1e-6)?,
            &shift_mlp,
            &scale_mlp,
        )?;
        let ff_encoder = modulate(
            &layer_norm_no_affine(&encoder, 1e-6)?,
            &c_shift_mlp,
            &c_scale_mlp,
        )?;
        let hidden = hidden.broadcast_add(&gate(&self.ff.forward(&ff_hidden)?, &gate_mlp)?)?;
        let encoder =
            encoder.broadcast_add(&gate(&self.ff_context.forward(&ff_encoder)?, &c_gate_mlp)?)?;
        Ok((hidden, encoder))
    }
}

struct LongCatSingleBlock {
    norm: LongCatAdaLayerNormZeroSingle,
    proj_mlp: candle_nn::Linear,
    attn: LongCatAttention,
    proj_out: candle_nn::Linear,
}

impl LongCatSingleBlock {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: LongCatAdaLayerNormZeroSingle::new(vb.pp("norm"))?,
            proj_mlp: candle_nn::linear(3072, 12288, vb.pp("proj_mlp"))?,
            attn: LongCatAttention::new_single(
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                vb.pp("attn"),
            )?,
            proj_out: candle_nn::linear(15360, 3072, vb.pp("proj_out"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, emb: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (norm_hidden, gate_value) = self.norm.forward(hidden, emb)?;
        let mlp = self
            .proj_mlp
            .forward(&norm_hidden)?
            .apply(&candle_nn::Activation::GeluPytorchTanh)?;
        let (attn, _) = self.attn.forward(&norm_hidden, None, cos, sin)?;
        let output = Tensor::cat(&[&attn, &mlp], 2)?;
        Ok(hidden.broadcast_add(&gate(&self.proj_out.forward(&output)?, &gate_value)?)?)
    }
}

fn smoke_forward_transformer(
    model_path: &Path,
    cfg: &TransformerConfig,
    device: &Device,
) -> Result<String> {
    let dtype = match device {
        Device::Cpu => bail!("LongCat transformer smoke forward refuses CPU device"),
        _ => DType::F16,
    };
    let batch = 1usize;
    let text_tokens = 3usize;
    let image_tokens = 4usize;
    let latents = Tensor::zeros((batch, image_tokens, cfg.in_channels), dtype, device)?;
    let prompt = Tensor::zeros((batch, text_tokens, cfg.joint_attention_dim), dtype, device)?;
    let out = forward_complete_transformer(model_path, cfg, &latents, &prompt, 0.5, device)?;
    Ok(format!(
        "metal/candle full-transformer out={:?}",
        out.dims()
    ))
}

fn forward_complete_transformer(
    model_path: &Path,
    cfg: &TransformerConfig,
    packed_latents: &Tensor,
    prompt_embeds: &Tensor,
    timestep: f32,
    device: &Device,
) -> Result<Tensor> {
    let dtype = packed_latents.dtype();
    let (_, image_tokens, in_channels) = packed_latents.dims3()?;
    let (_, text_tokens, joint_dim) = prompt_embeds.dims3()?;
    if in_channels != cfg.in_channels || joint_dim != cfg.joint_attention_dim {
        bail!("LongCat transformer input shape mismatch");
    }

    let path = model_path.join("transformer/diffusion_pytorch_model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    let x_embedder = candle_nn::linear(cfg.in_channels, 3072, vb.pp("x_embedder"))?;
    let context_embedder =
        candle_nn::linear(cfg.joint_attention_dim, 3072, vb.pp("context_embedder"))?;
    let time_embed = LongCatTimeEmbed::new(vb.pp("time_embed"))?;
    let norm_out = LongCatAdaLayerNormContinuous::new(vb.pp("norm_out"))?;
    let proj_out = candle_nn::linear(3072, cfg.in_channels, vb.pp("proj_out"))?;

    let mut hidden = x_embedder.forward(packed_latents)?;
    let mut encoder = context_embedder.forward(prompt_embeds)?;
    let emb = time_embed.forward(timestep, device, dtype)?;
    let (cos, sin) = rotary_embeddings(text_tokens, image_tokens, cfg, device, dtype)?;

    for layer_idx in 0..cfg.num_layers {
        let block = LongCatJointBlock::new(cfg, vb.pp("transformer_blocks").pp(layer_idx))?;
        (hidden, encoder) = block.forward(&hidden, &encoder, &emb, &cos, &sin)?;
    }

    let mut combined = Tensor::cat(&[&encoder, &hidden], 1)?;
    for layer_idx in 0..cfg.num_single_layers {
        let block = LongCatSingleBlock::new(cfg, vb.pp("single_transformer_blocks").pp(layer_idx))?;
        combined = block.forward(&combined, &emb, &cos, &sin)?;
    }

    let hidden = combined.narrow(1, text_tokens, image_tokens)?;
    let hidden = norm_out.forward(&hidden, &emb)?;
    proj_out.forward(&hidden).map_err(Into::into)
}

fn load_vae(
    model_path: &Path,
    cfg: &VaeConfig,
    device: &Device,
    dtype: DType,
) -> Result<AutoEncoderKL> {
    let vae_weights = model_path.join("vae/diffusion_pytorch_model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_weights], dtype, device)? };
    let candle_cfg = CandleVaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: cfg.latent_channels,
        block_out_channels: cfg.block_out_channels.clone(),
        layers_per_block: 2,
        scaling_factor: cfg.scaling_factor,
        shift_factor: cfg.shift_factor,
        norm_num_groups: 32,
    };
    AutoEncoderKL::new(&candle_cfg, vb).map_err(Into::into)
}

fn preprocess_image_tensor(
    image_bytes: &[u8],
    width: usize,
    height: usize,
    device: &Device,
) -> Result<Tensor> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| anyhow!("failed to decode LongCat input image: {e}"))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &img,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut values = vec![0f32; 3 * width * height];
    for y in 0..height {
        for x in 0..width {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let offset = y * width + x;
            values[offset] = pixel[0] as f32 / 127.5 - 1.0;
            values[width * height + offset] = pixel[1] as f32 / 127.5 - 1.0;
            values[2 * width * height + offset] = pixel[2] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_vec(values, (1, 3, height, width), device)?.to_dtype(DType::F32)?)
}

fn encode_png_tensor(image: &Tensor) -> Result<Vec<u8>> {
    let image = image.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let (b, c, h, w) = image.dims4()?;
    if b != 1 || c != 3 {
        bail!(
            "expected decoded LongCat image tensor [1,3,H,W], got {:?}",
            image.dims()
        );
    }
    let values = image.squeeze(0)?.to_vec3::<f32>()?;
    let mut out = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let r = tensor_pixel_to_u8(values[0][y][x]);
            let g = tensor_pixel_to_u8(values[1][y][x]);
            let b = tensor_pixel_to_u8(values[2][y][x]);
            out.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(out)
        .write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)
        .context("failed to encode LongCat native output PNG")?;
    Ok(bytes)
}

fn tensor_pixel_to_u8(value: f32) -> u8 {
    (((value + 1.0) * 127.5).round().clamp(0.0, 255.0)) as u8
}

fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = latents.dims4()?;
    if h % 2 != 0 || w % 2 != 0 {
        bail!("LongCat latent dimensions must be even for packing");
    }
    Ok(latents
        .reshape((b, c, h / 2, 2, w / 2, 2))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((b, (h / 2) * (w / 2), c * 4))?)
}

fn unpack_latents(packed: &Tensor, channels: usize, height: usize, width: usize) -> Result<Tensor> {
    let (b, tokens, packed_channels) = packed.dims3()?;
    if packed_channels != channels * 4 || tokens != (height / 2) * (width / 2) {
        bail!("LongCat packed latent shape does not match requested unpack shape");
    }
    Ok(packed
        .reshape((b, height / 2, width / 2, channels, 2, 2))?
        .permute((0, 3, 1, 4, 2, 5))?
        .reshape((b, channels, height, width))?)
}

fn normalize_draft_delta(latent: &Tensor, delta: &Tensor) -> Result<Tensor> {
    let latent_rms = tensor_rms(latent)?;
    let delta_rms = tensor_rms(delta)?;
    let scale = if latent_rms.is_finite() && delta_rms.is_finite() && delta_rms > 1e-6 {
        (latent_rms / delta_rms).min(native_draft_delta_rms_ratio())
    } else {
        0.0
    };
    tracing::info!(
        "LongCat native draft delta RMS guard: latent_rms={latent_rms:.6}, delta_rms={delta_rms:.6}, scale={scale:.6}"
    );
    (delta * scale as f64).map_err(Into::into)
}

fn tensor_rms(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?
        .sqrt())
}

struct QwenVisionAttention {
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
    heads: usize,
    head_dim: usize,
}

impl QwenVisionAttention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: candle_nn::linear(cfg.hidden_size, cfg.hidden_size * 3, vb.pp("attn.qkv"))?,
            proj: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn.proj"))?,
            heads: cfg.num_heads,
            head_dim: cfg.hidden_size / cfg.num_heads,
        })
    }

    fn forward(&self, hidden: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (tokens, hidden_size) = hidden.dims2()?;
        let qkv = self
            .qkv
            .forward(hidden)?
            .reshape((tokens, 3, self.heads, self.head_dim))?;
        let q = qkv.narrow(1, 0, 1)?.squeeze(1)?;
        let k = qkv.narrow(1, 1, 1)?.squeeze(1)?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;
        let q = apply_qwen_vision_rotary(&q, cos, sin)?;
        let k = apply_qwen_vision_rotary(&k, cos, sin)?;
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scores = q
            .matmul(&k.transpose(1, 2)?)?
            .affine((self.head_dim as f64).sqrt().recip(), 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((tokens, hidden_size))?;
        self.proj.forward(&out).map_err(Into::into)
    }
}

struct QwenVisionMlp {
    gate: candle_nn::Linear,
    up: candle_nn::Linear,
    down: candle_nn::Linear,
}

impl QwenVisionMlp {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.gate_proj"),
            )?,
            up: candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))?,
            down: candle_nn::linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("mlp.down_proj"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate
            .forward(hidden)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up.forward(hidden)?;
        self.down.forward(&(gate * up)?).map_err(Into::into)
    }
}

struct QwenVisionBlock {
    norm1: LongCatRmsNorm,
    norm2: LongCatRmsNorm,
    attn: QwenVisionAttention,
    mlp: QwenVisionMlp,
}

impl QwenVisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: LongCatRmsNorm::new(cfg.hidden_size, vb.pp("norm1"))?,
            norm2: LongCatRmsNorm::new(cfg.hidden_size, vb.pp("norm2"))?,
            attn: QwenVisionAttention::new(cfg, vb.clone())?,
            mlp: QwenVisionMlp::new(cfg, vb)?,
        })
    }

    fn forward(&self, hidden: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let attn = self.attn.forward(&self.norm1.forward(hidden)?, cos, sin)?;
        let hidden = (hidden + attn)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&hidden)?)?;
        (hidden + mlp).map_err(Into::into)
    }
}

struct QwenPatchMerger {
    norm: LongCatRmsNorm,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    merged_dim: usize,
}

impl QwenPatchMerger {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let merged_dim = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        Ok(Self {
            norm: LongCatRmsNorm::new(cfg.hidden_size, vb.pp("ln_q"))?,
            fc1: candle_nn::linear(merged_dim, merged_dim, vb.pp("mlp.0"))?,
            fc2: candle_nn::linear(merged_dim, cfg.out_hidden_size, vb.pp("mlp.2"))?,
            merged_dim,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let tokens = hidden.dim(0)?;
        let hidden = self.norm.forward(hidden)?;
        let hidden = hidden.reshape((tokens / 4, self.merged_dim))?;
        let hidden = self
            .fc1
            .forward(&hidden)?
            .apply(&candle_nn::Activation::Gelu)?;
        self.fc2.forward(&hidden).map_err(Into::into)
    }
}

struct QwenTextAttention {
    q: candle_nn::Linear,
    k: candle_nn::Linear,
    v: candle_nn::Linear,
    o: candle_nn::Linear,
    heads: usize,
    kv_heads: usize,
    kv_groups: usize,
    head_dim: usize,
}

impl QwenTextAttention {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q: candle_nn::linear(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                vb.pp("self_attn.q_proj"),
            )?,
            k: candle_nn::linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                vb.pp("self_attn.k_proj"),
            )?,
            v: candle_nn::linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                vb.pp("self_attn.v_proj"),
            )?,
            o: candle_nn::linear_no_bias(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                vb.pp("self_attn.o_proj"),
            )?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, tokens, _) = hidden.dims3()?;
        let q = self
            .q
            .forward(hidden)?
            .reshape((b, tokens, self.heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(hidden)?
            .reshape((b, tokens, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(hidden)?
            .reshape((b, tokens, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = apply_qwen_mrope(&q, cos, sin)?;
        let k = apply_qwen_mrope(&k, cos, sin)?;
        let k = repeat_qwen_kv(k, self.kv_groups)?.contiguous()?;
        let v = repeat_qwen_kv(v, self.kv_groups)?.contiguous()?;
        let scores = q
            .matmul(&k.transpose(2, 3)?)?
            .affine((self.head_dim as f64).sqrt().recip(), 0.0)?
            .broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out =
            probs
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, tokens, self.heads * self.head_dim))?;
        self.o.forward(&out).map_err(Into::into)
    }
}

struct QwenTextMlp {
    gate: candle_nn::Linear,
    up: candle_nn::Linear,
    down: candle_nn::Linear,
}

impl QwenTextMlp {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.gate_proj"),
            )?,
            up: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.up_proj"),
            )?,
            down: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("mlp.down_proj"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate
            .forward(hidden)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up.forward(hidden)?;
        self.down.forward(&(gate * up)?).map_err(Into::into)
    }
}

struct QwenTextLayer {
    attn: QwenTextAttention,
    mlp: QwenTextMlp,
    input_norm: LongCatRmsNorm,
    post_norm: LongCatRmsNorm,
}

impl QwenTextLayer {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: QwenTextAttention::new(cfg, vb.clone())?,
            mlp: QwenTextMlp::new(cfg, vb.clone())?,
            input_norm: LongCatRmsNorm::new(cfg.hidden_size, vb.pp("input_layernorm"))?,
            post_norm: LongCatRmsNorm::new(cfg.hidden_size, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let attn = self
            .attn
            .forward(&self.input_norm.forward(hidden)?, cos, sin, mask)?;
        let hidden = (hidden + attn)?;
        let mlp = self.mlp.forward(&self.post_norm.forward(&hidden)?)?;
        (hidden + mlp).map_err(Into::into)
    }
}

fn prompt_context_tensor(
    model_path: &Path,
    prompt: &str,
    image_bytes: &[u8],
    cfg: &TextEncoderConfig,
    processor: &ProcessorConfig,
    device: &Device,
    output_dtype: DType,
) -> Result<Tensor> {
    if matches!(device, Device::Cpu) {
        bail!("LongCat Qwen2.5-VL encoder requires GPU execution");
    }
    let dtype = DType::F32;
    let weights = text_encoder_weight_files(model_path);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? };

    let (patches, grid_t, grid_h, grid_w) = preprocess_qwen_image(image_bytes, processor)?;
    let vision_tokens = grid_t * grid_h * grid_w / 4;
    let patches = Tensor::from_vec(
        patches,
        (grid_t * grid_h * grid_w, qwen_patch_features(processor)),
        device,
    )?
    .to_dtype(dtype)?;
    let visual = qwen_vision_forward(
        &patches,
        grid_t,
        grid_h,
        grid_w,
        cfg,
        vb.pp("visual"),
        dtype,
        device,
    )?;

    let tokenizer_path = model_path.join("text_processor/tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer {}: {e}", tokenizer_path.display()))?;
    let token_ids = longcat_qwen_prompt_ids(&tokenizer, prompt, vision_tokens)?;
    let image_positions: Vec<usize> = token_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, id)| (*id == cfg.image_token_id).then_some(idx))
        .collect();
    if image_positions.len() != vision_tokens {
        bail!(
            "LongCat Qwen prompt produced {} image token slots but vision encoder returned {} tokens",
            image_positions.len(),
            vision_tokens
        );
    }

    let ids_tensor = Tensor::new(token_ids.as_slice(), device)?.unsqueeze(0)?;
    let embed_tokens = vb
        .pp("model.embed_tokens")
        .get((152064, cfg.hidden_size), "weight")?;
    let mut hidden = embed_tokens
        .index_select(&ids_tensor.flatten_all()?, 0)?
        .reshape((1, token_ids.len(), cfg.hidden_size))?;
    drop(embed_tokens);

    hidden = splice_visual_embeds(hidden, &visual, &image_positions, dtype, device)?;
    let (cos, sin) = qwen_text_mrope(&token_ids, (grid_t, grid_h, grid_w), cfg, device, dtype)?;
    let mask = qwen_causal_mask(token_ids.len(), device, dtype)?;

    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = QwenTextLayer::new(cfg, vb.pp("model.layers").pp(layer_idx))?;
        hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
    }

    let norm = LongCatRmsNorm::new(cfg.hidden_size, vb.pp("model.norm"))?;
    let hidden = norm.forward(&hidden)?;
    tracing::info!(
        "LongCat Qwen2.5-VL prompt context RMS={:.6}",
        tensor_rms(&hidden)?
    );
    hidden.to_dtype(output_dtype).map_err(Into::into)
}

fn qwen_patch_features(cfg: &ProcessorConfig) -> usize {
    3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
}

fn preprocess_qwen_image(
    image_bytes: &[u8],
    cfg: &ProcessorConfig,
) -> Result<(Vec<f32>, usize, usize, usize)> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| anyhow!("failed to decode LongCat Qwen image: {e}"))?
        .to_rgb8();
    let source_w = img.width() as usize;
    let source_h = img.height() as usize;
    let max_side = native_qwen_max_side();
    let ratio = source_w as f64 / source_h as f64;
    let (bounded_w, bounded_h) = if source_w >= source_h {
        (max_side, (max_side as f64 / ratio).round() as usize)
    } else {
        ((max_side as f64 * ratio).round() as usize, max_side)
    };
    let factor = cfg.patch_size * cfg.merge_size;
    let max_pixels = (bounded_w.max(factor) * bounded_h.max(factor)).min(cfg.max_pixels);
    let min_pixels = cfg.min_pixels.min(max_pixels);
    let (height, width) = smart_resize(source_h, source_w, factor, min_pixels, max_pixels);
    let resized = image::imageops::resize(
        &img,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let grid_t = 1usize;
    let grid_h = height / cfg.patch_size;
    let grid_w = width / cfg.patch_size;
    let patch_features = qwen_patch_features(cfg);
    let mut patches = vec![0f32; grid_t * grid_h * grid_w * patch_features];
    let merge = cfg.merge_size;
    for bh in 0..(grid_h / merge) {
        for bw in 0..(grid_w / merge) {
            for ih in 0..merge {
                for iw in 0..merge {
                    let gh = bh * merge + ih;
                    let gw = bw * merge + iw;
                    let patch_idx = bh * (grid_w / merge) * merge * merge
                        + bw * merge * merge
                        + ih * merge
                        + iw;
                    let start_y = gh * cfg.patch_size;
                    let start_x = gw * cfg.patch_size;
                    for c in 0..3 {
                        let mean = cfg.image_mean.get(c).copied().unwrap_or(0.5);
                        let std = cfg.image_std.get(c).copied().unwrap_or(0.5);
                        for t in 0..cfg.temporal_patch_size {
                            for ph in 0..cfg.patch_size {
                                for pw in 0..cfg.patch_size {
                                    let pixel = resized
                                        .get_pixel((start_x + pw) as u32, (start_y + ph) as u32);
                                    let val = pixel[c] as f32 / 255.0;
                                    let feature_idx = c
                                        * cfg.temporal_patch_size
                                        * cfg.patch_size
                                        * cfg.patch_size
                                        + t * cfg.patch_size * cfg.patch_size
                                        + ph * cfg.patch_size
                                        + pw;
                                    patches[patch_idx * patch_features + feature_idx] =
                                        (val - mean) / std;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((patches, grid_t, grid_h, grid_w))
}

fn qwen_vision_forward(
    patches: &Tensor,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    text_cfg: &TextEncoderConfig,
    vb: VarBuilder,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let cfg = &text_cfg.vision_config;
    let patch_weight = vb.pp("patch_embed.proj").get(
        (
            cfg.hidden_size,
            cfg.in_channels,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
        ),
        "weight",
    )?;
    let patch_weight = patch_weight.reshape((
        cfg.hidden_size,
        cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size,
    ))?;
    let mut hidden = patches.matmul(&patch_weight.transpose(0, 1)?)?;
    let (cos, sin) = qwen_vision_rotary(grid_t, grid_h, grid_w, cfg, device, dtype)?;
    for layer_idx in 0..cfg.depth {
        let block = QwenVisionBlock::new(cfg, vb.pp("blocks").pp(layer_idx))?;
        hidden = block.forward(&hidden, &cos, &sin)?;
    }
    let merger = QwenPatchMerger::new(cfg, vb.pp("merger"))?;
    merger.forward(&hidden)
}

fn qwen_vision_rotary(
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    cfg: &VisionConfig,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..(half / 2))
        .map(|i| 1.0f32 / 10_000f32.powf((2 * i) as f32 / half as f32))
        .collect();
    let mut cos = Vec::with_capacity(grid_t * grid_h * grid_w * head_dim);
    let mut sin = Vec::with_capacity(grid_t * grid_h * grid_w * head_dim);
    let merge = cfg.spatial_merge_size;
    for _t in 0..grid_t {
        for bh in 0..(grid_h / merge) {
            for bw in 0..(grid_w / merge) {
                for ih in 0..merge {
                    for iw in 0..merge {
                        let gh = bh * merge + ih;
                        let gw = bw * merge + iw;
                        let mut freqs = Vec::with_capacity(half);
                        for f in &inv {
                            freqs.push(gh as f32 * f);
                        }
                        for f in &inv {
                            freqs.push(gw as f32 * f);
                        }
                        for value in freqs.iter().chain(freqs.iter()) {
                            cos.push(value.cos());
                            sin.push(value.sin());
                        }
                    }
                }
            }
        }
    }
    let tokens = grid_t * grid_h * grid_w;
    Ok((
        Tensor::from_vec(cos, (tokens, head_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (tokens, head_dim), device)?.to_dtype(dtype)?,
    ))
}

fn apply_qwen_vision_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (tokens, _, head_dim) = x.dims3()?;
    let cos = cos.reshape((tokens, 1, head_dim))?;
    let sin = sin.reshape((tokens, 1, head_dim))?;
    let rotated = rotate_half_standard(x)?;
    Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
}

fn longcat_qwen_prompt_ids(
    tokenizer: &Tokenizer,
    prompt: &str,
    image_tokens: usize,
) -> Result<Vec<u32>> {
    let formatted = format!(
        "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|>{}<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n",
        "<|image_pad|>".repeat(image_tokens),
        prompt
    );
    tokenizer
        .encode(formatted, false)
        .map(|enc| enc.get_ids().to_vec())
        .map_err(|e| anyhow!("failed to tokenize LongCat Qwen prompt: {e}"))
}

fn splice_visual_embeds(
    hidden: Tensor,
    visual: &Tensor,
    image_positions: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (_, seq_len, hidden_size) = hidden.dims3()?;
    let visual = visual
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    let mut values = hidden
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec3::<f32>()?;
    for (visual_idx, token_idx) in image_positions.iter().copied().enumerate() {
        values[0][token_idx].copy_from_slice(&visual[visual_idx]);
    }
    Tensor::from_vec(
        values
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        (1, seq_len, hidden_size),
        device,
    )?
    .to_dtype(dtype)
    .map_err(Into::into)
}

fn qwen_text_mrope(
    token_ids: &[u32],
    grid: (usize, usize, usize),
    cfg: &TextEncoderConfig,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let (position_ids, _) = qwen_rope_index(token_ids, cfg.image_token_id, grid);
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let inv: Vec<f32> = (0..(head_dim / 2))
        .map(|i| 1.0f32 / (cfg.rope_theta as f32).powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(token_ids.len() * head_dim);
    let mut sin = Vec::with_capacity(token_ids.len() * head_dim);
    let sections: Vec<usize> = cfg
        .rope_scaling
        .mrope_section
        .iter()
        .map(|v| v * 2)
        .collect();
    for idx in 0..token_ids.len() {
        for (axis, section) in sections.iter().enumerate() {
            let pos = position_ids[axis][idx] as f32;
            for i in 0..(*section / 2) {
                let angle = pos * inv[i];
                cos.push(angle.cos());
                cos.push(angle.cos());
                sin.push(angle.sin());
                sin.push(angle.sin());
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (token_ids.len(), head_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (token_ids.len(), head_dim), device)?.to_dtype(dtype)?,
    ))
}

fn qwen_rope_index(
    token_ids: &[u32],
    image_token_id: u32,
    grid: (usize, usize, usize),
) -> (Vec<Vec<usize>>, usize) {
    let seq_len = token_ids.len();
    let (grid_t, grid_h, grid_w) = grid;
    let llm_grid_t = grid_t;
    let llm_grid_h = grid_h / 2;
    let llm_grid_w = grid_w / 2;
    let mut t_ids = Vec::with_capacity(seq_len);
    let mut h_ids = Vec::with_capacity(seq_len);
    let mut w_ids = Vec::with_capacity(seq_len);
    let mut idx = 0usize;
    let mut pos = 0usize;
    while idx < seq_len {
        if token_ids[idx] == image_token_id {
            let start = pos;
            for gt in 0..llm_grid_t {
                for gh in 0..llm_grid_h {
                    for gw in 0..llm_grid_w {
                        t_ids.push(start + gt);
                        h_ids.push(start + gh);
                        w_ids.push(start + gw);
                        idx += 1;
                    }
                }
            }
            pos = start + llm_grid_t.max(llm_grid_h).max(llm_grid_w);
        } else {
            t_ids.push(pos);
            h_ids.push(pos);
            w_ids.push(pos);
            idx += 1;
            pos += 1;
        }
    }
    (vec![t_ids, h_ids, w_ids], pos)
}

fn qwen_causal_mask(tokens: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut values = Vec::with_capacity(tokens * tokens);
    for i in 0..tokens {
        for j in 0..tokens {
            values.push(if j <= i { 0f32 } else { -10_000.0 });
        }
    }
    Tensor::from_vec(values, (1, 1, tokens, tokens), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn apply_qwen_mrope(q_or_k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, tokens, head_dim) = q_or_k.dims4()?;
    let cos = cos.reshape((1, 1, tokens, head_dim))?;
    let sin = sin.reshape((1, 1, tokens, head_dim))?;
    let rotated = rotate_half_standard(q_or_k)?;
    Ok((q_or_k.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
}

fn rotate_half_standard(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let half = last_dim / 2;
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&second.neg()?, &first], D::Minus1).map_err(Into::into)
}

fn repeat_qwen_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, kv_heads, tokens, head_dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, tokens, head_dim))?
        .reshape((b, kv_heads * n_rep, tokens, head_dim))
        .map_err(Into::into)
}

fn native_qwen_max_side() -> usize {
    std::env::var("BLOOM_LONGCAT_QWEN_MAX_SIDE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(224)
        .clamp(56, 448)
}

fn native_draft_strength() -> f64 {
    std::env::var("BLOOM_LONGCAT_NATIVE_DRAFT_STRENGTH")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.12)
}

fn native_draft_delta_rms_ratio() -> f32 {
    std::env::var("BLOOM_LONGCAT_NATIVE_DELTA_RMS_RATIO")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(1.0)
        .clamp(0.01, 4.0)
}

fn native_draft_max_side() -> usize {
    std::env::var("BLOOM_LONGCAT_NATIVE_MAX_SIDE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256)
        .clamp(64, 512)
}

fn draft_dimensions(source_width: usize, source_height: usize, max_side: usize) -> (usize, usize) {
    let ratio = source_width as f64 / source_height as f64;
    let (width, height) = if source_width >= source_height {
        (max_side, (max_side as f64 / ratio).round() as usize)
    } else {
        ((max_side as f64 * ratio).round() as usize, max_side)
    };
    (
        round_up_to(width.max(64), 16),
        round_up_to(height.max(64), 16),
    )
}

fn project_heads(
    linear: &candle_nn::Linear,
    hidden: &Tensor,
    batch: usize,
    tokens: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    Ok(linear
        .forward(hidden)?
        .reshape((batch, tokens, heads, head_dim))?)
}

fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean = x_f32.mean_keepdim(D::Minus1)?;
    let var = x_f32.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    x_f32
        .broadcast_sub(&mean)?
        .broadcast_mul(&inv_std)?
        .to_dtype(x_dtype)
        .map_err(Into::into)
}

fn modulate(hidden: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    Ok(hidden
        .broadcast_mul(&scale.affine(1.0, 1.0)?.unsqueeze(1)?)?
        .broadcast_add(&shift.unsqueeze(1)?)?)
}

fn gate(hidden: &Tensor, gate: &Tensor) -> Result<Tensor> {
    Ok(hidden.broadcast_mul(&gate.unsqueeze(1)?)?)
}

fn timestep_embedding(timestep: f32, dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let max_period = 10_000f32;
    let mut values = Vec::with_capacity(dim);
    for i in 0..half {
        let freq = (-max_period.ln() * i as f32 / half as f32).exp();
        values.push((timestep * 1000.0 * freq).cos());
    }
    for i in 0..half {
        let freq = (-max_period.ln() * i as f32 / half as f32).exp();
        values.push((timestep * 1000.0 * freq).sin());
    }
    Ok(Tensor::from_vec(values, (1, dim), device)?)
}

fn rotary_embeddings(
    text_tokens: usize,
    image_tokens: usize,
    cfg: &TransformerConfig,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let axes = cfg
        .axes_dims_rope
        .clone()
        .unwrap_or_else(|| vec![16, 56, 56]);
    if axes.iter().sum::<usize>() != cfg.attention_head_dim {
        bail!("LongCat rope axis dims do not sum to attention_head_dim");
    }
    let mut positions = Vec::with_capacity(text_tokens + image_tokens);
    for idx in 0..text_tokens {
        positions.push((0usize, 0usize, idx));
    }
    let side = (image_tokens as f64).sqrt() as usize;
    for idx in 0..image_tokens {
        positions.push((0usize, idx / side.max(1), idx % side.max(1)));
    }

    let mut cos = Vec::with_capacity(positions.len() * cfg.attention_head_dim);
    let mut sin = Vec::with_capacity(positions.len() * cfg.attention_head_dim);
    for (t, y, x) in positions {
        let axis_positions = [t, y, x];
        for (axis, axis_dim) in axes.iter().enumerate() {
            if axis_dim % 2 != 0 {
                bail!("LongCat rope axis dim {axis_dim} is not even");
            }
            let half = axis_dim / 2;
            for i in 0..half {
                let freq = 1.0f32 / 10_000f32.powf((2 * i) as f32 / *axis_dim as f32);
                let angle = axis_positions[axis] as f32 * freq;
                cos.push(angle.cos());
                cos.push(angle.cos());
                sin.push(angle.sin());
                sin.push(angle.sin());
            }
        }
    }
    let tokens = text_tokens + image_tokens;
    Ok((
        Tensor::from_vec(cos, (tokens, cfg.attention_head_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (tokens, cfg.attention_head_dim), device)?.to_dtype(dtype)?,
    ))
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, tokens, _, head_dim) = x.dims4()?;
    let cos = cos
        .narrow(0, 0, tokens)?
        .reshape((1, tokens, 1, head_dim))?;
    let sin = sin
        .narrow(0, 0, tokens)?
        .reshape((1, tokens, 1, head_dim))?;
    let rotated = rotate_half_interleaved(x)?;
    Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
}

fn rotate_half_interleaved(x: &Tensor) -> Result<Tensor> {
    let (b, tokens, heads, head_dim) = x.dims4()?;
    if head_dim % 2 != 0 {
        bail!("rotary head_dim must be even");
    }
    let pairs = x.reshape((b, tokens, heads, head_dim / 2, 2))?;
    let even = pairs.narrow(4, 0, 1)?;
    let odd = pairs.narrow(4, 1, 1)?;
    Ok(Tensor::cat(&[&odd.neg()?, &even], 4)?.reshape((b, tokens, heads, head_dim))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_are_multiple_of_sixteen() {
        let (w, h) = calculate_dimensions(1024 * 1024, 16.0 / 9.0);
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
    }

    #[test]
    fn flow_sigmas_are_monotonic() {
        let sigmas = flow_match_sigmas(8, 0.63);
        assert_eq!(sigmas.len(), 9);
        for pair in sigmas.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }
}
