// The top-level diffusion call keeps checkpoint inputs explicit at the boundary.
#![allow(clippy::too_many_arguments)]

//! Wan2.1 video generation model — the main inference pipeline.
//!
//! Orchestrates the full text-to-video generation:
//! 1. T5 text encoding
//! 2. Noise initialization
//! 3. Iterative DiT denoising with CFG
//! 4. VAE decoding
//! 5. Frame extraction

use std::path::Path;

use anyhow::{anyhow, Result};
use bloomai_core::{GenerationParams, Modality, ModelManifest};
use candle_core::{DType, Device, Tensor};

use super::dit::{WanConfig, WanModel};
use super::loader;
use super::scheduler::FlowUniPCScheduler;
use super::t5_encoder::T5TextEncoder;
use super::vae::{self, WanVAE};

use crate::io::OutputChunk;
use crate::io::{ModelInput, ModelOutput, VideoOutput};
use crate::model::{LoadedModel, ModelMetadata, OutputSink};

/// Wan2.1 video generation model.
///
/// Implements the `LoadedModel` trait for integration with the bloom engine.
pub struct WanVideoModel {
    /// Path to model directory.
    model_path: std::path::PathBuf,
    /// T5 text encoder.
    text_encoder: T5TextEncoder,
    /// VAE decoder.
    vae: WanVAE,
    /// Device.
    device: Device,
    /// Model metadata for the engine interface.
    metadata: ModelMetadata,
}

impl WanVideoModel {
    /// Load the Wan2.1 model from a path.
    ///
    /// This loads:
    /// - DiT weights from GGUF/Safetensors (the primary model file)
    /// - T5 encoder weights (companion file, optional)
    /// - VAE decoder weights (companion file, optional)
    pub fn load(
        model_path: &Path,
        device_kind: bloomai_core::DeviceKind,
    ) -> Result<Box<dyn LoadedModel>> {
        let device = match device_kind {
            bloomai_core::DeviceKind::Cpu => Device::Cpu,
            bloomai_core::DeviceKind::Gpu => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0)
                        .map_err(|e| anyhow!("failed to initialize CUDA device: {}", e))?
                }
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0)
                        .map_err(|e| anyhow!("failed to initialize Metal device: {}", e))?
                }
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                {
                    return Err(anyhow!(
                        "GPU not available: bloom was compiled without cuda or metal features"
                    ));
                }
            }
            other => {
                return Err(anyhow!(
                    "unsupported device {:?} for Wan engine (use cpu or gpu)",
                    other
                ));
            }
        };

        let offload = std::env::var("BLOOM_WAN_OFFLOAD")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
            || std::env::var("BLOOM_GPU_LAYERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
                > 0;

        let host_device = if offload { Device::Cpu } else { device.clone() };

        tracing::info!("Loading Wan2.1 model from {}", model_path.display());

        // Find weight files to verify they exist and determine quantization
        let weight_files = loader::find_weight_files(model_path)?;

        // Load T5 encoder (optional)
        let text_encoder = T5TextEncoder::new(model_path, &host_device)?;
        tracing::info!("T5 text encoder initialized on {:?}", host_device);

        // Load VAE decoder (optional)
        let vae_decoder = WanVAE::new(model_path, &host_device)?;
        tracing::info!("VAE decoder initialized on {:?}", host_device);

        let model_id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("wan2.1-t2v")
            .to_string();

        let metadata = ModelMetadata {
            id: model_id.clone(),
            modality: Modality::Vision, // Video output uses Vision modality
            quantized: matches!(weight_files, loader::WeightFiles::Gguf(_)),
            manifest: ModelManifest {
                id: model_id,
                family: bloomai_core::ModelFamily::Custom("wan".to_string()),
                ..ModelManifest::default()
            },
        };

        Ok(Box::new(Self {
            model_path: model_path.to_path_buf(),
            text_encoder,
            vae: vae_decoder,
            device,
            metadata,
        }))
    }

    /// Run the full video generation pipeline.
    fn generate_video(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        width: u32,
        height: u32,
        num_frames: u32,
        fps: f32,
        guidance_scale: f64,
        num_steps: u32,
        seed: Option<u64>,
        mut sink: Option<&mut dyn OutputSink>,
    ) -> Result<VideoOutput> {
        let cfg = WanConfig::t2v_1_3b();

        // Compute latent dimensions from video dimensions
        // Video: [F, H, W] -> Latent: [F', H', W']
        // F' = (F - 1) / temporal_stride + 1
        // H' = H / spatial_stride
        // W' = W / spatial_stride
        let t_stride = self.vae.temporal_stride();
        let s_stride = self.vae.spatial_stride();
        let latent_f = ((num_frames as usize) - 1) / t_stride + 1;
        let latent_h = height as usize / s_stride;
        let latent_w = width as usize / s_stride;

        // 1. Encode text
        tracing::info!("Encoding text prompt: {}", &prompt[..prompt.len().min(80)]);
        let texts = if guidance_scale > 1.0 {
            let neg = negative_prompt.unwrap_or("");
            vec![prompt.to_string(), neg.to_string()]
        } else {
            vec![prompt.to_string()]
        };
        let text_embeddings = self.text_encoder.encode(&texts)?;
        // text_embeddings: [batch_text, text_len=512, text_dim=4096]

        // 2. Initialize noise
        let seed = seed.unwrap_or(42);
        let latent_shape = (1, cfg.in_dim, latent_f, latent_h, latent_w);
        let mut latent = self.random_noise(latent_shape, seed)?;

        // 3. Set up scheduler
        let shift = 5.0; // Default shift for Wan2.1
        let mut scheduler = FlowUniPCScheduler::new(1000, shift, false);
        scheduler.set_timesteps(num_steps, shift);
        let total_steps = scheduler.num_steps();

        // 4. Load DiT model to GPU natively (lazy QMatMul ensures it fits in VRAM)
        tracing::info!("Loading DiT model weights...");
        let weight_files = loader::find_weight_files(&self.model_path)?;

        let vb_gpu = loader::build_var_builder(&weight_files, DType::F32, &self.device, false)?;

        let dit = WanModel::new(&cfg, vb_gpu, None, &self.device)
            .map_err(|e| anyhow!("failed to load DiT model: {}", e))?;

        tracing::info!(
            "Starting diffusion: {} steps, latent shape {:?}",
            total_steps,
            latent_shape
        );

        // Diffusion loop with CFG
        for step_idx in 0..total_steps {
            let timestep = scheduler.timesteps[step_idx];
            let t = Tensor::new(&[timestep as f32], &self.device)?;

            if guidance_scale > 1.0 {
                // Classifier-free guidance: run conditional + unconditional
                let latent_double = Tensor::cat(&[&latent, &latent], 0)?;
                let t_double = Tensor::cat(&[&t, &t], 0)?;
                let ctx = text_embeddings.narrow(0, 0, 2)?.to_device(&self.device)?;

                let noise_pred = dit.forward(&latent_double, &t_double, &ctx)?;

                // Split conditional and unconditional predictions
                let noise_cond = noise_pred.narrow(0, 0, 1)?;
                let noise_uncond = noise_pred.narrow(0, 1, 1)?;

                // CFG: guided = uncond + scale * (cond - uncond)
                let guided = noise_uncond
                    .add(&noise_cond.sub(&noise_uncond)?.affine(guidance_scale, 0.0)?)?;

                latent = scheduler.step(&guided, step_idx, &latent)?;
            } else {
                // No CFG
                let ctx = text_embeddings.narrow(0, 0, 1)?.to_device(&self.device)?;
                let noise_pred = dit.forward(&latent, &t, &ctx)?;
                latent = scheduler.step(&noise_pred, step_idx, &latent)?;
            }

            // Report progress
            if let Some(sink) = sink.as_mut() {
                sink.on_chunk(OutputChunk::DiffusionProgress {
                    step: (step_idx + 1) as u32,
                    total_steps: total_steps as u32,
                })?;
            }

            if (step_idx + 1) % 10 == 0 || step_idx == total_steps - 1 {
                tracing::info!("Diffusion step {}/{}", step_idx + 1, total_steps);
            }
        }

        // Drop DiT model to free VRAM before VAE decoding
        std::mem::drop(dit);

        // 5. VAE decode
        tracing::info!("Decoding latent to video frames...");
        // Ensure latent is on the same device as the VAE decoder (e.g. CPU if offloaded)
        let latent_vae = latent.to_device(self.vae.device())?;
        let video_tensor = self.vae.decode(&latent_vae)?;
        // video_tensor: [1, 3, F, H, W]

        // 6. Extract frames
        let video_tensor = video_tensor.squeeze(0)?; // [3, F, H, W]
        let frame_count = video_tensor.dim(1)?;
        let frame_h = video_tensor.dim(2)?;
        let frame_w = video_tensor.dim(3)?;

        let mut frames = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let frame = video_tensor.narrow(1, i, 1)?.squeeze(1)?; // [3, H, W]
            let rgb = vae::tensor_to_rgb_frame(&frame)?;
            // Stream individual frames
            if let Some(sink) = sink.as_mut() {
                sink.on_chunk(OutputChunk::VideoFrame(rgb.clone()))?;
            }
            frames.push(rgb);
        }

        tracing::info!(
            "Video generation complete: {}x{}x{} @ {} fps",
            frame_w,
            frame_h,
            frame_count,
            fps
        );

        // Signal completion
        if let Some(sink) = sink.as_mut() {
            sink.on_chunk(OutputChunk::VideoComplete {
                width,
                height,
                fps,
                frame_count: frame_count as u32,
            })?;
        }

        Ok(VideoOutput {
            width,
            height,
            fps,
            frame_count: frame_count as u32,
            frames,
        })
    }

    /// Generate random noise tensor with deterministic seed.
    fn random_noise(
        &self,
        shape: (usize, usize, usize, usize, usize),
        seed: u64,
    ) -> Result<Tensor> {
        // Use a simple deterministic approach with the seed
        // Candle doesn't have a built-in seeded random, so we generate manually
        let (b, c, f, h, w) = shape;
        let total = b * c * f * h * w;
        let mut data = Vec::with_capacity(total);
        let mut rng = PcgRng::new(seed);
        for _ in 0..total {
            data.push(rng.next_normal_f32());
        }
        Tensor::from_vec(data, shape, &self.device)
            .map_err(|e| anyhow!("noise tensor error: {}", e))
    }
}

impl LoadedModel for WanVideoModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        match input {
            ModelInput::VideoGeneration {
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                fps,
                guidance_scale,
                num_steps,
                seed,
            } => {
                let video = self.generate_video(
                    &prompt,
                    negative_prompt.as_deref(),
                    width,
                    height,
                    num_frames,
                    fps,
                    guidance_scale,
                    num_steps,
                    seed.or(params.seed),
                    None,
                )?;
                Ok(ModelOutput {
                    text: None,
                    logits: None,
                    image: None,
                    audio: None,
                    video: Some(video),
                })
            }
            ModelInput::Text { prompt } => {
                // Treat as video generation with default params (configurable via env vars)
                let width = std::env::var("BLOOM_WAN_WIDTH")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(832);
                let height = std::env::var("BLOOM_WAN_HEIGHT")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(480);
                let num_frames = std::env::var("BLOOM_WAN_FRAMES")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(81);
                let num_steps = std::env::var("BLOOM_WAN_STEPS")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(50);
                let guidance_scale = std::env::var("BLOOM_WAN_GUIDANCE_SCALE")
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(5.0);

                let video = self.generate_video(
                    &prompt,
                    None,
                    width,
                    height,
                    num_frames,
                    16.0, // default fps
                    guidance_scale,
                    num_steps,
                    params.seed,
                    None,
                )?;
                Ok(ModelOutput {
                    text: None,
                    logits: None,
                    image: None,
                    audio: None,
                    video: Some(video),
                })
            }
            _ => Err(anyhow!(
                "WanVideoModel expects VideoGeneration or Text input, got {:?}",
                std::mem::discriminant(&input)
            )),
        }
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        let (prompt, neg_prompt, width, height, num_frames, fps, guidance_scale, num_steps, seed) =
            match input {
                ModelInput::VideoGeneration {
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    num_frames,
                    fps,
                    guidance_scale,
                    num_steps,
                    seed,
                } => (
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    num_frames,
                    fps,
                    guidance_scale,
                    num_steps,
                    seed,
                ),
                ModelInput::Text { prompt } => {
                    let width = std::env::var("BLOOM_WAN_WIDTH")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(832);
                    let height = std::env::var("BLOOM_WAN_HEIGHT")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(480);
                    let num_frames = std::env::var("BLOOM_WAN_FRAMES")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(81);
                    let num_steps = std::env::var("BLOOM_WAN_STEPS")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(50);
                    let guidance_scale = std::env::var("BLOOM_WAN_GUIDANCE_SCALE")
                        .ok()
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(5.0);
                    (
                        prompt,
                        None,
                        width,
                        height,
                        num_frames,
                        16.0,
                        guidance_scale,
                        num_steps,
                        params.seed,
                    )
                }
                _ => {
                    return Err(anyhow!(
                        "WanVideoModel expects VideoGeneration or Text input"
                    ));
                }
            };

        let _video = self.generate_video(
            &prompt,
            neg_prompt.as_deref(),
            width,
            height,
            num_frames,
            fps,
            guidance_scale,
            num_steps,
            seed.or(params.seed),
            Some(sink),
        )?;

        sink.on_chunk(OutputChunk::End)?;
        Ok(())
    }
}

/// Simple PCG-based RNG for deterministic noise generation.
struct PcgRng {
    state: u64,
    inc: u64,
}

impl PcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
            inc: (seed << 1) | 1,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((-(rot as i32)) & 31) as u32)
    }

    /// Generate a standard normal random float using Box-Muller.
    fn next_normal_f32(&mut self) -> f32 {
        let u1 = (self.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let u2 = (self.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z0 as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcg_rng_deterministic() {
        let mut rng1 = PcgRng::new(42);
        let mut rng2 = PcgRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_pcg_rng_normal_distribution() {
        let mut rng = PcgRng::new(42);
        let samples: Vec<f32> = (0..1000).map(|_| rng.next_normal_f32()).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.2, "mean should be close to 0, got {}", mean);
    }
}
