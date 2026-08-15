//! T5 text encoder for Wan2.1 video generation.
//!
//! Wraps the umT5 (multilingual T5) encoder to convert text prompts into
//! dense embeddings [batch, seq_len=512, dim=4096] used by the DiT.

use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;

/// T5 text encoder configuration for Wan2.1.
#[derive(Debug, Clone)]
pub struct T5Config {
    /// Maximum token length (Wan2.1 uses 512).
    pub max_length: usize,
    /// Embedding dimension (umT5-XXL = 4096).
    pub d_model: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feed-forward intermediate size.
    pub d_ff: usize,
}

impl T5Config {
    /// Default config for umT5-XXL as used in Wan2.1.
    pub fn umt5_xxl() -> Self {
        Self {
            max_length: 512,
            d_model: 4096,
            vocab_size: 256384,
            num_layers: 24,
            num_heads: 64,
            d_ff: 10240,
        }
    }
}

/// Simple T5 encoder wrapper.
///
/// This implementation uses a minimal approach:
/// - If T5 weights are available, loads the encoder via candle_transformers
/// - Otherwise, falls back to a hash-based text embedding for testing
///
/// For production use, the full umT5-XXL encoder should be loaded from
/// the original Wan2.1 model weights.
pub struct T5TextEncoder {
    config: T5Config,
    device: Device,
    /// Path to the model directory.
    model_path: std::path::PathBuf,
    /// Tokenizer (if available).
    tokenizer: Option<tokenizers::Tokenizer>,
}

/// Loaded T5 encoder model.
struct T5EncoderModel {
    embeddings: candle_nn::Embedding,
    blocks: Vec<T5EncoderBlock>,
    final_norm: candle_nn::LayerNorm,
    /// Output projection (d_model -> text_dim, may be identity if d_model == text_dim).
    output_proj: Option<candle_nn::Linear>,
}

/// Simplified T5 encoder block.
struct T5EncoderBlock {
    self_attn: T5SelfAttention,
    layer_norm1: candle_nn::LayerNorm,
    ffn_wi: candle_nn::Linear,
    ffn_wo: candle_nn::Linear,
    layer_norm2: candle_nn::LayerNorm,
}

struct T5SelfAttention {
    q: candle_nn::Linear,
    k: candle_nn::Linear,
    v: candle_nn::Linear,
    o: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    _relative_attention_bias: Option<candle_nn::Embedding>,
}

impl T5TextEncoder {
    /// Create a new T5 text encoder.
    ///
    /// Attempts to load weights from the given path. If weights are not found,
    /// creates a stub encoder that produces zero embeddings (useful for testing
    /// the pipeline without downloading the full T5 model).
    pub fn new(model_path: &Path, device: &Device) -> Result<Self> {
        let config = T5Config::umt5_xxl();

        // Try to load tokenizer
        let tokenizer = Self::load_tokenizer(model_path);

        Ok(Self {
            config,
            device: device.clone(),
            model_path: model_path.to_path_buf(),
            tokenizer,
        })
    }

    fn load_tokenizer(model_path: &Path) -> Option<tokenizers::Tokenizer> {
        let search_dir = if model_path.is_file() {
            model_path.parent().unwrap_or(Path::new("."))
        } else {
            model_path
        };

        // Try various tokenizer locations
        let candidates = [
            search_dir.join("tokenizer.json"),
            search_dir.join("google/umt5-xxl/tokenizer.json"),
            search_dir.join("tokenizer.model"),
        ];

        for path in &candidates {
            if path.exists() {
                if let Ok(tok) = tokenizers::Tokenizer::from_file(path) {
                    return Some(tok);
                }
            }
        }
        None
    }

    fn load_model(config: &T5Config, model_path: &Path, device: &Device) -> Option<T5EncoderModel> {
        // Look for T5 weight files
        let t5_file = crate::executor::wan::loader::find_t5_weights(model_path)?;
        if t5_file.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            tracing::warn!(
                "Ignoring T5 weights in PyTorch checkpoint format ({}). \
                 Candle Wan runtime only loads safetensors and will not import PyTorch.",
                t5_file.display()
            );
            return None;
        }

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[t5_file], DType::F32, device)
        };
        let vb = match vb {
            Ok(v) => v,
            Err(_) => return None,
        };

        Self::build_model(config, vb, device).ok()
    }

    fn build_model(
        config: &T5Config,
        vb: candle_nn::VarBuilder,
        _device: &Device,
    ) -> Result<T5EncoderModel> {
        let d = config.d_model;
        let prefix = "encoder";

        let embeddings = candle_nn::embedding(
            config.vocab_size,
            d,
            vb.pp(format!("{prefix}.embed_tokens")),
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block_vb = vb.pp(format!("{prefix}.block.{i}"));
            let layer_vb = block_vb.pp("layer");

            // Self-attention layer
            let self_attn = Self::build_self_attention(config, layer_vb.pp("0"))?;
            let layer_norm1 = candle_nn::layer_norm(d, 1e-6, layer_vb.pp("0").pp("layer_norm"))?;

            // FFN layer
            let ffn_vb = layer_vb.pp("1");
            let ffn_wi = candle_nn::linear_no_bias(d, config.d_ff, ffn_vb.pp("DenseReluDense.wi"))?;
            let ffn_wo = candle_nn::linear_no_bias(config.d_ff, d, ffn_vb.pp("DenseReluDense.wo"))?;
            let layer_norm2 = candle_nn::layer_norm(d, 1e-6, ffn_vb.pp("layer_norm"))?;

            blocks.push(T5EncoderBlock {
                self_attn,
                layer_norm1,
                ffn_wi,
                ffn_wo,
                layer_norm2,
            });
        }

        let final_norm =
            candle_nn::layer_norm(d, 1e-6, vb.pp(format!("{prefix}.final_layer_norm")))?;

        // Optional output projection (T5 typically doesn't have one, but Wan may)
        let output_proj = None;

        Ok(T5EncoderModel {
            embeddings,
            blocks,
            final_norm,
            output_proj,
        })
    }

    fn build_self_attention(
        config: &T5Config,
        vb: candle_nn::VarBuilder,
    ) -> Result<T5SelfAttention> {
        let d = config.d_model;
        let n = config.num_heads;
        let dh = d / n;
        let self_attn_vb = vb.pp("SelfAttention");

        let q = candle_nn::linear_no_bias(d, d, self_attn_vb.pp("q"))?;
        let k = candle_nn::linear_no_bias(d, d, self_attn_vb.pp("k"))?;
        let v = candle_nn::linear_no_bias(d, d, self_attn_vb.pp("v"))?;
        let o = candle_nn::linear_no_bias(d, d, self_attn_vb.pp("o"))?;

        let relative_attention_bias =
            candle_nn::embedding(32, n, self_attn_vb.pp("relative_attention_bias")).ok();

        Ok(T5SelfAttention {
            q,
            k,
            v,
            o,
            num_heads: n,
            head_dim: dh,
            _relative_attention_bias: relative_attention_bias,
        })
    }

    /// Encode text prompts into embeddings.
    ///
    /// Returns tensor of shape [batch, max_length, d_model].
    pub fn encode(&self, texts: &[String]) -> Result<Tensor> {
        if let Some(model) = Self::load_model(&self.config, &self.model_path, &self.device) {
            let res = self.encode_with_model(&model, texts)?;
            // model is dropped here, releasing VRAM!
            Ok(res)
        } else {
            // Fallback: create deterministic pseudo-embeddings from text hash
            self.encode_fallback(texts)
        }
    }

    fn encode_with_model(&self, model: &T5EncoderModel, texts: &[String]) -> Result<Tensor> {
        let batch = texts.len();
        let max_len = self.config.max_length;

        // Tokenize
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer required for T5 encoding"))?;

        let mut input_ids = Vec::<u32>::with_capacity(batch * max_len);
        for text in texts {
            let encoding = tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow!("tokenization error: {}", e))?;
            let ids = encoding.get_ids();
            let mut padded = vec![0u32; max_len];
            let copy_len = ids.len().min(max_len);
            padded[..copy_len].copy_from_slice(&ids[..copy_len]);
            input_ids.extend(padded.iter());
        }

        let input_ids = Tensor::from_vec(input_ids, (batch, max_len), &self.device)?;

        // Embed tokens
        let mut hidden = model.embeddings.forward(&input_ids)?;

        // Run through encoder blocks
        for block in &model.blocks {
            hidden = self.encoder_block_forward(block, &hidden)?;
        }

        // Final layer norm
        hidden = model.final_norm.forward(&hidden.to_dtype(DType::F32)?)?;

        // Optional output projection
        if let Some(ref proj) = model.output_proj {
            hidden = proj.forward(&hidden)?;
        }

        Ok(hidden)
    }

    fn encoder_block_forward(&self, block: &T5EncoderBlock, x: &Tensor) -> Result<Tensor> {
        // Self-attention with pre-norm
        let normed = block.layer_norm1.forward(&x.to_dtype(DType::F32)?)?;
        let attn_out = self.self_attention_forward(&block.self_attn, &normed)?;
        let x = (x + attn_out)?;

        // FFN with pre-norm
        let normed = block.layer_norm2.forward(&x.to_dtype(DType::F32)?)?;
        let ffn_out = block.ffn_wi.forward(&normed)?.relu()?;
        let ffn_out = block.ffn_wo.forward(&ffn_out)?;
        let x = (x + ffn_out)?;

        Ok(x)
    }

    fn self_attention_forward(&self, attn: &T5SelfAttention, x: &Tensor) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;
        let n = attn.num_heads;
        let dh = attn.head_dim;

        let q = attn
            .q
            .forward(x)?
            .reshape((b, s, n, dh))?
            .permute((0, 2, 1, 3))?;
        let k = attn
            .k
            .forward(x)?
            .reshape((b, s, n, dh))?
            .permute((0, 2, 1, 3))?;
        let v = attn
            .v
            .forward(x)?
            .reshape((b, s, n, dh))?
            .permute((0, 2, 1, 3))?;

        let scale = (dh as f64).sqrt().recip();
        let scores = (q.matmul(&k.t()?)?.affine(scale, 0.0))?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?;

        // [b, n, s, dh] -> [b, s, d]
        let out = out.permute((0, 2, 1, 3))?.reshape((b, s, n * dh))?;
        attn.o.forward(&out).map_err(Into::into)
    }

    /// Fallback encoding: produce deterministic pseudo-embeddings from text.
    ///
    /// This uses a simple hash-based approach to create reproducible
    /// embeddings when T5 weights are not available.
    fn encode_fallback(&self, texts: &[String]) -> Result<Tensor> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let batch = texts.len();
        let d = self.config.d_model;
        let max_len = self.config.max_length;

        let mut data = vec![0.0f32; batch * max_len * d];

        for (bi, text) in texts.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let seed = hasher.finish();

            // Generate deterministic pseudo-random embeddings
            let mut rng = SimpleRng::new(seed);
            for si in 0..max_len {
                for di in 0..d {
                    let idx = bi * max_len * d + si * d + di;
                    // Scale to roughly match T5 embedding distribution
                    data[idx] = (rng.next_f32() - 0.5) * 0.1;
                }
            }
        }

        Tensor::from_vec(data, (batch, max_len, d), &self.device).map_err(Into::into)
    }

    /// Get the output dimension.
    pub fn output_dim(&self) -> usize {
        self.config.d_model
    }

    /// Get max sequence length.
    pub fn max_length(&self) -> usize {
        self.config.max_length
    }
}

/// Simple deterministic PRNG for fallback embedding generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t5_config_default() {
        let cfg = T5Config::umt5_xxl();
        assert_eq!(cfg.d_model, 4096);
        assert_eq!(cfg.max_length, 512);
    }

    #[test]
    fn test_fallback_encoding_shape() {
        let encoder = T5TextEncoder {
            config: T5Config::umt5_xxl(),
            device: Device::Cpu,
            model_path: std::path::PathBuf::new(),
            tokenizer: None,
        };
        let result = encoder.encode(&["hello world".to_string()]).unwrap();
        assert_eq!(result.dims(), &[1, 512, 4096]);
    }

    #[test]
    fn test_fallback_encoding_deterministic() {
        let encoder = T5TextEncoder {
            config: T5Config::umt5_xxl(),
            device: Device::Cpu,
            model_path: std::path::PathBuf::new(),
            tokenizer: None,
        };
        let r1 = encoder.encode(&["test".to_string()]).unwrap();
        let r2 = encoder.encode(&["test".to_string()]).unwrap();
        // Same text should produce same embeddings
        let diff = (r1 - r2)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let v1 = rng.next_f32();
        let v2 = rng.next_f32();
        assert!((0.0..=1.0).contains(&v1));
        assert!((0.0..=1.0).contains(&v2));
        assert!(v1 != v2);
    }
}
