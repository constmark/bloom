// Image resampling indexes source and destination coordinates explicitly.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use bloomai_core::{Modality, ProcessorKind, ProcessorSpec};

use crate::io::{DataBlock, MAX_AUDIO_SAMPLES, MAX_IMAGE_BYTES, MAX_TENSOR_VALUES};

pub trait Processor: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> ProcessorKind;
    fn input_modalities(&self) -> Vec<Modality>;
    fn output_modalities(&self) -> Vec<Modality>;
    fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>>;

    fn spec(&self) -> ProcessorSpec {
        ProcessorSpec {
            name: self.name().to_string(),
            kind: self.kind(),
            version: "1".to_string(),
            inputs: self.input_modalities(),
            outputs: self.output_modalities(),
            parameters: HashMap::new(),
        }
    }
}

pub struct IdentityProcessor {
    name: String,
    kind: ProcessorKind,
    modalities: Vec<Modality>,
}

impl IdentityProcessor {
    pub fn new(name: impl Into<String>, kind: ProcessorKind, modalities: Vec<Modality>) -> Self {
        Self {
            name: name.into(),
            kind,
            modalities,
        }
    }
}

impl Processor for IdentityProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> ProcessorKind {
        self.kind.clone()
    }

    fn input_modalities(&self) -> Vec<Modality> {
        self.modalities.clone()
    }

    fn output_modalities(&self) -> Vec<Modality> {
        self.modalities.clone()
    }

    fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>> {
        Ok(blocks)
    }
}

#[cfg(feature = "candle-engine")]
pub struct TokenizerProcessor {
    name: String,
    tokenizer: tokenizers::Tokenizer,
    add_special_tokens: bool,
}

#[cfg(feature = "candle-engine")]
impl TokenizerProcessor {
    pub fn new(name: impl Into<String>, tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            name: name.into(),
            tokenizer,
            add_special_tokens: false,
        }
    }

    pub fn new_with_special_tokens(
        name: impl Into<String>,
        tokenizer: tokenizers::Tokenizer,
        add_special_tokens: bool,
    ) -> Self {
        Self {
            name: name.into(),
            tokenizer,
            add_special_tokens,
        }
    }
}

#[cfg(feature = "candle-engine")]
impl Processor for TokenizerProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> ProcessorKind {
        ProcessorKind::TextTokenizer
    }

    fn input_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn output_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>> {
        let mut out = Vec::new();
        for block in blocks {
            match block {
                DataBlock::Text(text) => {
                    let encoding = self
                        .tokenizer
                        .encode(text, self.add_special_tokens)
                        .map_err(|e| anyhow!("tokenizer encode error: {}", e))?;
                    out.push(DataBlock::Tokens(encoding.get_ids().to_vec()));
                }
                DataBlock::Tokens(ids) => {
                    anyhow::ensure!(
                        ids.iter()
                            .all(|id| self.tokenizer.id_to_token(*id).is_some()),
                        "token sequence contains an unknown token ID"
                    );
                    let text = self
                        .tokenizer
                        .decode(&ids, true)
                        .map_err(|e| anyhow!("tokenizer decode error: {}", e))?;
                    out.push(DataBlock::Text(text));
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }
}

#[cfg(not(feature = "candle-engine"))]
pub struct TokenizerProcessor {
    name: String,
}

#[cfg(not(feature = "candle-engine"))]
impl TokenizerProcessor {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[cfg(not(feature = "candle-engine"))]
impl Processor for TokenizerProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> ProcessorKind {
        ProcessorKind::TextTokenizer
    }

    fn input_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn output_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }

    fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>> {
        Ok(blocks)
    }
}

pub struct AudioProcessor {
    name: String,
    config: AudioProcessorConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioProcessorConfig {
    pub target_sample_rate: Option<u32>,
    pub trim_silence: bool,
    pub vad_threshold: f32,
    pub min_speech_samples: usize,
    pub write_wav: bool,
}

impl Default for AudioProcessorConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: None,
            trim_silence: false,
            vad_threshold: 0.01,
            min_speech_samples: 1,
            write_wav: true,
        }
    }
}

impl AudioProcessor {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: AudioProcessorConfig::default(),
        }
    }

    pub fn with_config(name: impl Into<String>, config: AudioProcessorConfig) -> Self {
        Self {
            name: name.into(),
            config,
        }
    }

    pub fn with_target_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.target_sample_rate = Some(sample_rate);
        self
    }

    pub fn with_vad(mut self, threshold: f32, min_speech_samples: usize) -> Self {
        self.config.trim_silence = true;
        self.config.vad_threshold = threshold;
        self.config.min_speech_samples = min_speech_samples;
        self
    }
}

impl Processor for AudioProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> ProcessorKind {
        ProcessorKind::Audio
    }

    fn input_modalities(&self) -> Vec<Modality> {
        vec![Modality::Audio]
    }

    fn output_modalities(&self) -> Vec<Modality> {
        vec![Modality::Audio]
    }

    fn spec(&self) -> ProcessorSpec {
        let mut parameters = HashMap::new();
        if let Some(rate) = self.config.target_sample_rate {
            parameters.insert("target_sample_rate".to_string(), serde_json::json!(rate));
        }
        parameters.insert(
            "trim_silence".to_string(),
            serde_json::json!(self.config.trim_silence),
        );
        parameters.insert(
            "vad_threshold".to_string(),
            serde_json::json!(self.config.vad_threshold),
        );
        parameters.insert(
            "min_speech_samples".to_string(),
            serde_json::json!(self.config.min_speech_samples),
        );
        parameters.insert(
            "write_wav".to_string(),
            serde_json::json!(self.config.write_wav),
        );

        ProcessorSpec {
            name: self.name().to_string(),
            kind: self.kind(),
            version: "1".to_string(),
            inputs: self.input_modalities(),
            outputs: self.output_modalities(),
            parameters,
        }
    }

    fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>> {
        let mut out = Vec::new();
        let mut staged_files = Vec::new();
        for block in blocks {
            match block {
                DataBlock::AudioPcm {
                    mut samples,
                    mut sample_rate,
                } => {
                    validate_pcm(&samples, sample_rate)?;
                    if !self.config.vad_threshold.is_finite() {
                        bail!("audio VAD threshold must be finite");
                    }
                    if let Some(target_rate) = self.config.target_sample_rate {
                        samples = resample_linear(&samples, sample_rate, target_rate)?;
                        sample_rate = target_rate;
                    }
                    if self.config.trim_silence {
                        samples = trim_silence(
                            &samples,
                            self.config.vad_threshold,
                            self.config.min_speech_samples,
                        );
                    }
                    validate_pcm(&samples, sample_rate)?;
                    if !self.config.write_wav {
                        out.push(DataBlock::AudioPcm {
                            samples,
                            sample_rate,
                        });
                        continue;
                    }
                    // Create the file atomically with private permissions. A timestamp
                    // plus File::create permits symlink overwrite in a shared temp dir.
                    let mut file = tempfile::Builder::new()
                        .prefix("bloom_audio_pcm_")
                        .suffix(".wav")
                        .tempfile()?;
                    write_wav(&mut file, &samples, sample_rate)?;
                    out.push(DataBlock::AudioFile {
                        path: file.path().to_string_lossy().to_string(),
                        language: None,
                    });
                    staged_files.push(file);
                }
                other => out.push(other),
            }
        }
        // A later invalid block must not strand an earlier generated WAV.
        // Successful AudioFile outputs transfer cleanup responsibility to the caller.
        for file in staged_files {
            file.keep()?;
        }
        Ok(out)
    }
}

fn validate_pcm(samples: &[f32], sample_rate: u32) -> Result<()> {
    if !(8_000..=192_000).contains(&sample_rate) {
        bail!("audio sample_rate must be between 8000 and 192000 Hz");
    }
    if samples.is_empty() || samples.len() > MAX_AUDIO_SAMPLES {
        bail!("audio must contain between 1 and {MAX_AUDIO_SAMPLES} samples");
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        bail!("audio contains a non-finite sample");
    }
    Ok(())
}

fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    validate_pcm(samples, source_rate)?;
    if !(8_000..=192_000).contains(&target_rate) {
        bail!("target audio sample_rate must be between 8000 and 192000 Hz");
    }
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }

    let target_len = (samples.len() as u128 * target_rate as u128 + (source_rate / 2) as u128)
        / source_rate as u128;
    if target_len == 0 || target_len > MAX_AUDIO_SAMPLES as u128 {
        bail!("resampled audio must contain between 1 and {MAX_AUDIO_SAMPLES} samples");
    }
    let target_len = usize::try_from(target_len)?;
    let mut out = Vec::with_capacity(target_len);
    for i in 0..target_len {
        // Integer coordinates prevent f32 rounding from selecting samples[len]
        // at the tail of a long clip; f64 interpolation cannot overflow f32 inputs.
        let position = i as u64 * u64::from(source_rate);
        let left = (position / u64::from(target_rate)) as usize;
        let left = left.min(samples.len() - 1);
        let right = (left + 1).min(samples.len() - 1);
        let frac = (position % u64::from(target_rate)) as f64 / f64::from(target_rate);
        out.push(
            (f64::from(samples[left]) * (1.0 - frac) + f64::from(samples[right]) * frac) as f32,
        );
    }
    Ok(out)
}

fn trim_silence(samples: &[f32], threshold: f32, min_speech_samples: usize) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let threshold = threshold.abs();
    let first = samples.iter().position(|s| s.abs() >= threshold);
    let Some(start) = first else {
        return Vec::new();
    };
    let end = samples
        .iter()
        .rposition(|s| s.abs() >= threshold)
        .unwrap_or(start);
    let voiced = samples[start..=end]
        .iter()
        .filter(|s| s.abs() >= threshold)
        .count();
    if voiced < min_speech_samples {
        Vec::new()
    } else {
        samples[start..=end].to_vec()
    }
}

pub(crate) fn write_wav_file(
    path: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
) -> Result<()> {
    validate_pcm(samples, sample_rate)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    write_wav(&mut file, samples, sample_rate)
}

fn write_wav(file: &mut impl std::io::Write, samples: &[f32], sample_rate: u32) -> Result<()> {
    validate_pcm(samples, sample_rate)?;
    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let bytes_per_sample = bits_per_sample / 8;
    let num_samples = samples.len();
    let data_size = num_samples * bytes_per_sample as usize;
    let file_size = 36 + data_size;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&(file_size as u32).to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt subchunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * num_channels as u32 * bytes_per_sample as u32;
    file.write_all(&byte_rate.to_le_bytes())?;
    let block_align = num_channels * bytes_per_sample;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data subchunk
    file.write_all(b"data")?;
    file.write_all(&(data_size as u32).to_le_bytes())?;

    // Convert f32 samples to i16 PCM
    for &sample in samples {
        let sample_i16 = (sample * 32767.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        file.write_all(&sample_i16.to_le_bytes())?;
    }

    Ok(())
}

pub struct ImageProcessor {
    name: String,
    config: ImageProcessorConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessorConfig {
    pub target_width: u32,
    pub target_height: u32,
    pub preserve_aspect_ratio: bool,
    pub normalize_to_tensor: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            target_width: 512,
            target_height: 512,
            preserve_aspect_ratio: true,
            normalize_to_tensor: false,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }
}

impl ImageProcessor {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: ImageProcessorConfig::default(),
        }
    }

    pub fn with_config(name: impl Into<String>, config: ImageProcessorConfig) -> Self {
        Self {
            name: name.into(),
            config,
        }
    }

    pub fn with_target_size(mut self, width: u32, height: u32) -> Self {
        self.config.target_width = width;
        self.config.target_height = height;
        self
    }

    pub fn with_tensor_normalization(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.config.normalize_to_tensor = true;
        self.config.mean = mean;
        self.config.std = std;
        self
    }
}

impl Processor for ImageProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> ProcessorKind {
        ProcessorKind::Image
    }

    fn input_modalities(&self) -> Vec<Modality> {
        vec![Modality::Vision]
    }

    fn output_modalities(&self) -> Vec<Modality> {
        vec![Modality::Vision]
    }

    fn spec(&self) -> ProcessorSpec {
        let mut parameters = HashMap::new();
        parameters.insert(
            "target_width".to_string(),
            serde_json::json!(self.config.target_width),
        );
        parameters.insert(
            "target_height".to_string(),
            serde_json::json!(self.config.target_height),
        );
        parameters.insert(
            "preserve_aspect_ratio".to_string(),
            serde_json::json!(self.config.preserve_aspect_ratio),
        );
        parameters.insert(
            "normalize_to_tensor".to_string(),
            serde_json::json!(self.config.normalize_to_tensor),
        );
        parameters.insert("mean".to_string(), serde_json::json!(self.config.mean));
        parameters.insert("std".to_string(), serde_json::json!(self.config.std));

        ProcessorSpec {
            name: self.name().to_string(),
            kind: self.kind(),
            version: "1".to_string(),
            inputs: self.input_modalities(),
            outputs: self.output_modalities(),
            parameters,
        }
    }

    fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>> {
        let mut out = Vec::new();
        for block in blocks {
            match block {
                DataBlock::Image { bytes, mime } => {
                    if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
                        bail!("image must contain between 1 and {MAX_IMAGE_BYTES} bytes");
                    }
                    validate_image_size(self.config.target_width, self.config.target_height)?;
                    if self.config.normalize_to_tensor {
                        let values = u64::from(self.config.target_width)
                            * u64::from(self.config.target_height)
                            * 3;
                        if values > MAX_TENSOR_VALUES as u64 {
                            bail!("image tensor exceeds the {MAX_TENSOR_VALUES}-value limit");
                        }
                        validate_image_normalization(self.config.mean, self.config.std)?;
                    }
                    let format = match mime.as_str() {
                        "image/png" => image::ImageFormat::Png,
                        "image/jpeg" | "image/jpg" => image::ImageFormat::Jpeg,
                        _ => bail!("image processor supports only PNG and JPEG MIME types"),
                    };
                    let (width, height) =
                        image::io::Reader::with_format(std::io::Cursor::new(&bytes), format)
                            .into_dimensions()?;
                    validate_image_size(width, height)?;
                    let mut reader =
                        image::io::Reader::with_format(std::io::Cursor::new(&bytes), format);
                    let mut limits = image::io::Limits::default();
                    limits.max_image_width = Some(MAX_PROCESSOR_IMAGE_DIMENSION);
                    limits.max_image_height = Some(MAX_PROCESSOR_IMAGE_DIMENSION);
                    limits.max_alloc = Some(MAX_PROCESSOR_IMAGE_PIXELS * 8);
                    reader.limits(limits);
                    let img = reader
                        .decode()
                        .map_err(|e| anyhow!("Failed to decode image: {}", e))?;
                    let resized = if self.config.preserve_aspect_ratio {
                        img.resize(
                            self.config.target_width,
                            self.config.target_height,
                            image::imageops::FilterType::Lanczos3,
                        )
                    } else {
                        img.resize_exact(
                            self.config.target_width,
                            self.config.target_height,
                            image::imageops::FilterType::Lanczos3,
                        )
                    };
                    if self.config.normalize_to_tensor {
                        out.push(DataBlock::Tensor(image_to_chw_tensor(
                            &resized,
                            self.config.mean,
                            self.config.std,
                        )));
                        continue;
                    }
                    let mut cursor = std::io::Cursor::new(Vec::new());
                    resized
                        .write_to(&mut cursor, format)
                        .map_err(|e| anyhow!("Failed to encode resized image: {}", e))?;
                    let image = DataBlock::Image {
                        bytes: cursor.into_inner(),
                        mime,
                    };
                    image.validate().map_err(anyhow::Error::msg)?;
                    out.push(image);
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }
}

const MAX_PROCESSOR_IMAGE_DIMENSION: u32 = 16_384;
const MAX_PROCESSOR_IMAGE_PIXELS: u64 = 16_777_216;

fn validate_image_size(width: u32, height: u32) -> Result<()> {
    if width == 0
        || height == 0
        || width > MAX_PROCESSOR_IMAGE_DIMENSION
        || height > MAX_PROCESSOR_IMAGE_DIMENSION
        || u64::from(width) * u64::from(height) > MAX_PROCESSOR_IMAGE_PIXELS
    {
        bail!("image dimensions exceed the supported pixel or dimension limit");
    }
    Ok(())
}

fn validate_image_normalization(mean: [f32; 3], std: [f32; 3]) -> Result<()> {
    for (mean, std) in mean.into_iter().zip(std) {
        if !mean.is_finite() || !std.is_finite() || std == 0.0 {
            bail!("image normalization requires finite means and finite non-zero std values");
        }
        for endpoint in [0.0, 1.0] {
            if !(((endpoint - f64::from(mean)) / f64::from(std)) as f32).is_finite() {
                bail!("image normalization would produce non-finite tensor values");
            }
        }
    }
    Ok(())
}

fn image_to_chw_tensor(img: &image::DynamicImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let plane = width as usize * height as usize;
    let mut tensor = vec![0.0f32; plane * 3];

    let mut scale = [0.0f64; 3];
    let mut bias = [0.0f64; 3];
    for c in 0..3 {
        scale[c] = 1.0 / (255.0 * f64::from(std[c]));
        bias[c] = -f64::from(mean[c]) / f64::from(std[c]);
    }

    let raw_pixels = rgb.as_raw();

    let (r_part, rest) = tensor.split_at_mut(plane);
    let (g_part, b_part) = rest.split_at_mut(plane);

    for idx in 0..plane {
        r_part[idx] = ((raw_pixels[idx * 3] as f64) * scale[0] + bias[0]) as f32;
    }
    for idx in 0..plane {
        g_part[idx] = ((raw_pixels[idx * 3 + 1] as f64) * scale[1] + bias[1]) as f32;
    }
    for idx in 0..plane {
        b_part[idx] = ((raw_pixels[idx * 3 + 2] as f64) * scale[2] + bias[2]) as f32;
    }

    tensor
}

#[derive(Default)]
pub struct ProcessorRegistry {
    processors: HashMap<String, Box<dyn Processor>>,
}

impl ProcessorRegistry {
    pub fn register(&mut self, processor: Box<dyn Processor>) {
        self.processors
            .insert(processor.name().to_string(), processor);
    }

    pub fn get(&self, name: &str) -> Result<&dyn Processor> {
        self.processors
            .get(name)
            .map(|p| p.as_ref())
            .ok_or_else(|| anyhow!("processor '{}' not found", name))
    }

    pub fn specs(&self) -> Vec<ProcessorSpec> {
        let mut specs: Vec<_> = self.processors.values().map(|p| p.spec()).collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_processor_round_trips_blocks() {
        let processor = IdentityProcessor::new(
            "text.identity",
            ProcessorKind::TextTokenizer,
            vec![Modality::Text],
        );
        let blocks = vec![DataBlock::Text("hello".into())];
        let out = processor.process(blocks.clone()).unwrap();
        assert_eq!(out, blocks);
        assert_eq!(processor.spec().name, "text.identity");
    }

    #[test]
    fn registry_returns_specs() {
        let mut registry = ProcessorRegistry::default();
        registry.register(Box::new(IdentityProcessor::new(
            "image.identity",
            ProcessorKind::Image,
            vec![Modality::Vision],
        )));
        assert!(registry.get("image.identity").is_ok());
        assert_eq!(registry.specs()[0].kind, ProcessorKind::Image);
    }

    #[test]
    fn audio_processor_converts_pcm_to_file() {
        let processor = AudioProcessor::new("audio.pcm2wav");
        let samples = vec![0.0; 16000];
        let blocks = vec![DataBlock::AudioPcm {
            samples,
            sample_rate: 16000,
        }];
        let out = processor.process(blocks).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            DataBlock::AudioFile { path, language } => {
                assert!(path.contains("bloom_audio_pcm_"));
                assert!(path.ends_with(".wav"));
                assert_eq!(*language, None);
                let path = std::path::Path::new(path);
                assert!(path.exists());
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    assert_eq!(
                        std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                        0o600
                    );
                }
                let _ = std::fs::remove_file(path);
            }
            _ => panic!("Expected AudioFile block"),
        }
    }

    #[test]
    fn audio_processor_resamples_and_trims_pcm() {
        let processor = AudioProcessor::with_config(
            "audio.normalize",
            AudioProcessorConfig {
                target_sample_rate: Some(8_000),
                trim_silence: true,
                vad_threshold: 0.1,
                min_speech_samples: 2,
                write_wav: false,
            },
        );
        let blocks = vec![DataBlock::AudioPcm {
            samples: vec![0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
            sample_rate: 16_000,
        }];

        let out = processor.process(blocks).unwrap();
        match &out[0] {
            DataBlock::AudioPcm {
                samples,
                sample_rate,
            } => {
                assert_eq!(*sample_rate, 8_000);
                assert_eq!(samples.len(), 2);
                assert!(samples.iter().all(|s| *s >= 0.1));
            }
            _ => panic!("Expected normalized AudioPcm block"),
        }

        let spec = processor.spec();
        assert_eq!(
            spec.parameters.get("target_sample_rate"),
            Some(&serde_json::json!(8_000))
        );
        assert_eq!(
            spec.parameters.get("trim_silence"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn image_processor_resizes_and_normalizes_to_chw_tensor() {
        let mut img = image::RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 127]));
        img.put_pixel(1, 0, image::Rgb([0, 255, 255]));

        let mut cursor = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();

        let processor = ImageProcessor::with_config(
            "image.tensor",
            ImageProcessorConfig {
                target_width: 2,
                target_height: 1,
                preserve_aspect_ratio: false,
                normalize_to_tensor: true,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
            },
        );
        let out = processor
            .process(vec![DataBlock::Image {
                bytes: cursor.into_inner(),
                mime: "image/png".to_string(),
            }])
            .unwrap();

        match &out[0] {
            DataBlock::Tensor(values) => {
                assert_eq!(values.len(), 2 * 3);
                assert!((values[0] - 1.0).abs() < 1e-6);
                assert!((values[1] + 1.0).abs() < 1e-6);
                assert!((values[2] + 1.0).abs() < 1e-6);
                assert!((values[3] - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected normalized Tensor block"),
        }

        let spec = processor.spec();
        assert_eq!(
            spec.parameters.get("normalize_to_tensor"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn audio_processor_rejects_invalid_pcm_and_unbounded_resampling() {
        let processor = AudioProcessor::with_config(
            "audio",
            AudioProcessorConfig {
                write_wav: false,
                ..Default::default()
            },
        );
        for (samples, sample_rate) in [
            (vec![0.0], 0),
            (vec![0.0], u32::MAX),
            (vec![f32::NAN], 16_000),
            (vec![f32::INFINITY], 16_000),
            (Vec::new(), 16_000),
        ] {
            assert!(
                processor
                    .process(vec![DataBlock::AudioPcm {
                        samples,
                        sample_rate
                    }])
                    .is_err()
            );
        }
        assert!(resample_linear(&[0.0], 8_000, u32::MAX).is_err());
        // A bounded input can still expand beyond the output admission limit.
        let samples = vec![0.0; MAX_AUDIO_SAMPLES / 24 + 1];
        assert!(resample_linear(&samples, 8_000, 192_000).is_err());
        let out = resample_linear(&[f32::MAX, -f32::MAX], 8_000, 192_000).unwrap();
        assert_eq!(out.len(), 48);
        assert!(out.iter().all(|value| value.is_finite()));
        assert_eq!(out[47], -f32::MAX);
    }

    #[test]
    fn wav_writer_preserves_existing_files_and_rejects_invalid_input_before_creation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("existing.wav");
        std::fs::write(&path, b"preserve me").unwrap();
        assert!(write_wav_file(&path, &[0.0], 16_000).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"preserve me");
        let invalid = directory.path().join("invalid.wav");
        assert!(write_wav_file(&invalid, &[f32::NAN], 16_000).is_err());
        assert!(!invalid.exists());
        #[cfg(unix)]
        {
            let link = directory.path().join("link.wav");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert!(write_wav_file(&link, &[0.0], 16_000).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"preserve me");
        }
    }

    #[test]
    fn image_processor_rejects_oversized_resize_and_invalid_normalization_before_decode() {
        let block = || DataBlock::Image {
            bytes: vec![1],
            mime: "image/png".into(),
        };
        for (width, height) in [(0, 1), (u32::MAX, 1), (8_192, 8_192)] {
            let processor = ImageProcessor::new("image").with_target_size(width, height);
            let error = processor.process(vec![block()]).unwrap_err().to_string();
            assert!(error.contains("dimensions"), "{error}");
        }
        let processor = ImageProcessor::new("image")
            .with_target_size(4_096, 4_096)
            .with_tensor_normalization([0.0; 3], [1.0; 3]);
        assert!(
            processor
                .process(vec![block()])
                .unwrap_err()
                .to_string()
                .contains("tensor exceeds")
        );
        for (mean, std) in [
            ([f32::NAN; 3], [1.0; 3]),
            ([0.0; 3], [f32::INFINITY; 3]),
            ([0.0; 3], [0.0; 3]),
            ([0.0; 3], [f32::from_bits(1); 3]),
        ] {
            let processor = ImageProcessor::new("image").with_tensor_normalization(mean, std);
            assert!(
                processor
                    .process(vec![block()])
                    .unwrap_err()
                    .to_string()
                    .contains("normalization")
            );
        }
    }

    #[test]
    fn image_processor_bounds_decoded_dimensions_and_requires_supported_mime() {
        // JPEG SOF dimensions can be inspected without decoding pixel data.
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::new(1, 1))
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Jpeg,
            )
            .unwrap();
        let offset = bytes
            .windows(2)
            .position(|pair| pair == [0xff, 0xc0])
            .unwrap();
        bytes[offset + 5..offset + 7].copy_from_slice(&16_385u16.to_be_bytes());
        let error = ImageProcessor::new("image")
            .process(vec![DataBlock::Image {
                bytes: bytes.clone(),
                mime: "image/jpeg".into(),
            }])
            .unwrap_err()
            .to_string();
        assert!(error.contains("dimensions"), "{error}");
        assert!(
            ImageProcessor::new("image")
                .process(vec![DataBlock::Image {
                    bytes,
                    mime: "application/octet-stream".into(),
                }])
                .unwrap_err()
                .to_string()
                .contains("MIME")
        );
    }

    #[test]
    fn test_tokenizer_processor() {
        #[cfg(feature = "candle-engine")]
        {
            let model = tokenizers::models::bpe::BPE::default();
            let tokenizer = tokenizers::Tokenizer::new(model);
            let processor = TokenizerProcessor::new("test.tokenizer", tokenizer);
            assert_eq!(processor.name(), "test.tokenizer");
            assert!(
                processor
                    .process(vec![DataBlock::Tokens(vec![u32::MAX])])
                    .is_err(),
                "unknown token IDs must not silently decode to empty text"
            );

            let blocks = vec![DataBlock::Text("hello".to_string())];
            let out = processor.process(blocks).unwrap();
            assert_eq!(out.len(), 1);
            match &out[0] {
                DataBlock::Tokens(ids) => {
                    assert!(ids.is_empty());
                    let decoded_out = processor
                        .process(vec![DataBlock::Tokens(ids.clone())])
                        .unwrap();
                    match &decoded_out[0] {
                        DataBlock::Text(text) => {
                            assert_eq!(text, "");
                        }
                        _ => panic!("Expected Text block"),
                    }
                }
                _ => panic!("Expected Tokens block"),
            }
        }

        #[cfg(not(feature = "candle-engine"))]
        {
            let processor = TokenizerProcessor::new("test.tokenizer");
            assert_eq!(processor.name(), "test.tokenizer");
            let blocks = vec![DataBlock::Text("hello".to_string())];
            let out = processor.process(blocks.clone()).unwrap();
            assert_eq!(out, blocks);
        }
    }
}
