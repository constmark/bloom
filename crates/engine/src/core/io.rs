use serde::{Deserialize, Serialize};

/// Safety limits shared by embedded callers and every transport adapter.
///
/// HTTP handlers have tighter, protocol-specific limits, but the engine also
/// has public Rust/FFI/plugin entry points.  Keeping a second bounded layer
/// here prevents a direct caller from handing an external runtime an
/// unbounded allocation or non-finite tensor values.
pub const MAX_TEXT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_TOKEN_COUNT: usize = 4 * 1024 * 1024;
pub const MAX_AUDIO_SAMPLES: usize = 32 * 1024 * 1024;
pub const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_VIDEO_FRAMES: usize = 2_048;
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_TENSOR_VALUES: usize = 16 * 1024 * 1024;
pub const MAX_STATE_ID_BYTES: usize = 512;

fn bounded_text(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(format!("{field} exceeds the {MAX_TEXT_BYTES}-byte limit"));
    }
    Ok(())
}

fn validate_audio(samples: &[f32], sample_rate: u32, field: &str) -> Result<(), String> {
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(format!(
            "{field} sample_rate must be between 8000 and 192000 Hz"
        ));
    }
    if samples.is_empty() || samples.len() > MAX_AUDIO_SAMPLES {
        return Err(format!(
            "{field} must contain between 1 and {MAX_AUDIO_SAMPLES} samples"
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(format!("{field} contains a non-finite sample"));
    }
    Ok(())
}

fn validate_image(bytes: &[u8], mime: &str, field: &str) -> Result<(), String> {
    if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "{field} must contain between 1 and {MAX_IMAGE_BYTES} bytes"
        ));
    }
    let mime = mime.trim();
    if mime.is_empty() || mime.len() > 128 || mime.chars().any(char::is_control) {
        return Err(format!("{field} has an invalid MIME type"));
    }
    // application/octet-stream is retained for legacy/plugin callers that
    // carry a model-specific image container and validate it downstream.
    if !mime.to_ascii_lowercase().starts_with("image/") && mime != "application/octet-stream" {
        return Err(format!("{field} MIME type must be image/*"));
    }
    Ok(())
}

fn validate_tensor(values: &[f32], field: &str) -> Result<(), String> {
    if values.is_empty() || values.len() > MAX_TENSOR_VALUES {
        return Err(format!(
            "{field} must contain between 1 and {MAX_TENSOR_VALUES} values"
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{field} contains a non-finite value"));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataBlock {
    Text(String),
    Tokens(Vec<u32>),
    AudioPcm {
        samples: Vec<f32>,
        sample_rate: u32,
    },
    AudioFile {
        path: String,
        language: Option<String>,
    },
    Image {
        bytes: Vec<u8>,
        mime: String,
    },
    VideoFrames(Vec<Vec<u8>>),
    Tensor(Vec<f32>),
    /// World state snapshot: state_id, observations, optional latent.
    WorldState {
        state_id: String,
        latent: Option<Vec<u8>>,
        step: u64,
    },
    /// Action from a policy model: action space name + values.
    Action {
        action_space: String,
        values: Vec<f32>,
    },
}

impl DataBlock {
    /// Validate one block before processor or backend execution.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Text(text) => bounded_text(text, "text block"),
            Self::Tokens(tokens) => {
                if tokens.is_empty() || tokens.len() > MAX_TOKEN_COUNT {
                    Err(format!(
                        "token block must contain between 1 and {MAX_TOKEN_COUNT} tokens"
                    ))
                } else {
                    Ok(())
                }
            }
            Self::AudioPcm {
                samples,
                sample_rate,
            } => validate_audio(samples, *sample_rate, "audio block"),
            Self::AudioFile { path, language } => {
                if path.is_empty() || path.len() > 4 * 1024 {
                    return Err("audio file path is empty or exceeds 4096 bytes".to_string());
                }
                if path.chars().any(char::is_control) {
                    return Err("audio file path contains control characters".to_string());
                }
                if let Some(language) = language
                    && (language.is_empty()
                        || language.len() > 64
                        || language.chars().any(char::is_control))
                {
                    return Err("audio language is invalid".to_string());
                }
                Ok(())
            }
            Self::Image { bytes, mime } => validate_image(bytes, mime, "image block"),
            Self::VideoFrames(frames) => {
                if frames.is_empty() || frames.len() > MAX_VIDEO_FRAMES {
                    return Err(format!(
                        "video block must contain between 1 and {MAX_VIDEO_FRAMES} frames"
                    ));
                }
                for (index, frame) in frames.iter().enumerate() {
                    if frame.is_empty() || frame.len() > MAX_FRAME_BYTES {
                        return Err(format!("video frame {index} is empty or too large"));
                    }
                }
                Ok(())
            }
            Self::Tensor(values) => validate_tensor(values, "tensor block"),
            Self::WorldState {
                state_id, latent, ..
            } => {
                if state_id.is_empty() || state_id.len() > MAX_STATE_ID_BYTES {
                    return Err("world state id is empty or too long".to_string());
                }
                if state_id.chars().any(char::is_control) {
                    return Err("world state id contains control characters".to_string());
                }
                if latent
                    .as_ref()
                    .is_some_and(|bytes| bytes.len() > MAX_IMAGE_BYTES)
                {
                    return Err("world state latent exceeds the size limit".to_string());
                }
                Ok(())
            }
            Self::Action {
                action_space,
                values,
            } => {
                bounded_text(action_space, "action space")?;
                validate_tensor(values, "action values")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    #[serde(default)]
    pub response_format: Option<bloomai_core::ResponseFormat>,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            seed: None,
            response_format: None,
        }
    }
}

impl InferenceParams {
    pub fn validate(&self) -> Result<(), String> {
        bloomai_core::GenerationParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            response_format: self.response_format.clone(),
        }
        .validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub blocks: Vec<DataBlock>,
    pub params: InferenceParams,
}

impl InferenceRequest {
    pub fn validate(&self) -> Result<(), String> {
        self.params.validate()?;
        if self.blocks.is_empty() {
            return Err("inference request must contain at least one block".to_string());
        }
        for block in &self.blocks {
            block.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputChunk {
    TextDelta(String),
    Tokens(Vec<u32>),
    AudioDelta(Vec<f32>),
    Image(Vec<u8>),
    Embedding(Vec<f32>),
    /// Single video frame (RGB bytes, width*height*3).
    VideoFrame(Vec<u8>),
    /// Signals that video generation is complete with metadata.
    VideoComplete {
        width: u32,
        height: u32,
        fps: f32,
        frame_count: u32,
    },
    /// Diffusion denoising progress (step / total).
    DiffusionProgress {
        step: u32,
        total_steps: u32,
    },
    /// Action output from a policy model.
    Action {
        action_space: String,
        values: Vec<f32>,
    },
    /// State delta from a world model.
    StateDelta {
        from_state_id: String,
        to_state_id: String,
        latent_update: Option<Vec<u8>>,
    },
    /// Predicted future state from a world model.
    PredictedState {
        state_id: String,
        confidence: f32,
        horizon_step: u32,
    },
    Metrics {
        compute_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speculative_acceptance_rate: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speculative_draft_tokens: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speculative_accepted_tokens: Option<usize>,
    },
    /// Partially decoded ASR (Speech-to-Text) result.
    AsrPartial {
        text: String,
        tokens: Vec<u32>,
    },
    /// Audio chunk delta for Text-to-Speech (TTS).
    TtsAudioChunk {
        samples: Vec<f32>,
        sample_rate: u32,
        is_final: bool,
    },
    /// Vision Language Model (VLM) generated text token, optionally referencing predicted bounding boxes.
    VlmToken {
        text: String,
        bounding_box: Option<Vec<f32>>,
    },
    End,
}

impl OutputChunk {
    /// Validate an event emitted by a backend before it reaches a transport or
    /// caller-owned sink. This prevents a faulty/native backend from turning
    /// one request into an unbounded allocation or leaking NaN values.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::TextDelta(text) => {
                if text.len() > MAX_TEXT_BYTES {
                    Err("text output chunk exceeds the byte limit".to_string())
                } else {
                    Ok(())
                }
            }
            Self::Tokens(tokens) => {
                if tokens.len() > MAX_TOKEN_COUNT {
                    Err("token output chunk exceeds the token limit".to_string())
                } else {
                    Ok(())
                }
            }
            Self::AudioDelta(samples) => {
                if samples.len() > MAX_AUDIO_SAMPLES {
                    return Err("audio output chunk exceeds the sample limit".to_string());
                }
                if samples.iter().any(|sample| !sample.is_finite()) {
                    return Err("audio output chunk contains a non-finite sample".to_string());
                }
                Ok(())
            }
            Self::Image(bytes) => {
                if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
                    Err("image output chunk is empty or too large".to_string())
                } else {
                    Ok(())
                }
            }
            Self::Embedding(values) => validate_tensor(values, "embedding output"),
            Self::VideoFrame(frame) => {
                if frame.is_empty() || frame.len() > MAX_FRAME_BYTES {
                    Err("video output frame is empty or too large".to_string())
                } else {
                    Ok(())
                }
            }
            Self::VideoComplete {
                width,
                height,
                fps,
                frame_count,
            } => {
                if *width == 0 || *height == 0 || *width > 16_384 || *height > 16_384 {
                    return Err("video output dimensions are invalid".to_string());
                }
                if !fps.is_finite() || !(0.1..=240.0).contains(fps) || *frame_count == 0 {
                    return Err("video output metadata is invalid".to_string());
                }
                if usize::try_from(*frame_count).unwrap_or(usize::MAX) > MAX_VIDEO_FRAMES {
                    return Err("video output frame count exceeds the limit".to_string());
                }
                Ok(())
            }
            Self::DiffusionProgress { step, total_steps } => {
                if *total_steps == 0 || *step > *total_steps {
                    Err("diffusion progress is outside its declared range".to_string())
                } else {
                    Ok(())
                }
            }
            Self::Action {
                action_space,
                values,
            } => {
                bounded_text(action_space, "action space")?;
                validate_tensor(values, "action output")
            }
            Self::StateDelta {
                from_state_id,
                to_state_id,
                latent_update,
            } => {
                DataBlock::WorldState {
                    state_id: from_state_id.clone(),
                    latent: latent_update.clone(),
                    step: 0,
                }
                .validate()?;
                if to_state_id.is_empty() || to_state_id.len() > MAX_STATE_ID_BYTES {
                    return Err("state delta target id is empty or too long".to_string());
                }
                Ok(())
            }
            Self::PredictedState {
                state_id,
                confidence,
                ..
            } => {
                if state_id.is_empty() || state_id.len() > MAX_STATE_ID_BYTES {
                    return Err("predicted state id is empty or too long".to_string());
                }
                if !confidence.is_finite() || !(0.0..=1.0).contains(confidence) {
                    return Err("predicted state confidence must be between 0 and 1".to_string());
                }
                Ok(())
            }
            Self::Metrics {
                speculative_acceptance_rate,
                speculative_draft_tokens,
                speculative_accepted_tokens,
                ..
            } => {
                if let Some(rate) = speculative_acceptance_rate
                    && (!rate.is_finite() || !(0.0..=1.0).contains(rate))
                {
                    return Err("speculative acceptance rate must be between 0 and 1".to_string());
                }
                if speculative_accepted_tokens.unwrap_or(0)
                    > speculative_draft_tokens.unwrap_or(usize::MAX)
                {
                    return Err("accepted speculative tokens exceed draft tokens".to_string());
                }
                Ok(())
            }
            Self::AsrPartial { text, tokens } => {
                bounded_text(text, "ASR partial text")?;
                if tokens.len() > MAX_TOKEN_COUNT {
                    return Err("ASR partial token count exceeds the limit".to_string());
                }
                Ok(())
            }
            Self::TtsAudioChunk {
                samples,
                sample_rate,
                ..
            } => validate_audio(samples, *sample_rate, "TTS audio output"),
            Self::VlmToken { text, bounding_box } => {
                if !text.is_empty() {
                    bounded_text(text, "VLM token")?;
                }
                if let Some(bounding_box) = bounding_box
                    && (bounding_box.len() != 4
                        || bounding_box.iter().any(|value| !value.is_finite()))
                {
                    return Err("VLM bounding box must contain four finite values".to_string());
                }
                Ok(())
            }
            Self::End => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelInput {
    Text {
        prompt: String,
    },
    Audio {
        samples: Vec<f32>,
        sample_rate: u32,
    },
    AudioFile {
        path: String,
        language: Option<String>,
    },
    Vision {
        bytes: Vec<u8>,
        mime: String,
    },
    Multi {
        text: Option<String>,
        audio: Option<Vec<f32>>,
        image: Option<Vec<u8>>,
    },
    /// Video generation request (text-to-video).
    VideoGeneration {
        prompt: String,
        negative_prompt: Option<String>,
        width: u32,
        height: u32,
        num_frames: u32,
        fps: f32,
        guidance_scale: f64,
        num_steps: u32,
        seed: Option<u64>,
    },
}

impl ModelInput {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Text { prompt } => bounded_text(prompt, "prompt"),
            Self::Audio {
                samples,
                sample_rate,
            } => validate_audio(samples, *sample_rate, "audio input"),
            Self::AudioFile { path, language } => DataBlock::AudioFile {
                path: path.clone(),
                language: language.clone(),
            }
            .validate(),
            Self::Vision { bytes, mime } => validate_image(bytes, mime, "vision input"),
            Self::Multi { text, audio, image } => {
                if text.is_none() && audio.is_none() && image.is_none() {
                    return Err("multimodal input must contain at least one modality".to_string());
                }
                if let Some(text) = text {
                    bounded_text(text, "multimodal text")?;
                }
                if let Some(audio) = audio {
                    validate_audio(audio, 16_000, "multimodal audio")?;
                }
                if let Some(image) = image {
                    validate_image(image, "image/jpeg", "multimodal image")?;
                }
                Ok(())
            }
            Self::VideoGeneration {
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                fps,
                guidance_scale,
                num_steps,
                ..
            } => {
                bounded_text(prompt, "video prompt")?;
                if let Some(negative_prompt) = negative_prompt {
                    bounded_text(negative_prompt, "negative prompt")?;
                }
                if *width == 0 || *height == 0 || *width > 16_384 || *height > 16_384 {
                    return Err("video dimensions must be between 1 and 16384 pixels".to_string());
                }
                if *num_frames == 0
                    || usize::try_from(*num_frames).unwrap_or(usize::MAX) > MAX_VIDEO_FRAMES
                {
                    return Err(format!(
                        "video frame count must be between 1 and {MAX_VIDEO_FRAMES}"
                    ));
                }
                if !fps.is_finite() || !(0.1..=240.0).contains(fps) {
                    return Err("video fps must be finite and between 0.1 and 240".to_string());
                }
                if !guidance_scale.is_finite() || !(0.0..=100.0).contains(guidance_scale) {
                    return Err(
                        "video guidance_scale must be finite and between 0 and 100".to_string()
                    );
                }
                if *num_steps == 0 || *num_steps > 10_000 {
                    return Err("video num_steps must be between 1 and 10000".to_string());
                }
                Ok(())
            }
        }
    }
}

impl From<ModelInput> for InferenceRequest {
    fn from(input: ModelInput) -> Self {
        let mut blocks = Vec::new();
        match input {
            ModelInput::Text { prompt } => blocks.push(DataBlock::Text(prompt)),
            ModelInput::Audio {
                samples,
                sample_rate,
            } => blocks.push(DataBlock::AudioPcm {
                samples,
                sample_rate,
            }),
            ModelInput::AudioFile { path, language } => {
                blocks.push(DataBlock::AudioFile { path, language });
            }
            ModelInput::Vision { bytes, mime } => blocks.push(DataBlock::Image { bytes, mime }),
            ModelInput::Multi { text, audio, image } => {
                if let Some(t) = text {
                    blocks.push(DataBlock::Text(t));
                }
                if let Some(a) = audio {
                    blocks.push(DataBlock::AudioPcm {
                        samples: a,
                        sample_rate: 16000,
                    });
                }
                if let Some(i) = image {
                    blocks.push(DataBlock::Image {
                        bytes: i,
                        mime: "image/jpeg".to_string(),
                    });
                }
            }
            ModelInput::VideoGeneration { prompt, .. } => {
                blocks.push(DataBlock::Text(prompt));
            }
        }
        Self {
            blocks,
            params: InferenceParams::default(),
        }
    }
}

/// Video output metadata and frame data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOutput {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frames per second.
    pub fps: f32,
    /// Number of frames.
    pub frame_count: u32,
    /// Raw RGB frame data: Vec of (width * height * 3) bytes per frame.
    pub frames: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    pub text: Option<String>,
    pub logits: Option<Vec<f32>>,
    /// Image output from image generation or editing models.
    pub image: Option<Vec<u8>>,
    /// Audio output from TTS models: (pcm samples, sample_rate).
    pub audio: Option<(Vec<f32>, u32)>,
    /// Video output from video generation models.
    pub video: Option<VideoOutput>,
}

impl ModelOutput {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(text) = &self.text {
            bounded_text(text, "text output")?;
        }
        if let Some(logits) = &self.logits {
            validate_tensor(logits, "logits")?;
        }
        if let Some(image) = &self.image
            && (image.is_empty() || image.len() > MAX_IMAGE_BYTES)
        {
            return Err("image output is empty or too large".to_string());
        }
        if let Some((samples, sample_rate)) = &self.audio {
            validate_audio(samples, *sample_rate, "audio output")?;
        }
        if let Some(video) = &self.video {
            if video.frames.len() > MAX_VIDEO_FRAMES
                || video.frames.len() != usize::try_from(video.frame_count).unwrap_or(usize::MAX)
            {
                return Err("video output frame count does not match frames".to_string());
            }
            OutputChunk::VideoComplete {
                width: video.width,
                height: video.height,
                fps: video.fps,
                frame_count: video.frame_count,
            }
            .validate()?;
            for frame in &video.frames {
                OutputChunk::VideoFrame(frame.clone()).validate()?;
            }
        }
        Ok(())
    }
}

impl From<Vec<OutputChunk>> for ModelOutput {
    fn from(chunks: Vec<OutputChunk>) -> Self {
        let mut text = String::new();
        let mut image = None;
        let mut audio_samples = Vec::new();
        let mut audio_sr = 0u32;
        let mut video_frames: Vec<Vec<u8>> = Vec::new();
        let mut video_meta: Option<(u32, u32, f32, u32)> = None;
        for chunk in chunks {
            match chunk {
                OutputChunk::TextDelta(delta) => text.push_str(&delta),
                OutputChunk::Image(bytes) => image = Some(bytes),
                OutputChunk::AudioDelta(samples) => audio_samples.extend(samples),
                OutputChunk::TtsAudioChunk {
                    samples,
                    sample_rate,
                    ..
                } => {
                    if audio_sr == 0 {
                        audio_sr = sample_rate;
                    }
                    audio_samples.extend(samples);
                }
                OutputChunk::VideoFrame(frame) => video_frames.push(frame),
                OutputChunk::VideoComplete {
                    width,
                    height,
                    fps,
                    frame_count,
                } => {
                    video_meta = Some((width, height, fps, frame_count));
                }
                _ => {}
            }
        }
        let video = video_meta.map(|(w, h, fps, fc)| VideoOutput {
            width: w,
            height: h,
            fps,
            frame_count: fc,
            frames: video_frames,
        });
        Self {
            text: if text.is_empty() { None } else { Some(text) },
            logits: None,
            image,
            audio: if audio_samples.is_empty() {
                None
            } else {
                Some((audio_samples, audio_sr))
            },
            video,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_input_text_serde() {
        let input = ModelInput::Text {
            prompt: "hello".to_string(),
        };
        let serialized = serde_json::to_string(&input).unwrap();
        assert!(serialized.contains("Text"));
        assert!(serialized.contains("prompt"));
        assert!(serialized.contains("hello"));

        let deserialized: ModelInput = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            ModelInput::Text { prompt } => assert_eq!(prompt, "hello"),
            _ => panic!("Expected ModelInput::Text"),
        }
    }

    #[test]
    fn test_model_input_audio_serde() {
        let input = ModelInput::Audio {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 16000,
        };
        let serialized = serde_json::to_string(&input).unwrap();
        assert!(serialized.contains("Audio"));
        assert!(serialized.contains("samples"));
        assert!(serialized.contains("sample_rate"));

        let deserialized: ModelInput = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            ModelInput::Audio {
                samples,
                sample_rate,
            } => {
                assert_eq!(samples, vec![0.1, 0.2, 0.3]);
                assert_eq!(sample_rate, 16000);
            }
            _ => panic!("Expected ModelInput::Audio"),
        }
    }

    #[test]
    fn test_model_input_vision_serde() {
        let input = ModelInput::Vision {
            bytes: vec![1, 2, 3, 4],
            mime: "image/png".to_string(),
        };
        let serialized = serde_json::to_string(&input).unwrap();
        assert!(serialized.contains("Vision"));
        assert!(serialized.contains("bytes"));
        assert!(serialized.contains("mime"));

        let deserialized: ModelInput = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            ModelInput::Vision { bytes, mime } => {
                assert_eq!(bytes, vec![1, 2, 3, 4]);
                assert_eq!(mime, "image/png");
            }
            _ => panic!("Expected ModelInput::Vision"),
        }
    }

    #[test]
    fn test_model_input_multi_serde() {
        let input = ModelInput::Multi {
            text: Some("hello".to_string()),
            audio: Some(vec![0.5, 0.6]),
            image: None,
        };
        let serialized = serde_json::to_string(&input).unwrap();
        assert!(serialized.contains("Multi"));
        assert!(serialized.contains("text"));
        assert!(serialized.contains("audio"));

        let deserialized: ModelInput = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            ModelInput::Multi { text, audio, image } => {
                assert_eq!(text, Some("hello".to_string()));
                assert_eq!(audio, Some(vec![0.5, 0.6]));
                assert_eq!(image, None);
            }
            _ => panic!("Expected ModelInput::Multi"),
        }
    }

    #[test]
    fn test_model_output_serde() {
        let output = ModelOutput {
            text: Some("result".to_string()),
            logits: Some(vec![0.9, 0.1]),
            image: None,
            audio: None,
            video: None,
        };
        let serialized = serde_json::to_string(&output).unwrap();
        assert!(serialized.contains("text"));
        assert!(serialized.contains("logits"));

        let deserialized: ModelOutput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.text, Some("result".to_string()));
        assert_eq!(deserialized.logits, Some(vec![0.9, 0.1]));
    }

    #[test]
    fn test_model_output_from_tts_chunks_preserves_sample_rate() {
        let output = ModelOutput::from(vec![
            OutputChunk::TtsAudioChunk {
                samples: vec![0.1, 0.2],
                sample_rate: 24000,
                is_final: false,
            },
            OutputChunk::TtsAudioChunk {
                samples: vec![0.3],
                sample_rate: 24000,
                is_final: true,
            },
            OutputChunk::End,
        ]);

        assert_eq!(output.audio, Some((vec![0.1, 0.2, 0.3], 24000)));
    }

    #[test]
    fn test_model_output_from_audio_delta_uses_unknown_sample_rate() {
        let output = ModelOutput::from(vec![OutputChunk::AudioDelta(vec![0.4, 0.5])]);

        assert_eq!(output.audio, Some((vec![0.4, 0.5], 0)));
    }

    #[test]
    fn validation_rejects_unbounded_or_non_finite_blocks() {
        let oversized = DataBlock::Text("x".repeat(MAX_TEXT_BYTES + 1));
        assert!(oversized.validate().unwrap_err().contains("byte limit"));

        let invalid_audio = DataBlock::AudioPcm {
            samples: vec![f32::NAN],
            sample_rate: 16_000,
        };
        assert!(invalid_audio.validate().unwrap_err().contains("non-finite"));

        let invalid_image = DataBlock::Image {
            bytes: vec![1],
            mime: "text/plain".to_string(),
        };
        assert!(invalid_image.validate().unwrap_err().contains("MIME"));
    }

    #[test]
    fn request_validation_is_shared_by_direct_callers() {
        let request = InferenceRequest {
            blocks: vec![DataBlock::Text("hello".to_string())],
            params: InferenceParams {
                max_tokens: 0,
                ..InferenceParams::default()
            },
        };
        assert!(request.validate().unwrap_err().contains("max_tokens"));
    }

    #[test]
    fn model_input_validation_covers_video_bounds() {
        let input = ModelInput::VideoGeneration {
            prompt: "video".to_string(),
            negative_prompt: None,
            width: 64,
            height: 64,
            num_frames: (MAX_VIDEO_FRAMES as u32) + 1,
            fps: 24.0,
            guidance_scale: 7.5,
            num_steps: 20,
            seed: None,
        };
        assert!(input.validate().unwrap_err().contains("frame count"));
    }

    #[test]
    fn output_validation_rejects_non_finite_embeddings_and_bad_terminal_metadata() {
        let embedding = OutputChunk::Embedding(vec![1.0, f32::INFINITY]);
        assert!(embedding.validate().unwrap_err().contains("non-finite"));

        let progress = OutputChunk::DiffusionProgress {
            step: 4,
            total_steps: 3,
        };
        assert!(progress.validate().unwrap_err().contains("range"));

        let output = ModelOutput {
            text: None,
            logits: None,
            image: None,
            audio: None,
            video: Some(VideoOutput {
                width: 4,
                height: 4,
                fps: 24.0,
                frame_count: 2,
                frames: vec![vec![0; 4 * 4 * 3]],
            }),
        };
        assert!(output.validate().unwrap_err().contains("does not match"));
    }
}
