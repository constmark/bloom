use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub blocks: Vec<DataBlock>,
    pub params: InferenceParams,
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
}
