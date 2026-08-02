use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{DType, Modality, ModelFormat, QuantizationInfo};

fn default_schema_version() -> String {
    "1".to_string()
}

fn default_processor_version() -> String {
    "1".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelFamily {
    Llama,
    Qwen,
    Gemma,
    Bert,
    Whisper,
    FunAsr,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub name: String,
    pub format: ModelFormat,
    pub size_bytes: usize,
    pub hash_sha256: Option<String>,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMemoryProfile {
    pub min_ram_bytes: usize,
    pub min_vram_bytes: usize,
    pub recommended_ram_bytes: usize,
    pub recommended_vram_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeHints {
    #[serde(default)]
    pub preferred_backends: Vec<String>,
    #[serde(default)]
    pub supports_mmap: bool,
    #[serde(default)]
    pub requires_streaming: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIoSchema {
    #[serde(default)]
    pub inputs: Vec<Modality>,
    #[serde(default)]
    pub outputs: Vec<Modality>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessorSpec {
    pub name: String,
    pub kind: ProcessorKind,
    #[serde(default = "default_processor_version")]
    pub version: String,
    #[serde(default)]
    pub inputs: Vec<Modality>,
    #[serde(default)]
    pub outputs: Vec<Modality>,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProcessorKind {
    TextTokenizer,
    Audio,
    Image,
    Video,
    Tensor,
    WorldState,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub id: String,
    pub family: ModelFamily,
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub source_url: Option<String>,
    pub io_schema: ModelIoSchema,
    pub memory_profile: ModelMemoryProfile,
    #[serde(default)]
    pub files: Vec<ModelFile>,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    pub runtime_hints: RuntimeHints,
    pub primary_dtype: DType,
    /// Quantization configuration, if the model uses quantized weights.
    #[serde(default)]
    pub quantization: Option<QuantizationInfo>,
    #[serde(default)]
    pub processors: Vec<ProcessorSpec>,
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            id: "unknown".to_string(),
            family: ModelFamily::Custom("unknown".to_string()),
            version: "0.1.0".to_string(),
            description: None,
            license: None,
            source_url: None,
            io_schema: ModelIoSchema {
                inputs: vec![Modality::Text],
                outputs: vec![Modality::Text],
            },
            memory_profile: ModelMemoryProfile {
                min_ram_bytes: 0,
                min_vram_bytes: 0,
                recommended_ram_bytes: 0,
                recommended_vram_bytes: 0,
            },
            files: vec![],
            parameters: HashMap::new(),
            runtime_hints: RuntimeHints {
                preferred_backends: vec![],
                supports_mmap: false,
                requires_streaming: false,
            },
            primary_dtype: DType::F32,
            quantization: None,
            processors: vec![],
        }
    }
}
