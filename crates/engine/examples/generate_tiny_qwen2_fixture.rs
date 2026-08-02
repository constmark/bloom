//! Generate a deterministic, untrained Qwen2 fixture for CPU runtime tests.
//!
//! The fixture is deliberately tiny and is not useful for language quality
//! evaluation. It contains a real tokenizer, Safetensors checkpoint, and
//! Qwen2 configuration so integration tests exercise Bloom's native Candle
//! loading, forward, decoding, embedding, reranking, function calling, and HTTP
//! streaming paths without downloading third-party model weights.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Shape, Tensor};
use clap::{Parser, ValueEnum};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

const HIDDEN_SIZE: usize = 8;
const INTERMEDIATE_SIZE: usize = 16;
const NUM_ATTENTION_HEADS: usize = 2;
const NUM_KEY_VALUE_HEADS: usize = 1;
const NUM_HIDDEN_LAYERS: usize = 1;
const MAX_POSITION_EMBEDDINGS: usize = 512;
const PREDICTED_TOKEN_ID: usize = 1;
const EOS_TOKEN: &str = "<eos>";
const STRUCTURED_TOKEN: &str = r#"{"ok":true}"#;
const TOOL_CALL_TOKEN: &str =
    r#"{"type":"function_calls","calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}"#;

const VOCABULARY: &[&str] = &[
    "<unk>",
    "fixture",
    "hello",
    "Bloom",
    "{",
    "}",
    "\"",
    ":",
    "true",
    "false",
    "ok",
    ",",
    "[",
    "]",
    "Paris",
    ".",
    "local",
    "AI",
    "runtime",
    "bounded",
    "retrieval",
    "banana",
    "orchard",
    EOS_TOKEN,
    STRUCTURED_TOKEN,
    TOOL_CALL_TOKEN,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum FixtureProfile {
    /// Repeated ordinary text generation.
    #[default]
    Text,
    /// Distinct mechanical embedding vectors with an explicit task marker.
    Embedding,
    /// One schema-valid JSON object followed by EOS.
    Structured,
    /// One valid function call followed by EOS.
    Tool,
}

#[derive(Debug, Parser)]
#[command(
    name = "generate_tiny_qwen2_fixture",
    about = "Generate Bloom's deterministic untrained Qwen2 CPU test fixture"
)]
struct Args {
    /// New directory that will receive the model fixture files.
    #[arg(long)]
    output: PathBuf,

    /// Deterministic behavior exercised by the generated weights.
    #[arg(long, value_enum, default_value_t)]
    profile: FixtureProfile,

    /// Write two indexed model-0000N-of-00002.safetensors shards.
    #[arg(long)]
    sharded: bool,
}

fn tensor_zeros<S: Into<Shape>>(shape: S, device: &Device) -> Result<Tensor> {
    Tensor::zeros(shape, DType::F32, device).map_err(Into::into)
}

fn tensor_ones<S: Into<Shape>>(shape: S, device: &Device) -> Result<Tensor> {
    Tensor::ones(shape, DType::F32, device).map_err(Into::into)
}

fn vocabulary_id(token: &str) -> Result<usize> {
    VOCABULARY
        .iter()
        .position(|candidate| *candidate == token)
        .ok_or_else(|| anyhow!("fixture vocabulary omitted {token:?}"))
}

fn deterministic_output_token(profile: FixtureProfile) -> Option<&'static str> {
    match profile {
        FixtureProfile::Text | FixtureProfile::Embedding => None,
        FixtureProfile::Structured => Some(STRUCTURED_TOKEN),
        FixtureProfile::Tool => Some(TOOL_CALL_TOKEN),
    }
}

fn build_tensors(profile: FixtureProfile) -> Result<HashMap<String, Tensor>> {
    let device = Device::Cpu;
    let vocab_size = VOCABULARY.len();
    let head_dim = HIDDEN_SIZE / NUM_ATTENTION_HEADS;
    let kv_size = NUM_KEY_VALUE_HEADS * head_dim;
    let mut tensors = HashMap::new();

    // Zero attention/MLP projections preserve the token residual while still
    // executing every transformer operation. The text profile uses positive
    // embeddings with two token-specific dimensions so the rerank gate sees
    // distinct directions instead of universally tied vectors. Deterministic
    // output profiles use two orthogonal states: ordinary prompt tokens select
    // the complete JSON token, which then selects EOS.
    let mut embeddings = match deterministic_output_token(profile) {
        None => vec![1.0_f32; vocab_size * HIDDEN_SIZE],
        Some(output_token) => {
            let mut values = vec![0.0_f32; vocab_size * HIDDEN_SIZE];
            for row in values.chunks_exact_mut(HIDDEN_SIZE) {
                row[0] = 1.0;
            }
            let output_row = vocabulary_id(output_token)? * HIDDEN_SIZE;
            values[output_row..output_row + HIDDEN_SIZE].fill(0.0);
            values[output_row + 1] = 1.0;
            values
        }
    };
    if matches!(profile, FixtureProfile::Text | FixtureProfile::Embedding) {
        for (token, dimension) in [("runtime", 0_usize), ("orchard", 1_usize)] {
            let token_id = vocabulary_id(token)?;
            embeddings[token_id * HIDDEN_SIZE + dimension] = 2.0;
        }
    }
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::from_vec(embeddings, (vocab_size, HIDDEN_SIZE), &device)?,
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        tensor_ones(HIDDEN_SIZE, &device)?,
    );

    for layer in 0..NUM_HIDDEN_LAYERS {
        let prefix = format!("model.layers.{layer}");
        for (projection, output_size) in [
            ("q_proj", HIDDEN_SIZE),
            ("k_proj", kv_size),
            ("v_proj", kv_size),
        ] {
            tensors.insert(
                format!("{prefix}.self_attn.{projection}.weight"),
                tensor_zeros((output_size, HIDDEN_SIZE), &device)?,
            );
            tensors.insert(
                format!("{prefix}.self_attn.{projection}.bias"),
                tensor_zeros(output_size, &device)?,
            );
        }
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            tensor_zeros((HIDDEN_SIZE, HIDDEN_SIZE), &device)?,
        );
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            tensor_zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), &device)?,
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            tensor_zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), &device)?,
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            tensor_zeros((HIDDEN_SIZE, INTERMEDIATE_SIZE), &device)?,
        );
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            tensor_ones(HIDDEN_SIZE, &device)?,
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            tensor_ones(HIDDEN_SIZE, &device)?,
        );
    }

    let mut lm_head = vec![0.0_f32; vocab_size * HIDDEN_SIZE];
    match profile {
        FixtureProfile::Text | FixtureProfile::Embedding => {
            // One positive row produces the ordinary token "fixture" rather
            // than EOS. The smaller feature rows keep embedding directions
            // observable through the causal LM head.
            let row = PREDICTED_TOKEN_ID * HIDDEN_SIZE;
            lm_head[row..row + HIDDEN_SIZE].fill(1.0);
            lm_head[2 * HIDDEN_SIZE] = 0.25;
            lm_head[3 * HIDDEN_SIZE + 1] = 0.25;
        }
        FixtureProfile::Structured | FixtureProfile::Tool => {
            let output_token = deterministic_output_token(profile)
                .ok_or_else(|| anyhow!("output profile omitted its deterministic token"))?;
            lm_head[vocabulary_id(output_token)? * HIDDEN_SIZE] = 2.0;
            lm_head[vocabulary_id(EOS_TOKEN)? * HIDDEN_SIZE + 1] = 2.0;
        }
    }
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(lm_head, (vocab_size, HIDDEN_SIZE), &device)?,
    );
    Ok(tensors)
}

fn write_config(output: &Path, profile: FixtureProfile) -> Result<()> {
    let eos_token_id = match profile {
        FixtureProfile::Text | FixtureProfile::Embedding => 100,
        FixtureProfile::Structured | FixtureProfile::Tool => vocabulary_id(EOS_TOKEN)?,
    };
    let mut config = serde_json::json!({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": VOCABULARY.len(),
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "sliding_window": MAX_POSITION_EMBEDDINGS,
        "max_window_layers": NUM_HIDDEN_LAYERS,
        "tie_word_embeddings": false,
        "rope_theta": 10000.0,
        "rms_norm_eps": 0.000001,
        "use_sliding_window": false,
        "hidden_act": "silu",
        "bos_token_id": 2,
        "eos_token_id": eos_token_id,
        "torch_dtype": "float32"
    });
    if profile == FixtureProfile::Embedding {
        config["bloom_task"] = serde_json::json!("embedding");
    }
    let bytes = serde_json::to_vec_pretty(&config).context("failed to encode fixture config")?;
    fs::write(output.join("config.json"), bytes).context("failed to write fixture config")
}

fn write_tokenizer(output: &Path) -> Result<()> {
    let vocab = VOCABULARY
        .iter()
        .enumerate()
        .map(|(id, token)| ((*token).to_string(), id as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token(VOCABULARY[0].to_string())
        .build()
        .map_err(|error| anyhow!("failed to build fixture tokenizer: {error}"))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .save(output.join("tokenizer.json"), true)
        .map_err(|error| anyhow!("failed to write fixture tokenizer: {error}"))
}

fn write_weights(output: &Path, profile: FixtureProfile, sharded: bool) -> Result<()> {
    let tensors = build_tensors(profile)?;
    if !sharded {
        return candle_core::safetensors::save(&tensors, output.join("model.safetensors"))
            .context("failed to write fixture Safetensors checkpoint");
    }

    let total_size = tensors.values().try_fold(0_u64, |total, tensor| {
        let bytes = tensor
            .elem_count()
            .checked_mul(tensor.dtype().size_in_bytes())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| anyhow!("fixture tensor byte count overflow"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("fixture checkpoint byte count overflow"))
    })?;
    let mut tensor_names = tensors.keys().cloned().collect::<Vec<_>>();
    tensor_names.sort();
    let shard_names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let mut shards = [HashMap::new(), HashMap::new()];
    let mut weight_map = BTreeMap::new();
    for (position, tensor_name) in tensor_names.into_iter().enumerate() {
        let shard_index = position % shards.len();
        let tensor = tensors
            .get(&tensor_name)
            .ok_or_else(|| anyhow!("fixture tensor disappeared: {tensor_name}"))?
            .clone();
        shards[shard_index].insert(tensor_name.clone(), tensor);
        weight_map.insert(tensor_name, shard_names[shard_index]);
    }
    for (shard_name, shard) in shard_names.iter().zip(&shards) {
        candle_core::safetensors::save(shard, output.join(shard_name))
            .with_context(|| format!("failed to write fixture Safetensors shard {shard_name}"))?;
    }
    let index = serde_json::json!({
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    });
    fs::write(
        output.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).context("failed to encode fixture Safetensors index")?,
    )
    .context("failed to write fixture Safetensors index")
}

fn generate(output: &Path, profile: FixtureProfile, sharded: bool) -> Result<()> {
    if output.exists() {
        return Err(anyhow!(
            "output directory already exists: {}; refusing to overwrite it",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(anyhow!(
            "output parent directory does not exist: {}",
            parent.display()
        ));
    }

    fs::create_dir(output).with_context(|| {
        format!(
            "failed to create fixture output directory {}",
            output.display()
        )
    })?;
    write_config(output, profile)?;
    write_tokenizer(output)?;
    write_weights(output, profile, sharded)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    generate(&args.output, args.profile, args.sharded)?;
    println!(
        "Generated deterministic untrained Qwen2 {:?} {} fixture at {}",
        args.profile,
        if args.sharded {
            "sharded"
        } else {
            "single-file"
        },
        args.output.display()
    );
    Ok(())
}
