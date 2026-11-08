/// A small, untrained checkpoint exercises the actual native loader and
/// forward pass without network access or production model weights.
pub(crate) fn tiny_checkpoint() -> tempfile::TempDir {
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("encoder")).unwrap();
    std::fs::create_dir_all(dir.path().join("tokenizer")).unwrap();
    let config = serde_json::json!({
        "model_type": "modernbert", "hidden_size": 8,
        "vocab_size": 5, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": 16,
        "max_position_embeddings": 64, "pad_token_id": 0,
        "torch_dtype": "float32"
    });
    std::fs::write(dir.path().join("encoder/config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.path().join("rl_agent_config.json"),
        r#"{"max_len":64,"head_max_len":32,"head_layers":1}"#,
    )
    .unwrap();
    let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
    let vocab = specials
        .iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), id as u32))
        .collect();
    let wordlevel = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(wordlevel);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer
        .add_special_tokens(
            specials
                .iter()
                .map(|token| tokenizers::AddedToken::from(*token, true)),
        )
        .unwrap();
    tokenizer
        .save(dir.path().join("tokenizer/tokenizer.json"), false)
        .unwrap();

    let mut weights = HashMap::<String, Tensor>::new();
    let mut add = |name: &str, shape: &[usize], norm: bool| {
        let tensor = if norm {
            Tensor::ones(shape, DType::F32, &Device::Cpu)
        } else {
            Tensor::zeros(shape, DType::F32, &Device::Cpu)
        }
        .unwrap();
        weights.insert(name.to_string(), tensor);
    };
    add("encoder.embeddings.tok_embeddings.weight", &[5, 8], false);
    add("encoder.embeddings.norm.weight", &[8], true);
    add("encoder.final_norm.weight", &[8], true);
    add("encoder.layers.0.attn.Wqkv.weight", &[24, 8], false);
    add("encoder.layers.0.attn.Wo.weight", &[8, 8], false);
    add("encoder.layers.0.attn_norm.weight", &[8], true);
    add("encoder.layers.0.mlp_norm.weight", &[8], true);
    add("encoder.layers.0.mlp.Wi.weight", &[32, 8], false);
    add("encoder.layers.0.mlp.Wo.weight", &[8, 16], false);
    add("type_emb.weight", &[3, 8], false);
    add("head.layers.0.self_attn.in_proj_weight", &[24, 8], false);
    add("head.layers.0.self_attn.in_proj_bias", &[24], false);
    for (name, out, input) in [
        ("head.layers.0.self_attn.out_proj", 8, 8),
        ("head.layers.0.linear1", 32, 8),
        ("head.layers.0.linear2", 8, 32),
        ("scorer.1", 8, 8),
        ("scorer.3", 1, 8),
        ("act_head.0", 256, 12),
        ("act_head.2", 1, 256),
    ] {
        add(&format!("{name}.weight"), &[out, input], false);
        add(&format!("{name}.bias"), &[out], false);
    }
    for name in ["head.layers.0.norm1", "head.layers.0.norm2", "scorer.0"] {
        add(&format!("{name}.weight"), &[8], true);
        add(&format!("{name}.bias"), &[8], false);
    }
    candle_core::safetensors::save(&weights, dir.path().join("model.safetensors")).unwrap();
    dir
}
