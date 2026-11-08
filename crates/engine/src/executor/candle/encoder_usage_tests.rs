use super::*;

#[test]
fn native_bert_usage_includes_special_tokens_and_excludes_batch_padding() {
    let directory = tempfile::tempdir().unwrap();
    let config = candle_transformers::models::bert::Config {
        vocab_size: 8,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        hidden_dropout_prob: 0.0,
        max_position_embeddings: 16,
        type_vocab_size: 2,
        pad_token_id: 0,
        classifier_dropout: None,
        model_type: Some("bert".into()),
        ..Default::default()
    };
    let variables = candle_nn::VarMap::new();
    let builder = candle_nn::VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    let _model = candle_transformers::models::bert::BertModel::load(builder, &config).unwrap();
    variables
        .save(directory.path().join("model.safetensors"))
        .unwrap();
    std::fs::write(
        directory.path().join("config.json"),
        serde_json::json!({
            "model_type":"bert", "architectures":["BertModel"], "vocab_size":8,
            "hidden_size":8, "num_hidden_layers":1, "num_attention_heads":2,
            "intermediate_size":16, "hidden_dropout_prob":0.0, "max_position_embeddings":16,
            "type_vocab_size":2, "pad_token_id":0, "hidden_act":"gelu", "layer_norm_eps":1e-12,
            "initializer_range":0.02
        })
        .to_string(),
    )
    .unwrap();
    let vocab = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "a", "b",
    ]
    .into_iter()
    .enumerate()
    .map(|(id, text)| (text.to_string(), id as u32))
    .collect();
    let wordlevel = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(wordlevel);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.with_post_processor(Some(tokenizers::processors::bert::BertProcessing::new(
        ("[SEP]".into(), 3),
        ("[CLS]".into(), 2),
    )));
    tokenizer
        .save(directory.path().join("tokenizer.json"), false)
        .unwrap();
    let model = CandleEngine
        .load(directory.path(), DeviceKind::Cpu)
        .unwrap();
    let inputs: Vec<String> = vec!["hello world".into(), "hello".into()];
    assert_eq!(model.tokenize(&inputs[0]).unwrap().unwrap(), [2, 4, 5, 3]);
    assert_eq!(model.embedding_token_count(&inputs).unwrap().unwrap(), 7);
    let vectors = model.embed_batch(&inputs).unwrap();
    assert_eq!(vectors.len(), 2);
    assert!(
        vectors
            .iter()
            .all(|vector| vector.len() == 8 && vector.iter().all(|value| value.is_finite()))
    );
}
