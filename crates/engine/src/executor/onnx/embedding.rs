use anyhow::{Result, anyhow, ensure};
use bloomai_core::ModelManifest;

/// An explicit export contract, never inferred from an arbitrary first output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub pooling: String,
    pub output_name: String,
    pub max_length: usize,
    pub pad_token_id: u32,
    #[serde(default)]
    pub normalize: bool,
    #[serde(default)]
    pub truncate: bool,
}
impl Config {
    pub fn from_manifest(manifest: &ModelManifest) -> Result<Option<Self>> {
        if !crate::model_manifest_supports_embeddings(manifest) {
            return Ok(None);
        }
        let config: Self = serde_json::from_value(
            manifest
                .parameters
                .get("onnx_embedding")
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "ONNX encoder requires explicit parameters.onnx_embedding configuration"
                    )
                })?,
        )?;
        ensure!(
            matches!(config.pooling.as_str(), "mean" | "cls" | "none"),
            "ONNX embedding pooling must be mean, cls or none"
        );
        ensure!(
            !config.output_name.is_empty() && config.output_name.len() <= 256,
            "ONNX embedding output_name is invalid"
        );
        ensure!(
            (1..=32768).contains(&config.max_length),
            "ONNX embedding max_length exceeds token limit"
        );
        Ok(Some(config))
    }
}

pub(super) fn validate_inputs(inputs: &[String]) -> Result<()> {
    ensure!(
        !inputs.is_empty() && inputs.len() <= 128,
        "ONNX embedding batch must contain 1 to 128 inputs"
    );
    for text in inputs {
        ensure!(
            !text.trim().is_empty() && text.len() <= 32 * 1024,
            "ONNX embedding input must be nonempty and at most 32 KiB"
        );
    }
    Ok(())
}

pub(super) fn parse_response(
    value: serde_json::Value,
    count: usize,
    max_length: usize,
) -> Result<crate::model::EmbeddingBatchOutput> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Response {
        embeddings: Vec<Vec<f32>>,
        prompt_tokens: u64,
        usage_source: String,
    }
    let output: Response = serde_json::from_value(value)?;
    ensure!(
        output.usage_source == "input_ids_attention_mask",
        "ONNX bridge did not report input-ID token usage"
    );
    ensure!(
        output.prompt_tokens > 0 && output.prompt_tokens <= count.saturating_mul(max_length) as u64,
        "ONNX bridge returned invalid token usage"
    );
    ensure!(
        output.embeddings.len() == count,
        "ONNX bridge returned wrong embedding count"
    );
    let dim = output.embeddings.first().map_or(0, Vec::len);
    ensure!(
        dim > 0 && count.saturating_mul(dim) <= crate::MAX_TENSOR_VALUES,
        "ONNX bridge returned invalid embedding dimensions"
    );
    ensure!(
        output
            .embeddings
            .iter()
            .all(|row| row.len() == dim && row.iter().all(|v| v.is_finite())),
        "ONNX bridge returned inconsistent/nonfinite embeddings"
    );
    Ok(crate::model::EmbeddingBatchOutput {
        embeddings: output.embeddings,
        prompt_tokens: Some(output.prompt_tokens),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn exact_bridge_usage_requires_valid_source_shape_and_count() {
        let valid = json!({"embeddings":[[1.0,2.0],[3.0,4.0]], "prompt_tokens":8,
            "usage_source":"input_ids_attention_mask"});
        assert_eq!(
            parse_response(valid.clone(), 2, 16).unwrap().prompt_tokens,
            Some(8)
        );
        for (key, value) in [
            ("prompt_tokens", json!(-1)),
            ("prompt_tokens", json!(0)),
            ("prompt_tokens", json!(33)),
            ("prompt_tokens", json!(1.5)),
            ("usage_source", json!("characters")),
            ("embeddings", json!([[1.0, 2.0]])),
            ("embeddings", json!([[1.0, 2.0], [3.0]])),
            ("embeddings", json!([[], []])),
        ] {
            let mut invalid = valid.clone();
            invalid[key] = value;
            assert!(parse_response(invalid, 2, 16).is_err(), "accepted {key}");
        }
        let mut absent = valid;
        absent.as_object_mut().unwrap().remove("prompt_tokens");
        assert!(parse_response(absent, 2, 16).is_err());
    }

    #[test]
    fn generic_onnx_is_not_misclassified_as_an_encoder() {
        let mut manifest = ModelManifest::default();
        assert!(Config::from_manifest(&manifest).unwrap().is_none());
        manifest
            .parameters
            .insert("bloom_task".into(), json!("embedding"));
        assert!(Config::from_manifest(&manifest).is_err());
        manifest.parameters.insert(
            "onnx_embedding".into(),
            json!({
                "pooling":"mean", "output_name":"hidden_states", "max_length":128, "pad_token_id":0
            }),
        );
        assert!(Config::from_manifest(&manifest).unwrap().is_some());
        manifest.parameters.get_mut("onnx_embedding").unwrap()["max_length"] = json!(32769);
        assert!(Config::from_manifest(&manifest).is_err());
    }
}
