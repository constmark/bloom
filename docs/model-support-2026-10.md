# Small-model support added in October 2026

This record describes the model-family routing added in this iteration. A
route is executable when Bloom can classify the package, select a compatible
backend, and fail before load when the package needs an unavailable native
implementation. It is still `experimental` until a pinned model revision,
license payload, real inference, protocol smoke, and benchmark record are
published as described by the [support matrix](support-matrix.md).

## Native Candle paths

The existing Qwen2, Qwen3, Llama, and BERT loaders now cover these small-model
families without requiring a model-specific name list:

| Family | Examples | Route |
| --- | --- | --- |
| Qwen2 | Qwen2.5-1.5B/3B-Instruct; DeepSeek-R1-Distill-Qwen-1.5B | `model_type=qwen2` and Qwen2 aliases; DeepSeek's Qwen2 config is treated as Qwen2 |
| Qwen3 | Qwen3-1.7B/4B | `model_type=qwen3`, including current RoPE and missing-default normalization |
| Llama | Llama 3.2 1B/3B-Instruct | `model_type=llama` and Llama 3 aliases; Llama 3 chat templates are classified as `llama3` |
| BERT embeddings | BAAI/bge-small-en-v1.5; intfloat/multilingual-e5-small | Sentence-Transformers metadata with either CLS-only or mean-token pooling |
| BERT reranking | cross-encoder/ms-marco-MiniLM-L6-v2 | `BertForSequenceClassification`; `id2label`/`label2id` can supply the classifier width when `num_labels` is absent |

BGE's CLS pooling is now preserved through manifest inference and the Candle
embedding batch path. E5 and the existing MiniLM profile continue to use
masked mean pooling. Unsupported or mixed Sentence-Transformers pooling modes
remain rejected before model loading.

## Multimodal path

`model_type=qwen3_vl` receives an explicit `qwen3_vl` preferred backend and is
selected by both the CLI and server auto-routing. The native executor accepts
text plus image input and Safetensors weights. Qwen3-VL remains experimental;
the 2B checkpoint still needs a pinned trained-model quality and hardware
record before promotion.

## External llama.cpp path

GGUF metadata for the following architectures is classified as an external
runtime package and automatically routed to `llamacpp` by the CLI and server:

- Phi-3/Phi-4 (`phi3`, `phi3moe`, `phi4`, `phi4mm`)
- Gemma 3 (`gemma3`)
- SmolLM3 (`smollm3`)
- Ministral/Mistral 3 (`ministral3`, `mistral3`)
- Granite 3/3.3 (`granite`, `granite3`, `granite3.3`)
- Liquid LFM2 (`lfm2`)

Bloom does not claim a native Candle loader for these families. A recent
`llama-server` with the relevant GGUF architecture is required, and the
external adapter currently exposes text completion/streaming only. Conversion
from Safetensors to GGUF, model-specific chat templates, and the selected
llama.cpp version remain deployment inputs.

## Verification

The routing and metadata contracts are covered by focused tests:

```bash
cargo test -p bloomai-engine --lib native_text_config_normalizes_current_hf_aliases
cargo test -p bloomai-engine --lib gguf_external_architectures_declare_llamacpp_route
cargo test -p bloomai-engine --lib hf_bge_cls_pooling_fixture_routes_to_embedding_runtime
cargo test -p bloomai-engine --lib cross_encoder_num_labels_accepts_legacy_id2label_metadata
cargo test -p bloomai-app --bin bloom_infer select_backend_routes_external_gguf_architectures_to_llamacpp
cargo test -p bloomai-server select_backend_routes_manifest_external_runtime_hints
```

These tests validate routing and shape metadata without storing model weights
in the repository. Real-model promotion still uses the pinned validation
scripts and requires an exact revision and SHA-256.
