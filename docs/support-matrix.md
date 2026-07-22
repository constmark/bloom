# Support Matrix

This page describes executable support in the current repository. Adapter code
or file detection alone does not mean that inference works.

## Status

| Status | Meaning |
| --- | --- |
| `stable` | Reproducible real-model validation and benchmark evidence are published |
| `experimental` | The path executes but still needs broader model or hardware coverage |
| `external-runtime` | Execution requires another process, SDK, or hardware runtime |
| `skeleton` | Inspection and capability diagnostics only; no end-to-end inference |
| `blocked` | The required loader, fixture, dependency, or hardware is unavailable |

## Models and formats

| Model or format | Engine | Status | Notes |
| --- | --- | --- | --- |
| Qwen text, Safetensors | Candle | `experimental` | Native CPU path; Metal and CUDA use feature flags |
| Llama, Qwen2, and Qwen3 GGUF | Candle | `experimental` | Architecture-specific quantized loaders |
| Gemma text, Safetensors | Candle | `experimental` | Native execution |
| Gemma, Mistral, and DeepSeek GGUF | Candle | `blocked` | Requires architecture-specific loaders |
| FunASR and Qwen ASR | Python bridge | `external-runtime` | Requires the model-specific Python environment |
| OpenVINO IR | OpenVINO | `external-runtime` | Requires OpenVINO and exported IR files |
| GGUF fallback and MTP | llama.cpp | `external-runtime` | Launches a compatible `llama-server` |
| ONNX graph | ONNX Runtime adapter | `skeleton` | Inspection and runtime diagnostics only |
| TensorRT plan | TensorRT adapter | `skeleton` | Plan detection and diagnostics only |

## Backends

| Backend | Device | Build or runtime requirement | Status |
| --- | --- | --- | --- |
| Candle CPU | CPU | Default features | `experimental` |
| Candle Metal | Apple GPU | `--features metal` | `experimental` |
| Candle CUDA | NVIDIA GPU | `--features cuda` and CUDA toolchain | `experimental` |
| OpenVINO | Intel CPU, GPU, or NPU | OpenVINO runtime | `external-runtime` |
| llama.cpp | CPU or GPU | Compatible `llama-server` | `external-runtime` |
| CoreML, MLX, Vulkan | Platform-specific | Runtime implementation not connected | `skeleton` |

## Server capabilities

| Capability | Status | Limitation |
| --- | --- | --- |
| Chat and text completions | `experimental` | Depends on the selected model path |
| SSE streaming | `experimental` | Available on generative engines that emit text deltas |
| Embeddings | `experimental` | Requires a compatible embedding model |
| Reranking | `experimental` | Uses embedding cosine similarity; needs real-model validation |
| Backend discovery | `stable` | Reports compiled and detected capabilities |
| JSON object and JSON Schema responses | `experimental` | Native paths support validation and grammar filtering with backend-specific limits |
| In-flight batching and long-context policies | `experimental` | Paged-cache integration is still being hardened with real models |

## Validation

Run the default repository checks:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/validate_json_artifacts.py
```

Real-model validation is opt-in and must not silently skip when used as release
evidence:

```bash
BLOOM_REQUIRE_MODEL=1 \
BLOOM_MODEL_PATH=/path/to/model.gguf \
./scripts/ci_smoke_test.sh

BLOOM_REQUIRE_MODEL=1 \
BLOOM_MODEL_PATH=/path/to/model.gguf \
./scripts/openai_compat_smoke.py --require-openai-sdk
```

Promotion to `stable` requires a pinned model source and hash, a reproducible
command, successful generation, and a benchmark record containing backend,
dtype, quantization, TTFT, TBT, throughput, and peak memory. See
[RELEASE.md](../RELEASE.md) for the release gate.
