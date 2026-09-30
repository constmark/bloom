# Backend Adapters

Adapters connect Bloom to another execution runtime or hardware stack without
making that dependency part of the default build.

## Requirements

Every adapter must:

- publish an `EngineCapability` with model, format, dtype, device, modality,
  streaming, quantization, and maturity fields;
- probe missing hardware or runtimes without crashing;
- return an actionable error when execution is unavailable;
- keep vendor SDKs behind a feature, plugin, or external process boundary;
- provide mock or probe tests for default CI; and
- gate real-hardware tests behind an explicit feature or dedicated runner.

## Current adapters

| Adapter | Boundary | Status | Executable behavior |
| --- | --- | --- | --- |
| OpenVINO | External runtime | `external-runtime` | Loads OpenVINO IR when the runtime is installed |
| llama.cpp | External process | `external-runtime` | Starts `llama-server` for GGUF fallback and MTP |
| FunASR / Qwen ASR | External Python | `external-runtime` | Runs model-specific ASR scripts |
| Laya | Native Rust (`laya-decision`) | `experimental` | Runs ModernBERT decision checkpoints and returns typed JSON answers |
| ONNX Runtime | In-process runtime | `skeleton` | File probing and diagnostics only |
| TensorRT | Vendor runtime | `skeleton` | Plan detection and diagnostics only |
| CoreML | Apple framework | `skeleton` | Package probing and diagnostics only |
| MLX | Apple runtime | `skeleton` | Weight probing and diagnostics only |
| Vulkan | GPU runtime | `skeleton` | Shader probing and diagnostics only |

The [support matrix](support-matrix.md) is authoritative when this table and
runtime behavior differ.

## ONNX inspection

The ONNX adapter does not execute graphs in the default build. It can inspect a
single file or a directory containing `model.onnx`, `encoder.onnx`, or
`decoder.onnx`:

```bash
cargo run --bin bloom_infer -- \
  --model /path/to/model.onnx \
  --backend onnxruntime \
  --inspect
```

## llama.cpp

Set the server path explicitly when it is not available on `PATH`:

```bash
BLOOM_LLAMA_CPP_SERVER=/path/to/llama-server \
cargo run --release --bin bloom_infer -- \
  --model /path/to/model.gguf \
  --backend llamacpp \
  --prompt "Hello"
```

`--speculative mtp` selects this adapter automatically and requires a
`llama-server` that advertises `draft-mtp` support.

## Python-backed probes

MLX, Metal/MPS, and CoreML diagnostics use a Python interpreter only when the
host platform supports that probe. Set `BLOOM_PYTHON` to an interpreter path
when `python3` is not on `PATH` (for example, Windows `python.exe`, a virtual
environment, or a non-default Python installation). Bloom falls back to
`python3` then `python` on Unix, and `python` then the `py` launcher on Windows.

## Laya decision models

Laya checkpoints use a nested Hugging Face layout (`rl_agent_config.json`,
`encoder/config.json`, `tokenizer/tokenizer.json`, and a single
`model.safetensors`). Bloom detects that layout and selects the native `laya`
engine automatically. The prompt is a JSON object with `state` and
`questions`; the output is the Jev-compatible typed decision document.

```bash
cargo run --release --bin bloom_infer -- \
  --model /path/to/convaiinnovations-laya \
  --prompt '{"state":"I was charged twice; please refund me.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}' \
  --device cpu
```

The same raw JSON can be sent as the `prompt` to the OpenAI-compatible
`/v1/completions` endpoint. The chat endpoint formats messages for
autoregressive chat models, so it is not the request interface for Laya.

The runtime does not generate tokens. `--stream` emits the complete JSON
document as one output chunk for compatibility with Bloom's existing CLI
boundary. GGUF/ggmlc Laya artifacts are a separate format and are not accepted
by this adapter.

## Adding an adapter

Start with the manifests in `examples/plugins/` and the vendor example in
`examples/vendor_sdk_adapter/`. Keep runtime-specific types out of public core
traits and include license and redistribution notes for vendor SDKs.
