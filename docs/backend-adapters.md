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
| ONNX Runtime | Python `onnxruntime` bridge | `experimental` | Executes the first model input/output; text, audio, image bytes and numeric outputs are supported |
| TensorRT | Vendor runtime | `skeleton` | Plan detection and diagnostics only |
| CoreML | Apple framework | `skeleton` | Package probing and diagnostics only |
| MLX | Apple runtime | `experimental` | Text generation through the bundled `mlx_lm` Python bridge; install `requirements/mlx.txt` and use an MLX-LM model package |
| Vulkan | GPU runtime | `skeleton` | Shader probing and diagnostics only |

The [support matrix](support-matrix.md) is authoritative when this table and
runtime behavior differ.

## ONNX Runtime execution

The ONNX adapter executes through the bundled Python bridge when the selected
Python interpreter has `onnxruntime` (or `onnxruntime-gpu`) installed. It can
load a single file or a directory containing `model.onnx`, `encoder.onnx`, or
`decoder.onnx`:

```bash
cargo run --bin bloom_infer -- \
  --model /path/to/model.onnx \
  --backend onnxruntime \
  --inspect
```

For inference, set `BLOOM_ONNX_PYTHON` or `BLOOM_PYTHON` when Python is not on
`PATH`. `BLOOM_ONNX_EXECUTION_PROVIDER=CUDA|OpenVINO|CPU` selects the provider;
the bridge falls back to CPU only when that provider is unavailable. The first
model input is populated from the request and the first output is returned as
text or flattened numeric logits. Model loading first invokes the bridge with
`--probe --model-path PATH --device DEVICE --provider PROVIDER`; it must create
the session with the requested execution provider and print
`{"ok":true,"device":"...","provider":"..."}`. Inference then writes one
serialized `ModelInput` object to stdin and expects one `ModelOutput` object on
stdout (at minimum, `text` and `logits`; omitted optional fields default to
`null`). Models that require a tokenizer or multiple named inputs should
provide a custom `BLOOM_ONNX_SCRIPT` implementing both parts of this contract.
The bridge retains at most 64 MiB of stdout and 64 KiB of stderr while
draining both pipes, and aborts after ten minutes by default. Set
`BLOOM_ONNX_TIMEOUT_SECS` to a value from 1 to 3600 to change the deadline.

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

Bloom's MTP support is provided by this adapter: the GGUF must contain the
model's native next-token heads and the installed llama.cpp server must expose
`--spec-type draft-mtp`. The Candle backend does not load MTP auxiliary heads;
an explicitly selected Candle backend therefore fails during model loading
with a remediation pointing to this adapter instead of silently running a
different speculative strategy.

## Python-backed runtimes and probes

MLX generation and Metal/MPS/CoreML diagnostics use a Python interpreter only
when the host platform supports that path. Set `BLOOM_MLX_PYTHON` for MLX or
`BLOOM_PYTHON` for shared probes when `python3` is not on `PATH` (for example,
Windows `python.exe`, a virtual environment, or a non-default Python
installation). Bloom falls back to `python3` then `python` on Unix, and
`python` then the `py` launcher on Windows.
Non-streaming MLX inference and probing retain at most 16 MiB of stdout and
64 KiB of stderr, drain both pipes concurrently, and stop after ten minutes by
default. Set `BLOOM_MLX_TIMEOUT_SECS` to a value from 1 to 3600 to change that
deadline.
The NPU TTS bridge uses the same bounded runner. Inference defaults to ten
minutes (`BLOOM_TTS_TIMEOUT_SECS`), while ModelScope downloads default to one
hour (`BLOOM_TTS_DOWNLOAD_TIMEOUT_SECS`); both settings accept 1–3600 seconds.

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
