# GGUF

Bloom accepts a single `.gguf` file for inspection, inference, serving, and
benchmarking. A `bloom.json` manifest is optional.

## Direct Inspection

```bash
# Human-readable output
cargo run --bin inspect_gguf -- /path/to/model.gguf

# Machine-readable output, suitable for scripts and compatibility matrix collection
cargo run --bin inspect_gguf -- --json --limit 20 /path/to/model.gguf

# Synthesize manifest, routing, and memory estimation without loading full weights
cargo run --bin bloom_infer -- --model /path/to/model.gguf --inspect
```

`bloom_infer --inspect` extracts as much as possible from the GGUF header:

- `general.name`
- `general.architecture`
- context length
- block/layer count
- hidden size
- attention head / KV head
- head dim
- rope theta / rope scaling
- tokenizer model / vocab size / BOS / EOS
- tensor quantization dtype
- file size and format

## Direct Inference

```bash
cargo run --release --bin bloom_infer -- \
  --model /path/to/model.gguf \
  --prompt "Describe Bloom in one sentence" \
  --stream \
  --max-tokens 128
```

A GGUF directory can also be passed directly, as long as it contains exactly one `.gguf` file:

```bash
cargo run --release --bin bloom_infer -- \
  --model /path/to/gguf-dir \
  --prompt "Hello"
```

## OpenAI-Compatible Service Acceptance

```bash
BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/openai_compat_smoke.py

# Optional: force streaming acceptance with the official OpenAI Python SDK
python3 -m pip install openai
BLOOM_MODEL_PATH=/path/to/model.gguf \
./scripts/openai_compat_smoke.py --require-openai-sdk
```

## Benchmark and llama.cpp Comparison

```bash
cargo run --release --bin bloom_bench -- \
  --model /path/to/model.gguf \
  --max-tokens 64 \
  --repetitions 3

BLOOM_MODEL_PATH=/path/to/model.gguf \
LLAMA_CPP_BIN=/path/to/llama-cli \
./scripts/compare_llamacpp.py --max-tokens 64
```

## Converting Hugging Face to GGUF

Bloom does not ship a built-in model converter. Use the official llama.cpp conversion tooling to produce GGUF, then load the result with Bloom.

Typical workflow:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
python3 -m pip install -r requirements.txt

# Produce an F16 GGUF
python3 convert_hf_to_gguf.py /path/to/hf-model \
  --outfile /path/to/model-f16.gguf

# Quantize to Q4_K_M
cmake -B build
cmake --build build --config Release -j
./build/bin/llama-quantize /path/to/model-f16.gguf /path/to/model-q4_k_m.gguf Q4_K_M
```

Depending on the llama.cpp version, the binary path may be `build/bin/llama-quantize`, `quantize`, or a platform-specific name. Always refer to your local build output.

## Common Failure Diagnosis

| Symptom | Likely Cause | How to Fix |
| :--- | :--- | :--- |
| `GGUF single-file loading requires the candle-engine feature` | Built with the default `candle-engine` feature disabled | Use the default features, or explicitly enable `--features candle-engine`. |
| `failed to read GGUF header` | File is corrupted, not GGUF, or the GGUF version is too new | Cross-verify the file with `inspect_gguf --json` and llama.cpp tools. |
| `missing tokenizer.ggml.tokens in GGUF` | The GGUF does not embed a tokenizer | Prefer a GGUF with tokenizer metadata, or provide a directory path containing an HF `tokenizer.json`. |
| `unsupported model_type` | `general.architecture` is not yet mapped to a Candle model structure | Record the architecture with `--inspect` first, then add an adapter or use a third-party engine/plugin. |
| `llama-server not found` | The llama.cpp server was not found when using `--speculative mtp` | Set `BLOOM_LLAMA_CPP_SERVER=/path/to/llama-server`, or put a recent `llama-server` on PATH. |
| `does not advertise speculative mode 'draft-mtp'` | llama.cpp is too old to support MTP speculative | Switch to a newer llama.cpp / LM Studio server build that supports `--spec-type draft-mtp`. |
| `response did not confirm speculative mode` | The external server did not actually enable the requested speculative mode | Check whether the MTP GGUF contains next-n heads, whether `--speculative mtp` was passed, and the server logs. |
| OOM or insufficient VRAM | Context, batch, model size, or quantization scheme exceeds the device budget | Lower `--context-size` / `--max-tokens`, switch to a Q4/Q5/Q8 GGUF, or fall back to CPU. |
| tokens/s far behind llama.cpp | Backend kernels, quantization type, or batch strategy not tuned for this device | Pin the same model and prompt with `scripts/compare_llamacpp.py` and record the differences. |

## Status Promotion Requirements

Before a GGUF model family can be promoted from `experimental` to `stable`, at minimum:

- `inspect_gguf --json` can read out the architecture, tensor dtype, and tensor count.
- `bloom_infer --inspect` can generate a manifest and memory estimation.
- `bloom_infer` or `bloom_server` can complete real generation.
- `bloom_bench` output conforms to `examples/benchmark-schema.json`.
- A same-machine, same-model comparison against llama.cpp is recorded in `docs/support-matrix.md`.
