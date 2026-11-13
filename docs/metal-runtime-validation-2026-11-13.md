# Candle Metal runtime validation — 2026-11-13

The local Apple M5 GPU executed Bloom's native Candle Metal path. This is
hardware execution evidence for the listed generated fixtures, not a trained
model quality, performance, sustained-memory or deployment acceptance result.

## Reproduce offline

Build the native binaries and the fixture generator from the same checkout:

```bash
CARGO_TARGET_DIR=/tmp/bloom-metal-target cargo build --locked --features metal \
  --bin bloom_server --bin bloom_infer
cargo build --locked -p bloomai-engine --example generate_tiny_qwen2_fixture
CARGO_TARGET_DIR=/tmp/bloom-metal-target cargo test --locked -p bloomai-engine \
  --features metal,hardware-tests test_metal_ -- --nocapture
python scripts/metal_runtime_smoke.py \
  --server-bin /tmp/bloom-metal-target/debug/bloom_server \
  --infer-bin /tmp/bloom-metal-target/debug/bloom_infer \
  --fixture-generator target/debug/examples/generate_tiny_qwen2_fixture \
  --output-dir /tmp/bloom-metal-report \
  --require-openai-sdk
```

The Python interpreter must already have the official `openai` package when
`--require-openai-sdk` is supplied. Nothing in the smoke script downloads
models, installs dependencies or substitutes CPU execution for a GPU request.
The output directory must be new. The optional `--hardware-test-bin` argument
includes the engine test executable from the preceding Cargo command in the
same report and requires both AWQ and GPTQ CPU/Metal comparison tests to run.

The gate checks native macOS arm64, device admission via `--doctor=json`, exact
deterministic CPU/Metal text parity, and seven real HTTP scenarios:

- Text, structured output and function tools, each with direct and continuous
  batching execution, including multiple candidates and streamed responses.
- Embedding projection and embedding-based reranking.
- Official OpenAI SDK decoding for each selected HTTP profile.

Each subprocess has a deadline and private process group. The report records
binary and model SHA-256 digests, commands, results and per-phase logs, with
isolated configuration/catalog paths. Binary digests must remain unchanged
throughout the run. GPU metadata excludes attached-display identifiers.

## Observed first pass

The first pass used native Darwin 25.5.0 arm64 and an Apple M5 with ten GPU
cores and Metal 4. All 16 phases passed, including the two real Metal kernel
tests; the generated text fixture emitted `fixture fixture fixture` on both
CPU and GPU. The device check reported Metal available. Overall doctor status
was `warn` because the deliberately empty catalog and absent UI were unrelated
to device admission.

The local report is
`/tmp/bloom-metal-acceptance-2026-11-13-run2/report.json`. Its server SHA-256 was
`c1bac749497ccf117fc666693a4d3b8fef9519a80ab5c6844605a5f871c5548f`.
This snapshot predates the subsequent CPU token-logprob change.

## Final rerun

After the token-logprob implementation was frozen, the Metal-enabled server,
CLI and hardware test executable were rebuilt. All 16 phases passed again,
including both real Metal kernel tests and all seven HTTP scenarios with
OpenAI Python SDK 3.37.0. The final local report is
`/tmp/bloom-metal-final-2026-11-13/report.json`; its server SHA-256 is
`4807e9a44bc846428bf3e49b487e4a451c5e4dcfcc4a2405dccd19dc79d4b491`.
The separate build and kernel logs are `/tmp/bloom-metal-final-build.log` and
`/tmp/bloom-metal-final-kernels.log`. The report retains the CLI, hardware-test
and fixture-generator digests, plus hashes of every generated model artifact.
The final report and its logs are also archived under
`output/capability-logprobs-metal-2026-11-13/metal/`, with build/kernel logs in
the parent evidence directory.

## Limits

The four tiny Qwen2 profiles use generated weights. They validate tensor
execution, protocol contracts, Unicode/stream handling, resource cleanup and
device admission; they do not demonstrate useful model answers or quality of
trained quantization. This run does not cover CUDA, Vulkan, Intel NPU or other
hardware. Token log probabilities remain an explicitly CPU-only capability.
