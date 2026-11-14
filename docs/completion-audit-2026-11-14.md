# Cross-layer correctness and client completion audit — 2026-11-14

Bloom already has a broad implementation, but the audit found correctness gaps
in recently extended execution paths and user-facing operations that were
implemented in the server but inaccessible in clients. This pass fixes the
reproducible issues below and expands the gates that exposed them. It does not
claim full llama.cpp parity or generally supported production deployment.

This is the initial audit's historical record. The subsequent
[completion continuation](completion-continuation-2026-11-14.md) records the
typed native generation contract, trained cross-encoder acceptance, release
preservation and final installed-SDK validation performed before the dev push.

## Implemented corrections

| Area | Observed problem | Delivered behavior |
| --- | --- | --- |
| Resident KV reuse | The retained prefix included a final sampled token that had never been forwarded | Retain only tokens actually represented by the KV state, including token-limit and EOS boundaries |
| Seeded decoding | Replaying a cached prompt suffix consumed random draws | Sample only the final prompt position; equal seeded requests no longer depend on warm versus cold prefix state |
| Unicode streaming | Byte-fragment decoding could emit a replacement character before the completed character, e.g. `�€` | Emit only stable decoded prefixes, flush at completion and preserve output bounds; strict logprob byte validation remains separate |
| Buffered embeddings | Candle BERT returned vectors in streams but discarded them in buffered inference | Preserve embedding chunks in `ModelOutput.logits` |
| Default model adapter | The default streaming implementation forwarded only text | Validate and forward numeric vectors, image bytes, audio with its sample rate and video frames/metadata; errors stop delivery before success |
| Direct HTTP streams | Single-candidate execution duplicated the bounded multi-candidate path and could report success after worker failure | Share the bounded collector for direct text choices, retaining cancellation, validation, output limits, tool identities and one successful terminal lifecycle |
| Responses tools | An initial empty content field was interpreted as a text item, rejecting later tool deltas | Ignore empty role announcements; accept tool arguments and finish markers in separate events, verified through the real runtime and official clients |
| CLI and benchmarks | Text chunks or whitespace were counted as tokens; buffered timings fabricated first-token observations | Prefer cumulative native usage, otherwise tokenize complete text; preserve empty/EOS behavior and truthful inference timing |
| Interactive CLI | Invalid parameter changes and generation errors ended the session | Report errors while preserving the session and validated settings |
| CLI logprobs | Native token probabilities had no CLI surface | `--logprobs N --json`, configuration and JSON-request propagation, buffered records and streaming NDJSON |
| macOS reranking | Native model/embedding panels lacked document ranking | Bounded query/documents/top-N, cancellation, response identity/usage/order checks, results and versioned JSON export |
| macOS recovery | Reconnection could leave old encoder output; embedding responses lacked complete usage/normalization checks | Invalidate old operations/results and reject inconsistent responses |
| Browser routing | Rerank-only cross encoders passed readiness but could not open the encoder workspace | Select the supported rerank tab and disable unsupported embeddings |
| Browser size bounds | Valid raw documents expanded beyond response/export limits after JSON escaping | A bounded 6 MiB response/export envelope covers the full 768 KiB raw-content contract and metadata |
| Browser accessibility | Ready/success text and rank badges failed contrast checks; gradient backgrounds prevented automated contrast verification | Separate darker text colors and opaque encoder backgrounds, with browser scans of a populated rerank workspace |
| ONNX provider selection | Unknown providers became CPU; ORT could fall back during construction or execution | Reject unknown/unavailable/inactive providers and disable execution-time fallback |
| ONNX numeric I/O | Unsigned types could be interpreted as signed, integer casts could truncate and scalar/string/output handling was inconsistent | Exact dtype mapping, bounded shape/range checks, finite float32-compatible outputs, UTF-8 strings and real CPU identity-graph tests |
| Performance evidence | Missing references could hide Bloom failure or bypass an enabled ratio gate; throughput parsing missed latency-prefixed output and accepted numeric suffixes | Fail executed errors and missing required evidence, validate thresholds, and parse complete finite nonnegative values |
| CI and documentation | Real ONNX regression tests were not in CI; production notes still described completed historical work as pending implementation | Pin CPU ONNX test dependencies and run graph/tokenizer gates; reconcile historical soak and native-wheel implementation without promoting current deployment support |

The direct collector preserves ordinary HTTP's existing visible-text usage
convention. Exact accepted-token/EOS and finish-reason unification across every
adapter still needs a separate typed output contract. Native CLI streams use
model-owned cumulative usage when available. Buffered CLI results and external
adapters without usage/tokenizer support retain the boundaries documented in
[performance checks](performance_budgets.md).

The ONNX bridge remains a bounded first-input/first-output numeric/string
adapter plus its explicit embedding package path. The changes do not add an
autoregressive ONNX language-model implementation or prove accelerator
placement on hardware. See [backend adapters](backend-adapters.md).

## Validation

Validation was performed on macOS arm64 with Rust 1.97.1. The local evidence
directory is `output/completion-audit-2026-11-14/`; generated binaries, models
and logs are excluded from Git.

| Gate | Result |
| --- | --- |
| Rust workspace | 1,395 passed, zero failed; one existing external Laya/Torch parity test ignored |
| Server after the final Responses follow-up | All 463 tests passed again |
| Strict workspace/all-target Clippy and formatting | Passed |
| Workspace/all-target build without default features | Passed |
| Web UI | 151 tests and strict UI Clippy passed; production WASM assets rebuilt |
| Native macOS client | Three regression groups, debug and release builds passed |
| Python SDK with native FFI required | 42 passed |
| Python script regressions with installed optional CPU dependencies | 247 passed, no skips; includes 16 real/numeric ONNX bridge tests |
| Generated native Qwen2 integration | Single-file/sharded, direct/IFB, embeddings, reranking, JSON/schema and function-call flows passed with pinned OpenAI/Ollama clients |
| Trained Qwen2 CPU | CLI buffered/streamed logprobs reconstructed `Bloom`; OpenAI/Ollama buffered/streamed semantic output and official SDK checks passed |
| Browser engines | Chromium, Firefox and WebKit passed shell, dialogs, clipboard, conversation export and rerank-only request/result/export checks; all five axe scan contexts had zero violations and unresolved items |
| Process and HTTP boundary | Real-server graceful shutdown/escalation/deadline and final embedded-server routing/authentication/origin/readiness checks passed |
| Repository contracts | Architecture, JSON artifacts, toolchain consistency, immutable actions, release policy and local Markdown links passed |

Trained HTTP reports bind to their own unchanged server snapshot, SHA-256
`3725cef6317f52e861702f094d8ac053a03d6bbdda5849ae5a15f46c05495e4d`, and Qwen2
model SHA-256 `aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f`.
The later Responses empty-content correction is covered by the final native
tool/SDK profiles, complete tiny-model gate and final server tests. Browser and
HTTP-boundary reports use `bloom_server-final`; each browser report records
unchanged server and asset hashes. `EVIDENCE.json` inventories the final local
sources and retained reports rather than conflating these artifact identities.

Representative reproduction commands:

```bash
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings -A missing_docs
cargo check --workspace --all-targets --no-default-features --locked
cargo test --manifest-path ui/Cargo.toml --locked
./scripts/test_macos_client.sh
cargo build -p bloomai-ffi --locked
BLOOM_TEST_NATIVE_FFI=1 python3 -m unittest discover -s python/tests -v
python3 -m pip install -r requirements/schema-validation.txt -r requirements/onnx-runtime-test.txt -r requirements/python-package.txt -r requirements/compat-smoke.txt
python3 -m unittest discover -s scripts -p 'test_*.py' -v
./scripts/test_tiny_model_runtime.sh --require-official-clients
./scripts/build_ui.sh
python3 scripts/test_ui_browser.py --browser chromium
python3 scripts/test_ui_browser.py --browser firefox
python3 scripts/test_ui_browser.py --browser webkit
```

The browser gate exercises the real built WASM application. Its rerank-only
readiness, request and result fixtures test client routing and export without
claiming trained cross-encoder quality. Real ONNX tests use generated CPU
graphs and package tokenizers; TensorRT tests do not claim CUDA execution.

The existing Laya/Torch parity test remains explicitly ignored because its
external weights/reference are not available to the default suite. Hardware,
training quality and hosted installation requirements below are not converted
into passing mock checks.

## Remaining completion requirements

| Priority | Remaining work | Required evidence or implementation |
| --- | --- | --- |
| P0 | A supported current-build deployment cell | Exact final binary/model identity, sustained load, budgets, target service-manager and filesystem upgrade/rollback drills |
| P0 | Verified hosted release artifacts | Clean Linux/macOS/Windows installation, full native-wheel Python matrix, published container digests, SBOM/provenance and final-layer scans |
| P1 | Wider model and quantization parity | Native loaders and executable packed kernels, pinned licensed trained models, correctness and memory/performance evidence per architecture |
| P1 | Accelerator coverage | Real Vulkan execution and broader CUDA/Metal/CoreML/TensorRT/NPU task validation on their target devices |
| P1 | Sampling/protocol breadth | IFB/GPU/multimodal logprobs, broader grammar/tools, multimodal multiple choices and consistent exact usage/finish metadata |
| P1 | Trained task quality | Public cross-encoder, tool-selection, converted/quantized and multimodal quality sets with task-specific thresholds |
| P1 | Fleet operation and accessibility | Multi-host revocation/recovery drills, target assistive-technology and mobile-device acceptance |
| P2 | Maintainability | Further extraction of oversized handler, UI, executor and scheduler modules; keep boundaries covered during feature work |

The earlier 24-hour workload and Metal validation remain useful historical
evidence for their recorded builds. They do not establish 24-hour stability
for this pass. No release, registry publication or production deployment was
performed. Track these remaining items in the [production gap register](production-readiness.md),
[roadmap](roadmap.md), [support matrix](support-matrix.md) and
[reference parity inventory](llamacpp-parity.md).
