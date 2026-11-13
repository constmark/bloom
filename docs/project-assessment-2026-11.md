# Capability and reliability assessment — 2026-11-03

This review covers the vendored Bloom workspace, its engine operations, HTTP
compatibility, resource admission, FFI/Python boundary, browser settings and
promotion evidence tooling. The support matrix remains the executable support
contract. This iteration closes the concrete software defects below; it does
not establish complete llama.cpp parity or promote an unmeasured deployment.

The [2026-11-13 continuation](capability-completion-2026-11-13.md) supersedes
this historical review for multi-candidate structured/tool streaming, KV retention,
CLI sampling, RTN export and SDK tokenization.

## Closed software gaps

| Area | Previous gap | Delivered behavior and regression evidence |
| --- | --- | --- |
| External llama.cpp | Completion-only operation surface; structured parameters ignored; duplicated stream terminal | Executable text/JSON grammar, exact tokenization, ordered embedding batches and cross-encoder reranking, with task isolation, bounded result validation, error/truncation/cancellation handling, an absolute HTTP deadline, strict body framing and one successful terminal. Local HTTP fixture tests cover both default and no-default-feature builds. See the [adapter contract](llamacpp-adapter.md). |
| OpenAI streaming | `n > 1` and prompt arrays rejected in streaming mode | Direct text Chat/Completions stream up to eight independent choices; legacy Completions stream up to eight prompts. Prompt-major indices, per-choice terminals, stop filtering, echo, seed offsets, Chat usage, cancellation, bounded backpressure and worker/runtime ownership have regression coverage and model-backed smoke coverage. |
| Python SDK | Missing top-k, presence/frequency penalties and logit bias; legitimate Rust `"End"` stream event rejected | Buffered/streamed calls expose and validate the shared controls; normal native stream completion is accepted. Unknown schema wrapper fields and unknown/duplicate FFI generation parameters fail explicitly. Native ABI, SDK and isolated package gates cover the boundary. |
| Browser controls | Sampling supported by the server was inaccessible through settings or dropped during persistence/request construction | Top-k, both penalties and bounded JSON logit bias are editable, validated, serialized and persisted with legacy defaults. Image requests reject unsupported active controls before changing conversation state. |
| Memory estimates | Overflow could turn an impossible model estimate into a small reservation | Checked model-size/parameter arithmetic and saturating totals prevent wraparound. Large-budget percentage calculations preserve their intended value. |
| Resource ownership | Cache reservations were absent from budget accounting; legacy UMA unload could release another model's reservation | Cache bytes participate in admission and are reclaimed by unload/eviction; only owned leases release reservations. Impossible or overflowing requests fail before evicting current residents. |
| Deployment evidence | Configured duration could masquerade as measured soak; inconsistent/bool/non-finite evidence could pass | Promotion requires measured duration, consistent request/concurrency/wave bounds, coherent memory/drift data, available physical-device evidence and strict SHA-256 identity. Adversarial evidence regressions cover each rejection. |

## Continuation — 2026-11-04: executable plugins, numeric runtimes and buffered choices

The follow-up closes these additional implementation gaps:

| Area | Delivered behavior | Evidence boundary |
| --- | --- | --- |
| WASM processing | Versioned ABI through the real Processor/ProcessorRegistry; import/WASI/start and file-reference rejection; fresh instances; fuel, memory, stack, table, I/O, concurrency, deadline and cancellation limits | Actual modules and adversarial fixtures execute in default/no-default builds. This proves bounded processors, not WASI or GPU kernels. See [contract](wasm-plugins.md). |
| CoreML | Embedded Swift compiles real CoreML packages into a private workspace and executes fixed-shape numeric tensors with CPU-only placement | Real affine GLM and identity Scaler models run on this macOS host, including buffered, typed Tensor, stream and CLI output. Trained quality and accelerator-exclusive execution are not claimed. See [contract](coreml-tensor.md). |
| TensorRT | Previously disconnected executor is now compiled and registered in CLI/server/FFI; bundled TensorRT 10.x/CUDA worker executes named numeric tensors, enforces dtype/shape/LINEAR bounds and immutable plan identity, and cleans up CUDA resources. Fake engine-building and attention successes are removed | Rust process fixtures and Python simulated CUDA/TensorRT interfaces prove the boundary; real NVIDIA plan/provider execution remains pending. See [contract](tensorrt-tensor.md). |
| Buffered multi-candidate output | Direct structured Chat/Completion and function-tool Chat support n=1..8, with per-candidate validation, prompt-major ordering, independent seed offsets, aggregate usage and unique call identities; one bad candidate fails the entire response | Native tiny Qwen2 structured/tool fixtures pass raw HTTP and OpenAI SDK checks. IFB, multimodal and streamed structured/tool multiple choices still fail explicitly. |
| Numeric acquisition | TensorRT .plan/.engine verified import/download, catalog scan/resolve, source inspection and signed-index publication/consumption; browser validators and file pickers share the contract | Two-suffix transaction, exact identity, schema positive/negative and UI tests preserve operator authorization, source-host, SHA-256, quota and no-overwrite checks. Acquisition does not establish GPU execution. |
| Numeric task/output | Tensor task identity, zero KV accounting, readiness/preflight/client validation and generation isolation; CLI preserves numeric buffered/stream output and --json envelopes; collecting numeric stream chunks retains their vectors | macOS client, UI, Rust/CLI and protocol regressions distinguish numeric prediction from generation and sentence embeddings. See [CLI](numeric-cli.md). |

## Continuation — 2026-11-07: lifecycle and streaming reliability

A cross-layer source review found additional defects despite the existing
passing suites. This pass preserves the in-progress Laya changes and focuses
on observable failures, resource ownership and regression coverage.

| Area | Defect | Implemented correction |
| --- | --- | --- |
| Resource eviction | Failed or panicking offload callbacks had already released their memory accounting; concurrent reservations could reuse memory still occupied by the model | Retain RAM/VRAM/cache and residency until successful offload, exclude in-flight lease IDs, skip failed victims within a reservation, and commit by lease identity. Reentrant callbacks remain outside the coordinator lock. |
| KV scheduling | Aegaeon scoring multiplied by cache cost despite the intended inverse-cost policy; invalid thresholds bypassed admission; large chunk/token counts overflowed | Divide by cache cost, reject non-finite/out-of-range thresholds, count zero-capacity rejections, skip empty victims, saturate telemetry and check request token arithmetic before any eviction. |
| Server streaming | Internal SSE framing repeatedly scanned/copied accumulated data and omitted legal line endings and empty data fields | Single-pass incremental framing with existing size/UTF-8/truncation limits; split-boundary, mixed CR/LF/CRLF and large-frame regressions. |
| Reranking | Missing/extra backend scores could be concealed by top-N projection | Require exactly one score per input document before sorting or projection. |
| Readiness | Replacing a resident model could publish `ready` with loading/progress/error fields that violated the client's contract | Derive HTTP readiness from the published snapshot, returning 503 during replacement/failure while preserving the old runtime's inference lease. |
| Desktop launcher | Any HTTP 200/503 was accepted; fragmented responses failed; config-file ports and short `-p` were ignored; errors could leave the server child alive | Validate the complete schema-v3 document and HTTP status within header/body/time budgets, honor the server configuration and CLI overrides, register signal handlers before child creation, and own cleanup through a drop guard. |
| Native macOS client | Stop-and-resend let stale requests clear newer state; edited connection settings reused old readiness; readiness/SSE could buffer without limits | Isolate generation and connection identities, snapshot the validated client, invalidate edited credentials/endpoints, cap readiness while reading, and parse bounded complete SSE events incrementally. |

The native regression harness now exercises asynchronous stale-success/error,
connection cancellation, bounded response parsing and real URLSession protocol
fixtures without adding an XCTest dependency. The launcher uses the packaged
readiness example as a fixture and tests missing fields, incompatible ranges,
inconsistent state, truncation, fragmentation, configuration and child reaping.

Validation on macOS/aarch64 with Rust 1.97.1:

- Workspace: **1,218 passed**, zero failures/ignored cases, including doc tests.
- Strict workspace/all-target Clippy (`-D warnings -A missing_docs`), default
  formatting and no-default-feature/all-target checking passed.
- Elderwand integration workspace: **936 passed**, including doc tests.
- Web UI: **144 passed**. Python SDK: **31 run**, one native-library case
  skipped. Tooling scripts: **149 run**, three environment-gated cases skipped.
- Native macOS parser/readiness/async lifecycle checks and Swift build passed.
- Generated, untrained Qwen2 single-file/sharded, IFB, embedding, rerank,
  structured-output and function-call HTTP smoke gates passed. The optional
  Ollama Python SDK was unavailable and its subchecks were skipped; this run
  does not claim that SDK coverage.
- A real no-model server launched on a config-file port passed the launcher's
  bounded handshake; SIGTERM reached the child, both exited successfully and
  the listening socket closed.
- Architecture, readiness-contract, JSON artifact, toolchain consistency and
  local Markdown-link gates passed. The shell initially selected Homebrew Rust
  1.94 despite the installed pin; final Rust checks explicitly used the pinned
  toolchain. A concurrent intermediate build invalidated a doctest artifact;
  the serial full rerun above passed.

Remaining product gaps include the native macOS client's chat-only workspace
(no persistent conversations, model management or encoder UI), the declared but
unused KV `retention_ratio` policy, large scheduler/executor modules, and the
model/hardware/evidence inventory below. These are not represented as completed
features or stable deployment claims.

## Interface completion — 2026-11-07

This follow-up adds four executable HTTP routes while reusing the existing
engine contracts and request/runtime lifecycle:

| Interface | Implemented behavior | Explicit boundary |
| --- | --- | --- |
| `POST /v1/tokenize` | Exact model-owned token IDs and count, versioned response and exact model identity | No word-count approximation, chat templating or invented IDs; 256 KiB text and 65,536 tokens. See [contract](tokenization-api.md). |
| `POST /v1/detokenize` | Exact decoding with vocabulary validation on Candle and declared tokenizer processors | Special-token/normalization behavior belongs to the adapter; unsupported decoding fails explicitly. |
| `POST /v1/tensors/infer` | CoreML flat arrays and TensorRT named tensors, checked dimensions, preserved integer JSON and finite flat output | Numeric output is not an embedding and has no invented dimensional shape; target vendor runtime remains required. See [contract](tensor-api.md). |
| `POST /v1/audio/transcriptions` | Multipart WAV input and JSON/text transcription with exact model selection and bounded output | PCM16 mono, 8–48 kHz, nonempty recording up to 60 seconds; unsupported controls/formats fail explicitly. See [contract](audio-transcriptions.md). |

All routes share authentication, protocol error normalization, correlation and
no-store policies. Blocking workers retain the exact model lease and concurrency
permit until execution finishes after disconnect or cancellation. Numeric/audio
results do not inflate generated language-token counters. WAV parsing validates
bounded metadata before admission; decoded floats are allocated only inside the
admitted blocking worker. ASR partial snapshots and final deltas do not duplicate
text, and an empty delta cannot erase a prior snapshot.

The Elderwand remote proxy now forwards tokenizer/tensor routes, preserves
query controls for worker rejection, and forwards transcription multipart bytes
unchanged. Invalid query requests do not trigger model switches. Multipart
model selection stays with the active worker rather than parsing audio bytes
as JSON. Worker discovery lists the new available interfaces.

Real CoreML startup exposed a stale server allowlist: the adapters already
reported verified CPU-only/CoreML and CUDA/TensorRT placement, but strict server
admission rejected their names. The allowlist now includes both while retaining
the adapter probes, exact physical-device check and memory accounting.

Default workspace regression passes **1,238 tests**. Engine/server tests with
no default features pass **879 tests**; this also fixes existing preflight
fixtures that incorrectly assumed Candle was compiled and could wait forever
for rejected model loads. Lifecycle/memory tests now use ONNX metadata fixtures,
while feature-off tests explicitly assert native Candle rejection. Elderwand
integration passes **939 tests**, including native-route proxy checks.

The reproducible [HTTP smoke script](../scripts/native_operation_smoke.py) uses
an isolated configuration, local existing models and no downloads. A generated
untrained Qwen2 fixture verifies exact tokenization/decoding plus authenticated
HTTP and task-isolation behavior. A real macOS CoreML CPU affine model also
passes the HTTP route with `[1,2,3] -> [4.5]`, including exact model identity,
authentication, unsupported-field/query rejection and tokenizer task isolation.
Broader trained tokenizer coverage, TensorRT
GPU execution and trained ASR quality remain target-specific evidence gates.

## Remaining capability inventory

| Domain | Still incomplete | Required closure |
| --- | --- | --- |
| Native models and quantization | GGUF/Safetensors architecture, quantization and conversion/export coverage is bounded; some families still need external llama.cpp | Architecture-specific loaders/kernels and hash-pinned trained model/device/quantization fixtures. An external route does not prove native parity. |
| Vendor backends | Vulkan remains a discovery/diagnostic skeleton; CoreML and TensorRT now execute bounded numeric contracts, while richer model semantics and target-runtime breadth remain incomplete | Broaden CoreML beyond macOS CPU fixed-shape numeric models; publish real NVIDIA plan/provider evidence for TensorRT; implement a Vulkan execution runtime. ONNX, MLX, OpenVINO, ASR and NPU bridges still require their documented external packages and hardware evidence. |
| Advanced compatibility | IFB and multimodal multiple choices; streamed tool/structured multiple choices; token log probabilities; broader reference sampling/grammar; custom/built-in tools; broader image/audio/video semantics | Implement each through a typed engine contract, scheduler/adapters, output validation and client fixtures before admitting its request fields. Unsupported active fields continue to fail closed. |
| Plugin breadth | Bounded WASM processors now execute; WASI, arbitrary imports and WASM engine/backend replacement remain outside the contract | Add any future host functions or plugin roles through explicit typed contracts and resource/cancellation tests. Do not infer WASI or hardware-engine support from the processor ABI. |
| Scheduler and acceleration | Wider paged attention, speculative decoding, quantized Metal/CUDA and multi-device performance are unproven | Exact target-runner correctness, memory, throughput, cancellation and fallback evidence, with native and external results reported separately. |
| Trained quality | Broader cross-encoder, tool selection and multimodal quality evidence | Immutable license-reviewed packages and task-specific quality thresholds, distinct from deterministic untrained mechanical fixtures. |
| Stable deployment | No generally stable cell has met every published acceptance criterion | Measured reference comparison, 24-hour/1,000-request soak, cancellation/restart and physical disk-failure drills, upgrade/rollback and bounded resource drift on the declared target. |
| Distribution and clients | Official multi-architecture container supply chain, binary-wheel platform/ABI matrix and broader browser/assistive-technology validation | Published digest/SBOM/provenance and runtime scans, platform-native wheel/ABI gates and target browser/device accessibility records. A local sdist-to-wheel check covers packaging mechanics only. |

The [parity matrix](llamacpp-parity.md), [roadmap](roadmap.md),
[production gap register](production-readiness.md) and
[support matrix](support-matrix.md) define the acceptance criteria for these
remaining rows. Missing vendor runtimes, devices, trained fixtures or long-run
measurements must remain visible rather than being represented by a capability
flag or a successful mock test.

## Initial-pass local validation record

The macOS/aarch64 validation run passed:

- Bloom workspace: 1,143 tests, no failures or ignored cases; strict workspace
  Clippy and formatting; no-default-feature all-target checks.
- Elderwand integration workspace: 866 tests, no failures.
- UI: 143 tests, WASM target compilation and strict Clippy.
- Python SDK: 31 tests including the native FFI; ABI compatibility and isolated
  sdist-to-wheel build/install checks.
- Evidence/tooling scripts: 120 tests, architecture/readiness checks, JSON
  artifact validation and local documentation links.
- Deterministic native Qwen2 single-file/sharded, IFB, embedding, rerank,
  structured-output and function-lifecycle protocol gates, requiring pinned
  OpenAI 3.37.0 and Ollama 0.6.3 clients. The new direct text gates assert two
  Chat candidates and four prompt-major Completion candidates; official OpenAI
  streaming decoding also passes.
- External llama.cpp boundary: 27 local HTTP fixture tests in each default and
  no-default-feature configuration, including absolute-deadline and strict
  body-framing regressions.

No new trained-model, physical accelerator, visual browser, 24-hour soak or
release-publication claim follows from these local checks. The historical
trained-model records and support labels retain their existing scope.

## Continuation validation record

The follow-up passed the real CoreML CPU gate (eight cases, including numeric
buffered/stream/Tensor inputs and single-worker busy admission), twelve actual
WASM sandbox/adversarial cases, and nine focused TensorRT Rust process/manifest
cases. TensorRT's 24 Python contract tests pass with the pinned NumPy 2.5.4,
including float16/float32 boundaries that previously rounded before comparison.
The NumPy test dependencies are installed by the Linux CI contract gate; macOS
CI now runs the CoreML CPU gate explicitly.

The numeric CoreML model also passed CLI plain/JSON and stream/NDJSON checks,
and actual native ABI revision 2 calls through Python's `generate` and
`generate_stream`. Buffered structured/tool n=2 native Qwen2 fixtures passed raw
HTTP and OpenAI 3.37.0 SDK checks, including prompt-major Completion choices,
legacy function aliases and unique call identities.

The final runtime workspace passed 1,175 tests; Elderwand integration passed
892; the UI passed 144; Python SDK/native ABI checks passed 31. The complete
149-case script suite includes the new TensorRT, catalog-schema and documentation
traversal regressions. A reproduced concurrent-build documentation-scan race
was fixed by pruning excluded directories before traversal, retaining errors
for real documentation directories.

Strict workspace/UI Clippy, no-default-feature all-target checks, WASM UI
compilation, native macOS client checks, JSON Schemas, architecture checks,
policy-checked dependency SBOM generation and Markdown links passed. The new
Wasmi/Wat dependency graph satisfies the existing license/source policy without
adding an exception. Real NVIDIA plan execution, broader trained quality,
visual browser/assistive-technology validation, long soak and publication
supply-chain evidence remain pending.

## Reproduction

Use the Rust toolchain pinned in `rust-toolchain.toml` for `cargo`, `rustc`,
`rustdoc`, `rustfmt` and Clippy; mixed PATH toolchains can otherwise select an
older compiler despite the installed pin.

```sh
cargo test --workspace --locked
cargo check --workspace --all-targets --no-default-features --locked
cargo clippy --workspace --all-targets --locked -- -D warnings -A missing_docs
cargo fmt --all -- --check
cargo test --manifest-path ui/Cargo.toml --locked
cargo check --manifest-path ui/Cargo.toml --target wasm32-unknown-unknown --locked
# In an isolated Python environment, install the pinned gate dependencies:
python3 -m pip install -r requirements/schema-validation.txt -r requirements/tensor-runtime-test.txt -r requirements/compat-smoke.txt
BLOOM_TEST_NATIVE_FFI=1 python3 -m unittest discover -s python/tests -v
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/check_architecture.py
python3 scripts/validate_json_artifacts.py
python3 scripts/check_markdown_links.py
./scripts/test_tiny_model_runtime.sh --require-official-clients
python3 scripts/test_python_package.py
```

The generated tiny model proves execution and protocols, including the new
streaming choices. It does not prove semantic quality or a stable deployment.
