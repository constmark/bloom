# Cross-layer completion review — 2026-11-13

Bloom has a substantial implemented inference stack, but full llama.cpp parity
and generally supported production deployment remain unfinished. This review
traces concrete gaps through core policy, native execution, HTTP adapters,
CLI/export tooling and C/Python clients. The changes below are executable
implementations with regression coverage, not new support labels.

The [next continuation](capability-continuation-2026-11-13.md) adds packed
loadable RTN packages, scheduled choices, native macOS conversation persistence
and bundled SDK wheels. It supersedes the corresponding remaining rows and
unpacked-export limitations below; this initial-pass record is retained.

## Implemented corrections and capabilities

| Layer | Observed gap | Delivered behavior |
| --- | --- | --- |
| Native sampling | Penalties/bias were applied after softmax without matching temperature; bias was clipped to ±20 | Apply full bounded bias and presence/frequency penalties to logits before temperature and probability filtering |
| Batched/speculative sampling | LCG output covered only half of the probability interval; fixed seed restarted batch sampling each step; stochastic draft proposals used argmax despite declaring softmax probabilities | Full-interval RNG, position-dependent batch draws, request-owned draft RNG and actual sampling from the verifier's proposal distribution |
| Speculative controls | Some active sampling/grammar controls bypassed the verifier | Ordinary decoding fallback for combinations the speculative verifier cannot preserve |
| KV admission | Declared retention ratio had no effect; an insufficient eviction plan could still cancel existing work | Enforce rounded-up token retention, whole-session victims and all-or-nothing sufficient eviction plans; reject invalid policy values |
| OpenAI streams | Multiple structured/tool choices and structured Completion prompt arrays were rejected | Bounded per-candidate generation and validation, candidate indices/seeds, unique call IDs, aggregate success usage, cancellation and one terminal stream lifecycle |
| Completion echo | Schema guidance could be reflected back as if it were the user prompt | Echo the original prompt in buffered, direct streaming and IFB paths |
| CLI | Native penalties, bias and structured-output controls were unavailable; arbitrary JSON could become an empty prompt; invalid interactive updates poisoned settings | Flags/config/JSON/interactive propagation, literal JSON preservation, explicit request errors and validated transactional interactive updates; benchmarks honor requested sampling controls |
| RTN export | Invalid methods silently selected INT8; fake AWQ changed weights without inverse scales; rounded INT4 offsets corrupted fractional values; multiple shards could overwrite one path | Explicit RTN methods and documented tensor format, correct symmetric/asymmetric quantization, precision/group/name checks, staged multi-shard handling and no-overwrite destinations |
| C/Python | Exact tokenization existed in engines and HTTP but had no SDK surface | Optional ABI v2 tokenize/detokenize extensions and Python methods, legacy-library compatibility, bounded UTF-8/token inputs, output ownership and errors |
| Python boundaries | Identifier/JSON limits differed from native limits; huge stream timeout overflowed platform waiting APIs | Early matching limits and bounded condition-variable waits |

See the updated [OpenAI contract](openai-compatibility.md),
[SDK contract](ffi-python.md), [RTN format](quantization-export.md) and
[scheduler policy](scheduler.md). CLI usage is in the [README](../README.md).

The sampling fixes intentionally change sequences produced by the previously
incorrect seeded sampler. Export no longer accepts `awq_int4` or unknown method
names. RTN INT4 values are stored unpacked as U8 and are not an inference-ready
AWQ/GPTQ package. F64 export requires source values exactly representable as F32.
These behavioral boundaries are deliberate and documented.

## Existing long-run evidence reconciled

The retained 2026-11-11 CPU workload had finished even though the launch review
still described it as running. Current production-profile revalidation passes:
86,400.270313 measured seconds, 132,378 requests, concurrency 1/2, p95 first
content 1,279.559 ms, p95 total latency 1,342.682 ms and peak RSS 1,589,280,768
bytes. Failure injection and disconnect recovery passed their declared budgets.

This belongs exclusively to server SHA-256
`bed498b42190e7c1eabf35e3add8522e88e3433a87c9f2796ac54411ec501172` and trained
Qwen2 model SHA-256
`aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f`.
The report remains `stable_claim: false`. It is historical evidence, not a
24-hour validation of these new changes. See the reconciled
[launch review](launch-readiness-2026-11-11.md).

## Remaining work and acceptance requirements

| Domain | Remaining work | Acceptance required |
| --- | --- | --- |
| Native model/quantization parity | Wider architectures, conversions and packed executable quantization formats | Architecture/quantization-specific implementation and immutable trained-model fixtures |
| Hardware execution | Vulkan runtime; wider CoreML/TensorRT/Metal/CUDA/NPU task coverage | Real target-device execution, correctness, peak memory and throughput; mocks cannot close these rows |
| API and scheduling breadth | IFB/multimodal multiple choices, log probabilities, broader grammar and tools, accelerated sampling combinations | Typed engine contract, scheduler implementation, protocol fixtures and trained/runtime checks |
| Native macOS application | Persistent conversations, model management and encoder workflows | Client implementation, state/recovery tests and platform UI acceptance |
| Quality | Broader trained cross-encoder, tool selection and multimodal evidence | Pinned licensed models and task-specific quality thresholds |
| Production | Reference-engine performance, current-build sustained workload and actual deployment operations | Matching build/model identity, budgets, service manager/filesystem drills and recovery evidence |
| Distribution | Hosted cross-platform releases, official containers and platform-native SDK wheels | Clean installation, immutable artifacts, SBOM/provenance and platform/ABI matrix |
| Accessibility | Target assistive-technology/mobile device coverage | Real device and screen-reader acceptance |

Full capability completion is therefore **not claimed**. The support matrix and
[parity inventory](llamacpp-parity.md) remain the source of truth. These rows
require additional implementation or external acceptance environments; they
are not made complete by changing metadata or passing local mock tests.

## Validation

On macOS/aarch64 with Rust 1.97.1:

- Default workspace: **1,356 passed, zero failures**, one pre-existing ignored
  upstream Laya real-weight/Torch parity test. Baseline was 1,335 passed.
- App tests: **45 passed**, including eight export regressions and CLI parameter,
  JSON interpretation and interactive-update coverage.
- Web UI: **150 passed**. Python SDK: **38 passed with native FFI required**.
- No-default-feature server: **452 passed**. Workspace no-default-feature
  all-target compilation, strict all-target Clippy and formatting passed.
- C11/C++17 public header, optional tokenizer extension ownership/error tests,
  and native ABI compatibility checks passed.
- Generated untrained Qwen2 ran new structured/tool multiple-choice HTTP cases
  through Candle CPU and the installed OpenAI 3.37.0 SDK. Completion prompt
  arrays, original echo, tool identities, usage and terminal behavior passed.
  The same native structured fixture produced `{"ok":true}` through the CLI
  with the new response-format, penalty and bias flags.
- Cached, hash-verified trained Qwen2 Q4_0 passed exact SDK tokenization round
  trips for English, Chinese, emoji, multilingual text, newlines and empty
  input; invalid vocabulary IDs failed explicitly. For example,
  `Hello, Bloom!` encoded as `[9707, 11, 24503, 0]` and decoded identically.
- Architecture checks and 13 architecture regressions, Rust toolchain
  consistency, JSON artifact validation and local Markdown links passed.

Reproduce the local regression gates:

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings -A missing_docs
cargo check --workspace --all-targets --no-default-features --locked
cargo fmt --all -- --check
cargo test --manifest-path ui/Cargo.toml --locked
cargo build -p bloomai-ffi --locked
BLOOM_TEST_NATIVE_FFI=1 python3 -m unittest discover -s python/tests -v
python3 scripts/check_architecture.py
python3 scripts/test_architecture.py
python3 scripts/validate_json_artifacts.py
python3 scripts/check_markdown_links.py
```

The existing `scripts/test_tiny_model_runtime.sh --require-official-clients`
now includes the new multiple-choice stream assertions. For an installed SDK,
point `BLOOM_FFI_LIB` to the default-feature native build, rather than a
no-default-feature library without Candle.

Local logs, native HTTP reports and checksums are retained in
`output/capability-review-2026-11-13`. Untrained fixture execution proves runtime
mechanics, not semantic model quality. No new accelerator, 24-hour current-build,
clean-platform installation or release-publication claim follows from these
checks. No release was published.
