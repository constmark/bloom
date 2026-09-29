# Bloom project assessment — 2026-09-29

## Scope and conclusion

This review examined the current working tree, including the existing
application/engine separation and runtime reliability changes. It covered the
seven native crates, the separate Dioxus UI, Python/FFI integration, protocol
adapters, model support, release/build scripts, CI, and operational documents.
The existing changes were preserved. This is a repository and local execution
assessment, not a new claim of hardware or production certification.

Bloom already has a substantial local inference product: native text and
embedding execution, OpenAI/Ollama adapters, model acquisition and integrity,
bounded HTTP admission, runtime leases, a browser client, and release checks.
The largest remaining gap is proving a small supported deployment combination
under sustained real use. Adding more backend names or HTTP routes will not
close that gap.

The review also found reproducible defects in the development baseline. This
iteration fixes SDK streaming resource bounds, SDK distribution, misleading
performance gates, and Unix executable-cache permissions before extending
feature scope.

## Reference projects and the practical gap

The comparison is about publicly documented product behavior, not a claim
that the projects have identical architecture or measured performance.

| Reference | Public capability used as a yardstick | Bloom gap that matters now |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Prebuilt local CLI/server and broad device-oriented inference; its [server internals](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md) describe shared batching across active slots. | Bloom has an optional IFB path and protocol tests, but no published, sustained model/device/concurrency cell with comparable measurements. |
| [Ollama](https://github.com/ollama/ollama/blob/main/README.md) | Local install, model run/management, REST API and client libraries. | Bloom has substantial model catalog and API mechanics, but lacks a promoted deployment artifact and long-running recovery evidence for one exact model and host. |
| [MLX LM benchmarks](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md) | Model-specific throughput, memory, quantization and hardware/version context. | Bloom's current generic budgets and CLI reports do not yet form a versioned model-specific latency/throughput/memory baseline. |

This comparison prioritizes measured reliability over adding another backend
or route. Bloom's broader surface should not be mistaken for stable support
across every exposed model/device combination.

## Main remaining gaps

| Priority | Gap and current evidence | Concrete next acceptance condition |
| --- | --- | --- |
| P0 | **No supported inference deployment cell.** The [support matrix](support-matrix.md) keeps executable inference experimental. The trained CPU gates are useful correctness evidence, but do not establish sustained-load reliability. | Pick one immutable model revision, dtype/quantization, backend, device, OS and client set. Publish cold/warm startup, concurrent saturation, disconnect/cancellation and multi-hour soak results; test process interruption, bad model load, disk-full, upgrade and rollback. Promote only that combination. |
| P1 | **Performance evidence is incomplete.** Existing thresholds are generic tiers, not model/context-specific measured baselines. The benchmark records process RSS on Linux/macOS and falls back to estimates elsewhere; it has no independent per-device peak-memory guarantee. | Store versioned reports tied to model digest, build features and exact hardware. Add TTFT/TBT latency distributions, concurrency/throughput curves, explicit host/device measurement provenance and resource drift after cancellation. Define separate ARM CPU budgets from actual measurements. |
| P1 | **Trained model/task coverage is narrower than the API surface.** Pinned Qwen2/Qwen3/SmolLM2/MiniLM CPU profiles exist, but vision and tool-selection quality, public trained sharded packages, and more architectures lack equivalent evidence. | Add an immutable trained vision fixture across native, OpenAI buffered/streamed and browser paths, plus a trained tool-use fixture that checks tool choice and arguments. Keep deterministic tiny fixtures for protocol mechanics, separately from model quality. |
| P1 | **Accelerator support lacks target execution evidence.** Metal/CUDA feature builds cannot establish kernel correctness, memory peaks or useful speedups. ONNX/CoreML/MLX/Vulkan/TensorRT remain detection or skeleton paths as documented. | Use dedicated target runners and publish correctness/performance artifacts for each supported accelerator combination. Choose one adapter with an owner, hardware and pinned model before implementing another runtime. |
| P1 | **SDK compatibility and native distribution are unfinished.** ABI revisions 1/2 exist; this iteration makes wrapper packaging reproducible and streaming bounded. It still produces a pure-Python wheel requiring a separately supplied native library. | Build platform binary artifacts/wheels with provenance and dependency policy, test the declared Python/platform range, and run installed-wrapper tests against versioned native libraries before promising a compatibility window. |
| P1 if containers are supported | **No complete official container deployment contract.** The Docker build is hardened and smoke-tested, while official multi-architecture publication, final-layer inventory and operational reference deployment remain unproven. | Publish by digest with SBOM/provenance, scan the final runtime, and validate read-only root filesystem, persistent data ownership, probe ACLs, proxy configuration and signal drain on the target orchestrator. |
| P2 | **Large protocol/UI implementation modules still concentrate review risk.** The application services are separated and architecture checks pass; handlers and UI still combine many concerns. | Extract protocol-specific request/response adapters and UI workspaces without changing wire behavior. Keep boundary tests and the no-default-features build required for every extraction. |
| P2 | **Accessibility and incident operations need environment-level validation.** Chromium/axe gates and local signed revocation enforcement exist; cross-browser/assistive-technology behavior and multi-host revocation recovery are not established. | Add Firefox/WebKit and a target screen-reader workflow; run a signed-index incident drill measuring refresh latency, alerting, durable state restore and replacement-digest recovery. |

These priorities distinguish intentionally bounded behavior from defects.
Process-local Responses retention, bi-encoder reranking, single-image Chat
input, unsupported reasoning controls, and delegated TLS/network rate limits
are published product boundaries. Extend them only against an explicit use
case and an achievable verification contract. In particular, state persistence
would add a migration and recovery obligation; it should not be introduced as
an incidental route-level change.

## Reproduced defects and implemented iteration

### Python stream backpressure and lifecycle

[`pipeline.py`](../python/bloom_sdk/pipeline.py) previously handed every callback
to an unbounded `queue.Queue`. A native producer could finish a large stream
while its consumer paused, retaining every parsed output object. Adding only
`maxsize` would introduce another problem: pipeline close could wait on a
native call blocked behind the abandoned consumer.

[`_stream.py`](../python/bloom_sdk/_stream.py) now bounds queued serialized output
to 64 chunks and 16 MiB. A full handoff blocks the producer until consumption
or stop. Terminal state occupies a separate slot, so completion and errors
cannot wait behind a full queue. The consumer parses one chunk at a time.
Oversized revision 2 callbacks fail before dereferencing their declared byte
range. Stream close and pipeline close release blocked callbacks, request
cooperative cancellation, and preserve native handle/token ownership until
the native call finishes. A worker that cannot start releases its registration
and token. Legacy native calls drain without accumulating discarded output.

This bounds the handoff, not application-retained objects or model memory.
Native cancellation remains cooperative, and revision 1 does not support
native cancellation. A blocked external runtime can still delay pipeline close.
Calls on a single pipeline remain serialized.

### SDK source and wheel packaging

The original `readme = "../README.md"` in
[`pyproject.toml`](../python/pyproject.toml) caused the build backend to reject
the source distribution because its README was outside the package root.
The package now includes a local README and the repository license.

[`test_python_package.py`](../scripts/test_python_package.py) copies the SDK into
a temporary tree, builds its sdist and then its wheel from the sdist, checks
metadata/license contents, installs with no package-index lookup into a fresh
environment, verifies import without native code, and exercises the installed
wrapper against the real ABI v2 mock engine. The gate checks license equality
with the repository to prevent copied-license drift. CI runs it after building
the native library. This is a wrapper distribution gate, not binary-wheel
publication or trained-model validation.

### Performance evidence and classification

The old checker returned `PASS` and exit code 0 for an x86-like record containing
`ttft_ms: -1` with no TBT or peak memory. It also allowed invalid document shapes
to crash, overwrote top-level timing averages with nested fallback values, and
could report exact 5% regressions as failures because of floating-point
subtraction. Its hardware heuristic classified macOS ARM CPU as x86 and
OpenVINO CPU as NPU.

[`bench_budget_check.py`](../scripts/bench_budget_check.py) now validates required
measurements and metadata before hardware classification. Invalid evidence is
exit code 4 and always fails the smoke gate; exit code 3 only means valid data
has no matching hardware budget. Timing fallback has explicit precedence,
model estimates cannot fill an absent peak measurement, and the 5% WARN
boundary is inclusive. `bloom_bench` publishes architecture, and CPU selection
takes precedence over backend capability. Existing budget values are unchanged;
the documentation now correctly labels their MiB units.

This closes a false-positive quality gate. It does not create a measured
performance baseline or prove that all reported peak memory is observed.

### TileLang private executable cache

The full native baseline failed
`compiler::tests::compiler_uses_a_private_process_cache`: the cache inherited
group/other permissions through the normal umask. The
[`compiler`](../crates/tilelang/src/compiler.rs) now requests Unix mode `0700`
from `tempfile::Builder` at directory creation, before publishing the cache
path. The existing regression test passes. This validates the directory
boundary, not actual accelerator kernel execution or Windows ACL behavior.

Existing Rust changes also had formatting failures. `cargo fmt --all` restores
the workspace formatting gate without reverting the ongoing architecture work.

## Architecture interpretation

The seven-crate dependency graph and backend-neutral batch/model contracts are
real improvements and the architecture regression gate passes. File length
alone would exaggerate some remaining problems: server `lib.rs` has 11,202
lines, but its inline test module starts at line 985. Its production composition
root is about 983 lines. The clearer extraction candidates are the 5,276-line
handler module, roughly 4,250 lines of Ollama adapter production code, roughly
3,514 lines before Candle's inline tests, and roughly 6,208 lines before UI
inline tests. These are reviewability indicators, not proof of a defect.

Avoid another broad layer rewrite solely to reduce file counts. Extract
well-defined protocol or UI responsibilities with existing behavioral tests,
and keep backend implementations out of application contracts.

## Validation and limits

Local host: macOS/aarch64, pinned Rust 1.97.1. Python boundary checks ran on
3.9 and 3.12; isolated packaging and native interoperation used Python 3.12.

- Native workspace: **933 tests passed**, including the previously failing
  private-cache test.
- UI host tests: **134 passed**.
- Python SDK: **20 passed** with native FFI enabled, including count/byte
  backpressure, both ABI lifecycle paths, malformed/oversized output,
  terminal delivery and allocation/start cleanup.
- Benchmark evidence: **8 regression tests passed**, with parameterized
  invalid-data, hardware selection, timing precedence and exit-code cases.
  A real tiny-model `bloom_bench` run emitted `hardware.arch: aarch64`; its valid
  measurements produced exit code 3 because there is no ARM CPU budget, rather
  than claiming an x86 performance pass.
- Architecture: source/dependency checks and **10 regression tests passed**.
- Strict workspace Clippy and the all-target no-default-features build passed.
- The SDK sdist/wheel clean-install gate passed using a freshly built native
  library; import and native buffered/streamed mock inference both succeeded.
- The deterministic tiny Qwen2 CPU gate passed single-file/sharded text, IFB,
  embedding, reranking, structured-output and function-call paths through the
  OpenAI/Ollama adapters. Optional official-client checks were not required in
  this local run; this does not replace CI's pinned client matrix.
- Live HTTP boundary and SIGTERM drain/escalation/deadline checks passed.
- Workspace formatting, local Markdown links, toolchain consistency, immutable
  GitHub Action references and whitespace checks passed.
- The HTTP workload collector passed five SSE/evidence regressions and the
  full deterministic tiny Qwen2 CPU gate. A separately downloaded pinned Qwen2
  Q4_0 GGUF file matched SHA-256
  `aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f`;
  a local macOS/ARM debug server completed the `1,2,4` streaming workload
  (seven requests, 42 reported completion tokens), disconnect recovery, and
  clean shutdown. Debug timing is not a performance baseline.
- A rebuilt release server with the same verified Qwen2 file passed a 30-second
  Apple M5 CPU run: 49 completed streaming requests, 441 reported completion
  tokens, first-content p95 2,534 ms, two accepted disconnects with recovery,
  and clean shutdown. The exact binary/model hashes and measured host RSS are
  retained in the local workload report. This single short run is not a stable
  performance budget or a soak result.

This iteration does not re-run the
downloaded trained-model matrix, GPU execution, a release-container/browser
build, multi-hour soak, or Windows/macOS binary-wheel validation. Those remain
separate acceptance work; the support labels stay unchanged.

## Suggested next iteration

The first candidate cell is the pinned Qwen2 0.5B Instruct Q4_0 GGUF profile
on native Candle CPU with OpenAI streaming. The bounded
[`http_workload.py`](../scripts/http_workload.py) runner now records exact
model/binary hashes, cold startup, concurrency, first-content and inter-delta
latency, completion usage, host RSS, disconnect recovery, and clean shutdown.
The deterministic tiny-model gate and pinned Qwen2 trained gate run short
`1,2,4` concurrency workloads; the [operator procedure](http-workload.md)
uses the same report for a longer run. A local short tiny-model run passed on
macOS/ARM, but that untrained fixture is only a mechanical check. The pinned
trained workload also passed locally with the verified Qwen2 GGUF, and a
separate release run established one short host-specific observation. No
multi-hour target deployment result is available yet.

Next, retain repeated multi-hour reports on the chosen immutable host and
perform bad model load, disk-full, interruption, upgrade, and rollback drills.
Publish the measured baseline and failure-injection results before widening
the stable-support claim. The corrected evidence gate and SDK lifecycle
boundary provide a safer foundation for that work.
