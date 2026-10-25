# Bloom project assessment — 2026-10

## Scope

This follow-up reviewed the native backend probes, FFI boundary, Python SDK,
Windows and macOS path handling, NPU/TTS defaults, workload evidence tooling,
and the local regression suite. The focus was cross-platform usability and
Unicode/multilingual input rather than adding another inference backend.

## Findings and remediation

| Area | Finding | Remediation |
| --- | --- | --- |
| Windows memory accounting | Windows capability reports used a fixed fallback and did not expose available physical memory. | Added `GlobalMemoryStatusEx` probing for total and available memory, with bounded conversion and scheduler-safe Linux clamping. |
| Backend discovery | The generic `gpu` alias always selected CUDA, and Python-backed probes assumed `python3`. Intel NPU discovery could treat any existing `PATH` entry as an OpenVINO installation. | `gpu` selects Metal on macOS; interpreter discovery honors `BLOOM_PYTHON` and platform launchers; NPU probes use explicit OpenVINO paths and common ARM64/runtime locations. |
| Native paths | ABI v2 limited model paths to the identifier limit, excluding long and multibyte Windows paths. Home-directory lookup ignored `HOMEDRIVE`/`HOMEPATH`; Llama.cpp and NPU/TTS helpers had Unix-only or hard-coded roots. | Added a 128 KiB UTF-8 model-path limit, Windows home fallbacks, portable Llama.cpp lookup, and configurable per-user model roots. |
| Python SDK | `PathLike` model paths, non-ASCII JSON, structured-output parameters, and localized diagnostics were incomplete. | Added UTF-8-preserving `PathLike` support, BCP-47 aliases for English/Chinese diagnostics, OpenAI/native response-format normalization, and response-shape validation. |
| Packaging | Setuptools 84 emitted deprecation warnings for the old license metadata form. | Migrated to SPDX `license` and `project.license-files`, requiring the supporting setuptools version. |
| Workload evidence | RSS and host-memory collection relied on POSIX commands, so Windows reports could be invalid or localized; the requested `auto` device was not independently represented in the workload artifact. | Added locale-independent Windows PSAPI and `GlobalMemoryStatusEx` collection, report the measurement source, and test the sampler in-process. The workload now records a bounded, authenticated `/v1/backends` physical-device inventory after readiness, so accelerator availability is evidence rather than a CLI selector claim. |

## Validation

The local macOS/aarch64 run passed the full Rust workspace test suite, strict
Clippy, formatting, architecture checks, JSON artifact validation, HTTP
workload and benchmark evidence tests, the C ABI tests (including long
multibyte paths), the Python SDK unit and native ABI integration suites, and
the isolated SDK sdist-to-wheel installation gate. Cross-target checks passed
for the backend on Windows MSVC/MinGW and macOS ARM targets. The FFI Windows
MinGW check remains dependent on an upstream `esaxx-rs` build script that
unconditionally adds a libc++ flag; Windows host CI is the authoritative check
for that toolchain.

The broader support matrix remains accurate: most model/device combinations
are experimental until they have target-hardware correctness, performance,
and soak evidence. This iteration improves the boundary and deployment
experience without promoting an unmeasured accelerator path to stable.

## Edge GUI iteration (2026-10-10)

The repository now has a complete inventory path for the graphical runtime:
the server exposes engine declarations and live physical-device probes from
`GET /v1/backends`, the model manager can inspect/import/download/preflight
packages, and the connection indicator is an actionable control. On desktop
and mobile layouts, clicking the indicator opens the next useful surface:
settings for connection failures, model management for an unavailable model,
and runtime diagnostics when the active model is ready. The full state remains
available to assistive technology through the button label and tooltip.

Startup no longer requires a hardware guess. `--device auto` is the default
for the server, standalone inference, and benchmarking commands. A shared
backend-registry policy checks the selected engine's declared devices, probes
compiled/runtime availability, prefers GPU then NPU, and falls back to CPU.
Metal is reported unavailable when the binary was not built with its feature,
which prevents a GUI or auto-selection from promising an accelerator that
cannot execute in the current binary. Windows now contributes real available
physical memory to runtime admission, and malformed EOS token metadata is
rejected instead of being silently truncated.

The remaining release blockers are evidence and vendor coverage rather than
missing control flow: CoreML/Vulkan/TensorRT adapters still need executable
runtime integrations; the MLX adapter now executes text generation through the
installed MLX-LM bridge but remains experimental; external OpenVINO/ASR/NPU paths now ship
their Python bridges but still require the corresponding packages and model
profiles; ONNX now has an executable, packaged Python `onnxruntime` bridge but
remains experimental until real model/provider evidence is published; and a stable deployment cell still needs target-hardware soak,
performance, cancellation, upgrade/rollback, and trained multimodal quality
evidence. Those paths remain explicitly labelled experimental, external, or
skeleton in the support matrix and in the GUI capability response.

## 2026-10-14 follow-up

The repository was fetched from `origin` and is current at `b277190`; the
remote has no newer `main` commit or release tag. This follow-up closed two
runtime correctness gaps in the CacheMesh/IFB path: both the shared Qwen KV
hook and the production handle-keyed Candle hook now truncate model-owned KV
state when speculative or structured decoding rolls back tokens. Regression
coverage exercises the shared hook through extraction, CacheMesh restoration,
batch execution, and rollback.

The lockfiles now pin `quinn-proto` 0.11.18, and the architecture gate documents
and tests the intentional `bloomai-app -> bloomai-backend` device-selection
edge. Workspace Rust tests, no-default-feature checks, strict architecture
tests, and Python SDK tests pass after these changes.

This environment also completed the MLX gap: the M5/Metal 4 bridge loads a
fixed Qwen2.5 0.5B MLX package and passes model-backed OpenAI/Ollama server
protocol smoke checks. The Safetensors resolver and package validator now
accept a matching single-file index while continuing to reject ambiguous
shard layouts. The MLX record does not claim an exact semantic answer for this
checkpoint. Metal has pinned Qwen2 Q4_0 trained-model and short
concurrency/disconnect evidence; these observations still do not replace a
long soak or broader model matrix.

The Metal trained profile used the immutable Qwen2 Q4_0 file
`aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f` (352,969,408
bytes). A 1/2/4-concurrency workload completed 7 requests and 42 tokens,
validated the semantic `Bloom` response, recovered from one accepted disconnect
in 1.531 seconds, and measured a 6.199-second first-content p95 with 2.763 GB
peak host RSS. This is a short M5 observation, not a long-term budget.

The highest-value remaining work is still hardware- and evidence-bound: make
one of CoreML, Vulkan, or TensorRT executable behind its vendor runtime;
publish real ONNX/OpenVINO/ASR provider and model gates; add trained
multimodal and cross-encoder evidence; then run the multi-hour deployment,
cancellation, disk-failure, upgrade/rollback, and cross-platform wheel drills
before promoting any experimental support label.

## 2026-10-24 follow-up

This pass closed the software-completable production gaps found in the runtime
and model-management audit. Native Qwen2-MoE batching now shares immutable
weights and bounded GPU expert storage across independent request KV state;
eviction respects active leases and reports cache pressure. Request teardown
also clears speculative-decoding state, preventing handle reuse from carrying
strategy state into a later request. The embedded server's tracing setup is
idempotent, so repeated in-process startup no longer panics.

Model inventory restore now has an authenticated, bounded bulk endpoint for up
to 32 exact source/checksum entries. Admission is fail-closed, transfers are
serialized through the verified no-overwrite path, and each entry returns an
explicit rejection when it cannot be restored. The GUI exposes the same action
with busy-state and result feedback. Legacy v1 index watermarks have a
regression test covering migration into the v2 history, and CI now compiles
the embedded UI feature as a release-facing gate.

Validation passed for the full workspace (including no-default-feature and
WASM UI checks), UI tests, strict Clippy, formatting, architecture tests, JSON
artifact validation, and the server/UI bulk-restore contracts. These changes
improve correctness and operability but do not change the support labels:
stable deployment still requires target-hardware multi-hour soak,
failure-injection and upgrade/rollback drills, official multi-architecture
container provenance, binary-wheel ABI matrices, executable CoreML/Vulkan or
TensorRT adapters, and trained multimodal/tool-quality evidence.

## 2026-10-25 static follow-up

The repository was refreshed against all advertised `origin` refs. `dev`
already contains the current `origin/main` history and the 2026-10-24 runtime
work. The remaining stale dependency declarations were reconciled without
starting a model or server: the direct Metal integration now uses 0.33.0 while
the Candle 0.29 transitive path remains explicit in `Cargo.lock`; the MLX,
MLX-LM, and OpenAI compatibility pins now match the latest reviewed remote
updates.

The static audit also found three software-level reliability gaps. External
process diagnostics had a reusable bounded reader only inside the ONNX
adapter, while MLX used unbounded `wait_with_output`; the bounded runner now
lives at the engine process boundary, applies a hard deadline to both Python
bridges, and drains noisy pipes concurrently. ffmpeg retained all stderr in
memory and drained it only after writing every frame; it now drains
concurrently while retaining at most 64 KiB. Server metrics could wrap the
in-flight gauge on a duplicate completion, admit NaN/negative latency
observations, or underflow KV utilization when a malformed snapshot reported
more free blocks than total; all three paths now fail closed and have
regression coverage.

No inference service, model loader, external bridge, browser workload, or
hardware test was run in this pass. The unclosed release gaps therefore remain
unchanged: a measured stable deployment cell, multi-hour and failure-injection
evidence, official container provenance, binary-wheel compatibility matrices,
and executable CoreML/Vulkan/TensorRT adapters.
