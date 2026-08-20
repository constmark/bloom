# Bloom Code Structure and Open-Source Readiness Assessment

> Assessment date: 2026-07-22  
> Method: static review of dependencies, core traits, error handling, CLI and
> server entry points, engine implementations, CI, project governance, and
> technical-debt indicators.
>
> **Current status:** Sections 1–5 preserve the original assessment. Sections
> 6–8 record subsequent remediation; Section 8 is the current 2026-08-15
> reassessment and remaining-risk list.

## Executive Summary

Bloom has a clean high-level architecture and real MVP execution paths, but the
implementation still carries substantial cleanup work. End-to-end correctness
is not enforced by default CI because real-model tests may skip when model files
are unavailable. The repository contains the core governance, CI, examples, and
documentation expected from a public open-source project.

| Area | Assessment | Score |
| --- | --- | ---: |
| Architecture and module boundaries | Clear layering, no cyclic dependencies, strong abstractions | 8/10 |
| Implementation quality | Excessive panic-prone calls and several very large files | 6/10 |
| MVP usability | Real code paths exist, but golden end-to-end coverage is missing | 7/10 |
| Open-source readiness | License, CI, documentation, and examples are present | 8/10 |
| Overall | Ready to open with visible follow-up work | 7/10 |

## 1. Architecture and Module Boundaries

### 1.1 Dependency graph

The six workspace crates form a directed acyclic graph:

```text
core
  ├── backend ──┐
  └─────────────┼── engine ── ffi
tilelang ───────┘       └──── server
backend ─────────────────────┘
```

`core` does not depend on `engine`, and dependencies flow toward lower-level
contracts. This is one of the repository's strongest design choices.

### 1.2 Engine routing

The `Engine` abstraction in `crates/engine/src/core/engine.rs` is a notable
strength:

- `EngineCapability` declares supported model families, dtypes, formats,
  devices, modalities, quantization methods, and maturity before loading.
- `SupportLevel` distinguishes `Native`, `Fallback(reason)`, and
  `Unsupported(reason)` instead of returning an unexplained Boolean.
- `EngineRouter::select_engine` prefers native support, then the best fallback,
  and returns structured reasons when no engine can run a request.
- Unit tests cover registration, capability matching, and routing branches.

### 1.3 Error diagnostics

`crates/core/src/error.rs` defines structured `BloomError` and `ResourceError`
types. Recovery hints make failures actionable, while error categories support
monitoring and API-level reporting.

### 1.4 Unsafe boundaries

Unsafe code is concentrated around appropriate systems boundaries: the C ABI,
dynamic plugin loading, JIT kernels, and native GPU interoperability. It is not
widely distributed through orchestration logic.

## 2. Implementation Quality

### 2.1 Panic-prone recoverable paths

The engine crate contains many uses of `unwrap`, `expect`, and `panic`, including
model loading, weight handling, and scheduling paths where errors should be
recoverable. These should return structured `BloomError` values instead. This is
the largest gap between the current implementation and production reliability.

### 2.2 Very large source files

Several source files combine multiple responsibilities. Examples include the
server entry point, the Candle executor, the scheduler, multimodal executors,
and the standalone CLI. Splitting routing, request handling, interactive UI,
benchmarking, inspection, model loading, and generation loops into focused
modules would improve reviewability and testability.

### 2.3 Repeated streaming logic

The Qwen, Gemma, and LongCat streaming paths repeat KV-hook and sampling-loop
patterns. A shared streaming abstraction would reduce backend maintenance cost
and make behavior more consistent.

### 2.4 Abstraction leakage

Some public engine abstractions expose Candle-specific tensor types behind
feature gates. Crate-root aliases also expose internal module naming. Both are
reasonable short-term facade choices, but they narrow portability and should be
reviewed before a stable API release.

### 2.5 Broad lint allowances

The engine crate allows a large set of Clippy lints at crate scope. This weakens
the practical effect of the CI lint gate. Prefer targeted allowances with a
local explanation.

### 2.6 Runtime and scheduler ownership

`crates/core` contains runtime, scheduler, VRAM, token scheduling, unified
memory, KV overlay, and online-switching modules, while `crates/engine` also has
an engine-level scheduler. The ownership boundary between those layers, and
between Bloom and external runtimes, should be documented explicitly.

## 3. MVP Usability

### 3.1 Real execution paths

The Candle executor reads model configuration, tokenizers, and weights; builds
architecture-specific loaders; performs attention computation; and implements
streaming decode. The CLI registers multiple engines, while `bloom_server`
provides OpenAI-compatible generation, embeddings, reranking, metrics, and
inspection endpoints. This is a functional product foundation rather than a
pure interface demonstration.

### 3.2 Validation risks

Default CI can compile, lint, test, validate schemas, and exercise smoke-test
control flow without proving that a real model emits correct output. Real-model
checks may skip when no model is configured. Production claims therefore need a
separate required gate with pinned model provenance and reproducible output.

Support labels must also stay consistent between the README, architecture
documents, and `docs/support-matrix.md`. Until a path has reproducible
real-model evidence, `experimental` is the appropriate public status.

## 4. Open-Source Readiness

### Strengths

- Apache-2.0 license and standard governance documents
- CI covering formatting, compilation, Clippy, tests, packaging, feature
  checks, schema validation, smoke tests, documentation, and security auditing
- Architecture, security, plugin, scheduling, roadmap, and support documents
- Public schemas, model manifests, plugin examples, a C++ FFI example, and a
  vendor SDK adapter template
- Explicit crate publishing order and release gates

### Follow-up items

1. Decide whether `Cargo.lock` should be committed for reproducible binary
   builds; it is currently ignored.
2. Keep the README's repository layout synchronized with the workspace,
   including the FFI crate and the full scope of `core`.
3. Keep example build outputs ignored and out of source distributions.
4. Track major cleanup work explicitly instead of leaving technical debt
   implicit in large files and panic-prone paths.

## 5. Recommended Priorities

| Priority | Action | Expected benefit |
| --- | --- | --- |
| P0 | Replace panic-prone recoverable engine and scheduler paths with structured errors | Prevent production crashes and improve diagnostics |
| P0 | Add a reproducible real-model golden test or keep support claims experimental | Make MVP correctness independently verifiable |
| P1 | Split the largest CLI, server, executor, and scheduler files by responsibility | Improve readability, testing, and onboarding |
| P1 | Introduce a shared streaming abstraction | Reduce duplicated backend logic |
| P1 | Clarify scheduler, runtime, memory, and external runtime ownership boundaries | Reduce architectural ambiguity |
| P2 | Review lockfile policy and keep repository layout documentation current | Improve release reproducibility and trust |
| P2 | Narrow crate-level Clippy allowances | Strengthen the lint gate |

## Conclusion

Bloom is ready to be developed in public and has stronger architecture,
documentation, CI, and extension contracts than many projects at a similar
stage. It should still present itself as experimental until recoverable errors,
large implementation modules, and real-model validation are addressed. The
first post-open-source priorities should be structured engine error handling
and decomposing the largest source files.

## 6. Remediation Log (2026-07-22)

Two P0/P1 items from Section 5 were addressed.

### 6.1 Split the 3800-line `bloom_server` binary (P1)

`crates/engine/src/bin/bloom_server/main.rs` (3849 lines) was decomposed without
changing behavior:

The later application-layer split moved this composition root and its modules
to `crates/server`; the table below records the earlier remediation state.

| File | Lines | Contents |
| --- | --- | --- |
| `main.rs` | 1310 | `Args` config struct, `ServerState`, `ApiError`, DTOs, error helpers, `main()` wiring, `mod` declarations |
| `cli.rs` | 303 | CLI arg parsing helpers (`parse_args`, `apply_config`, `select_backend_name`, `build_long_context_policy`) + the two `apply_config_*` macros |
| `handlers.rs` | 1897 | All HTTP handlers (`/health`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/world/step`, `/cancel`, `/backends`, …) |
| `helpers.rs` | 356 | Prompt building, structured-output JSON-schema validation, embedding math, request-id generation |

Mechanism: each submodule re-exports its `pub(crate)` items to `main.rs` via
`use cli::*; use handlers::*; use helpers::*;`, and reaches the others through
`main.rs`'s glob re-exports (`use super::*;` inside each submodule). The split
**verified green**: `cargo check --bin bloom_server --workspace` and
`cargo clippy --bin bloom_server --workspace` (the repo's `-D warnings` gate)
both pass.

### 6.2 Panic convergence on recoverable paths (P0) — reassessment + fix

**Reassessment:** the review's "918 engine-layer unwraps" overstated the
user-facing panic risk. On inspection, the default Candle backend's
**model-loading path is already structured-error-clean**: `CandleEngine::load`
returns `anyhow::Result` and uses `anyhow!` / `?` / `.map_err()` for config
reading, config parsing, tokenizer loading, and CUDA/Metal device init. A bad
model file, missing tokenizer, or absent device therefore already returns a
diagnostic error rather than panicking. The 687 bare `.unwrap()`/`.expect()`
sites (executor + scheduler) are concentrated in: (a) runtime-inference internal
invariants (`Mutex` guards, `model.as_mut()`, `.last()` on non-empty buffers),
(b) structured-output JSON-shape assumptions, and (c) `#[cfg(test)]` code.

**Fix applied — the highest-severity, server-crash class:** 196 `Mutex::lock()`
/ `RwLock::read()` / `RwLock::write()` panic sites across `executor/*` and
`scheduler/*` were converted from `.unwrap()` to
`.unwrap_or_else(|e| e.into_inner())`. This **recovers from lock poisoning**
instead of panicking the whole process — a poisoned mutex (caused by another
thread panicking) now yields the guard and keeps serving, rather than taking
down the server. This is a drop-in change valid in any function return type
(`anyhow::Error` cannot wrap a `!Send` `MutexGuard`, so `?` does not apply here).

**Governance:** `#![warn(clippy::unwrap_used)]` was added to
`crates/engine/src/lib.rs`. It surfaces the remaining unwraps in CI as warnings
(without breaking the build) and blocks new recoverable-path panics from
regressing. Default-feature clippy currently reports 59 `unwrap_used` warnings;
more appear under the full backend feature set. Recommended next phase: convert
the remaining internal-invariant `.unwrap()`/`.expect()` calls (preferring
`.expect("clear invariant reason")` where a logic bug is intended) and flip the
lint to `#![deny(clippy::unwrap_used)]` once the count is driven down.

### 6.3 Verification

- `cargo check --workspace` — passes.
- `cargo clippy --bin bloom_server --workspace` — passes (warnings only).
- `cargo clippy -p bloomai-engine` — passes; 59 `unwrap_used` warnings now
  tracked (were silently allowed before).

## 7. Reassessment and Iteration Log (2026-08-14)

The default workspace baseline passed formatting, all-target compilation,
strict Clippy, and 772 Rust tests before this iteration. The original report's
real-model-gate and lockfile concerns are now stale: `Cargo.lock` is committed,
and Linux CI requires deterministic native fixtures plus pinned trained Qwen2,
Qwen3, Llama, Safetensors, and MiniLM embedding gates instead of silently
skipping every real execution path.

### 7.1 Completed in this iteration

- Removed the remaining runtime `unimplemented!` crash paths. Unavailable Gemma
  FlashAttention and placeholder Conv3d execution now return actionable Candle
  errors, with regression tests.
- Hardened the pre-1.0 C ABI. NULL stream callbacks and zero context sizes fail
  closed, load/run/stream entry points catch internal Rust panics, unsafe entry
  points carry explicit safety contracts, and the crate no longer needs broad
  `missing_safety_doc` or `needless_return` lint allowances.
- Hardened the Python wrapper. Import no longer loads the native library;
  generation parameters and NUL-bearing identifiers are validated; returned C
  strings are freed on decode failures; and a per-pipeline lock prevents
  `close()` from freeing a handle during active buffered or streaming work.
- Added model-free Python contract tests and a real Python → `ctypes` → Rust
  mock-engine round trip to Linux CI.
- Reconciled release metadata after the server application-layer split. All six
  publishable crates now declare Rust 1.97.1, the consistency checker enforces
  workspace/UI declarations, and the release checklist includes
  `bloomai-server`, the Python/FFI gate, and the trained embedding gate.
- Corrected documentation that prematurely called the ABI stable and added an
  ownership, threading, and failure contract for C and Python consumers.

### 7.2 Remaining priorities at that checkpoint

| Priority | Remaining gap | Exit condition |
| --- | --- | --- |
| P1 | Production modules remain too large: UI `main.rs` (~6.2k lines), UI `api.rs` (~5.5k), server `handlers.rs` (~5k), and Candle executor (~3.4k before tests) | Split by feature boundary while preserving focused module-level tests |
| P1 | The engine crate still carries broad crate-level Clippy allowances | Move each necessary allowance to the smallest item/module and remove obsolete entries |
| P1 | C ABI streaming has no cancellation function, and string inputs are NUL-terminated rather than length-delimited | Add a versioned, length-aware ABI with cancellation before declaring stability |
| P1 | Metal/CUDA benchmark evidence is not a required cross-platform release artifact | Publish reproducible hardware profiles using the benchmark schema |
| P2 | ONNX, TensorRT, CoreML, MLX, and Vulkan remain truthful diagnostic skeletons | Keep them non-routable until an executable adapter has pinned runtime evidence |

The next implementation iteration should split the HTTP handler and UI feature
surfaces before adding more protocol behavior. That work is primarily a review
and ownership improvement, so it should be done in small behavior-preserving
changes with the existing HTTP and UI state suites kept green.

## 8. CI Reliability and Lint-Scope Iteration (2026-08-15)

The failing `main` workflow at commit `a28da32` was reproduced against Linux
process behavior and a locally cross-compiled Windows workspace. Its three
blocking jobs had independent causes rather than one shared toolchain failure.

### 8.1 GitHub CI failures remediated

- The Linux shutdown test used a fixed 50 ms sleep after writing a partial HTTP
  request. On a contended runner the server could begin shutdown before it had
  accepted that connection. The test now waits for a unique request ID in the
  server's HTTP trace before sending SIGTERM, making the active drain observable
  instead of timing-dependent.
- Windows Clippy exposed an Unix-only local variable and an implicit
  `OpenOptions` truncation policy. The variable is now conditionally compiled,
  while the persistent catalog lock explicitly uses `truncate(false)` so an
  acquisition attempt cannot erase an invalid non-empty lock file.
- The RustSec Action tried to create a GitHub Check while the workflow correctly
  defaulted to `contents: read`. CI now runs pinned `cargo-audit` directly; it
  requires no write-capable token and therefore works for push, Dependabot, and
  fork pull-request contexts.
- Node 20 actions were upgraded to current full-SHA-pinned releases across CI
  and release workflows. The immutable-action and release-provenance contract
  tests remain blocking.

### 8.2 Engine Clippy policy narrowed

The 20-entry crate-root `clippy::allow` block was removed. Clippy's safe fixes
resolved the mechanical findings, including derivable defaults, redundant
fields, needless borrows/returns/question marks, manual range/repeat/divisibility
operations, and an actual repeated-`Vec::with_capacity` allocation bug. The only
remaining exceptions are `needless_range_loop`, `too_many_arguments`, and
`type_complexity`; each is now scoped to the specific tensor/model/scheduler
module with a rationale instead of disabling the lint across the engine crate.

### 8.3 First server ownership split

Cross-platform signal listeners and graceful-shutdown deadline coordination now
live in `crates/server/src/shutdown.rs`. This is a small first extraction from
the server composition module and gives subsequent HTTP/runtime splits a clear
process-lifecycle boundary.

### 8.4 Current remaining priorities

| Priority | Remaining gap | Exit condition |
| --- | --- | --- |
| P1 | UI `main.rs`/`api.rs`, server handlers and model-management modules, and Candle executors remain too large | Continue behavior-preserving splits by protocol and runtime ownership, with focused tests per extracted module |
| P1 | C ABI streaming still lacks cancellation and length-delimited string inputs | Ship a versioned ABI revision and migration path before declaring stability |
| P1 | Metal/CUDA benchmark evidence is not a required cross-platform release artifact | Publish reproducible hardware profiles using the benchmark schema |
| P2 | `paste` 1.0.15 is an unmaintained transitive dependency through Candle/gemm/tokenizers | Track upstream replacement and remove it when the model stack supports a compatible release |
| P2 | ONNX, TensorRT, CoreML, MLX, and Vulkan remain truthful diagnostic skeletons | Keep them non-routable until an executable adapter has pinned runtime evidence |

## 9. Versioned Native-Integration Iteration (2026-08-20)

The C/Python P1 from Sections 7 and 8 is now implemented without removing the
revision 1 entry points:

- `bloom_abi_version()` reports ABI revision 2, while the original symbols
  remain available for existing binaries.
- New `_v2` load, buffered-run, and streaming entry points use bounded
  length-delimited UTF-8 inputs. Buffered output is length-delimited and has an
  explicit clearing free function.
- Stable status codes distinguish invalid arguments, UTF-8, each JSON input,
  inference, output serialization, caught panics, and cooperative cancellation.
- A thread-safe cancellation token stops streaming at the next output-sink
  boundary without freeing a token that is still owned by a native worker.
- The Python SDK negotiates revision 2 automatically, retains a tested revision
  1 fallback, and cancels native work when a partially consumed generator is
  closed.

The remaining ABI work is release engineering rather than the original unsafe
shape: publish binary wheels, declare a compatibility window, and test packaged
old/new shared-library combinations on supported targets. The highest-priority
code-structure gap remains the oversized server/UI modules. At that checkpoint,
the highest production-evidence gaps were one stable deployment cell, real
browser and accessibility coverage, hardware benchmarks, and artifact
SBOM/license policy; Section 10 records the native archive remediation.

## 10. Release Supply-Chain Iteration (2026-08-20)

The native artifact SBOM/license-policy gap is now closed for application
archives:

- A versioned repository policy admits only the reviewed crates.io registry,
  workspace-local path packages, and the exact declared license expressions in
  the locked dependency graph. Git, external path, alternate registry, missing
  license, and expression drift fail closed.
- Packaging resolves and validates the native target graph plus the independent
  wasm UI graph, when embedded, before either build starts. It emits a
  deterministic CycloneDX 1.5 SBOM without timestamps or local paths.
- Each schema-version-2 archive carries the SBOM and the policy that admitted
  it. The offline validator binds the SBOM to the Bloom version, target,
  embedded-UI setting, component inventory, dependency edges, sources, and
  licenses while retaining validation for legacy version-1 archives.
- CI runs policy drift and negative contract tests independently of full
  packaging.

This does not close the separate official-container gap: a future published
image still needs final-layer inventory, SBOM/provenance attachment, scanning,
and orchestrator evidence. For the native application, the next code-level P1
is real-browser/accessibility coverage or behavior-preserving decomposition of
the largest server and UI modules.

## 11. Browser Semantics Iteration (2026-08-20)

A manual real-browser pass against the embedded empty-model application now
verifies the baseline navigation and modal lifecycle instead of relying only on
host-side state tests:

- The document has one descriptive title, a favicon, one level-one Bloom
  heading, and a named conversations landmark.
- Runtime connection changes use a polite atomic status region.
- The Models dialog receives initial focus, contains forward and reverse Tab
  navigation, closes with Escape, and restores focus to its opener. Settings
  controls expose accessible names in the browser accessibility tree.

This is evidence for the audited empty state, not an automated release gate.
Cross-browser keyboard flows, browser downloads/clipboard, screen-reader
behavior, and an automated accessibility scanner remain in the high-priority
register.
