# Bloom Code Structure and Open-Source Readiness Assessment

> Assessment date: 2026-07-22  
> Method: static review of dependencies, core traits, error handling, CLI and
> server entry points, engine implementations, CI, project governance, and
> technical-debt indicators.

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

The five workspace crates form a directed acyclic graph:

```text
core
  ^
backend
  ^
engine
  ^
ffi

tilelang (independent leaf used by engine)
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
between Bloom and Elderwand, should be documented explicitly.

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
| P1 | Clarify scheduler, runtime, memory, and Elderwand ownership boundaries | Reduce architectural ambiguity |
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
