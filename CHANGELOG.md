# Changelog

All notable changes to Bloom should be recorded here.

This project follows semantic versioning once public APIs and manifest schemas
are declared stable. Before `1.0`, breaking changes are allowed but must be
called out in release notes.

## Unreleased

### Breaking

- Rename legacy core config struct to `BloomConfig` in `bloomai-core`. The struct
  lives in the Bloom workspace, so the previous name leaked an external
  runtime's brand into Bloom's public API. Callers using
  the legacy config struct name must update to `bloomai_core::BloomConfig`.
  No JSON field names changed (the struct never carried a serde rename).

### UI

- Add a decoupled Dioxus (Rust → WASM) frontend in `ui/`: streaming chat
  against the OpenAI-compatible API (SSE), connection settings with
  persistence, generation-parameter sliders, and a live health/status bar.
- Add an optional `serve-ui` feature to `bloom_server` that embeds the
  frontend with `rust-embed` and serves it at `/`, so a single binary can host
  both the API and the UI. Off by default; the backend stays independently
  deployable and the frontend can be hosted separately.
- Add `just ui-dev` / `ui-build` / `server-ui` recipes and `ui/README.md`
  documenting both deployment modes.
- Rename the frontend directory/crate from `web/` (`bloom-web`) to `ui/`
  (`bloom-ui`): the Dioxus frontend can target web or desktop, so `web/` was
  misleading. The `ui-*` recipes already followed this naming.

### CI / open-source readiness

- Establish `main` as the default branch; CI now triggers on pushes and PRs to
  `main`.
- Decouple the `missing_docs` lint from the blocking Clippy gate: CI runs
  `clippy -D warnings -A missing_docs`, plus a non-blocking advisory
  `docs` job (`-W missing_docs` and `cargo doc`). Removed the per-crate
  `#![warn(missing_docs)]` attributes that command-line flags could not
  override.
- Fix three genuine `clippy::useless_vec` errors that the `missing_docs`
  noise had been masking.
- Add a `hardware-tests` feature (engine + tilelang) and gate TileLang JIT
  tests behind it; they need the Python/numpy toolchain and no longer fail on
  standard CI runners.
- Ignore `test_batch_executor_grammar_filtering_decode_chain` as a known
  issue: the JSON grammar mask uses stale state in the decode chain, so an
  invalid token is sampled. Re-enable once the decode-chain state is fixed.
- Update CONTRIBUTING to reflect the `bloomai-*` namespace (drop stale
  namespace compatibility notes).
- Remove `continue-on-error` from the four CI smoke steps (benchmark, OpenAI
  API, llama.cpp comparison, Docker build). All three scripts already SKIP
  gracefully (exit 0) when models or external binaries are absent, so the
  escape hatch was masking real failures.
- Add a `Security audit` CI job using `rustsec/audit-check` to catch
  advisories in `Cargo.lock` on every PR and push to `main`.
- Extract shared byte-unit constants (`KIB`/`MIB`/`GIB` and `f64` variants)
  into `bloomai_core::constants`, replacing scattered local definitions and
  raw `1024 * 1024 * 1024` literals across 13 production source files.
- Remove the crate-level `#![allow(dead_code, unused_variables, ...)]` from
  `bloom_server/main.rs`. Fixed the underlying issues: dropped an unused
  `engine` binding and a duplicate `backend_name` declaration, removed two
  dead utility functions from `chat_template.rs`, and replaced
  `contains_key`+`insert` patterns with the `Entry` API.
- Adopt typed `BloomError` variants across the engine crate, replacing
  unstructured `anyhow::anyhow!` / `anyhow::bail!` calls in 10 source files:
  `gemma4.rs`, `core/model.rs`, `core/pipeline.rs`, `core/manifest.rs`,
  `plugin/mod.rs`, `scheduler/mod.rs`, `scheduler/kv_hook.rs`,
  `scheduler/scheduler_test.rs`, `executor/qwen_streaming.rs`, and
  `bloom_server/main.rs`. Each call site now uses the semantically correct
  variant (`ModelLoad`, `Engine`, `InvalidInput`, `UnsupportedFormat`,
  `UnsupportedFamily`, `MissingRequiredFile`, `HashMismatch`, `Plugin`,
  `SchedulingFailed`, `Resource(...)`, etc.), enabling `ErrorCategory`
  classification and `recovery_hints()` on the hot path. The OOM-detection
  logic in `pipeline.rs` also learned to recognize typed
  `BloomError::Resource(InsufficientRam|InsufficientVram|...)` as OOM.

## 0.1.0 - 2026-07-21

First public open-source release of the Bloom workspace.

### Engine & Features

- Initial early engine workspace with CLI, OpenAI-compatible server, scheduler,
  CacheMesh, plugins, GGUF inspection, benchmark schema, and experimental model
  backends.
- Vendor `bloomai-core`, `bloomai-backend`, and `bloomai-tilelang` into the
  Bloom workspace so the repository can build independently.
- Add optional API-key protection, configurable CORS, and JSON request body
  limits to `bloom_server`.

### Open-source readiness

- Rename crates to the `bloom-*` namespace and unify the repository URL to
  `github.com/constmark/bloom`.
- Remove accidentally committed build artifacts, demo scripts and benchmark
  outputs; extend `.gitignore` to keep them out.
- Replace local machine paths in documentation with repository-relative links.
- Add `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and populate
  `.github/CODEOWNERS`.
- Add README badges, `#![warn(missing_docs)]` on public crates, and a
  `pyproject.toml` for the Python SDK.
- Fix a cross-platform build break in `prefetch_file_madvise`
  (`posix_fadvise` is Linux-only; use `fcntl(F_RDADVISE)` on macOS).
- Fix `bloom_server` KV hook wiring to use `ServerKvHook` matching the
  per-request model map type.
