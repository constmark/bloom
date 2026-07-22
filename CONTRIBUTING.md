# Contributing to Bloom

Thanks for helping make Bloom better. This repository is still early, so small,
well-scoped changes are easiest to review.

## Development Setup

Install Rust using rustup, then run:

```bash
cargo check --workspace
cargo test --workspace
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

CUDA checks are intentionally separate because they require `nvcc`:

```bash
cargo clippy --workspace --all-targets --features cuda -- -D warnings
```

## Pull Request Expectations

- Keep API and crate boundary changes explicit in the PR description.
- Add or update tests for behavior changes.
- Keep generated models, downloaded weights, local virtualenvs, and build
  artifacts out of git.
- Prefer small PRs that touch one area: backend, model, CLI/server, or docs.

## Project Boundaries

The repository is a standalone workspace. All crates are published under the
`bloomai-*` namespace and build independently of any external project.

- `crates/core` (`bloomai-core`): shared runtime types, manifests, resource
  tickets, scheduler configuration, world-state abstractions, and benchmark
  result types.
- `crates/backend` (`bloomai-backend`): device backend traits, CPU/default
  backend, Intel NPU probing, and backend registry helpers.
- `crates/tilelang` (`bloomai-tilelang`): TileLang kernel compilation and
  dynamic loading helpers.
- `crates/engine` (`bloomai-engine`): model loading, concrete inference engines,
  CLI binaries, OpenAI-compatible server, plugins, processors, CacheMesh, and
  engine-internal scheduling.

Do not reintroduce dependencies on files outside this repository without an
explicit design note and CI coverage.

## Plugin Contributions

Bloom supports five plugin types. Each plugin is described by a JSON manifest and
shipped as a separate crate or shared library. Templates live under `examples/plugins/`.

### Plugin Types

| Plugin | Purpose | Template | Required Manifest |
| --- | --- | --- | --- |
| Backend | New chip, vendor SDK, remote device | `examples/plugins/backend-plugin.json` | `BackendPluginManifest` |
| Engine | New model family or execution framework | `examples/plugins/engine-plugin.json` | `EnginePluginManifest` |
| Processor | Tokenizer, image/audio/video preprocessor | `examples/plugins/processor-plugin.json` | `ProcessorPluginManifest` |
| Operator | Custom kernel (attention, quantized matmul) | `examples/plugins/operator-plugin.json` | `OperatorPluginManifest` |
| Model Package | Weights + manifest + processor config | `examples/plugins/model-package.json` | `ModelPackageManifest` |

### Testing Requirements

Every plugin PR **must** include:

1. **Availability / probe test** — verifies the plugin can detect its hardware
   or dependency without crashing. Must run on CI without special hardware
   (use mock or stub where needed).
2. **Minimal functional test** — for engines: load a tiny mock model and run
   one inference step; for processors: round-trip a sample input through the
   processor; for operators: verify output shape and dtype.
3. **Error path test** — confirm the plugin returns structured `BloomError`
   variants (not raw strings or panics) when dependencies or hardware are
   missing.

Hardware-specific tests should be gated behind a CI label or feature flag:

```rust
#[cfg(feature = "hardware-tests")]
#[test]
fn test_real_device_inference() { /* ... */ }
```

### License Compatibility

- Plugin code must be licensed under **Apache-2.0** or **MIT** (dual-licensed
  preferred, matching Bloom's own license).
- If the plugin wraps a vendor SDK, document the SDK license and any
  redistribution restrictions in the plugin README.
- Model package plugins must include the model's original license identifier
  in the `license` field.

### Platform Coverage

- Declare supported platforms in the `platforms` array of the manifest
  (e.g. `["linux-x86_64", "macos-aarch64"]`).
- At least one platform must pass CI. If the plugin requires proprietary
  hardware, provide a mock/probe test that runs on standard CI runners.

### CI Labels

Use the following PR labels to help triage plugin CI runs:

| Label | When to use |
| --- | --- |
| `plugin/backend` | PR adds or modifies a backend plugin |
| `plugin/engine` | PR adds or modifies an engine plugin |
| `plugin/processor` | PR adds or modifies a processor plugin |
| `plugin/operator` | PR adds or modifies an operator plugin |
| `plugin/model-package` | PR adds or modifies a model package |
| `hardware-required` | Tests require specific hardware not available on default CI |

### Version Strategy

- Public traits (`Backend`, `Engine`, `LoadedModel`, `Processor`) and manifest
  schemas follow semantic versioning. Breaking changes require a migration
  note in the PR description and a `CHANGELOG` entry.
- Plugin manifests should declare `min_runtime_version` to pin the minimum
  compatible Bloom runtime.

## Test Classification

Bloom tests are divided into three categories to keep CI fast and hardware-free
by default.

### Mock Tests (default — always run)

- Pure data-structure and logic tests that require **no** hardware, network,
  or external model files.
- Use synthetic / stub implementations (e.g. `EchoTextModel`, `MockWorldModel`,
  mock backend capabilities).
- Must pass on every platform in CI (Linux, macOS, Windows).
- Location: inline `#[cfg(test)] mod tests` in each module.

### Probe Tests (default — always run)

- Verify that hardware detection and availability reporting works correctly
  on the CI runner, including the “not available” path.
- Must **not** crash or panic when hardware is absent.
- Examples: backend `availability()` returns structured `BackendAvailability`
  with reason strings; Intel NPU probe on macOS returns `available: false`
  with a clear reason.
- Location: inline tests in `crates/backend/src/*.rs` and engine/backend
  integration tests.

### Hardware Tests (gated — opt-in)

- Require specific hardware (NVIDIA GPU, Intel NPU, Apple Metal, etc.).
- Gated behind `#[cfg(feature = "hardware-tests")]` so they are skipped
  on standard CI.
- To run locally: `cargo test --workspace --features hardware-tests`.
- These tests validate real model loading, real inference output, and
  performance characteristics.
- Location: `tests/` directory or inline with feature gate.

### Running Tests

```bash
# Default (mock + probe, no hardware needed)
cargo test --workspace

# With hardware tests
cargo test --workspace --features hardware-tests

# With CUDA hardware tests
cargo test --workspace --features cuda,hardware-tests
```

## Support Matrix

The source of truth is `docs/support-matrix.md`. Do not mark a row as
`stable` unless it has a reproducible command, real-model evidence, and
benchmark data.

**Legend**: `stable` = production-claim path with evidence; `experimental` =
code path exists but needs more coverage; `external-runtime` = requires an
external process, SDK, or runtime; `skeleton` = API shape only.
