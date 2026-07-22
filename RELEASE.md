# Release Checklist

Use this checklist before tagging a release or marking a model/backend path as
`stable`.

## Required Checks

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/validate_json_artifacts.py
cargo package --workspace --allow-dirty
```

## Real-Model Gates

At least one public, reproducible model path must pass:

```bash
BLOOM_REQUIRE_MODEL=1 BLOOM_STRICT_MEMORY_BUDGET=1 BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/openai_compat_smoke.py --build --require-openai-sdk
BLOOM_REQUIRE_MODEL=1 BLOOM_STRICT_MEMORY_BUDGET=1 BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/ci_smoke_test.sh
BLOOM_REQUIRE_MODEL=1 BLOOM_REQUIRE_LLAMA_CPP=1 BLOOM_STRICT_MEMORY_BUDGET=1 BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/compare_llamacpp.py --build
```

Record the model source, license, SHA-256 hash, backend, dtype, quantization,
TTFT, TBT, tokens/s, peak memory, OS, and hardware in
`docs/support-matrix.md`.

## Artifacts

- Linux `bloom_infer`, `bloom_server`, `bloom_bench`, `inspect_gguf`.
- macOS `bloom_infer`, `bloom_server`, `bloom_bench`, `inspect_gguf`.
- Docker image for CPU/local serving.
- SHA-256 checksums for all published binaries.
- Release notes listing support matrix changes and known limitations.

## Publishing to crates.io

The workspace publishes five crates under the `bloomai-*` namespace. Publish
them in dependency order so each crate's path dependencies already exist on
the registry:

```bash
cargo login                          # one-time, needs a crates.io API token
cargo publish -p bloomai-core
cargo publish -p bloomai-backend
cargo publish -p bloomai-tilelang
cargo publish -p bloomai-engine
cargo publish -p bloomai-ffi
```

> **Namespace:** crates are published as `bloomai-*` because the bare
> `bloom-core` name was already taken on crates.io by an unrelated project.
> All five `bloomai-*` names were verified free on 2026-07-21. The FFI dynamic
> library keeps the file name `bloom_ffi` (`libbloom_ffi.{so,dylib,dll}`) so
> the Python SDK and downstream loaders are unaffected by the crate rename.

Python SDK: `python/` has its own `pyproject.toml`; build the native library
first (`cargo build --release -p bloomai-ffi`), then `pip install ./python`.

## Git tag & GitHub Release

```bash
git tag -a v0.1.0 -m "Bloom v0.1.0"
git push origin main --tags        # triggers .github/workflows/release.yml
```

Build tarballs are uploaded by the release workflow; do **not** commit
`release-artifacts/` (git-ignored).

## Do Not Release If

- The repository cannot build without files outside this repository.
- A default CI smoke silently skips every real-model validation and the release
  notes still claim production support.
- Any declared `bloom.json` file hash fails verification. Do not set
  `BLOOM_ALLOW_HASH_MISMATCH` in release or production validation.
- The model cannot load with `BLOOM_STRICT_MEMORY_BUDGET=1` on the documented
  target hardware.
- New public traits or schema fields lack migration notes.
- Any `stable` row in the support matrix lacks a reproducible command.
