# Development

## Common Commands

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Use `just --list --unsorted` for the maintained command shortcuts.

## Feature Policy

The default feature set must build on a standard Linux/macOS development
machine. Hardware-specific features such as CUDA are checked separately because
they require vendor toolchains.

## Local Artifacts

Do not commit model weights, generated IR files, virtualenvs, compiled kernels,
or installer binaries. Put reproducible setup steps in `docs/` or `scripts/`
instead.
