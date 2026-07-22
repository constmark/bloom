# Bloom - Edge Multimodal Inference Engine
#
# Install `just`: https://github.com/casey/just
#
# Usage:
#   just <recipe>            # Run a recipe
#   just --list              # List all recipes
#   just --list --unsorted   # List recipes in definition order

# ===== Configuration =====

profile := "debug"
target_dir := if profile == "release" { "target/release" } else { "target/debug" }

# ===== Help =====

default:
    @just --list --unsorted

# ===== Build =====

# Build the workspace.
build:
    cargo build --workspace

# Build the workspace in release mode.
build-release:
    cargo build --workspace --release

# ===== Inference =====

# Run streaming inference with a local model (example: just infer /path/to/model "your prompt").
infer model_path prompt="Hello!":
    cargo run --bin bloom_infer -- --model {{model_path}} --prompt "{{prompt}}" --stream --max-tokens 128

# Run non-streaming inference with a local model.
infer-sync model_path prompt="Hello!":
    cargo run --bin bloom_infer -- --model {{model_path}} --prompt "{{prompt}}" --max-tokens 128

# ===== Checks =====

# Run a fast type check.
check:
    cargo check --workspace

# ===== Tests =====

# Run all tests.
test:
    cargo test --workspace

# ===== Code Quality =====

# Format the codebase.
fmt:
    cargo fmt --all

# Check formatting.
fmt-check:
    cargo fmt --all -- --check

# Run Clippy.
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run Clippy with the CUDA feature.
clippy-cuda:
    cargo clippy --workspace --all-targets --features cuda -- -D warnings

# Run the complete lint suite.
lint: fmt-check clippy

# ===== Documentation =====

# Generate and open API documentation.
doc:
    cargo doc --workspace --no-deps --open

# ===== Maintenance =====

# Remove build artifacts.
clean:
    cargo clean

# ===== Frontend (Dioxus UI) =====

# Run the UI development server (start bloom_server first; default: 127.0.0.1:3000).
ui-dev:
    cd ui && dx serve --platform web

# Build release UI assets into ui/dist for serve-ui embedding.
ui-build:
    cd ui && dx build --platform web --release && rm -rf dist && mkdir -p dist && cp -r target/dx/bloom-ui/release/web/public/* dist/

# Build a single binary containing both the API and UI.
server-ui: ui-build
    cargo build --release --bin bloom_server --features serve-ui

# Update dependencies.
update:
    cargo update
