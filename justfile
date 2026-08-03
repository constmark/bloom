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
    cargo build --workspace --locked

# Build the workspace in release mode.
build-release:
    cargo build --workspace --release --locked

# ===== Inference =====

# Run streaming inference with a local model (example: just infer /path/to/model "your prompt").
infer model_path prompt="Hello!":
    cargo run --locked --bin bloom_infer -- --model {{model_path}} --prompt "{{prompt}}" --stream --max-tokens 128

# Run non-streaming inference with a local model.
infer-sync model_path prompt="Hello!":
    cargo run --locked --bin bloom_infer -- --model {{model_path}} --prompt "{{prompt}}" --max-tokens 128

# ===== Checks =====

# Run a fast type check for the Rust workspace and UI.
check: ui-check
    cargo check --workspace --locked

# Inspect the effective server deployment without loading a model or binding a port.
doctor:
    cargo run --locked --bin bloom_server -- --doctor

# Emit the versioned server deployment report as JSON.
doctor-json:
    cargo run --locked --bin bloom_server -- --doctor=json

# ===== Tests =====

# Run all workspace, UI, process-boundary, and native CPU runtime tests.
test: ui-test server-shutdown-test server-http-boundary-test tiny-model-runtime-test
    cargo test --workspace --locked

# Exercise clean drain, deadline expiry, and repeated-signal escalation.
server-shutdown-test:
    python3 scripts/test_server_shutdown.py

# Exercise browser-origin, authentication, correlation, and routing boundaries.
server-http-boundary-test:
    python3 scripts/test_server_http_boundary.py

# Generate a deterministic tiny Qwen2 model and cross both public API adapters.
tiny-model-runtime-test:
    ./scripts/test_tiny_model_runtime.sh

# Download and verify all pinned trained packages, then gate native CPU
# instruction following, semantic retrieval, and both public API adapters.
trained-model-runtime-test: trained-qwen2-runtime-test trained-qwen3-runtime-test trained-llama-runtime-test trained-safetensors-runtime-test trained-embedding-runtime-test

trained-qwen2-runtime-test:
    ./scripts/test_trained_qwen2_runtime.sh

trained-qwen3-runtime-test:
    ./scripts/test_trained_qwen3_runtime.sh

trained-llama-runtime-test:
    ./scripts/test_trained_llama_runtime.sh

trained-safetensors-runtime-test:
    ./scripts/test_trained_safetensors_runtime.sh

trained-embedding-runtime-test:
    ./scripts/test_trained_embedding_runtime.sh

# Require the pinned OpenAI and Ollama clients during the tiny-model runtime gate.
tiny-model-runtime-test-official:
    ./scripts/test_tiny_model_runtime.sh --require-official-clients

# Exercise Ollama discovery and admission; generation runs when BLOOM_MODEL_PATH exists.
ollama-smoke:
    ./scripts/ollama_compat_smoke.py

# ===== Code Quality =====

# Format the codebase.
fmt:
    cargo fmt --all

# Check formatting.
fmt-check:
    cargo fmt --all -- --check

# Run Clippy.
clippy:
    cargo clippy --workspace --all-targets --locked -- -D warnings -A missing_docs

# Run Clippy with the CUDA feature.
clippy-cuda:
    cargo clippy --workspace --all-targets --features cuda --locked -- -D warnings -A missing_docs

# Run the complete workspace and UI lint suite.
lint: fmt-check clippy ui-fmt-check ui-clippy

# Package and compile the exact publishable crate archive set.
crate-package-test:
    ./scripts/test_crate_packages.sh

# ===== Documentation =====

# Generate and open API documentation.
doc:
    cargo doc --workspace --no-deps --locked --open

# ===== Maintenance =====

# Remove build artifacts.
clean:
    cargo clean

# ===== Frontend (Dioxus UI) =====

# Run the standalone UI on the documented exact development origin.
ui-dev:
    cd ui && dx serve --platform web --addr 127.0.0.1 --port 8080

# Check the UI for its actual WebAssembly target.
ui-check:
    cargo check --manifest-path ui/Cargo.toml --target wasm32-unknown-unknown --all-targets --locked

# Run host-side UI state and protocol tests.
ui-test:
    cargo test --manifest-path ui/Cargo.toml --locked

# Check UI formatting.
ui-fmt-check:
    cargo fmt --manifest-path ui/Cargo.toml -- --check

# Run strict UI lints.
ui-clippy:
    cargo clippy --manifest-path ui/Cargo.toml --all-targets --locked -- -D warnings

# Build release UI assets into ui/dist for serve-ui embedding.
ui-build:
    ./scripts/build_ui.sh

# Build a single binary containing both the API and UI.
server-ui: ui-build
    cargo build --locked --release --bin bloom_server --features serve-ui

# Build and run the single-binary local app against a managed model directory.
app models_dir: server-ui
    ./target/release/bloom_server --models-dir "{{models_dir}}" --open-browser

# Build the native SwiftUI macOS client as target/macos/Bloom Desktop.app.
desktop-build:
    ./scripts/build_macos_client.sh

# Run native macOS client unit tests.
desktop-test:
    ./scripts/test_macos_client.sh

# Build and open the native macOS client. Start bloom_server separately first.
desktop-run: desktop-build
    open "target/macos/Bloom Desktop.app"

# Build a self-checked release archive with the embedded UI.
package-release:
    ./scripts/package_release.sh

# Update dependencies.
update:
    cargo update
