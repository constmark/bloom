# Bloom

[![CI](https://github.com/constmark/bloom/actions/workflows/ci.yml/badge.svg)](https://github.com/constmark/bloom/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/bloomai-engine.svg)](https://crates.io/crates/bloomai-engine)
[![docs.rs](https://docs.rs/bloomai-engine/badge.svg)](https://docs.rs/bloomai-engine)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](rust-toolchain.toml)

Bloom is a standalone, multimodal inference engine written in Rust. It loads
local models directly, runs inference without an external scheduler, and
provides both a command-line interface and an OpenAI-compatible HTTP API.

Bloom can run on its own or as the inference engine used by
[Elderwand](https://github.com/constmark/elderwand).

> [!IMPORTANT]
> Bloom is pre-1.0 software. The default Candle paths are currently
> **experimental**. Other backends may require external runtimes or may only
> provide capability-detection skeletons. See the
> [support matrix](docs/support-matrix.md) before choosing a production path.

## Highlights

- Standalone local inference through `bloom_infer`
- OpenAI-compatible chat, completion, embedding, and reranking endpoints
- Streaming generation and an interactive terminal mode
- CPU, Metal, and CUDA execution through Candle
- GGUF and Hugging Face-style model package support
- Pluggable engines, backends, processors, operators, and model packages
- Memory estimation, KV-cache management, in-flight batching, and CacheMesh
- Optional browser UI built with Dioxus
- C ABI and a small Python SDK for downstream integrations

## Quickstart

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) with the toolchain pinned in
  `rust-toolchain.toml`
- Git
- A compatible local model; start with the
  [support matrix](docs/support-matrix.md)

Clone and build Bloom:

```bash
git clone https://github.com/constmark/bloom.git
cd bloom
cargo build --release --bin bloom_infer --bin bloom_server
```

Run a local model with the default CPU backend:

```bash
cargo run --release --bin bloom_infer -- \
  --model /path/to/model.gguf \
  --prompt "Explain edge inference in one sentence." \
  --max-tokens 128 \
  --stream
```

Model paths may point to a single `.gguf` file or to a model directory. A
Hugging Face-style directory normally contains `config.json`, `tokenizer.json`,
and Safetensors weights. A Bloom model package contains a `bloom.json`
manifest. Use `--inspect` to validate routing and estimate memory without
loading the full model:

```bash
cargo run --release --bin bloom_infer -- \
  --model /path/to/model \
  --inspect
```

Useful CLI modes:

```bash
# Interactive terminal chat
cargo run --release --bin bloom_infer -- --model /path/to/model --interactive

# List engines compiled into the binary
cargo run --release --bin bloom_infer -- --list-engines

# Use Apple Metal or NVIDIA CUDA
cargo run --release --features metal --bin bloom_infer -- \
  --model /path/to/model --device gpu --prompt "Hello" --stream
cargo run --release --features cuda --bin bloom_infer -- \
  --model /path/to/model --device gpu --prompt "Hello" --stream
```

Run `cargo run --release --bin bloom_infer -- --help` for the complete CLI
reference.

## OpenAI-Compatible Server

Start the API server:

```bash
cargo run --release --bin bloom_server -- \
  --model /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 3000
```

Send a chat request:

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello from Bloom"}],
    "max_tokens": 64,
    "stream": false
  }'
```

The server exposes:

- `/health`, `/ready`, and `/metrics`
- `/v1/models`
- `/v1/chat/completions` and `/v1/completions`
- `/v1/embeddings` and `/v1/rerank`
- `/v1/backends` and `/v1/kv-cache-stats`
- `/v1/cancel/{request_id}`

For anything beyond localhost, set `BLOOM_API_KEY`, restrict
`BLOOM_CORS_ALLOW_ORIGIN`, and protect health and metrics endpoints with a
reverse proxy or network ACL. See the [production guide](docs/production.md)
and [security policy](SECURITY.md).

## Configuration

Bloom reads `~/.bloom/config.json` by default. Override the path with
`BLOOM_CONFIG` or `--config`, and generate an example with:

```bash
cargo run --bin bloom_infer -- --init-config
```

Explicit command-line arguments take precedence over the configuration file.
Common environment variables include:

| Variable | Purpose |
| --- | --- |
| `BLOOM_CONFIG` | Runtime configuration path |
| `BLOOM_API_KEY` | API key required by `/v1/*` routes |
| `BLOOM_CORS_ALLOW_ORIGIN` | Allowed browser origin |
| `BLOOM_MEMORY_UTILIZATION` | Fraction of available memory usable at startup |
| `BLOOM_STRICT_MEMORY_BUDGET` | Fail before loading when the estimate exceeds the budget |

## Model and Backend Support

Bloom separates executable, external-runtime, experimental, and skeleton
paths. The current source of truth is the
[model and backend support matrix](docs/support-matrix.md).

| Path | Status | Notes |
| --- | --- | --- |
| Candle on CPU | Experimental | Default development and CI path |
| Candle on Metal | Experimental | Build with `--features metal` |
| Candle on CUDA | Experimental | Build with `--features cuda` |
| OpenVINO and ASR bridges | External runtime | Require additional runtimes or Python packages |
| ONNX Runtime | Skeleton | Inspection and capability diagnostics only |

Do not infer production readiness from the presence of an engine adapter. A
path is promoted only after reproducible real-model validation and benchmark
evidence are recorded.

## Web UI

The optional Dioxus UI can run separately or be embedded in `bloom_server`.
Install a Dioxus CLI version compatible with Dioxus 0.7, then run:

```bash
cargo install dioxus-cli

# Terminal 1: API server
cargo run --bin bloom_server -- --model /path/to/model

# Terminal 2: UI development server
just ui-dev
```

See [ui/README.md](ui/README.md) for standalone and single-binary deployment.

## Documentation

See the [documentation index](docs/README.md). Start with the
[architecture](docs/architecture.md), [support matrix](docs/support-matrix.md),
or [production checklist](docs/production.md).

Public JSON schemas and examples live under `examples/`. Validate them with:

```bash
./scripts/validate_json_artifacts.py
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `crates/core` | Shared types, manifests, scheduling, memory, world-state, and plugin contracts |
| `crates/backend` | Device probing, backend traits, and backend registry helpers |
| `crates/engine` | Model loading, inference engines, pipelines, server, CLI, and CacheMesh |
| `crates/tilelang` | TileLang kernel compilation and loading |
| `crates/ffi` | Stable C ABI for native consumers |
| `python` | Python SDK bindings |
| `ui` | Optional Dioxus web interface |
| `docs` | Architecture, operations, support, and roadmap documentation |
| `examples` | Schemas, manifests, plugins, and integration examples |
| `scripts` | Validation, smoke-test, benchmark, and external-runtime helpers |

## Development

Install [just](https://github.com/casey/just) for maintained command shortcuts,
or run the underlying commands directly:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/validate_json_artifacts.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements and
[RELEASE.md](RELEASE.md) for release gates.

## Community and Security

- Use GitHub issues for reproducible bugs and focused feature proposals.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License

Bloom is licensed under the [Apache License 2.0](LICENSE).
