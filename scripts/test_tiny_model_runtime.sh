#!/usr/bin/env bash
# Exercise Bloom's native CPU inference path with a generated Qwen2 fixture.
#
# The fixture uses deterministic, untrained weights. This gate proves that the
# tokenizer, Safetensors loader, Candle forward pass, decoding, embeddings,
# reranking, structured outputs, function calls, HTTP adapters, and streaming
# lifecycles execute together; it does not measure model quality.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SERVER_BIN="${BLOOM_TINY_RUNTIME_SERVER_BIN:-${WORKSPACE_ROOT}/target/debug/bloom_server}"
if [[ -n "${BLOOM_TINY_RUNTIME_SERVER_BIN:-}" ]]; then
    BUILD_SERVER=0
else
    BUILD_SERVER=1
fi
REQUIRE_OFFICIAL_CLIENTS="${BLOOM_TINY_RUNTIME_REQUIRE_OFFICIAL_CLIENTS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --require-official-clients)
            REQUIRE_OFFICIAL_CLIENTS=1
            shift
            ;;
        --server-bin)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "--server-bin requires a path" >&2
                exit 2
            fi
            SERVER_BIN="$2"
            BUILD_SERVER=0
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "$REQUIRE_OFFICIAL_CLIENTS" in
    0|false|FALSE|no|NO)
        OPENAI_SDK_ARGS=()
        OLLAMA_SDK_ARGS=()
        ;;
    1|true|TRUE|yes|YES)
        OPENAI_SDK_ARGS=(--require-openai-sdk)
        OLLAMA_SDK_ARGS=(--require-ollama-sdk)
        ;;
    *)
        echo "BLOOM_TINY_RUNTIME_REQUIRE_OFFICIAL_CLIENTS must be a boolean value" >&2
        exit 2
        ;;
esac

TINY_RUNTIME_DIR=$(mktemp -d)
trap 'rm -rf "$TINY_RUNTIME_DIR"' EXIT
MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-fixture"
REPEATED_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-fixture-repeat"
SHARDED_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-sharded-fixture"
REPEATED_SHARDED_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-sharded-fixture-repeat"
EMBEDDING_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-embed-fixture"
REPEATED_EMBEDDING_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-embed-fixture-repeat"
STRUCTURED_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-structured-fixture"
REPEATED_STRUCTURED_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-structured-fixture-repeat"
TOOL_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-tool-fixture"
REPEATED_TOOL_MODEL_DIR="${TINY_RUNTIME_DIR}/tiny-qwen2-tool-fixture-repeat"

cd "$WORKSPACE_ROOT"

echo "Generating deterministic Qwen2 CPU fixture..."
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$MODEL_DIR"
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$REPEATED_MODEL_DIR"
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$SHARDED_MODEL_DIR" \
    --sharded
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$REPEATED_SHARDED_MODEL_DIR" \
    --sharded
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$EMBEDDING_MODEL_DIR" \
    --profile embedding
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$REPEATED_EMBEDDING_MODEL_DIR" \
    --profile embedding
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$STRUCTURED_MODEL_DIR" \
    --profile structured
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$REPEATED_STRUCTURED_MODEL_DIR" \
    --profile structured
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$TOOL_MODEL_DIR" \
    --profile tool
cargo run --quiet --locked -p bloomai-engine \
    --example generate_tiny_qwen2_fixture -- \
    --output "$REPEATED_TOOL_MODEL_DIR" \
    --profile tool
for fixture_file in config.json tokenizer.json model.safetensors; do
    cmp "$MODEL_DIR/$fixture_file" "$REPEATED_MODEL_DIR/$fixture_file"
    cmp "$EMBEDDING_MODEL_DIR/$fixture_file" \
        "$REPEATED_EMBEDDING_MODEL_DIR/$fixture_file"
    cmp "$STRUCTURED_MODEL_DIR/$fixture_file" \
        "$REPEATED_STRUCTURED_MODEL_DIR/$fixture_file"
    cmp "$TOOL_MODEL_DIR/$fixture_file" "$REPEATED_TOOL_MODEL_DIR/$fixture_file"
done
for fixture_file in \
    config.json \
    tokenizer.json \
    model.safetensors.index.json \
    model-00001-of-00002.safetensors \
    model-00002-of-00002.safetensors; do
    cmp "$SHARDED_MODEL_DIR/$fixture_file" \
        "$REPEATED_SHARDED_MODEL_DIR/$fixture_file"
done

if [[ "$BUILD_SERVER" -eq 1 ]]; then
    echo "Building bloom_server..."
    cargo build --quiet --locked --bin bloom_server
fi
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "bloom_server is not executable: $SERVER_BIN" >&2
    exit 1
fi

echo "Running OpenAI-compatible native-model smoke..."
python3 scripts/openai_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$MODEL_DIR" \
    --require-model \
    --max-tokens 3 \
    --api-key bloom-tiny-runtime \
    "${OPENAI_SDK_ARGS[@]}"

echo "Running Ollama-compatible native-model smoke..."
python3 scripts/ollama_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$MODEL_DIR" \
    --require-model \
    --catalog-only \
    --max-tokens 3 \
    --api-key bloom-tiny-runtime \
    "${OLLAMA_SDK_ARGS[@]}"

echo "Running OpenAI-compatible sharded native-model smoke..."
python3 scripts/openai_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$SHARDED_MODEL_DIR" \
    --require-model \
    --max-tokens 3 \
    --api-key bloom-tiny-runtime \
    "${OPENAI_SDK_ARGS[@]}"

echo "Running Ollama-compatible sharded native-model smoke..."
python3 scripts/ollama_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$SHARDED_MODEL_DIR" \
    --require-model \
    --catalog-only \
    --max-tokens 3 \
    --api-key bloom-tiny-runtime \
    "${OLLAMA_SDK_ARGS[@]}"

echo "Running OpenAI-compatible native embedding and rerank smoke..."
python3 scripts/openai_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$EMBEDDING_MODEL_DIR" \
    --require-model \
    --embedding-only \
    --api-key bloom-tiny-runtime \
    "${OPENAI_SDK_ARGS[@]}"

echo "Running Ollama-compatible native embedding smoke..."
python3 scripts/ollama_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$EMBEDDING_MODEL_DIR" \
    --require-model \
    --catalog-only \
    --api-key bloom-tiny-runtime \
    "${OLLAMA_SDK_ARGS[@]}"

echo "Running OpenAI-compatible native structured-output smoke..."
python3 scripts/openai_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$STRUCTURED_MODEL_DIR" \
    --require-model \
    --structured-only \
    --max-tokens 8 \
    --api-key bloom-tiny-runtime \
    "${OPENAI_SDK_ARGS[@]}"

echo "Running Ollama-compatible native structured-output smoke..."
python3 scripts/ollama_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$STRUCTURED_MODEL_DIR" \
    --require-model \
    --catalog-only \
    --structured-only \
    --max-tokens 8 \
    --api-key bloom-tiny-runtime \
    "${OLLAMA_SDK_ARGS[@]}"

echo "Running OpenAI-compatible native function-call smoke..."
python3 scripts/openai_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$TOOL_MODEL_DIR" \
    --require-model \
    --tool-only \
    --max-tokens 64 \
    --api-key bloom-tiny-runtime \
    "${OPENAI_SDK_ARGS[@]}"

echo "Running Ollama-compatible native function-call smoke..."
python3 scripts/ollama_compat_smoke.py \
    --server-bin "$SERVER_BIN" \
    --model "$TOOL_MODEL_DIR" \
    --require-model \
    --catalog-only \
    --tool-only \
    --max-tokens 64 \
    --api-key bloom-tiny-runtime \
    "${OLLAMA_SDK_ARGS[@]}"

echo "OK: deterministic Qwen2 fixtures passed native CPU single-file/sharded text, embedding, rerank, structured-output, and function-call runtime smokes"
