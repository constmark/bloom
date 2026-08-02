#!/usr/bin/env bash
# Run Bloom's immutable official MiniLM native-encoder CPU acceptance profile.

set -euo pipefail

WORKSPACE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_REPOSITORY="sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_PATH="${BLOOM_TRAINED_EMBEDDING_CACHE:-/tmp/bloom-trained-embedding-runtime}"
SERVER_BIN="${BLOOM_TRAINED_EMBEDDING_SERVER_BIN:-${WORKSPACE_DIR}/target/release/bloom_server}"
INFER_BIN="${BLOOM_TRAINED_EMBEDDING_INFER_BIN:-${WORKSPACE_DIR}/target/release/bloom_infer}"
BENCHMARK_OUTPUT="${BLOOM_TRAINED_EMBEDDING_BENCHMARK_OUTPUT:-}"
BUILD_BINARIES=1
REQUIRE_OFFICIAL_CLIENTS=0

usage() {
    echo "Usage: $0 [--model-path PATH] [--server-bin PATH] [--infer-bin PATH] [--benchmark-output PATH] [--require-official-clients] [--skip-build]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-path)
            MODEL_PATH="${2:?--model-path requires a value}"
            shift 2
            ;;
        --server-bin)
            SERVER_BIN="${2:?--server-bin requires a value}"
            shift 2
            ;;
        --infer-bin)
            INFER_BIN="${2:?--infer-bin requires a value}"
            shift 2
            ;;
        --benchmark-output)
            BENCHMARK_OUTPUT="${2:?--benchmark-output requires a value}"
            shift 2
            ;;
        --require-official-clients)
            REQUIRE_OFFICIAL_CLIENTS=1
            shift
            ;;
        --skip-build)
            BUILD_BINARIES=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for command_name in cargo curl grep python3 sha256sum; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "FAIL: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

if [ -L "$MODEL_PATH" ] || { [ -e "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; }; then
    echo "FAIL: trained embedding cache is not a real directory: ${MODEL_PATH}" >&2
    exit 1
fi
mkdir -p "$MODEL_PATH"

verify_model_file() {
    local relative_path="$1"
    local expected_size="$2"
    local expected_sha256="$3"
    case "$relative_path" in
        ""|/*|*\\*|../*|*/../*|*/..|..)
            echo "FAIL: unsafe trained embedding path: ${relative_path}" >&2
            exit 1
            ;;
    esac
    local path="${MODEL_PATH}/${relative_path}"
    local partial_path="${path}.partial"
    local url="https://huggingface.co/${MODEL_REPOSITORY}/resolve/${MODEL_REVISION}/${relative_path}"
    mkdir -p "$(dirname "$path")"
    if [ ! -f "$path" ]; then
        if [ -e "$path" ]; then
            echo "FAIL: trained embedding path exists but is not a regular file: ${path}" >&2
            exit 1
        fi
        if [ -L "$partial_path" ]; then
            echo "FAIL: refusing a symlinked partial file: ${partial_path}" >&2
            exit 1
        fi
        echo "Downloading pinned ${relative_path} (${expected_size} bytes)..." >&2
        curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
            --retry 4 --retry-all-errors --continue-at - \
            --output "$partial_path" "$url"
        printf '%s  %s\n' "$expected_sha256" "$partial_path" | sha256sum -c - >&2
        mv "$partial_path" "$path"
    fi
    if [ -L "$path" ]; then
        echo "FAIL: refusing a symlinked trained embedding file: ${path}" >&2
        exit 1
    fi
    python3 - "$path" "$expected_size" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
actual = path.stat().st_size
if actual != expected:
    raise SystemExit(f"FAIL: {path.name} is {actual} bytes, expected {expected}")
PY
    printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum -c - >&2
}

echo "Validating pinned ${MODEL_REPOSITORY} package provenance..." >&2
while IFS='|' read -r relative_path expected_size expected_sha256; do
    [ -n "$relative_path" ] || continue
    verify_model_file "$relative_path" "$expected_size" "$expected_sha256"
done <<'FILES'
model.safetensors|90868376|53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
config.json|612|953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41
tokenizer.json|466247|be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
tokenizer_config.json|350|acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b
special_tokens_map.json|112|303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3
sentence_bert_config.json|53|fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7
config_sentence_transformers.json|116|061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3
modules.json|349|84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf
1_Pooling/config.json|190|4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23
README.md|10502|dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0
FILES

python3 - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
pooling = json.loads((root / "1_Pooling/config.json").read_text(encoding="utf-8"))
if pooling.get("word_embedding_dimension") != 384:
    raise SystemExit("FAIL: pinned pooling metadata does not declare 384 dimensions")
if pooling.get("pooling_mode_mean_tokens") is not True:
    raise SystemExit("FAIL: pinned pooling metadata does not declare mean pooling")
if any(
    pooling.get(name) is True
    for name in (
        "pooling_mode_cls_token",
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
    )
):
    raise SystemExit("FAIL: pinned pooling metadata enables an unsupported pooling mode")
sentence = json.loads((root / "sentence_bert_config.json").read_text(encoding="utf-8"))
if sentence.get("max_seq_length") != 256:
    raise SystemExit("FAIL: pinned sentence-transformer context limit changed")
PY

if [ "$BUILD_BINARIES" -eq 1 ]; then
    echo "Building release inspection and server binaries..." >&2
    cargo build --manifest-path "${WORKSPACE_DIR}/Cargo.toml" --locked --release \
        --bin bloom_infer --bin bloom_server >&2
fi
if [ ! -x "$INFER_BIN" ] || [ ! -x "$SERVER_BIN" ]; then
    echo "FAIL: release bloom_infer or bloom_server is unavailable" >&2
    exit 1
fi

RUN_DIR=$(mktemp -d)
trap 'rm -r -- "$RUN_DIR"' EXIT
INSPECT_JSON="${RUN_DIR}/inspect.json"
CLI_STDERR="${RUN_DIR}/cli-stderr.txt"
OPENAI_JSON="${RUN_DIR}/openai.json"
OLLAMA_JSON="${RUN_DIR}/ollama.json"

echo "Inspecting native BERT routing and CPU memory accounting..." >&2
"$INFER_BIN" \
    --model "$MODEL_PATH" \
    --device cpu \
    --context-size 4096 \
    --inspect \
    > "$INSPECT_JSON"
python3 - "$INSPECT_JSON" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = report.get("manifest", {})
memory = report.get("memory_estimate", {})
parameters = manifest.get("parameters", {})
expected = {
    "family": "Bert",
    "primary_dtype": "F32",
}
for name, value in expected.items():
    if manifest.get(name) != value:
        raise SystemExit(
            f"FAIL: expected manifest {name}={value!r}, got {manifest.get(name)!r}"
        )
if parameters.get("bloom_task") != "embedding":
    raise SystemExit("FAIL: native BERT manifest omitted the embedding task")
if parameters.get("max_seq_length") != 256:
    raise SystemExit("FAIL: native BERT manifest omitted the 256-token task limit")
expected_parameters = {
    "embedding_pooling": "mean",
    "embedding_dimensions": 384,
    "embedding_normalization": "l2",
}
for name, value in expected_parameters.items():
    if parameters.get(name) != value:
        raise SystemExit(
            f"FAIL: native BERT manifest omitted {name}={value!r}: {parameters!r}"
        )
if memory.get("weight_dtype") != "F32" or memory.get("weight_bytes") != 90868376:
    raise SystemExit(f"FAIL: native BERT weight accounting changed: {memory!r}")
if memory.get("kv_cache_bytes") != 0 or memory.get("kv_cache_bytes_per_token") != 0:
    raise SystemExit(f"FAIL: stateless BERT incorrectly reserved a KV cache: {memory!r}")
PY

if "$INFER_BIN" \
    --model "$MODEL_PATH" \
    --device cpu \
    --prompt "This text-generation request must not run." \
    > /dev/null 2> "$CLI_STDERR"; then
    echo "FAIL: bloom_infer accepted an embedding encoder for text generation" >&2
    exit 1
fi
if ! grep -F "cannot run an embedding encoder" "$CLI_STDERR" >/dev/null; then
    echo "FAIL: bloom_infer did not return the expected task-isolation diagnostic" >&2
    exit 1
fi

OPENAI_CLIENT_ARGS=()
OLLAMA_CLIENT_ARGS=()
if [ "$REQUIRE_OFFICIAL_CLIENTS" -eq 1 ]; then
    OPENAI_CLIENT_ARGS=(--require-openai-sdk)
    OLLAMA_CLIENT_ARGS=(--require-ollama-sdk)
fi

echo "Running trained semantic embedding, rerank, and task-isolation gates..." >&2
python3 "${WORKSPACE_DIR}/scripts/openai_compat_smoke.py" \
    --model "$MODEL_PATH" \
    --server-bin "$SERVER_BIN" \
    --device cpu \
    --embedding-only \
    --embedding-quality-dimensions 384 \
    --expected-context-size 256 \
    --startup-timeout 180 \
    --api-key bloom-trained-embedding \
    --require-model \
    "${OPENAI_CLIENT_ARGS[@]}" \
    > "$OPENAI_JSON"

echo "Running Ollama current, legacy, discovery, and official-client embedding gates..." >&2
python3 "${WORKSPACE_DIR}/scripts/ollama_compat_smoke.py" \
    --model "$MODEL_PATH" \
    --server-bin "$SERVER_BIN" \
    --device cpu \
    --catalog-only \
    --startup-timeout 180 \
    --request-timeout 180 \
    --api-key bloom-trained-embedding \
    --require-model \
    "${OLLAMA_CLIENT_ARGS[@]}" \
    > "$OLLAMA_JSON"

python3 - "$OPENAI_JSON" "$OLLAMA_JSON" "$MODEL_REPOSITORY" "$MODEL_REVISION" <<'PY'
import json
import math
import pathlib
import sys

openai = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
ollama = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
quality = openai.get("trained_quality")
if openai.get("status") != "ok" or not isinstance(quality, dict):
    raise SystemExit(f"FAIL: trained OpenAI embedding gate failed: {openai!r}")
expected_quality = {
    "dimensions": 384,
    "context_window": 256,
    "generation_routes_rejected": 5,
    "context_limit_checks": 3,
}
for name, value in expected_quality.items():
    if quality.get(name) != value:
        raise SystemExit(f"FAIL: trained embedding quality omitted {name}={value}: {quality!r}")
for name in ("similarity_margin", "rerank_margin"):
    value = quality.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.50
    ):
        raise SystemExit(f"FAIL: trained embedding quality has invalid {name}: {quality!r}")
latency = quality.get("batch_latency_ms")
if (
    isinstance(latency, bool)
    or not isinstance(latency, (int, float))
    or not math.isfinite(latency)
    or latency <= 0
):
    raise SystemExit(f"FAIL: trained embedding benchmark latency is invalid: {quality!r}")
if ollama.get("status") != "ok" or ollama.get("embedding") != "ok":
    raise SystemExit(f"FAIL: trained Ollama embedding gate failed: {ollama!r}")
result = {
    "status": "ok",
    "model": sys.argv[3],
    "revision": sys.argv[4],
    "license": "Apache-2.0",
    "sha256": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "protocols": {
        "openai_embeddings": "ok",
        "openai_rerank": "ok",
        "ollama_embed": "ok",
        "ollama_embeddings_legacy": "ok",
        "generation_task_isolation": "ok",
        "cli_task_isolation": "ok",
    },
    "quality": quality,
}
print(json.dumps(result, indent=2, sort_keys=True))
PY

if [ -n "$BENCHMARK_OUTPUT" ]; then
    mkdir -p "$(dirname "$BENCHMARK_OUTPUT")"
    cp "$OPENAI_JSON" "$BENCHMARK_OUTPUT"
    echo "Embedding benchmark JSON saved to ${BENCHMARK_OUTPUT}" >&2
fi
