#!/usr/bin/env bash
# Shared pinned trained-text CPU acceptance runner. Model-specific wrappers
# provide immutable provenance, layout, and semantic prompt configuration.

set -euo pipefail

MODEL_PROFILE="${BLOOM_TRAINED_MODEL_PROFILE:?BLOOM_TRAINED_MODEL_PROFILE is required}"
MODEL_REPOSITORY="${BLOOM_TRAINED_MODEL_REPOSITORY:?BLOOM_TRAINED_MODEL_REPOSITORY is required}"
MODEL_REVISION="${BLOOM_TRAINED_MODEL_REVISION:?BLOOM_TRAINED_MODEL_REVISION is required}"
MODEL_FILENAME="${BLOOM_TRAINED_MODEL_FILENAME:?BLOOM_TRAINED_MODEL_FILENAME is required}"
MODEL_SIZE_BYTES="${BLOOM_TRAINED_MODEL_SIZE_BYTES:?BLOOM_TRAINED_MODEL_SIZE_BYTES is required}"
MODEL_SHA256="${BLOOM_TRAINED_MODEL_SHA256:?BLOOM_TRAINED_MODEL_SHA256 is required}"
MODEL_LICENSE="${BLOOM_TRAINED_MODEL_LICENSE:?BLOOM_TRAINED_MODEL_LICENSE is required}"
LICENSE_SHA256="${BLOOM_TRAINED_LICENSE_SHA256:?BLOOM_TRAINED_LICENSE_SHA256 is required}"
LICENSE_REPOSITORY="${BLOOM_TRAINED_LICENSE_REPOSITORY:-${MODEL_REPOSITORY}}"
LICENSE_REVISION="${BLOOM_TRAINED_LICENSE_REVISION:-${MODEL_REVISION}}"
LICENSE_FILENAME="${BLOOM_TRAINED_LICENSE_FILENAME:-LICENSE}"
SEMANTIC_SYSTEM="${BLOOM_TRAINED_SEMANTIC_SYSTEM:?BLOOM_TRAINED_SEMANTIC_SYSTEM is required}"
SEMANTIC_PROMPT="${BLOOM_TRAINED_SEMANTIC_PROMPT:?BLOOM_TRAINED_SEMANTIC_PROMPT is required}"
SEMANTIC_EXPECTED="${BLOOM_TRAINED_SEMANTIC_EXPECTED:?BLOOM_TRAINED_SEMANTIC_EXPECTED is required}"
CHAT_PROMPT="${BLOOM_TRAINED_CHAT_PROMPT:?BLOOM_TRAINED_CHAT_PROMPT is required}"
MAX_TOKENS="${BLOOM_TRAINED_MAX_TOKENS:?BLOOM_TRAINED_MAX_TOKENS is required}"
EXPECTED_DTYPE="${BLOOM_TRAINED_EXPECTED_DTYPE:?BLOOM_TRAINED_EXPECTED_DTYPE is required}"
MODEL_LAYOUT="${BLOOM_TRAINED_MODEL_LAYOUT:-single-file}"
ADDITIONAL_FILES="${BLOOM_TRAINED_ADDITIONAL_FILES:-}"
EXPECTED_FAMILY="${BLOOM_TRAINED_EXPECTED_FAMILY:-}"
EXPECTED_STORAGE_DTYPE="${BLOOM_TRAINED_EXPECTED_STORAGE_DTYPE:-}"
EXPECTED_CHAT_TEMPLATE_KIND="${BLOOM_TRAINED_EXPECTED_CHAT_TEMPLATE_KIND:-}"
EXPECTED_RUNTIME_WEIGHT_BYTES="${BLOOM_TRAINED_EXPECTED_RUNTIME_WEIGHT_BYTES:-}"

WORKSPACE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_CACHE_DIR="${BLOOM_TRAINED_MODEL_CACHE:-/tmp/bloom-trained-${MODEL_PROFILE}-runtime}"
if [ "$MODEL_LAYOUT" = "directory" ]; then
    MODEL_PATH="${BLOOM_TRAINED_MODEL_PATH:-${MODEL_CACHE_DIR}}"
else
    MODEL_PATH="${BLOOM_TRAINED_MODEL_PATH:-${MODEL_CACHE_DIR}/${MODEL_FILENAME}}"
fi
INFER_BIN="${BLOOM_TRAINED_INFER_BIN:-${WORKSPACE_DIR}/target/release/bloom_infer}"
BENCH_BIN="${BLOOM_TRAINED_BENCH_BIN:-${WORKSPACE_DIR}/target/release/bloom_bench}"
SERVER_BIN="${BLOOM_TRAINED_SERVER_BIN:-${WORKSPACE_DIR}/target/release/bloom_server}"
BENCHMARK_OUTPUT="${BLOOM_TRAINED_BENCHMARK_OUTPUT:-}"
BUILD_BINARIES="${BLOOM_TRAINED_BUILD_BINARIES:-1}"
COMMAND_NAME="${BLOOM_TRAINED_COMMAND_NAME:-$0}"

usage() {
    echo "Usage: ${COMMAND_NAME} [--model-path PATH] [--infer-bin PATH] [--bench-bin PATH] [--server-bin PATH] [--benchmark-output PATH] [--skip-build]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-path)
            MODEL_PATH="${2:?--model-path requires a value}"
            shift 2
            ;;
        --infer-bin)
            INFER_BIN="${2:?--infer-bin requires a value}"
            shift 2
            ;;
        --bench-bin)
            BENCH_BIN="${2:?--bench-bin requires a value}"
            shift 2
            ;;
        --server-bin)
            SERVER_BIN="${2:?--server-bin requires a value}"
            shift 2
            ;;
        --benchmark-output)
            BENCHMARK_OUTPUT="${2:?--benchmark-output requires a value}"
            shift 2
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

for command_name in curl python3 sha256sum timeout; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "FAIL: required command is unavailable: ${command_name}" >&2
        exit 1
    fi
done

RUN_DIR=$(mktemp -d)
trap 'rm -rf "$RUN_DIR"' EXIT
LICENSE_EVIDENCE_PATH="${RUN_DIR}/license-evidence"
SEMANTIC_OUTPUT="${RUN_DIR}/semantic-output.txt"
BENCHMARK_JSON="${RUN_DIR}/benchmark.json"
INSPECT_JSON="${RUN_DIR}/inspect.json"
OPENAI_JSON="${RUN_DIR}/openai.json"
OLLAMA_JSON="${RUN_DIR}/ollama.json"

LICENSE_URL="https://huggingface.co/${LICENSE_REPOSITORY}/resolve/${LICENSE_REVISION}/${LICENSE_FILENAME}"

echo "Validating pinned ${MODEL_REPOSITORY} provenance..." >&2
curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
    --output "$LICENSE_EVIDENCE_PATH" "$LICENSE_URL"
printf '%s  %s\n' "$LICENSE_SHA256" "$LICENSE_EVIDENCE_PATH" | sha256sum -c - >&2

case "$MODEL_LAYOUT" in
    single-file)
        MODEL_ROOT=$(dirname "$MODEL_PATH")
        PRIMARY_MODEL_PATH="$MODEL_PATH"
        ;;
    directory)
        if [ -L "$MODEL_PATH" ] || { [ -e "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; }; then
            echo "FAIL: trained model directory is not a real directory: ${MODEL_PATH}" >&2
            exit 1
        fi
        MODEL_ROOT="$MODEL_PATH"
        PRIMARY_MODEL_PATH="${MODEL_ROOT}/${MODEL_FILENAME}"
        ;;
    *)
        echo "FAIL: BLOOM_TRAINED_MODEL_LAYOUT must be single-file or directory" >&2
        exit 1
        ;;
esac
mkdir -p "$MODEL_ROOT"

verify_model_file() {
    local filename="$1"
    local expected_size="$2"
    local expected_sha256="$3"
    local path="${4:-${MODEL_ROOT}/${filename}}"
    case "$filename" in
        ""|*/*|*\\*|.|..)
            echo "FAIL: trained-model filename is unsafe: ${filename}" >&2
            exit 1
            ;;
    esac
    local partial_path="${path}.partial"
    local url="https://huggingface.co/${MODEL_REPOSITORY}/resolve/${MODEL_REVISION}/${filename}"
    if [ ! -f "$path" ]; then
        if [ -e "$path" ]; then
            echo "FAIL: trained model path exists but is not a regular file: ${path}" >&2
            exit 1
        fi
        if [ -L "$partial_path" ]; then
            echo "FAIL: refusing a symlinked trained-model partial file: ${partial_path}" >&2
            exit 1
        fi
        echo "Downloading pinned ${filename} (${expected_size} bytes)..." >&2
        curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
            --retry 4 --retry-all-errors --continue-at - \
            --output "$partial_path" "$url"
        printf '%s  %s\n' "$expected_sha256" "$partial_path" | sha256sum -c - >&2
        mv "$partial_path" "$path"
    fi
    if [ -L "$path" ]; then
        echo "FAIL: refusing a symlinked trained model file: ${path}" >&2
        exit 1
    fi
    python3 - "$path" "$expected_size" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
actual = path.stat().st_size
if actual != expected:
    raise SystemExit(f"FAIL: trained model size is {actual}, expected {expected}")
PY
    printf '%s  %s\n' "$expected_sha256" "$path" | sha256sum -c - >&2
}

verify_model_file "$MODEL_FILENAME" "$MODEL_SIZE_BYTES" "$MODEL_SHA256" "$PRIMARY_MODEL_PATH"
if [ -n "$ADDITIONAL_FILES" ]; then
    while IFS='|' read -r filename expected_size expected_sha256; do
        [ -n "$filename" ] || continue
        if [ -z "$expected_size" ] || [ -z "$expected_sha256" ]; then
            echo "FAIL: invalid BLOOM_TRAINED_ADDITIONAL_FILES entry" >&2
            exit 1
        fi
        verify_model_file "$filename" "$expected_size" "$expected_sha256"
    done <<< "$ADDITIONAL_FILES"
fi

case "$BUILD_BINARIES" in
    1|true|TRUE|yes|YES)
        echo "Building release inference, benchmark, and server binaries..." >&2
        cargo build --manifest-path "${WORKSPACE_DIR}/Cargo.toml" --locked --release \
            --bin bloom_infer --bin bloom_bench --bin bloom_server >&2
        ;;
    0|false|FALSE|no|NO)
        ;;
    *)
        echo "FAIL: BLOOM_TRAINED_BUILD_BINARIES must be a boolean" >&2
        exit 1
        ;;
esac
if [ ! -x "$INFER_BIN" ] || [ ! -x "$BENCH_BIN" ] || [ ! -x "$SERVER_BIN" ]; then
    echo "FAIL: release inference, benchmark, or server binary is unavailable" >&2
    exit 1
fi

echo "Inspecting trained-model routing and CPU memory plan..." >&2
"$INFER_BIN" \
    --model "$MODEL_PATH" \
    --device cpu \
    --context-size 512 \
    --inspect \
    > "$INSPECT_JSON"
python3 - "$INSPECT_JSON" "$EXPECTED_FAMILY" "$EXPECTED_STORAGE_DTYPE" \
    "$EXPECTED_CHAT_TEMPLATE_KIND" "$EXPECTED_DTYPE" \
    "$EXPECTED_RUNTIME_WEIGHT_BYTES" <<'PY'
import json
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as handle:
    report = json.load(handle)
manifest = report.get("manifest", {})
memory = report.get("memory_estimate", {})
expected_family, expected_storage, expected_template, expected_runtime, expected_weight_bytes = sys.argv[2:]
if expected_family and manifest.get("family") != expected_family:
    raise SystemExit(
        f"FAIL: expected model family {expected_family!r}, got {manifest.get('family')!r}"
    )
if expected_storage and manifest.get("primary_dtype") != expected_storage:
    raise SystemExit(
        f"FAIL: expected storage dtype {expected_storage!r}, got {manifest.get('primary_dtype')!r}"
    )
if expected_template and manifest.get("parameters", {}).get("chat_template_kind") != expected_template:
    raise SystemExit(
        "FAIL: trained model did not select the expected safe chat-template contract"
    )
if expected_runtime and memory.get("weight_dtype") != expected_runtime:
    raise SystemExit(
        f"FAIL: expected runtime dtype {expected_runtime!r}, got {memory.get('weight_dtype')!r}"
    )
if expected_weight_bytes:
    expected = int(expected_weight_bytes)
    if memory.get("weight_bytes") != expected:
        raise SystemExit(
            f"FAIL: expected runtime weight estimate {expected}, got {memory.get('weight_bytes')!r}"
        )
PY

echo "Running deterministic trained-model semantic check..." >&2
timeout 180 "$INFER_BIN" \
    --model "$MODEL_PATH" \
    --device cpu \
    --context-size 512 \
    --disable-memory-prealloc \
    --system-prompt "$SEMANTIC_SYSTEM" \
    --prompt "$SEMANTIC_PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0 \
    --seed 42 \
    --quiet \
    > "$SEMANTIC_OUTPUT"
python3 - "$SEMANTIC_OUTPUT" "$SEMANTIC_EXPECTED" <<'PY'
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
expected = sys.argv[2]
if len(text.encode("utf-8")) > 1024:
    raise SystemExit("FAIL: trained-model semantic output exceeded 1024 bytes")
if text.strip() != expected:
    raise SystemExit(f"FAIL: expected exact trained-model output {expected!r}, got {text!r}")
PY

echo "Running trained-model CPU benchmark..." >&2
timeout 240 "$BENCH_BIN" \
    --model "$MODEL_PATH" \
    --prompt "$CHAT_PROMPT" \
    --device cpu \
    --context-size 512 \
    --max-tokens "$MAX_TOKENS" \
    --repetitions 1 \
    --warmup 0 \
    > "$BENCHMARK_JSON"

echo "Running trained-model OpenAI protocol smoke..." >&2
python3 "${WORKSPACE_DIR}/scripts/openai_compat_smoke.py" \
    --model "$MODEL_PATH" \
    --server-bin "$SERVER_BIN" \
    --device cpu \
    --max-tokens "$MAX_TOKENS" \
    --semantic-system "$SEMANTIC_SYSTEM" \
    --semantic-prompt "$SEMANTIC_PROMPT" \
    --expected-output "$SEMANTIC_EXPECTED" \
    --startup-timeout 180 \
    --require-model \
    > "$OPENAI_JSON"

echo "Running trained-model Ollama protocol smoke..." >&2
python3 "${WORKSPACE_DIR}/scripts/ollama_compat_smoke.py" \
    --model "$MODEL_PATH" \
    --server-bin "$SERVER_BIN" \
    --device cpu \
    --max-tokens "$MAX_TOKENS" \
    --semantic-system "$SEMANTIC_SYSTEM" \
    --semantic-prompt "$SEMANTIC_PROMPT" \
    --expected-output "$SEMANTIC_EXPECTED" \
    --startup-timeout 180 \
    --request-timeout 180 \
    --require-model \
    > "$OLLAMA_JSON"

python3 - "$BENCHMARK_JSON" "$OPENAI_JSON" "$OLLAMA_JSON" \
    "$MODEL_REPOSITORY" "$MODEL_REVISION" "$MODEL_FILENAME" "$MODEL_SHA256" \
    "$MODEL_LICENSE" "$SEMANTIC_EXPECTED" "$EXPECTED_DTYPE" "$MODEL_LAYOUT" \
    "$EXPECTED_RUNTIME_WEIGHT_BYTES" <<'PY'
import json
import math
import pathlib
import sys

benchmark_path = pathlib.Path(sys.argv[1])
with benchmark_path.open(encoding="utf-8") as handle:
    benchmark = json.load(handle)
required = {
    "backend",
    "model_id",
    "dtype",
    "tokens_per_second",
    "tokens_generated",
    "peak_memory_bytes",
    "duration_secs",
    "timestamp",
    "hardware",
    "timing_breakdown",
}
missing = sorted(required.difference(benchmark))
if missing:
    raise SystemExit(f"FAIL: trained benchmark omitted required fields: {missing}")
for field in ("tokens_per_second", "duration_secs"):
    value = benchmark[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"FAIL: trained benchmark has invalid {field}: {value!r}")
if not isinstance(benchmark["tokens_generated"], int) or benchmark["tokens_generated"] <= 0:
    raise SystemExit("FAIL: trained benchmark generated no tokens")
if benchmark["backend"] != "cpu":
    raise SystemExit(f"FAIL: trained benchmark used unexpected backend: {benchmark['backend']!r}")
if benchmark["dtype"] != sys.argv[10]:
    raise SystemExit(
        f"FAIL: trained benchmark used unexpected dtype: {benchmark['dtype']!r}"
    )
if sys.argv[12]:
    memory = benchmark.get("memory_breakdown", {})
    if memory.get("weight_bytes") != int(sys.argv[12]):
        raise SystemExit(
            "FAIL: trained benchmark did not preserve the inspected runtime weight estimate"
        )
with pathlib.Path(sys.argv[2]).open(encoding="utf-8") as handle:
    openai = json.load(handle)
with pathlib.Path(sys.argv[3]).open(encoding="utf-8") as handle:
    ollama = json.load(handle)
if (
    openai.get("status") != "ok"
    or openai.get("stream_events", 0) <= 0
    or openai.get("semantic_output") != "validated"
):
    raise SystemExit(f"FAIL: trained OpenAI protocol smoke failed: {openai!r}")
if (
    ollama.get("status") != "ok"
    or ollama.get("generation") != "ok"
    or ollama.get("semantic_output") != "validated"
):
    raise SystemExit(f"FAIL: trained Ollama protocol smoke failed: {ollama!r}")
result = {
    "status": "ok",
    "model": sys.argv[4],
    "revision": sys.argv[5],
    "filename": sys.argv[6],
    "sha256": sys.argv[7],
    "license": sys.argv[8],
    "semantic_output": sys.argv[9],
    "layout": sys.argv[11],
    "protocols": {
        "openai": "streaming_ok",
        "ollama": "generation_ok",
    },
    "benchmark": benchmark,
}
print(json.dumps(result, indent=2, sort_keys=True))
PY

if [ -n "$BENCHMARK_OUTPUT" ]; then
    mkdir -p "$(dirname "$BENCHMARK_OUTPUT")"
    cp "$BENCHMARK_JSON" "$BENCHMARK_OUTPUT"
    echo "Benchmark JSON saved to ${BENCHMARK_OUTPUT}" >&2
fi
