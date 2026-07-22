#!/usr/bin/env bash
# CI Smoke Test — runs bloom_bench with a small model and validates output JSON structure.
#
# Usage:
#   ./scripts/ci_smoke_test.sh [--model-path PATH] [--require-model]
#
# Environment variables:
#   BLOOM_MODEL_PATH  — path to a small GGUF model (default: $MODEL_DIR or /tmp/smoke_model)
#   BLOOM_MAX_TOKENS  — max tokens to generate (default: 16)
#   BLOOM_TIMEOUT     — max wall-clock seconds (default: 120)
#   BLOOM_REQUIRE_MODEL — set to 1/true to fail instead of SKIP when the model is missing

set -euo pipefail

# --- Parse arguments ---
MODEL_PATH="${BLOOM_MODEL_PATH:-${MODEL_DIR:-/tmp/smoke_model}}"
MAX_TOKENS="${BLOOM_MAX_TOKENS:-16}"
TIMEOUT_SECS="${BLOOM_TIMEOUT:-120}"
REQUIRE_MODEL="${BLOOM_REQUIRE_MODEL:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --require-model)
            REQUIRE_MODEL=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Bloom CI Smoke Test"
echo "============================================"
echo "Model path:   $MODEL_PATH"
echo "Max tokens:   $MAX_TOKENS"
echo "Timeout:      ${TIMEOUT_SECS}s"
echo "============================================"

# --- Check model exists ---
if [ ! -e "$MODEL_PATH" ]; then
    case "$REQUIRE_MODEL" in
        1|true|TRUE|yes|YES)
            echo "FAIL: Model not found at $MODEL_PATH (required by --require-model/BLOOM_REQUIRE_MODEL)"
            exit 1
            ;;
    esac
    echo "SKIP: Model not found at $MODEL_PATH (set BLOOM_MODEL_PATH or pass --model-path)"
    exit 0
fi

# --- Build bloom_bench ---
echo "[1/5] Building bloom_bench..."
cargo build --release --bin bloom_bench 2>&1 | tail -5

BENCH_BIN="./target/release/bloom_bench"
if [ ! -x "$BENCH_BIN" ]; then
    echo "FAIL: bloom_bench binary not found at $BENCH_BIN"
    exit 1
fi

# --- Run benchmark with timeout ---
echo "[2/5] Running benchmark..."
OUTPUT_FILE=$(mktemp)
trap "rm -f $OUTPUT_FILE" EXIT

set +e
if command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT_SECS" "$BENCH_BIN" \
        --model "$MODEL_PATH" \
        --max-tokens "$MAX_TOKENS" \
        --repetitions 1 \
        --warmup 0 \
        --device cpu \
        > "$OUTPUT_FILE" 2>/dev/null
    EXIT_CODE=$?
elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$TIMEOUT_SECS" "$BENCH_BIN" \
        --model "$MODEL_PATH" \
        --max-tokens "$MAX_TOKENS" \
        --repetitions 1 \
        --warmup 0 \
        --device cpu \
        > "$OUTPUT_FILE" 2>/dev/null
    EXIT_CODE=$?
else
    echo "Warning: timeout/gtimeout command not found, running benchmark without timeout control..."
    "$BENCH_BIN" \
        --model "$MODEL_PATH" \
        --max-tokens "$MAX_TOKENS" \
        --repetitions 1 \
        --warmup 0 \
        --device cpu \
        > "$OUTPUT_FILE" 2>/dev/null
    EXIT_CODE=$?
fi
set -e

if [ $EXIT_CODE -eq 124 ]; then
    echo "FAIL: Benchmark timed out after ${TIMEOUT_SECS}s"
    exit 1
elif [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: Benchmark exited with code $EXIT_CODE"
    cat "$OUTPUT_FILE"
    exit 1
fi

echo "[3/5] Validating output JSON structure..."

# Check it's valid JSON
if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('Valid JSON')" "$OUTPUT_FILE" 2>/dev/null; then
    echo "FAIL: Output is not valid JSON"
    cat "$OUTPUT_FILE"
    exit 1
fi

# Validate required fields
python3 -c "
import json, sys

with open(sys.argv[1]) as f:
    d = json.load(f)

required = ['backend', 'model_id', 'tokens_per_second', 'tokens_generated', 'hardware', 'timing_breakdown']
missing = [k for k in required if k not in d]
if missing:
    print(f'FAIL: Missing required fields: {missing}')
    sys.exit(1)

# Check tokens/s > 0
tps = d.get('tokens_per_second', 0)
if tps <= 0:
    print(f'FAIL: tokens_per_second = {tps} (expected > 0)')
    sys.exit(1)

# Check TTFT < 30s (loose threshold for CI)
ttft = d.get('ttft_ms')
if ttft is not None and ttft > 30000:
    print(f'FAIL: TTFT = {ttft}ms (expected < 30000ms)')
    sys.exit(1)

print(f'OK: tokens/s={tps:.2f}, TTFT={ttft}ms, model={d[\"model_id\"]}')
" "$OUTPUT_FILE"

RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "FAIL: Output validation failed"
    exit 1
fi

# --- Performance budget gate ---
# Compares TTFT/TBT/Peak Memory against docs/performance_budgets.md.
# SKIPs gracefully when hardware tier cannot be classified (e.g. CI runner
# with an unknown device) so a CI run on a generic Linux box doesn't fail
# just because it's not in the budget table. Override with
# BLOOM_REQUIRE_BUDGET=1 to make SKIP a hard failure.
echo "[4/5] Performance budget check..."
BUDGET_REQUIRE="${BLOOM_REQUIRE_BUDGET:-0}"
set +e
python3 ./scripts/bench_budget_check.py "$OUTPUT_FILE"
BUDGET_EXIT=$?
set -e
case "$BUDGET_EXIT" in
    0)
        echo "Budget: PASS"
        ;;
    1)
        echo "Budget: WARN (within tolerance, not gating)"
        ;;
    2)
        echo "FAIL: Budget exceeded beyond tolerance"
        exit 1
        ;;
    3)
        if [ "$BUDGET_REQUIRE" = "1" ]; then
            echo "FAIL: Could not classify hardware tier (BLOOM_REQUIRE_BUDGET=1)"
            exit 1
        fi
        echo "SKIP: Hardware tier not in budget table (set BLOOM_REQUIRE_BUDGET=1 to enforce)"
        ;;
    *)
        echo "FAIL: bench_budget_check.py exited with $BUDGET_EXIT"
        exit 1
        ;;
esac

echo "[5/5] Smoke test PASSED"
exit 0
