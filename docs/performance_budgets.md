# Performance Checks

Bloom records three primary runtime metrics:

- **TTFT:** time from request admission to the first generated token.
- **TBT:** average latency between generated tokens after the first token.
- **Peak memory:** maximum resident host or device memory during the run.

Use identical model files, prompts, context sizes, generation lengths, build
profiles, and power settings when comparing results.

## Benchmark

```bash
cargo run --release --bin bloom_bench -- \
  --model /path/to/model.gguf \
  --max-tokens 64 \
  --repetitions 3 > bench.json

./scripts/bench_budget_check.py bench.json
```

The benchmark output must validate against
[`examples/benchmark-schema.json`](../examples/benchmark-schema.json).

## Current gate thresholds

`scripts/bench_budget_check.py` currently classifies four hardware tiers. These
values are regression thresholds used by the script, not published performance
claims.

| Tier | TTFT | TBT | Peak memory |
| --- | ---: | ---: | ---: |
| Apple Silicon | 150 ms | 30 ms | 6,500 MiB |
| NVIDIA RTX | 50 ms | 15 ms | 8,000 MiB |
| x86 CPU | 1,500 ms | 120 ms | 5,500 MiB |
| Intel NPU | 200 ms | 40 ms | 6,000 MiB |

`bloom_bench` includes `hardware.arch`. CPU results require explicit x86
architecture evidence before using the x86 tier; ARM CPU results remain
unclassified until they have a separate budget. Apple GPU classification
requires macOS and ARM architecture. A CPU selection takes precedence over the
backend name, and OpenVINO alone does not imply NPU execution. Older records
without architecture may need to be regenerated or remain unclassified.

The checker returns:

| Exit code | Result |
| ---: | --- |
| `0` | All three required metrics are valid and within the configured thresholds |
| `1` | At least one metric is no more than 5% over its threshold |
| `2` | At least one metric is more than 5% over its threshold |
| `3` | Measurements are valid but hardware could not be classified |
| `4` | Data is malformed, missing, negative, non-finite, or otherwise invalid |

`bloom_bench` also emits `run_samples` plus p50/p95 TTFT, TBT, and decode
throughput fields when measured generation runs produce tokens. The existing
budget checker intentionally uses the average fields for backward-compatible
regression gates; promotion reports should retain the percentile fields and
use p95 latency for the release decision. When those optional fields are
present, the checker still validates that they are finite, non-negative,
ordered (`p50 <= p95`), and accompanied by a non-negative integer
`run_samples`; malformed distribution metadata exits with code `4`.

Set `BLOOM_REQUIRE_BUDGET=1` in environments where an unclassified result must
fail the job.

Invalid data always fails the smoke job, independently of
`BLOOM_REQUIRE_BUDGET`. TTFT and TBT must be finite non-negative numbers, and
peak memory must be a positive integer byte count. Timing lookup prefers the
primary field, then the top-level average, then the nested timing average;
an invalid present value is not replaced by a more favorable fallback.
Missing peak memory is not filled from a model's estimated memory breakdown.
The 5% WARN boundary is inclusive. A run too short to measure TBT cannot pass
all three budgets; generate multiple output tokens before checking.

These checks validate reported values. Read the benchmark's `notes` to
distinguish observed process RSS from a runtime estimate; the checker does not
prove independent device-memory sampling or production performance.

Run the model-free evidence regression gate with
`python3 scripts/test_bench_budget_check.py`.

## Recording results

Every published result should include:

- Bloom commit and build features
- model source, file hash, format, and quantization
- device, operating system, driver, and runtime versions
- prompt tokens, generated tokens, context limit, and batch settings
- TTFT, TBT, throughput, and peak host/device memory

Compare Bloom and llama.cpp on the same machine and model with:

```bash
BLOOM_MODEL_PATH=/path/to/model.gguf \
LLAMA_CPP_BIN=/path/to/llama-cli \
./scripts/compare_llamacpp.py \
  --context-size 4096 --device cpu --backend candle \
  --threads 8 --warmup 2 --repetitions 3 --max-tokens 64 \
  --seed 42 --temperature 0.7 --top-p 0.9
```

The comparison report records these workload settings and passes the context,
seed, temperature and top-p to both `bloom_bench` and the llama.cpp CLI. Bloom
receives the same backend/device, warmup and repetition settings.
`llama-bench` accepts a token-count workload rather than prompt text; use
`llama-cli` or `main` when exact prompt text parity is required. The harness
remains a measurement aid; release promotion still requires the full protocol
and raw artifacts described in [llama.cpp parity](llamacpp-parity.md).
