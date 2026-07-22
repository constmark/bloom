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
| Apple Silicon | 150 ms | 30 ms | 6.5 GiB |
| NVIDIA RTX | 50 ms | 15 ms | 8.0 GiB |
| x86 CPU | 1,500 ms | 120 ms | 5.5 GiB |
| Intel NPU | 200 ms | 40 ms | 6.0 GiB |

The checker returns:

| Exit code | Result |
| ---: | --- |
| `0` | All measured metrics are within the configured thresholds |
| `1` | At least one metric is no more than 5% over its threshold |
| `2` | At least one metric is more than 5% over its threshold |
| `3` | Hardware could not be classified or required fields are missing |

Set `BLOOM_REQUIRE_BUDGET=1` in environments where an unclassified result must
fail the job.

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
./scripts/compare_llamacpp.py --max-tokens 64
```
