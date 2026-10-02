# HTTP workload evidence

`scripts/http_workload.py` exercises one local Bloom generation deployment
through the OpenAI-compatible streaming API. It starts an isolated server with
an exact model path and binary, waits for generation readiness, runs warmup and
concurrency waves, disconnects clients after HTTP acceptance, waits for zero
in-flight requests, verifies a later request, and sends SIGTERM. A JSON report
is written even when the runner detects a failure.

The first candidate deployment cell is the pinned Qwen2 0.5B Instruct Q4_0
GGUF file in [`test_trained_qwen2_runtime.sh`](../scripts/test_trained_qwen2_runtime.sh),
with native Candle CPU and the OpenAI streaming API. Its revision, file size,
SHA-256 and license evidence are fixed by that script. The trained Qwen2 gate
now runs a short `1,2,4` concurrency workload after semantic and protocol
checks. The deterministic tiny Qwen2 gate runs the same mechanics without
claiming model quality or useful throughput. CI retains the trained Qwen2
report as a 14-day build artifact.

## Run an operator workload

Build or select the exact binary to deploy. Verify the model against the
pinned profile before collecting a result. For example:

```bash
cargo build --release --locked --bin bloom_server
python3 scripts/http_workload.py \
  --server-bin target/release/bloom_server \
  --model /path/to/qwen2-0_5b-instruct-q4_0.gguf \
  --expected-model-sha256 aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f \
  --concurrency 1,2,4 \
  --duration-seconds 7200 \
  --max-requests 100000 \
  --max-tokens 64 \
  --collect-runtime-stats \
  --output qwen2-cpu-http-workload.json
```

The report contains the model and server SHA-256, server version, configured
backend/device, CPU model, host RAM, OS and architecture, requested settings, cold-start time, completed
request and reported token counts, latency distributions, first-content
latency, per-wave completion throughput, disconnect recovery, and host RSS
baseline/observed peak/after-recovery drift. Each streaming response must
report the active model, a finish event, positive completion-token usage, and
`[DONE]`. A positive target duration is mandatory: reaching `--max-requests`
before it expires fails the run. `--duration-seconds 0` runs one wave at each
concurrency level for short CI mechanics checks. The request cap and
per-request timeout bound the run when a server becomes slow.

`--collect-runtime-stats` enables authenticated, bounded snapshots from
`/v1/observability` and `/v1/kv-cache-stats`. The runner captures one baseline
after warmup and one snapshot after each completed wave. Endpoint probes run
outside each wave timer, so their latency cannot change reported request
throughput. The report records only allow-listed counters and gauges; it never
stores the API key, prompts, responses, model paths, device labels or raw
endpoint bodies. The default is disabled, which avoids adding diagnostics
requests. A collector can be `available`, `unavailable` (an endpoint is
missing, unauthorized, malformed or timed out), or `reset` (a process-local
counter decreased or uptime moved backwards). Counter deltas are reported only
for monotonic counters; gauges such as queue depth, in-flight requests, block
utilization and memory are reported as observed values. A reset produces a
`null` delta instead of a negative value.

`first_content_ms` is measured from client send to the first nonempty SSE
content delta; it includes connection and admission time. `inter_content_delta_ms`
measures gaps between SSE content deltas, which may each contain multiple
tokens. It is not token-by-token TBT. Host RSS is sampled every 100 ms plus
warm/recovery endpoints; short peaks may be missed. The runner does not measure
accelerator memory. Closing a client after HTTP acceptance proves recovery of
the public request path, but does not prove interruption during an active model
step. Model and binary hashing occurs before the cold-start timer and is
checked again after shutdown. Host RSS sampling uses Linux `/proc`, `ps` on
other POSIX hosts, and the locale-independent Windows PSAPI
`WorkingSetSize` API on Windows. The report states the measurement limits
explicitly.

The workload requests a graceful `SIGTERM` on POSIX and a console Ctrl-Break
on Windows (with a hard-termination fallback when no console is available).

The workload is a standalone HTTP client that drives the server's public
request path. It does not reproduce an in-process benchmark or bypass the
scheduler, and runtime snapshots are observational rather than transactional
benchmark records. `inter_content_delta_ms` is an inter-event gap, not TPOT:
one SSE content event may contain multiple tokens. Use the reported counters
and gauges to explain a run, not as an arbitrary performance gate or a claim
that another host, model or backend will achieve the same result.

This workload is one part of the [deployment exit criteria](production-readiness.md).
Before promoting the Qwen2 CPU cell, run it on an immutable OS/hardware/build
combination and retain the report with cold/warm baselines, server logs, power
settings, CPU model, and deployment configuration. Then separately exercise
bad model load, disk-full, process interruption, upgrade/rollback and long-run
memory behavior. Neither the short CI run nor a single two-hour pass completes
those failure-injection and repeatability requirements.
