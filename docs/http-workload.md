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

`first_content_ms` is measured from client send to the first nonempty SSE
content delta; it includes connection and admission time. `inter_content_delta_ms`
measures gaps between SSE content deltas, which may each contain multiple
tokens. It is not token-by-token TBT. Host RSS is sampled every 100 ms plus
warm/recovery endpoints; short peaks may be missed. The runner does not measure
accelerator memory. Closing a client after HTTP acceptance proves recovery of
the public request path, but does not prove interruption during an active model
step. Model and binary hashing occurs before the cold-start timer and is
checked again after shutdown. The runner currently requires Linux `/proc` or
macOS `ps` for host RSS; Windows memory sampling has not been implemented. The
report states the measurement limits explicitly.

This workload is one part of the [deployment exit criteria](production-readiness.md).
Before promoting the Qwen2 CPU cell, run it on an immutable OS/hardware/build
combination and retain the report with cold/warm baselines, server logs, power
settings, CPU model, and deployment configuration. Then separately exercise
bad model load, disk-full, process interruption, upgrade/rollback and long-run
memory behavior. Neither the short CI run nor a single two-hour pass completes
those failure-injection and repeatability requirements.
