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
backend/device, a sanitized `/v1/backends` physical-device inventory, CPU model, host RAM, OS and architecture, requested settings, cold-start time, completed
request and reported token counts, latency distributions, first-content
latency, per-wave completion throughput, disconnect recovery, and host RSS
baseline/observed peak/after-recovery drift. Each streaming response must
report the active model, a finish event, positive completion-token usage, and
`[DONE]`. A positive target duration is mandatory: reaching `--max-requests`
before it expires fails the run. `--duration-seconds 0` runs one wave at each
concurrency level for short CI mechanics checks. The request cap and
per-request timeout bound the run when a server becomes slow.

The `backend_inventory` report section is collected after readiness and before
the timed waves. It retains only bounded physical probe results (availability,
vendor, memory topology, supported formats/dtypes, and execution flags) and
never includes the active model ID, local paths, prompts, or engine diagnostic
text. An unavailable inventory is recorded as evidence failure rather than
silently treated as proof of hardware support; the workload itself can still
finish so operators can inspect a partial report.

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
Both workload and failure-injection children use a private configuration path
and discard inherited `BLOOM_*` settings. Personal credentials, saved model
paths and development escape hatches cannot silently alter the measured cell.
The failure runner validates the complete readiness contract and rehashes the
binary and model after the drills; replacement during a drill invalidates it.

The workload is a standalone HTTP client that drives the server's public
request path. It does not reproduce an in-process benchmark or bypass the
scheduler, and runtime snapshots are observational rather than transactional
benchmark records. `inter_content_delta_ms` is an inter-event gap, not TPOT:
one SSE content event may contain multiple tokens. Use the reported counters
and gauges to explain a run, not as an arbitrary performance gate or a claim
that another host, model or backend will achieve the same result.

This workload is one part of the [deployment exit criteria](production-readiness.md).

## Enforce the deployment-cell evidence gate

After collecting a target-host report, run the fail-closed validator. It
requires a two-hour run by default, at least two distinct concurrency levels,
fixed model/build identity, physical backend inventory, host-memory evidence,
and disconnect recovery. A passing result means only that the workload
evidence is complete; it does not promote the cell to stable support.

```bash
python3 scripts/validate_deployment_cell.py qwen2-cpu-http-workload.json \
  --expected-model-sha256 aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f \
  --backend candle --device cpu \
  --failure-report qwen2-cpu-failure-injection.json \
  > qwen2-cpu-cell-validation.json
```

The validator deliberately leaves physical disk-full, target-host
upgrade/rollback, accelerator-memory, and token-level TBT checks as separate
acceptance gates. This prevents a short HTTP smoke test from being mistaken
for a complete production claim.

For the production workload portion of acceptance, select service budgets on
the target host **before** running the measurement. Use `--production-profile`
to require at least 24 measured hours and 1,000 completed requests, explicit
expected model/build/backend/device identity and all six budget limits. Smaller
`--min-duration-seconds` or `--min-requests` values cannot weaken this profile.
Both latency distributions must have one measurement per request and ordered,
positive finite percentiles. Missing metrics, excessive memory growth and
violated latency limits fail rather than being treated as skipped checks.

For example, write `cell-budgets.json` with operator-selected limits (the values
below illustrate the format; they are not Bloom performance promises):

```json
{
  "cold_start_ms": 60000,
  "latency_p95_ms": 30000,
  "first_content_p95_ms": 5000,
  "recovery_ms": 10000,
  "peak_rss_bytes": 4294967296,
  "rss_growth_bytes": 67108864
}
```

Collect a workload with `--duration-seconds 86400` and a sufficiently large
`--max-requests`, then validate it against the exact deployed binary hash:

```bash
python3 scripts/validate_deployment_cell.py qwen2-cpu-http-workload.json \
  --production-profile --budgets cell-budgets.json \
  --expected-model-sha256 aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f \
  --expected-server-sha256 "$DEPLOYED_SERVER_SHA256" \
  --backend candle --device cpu \
  --failure-report qwen2-cpu-failure-injection.json
```

The result still explicitly reports `stable_claim: false`: the workload gate
cannot substitute for model quality, physical disk-full, upgrade/rollback,
target-browser accessibility or release supply-chain evidence.

## Durable local CPU soak

On POSIX hosts, `run_deployment_soak.py` runs the initial Candle CPU profile
without keeping a terminal open. It copies the exact server and validation
scripts into a new private run directory, records their hashes, runs process
failure drills, executes a 24-hour workload at concurrency 1 and 2, then checks
all budgets with the production profile. Inference uses strict memory admission.
Rayon/OpenMP thread limits default to two. A failed or cancelled run never becomes
passing evidence, and the tool never promotes a support-matrix row.

```bash
python3 scripts/run_deployment_soak.py start \
  --server-bin target/release/bloom_server \
  --model /path/to/qwen2-0_5b-instruct-q4_0.gguf \
  --expected-model-sha256 aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f \
  --budgets cell-budgets.json --run-dir output/deployment-soak/qwen2-cpu

python3 scripts/run_deployment_soak.py status output/deployment-soak/qwen2-cpu
python3 scripts/run_deployment_soak.py cancel output/deployment-soak/qwen2-cpu
```

`status.json` records the final outcome; `progress.json` is atomically updated
every 30 seconds during inference. `workload.json`, `failures.json` and
`validation.log` retain the measured evidence. The directory also contains local
model/binary paths and private logs; review it before sharing. `cancel` writes a
control marker and the owning runner terminates and reaps its process group;
it does not signal a PID copied from a stale status file. Each phase also has a
separate owner-monitoring deadline guard: if the runner is forcibly killed,
the guard detects parent loss and reaps the phase rather than letting it run
for the remainder of the day. This produces interrupted, not passing evidence.
Keep the machine awake
and the model available throughout the run. Interruption/reboot requires a fresh
run directory, not resuming a partial report as a complete soak.

For an orchestration-only check, add `--smoke-seconds 5`. Such a run is always
labelled `smoke`, even if its individual commands succeed. The request cap is
one million: reaching it before the measured duration fails the run. The helper
retains wave measurements in memory, so budget the driver as well as the server
on very fast or very small-model workloads. The initial runner supports local
POSIX CPU cells only; the underlying workload can still be used independently
for other target-specific acceptance procedures.

The process-boundary and storage-quota checks are available as a separate gate:

```bash
python3 scripts/deployment_failure_injection.py \
  --server-bin target/release/bloom_server \
  --model /path/to/qwen2-0_5b-instruct-q4_0.gguf \
  --scenario all --backend candle --device cpu \
  --output qwen2-cpu-failure-injection.json
```

`invalid-model` verifies that an invalid package becomes a terminal readiness
or load failure. `process-interruption` sends a hard process interruption and
then starts the same model catalog again, proving that the operating-system
lease is recoverable. `storage-quota` starts bounded local imports with a
one-byte catalog quota, verifies the expected 409 conflict, and checks that no
staging file remains. Pass the resulting report to the evidence gate with
`--failure-report`; a failed or incomplete scenario is rejected.
Before promoting the Qwen2 CPU cell, run it on an immutable OS/hardware/build
combination and retain the report with cold/warm baselines, server logs, power
settings, CPU model, and deployment configuration. Then separately exercise
bad model load, disk-full, process interruption, upgrade/rollback and long-run
memory behavior. Neither the short CI run nor a single two-hour pass completes
those failure-injection and repeatability requirements.
