# Runtime diagnostics

Bloom exposes a versioned runtime snapshot at `GET /v1/observability`. The
endpoint is part of the authenticated `/v1` API and returns `Cache-Control:
no-store`.

```bash
curl http://127.0.0.1:3000/v1/observability \
  -H "Authorization: Bearer $BLOOM_API_KEY"
```

The response identifies itself with:

```json
{
  "schema_version": 1,
  "object": "bloom.observability_snapshot"
}
```

Clients must reject unknown schema versions instead of guessing their meaning.
The complete contract and a synthetic response are published as the
[JSON Schema](../examples/observability-snapshot.schema.json) and
[example](../examples/observability-snapshot.json).

## Snapshot contents

The snapshot includes:

- Bloom server version and monotonic process uptime;
- active and last-requested model identifiers plus bounded load phase and
  progress;
- request, token, scheduler, KV-cache, and optional CacheMesh counters;
- current process RAM/VRAM observations and peak values;
- the active runtime's startup memory estimate when a model is loaded.

Counters are process-local and reset when the server restarts. A snapshot is an
operational observation, not a benchmark result or billing record. Values can
also change between adjacent fields while requests are running, so consumers
must not assume a transactional metrics read.

Load failures are represented by `load.phase: "failed"` and
`failure_present: true`; the snapshot intentionally omits the raw loader error.
Use the authenticated Models UI or model-management endpoints for actionable
preflight and lifecycle details.

## Browser UI

The Diagnostics drawer polls the endpoint every five seconds and also supports
an explicit refresh. It presents model readiness, uptime, request and token
counters, scheduler queues, memory use, load planning, KV cache, and CacheMesh.

`Export JSON` downloads the validated snapshot as
`bloom-diagnostics.json`. The export contains runtime counters, resource sizes,
and model identifiers. It excludes the server address, API key, prompts,
responses, conversation history, model paths, and raw load errors.

The UI limits a diagnostics response to 256 KiB, validates the object identity,
schema version, bounded labels, load-state consistency, and cache metric ranges,
and rejects malformed data rather than rendering it as trusted diagnostics.

## Operational use

- Capture a snapshot immediately before and after reproducing an issue.
- Pair it with the server log and the exact model inventory when filing a
  private support report.
- Treat model identifiers, resource sizes, and usage counters as deployment
  metadata. Review exported files before sharing them publicly.
- Use `/metrics` for Prometheus scraping. Use `/v1/observability` for bounded
  interactive diagnostics and support snapshots.
