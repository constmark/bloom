# Model Load Preflight

Bloom exposes an authenticated, read-only model inspection contract at
`POST /v1/model-management/preflight`. It lets a browser or deployment tool
review task identity, runtime compatibility, and the loader's conservative
memory plan before model weights enter the load queue.

## Request

Send the exact `id` published by `GET /v1/model-management/models`:

```json
{
  "id": "local-model"
}
```

The identifier names one recognized direct child of the configured model
catalog. It is not a filesystem path. Normal model-management authentication,
origin policy, request-size limits, and no-store response policy apply.

## Version 1 response

A successful response has this identity:

```json
{
  "schema_version": 1,
  "object": "bloom.model_preflight",
  "data": {}
}
```

`data` contains:

- `model_id`, `inspected_at`, `loadable`, and an explicit nullable
  `load_blocker`;
- a bounded manifest summary, including the trusted task set;
- the configured and selected engine, maturity, device backend, support level,
  and bounded diagnostics;
- the per-request and aggregate context plan, weight/KV/temporary allocation
  estimate, available budget, reserve, and resulting admission decision.

The task set is exactly `generation` or `embedding` plus `rerank`. Bloom derives
it from the parsed manifest family and trusted `bloom_task` metadata; filenames
and directory names do not grant capabilities.

The complete [example response](../examples/model-preflight.json) validates
against the strict [Draft-07 JSON Schema](../examples/model-preflight.schema.json).
The schema rejects unknown fields and bounds strings and collections. The UI
also caps the complete response at 256 KiB before decoding.

## Decision invariants

Bloom's server and browser independently enforce these invariants:

- `model_id` exactly matches the requested catalog ID;
- the task set is one of the two supported ordered sets;
- `planned_context_tokens` equals `per_request_context_tokens` multiplied by
  `max_concurrent`;
- `max_concurrent` is a positive u64 projection of the server value already
  admitted below the target runtime's semaphore capacity;
- `memory_utilization` is finite and between zero and one;
- `loadable` is true exactly when `load_blocker` is null;
- a loadable report has an available backend, native or fallback runtime
  support, and a memory plan that fits its budget.

Preflight is advisory. Catalog switches run it again, and the transactional
loader remains authoritative because model bytes, host resources, or an
external runtime can change after inspection.

## Caching and disclosure

Inspections run on a blocking worker and are serialized. Successful reports
use a maximum 128-entry, 30-second cache keyed by catalog ID and file or
descriptor metadata. Responses expose neither the catalog root nor a resolved
filesystem path, arbitrary manifest parameters, environment values, or API
credentials.

## Compatibility policy

Bloom UI currently supports exactly `schema_version: 1` with object identity
`bloom.model_preflight`. Missing, older, newer, partial, or unknown-field
documents fail closed as incompatible; the browser does not guess how a future
load decision should be interpreted. Independently deployed UI and server
builds must agree on this schema before model loading is enabled.

Non-success responses use Bloom's normal protocol error envelope and are not
instances of the successful-response schema.
