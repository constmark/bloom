# Model Catalog Contract

Bloom exposes the authenticated model-management snapshot at
`GET /v1/model-management/models`. The Models drawer polls this read-only
endpoint to render local models, current residency and load state, optional
acquisition capabilities, storage accounting, and integrity progress.

This endpoint is distinct from OpenAI-compatible `GET /v1/models`, which is a
small active-runtime discovery response used during connection testing.

## Version 1 identity

A successful response starts with:

```json
{
  "schema_version": 1,
  "object": "bloom.model_catalog"
}
```

Bloom UI supports exactly schema version 1. Missing, older, newer, partial, and
unknown-field documents fail closed instead of silently turning missing data
into an empty catalog or disabled capability.

The complete [active response](../examples/model-catalog.json) and
[empty-runtime response](../examples/model-catalog-empty.json) validate against
the strict [Draft-07 JSON Schema](../examples/model-catalog.schema.json).

## Snapshot contents

The response contains:

- `root` and `root_exists` for the configured server-side catalog directory;
- at most 4,096 recognized direct-child model entries;
- the active runtime and its catalog association, when present;
- lifecycle phase, progress, requested model, and a bounded load error;
- download and browser-import capability, policy, progress, and at most 1,000
  resumable staged entries for each acquisition path;
- signed-index trust configuration without publisher keys or source paths;
- exact installed, staged, reserved, committed, quota, and cleanup accounting;
- the current cancellable integrity-verification state.

The browser limits the complete response to 16 MiB before decoding. The server
also bounds direct-child inspection and fails the scan instead of returning a
partial catalog when its published-model or inspection ceiling is exceeded.

## Validated invariants

The UI validates the complete document before replacing the last good drawer
state. Important invariants include:

- model IDs are bounded, path-free, unique direct-child selectors;
- no more than one entry is active, and a catalog-backed active runtime names
  that exact entry;
- lifecycle phases, acquisition phases, formats, modalities, and integrity
  phases use known values;
- disabled acquisition or index capabilities carry their exact inert state;
- staged filenames are unique and progress never exceeds a declared total;
- license allowlists are bounded and consistent with enforcement;
- `used_bytes` is the sum of installed and staged bytes, while
  `committed_bytes` also includes reservations;
- quota availability, integrity hashes, timestamps, and terminal state are
  internally consistent.

The response is a snapshot, not a transaction token. Every mutating endpoint
revalidates its exact catalog selector and lifecycle constraints independently.

## Security and disclosure

The endpoint requires the same API authentication and browser-origin policy as
other model-management routes and is always non-cacheable. It never accepts a
filesystem path from the client.

`root` intentionally reveals the configured server-side catalog directory to
an authenticated operator because the local Models drawer displays where model
files must be installed. Treat the response as private deployment metadata. Do
not expose the endpoint to untrusted users, and use the separate model inventory
export when a portable path-free artifact is required.

The response omits resolved per-model paths, model file contents, provenance
record paths, API credentials, signed-index public keys and source locations,
and invalid provenance details.

Non-success responses use Bloom's normal protocol error envelope and are not
instances of the successful-response schema.
