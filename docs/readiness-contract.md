# Readiness Contract

`GET /ready` is Bloom's public, model-free UI/server handshake and readiness
probe. It returns HTTP 200 when new inference can be admitted and HTTP 503 in
every other state. Both statuses carry the same bounded JSON shape,
`Cache-Control: no-store`, and `x-request-id`.

The response identity is:

```json
{
  "schema_version": 3,
  "object": "bloom.readiness",
  "protocol_version": 3,
  "minimum_ui_protocol_version": 3,
  "maximum_ui_protocol_version": 3,
  "server_version": "0.1.0"
}
```

`schema_version` identifies the response shape. `protocol_version` identifies
the behavioral contract implemented by the server. The inclusive
`minimum_ui_protocol_version` and `maximum_ui_protocol_version` fields are the
server's explicit supported UI protocol range. A client must require the Bloom
object name and a schema it can decode, validate that the range is positive and
ordered, require the server protocol itself to fall inside the range, and
require its own protocol version to fall inside the range. A matching HTTP
status or a partial set of familiar fields is not sufficient proof that the
endpoint is compatible.

The Bloom UI currently implements protocol 3. A future protocol 4 server may
remain compatible by preserving schema 3 and advertising `3..4` only when it
still implements every protocol 3 invariant. A server advertising `4..4` is
rejected by the current UI. New optional fields may be added to one schema
version. Removing a field, changing its type or meaning, or changing a required
invariant requires a new schema version. An incompatible behavioral change
requires a new protocol version and must raise the minimum supported UI
protocol when backward behavior cannot be preserved.

## Required fields

| Field | Meaning |
| --- | --- |
| `minimum_ui_protocol_version` | Oldest UI protocol the server explicitly supports |
| `maximum_ui_protocol_version` | Newest UI protocol the server explicitly supports |
| `status` | `ready` only when the server can admit inference; otherwise `not_ready` |
| `progress` | Current model-load progress from 0 through 100 |
| `model` | Bounded active or requested model identity; never a filesystem path |
| `loading` | Whether a model lifecycle operation is in progress |
| `load_error` | A bounded, path-free public failure summary or `null` |
| `input_modalities` | Bounded active-model input capability labels |
| `model_tasks` | Bounded active-model tasks: `generation`, or `embedding` plus `rerank` |
| `context_window` | Positive u64 active-model context limit or `null` |
| `in_flight_requests` | Process-local admitted inference count as a u64 |
| `available_permits` | Immediately available inference permits as a platform-independent u64 |
| `memory_pressure_high` | Whether memory pressure currently blocks readiness |
| `ram_utilization` | Current normalized RAM utilization in `[0, 1]` |

A `ready` response is internally consistent: model loading is complete,
progress is 100, no public load error is present, at least one model task is
declared, the context window is known, at least one permit is available, and
high memory pressure is false. Consumers must still handle readiness changing
immediately after a response.

The [JSON Schema](../examples/readiness.schema.json) and
[example response](../examples/readiness.json) are packaged with release
archives. `scripts/test_server_http_boundary.py` validates the exact staged
binary's HTTP status, identity, required fields, cache directive, and request
correlation before an archive is created.

## Browser behavior

The Bloom UI validates the complete required envelope within a 64 KiB response
budget. A reachable endpoint with an unknown object or schema, an invalid or
unsupported protocol range, a missing field, invalid type, unsupported task
set, or inconsistent ready state is shown
as **Incompatible Bloom server**. The task set decides whether the browser
exposes generation or embedding/rerank controls. Network and HTTP availability
failures remain a separate
**Bloom server is unavailable** state. This distinction prevents an old Bloom
server, an arbitrary health endpoint, or a misrouted reverse proxy from being
treated as a valid chat runtime.

Readiness validation alone does not publish the browser's final connection
state. The UI next issues a bounded authenticated `GET /v1/models` probe and
requires Bloom's empty-or-singleton Models list. HTTP 401 becomes the separate
**API key required** state, other HTTP failures remain unavailable, and a
malformed successful response remains incompatible. This second stage validates
the configured credential without loading a model or exposing catalog entries.

The endpoint intentionally does not require the `/v1` API key so orchestrators
and the first UI connection can probe it. It exposes bounded deployment and
model metadata, so non-local deployments should still restrict it with a
network ACL or reverse proxy.
