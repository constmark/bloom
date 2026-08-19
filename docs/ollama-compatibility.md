# Ollama API Compatibility

Bloom exposes a deliberately bounded subset of Ollama's local HTTP API. The
goal is to let existing local chat and generation clients reuse their request
path while Bloom remains the model runtime and lifecycle owner. This is an
experimental compatibility layer, not a claim that Bloom is an Ollama daemon or
that every Ollama client workflow is supported.

Every `/api` response, including NDJSON streams and Ollama-shaped errors,
carries `Cache-Control: no-store`; deployments must preserve that directive
through any reverse proxy.
Before namespace routing, Bloom's default same-origin browser guard rejects an
untrusted, opaque, malformed, or loopback DNS-rebinding `Origin` with HTTP 403.
Origin-free Ollama clients remain compatible. A separately hosted browser
client must be configured as the one exact HTTP(S) origin; rejected requests do
not receive CORS admission.

## Protocol errors

An unknown path anywhere in the `/api` namespace returns HTTP 404 with Ollama's
`{"error":"..."}` shape. A supported path called with the wrong HTTP method
returns HTTP 405 in the same shape and retains the route's `Allow` header. The
messages are fixed and bounded and never reflect the request path, query, or
credentials. No route handler runs and no model or catalog state is exposed.

These fallbacks remain distinguishable from authentication: unknown and
method-mismatched paths retain 404 or 405 when API-key protection is enabled,
while every recognized `/api` route still requires the configured credential.
Reverse proxies must preserve the JSON body, status, `Allow`, `Cache-Control`,
and `x-request-id` instead of substituting an HTML error page.
Inference-capacity HTTP 429 responses also carry `Retry-After`; proxies must
preserve it, and CORS exposes it for browser clients.
Recognized routes rejected for a missing or invalid key return the Ollama error
shape with HTTP 401 and `WWW-Authenticate: Bearer realm="Bloom"`. CORS exposes
the challenge. Public 404/405 namespace fallbacks remain outside authentication
and do not carry it.

Extractor failures also retain the Ollama error shape. Malformed JSON, schema
mismatches, and missing JSON media types return HTTP 400, consistent with the
adapter's bounded invalid-request behavior. A request body over the configured
transport limit retains HTTP 413. These messages are fixed and do not include
parser internals or submitted content.

The upstream contracts used by this adapter are Ollama's
[API reference](https://docs.ollama.com/api/introduction),
[streaming contract](https://docs.ollama.com/api/streaming), and
[tool-calling message format](https://docs.ollama.com/capabilities/tool-calling).
Embedding behavior follows the current
[`/api/embed` contract](https://docs.ollama.com/api/embed) and
[embedding capability guide](https://docs.ollama.com/capabilities/embeddings).
Removal follows Ollama's
[`DELETE /api/delete` contract](https://docs.ollama.com/api/delete).
Verified acquisition follows the wire shape of Ollama's
[`POST /api/pull` contract](https://docs.ollama.com/api/pull), with a narrower
Bloom trust and identity model described below.

## Supported endpoints

| Endpoint | Status | Bloom behavior |
| --- | --- | --- |
| `GET /api/version` | Supported | Returns the Bloom package version in Ollama's `version` envelope. |
| `GET /api/tags` | Supported | Lists bounded Bloom catalog metadata. A signed pull is listed under its persistent signed-index ID rather than its destination filename. |
| `GET /api/ps` | Supported | Reports Bloom's one active runtime under its stable Ollama selector, memory estimate, context length, and model details. |
| `POST /api/show` | Supported subset | Accepts `model` or the legacy `name`, including a persisted signed-index alias; returns bounded details, declared license metadata, capabilities, and context metadata. `verbose: true` is rejected. |
| `POST /api/pull` | Verified subset | Resolves one exact signed-index ID, enforces the signed URL, filename, size, SHA-256, and license, and returns Ollama-compatible progress in NDJSON or one terminal JSON object. |
| `DELETE /api/delete` | Supported subset | Permanently removes one exact, inactive Bloom catalog entry selected by catalog ID or persistent signed-index alias through the shared guarded lifecycle operation. Success has an empty HTTP 200 body. |
| `POST /api/chat` | Supported subset | On-demand exact model activation, text chat, system/user/assistant/tool history, function tools, structured output, streaming and non-streaming responses, plus empty-message preload/unload. |
| `POST /api/generate` | Supported subset | On-demand exact model activation, text prompt plus optional system prompt, structured output, streaming and non-streaming responses, plus empty-prompt preload/unload. |
| `POST /api/embed` | Supported subset | On-demand exact model activation, one string or a batch of strings, default context truncation, optional dimensionality reduction, normalized float vectors, token count, and nanosecond duration. |
| `POST /api/embeddings` | Legacy subset | On-demand exact model activation for one nonblank `prompt`; returns a finite L2-normalized float vector in the superseded Ollama response shape. |

Model creation, copy, push, blob, and cloud endpoints are not exposed under
`/api`. General source inspection, import, cancellation, staged-download
management, and loading remain under `/v1/model-management/*`; the browser UI
is the recommended interactive path. The pull adapter never converts a model
name, client URL, or registry reference into an unverified download.

## Start and connect

Start Bloom with one active model:

```bash
./bloom_server \
  --model /path/to/model.gguf \
  --models-dir /path/to/catalog \
  --host 127.0.0.1 \
  --port 3000
```

The official Ollama Python client can point at Bloom directly:

```python
from ollama import Client

client = Client(host="http://127.0.0.1:3000")
response = client.chat(
    model="default",
    messages=[{"role": "user", "content": "Hello from Bloom"}],
    options={"num_predict": 64, "temperature": 0},
)
print(response.message.content)
```

With verified downloads and a trusted signed index configured, the same client
can acquire an indexed entry:

```python
progress = client.pull("exact-signed-index-id", stream=False)
assert progress.status == "success"

response = client.chat(
    model="exact-signed-index-id",
    messages=[{"role": "user", "content": "Use the pulled model"}],
)
```

For inference, pass the inference API key as a client header:

```python
client = Client(
    host="https://bloom.example.test",
    headers={"Authorization": "Bearer " + BLOOM_API_KEY},
)
```

`X-API-Key` is also accepted. All `/api/*` routes use the same protection as
`/v1/*`; this intentionally differs from an unauthenticated default local
Ollama installation. Keep `/health`, `/ready`, and `/metrics` behind a network
ACL or reverse proxy when the server is not localhost-only.

Set `BLOOM_OPERATOR_API_KEY` to a different value and use it for `/api/pull`,
`DELETE /api/delete`, empty-prompt load/unload requests, explicit `keep_alive`,
or any request that must switch to an inactive model. The operator key is also
accepted for ordinary inference. Once separation is configured, the inference
key can query the catalog and use the already active model but receives HTTP
403 for lifecycle or destructive operations. Omitting the operator key retains
the pre-separation single-key behavior; strict non-loopback deployments reject
that compatibility mode.

## Model identity and lifecycle

Bloom owns one active model runtime. `default` aliases that runtime. Any other
chat, generation, or embedding selector must resolve exactly to one safe local
catalog ID or persisted signed-index alias. If it is inactive and the request
has operator authority, Bloom runs the same integrity and preflight checks as
an explicit native switch, closes inference admission, drains the previous
runtime, and waits for the requested
load to finish before inference. An inference-scoped credential receives HTTP
403 instead of changing the runtime. The previous runtime remains published if
the replacement fails.

Every load has a monotonic process-local sequence and its own terminal result.
Concurrent requests for the same canonical path and selector join that result;
they do not queue duplicate loads or poll global status. A request for a
different model while lifecycle work is active returns HTTP 409. The runtime's
internal metadata ID may differ from the Ollama selector: internal inference is
bound to the actual runtime ID, while JSON and NDJSON responses preserve the
selector supplied by the client.

`DELETE /api/delete` accepts an exact catalog ID or persistent signed-index
alias returned by `/api/tags`; display names and the `default` alias are not
removal selectors.
The ID is bounded and may not contain a path, surrounding whitespace, or
control characters. Bloom refreshes the catalog under its storage gate,
refuses active models, integrity checks, and concurrent lifecycle work, then
performs the same containment-checked removal used by
`POST /v1/model-management/remove`. A missing model returns HTTP 404, unsafe
input returns 400, and a lifecycle race returns 409, all with Ollama's
`{"error":"..."}` shape. Success returns an empty HTTP 200 body, which the
official Python client maps to a successful status. Removal is permanent and
does not unload a runtime; restrict this endpoint to trusted operators.

The Bloom UI and `POST /v1/model-management/switch` remain explicit lifecycle
controls. Ollama also supports its load-only forms: an empty `messages` array
or empty `prompt` activates the exact model and returns a terminal load object.
The same empty request with numeric `0`, string `"0"`, or string `"0s"` as
`keep_alive` unloads that exact active model. Omitted or null values use the
Ollama-compatible five-minute default. Positive JSON numbers are seconds, and
duration strings accept bounded Go-style `ns`, `us`, `µs`, `ms`, `s`, `m`, and
`h` components such as `"250ms"`, `"5m"`, or `"1h30m"`; the maximum is 365
days. Any negative number or valid negative duration retains the model
indefinitely. A zero value on a nonempty chat, generate, or embedding request
unloads only after its buffered response or stream has completed.

Each successfully activated Ollama request cancels the previous residency
timer. Activation and unload commit their new residency policy atomically;
validation, lookup, or load failure leaves the current deadline unchanged.
Inference-scoped requests use the active model without changing an
operator-established deadline. Timers bind to the exact loaded runtime instance,
never unload a replacement
that happens to use the same path, retry while other requests are in flight,
and publish their deadline as `expires_at` through `/api/ps`. Infinite or
non-Ollama-managed residency retains the documented far-future marker.

Bloom reports a catalog digest only when verified acquisition provenance
contains a SHA-256. It does not hash large unverified model files merely to
answer discovery requests. Empty digests and parameter-size strings therefore
mean unknown, not verified-empty.

## Verified pull

`POST /api/pull` requires both `--enable-model-downloads` and a trusted signed
model index. The `model` value must exactly match the index entry's bounded
lowercase ID; Bloom does not resolve Ollama registry names, tags, namespaces,
or arbitrary URLs. `insecure` must be absent, `null`, or `false`. Unknown
non-null fields fail with HTTP 400. This deliberately keeps source selection
and content identity under the operator's signed policy rather than the
calling client's control.

Before starting any network request, Bloom verifies the index signature,
expiry, persistent rollback watermark, immutable Hugging Face commit URL,
license policy, and configured size limit. The shared downloader then enforces
trusted redirects, storage quota, the signed byte size, the signed SHA-256,
atomic no-overwrite installation, and durable provenance. Provenance records
the signed index ID so it survives restart independently of the current index
cache. A catalog filename that already exists succeeds idempotently only when
its kind, format, complete size, digest, package file count, license, download
acquisition, exact signed-index ID, and integrity state match the signed entry.
One clean prior download with the same signed-index ID is upgraded
transactionally, even when the signed filename or file/directory shape changed.
Bloom retains the old payload while staging the new one, admits their combined
peak size, rehashes both identities at commit, and records rollback state under
`.bloom-upgrade`. Startup then restores the old entry or completes a fully
verified replacement after interruption. The source must be inactive, and
load, removal, and integrity operations reject it while the upgrade is active.
An unaliased legacy or manually acquired file is not silently adopted;
otherwise Bloom could report success without preserving the requested Ollama
selector. Occupied destinations, duplicate aliases, quarantined sources, and
ambiguous transaction state fail closed without in-place overwrite. The native
signed-index download endpoint and Models drawer use the same identity rules.

`stream` defaults to `true`. Streaming returns bounded
`application/x-ndjson` progress objects with `status`, a `sha256:` digest,
completed bytes, and a total when the source publishes one, followed by
`{"status":"success"}`. A failure after streaming begins is an NDJSON
`{"error":"..."}` object. `stream: false` waits for the terminal state and
returns one success object or an Ollama-shaped HTTP error.

Concurrent pulls join only when filename, SHA-256, signed size, index ID, and
upgrade source identity all match; other acquisition work returns HTTP 409.
Dropping a progress response stops
only that client's bounded progress relay. The shared verified download keeps
running so another identical call can rejoin it, and a cancelled or transiently
failed download can reuse its matching staged bytes. Use Bloom's authenticated
model-management API or UI to cancel, discard, or inspect staged work.

A successful pull adds the signed index ID to `/api/tags`; it does not
immediately load or switch Bloom's one active runtime. A later chat, generate,
embedding, or empty preload request using that same ID safely activates the
installed destination. Legacy matching acquisitions without the optional alias
can still resolve through the currently verified signed index.

## Chat, generation, and options

`/api/chat` requires `model` and at least one text message for inference. Supported roles are
`system`, `user`, `assistant`, and `tool`. `/api/generate` requires `model` and a
nonblank `prompt` for inference; it also accepts a text `system` prompt. The
empty load/unload forms are described under model lifecycle above.

The following `options` are mapped into Bloom's shared generation admission:

| Ollama option | Bloom behavior |
| --- | --- |
| `num_predict` | Integer from 1 through 32,768; defaults to 128. |
| `temperature` | Finite number, validated by the shared generation layer; defaults to 0.7. |
| `top_p` | Finite number, validated by the shared generation layer; defaults to 0.9. |
| `seed` | `-1` for no fixed seed or a non-negative integer. |
| `stop` | One non-empty string or up to four strings; each is limited to 1,024 characters and all are limited to 16 KiB. Matches are excluded from output and end generation early. |

Unknown options and non-neutral unsupported request fields return HTTP 400.
Bloom does not silently ignore sampling semantics. `think: true` and nonempty
thinking history are rejected because Bloom does not expose a separate
reasoning channel. For Qwen3, normal requests use the model's disabled-thinking
prompt form so raw `<think>` markers do not leak into answer text. Log
probabilities, raw prompt mode, nonempty suffixes, and inline Ollama images are
currently unsupported.
Use Bloom's bounded multimodal endpoints for image or PCM input.

## Embeddings

`/api/embed` requires `model` and accepts `input` as one nonblank string or an
array of 1 through 256 nonblank strings. Each string is limited to 262,144
characters and the batch to 768 KiB. Bloom tokenizes the complete batch before
inference. `truncate` defaults to `true`; over-context inputs are truncated to
the active model's context window. With `truncate: false`, the same condition
returns HTTP 400 without inference.

`dimensions` may be zero or omitted for the model's native width, or an integer
from 1 through 16,384. A smaller value takes the leading dimensions and
renormalizes the projected vector. A value larger than the native width leaves
the native width unchanged, matching current Ollama behavior. Current
`/api/embed` vectors are always finite, nonempty, dimensionally consistent, and
L2-normalized. Bloom rejects zero-norm or non-finite model output and caps a
batch at 1,048,576 float values.

The legacy `/api/embeddings` route accepts one nonblank `prompt`, does not
truncate it, and returns a finite L2-normalized `{"embedding":[...]}` without
metrics. It exists for older clients; new integrations should use `/api/embed`.

The selected runtime must advertise embedding support through its BERT family
or trusted `bloom_task` manifest metadata; a suggestive directory name is not
enough. Both routes activate an
exact inactive local selector before entering the shared embedding executor;
the current `/api/embed` response preserves the requested Ollama selector even
when runtime metadata uses another internal ID. Both embedding routes share the
same timed, infinite, and response-safe zero `keep_alive` behavior as text
inference. `options` must be absent, null, or contain only null-valued fields.
Active per-request runtime options fail with HTTP 400 instead of being ignored.
Empty input remains invalid; use an empty chat or generate request for
lifecycle-only preload or immediate unload.

## Structured output

Both generation endpoints accept:

- `format: "json"` for a JSON object; or
- a bounded JSON Schema object for schema-constrained output.

Schemas pass through Bloom's shared supported-subset validator. Unsupported
keywords fail before inference. Model output is validated at the terminal
boundary; invalid output fails instead of being mislabeled as a successful
structured response. See [Structured output](structured-output.md) for limits
and supported constraints.

## Function tools

`/api/chat` accepts Ollama's nested `type: "function"` tool definitions and up
to eight parallel calls. Bloom never executes a tool. It instructs the active
model to return a private control envelope, validates each selected name and
arguments object against the declared bounded schema, then emits Ollama
`message.tool_calls`.

Ollama tool-result history identifies a call by `tool_name` rather than a call
ID. Bloom assigns non-public internal IDs while normalizing history and matches
each result to the first outstanding call with that name. Every call must have
exactly one following result before another user or assistant turn. This keeps
the shared strict pairing validator intact while preserving Ollama's public
wire shape. Internal IDs are never returned under `/api`.

Function quality depends on the active model's instruction following. Active
tools cannot currently be combined with structured response formats. Custom or
built-in tool types and server-side execution are rejected.

The mandatory native CPU gate uses a deterministic tool profile to require one
valid `get_weather` call, buffered and NDJSON decoding, private internal-ID
non-disclosure, and a name-correlated result continuation. Linux CI requires
the pinned official client to complete the same call, stream, and continuation.
For a compatible generation model, run this proof directly with:

```bash
BLOOM_MODEL_PATH=/path/to/tool-capable-model \
./scripts/ollama_compat_smoke.py \
  --require-model \
  --tool-only \
  --require-ollama-sdk
```

A model that reaches its token limit or emits invalid arguments fails this
mode; an admission-only or fail-closed response is not counted as tool success.

## Streaming and errors

Streaming defaults to `true`. Bloom returns `application/x-ndjson`, one JSON
object per line. Text or tool-call events use `done: false`; one final empty
message or response uses `done: true` and includes token counts, completion
reason, and nanosecond duration fields. Non-streaming requests return one JSON
object with the same terminal metrics.

The adapter bounds decoded internal responses, event count, encoded tool calls,
and output bytes. Dropping the Ollama response body drops the owned internal
stream, which activates Bloom's normal disconnect cancellation path. A failure
before response streaming uses an HTTP error with `{"error":"..."}`. A failure
after streaming starts is a final NDJSON error object, matching Ollama's error
transport convention.

Durations measure Bloom request handling around generation. `load_duration` and
`prompt_eval_duration` are currently zero because the shared runtime does not
publish those phases separately. `total_duration` and `eval_duration` use the
bounded adapter elapsed time; token counts come from Bloom's tokenizer-backed
usage accounting.

## Validation

Discovery and request admission can be checked on a CPU-only machine without a
model:

```bash
python3 -m pip install -r requirements/compat-smoke.txt
./scripts/ollama_compat_smoke.py \
  --build \
  --api-key local-compat-smoke \
  --require-ollama-sdk
```

The command starts an empty Bloom server, validates discovery, API-key
errors, official-client discovery and error decoding, non-mutating missing-model
deletion admission, fail-closed unsupported options, and embedding admission,
then reports model-backed inference as skipped. Run a complete model path with:

```bash
BLOOM_MODEL_PATH=/path/to/model.gguf \
./scripts/ollama_compat_smoke.py --require-model --require-ollama-sdk
```

The repository requirement pins the official client version exercised in CI.
The smoke test reads `/api/show` capabilities: a generation model exercises
non-streaming chat/generate and NDJSON streaming, while an embedding model
exercises current batched projection, legacy output, normalization, and the
official client's `embed` decoder.

`scripts/test_trained_embedding_runtime.sh` pins the official MiniLM package
and adds trained semantic, native-width, context-limit, rerank, and
encoder-only task-isolation evidence to this protocol smoke.

To require successful JSON and JSON Schema output rather than ordinary text,
use a generation model that deterministically follows the requested format:

```bash
BLOOM_MODEL_PATH=/path/to/structured-output-model \
./scripts/ollama_compat_smoke.py \
  --require-model \
  --structured-only \
  --require-ollama-sdk
```

This mode checks buffered and streamed `/api/chat` and `/api/generate`, both
`format: "json"` and a bounded schema, and exact decoding through the pinned
official client. A validation error is a failure in this mode. The mandatory
CPU gate supplies a byte-reproducible native fixture for this protocol proof;
trained-model instruction-following quality remains separate release evidence.
