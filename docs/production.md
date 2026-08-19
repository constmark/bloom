# Production Checklist

Bloom is pre-1.0. Only deploy model/backend combinations with your own
real-model and hardware validation.

## Server

- Set `BLOOM_API_KEY` or `--api-key`.
- Non-loopback listeners fail closed without an API key. Do not enable
  `--allow-unauthenticated-network` or
  `BLOOM_ALLOW_UNAUTHENTICATED_NETWORK` in production; that explicit escape
  hatch exists only for isolated development environments and is rejected by
  strict security mode.
- The official Dockerfile runs with UID/GID `10001`, uses
  `/var/lib/bloom` for mutable state, and enables strict security and memory
  admission. Supply the API key at runtime rather than baking it into an image.
  Keep those defaults in derived images, use a named volume or make bind mounts
  writable by `10001:10001`, and preserve a read-only root filesystem whenever
  the selected external runtimes permit it.
- Preserve the Docker builder's download-disabled Dioxus CLI and checksummed
  Linux UI tools. Updating `dx`, `wasm-bindgen`, `esbuild`, or `wasm-opt`
  requires reviewing both architecture-specific digests and passing the full
  image build; do not enable Dioxus's runtime tool bootstrap in a derived image.
- Keep the default `same-origin` browser policy for the embedded UI. For one
  separately hosted UI or an HTTPS reverse proxy, set
  `BLOOM_CORS_ALLOW_ORIGIN` to its exact public HTTP(S) origin. Do not use `*`;
  strict security rejects it and the doctor warns when it is otherwise enabled.
- Preserve Bloom's `x-request-id` response header in reverse proxies and support
  bundles. Proxies may supply their own correlation value only within Bloom's
  documented 128-character ASCII alphabet; invalid values are replaced. Never
  use this header as authentication or as a generation cancellation ID.
- Configure reverse proxies and CDNs to preserve Bloom's
  `Cache-Control: no-store` on `/v1/*`, `/api/*`, `/health`, `/ready`, and
  `/metrics`. Do not add cache overrides for inference, authentication errors,
  model-management state, probes, or streaming responses.
- Set `--max-body-bytes`, `--max-concurrent`, `--timeout`, and
  `--shutdown-timeout-seconds`. Keep concurrency far below the reported
  platform semaphore ceiling according to measured model memory and latency;
  Bloom rejects a value above that hard runtime ceiling before startup.
- Set the service manager's termination grace period longer than Bloom's
  shutdown timeout, with enough margin for signal delivery and container
  teardown. Kubernetes `terminationGracePeriodSeconds`, for example, must
  exceed Bloom's configured HTTP drain window.
- Set `--max-upload-bytes` to the smallest image size your deployed models need.
- Leave model downloads disabled for immutable deployments. If enabled, set
  `--max-model-download-bytes`, require API authentication, and budget catalog
  disk space for retained partials and installed files.
- Leave browser imports disabled for immutable deployments. If enabled, set
  `--max-model-import-bytes` and `--max-model-import-chunk-bytes`, require API
  authentication and TLS, restrict CORS, rate-limit chunk requests, and budget
  catalog disk space for retained partials and installed files.
- Set `--allowed-model-licenses` (or `BLOOM_ALLOWED_MODEL_LICENSES`) when the
  deployment has an acquisition policy. Treat declarations as operator-supplied
  metadata: the allowlist controls admission but does not prove the upstream
  repository's legal terms.
- If using signed discovery, distribute `--model-index-public-key` and any
  temporary `--model-index-public-keys` rotation values separately from the
  index, prefer a read-only local index for fixed deployments, and verify the
  documented overlap rotation procedure. Keep the one-to-eight-key trust set
  minimal, confirm the signer `key_id` after rotation, then remove the retired
  key from every instance. Never install a signing seed on a Bloom host. An
  index signature authenticates publisher metadata; it does not establish model
  safety or license accuracy.
- Keep `--model-index-state-dir` on private durable storage and include it in
  backups. The default is `model-index-watermarks` beside the effective config
  file. Do not clear it to work around a rollback error, especially during a
  signing-key incident; investigate the publisher generation first.
- Set `--max-model-storage-bytes` for every writable catalog and keep the value
  below the filesystem or container limit. This shared commitment budget covers
  installed data, staged payloads, declared import remainder, and active
  download remainder.
- Set `--staged-model-retention-seconds` only after choosing an operational
  resume window. Automatic cleanup is disabled at `0`; when enabled it runs at
  startup and periodically, skips active downloads, and can expire paused
  imports.
- Budget signed-model upgrades for the previous and replacement payloads at the
  same time. Do not remove `.bloom-upgrade` manually after an interrupted
  commit: restart Bloom with the same catalog root and let bounded recovery
  restore the previous entry or complete the verified replacement. Treat a
  fail-closed recovery error as catalog corruption requiring backup-led
  operator review, not permission to delete the marker or provenance.
- Assign a distinct `--models-dir` to every concurrently running Bloom server.
  A process-lifetime operating-system lease intentionally rejects a second
  server before it can recover or mutate the same catalog. Do not delete
  `.bloom-catalog.lock`: the empty file is harmless after shutdown, and kernel
  lock ownership—not its presence—identifies a live owner. Verify advisory-lock
  behavior on the exact network or userspace filesystem before deployment;
  local CPU tests cannot establish a remote filesystem's lock guarantees.
- Restrict model removal and staged-acquisition discard endpoints to operators;
  they permanently delete server-side data. This includes
  `DELETE /api/delete`, which has no browser confirmation step.
- Reverify acquired models after transfer or restore and before first load in a
  new environment. Investigate rather than bypass a persistent integrity
  mismatch; replace the file from a trusted source and verify it again.
- Export and version-control `bloom-model-inventory.json` after catalog changes.
  Review untracked models, invalid provenance, quarantine, mutable source refs,
  checksum changes, and license changes as deployment drift.
- Run the authenticated inventory reconciliation preview after restore or
  deployment. Treat blocking drift as a stop condition and remediate through
  normal verified acquisition or removal workflows; reconciliation itself does
  not mutate the catalog. Inventory restore is intentionally per-model: inspect
  the exact source and checksum in the saved inventory before confirmation, and
  compare again after the verified download finishes.
- Review the Models drawer preflight on the deployment host before switching.
  Treat its memory and capability verdict as an early admission check, not as
  real-model validation or a guarantee that an external runtime will start.
  Require the response to retain `schema_version: 1` and the
  `bloom.model_preflight` object identity through any proxy; independently
  deployed UIs reject incompatible documents before enabling `Load`.
- Require authenticated `/v1/model-management/models` responses to retain
  `schema_version: 1` and the `bloom.model_catalog` object identity. The
  response exposes the server's configured catalog root to authorized
  operators; do not publish it through an unauthenticated diagnostics proxy.
  Independently deployed UIs reject incompatible catalog snapshots instead of
  assuming omitted capabilities are disabled.
- Put TLS, rate limiting, and request logging policy in a reverse proxy.
- Keep `/metrics`, `/health`, and `/ready` on an internal network or proxy ACL.
  Require `/ready` to retain its `bloom.readiness` identity and supported
  UI protocol range through the proxy; do not replace it with a generic load
  balancer health document. Independently deployed UIs fail closed unless
  their protocol falls inside the server's positive ordered range.
  Keep authenticated `/v1/observability` access limited to operators who may
  inspect model identifiers, resource sizes, and process-local usage counters.
- Run `bloom_server --doctor` with the deployment's effective environment on
  the target host. Resolve every failure and review warnings before binding the
  service; the check does not load model weights or replace a real-model test.

Verify authentication before deployment:

```bash
BLOOM_MODEL_PATH=/path/to/model.gguf \
./scripts/openai_compat_smoke.py --api-key change-me

BLOOM_MODEL_PATH=/path/to/model.gguf \
./scripts/ollama_compat_smoke.py --api-key change-me --require-model
```

Both protected namespaces must return their documented JSON 401 together with
`WWW-Authenticate: Bearer realm="Bloom"`, `Cache-Control: no-store`, and a
bounded `x-request-id` for missing and invalid credentials. Confirm Bearer and
`X-API-Key` credentials both succeed, CORS exposes `www-authenticate`, and
unknown or method-mismatched public route fallbacks remain 404/405 without an
authentication challenge. In the browser, an invalid saved key must produce
**API key required** in both connection testing and the live status poll.

Verify the browser-origin boundary separately. With the default configuration,
the embedded HTTP origin must succeed, an unrelated or opaque `Origin` and a
loopback DNS-rebinding `Host`/`Origin` pair must receive HTTP 403, and an
origin-free SDK request must still succeed. When an exact standalone UI origin
is configured, its actual request and preflight must receive that exact
`Access-Control-Allow-Origin`; every other origin and preflight must be rejected
before routing without an allow-origin response. Rejected `/v1` and `/api`
requests must retain their protocol JSON, no-store, and correlation headers.

## Memory

- Set `BLOOM_STRICT_MEMORY_BUDGET=1`.
- Tune `--memory-utilization`, context size, and concurrency on the target host.
- Record peak host memory and device memory with the production model.
- Configure operating-system or container memory limits as a final guard.

## Models and plugins

- Use fixed, read-only model directories.
- Point `--models-dir` at the smallest dedicated catalog root. Keep it writable
  only by the deployment pipeline, or by Bloom itself when verified downloads
  are explicitly enabled.
- Include `bloom.json`, model license information, immutable source revisions,
  and SHA-256 hashes. Treat Bloom's acquisition license field as an operator
  declaration and enforce organizational license policy outside the runtime.
- Allowlist every external script, runner, and plugin.
- Expose only catalog IDs to API users; never add an endpoint that accepts an
  arbitrary model path.
- Do not deploy `skeleton` paths as executable backends.
- Confirm preflight selects the intended task, engine, device backend, context
  budget, and maturity for every production catalog entry.
- Prefer exact Hugging Face commit revisions. Inventory `source_locked: false`
  means the recorded source uses a mutable ref or cannot be proven immutable;
  the SHA-256 remains the byte-integrity authority.

## Release gate

- Complete the checks in [RELEASE.md](../RELEASE.md).
- Require a valid `bloom_server --doctor=json` report from the staged native
  binary and verify `embedded_ui` passes for the official application package.
- Run `scripts/test_server_http_boundary.py` against the staged binary. Require
  browser HTML navigation to reach the embedded app shell while unknown
  `/v1/*` and `/api/*` paths, missing assets, JSON clients, and non-GET requests
  retain 404 status and a bounded correlation ID.
- Run the OpenAI compatibility smoke test with authentication enabled and the
  pinned official client. Confirm empty discovery, missing-model retrieval, 401,
  unavailable-model chat, and missing Responses state decode correctly before
  adding the production model. Then confirm exact-ID and `default` model
  retrieval return the same stable identity before exercising streaming
  generation or embeddings.
- Run the Ollama compatibility smoke test with authentication enabled. Confirm
  the official client decodes list, process, and show plus chat/generate/NDJSON
  for a text model or batched normalized embed output for an embedding model
  when that SDK is part of the deployed client set. Confirm the smoke's
  disabled-download pull probe and non-mutating missing-model delete probe
  return authenticated Ollama-shaped 403 and 404 errors. On a host configured
  with a production signed index, pull one small entry by exact index ID in
  streaming and non-streaming mode; verify its source, signed size, SHA-256,
  license provenance, persistent signed-index alias, idempotent replay, and
  exact-work acquisition concurrency join. Confirm pull alone leaves the
  runtime unchanged, then invoke chat/generate or embedding with the same index
  ID and verify exact on-demand activation, stable response identity after
  restart, and same-target load joining. Exercise empty-request preload,
  negative indefinite residency, response-safe `keep_alive: 0`, and a short
  positive duration whose `/api/ps` deadline ends in automatic unload. Then
  verify an isolated inactive fixture can be deleted by its alias while an
  active fixture and unauthenticated request cannot. Use the repository's pinned
  compatibility requirement for the baseline, then repeat with the exact client
  version deployed by the operator.
- Run both OpenAI and Ollama embedding requests against the production
  embedding model. Record native dimensions, projected dimensions, context
  overflow behavior with truncation enabled and disabled, vector norm,
  throughput, peak memory, and client-disconnect drain behavior.
- Run `scripts/openai_compat_smoke.py --embedding-only --require-model` against
  the production embedding model. Confirm stable rerank ties, descending finite
  scores in `[-1, 1]`, `top_n`, returned document identity, unique response IDs,
  tokenizer-backed usage, and bounded drain after disconnect.
- Exercise Responses `text.format` with both `json_object` and `json_schema` in
  streaming and non-streaming modes. Confirm valid output preserves the format
  in response lifecycle objects, invalid non-streaming output returns HTTP 422,
  and an invalid stream terminates with `response.failed` rather than
  `response.completed`.
- Verify an exact active model ID and the `default` alias succeed, a different
  well-formed model ID returns `404 model_not_found`, and successful response
  metadata names the model that actually ran.
- Run a real-model benchmark on the target hardware.
- Confirm startup, readiness, and request cancellation. Send `SIGTERM` on Unix,
  verify readiness is withdrawn and a normally drained server exits with status
  0, then hold an incomplete connection open and verify Bloom exits with status
  1 within the configured shutdown deadline. Repeat with a long deadline and
  send a second signal, which must force status 1 immediately. Exercise
  `Ctrl-C` separately on every deployed platform.
- Disconnect streaming clients before the first token and during generation;
  confirm execution is cancelled, IFB sender and request registrations are
  removed, in-flight metrics return to zero exactly once, and a new request
  cannot acquire the configured concurrency slot before a blocking worker exits.
- Repeat the disconnect and configured HTTP-timeout checks with non-streaming
  chat and legacy completion requests. Confirm scheduled work is cancelled and
  a non-cooperative blocking model retains its permit and in-flight ownership
  until the real worker exits rather than admitting overlapping CPU work.
- Confirm malformed, oversized, and path-like cancellation IDs return HTTP 400
  and cannot reach another authenticated route.
- Confirm every success, authentication failure, body-limit rejection, timeout,
  and unknown route carries one bounded `x-request-id`; missing IDs must receive
  distinct UUIDs, safe proxy values must round-trip, and unsafe or oversized
  values must be replaced. From an allowed browser origin, confirm
  `Access-Control-Expose-Headers` includes `x-request-id`. At debug trace level,
  confirm request and response events share that ID and log the path without the
  query string. Trigger a non-successful UI request and confirm its error banner
  includes the same safe ID; an unsafe or oversized proxy value must be replaced
  by the server and must never be displayed verbatim by the browser.
- Confirm the default origin policy admits the embedded HTTP authority and
  origin-free official clients but returns a fixed 403 for cross-origin,
  opaque, malformed, duplicate, and loopback DNS-rebinding browser requests.
  Configure one exact standalone UI origin and verify only it receives CORS
  admission on actual requests and preflights. `--strict-security` must reject
  an explicit wildcard, and the doctor must warn about a non-strict wildcard.
- Confirm OpenAI- and Ollama-shaped 401 responses carry the fixed Bearer
  challenge, no-store, and a correlation ID; an explicit downstream challenge
  must survive, 403 must remain unchallenged, and CORS must expose
  `www-authenticate`. Verify the UI does not publish ready/loading/model state
  until its bounded protected Models probe succeeds.
- Saturate the configured inference concurrency with a controlled test and
  confirm the next request receives protocol JSON, HTTP 429, `Retry-After: 1`,
  `no-store`, and a correlation ID. Confirm CORS exposes `retry-after`, an
  explicit handler hint is retained, 503 does not gain a synthetic hint, and
  the UI displays only bounded delta-seconds without retrying automatically.
- Confirm successful JSON, SSE, and NDJSON responses plus authentication,
  readiness, timeout, body-limit, and unknown-route errors carry exactly
  `Cache-Control: no-store` throughout the `/v1` and `/api` namespaces and on
  all three probe endpoints. Verify a static UI asset path is not accidentally
  classified as dynamic API state.
- Confirm an unknown `/v1` path returns an OpenAI-shaped JSON 404 and an unknown
  `/api` path returns an Ollama-shaped JSON 404, including when API-key
  protection is enabled. Call known routes with the wrong method and require
  the matching JSON 405 plus the documented `Allow` header. Verify fallback
  messages do not reflect paths, queries, or credentials, known routes still
  return 401 without credentials, and the reverse proxy does not replace these
  responses with redirects or HTML.
- Send authenticated malformed JSON, a valid JSON value with the wrong endpoint
  shape, a request without the required media type, an oversized body, and an
  invalid multipart boundary. Require protocol JSON with the documented
  400/413/415/422 status, `no-store`, and a bounded correlation ID. Include a
  private marker in the rejected input and confirm neither the marker nor raw
  parser diagnostics appear in the response. Verify probes and static assets
  remain untouched.
- Confirm empty chat arrays, unsupported roles, more than 2,048 messages,
  oversized user or system messages, and more than 768 KiB of combined content
  return HTTP 400 before inference admission. Repeat with a raised general JSON
  body limit to verify semantic limits remain active.
- Confirm string content and bounded ordered OpenAI text-part arrays are both
  admitted. Empty, malformed, over-256-part, image, audio, file, and refusal
  arrays must return HTTP 400 before inference admission. Verify
  `max_completion_tokens`, leading `developer` instruction mapping, rejection
  of late developer messages, and conflicting dual token-limit fields.
- Confirm `/v1/responses` accepts bounded string and `input_text` message input,
  returns a response/output/usage envelope accepted by the current OpenAI SDK,
  and reports token-limit termination as `incomplete`. With `stream: true`,
  require named created/progress/item/content/text-delta/done events, monotonic
  sequence numbers, stable metadata, full terminal usage, and a typed completed
  or incomplete terminal event without Chat's `[DONE]`. Disconnect before and
  during output and confirm the underlying generation lifecycle drains and no
  requested state is committed. With explicit `store: true`, retrieve the
  terminal response, page its input items in both orders, chain a turn with
  `previous_response_id`, verify prior top-level instructions are not inherited,
  confirm bounded metadata survives streaming and retrieval but is neither
  added to the prompt nor inherited, and delete both responses. Confirm omitted
  or false `store` values remain
  unretained, deleted/expired/evicted IDs return HTTP 404, a chained request
  remains bound to its original model, and restart loses all retained state.
  Exercise flat Responses function definitions with `auto`, `required`, named,
  and `none` choices; parallel calls enabled and disabled; schema-invalid model
  arguments; native non-streaming call items; native argument-delta/done stream
  events; and a `previous_response_id` continuation containing matching
  `function_call_output` items. Confirm retained input-item pages include both
  calls and results, tool definitions are resent rather than inherited, and the
  server never executes a function. Verify custom or built-in tools, image/file
  function outputs, background execution, automatic truncation, and unsupported
  content parts return HTTP 400 before inference.
- Send the neutral extension defaults documented in
  [OpenAI API compatibility](openai-compatibility.md) and confirm admission
  continues normally. Exercise Chat function tools with `auto`, `required`,
  named, and `none` choices; parallel calling disabled and enabled; valid and
  schema-invalid arguments; buffered streaming `delta.tool_calls`; and a full
  assistant-call, external-result, final-message continuation. Confirm Bloom
  never executes the function and never exposes its private control envelope.
  Then activate a custom tool, deprecated `functions`, `n`, penalties, log
  probabilities, an unknown field, a malformed or unpaired message
  `tool_calls`, and an unknown `stream_options` field. Each unsupported or
  malformed case must return a bounded HTTP 400 `invalid_request_error` before
  request counters or concurrency permits change; long or control-bearing field
  names must not be reflected verbatim. A model-generated invalid tool control
  object must fail with `invalid_tool_call`, never a partially trusted call.
- Exercise Chat, legacy Completion, `/api/chat`, and `/api/generate` with one
  stop marker split across model output chunks. Confirm neither the marker nor
  later text reaches buffered or streaming clients, the finish reason is
  `stop`, blocking or scheduled execution ends early, lifecycle metrics settle
  successfully exactly once, and an unmatched partial suffix is flushed on
  natural termination. Reject empty, non-string, over-four,
  over-1,024-character, and over-16-KiB controls before inference. Confirm the
  browser persists valid JSON-array settings and rejects active stop sequences
  for image chat before conversation mutation.
- Confirm nonblank/count/byte budgets for completion, embedding, and reranking
  return HTTP 400 before model readiness and leave inference metrics and permits
  unchanged. Repeat with a raised general JSON body limit.
- Confirm `/v1/multimodal/stream` rejects duplicate or internal block variants,
  non-finite/out-of-range/over-duration PCM, invalid or oversized images, and
  every `AudioFile` value without opening or reflecting the submitted server
  path. Verify only bounded inline `Text`, `AudioPcm`, and JPEG/PNG `Image`
  combinations reach modality admission.
- Confirm the browser rejects invalid connection settings and persisted
  generation settings, encoded chat JSON above 1 MiB, multimodal prompts above
  1 MiB, and JPEG/PNG attachments above 10 MiB without sending or truncating the
  request. Verify initial-send rejection preserves the draft and attachment,
  retry rejection preserves the prior response, and edit rejection keeps the
  dialog content open with an inline error before any reactive history change.
- Confirm the browser rejects a non-SSE success response, an SSE frame or
  response beyond its documented byte budget, and EOF without `[DONE]`; verify
  that bounded partial output is retained as failed rather than completed.
- Confirm ordinary browser API reads reject invalid, declared-oversized, and
  incrementally oversized bodies. Test success, error, missing `Content-Length`,
  multibyte UTF-8, and decoding-failure paths through the deployment proxy.
- Open every drawer and dialog from the keyboard. Confirm initial focus is
  inside the surface, Tab and Shift+Tab cannot escape it, Escape closes it,
  focus returns to the invoking control, and screen readers announce the
  expected name and description. Repeat with reduced motion enabled in each
  supported target browser.
- Branch from an earlier user message and assistant response. Confirm each new
  conversation stops at the selected message, receives a unique title, survives
  reload, and leaves its source unchanged. Repeat with browser storage denied or
  quota-exhausted and verify new-chat creation, selection, deletion, branching,
  and send admission leave the current visible conversation and draft intact.
- Import a conversation containing more than 300 messages. Confirm only the
  latest 100 mount initially, each **Show earlier** action reveals at most 100
  more, absolute copy and branch targets remain correct, and generation
  admission still evaluates the complete history rather than the visible page.
- Import a valid archive with both **Merge** and **Replace all**. Confirm Merge
  retains current conversations and selection, appends archive-order copies with
  fresh IDs, and preserves exact duplicates; confirm Replace all restores only
  the archive. Repeat Merge at the combined conversation/message limits, with
  exhausted ID space, denied storage, and recovery-locked unreadable history.
  Every failure must keep visible and persisted history unchanged, and the
  recovery-locked dialog must disable Merge while retaining explicit Replace
  all and Cancel choices.
- Confirm typed success responses preserve an accepted JSON media type through
  the proxy. Exercise response-header, body-idle, and total-body timeouts, plus
  UI task disposal, and verify the upstream request or reader is cancelled.
- Confirm a newly saved API key is absent from the persistent connection record,
  survives a same-tab reload through session storage, and becomes persistent
  only after **Remember API key in this browser** is selected. Disable the option
  again and verify the persistent record is scrubbed. Protect both storage modes
  from untrusted same-origin scripts and extensions.
- Document supported model hashes, runtime versions, and known limits.
