# Security Boundaries

Bloom loads model files and can launch external runtimes or plugins. Treat all
three as executable supply-chain inputs.

For vulnerability reporting, see [SECURITY.md](../SECURITY.md).

## Model files

- Load models only from controlled directories.
- Do not build model paths from untrusted request input.
- Configure a dedicated `--models-dir`; the model-management API accepts only
  discovered direct-child IDs and rejects traversal and escaping symlinks.
- Do not load files while they are still being downloaded.
- Keep model downloads disabled unless the catalog is intentionally writable.
  The built-in downloader accepts only HTTPS URLs on Hugging Face and its CDN
  hosts, rejects URL credentials and untrusted redirects, requires SHA-256,
  stages partial files in a private directory, and installs with a no-overwrite
  filesystem operation.
- Source inspection is authenticated, accepts only public Hugging Face
  repository file URLs, strips query strings and fragments, uses bounded HEAD
  requests and trusted redirects, and pins the returned URL to published commit
  metadata. A discovered digest is accepted as SHA-256 only when it is exactly
  64 hexadecimal characters; missing metadata never enables an unverified
  download.
- Signed discovery indexes are optional and authenticated. Bloom accepts one
  regular non-symlink file or constrained HTTPS source, bounds reads and
  redirects, verifies a domain-separated Ed25519 signature with a pinned
  non-weak public key, rejects expired payloads, unknown fields, duplicate
  destinations, and mutable model revisions, and never returns an expired
  cache. A valid signature proves only that the configured publisher signed the
  metadata. It does not prove model safety, checksum truth, license truth, or
  permission to redistribute; complete downloaded bytes still require SHA-256.
- Treat each model-index signing seed as an offline release secret. Bloom
  servers need only one to eight bounded public keys. Rotation uses a temporary
  old/new overlap and a strictly newer signed generation; unknown signers,
  older generations, and conflicting equal-time generations fail closed across
  restarts. Bloom atomically publishes bounded, source-scoped watermark records
  before exposing a new generation. There is no remote revocation list, so
  suspected compromise still requires immediate key removal or index
  disablement, short validity windows, and review of installed provenance.
- Protect and back up the private model-index state directory. Corrupt,
  unexpected, symlinked, or unwritable state fails closed. A local actor that
  can delete both this directory and process memory can reset the watermark;
  signed-index rollback protection is aimed at a compromised or stale remote
  publisher path, not a privileged local-filesystem attacker. On Unix, the
  directory is created with mode `0700` and group/other-writable state is
  rejected.
- Same-ID signed upgrades are replacement transactions, not in-place writes.
  Bloom retains and rehashes the previous payload and provenance, verifies the
  complete replacement, and records bounded rollback state under the private
  `.bloom-upgrade` directory before renaming either side. Startup recovery runs
  before configured model-path validation, stale-stage cleanup, or runtime
  loading. Malformed, oversized, symlinked, unexpected, or ambiguous
  transaction state fails startup closed rather than guessing which model is
  authoritative. Protect this directory under the same administrative boundary
  as model payloads and `.bloom-metadata`.
- Treat one model catalog as one mutable `bloom_server` ownership domain. The
  server takes a non-blocking exclusive operating-system lease on the empty
  `.bloom-catalog.lock` regular file before recovery or cleanup and retains it
  through runtime teardown. A persistent file after exit is not a stale lock;
  kernel ownership is released automatically. Unix rejects a symbolic-link or
  group/other-writable lock path. This coordinates cooperating Bloom processes,
  not a privileged actor able to unlink files or a network filesystem that does
  not honor advisory locks; validate the deployment filesystem and never run an
  older lock-unaware Bloom against the same writable root.
- Resume and discard operations accept only a validated single-file name and
  reject a symlinked staging directory or entry. Stored download URLs are not
  returned by the API.
- Set `BLOOM_ALLOWED_MODEL_LICENSES` on governed writable catalogs. A non-empty
  comma-separated allowlist makes a matching license declaration mandatory for
  new downloads, download resumes, import declarations, and final import
  publication. Matching is exact and case-insensitive; Bloom does not interpret
  SPDX expression semantics or independently verify a repository's license.
- Keep browser model imports disabled unless authenticated users are allowed to
  write to the catalog. Imports require a declared size and SHA-256, enforce
  per-file and per-request limits, require exact append offsets, and publish
  only after complete-file verification. Staging directories and entries reject
  symlinks, and installation never overwrites an existing catalog entry.
- A browser-local import is still a network upload to `bloom_server`. If the UI
  and server run on different hosts, model bytes cross that network boundary;
  use TLS, a narrow CORS origin, authentication, rate limiting, and disk quotas.
  The expected checksum proves integrity, not that a model is trustworthy.
- Successful downloads and imports persist acquisition records under the
  private `.bloom-metadata` directory. Source URLs must use HTTPS and are stored
  without query strings or fragments. License or SPDX values are user-declared
  metadata, not proof of ownership, permission, or policy compliance.
- Signed upgrade admission requires the source to be inactive and serializes
  against load, removal, and integrity work. It reserves the complete previous
  and replacement payloads until commit, so filesystem free-space and
  application quota must cover the peak even though the old model is retired
  afterward.
- Re-run model integrity verification after copying catalogs, restoring
  backups, or changing filesystem ownership. Bloom hashes only inactive
  acquired single-file models, detects path/identity changes during the scan,
  and persists mismatches. A quarantined entry cannot be loaded until its bytes
  match the acquisition checksum again. Keep the provenance directory and
  model payload under the same trusted administrative boundary.
- Model preflight parses model-controlled headers or manifests on a bounded
  blocking worker and returns only selected summary fields. It rejects path
  traversal and escaping symlinks, serializes inspections, and bounds its
  cache. Successful responses use the strict version 1
  `bloom.model_preflight` contract; the server validates its bounded output and
  the browser rejects missing, unknown, or unsupported-version documents. This
  is not a malware scanner or sandbox. Keep untrusted model payloads outside
  the catalog until they have passed the deployment's supply chain controls.
- GGUF `tokenizer.chat_template` content remains model-controlled, inert data.
  Bloom reads at most the documented bound, classifies only known token
  contracts, and selects a hard-coded formatter; it never evaluates embedded
  Jinja or other template code. Preserve that boundary when adding formats or
  templates.
- Hugging Face `tokenizer_config.json` is also model-controlled, inert data.
  Bloom rejects oversized or non-regular metadata, parses it within a fixed
  bound, stores only a recognized hard-coded chat-template kind, and never logs
  or evaluates the supplied template source.
- Model inventory export is authenticated and intentionally omits the catalog
  root, absolute paths, transient runtime state, URL query strings, fragments,
  credentials, and invalid provenance details. It still exposes model IDs,
  source repositories, checksums, license declarations, and install times;
  handle the file as deployment metadata rather than publishing it blindly.
- The live model catalog is authenticated, non-cacheable, bounded, and uses the
  strict version 1 `bloom.model_catalog` contract. It intentionally includes
  the configured catalog root for the local operator UI, plus model IDs,
  acquisition status, storage accounting, and sanitized provenance. Treat it
  as private deployment metadata. It omits resolved per-model paths, metadata
  record paths, credentials, index keys and sources, and invalid provenance
  details; use the inventory export when a path-free portable artifact is
  required.
- Inventory reconciliation is authenticated, limited to 16 MiB and 20,000
  entries, rejects unknown fields and inconsistent provenance, and returns
  changed field names rather than uploaded values. It is a read-only preview:
  uploaded inventory data never triggers installation, removal, loading, or
  provenance writes.
- Inventory restore is a separate authenticated mutation requiring a model ID
  and the complete validated inventory on every request. It accepts only a
  currently missing single-file download pinned to an exact Hugging Face commit,
  checks the expected size against the configured limit, and then reuses the
  trusted-host, SHA-256-verified, quota-aware, no-overwrite download pipeline.
  It never performs bulk restore or automatically loads the result.
- Model removal through either `/v1/model-management/remove` or
  `DELETE /api/delete` accepts only a bounded, resolved direct-child catalog
  ID, refuses the active model and lifecycle races, and does not follow nested
  symlinks. It is a permanent operation and should be limited to trusted
  administrators.
- Model discovery and `GET /v1/models/{model}` expose only the active runtime's
  bounded public ID, process-local publication time, object type, and Bloom
  ownership label. Retrieval rejects unknown query parameters, never returns a
  source path, never exposes an inactive catalog entry, and cannot trigger a
  model load. Keep both routes under the shared `/v1` API-key boundary.
- Declare SHA-256 hashes in `bloom.json` for released model packages.
- Do not set `BLOOM_ALLOW_HASH_MISMATCH=1` in production.

Bloom verifies every declared `files[].hash_sha256` before loading. Files without
a declared hash are not verified.

Generation stop sequences are untrusted request data. Bloom limits them to four
non-empty strings, 1,024 characters each, and 16 KiB combined before runtime
admission. Streaming retains at most a possible matching suffix and never emits
a confirmed marker or later model output. A stop match ends model or scheduler
work but does not cancel the authenticated request lifecycle; external client
cancellation remains a distinct failed outcome. Operators should still avoid
placing secrets in stop strings because direct API requests may be logged by a
reverse proxy outside Bloom.

## External runtimes

The following paths can execute external code:

| Path | Trigger |
| --- | --- |
| OpenVINO export | `BLOOM_OPENVINO_AUTO_EXPORT=1` or `BLOOM_NPU_AUTO_EXPORT=1` |
| FunASR or Qwen ASR | `--backend funasr` |
| Intel NPU tooling | `--backend intel-npu` |
| TTS bridge | `--backend npu-tts` |
| LongCat runner | `BLOOM_LONGCAT_RUNNER` or `BLOOM_MNN_DIFFUSION_DEMO` |
| llama.cpp | `BLOOM_LLAMA_CPP_SERVER` or a discovered `llama-server` |

Pin executable paths, runtime versions, and Python environments. Do not point
these settings at user-uploaded scripts or binaries.

## Plugins

A plugin manifest is a capability declaration, not a sandbox.

- Native plugins run in the Bloom process with its permissions.
- Subprocess plugins inherit an environment and filesystem context unless the
  deployment restricts them.
- Remote plugins can send model input over the network.
- WASM manifests are validated, but a sandboxed execution runtime is not yet a
  default capability.

Production deployments should load plugins from an allowlist and keep native
libraries outside user-writable directories.

## Strict mode

Enable strict security checks with:

```bash
BLOOM_STRICT_SECURITY=1 bloom_server --model /models/example
```

Custom external components must then be listed explicitly:

| Variable | Value |
| --- | --- |
| `BLOOM_ALLOWED_SCRIPTS` | Comma-separated script paths or names |
| `BLOOM_ALLOWED_RUNNERS` | Comma-separated executable paths or names |
| `BLOOM_ALLOWED_PLUGINS` | Comma-separated plugin manifest names |

Strict mode reduces accidental execution but does not sandbox an allowed
component.

## Network and data

- `bloom_server --doctor` is side-effect-free and does not serialize either API key,
  absolute model paths, source URLs, prompts, responses, or raw loader errors.
  Its engine names, device names, model counts, and deployment warnings are
  still operational metadata; review JSON reports before sharing them.
- Keep `/metrics` and health endpoints behind an internal network or proxy ACL.
- Bloom rejects a non-loopback listener without an API credential before
  storage mutation or socket binding. Strict mode also requires different
  inference and operator keys. The explicit
  `BLOOM_ALLOW_UNAUTHENTICATED_NETWORK` development override degrades that
  failure to a warning only outside strict mode; never enable it on a shared or
  untrusted network.
- Keep the default `same-origin` browser policy for the embedded UI. Bloom
  rejects every malformed, opaque, duplicate, untrusted, or loopback
  DNS-rebinding `Origin` before CORS, preflight handling, authentication, or a
  route can mutate state. Origin-free SDK and CLI requests remain allowed. For
  a separately hosted UI or HTTPS-terminating reverse proxy, configure its one
  exact public HTTP(S) origin; Bloom intentionally does not trust forwarded
  scheme or host headers. Treat `*` as an explicit development exception: the
  doctor warns on every listener and strict security rejects it. Origin checks
  complement API keys and TLS; they do not protect against browser extensions,
  a compromised allowed origin, or non-browser clients.
- Treat the active model ID as execution metadata, not a cosmetic response
  field. Bloom accepts an omitted selector or the `default` alias, but any other
  explicit selector must exactly match the active runtime before inference is
  admitted. Do not remove this check or echo an unvalidated requested model in
  a response.
- Treat streamed request IDs as cancellation capabilities. The browser accepts
  only a bounded ASCII path-segment alphabet, rejects missing or changing IDs
  before displaying stream content, then revalidates and URI-encodes the value
  before an authenticated cancellation POST. The server applies the same
  contract after path decoding. Never interpolate an unchecked response ID into
  a URL.
- Do not confuse the HTTP `x-request-id` correlation header with a streamed
  generation ID. Bloom bounds incoming correlation values to 128 ASCII letters,
  digits, hyphens, underscores, dots, or colons and replaces anything else with
  a UUID before tracing. The value is attacker-controlled when accepted, may
  repeat, and proves neither identity nor authorization. Tracing records the
  normalized ID, method, and URL path but deliberately omits the query string;
  keep downstream proxy logs equally conservative. The UI independently
  validates and bounds the response header before adding it to an HTTP error
  banner, and silently omits invalid values rather than reflecting them.
- Treat `Retry-After` as untrusted display guidance, not permission to replay a
  request. Bloom emits a default one-second hint for capacity 429 responses and
  exposes it through CORS. The UI accepts only strict 1-to-300-second decimal
  values on 429, ignores every other form or status, and does not automatically
  resend prompts, credentials, uploads, downloads, or lifecycle operations.
- Preserve Bloom's fixed `WWW-Authenticate: Bearer realm="Bloom"` challenge on
  protected 401 responses. The challenge advertises one supported HTTP scheme;
  it is not proof that a submitted credential was close, valid elsewhere, or
  safe to persist. Bloom retains handler-specific challenges, leaves 403 and
  public 404/405 fallbacks unchanged, and exposes the header through CORS. The
  UI follows a validated public readiness request with a bounded protected
  Models probe, classifies only HTTP 401 as an authentication failure, and never
  copies the key into error text, diagnostics, or conversation storage.
- Preserve `Cache-Control: no-store` across every `/v1` and `/api` response and
  the health, readiness, and metrics probes. Bloom applies it outside normal
  routing so successful inference, streams, credential errors, body limits,
  timeouts, and unknown protocol paths share one policy. A reverse proxy must
  not replace it with a public or heuristic cache rule; prompts, generated
  output, model state, and operational counters can all be sensitive or stale.
- Keep namespace fallbacks fixed and non-reflective. Bloom's public 404 and 405
  handlers disclose only the protocol family and method mismatch, return no
  model or catalog state, and never include a request path, query, credential,
  or downstream loader error. They intentionally remain outside route-level
  API-key checks so missing paths are not misreported as authentication
  failures; every recognized `/v1` and `/api` handler remains inside the shared
  authentication layer. Preserve the status, JSON envelope, `Allow`,
  `x-request-id`, and no-store policy at the proxy boundary.
- Treat framework rejection text as untrusted diagnostic output. Bloom's outer
  protocol normalizer replaces non-JSON `/v1` and `/api` error bodies from JSON
  parsing, body buffering, multipart extraction, media-type admission, and
  timeouts with fixed bounded messages while retaining status and safe headers.
  It never buffers the rejected body or reflects parser details and submitted
  content. Existing protocol JSON and streaming media types pass unchanged;
  probes and static UI paths remain outside the classifier. Keep this layer
  outside request limits and timeouts but inside authoritative correlation and
  no-store middleware.
- Preserve generation lifecycle ownership through streaming response bodies
  and non-streaming handler futures. Bloom holds the concurrency permit and
  cancellation registration until terminal completion or confirmed worker
  exit; an early disconnect or HTTP timeout cancels execution, removes
  scheduler senders, and settles metrics exactly once. A blocking worker that
  cannot stop mid-forward retains admission until it exits. Returning these
  resources earlier would let abandoned generations evade CPU admission limits
  and model-switch draining.
- Treat bounded process shutdown as an availability boundary. On a shutdown
  signal Bloom withdraws readiness before draining HTTP requests, then exits
  with status 1 if the configured deadline expires. Set the orchestrator's
  termination grace period longer than Bloom's drain window. A forced exit has
  process-loss semantics and may interrupt background acquisition work; Bloom's
  hidden staging and startup recovery keep partial model data out of the live
  catalog, but operators must still review recovery and integrity state after
  restart. A repeated shutdown signal deliberately forces the same non-zero
  process-loss path immediately, so restrict service-control access to trusted
  operators.
- Treat `/ready` as a public compatibility and deployment-metadata boundary,
  not as authentication. Its versioned identity prevents the UI from accepting
  a generic or legacy health response, while its validated inclusive protocol
  range prevents independently deployed UI/server versions from assuming
  compatibility. The fields still reveal bounded model identity, package
  version, load state, admission counts, and memory pressure. Restrict the probe
  at the network edge when that metadata is not intended to be public, and
  never route a proxy-generated health document to the browser as Bloom
  readiness.
- `/v1/observability` is authenticated and returns `no-store`, but its model
  identifiers, resource sizes, and process-local counters remain deployment
  metadata. Review diagnostics exports before sharing them; they intentionally
  exclude connection credentials, prompts, model paths, and raw loader errors.
- Use a fixed remote-plugin endpoint with authentication, timeouts, and request
  size limits.
- Do not log sensitive prompts, images, audio, tokens, or private model paths.
- Verify whether vendor runtimes collect telemetry before production use.
- Use pinned offline model directories when network access is not required.
- Treat URL query strings as secrets. Bloom does not expose download URLs in
  API status or errors, but resumable download metadata retains the request URL
  locally until installation, checksum failure, or explicit discard. The final
  provenance record and catalog response omit its query string and fragment.
- Partial imports and their checksum metadata remain in `.bloom-imports` until
  installation or explicit discard. Apply retention and filesystem monitoring
  appropriate to the model catalog.
- Set `BLOOM_MAX_MODEL_STORAGE_BYTES` for writable catalogs. The shared quota
  includes installed/staged payloads and outstanding acquisition commitments,
  so concurrent download and import paths cannot each consume the same free
  budget. This is an application quota; retain operating-system filesystem and
  container limits as the final availability boundary.
- `BLOOM_STAGED_MODEL_RETENTION_SECONDS` enables startup and periodic removal
  of inactive staging sessions. It is disabled by default. Choose a value long
  enough for expected transfers; active downloads are protected, while paused
  imports become eligible after their files remain unchanged for the complete
  retention interval.

## Browser UI

- Treat assistant output as untrusted model-generated text. Bloom's UI parses a
  constrained Markdown subset, escapes raw HTML, removes every Markdown image,
  and creates links only for `http`, `https`, `mailto`, and same-page fragment
  destinations. User and system messages are always plain text.
- The renderer's HTML insertion boundary must receive only its constrained
  output. Do not pass API response text, conversation storage, tool output, or
  plugin output directly to Dioxus `dangerous_inner_html`.
- Embedded UI responses set a Content Security Policy plus clickjacking,
  MIME-sniffing, referrer, and browser-permission protections. A standalone
  static host must configure equivalent response headers; the Rust server
  cannot protect assets served by another origin.
- Keep SPA history fallback content-negotiated and outside protocol namespaces.
  Bloom serves the embedded app shell only to extensionless `GET` navigation
  requests that explicitly accept HTML; `/v1/*`, `/api/*`, probe namespaces,
  missing assets, non-browser media types, and unsafe methods stay 404. A global
  index fallback can turn misspelled API calls into false HTTP 200 responses.
- The configured API base URL is browser-local. A new connection uses per-tab
  `sessionStorage` for its API key by default and enters persistent
  `localStorage` only after the user selects **Remember API key in this
  browser**. Existing locally persisted
  keys remain explicitly marked as remembered for compatibility. Disabling the
  option removes the key from the persistent connection record. Neither storage
  mode protects a key from script executing in the same origin, browser
  extensions, or a compromised profile; use TLS for non-loopback servers, serve
  a restrictive Content Security Policy, and protect origin storage as
  credential-bearing state.
- Treat decoded inference payloads as a capability boundary, not merely valid
  JSON. The server bounds completion prompts, embedding batches, reranking
  inputs, generated tokens, and public inline multimodal blocks before runtime
  readiness checks, permit acquisition, or request accounting. Public
  multimodal JSON accepts only one each of bounded `Text`, normalized
  `AudioPcm`, and signature-checked JPEG/PNG `Image`; it rejects `AudioFile` so
  a remote request cannot name a path in the server's filesystem, and rejects
  internal tensor, token, video, world-state, and action variants. Retain these
  semantic checks even when raising the transport body limit.
- Treat embedding output as untrusted model data. The shared executor requires
  finite, nonempty, stable-dimension vectors, caps native width at 16,384 and a
  batch at 1,048,576 values, and rejects zero-norm output before normalized
  projection. Blocking embedding work retains its concurrency permit and
  exactly-once accounting after client disconnect until the worker exits.
  Rerank normalization, finite score validation, range clamping, `top_n`, and
  deterministic tie ordering execute inside that same worker boundary; never
  restore a per-document task loop or silently truncate mismatched dimensions.
- Treat `/api/*` as an authenticated compatibility view over the same Bloom
  runtime, not as an unauthenticated Ollama daemon. Unknown and non-neutral
  Ollama fields fail closed, tool calls remain untrusted model output, and the
  server never executes a client-declared function. Configure a distinct
  operator key so inference clients cannot mutate the catalog or model
  lifecycle. `/api/tags` can reveal
  inactive catalog identifiers; restrict it to the same principals as `/v1`.
  `DELETE /api/delete` is intentionally destructive and delegates to Bloom's
  guarded removal operation; authorize it only for operators. `/api/pull`
  accepts only an exact operator-signed index ID and delegates to Bloom's
  verified downloader, including immutable-source, signed-size, checksum,
  license, quota, resumability, provenance, and no-overwrite controls. It must
  never accept a client URL, registry fallback, insecure transport, or silent
  policy downgrade. The signed ID persisted in provenance is a public selector,
  not a path; reject duplicate aliases and never derive a destination from it.
  Operator-scoped chat, generate, and embedding requests may activate only an
  exact contained catalog ID or that validated persisted alias, after the
  shared integrity and preflight gates. Inference-scoped requests must use the
  already active runtime and cannot set `keep_alive`. Same-target concurrent
  requests may join only their own
  sequenced terminal channel; never infer completion from mutable global load
  status or let a different selector join. Empty lifecycle requests may preload
  or synchronously unload the exact active runtime. Timed expiry must remain
  bounded, cancelable by newer activity, tied to the exact runtime instance,
  and deferred until its response lease is released; `keep_alive: 0` must never
  unload mid-stream. Keep create, copy, push, registry resolution, and
  client-controlled sources outside this adapter.
- Treat OpenAI-compatible extension fields as request semantics, not disposable
  metadata. Responses, chat, message, stream-option, response-format, and legacy
  completion decoders retain them; only JSON `null` and documented exact no-op
  defaults pass admission. Non-neutral or unknown fields fail before runtime
  accounting, and error text reports only a bounded number of sanitized field
  names. Do not silently drop tool calls, response state, stop sequences,
  output-count controls, or sampling behavior. Chat function definitions,
  choices, call counts, IDs, history pairing, and generated arguments cross a
  bounded fail-closed validator; the server exposes a call only after its
  arguments satisfy the declared schema and never executes it. The Responses
  adapter translates flat function declarations into that same boundary,
  admits only bounded native call/result items, requires exact outstanding-call
  pairing, and retains calls/results only under explicit response storage.
  Treat returned tool arguments and externally supplied tool results as
  untrusted data despite structural validation. The Responses adapter places
  `instructions` in the leading
  developer/system instruction position. Its `text.format` decoder accepts only
  the documented bounded JSON object and JSON Schema subset, rejects unknown
  active configuration before inference, and revalidates accumulated structured
  stream output before publishing success. Its streaming translator requires an
  internal SSE media type, bounded valid UTF-8/JSON frames, stable ID/model/time
  metadata, ordered finish and usage records, a bounded event/output count, and
  the internal terminal marker. The outer stream owns the inner stream so a
  disconnected client still triggers the established cancellation lifecycle.
- Treat `store: true` as explicit authorization to retain sensitive prompt and
  output text in Bloom's process memory. Omitted and false values retain
  nothing. The store is capped at 256 records, 64 MiB total, 40 MiB per record,
  and 24 hours, uses oldest-first eviction, never writes response state to disk,
  and is cleared by restart. Retrieval, input-item listing, and deletion remain
  under `/v1` authentication and return `Cache-Control: no-store`; they do not
  protect memory from a privileged debugger, core dump, or host compromise.
  Delete state as soon as continuation is no longer needed, disable or encrypt
  process dumps where prompts are sensitive, and do not assume the 24-hour
  ceiling guarantees availability because capacity eviction may occur earlier.
  Responses metadata is also client-controlled retained data. Bloom limits it
  to 16 control-free string pairs with bounded keys and values, does not log or
  inject it into prompts, and does not inherit it across chained responses;
  avoid putting credentials or secrets in it nevertheless.
- Treat outbound request construction as an availability boundary. The browser
  validates bounded HTTP(S) connection settings, rejects invalid persisted
  generation settings, limits chat to 2,048 messages and 768 KiB of content,
  limits encoded chat JSON and formatted multimodal prompts to 1 MiB, and admits
  one JPEG or PNG of at most 10 MiB. The server independently validates chat
  roles and semantic message limits even if its general JSON body limit is
  raised. Its OpenAI text-part normalizer admits one to 256 text parts per
  message, accounts concatenated bytes before allocating beyond the shared
  budget, and rejects non-text semantics. Do not replace these failures with
  silent history truncation.
- Treat the configured generation endpoint as an untrusted streaming peer. The
  browser requires the SSE media type, limits total decoded transport bytes,
  individual frame bytes, error text, and accumulated output before updating
  UI state, and requires `[DONE]` rather than accepting EOF as success. Keep
  these limits when adding compatible providers; token limits alone do not
  bound malformed transport framing or proxy-generated response bodies.
- Treat every ordinary API response as untrusted transport too. The browser
  incrementally reads at most 16 MiB for a successful body, applies smaller
  endpoint limits where defined, and limits error bodies to 64 KiB. It rejects
  invalid or oversized `Content-Length` values, counts both streamed bytes and
  UTF-8-decoded text when the header is absent or inaccurate, cancels failed
  readers, and exposes at most 4 KiB of whitespace-normalized error detail.
- Successful typed responses must declare `application/json` or an
  `application/*+json` media type. Ordinary control requests abort if response
  headers take more than 120 seconds; response bodies fail after 30 idle seconds
  or 300 total seconds. Dropping the corresponding UI future aborts the pending
  fetch or cancels its body reader. Keep explicit user cancellation for
  generation and model-import transfers instead of applying control-operation
  timing assumptions to CPU inference or large uploads.
- The diagnostics client bounds and validates the versioned server snapshot
  before rendering it. Its export serializes only typed runtime fields and does
  not include the connection configuration or conversation state.
- Conversation exports omit connection and generation settings but contain
  prompt and response text. Treat them as private user data. Import is bounded,
  schema-checked, and confirmed before replacement; imported assistant content
  remains subject to the constrained Markdown or escaped structured-code renderer.
- Browser-local assistant messages can include a bounded confirmed execution
  model, token counts, request time, first-token latency, and completion outcome.
  The stream client rejects a model that differs from the request or changes
  during one response. Portable version 2 exports retain the bounded assistant
  model identity, but omit timing, token usage, request IDs, and generation
  settings. Treat the archive, browser profile, and raw recovery copy as private
  usage data.
- JSON object or JSON Schema validation constrains output shape, not meaning or
  authority. Unknown schema constraints fail admission instead of being ignored,
  and structured UI output is escaped rather than interpreted as Markdown.
  Consumers must still validate any value used in commands, paths, queries, or
  external requests.
- Malformed browser-local conversation state is retained verbatim and blocks
  conversation writes. Its recovery download can contain prompts, responses,
  attachment filenames, and other user-entered secrets; protect it like the
  conversation archive. Overwriting requires a confirmed fresh start or a
  validated archive import.
- Message copy is an explicit user action and writes original plain text rather
  than rendered HTML. Clipboard availability and permission are enforced by the
  browser; deployments should use HTTPS outside localhost.
- Response regeneration and latest-prompt editing replay stored text with the
  current model and settings. Bloom refuses either operation when the current
  user request included an image because the original bytes are unavailable; it
  does not silently replace multimodal input with its display marker. If the
  current model differs from the conversation's latest recorded response model,
  replay remains blocked until the exact transition is explicitly acknowledged.

## Memory safety at startup

Set `BLOOM_STRICT_MEMORY_BUDGET=1` or pass `--strict-memory-budget` to fail
before loading when the estimated model footprint exceeds available memory.
This is an availability control, not a substitute for process isolation or
operating-system resource limits.

Bloom also validates `max_concurrent` before creating its admission semaphore.
Zero and values above Tokio's target-specific permit capacity fail with an
actionable configuration error rather than a process panic. This hard ceiling
only protects runtime construction; operators must choose a much smaller value
that fits the selected model, context, latency objective, and host memory.
