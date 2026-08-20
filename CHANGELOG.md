# Changelog

All notable changes to Bloom should be recorded here.

This project follows semantic versioning once public APIs and manifest schemas
are declared stable. Before `1.0`, breaking changes are allowed but must be
called out in release notes.

## Unreleased

### Breaking

- Add `BLOOM_OPERATOR_API_KEY`/`--operator-api-key` and enforce a separate
  operator scope for `/v1/model-management/*`, Ollama pull/delete, inactive
  model activation, empty-prompt load/unload, and explicit `keep_alive`.
  Operator credentials remain valid for inference; inference credentials now
  receive protocol-shaped HTTP 403 responses for control-plane operations when
  both keys are configured. Omitting the operator key retains legacy single-key
  behavior, while strict non-loopback deployments require two different keys.
- Reject non-loopback `bloom_server` listeners without an API credential by default.
  Isolated development environments can opt back into the old behavior only
  with `--allow-unauthenticated-network` or
  `BLOOM_ALLOW_UNAUTHENTICATED_NETWORK`; strict security rejects that override.
- Change the Docker runtime to fixed unprivileged UID/GID `10001`, move its
  mutable state to `/var/lib/bloom`, and enable strict security and memory
  admission by default. Existing bind mounts must be writable by `10001:10001`,
  and public container listeners in strict mode now require distinct inference
  and operator API keys. The Dockerfile
  frontend and both tagged base images are digest-pinned, with a regression
  validator for immutable bases and hardened runtime settings.
- Change the application-layer `bloom_server::run_cli` bootstrap from async to
  synchronous and unsafe so it can freeze process-wide engine settings before
  creating Tokio worker threads. Embedders must call it directly from their
  single-threaded process entry point, uphold its documented environment safety
  contract, and no longer await it inside an existing runtime.
- Version successful `GET /v1/model-management/models` responses as strict
  schema version 1 `bloom.model_catalog` documents. Bloom UI now requires the
  complete lifecycle, acquisition, index, storage, and integrity contract;
  rejects unknown fields and unsupported versions; and validates cross-field
  invariants before updating the Models drawer. Catalog scans also reject
  unsafe IDs and bound inspected and published direct children. A current UI
  therefore requires a server that publishes the new identity fields.
- Version successful `POST /v1/model-management/preflight` responses as strict
  schema version 1 `bloom.model_preflight` documents. The browser now requires
  every published field, rejects unknown fields and unsupported versions, and
  verifies the bounded load-decision contract before enabling `Load`. Custom
  clients must check the new identity fields when upgrading.
- Advance conversation archive exports to version 2. Assistant messages now
  preserve a bounded exact `model` identity so imported history retains
  cross-model continuation safeguards. Imports remain backward compatible with
  strict version 1 archives, which have unknown model provenance.
- Advance the public `/ready` schema and UI/server protocol to v3. Servers now
  publish an explicit inclusive minimum and maximum supported UI protocol range;
  the browser validates the range and accepts a newer server protocol only when
  that range retains UI protocol 3. Ready responses also require a bounded
  `model_tasks` set: generation runtimes publish `generation`, while encoder
  runtimes publish `embedding` and `rerank`. Clients implementing the earlier
  handshake must add range-aware and task-aware admission before upgrading.
- Change `bloom_server`'s default `BLOOM_CORS_ALLOW_ORIGIN` policy from `*` to
  `same-origin`. Embedded UI requests and origin-free SDK/CLI clients continue
  to work without configuration; a separately hosted browser UI must configure
  its one exact HTTP(S) origin. Wildcard access now requires an explicit `*`
  and is rejected when strict security is enabled.
- Model removal now rejects IDs with surrounding whitespace instead of trimming
  them, and a missing catalog ID returns HTTP 404 `model_not_found` instead of a
  generic invalid-request response. Send the exact ID published by the catalog.
- `/v1/embeddings` now returns finite L2-normalized float vectors, rejects
  unknown non-null request fields and invalid model output, and enforces the
  active model's context window. The legacy Ollama `/api/embeddings` route now
  follows the same normalization contract; clients that depended on raw
  vectors must recalibrate.
- `/v1/rerank` now rejects unknown non-null fields and over-context inputs,
  validates complete vector dimensions, and uses stable score/index ordering.
  Requests that relied on ignored extension fields or mismatched-vector prefix
  scoring now fail explicitly.
- Embedding capability no longer depends on `embed`, `bert`, `bge`, or
  `rerank` appearing in a directory name. Native BERT packages advertise the
  capability by family; custom packages must declare trusted
  `bloom_task=embedding` or `bloom_task=rerank` manifest metadata. Renaming a
  model directory can no longer change its task.
- Restrict the public `/v1/multimodal/stream` JSON contract to bounded inline
  `Text`, `AudioPcm`, and JPEG/PNG `Image` blocks. Requests can no longer submit
  server-local `AudioFile` paths or internal `Tokens`, `Tensor`, `VideoFrames`,
  `WorldState`, and `Action` variants. Structured response formats remain
  available only on text completion routes.
- Rename legacy core config struct to `BloomConfig` in `bloomai-core`. The struct
  lives in the Bloom workspace, so the previous name leaked an external
  runtime's brand into Bloom's public API. Callers using
  the legacy config struct name must update to `bloomai_core::BloomConfig`.
  No JSON field names changed (the struct never carried a serde rename).

### Runtime

- Freeze legacy environment-backed engine settings before constructing the
  multi-threaded server runtime. The server bootstrap and standalone inference
  CLI no longer mutate the process environment after worker threads may exist.
- Accept documented bool-like environment values (`1`/`0`, `yes`/`no`, and
  `on`/`off` in addition to `true`/`false`) for server and standalone CLI
  switches. This prevents strict container defaults from failing during Clap
  parsing before startup validation.
- Make the Docker UI build fail closed and cacheable: install the Dioxus CLI
  before copying application sources, compile it with downloads disabled, and
  verify fixed Linux amd64/arm64 `wasm-bindgen`, `esbuild`, and `wasm-opt`
  artifacts by SHA-256 before use. Link the compiler from the digest-pinned
  builder under a local toolchain name so `rust-toolchain.toml` cannot trigger
  an unnecessary network refresh during the build.
- Make scheduler token-budget admission overflow-safe across shared admission,
  batch execution, and live prefill/decode accounting. Boundary-sized or
  inconsistent counters now fail closed identically in debug and release builds
  instead of panicking or wrapping into an incorrectly admitted batch. An
  enabled zero-sized prefill chunk is now rejected during shared doctor/startup
  validation instead of being silently rewritten to one token.
- Reject `max_concurrent` values above Tokio's platform semaphore capacity
  during shared startup/doctor validation instead of allowing semaphore
  construction to panic. CLI and configuration-file precedence are covered,
  and the live HTTP boundary launches an oversized configuration to require an
  actionable nonzero exit before catalog mutation or listener binding.
  Readiness permit counts now use a platform-independent u64 browser wire type
  with matching schema and validator bounds.
- Make the standalone release-archive validator parse and validate the packaged
  readiness example and compatibility-critical Draft-07 schema structure. A
  nonempty but stale, malformed, range-inconsistent, or internally inconsistent
  public handshake can no longer pass offline package verification. Live HTTP,
  bundled-example, OpenAI and Ollama smoke, and archive checks now share one
  bounded standard-library readiness validator to prevent policy drift.
- Isolate `/ready` in a strongly typed protocol projection with compile-time
  compatibility-range invariants. Server evolution can no longer omit a
  required field through an ad hoc JSON map or publish an inverted, zero, or
  self-excluding UI protocol range.
- Add exclusive process-lifetime model-catalog ownership. `bloom_server` now
  acquires a non-blocking operating-system lease on a bounded empty
  `.bloom-catalog.lock` file before upgrade recovery, stale cleanup, quota
  admission, or mutation. A second server using the same canonical catalog
  fails startup instead of racing in-process storage guards. Kernel ownership
  disappears on exit or crash while the reusable file remains; subprocess,
  stale-file, unsafe-root, permission, and Unix symbolic-link behavior are
  covered by CPU tests.
- Add crash-safe same-ID upgrades for signed model-index downloads and Ollama
  pull. Bloom now recognizes one clean prior signed alias as upgradable across
  filename and file/package-shape changes, retains it while staging, reserves
  old-plus-new peak storage, persists exact resume identity, and rehashes both
  sides before a storage-serialized commit. A bounded `.bloom-upgrade`
  transaction backs up provenance and lets startup restore the previous model
  or finish a verified replacement after interruption. Load, removal, and
  integrity races are blocked; occupied, duplicate, quarantined, symlinked,
  corrupt, or ambiguous states fail closed, and the Models drawer exposes the
  same installed/upgrade/conflict decision.
- Include nested multi-file download staging in persistent storage accounting
  and retention. Restarted package payloads can no longer disappear from quota
  usage, and stale cleanup now treats a package directory plus metadata as one
  bounded session, uses the newest nested modification time, preserves active
  work, and unlinks package-shaped symlinks without following their targets.
- Make signed-index acquisition state explicit and idempotent across the native
  `/v1` endpoint, Ollama pull, and the Models drawer. Bloom now requires an
  installed entry's exact kind, format, size, digest, package file count,
  license, download provenance, persistent signed-index alias, and clean
  integrity state before reporting success. Identical active work is joined;
  unaliased, stale, quarantined, duplicate-alias, or otherwise conflicting
  entries fail with HTTP 409, and the GUI labels installed/conflicting entries
  without offering a misleading duplicate download.
- Add signed model-index schema version 2 for bounded multi-file
  Safetensors/Transformers packages. Bloom authenticates one exact-commit file
  manifest, derives a domain-separated aggregate digest, reserves package-wide
  storage, resumes only matching hidden staging state, verifies every file, and
  validates canonical shard numbering, actual index membership, tensor-header
  ownership, offsets, duplicates, and total size before atomically publishing
  the complete no-overwrite directory. Unix publication synchronizes the
  verified tree, provenance directory, and rename parents. Versioned package
  provenance supports exact-tree integrity checks and removal cleanup. The GUI
  and CPU-tested idempotent Ollama package pull use a server-authoritative
  signed-ID acquisition path; v1
  single-file indexes remain supported by the offline signer, schemas, and
  runtime.
- Repair structured-output grammar state on the scheduler-driven Candle path.
  Prefill, scalar decode, and native batched decode now derive parser state from
  the scheduler's authoritative generated-token history and share one
  fail-closed sampling path. JSON object and JSON Schema requests disable
  unconstrained speculative token runs, while ordinary n-gram speculation no
  longer duplicates the current token in its context. The formerly ignored
  decode-chain regression is enabled, with additional independent-batch and
  structured-speculation CPU coverage.
- Add fail-closed Hugging Face Safetensors shard admission shared by manifest
  inference and Candle loading. Indexed checkpoints now require bounded regular
  files, canonical consecutive shard names, an exact discovered/indexed shard
  set, non-overlapping bounded tensor offsets, exact header-to-index tensor
  ownership, duplicate-tensor rejection, and truthful `metadata.total_size`.
  The deterministic Qwen2 CPU gate now byte-compares two independently
  generated indexed two-shard packages and executes one through buffered and
  streamed OpenAI and Ollama paths.
- Add native variable-length BERT embedding batches. The server dispatches
  bounded cancellable microbatches through a new runtime capability while
  preserving scalar adapters; Candle length-buckets inputs under an explicit
  padded-token budget, masks padding in encoder attention and mean pooling, and
  restores exact request order for embeddings and bi-encoder reranking.
- Centralize trusted manifest task classification across preflight, server
  routing, standalone inference, and benchmarking. Load preflight now publishes
  `generation` or `embedding` plus `rerank` before model weights are loaded.
- Add native BERT Safetensors execution with CLS/SEP tokenization, hidden-state
  mean pooling, model-bounded context, and zero KV-cache accounting. An
  immutable official Apache-2.0 `all-MiniLM-L6-v2` CPU gate now proves
  384-dimensional normalized output, semantic separation, bi-encoder
  reranking, OpenAI and Ollama compatibility, and fail-closed HTTP 422 task
  isolation for encoder-only models.
- Clamp standalone and legacy pipeline context windows to the model's declared
  task or architecture limit. Sentence-Transformer `max_seq_length` metadata is
  read through a bounded regular-file parser and takes precedence over the
  larger positional-embedding capacity.
- Validate bounded Sentence Transformers module and pooling metadata before
  loading BERT packages. Unsupported composition modules, non-mean pooling,
  unsafe module paths, and encoder/pooling dimension mismatches fail closed
  instead of silently changing vector semantics.

- Add an immutable official SmolLM2 360M Instruct BF16 Safetensors CPU gate.
  Raw Hugging Face packages now publish exact weight-file metadata, classify a
  bounded tokenizer chat template into an inert hard-coded contract, and prove
  deterministic CLI, benchmark, OpenAI, and Ollama execution with trained
  weights.
- Make Safetensors memory planning device-aware. CPU BF16 storage is reported
  at its actual F32 runtime footprint, missing attention head dimensions are
  safely derived from model configuration, and unsupported explicit CPU
  F16/BF16 precision fails before loading rather than during matmul.
- Release a stale non-resettable model wrapper before rebuilding its KV state,
  avoiding transient duplicate weight residency. Also require `oom` to be a
  standalone token or explicit memory-exhaustion phrase so configuration names
  such as `BLOOM_DTYPE` cannot trigger false OOM degradation.
- Repair GGUF tokenizer synthesis for current `tokenizers` JSON requirements,
  preserve GGUF CONTROL and USER_DEFINED tokens, and reproduce Qwen2's NFC,
  regex pre-tokenization, ByteLevel, and BPE settings. Standalone
  `--system-prompt` now uses the same ChatML contract as message input, and
  non-streaming `--quiet` emits only generated text.
- Make mixed-dtype GGUF preflight select the primary transformer weight dtype
  deterministically by covered elements instead of hash-map order. Candle's
  supported Q4_1/Q5_1 tensors are no longer rejected, and compatibility smoke
  scripts fail immediately when readiness reports a terminal model-load error.
- Add an opt-in, self-downloading trained-model CPU gate pinned to the official
  Apache-2.0 Qwen2 0.5B Instruct Q4_0 revision, size, model SHA-256, and license
  SHA-256. It requires exact deterministic instruction following, positive
  benchmark evidence, OpenAI streaming, and Ollama generation without storing
  weights in the repository or release archive.
- Load GGUF `qwen3` metadata with Candle's architecture-specific quantized
  Qwen3 implementation instead of routing it through Qwen2. Exact architecture
  mapping now also keeps synthesized Hugging Face configuration truthful.
- Add an immutable official Qwen3 0.6B Q8_0 trained-model CPU profile on top of
  a shared GGUF acceptance runner. CLI and server ChatML prompts prefill Qwen3's
  official empty reasoning block so ordinary output does not expose raw
  `<think>` markers while Bloom lacks a separate reasoning channel.
- Classify bounded GGUF chat-template metadata into hard-coded SmolLM2,
  ChatML, Llama 2, Llama 3, and Gemma prompt contracts without executing
  model-provided Jinja. The CLI and server now use the same safe selection,
  including SmolLM2's documented default system prompt.
- Keep ordinary text generation outside the structured-output grammar path.
  This restores deterministic token selection and exact CLI/server agreement
  instead of applying a text grammar that can perturb equal-score logits.
- Add an immutable official SmolLM2 360M Instruct Q8_0 Llama-architecture CPU
  profile. All maintained trained-model gates now require exact semantic output
  from buffered and streamed OpenAI and Ollama chat routes in addition to CLI,
  provenance, architecture, and benchmark checks.
- Create model-index watermark directories with their restrictive Unix mode in
  the creation syscall, closing a first-use race between concurrent admissions
  that could otherwise observe the directory before a follow-up permission
  change.

### C ABI / Python SDK

- Add C ABI revision 2 with runtime version negotiation, bounded
  length-delimited UTF-8 inputs, length-delimited owned output buffers, stable
  status codes, length-aware stream callbacks, and thread-safe cooperative
  cancellation tokens. The revision 1 symbols remain exported for migration.
- Prefer ABI revision 2 automatically in the Python SDK while retaining a
  tested revision 1 fallback. Closing a partially consumed Python stream now
  cancels native decode at the next output boundary and the worker owns its
  token until the native call returns.

### UI

- Add explicit, non-destructive recent-context continuations. `Continue` on any
  user message after the first creates a bounded, uniquely titled conversation
  containing that message and every later turn while retaining the complete
  source. Context-window warnings point to the action; Bloom does not silently
  truncate history. Continuations preserve response metadata and fail without
  mutation at storage, conversation, total-message, ID-space, and unsafe-role
  boundaries.
- Detect when the active generation model differs from the most recent model
  recorded in a conversation. Sending, retrying, and prompt editing remain
  unavailable until the user explicitly starts a new chat or confirms sending
  the existing history to that exact new model. Persist and display response
  model provenance independently from runtime measurements, including through
  version 2 conversation export, import, merge, replace, and branching.
- Require a successful model review before enabling `Load`. The review shows
  the model's generation or encoder tasks and strictly validates the bounded
  preflight object, requested-model identity, task set, memory plan, and load
  decision before trusting it in the browser.
- Preserve the exact input associated with every validated embedding vector and
  the exact query associated with every rerank batch. Add complete per-result
  clipboard actions plus independently bounded, versioned JSON downloads that
  revalidate shape, indices, dimensions, normalization, finite values, stable
  ordering, content budgets, and model identity before export. Package a
  Draft-07 schema and examples for both artifact identities.
- Select the primary browser workspace from readiness v3 model tasks. An
  encoder-only model now opens bounded embedding and bi-encoder reranking tools
  instead of a chat composer that would inevitably fail. Validate exact model
  identity, counts, indices, vector dimensions and normalization, finite values,
  usage, stable ranking order, and returned document identity before rendering
  compact results.
- Append the server's validated, bounded `x-request-id` to every shared
  non-successful HTTP error message. Chat, readiness, diagnostics, and model
  management failures can now be matched to server or proxy logs, while empty,
  unsafe, non-ASCII, and oversized response-header values remain undisplayed
  and separate from generation cancellation IDs.
- Show bounded `Retry-After` delta-seconds on HTTP 429 error banners without
  automatically replaying requests. Ignore dates, whitespace, signs, decimals,
  zero, values above five minutes, and hints attached to other statuses.
- Make connection testing and the live status poll verify both the public
  readiness handshake and a bounded protected Models response. Missing or
  invalid credentials now produce an actionable **API key required** state with
  the correlated 401 detail instead of appearing ready or generically offline.
- Add a locally persisted, bounded JSON-array stop-sequence control for text
  chat. Validate one to four exact strings before request construction and
  reject active stop controls for image chat during transactional preflight.
- Add explicit transactional **Merge** and **Replace all** choices to validated
  conversation archive imports. Merge retains local history and active
  selection, appends archive-order copies with fresh IDs, and fails without
  mutation at combined conversation, message, ID-space, or storage limits.
  Recovery-locked unreadable history disables Merge while preserving the
  explicit replacement and recovery paths.
- Preflight the complete text or multimodal submission before initial-send,
  retry, or edit state changes. Reuse exact text-body encoding and multimodal
  prompt construction, retain drafts and prior responses on rejection, and show
  edit-admission failures inside the open dialog. Render only the latest 100
  messages initially and reveal earlier history in tested 100-message pages
  without changing absolute message actions or request history.
- Add bounded per-message conversation branching. A branch copies history only
  through the selected user or assistant message, preserves local response
  metadata, assigns a unique bounded title and ID, selects the new conversation,
  and never mutates its source. Fail closed at conversation, message, and ID
  limits, and warn when copied image turns lack replayable bytes. Make new-chat
  creation, selection, deletion, branching, and prompt insertion publish to
  browser storage before replacing reactive state.
- Give every drawer, editing dialog, and destructive confirmation a shared
  accessible focus lifecycle: labelled descriptions, initial in-dialog focus,
  Tab and Shift+Tab containment, Escape dismissal, opener-focus restoration,
  visible keyboard focus, unique confirmation IDs, and reduced-motion support.
  Add host tests for keyboard dispatch and focus-boundary selection.
- Add a decoupled Dioxus (Rust → WASM) frontend in `ui/`: streaming chat
  against the OpenAI-compatible API (SSE), connection settings with
  persistence, generation-parameter sliders, and a live health/status bar.
- Add an optional `serve-ui` feature to `bloom_server` that embeds the
  frontend with `rust-embed` and serves it at `/`, so a single binary can host
  both the API and the UI. Off by default; the backend stays independently
  deployable and the frontend can be hosted separately.
- Add `just ui-dev` / `ui-build` / `server-ui` recipes and `ui/README.md`
  documenting both deployment modes.
- Rename the frontend directory/crate from `web/` (`bloom-web`) to `ui/`
  (`bloom-ui`): the Dioxus frontend can target web or desktop, so `web/` was
  misleading. The `ui-*` recipes already followed this naming.
- Add persistent browser-local conversations with creation, selection,
  automatic titles, and guarded deletion.
- Add generation cancellation using both `AbortController` and Bloom's
  request-cancellation endpoint, while preserving partial responses.
- Poll `/ready` so the UI distinguishes an available model from a server that
  is alive but still loading, and disable generation until the model is ready.
- Harden the SSE client for CRLF framing, split UTF-8 input, request IDs, and
  streamed server errors, with host-side regression tests.
- Add a dedicated UI CI job covering formatting, host tests, strict Clippy,
  and the production WebAssembly compilation target.
- Persist generation settings and add system prompt, `top_p`, and deterministic
  seed controls; validate supported ranges before requests are sent.
- Add an English model manager that discovers the server's safe local catalog,
  shows active/loading/error state, switches models, and unloads the runtime.
- Add one-request JPEG/PNG attachments to chat. Images use bounded multipart
  uploads, reuse streamed multimodal output and request cancellation, and are
  not persisted in browser-local conversation storage.
- Add an opt-in verified download form to the Models drawer with progress,
  cancellation, retry, and resume state.
- Add metadata-only Hugging Face source inspection to the download form. It
  accepts `/blob/` and `/resolve/` links, fills safe filename/size/SHA-256
  fields, pins published repository commits, requires explicit confirmation,
  and retains manual checksum entry as a fail-closed fallback.
- Render administrator-approved license choices in both acquisition forms when
  the server enables license admission, and prevent new/resumed actions until a
  permitted declaration is selected.
- Add a searchable publisher-signed model index to the Models drawer. The UI
  strictly validates bounded server snapshots, explains signature and checksum
  trust boundaries, disables size/license-policy conflicts, and copies a chosen
  immutable source into the existing explicit verified-download form.
- Rediscover partial downloads after restart and add guarded resume, discard,
  and permanent inactive-model removal controls.
- Add opt-in browser-local GGUF, ONNX, and Core ML imports with bounded chunk
  transfer, progress, cancellation, restart-safe resume, staged-data discard,
  and mandatory SHA-256 verification.
- Show shared model storage usage, outstanding reservations, configured quota,
  available capacity, retention policy, and the latest cleanup result in the
  Models drawer.
- Let verified downloads and browser imports declare license metadata, let
  imports declare a public HTTPS source, and show acquisition provenance,
  checksum, source, missing metadata, and invalid-record warnings on model cards.
- Add cancellable integrity verification controls with progress and persistent
  verified/mismatch states. Model cards warn about quarantined entries and
  disable loading until a later checksum check succeeds.
- Add model details backed by load preflight. The Models drawer shows bounded
  architecture, precision, runtime compatibility, and memory-budget details,
  and disables loading after a known incompatible verdict.
- Add `Export JSON` to the Models drawer for a stable, versioned, path-free
  catalog inventory with provenance, integrity, and exact-commit source-lock
  summaries.
- Add `Compare JSON` with bounded browser file handling and a clear, read-only
  summary of matching, missing, unexpected, changed, and blocking catalog drift.
- Add explicit per-model `Restore` for eligible missing inventory entries, with
  exact-commit and SHA-256 requirements, confirmation, download-state guards,
  and no bulk execution.
- Render assistant responses as constrained Markdown with readable headings,
  lists, tables, links, inline code, and fenced code blocks. Raw HTML is escaped,
  unsafe link protocols are suppressed, and remote images become alt text.
- Add a versioned conversation JSON archive with an 8 MiB input limit, strict
  validation, no connection configuration or credentials, import count preview,
  and explicit confirmation before replacing browser-local history.
- Add accessible per-message copy controls that write original plain text and
  report secure-context or clipboard-permission failures without injecting HTML.
- Add guarded regeneration for the latest text response through the shared
  streaming/cancellation path. Empty failures restore the prior answer, partial
  output is retained, and requests with unavailable image bytes are not replayed
  as text.
- Add transactional editing and resubmission for the latest text prompt. Empty
  failure restores the original prompt, response, and automatic title; failed
  user-only turns expose retry/edit recovery without duplicating their context.
- Add case-insensitive browser-local conversation search over titles and message
  text, plus validated renaming that persists a copy of the updated store before
  changing reactive UI state.
- Add a strict versioned v2 envelope for browser-local conversations with v1
  migration. Malformed data is never silently overwritten: conversation writes
  pause until the raw recovery copy is downloaded and a fresh start is confirmed,
  or a validated archive is imported.
- Add OpenAI-compatible streaming usage chunks and per-assistant generation
  diagnostics in the UI. Prompt/output tokens, total time, TTFT, throughput, and
  completed/stopped/failed outcome persist locally while portable archives omit
  runtime-specific measurements.
- Publish the active context window in readiness and reject invalid generation
  controls or prompt-plus-output budgets before inference. The UI shows previous
  turn utilization, warns about likely overflow, and never silently truncates
  conversation history; unsupported batched completion prompts now fail clearly.
- Add persistent Text, JSON object, and JSON Schema controls to the UI. Schema
  admission is bounded and fail-closed on both client and server, structured JSON
  renders as escaped code, and portable archives retain rendering semantics
  without exporting schemas or runtime settings.
- Bind every UI text and multipart image generation to the active readiness
  model captured at send time, preventing a concurrent model switch from
  silently changing execution identity.
- Validate bounded execution-model metadata across every text and multimodal
  stream event, reject request mismatches or mid-stream identity changes, and
  persist the confirmed model with local assistant diagnostics while keeping it
  out of portable conversation archives.
- Treat streamed request IDs as bounded cancellation capabilities: require a
  safe and stable ID before content, revalidate and URI-encode it before the Stop
  action, and prevent a malicious stream from redirecting an authenticated POST.
- Bound browser SSE transport independently of token settings: require the
  event-stream media type, cap response, frame, error, and accumulated-output
  bytes before UI updates, and reject EOF without `[DONE]`. Multimodal `End`
  chunks no longer terminate the HTTP reader before the server's final event.
- Replace unbounded browser `Response.text()` reads with incremental ordinary
  response decoding. Success bodies have a 16 MiB ceiling or a smaller endpoint
  budget, error bodies are capped at 64 KiB, displayed detail is capped at 4 KiB,
  and both declared length and actual raw/decoded bytes fail closed before JSON
  parsing or UI updates.
- Require JSON media types for successful typed browser responses and bound
  ordinary HTTP progress with a 120-second response-header deadline, 30-second
  body-idle deadline, and 300-second total-body deadline. Fetch and reader guards
  also cancel network work when their owning UI future is dropped, while slow
  CPU generation and large imports retain explicit user cancellation.
- Bound outbound browser request construction before large serialization or `fetch`:
  validate connection and persisted generation settings, cap chat message count,
  content, and encoded JSON, cap multimodal prompt/image metadata, reject unsafe
  import chunk sizes, and avoid duplicate prompt construction for text-only
  generation. Limits fail explicitly without truncating conversation history.
- Keep API keys for new connections in per-tab session storage by default and
  add an explicit **Remember API key in this browser** opt-in for persistent
  local storage.
  Version-tolerant connection decoding preserves legacy credentials as visibly
  remembered, session-only serialization omits the secret, policy changes clear
  stale storage, and invalid or unknown records fail closed.
- Add connection-aware first-run guidance and a live Runtime diagnostics drawer.
  The UI validates a 256 KiB versioned snapshot, shows model/load, request,
  token, scheduler, memory, KV-cache, and CacheMesh state, and exports a
  credential-free support JSON file.
- Add explicit `--open-browser` / `BLOOM_OPEN_BROWSER` startup for embedded
  application builds. Bloom waits until the listener is bound, converts
  wildcard listeners to a local URL, supports IPv6, falls back across native
  browser launchers, and keeps headless server defaults unchanged.

### Server

- Reset every start-at-zero native Candle sequence before execution, including
  the first embedding after startup verification and every item in an embedding
  batch. Qwen2, Qwen3, Gemma, and streaming wrappers now clear their KV caches
  in place, while wrappers without a safe reset API are recreated. This prevents
  cross-input context leakage and causal attention-mask shape failures without
  remapping supported model weights for every fresh request. Prompt reuse now
  requires the new input to extend the entire physical cached sequence because
  these wrappers cannot safely truncate a divergent suffix; OnDemand offload
  clears the matching logical prefix together with its physical cache.
- Close the verified Ollama acquisition-to-inference lifecycle. Signed pulls
  now persist their index ID as a backward-compatible acquisition alias used by
  tags, process discovery, show, delete, chat, generate, and both embedding
  routes. Exact inactive selectors run shared integrity/preflight checks and
  atomically activate on demand; same-target concurrent requests join one
  sequenced terminal result, different lifecycle work conflicts, load failure
  preserves the previous runtime, and client responses retain the requested
  public selector even when runtime metadata differs. Add empty chat/generate
  preload, negative indefinite residency, response-safe `keep_alive: 0`, and
  bounded positive timed expiry with five-minute defaults, numeric seconds, Go
  duration strings, `/api/ps` deadlines, revision-based refresh, exact-runtime
  identity checks, cancellation-safe automatic unload, and transactional policy
  updates that preserve the old deadline when activation fails. Model inventory
  v2 exports and restores the optional alias while the server continues
  accepting legacy v1 backups. Extend CPU provenance, identity,
  concurrency, lifecycle, stream, pull, unload, and server-route coverage plus
  compatibility, architecture, security, operations, and release guidance.
- Add a bounded Ollama-compatible `POST /api/pull` projection over Bloom's
  operator-trusted signed model index and verified downloader. Clients supply
  only an exact signed entry ID; Bloom enforces immutable source, destination,
  signed size, SHA-256, license, quota, provenance, resumability, and
  no-overwrite installation. Streaming and non-streaming official-client wire
  shapes are supported, identical concurrent work shares progress, progress
  disconnects leave the background download resumable, and successful pulls do
  not automatically load a model. Registry resolution and insecure pulls fail
  closed. Extend CPU downloader/server tests, official-client smoke, security,
  compatibility, production, and release documentation.
- Add a fail-closed browser-origin request boundary ahead of CORS and routing.
  It accepts the embedded same origin by default, one explicitly configured
  cross-origin HTTP(S) UI, or an explicit wildcard; rejects malformed, opaque,
  duplicate, untrusted, and loopback DNS-rebinding origins with HTTP 403; and
  leaves origin-free SDK/CLI traffic unchanged. Align startup validation,
  deployment doctor warnings, protocol-shaped errors, unit tests, three-process
  live tests, official-client smoke, and staged-package verification.
- Add an authoritative HTTP authentication-challenge boundary. Every 401
  without a handler-specific challenge receives
  `WWW-Authenticate: Bearer realm="Bloom"`; explicit challenges are retained,
  403 and public route fallbacks are unchanged, and CORS exposes the header.
  Extend OpenAI/Ollama unit, live-process, official-client, and packaged-binary
  coverage for missing, invalid, Bearer, and `X-API-Key` credentials.
- Add a shared transient-overload response boundary. Every HTTP 429 without a
  handler-specific hint now receives `Retry-After: 1`; existing hints are
  retained, 503 responses are unchanged, and CORS exposes the header to browser
  clients. Add capacity, middleware, live-process, UI validation, and packaged
  binary release coverage.
- Add an outer protocol error normalizer for framework-level rejections.
  Malformed JSON, endpoint-schema mismatches, missing media types, body limits,
  invalid multipart extraction, timeouts, and other non-protocol errors now
  retain their meaningful status and safe headers but return fixed, bounded
  OpenAI or Ollama JSON instead of framework text or empty bodies. Existing
  JSON/SSE/NDJSON, probes, and static UI paths pass unchanged. Ollama body-limit
  rejection now retains HTTP 413. Extend CPU middleware, live authenticated
  process, official-client smoke, documentation, and staged-package gates.
- Add protocol-owned route and method fallbacks. Unknown `/v1` paths now return
  bounded, non-reflective OpenAI error envelopes, unknown `/api` paths return
  Ollama error envelopes, and known routes called with the wrong method retain
  HTTP 405 plus `Allow`. The fixed fallbacks remain separate from API-key
  authentication while recognized routes stay protected. Extend router,
  real-process, official-client smoke, documentation, and staged-package gates.
- Version the public `/ready` response as a Bloom UI/server compatibility
  handshake with schema, object, protocol, and package identities plus bounded
  required admission fields. Make the browser reject partial, legacy,
  inconsistent, or non-Bloom documents as **Incompatible Bloom server**, keep
  transport failures distinct, publish a Draft-07 schema and example, and
  extend live and staged-binary release gates.
- Add one authoritative dynamic-response cache boundary. Every `/v1` and
  `/api` response plus health, readiness, and metrics now carries
  `Cache-Control: no-store`, including JSON, SSE, NDJSON, authentication and
  admission failures, timeouts, body-limit errors, and unknown protocol routes.
  Downstream handler directives cannot weaken the policy, while embedded static
  paths remain separate. Extend CPU router, live no-model, CI, and staged-binary
  package gates.
- Replace the misordered request-ID middleware with one authoritative HTTP
  correlation boundary. Every success and error response now carries a
  CORS-exposed `x-request-id`; missing or unsafe inbound values become UUIDs,
  while bounded proxy values using a conservative ASCII alphabet are retained.
  Trace spans include the normalized ID, method, and path without query strings.
  Add CPU router coverage and a live no-model smoke gate for generation,
  preservation, unsafe replacement, CORS exposure, and unknown-route behavior.
- Constrain the embedded UI's SPA fallback to extensionless HTML `GET`
  navigation. Unknown `/v1/*` and `/api/*` routes, probe namespaces, missing
  assets, JSON clients, and non-GET requests now retain real 404 behavior rather
  than receiving a false `200 text/html`. Add serve-ui router tests and a
  cross-platform staged-binary HTTP boundary gate to release packaging.
- Add bounded, authenticated `GET /v1/models/{model}` compatibility for the
  exact active model and Bloom's `default` alias. List and retrieve now share a
  standard Model projection with the runtime's stable process-local publication
  time instead of an arbitrary fixed timestamp. Missing models return typed 404
  errors; invalid selectors and unknown query parameters fail with 400. Add
  CPU HTTP coverage and mandatory official OpenAI SDK missing/success retrieval
  checks across model-free and real-model smoke paths.
- Add authenticated `DELETE /api/delete` compatibility with Ollama's empty
  HTTP 200 success contract and Ollama-shaped 400/404/409/500 failures. Both API
  surfaces now share one exact-ID, fresh-catalog, storage-serialized removal
  operation that refuses active models, integrity checks, lifecycle races,
  unsafe paths, and ambiguous whitespace. Add CPU-only HTTP and no-model smoke
  coverage, including non-destructive authorization and rejection assertions.
- Support bounded OpenAI Chat/Completion and Ollama chat/generate stop
  sequences across buffered, SSE, and NDJSON responses. Incrementally retain
  possible cross-delta prefixes, exclude matched markers and later text, end
  blocking or continuously batched generation early, preserve successful
  lifecycle accounting, and report `stop` consistently. Add a CPU-only fake
  model HTTP test proving early termination across all four protocol routes.
- Move `/v1/rerank` from one blocking task per query/document onto the shared
  bounded embedding executor. Normalize and score the whole batch inside the
  worker-owned lifecycle, retain cancellation registration and concurrency
  admission through scoring, return finite clamped cosine scores with stable
  tie ordering and process-unique IDs, and remove the dimension-truncating
  legacy cosine path. Add an embedding-only OpenAI smoke mode covering SDK
  projection, norms, usage, rerank ordering, `top_n`, and document identity.
- Add authenticated current `/api/embed` and superseded `/api/embeddings`
  Ollama compatibility. Support bounded string batches, default context
  truncation, explicit no-truncation errors, dimensionality projection,
  normalized current vectors, raw legacy vectors, token/duration metadata, and
  Ollama-shaped errors while preserving Bloom's one-active-model lifecycle.
  Refactor OpenAI embeddings onto the same output-bounded, disconnect-aware
  blocking executor so cancellation registration, concurrency permits, and
  exactly-once metrics remain owned until the worker exits. Reject non-finite,
  inconsistent, oversized, aggregate-over-limit, or zero-norm vectors and
  non-neutral unknown request semantics.

- Add an authenticated, fail-closed Ollama API adapter for bounded
  `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, and
  `/api/generate` compatibility. Preserve Bloom's single-active-model and
  verified-acquisition boundaries; translate text, JSON/JSON Schema output,
  function tools and name-correlated result history through the shared Chat
  core; convert SSE to bounded Ollama NDJSON with terminal metrics; and return
  Ollama-shaped HTTP or in-stream errors. Pull, creation, copy, automatic
  loading, thinking, log probabilities, raw/suffix/image generation, residency
  changes, and unknown semantics remain explicit errors; guarded deletion was
  added later through the shared Bloom lifecycle operation.
- Add a fail-closed `POST /v1/responses` adapter for bounded text generation.
  Support instructions, string or `input_text` message input,
  modern generation controls, current SDK-shaped non-streaming output, and
  Responses-native SSE lifecycle, delta, item-done, usage, completed,
  incomplete, and failed events. Bound and validate internal framing, stable
  metadata, sequence and output growth, and disconnect ownership. Add
  privacy-first explicit `store: true`, bounded process-local retention,
  same-model `previous_response_id` history without stale top-level
  instructions, bounded non-inherited response metadata, retrieve/delete,
  cursor-paged input-item listing, and
  commit-before-terminal streaming semantics. Reject background work,
  automatic truncation, and unsupported content semantics before runtime
  admission.
- Add native Responses function calling through the shared bounded Chat tool
  core. Accept flat function definitions, `none`/`auto`/`required`/named
  choices, up to eight parallel calls, native `function_call` and string
  `function_call_output` input items, native non-streaming output items, and
  function-argument streaming events. Retain call/result lifecycle items for
  strict `previous_response_id` continuation, validate all IDs, pairings,
  strict schemas, and generated arguments, and continue to reject custom or
  hosted/built-in tools without executing functions server-side.
- Add Responses `text.format` support for bounded `json_object` and
  `json_schema` structured output. Map the direct Responses schema shape into
  the shared prompt/engine constraint, preserve the normalized format across
  lifecycle objects, and turn invalid streamed output into `response.failed`
  instead of reporting completion.
- Emit a standard assistant-role start chunk for Chat streaming and keep its
  creation timestamp stable across start, delta, stop, and usage chunks.
- Accept leading OpenAI `developer` chat messages as explicit local system
  instructions and reject late developer turns that cannot preserve instruction
  priority. Correct the Llama prompt formatter so consecutive leading
  instruction messages are never folded into a user turn.
- Accept modern OpenAI chat text-part arrays and `max_completion_tokens` while
  retaining string content and `max_tokens`. Text parts are concatenated in
  order under existing content budgets; empty, malformed, non-text, excessive,
  or semantically extended parts and conflicting dual token limits fail with a
  pre-runtime HTTP 400 instead of being ignored.
- Retain extension fields on chat, message, stream-option, and legacy
  completion payloads. Admit documented exact no-op OpenAI defaults, but reject
  tools, multiple choices, log probabilities, penalties, logit-bias controls,
  message tool calls, and unknown non-neutral semantics
  with a bounded HTTP 400 error before runtime admission instead of silently
  ignoring them.
- Handle `Ctrl-C` on supported hosts and `SIGTERM` on Unix as one production
  shutdown lifecycle: withdraw readiness immediately, drain existing HTTP
  requests for a configurable bounded 1-to-3,600-second window, and exit with
  status 1 when the deadline expires or a second shutdown signal requests
  immediate escalation. The default is 30 seconds, Docker declares `STOPSIGNAL
  SIGTERM`, and POSIX CI plus native Unix packaging exercise clean,
  deadline-forced, and repeated-signal exits with the real server process.

- Add endpoint-specific admission budgets for generation output, legacy
  completion prompts, embedding batches, reranking query/document sets, and
  inline multimodal text, PCM, and images. Invalid decoded requests now return
  HTTP 400 before runtime readiness checks, inference permits, or request
  metrics, independently of the configurable general JSON body limit.
- Validate chat message shape before inference admission: require a non-empty
  `system`/`user`/`assistant` sequence, cap message count, per-user and per-system
  character counts, and combined content bytes independently of the configurable
  general JSON body limit.

- Enforce optional OpenAI-compatible `model` selectors across chat, legacy
  completion, embeddings, and reranking before inference admission. The
  `default` alias remains compatible, mismatches return `404 model_not_found`,
  multipart UI uploads carry the same binding, and response metadata always
  reports the runtime that actually executed the request.
- Validate decoded `/v1/cancel/{request_id}` path values against the same bounded
  ASCII contract used by generated IDs and return HTTP 400 before lookup or
  reflection when the value is unsafe.
- Keep streaming and non-streaming generation permits, cancellation
  registrations, and metrics owned until the response and confirmed execution
  exit. Early body/handler drops and HTTP timeouts cancel scheduled or
  cooperatively cancellable text and multimodal work, drain blocking workers
  before releasing admission, remove IFB token senders, and settle request
  accounting exactly once.

- Add side-effect-free `bloom_server --doctor[=text|json]` deployment checks
  for effective arguments, network security, engine/device availability,
  bounded catalog discovery, startup-model compatibility and memory planning,
  storage policy, and embedded UI presence. Normal startup now fails on the
  same invalid numeric and feature combinations before storage mutation or
  port binding; the versioned JSON report has a public schema and example.
- Version `/v1/observability`, add server version, monotonic uptime and bounded
  model-load state, return `Cache-Control: no-store`, and publish a Draft-07
  JSON Schema plus validated synthetic example.

- Allow `bloom_server` to start without `--model`, using `--models-dir` or
  `BLOOM_MODELS_DIR` as an authenticated model catalog.
- Add `/v1/model-management/models`, `/switch`, and `/unload`. Catalog requests
  use discovered direct-child IDs rather than arbitrary paths, reject traversal
  and escaping symlinks, serialize lifecycle operations, and retain the prior
  runtime when a replacement fails to load.
- Publish runtime state as one snapshot, close inference admission while
  switching, drain in-flight work before replacement, and terminate the old
  continuous-batching worker when its runtime is released.
- Include model load progress and a path-safe failure notice in public
  `/ready`, including the useful 503 state of a healthy server with no active
  model; detailed errors remain on the authenticated management endpoint.
- Cache model catalog scans for a short interval and invalidate them when the
  active model changes, avoiding repeated recursive size walks from UI polling.
- Add `/v1/multimodal/upload` for one bounded JPEG/PNG attachment plus prompt
  and generation fields. The endpoint emits an immediate request-ID event,
  then reuses the existing multimodal SSE execution and cancellation path.
- Add `/v1/model-management/downloads` for opt-in, resumable downloads from
  trusted Hugging Face HTTPS hosts. Downloads require SHA-256, enforce a size
  limit, reject unsafe redirects, stage outside the visible catalog, and use a
  no-overwrite atomic install after verification.
- Add authenticated `/v1/model-management/downloads/inspect` for bounded HEAD
  metadata discovery. It strips URL secrets, normalizes repository browser
  links, trusts only 64-hex content hashes, applies the configured download
  limit before transfer, and returns a commit-pinned source when available.
- Add a bounded, optional acquisition license allowlist through config,
  `--allowed-model-licenses`, and `BLOOM_ALLOWED_MODEL_LICENSES`. The server
  canonicalizes exact case-insensitive matches, enforces them for downloads and
  imports, rechecks restart-time publication, exposes approved declarations to
  the GUI, and reports missing governance in `--doctor`.
- Add optional Ed25519-signed model discovery from one local file or constrained
  HTTPS source. Bloom strictly verifies domain-separated signatures, key IDs,
  expiry, payload bounds, immutable Hugging Face commits, duplicate IDs and
  filenames, and entry metadata; only an unexpired verified cache can survive a
  refresh failure. Authenticated GET/POST index endpoints, doctor checks,
  environment/config support, Draft-07 schemas, a signed example, and a safe
  offline signing helper complete the publisher workflow.
- Extend signed discovery with a backward-compatible one-to-eight-key trust
  set for controlled old/new overlap rotation. Trust-set fingerprints are
  order-independent, doctor output remains key-material-free, the GUI polls at
  the bounded server refresh interval, and Bloom rejects signed rollback or
  conflicting equal-time generations while retaining only a newer unexpired
  snapshot.
- Persist signed-index rollback watermarks in a bounded source-scoped state
  directory before exposing a generation. Immutable atomic publication prevents
  concurrent equal-time overwrite, restart and trust-set-reduction tests fail
  closed, doctor validates the private state without path disclosure, and the
  GUI reports persistent rollback protection.
- Add server-side staged-download inventory, filename-only resume/discard
  actions, and guarded inactive-model removal. Storage and model-switch
  operations share a lifecycle gate, and removal does not follow nested
  symlinks.
- Add `/v1/model-management/imports*` for opt-in, offset-checked local-file
  chunks. Imports enforce file and request limits, retain safe partials across
  restarts, verify SHA-256, reject staging symlinks, and install without
  overwriting an existing catalog entry.
- Add a shared model-storage coordinator and optional
  `BLOOM_MAX_MODEL_STORAGE_BYTES` commitment quota across installed models,
  partial downloads, and declared imports. Add configurable startup and
  periodic stale-session cleanup with active-download protection, bounded
  filesystem accounting, and catalog API status.
- Persist versioned provenance records for verified downloads and imports in a
  private `.bloom-metadata` directory. Strip URL secrets, validate records at
  catalog scan time, roll back installs when record publication fails, and
  clean records when single-file models are removed.
- Add an asynchronous `/v1/model-management/integrity` operation that hashes
  inactive acquired files, detects identity changes, persists its result, and
  blocks loading or removal races. Recorded mismatches survive restarts and
  prevent loading until a matching verification clears them.
- Add authenticated `/v1/model-management/preflight` inspection with bounded
  manifest summaries, engine and device capability checks, loader-equivalent
  memory planning, serialized work, and a short bounded cache. Catalog switches
  enforce the verdict before entering the load queue.
- Add authenticated `/v1/model-management/inventory` export with deterministic
  ordering, URL-secret redaction, transient-state omission, exact Hugging Face
  commit detection, a documented format, and a Draft-07 JSON Schema.
- Add authenticated `/v1/model-management/inventory/reconcile` with strict
  version `1` validation, bounded input and output, deterministic value-redacted
  drift reporting, and a documented Draft-07 response schema.
- Add authenticated `/v1/model-management/inventory/restore/{id}`. It revalidates
  the complete inventory and current catalog, accepts only missing exact-commit
  download records, and delegates to the existing bounded, resumable,
  SHA-256-verified, atomic no-overwrite download manager.
- Add Content Security Policy, clickjacking, MIME-sniffing, referrer, and browser
  permissions headers to every response served by the embedded UI.

### CI / open-source readiness

- Give the embedded UI a single descriptive document title and favicon, promote
  the product name to the page's level-one heading, name the conversations
  landmark, and announce connection/runtime state through a polite live status.
  A real-browser empty-state audit now records dialog focus entry, forward and
  reverse focus containment, Escape dismissal, and opener-focus restoration;
  automated cross-browser and assistive-technology gates remain future work.
- Add a fail-closed locked-dependency policy for reviewed Cargo registry
  sources and exact declared license expressions. CI and release packaging now
  validate before building and generate a deterministic CycloneDX 1.5 SBOM
  covering the native target plus the independent wasm UI workspace when
  embedded. Schema-version-2 archives bundle it with the reviewed policy; the
  offline validator retains legacy version-1 support and binds current SBOM
  identity, components, sources, licenses, and graph to the release contract.
- Replace loose Markdown issue templates with required, English-only GitHub
  forms for bugs, features, and model/backend support. Route vulnerabilities to
  private advisories, disable public blank issues, use only existing repository
  labels, and add a pinned YAML parser plus an offline fail-closed metadata
  contract with negative regression tests.
- Add weekly Dependabot coverage for both Cargo workspaces, Python requirements
  and package metadata, the Docker base image, and GitHub Actions. Add a
  standard-library local Markdown-link gate to pull-request CI and native
  release jobs, repair the stale support-matrix path in the PR template, and
  align contributor commands with the locked CI workspaces. Pin the currently
  tested Rust 1.97.1 patch release across local rustup, every CI/release job, and
  the Docker builder, with an offline consistency gate that prevents partial
  upgrades without presenting the pin as an unverified MSRV guarantee. Pin all
  37 external GitHub Action uses to verified upstream commit SHAs, retain
  same-line release comments for Dependabot, enforce the rule offline, default
  workflow tokens to read-only, and grant write access only to release
  publication. Official archives and checksum files now receive signed
  GitHub/Sigstore build-provenance attestations before the draft release is
  created, with consumer verification documented separately from local SHA-256
  checking. A standard-library workflow contract gate preserves the required
  permission scope, complete archive/checksum subjects, and download-attest-
  publish ordering. Regression tests inject permission, subject, action-pin,
  ordering, and duplicate-step failures into that workflow contract. Release
  tar.gz and zip creation now normalizes ordering, timestamps, owners, and
  modes from `SOURCE_DATE_EPOCH`, publishes atomically, and has byte-level
  standard-library regression tests for both formats. The offline release
  validator independently enforces that metadata contract for hosted builds.
- Add an exact multi-crate archive gate for coordinated unpublished workspace
  releases. It performs locked packaging, extracts every publishable `.crate`,
  and compiles those archive contents against the sibling release set instead
  of accidentally verifying against older same-version crates.io packages.
- Add a deterministic, untrained tiny Qwen2 fixture generator and a mandatory
  Linux CPU runtime gate. The gate verifies byte-for-byte reproducible config,
  tokenizer, and Safetensors artifacts, then crosses the native Candle loader,
  forward pass, decoding, authenticated OpenAI/Ollama buffered and streaming
  routes, inactive-catalog activation, `/api/ps` publication, OpenAI embedding
  projection and reranking, current and legacy Ollama embeddings, clean
  per-input KV state, successful raw and pinned-SDK OpenAI/Ollama JSON and JSON
  Schema output, successful OpenAI Chat/Responses and Ollama chat function
  calls, buffered structured and function streams, result continuation,
  private-control non-disclosure, retained state, and both pinned official
  clients without downloading or committing model weights.
  Keep trained-model semantic and instruction-following quality plus
  architecture breadth as separate release evidence.
- Pin the Draft-07 JSON Schema validator and require complete public schema
  checks in Linux CI and every native release job. Keep the standard-library
  structural fallback for minimal local environments without treating it as
  full release evidence.
- Pin the official OpenAI and Ollama Python clients used by compatibility
  testing and make both Linux CI smokes require them with API-key protection.
  The model-free gates now prove official-client decoding for empty discovery,
  HTTP 401, unavailable-model chat, missing Responses state, and a non-mutating
  Ollama delete 404 instead of silently skipping every SDK assertion when the
  packages are absent.
- Build and embed the Dioxus UI in official release archives and the Docker
  image. Native release binaries run their versioned doctor report before
  packaging, and the release fails if the executable is unhealthy or the UI
  is absent. Archives now include documentation, examples, and checksums.
- Add one fail-closed production UI build wrapper used by local packaging,
  Docker, Just, and GitHub Releases. It disables release DWARF before
  `wasm-opt`, rejects Dioxus's zero-exit optimizer failure, verifies the
  generated entry point, JavaScript glue, and WebAssembly payload, and only
  then publishes `ui/dist`.
- Make GitHub Releases call the same packaging script used locally. Archives
  now carry an archive-first `QUICKSTART.md` and schema-versioned
  `BLOOM-RELEASE.json` with target, UI, self-check, size, and per-binary
  SHA-256 metadata; hosted archive names use the Rust target triple.
- Add native Windows CI and release jobs, portable Python-based zip and
  checksum packaging, PowerShell first-run guidance, and a fail-closed tar.gz
  and zip release artifact validator.
- Establish `main` as the default branch; CI now triggers on pushes and PRs to
  `main`.
- Decouple the `missing_docs` lint from the blocking Clippy gate: CI runs
  `clippy -D warnings -A missing_docs`, plus a non-blocking advisory
  `docs` job (`-W missing_docs` and `cargo doc`). Removed the per-crate
  `#![warn(missing_docs)]` attributes that command-line flags could not
  override.
- Fix three genuine `clippy::useless_vec` errors that the `missing_docs`
  noise had been masking.
- Add a `hardware-tests` feature (engine + tilelang) and gate TileLang JIT
  tests behind it; they need the Python/numpy toolchain and no longer fail on
  standard CI runners.
- Update CONTRIBUTING to reflect the `bloomai-*` namespace (drop stale
  namespace compatibility notes).
- Remove `continue-on-error` from the four CI smoke steps (benchmark, OpenAI
  API, llama.cpp comparison, Docker build). All three scripts already SKIP
  gracefully (exit 0) when models or external binaries are absent, so the
  escape hatch was masking real failures.
- Add a `Security audit` CI job using `rustsec/audit-check` to catch
  advisories in `Cargo.lock` on every PR and push to `main`.
- Commit application lockfiles, use `--locked` in CI and packaging paths, and
  enforce panic-free production paths across core resources, backends, engine
  executables, schedulers, model loaders, and HTTP streaming handlers.
- Exclude nested Rust build directories from Docker contexts, preventing the
  standalone UI target directory from adding gigabytes to container builds.
- Normalize Rust formatting across the workspace so the existing blocking
  `cargo fmt --check` gate is reproducible on the current stable toolchain.
- Extract shared byte-unit constants (`KIB`/`MIB`/`GIB` and `f64` variants)
  into `bloomai_core::constants`, replacing scattered local definitions and
  raw `1024 * 1024 * 1024` literals across 13 production source files.
- Remove the crate-level `#![allow(dead_code, unused_variables, ...)]` from
  `bloom_server/main.rs`. Fixed the underlying issues: dropped an unused
  `engine` binding and a duplicate `backend_name` declaration, removed two
  dead utility functions from `chat_template.rs`, and replaced
  `contains_key`+`insert` patterns with the `Entry` API.
- Adopt typed `BloomError` variants across the engine crate, replacing
  unstructured `anyhow::anyhow!` / `anyhow::bail!` calls in 10 source files:
  `gemma4.rs`, `core/model.rs`, `core/pipeline.rs`, `core/manifest.rs`,
  `plugin/mod.rs`, `scheduler/mod.rs`, `scheduler/kv_hook.rs`,
  `scheduler/scheduler_test.rs`, `executor/qwen_streaming.rs`, and
  `bloom_server/main.rs`. Each call site now uses the semantically correct
  variant (`ModelLoad`, `Engine`, `InvalidInput`, `UnsupportedFormat`,
  `UnsupportedFamily`, `MissingRequiredFile`, `HashMismatch`, `Plugin`,
  `SchedulingFailed`, `Resource(...)`, etc.), enabling `ErrorCategory`
  classification and `recovery_hints()` on the hot path. The OOM-detection
  logic in `pipeline.rs` also learned to recognize typed
  `BloomError::Resource(InsufficientRam|InsufficientVram|...)` as OOM.

## 0.1.0 - 2026-07-21

First public open-source release of the Bloom workspace.

### Engine & Features

- Initial early engine workspace with CLI, OpenAI-compatible server, scheduler,
  CacheMesh, plugins, GGUF inspection, benchmark schema, and experimental model
  backends.
- Vendor `bloomai-core`, `bloomai-backend`, and `bloomai-tilelang` into the
  Bloom workspace so the repository can build independently.
- Add optional API-key protection, configurable CORS, and JSON request body
  limits to `bloom_server`.

### Open-source readiness

- Rename crates to the `bloom-*` namespace and unify the repository URL to
  `github.com/constmark/bloom`.
- Remove accidentally committed build artifacts, demo scripts and benchmark
  outputs; extend `.gitignore` to keep them out.
- Replace local machine paths in documentation with repository-relative links.
- Add `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and populate
  `.github/CODEOWNERS`.
- Add README badges, `#![warn(missing_docs)]` on public crates, and a
  `pyproject.toml` for the Python SDK.
- Fix a cross-platform build break in `prefetch_file_madvise`
  (`posix_fadvise` is Linux-only; use `fcntl(F_RDADVISE)` on macOS).
- Fix `bloom_server` KV hook wiring to use `ServerKvHook` matching the
  per-request model map type.
