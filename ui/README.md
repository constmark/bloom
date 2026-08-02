# Bloom UI

Bloom UI is a [Dioxus](https://dioxuslabs.com) client for `bloom_server`.
It currently compiles to a WebAssembly application and can later share the same
codebase with a desktop client.

The UI is fully decoupled from the backend. It communicates only through the
OpenAI-compatible HTTP API, including streaming `/v1/chat/completions`, plus
Bloom's readiness and model-management endpoints. Deploy it as a standalone
static site or embed it in the server binary.

## Features

- Streaming chat with reliable LF/CRLF event parsing and server error handling
- Per-response prompt/output tokens, request time, TTFT, throughput, and outcome metadata
- Safe assistant Markdown for headings, lists, tables, links, and code blocks
- Plain-text copy controls for user and assistant messages
- Guarded regeneration of the latest text response with failure rollback
- Transactional editing and resubmission of the latest text prompt
- Bounded branching from any historical message without changing the source conversation
- Explicit recent-context continuations that preserve the complete source conversation
- Incremental rendering of large histories in deterministic 100-message windows
- Stop generation through browser aborts and server-side request cancellation
- Multiple browser-local conversations with automatic titles and guarded deletion
- Case-insensitive conversation search across titles and message text, plus safe renaming
- Versioned, bounded conversation backup with transactional merge or confirmed full-history restore
- Configurable server URL and API key with session-only storage by default and explicit opt-in persistence
- Versioned UI/server readiness handshake with explicit unavailable and incompatible states
- Task-aware generation or embedding/rerank workspace selection from readiness v3
- Bounded embedding batches with optional dimensions and compact normalized-vector summaries
- Bounded bi-encoder reranking with exact document and stable-order response validation
- Complete per-result clipboard copy and versioned embedding/rerank JSON downloads
- Authenticated bounded Models probe with an explicit API-key-required state
- Connection test and live readiness/loading status with the validated server version
- Connection-aware first-run guidance for authentication, offline, incompatible, empty, loading, and failed states
- Live runtime diagnostics with bounded validation and credential-free JSON export
- Server-side local model catalog with load, safe switch, failure state, unload, and guarded permanent removal
- Opt-in resumable Hugging Face downloads with mandatory SHA-256 verification
- Searchable publisher-signed model discovery with bounded periodic and keyring-change refresh, visible persistent rollback protection, and immutable-source handoff to the verified download form
- Opt-in resumable browser-local model imports with bounded chunks and mandatory SHA-256 verification
- Shared catalog storage usage, reservation, quota, and stale-cleanup visibility
- Verified acquisition provenance with optional source and license declarations
- Cancellable on-demand SHA-256 re-verification with durable mismatch quarantine
- JPEG/PNG image attachments with bounded multipart upload, streaming output, and cancellation
- Persistent system prompt, `max_tokens`, `temperature`, `top_p`, seed, and bounded stop-sequence controls
- Persistent Text, JSON object, and bounded JSON Schema response modes
- Persistent message history, empty states, and actionable error banners with
  validated HTTP request IDs for log correlation
- Versioned browser persistence with legacy migration and corruption recovery
- Labelled, keyboard-contained drawers and dialogs with Escape dismissal, focus restoration, and reduced-motion support
- Responsive layout for desktop and mobile browsers

## Development

On Debian or Ubuntu, install the native dependencies required to compile the
Dioxus CLI:

```bash
sudo apt-get install libssl-dev pkg-config
```

Install a [`dx`](https://github.com/DioxusLabs/dioxus/tree/master/packages/cli)
version matching the locked UI dependencies:

```bash
cargo install dioxus-cli --version 0.7.10 --locked
```

Start the backend and UI in separate terminals:

```bash
# Terminal 1: standalone backend
cargo run --bin bloom_server -- --models-dir /path/to/models \
  --cors-allow-origin http://127.0.0.1:8080

# Terminal 2: UI development server at http://127.0.0.1:8080
just ui-dev
```

The UI connects to `http://127.0.0.1:3000` by default. Change the base URL in
Settings when the backend is hosted elsewhere.

The embedded application uses Bloom's default `same-origin` browser policy. A
UI served by the Dioxus development server or another static host is
cross-origin and requires `bloom_server --cors-allow-origin ORIGIN`, where
`ORIGIN` is that host's exact HTTP(S) origin without a path. The server rejects
unconfigured, opaque, malformed, and loopback DNS-rebinding origins before any
API route. Do not use `*` outside an explicitly reviewed development setup;
strict security rejects the wildcard.

The configured endpoint must implement the v3 `bloom.readiness` handshake
returned by `/ready`. The UI validates its schema and behavioral protocol
versions, required bounded fields, active-model task set, and ready-state
invariants before treating the endpoint as Bloom. A reachable old, partial,
unrelated, or misrouted response is shown as **Incompatible Bloom server**,
separately from a network failure. See the
[readiness contract](../docs/readiness-contract.md).

After the public readiness document passes, the connection test and five-second
status poll issue a bounded authenticated `GET /v1/models` probe. A valid key or
an authentication-disabled server returns Bloom's empty-or-singleton Models
list. HTTP 401 becomes **API key required**, preserves the correlated error for
support, and links directly to Settings; malformed success data remains an
incompatible-server failure. The probe never loads, switches, or enumerates the
local model catalog.

The default policy for a new connection keeps its API key in this tab's
`sessionStorage` and does not write it into persistent `localStorage`. Select
**Remember API key in this browser** only when the browser profile and origin
storage are trusted; the key then remains in `localStorage` until it is cleared
or replaced. Connections saved by an older Bloom UI retain their existing key
and are marked as remembered, avoiding a surprise logout during migration.
Disabling the option moves the current key into session storage and removes it
from the persistent connection record. Conversation and diagnostics exports
never include either form of the credential.

The main workspace follows the current readiness state. It links offline users
directly to connection settings, sends an empty or failed runtime to the Models
drawer, and shows model-loading progress. A ready model advertising
`generation` opens chat. A ready model advertising `embedding` instead opens an
encoder workspace; `rerank` enables its ranking tab. The UI never sends a chat
request merely because some model is loaded.

Text chat settings accept up to four exact stop sequences as a JSON string
array, for example `["END", "\nUser:"]`. Each entry is limited to 1,024
characters and all entries to 16 KiB. Bloom stops before the first matching
sequence even when it spans streamed model deltas. Image chat rejects an active
stop setting during preflight and leaves the draft and conversation unchanged.

The Diagnostics drawer polls the authenticated, versioned
`/v1/observability` snapshot and presents server uptime, model/load state,
request and token counters, scheduling queues, memory observations, the startup
memory plan, KV cache, and CacheMesh. Its JSON export excludes the server URL,
API key, prompts, responses, conversation history, model paths, and raw loader
errors. See the [runtime diagnostics guide](../docs/runtime-diagnostics.md).

Every shared non-successful HTTP response path reads the CORS-exposed
`x-request-id` before consuming the response body. The UI appends the value to
the error banner only when it contains 1 to 128 ASCII letters, digits, hyphens,
underscores, dots, or colons. This support correlation value is not persisted,
exported, placed in a URL, or reused as the streamed generation ID sent to the
cancellation endpoint.

For HTTP 429 only, the same path reads `Retry-After` and displays a wait hint
when the value is a strict decimal from 1 to 300 seconds. HTTP-date values,
whitespace, signs, decimals, zero, larger delays, and hints on other statuses
are ignored. The value is guidance only: Bloom UI does not automatically retry
chat, model-management, or support requests.

The Models drawer lists recognized direct children of `--models-dir` (or
`BLOOM_MODELS_DIR`). It never submits arbitrary filesystem paths. Add a GGUF,
ONNX, or Core ML file, or a recognized model directory. Recognition means Bloom
can route the package; execution still follows the support matrix. You can
still pass `--model` to preload a model at startup.

The drawer accepts only the strict version 1 `bloom.model_catalog` response
from the authenticated server. Required fields, known phases and formats,
unique selectors, active-runtime ownership, capability state, storage sums,
integrity state, and response size are validated before display. Missing,
unknown, malformed, or unsupported-version data leaves the previous snapshot
unchanged and surfaces an error. See the published
[model catalog contract](../docs/model-catalog.md).

Signed discovery cards distinguish `Installed`, a verified update, and a local
conflict with the same bounded identity comparison used by the server. A clean
inactive download carrying the same signed-index ID can show `Upgrade signed
model`, including a filename or file/package-shape change. The previous model
remains available until the verified replacement commits; the source cannot be
loaded, removed, or reverified while that upgrade is active. Duplicate aliases,
an occupied destination, quarantine, and ambiguous local state disable the
action instead of guessing which entry to replace.

Each model card has a `Review` action. It asks the authenticated server to
inspect bounded manifest metadata and shows the trusted generation or
embedding/rerank tasks, architecture, precision or quantization, modalities,
context limits, selected engine maturity, device support, and the loader's
conservative memory estimate. `Load` remains disabled until a matching,
strictly validated version 1 `bloom.model_preflight` review succeeds. Missing,
unknown-field, or unsupported-version documents fail closed; a known preflight
failure keeps `Load` disabled,
while the server independently enforces the same guard for direct switch API
calls. Preflight does not read full model
weights into browser memory and does not guarantee that an external runtime or
the complete payload will load successfully. The published
[contract](../docs/model-preflight.md) includes a strict JSON Schema and example.

`Export JSON` downloads a versioned inventory for audit and backup. It includes
every recognized entry, acquisition SHA-256, license declaration, integrity
state, and exact-commit source-lock status. The server omits the catalog root,
absolute paths, transient runtime state, and source URL query strings or
fragments. `Compare JSON` sends a bounded inventory to the authenticated server
for strict validation and a deterministic missing, unexpected, and changed
preview. The comparison does not install, remove, or modify models. When
verified downloads are enabled, a missing single-file model with a recorded
exact-commit download source exposes `Restore`. Each restore requires a
confirmation and reuses the bounded, SHA-256-verified, no-overwrite download
pipeline; Bloom never performs a bulk restore. The format is documented in
[`docs/model-inventory.md`](../docs/model-inventory.md).

Start the server with `--enable-model-downloads` to expose the download form.
Downloads accept public HTTPS URLs on `huggingface.co` and its trusted CDN
hosts and install only single-file GGUF, ONNX, or Core ML models. Paste a
repository `/blob/` or `/resolve/` file URL and use `Inspect source` to fill the
safe filename, declared size, published 64-character SHA-256, and an immutable
commit URL through a metadata-only HEAD request. Inspection never starts the
download. If the source does not publish a SHA-256, the form stays blocked
until an independently obtained digest is entered. Cancellation retains the
partial file so the same URL, filename, and checksum can resume later. Set
`BLOOM_MAX_MODEL_DOWNLOAD_BYTES` to enforce an appropriate disk budget. Bloom
rediscovers valid partial downloads after restart; the Models drawer can resume
them without exposing stored source URLs, or permanently discard their bytes.
An optional license or SPDX expression is retained with the verified install.
If the server configures `BLOOM_ALLOWED_MODEL_LICENSES`, both acquisition forms
replace free-form license entry with the approved declarations and block until
one is selected. The server rechecks the policy when a staged transfer resumes
or an import is published, so a policy change cannot be bypassed with old
staging metadata. Selecting an approved license can resume otherwise matching
partial download bytes without restarting the transfer.

Start the server with `--enable-model-imports` to expose the local import form.
The UI accepts one `.gguf`, `.onnx`, or `.mlmodel` file and a 64-character
SHA-256 digest. It slices the browser `File` into chunks (4 MiB by default,
capped by `BLOOM_MAX_MODEL_IMPORT_CHUNK_BYTES`) and never materializes the full
file in WebAssembly or in one HTTP request. Set `BLOOM_MAX_MODEL_IMPORT_BYTES`
to match the catalog disk budget. Cancellation retains partial staging state;
select the same file again to resume. After a page reload the
browser requires the user to reselect the file because Bloom does not request
persistent filesystem access. Staged imports survive server restarts and can
be explicitly discarded from the drawer. Imports can include an optional
public HTTPS source URL and license declaration.

For models installed through either acquisition path, the drawer shows the
verified SHA-256, acquisition type, source host/link when available, and the
declared license. Query strings and fragments are removed from persisted source
links. Manually copied files are labeled `Provenance not recorded`; license
declarations are informational and are not verified by Bloom.

The `Verify` action is available for inactive single-file models with an
acquisition record. It streams server-side hashing progress without loading the
file into browser memory. A mismatch is persisted, shown as a model-card
warning, and disables loading until a later verification succeeds. Bloom
requires the active model to be unloaded or switched away before checking its
on-disk bytes.

The drawer reports installed, staged, reserved, and committed model bytes.
Configure `BLOOM_MAX_MODEL_STORAGE_BYTES` to prevent downloads and imports from
independently overcommitting one catalog; `0` keeps the shared quota disabled.
Configure `BLOOM_STAGED_MODEL_RETENTION_SECONDS` to remove inactive staged
sessions at startup and on a bounded periodic schedule. Automatic cleanup is
disabled by default and never removes a download currently marked active by
the server.

Removing a catalog model is separate from unloading it. The UI never offers
removal for the active entry, requires an explicit confirmation, and treats the
operation as irreversible. Bloom rejects removal while another model lifecycle
operation is running.

Image attachments are kept only for the active request; browser-local history
stores the file name and prompt, not the image bytes. The UI accepts one JPEG
or PNG up to 10 MiB. Keep the server's total multipart request limit above the
image size with `BLOOM_MAX_UPLOAD_BYTES`.

Assistant responses render a constrained Markdown subset as the text streams.
Raw HTML is displayed as text, Markdown images are reduced to their alt text,
and only `http`, `https`, `mailto`, and same-page fragment links become
clickable. External links open in an isolated tab. User and system messages
remain plain text. This boundary prevents model-generated text from injecting
active HTML or using the browser as an automatic image-request channel.
Every non-empty message has a copy action that writes its original plain text,
not rendered HTML, to the browser clipboard. Clipboard access requires a secure
context such as HTTPS or localhost and may require browser permission.

Settled assistant responses show the confirmed execution model, total request
time, and browser-observed time to first token. Text chat requests ask compatible
OpenAI endpoints for a final stream usage chunk, allowing Bloom to also show
server-reported prompt and completion tokens plus completion throughput. Servers
that omit usage and Bloom's current multimodal stream still retain timing without
inventing token counts. Partial responses are labeled `Stopped` or `Failed`.
These bounded diagnostics persist with browser-local history. Portable version
2 archives retain only the exact assistant execution model needed for
cross-model continuation safety; timing, token usage, request IDs, and the rest
of the runtime measurements remain excluded. Strict version 1 archives remain
importable with unknown model provenance.

Each generation snapshots the active model from readiness before mutating the
conversation and sends that identifier with both text and multipart image
requests. Bloom servers reject the request if a model switch wins the race, so
the UI cannot silently display output from a model other than the one shown when
the user sent the prompt. The stream parser also requires bounded model metadata,
rejects a mismatch or any mid-stream identity change, and stores the confirmed
model only after receiving it from the server.

When a conversation's most recently recorded execution model differs from the
active generation model, the composer, response retry, and prompt edit paths
fail closed. An inline warning offers **Start new chat** or an exact
**Continue with active model** acknowledgement. The acknowledgement is scoped to the
conversation, previous model, and current model; another switch requires a new
decision. Imported version 2 archives and local branches retain the same guard.

The request ID follows the same fail-closed stream boundary. It must use the
bounded ASCII cancellation alphabet, arrive before content, and remain unchanged
for the response. The Stop action revalidates and URI-encodes that ID before
constructing `/v1/cancel/{request_id}`, so an untrusted stream cannot turn the
configured API key into a credential for a different same-origin POST path.

Outbound request construction is bounded before conversation state changes or
`fetch`. Text chat admits at
most 2,048 messages, 768 KiB of combined content, 262,144 characters in one
user message, and 65,536 characters in one system message. The actual encoded
chat JSON must fit 1 MiB. A multimodal turn admits a formatted prompt of at most
1 MiB and one JPEG or PNG of at most 10 MiB with a bounded safe filename. Other
JSON control bodies have a 16 MiB ceiling. Settings constrain the HTTP(S) server
address and printable-ASCII API key, and invalid locally persisted connection or
generation settings fall back safely. The same pure preflight builds the exact
multimodal prompt or complete encoded text body before an initial send, retry,
or prompt edit can publish a placeholder. Rejected initial sends retain the
draft and attachment; rejected retries leave the prior response unchanged; and
rejected edits remain open with an inline error. Bloom never silently removes
history. Text-only requests also avoid constructing a duplicate multimodal
prompt.

Stream transport is bounded independently of generation settings. The client
requires `text/event-stream`, accepts at most 64 MiB of decoded response bytes,
1 MiB per event, 4 KiB per streamed error message, and 16 MiB of accumulated
generated text. Limits are enforced before callbacks update the conversation.
Both text and multimodal streams require `[DONE]`; a multimodal `End` chunk does
not substitute for transport completion, and an early EOF records the bounded
partial response as failed.

Non-streaming API reads are bounded independently. General browser control
responses have a 16 MiB maximum success budget. Source inspection is limited to
16 KiB; readiness, restore, and import responses to 64 KiB; runtime diagnostics
and model preflight to 256 KiB; the signed index to 512 KiB; reranking to 2 MiB;
and embeddings to 32 MiB with at most 1,048,576 total float values.
It treats `Content-Length` only as an early admission check, also counts raw and
decoded bytes, cancels an over-budget reader, limits error bodies to 64 KiB, and
shows no more than 4 KiB of normalized error detail. Typed success responses
must use `application/json` or `application/*+json`. Ordinary control requests
allow 120 seconds for response headers; body reads allow 30 seconds between
chunks and 300 seconds overall. Dropping an in-progress UI task aborts its fetch
or cancels its reader. Generation and model-import uploads retain explicit Stop
controls instead of inheriting these ordinary control-request assumptions.

Bloom readiness also reports the active model's context window. The composer
shows the latest server-reported turn usage against that window and warns when
the current maximum output reservation makes the next turn likely to overflow.
Settings reject an output limit that alone consumes the complete window. The
server remains authoritative: it tokenizes the fully formatted prompt and
returns an actionable error before inference if prompt plus response budget does
not fit. The UI and server never silently remove older messages.

The encoder workspace accepts at most 256 non-empty lines and 768 KiB of
combined embedding content. It optionally requests 1 through 16,384 dimensions
and requires the response to contain the exact active model, input count,
contiguous indices, one consistent width, finite values, L2-normalized vectors,
and consistent usage. It displays only each vector's width, norm, and first
eight values. Reranking accepts one bounded query, at most 256 one-line
documents, and a positive result count no larger than the document count. The
response must preserve the exact submitted document for every unique index and
use descending finite scores with stable index tie ordering.

Each vector card keeps a compact 160-character input preview and can copy its
complete JSON float array. Each ranking card can copy its index, score, and
document as JSON. Batch downloads retain the exact submitted input or query,
model, token usage, indices, and complete validated results under a versioned
object identity. Export encoding is independently bounded to 40 MiB for
embeddings and 4 MiB for reranking; clipboard payloads are capped at 1 MiB.
Results are not persisted automatically. See the
[encoder result export guide](../docs/encoder-result-export.md).

Settings can constrain text chat to a JSON object or the documented JSON Schema
subset. Active schemas are syntax-checked, bounded, and rejected on unknown
keywords before being saved or sent. Structured responses render as escaped
code rather than Markdown, and that rendering marker survives conversation
export/import without exporting the schema itself. Image attachment controls are
disabled while structured output is selected. See the
[`structured-output guide`](../docs/structured-output.md).

The latest assistant response can be regenerated with the currently selected
model, system prompt, and generation settings. Bloom replaces only that response
and reuses the preceding text history. If cancellation or a request error occurs
before any new text arrives, the prior response is restored; a partial new
response is retained. Regeneration is disabled when the current response used
an image because attachment bytes are intentionally not persisted. Older
browser-local image markers are detected conservatively for the same reason.

The latest text prompt can also be edited and resent from its message action.
Bloom updates only that final turn and regenerates through the same streaming
path. If no replacement text arrives, the original prompt, response, and
automatic conversation title are restored together. If an earlier request
failed before producing an assistant message, its user-only turn exposes both
`Retry` and `Edit` instead of forcing a duplicate prompt. Image prompts cannot
be edited and replayed without selecting the original image again.

Every message also exposes a `Branch` action. Bloom creates and selects a new
conversation containing history through that exact message while leaving the
source untouched. Branch titles are bounded and made unique, and creation fails
without mutation at the 1,000-conversation, 50,000-total-message, or ID-space
limits. Generation diagnostics and structured-output rendering metadata remain
with copied assistant messages. Image bytes are still intentionally absent, so
the success notice identifies branches whose image turns need reattachment
before replay.

Every user message after the first also exposes a `Continue` action. Bloom
creates a new conversation that starts at that selected user message and keeps
all later messages, while the complete source remains unchanged. This gives a
user an explicit way to omit older context when a model's context window is
nearly full; Bloom never guesses a cutoff or silently truncates submitted
history. Continuations preserve response metadata, use bounded unique titles,
and share the same storage, capacity, ID-space, and unavailable-image guards as
branches. Context-budget warnings point to this action when the previous turn
plus the configured output allowance may exceed the active model's window.

To keep imported and long-running histories responsive, the message list mounts
only the latest 100 messages initially. A labelled control reveals earlier
history in deterministic 100-message pages. Copy, edit, retry, branch, and
continue actions use absolute message positions, and request construction still
sees the complete active conversation; rendering windows never truncate stored
or submitted history.

The conversation sidebar exports a versioned JSON backup containing titles,
user messages, assistant messages, bounded assistant model provenance, ordering,
and the active selection. It
excludes connection credentials, generation settings, system prompts, request
IDs, and image bytes. Import validates an 8 MiB bounded archive, previews its
conversation and message counts, and then offers two explicit transactional
choices. **Merge** appends every archived conversation in archive order with
fresh collision-free local IDs while retaining current conversations and the
active selection; exact duplicate conversations remain separate. **Replace
all** writes the imported store before replacing browser-local history. Both
paths fail without visible mutation at the shared 1,000-conversation,
50,000-message, ID-space, validation, or storage limits. Merge is disabled while
unreadable saved history is recovery-locked, preventing a valid archive from
silently overwriting raw recovery data. See the
[`conversation archive format`](../docs/conversation-archive.md).

The sidebar search filters browser-local conversations by title and message
content without sending a query to the server. Renamed titles are normalized to
one line and limited to 80 characters. New-chat creation, selection, deletion,
branching, prompt insertion, renaming, archive merge, and archive replacement
all build a candidate store and write it before replacing reactive state. A
full or unavailable `localStorage` therefore does not leave visible navigation
or history ahead of persisted history, and a failed send keeps the current
draft and attachment available.

Conversation persistence uses a strict v2 envelope with an object identifier;
the previous unwrapped v1 store migrates automatically after successful
decoding. Bloom never silently overwrites malformed saved history. Instead, it
blocks conversation mutations, offers the original text as a recovery download,
and requires either a confirmed `Start fresh` action or a validated archive
import before writes resume. A full or unavailable `localStorage` produces a
visible warning rather than an empty-history illusion.

## Deployment

### Standalone static site

Build static assets and deploy `ui/dist/` to a static host, CDN, or web server:

```bash
just ui-build
```

The maintained build wrapper disables release DWARF before optimization,
rejects a silently failed `wasm-opt`, verifies the expected output files, and
publishes `ui/dist/` only after every check succeeds.

Configure the deployed UI to use the URL of your separately running
`bloom_server` instance. The static host must set an equivalent Content Security
Policy, clickjacking protection, MIME-sniffing protection, referrer policy, and
permissions policy; the embedded server headers do not apply to a separately
hosted build.

## Validation

The UI is a separate WebAssembly crate, so validate it explicitly:

```bash
rustup target add wasm32-unknown-unknown
just ui-check
just ui-test
just ui-clippy
```

Pure conversation-state, archive-merge, submission-preflight, history-window,
streaming-protocol, and modal keyboard-boundary tests run on the host. The UI
type check targets `wasm32-unknown-unknown`, matching production builds. Before
a release, validate initial focus, Tab and Shift+Tab
cycling, Escape dismissal, opener-focus restoration, accessible names and
descriptions, and reduced-motion behavior in every supported browser with the
target assistive technology.

### Embedded in bloom_server

Build the UI and embed it in a single server binary:

```bash
just server-ui
./target/release/bloom_server --models-dir /path/to/models --open-browser
```

Open `http://127.0.0.1:3000/` for the UI. The server continues to expose its API
under `/v1/*`. Embedded responses include a restrictive Content Security Policy
and headers that disable framing, MIME sniffing, referrer disclosure, camera,
microphone, and geolocation access.

The `serve-ui` feature is disabled by default and requires `ui/dist/` to exist
at compile time. `just server-ui` performs the required steps in order.
Official release archives and the Docker image enable this feature and verify
it through the side-effect-free [server doctor](../docs/server-doctor.md).
Browser launching is explicit and disabled by default, so headless services and
containers are unchanged. A missing operating-system launcher produces a
warning with the local URL and does not stop the server.

## Layout

```text
ui/
|-- Cargo.toml       # Independent WASM crate
|-- Dioxus.toml
|-- index.html       # Application mount point
|-- assets/style.css # Global styles
`-- src/
    |-- main.rs      # Components and application state
    |-- api.rs       # OpenAI client and SSE parser
    |-- chat.rs      # Conversation state and title management
    |-- markdown.rs  # Constrained assistant Markdown renderer
    `-- storage.rs   # Browser-local settings and history persistence
```

`ui/dist/` and `ui/target/` are generated artifacts and are not committed.
