# Bloom

[![CI](https://github.com/constmark/bloom/actions/workflows/ci.yml/badge.svg)](https://github.com/constmark/bloom/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/bloomai-engine.svg)](https://crates.io/crates/bloomai-engine)
[![docs.rs](https://docs.rs/bloomai-engine/badge.svg)](https://docs.rs/bloomai-engine)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](rust-toolchain.toml)

Bloom is a standalone, multimodal inference engine written in Rust. It loads
local models directly, runs inference without an external scheduler, and
provides both a command-line interface and bounded OpenAI- and
Ollama-compatible HTTP APIs.

Bloom can run on its own or serve as an inference engine integrated into larger runtime systems.

> [!IMPORTANT]
> Bloom is pre-1.0 software. The default Candle paths are currently
> **experimental**. Other backends may require external runtimes or may only
> provide capability-detection skeletons. See the
> [support matrix](docs/support-matrix.md) before choosing a production path.

## Highlights

- Standalone local inference through `bloom_infer`
- OpenAI-compatible Responses, chat, completion, embedding, and reranking
  endpoints, including bounded Chat function tools with schema-validated calls
- Ollama-compatible discovery, signed-index verified pull, guarded model
  deletion, chat, generate, embedding, structured-output, and function-tool
  endpoints
- Streaming generation and an interactive terminal mode
- CPU, Metal, and CUDA execution through Candle
- GGUF and Hugging Face-style model package support
- Native BERT sentence embeddings with mask-aware bounded batching, mean
  pooling, and CPU-trained quality gates
- Safe local model catalog with load preflight, portable inventory export, transactional lifecycle, verified downloads, and resumable browser imports
- Pluggable engines, backends, processors, operators, and model packages
- Memory estimation, KV-cache management, in-flight batching, and CacheMesh
- Task-aware browser UI for streaming chat, embeddings, and bi-encoder reranking,
  with session-only credentials, keyboard-contained dialogs, and reduced-motion
  support, built with Dioxus and embedded in official release artifacts
- C ABI and a small Python SDK for downstream integrations

## Quickstart

### Release application

Official application archives contain the API, embedded browser UI, command-line
tools, documentation, per-binary checksums, and a target-specific quickstart.
GitHub release archives also carry signed build provenance verifiable with
`gh attestation verify ARCHIVE --repo constmark/bloom`. After provenance and
checksum verification, extract the archive and run:

```bash
./bloom_server --doctor
./bloom_server --open-browser
```

On Windows PowerShell, use the corresponding `.exe` commands:

```powershell
.\bloom_server.exe --doctor
.\bloom_server.exe --open-browser
```

The application binds to localhost, opens `http://127.0.0.1:3000/`, and can
start with an empty model catalog. Read the bundled `QUICKSTART.md` to add a
supported model or install the commands into `PATH`.

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) with the toolchain pinned in
  `rust-toolchain.toml`
- Git
- A compatible local model; start with the
  [support matrix](docs/support-matrix.md)

Clone and build Bloom:

```bash
git clone https://github.com/constmark/bloom.git
cd bloom
cargo build --release --bin bloom_infer --bin bloom_server
```

Run a local model with the default CPU backend:

```bash
cargo run --release --bin bloom_infer -- \
  --model /path/to/model.gguf \
  --prompt "Explain edge inference in one sentence." \
  --max-tokens 128 \
  --stream
```

Model paths may point to a single `.gguf` file or to a model directory. A
Hugging Face-style directory normally contains `config.json`, `tokenizer.json`,
and either `model.safetensors` or an indexed, canonically numbered Safetensors
shard set. Sharded packages must include `model.safetensors.index.json`; Bloom
rejects incomplete, unreferenced, symlinked, duplicate-tensor, or
header/index-mismatched shards before mmap loading. A Bloom model package
contains a `bloom.json` manifest. Use `--inspect` to validate routing and
estimate memory without loading the full model:

```bash
cargo run --release --bin bloom_infer -- \
  --model /path/to/model \
  --inspect
```

Maintainers can reproduce the pinned trained Qwen2 Q4_0, Qwen3 Q8_0,
SmolLM2 Llama-architecture Q8_0 and native BF16 Safetensors, plus MiniLM BERT
embedding CPU gates with:

```bash
./scripts/test_trained_qwen2_runtime.sh
./scripts/test_trained_qwen3_runtime.sh
./scripts/test_trained_llama_runtime.sh
./scripts/test_trained_safetensors_runtime.sh
./scripts/test_trained_embedding_runtime.sh
```

Each profile verifies the exact model revision, file size, SHA-256, and
Apache-2.0 license evidence. Text profiles test deterministic instruction
following, benchmark output, and exact buffered and streamed responses. The
91 MB MiniLM profile tests native 384-dimensional embeddings, semantic
separation, bi-encoder reranking, and fail-closed generation through both
OpenAI and Ollama paths. The primary model files are cached under `/tmp` by
default; no model weights are stored in this repository or in release archives. See the
[trained-model validation record](docs/trained-model-validation.md).

Useful CLI modes:

```bash
# Interactive terminal chat
cargo run --release --bin bloom_infer -- --model /path/to/model --interactive

# List engines compiled into the binary
cargo run --release --bin bloom_infer -- --list-engines

# Use Apple Metal or NVIDIA CUDA
cargo run --release --features metal --bin bloom_infer -- \
  --model /path/to/model --device gpu --prompt "Hello" --stream
cargo run --release --features cuda --bin bloom_infer -- \
  --model /path/to/model --device gpu --prompt "Hello" --stream
```

Run `cargo run --release --bin bloom_infer -- --help` for the complete CLI
reference.

## OpenAI-Compatible Server

Start the API-only server. This is the default build and does not require the
Dioxus toolchain or generated browser assets:

```bash
cargo run --release --bin bloom_server -- \
  --model /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 3000
```

The server can also start without an active model so the UI can select one
from a managed directory:

```bash
mkdir -p ~/.bloom/models
cargo run --release --bin bloom_server -- --models-dir ~/.bloom/models
```

The catalog scans supported direct children only. A switch temporarily closes
inference admission, drains active requests, builds the replacement runtime,
and publishes it only after every component is ready. If loading fails, the
previous runtime remains available and the error is reported by `/ready` and
the model-management API.

Verified downloads are opt-in. When enabled, the Models drawer and API accept
public Hugging Face HTTPS URLs for GGUF, ONNX, and Core ML files. Paste either a
repository `/blob/` or `/resolve/` file URL and choose `Inspect source`; Bloom
uses a bounded HEAD request to derive the filename, declared size, published
SHA-256, and immutable repository commit without transferring model bytes. The
user still confirms the download, every transfer still requires a SHA-256, and
missing hash metadata falls back to manual entry rather than an unverified
install. Bloom stages partial bytes outside the catalog, resumes with HTTP
Range, verifies the complete file, and installs it without overwriting an
existing entry:

```bash
cargo run --release --bin bloom_server -- \
  --models-dir ~/.bloom/models \
  --enable-model-downloads
```

Source inspection is also authenticated and returns a sanitized, commit-pinned
download URL when Hugging Face publishes the commit metadata:

```bash
curl -X POST http://127.0.0.1:3000/v1/model-management/downloads/inspect \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://huggingface.co/OWNER/REPO/blob/main/model.gguf"}'
```

The download operation remains available over the authenticated API. The
filename must be a direct catalog child and the checksum must contain 64
hexadecimal characters:

```bash
curl -X POST http://127.0.0.1:3000/v1/model-management/downloads \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://huggingface.co/OWNER/REPO/resolve/REVISION/model.gguf",
    "filename": "model.gguf",
    "sha256": "EXPECTED_SHA256",
    "license": "Apache-2.0"
  }'
```

Send `DELETE` to the same endpoint to cancel the active transfer. Partial data
is retained for later recovery and appears in the Models drawer after a server
restart. The `/downloads/resume` and `/downloads/discard` actions accept only a
catalog-safe filename; the source URL and checksum remain server-side.

Deployers can turn recorded license metadata into an acquisition allowlist with
`--allowed-model-licenses Apache-2.0,MIT` or
`BLOOM_ALLOWED_MODEL_LICENSES`. When configured, every new or resumed download
and every import publication must declare an exact case-insensitive match. The
GUI replaces free-form entry with the approved choices, while the server remains
authoritative for direct API clients. An empty allowlist preserves the
backward-compatible record-only behavior.

Operators can also configure a publisher-signed discovery index. Bloom accepts
one bounded local file or HTTPS source and one to eight trusted Ed25519 public
keys, rejects expired, tampered, unknown-signer, or rolled-back metadata across
restarts through a bounded persistent watermark, and shows only
immutable commit-pinned entries in the Models drawer. The private watermark
directory defaults next to the effective Bloom configuration file.
The bounded key overlap supports controlled publisher rotation. Selecting an
entry calls a server-authoritative endpoint by signed ID; the browser cannot
replace its source, destination, size, hashes, license, or file manifest.
The Models drawer compares each entry with the current catalog and labels an
exact signed acquisition as installed. A clean prior download carrying the same
persistent signed-index ID is shown as an available upgrade, including when the
new generation changes its catalog filename or moves between a single file and
a package directory. Bloom downloads and verifies the complete replacement
beside the still-usable old version, accounts for their peak combined storage,
rehashes both identities at commit, and then publishes through a private
crash-recoverable transaction. Loading, removal, and integrity work for the old
entry are blocked during that transaction; an already active model must be
unloaded first. Startup either restores the old entry or completes the verified
replacement after an interruption. A same-name entry without the exact alias,
duplicate aliases, stale or quarantined provenance, or an occupied new
destination remains a local conflict and receives HTTP 409. Exact installed,
in-progress download, and in-progress upgrade requests remain idempotent.
Version 1 entries install one verified model file. Version 2 entries can install
a complete Safetensors/Transformers directory from one exact repository commit:
Bloom bounds and resumes the hidden package stage, enforces aggregate quota,
verifies every signed file, validates indexed shard names against the actual
Safetensors index and tensor headers, and exposes a new directory only after one
atomic no-overwrite rename. Transactional upgrades preserve the previous entry
under `.bloom-upgrade` until replacement data and provenance are committed.
Each server also holds a non-blocking operating-system lease on
`.bloom-catalog.lock` for its complete process lifetime. A second Bloom server
cannot share that mutable catalog, including during startup recovery and Tokio
runtime teardown; use a different `--models-dir` instead. The empty lock file
may remain after shutdown because kernel ownership, not file existence, is
authoritative.
Unix builds also sync the verified tree,
provenance, and rename parents before reporting success. Entries outside the
server size or license policy stay
visible but disabled. See the
[signed model index guide](docs/model-index.md),
[payload schema](examples/model-index-payload.schema.json), and
[envelope schema](examples/model-index-envelope.schema.json). The authenticated
API has its own [normalized response schema](examples/model-index-response.schema.json).

```bash
cargo run --release --bin bloom_server -- \
  --models-dir ~/.bloom/models \
  --enable-model-downloads \
  --model-index-url https://models.example.org/bloom/index.json \
  --model-index-public-key TRUSTED_ED25519_PUBLIC_KEY
```

Browser-local imports are independently opt-in. They accept one `.gguf`,
`.onnx`, or `.mlmodel` file, require its expected SHA-256 digest, and transfer
bounded chunks rather than loading the whole model into browser or server
memory:

```bash
cargo run --release --bin bloom_server -- \
  --models-dir ~/.bloom/models \
  --enable-model-imports
```

The Models drawer can pause an import by cancellation and resume it by
selecting the same local file with the same size and checksum. Bloom retains
valid partial data under the catalog's private `.bloom-imports` staging
directory across server restarts, verifies the completed file, and publishes
it with a no-overwrite filesystem operation. The browser cannot retain a file
handle across a page reload, so the file must be selected again. A staged
entry can also be discarded from the drawer.

The authenticated wire protocol is:

- `POST /v1/model-management/imports` declares `filename`, `total_bytes`, and
  `sha256`; it may also include a public HTTPS `source_url` and a user-declared
  `license` or SPDX expression.
- `PUT /v1/model-management/imports/{filename}` appends a chunk with an `Upload-Offset` header.
- `POST /v1/model-management/imports/{filename}/complete` verifies and installs the file.
- `DELETE /v1/model-management/imports/{filename}` discards staged bytes.

Offset conflicts return the server's authoritative offset.

Successful downloads and imports write a versioned provenance record under the
catalog's private `.bloom-metadata` directory. The authenticated catalog API
and Models drawer show the acquisition path, verified SHA-256, optional source,
optional license declaration, and installation time. Bloom strips URL query
strings and fragments before persistence or display. These records describe
the verified acquisition; Bloom does not independently validate license claims
or retroactively infer provenance for files copied into the catalog manually.
Package provenance additionally records the exact canonical file manifest and
its aggregate digest.

The Models drawer can reverify any inactive acquired model. Single-file entries
recompute one SHA-256; package entries reject an altered tree and recompute
every signed file plus the aggregate package digest.
`POST /v1/model-management/integrity` accepts its
catalog `id`, returns immediately, and exposes progress through the catalog
response; `DELETE` on the same endpoint cancels the active check. Successful
checks and mismatches are written back to `.bloom-metadata`. A recorded
mismatch survives server restarts and blocks later load attempts until the file
passes verification. Active models must be unloaded or switched away first.

The drawer polls authenticated `GET /v1/model-management/models`. Successful
responses use the strict version 1 `bloom.model_catalog` contract and include
the configured catalog location, recognized models, active/load state,
acquisition capabilities, signed-index trust identity, storage accounting, and
integrity progress. The browser rejects missing, partial, unknown-field, or
unsupported-version documents instead of interpreting them as disabled
features. See the [contract](docs/model-catalog.md),
[JSON Schema](examples/model-catalog.schema.json), and
[example response](examples/model-catalog.json).

The `Review` action performs an authenticated load preflight without loading
model weights. `POST /v1/model-management/preflight` accepts a catalog `id`
and returns the strict version 1 `bloom.model_preflight` document: a bounded
manifest summary, the trusted `generation` or
`embedding` plus `rerank` task identity, the selected engine and its maturity,
device compatibility, planned context, and the same conservative memory budget
used by the loader. The browser requires this review before enabling `Load` and
rejects missing, unknown, unsupported-version, or model-mismatched reports.
Results are cached briefly using a file/manifest fingerprint. Catalog switches
run this check again and reject models with an
unavailable device, unsupported or skeleton engine, incompatible format, or an
exceeded memory budget before entering the load queue. A successful preflight
is still advisory: model payloads and external runtimes can fail later, so the
transactional loader remains authoritative. See the
[contract](docs/model-preflight.md), [JSON Schema](examples/model-preflight.schema.json),
and [example response](examples/model-preflight.json).

`GET /v1/model-management/inventory` exports the complete catalog as stable,
versioned JSON for audit and backup. The document is sorted by catalog ID and
omits the local catalog root, absolute paths, active runtime state, modification
times, and source URL secrets. It includes acquisition checksums, license
declarations, integrity quarantine, and whether a Hugging Face source is pinned
to an exact commit. The Models drawer downloads the same document with
`Export JSON`. `Compare JSON` uploads a bounded version `1` inventory for strict
server-side validation and a read-only drift preview. For a missing model with
a verified exact-commit download record, an operator can explicitly queue one
no-overwrite `Restore` through the existing SHA-256-verified download pipeline.
See the [model inventory format](docs/model-inventory.md), its
[inventory JSON Schema](examples/model-inventory.schema.json), and the
[reconciliation response schema](examples/model-inventory-reconciliation.schema.json).

For writable catalogs, `--max-model-storage-bytes` applies one shared budget
to installed model data, partial downloads, uploaded import bytes, the
remaining size declared by staged imports, and the remaining size reserved by
active downloads. The default `0` leaves the total budget unlimited; per-file
limits still apply. Multi-file package staging is recursively counted across
restarts, including completed and partial nested files. `--staged-model-retention-seconds`
optionally removes inactive single-file and package download/import staging at
startup and periodically thereafter. Active package directories and metadata
remain protected as one session, and cleanup never follows staging symlinks.
Its default `0` disables automatic deletion. Current usage, commitments,
available capacity, and the last cleanup result are visible in the Models
drawer and the model catalog response.

Inactive catalog entries can be permanently removed from the Models drawer or
with `POST /v1/model-management/remove`. Removal accepts only a discovered
direct-child ID, is blocked during model loading, and refuses to remove the
active runtime. Single-file removal also cleans its provenance record. The
operation is intentionally irreversible.

Send a chat request:

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello from Bloom"}],
    "max_tokens": 64,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

The active runtime can be listed with `GET /v1/models` and retrieved with
`GET /v1/models/{model}` using its exact ID or the `default` alias. The latter
always returns the real runtime ID and never loads an inactive catalog entry.

The `model` field is optional for backward compatibility. When present on chat,
legacy completion, embedding, or reranking requests, it must be `default` or the
exact active identifier returned by `/v1/models`; Bloom returns HTTP 404 with
`model_not_found` instead of silently running a different model. `default` is
an alias for the single active runtime, and response metadata always reports
the identifier that actually performed the work. The browser also binds text
and multipart image requests to the model reported by readiness at send time.

Chat request shape is admitted before runtime work. Both server and browser
accept at most 2,048 messages, 768 KiB of combined message content, 262,144
characters in one user message, and 65,536 characters in one system message.
Direct API clients may use `developer`, `system`, `user`, and `assistant` roles
and send each `content` as a string or as one to 256 ordered OpenAI text parts;
leading `developer` messages map explicitly to local system instructions, while
late developer messages and non-text parts fail explicitly. The browser
additionally caps the encoded chat JSON at 1 MiB. These are explicit errors,
not truncation. Raising the server's general JSON body limit does not disable
the semantic chat limits.

Text chat and legacy completions accept `stop` as one non-empty string or an
array of up to four exact strings. Each sequence is limited to 1,024 characters
and the combined wire value to 16 KiB. Bloom withholds possible partial matches
across streaming chunks, excludes the matched sequence and all later text from
the response, and ends blocking or continuously batched generation as soon as
a match is confirmed. The browser exposes the same control as a JSON-array
setting. Stop sequences are currently text-only; image chat rejects an active
setting before changing conversation state.

Chat Completions supports bounded OpenAI function tools. Requests may provide
up to 32 `type: "function"` definitions, use `tool_choice` values `none`,
`auto`, `required`, or one named function, and permit up to eight calls when
`parallel_tool_calls` is enabled. Bloom returns standard assistant
`tool_calls`, accepts paired assistant-call and `role: "tool"` result history,
and emits streaming calls through `delta.tool_calls`. The server never executes
functions. It asks the local model for a private JSON control envelope, buffers
tool-enabled streams until that envelope is complete, then validates every
selected name and arguments object against the declared bounded JSON Schema
before exposing it. Invalid model output fails with `invalid_tool_call` instead
of becoming an executable call. Function quality therefore depends on the
loaded model's instruction-following ability; custom and built-in tool types,
the deprecated `functions`/`function_call` interface, and combining active
tools with `response_format` remain unsupported.

All generation routes admit `max_tokens` from 1 through 32,768. Chat also
accepts the current `max_completion_tokens` spelling; if both are sent, their
values must match. The legacy completion route accepts one nonblank prompt of
at most 262,144 characters and 768 KiB. Embeddings accept 1 through 256 nonblank
strings, each at most 262,144 characters and together at most 768 KiB; only
finite, L2-normalized float output is currently supported, with optional
dimensionality reduction from 1 through 16,384. Native BERT encoders execute
bounded padded microbatches with attention-mask-aware mean pooling and restore
the original input order; other embedding adapters keep their scalar fallback.
Embedding batches retain cancellation, concurrency admission, and request
accounting until their blocking worker exits. Reranking accepts a nonblank query
of at most 65,536 characters and 1 through 256 nonblank documents, each at most
262,144 characters, with 768 KiB available to the query and documents together.
If supplied, `top_n` must be between one and the submitted document count.
Reranking shares the embedding worker lifecycle, returns finite cosine scores
from -1 through 1, orders ties by original document index, and can include the
selected source documents. These checks run before runtime availability,
inference admission, or request accounting.

The public JSON multimodal route accepts 1 through 3 distinct inline blocks:
at most one `Text`, one `AudioPcm`, and one JPEG or PNG `Image`. Text has the
same 262,144-character and 768 KiB ceiling, image bytes are limited to 10 MiB,
and normalized finite PCM is limited to 8--48 kHz, 4.8 million samples, and 600
seconds. Server-local
`AudioFile` paths and internal `Tokens`, `Tensor`, `VideoFrames`, `WorldState`,
or `Action` blocks are rejected. Structured response formats remain text-route
only. The configurable JSON body limit is an additional transport ceiling, not
a replacement for these semantic limits.

For streaming chat, `stream_options.include_usage` emits an OpenAI-shaped final
usage chunk with empty `choices` immediately before `[DONE]`. It reports prompt,
completion, and total token counts; clients that omit the option keep the
previous stream shape. Every chat stream now begins with an assistant-role
chunk and retains one creation timestamp across all chunks.

The browser validates the bounded `model` metadata on every text and multimodal
stream event. It rejects a response that names a different model or changes
models mid-stream, then records and displays the confirmed execution model with
the assistant response. If the active model later changes, Send, Retry, and Edit
remain unavailable until the user starts a new chat or explicitly confirms
sending the existing history to that exact model. Version 2 portable
conversation archives retain this bounded model provenance while continuing to
exclude timing, token usage, request IDs, credentials, and generation settings;
legacy version 1 imports remain supported with unknown model provenance.

Streaming request IDs are also limited to 128 ASCII letters, digits, hyphens,
or underscores and must remain constant for the complete response. Bloom
confirms both the model and request ID before accepting content. The UI validates
and URI-encodes the ID again before sending an authenticated cancellation POST,
and the server rejects malformed cancellation path values with HTTP 400.

The browser admits streaming responses only when their media type is
`text/event-stream`. It limits the decoded transport to 64 MiB, each SSE frame
to 1 MiB, streamed error text to 4 KiB, and accumulated generated text to
16 MiB. Text and multimodal streams must finish with an explicit `[DONE]`
event; EOF alone is treated as a truncated failure, preserving any bounded
partial text without labeling it complete.

Server-side request ownership lasts until the response completes and execution
has exited, not merely until headers are returned or an HTTP handler is dropped.
The concurrency permit, cancellation registration, and in-flight metrics
therefore remain live for streaming and non-streaming text generation as well
as multimodal generation. Disconnects and server-side HTTP timeouts cancel
scheduled or cooperatively cancellable blocking execution, remove IFB token
senders, record one failed completion, and release admission only after any
blocking worker has drained.

Process shutdown accepts `Ctrl-C` on every supported host and `SIGTERM` on
Unix. Bloom immediately reports not ready, stops accepting new connections,
and gives existing HTTP requests a bounded drain window. Configure the window
from 1 through 3,600 seconds with `--shutdown-timeout-seconds` or
`BLOOM_SHUTDOWN_TIMEOUT_SECONDS`; the default is 30 seconds. If the window
expires, Bloom logs the failure and exits with status 1 instead of leaving a
container indefinitely stuck in termination. Send a second shutdown signal to
skip the remaining drain window and force the same non-zero exit immediately.

`/ready` reports the active model's per-request `context_window` when a runtime
is loaded. Before chat or completion inference is admitted, Bloom validates
sampling controls, tokenizes the fully formatted prompt, and requires the prompt
plus `max_tokens` reservation to fit that window. An oversized request returns
HTTP 400 with `context_length_exceeded`; Bloom does not silently discard earlier
messages. The legacy completions endpoint currently accepts one string prompt
or a one-item string array and explicitly rejects batched prompt arrays.

Text endpoints accept bounded `json_object` and fail-closed `json_schema`
response formats through Chat/Completions `response_format` and Responses
`text.format`. Unknown schema constraints are rejected rather than ignored;
the scheduler-driven Candle path applies token grammar from each request's
authoritative generated-token history across prefill and native batched decode.
Structured requests do not use unconstrained speculative token runs. See the
[structured-output guide](docs/structured-output.md) for the supported subset,
limits, streaming behavior, and UI controls.

Chat and legacy completion requests also capture fields outside Bloom's
supported OpenAI subset. Exact no-op defaults such as `n: 1`, empty `tools`,
`tool_choice: "none"`, and zero penalties remain compatible; any
unknown or unsupported semantics fail with HTTP 400 before runtime admission
instead of being silently ignored. Multiple choices, log probabilities,
penalty/logit-bias sampling, custom tools, and the deprecated function-call
request fields are not yet implemented. See the
[OpenAI compatibility guide](docs/openai-compatibility.md) for the precise field
matrix and error behavior.

Bloom also provides an experimental, bounded Ollama-native surface at
`/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/pull`,
`DELETE /api/delete`, `/api/chat`, and `/api/generate`, plus current
`/api/embed` and legacy `/api/embeddings`. It supports guarded inactive-model
deletion, exact-ID acquisition through an operator-trusted signed index, text
streaming as NDJSON, structured output, function tools, and bounded normalized
embedding batches. A chat, generate, or embedding request automatically loads
an exact inactive catalog selector; a signed pull persists its index ID as the
stable Ollama selector used by tags, show, inference, and delete. Matching
concurrent requests join one sequenced load result, while a different lifecycle
operation returns a conflict. Empty chat/generate requests support preload and
`keep_alive: 0` unload. Bounded positive durations schedule automatic unload
after the response or stream completes, refresh on newer Ollama activity, and
are published through `/api/ps`; omitted values use Ollama's five-minute
default and negative values remain resident indefinitely. Bloom retains its
one-active-runtime and verified-acquisition boundaries: pull itself does not
immediately load, and registry pull, create, copy, and push workflows are
intentionally not emulated. Unsupported options fail instead of being ignored.
See the
[Ollama compatibility guide](docs/ollama-compatibility.md) for the endpoint and
field matrix.

`POST /v1/responses` offers a modern SDK-shaped adapter for bounded text
generation and function calling. It accepts `instructions`, string,
`input_text` message, `function_call`, and string `function_call_output` input,
`max_output_tokens`, `temperature`, `top_p`, and `stream: true`. Flat function
definitions, `none`/`auto`/`required`/named selection, up to eight parallel
calls, schema-validated native call items, and native function-argument stream
events reuse the same bounded fail-closed function layer as Chat Completions.
Responses streams emit current named lifecycle, text-delta, item-completion,
usage, and terminal events with monotonic sequence numbers; their typed terminal
event replaces Chat's `[DONE]` marker. Explicit `store: true` enables bounded,
24-hour, process-local retention; `previous_response_id` continues same-model
history, including outstanding function calls and matching results, without
carrying prior top-level instructions or metadata. Bounded
response metadata is preserved without entering the model prompt. Current
SDK-shaped retrieve, delete, and cursor-paged input-item endpoints are available. Bloom
defaults to no retention, never writes this state to disk, and clears it on
restart. Bloom never executes functions. Custom or hosted/built-in tools,
background work, automatic truncation, image/file tool results, and unsupported
content return HTTP 400 instead of being silently discarded.

The server exposes:

- `/health`, the versioned `/ready` UI/server handshake, and `/metrics`
- `/v1/models`, `/v1/models/{model}`, and the versioned `/v1/observability` diagnostics snapshot
- `/v1/model-management/models`, `/index*`, `/inventory`, `/inventory/reconcile`, `/inventory/restore/{id}`, `/preflight`, `/switch`, `/unload`, `/remove`, `/integrity`, `/downloads*`, and `/imports*`
- `/v1/responses`, `/v1/responses/{response_id}`,
  `/v1/responses/{response_id}/input_items`, `/v1/chat/completions`, and
  `/v1/completions`
- `/v1/multimodal/stream` (bounded inline JSON) and `/v1/multimodal/upload` (bounded multipart image upload)
- `/v1/embeddings` and `/v1/rerank`
- `/v1/backends` and `/v1/kv-cache-stats`
- `/v1/cancel/{request_id}`
- `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/embed`, and `/api/embeddings`

Every HTTP response, including health probes, authentication failures, rejected
requests, and unknown routes, carries `x-request-id`. Bloom preserves an
incoming value only when it contains 1 to 128 ASCII letters, digits, hyphens,
underscores, dots, or colons; otherwise it replaces the value with a UUID. The
header is exposed through CORS so browser clients can include it in support
reports. It is an HTTP correlation value, not a generation cancellation ID or
an authorization credential. On non-successful HTTP responses, the Bloom UI
appends a validated copy of this value to its error banner so an operator can
match the failure to server or proxy logs without exposing an unsafe header.

Transient inference-capacity failures return HTTP 429 with `Retry-After`. Bloom
adds a one-second delta only when the handler did not provide a more specific
standard hint, and exposes the header through CORS. The UI displays only strict
1-to-300-second values on 429 errors; it never automatically replays a prompt or
model-management request.

Every response under `/v1` or `/api`, plus `/health`, `/ready`, and `/metrics`,
is authoritatively marked `Cache-Control: no-store`. The policy covers normal
JSON, SSE and NDJSON streams, authentication and admission failures, timeouts,
body-limit responses, and unknown protocol routes. Embedded UI assets remain a
separate static-content policy.

Unknown `/v1` routes return OpenAI-shaped JSON 404 errors and unknown `/api`
routes return Ollama-shaped JSON 404 errors. Calling a known route with the
wrong method returns a matching JSON 405 while preserving `Allow`. These fixed,
non-reflective fallbacks remain separate from authentication, and the embedded
SPA never replaces them with HTML.

Malformed JSON, endpoint-schema mismatches, unsupported media types, oversized
bodies, and multipart extraction failures are normalized at the same outer
protocol boundary. Bloom preserves meaningful 400/413/415/422 statuses and safe
headers while replacing framework text with bounded, non-reflective OpenAI or
Ollama error JSON. Static UI responses stay outside this normalization.

`/ready` is also a fail-closed compatibility boundary. Its v3 contract publishes
the Bloom object identity, readiness schema version, server protocol, explicit
minimum and maximum supported UI protocols, package version, bounded admission
state, and the active model's `generation`, `embedding`, or `rerank` tasks. The
browser accepts its protocol only inside that validated inclusive range,
distinguishes an unreachable server from a reachable but incompatible endpoint,
and never presents an encoder-only model as chat-capable. See the
[readiness contract](docs/readiness-contract.md) and its
[JSON Schema](examples/readiness.schema.json).

For anything beyond localhost, set `BLOOM_API_KEY`, restrict
`BLOOM_CORS_ALLOW_ORIGIN`, and protect health and metrics endpoints with a
reverse proxy or network ACL. See the [production guide](docs/production.md)
and [security policy](SECURITY.md).

Protected OpenAI and Ollama routes accept `Authorization: Bearer ...` or
`X-API-Key`. A rejected credential returns the protocol's JSON 401 plus
`WWW-Authenticate: Bearer realm="Bloom"`, a correlation ID, and `no-store`;
CORS exposes the challenge. The UI validates `/ready` first and then probes the
bounded protected Models API, so a missing or invalid key becomes an explicit
**API key required** state instead of a false successful connection.

Browser access is same-origin by default. Bloom validates a single `Origin`
before CORS or routing, permits the embedded HTTP origin, and rejects malformed,
opaque, cross-origin, and loopback DNS-rebinding requests with HTTP 403. CLI and
SDK requests without `Origin` are unchanged. A separately hosted UI must set
`BLOOM_CORS_ALLOW_ORIGIN` to its exact HTTP(S) origin; `*` is an explicit,
doctor-visible development escape hatch and is incompatible with strict
security.

## Configuration

Bloom reads `~/.bloom/config.json` by default. Override the path with
`BLOOM_CONFIG` or `--config`, and generate an example with:

```bash
cargo run --bin bloom_infer -- --init-config
```

Explicit command-line arguments take precedence over the configuration file.
Common environment variables include:

| Variable | Purpose |
| --- | --- |
| `BLOOM_CONFIG` | Runtime configuration path |
| `BLOOM_API_KEY` | API key required by `/v1/*` and `/api/*` routes |
| `BLOOM_OPEN_BROWSER` | Open the embedded local UI after the listener is ready |
| `BLOOM_SHUTDOWN_TIMEOUT_SECONDS` | Maximum graceful HTTP drain before forced exit (default 30 seconds) |
| `BLOOM_MODELS_DIR` | Root scanned by the authenticated model-management API |
| `BLOOM_MAX_UPLOAD_BYTES` | Multipart request limit (default 12 MiB) |
| `BLOOM_ENABLE_MODEL_DOWNLOADS` | Enable authenticated, verified downloads from trusted hosts |
| `BLOOM_MAX_MODEL_DOWNLOAD_BYTES` | Per-acquisition download limit, including an entire signed package (default 20 GiB) |
| `BLOOM_ENABLE_MODEL_IMPORTS` | Enable authenticated, resumable browser-local model imports |
| `BLOOM_MAX_MODEL_IMPORT_BYTES` | Per-file import limit (default 20 GiB) |
| `BLOOM_MAX_MODEL_IMPORT_CHUNK_BYTES` | Per-request import chunk limit (default 8 MiB) |
| `BLOOM_ALLOWED_MODEL_LICENSES` | Comma-separated acquisition license allowlist (default empty = record only) |
| `BLOOM_MAX_MODEL_STORAGE_BYTES` | Shared installed/staged model commitment limit (default 0 = unlimited) |
| `BLOOM_STAGED_MODEL_RETENTION_SECONDS` | Inactive staging retention before automatic cleanup (default 0 = disabled) |
| `BLOOM_CORS_ALLOW_ORIGIN` | Browser origin policy: `same-origin` (default), one exact HTTP(S) origin, or explicit `*` |
| `BLOOM_MEMORY_UTILIZATION` | Fraction of available memory usable at startup |
| `BLOOM_STRICT_MEMORY_BUDGET` | Fail before loading when the estimate exceeds the budget |

Inspect the effective configuration and host capabilities without loading a
model, creating storage, or binding a port:

```bash
cargo run --locked --bin bloom_server -- --doctor
cargo run --locked --bin bloom_server -- --doctor=json
```

Warnings describe an incomplete or risky deployment; blocking failures exit
non-zero. See the [server doctor contract](docs/server-doctor.md).

`--max-concurrent` must be at least one and cannot exceed Tokio's semaphore
capacity for the build target. Normal startup and `--doctor` apply the same
check before storage mutation, background work, or listener binding, and report
the exact platform limit instead of allowing semaphore construction to panic.

## Model and Backend Support

Bloom separates executable, external-runtime, experimental, and skeleton
paths. The current source of truth is the
[model and backend support matrix](docs/support-matrix.md).

| Path | Status | Notes |
| --- | --- | --- |
| Candle on CPU | Experimental | Default development and CI path |
| Candle on Metal | Experimental | Build with `--features metal` |
| Candle on CUDA | Experimental | Build with `--features cuda` |
| OpenVINO and ASR bridges | External runtime | Require additional runtimes or Python packages |
| ONNX Runtime | Skeleton | Inspection and capability diagnostics only |

Do not infer production readiness from the presence of an engine adapter. A
path is promoted only after reproducible real-model validation and benchmark
evidence are recorded.

## Native macOS Client

Bloom includes a native SwiftUI client for macOS 13 and newer. It connects
directly to a local or remote `bloom_server`, validates the readiness v3
contract, and streams OpenAI-compatible chat completions without loading a Web
UI or embedding a browser runtime. API keys remain in memory and the server URL
is saved as a local preference.

Start a Metal backend, then build and open the desktop client:

```bash
cargo run --release --features metal --bin bloom_server -- \
  --model /path/to/model \
  --device gpu

just desktop-run
```

The application bundle is written to `target/macos/Bloom Desktop.app`. Run the
native parser and readiness tests with `just desktop-test`.

## Web UI

The optional Dioxus UI can run separately or be embedded in `bloom_server`. It
uses the readiness task contract to select either streaming chat or a bounded
embedding and bi-encoder reranking workspace. Encoder results are validated for
model identity, vector shape and normalization, finite values, document identity,
and stable ranking order before rendering compact summaries. Complete vectors
and ranked results can be copied explicitly or downloaded with their original
input association in a bounded, versioned JSON format. Model review exposes
that same task identity before loading, and changing the active model remounts
the task workspace so results cannot be mistaken for output from its successor.
The chat workspace
supports browser-local conversations, safe assistant Markdown, per-response
generation diagnostics, guided first-run recovery, a live exportable runtime
diagnostics drawer, guarded model lifecycle operations, and accessible keyboard
focus across drawers and dialogs. Any historical message can be forked
into a bounded, independently persisted conversation branch, while large
histories render incrementally instead of mounting every message at once.
Versioned conversation archives can be merged with fresh local IDs without
replacing current history, or restored through an explicitly destructive
Replace all choice. Version 2 preserves assistant model provenance so restored
histories retain explicit cross-model continuation checks. In an embedded application build, the SPA fallback is
limited to extensionless `GET` requests that explicitly accept HTML. Unknown
`/v1/*` and `/api/*` routes, missing assets, non-browser requests, and unsafe
methods retain real HTTP 404 behavior instead of returning the app shell.
Install the Dioxus CLI version used by the locked UI dependencies, then run:

```bash
cargo install dioxus-cli --version 0.7.10 --locked

# Build one binary containing the UI and API, then open http://127.0.0.1:3000/
just app /path/to/models

# Terminal 1: API server
cargo run --bin bloom_server -- --models-dir /path/to/models \
  --cors-allow-origin http://127.0.0.1:8080

# Terminal 2: UI development server at http://127.0.0.1:8080
just ui-dev
```

See [ui/README.md](ui/README.md) for standalone and single-binary deployment.

## Documentation

See the [documentation index](docs/README.md). Start with the
[architecture](docs/architecture.md), [support matrix](docs/support-matrix.md),
or [production checklist](docs/production.md).

Public JSON schemas and examples live under `examples/`. Validate them with:

```bash
./scripts/validate_json_artifacts.py
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `crates/core` | Shared types, manifests, scheduling, memory, world-state, and plugin contracts |
| `crates/backend` | Device probing, backend traits, and backend registry helpers |
| `crates/engine` | Model loading, inference engines, pipelines, native CLI tools, and CacheMesh |
| `crates/server` | HTTP application layer, model lifecycle, protocol adapters, and optional UI embedding |
| `crates/tilelang` | TileLang kernel compilation and loading |
| `crates/ffi` | Stable C ABI for native consumers |
| `python` | Python SDK bindings |
| `clients/macos` | Native SwiftUI macOS client |
| `ui` | Optional Dioxus web interface |
| `docs` | Architecture, operations, support, and roadmap documentation |
| `examples` | Schemas, manifests, plugins, and integration examples |
| `scripts` | Validation, smoke-test, benchmark, and external-runtime helpers |

## Development

Install [just](https://github.com/casey/just) for maintained command shortcuts,
or run the underlying commands directly:

```bash
cargo fmt --all -- --check
cargo check --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings -A missing_docs
cargo test --workspace --locked
cargo test --manifest-path ui/Cargo.toml --locked
./scripts/test_tiny_model_runtime.sh
./scripts/validate_json_artifacts.py
python3 -m pip install -r requirements/compat-smoke.txt
./scripts/openai_compat_smoke.py --api-key local-smoke --require-openai-sdk
./scripts/ollama_compat_smoke.py --api-key local-smoke --require-ollama-sdk
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements and
[RELEASE.md](RELEASE.md) for release gates.

## Community and Security

- Use the structured GitHub issue forms for reproducible bugs, focused feature
  proposals, and evidence-backed model or backend requests. Reports must be in
  English and must not contain credentials, private paths, model data, prompts,
  or responses.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License

Bloom is licensed under the [Apache License 2.0](LICENSE).
