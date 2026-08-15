# Release Checklist

Use this checklist before tagging a release or marking a model/backend path as
`stable`.

## Required Checks

```bash
python3 -m pip install -r requirements/compat-smoke.txt
python3 -m pip install -r requirements/schema-validation.txt
cargo fmt --all -- --check
cargo fmt --manifest-path ui/Cargo.toml -- --check
cargo check --workspace --all-targets --locked
cargo check --manifest-path ui/Cargo.toml --target wasm32-unknown-unknown --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings -A missing_docs
cargo clippy --workspace --all-targets --features serve-ui --locked -- -D warnings -A missing_docs
cargo clippy --manifest-path ui/Cargo.toml --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test --manifest-path ui/Cargo.toml --locked
cargo build -p bloomai-ffi --locked
BLOOM_TEST_NATIVE_FFI=1 python3 -m unittest discover -s python/tests -v
python3 scripts/test_server_shutdown.py
python3 scripts/test_server_http_boundary.py
./scripts/test_tiny_model_runtime.sh --require-official-clients
./scripts/test_trained_qwen2_runtime.sh
./scripts/test_trained_qwen3_runtime.sh
./scripts/test_trained_llama_runtime.sh
./scripts/test_trained_safetensors_runtime.sh
./scripts/test_trained_embedding_runtime.sh --require-official-clients
./scripts/validate_json_artifacts.py
python3 scripts/check_readiness_contract.py
python3 scripts/check_markdown_links.py
python3 scripts/check_community_metadata.py
python3 scripts/test_community_metadata.py
python3 scripts/check_github_action_pins.py
python3 scripts/check_release_workflow_security.py
python3 scripts/test_release_workflow_security.py
python3 scripts/test_create_release_archive.py
python3 scripts/check_toolchain_consistency.py
python3 scripts/test_validate_release_artifact.py
cargo test --locked -p bloomai-server --example sign_model_index
./scripts/test_crate_packages.sh
./scripts/openai_compat_smoke.py --api-key release-compat-smoke --require-openai-sdk
./scripts/ollama_compat_smoke.py --api-key release-compat-smoke --require-ollama-sdk
./scripts/package_release.sh
./scripts/validate_release_artifact.py release-artifacts/bloom-*.tar.gz \
  --checksum release-artifacts/SHA256SUMS \
  --require-embedded-ui \
  --require-native-self-check \
  --require-deterministic-metadata
```

The archive-validator regression suite mutates packaged readiness identity,
protocol ranges, ready-state invariants, and schema versions. The standalone
validator must reject each mutation and must parse the actual bundled v3
example and compatibility-critical Draft-07 schema structure rather than
treating those files as opaque nonempty assets.
It also accepts both deterministic archive formats and rejects legacy metadata
when the release-only deterministic-metadata gate is enabled.

## Real-Model Gates

The required-check phase first generates a deterministic untrained Qwen2
single-file package plus a byte-reproducible indexed two-shard variant and
crosses the real native tokenizer, strict shard resolver, Safetensors loader,
Candle forward pass, decoding, OpenAI/Ollama adapters, streaming, embedding
projection, reranking, current and legacy Ollama embedding routes, and pinned
official client decoders on CPU. A deterministic structured-output profile must produce
schema-valid JSON through buffered and streamed OpenAI Chat/Responses and
Ollama chat/generate routes using raw HTTP and both pinned clients. A separate
executor gate must keep grammar state across decode steps, isolate histories in
one native batch, and prove that enabled speculation cannot emit unchecked
structured tokens. A separate
tool profile must successfully complete raw and SDK-decoded OpenAI
Chat/Responses and Ollama chat function calls, buffered tool streams, result
continuations, private-control non-disclosure, retained input-item history, and
deletion; fail-closed output validation does not count as success in either
mode. The gate also verifies clean native KV state after startup verification
and between embedding batch items. This prevents model-free CI from silently
skipping runtime execution, but it is not trained-model language, semantic, or
tool-selection quality evidence.

The maintained text-model gates are pinned and self-contained:

```bash
./scripts/test_trained_qwen2_runtime.sh
./scripts/test_trained_qwen3_runtime.sh
./scripts/test_trained_llama_runtime.sh
./scripts/test_trained_safetensors_runtime.sh
```

They must verify the documented Qwen2 Q4_0, Qwen3 Q8_0, SmolLM2
Llama-architecture Q8_0, and SmolLM2 native BF16 Safetensors revisions, every
required package file, license evidence, exact instruction-following output,
positive benchmark records, and exact buffered and streamed output through the
OpenAI and Ollama adapters. They download into external `/tmp` caches and must
not add weights to the source tree or release archive. The following commands
remain required when promoting any additional model/backend path:

```bash
BLOOM_REQUIRE_MODEL=1 BLOOM_STRICT_MEMORY_BUDGET=1 BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/openai_compat_smoke.py --build --require-openai-sdk
BLOOM_REQUIRE_MODEL=1 BLOOM_STRICT_MEMORY_BUDGET=1 BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/ollama_compat_smoke.py --build --require-model --require-ollama-sdk
BLOOM_REQUIRE_MODEL=1 BLOOM_STRICT_MEMORY_BUDGET=1 BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/ci_smoke_test.sh
BLOOM_REQUIRE_MODEL=1 BLOOM_REQUIRE_LLAMA_CPP=1 BLOOM_STRICT_MEMORY_BUDGET=1 BLOOM_MODEL_PATH=/path/to/model.gguf ./scripts/compare_llamacpp.py --build
```

Run the OpenAI and Ollama compatibility smoke commands a second time with a
pinned embedding model. The Ollama smoke detects the runtime's `embedding`
capability and validates normalized batched `/api/embed`, legacy
`/api/embeddings`, dimensionality projection, and official-client decoding.
The trained MiniLM gate submits variable-length OpenAI embedding and rerank
batches, exercising native padded attention masks, masked mean pooling, and
stable request-order restoration on CPU.
The no-model phase also sends a non-mutating `DELETE /api/delete` request for a
deliberately absent catalog ID and requires an authenticated Ollama-shaped 404.
It calls `POST /api/pull` with downloads disabled and requires the pinned
official client to decode the authenticated Ollama-shaped 403; a separate raw
request proves `insecure: true` fails before acquisition admission. CPU server
fixtures cover signed-index idempotency and both pull response modes, while the
verified downloader fixtures enforce signed size, SHA-256, atomic installation,
exact active-work identity, and persistent signed-index aliases without a GPU
or public network request. Schema-v2 package fixtures must additionally prove
exact-manifest restart resume, aggregate identity and quota admission,
every-file verification, no publication after a bad checksum, atomic directory
installation, valid indexed-shard header ownership, rejection of logically
invalid index/header mappings, directory integrity quarantine, Unix durability
sync, recursive restart-time staging accounting, active-session-safe stale
package cleanup, no-follow symlink cleanup, and server-authoritative signed-ID
acquisition through both `/v1` and
Ollama pull. Repeated `/v1` and Ollama pulls must prove exact installed
idempotency, identical in-progress joining, and HTTP 409 for unaliased or
otherwise conflicting local entries; UI tests must make the same installed or
conflict decision before enabling acquisition. Same-ID upgrade fixtures must
also cover old-plus-new quota admission, persisted restart resume, checksum
failure with the previous model intact, destination and file/package-shape
changes, lifecycle exclusion, interrupted commit on both sides of the rename,
corrupt replacement rollback, and fail-closed symlink transaction state. The
native and Ollama routes and the UI must agree on the upgrade decision. The
catalog ownership gate must additionally prove that a second process cannot
acquire the same root, process exit releases ownership, an unlocked stale file
is reusable, and an unsafe root or symbolic-link lock fails closed. Run these
tests on every supported host filesystem and record any network-filesystem lock
limitations. The artifact schema gate must
recompute the package digest, and the
offline signing-helper tests must accept v2 while retaining the v1 signature
domain. CPU lifecycle fixtures must also prove one queued
load, exact same-target joining, different-target conflict, terminal success
and failure delivery, public-selector preservation, and empty-request unload.
The required-check phase uses both pinned official clients. OpenAI must decode
empty discovery, missing-model retrieval, authentication failure,
unavailable-model chat, and missing Responses state; Ollama must decode
discovery, authentication failure, and the deletion failure before any
model-specific release evidence is collected. With a real model, OpenAI must
also decode a retrieved Model resource whose ID matches generation metadata.
The model-free OpenAI smoke also requires every exercised HTTP path to publish a
bounded `x-request-id`, preserve a safe proxy-supplied value, replace an unsafe
one on a 404 response, and expose the header to browser clients through CORS.
It configures one exact test origin, requires that exact allow-origin response,
and requires an unrelated origin to fail with HTTP 403 and no CORS admission.
The live HTTP boundary additionally launches the default policy, proves the
embedded origin and origin-free SDK path remain admitted, and rejects opaque,
cross-origin, preflight, and loopback DNS-rebinding requests before routing.
The same CORS gate must expose `Retry-After`; CPU server tests hold an inference
permit to require a default one-second 429 hint and prove that explicit hints
are retained while 503 remains untouched. UI tests accept only 1-to-300-second
delta values on 429 and never schedule an automatic replay.
It must also expose `WWW-Authenticate`. Missing and invalid credentials in both
namespaces must retain their protocol JSON and fixed Bearer challenge, while
Bearer plus `X-API-Key` credentials succeed and public fallbacks remain
unchallenged. The HTTP-boundary gate launches an isolated authenticated server,
and the pinned OpenAI client must observe the challenge on its decoded 401.
UI tests require public readiness plus the bounded protected Models probe before
publishing connection state and a distinct authentication failure on 401.
UI host tests require the shared non-successful-response formatter to append
only values accepted by the same bounded correlation alphabet. Together these
checks cover the server and browser halves of error correlation without a GPU;
confirm the rendered banner in each supported target browser before promotion.
It validates the complete versioned readiness identity, positive ordered UI
protocol range containing the server and current UI protocols, and required
admission fields before invoking the SDK. The real-model path additionally
requires a consistent HTTP 200 ready state.
Run the OpenAI command with `--embedding-only --require-model
--require-openai-sdk`; it must validate projected vectors, normalization,
token usage, stable rerank ordering, bounded scores, `top_n`, and returned
documents without invoking text generation.

The text-model OpenAI and Ollama smoke paths send a bounded stop sequence on
chat/generate requests. Before release, also use a deterministic fixture or
model prompt that emits a marker split across deltas and confirm buffered,
SSE, and NDJSON output omit both the marker and later text while the request
settles successfully with a `stop` reason.

For an index-backed release candidate, restart after a verified pull, confirm
`/api/tags` still publishes the signed ID, and use that same ID for show,
on-demand generation or embedding, and guarded deletion. Record that pull alone
does not switch the runtime, two exact concurrent inference requests share one
load, a different selector receives 409, a failed replacement leaves the old
runtime usable, empty preload succeeds, negative `keep_alive` remains resident,
`keep_alive: 0` unloads an empty request immediately or a nonempty request only
after its response finishes, and a short positive duration appears in `/api/ps`
before automatically unloading the exact runtime instance.

Record the model source, license, SHA-256 hash, backend, dtype, quantization,
TTFT, TBT, tokens/s, peak memory, OS, and hardware in
`docs/support-matrix.md`.

## Artifacts

- Linux application archive containing the embedded browser UI,
  `bloom_infer`, `bloom_server`, `bloom_bench`, `inspect_gguf`, and docs.
- macOS application archive containing the embedded browser UI,
  `bloom_infer`, `bloom_server`, `bloom_bench`, `inspect_gguf`, and docs.
- Windows zip application archive containing the embedded browser UI,
  `bloom_infer.exe`, `bloom_server.exe`, `bloom_bench.exe`, `inspect_gguf.exe`,
  and docs.
- Docker image for CPU/local serving.
- SHA-256 checksums for every published archive.
- A signed GitHub build-provenance attestation for every archive and checksum
  file, emitted before the draft release is created.
- Deterministic tar.gz and zip container metadata derived from
  `SOURCE_DATE_EPOCH`, with atomic publication of each completed archive.
- Bundled `QUICKSTART.md` and schema-versioned `BLOOM-RELEASE.json` containing
  the target, embedded-UI state, self-check result, and per-binary SHA-256.
- Release notes listing support matrix changes and known limitations.

The native staged `bloom_server` must pass `--doctor=json` without failures,
and its `embedded_ui` check must pass. The release workflow and packaging
script enforce both conditions without binding a port or loading a model.
Native Unix packaging also runs the clean, deadline-forced, and
repeated-signal lifecycle tests against that exact staged release binary.
Every native package runs the HTTP boundary test against that same binary:
default and exact browser-origin policies, rejected origins and preflights,
origin-free client compatibility, generated and supplied correlation IDs, CORS
exposure, query-free trace fields,
the versioned readiness handshake, protocol-shaped namespace 404s, JSON 405s
with `Allow`, malformed JSON, schema, media-type, body-limit and multipart
rejections, authentication isolation, asset 404s, content-negotiated SPA
navigation, and unsafe-method rejection must all pass before the archive is
created. Error messages must remain bounded and non-reflective. The same gate
requires `Cache-Control: no-store` on probes, known OpenAI/Ollama resources, and
all protocol errors while leaving static UI paths outside the dynamic policy.

## Publishing to crates.io

The workspace publishes six crates under the `bloomai-*` namespace. Publish
them in dependency order so each crate's path dependencies already exist on
the registry. The server and FFI crates can be published in either order after
the engine because neither depends on the other:

```bash
cargo login                          # one-time, needs a crates.io API token
cargo publish -p bloomai-core
cargo publish -p bloomai-backend
cargo publish -p bloomai-tilelang
cargo publish -p bloomai-engine
cargo publish -p bloomai-ffi
cargo publish -p bloomai-server
```

> **Namespace:** crates are published as `bloomai-*` because the bare
> `bloom-core` name was already taken on crates.io by an unrelated project.
> The original five library names were verified free on 2026-07-21; verify the
> newer `bloomai-server` name again immediately before its first publication.
> The FFI dynamic
> library keeps the file name `bloom_ffi` (`libbloom_ffi.{so,dylib,dll}`) so
> the Python SDK and downstream loaders are unaffected by the crate rename.

Python SDK: `python/` has its own `pyproject.toml`; build the native library
first (`cargo build --release -p bloomai-ffi`), then `pip install ./python`.

## Git tag & GitHub Release

```bash
git tag -a v0.1.0 -m "Bloom v0.1.0"
git push origin main --tags        # triggers .github/workflows/release.yml
```

Target-specific tar.gz and zip archives are uploaded by the release workflow;
do **not** commit `release-artifacts/` (git-ignored).
The workflow calls `scripts/package_release.sh` directly so local and hosted
archives pass the same UI, doctor, documentation, and manifest gates.
The script uses `SOURCE_DATE_EPOCH` when provided and otherwise uses the current
Git commit time. This makes the archive layer reproducible for an identical
staged tree; it is not a claim that separate compiler or linker runs produce
identical native binaries.
Before publishing the draft, verify at least one downloaded archive with
`gh attestation verify ARCHIVE --repo constmark/bloom`; the attestation must
name this repository and the tagged release workflow revision.

## Do Not Release If

- The repository cannot build without files outside this repository.
- An official archive or Docker image does not serve the embedded UI.
- An embedded UI fallback returns its HTML shell for an unknown API route,
  missing asset, non-HTML client, or non-GET request.
- An unknown `/v1` or `/api` path does not return its documented JSON 404
  envelope, or a method mismatch loses HTTP 405, its protocol envelope, or the
  route's `Allow` header. Fixed fallback messages must not reflect request paths,
  queries, or credentials, and recognized routes must remain authenticated.
- A malformed JSON, schema mismatch, missing media type, oversized body, invalid
  multipart request, timeout, or other framework rejection escapes `/v1` or
  `/api` as plain text, HTML, an empty body, a reflected parser detail, or the
  wrong protocol envelope. Meaningful status and safe response headers must be
  retained, while probes and static UI paths must remain unchanged.
- A proxy, middleware, or handler can make a `/v1`, `/api`, health, readiness,
  or metrics response cacheable.
- The UI omits a valid `x-request-id` from a non-successful HTTP error, displays
  an unsafe or oversized value, or treats the correlation ID as a generation
  cancellation or authorization capability.
- A capacity HTTP 429 can omit `Retry-After`, CORS hides the hint, the UI
  displays an unsafe or unbounded value, or any client path automatically
  replays a user request in response to the header.
- A protected 401 omits its fixed Bearer challenge, CORS hides the challenge, a
  403 or public route fallback gains one, or the GUI can report a successful
  connection before its protected Models probe accepts the configured key.
- The default browser policy admits a cross-origin, opaque, malformed,
  duplicate, or loopback DNS-rebinding request; rejects an embedded same-origin
  or origin-free SDK request; lets CORS answer an untrusted preflight; or emits
  an allow-origin header on a rejected response. Do not release if strict
  security accepts `*` or the doctor fails to warn about a non-strict wildcard.
- `/ready` lacks the documented Bloom identity or supported UI protocol range,
  excludes either its server protocol or the current UI protocol, violates its
  required-field or ready-state invariants, or the UI accepts an unknown
  readiness contract as compatible.
- The staged native server fails its versioned doctor self-check.
- A hosted archive or checksum lacks its GitHub build-provenance attestation,
  or the attestation cannot be verified against `constmark/bloom`.
- An oversized `--max-concurrent` reaches semaphore construction, panics, or
  binds a listener instead of returning the documented configuration error.
- A Unix application build does not handle `SIGTERM` with a clean status-0
  drain, fails to force a status-1 exit when its bounded drain expires, or does
  not exit immediately with status 1 after a repeated shutdown signal.
- A default CI smoke silently skips every real-model validation and the release
  notes still claim production support.
- Any declared `bloom.json` file hash fails verification. Do not set
  `BLOOM_ALLOW_HASH_MISMATCH` in release or production validation.
- The model cannot load with `BLOOM_STRICT_MEMORY_BUDGET=1` on the documented
  target hardware.
- New public traits or schema fields lack migration notes.
- Any `stable` row in the support matrix lacks a reproducible command.
