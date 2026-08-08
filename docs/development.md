# Development

## Common Commands

```bash
cargo fmt --all -- --check
cargo check --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings -A missing_docs
cargo test --workspace --locked
python3 scripts/test_server_shutdown.py
python3 scripts/test_server_http_boundary.py
./scripts/test_tiny_model_runtime.sh
python3 -m pip install -r requirements/schema-validation.txt
./scripts/validate_json_artifacts.py
```

The process-lifecycle test builds and launches the real server. It exercises a
clean `SIGTERM`, bounded deadline expiry, and repeated-signal escalation on
POSIX hosts, and reports an explicit skip on unsupported hosts. Set
`BLOOM_TEST_SERVER_BINARY` to exercise an existing native binary without
rebuilding it; native Unix release packaging uses this path for the staged
release server.

The HTTP-boundary test is cross-platform and model-free. It checks generated,
proxy-supplied, and unsafe `x-request-id` values; default same-origin and exact
cross-origin admission; rejected actual requests and preflights; opaque and
loopback DNS-rebinding origins; origin-free client compatibility; CORS exposure;
query-free trace fields; the versioned readiness identity, supported UI
protocol range, and required handshake fields; authoritative no-store headers
on dynamic responses; fail-closed oversized-concurrency and zero-sized chunked
prefill startup without a panic; API and asset 404 behavior;
OpenAI- and Ollama-shaped namespace 404s; method-specific JSON 405s and `Allow`
headers; protocol-shaped malformed JSON, schema, media-type, body-limit, and
multipart rejections; authentication isolation; and content-negotiated SPA
navigation. Set
`BLOOM_EXPECT_EMBEDDED_UI=1` with `BLOOM_TEST_SERVER_BINARY` when validating an
application build. Native release packaging applies both variables to the exact
staged server before creating the archive.

Readiness validation used by this live boundary, the OpenAI and Ollama smokes,
bundled JSON artifact checks, and offline release-archive validation lives in
`scripts/readiness_contract.py`. Keep identity, compatibility-range, field, and
ready-state rules there so these independent gates cannot silently diverge.
`scripts/test_validate_release_artifact.py` mutates the packaged example and
schema to prove stale or inconsistent contracts fail closed. It also verifies
both deterministic archive formats and proves the release metadata requirement
rejects an archive made by the legacy non-normalizing path.

UI host tests independently cover the browser half of correlation: the shared
HTTP error formatter accepts the server's conservative 128-byte ASCII alphabet,
appends a valid ID after bounded body formatting, and omits empty, unsafe,
non-ASCII, or oversized values. The live boundary test proves the corresponding
header and CORS contract, including exposure of `retry-after`. Server tests hold
the only inference permit to exercise a real 429 producer, while UI host tests
admit only bounded delta-seconds on 429 and prove no retry scheduler is involved.
Target-browser rendering remains a manual release check.

The same live boundary gate launches isolated API-key and default-origin server
processes in addition to the exact-origin process. It requires OpenAI- and
Ollama-shaped 401 responses, the fixed Bearer challenge, CORS exposure,
successful Bearer and `X-API-Key` access, and
unchallenged public 404 behavior, while proving that the default policy admits
only the embedded origin and origin-free clients. UI host tests cover the
bounded Models probe, authentication error classification, and the actionable
first-run state; official-client smokes confirm the challenge on their decoded
401 path.

Install `requirements/schema-validation.txt` before validating public artifacts
to run complete Draft-07 checks. The validator retains a standard-library
structural fallback for minimal release environments, but CI and release
evidence must include the pinned full validator.

Signed-index changes must keep both protocol generations executable. The JSON
artifact gate validates the version 1 single-file examples and the version 2
package examples, including a recomputed domain-separated package digest. Run
the offline publisher helper's own admission tests as well:

```bash
cargo test -p bloomai-server --example sign_model_index --locked
```

Server unit coverage must prove that one bad package-file checksum leaves no
catalog directory or provenance, that a matching hidden package stage resumes
after restart, that an index/header ownership mismatch is rejected before
publication, that a byte-valid indexed two-shard package installs, that
integrity verification hashes the exact installed tree, and that both
`POST /v1/model-management/index/{id}/download` and Ollama pull derive all
fields from the verified snapshot. Both routes must also prove exact
installation idempotency, identical in-progress joining, and fail-closed
conflict handling for missing aliases or changed provenance. UI tests must
render the same missing, installed, and conflict decision from the bounded
catalog snapshot. These tests are model-free and require no GPU or public
network access.

Storage policy tests must include nested completed and partial package files
left by a previous process. Their bytes remain part of quota admission after
restart, stale package directories and metadata are removed as one session,
active package sessions survive cleanup, and a package-shaped symlink is
unlinked without traversing or deleting its target.

Use `just --list --unsorted` for the maintained command shortcuts.

`scripts/test_crate_packages.sh` packages every publishable workspace member
with the workspace lockfile, extracts the exact `.crate` archives, and compiles
them as one locally patched release set. Use this instead of plain
`cargo package --workspace` when coordinated unpublished workspace crates share
a version: Cargo's built-in independent verification would otherwise resolve
an older same-version dependency from crates.io rather than the sibling archive.

## Deterministic Native CPU Runtime Gate

`scripts/test_tiny_model_runtime.sh` generates small, untrained Qwen2 text and
explicitly marked embedding profiles twice each, a structured-output profile
twice, and a tool-call profile twice in a temporary directory. It also writes
the text profile as two indexed Safetensors shards twice. It requires every
same-profile config, tokenizer, index, and Safetensors byte to match. No model
weights are downloaded or committed. The single-file packages and one
byte-reproducible sharded package then run through Bloom's native Candle loader
and forward pass via the OpenAI- and Ollama-compatible HTTP adapters:

```bash
./scripts/test_tiny_model_runtime.sh
```

Linux CI additionally passes `--require-official-clients`, so the pinned
OpenAI and Ollama Python clients must decode responses produced by the same
native inference path. The text phases cover startup readiness, model
publication, authenticated buffered and streaming generation,
structured-output failure shaping, and lifecycle cleanup on ordinary CPU
runners. The OpenAI text phase uses startup loading; the Ollama text phase
starts with an inactive managed-catalog entry, requires empty-prompt preload to
publish it through `/api/ps`, and then runs buffered and streamed generation
through that activated runtime. Every Ollama model phase finishes with a short
positive `keep_alive`, requires `/api/ps` to publish a finite deadline, and
waits for automatic unload of the exact runtime. Both adapters repeat the text
lifecycle against the indexed two-shard package, proving that strict shard
admission reaches a real CPU forward pass.

The embedding profile carries trusted `bloom_task=embedding` metadata; changing
a directory name cannot grant the capability. That phase requires each
start-at-zero input to receive clean native KV state, including the first
request after startup verification and every item in a batch. It validates OpenAI batched embeddings, finite L2 normalization,
dimensionality projection, stable duplicate ordering in reranking, Ollama
on-demand activation, current `/api/embed`, legacy `/api/embeddings`, and the
pinned official-client decoders in CI.

The structured-output profile emits exactly `{"ok":true}` followed by EOS. Its
gate requires successful JSON object and strict JSON Schema output through
OpenAI Chat Completions and Responses, including both streaming protocols, plus
Ollama chat and generate in buffered and NDJSON forms. Linux CI also requires
the pinned OpenAI and Ollama SDKs to decode schema-valid output; a terminal
validation error does not count as success in this mode. Executor unit gates
add adversarial logits: they require grammar state to advance across decode,
keep two histories independent in one native batch, and prevent an enabled
n-gram speculative strategy from bypassing structured token constraints.

The tool profile emits one schema-valid `get_weather` call followed by EOS. Its
gate requires successful OpenAI Chat Completions and Responses calls plus
Ollama chat calls, buffered tool streams that never expose Bloom's private
control JSON or correlation IDs, function-result continuations, retained
Responses input-item history, and explicit state deletion. Linux CI also
requires the pinned OpenAI and Ollama SDKs to complete their native function
lifecycles; fail-closed output validation does not count as tool-call success in
this mode.

This causal, untrained fixture proves deterministic runtime mechanics,
cross-input isolation, structured-output protocol mechanics, and
function-protocol mechanics, not language, semantic embedding, or trained-model
instruction-following quality, architecture breadth, quantized-kernel behavior,
or compatibility with a trained public checkpoint. Release promotion still
requires the separately documented pinned text and embedding model gates.

## Pinned Trained CPU Gates

Run the opt-in trained-model gate when changing GGUF or Safetensors metadata,
tokenization, native or quantized execution, prompt formatting, memory
accounting, benchmark reporting, or either HTTP adapter:

```bash
./scripts/test_trained_qwen2_runtime.sh
./scripts/test_trained_qwen3_runtime.sh
./scripts/test_trained_llama_runtime.sh
./scripts/test_trained_safetensors_runtime.sh
./scripts/test_trained_embedding_runtime.sh
```

The profiles download official Apache-2.0 Qwen2 0.5B Instruct Q4_0, Qwen3 0.6B
Q8_0, SmolLM2 360M Instruct Q8_0 GGUF, and SmolLM2 360M Instruct BF16
Safetensors packages into separate `/tmp` caches. They verify fixed revisions,
sizes, every required SHA-256 value, and immutable license evidence, then
require exact `Bloom` instruction-following output. They also require positive
CPU benchmarks with hardware and timing breakdowns plus exact buffered and
streamed output through both HTTP adapters. The Qwen3 profile crosses
architecture-specific loading and disabled-thinking formatting; both SmolLM2
profiles cross Llama-architecture loading and classified ChatML prompt
selection. The Safetensors profile additionally proves bounded Hugging Face
metadata parsing and truthful BF16-storage/F32-runtime memory reporting. Use
`BLOOM_TRAINED_MODEL_CACHE` to select a persistent external cache or
`--model-path` to reuse an already downloaded exact file. The model is never
copied into the workspace or release artifact. Full provenance and the current
reference observation are recorded in
[trained-model-validation.md](trained-model-validation.md).

The MiniLM profile downloads the official 91 MB F32 Sentence-Transformer
package into its own external cache. It proves native BERT loading, CLS/SEP
tokenization, mean pooling, 384-dimensional normalized output, zero KV-cache
accounting, the model's 256-token task limit, semantic separation, and
bi-encoder reranking through OpenAI and both Ollama embedding contracts. It
also requires encoder-only task isolation across five generation routes. Pass
`--require-official-clients` after installing `requirements/compat-smoke.txt`
to make both pinned SDK decoders mandatory.

## Official Client Compatibility

CI installs the official OpenAI and Ollama Python client versions pinned in
`requirements/compat-smoke.txt` and requires both model-free compatibility
smokes to pass with API-key protection enabled:

```bash
python3 -m pip install -r requirements/compat-smoke.txt
./scripts/openai_compat_smoke.py \
  --api-key local-compat-smoke \
  --require-openai-sdk
./scripts/ollama_compat_smoke.py \
  --api-key local-compat-smoke \
  --require-ollama-sdk
```

These paths require no model or accelerator. They validate official-client
decoding for empty discovery, 401 authentication errors, unavailable-model
admission, missing Responses state, a versioned no-model readiness handshake,
protocol-shaped 404/405 fallbacks, framework-level request rejection, and a
non-mutating Ollama delete 404. The Ollama client also exercises `/api/pull`
and must decode its explicit disabled-download admission without starting
network or filesystem work; the raw smoke separately rejects `insecure: true`.
Update the pins deliberately, run both gates, and record the compatibility
change before merging an SDK upgrade. A successful signed-index pull is covered
by CPU server and downloader fixtures. The generated Qwen2 gate adds full
native text loading, forward, chat, generation, streaming, embedding,
projection, reranking, and successful function-protocol mechanics without
external weights. The pinned text and MiniLM profiles add trained language and
semantic-retrieval evidence; trained tool-use quality and broad architecture
compatibility remain release gates.

Signed-model upgrade recovery is also model-free. Its focused CPU gates cover
same-path and cross-path commit, file/package-shape changes, old-plus-new quota,
persisted resume identity, checksum failure, route-level lifecycle exclusion,
and restart recovery before and after replacement publication:

```bash
cargo test --locked -p bloomai-server --lib model_upgrade::tests
cargo test --locked -p bloomai-server --lib \
  signed_index_download_endpoint_transactionally_upgrades_an_installed_alias
cargo test --locked -p bloomai-server --lib \
  ollama_pull_transactionally_upgrades_and_reuses_a_signed_model_package
cargo test --locked -p bloomai-server --lib \
  upgrades_reserve_peak_space_without_treating_the_old_model_as_reclaimable
cargo test --locked -p bloomai-server --lib \
  restart_resume_retains_the_exact_signed_upgrade_identity
cargo test --locked -p bloomai-server --lib \
  failed_upgrade_verification_leaves_the_installed_model_usable
cargo test --locked -p bloomai-server --lib catalog_lock::tests
```

## Feature Policy

The default feature set must build on standard Linux, macOS, and Windows
development machines. Hardware-specific features such as CUDA are checked
separately because they require vendor toolchains.

## Local Artifacts

Do not commit model weights, generated IR files, virtualenvs, compiled kernels,
or installer binaries. Put reproducible setup steps in `docs/` or `scripts/`
instead.
