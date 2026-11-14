# Protocol, sampling and client iteration — 2026-11-14

This iteration follows the [completion continuation](completion-continuation-2026-11-14.md).
Three independent read-only audits covered the HTTP protocol layer, the engine
sampling/scheduler paths and the client surfaces (CLI, C ABI/Python SDK, web UI
and macOS). Each finding below was checked against the code before it was
changed, and each one has a regression test. The iteration does not change
any support classification.

## Engine sampling

| Observed problem | Delivered behavior |
| --- | --- |
| Direct Candle generation counted prompt tokens toward `presence_penalty`/`frequency_penalty`. A multi-turn prompt full of `<|im_end|>` could suppress EOS until the token limit; IFB counted only generated tokens | All direct sampling and logprob sites pass generated-only history, matching IFB and the OpenAI contract. A cycle-fixture test fails if prompt tokens are penalized, with or without n-gram speculation |
| `top_k` with `top_p` used Candle's `TopKThenTopP`, which compares `p` with the unnormalized top-k mass. The direct nucleus was wider than the batch sampler's (and HF/vLLM/llama.cpp) | The direct path masks to exactly k tokens and then applies nucleus sampling over the renormalized distribution. Both paths have a `k=2, p=0.3` test that must select a single token |
| The batch sampler mapped `+inf` to impossible and fell back to a raw argmax, which could return a NaN or masked token id when nothing was sampleable | NaN/`+inf` logits, empty logits and fully masked steps are request errors. `-inf` masks are never selected |
| Greedy cutoffs differed (`1e-7` direct, `1e-6` batch/speculative) | One shared `GREEDY_TEMPERATURE` constant |
| Qwen3-VL ignored `top_k`, penalties and `logit_bias`, which both image adapters validate and forward. It silently capped output at 4096 tokens and always reported `stop` | Qwen3-VL uses the shared direct sampler, emits native `TokenUsage`/`GenerationFinished` metadata (validated by the pipeline) and rejects budgets above 4096 instead of truncating |

## HTTP protocols

| Observed problem | Delivered behavior |
| --- | --- |
| Legacy `functions` requests were normalized to tools, but replies used `tool_calls` and follow-up `function_call`/`role:"function"` history returned 400 | Legacy requests are single-call and return `message.function_call` / `delta.function_call` with `finish_reason:"function_call"` (buffered, streaming, n>1 and IFB). Legacy history is rewritten to paired tool history with deterministic IDs; invalid pairing fails closed. Mixing legacy and modern fields is rejected |
| OpenAI and Ollama image responses always reported `stop` and omitted usage | Both adapters consume native multimodal metadata: `length` maps to `length`, buffered OpenAI responses carry `usage`, and Ollama terminals carry `prompt_eval_count`/`eval_count`. Models without metadata keep the previous mapping |
| `/v1/multimodal/stream` validated only some controls before committing a 200 SSE response | All generation controls are validated before admission (400) |
| IFB legacy Completions streaming used `text_completion.chunk` and a per-chunk `created` | `text_completion` and one creation time per stream, matching the direct path |
| Chat `json_schema` rejected `"strict": null` / `"description": null`, which Responses accepts | Null annotations are treated as absent |
| Embedding/rerank backend failures returned 501 `unsupported_operation`, which clients treat as permanent | 500 `internal_error`; unsupported models are still rejected with 501 before admission |
| `top_n` above the document count returned 400, breaking LangChain-style retrievers | Values above the document count return every document; `top_n=0` remains invalid |

Ollama `/api/generate` with images and an empty prompt still returns 400.
Ollama treats an empty prompt as a model-load request, so Bloom keeps its
fail-closed rejection instead of silently ignoring the images.

## Clients

| Observed problem | Delivered behavior |
| --- | --- |
| A config `prompt_file` overrode `--prompt`; config booleans could not be disabled; config `logprobs` broke `--list-engines`, `--inspect`, `-i` and `--bench` | A CLI prompt suppresses both config prompt sources. `--no-stream`, `--no-quiet`, `--no-interactive` and `--no-bench` were added. Config `logprobs` applies only with `--json` |
| `--bench` ignored prompt files, stdin, `--system-prompt` and chat templates; `--prompt -` benchmarked a literal `-`; dtype labels dropped q4/q8 | The benchmark uses the normal input resolver and the shared dtype parser |
| `bloom_bench` had a stale backend-selection copy: no `preferred_backends`, MTP routing, TensorRT or task check | `crates/app/src/backend.rs` provides one registry, selection and task-validation implementation for both binaries |
| JSON chat prompts silently dropped `--system-prompt` | The system message is prepended; a conflicting leading system message is rejected |
| Interactive mode always streamed, and an empty reply left consecutive user turns | Mode starts from `--stream`; an empty reply drops the whole exchange |
| C ABI/Python rejected `device="auto"` | `auto` resolves against the selected engine's declared devices, like the CLI |
| Buffered C ABI/Python generation had no finish reason or token counts | Opted-in models add a `generation` object (`input_tokens`, `output_tokens`, `reason`). The key is additive, ABI revision 2 is unchanged, and the SDK validates its shape |
| Web UI and macOS treated token-limit truncation as normal completion; macOS hard-coded `max_tokens: 256` | Both clients show an accessible truncation notice. macOS adds a persisted, bounded Max tokens setting (default 256) |

## Validation

Run on macOS arm64 with Rust 1.97.1.

| Gate | Result |
| --- | --- |
| Rust workspace (`cargo test --workspace --locked`) | 1,424 passed, 0 failed, 1 existing external Laya/Torch parity test ignored (baseline before this iteration: 1,403) |
| Formatting and strict workspace/all-target Clippy | Passed |
| Workspace/all-target check without default features | Passed |
| Web UI tests, strict UI Clippy, production asset build, Chromium browser/axe gate | 157 passed; Clippy clean; assets built; browser gate passed |
| Native macOS client checks and SwiftUI release-product build | Passed |
| Python SDK with native FFI (fresh debug library) and ABI compatibility exercise | 46 passed; ABI revision 2 buffered and streamed calls passed |
| Python script regressions with pinned optional dependencies | 258 passed |

| Architecture contract and no-default-features batching tests | Passed |
| Real-server shutdown and HTTP-boundary gates | Passed |
| Tiny-model runtime with pinned official OpenAI/Ollama clients, including the updated legacy-function check | Passed |
| Trained MS MARCO MiniLM cross-encoder runtime | Passed |
| Repository contracts (Markdown links, JSON artifacts, readiness, toolchain, action pins, Dockerfile/release/container workflow, Laya upstream, community metadata) | Passed |

The browser gate runs without a model, so it does not render a truncated
answer. The truncation notice reuses the existing storage-notice color pair.
Qwen3-VL metadata is covered by compilation, adapter tests and the pipeline
validator. No pinned trained Qwen3-VL package exists for an end-to-end run.

## Remaining work

This iteration does not change the release requirements in the
[production gap register](production-readiness.md). IFB still derives its finish
reason in the server rather than through a typed scheduler event. IFB
logprobs are still rejected; adding them requires per-request adjusted-logit
capture in the batch sampler. Qwen3-VL budgets above 4096 are rejected inside
the stream rather than before the HTTP response commits.
