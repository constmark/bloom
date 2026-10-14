# llama.cpp parity and acceptance matrix

This document turns the edge-engine goal into reviewable evidence. It does not
claim that Bloom currently has full parity. The first reference is the pinned
llama.cpp revision [`1b70350c92563449576bcbbf552994b1d1ed847d`](https://github.com/ggml-org/llama.cpp/tree/1b70350c92563449576bcbbf552994b1d1ed847d).
Advance the pin only after regenerating the inventory and rerunning the cells.

## Capability matrix

| Domain | Current Bloom position | Parity target | Required evidence |
| --- | --- | --- | --- |
| GGUF loading and metadata | Native Candle loaders for a bounded set; external `llama-server` bridge | Every reference-supported architecture/metadata path has a declared route | Pinned model fixtures, malformed-package rejection, load and unload tests |
| Safetensors and model packages | Native paths for selected families; strict package validation | Keep native support explicit and add conversion/import coverage where the reference offers it | Hash-pinned trained packages and preflight/load agreement |
| Quantization | Q4/Q8 and selected native methods; broader methods remain experimental | Reference-supported quantization has a native or explicitly external cell | Correctness plus memory/throughput report per quantization and device |
| Tokenization and templates | Native tokenizers and bounded templates | Same prompt formatting and token accounting for the reference cell | Token-ID/template fixtures and exact prompt metadata |
| Sampling | Temperature, top-p, seed and bounded generation parameters | Match reference-visible sampling, stop, seed, and logit behavior that is in scope | Golden outputs with fixed seed and parameter matrix |
| Structured/grammar output | Bloom has bounded structured output on native paths | Reference grammar/JSON behavior is available through the same typed contract | Valid/invalid schema, streaming terminal, and refusal tests |
| Streaming | OpenAI/Ollama SSE/NDJSON adapters and external streaming | Equivalent delta, terminal, cancellation, and error semantics | Raw HTTP and official-client stream fixtures |
| Batching and scheduling | IFB, chunked prefill, paged KV and CacheMesh are present but experimental | Comparable slot admission, fairness, context policy, and backpressure | Concurrency curve, queue/cancellation tests, long soak |
| KV cache and prefix reuse | Native KV policies and CacheMesh | Stable cache accounting and observable reuse under the reference workload | Peak memory, hit-rate, eviction, restart and leak reports |
| Speculative decoding | Native/external paths are bounded and capability-gated | Reference-supported draft/MTP behavior with truthful fallback | Draft/eos correctness, speedup, and fallback tests |
| Embedding and reranking | Native BERT embedding and bi-encoder reranking; external llama.cpp text path does not prove these | Every advertised operation has an implementation and a trained evidence cell | Dimension, ordering, semantic quality, and task-isolation tests |
| Multimodal | Selected native/external adapters; broad coverage remains experimental | Reference-supported modalities have a model/device/task cell | Trained image/audio/video fixtures and stream/error coverage |
| Devices | CPU evidence; Metal/CUDA experimental, other adapters external or skeleton | Each promoted device has correctness, memory, performance, and recovery evidence | Target runner artifacts with driver/build metadata |
| Model management | Catalog, import, signed acquisition, preflight, switch and unload | Install/run/list/delete/recover flows are at least as clear as the local reference | Fresh-machine, interrupted-download, replacement and rollback drills |
| CLI and server | `bloom_infer`, `bloom_server`, OpenAI/Ollama compatibility | Common options, doctor output, health/readiness, and errors are predictable | CLI snapshots, HTTP contract tests, and client smoke tests |

The current `llamacpp` engine starts an external `llama-server`. Until its
loaded-model operations forward embeddings, reranking, and structured output,
those capability flags must not be used for routing or support claims.

## Benchmark protocol

`scripts/compare_llamacpp.py` is a smoke harness and currently compares a
limited throughput result. It is useful for discovering a regression, but it
does not by itself establish parity: Bloom is forced to CPU, context and
threads are not fully normalized, and `llama-bench` does not use the same
prompt. Release evidence must use the following protocol:

1. Use the same model file and SHA-256, tokenizer/template, prompt, context
   limit, generated-token limit, seed, temperature, top-p, batch settings,
   thread count, device layers, power mode, and runtime precision.
2. Record the exact Bloom commit/features and llama.cpp commit/build flags,
   hardware, operating system, driver, and runtime versions.
3. Discard cold-start and warm up both systems with the same number of runs.
   Run at least three measured repetitions per point and retain raw output.
4. Measure TTFT, inter-token latency/TBT, prompt and decode tokens/sec, total
   latency, peak host/device memory, concurrency/queue behavior, cancellation
   recovery, and output correctness. Report median and p95 where applicable.
5. Compare native Bloom and the external `llama-server` bridge separately.
   A bridge result may validate protocol compatibility but cannot substitute for
   native backend evidence.

Store reports under a versioned evidence directory with the benchmark schema;
each report must include model digest, reference revision, build features,
device, context, token counts, sampling parameters, and measurement source.

## Acceptance gates

These are promotion gates for a declared reference cell, not universal claims
about every model or device:

| Gate | Initial promotion threshold | Evidence |
| --- | --- | --- |
| Correctness | 100% of declared request, stream, stop, structured-output, and error cases; no silent fallback | Deterministic fixtures and raw protocol/client tests |
| Performance | Bloom median decode throughput ≥ 90% of llama.cpp; p95 TTFT ≤ 1.25×; peak memory ≤ 1.20× on the same cell | Three-run report with raw measurements |
| Stability | 24-hour soak and 1,000 requests without crash; cancellation/restart recovery 100/100; resource drift ≤ 10% | Workload, interruption, and recovery artifacts |
| Model lifecycle | Import, preflight, activation, replacement, unload, interrupted download, and rollback all leave a valid catalog | Catalog and release-drill logs |
| Client usability | Web, desktop launcher, and macOS client agree on readiness/version/model identity; mobile browser completes first-run, stream, stop, and recovery flows | Pinned browser/client checks and screenshots or traces where needed |
| Maintenance | Layer check, no-default-features check, formatting, schemas, and documentation links pass | CI artifacts |

The long-term parity target raises the performance threshold to 95% median
decode throughput, p95 TTFT ≤ 1.15×, and peak memory ≤ 1.15× for each promoted
cell. A cell may remain supported with a documented lower result only when the
tradeoff is deliberate, reproducible, and visible in the support matrix.

## Evidence rules

- A capability declaration without an executable operation and a test is not
  evidence.
- A deterministic untrained fixture proves mechanics only; trained quality
  claims require a pinned, license-reviewed model.
- A successful external process proves the adapter boundary, not native kernel
  or memory parity.
- Missing hardware or runtime evidence keeps the row `experimental`,
  `external-runtime`, or `skeleton` as appropriate.
- Every reference update requires regenerating this matrix, rerunning the
  benchmark protocol, and recording changed behavior before promotion.

