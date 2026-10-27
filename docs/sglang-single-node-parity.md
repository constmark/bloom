# Bloom and SGLang single-node optimization parity

This review compares Bloom's local inference path with the optimizations that
SGLang exposes for one host.  The comparison is about serving a loaded model
on one CPU/GPU machine; SGLang's distributed tensor/data/expert parallel and
prefill/decode disaggregation are outside this scope.

SGLang's current single-node serving path is built around continuous batching,
Radix prefix caching, paged KV storage, chunked prefill, fused attention
backends, CUDA graphs, quantization, and speculative decoding.  See the
[SGLang attention backend guide](https://docs.sglang.io/docs/advanced_features/attention_backend),
[bench serving guide](https://docs.sglang.io/docs/developer_guide/bench_serving), and
[session-aware radix cache guide](https://docs.sglang.io/docs/advanced_features/session_radix_cache).

| Optimization | Bloom status before this review | Current status | Scope and limitation |
| --- | --- | --- | --- |
| Continuous/in-flight batching | Present | Present | `InferenceScheduler` admits decode and prefill work under token budgets. Native Candle Qwen batching still runs one private KV model per request, so it is scheduler-level batching rather than one fused model graph. |
| Chunked prefill and decode priority | Present | Present | Configured through IFB/chunked-prefill settings; decode receives budget first. |
| Paged KV allocation and prefix reuse | Present | Present | Block allocation, prefix matching, LRU/priority eviction, long-context policies, and CacheMesh are implemented. Real KV extraction requires the streaming Qwen path. |
| GPU-resident KV bridge | Host copy on every tensor write/restore | **Filled** | F16/BF16/F32 local blocks now retain Candle tensors and restore prefixes on-device. CacheMesh and one-byte KV formats retain the vector/quantization path. |
| KV-cache compression | Model-specific Int8/Int4 support; paged cache was fixed F16 | **Filled** | `BLOOM_KV_CACHE_DTYPE=int8|fp8|bf16|f32|f16` now configures the paged cache. `fp8` uses the checked one-byte INT8-compatible payload until a native FP8 kernel exists. |
| Weight residency across requests | MoE template sharing only | **Filled** | Streaming Qwen2/Qwen3/Qwen2-MoE requests fork an empty-KV template and share immutable mmap-backed weights. Other Candle wrappers retain their existing reload semantics. |
| Batch sampling controls | `top_k`/penalties/bias were ignored on the IFB path | **Filled** | Batched sampling now applies top-k before top-p, presence/frequency penalties, and logit bias; top-k avoids sorting the full vocabulary. |
| Fused attention | Optional `candle-flash-attn` existed in the cache helper | **Filled for streaming Qwen/Gemma4** | With `--features flash-attn`, CUDA/Metal global cold prefill and decode use fused FlashAttention; sliding-window, soft-capped, or suffix-prefill paths keep the correctness-first eager mask path. There is no FlashInfer backend. |
| CUDA graphs / static graph replay | Missing | Missing | Candle has no equivalent serving graph capture in this path. This remains a major throughput/launch-overhead gap on NVIDIA GPUs. |
| Fused GPU sampling | Missing | Missing | Bloom samples logits on the host with temperature/top-p logic. SGLang's fused sampling backends are not represented. |
| Speculative decoding | Native n-gram/draft paths; MTP through llama.cpp | Present | Structured output disables unconstrained multi-token speculation. Native Candle MTP remains unavailable. |
| Tensor/data/expert parallelism | Abstractions and no-op collectives | Missing for native Candle | TensorRT can advertise external parallel execution, but Candle's collectives are single-device stubs. |
| Prefill/decode disaggregation | Missing | Intentionally missing | Not needed for a single-machine process; it is a multi-process/multi-device deployment feature. |

The remaining high-impact gaps are therefore CUDA graph capture, a fused
sampling backend, and a true fused ragged model forward.  Adding those requires
device-specific kernels or a runtime such as TensorRT-LLM/FlashInfer; a Rust
cache or scheduler change alone cannot provide them without changing the
execution backend.

## Runtime knobs

For a streaming Qwen model, build with `--features flash-attn` on a machine
with the required CUDA/Metal toolchain.  The local paged KV dtype can be set
with `BLOOM_KV_CACHE_DTYPE`; `int8` is the useful memory-saving choice today.
Keep CacheMesh enabled when KV blocks must survive local eviction or be shared
with a persistent L2/L3 tier; its serialization path intentionally materializes
vectors.

## Validation

The CPU/default-feature engine suite covers the tensor/vector KV round trip,
CacheMesh restore, batch executor integration, and Qwen streaming KV hooks.  A
hardware build with `flash-attn` still needs to be run on the target CUDA/Metal
machine because the local environment does not provide `nvcc`.
