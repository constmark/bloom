# Bloom versus open-source edge inference engines

This review records where Bloom is differentiated, where it trails the most
useful open-source edge runtimes, and which work should close the gaps. It is a
decision document rather than a support claim. The [support matrix](support-matrix.md)
remains the source of truth for executable Bloom behavior.

The comparison was refreshed on 2026-10-25 against the repository at the
current `main` history and the upstream documentation linked below. The review
uses the same boundary throughout: a capability counts only when a model can
load, execute, and be measured on the declared device.

## Decision in one paragraph

Bloom's strongest position is the product and contract layer around an
inference runtime. Its Rust workspace combines native generation and encoder
paths with OpenAI/Ollama protocols, a browser UI, C ABI/Python access, model
preflight and signed acquisition, transactional switching, diagnostics, and
bounded lifecycle behavior. Its weakest position is the execution ecosystem:
native model and quantization coverage is still narrow, most accelerator paths
are experimental or external, three advertised adapters are skeletons, and no
single target has yet reached a stable deployment cell with long-soak and
failure-recovery evidence. Bloom should keep the product layer, make one
CPU/Metal or CPU/CUDA cell measurably excellent, then add one executable
accelerator and broaden GGUF coverage. Implementing every mobile or vendor
runtime in the default server would dilute the current advantage.

## What the reference engines optimize for

| Engine | Primary design center | Artifact and hardware strengths | Cost or limitation Bloom should account for |
| --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | Portable local LLM runtime | Mature GGUF ecosystem; CPU SIMD on x86/ARM/RISC-V; CUDA, HIP, Metal, Vulkan, SYCL and other backends; 1.5/2/3/4/5/6/8-bit quantization; CPU+GPU hybrid execution; server-side streaming, batching, grammar/JSON, tools, embeddings and reranking | C/C++ model/backend code is tightly coupled to the ggml/LLM center of gravity. It is the benchmark and compatibility reference for Bloom, but not a reason to abandon Bloom's typed application and lifecycle layers. |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) and [WebLLM](https://github.com/mlc-ai/web-llm) | Compile once for a target, then run a high-performance LLM artifact | TVM-generated code across CUDA, ROCm, Metal, Vulkan, WebGPU/WASM, iOS and Android; one MLCEngine API across REST, Python, JavaScript and native apps | Target compilation and artifact/version management are heavier than loading GGUF. A compiled artifact is target-specific, so a portable model catalog needs an explicit build matrix. |
| [ExecuTorch](https://github.com/pytorch/executorch) | PyTorch export and delegate deployment | `.pte` programs, torch export/torchao ecosystem, XNNPACK fallback, Core ML, Vulkan, Qualcomm, MediaTek, Arm and Samsung delegates, plus embedded and WebAssembly targets | Export and lowering are part of the product. Backend operator coverage and per-target artifacts require dedicated validation; it is a poor default dependency for a small standalone GGUF server. |
| [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/) / [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) | General graph runtime with provider selection | Broad C/C++ and language bindings, CPU/CUDA/TensorRT/DirectML/OpenVINO/NNAPI/CANN and other execution providers; GenAI adds generation loops, sampling, KV cache, grammar and tool-call building blocks | GenAI needs an ONNX graph, tokenizer and provider packaging. Zero-copy and LLM performance depend on the selected provider, so an adapter without provider evidence is not a deployment cell. |
| [MNN](https://github.com/alibaba/MNN) | Compact mobile and embedded C++ runtime | Android/iOS/POSIX support, CPU/GPU hybrid execution, Metal/OpenCL/Vulkan/CUDA, model conversion from several frameworks, LLM and diffusion paths | Mobile packaging and conversion are strong, while GGUF interoperability and the surrounding server ecosystem are smaller than llama.cpp's. It belongs behind a mobile package boundary until Bloom has a mobile owner and device cell. |
| [LiteRT/TFLite delegates](https://android.googlesource.com/platform/external/tensorflow/+/f13ea73a9eb/tensorflow/lite/g3doc/performance/delegates.md) | Android and Google hardware delegates | XNNPACK, GPU, NNAPI, Hexagon and Core ML delegate options with CPU fallback | Delegate behavior varies by operator and device. It is valuable for a Google/Android product cell, not a generic desktop backend. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple Silicon Python-first local LLM work | Unified memory and Metal are integrated into MLX; simple model loading and quantization on Apple hardware | The Python/runtime boundary and Apple-only scope make it an excellent isolated Apple adapter, but a weak universal backend. |

The comparison has two implications. First, llama.cpp is the right near-term
reference for Bloom's GGUF, sampler, streaming, KV and benchmark behavior.
Second, MLC, ExecuTorch, ONNX GenAI and MNN solve different deployment
problems. Bloom should select one based on a target product cell instead of
turning all of them into partially implemented default features.

## Bloom's advantages

| Advantage | Evidence in this repository | How to extend it |
| --- | --- | --- |
| Layered, replaceable runtime | `bloomai-core` contracts, `bloomai-backend` device policy, `bloomai-engine` executors, and a server that does not own Candle tensors ([architecture](architecture.md)) | Keep new vendor SDKs behind an adapter or process boundary. Add capability evidence to the same contract rather than routing by file name. |
| Product-level model lifecycle | Signed index, SHA-256 verification, bounded downloads/imports, preflight, memory admission, atomic replacement, rollback and guarded deletion ([model index](model-index.md), [model catalog](model-catalog.md)) | Make lifecycle evidence part of every new backend cell: install, interruption, replacement, unload and recovery must be measured with inference. |
| One client contract | OpenAI Responses/chat/completions/embeddings/reranking, Ollama compatibility, streaming, browser UI, C ABI and Python SDK | Preserve one typed application service and project it into protocols. Add a capability-aware client error when an engine cannot perform an operation. |
| Edge resource controls | IFB, chunked prefill, paged KV, CacheMesh, speculative decoding, memory estimates, cancellation and request admission | Turn the current experimental mechanics into a measured curve: TTFT/TBT, fairness, memory, cache hit rate, cancellation recovery and long-context behavior. |
| Multitask scope | Native text generation plus BERT embeddings and bi-encoder reranking, with multimodal and external adapters | Keep embeddings/reranking as first-class tasks and add a trained cross-encoder only after the current bi-encoder path has broader evidence. |
| Truthful failure behavior | Explicit `experimental`, `external-runtime` and `skeleton` statuses; capability routing rejects unsupported combinations before load | Treat every support promotion as a model/device/task evidence change, not an adapter declaration change. |
| Rust and bounded process boundaries | Memory-safe native code, bounded external process output/deadlines, cross-platform path handling, doctor and diagnostics | Publish a small stable backend ABI and versioned plugin contract after the first deployment cell is stable. |

These are differentiators that the reference engines do not provide as one
integrated product. They should be measured and documented rather than hidden
behind a larger feature list.

## Gaps that matter

### 1. Execution maturity is behind the interface surface

The current support matrix marks Candle CPU/Metal/CUDA as experimental and
keeps CoreML, Vulkan and TensorRT as skeletons. ONNX, MLX, OpenVINO and ASR
paths depend on a Python process or another runtime. There is trained CPU
evidence for Qwen2, Qwen3, SmolLM2 and MiniLM, and a Qwen2-MoE GPU observation,
but there is no stable Bloom deployment cell with the full correctness,
performance, cancellation, restart, upgrade/rollback and multi-hour soak gate.
The result is a polished contract sitting on a still-small set of proven
execution cells.

**Action:** freeze one exact model digest, template, quantization, device,
runtime and client set. Run the acceptance protocol in
[llama.cpp parity](llamacpp-parity.md), including three-run TTFT/TBT/throughput
and memory results, cancellation/restart drills, and a 24-hour soak. Promote
only that combination; keep every other path explicitly experimental.

### 2. Model and quantization coverage trails llama.cpp

Bloom's native GGUF and Safetensors loaders cover selected Qwen/Llama/Gemma,
BERT and MoE layouts. The most visible evidence is Q4_0/Q8_0 and selected
native methods. llama.cpp has a much wider GGUF architecture/quantization
ecosystem and optimized kernels across more CPU and GPU families. Bloom also
has a quantization tool, but conversion, tokenizer/template compatibility and
per-quantization performance are not yet comparable.

**Action:** prioritize Llama 3, Mistral, Gemma and broader Qwen GGUF families,
then Q4_K/Q5_K/Q6_K/Q8 and mixed tensors. For each addition, require a pinned
model fixture, malformed-file rejection, tokenizer/template IDs, deterministic
sampling output and a benchmark record. Keep the manifest's quantization
method explicit so an unsupported method fails at preflight.

### 3. Hardware breadth is declared faster than it is proven

MLC and ExecuTorch have first-class mobile/WebGPU/NPU deployment stories;
ONNX Runtime has a broad execution-provider ecosystem; MNN and LiteRT are
purpose-built for mobile delegates. Bloom has device probing and adapter
contracts, but CoreML/Vulkan/TensorRT do not execute in the default build and
the other vendor paths require external environments. The parallelism module
also documents no-op collective implementations, so multi-device strategies
must not be presented as working acceleration.

**Action:** choose one owner-backed accelerator cell. Apple-focused work should
choose Metal/MLX or Core ML; cross-platform work should choose CUDA or ONNX
Runtime GenAI. Connect provider selection, cancellation, memory sampling,
packaging and real-model tests before adding another skeleton. Keep
multi-device parallelism disabled in routing until collectives and placement
are executable and measured.

### 4. The current benchmark comparison is a smoke test

`scripts/compare_llamacpp.py` now accepts explicit context, Bloom
backend/device, warmup/repetition and thread settings, records them in the
JSON report, and passes context/seed/sampling to both benchmark commands and
`llama-cli`/`main`. It still mainly compares one parsed decode throughput
number; `llama-bench` cannot use the same prompt, and device layers, p95
statistics, memory provenance, output correctness and queue behavior remain
outside the harness. That is useful for detecting a gross regression, but it
cannot justify parity or a support promotion.

**Action:** extend the report with model SHA-256, native Bloom versus external
llama.cpp labels, TTFT/TBT/throughput percentiles, peak-memory provenance,
output correctness and a fail-closed comparison status when a setting is not
shared. Use `llama-cli` for same-prompt runs and reserve `llama-bench` for a
separately labelled kernel-throughput mode.

### 5. Multimodal and ranking breadth is incomplete

Bloom accepts bounded text/image/audio protocol inputs and has native BERT
embedding plus bi-encoder cosine reranking, but cross-encoder reranking and
trained vision/audio/video task evidence remain incomplete. llama-server and
ONNX Runtime GenAI provide useful references for grammar/tooling and multimodal
generation; MNN and ExecuTorch provide stronger mobile encoder paths.

**Action:** first publish one trained multimodal cell with exact image/audio
limits, stream/error behavior and quality metrics. Then add a native
cross-encoder interface with a separate task capability. Do not route a
bi-encoder as if it were a cross-encoder.

### 6. Distribution and ecosystem are smaller

Bloom has a useful C ABI and Python SDK, but it does not yet have a stable
binary-wheel matrix, mobile SDK packages, a broad model conversion toolchain or
the community model breadth of llama.cpp/Ollama. Rust feature combinations and
external Python bridges also make a clean install more involved.

**Action:** finish one release archive and one platform wheel matrix after the
runtime cell is stable. Reuse the current signed catalog and doctor output to
make missing runtimes actionable. Keep mobile SDKs as separate packages that
consume the same readiness and model-manifest contracts.

## Prioritized work plan

| Stage | Scope | Exit evidence | Why this order |
| --- | --- | --- | --- |
| P0: deployment cell | Qwen2 0.5B Instruct Q4_0, CPU first and one selected accelerator second; common tokenizer/template, seed, context and sampling | Exact digest/license, correctness, 3-run p95 TTFT/TBT, throughput, peak host/device memory, cancellation, restart, rollback and 24-hour soak | Converts Bloom's strongest contracts into a trustworthy product claim |
| P1: native coverage | Llama 3/Mistral/Gemma/Qwen GGUF; Q4_K/Q5_K/Q6_K/Q8; tokenizer/template and quantization preflight | One fixture and one benchmark/quality record per family and quantization class | Closes the largest everyday gap against llama.cpp |
| P2: one accelerator | Metal/MLX/Core ML for Apple, or CUDA/ONNX GenAI for cross-platform | Executable packaged runtime, provider/device diagnostics, cancellation, memory and target hardware report | Adds real speed or reach without multiplying skeletons |
| P3: runtime behavior | IFB fairness, paged KV, prefix reuse, speculative decoding, long context and backpressure | Concurrency curves, cache hit/eviction data, recovery and long-context reports | Makes Bloom's scheduling/cache architecture a measurable advantage |
| P4: tasks and mobile | Trained multimodal cell, cross-encoder rerank, then an MNN/ExecuTorch/LiteRT mobile package | Quality gates, per-task capability routing, mobile artifact and battery/memory report | Extends scope after the core runtime is dependable |
| P5: ecosystem | Stable C ABI/Python wheels, plugin ABI, conversion/import helpers and release provenance | Cross-version ABI matrix, clean-machine install, upgrade/rollback and signed artifacts | Lowers adoption cost after behavior is stable |

## Guardrails for implementation

1. **Do not equate an adapter with support.** A capability appears in routing
   only when the selected loaded model implements the typed operation and an
   evidence cell exists.
2. **Keep native and external evidence separate.** An external `llama-server`,
   MLX-LM or ONNX process proves compatibility at that boundary; it does not
   prove Bloom's native kernels, memory use or scheduler behavior.
3. **Prefer one complete backend over many probes.** A working provider with
   model, cancellation, diagnostics and benchmark evidence is more valuable
   than another skeleton adapter.
4. **Preserve the model lifecycle boundary.** Downloads, signatures, checksums,
   preflight and atomic replacement remain Bloom responsibilities even when
   execution is delegated.
5. **Make unsupported behavior explicit.** Return a protocol-shaped error with
   the missing task/backend and a next action; never silently fall back to a
   slower or semantically different task.

The practical target is therefore not “implement every engine.” It is a small
set of excellent, measured execution cells behind Bloom's secure, typed and
cross-client product surface.
