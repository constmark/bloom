# Roadmap

Bloom is pre-1.0. Work is prioritized by reliability and reproducible model
support rather than fixed release dates.

## Direction and delivery stages

The long-term target is full user-visible capability parity with a pinned
llama.cpp reference, a maintainable layered implementation, and one shared
frontend across desktop and mobile browsers. See the
[edge-engine goal](edge-engine-goal.md) and
[parity matrix](llamacpp-parity.md) for acceptance requirements. The support
matrix records delivered behavior; roadmap entries describe intended work.

Deliver the goal in this order:

1. **Establish a truthful baseline.** Pin the reference revision, inventory its
   capabilities, reconcile advertised engine capabilities with implemented
   operations, restore agreement between dependency declarations and the
   architecture gate, and unify client readiness/version checks.
2. **Validate the first native deployment cell.** Complete correctness,
   measured comparison, sustained load, cancellation, failure recovery, and
   release/rollback evidence for the maintained Qwen2 Q4_0 CPU profile. Keep
   native results separate from external llama.cpp execution.
3. **Expand capability and hardware parity.** Close model, quantization,
   sampling, constrained-generation, scheduler/KV, accelerator, multimodal,
   conversion, and embedding/reranking gaps against the reference inventory.
   Promote each model/device/task combination from reproducible evidence.
4. **Validate the shared cross-platform experience.** Exercise install,
   first-model setup, streaming, stop, model switching, diagnostics, and
   recovery on Windows/macOS/Linux and mobile browsers using common API
   contracts; validate keyboard, accessibility, touch, and narrow layouts.
5. **Close and maintain full parity.** Resolve every reference capability row,
   publish performance and compatibility evidence, and re-run the inventory
   and regression gates when advancing the pinned reference.

Client usability and focused module extraction proceed throughout these
stages. Completing the first deployment cell is an initial milestone, not the
final parity claim.

## Current priorities

- Complete the first candidate deployment cell: pinned Qwen2 0.5B Instruct
  Q4_0 GGUF on Candle CPU through OpenAI streaming. A short concurrent HTTP
  workload now runs in CI and a 30-second local release run has been recorded;
  repeated multi-hour target-host runs, failure injection, measured budgets,
  and upgrade/rollback evidence are still required before stable support.
- Broaden pinned, license-reviewed trained-model evidence beyond the maintained
  official Qwen2 0.5B Instruct Q4_0, Qwen3 0.6B Q8_0, SmolLM2 360M
  Instruct Q8_0 GGUF and BF16 Safetensors, and MiniLM sentence-embedding CPU
  gates. These verify exact package
  and license-evidence hashes, architecture-specific loading, tokenizer and
  bounded prompt metadata, native and quantized forward execution, truthful
  runtime memory accounting, deterministic instruction following, benchmark
  output, and exact OpenAI/Ollama behavior. MiniLM additionally proves native
  BERT routing, attention-mask-aware variable-length batching, semantic
  separation, bi-encoder reranking, model-bounded context, and encoder-only task
  isolation. Wider Qwen, Llama, and embedding checkpoints, trained public
  sharded Safetensors packages, quantizations, cross-encoder reranking, and
  trained tool selection remain next. Deterministic untrained Qwen2 fixtures
  remain the exhaustive mechanical gates for single-file and indexed sharded
  Safetensors, embeddings, reranking, structured output, and native function
  lifecycles.
- Publish measured CPU, Metal, and CUDA results using the benchmark schema.
- Keep the README, support matrix, and runtime capability reports consistent.
- Split the largest CLI, server, scheduler, and executor modules.
- Exercise the implemented signed-index v3 revocation and replacement-digest
  recovery protocol in a multi-host incident drill, including refresh latency,
  alerting and durable state recovery.

## Next

- Expand native GGUF and Safetensors architecture coverage.
- Improve quantized kernels and paged-attention integration on Metal and CUDA.
- Expand trained embedding evidence beyond MiniLM and add a native
  cross-encoder reranking path; current reranking is normalized bi-encoder
  cosine similarity.
- Extend the bounded C ABI/Python SDK and isolated sdist/wheel gate to platform
  binary wheels, cross-version ABI tests, and a declared compatibility window.

## Later

- Move ONNX Runtime, CoreML, MLX, Vulkan, and vendor SDK paths from probing to
  executable adapters where maintainers and hardware are available.
- Add multi-device execution only after single-device behavior has stable tests
  and benchmarks.
- Extend native multimodal streaming as model support matures.

The [support matrix](support-matrix.md) is the source of truth for what works
today. An item appearing here is not a support claim.
