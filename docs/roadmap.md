# Roadmap

Bloom is pre-1.0. Work is prioritized by reliability and reproducible model
support rather than fixed release dates.

## Current priorities

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
- Evaluate an operator-controlled signed-index revocation and recovery format
  after the bounded overlap rotation and persistent watermark protocols gain
  deployment feedback.

## Next

- Expand native GGUF and Safetensors architecture coverage.
- Improve quantized kernels and paged-attention integration on Metal and CUDA.
- Expand trained embedding evidence beyond MiniLM and add a native
  cross-encoder reranking path; current reranking is normalized bi-encoder
  cosine similarity.
- Harden the C ABI and Python SDK before declaring a stable API.

## Later

- Move ONNX Runtime, CoreML, MLX, Vulkan, and vendor SDK paths from probing to
  executable adapters where maintainers and hardware are available.
- Add multi-device execution only after single-device behavior has stable tests
  and benchmarks.
- Extend native multimodal streaming as model support matures.

The [support matrix](support-matrix.md) is the source of truth for what works
today. An item appearing here is not a support claim.
