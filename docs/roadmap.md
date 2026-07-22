# Roadmap

Bloom is pre-1.0. Work is prioritized by reliability and reproducible model
support rather than fixed release dates.

## Current priorities

- Replace panic-prone model-loading and scheduling paths with structured errors.
- Add reproducible real-model tests for the native Candle text path.
- Publish measured CPU, Metal, and CUDA results using the benchmark schema.
- Keep the README, support matrix, and runtime capability reports consistent.
- Split the largest CLI, server, scheduler, and executor modules.

## Next

- Expand native GGUF and Safetensors architecture coverage.
- Improve quantized kernels and paged-attention integration on Metal and CUDA.
- Finish production-ready embeddings, reranking, and structured generation.
- Harden the C ABI and Python SDK before declaring a stable API.

## Later

- Move ONNX Runtime, CoreML, MLX, Vulkan, and vendor SDK paths from probing to
  executable adapters where maintainers and hardware are available.
- Add multi-device execution only after single-device behavior has stable tests
  and benchmarks.
- Extend native multimodal streaming as model support matures.

The [support matrix](support-matrix.md) is the source of truth for what works
today. An item appearing here is not a support claim.
