# Edge inference project shortlist — October 2026

This shortlist records projects that are practical references for Bloom's next
edge deployment cells. Upstream repository metadata and documentation links
were checked on 2026-10-13. A project appearing here is a selection input, not
a Bloom support claim.

| Project | Best fit | Model/runtime path | Integration effort | Why it is useful for Bloom |
| --- | --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Linux, macOS, Windows, Android, iOS; CPU, Metal, CUDA and other backends | GGUF; native C/C++ CLI and server | **Low** | Bloom already has a Llama.cpp executor and GGUF support. It is the shortest path to a reproducible CPU/Metal deployment cell and a useful reference for batching and streaming behavior. |
| [MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple Silicon | MLX arrays and quantized LLM checkpoints; Python-first tooling | **Medium** | High usability on macOS because unified memory and Metal are part of the runtime. It should be an isolated external adapter until a pinned model, Python environment, and cancellation contract are measured. |
| [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) | Cross-platform desktop and device deployments with execution providers | ONNX GenAI graphs; C++ API with language bindings | **Medium** | It is the clearest route for turning Bloom's existing ONNX bridge into a model-backed adapter with explicit provider selection and readiness checks. Conversion and provider packaging remain the main cost. |
| [MNN](https://github.com/alibaba/MNN) | Android, iOS and embedded C++ applications | MNN graphs plus mobile-oriented LLM and quantization paths | **Medium** | It has a compact native runtime and direct mobile focus, making it a strong candidate for a future Bloom-native mobile client or a separately packaged backend. |
| [ExecuTorch](https://github.com/pytorch/executorch) | PyTorch models on Android, iOS and embedded targets | Exported `.pte` programs with delegated backends | **Medium–high** | Best when the model pipeline already belongs to PyTorch. The export/delegate toolchain is heavier than GGUF, but it gives Bloom a principled mobile deployment path rather than a desktop runtime port. |
| [LiteRT](https://github.com/google-ai-edge/LiteRT) | Android and Google-supported edge hardware | Converted LiteRT models with hardware delegates | **Medium–high** | A good choice for Android and accelerator-specific delivery. It requires a conversion pipeline and delegate evidence, so it should follow a single-device acceptance test rather than enter the default server build. |
| [ncnn](https://github.com/Tencent/ncnn) | Small mobile applications, especially vision/encoder workloads | ncnn graphs with Vulkan acceleration | **Medium** | Mature mobile packaging and low native dependency count are attractive for encoder or multimodal preprocessing. It is less direct than llama.cpp or MNN for Bloom's generative LLM path. |

## Recommended order

1. **Promote one llama.cpp cell first.** Keep the current Qwen2 0.5B Instruct
   Q4_0 GGUF candidate, pin its digest and host, and publish cold/warm startup,
   streaming latency, concurrency, cancellation, memory and multi-hour soak
   artifacts. The streaming parser optimization in this iteration supports
   this path but does not promote its support level by itself.
2. **Choose one accelerator adapter with an owner and hardware.** Use MLX for a
   macOS/Apple-Silicon product cell, or ONNX Runtime GenAI for a cross-platform
   provider cell. Do not implement both before one has a trained model and
   repeatable benchmark evidence.
3. **Keep mobile runtimes behind a separate packaging boundary.** Select MNN
   for a compact native C++ mobile LLM path, or ExecuTorch when the source
   models are PyTorch exports. The mobile client should consume the same
   readiness and model-manifest contracts instead of widening the desktop
   server's default feature set.

## Acceptance gate for a new Bloom adapter

Before listing an adapter as executable in the support matrix, pin the runtime
version, model revision and license evidence; validate model loading and one
real inference path; expose truthful device/provider diagnostics; exercise
bounded cancellation and process shutdown; and publish TTFT, token rate,
peak-memory provenance, concurrency and failure-recovery results. Until those
artifacts exist, keep the adapter labelled external, experimental or skeleton.
