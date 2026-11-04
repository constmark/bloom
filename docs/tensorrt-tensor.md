# TensorRT numeric execution

The `tensorrt` engine executes real serialized TensorRT 10.x `.engine`/`.plan`
artifacts through the bundled `tensorrt_infer.py` worker. It uses NVIDIA's
named-I/O API and CUDA Python runtime bindings. Install compatible TensorRT
10.x, `cuda.bindings`, NumPy and an NVIDIA driver in the interpreter selected
by `BLOOM_TENSORRT_PYTHON` or `BLOOM_PYTHON`. The default interpreter is
`python3`; use an explicit path on Windows or in virtual environments.

This is an external-runtime tensor adapter. No tokenizer, autoregressive text
decoder, TensorRT-LLM session, engine builder, explicit fused-attention call,
CUDA-graph capture or multi-device placement is emulated. The old placeholder
engine writer and fake attention success now return explicit errors. Only
CUDA device 0 is selected; CPU/NPU fallback is rejected.

## Input and result

Pass a JSON tensor request as the Text prompt to the CLI or Rust/FFI API:

```json
{
  "inputs": {
    "input": {"shape": [1, 3], "data": [1, 2, 4]}
  },
  "output": "prediction"
}
```

Input names must match every engine input exactly. Each tensor requires exactly
`shape` and flat `data`. Shapes must match the static engine declaration or
its dynamic optimization profile; unresolved data-dependent output shapes,
host/shape-I/O tensors and non-LINEAR/vectorized layouts are rejected. Select
an output by name when there is more than one; all outputs are bound and copied
correctly, with only the selected one projected into `ModelOutput.logits`.

Supported I/O storage types are Float16/32, Int8/32/64, UInt8 and Bool. Integer
inputs require exact integral values in dtype range; input booleans and
non-finite values are rejected. Bool outputs project to numeric 0/1. Returned
vectors must be finite and representable as Float32. Quantized weights inside
a compatible serialized plan do not imply an arbitrary quantization converter.

The numeric model declares `bloom_task: "tensor"`, so readiness and preflight
report it separately from generation, sentence embedding and reranking.
Generation HTTP routes reject it. Use the Rust, CLI or FFI numeric interface;
public multimodal HTTP does not admit raw Tensor blocks. Use default generation
parameters (`max_tokens=128`, `temperature=0.7`, `top_p=0.9`, no active sampling
extensions); non-default controls fail instead of being ignored.

```sh
BLOOM_TENSORRT_PYTHON=/path/to/venv/bin/python \
  bloom_infer --backend tensorrt --device gpu --model model.plan \
  --max-tokens 128 --temperature 0.7 --top-p 0.9 \
  --prompt '{"inputs":{"input":{"shape":[1,3],"data":[1,2,4]}},"output":"prediction"}' \
  --json
```

Buffered `--json` returns the complete ModelOutput, including numeric logits.
Plain numeric output is a JSON array. The stream compatibility path emits one
numeric vector and one End; it does not claim incremental token streaming.

## Ownership and limits

- Plans must be nonempty regular files of at most 8 GiB. Load records a SHA-256
  digest, checked against the actual bytes before every worker deserialization.
  Replacing an active model file therefore fails instead of changing its identity.
- Requests are at most 8 MiB, have at most 32 I/O tensors and rank eight, and
  allocate at most 1,048,576 combined I/O values. Zero/overflowing dimensions,
  duplicate JSON keys, unknown fields and missing inputs fail explicitly.
- Each prediction starts a fresh process, deserializes the plan and creates a
  context. Each loaded model admits one prediction worker; concurrent callers
  receive an explicit busy error instead of launching duplicate resident engines.
  Memory estimates include the serialized plan's host copy as well as GPU weights. CUDA buffers and streams are synchronized/freed on both success and
  error; the parent process runner owns a hard 600-second deadline, configurable
  by `BLOOM_TENSORRT_TIMEOUT_SECS=1..3600`, with bounded stdout/stderr.
- Strict security trusts only the exact bundled bridge location or an explicit
  `BLOOM_ALLOWED_SCRIPTS` grant; custom interpreters follow the runner allowlist.
  Release archives ship the bridge beside their executables.

The plan's weight/context/device memory is additional to these bounded I/O
buffers. TensorRT plans and runtime versions must be compatible with the
actual target GPU. Process-per-request ownership adds startup cost and is not
an optimized resident server session.

## Validation boundary

Rust tests execute the actual bounded worker protocol with a local fixture,
including typed probes, numeric results, one terminal and immutable plan
identity. Python regressions exercise strict input parsing, shape/dtype
boundaries, named binding, multiple outputs, CUDA errors and cleanup using
simulated NVIDIA interfaces. Install `requirements/tensor-runtime-test.txt`
for all NumPy boundary tests; standard-library parser tests run without it.
These tests do not establish real NVIDIA execution or performance. This host
has no CUDA GPU; target TensorRT plan/model tests remain a promotion requirement.

The runtime calls follow the [TensorRT 10.x Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html)
and [official CUDA Python runtime](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html).
