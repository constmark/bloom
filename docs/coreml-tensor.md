# Core ML numeric tensor adapter

Bloom can execute a bounded Core ML model on macOS using Apple's system
framework. The implementation embeds a small Swift worker, compiles it inside
a private temporary workspace with `/usr/bin/xcrun --sdk macosx swiftc`, compiles
the model with `MLModel.compileModel`, and runs predictions with
`MLModelConfiguration.computeUnits = .cpuOnly`.

Install Apple's Command Line Tools (`xcode-select --install`). Python,
`coremltools`, and additional Rust native dependencies are not required. The
worker source is embedded in Bloom's binary, so release archives do not need a
separate script. Private generated sources, executables, and compiled models
are removed when the loaded model is dropped.

## Supported contract

- One `.mlmodel` file or `.mlpackage` directory. A containing directory must
  contain exactly one model artifact; ambiguous directories fail admission.
- Exactly one required, floating-point multiarray input with a declared fixed
  shape, and one floating-point multiarray or double scalar output.
- Float16, Float32, and Float64 array storage. Bloom's public numeric vectors
  remain Float32; values that overflow their model dtype or the returned
  Float32 representation fail explicitly.
- At most 16 dimensions and 1,048,576 input or output values. Empty, oversized,
  overflowing, non-finite, and mismatched shapes fail before prediction.
- CPU-only execution. GPU and NPU selectors fail admission because Apple's
  `cpuAndGPU` and `cpuAndNeuralEngine` policies also permit CPU execution.
- Numeric model metadata declares `bloom_task: "tensor"`. The engine does not
  declare text generation, sentence embeddings, reranking, token streaming,
  or constrained language output.

Multiple inputs or outputs, optional inputs, flexible shape sets, stateful
models, image/audio preprocessing, tokenizers, and language-model decoding need
separate adapters and are rejected by this contract. Source model compilation
and contract validation happen during load, so incompatible models are not
published as an active numeric runtime.

## Rust inputs and outputs

Pass one typed tensor block through `LoadedModel::infer_request` or the
corresponding pipeline operation:

```rust,ignore
let model = CoreMlEngine.load(model_path, DeviceKind::Cpu)?;
model.infer_request(
    InferenceRequest {
        blocks: vec![DataBlock::Tensor(vec![1.0, 2.0, 4.0])],
        params: InferenceParams::default(),
    },
    &mut sink,
)?;
```

The model's input shape determines the reshape; callers cannot supply a
contradictory shape or silently omit another model feature. Mixed or multiple
blocks are rejected.

For existing CLI/FFI callers that carry `ModelInput::Text`, the prompt must be
a JSON number array, for example `[1,2,4]`. This compatibility input performs
numeric prediction; a natural-language prompt is rejected. Buffered results
use `ModelOutput.logits`, with `text` absent. `infer_stream` and `infer_request`
emit one `OutputChunk::Embedding` numeric vector followed by one `End`. The
shared vector event name does not imply semantic embedding support.

Use default generation parameters. Active sampling, token budgets, seeds,
penalties, logit bias, and JSON language-response formats are rejected rather
than silently ignored. An explicit text response format and an empty bias map
remain neutral.

The native Python SDK preserves the same numeric result and stream events:

```python
from bloom_sdk import BloomPipeline

with BloomPipeline("numeric.mlmodel", engine="coreml", device="cpu") as model:
    output = model.generate("[1,2,4]", max_tokens=128)
    print(output["logits"])
    chunks = list(model.generate_stream("[1,2,4]", max_tokens=128))
```

The tensor contract still requires the model's declared input length and the
neutral controls above; the SDK's language-generation token default is 256,
so numeric calls explicitly pass 128.

## Worker boundary

Compilation has a 120-second deadline. Model preparation and prediction use a
60-second deadline, configurable through `BLOOM_COREML_TIMEOUT_SECS` with a
strict 1–3,600 second range. The existing bounded process runner owns, kills,
and reaps workers, caps prediction stdout at 32 MiB and diagnostic stderr at
64 KiB, and closes stdin deterministically. Swift also bounds JSON input before
decoding. Numeric results carry a versioned CPU policy identity; unexpected
fields, output counts, identities, non-numeric values, and non-finite values
are rejected in Rust.

Each prediction starts a fresh worker and loads the private compiled model.
A loaded model permits at most one active prediction worker, so concurrent calls
cannot duplicate its admitted weights. Admission uses a nonblocking lock: a
competing call returns an explicit busy error immediately, and no second worker
starts. The lock is released after success, failure, or timeout.
This provides bounded, stateless execution but adds process/model-load cost;
it is not a resident high-throughput Core ML session. Caller sink cancellation
suppresses the successful terminal event; the finite prediction itself is
bounded by the process deadline.

## Reproducible CPU gate

On a macOS host with Command Line Tools:

```bash
cargo test -p bloomai-engine --no-default-features --features hardware-tests \
  executor::coreml -- --nocapture
```

The hardware test constructs a real 87-byte Core ML GLMRegressor directly from
Apple's protobuf field definitions, with no model download or conversion
package. Its affine computation is `y = 2*x0 - x1 + 0.5*x2 + 3`. It verifies
CPU policy identity, numeric buffered output, typed Tensor input, numeric stream
termination, Unicode model paths, malformed inputs, non-finite values,
unsupported controls, mixed-block rejection, and sink cancellation. A separate
busy-admission regression holds the worker lock and verifies rejection before
process spawn, immediate return, and admission release after worker failure. A second, real Scaler model verifies Float64 array input and array output
through an identity transform. The test prints both fixture SHA-256 values,
inputs, expected outputs, and affine model-load time.

This mechanical CPU model proves compilation, framework prediction, and the
adapter contract. It does not establish trained-model quality, LLM support,
GPU/ANE execution, production throughput, or long-soak stability. The adapter
remains experimental.

API contracts are grounded in the installed Apple SDK headers and
[Apple's compute-unit documentation](https://developer.apple.com/documentation/coreml/mlcomputeunits),
[Model protobuf](https://github.com/apple/coremltools/blob/main/mlmodel/format/Model.proto),
[feature protobuf](https://github.com/apple/coremltools/blob/main/mlmodel/format/FeatureTypes.proto),
and [GLM protobuf](https://github.com/apple/coremltools/blob/main/mlmodel/format/GLMRegressor.proto).

### Local execution record — 2026-11-03

The gate above passed the seven original cases on macOS 26.5.1, arm64, with Apple Swift
6.3 and Rust 1.97.1. The actual Core ML CPU worker returned affine outputs
`5`, `0`, and `4` for the three input vectors in the gate, and returned
`[-1.5, 2, 4]` from the array Scaler model. Both buffered numeric prediction and
one-vector-plus-terminal emission passed. Compilation, preparation, and load
of the affine model took 1,702 ms in this short run; this is an observation,
not a performance budget.

| Fixture | Bytes | SHA-256 |
| --- | ---: | --- |
| GLM affine, Float32 input and double output | 87 | `f8972456d5e685d0823a400fe22112ea8021962b2c73a597cbd987614c9f1888` |
| Scaler identity, Float64 input and output arrays | 48 | `5f947fa8d49882282e026dddfc4d9b56d8fa7700369489c8e579d97f39d5d9a3` |

The fixtures are generated from source inside the test and need no downloaded
weights. No trained model, accelerator-only placement, sustained throughput,
model upgrade, or long-soak result is inferred from this record.
