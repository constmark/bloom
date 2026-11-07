# Numeric inference from the CLI

HTTP consumers can use the bounded [tensor inference API](tensor-api.md) with
the same active model and numeric input contract.

`bloom_infer` preserves numeric tensor input as JSON for models declaring the
`tensor` task. CoreML accepts a flat number array matching its fixed input
shape; TensorRT accepts the named shape/data contract documented by its
adapter. No chat template or system instruction is applied to these payloads.

```bash
bloom_infer --model numeric.mlmodel --device cpu --prompt '[1,2]' --quiet
bloom_infer --model numeric.mlmodel --device cpu --prompt '[1,2]' --json
bloom_infer --model numeric.mlmodel --device cpu --prompt '[1,2]' --stream --json
```

Buffered numeric output prints a parseable JSON number array. `--json` prints
the complete `ModelOutput` object, including `logits` and all other output
fields. Streaming numeric vector events print JSON arrays in ordinary mode;
`--stream --json` emits one complete serialized `OutputChunk` per line, including
the terminal `"End"` event. Text output retains its existing ordinary format.
Diagnostics and timing remain on stderr; tensor inference does not report
numeric values as generated language tokens.

Tensor adapters do not implement sampling. Keep `GenerationParams` at its
default values: `max_tokens=128`, `temperature=0.7`, `top_p=0.9`, with no seed,
Top K, penalties, logit biases, or constrained response format. The CLI passes
explicit controls unchanged, and the engine rejects active non-default
controls. It also rejects interactive chat, text-generation benchmarks,
system prompts, and image attachments for a tensor model. `--json` cannot be
combined with benchmark, interactive, inspection, or engine-list modes.

Routing or inspection proves format compatibility only. Actual model loading
still validates the payload and vendor runtime. CoreML requires macOS and
executes its declared CPU path; TensorRT requires its installed CUDA runtime
and compatible hardware. These adapters retain their experimental labels.

TensorRT catalog acquisition also accepts direct-child `.plan` and `.engine`
files. The Models drawer offers both suffixes for browser import and verified
Hugging Face download; signed single-file discovery, installation matching,
and catalog/inventory records use the same identities. Catalog entries expose
format `tensorrt`, signed-index entries use their `plan` or `engine` extension,
and preflight exposes weight format `tensorrt_engine`. SHA-256, source-host,
license, operator authorization, quota, no-overwrite, and quarantine checks
apply exactly as for the other supported model files. A failed checksum never
publishes a catalog model. Hash verification establishes payload identity;
actual loading validates the TensorRT engine and runtime compatibility.
