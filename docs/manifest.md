# Model Manifest

`ModelManifest` declares a model package's files, capabilities, I/O, and memory
requirements.

## bloom.json

Place `bloom.json` in the model directory.

The formal JSON Schema is located at [`examples/manifest/model-manifest.schema.json`](../examples/manifest/model-manifest.schema.json). The plugin manifest schema is located at [`examples/plugins/plugin-schema.json`](../examples/plugins/plugin-schema.json), and the benchmark output schema is located at [`examples/benchmark-schema.json`](../examples/benchmark-schema.json).

### Field Descriptions

- `id`: Unique identifier of the model.
- `family`: The model family it belongs to, e.g. `Qwen`, `Llama`, `FunAsr`.
- `version`: Version string.
- `io_schema`: Defines the input/output modalities of the model, e.g. `["Audio"] -> ["Text"]`.
- `memory_profile`: Declares the minimum and recommended RAM and VRAM required to run the model (in bytes).
- `files`: Contains the list of files that make up the model, their formats, and hash checksums. If any required file is missing, the engine will refuse to load and return a structured error.
- `runtime_hints`: Hints for the runtime, such as the preferred backend, whether mmap is supported, and whether streaming inference is enforced.
- `primary_dtype`: The primary data type, e.g. `F16`, `Q4`, `Q8`.

If the engine cannot find `bloom.json`, it will attempt to read the Hugging Face standard `config.json` to infer the above fields as a fallback adapter.
