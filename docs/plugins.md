# Plugins

Bloom's plugin system is used to extend backends, engines, processors, operators, and model packages. A plugin manifest is a capability declaration and a security boundary, not a sandbox; native, subprocess, and remote plugins may all execute external code.

## Entry Point Types

| Type | Purpose | Default validation | Execution boundary |
| :--- | :--- | :--- | :--- |
| `native` | Dynamic library plugins | Path exists; optionally check for the `bloom_plugin_init` symbol | Executes in-process and must be treated as trusted code. |
| `wasm` | WASM plugins | Path exists and ends with `.wasm` | Manifest validation only; WASM execution is not implemented. |
| `subprocess` | External binaries or scripts | Path exists | Executes in a separate process; working directory, environment variables, and input paths must be restricted. |
| `remote` | Network service plugins | Only `https://` or explicit local `http://127.0.0.1:` is allowed | Requires upper-layer authentication, timeouts, size limits, and auditing. |

`PluginManager::validate_entry_point` only performs a non-executing pre-check, which is suitable for default CI and mock plugin tests. `validate_native_library` actually loads the dynamic library and should only be invoked for trusted plugins or in dedicated tests.

## Mapping Capabilities to Core Traits

| Manifest field | Corresponding boundary | Description |
| :--- | :--- | :--- |
| `supported_families` | `Engine::supports` / `ModelManifest.family` | Declares the model families that can be handled. |
| `supported_dtypes` | `EngineCapability.supported_dtypes` | Declares the weight and compute dtype capabilities. |
| `supported_formats` | `ModelFile.format` | Declares input formats such as Safetensors, GGUF, OpenVINO IR, and ONNX. |
| `supported_devices` / `device_class` | `DeviceCapability` | Declares device classes such as CPU/GPU/NPU/Remote. |
| `supported_modalities` | `ModelIoSchema` / `ProcessorSpec` | Declares input/output domains such as Text, Audio, Vision, and Tensor. |
| `supports_streaming` | `LoadedModel::infer_stream` | Declares whether incremental output is supported. |
| `supports_quantized_models` | `QuantizationInfo` / backend kernels | Declares whether low-bit weights can be handled. |
| `max_context_tokens` | runtime admission / memory estimate | Used for request admission and context limits. |

Plugin declarations must not bypass core traits. When adding a new backend/engine/processor, the manifest capabilities need to be mapped to the corresponding trait's `supports`, `metadata`, or `spec` output. For the recommended boundary for third-party backends, see [`backend-adapters.md`](backend-adapters.md).

## Processors and the Model Manifest

A model manifest can declare the preprocessing pipeline required before and after loading via `processors`:

```json
{
  "processors": [
    {
      "name": "qwen-tokenizer",
      "kind": "TextTokenizer",
      "version": "1",
      "inputs": ["Text"],
      "outputs": ["Text"],
      "parameters": {
        "tokenizer_file": "tokenizer.json"
      }
    }
  ]
}
```

Processor plugins can declare behavior through `input_modalities`, `output_modalities`, `input_schema`, `output_schema`, and `deterministic`. When a model package references a processor, it must ensure that the processor's inputs and outputs are compatible with the model's `io_schema`.

## Default Testing Requirements

- Default CI only performs manifest/schema validation and entry point mock validation; it does not load unknown native libraries.
- Each plugin type needs error-path tests for missing dependencies and missing hardware.
- Tests for real hardware, vendor SDKs, and remote endpoints must be feature-gated or use a dedicated runner.
- Model packages must not commit large weights; only the manifest, hashes, license, and reproducible download/conversion instructions should be committed.
