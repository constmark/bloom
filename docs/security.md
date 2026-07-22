# Security Boundaries

Bloom loads model files and can launch external runtimes or plugins. Treat all
three as executable supply-chain inputs.

For vulnerability reporting, see [SECURITY.md](../SECURITY.md).

## Model files

- Load models only from controlled directories.
- Do not build model paths from untrusted request input.
- Do not load files while they are still being downloaded.
- Declare SHA-256 hashes in `bloom.json` for released model packages.
- Do not set `BLOOM_ALLOW_HASH_MISMATCH=1` in production.

Bloom verifies every declared `files[].hash_sha256` before loading. Files without
a declared hash are not verified.

## External runtimes

The following paths can execute external code:

| Path | Trigger |
| --- | --- |
| OpenVINO export | `BLOOM_OPENVINO_AUTO_EXPORT=1` or `BLOOM_NPU_AUTO_EXPORT=1` |
| FunASR or Qwen ASR | `--backend funasr` |
| Intel NPU tooling | `--backend intel-npu` |
| TTS bridge | `--backend npu-tts` |
| LongCat runner | `BLOOM_LONGCAT_RUNNER` or `BLOOM_MNN_DIFFUSION_DEMO` |
| llama.cpp | `BLOOM_LLAMA_CPP_SERVER` or a discovered `llama-server` |

Pin executable paths, runtime versions, and Python environments. Do not point
these settings at user-uploaded scripts or binaries.

## Plugins

A plugin manifest is a capability declaration, not a sandbox.

- Native plugins run in the Bloom process with its permissions.
- Subprocess plugins inherit an environment and filesystem context unless the
  deployment restricts them.
- Remote plugins can send model input over the network.
- WASM manifests are validated, but a sandboxed execution runtime is not yet a
  default capability.

Production deployments should load plugins from an allowlist and keep native
libraries outside user-writable directories.

## Strict mode

Enable strict security checks with:

```bash
BLOOM_STRICT_SECURITY=1 bloom_server --model /models/example
```

Custom external components must then be listed explicitly:

| Variable | Value |
| --- | --- |
| `BLOOM_ALLOWED_SCRIPTS` | Comma-separated script paths or names |
| `BLOOM_ALLOWED_RUNNERS` | Comma-separated executable paths or names |
| `BLOOM_ALLOWED_PLUGINS` | Comma-separated plugin manifest names |

Strict mode reduces accidental execution but does not sandbox an allowed
component.

## Network and data

- Keep `/metrics` and health endpoints behind an internal network or proxy ACL.
- Use a fixed remote-plugin endpoint with authentication, timeouts, and request
  size limits.
- Do not log sensitive prompts, images, audio, tokens, or private model paths.
- Verify whether vendor runtimes collect telemetry before production use.
- Use pinned offline model directories when network access is not required.

## Memory safety at startup

Set `BLOOM_STRICT_MEMORY_BUDGET=1` or pass `--strict-memory-budget` to fail
before loading when the estimated model footprint exceeds available memory.
This is an availability control, not a substitute for process isolation or
operating-system resource limits.
