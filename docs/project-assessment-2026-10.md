# Bloom project assessment — 2026-10

## Scope

This follow-up reviewed the native backend probes, FFI boundary, Python SDK,
Windows and macOS path handling, NPU/TTS defaults, workload evidence tooling,
and the local regression suite. The focus was cross-platform usability and
Unicode/multilingual input rather than adding another inference backend.

## Findings and remediation

| Area | Finding | Remediation |
| --- | --- | --- |
| Windows memory accounting | Windows capability reports used a fixed fallback and did not expose available physical memory. | Added `GlobalMemoryStatusEx` probing for total and available memory, with bounded conversion and scheduler-safe Linux clamping. |
| Backend discovery | The generic `gpu` alias always selected CUDA, and Python-backed probes assumed `python3`. Intel NPU discovery could treat any existing `PATH` entry as an OpenVINO installation. | `gpu` selects Metal on macOS; interpreter discovery honors `BLOOM_PYTHON` and platform launchers; NPU probes use explicit OpenVINO paths and common ARM64/runtime locations. |
| Native paths | ABI v2 limited model paths to the identifier limit, excluding long and multibyte Windows paths. Home-directory lookup ignored `HOMEDRIVE`/`HOMEPATH`; Llama.cpp and NPU/TTS helpers had Unix-only or hard-coded roots. | Added a 128 KiB UTF-8 model-path limit, Windows home fallbacks, portable Llama.cpp lookup, and configurable per-user model roots. |
| Python SDK | `PathLike` model paths, non-ASCII JSON, structured-output parameters, and localized diagnostics were incomplete. | Added UTF-8-preserving `PathLike` support, BCP-47 aliases for English/Chinese diagnostics, OpenAI/native response-format normalization, and response-shape validation. |
| Packaging | Setuptools 84 emitted deprecation warnings for the old license metadata form. | Migrated to SPDX `license` and `project.license-files`, requiring the supporting setuptools version. |
| Workload evidence | RSS and host-memory collection relied on POSIX commands, so Windows reports could be invalid or localized. | Added locale-independent Windows PSAPI and `GlobalMemoryStatusEx` collection, report the measurement source, and test the sampler in-process. |

## Validation

The local macOS/aarch64 run passed the full Rust workspace test suite, strict
Clippy, formatting, architecture checks, JSON artifact validation, HTTP
workload and benchmark evidence tests, the C ABI tests (including long
multibyte paths), the Python SDK unit and native ABI integration suites, and
the isolated SDK sdist-to-wheel installation gate. Cross-target checks passed
for the backend on Windows MSVC/MinGW and macOS ARM targets. The FFI Windows
MinGW check remains dependent on an upstream `esaxx-rs` build script that
unconditionally adds a libc++ flag; Windows host CI is the authoritative check
for that toolchain.

The broader support matrix remains accurate: most model/device combinations
are experimental until they have target-hardware correctness, performance,
and soak evidence. This iteration improves the boundary and deployment
experience without promoting an unmeasured accelerator path to stable.
