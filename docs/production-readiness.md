# Production Readiness Gap Register

Bloom has a substantial reliability baseline, but the project is not yet a
generally production-supported inference service. The support matrix remains
authoritative for individual capabilities; this document records the main exit
criteria that cut across those capabilities.

## Existing strengths

- Bounded request admission, cancellation ownership, readiness, request
  correlation, no-store policy, and graceful shutdown have automated coverage.
- Model acquisition verifies hashes, records provenance, bounds storage, and
  supports crash-safe signed upgrades plus persistent index rollback and
  permanent exact-version revocation protection.
- Release archives are deterministic at the archive layer, self-checked, and
  published with build-provenance attestations.
- Pinned trained CPU gates cover maintained Qwen2, Qwen3, SmolLM2, and MiniLM
  profiles through the native runtime and compatibility APIs.
- Non-loopback listeners fail closed without authentication. The Dockerfile
  uses immutable base-image digests, a non-root runtime, isolated mutable state,
  and strict security and memory admission.
- Distinct inference and operator credentials are enforced across native model
  management and Ollama acquisition, deletion, activation, unload, and
  residency controls. Strict non-loopback deployments require different values;
  legacy single-key behavior remains available only outside that baseline.

## Blocking gaps

### P0: establish a supported deployment cell

No end-to-end inference combination is classified `stable`; the default Candle
paths remain experimental. Select at least one exact model revision,
quantization, backend, device, operating system, and client set, then publish:

- repeatable quality and compatibility results;
- cold-start, steady-state, saturation, disconnect, and shutdown measurements;
- peak host/device memory and disk budgets under the deployed concurrency;
- a multi-hour soak plus injected client, model-load, disk-full, and process
  interruption failures;
- explicit upgrade and rollback evidence on the target filesystem and service
  manager.

Promotion should apply only to that measured cell. It must not implicitly
promote other model families, accelerators, or external-runtime adapters.

The pinned Qwen2 Q4_0 Candle CPU profile is the first candidate. Its short
CI workload records identity, concurrent streaming latency, disconnect
recovery, and host RSS using the [HTTP workload procedure](http-workload.md).
Multi-hour target-host evidence and the failure-injection drills above are
still required before `stable` classification.

### P0: complete the chosen deployment artifact

Release archives have provenance, while the Dockerfile is CI-built and receives
short-lived runtime, strict-default, doctor, and embedded-browser checks but no
official container publication contract exists. If containers are a supported
deployment, publish a multi-architecture image by digest, attach an SBOM and
signed provenance, scan the final runtime layer, document a read-only root
filesystem/security context, and exercise liveness/readiness plus signal drain
under the target orchestrator. The Docker build now installs fixed
Linux/amd64 and Linux/arm64 `wasm-bindgen`, `esbuild`, and `wasm-opt` binaries,
verifies their SHA-256 digests, selects the compiler already present in the
digest-pinned builder without refreshing its online channel, and runs a
download-disabled `dx`; extend that same fail-closed toolchain contract to the
non-container macOS and Windows archive builders.

## High-priority gaps

- Add cross-browser and target assistive-technology validation. The current
  release-container gate covers the real WASM application shell, security
  headers, empty-model admission, document/landmark/control semantics,
  Models/Settings/import-dialog focus behavior, bidirectional Tab containment,
  Escape dismissal, opener-focus restoration, reduced motion, real clipboard
  writes, versioned conversation downloads, request failures, page errors, and
  unexpected console errors. Pinned axe-core 4.13.0 scans the application shell
  and all three exercised dialogs against WCAG 2.0, 2.1, and 2.2 A/AA rules;
  the gate also enforces the scanner archive's npm SHA-512 integrity and exact
  browser-side version.
- Add dedicated target-hardware runners and published performance budgets for
  each claimed Metal or CUDA deployment. Feature compilation alone is not
  execution evidence.
- Continue stabilizing and package the C ABI and Python SDK. Revision 2 now
  provides version negotiation, bounded length-delimited buffers, explicit
  ownership/status contracts, cooperative stream cancellation, and revision 1
  fallback tests. Python streaming now has bounded byte/chunk backpressure and
  close-safe cancellation; a self-contained sdist-to-wheel gate verifies clean
  installation and native ABI v2 interoperation. Platform binary wheels, a
  declared compatibility window, packaged cross-version ABI tests, and the
  complete supported Python/platform matrix remain before production support.
- Exercise signed-index v3 urgent revocation through a multi-host deployment
  drill, including forced refresh latency, operator alerting, state backup, and
  replacement-digest recovery evidence. The protocol and local fail-closed
  enforcement are implemented; fleet incident operations are not yet proven.
- Extend the native archive's dependency policy and target-filtered CycloneDX
  SBOM gate to any future official container image, including final-layer
  package inventory and signed image provenance.
- Split the largest server, CLI, scheduler, executor, and browser modules so
  security and lifecycle boundaries remain reviewable as features grow. Loaded
  runtime ownership now has a focused bounded pool, exact draining leases, and
  aggregate host/device memory admission; HTTP origin, correlation, protocol
  error, cache, retry, and credential middleware now has a focused boundary;
  model loading/publication and optional IFB scheduling construction are also
  isolated with their coupled resource lifetimes. The remaining request state,
  route composition, handler, UI, and Candle modules remain oversized.

## Runtime hardening delivered in this iteration

The engine now applies one bounded contract to embedded, FFI, plugin, and HTTP
callers before backend execution. Text, token, audio, image, video, tensor,
world-state, and action inputs reject oversized payloads, invalid MIME/path
metadata, non-finite values, invalid sample rates, and impossible generation
controls. Backend output is checked through the same contract; streaming sinks
reject malformed events after a terminal event and model outputs reject
non-finite vectors or inconsistent video metadata.

Quantized KV writes and cache restores use checked shape/scale validation rather
than assertion-based helpers. Quantization configurations reject impossible bit
widths and incompatible manifest dtypes. Direct memory reservations reject
allocations larger than the platform can represent. The ONNX, MLX, and NPU TTS
external bridges now share a bounded process runner: retained stdout/stderr
are capped, stdin closes deterministically, noisy workers are drained without
pipe deadlock, and hung workers are killed at a bounded deadline
(`BLOOM_ONNX_TIMEOUT_SECS`, `BLOOM_MLX_TIMEOUT_SECS`, or
`BLOOM_TTS_TIMEOUT_SECS`; TTS downloads use
`BLOOM_TTS_DOWNLOAD_TIMEOUT_SECS`, each capped at one hour). ModelScope
arguments are passed as process arguments instead of interpolated into Python
source. The llama.cpp capability probe applies the same boundary with a
64 KiB-per-stream cap and ten-second deadline. OpenVINO and Intel NPU
auto-export workers use the boundary as well, with a one-hour conversion
deadline.

The core package also exposes a thread-safe operator registry with explicit
dtype/layout/backend matching, deterministic priority selection, duplicate
registration rejection, and cycle-checked fallbacks. A no-default-features
server keeps a diagnostics-only Candle registration so doctor and preflight
return an actionable incompatibility instead of a missing-engine 500 or a
startup panic. These changes improve failure behavior and compatibility
truthfulness; they do not promote a skeleton backend or an unmeasured hardware
cell to production support.

## Operational boundaries

Bloom intentionally delegates TLS, network rate limiting, health/metrics ACLs,
and centralized logs to a reverse proxy or service platform. A production
reference deployment must make those dependencies concrete and test them; a
documentation statement alone is not deployment evidence. Responses state is
bounded but process-local, so clients must not depend on it surviving restart
or failover.

Review this register together with the [support matrix](support-matrix.md),
[production checklist](production.md), [security guide](security.md), and
[release checklist](../RELEASE.md) before promoting a deployment.

The [September project assessment](project-assessment-2026-09.md) records the
latest repository-wide review, reproduced failures, and follow-up acceptance
criteria. Performance-budget checks now reject missing or invalid evidence
instead of treating skipped metrics as a passing result.
