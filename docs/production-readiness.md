# Production Readiness Gap Register

Bloom has a substantial reliability baseline, but the project is not yet a
generally production-supported inference service. The support matrix remains
authoritative for individual capabilities; this document records the main exit
criteria that cut across those capabilities.

## Existing strengths

- Bounded request admission, cancellation ownership, readiness, request
  correlation, no-store policy, and graceful shutdown have automated coverage.
- Model acquisition verifies hashes, records provenance, bounds storage, and
  supports crash-safe signed upgrades and persistent index rollback protection.
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

### P0: complete the chosen deployment artifact

Release archives have provenance, while the Dockerfile is only build-tested in
CI and no official container publication contract exists. If containers are a
supported deployment, publish a multi-architecture image by digest, attach an
SBOM and signed provenance, scan the final runtime layer, document a read-only
root filesystem/security context, and exercise liveness/readiness plus signal
drain under the target orchestrator. The Docker build now installs fixed
Linux/amd64 and Linux/arm64 `wasm-bindgen`, `esbuild`, and `wasm-opt` binaries,
verifies their SHA-256 digests, selects the compiler already present in the
digest-pinned builder without refreshing its online channel, and runs a
download-disabled `dx`; extend that same fail-closed toolchain contract to the
non-container macOS and Windows archive builders.

## High-priority gaps

- Add real-browser end-to-end and accessibility gates for the embedded UI;
  current interaction, clipboard, download, focus, and assistive-technology
  behavior still requires manual validation.
- Add dedicated target-hardware runners and published performance budgets for
  each claimed Metal or CUDA deployment. Feature compilation alone is not
  execution evidence.
- Stabilize and package the C ABI and Python SDK, including version negotiation,
  binary wheels for supported targets, ownership/error contracts, and ABI
  compatibility tests, before treating them as production integration APIs.
- Define a signed-index revocation and incident-recovery mechanism; expiry and
  rollback watermarks do not provide urgent remote revocation.
- Add dependency license/source policy enforcement and an artifact-level SBOM
  gate alongside the existing vulnerability audit.
- Split the largest server, CLI, scheduler, executor, and browser modules so
  security and lifecycle boundaries remain reviewable as features grow.

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
