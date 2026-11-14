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
- Release archives are deterministic at the archive layer and self-checked;
  the publication workflow requires build-provenance attestations.
- Pinned trained CPU gates cover maintained Qwen2, Qwen3, SmolLM2, and MiniLM
  profiles through the native runtime and compatibility APIs. The trained
  MS MARCO MiniLM cross-encoder gate adds raw-score ranking margins, padded
  batch consistency and task isolation; its small authored dataset does not
  establish general retrieval benchmark parity.
- Non-loopback listeners fail closed without authentication. The Dockerfile
  uses immutable base-image digests, a non-root runtime, isolated mutable state,
  and strict security and memory admission.
- Distinct inference and operator credentials are enforced across native model
  management and Ollama acquisition, deletion, activation, unload, and
  residency controls. Strict non-loopback deployments require different values;
  legacy single-key behavior remains available only outside that baseline.

## Blocking gaps

### Dependency audit closure — 2026-11-11

The native vulnerability gate now passes without advisory ignores. Official
Candle 0.11.1 and zip 9.5.1 replace the vulnerable ZIP graph. Laya remains
available through the explicitly named `bloomai-laya` compatibility package,
which preserves the verified upstream 0.2.20 code and Apache license while
migrating its Candle dependency. Original archive hashes, source commit, file
hashes and reviewed changes are recorded in `crates/laya/UPSTREAM.json` and
checked by `scripts/check_laya_upstream.py`. SBOMs expose this fork's provenance;
crate packaging resolves its actual registry dependency rather than relying on
a checkout-only Cargo patch. The fork must track future upstream/security
updates until an equivalent supported upstream dependency is available.

The release workflow depends on complete same-revision reusable CI, including
native and UI audits. Two upstream unmaintained-dependency notices remain
visible; they are not vulnerability ignores or a zero-risk claim. See the
[launch review](launch-readiness-2026-11-11.md) for measured closure evidence.

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
The previous packaged server completed a revalidated 24-hour local CPU run
with 132,378 requests; its exact build/model identity is retained in the
[completion review](capability-completion-2026-11-13.md). This historical result
does not validate subsequent code changes. Current-build sustained validation
remains required. Local physical HFS+ disk-full and
distinct-binary trained rollback drills have also passed. Exact target-host
evidence, reference comparisons and the service-manager checks above are still
required before `stable` classification; see the
[continuation record](launch-readiness-2026-11-11.md#continuation-validation-record).

The workload evidence gate now requires a passing report from
`scripts/deployment_failure_injection.py` covering invalid-model admission,
POSIX process interruption and catalog-lease reacquisition, and model-storage
quota rejection without staging residue. Signed upgrade recovery has focused
CPU coverage for file and package directory transactions, including restart
rollback before replacement publication and restart completion after it; a
target-host upgrade/rollback run remains required for deployment promotion.

### P0: complete the chosen deployment artifact

Release archives have a provenance publication gate, while the Dockerfile is CI-built and receives
short-lived runtime, strict-default, doctor, and embedded-browser checks but the
official multi-architecture publication workflow is now implemented but has not
yet produced verified public registry artifacts. If containers are a supported
deployment, publish a multi-architecture image by digest, attach an SBOM and
signed provenance, scan the final runtime layer, document a read-only root
filesystem/security context, and exercise liveness/readiness plus signal drain
under the target orchestrator. The Docker build now installs fixed
Linux/amd64 and Linux/arm64 `wasm-bindgen`, `esbuild`, and `wasm-opt` binaries,
verifies their SHA-256 digests, selects the compiler already present in the
digest-pinned builder without refreshing its online channel, and runs a
download-disabled `dx`. Native archive builders now also use checksum-pinned
platform helpers and download-disabled Dioxus; cross-platform hosted execution
remains part of the release gate.

## High-priority gaps

- Complete target assistive-technology validation. The same gate now runs on
  Chromium, Firefox and WebKit, with real per-engine reports; the local runs
  passed with zero axe violations and unresolved scanner items. It covers the real WASM application shell, security
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
- Complete hosted compatibility validation for the packaged C ABI and Python SDK. Revision 2 now
  provides version negotiation, bounded length-delimited buffers, explicit
  ownership/status contracts, cooperative stream cancellation, and revision 1
  fallback tests. Python streaming now has bounded byte/chunk backpressure and
  close-safe cancellation; a self-contained sdist-to-wheel gate verifies clean
  installation and native ABI v2 interoperation. Bundled host-native wheels and
  packaged ABI checks are implemented, with local macOS isolated installation
  evidence. The hosted Linux/macOS/Windows and Python 3.9/3.12/3.14 matrix,
  a declared compatibility window and broader binary portability still require
  release-candidate evidence before production support.
- Exercise signed-index v3 urgent revocation through a multi-host deployment
  drill, including forced refresh latency, operator alerting, state backup, and
  replacement-digest recovery evidence. The protocol and local fail-closed
  enforcement are implemented; fleet incident operations are not yet proven.
- Execute the implemented container dependency-policy, target-filtered
  CycloneDX SBOM, final-layer inventory/scanning and signed-provenance workflow
  successfully on the final candidate. The local new-image build was blocked
  by Debian package-index HTTP 500/502; an older image's runtime checks do not
  satisfy this gate.
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

The [November capability assessment](project-assessment-2026-11.md) records
the latest software fixes and remaining capability/evidence inventory. The
[September assessment](project-assessment-2026-09.md) retains the earlier
repository-wide review and acceptance criteria. Performance-budget checks now reject missing or invalid evidence
instead of treating skipped metrics as a passing result.
