# Public launch review — 2026-11-11

**Decision: not ready for a generally supported public release.** This review
closes reproducible software defects and makes release checks mandatory; it
does not turn an experimental backend into a stable service. The
[support matrix](support-matrix.md) remains authoritative. No release was
published, and no model/device row was promoted as part of this work.

The [2026-11-13 continuation](capability-continuation-2026-11-13.md) implements
native conversation persistence and bundled SDK wheels, with local checks;
the historical evidence below does not imply that their hosted/platform gates
have passed.

The smallest credible initial support scope is one pinned Qwen2 Q4_0 model,
Candle CPU, a declared OS/hardware configuration, the packaged Web UI and pinned
OpenAI/Ollama clients. Other model families, accelerators, external runtimes,
native clients and SDK binary wheels need their own acceptance evidence before
they become supported products. Broad llama.cpp parity is a separate roadmap,
not a prerequisite for a clearly bounded first release.

## Closed implementation defects

| Area | Observed defect | Correction |
| --- | --- | --- |
| Release dependency | A release tag could build archives without the complete test/audit pipeline succeeding | Release builds depend on the reusable CI workflow at the same revision, including native and UI lockfile audits. Failures prevent release draft creation. |
| Native UI build supply chain | macOS/Windows builders could bootstrap unverified auxiliary tools | Platform-specific hashes, bounded regular-file extraction, runtime version checks, and download-disabled Dioxus. Required Binaryen shared libraries are installed from the same verified archive. |
| Runtime preprocessing | Audio resampling and image resizing/normalization could allocate excessively or produce non-finite values | Shared payload limits, bounded output sizes, finite arithmetic and integer resampling positions. Invalid work fails before large allocation. |
| Temporary audio | Predictable names plus truncating file creation could follow a symlink | Random atomic private WAV creation, no-overwrite explicit writes and batch-failure cleanup. Unix files use mode 0600. |
| Persistent ASR worker | A worker that stopped responding could hold inference ownership indefinitely | Per-request deadline; timeout kills and reaps the worker, including its POSIX process group. Successful requests settle their watchdog before reusing the pipes. |
| Server configuration | An enormous HTTP timeout could overflow runtime deadline arithmetic | Startup and doctor both reject `--timeout` above 86400 seconds. Zero retains its documented behavior. |
| Native macOS connection | Public readiness alone could display Ready despite an invalid API key | Bounded authenticated Models probe; no credential-bearing redirects; ephemeral networking and stream bounds. |
| Streaming clients | Incomplete SSE framing, missing stream termination and indefinitely idle streams could be treated incorrectly | Incremental strict framing and UTF-8 validation, bounded reads, cancellation cleanup and stream contract checks in Web/native clients. |
| Multi-model UI | The authentication probe accepted only one model although the runtime supports multiple residents | Accept and validate up to 16 distinct model entries; a second resident no longer breaks connection validation. |
| Python installation | Python 3.8 was advertised despite a build backend requiring 3.9+ | Package metadata, documentation and clean-install gate now agree on Python 3.9+. |
| Dependency fixes | Compatible security fixes were absent from locks | Native `rustix` updated to 1.1.6; UI `rustix` and `pulldown-cmark` updated to fixed compatible versions. The native ZIP vulnerability gate is now closed by the dependency migration described below. |
| Deployment evidence | Inherited configuration could change a drill; weak readiness checks and missing latency/budget evidence could pass | Private config and environment isolation, full readiness validation, post-run hashes, latency consistency checks, complete service budgets and an explicit 24-hour/1,000-request production workload profile. |

Behavior changes: persistent ASR uses `BLOOM_ASR_TIMEOUT_SECS` (default 300,
accepted range 1–3600); after timeout, reload the model to restart its worker.
Large server request timeout values must be reduced to at most 86400 seconds.
Previously accepted malformed client streams now fail explicitly. Successful
processor-produced `AudioFile` values still transfer file cleanup to the caller.

## Remaining launch gates

| Priority / scope | Missing evidence or dependency | Concrete exit criterion |
| --- | --- | --- |
| Closed — native audit | Vulnerable `zip 7.4.0` transitive graph | Official Candle 0.11.1, explicit Laya compatibility package and official zip 9.5.1; zero known vulnerabilities in the native audit, with upstream provenance and SBOM checks. |
| P0 — stable support | No exact deployment cell has completed all published stability criteria | Retain a 24-hour, at least 1,000-request target-host report with preselected latency/memory budgets, identity-matched failure drills, trained quality and reference-comparison evidence. |
| P0 — operations | Isolated physical disk-full and cross-build rollback passed; production filesystem/service-manager configuration remains unverified | Test the real filesystem/service manager, recovery from a full disk, rollout and rollback, TLS/rate limits, readiness routing, termination grace, backup/restore and alert delivery; retain logs and run records. |
| P0 — distribution | Local macOS archive is packaged and self-checked; tagged cross-platform publication/provenance remains unverified | Pass the same-revision hosted Linux/macOS/Windows pipeline; validate downloaded archives, checksums, SBOMs and provenance; smoke-test installation on clean target systems. |
| Conditional — official containers | Digest/SBOM/provenance/scanning workflow is implemented; official published images remain unverified | Build and scan final runtime layers, publish immutable digests with signed provenance and inventory, validate read-only deployment and signal drain on the chosen orchestrator. |
| Conditional — accelerator claims | Target Metal/CUDA/NPU execution, peak memory and performance evidence is incomplete | Use dedicated real hardware and pinned models/runtimes; publish correctness, resource and latency budgets per cell. Compilation or simulated provider tests do not count. |
| Conditional — browser/accessibility claims | Chromium, Firefox and WebKit automated gates passed; target assistive-technology validation remains | Exercise packaged assets in each declared browser and target screen reader, including dialogs, streaming cancellation, clipboard, imports and storage failure. |
| Conditional — production SDK wheels | Platform-native wheels and cross-version ABI matrix are absent | Build/install native wheels on each advertised Python/OS/architecture, verify ABI compatibility and document the supported version window. |
| Conditional — signed fleet distribution | Multi-host revocation/rotation/restore operations remain unproven | Measure refresh/denial latency, alerts, watermark backups and replacement recovery across the actual fleet. |

Native macOS conversation persistence/model management, additional model
architectures, quantizations, Vulkan execution and broader compatibility fields
remain in the [capability inventory](project-assessment-2026-11.md) and
[parity matrix](llamacpp-parity.md). They must be either implemented and tested
for a later release or explicitly outside the first release's advertised scope.
They are not silently considered complete by this review.

### Dependency audit closure

The initial audit reported RUSTSEC-2026-0367, -0368, -0369 and -0429 for
transitive `zip 7.4.0`. A lockfile-only update could not satisfy Candle/Laya's
old semver constraints. The continuation upgrades to the official Candle 0.11.1
release and zip 9.5.1, and preserves Laya functionality through the explicitly
named `bloomai-laya 0.2.20-bloom.1` compatibility package. Its original upstream
source is unchanged; the manifest migrates Candle dependencies and records the
fork identity. The source/Apache license/commit/archive hashes and reviewed
changes are verified, and generated SBOMs retain the upstream provenance.
See the [fork notice](../crates/laya/NOTICE) and
[upstream record](../crates/laya/UPSTREAM.json).

The native audit now returns zero known vulnerabilities without ignores. The
two existing unmaintained dependency notices remain visible. The fork creates
a maintenance obligation: track upstream Laya and Candle changes, retain source
verification and rerun native model, Metal/API, package and ABI gates on every
upgrade. A passing audit does not imply that all future vulnerabilities are known.

## Initial validation record

The validation host is macOS/aarch64, using the repository's Rust 1.97.1 pin and
an isolated Python 3.12 environment. The shell's unrelated Homebrew Rust 1.94
must not be used; select the rustup shims first.

- Default Rust workspace: 1,259 tests passed, zero failures/ignored cases;
  strict workspace/all-target Clippy and formatting passed.
- Generated untrained Qwen2 single-file/sharded, IFB, embeddings, rerank,
  structured-output and function-call runtime gates passed with required
  OpenAI 3.37.0 and Ollama 0.6.3 clients. These prove integration mechanics,
  not trained-model quality.
- Real local process drills passed invalid-model rejection, hard interruption
  and catalog-lease reacquisition, and import quota rejection without residue.
  A short concurrent workload and its evidence validator passed. This is not a
  24-hour soak and is not used as production promotion evidence.
- No-default-feature engine/server: 890 tests passed. Elderwand integration:
  1,126 passed, five environment-gated cases ignored.
- Native macOS checks and release app build passed. Web UI: 150 host tests;
  WASM and strict Clippy checks passed; UI dependency audit reports zero vulnerabilities.
  Python SDK: 31 tests passed, including native ABI, plus clean sdist-to-wheel
  build and installation.
- Python tooling regression suite: 174 tests passed, including deployment
  evidence, release security, helper installation and archive contracts.
- Final production UI assets passed the real Chromium shell/security/semantics
  gate, axe-core 4.13.0 WCAG scans, focus containment, Escape/opener restoration,
  reduced motion, clipboard and conversation-download checks. Firefox and
  WebKit were subsequently verified in the continuation below; target
  assistive technologies remain unverified.
- Real server HTTP-boundary and shutdown gates passed, including catalog
  ownership, origin/authentication behavior, graceful drain and escalation.
- JSON schemas, readiness, architecture, toolchain and action-pin checks were
  run. The original native audit failure is resolved by the continuation above.

## Continuation validation record

The following results supersede the corresponding initial counts above. These
are local execution results, not claims that hosted release jobs have run.

- Bloom workspace: **1,335 passed, zero failures, one ignored** upstream Laya
  real-weight/Torch parity test. Strict workspace/all-target Clippy and
  formatting passed. Elderwand integration: **1,202 passed, zero failures,
  six ignored** environment-dependent cases. Both dependency graphs have zero
  known audit vulnerabilities; the two informational unmaintained notices
  remain. The parent application's lockfile is now eligible for version control.
- Real pinned Qwen2-0.5B-Instruct Q4_0 inference passed on Candle CPU, including
  the exact `Bloom` instruction result and the official OpenAI/Ollama clients.
  Generated single/sharded model, IFB, embedding, reranking, structured-output
  and tool-call integration checks also passed. The real model's SHA-256 is
  `aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f`.
- On the actual Apple M5, both AWQ/GPTQ Metal F32-scale reference tests passed,
  and trained Qwen2 Metal generation produced the exact expected `Bloom`.
  This closes the Candle Metal migration regression check; it does not measure
  a sustained Metal deployment or cover CUDA/NPU hardware.
- Chromium 157.0.8103.50, Firefox 153.0 and WebKit 26.5 passed against the same
  final embedded assets. All four scanned surfaces had zero axe violations
  and zero unresolved scanner items. Actual clipboard, keyboard containment,
  focus restoration, download and reduced-motion gates passed. The final UI
  has 150 host tests plus WASM and strict Clippy checks. This is not a manual
  screen-reader assessment or a Linux/Windows browser execution record.
- Python SDK: **31 passed with native FFI explicitly required**, plus isolated
  sdist-to-pure-Python-wheel build and installation. Equivalent native SDK jobs
  are wired for Linux/macOS/Windows; Windows execution and bundled native wheels
  remain unverified. All **eight crate archives** were extracted and compiled
  with all targets, including the independently named Laya package.
- Python release/deployment tooling: **216 tests passed**. The final sweep
  exposed a macOS transient `killpg` permission error when a process group
  contained only exiting zombies. A bounded Darwin-only retry preserves the
  original command exit status while persistent permission failures still
  fail cleanup. An additional 300 real subprocess cases passed. Failure-drill
  and deployment-evidence regression suites are now mandatory in CI.
- The macOS ARM64 archive was rebuilt with the final embedded UI and passed
  deterministic packaging, SBOM/policy, doctor, HTTP-boundary, shutdown and
  offline artifact checks. The packaged server's SHA-256 is retained in its
  release manifest and matched to the browser reports and final soak status.
  Final repackaging exposed embedded asset creation/modification timestamps
  changing the binary despite identical assets. Deterministic rust-embed
  timestamps remove that variability; this does not claim reproducibility
  across arbitrary compilers, hosts or build configurations.
- An isolated 64 MiB HFS+ disk image was actually filled until the OS returned
  `ENOSPC`. The server remained healthy, no partial model was published, and
  import/restart recovered after space was restored. An old→new→old trained
  Qwen2 rollback also passed with distinct binaries and equal deterministic
  outputs. This drill used candidate `bcc9e6e…` and previous `33ef65d…`, both
  version 0.1.0; it is cross-build evidence preceding the final UI binary,
  not a published-tag, APFS/ext4 or service-manager compatibility claim.
  See the [physical storage drill](storage-failure-drill.md).
- The container publication workflow, runtime security gate, final-layer
  inventory/scanning and digest/provenance checks are implemented, together
  with a reference deployment. A real current-source Docker attempt failed
  before compilation because Debian's signed package index returned HTTP
  500/502 even after bounded retries. No signature check was disabled. The
  hardened runtime smoke passed on an existing local image only; a new
  candidate image and remote publication are **not** validated. See the
  [container release procedure](container-release.md).

### Completed: real 24-hour CPU workload (verified 2026-11-13)

The final durable production workload is launched after packaging and browser
acceptance, with the same exact packaged server hash, the pinned trained
model, Candle CPU, two worker threads, concurrency 1/2 and preselected latency
and memory budgets. Its preflight failure drills passed. The run completed on 2026-11-12. Revalidation on 2026-11-13 with the current
production-profile evidence checker passed: 86,400.270313 measured seconds,
132,378 requests, concurrency 1/2, p95 first content 1,279.559 ms, p95 total
latency 1,342.682 ms and peak RSS 1,589,280,768 bytes. Disconnect recovery,
failure injection and all declared budgets passed. This evidence belongs to
server SHA-256 `bed498b42190e7c1eabf35e3add8522e88e3433a87c9f2796ac54411ec501172`;
it does not validate later code changes or imply a stable support promotion.
The retained report continues to declare `stable_claim: false`.

The private run directory is
`output/deployment-soak/qwen2-cpu-20261111-final-24h`. Its `status.json`, atomic progress,
workload, failure and validation records survive the initiating terminal. The
server, scripts and budget file are copied and hashed before launch. Each
phase has a deadline and an independent parent-death cleanup guard; cancellation
uses a marker in the owned directory instead of signaling a reused PID.
An idle-sleep assertion is tied to the worker and expires within 25 hours;
it does not prevent deliberate shutdown or system sleep. The earlier `-24h`
and `-24h-v2` runs were explicitly cancelled after the cleanup and timestamp
fixes so this run could snapshot the final helpers and packaged binary. Their
incomplete progress is retained and is not acceptance evidence.

```sh
python3 scripts/run_deployment_soak.py status output/deployment-soak/qwen2-cpu-20261111-final-24h
```

The short preflight is retained separately as
`output/deployment-soak/final-guarded-preflight`. Neither it nor intermediate
progress promotes this cell to stable support. Reference-engine comparisons,
the actual deployment filesystem/service manager, hosted cross-platform release
and downloaded attested artifacts still require their own acceptance records.
Local copies of the test, browser, trained-model, Metal, storage and failed
container-build records are retained under
`output/launch-readiness-2026-11-11`, with file hashes in `EVIDENCE.json`.

Use the [HTTP workload procedure](http-workload.md) for the production profile
and [release checklist](../RELEASE.md) for artifact acceptance. Evidence checks
validate report structure, consistency and declared limits; unsigned local JSON
reports are not attestations and cannot prove that a dishonest operator actually
ran a workload. Retain runner logs and signed build provenance with acceptance
records. A successful validator continues to emit `stable_claim: false`.
