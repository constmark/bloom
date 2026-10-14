# Edge-engine goal

## North-star outcome

Bloom should be a dependable local inference product whose user-visible
capabilities reach parity with a pinned llama.cpp release, while its code stays
layered, replaceable, and easy to operate. Ollama is the reference for local
model management and API ergonomics; MLX LM and platform runtimes are
references for device-specific performance. These references guide the work;
they do not turn an adapter or a passing fixture into a support claim.

The goal is complete only when every capability in the pinned reference
inventory has a Bloom implementation or an explicitly versioned compatibility
decision, with correctness, performance, recovery, and client evidence. A
single CPU model passing a smoke test is an initial deployment cell, not the
parity result.

## What the goal includes

### Runtime parity

Bloom will cover the reference's model formats and architecture families,
quantization, tokenizer and chat templates, sampling, grammar-constrained
output, streaming, continuous batching, KV-cache and prefix reuse,
speculative decoding, embeddings and reranking, multimodal inputs where the
reference supports them, model conversion/inspection, and observability. Each
capability is routed only when the selected engine can actually execute it.

### Maintainable layering

Dependencies point toward stable contracts. A model format or device backend
can be replaced without changing HTTP handlers or the browser application.
Protocol adapters project the same typed application services into OpenAI,
Ollama, CLI, FFI, and future client protocols. New support is added by a
capability and evidence cell, rather than by copying routing logic into every
entry point.

| Layer | Bloom owner today | Owns | Must not own |
| --- | --- | --- | --- |
| Contracts and policy | `bloomai-core` | manifests, typed inputs/outputs, resource and scheduling policies, errors | HTTP DTOs, concrete tensors, browser state |
| Device and backend | `bloomai-backend` | device discovery, capabilities, reservations, memory probes | prompt formatting, HTTP, UI |
| Model and runtime engine | `bloomai-engine` | loaders, processors, executors, scheduler, KV/cache, sampling, plugins | CLI argument parsing, Axum, Dioxus |
| Process/application | `bloomai-app` | configuration, device selection, native binaries, local diagnostics | protocol-specific response projection |
| Server application | `bloomai-server` | lifecycle, catalog, admission, application services, protocol adapters | Candle tensors and model-wrapper downcasts |
| Presentation | `ui/`, `clients/macos`, desktop launcher | connection, interaction, accessibility, local session state | model execution and backend selection |
| Native integration | `bloomai-ffi`, `python/` | bounded ABI and SDK contracts | private engine internals |

The dependency direction is `core → backend → engine → app/server → clients`.
`bloomai-server` may compose `bloomai-app`; transport and presentation never
flow downward into the engine. The architecture gate and Cargo manifests must
describe the same graph. Module extraction is driven by ownership and reuse,
not by line count alone.

### Ease of use

One installation should expose a doctor command, model inspection, a safe
model catalog, a useful default device (`auto`), streaming chat, diagnostics,
and a clear error when a requested capability is unavailable. The same model
and server can be used through:

- the terminal (`bloom_infer`, `bloom_server`, and `bloom_desktop`);
- the Dioxus WebAssembly frontend, hosted independently or embedded in the
  server;
- the macOS SwiftUI client; and
- OpenAI/Ollama clients, the C ABI, and the Python SDK.

The responsive browser frontend is the cross-terminal baseline for Windows,
macOS, Linux, and mobile browsers. Native iOS, Android, and additional desktop
shells may follow by reusing the same contracts; they are not claimed to
exist today.

All clients must consume the same `/ready` protocol range, model/task state,
request IDs, authentication errors, cancellation semantics, streaming event
rules, and model identity. The Web client currently performs the most complete
readiness validation. The desktop launcher and macOS client must reach the same
validation level before they are treated as interchangeable clients.

## Capability truthfulness

`EngineCapability` is a routing contract, not a feature wishlist. A flag is
`true` only if the loaded-model implementation accepts the corresponding typed
operation and has a test or external-runtime evidence cell. In particular, the
current external llama.cpp adapter executes text completion through
`llama-server`; its embedding, reranking, and structured-output declarations
must be reconciled with the operations it actually forwards before that path is
advertised as parity. Unsupported operations fail before inference with a
useful diagnostic.

Native Bloom execution and an external llama.cpp process are separate evidence
tracks. The latter is a compatibility bridge and reference oracle; it cannot
be used to claim that Candle or another native backend has equivalent kernel,
memory, or scheduler behavior.

## Delivery stages

1. **Truthful baseline.** Pin the llama.cpp reference, generate a capability
   inventory, correct capability declarations, align the architecture gate
   with manifests, and make every client validate the same readiness contract.
2. **First deployment cell.** Stabilize the pinned Qwen2 Q4_0 CPU profile with
   deterministic correctness, OpenAI/Ollama streaming, measured comparison,
   cancellation, interruption, disk-full, upgrade/rollback, and multi-hour
   soak evidence.
3. **Runtime parity.** Add the missing model, quantization, sampler/grammar,
   batching/KV, speculative, multimodal, embedding/rerank, and accelerator
   cells. Each cell includes a fixture, a capability route, and benchmark or
   quality evidence.
4. **Shared cross-terminal experience.** Validate install, first model,
   streaming, stop, switching, diagnostics, recovery, keyboard, touch,
   accessibility, and narrow layouts on desktop platforms and mobile browsers.
5. **Parity release and maintenance.** Require every reference inventory row
   to have passing evidence, publish reproducible artifacts, and rerun the
   inventory whenever the pinned reference advances.

## Definition of done

The goal can be called complete only when:

- the parity matrix has no unexplained capability gaps;
- supported model/device/task combinations are marked by evidence in the
  support matrix, with external and native paths distinguished;
- correctness is deterministic under the declared test conditions;
- the reference-cell performance and memory gates pass, with reports tied to
  model digest, Bloom commit, reference revision, hardware, drivers, and
  build features;
- cancellation, restart, bad input, model replacement, and rollback recover
  without leaked runtime or catalog state; and
- the shared Web/desktop/mobile-browser workflows pass the readiness, API,
  streaming, error, accessibility, and responsive-layout checks.

The [parity and acceptance matrix](llamacpp-parity.md) supplies the measurable
cells. The [support matrix](support-matrix.md) remains the source of truth for
what Bloom supports at any point in time.

