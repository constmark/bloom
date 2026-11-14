# Capability continuation: logprobs, native tools and Metal — 2026-11-13

The [2026-11-14 audit](completion-audit-2026-11-14.md) adds CLI logprobs and
native reranking and corrects cross-layer execution issues. The remaining-work
statements below describe this earlier snapshot.

This continues the [package/IFB/client-history pass](capability-continuation-2026-11-13.md).
It adds working token probabilities, native model/embedding tools and fresh
Apple GPU execution evidence. The changes remain experimental within the
explicit capability boundaries below.

## Token probabilities across native and HTTP APIs

The CPU Candle text executor now emits selected-token scores, exact token bytes
and bounded sorted alternatives. Rust, C ABI and Python preserve this metadata;
the server maps it to Chat and legacy Completion buffered/SSE shapes, including
multiple choices and legacy prompt arrays. The JSON ABI addition preserves the
existing ABI revision and omits the new parameter for ordinary Python calls.

The implementation validates UTF-8 reconstruction and terminal metadata/text
agreement, rejects missing/invalid usage, includes sampled EOS in HTTP usage,
and fails without success events if a candidate becomes inconsistent. Both
first-token EOS and EOS at the last reserved token produce a normal stop.
Metadata has per-token, per-candidate and whole-request memory bounds.

This is direct CPU text support for ByteLevel or decoder-free WordLevel
tokenizers. GPU logprobs, IFB, speculative decoding, multimodal input, active
tools, structured output, custom stop sequences and legacy prompt echo remain
explicitly rejected. The [logprob contract](token-logprobs.md) describes score
normalization, Unicode fragments, legacy offsets, limits and protocol details.

## Native macOS model and embedding workspaces

The SwiftUI client now discovers the configured server catalog, loads/switches
and unloads models, polls lifecycle state, displays failures and permits retry.
Operator credentials are separate, remain in memory and fall back to inference
credentials when omitted. Changing models cancels active chat/embedding work
and invalidates old readiness. Endpoint/credential epochs discard late network
callbacks. Stopping monitoring ends client polling, not the server load.

The embedding panel accepts 1–256 lines with optional output dimensions,
binds each request to the confirmed active model, validates finite vectors and
response identity/order, and provides preview plus JSON copy/save. It keeps
inputs/vectors out of conversation history. Native downloads/import and a
reranking workspace remain future work; see the
[native client guide](../clients/macos/README.md).

Live GUI acceptance used an isolated loopback Metal server with generated
text/embedding fixtures. The rebuilt application completed readiness, displayed
both catalog entries, switched to the embedding runtime and generated two
four-dimensional vectors with four input tokens. The native panels and result
preview were inspected through accessibility state and screenshots. This proves
the UI/network workflow, not trained embedding quality or screen-reader usability.

## Apple Metal acceptance

The final implementation was rebuilt with Metal and executed on an Apple M5
with ten GPU cores and Metal 4. All 16 phases passed: native AWQ/GPTQ kernel
parity, deterministic CPU/GPU CLI text equality, and real HTTP text,
structured-output, function-tool, embedding and reranking paths, including IFB
and pinned OpenAI SDK decoding. The tiny profiles demonstrate real native
execution and protocol mechanics, not trained language/tool quality.
See [reproduction and hardware evidence](metal-runtime-validation-2026-11-13.md).

## Validation

| Gate | Observed result |
| --- | --- |
| Workspace Rust tests | 1,385 passed, zero failed; one pre-existing Laya weights/Torch parity test ignored |
| Strict all-target Clippy, formatting, no-default-features build | Passed |
| Python with native FFI required | 42 passed, zero skipped |
| Swift regression harness | Stream/state, persistence and model/embedding groups passed; release app packaged |
| Native GUI | Actual Metal connection, catalog, model switch and two-input embedding result passed; layouts visually inspected |
| Final Metal runtime | 16/16 phases; includes 2/2 hardware kernel tests and seven HTTP scenarios |
| Tiny CPU logprobs | 10 groups: Chat/legacy × n=1/2 × top=0/2, buffered/SSE, plus both SDK routes |
| First-token EOS | Same 10 groups; empty metadata/text, normal stop and exactly one completion token per candidate |
| Trained Qwen2 logprobs | All 10 buffered/SSE/SDK groups passed against the cached trained GGUF, without a model download |
| Trained Qwen2 Unicode | Chat/legacy n=2, top=2; exact `你好你好`, raw-byte reconstruction and legacy offsets `[0, 2]` |
| Native wheel | Current release library bundled; ABI/hash/RECORD/SBOM and isolated installation passed; installed wheel reproduced trained Qwen2 text/logprobs and rejected structured-output combinations |
| Repository policy | Architecture, toolchain, action pinning and release workflow checks passed |

The Rust regressions additionally cover cross-token Unicode fragments, missing
or mismatched metadata, malformed native usage, and early candidate failure
without later candidates or success usage. Final runtime reports retain binary
and model digests and raw request/response artifacts.

Local evidence is collected in `output/capability-logprobs-metal-2026-11-13/`.
Its `EVIDENCE.json` records the source snapshot, build artifacts, commands and
log hashes. No external release or publication was performed.

## Remaining scope

The remaining work includes packed RTN execution kernels, wider native model
architectures and decoder support, GPU/IFB logprobs, multimodal multiple choices,
broader grammar/tool semantics, trained conversion/tool/multimodal quality,
CUDA/Vulkan/Intel target-hardware acceptance, current-build long-duration soak,
hosted cross-platform release execution and real assistive-technology checks.
Native wheel tags describe the host platform; Linux manylinux portability is
not claimed. The Metal tests do not establish accelerator performance or long
run memory stability. These boundaries remain explicit rather than inferred
from a successful CPU or tiny-fixture run.
