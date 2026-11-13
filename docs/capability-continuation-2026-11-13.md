# Capability continuation — 2026-11-13

This continues the [initial cross-layer review](capability-completion-2026-11-13.md)
with four implemented capabilities. It does not promote all models/platforms
to stable support or assert full llama.cpp parity.

A [subsequent continuation](capability-logprobs-metal-2026-11-13.md) adds token
logprobs, native model/embedding panels and current-build Metal evidence. The
remaining-scope and validation sections below describe this earlier snapshot.

## Native compressed model packages

`bloom_quantize --model-package` now exports dense Qwen2/Qwen3/Llama
Safetensors packages using versioned Bloom RTN v1 weights. INT4 stores two
values per byte, INT8 one per byte; metadata records precision, grouping and
matrix shape. The converter rewrites configuration and shard indexes and
preserves required tokenizer files. The loader checks dimensions, padding,
finite values, version/precision and package identity before restoring tensors.
Norm-only shards are valid inside a package; a purported RTN package with no
versioned quantized matrices is rejected.

The native Candle loader expands weights to floating point before execution.
This reduces stored weight size, not resident memory or matrix-multiply cost.
Admission reserves expansion/conversion peak memory without an mmap discount.
Single/sharded × INT4/INT8 tiny Qwen2 packages execute real CPU forward passes;
trained-quality and accelerator acceptance remain separate. See the
[format and conversion contract](quantization-export.md).

## Scheduled multiple-choice execution

The server now routes multiple choices and legacy prompt arrays through the
continuous-batching scheduler. Candidates execute sequentially within one HTTP
request's admission/runtime lease, while separate requests can batch together.
Candidate indices, seeds, token usage, tools/structured validation and stream
termination retain the direct-request contract. Cancelled candidates keep
scheduler ownership until cleanup; a later candidate cannot reuse live KV or
receive the prior candidate's token channel. Cumulative decoding preserves
Chinese/emoji across byte-level tokens; unexpected channel EOF fails explicitly,
and the producer caps the complete request's serialized event bytes at 16 MiB.
A stop match suppresses only an explicit stop sentinel, so an output-budget or
executor error in the same chunk cannot be converted into success. Existing
single-choice IFB routes also use cumulative Unicode decoding, exclude EOS
from content/usage, and reject premature channel closure.

Scheduler EOS detection now belongs to the loaded executor/model instead of
assuming token 2; the pipeline wrapper forwards this policy as well. This fixes
early termination on a normal vocabulary token and
failure to stop on other models' actual EOS IDs. See the
[OpenAI contract](openai-compatibility.md) for admitted combinations and limits.

## Persistent native macOS conversations

The SwiftUI application now provides a conversation sidebar, new/select/delete
operations, Cmd-N and restart recovery. It stores drafts and messages, including
periodic partial-generation checkpoints. Switching/deleting a conversation
cancels generation, and generation plus conversation identities prevent late
callbacks from modifying the next conversation.

A bounded versioned archive uses private temporary files and atomic replacement.
Malformed/unsupported archives remain intact until explicit recovery; write
failures are visible and retryable. Aggregate raw text is bounded before JSON
encoding, and the encoded archive is capped at 16 MiB. Credentials, endpoint
URLs and runtime state are excluded. See the [native client guide](../clients/macos/README.md).

Native build, packaging and state/storage regression tests are distinct from
visual acceptance. Computer-use inspection could identify the running app but
could not obtain a window (`cgWindowNotFound`); no screenshot or manual
screen-reader/visual acceptance is claimed for this continuation.

## Bundled Python native wheels

A maintained build command produces platform-specific ctypes wheels from the
SDK source distribution plus an ABI-checked native library. Wheels contain a
library inventory/hash, a policy-checked build-workspace SBOM and complete
RECORD, use non-pure platform placement,
and install without a Rust compiler or `BLOOM_FFI_LIB`. Explicit overrides remain
supported; bundled-library load failures cannot silently select a different
workspace/system runtime.

The builder snapshots the library and validates it in a subprocess, releasing
Windows DLL handles before temporary-directory cleanup. CI builds native wheels
on Linux/macOS/Windows and installs each artifact under Python 3.9, 3.12 and
3.14. Release jobs build release-mode libraries separately and attest wheels;
artifact selection is restricted to `release-bloom-*` so CI/debug wheels cannot
overwrite identically named release wheels. Policy regressions enforce that
separation and wheel attestation.

These are host-native wheels. Linux manylinux repair and broader binary
portability are not claimed. Local installation does not prove the unexecuted
hosted matrix or publication. See [SDK distribution](ffi-python.md).

## Remaining scope

The original gaps are reduced, not eliminated: wider native architectures,
packed execution kernels, Vulkan and other hardware coverage, multimodal
multiple choices, token log probabilities, broader grammar/tools, native-client
model management and encoder UI, trained conversion/tool/multimodal quality,
current-build long-run deployment evidence, hosted release validation and real
assistive-technology checks remain. No release or external publication was
performed by this continuation.

## Validation

Local validation on macOS arm64:

| Gate | Result |
| --- | --- |
| `cargo test --workspace --locked` | 1,369 passed, zero failed; one existing Laya parity test ignored because it requires weights and a Torch reference |
| Strict workspace/all-target Clippy and formatting | Passed |
| Workspace without default features | Passed |
| Python SDK with native FFI required | 41 passed |
| Native wheel contract / release-policy regressions | 6 / 11 passed |
| Isolated native wheel installation | Passed ABI/hash/RECORD/SBOM checks, buffered/streamed mock inference, and existing Qwen2 GGUF load, multilingual tokenizer round-trip and generation |
| Native macOS client | State/storage/stream/lifecycle checks and debug/release builds passed; packaged app rebuilt; visual acceptance remains unverified |
| Packed RTN execution | Single/sharded × INT4/INT8 package export/load/forward checks passed inside the quantizer tests |
| Native IFB HTTP profiles | Text, structured JSON/schema and tool profiles passed using real tiny Candle models and OpenAI SDK 3.37.0; includes n=1/n=2, prompt-array four-choice streams, Responses and tool-result continuation |
| Repository/workflow checks | Architecture, toolchain, Markdown links, immutable action pins and release-security checks passed |

The IFB regressions cover model-specific EOS forwarding, cross-token Unicode,
seed wraparound, prompt-major echo and usage, synchronous cleanup during active
execution/unrelated allocation, premature EOF, disconnects, bounded queues,
aggregate event bytes and stop/error precedence. Tiny fixtures prove actual
native execution and protocol behavior, not trained language/tool quality.

Local artifacts and copied logs are collected in
`output/capability-continuation-2026-11-13/`; `EVIDENCE.json` records source,
binary and evidence hashes. The native wheel in its `native/` directory uses
`py3-none-macosx_26_0_arm64`, matching this build host. Linux/Windows installation
and the Python 3.9/3.12/3.14 hosted matrix are configured but were not executed
in this local validation. No new 24-hour stability claim is made.
