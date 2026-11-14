# C ABI and Python SDK

Bloom exposes a pre-1.0 C ABI in `crates/ffi/bloom.h` and a small `ctypes`
wrapper in `python/bloom_sdk`. The API is usable for integrations, but its ABI
is not declared stable until the project reaches the corresponding roadmap
gate.

## Build and load the native library

Build the shared library with the default Candle engine:

```bash
cargo build --release -p bloomai-ffi
```

The artifact is named `libbloom_ffi.so` on Linux, `libbloom_ffi.dylib` on
macOS, and `bloom_ffi.dll` on Windows. The Python wrapper searches the workspace
release and debug directories. An application or installed wheel should set an
explicit path:

```bash
export BLOOM_FFI_LIB=/absolute/path/to/libbloom_ffi.so
python3 -m pip install ./python
```

Importing `bloom_sdk` does not load the native library. The first
`BloomPipeline` construction does, so packaging tools, documentation builders,
and applications that configure the path after import can still import the
module safely.

## Python usage

```python
from bloom_sdk import BloomPipeline

with BloomPipeline("/path/to/model.gguf", context_size=2048) as pipeline:
    output = pipeline.generate("Explain edge inference in one sentence.")
    print(output)

    for chunk in pipeline.generate_stream("Give me two short examples."):
        print(chunk)
```

`model_path` also accepts `pathlib.Path` values. The wrapper serializes text
and JSON as UTF-8 without ASCII-only escaping, so Chinese, Japanese, Arabic,
and other Unicode prompts are passed to the native engine unchanged:

```python
from pathlib import Path
from bloom_sdk import BloomPipeline

with BloomPipeline(Path("模型/问答.gguf"), locale="zh-CN") as pipeline:
    print(pipeline.generate("请用中文回答：什么是端侧推理？"))
```

`locale` selects the language of SDK-generated diagnostic prefixes. It accepts
`en`, `zh-CN`/`zh`, and `zh-TW`; an unknown tag falls back to English. Set
`BLOOM_LOCALE` to choose a process-wide default. Native diagnostic details
remain intact for troubleshooting. Generation also accepts the native
response-format mapping (or the `"json_object"` shorthand) when structured
output is needed:

```python
pipeline.generate(
    "Return a JSON object",
    response_format={"type": "json_object"},
)
```

Both `generate` and `generate_stream` accept `top_k` (1–32,768),
`presence_penalty` and `frequency_penalty` (−2 through 2), and `logit_bias`
(up to 1,024 token IDs, each 0–1,000,000, with a bias of −100 through 100)
as keyword-only controls. Logit-bias keys may be integers or canonical decimal
strings; aliases for the same token are rejected. For example:

```python
pipeline.generate(
    "Suggest a title", top_k=40, presence_penalty=0.5,
    frequency_penalty=0.25, logit_bias={17: -5.0}, seed=42,
)
```

Controls are validated before entering the native call. `max_tokens` must be
positive, `temperature` must be finite and between 0 and 2, and `top_p` must be
greater than 0 and at most 1. Unknown response-format or schema-wrapper fields
are rejected instead of silently discarded. Existing positional arguments,
including streaming `timeout`, retain their order.

Both generation methods also accept keyword-only `logprobs=None` or an integer
from 0 through 20. Zero requests only the selected token's log probability;
larger values add that many top alternatives. These requests require
`max_tokens <= 4096`. The native backend rejects unsupported models/devices,
structured-output combinations, and other unsupported decoding modes instead
of silently omitting metadata. `None` omits the new JSON field, so existing
calls remain compatible with older native libraries.

Buffered `generate` results from models with exact native accounting include
a `generation` dictionary with `input_tokens`, `output_tokens` (including an
accepted EOS token) and `reason` (`"eos"`, `"length"` or `"stop"`), so callers
can tell a natural stop from an exhausted `max_tokens` budget. The key is
absent for legacy adapters and older native libraries; malformed metadata
raises `BloomInferenceError`.

```python
result = pipeline.generate("Continue this sentence", max_tokens=32)
if result.get("generation", {}).get("reason") == "length":
    print("truncated after", result["generation"]["output_tokens"], "tokens")
```

`device` defaults to `"cpu"`. Pass `device="auto"` to use the same selection
as the CLI and server: the first available GPU, NPU or CPU declared by the
selected engine.

```python
result = pipeline.generate("Continue this sentence", max_tokens=32, logprobs=3)
for record in result["logprobs"]:
    print(record["token"]["token_id"], record["token"]["logprob"])

for chunk in pipeline.generate_stream("Continue this sentence", logprobs=0):
    if isinstance(chunk, dict) and "TokenLogprobs" in chunk:
        print(chunk["TokenLogprobs"])
```

Buffered `logprobs` arrays and streamed `TokenLogprobs` events preserve the
native record: `token` and `top_logprobs` each contain `token_id`, display
`token`, exact `bytes`, and `logprob`. Token bytes can contain partial UTF-8;
join the selected tokens' byte arrays before decoding complete text. Values
describe adjusted model logits before temperature/top-k/top-p sampling.

Calls on one Python pipeline are serialized; consume or close a stream before
starting another inference on that pipeline. Stream handoff is bounded to 64
serialized chunks and 16 MiB of queued bytes. A slow consumer backpressures the
native callback; an individual chunk above 16 MiB raises `BloomInferenceError`
(revision 2 checks its length before copying native memory). These limits bound
the handoff queue, not model memory or objects retained by the application.
Data-carrying stream variants are dictionaries such as `{"TextDelta": "hello"}`;
the native terminal unit variant is the string `"End"` and is yielded unchanged.

`close()` is idempotent. It stops registered streams, releases blocked
callbacks, and waits for active native calls before freeing the handle,
preventing a concurrent stream from using freed memory. With an ABI revision 2 native library, closing
the generator cooperatively cancels native decoding at the next output
boundary. Explicitly close a partially consumed stream instead of retaining it:

```python
stream = pipeline.generate_stream("Write a long answer.")
try:
    print(next(stream))
finally:
    stream.close()
```

The wrapper detects `bloom_abi_version()` at runtime. Revision 2 is preferred;
libraries that expose only the original symbols continue to work, without
stream cancellation or length-delimited buffers.

Revision 1 callbacks stop queueing output after close, but their native call
must still finish before the pipeline can be freed. For either revision, a
backend stuck in prefill or external I/O may delay `pipeline.close()`; the SDK
does not forcibly terminate threads or release a live native handle.

## Exact tokenization

Python callers can use the tokenizer owned by the loaded model without
running generation:

```python
tokens = pipeline.tokenize("你好, Bloom 🌸")
text = pipeline.detokenize(tokens)
```

`tokenize` accepts UTF-8 text up to 8 MiB and returns a list of unsigned 32-bit
token IDs. `detokenize` accepts a list or tuple of unsigned 32-bit token IDs,
bounded to 4M tokens and 16 MiB of serialized JSON, and returns text. Empty
inputs are allowed. The model adapter controls special-token insertion and
removal, so special-token spellings need not survive decoding. Both methods
use exact model tokenization; a model without a tokenizer raises
`BloomInferenceError` instead of fabricating IDs or estimating token counts.

These methods use optional ABI revision 2 extension symbols:

- `bloom_pipeline_tokenize_v2` accepts a raw UTF-8 `BloomSlice` and returns a
  JSON array of token IDs in `BloomOwnedBuffer`.
- `bloom_pipeline_detokenize_v2` accepts a JSON array of token IDs and returns
  a JSON string in `BloomOwnedBuffer`.

Both functions cap serialized output at 16 MiB, reset output on failure, and
return `BLOOM_STATUS_UNSUPPORTED` (`-9`) when the model has no exact tokenizer.
Initialize output to `{NULL, 0}` and release successful output with
`bloom_buffer_free`. Invalid input uses the existing argument/UTF-8/JSON status
codes; tokenizer execution errors use `BLOOM_STATUS_INFERENCE_ERROR` (`-5`),
and caught panics use `BLOOM_STATUS_PANIC` (`-7`).

Discover these symbols before calling an older revision 2 library: the ABI
version remains 2, and older libraries may lack these additive extensions.
The Python wrapper performs this check and reports an explicit upgrade error
when needed; generation continues to work with older revision 1/2 libraries.
Tokenization shares the pipeline call lock with generation and close, and
releases native output buffers even when decoding or validation fails.

The wrapper also checks engine/device identifiers against the native 4 KiB
UTF-8 limit and caps each generation input/parameter JSON document at 16 MiB
before invoking either ABI revision.

## Python distribution gate

The SDK has both the original source/pure-Python distribution and an optional
platform-native wheel. The pure distribution requires a separate library;
`build_python_native_wheel.py` builds from that source distribution and embeds
a validated ABI-2 library into a non-pure `py3-none-<platform>` wheel. The
wrapper uses `ctypes`, so it has no CPython extension ABI dependency. The tags
follow the [Python packaging compatibility specification](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/).

Native wheel metadata records the library SHA-256 and host platform; RECORD
covers every member. A policy-checked CycloneDX inventory records the locked
build-workspace dependency graph (which can include unlinked packages), with
its own hash and the Laya fork notice. Install validation checks inventory/hashes and exercises
buffered/streamed inference in an offline, fresh environment with no library
override. Packaged libraries take precedence over workspace/system discovery,
and a load failure does not silently select another runtime. An explicit
`BLOOM_FFI_LIB` override retains priority.

```sh
cargo build --release -p bloomai-ffi --locked
python scripts/build_python_native_wheel.py --native-library target/release/libbloom_ffi.dylib
python scripts/test_python_native_wheel.py release-artifacts/python/*.whl
```

Select the platform-native `.so`/`.dll` name on Linux/Windows. Wheels currently
use conservative host tags. Linux `linux_*` artifacts require compatible host
system libraries; manylinux repair is not claimed. CI installs each OS artifact
under Python 3.9/3.12/3.14. Release jobs independently build release-mode wheels
and attest them, selecting only `release-bloom-*` artifacts to avoid mixing
in CI/debug wheels with identical filenames. Local macOS installation passed;
hosted matrix execution and publication remain acceptance evidence, not local
claims. ABI-1/older ABI-2 compatibility remains covered through explicitly
supplied external libraries; bundled wheels require ABI 2.

The packaging gate builds an sdist, rebuilds the wheel from it, checks package
metadata and license contents, installs into a clean temporary environment,
imports with no native library, then exercises buffered and streamed inference
through the installed wrapper and the real ABI v2 mock engine. The native
contract gate also validates the exported revision-1 and revision-2 symbols,
the revision-2 cancellation lifecycle, and bounded invalid-input paths:

```bash
python3 -m pip install -r requirements/python-package.txt
cargo build -p bloomai-ffi --locked
python3 scripts/check_ffi_abi_compatibility.py \
  --library target/debug/libbloom_ffi.so
python3 scripts/test_python_package.py
```

`test_python_package.py` invokes the same native contract check for every
library passed through `--native-library` or `--compatibility-library` before
building the temporary wheel environment.

To exercise a previously released ABI-1 library through the wheel as well,
pass it as an additional compatibility input. Each library runs in a fresh
process so the wrapper cannot retain symbols or version state from the prior
load:

```bash
python3 scripts/test_python_package.py \
  --native-library target/debug/libbloom_ffi.so \
  --compatibility-library /path/to/legacy/libbloom_ffi.so
```

Pass `--native-library /absolute/path/to/library` to validate another build.
This proves package isolation and native interoperation, not trained-model
quality or compatibility with every supported Python/platform combination.

## ABI revision 2

New C consumers should use the `_v2` entry points in `crates/ffi/bloom.h`:

- `BloomSlice` carries an explicit pointer and byte length for every UTF-8
  input. Identifiers are bounded to 4 KiB and JSON inputs to 16 MiB before the
  memory is read.
- `bloom_pipeline_run_v2` returns a `BloomOwnedBuffer`, so output can be copied
  by its exact length and released with `bloom_buffer_free` without scanning
  for a terminator.
- `bloom_pipeline_run_stream_v2` invokes a length-aware callback and accepts an
  optional `BloomCancellationToken`. Calling
  `bloom_cancellation_token_cancel` from another thread is safe; the stream
  returns `BLOOM_STATUS_CANCELLED` (`-8`) at the next output boundary.
- `bloom_abi_version()` returns the newest implemented revision. Revision 1
  symbols remain exported for source and binary migration.

Cancellation is cooperative rather than forced thread termination. A backend
that is computing a long prefill or blocked in an external runtime may not
observe the token until it reaches its next output-sink boundary.

## C ownership and safety contract

Include `crates/ffi/bloom.h` and link the generated shared library. Revision 2
callers must observe these rules:

- Every `BloomSlice` points to exactly `len` readable bytes for the complete
  call. Input is UTF-8; JSON slices must contain the documented object shapes.
- Device names accept `auto`, `cpu`, `gpu`, `cuda`, `metal`, `npu`, and
  `intel-npu` (case-insensitive). The accelerator aliases map to Bloom's
  generic device class and are shared with the HTTP server's CLI
  configuration. `auto` is resolved at load time exactly like the CLI and
  server default: the first available of GPU, NPU and CPU that the selected
  engine declares in its supported devices. Explicit names are passed through
  unchanged.
- A non-NULL error buffer points to `error_buffer_len` writable bytes.
- A pipeline handle is freed exactly once and is not used concurrently with
  `bloom_pipeline_free`.
- `bloom_pipeline_run_v2` output is owned by Bloom and is released exactly once
  with `bloom_buffer_free`; the same rule applies to both tokenization
  extensions. The free call also clears the caller's structure.
- A cancellation token remains live until its stream returns. It may be marked
  concurrently, but it is freed exactly once and never concurrently with the
  stream.
- A stream callback is non-NULL. Its byte slice is borrowed only for that
  callback invocation; copy data that must outlive the callback.
- Callbacks must not throw or unwind across the C boundary.

Revision 2 reports stable `BLOOM_STATUS_*` codes: `0` for success, `-1` through
`-6` for argument, decoding, inference, or output failures, `-7` for a caught
internal panic, `-8` for cancellation, and `-9` for an unsupported exact
tokenizer. Invalid non-NULL pointers,
double-free, callback unwinding, and freeing a live handle or token during a
native call remain caller contract violations that no C ABI can validate
safely.

Buffered results (`bloom_pipeline_run_v2` and `bloom_pipeline_run`) are a
JSON `ModelOutput` object. When the loaded model reports exact native
generation accounting (currently the Candle text adapters), Bloom runs the
validated stream internally and adds a `generation` object, matching the
CLI's buffered `--json` output:

```json
{"text": "...", "generation": {"input_tokens": 12, "output_tokens": 32, "reason": "length"}}
```

`reason` is `eos`, `length` or `stop`; `output_tokens` counts accepted token
IDs including EOS. The key is omitted for legacy adapters. It is additive, so
the ABI revision is unchanged and consumers that ignore unknown keys keep
working.

Revision 1 consumers use NUL-terminated inputs, release
`bloom_pipeline_run` output with `bloom_string_free`, and cannot cancel native
streaming. Its historical stream error-code mapping is unchanged.

Both ABI revisions accept the same generation-parameter JSON fields:
`max_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty`,
`frequency_penalty`, `logit_bias`, `seed`, `response_format`, and optional
`logprobs` (null or 0–20). This is an additive JSON capability; the ABI version
remains unchanged. Older libraries reject an explicitly requested `logprobs`
field, so omit it when targeting a library without this capability.
`max_tokens`, `temperature`, and `top_p` are required; other controls default
to their native neutral values. Unknown or duplicate generation fields and
unknown response-format fields fail during parameter decoding. Revision 2
returns `BLOOM_STATUS_INVALID_PARAMS_JSON`; revision 1 preserves its NULL
buffered result / `-5` stream result with a diagnostic. This strict JSON
boundary prevents a misspelled control from producing apparently successful
inference with unintended defaults.

Run the model-free boundary gates with:

```bash
cargo test -p bloomai-ffi --locked
python3 -m unittest discover -s python/tests -v

# Also enable the Python-to-native integration case after cargo build.
BLOOM_TEST_NATIVE_FFI=1 python3 -m unittest discover -s python/tests -v
```
