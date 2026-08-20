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

Calls on one Python pipeline are serialized. `close()` is idempotent and waits
for an active native call before freeing the handle, preventing a concurrent
stream from using freed memory. With an ABI revision 2 native library, closing
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
- A non-NULL error buffer points to `error_buffer_len` writable bytes.
- A pipeline handle is freed exactly once and is not used concurrently with
  `bloom_pipeline_free`.
- `bloom_pipeline_run_v2` output is owned by Bloom and is released exactly once
  with `bloom_buffer_free`; the free call also clears the caller's structure.
- A cancellation token remains live until its stream returns. It may be marked
  concurrently, but it is freed exactly once and never concurrently with the
  stream.
- A stream callback is non-NULL. Its byte slice is borrowed only for that
  callback invocation; copy data that must outlive the callback.
- Callbacks must not throw or unwind across the C boundary.

Revision 2 reports stable `BLOOM_STATUS_*` codes: `0` for success, `-1` through
`-6` for argument, decoding, inference, or output failures, `-7` for a caught
internal panic, and `-8` for cancellation. Invalid non-NULL pointers,
double-free, callback unwinding, and freeing a live handle or token during a
native call remain caller contract violations that no C ABI can validate
safely.

Revision 1 consumers use NUL-terminated inputs, release
`bloom_pipeline_run` output with `bloom_string_free`, and cannot cancel native
streaming. Its historical stream error-code mapping is unchanged.

Run the model-free boundary gates with:

```bash
cargo test -p bloomai-ffi --locked
python3 -m unittest discover -s python/tests -v

# Also enable the Python-to-native integration case after cargo build.
BLOOM_TEST_NATIVE_FFI=1 python3 -m unittest discover -s python/tests -v
```
