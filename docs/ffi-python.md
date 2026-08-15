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
stream from using freed memory. The current C ABI has no cancellation function;
stopping iteration does not interrupt an already-running native stream.

## C ownership and safety contract

Include `crates/ffi/bloom.h` and link the generated shared library. Callers must
observe these rules:

- Input strings are valid, NUL-terminated C strings for the duration of a call.
- A non-NULL error buffer points to `error_buffer_len` writable bytes.
- A pipeline handle is freed exactly once and is not used concurrently with
  `bloom_pipeline_free`.
- `bloom_pipeline_run` output is owned by Bloom and is released exactly once
  with `bloom_string_free`.
- A stream callback is non-NULL. Its `chunk_json` pointer is borrowed only for
  that callback invocation; copy data that must outlive the callback.
- Callbacks must not throw or unwind across the C boundary.

Bloom validates NULL public inputs, reports a NULL stream callback as error
`-1`, and catches internal Rust panics at the load, run, and stream boundaries.
A caught streaming panic is reported as `-7`. Invalid non-NULL pointers,
double-free, callback unwinding, and freeing a handle during a native call
remain caller contract violations that no C ABI can validate safely.

Run the model-free boundary gates with:

```bash
cargo test -p bloomai-ffi --locked
python3 -m unittest discover -s python/tests -v

# Also enable the Python-to-native integration case after cargo build.
BLOOM_TEST_NATIVE_FFI=1 python3 -m unittest discover -s python/tests -v
```
