# Bloom Python SDK

A pre-1.0 Python wrapper for Bloom's native local inference engine. Python 3.8
or newer is required. Importing the package does not load a model or library.

## Install

```sh
python -m pip install ./python
cargo build --release -p bloomai-ffi
export BLOOM_FFI_LIB=/absolute/path/to/libbloom_ffi.so
```

Use `libbloom_ffi.dylib` on macOS or `bloom_ffi.dll` on Windows. The source
distribution and Python wheel contain the wrapper only; build or supply the
native library separately. `BLOOM_FFI_LIB` selects it explicitly. Source
checkouts also search the workspace's `target/release` and `target/debug`.

## Generate

```python
from bloom_sdk import BloomPipeline

with BloomPipeline("/path/to/model.gguf", context_size=2048) as pipeline:
    print(pipeline.generate("Explain edge inference in one sentence."))
    stream = pipeline.generate_stream("Give two short examples.")
    try:
        for chunk in stream:
            print(chunk)
    finally:
        stream.close()
```

Streaming buffers at most 64 serialized chunks and 16 MiB of queued bytes.
Slow consumers apply backpressure to native callbacks. Individual chunks over
16 MiB fail with `BloomInferenceError`. Close partially consumed generators to
release their workers; leaving the pipeline context also stops its streams and
waits for active native calls before freeing the handle.

The wrapper negotiates ABI revision 2 and supports revision 1 fallback.
Revision 2 cancellation is cooperative at output boundaries. Revision 1 has
no native cancellation and must finish its native call before pipeline close
can return. A long prefill or blocked external runtime can delay either close.

See the [full SDK contract](https://github.com/constmark/bloom/blob/main/docs/ffi-python.md)
and [model support matrix](https://github.com/constmark/bloom/blob/main/docs/support-matrix.md).
