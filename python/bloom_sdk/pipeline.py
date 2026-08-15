import ctypes
import json
import math
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union


class BloomError(Exception):
    """Base exception for all Bloom SDK errors."""
    pass

class BloomLoadError(BloomError):
    """Raised when the model pipeline fails to load."""
    pass

class BloomInferenceError(BloomError):
    """Raised when inference fails."""
    pass


# Locate the compiled FFI library
def _find_lib():
    env_path = os.environ.get("BLOOM_FFI_LIB")
    if env_path:
        return ctypes.CDLL(env_path)
    
    # Search target directories relative to workspace
    this_dir = Path(__file__).parent.resolve()
    workspace_dir = this_dir.parents[1]
    
    lib_names = []
    if sys.platform == "win32":
        lib_names = ["bloom_ffi.dll", "libbloom_ffi.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libbloom_ffi.dylib"]
    else:
        lib_names = ["libbloom_ffi.so"]
        
    search_dirs = [
        workspace_dir / "target" / "release",
        workspace_dir / "target" / "debug",
        workspace_dir / "crates" / "ffi",
        this_dir,
    ]
    
    for search_dir in search_dirs:
        for name in lib_names:
            lib_path = search_dir / name
            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path))
                except Exception:
                    pass
                    
    # Try system loader fallback
    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except Exception:
            pass
            
    raise RuntimeError(
        "Could not locate Bloom FFI shared library. "
        "Please compile crates/ffi with 'cargo build --release' or set "
        "BLOOM_FFI_LIB to the shared library path."
    )

# Define struct and callback type mappings
class BloomPipelineOpaque(ctypes.Structure):
    pass

BloomPipelinePtr = ctypes.POINTER(BloomPipelineOpaque)

BloomStreamCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_char_p
)


def _configure_lib(native_lib):
    native_lib.bloom_pipeline_load.argtypes = [
        ctypes.c_char_p,  # model_path
        ctypes.c_char_p,  # engine_name
        ctypes.c_char_p,  # device_name
        ctypes.c_size_t,  # context_size
        ctypes.c_char_p,  # error_buffer
        ctypes.c_size_t,  # error_buffer_len
    ]
    native_lib.bloom_pipeline_load.restype = BloomPipelinePtr

    native_lib.bloom_pipeline_free.argtypes = [BloomPipelinePtr]
    native_lib.bloom_pipeline_free.restype = None

    native_lib.bloom_pipeline_run.argtypes = [
        BloomPipelinePtr,
        ctypes.c_char_p,  # input_json
        ctypes.c_char_p,  # params_json
        ctypes.c_char_p,  # error_buffer
        ctypes.c_size_t,  # error_buffer_len
    ]
    native_lib.bloom_pipeline_run.restype = ctypes.c_void_p

    native_lib.bloom_pipeline_run_stream.argtypes = [
        BloomPipelinePtr,
        ctypes.c_char_p,  # input_json
        ctypes.c_char_p,  # params_json
        BloomStreamCallback,
        ctypes.c_void_p,  # user_data
        ctypes.c_char_p,  # error_buffer
        ctypes.c_size_t,  # error_buffer_len
    ]
    native_lib.bloom_pipeline_run_stream.restype = ctypes.c_int32

    native_lib.bloom_string_free.argtypes = [ctypes.c_void_p]
    native_lib.bloom_string_free.restype = None
    return native_lib


_lib = None
_lib_lock = threading.Lock()


def _get_lib():
    """Load the native library on first pipeline construction, not package import."""
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                _lib = _configure_lib(_find_lib())
    return _lib


def _decode_error(error_buffer) -> str:
    return error_buffer.value.decode("utf-8", errors="replace")


class BloomPipeline:
    """
    High-level Python wrapper for the Bloom inference engine.
    
    Supports non-streaming and streaming generation.
    """
    def __init__(
        self,
        model_path: str,
        engine: str = "candle",
        device: str = "cpu",
        context_size: int = 2048,
    ):
        self._call_lock = threading.RLock()
        self._pipeline = None
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("model_path must be a non-empty string")
        if not isinstance(engine, str) or not engine:
            raise ValueError("engine must be a non-empty string")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        for field_name, field_value in (
            ("model_path", model_path),
            ("engine", engine),
            ("device", device),
        ):
            if "\0" in field_value:
                raise ValueError(f"{field_name} must not contain NUL characters")
        if (
            not isinstance(context_size, int)
            or isinstance(context_size, bool)
            or context_size <= 0
            or context_size > ctypes.c_size_t(-1).value
        ):
            raise ValueError("context_size must be a positive platform-sized integer")

        try:
            self._lib = _get_lib()
        except (OSError, RuntimeError) as error:
            raise BloomLoadError(f"Could not load the Bloom native library: {error}") from error
        err_buf = ctypes.create_string_buffer(512)
        try:
            self._pipeline = self._lib.bloom_pipeline_load(
                model_path.encode("utf-8"),
                engine.encode("utf-8"),
                device.encode("utf-8"),
                context_size,
                err_buf,
                len(err_buf)
            )
        except Exception as error:
            raise BloomLoadError(f"Native pipeline loading failed: {error}") from error
        if not self._pipeline:
            raise BloomLoadError(
                f"Failed to load model pipeline: {_decode_error(err_buf)}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Free the pipeline resource."""
        call_lock = getattr(self, "_call_lock", None)
        if call_lock is None:
            return
        with call_lock:
            if self._pipeline:
                self._lib.bloom_pipeline_free(self._pipeline)
                self._pipeline = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not surface native-loader shutdown errors.
            pass

    def _prepare_input_params(
        self,
        prompt_or_input: Union[str, dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int],
    ):
        if isinstance(prompt_or_input, str):
            input_data = {"Text": {"prompt": prompt_or_input}}
        elif isinstance(prompt_or_input, dict):
            input_data = prompt_or_input
        else:
            raise TypeError("prompt_or_input must be a string or dictionary")

        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if (
            not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not math.isfinite(top_p)
            or top_p <= 0
            or top_p > 1
        ):
            raise ValueError("top_p must be greater than zero and at most one")
        if (
            seed is not None
            and (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or seed < 0
                or seed > (1 << 64) - 1
            )
        ):
            raise ValueError("seed must be None or an unsigned 64-bit integer")

        params_data = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed
        }
        return (
            json.dumps(input_data, allow_nan=False).encode("utf-8"),
            json.dumps(params_data, allow_nan=False).encode("utf-8"),
        )

    def generate(
        self,
        prompt_or_input: Union[str, dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run full non-streaming inference.
        
        :param prompt_or_input: Prompt string or dict representation of ModelInput.
        :return: Decoded ModelOutput dict.
        """
        input_bytes, params_bytes = self._prepare_input_params(prompt_or_input, max_tokens, temperature, top_p, seed)
        err_buf = ctypes.create_string_buffer(512)

        with self._call_lock:
            if not self._pipeline:
                raise BloomError("Pipeline is closed")
            try:
                res_ptr = self._lib.bloom_pipeline_run(
                    self._pipeline,
                    input_bytes,
                    params_bytes,
                    err_buf,
                    len(err_buf)
                )
            except Exception as error:
                raise BloomInferenceError(
                    f"Native inference call failed: {error}"
                ) from error
            if not res_ptr:
                raise BloomInferenceError(
                    f"Inference failed: {_decode_error(err_buf)}"
                )
            try:
                result_bytes = ctypes.cast(res_ptr, ctypes.c_char_p).value
                if result_bytes is None:
                    raise BloomInferenceError("Inference returned a NULL string")
                result_text = result_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise BloomInferenceError(
                    "Inference returned non-UTF-8 output"
                ) from error
            finally:
                self._lib.bloom_string_free(res_ptr)

        try:
            return json.loads(result_text)
        except json.JSONDecodeError as error:
            raise BloomInferenceError("Inference returned invalid JSON") from error

    def generate_stream(
        self,
        prompt_or_input: Union[str, dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run streaming inference.
        
        Yields parsed OutputChunk dicts progressively as they arrive from the engine.
        """
        input_bytes, params_bytes = self._prepare_input_params(prompt_or_input, max_tokens, temperature, top_p, seed)
        q = queue.Queue()

        # Define local callback
        @BloomStreamCallback
        def py_callback(user_data, chunk_json):
            try:
                if chunk_json is None:
                    raise BloomInferenceError("Streaming callback returned NULL data")
                chunk_str = chunk_json.decode("utf-8")
                q.put(("chunk", json.loads(chunk_str)))
            except Exception as e:
                q.put(("error", BloomInferenceError(f"Invalid streaming chunk: {e}")))

        err_buf = ctypes.create_string_buffer(512)
        
        # Execute streaming FFI in a background thread to allow yielding on main thread
        def run_thread():
            try:
                # Serialize calls on one native pipeline. This also prevents
                # close() from freeing the handle while inference is active.
                with self._call_lock:
                    if not self._pipeline:
                        q.put(("error", BloomError("Pipeline is closed")))
                        return
                    res = self._lib.bloom_pipeline_run_stream(
                        self._pipeline,
                        input_bytes,
                        params_bytes,
                        py_callback,
                        None,
                        err_buf,
                        len(err_buf)
                    )
                if res != 0:
                    q.put(("error", BloomInferenceError(
                        f"Streaming failed (code {res}): {_decode_error(err_buf)}"
                    )))
                else:
                    q.put(("done", None))
            except Exception as error:
                q.put(("error", BloomInferenceError(
                    f"Streaming native call failed: {error}"
                )))

        thread = threading.Thread(target=run_thread)
        thread.start()

        try:
            while True:
                status, value = q.get()
                if status == "chunk":
                    yield value
                elif status == "error":
                    raise value
                elif status == "done":
                    thread.join()
                    break
        finally:
            # The worker retains self and the callback until the native call
            # ends. Avoid blocking generator finalization when a consumer
            # intentionally stops reading an uncancellable stream.
            if not thread.is_alive():
                thread.join()
