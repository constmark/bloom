import ctypes
import os
import sys
import json
import threading
import queue
from pathlib import Path
from typing import Generator, Union, Dict, Any

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
        "Please compile crates/ffi with 'cargo build --release' or set BLOOM_FFI_LIB environment variable."
    )

lib = _find_lib()

# Define struct and callback type mappings
class BloomPipelineOpaque(ctypes.Structure):
    pass

BloomPipelinePtr = ctypes.POINTER(BloomPipelineOpaque)

BloomStreamCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_char_p
)

lib.bloom_pipeline_load.argtypes = [
    ctypes.c_char_p, # model_path
    ctypes.c_char_p, # engine_name
    ctypes.c_char_p, # device_name
    ctypes.c_size_t, # context_size
    ctypes.c_char_p, # error_buffer
    ctypes.c_size_t  # error_buffer_len
]
lib.bloom_pipeline_load.restype = BloomPipelinePtr

lib.bloom_pipeline_free.argtypes = [BloomPipelinePtr]
lib.bloom_pipeline_free.restype = None

lib.bloom_pipeline_run.argtypes = [
    BloomPipelinePtr,
    ctypes.c_char_p, # input_json
    ctypes.c_char_p, # params_json
    ctypes.c_char_p, # error_buffer
    ctypes.c_size_t  # error_buffer_len
]
lib.bloom_pipeline_run.restype = ctypes.c_void_p

lib.bloom_pipeline_run_stream.argtypes = [
    BloomPipelinePtr,
    ctypes.c_char_p, # input_json
    ctypes.c_char_p, # params_json
    BloomStreamCallback,
    ctypes.c_void_p, # user_data
    ctypes.c_char_p, # error_buffer
    ctypes.c_size_t  # error_buffer_len
]
lib.bloom_pipeline_run_stream.restype = ctypes.c_int32

lib.bloom_string_free.argtypes = [ctypes.c_void_p]
lib.bloom_string_free.restype = None


class BloomPipeline:
    """
    High-level Python wrapper for the Bloom inference engine.
    
    Supports non-streaming and streaming generation.
    """
    def __init__(self, model_path: str, engine: str = "candle", device: str = "cpu", context_size: int = 2048):
        self._lib = lib
        self._pipeline = None
        
        err_buf = ctypes.create_string_buffer(512)
        self._pipeline = self._lib.bloom_pipeline_load(
            model_path.encode("utf-8"),
            engine.encode("utf-8"),
            device.encode("utf-8"),
            context_size,
            err_buf,
            len(err_buf)
        )
        if not self._pipeline:
            raise BloomLoadError(f"Failed to load model pipeline: {err_buf.value.decode('utf-8')}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Free the pipeline resource."""
        if self._pipeline:
            self._lib.bloom_pipeline_free(self._pipeline)
            self._pipeline = None

    def __del__(self):
        self.close()

    def _prepare_input_params(self, prompt_or_input: Union[str, dict], max_tokens: int, temperature: float, top_p: float, seed: int):
        if isinstance(prompt_or_input, str):
            input_data = {"Text": {"prompt": prompt_or_input}}
        else:
            input_data = prompt_or_input

        params_data = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed
        }
        return json.dumps(input_data).encode("utf-8"), json.dumps(params_data).encode("utf-8")

    def generate(
        self,
        prompt_or_input: Union[str, dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = None
    ) -> Dict[str, Any]:
        """
        Run full non-streaming inference.
        
        :param prompt_or_input: Prompt string or dict representation of ModelInput.
        :return: Decoded ModelOutput dict.
        """
        if not self._pipeline:
            raise BloomError("Pipeline is closed")
            
        input_bytes, params_bytes = self._prepare_input_params(prompt_or_input, max_tokens, temperature, top_p, seed)
        err_buf = ctypes.create_string_buffer(512)
        
        res_ptr = self._lib.bloom_pipeline_run(
            self._pipeline,
            input_bytes,
            params_bytes,
            err_buf,
            len(err_buf)
        )
        if not res_ptr:
            raise BloomInferenceError(f"Inference failed: {err_buf.value.decode('utf-8')}")
            
        res_str = ctypes.c_char_p(res_ptr).value.decode("utf-8")
        self._lib.bloom_string_free(res_ptr)
        return json.loads(res_str)

    def generate_stream(
        self,
        prompt_or_input: Union[str, dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run streaming inference.
        
        Yields parsed OutputChunk dicts progressively as they arrive from the engine.
        """
        if not self._pipeline:
            raise BloomError("Pipeline is closed")

        input_bytes, params_bytes = self._prepare_input_params(prompt_or_input, max_tokens, temperature, top_p, seed)
        q = queue.Queue()

        # Define local callback
        @BloomStreamCallback
        def py_callback(user_data, chunk_json):
            try:
                chunk_str = chunk_json.decode("utf-8")
                q.put(("chunk", json.loads(chunk_str)))
            except Exception as e:
                q.put(("error", e))

        err_buf = ctypes.create_string_buffer(512)
        
        # Execute streaming FFI in a background thread to allow yielding on main thread
        def run_thread():
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
                q.put(("error", BloomInferenceError(f"Streaming failed (code {res}): {err_buf.value.decode('utf-8')}")))
            else:
                q.put(("done", None))

        thread = threading.Thread(target=run_thread)
        # Keep py_callback alive while thread runs
        thread._callback_ref = py_callback
        thread.start()

        while True:
            status, value = q.get()
            if status == "chunk":
                yield value
            elif status == "error":
                raise value
            elif status == "done":
                break
