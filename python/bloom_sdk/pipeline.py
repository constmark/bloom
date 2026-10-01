import ctypes
import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Mapping, Optional, Union

from ._stream import MAX_STREAM_BYTES, StreamBuffer


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


class BloomSlice(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("len", ctypes.c_size_t),
    ]


class BloomOwnedBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("len", ctypes.c_size_t),
    ]

BloomStreamCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_char_p
)

BloomStreamCallbackV2 = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
)

BLOOM_STATUS_CANCELLED = -8

# Keep the Python boundary's diagnostic vocabulary stable while allowing
# applications to present messages in their own UI language.  Native errors
# are retained verbatim after the localized prefix so no diagnostic context is
# lost.  Locale selection is deliberately small and deterministic: callers
# can pass either a BCP-47 language tag or set BLOOM_LOCALE in the process
# environment.  Unknown tags fall back to English rather than failing a model
# load because of a presentation preference.
_LOCALE_ALIASES = {
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-sg": "zh-CN",
    "zh-tw": "zh-TW",
    "zh-hk": "zh-TW",
}
_LOCALIZED_LABELS = {
    "en": {
        "load_failed": "Native pipeline loading failed",
        "model_load_failed": "Failed to load model pipeline",
        "inference_failed": "Inference failed",
        "stream_failed": "Streaming failed",
        "stream_native_failed": "Streaming native call failed",
        "pipeline_closed": "Pipeline is closed",
        "invalid_json": "Inference returned invalid JSON",
        "invalid_chunk": "Invalid streaming chunk",
    },
    "zh-CN": {
        "load_failed": "原生推理管道加载失败",
        "model_load_failed": "模型推理管道加载失败",
        "inference_failed": "推理失败",
        "stream_failed": "流式推理失败",
        "stream_native_failed": "原生流式调用失败",
        "pipeline_closed": "推理管道已关闭",
        "invalid_json": "推理返回了无效 JSON",
        "invalid_chunk": "无效的流式数据块",
    },
    "zh-TW": {
        "load_failed": "原生推理管線載入失敗",
        "model_load_failed": "模型推理管線載入失敗",
        "inference_failed": "推理失敗",
        "stream_failed": "串流推理失敗",
        "stream_native_failed": "原生串流呼叫失敗",
        "pipeline_closed": "推理管線已關閉",
        "invalid_json": "推理回傳了無效 JSON",
        "invalid_chunk": "無效的串流資料塊",
    },
}


def _normalize_locale(locale: Optional[str]) -> str:
    """Return the SDK's display locale for a BCP-47-ish language tag."""
    selected = locale if locale is not None else os.environ.get("BLOOM_LOCALE", "en")
    if not isinstance(selected, str):
        raise TypeError("locale must be a string or None")
    normalized = selected.strip().replace("_", "-").lower()
    if not normalized:
        return "en"
    return _LOCALE_ALIASES.get(normalized, "en")


def _label(locale: str, key: str) -> str:
    labels = _LOCALIZED_LABELS.get(locale, _LOCALIZED_LABELS["en"])
    return labels.get(key, _LOCALIZED_LABELS["en"].get(key, key))


def _normalize_response_format(
    response_format: Optional[Union[str, Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Convert OpenAI-style response formats to the native enum encoding.

    The C ABI deserializes ``GenerationParams`` directly. Its tagged enum uses
    ``{"type": "json_object"}`` and
    ``{"type": "json_schema", "json_schema": schema}``, while SDK callers
    commonly use the same OpenAI-style mapping. Accept strings as a convenient
    shorthand and normalize both forms to the exact Rust representation.
    """
    if response_format is None:
        return None
    if isinstance(response_format, str):
        if response_format in {"text", "json_object"}:
            return {"type": response_format}
        raise ValueError(
            "response_format string must be 'text' or 'json_object'"
        )
    if not isinstance(response_format, Mapping):
        raise TypeError("response_format must be a mapping, string, or None")

    format_type = response_format.get("type")
    if format_type is None and set(response_format) == {"json_schema"}:
        schema = response_format["json_schema"]
    elif format_type == "text":
        return {"type": "text"}
    elif format_type == "json_object":
        return {"type": "json_object"}
    elif format_type == "json_schema":
        schema = response_format.get("json_schema")
        if isinstance(schema, Mapping) and "schema" in schema:
            schema = schema["schema"]
    else:
        raise ValueError(
            "response_format.type must be 'text', 'json_object', or 'json_schema'"
        )
    if not isinstance(schema, Mapping):
        raise ValueError("response_format json_schema must contain a schema object")
    return {"type": "json_schema", "json_schema": dict(schema)}


def _bytes_slice(value: bytes):
    """Return a length-delimited ABI slice and an owner that keeps it alive."""
    owner = ctypes.create_string_buffer(value, max(1, len(value)))
    data = ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8))
    return BloomSlice(data, len(value)), owner


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

    v2_symbols = (
        "bloom_pipeline_load_v2",
        "bloom_pipeline_run_v2",
        "bloom_pipeline_run_stream_v2",
        "bloom_buffer_free",
        "bloom_cancellation_token_new",
        "bloom_cancellation_token_cancel",
        "bloom_cancellation_token_free",
    )
    has_version_symbol = hasattr(native_lib, "bloom_abi_version")
    has_v2 = False
    if has_version_symbol:
        native_lib.bloom_abi_version.argtypes = []
        native_lib.bloom_abi_version.restype = ctypes.c_uint32
        abi_version = native_lib.bloom_abi_version()
        if abi_version >= 2:
            missing = [
                symbol for symbol in v2_symbols if not hasattr(native_lib, symbol)
            ]
            if missing:
                raise RuntimeError(
                    "Bloom native library declares ABI revision "
                    f"{abi_version} but is missing required symbols: "
                    + ", ".join(missing)
                )
            has_v2 = True
    if has_v2:
        native_lib.bloom_pipeline_load_v2.argtypes = [
            BloomSlice,
            BloomSlice,
            BloomSlice,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        native_lib.bloom_pipeline_load_v2.restype = BloomPipelinePtr
        native_lib.bloom_pipeline_run_v2.argtypes = [
            BloomPipelinePtr,
            BloomSlice,
            BloomSlice,
            ctypes.POINTER(BloomOwnedBuffer),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        native_lib.bloom_pipeline_run_v2.restype = ctypes.c_int32
        native_lib.bloom_pipeline_run_stream_v2.argtypes = [
            BloomPipelinePtr,
            BloomSlice,
            BloomSlice,
            BloomStreamCallbackV2,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        native_lib.bloom_pipeline_run_stream_v2.restype = ctypes.c_int32
        native_lib.bloom_buffer_free.argtypes = [
            ctypes.POINTER(BloomOwnedBuffer)
        ]
        native_lib.bloom_buffer_free.restype = None
        native_lib.bloom_cancellation_token_new.argtypes = []
        native_lib.bloom_cancellation_token_new.restype = ctypes.c_void_p
        native_lib.bloom_cancellation_token_cancel.argtypes = [ctypes.c_void_p]
        native_lib.bloom_cancellation_token_cancel.restype = ctypes.c_int32
        native_lib.bloom_cancellation_token_free.argtypes = [ctypes.c_void_p]
        native_lib.bloom_cancellation_token_free.restype = None
    native_lib._bloom_uses_v2 = has_v2
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
        model_path: Union[str, os.PathLike],
        engine: str = "candle",
        device: str = "cpu",
        context_size: int = 2048,
        locale: Optional[str] = None,
    ):
        self.locale = _normalize_locale(locale)
        self._call_lock = threading.RLock()
        self._stream_lock = threading.Lock()
        self._stream_cancellations = set()
        self._closing = False
        self._pipeline = None
        # pathlib.Path and other os.PathLike implementations are common in
        # cross-platform applications.  Convert them once, while rejecting
        # arbitrary filesystem bytes that cannot be represented by the UTF-8
        # C ABI.  This keeps Unicode model paths lossless on all platforms.
        try:
            model_path = os.fspath(model_path)
        except TypeError as error:
            raise TypeError("model_path must be a string or os.PathLike") from error
        if isinstance(model_path, bytes):
            try:
                model_path = model_path.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("model_path bytes must be valid UTF-8") from error
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
            raise BloomLoadError(
                f"{_label(self.locale, 'load_failed')}: {error}"
            ) from error
        self._uses_v2 = bool(getattr(self._lib, "_bloom_uses_v2", False))
        err_buf = ctypes.create_string_buffer(512)
        try:
            if self._uses_v2:
                model_slice, model_owner = _bytes_slice(model_path.encode("utf-8"))
                engine_slice, engine_owner = _bytes_slice(engine.encode("utf-8"))
                device_slice, device_owner = _bytes_slice(device.encode("utf-8"))
                self._pipeline = self._lib.bloom_pipeline_load_v2(
                    model_slice,
                    engine_slice,
                    device_slice,
                    context_size,
                    err_buf,
                    len(err_buf),
                )
                # Keep the borrowed buffers alive through the native call.
                _ = (model_owner, engine_owner, device_owner)
            else:
                self._pipeline = self._lib.bloom_pipeline_load(
                    model_path.encode("utf-8"),
                    engine.encode("utf-8"),
                    device.encode("utf-8"),
                    context_size,
                    err_buf,
                    len(err_buf)
                )
        except Exception as error:
            raise BloomLoadError(
                f"{_label(self.locale, 'load_failed')}: {error}"
            ) from error
        if not self._pipeline:
            raise BloomLoadError(
                f"{_label(self.locale, 'model_load_failed')}: {_decode_error(err_buf)}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Cancel streams and wait for native calls before freeing the handle."""
        call_lock = getattr(self, "_call_lock", None)
        if call_lock is None:
            return
        with self._stream_lock:
            self._closing = True
            cancellations = tuple(self._stream_cancellations)
        # Never wait for native completion while a callback is blocked on a
        # consumer that has stopped reading (including context-manager exit).
        for cancel in cancellations:
            cancel()
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
        response_format: Optional[Union[str, Mapping[str, Any]]] = None,
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
        native_response_format = _normalize_response_format(response_format)

        params_data = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            # The native GenerationParams type supports text, json_object and
            # json_schema response formats.  Passing this through here keeps
            # the Python SDK feature-compatible with the HTTP adapters while
            # remaining backwards-compatible when omitted.
            "response_format": native_response_format,
        }
        try:
            # Avoid ASCII-only escaping so multilingual prompts and schema
            # descriptions remain inspectable in traces and examples.  The
            # native ABI consumes UTF-8 and receives the same JSON semantics.
            return (
                json.dumps(
                    input_data, ensure_ascii=False, allow_nan=False
                ).encode("utf-8"),
                json.dumps(
                    params_data, ensure_ascii=False, allow_nan=False
                ).encode("utf-8"),
            )
        except (TypeError, ValueError, UnicodeError) as error:
            raise ValueError(
                "input and generation parameters must be JSON serializable: "
                f"{error}"
            ) from error

    def generate(
        self,
        prompt_or_input: Union[str, dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        response_format: Optional[Union[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run full non-streaming inference.
        
        :param prompt_or_input: Prompt string or dict representation of ModelInput.
        :param response_format: Optional response format string or OpenAI-style
            mapping, for example ``"json_object"`` or
            ``{"type": "json_schema", "json_schema": {"schema": {...}}}``.
        :return: Decoded ModelOutput dict.
        """
        input_bytes, params_bytes = self._prepare_input_params(
            prompt_or_input, max_tokens, temperature, top_p, seed, response_format
        )
        err_buf = ctypes.create_string_buffer(512)

        with self._call_lock:
            if self._closing or not self._pipeline:
                raise BloomError(_label(self.locale, "pipeline_closed"))
            if self._uses_v2:
                input_slice, input_owner = _bytes_slice(input_bytes)
                params_slice, params_owner = _bytes_slice(params_bytes)
                output = BloomOwnedBuffer()
                try:
                    status = self._lib.bloom_pipeline_run_v2(
                        self._pipeline,
                        input_slice,
                        params_slice,
                        ctypes.byref(output),
                        err_buf,
                        len(err_buf),
                    )
                    _ = (input_owner, params_owner)
                except Exception as error:
                    raise BloomInferenceError(
                        f"Native inference call failed: {error}"
                    ) from error
                if status != 0:
                    raise BloomInferenceError(
                        f"{_label(self.locale, 'inference_failed')} "
                        f"(code {status}): {_decode_error(err_buf)}"
                    )
                try:
                    if not output.data and output.len:
                        raise BloomInferenceError(
                            "Inference returned an invalid output buffer"
                        )
                    result_bytes = ctypes.string_at(output.data, output.len)
                    result_text = result_bytes.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise BloomInferenceError(
                        "Inference returned non-UTF-8 output"
                    ) from error
                finally:
                    self._lib.bloom_buffer_free(ctypes.byref(output))
            else:
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
                        f"{_label(self.locale, 'inference_failed')}: {_decode_error(err_buf)}"
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
            result = json.loads(result_text)
        except json.JSONDecodeError as error:
            raise BloomInferenceError(_label(self.locale, "invalid_json")) from error
        if not isinstance(result, dict):
            raise BloomInferenceError(
                f"{_label(self.locale, 'invalid_json')}: expected a JSON object"
            )
        return result

    def generate_stream(
        self,
        prompt_or_input: Union[str, dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        response_format: Optional[Union[str, Mapping[str, Any]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run streaming inference.
        
        Yields parsed OutputChunks progressively, with bounded backpressure.
        """
        input_bytes, params_bytes = self._prepare_input_params(
            prompt_or_input, max_tokens, temperature, top_p, seed, response_format
        )
        buffer = StreamBuffer()
        err_buf = ctypes.create_string_buffer(512)
        token_lock = threading.Lock()
        token_holder = {"value": None}

        if self._uses_v2:
            token_holder["value"] = self._lib.bloom_cancellation_token_new()
            if not token_holder["value"]:
                raise BloomInferenceError("Could not allocate a native cancellation token")

        def cancel_v2_stream():
            if not self._uses_v2:
                return
            with token_lock:
                token = token_holder["value"]
                if token:
                    self._lib.bloom_cancellation_token_cancel(token)

        def free_v2_token():
            if not self._uses_v2:
                return
            with token_lock:
                token = token_holder["value"]
                token_holder["value"] = None
                if token:
                    self._lib.bloom_cancellation_token_free(token)

        def stop_stream():
            buffer.stop()
            cancel_v2_stream()

        def invalid_chunk(error):
            buffer.finish(BloomInferenceError(
                f"{_label(self.locale, 'invalid_chunk')}: {error}"
            ))
            cancel_v2_stream()

        @BloomStreamCallback
        def py_callback(_user_data, chunk_json):
            try:
                if buffer.stopped.is_set():
                    return
                if chunk_json is None:
                    raise BloomInferenceError("Streaming callback returned NULL data")
                buffer.put(chunk_json)
            except Exception as error:
                invalid_chunk(error)

        @BloomStreamCallbackV2
        def py_callback_v2(_user_data, chunk_json, chunk_json_len):
            try:
                if buffer.stopped.is_set():
                    return
                if not chunk_json and chunk_json_len:
                    raise BloomInferenceError("Streaming callback returned NULL data")
                if chunk_json_len > MAX_STREAM_BYTES:
                    raise BloomInferenceError("Streaming chunk exceeds the 16 MiB limit")
                chunk_bytes = ctypes.string_at(chunk_json, chunk_json_len)
                buffer.put(chunk_bytes)
            except Exception as error:
                invalid_chunk(error)
        
        # Execute streaming FFI in a background thread to allow yielding on main thread
        def run_thread():
            try:
                # Serialize calls on one native pipeline. This also prevents
                # close() from freeing the handle while inference is active.
                with self._call_lock:
                    if buffer.stopped.is_set():
                        return
                    if self._closing or not self._pipeline:
                        raise BloomError(_label(self.locale, "pipeline_closed"))
                    if self._uses_v2:
                        input_slice, input_owner = _bytes_slice(input_bytes)
                        params_slice, params_owner = _bytes_slice(params_bytes)
                        with token_lock:
                            token = token_holder["value"]
                        res = self._lib.bloom_pipeline_run_stream_v2(
                            self._pipeline,
                            input_slice,
                            params_slice,
                            py_callback_v2,
                            None,
                            token,
                            err_buf,
                            len(err_buf),
                        )
                        _ = (input_owner, params_owner)
                    else:
                        res = self._lib.bloom_pipeline_run_stream(
                            self._pipeline,
                            input_bytes,
                            params_bytes,
                            py_callback,
                            None,
                            err_buf,
                            len(err_buf)
                        )
                if res == BLOOM_STATUS_CANCELLED:
                    buffer.finish()
                elif res != 0:
                    buffer.finish(BloomInferenceError(
                        f"{_label(self.locale, 'stream_failed')} "
                        f"(code {res}): {_decode_error(err_buf)}"
                    ))
                else:
                    buffer.finish()
            except Exception as error:
                buffer.finish(BloomInferenceError(
                    f"{_label(self.locale, 'stream_native_failed')}: {error}"
                ))
            finally:
                free_v2_token()
                with self._stream_lock:
                    self._stream_cancellations.discard(stop_stream)

        with self._stream_lock:
            if self._closing:
                free_v2_token()
                raise BloomError(_label(self.locale, "pipeline_closed"))
            self._stream_cancellations.add(stop_stream)
        thread = threading.Thread(target=run_thread, name="bloom-stream")
        try:
            thread.start()
        except BaseException:
            with self._stream_lock:
                self._stream_cancellations.discard(stop_stream)
            free_v2_token()
            raise

        try:
            while True:
                chunk = buffer.receive()
                if chunk is None:
                    thread.join()
                    break
                try:
                    value = json.loads(chunk.decode("utf-8"))
                except (UnicodeDecodeError, ValueError, RecursionError) as error:
                    raise BloomInferenceError(
                        f"{_label(self.locale, 'invalid_chunk')}: {error}"
                    ) from error
                yield value
        finally:
            stop_stream()
            if self._uses_v2:
                # Closing or abandoning the generator now stops native decode
                # cooperatively instead of leaving an orphaned worker.
                thread.join(timeout=1)
            elif not thread.is_alive():
                # A revision 1 library cannot be cancelled. The worker retains
                # self and the callback until the native call ends.
                thread.join()
