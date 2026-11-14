#!/usr/bin/env python3
"""Validate packaged Bloom native libraries against the public C ABI contract.

The Python wrapper loads ``bloomai-ffi`` at runtime, from a bundled native
wheel or an explicitly configured library.  This gate therefore checks every native library supplied to it
independently, including a previously released ABI-1 library when one is
available.  It validates the exported symbol set and exercises the safe error
and cancellation paths without requiring a model or accelerator.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


ABI1_SYMBOLS = (
    "bloom_pipeline_load",
    "bloom_pipeline_free",
    "bloom_pipeline_run",
    "bloom_pipeline_run_stream",
    "bloom_string_free",
)
ABI2_SYMBOLS = (
    "bloom_abi_version",
    "bloom_pipeline_load_v2",
    "bloom_pipeline_run_v2",
    "bloom_pipeline_run_stream_v2",
    "bloom_buffer_free",
    "bloom_cancellation_token_new",
    "bloom_cancellation_token_cancel",
    "bloom_cancellation_token_free",
)


class BloomSlice(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]


class BloomOwnedBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]


def _symbol(library: ctypes.CDLL, name: str):
    try:
        return getattr(library, name)
    except AttributeError:
        return None


def _abi_version(library: ctypes.CDLL) -> int:
    version = _symbol(library, "bloom_abi_version")
    if version is None:
        # ABI revision 1 predates the version query.  Its required symbol set
        # is still sufficient to identify the legacy contract.
        return 1
    version.argtypes = []
    version.restype = ctypes.c_uint32
    return int(version())


def _check_symbols(library: ctypes.CDLL, version: int) -> List[str]:
    required = list(ABI1_SYMBOLS)
    if version >= 2:
        required.extend(ABI2_SYMBOLS)
    return [name for name in required if _symbol(library, name) is None]


def _exercise_legacy_error_path(library: ctypes.CDLL) -> None:
    load = library.bloom_pipeline_load
    load.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    load.restype = ctypes.c_void_p
    error = ctypes.create_string_buffer(512)
    # A zero context is rejected before touching the model path.  This gives
    # us a meaningful ABI call while remaining independent of model fixtures.
    pipeline = load(b".", b"mock", b"cpu", 0, error, len(error))
    if pipeline:
        library.bloom_pipeline_free.argtypes = [ctypes.c_void_p]
        library.bloom_pipeline_free.restype = None
        library.bloom_pipeline_free(pipeline)
        raise RuntimeError("ABI-1 invalid context unexpectedly loaded a pipeline")


def _exercise_v2_error_and_lifecycle(library: ctypes.CDLL) -> None:
    load = library.bloom_pipeline_load_v2
    load.argtypes = [
        BloomSlice,
        BloomSlice,
        BloomSlice,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    load.restype = ctypes.c_void_p
    error = ctypes.create_string_buffer(512)

    def slice_for(value: bytes) -> Tuple[BloomSlice, ctypes.Array]:
        owner = ctypes.create_string_buffer(value)
        pointer = ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8))
        return BloomSlice(pointer, len(value)), owner

    model, model_owner = slice_for(b".")
    engine, engine_owner = slice_for(b"mock")
    device, device_owner = slice_for(b"cpu")
    # Keep the owners alive through the FFI call.
    _ = (model_owner, engine_owner, device_owner)
    pipeline = load(model, engine, device, 0, error, len(error))
    if pipeline:
        library.bloom_pipeline_free.argtypes = [ctypes.c_void_p]
        library.bloom_pipeline_free.restype = None
        library.bloom_pipeline_free(pipeline)
        raise RuntimeError("ABI-2 invalid context unexpectedly loaded a pipeline")

    token_new = library.bloom_cancellation_token_new
    token_new.argtypes = []
    token_new.restype = ctypes.c_void_p
    token_cancel = library.bloom_cancellation_token_cancel
    token_cancel.argtypes = [ctypes.c_void_p]
    token_cancel.restype = ctypes.c_int32
    token_free = library.bloom_cancellation_token_free
    token_free.argtypes = [ctypes.c_void_p]
    token_free.restype = None
    token = token_new()
    if not token:
        raise RuntimeError("ABI-2 cancellation-token allocation returned NULL")
    try:
        if token_cancel(token) != 0:
            raise RuntimeError("ABI-2 cancellation-token cancellation failed")
    finally:
        token_free(token)

    buffer_free = library.bloom_buffer_free
    buffer_free.argtypes = [ctypes.POINTER(BloomOwnedBuffer)]
    buffer_free.restype = None
    empty = BloomOwnedBuffer()
    buffer_free(ctypes.byref(empty))


def _exercise_inference(library: ctypes.CDLL, version: int) -> None:
    """Exercise argument layouts, owned results and callbacks of both ABIs.

    This optional current-runtime gate uses only the built-in mock engine. Keep
    it opt-in for historical libraries that do not promise that engine.
    """
    prompt = "ABI smoke: 你好 🌸\0"
    expected = "echo: " + prompt
    encoded_input = json.dumps({"Text": {"prompt": prompt}}, ensure_ascii=False).encode()
    encoded_params = json.dumps({
        "max_tokens": 4, "temperature": 0.0, "top_p": 1.0,
        "seed": None, "response_format": None,
    }).encode()
    error = ctypes.create_string_buffer(512)
    free = library.bloom_pipeline_free
    free.argtypes = [ctypes.c_void_p]
    free.restype = None

    def read_output(data: bytes) -> None:
        if json.loads(data).get("text") != expected:
            raise RuntimeError("ABI inference returned a mismatched text result")

    def check_stream(chunks: list) -> None:
        if not chunks or chunks[-1] != "End" or chunks.count("End") != 1:
            raise RuntimeError("ABI stream omitted its unique final End event")
        if "".join(chunk.get("TextDelta", "") for chunk in chunks if isinstance(chunk, dict)) != expected:
            raise RuntimeError("ABI stream returned mismatched UTF-8 text")

    pointer = library.bloom_pipeline_load(b".", b"mock", b"cpu", 64, error, len(error))
    if not pointer:
        raise RuntimeError(f"ABI-1 mock load failed: {error.value!r}")
    try:
        run = library.bloom_pipeline_run
        run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        run.restype = ctypes.c_void_p
        free_string = library.bloom_string_free
        free_string.argtypes = [ctypes.c_void_p]
        free_string.restype = None
        output = run(pointer, encoded_input, encoded_params, error, len(error))
        if not output:
            raise RuntimeError(f"ABI-1 inference failed: {error.value!r}")
        try:
            read_output(ctypes.string_at(output))
        finally:
            free_string(output)
        chunks, callback_errors = [], []
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p)

        @callback_type
        def callback(_context, chunk):
            try:
                if _context != 0xB100:
                    raise ValueError("callback context was not preserved")
                chunks.append(json.loads(chunk))
            except Exception as exc:
                callback_errors.append(str(exc))

        stream = library.bloom_pipeline_run_stream
        stream.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, callback_type, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        stream.restype = ctypes.c_int32
        if stream(pointer, encoded_input, encoded_params, callback, 0xB100, error, len(error)) != 0 or callback_errors:
            raise RuntimeError(f"ABI-1 stream failed: {error.value!r}, {callback_errors}")
        check_stream(chunks)
    finally:
        free(pointer)
    if version < 2:
        return

    # All owners remain alive until both length-delimited calls have returned.
    values = (b".", b"mock", b"cpu", encoded_input, encoded_params)
    # Deliberately put invalid UTF-8 outside every borrowed slice. A library
    # that accidentally reads a CString instead of respecting len must fail.
    owners = [ctypes.create_string_buffer(value + b"\xff") for value in values]
    slices = [BloomSlice(ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8)), len(value)) for owner, value in zip(owners, values)]
    model, engine, device, input_slice, params_slice = slices
    pointer = library.bloom_pipeline_load_v2(model, engine, device, 64, error, len(error))
    if not pointer:
        raise RuntimeError(f"ABI-2 mock load failed: {error.value!r}")
    try:
        run = library.bloom_pipeline_run_v2
        run.argtypes = [ctypes.c_void_p, BloomSlice, BloomSlice, ctypes.POINTER(BloomOwnedBuffer), ctypes.c_char_p, ctypes.c_size_t]
        run.restype = ctypes.c_int32
        output = BloomOwnedBuffer()
        try:
            if run(pointer, input_slice, params_slice, ctypes.byref(output), error, len(error)) != 0:
                raise RuntimeError(f"ABI-2 inference failed: {error.value!r}")
            if not output.data or not 0 < output.len <= 64 * 1024:
                raise RuntimeError("ABI-2 smoke returned an invalid output buffer")
            read_output(ctypes.string_at(output.data, output.len))
        finally:
            library.bloom_buffer_free(ctypes.byref(output))
        if output.data or output.len:
            raise RuntimeError("ABI-2 buffer free did not reset ownership")
        chunks, callback_errors = [], []
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t)

        @callback_type
        def callback_v2(_context, chunk, length):
            try:
                if _context != 0xB100:
                    raise ValueError("callback context was not preserved")
                if not chunk or not 0 < length <= 64 * 1024:
                    raise ValueError("invalid callback slice")
                chunks.append(json.loads(ctypes.string_at(chunk, length)))
            except Exception as exc:
                callback_errors.append(str(exc))

        stream = library.bloom_pipeline_run_stream_v2
        stream.argtypes = [ctypes.c_void_p, BloomSlice, BloomSlice, callback_type, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        stream.restype = ctypes.c_int32
        if stream(pointer, input_slice, params_slice, callback_v2, 0xB100, None, error, len(error)) != 0 or callback_errors:
            raise RuntimeError(f"ABI-2 stream failed: {error.value!r}, {callback_errors}")
        check_stream(chunks)
        token = library.bloom_cancellation_token_new()
        if not token:
            raise RuntimeError("ABI-2 cancellation-token allocation returned NULL")
        try:
            if library.bloom_cancellation_token_cancel(token) != 0:
                raise RuntimeError("ABI-2 cancellation failed")
            chunks.clear()
            status = stream(pointer, input_slice, params_slice, callback_v2, 0xB100, token, error, len(error))
            if status != -8 or chunks or callback_errors:
                raise RuntimeError("ABI-2 cancelled stream emitted output or returned the wrong status")
        finally:
            library.bloom_cancellation_token_free(token)
    finally:
        free(pointer)


def inspect_library(path: Path, *, min_version: int = 1, max_version: int = 2, exercise_inference: bool = False) -> int:
    """Load and validate one library, returning its declared ABI revision."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"native library does not exist: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise RuntimeError(f"cannot load {path}: {error}") from error
    version = _abi_version(library)
    if not min_version <= version <= max_version:
        raise RuntimeError(
            f"{path}: ABI revision {version} is outside supported range "
            f"{min_version}..{max_version}"
        )
    missing = _check_symbols(library, version)
    if missing:
        raise RuntimeError(f"{path}: missing ABI-{version} symbols: {', '.join(missing)}")
    _exercise_legacy_error_path(library)
    if version >= 2:
        _exercise_v2_error_and_lifecycle(library)
    if exercise_inference:
        _exercise_inference(library, version)
    return version


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        action="append",
        type=Path,
        required=True,
        help="native library to validate; repeat for current and legacy builds",
    )
    parser.add_argument("--min-abi", type=int, default=1)
    parser.add_argument("--max-abi", type=int, default=2)
    parser.add_argument("--exercise-inference", action="store_true", help="also run UTF-8 buffered/streamed inference through both ABI revisions using the built-in mock engine")
    args = parser.parse_args(argv)
    if args.min_abi < 1 or args.max_abi < args.min_abi:
        parser.error("ABI range must be positive and ordered")
    try:
        paths = _unique_paths(args.library)
        if not paths:
            parser.error("at least one --library is required")
        reports = [
            (path, inspect_library(path, min_version=args.min_abi, max_version=args.max_abi, exercise_inference=args.exercise_inference))
            for path in paths
        ]
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    for path, version in reports:
        print(f"OK: {path} exports and exercises ABI revision {version}")
    print(f"PASS: validated {len(reports)} packaged native ABI library(s)")
    if args.exercise_inference:
        print("PASS: native ABI buffered/streamed UTF-8 calls and pre-cancelled ABI-2 stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
