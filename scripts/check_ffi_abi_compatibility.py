#!/usr/bin/env python3
"""Validate packaged Bloom native libraries against the public C ABI contract.

The Python wheel is intentionally pure Python and loads ``bloomai-ffi`` at
runtime.  This gate therefore checks every native library supplied to it
independently, including a previously released ABI-1 library when one is
available.  It validates the exported symbol set and exercises the safe error
and cancellation paths without requiring a model or accelerator.
"""

from __future__ import annotations

import argparse
import ctypes
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


def inspect_library(path: Path, *, min_version: int = 1, max_version: int = 2) -> int:
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
    args = parser.parse_args(argv)
    if args.min_abi < 1 or args.max_abi < args.min_abi:
        parser.error("ABI range must be positive and ordered")
    try:
        paths = _unique_paths(args.library)
        if not paths:
            parser.error("at least one --library is required")
        reports = [
            (path, inspect_library(path, min_version=args.min_abi, max_version=args.max_abi))
            for path in paths
        ]
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    for path, version in reports:
        print(f"OK: {path} exports and exercises ABI revision {version}")
    print(f"PASS: validated {len(reports)} packaged native ABI library(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
