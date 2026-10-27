#!/usr/bin/env python3
"""Unit tests for the packaged C ABI compatibility checker."""

from __future__ import annotations

import ctypes
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_ffi_abi_compatibility as contract


class _Function:
    def __init__(self, value=0):
        self.value = value
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.value


class _Library:
    def __init__(self, *, abi=None, symbols=()):
        if abi is not None:
            self.bloom_abi_version = _Function(abi)
        for symbol in symbols:
            if symbol != "bloom_abi_version":
                setattr(self, symbol, object())


class ContractTests(unittest.TestCase):
    def test_legacy_without_version_is_revision_one(self):
        library = _Library(symbols=contract.ABI1_SYMBOLS)
        self.assertEqual(contract._abi_version(library), 1)
        self.assertEqual(contract._check_symbols(library, 1), [])

    def test_revision_two_requires_every_new_symbol(self):
        library = _Library(abi=2, symbols=contract.ABI1_SYMBOLS)
        missing = contract._check_symbols(library, 2)
        self.assertEqual(
            missing,
            [symbol for symbol in contract.ABI2_SYMBOLS if symbol != "bloom_abi_version"],
        )

    def test_declared_version_uses_ctypes_unsigned_value(self):
        library = _Library(abi=2, symbols=contract.ABI1_SYMBOLS + contract.ABI2_SYMBOLS)
        self.assertEqual(contract._abi_version(library), 2)
        self.assertEqual(contract._check_symbols(library, 2), [])

    def test_empty_owned_buffer_matches_native_layout(self):
        buffer = contract.BloomOwnedBuffer()
        self.assertFalse(bool(buffer.data))
        self.assertEqual(buffer.len, 0)
        self.assertEqual(
            ctypes.sizeof(buffer),
            ctypes.sizeof(ctypes.c_void_p) + ctypes.sizeof(ctypes.c_size_t),
        )


if __name__ == "__main__":
    unittest.main()
