#!/usr/bin/env python3
"""Regression checks for TileLang's Windows symbol exports."""

from __future__ import annotations

import pathlib
import re
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "crates/tilelang/scripts/generate_kernel.py"


class TileLangWindowsExportTests(unittest.TestCase):
    def test_every_generated_kernel_has_an_explicit_msvc_export(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for operation, symbol in (
            ("vector_add", "vector_add_launch"),
            ("matmul", "matmul_launch"),
            ("softmax", "softmax_launch"),
            ("attention", "attention_launch"),
            ("mrope", "mrope_launch"),
        ):
            with self.subTest(operation=operation):
                self.assertRegex(
                    source,
                    rf'(?:if|elif) "{operation}" in name:\s+export_sym = "{symbol}"',
                )


if __name__ == "__main__":
    unittest.main()
