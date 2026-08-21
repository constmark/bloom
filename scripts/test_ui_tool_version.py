#!/usr/bin/env python3
"""Regression tests for the fail-closed Dioxus CLI version gate."""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("check_ui_tool_version.sh")
REPOSITORY_ROOT = SCRIPT.parents[1]


def locked_dioxus_version() -> str:
    lines = (REPOSITORY_ROOT / "ui/Cargo.lock").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line == 'name = "dioxus"':
            version_line = lines[index + 1]
            prefix = 'version = "'
            if version_line.startswith(prefix) and version_line.endswith('"'):
                return version_line[len(prefix) : -1]
    raise AssertionError("ui/Cargo.lock does not contain the dioxus package version")


LOCKED_DX_VERSION = locked_dioxus_version()


class UiToolVersionTests(unittest.TestCase):
    def run_with_fake_dx(
        self, version_output: str, version_status: int = 0
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            marker = temp / "unexpected-build-call"
            fake_dx = temp / "dx"
            fake_dx.write_text(
                "#!/bin/sh\n"
                'if [ "${1:-}" = "--version" ]; then\n'
                f"  printf '%s\\n' {version_output!r}\n"
                f"  exit {version_status}\n"
                "fi\n"
                'touch "$BLOOM_TEST_DX_MARKER"\n'
                "exit 99\n",
                encoding="utf-8",
            )
            fake_dx.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{temp}:{environment['PATH']}"
            environment["BLOOM_TEST_DX_MARKER"] = str(marker)
            result = subprocess.run(
                ["/bin/bash", str(SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            return result, marker.exists()

    def test_accepts_the_exact_locked_version_without_building(self) -> None:
        result, build_called = self.run_with_fake_dx(
            f"dioxus {LOCKED_DX_VERSION} (fixture)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("matches the locked", result.stdout)
        self.assertFalse(build_called)

    def test_rejects_mismatched_malformed_and_failed_version_probes(self) -> None:
        cases = (
            (
                "dioxus 0.0.0 (fixture)",
                0,
                f"{LOCKED_DX_VERSION} is required",
            ),
            ("unexpected", 0, "could not parse"),
            (f"dioxus {LOCKED_DX_VERSION}", 7, "--version failed"),
        )
        for output, status, expected in cases:
            with self.subTest(output=output, status=status):
                result, build_called = self.run_with_fake_dx(output, status)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertFalse(build_called)

    def test_rejects_a_missing_dx_binary(self) -> None:
        environment = os.environ.copy()
        environment["PATH"] = "/usr/bin:/bin"
        result = subprocess.run(
            ["/bin/bash", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dx is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
