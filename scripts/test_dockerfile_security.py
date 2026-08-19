#!/usr/bin/env python3
"""Regression tests for the Dockerfile security contract."""

from __future__ import annotations

import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from check_dockerfile_security import validate_text, validate_ui_toolchain_text  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
UI_TOOLCHAIN = (ROOT / "scripts/install_ui_toolchain_linux.sh").read_text(
    encoding="utf-8"
)


class DockerfileSecurityTests(unittest.TestCase):
    def assert_rejected(self, content: str, expected: str) -> None:
        self.assertTrue(
            any(expected in error for error in validate_text(content)),
            msg=f"expected an error containing {expected!r}",
        )

    def test_accepts_repository_dockerfile(self) -> None:
        self.assertEqual(validate_text(DOCKERFILE), [])
        self.assertEqual(validate_ui_toolchain_text(UI_TOOLCHAIN), [])

    def test_rejects_mutable_base(self) -> None:
        mutated = DOCKERFILE.replace(
            "debian:bookworm-slim@sha256:", "debian:bookworm-slim#sha256:", 1
        )
        self.assert_rejected(mutated, "base image is not tag-and-digest pinned")

    def test_rejects_root_runtime(self) -> None:
        mutated = DOCKERFILE.replace("USER 10001:10001", "USER 0:0", 1)
        self.assert_rejected(mutated, "non-root user")

    def test_rejects_disabled_strict_security(self) -> None:
        mutated = DOCKERFILE.replace("BLOOM_STRICT_SECURITY=1", "BLOOM_STRICT_SECURITY=0", 1)
        self.assert_rejected(mutated, "BLOOM_STRICT_SECURITY=1")

    def test_rejects_download_enabled_dioxus(self) -> None:
        mutated = DOCKERFILE.replace(" --features no-downloads", "", 1)
        self.assert_rejected(mutated, "fail-closed UI toolchain")

    def test_rejects_online_rustup_resolution(self) -> None:
        mutated = DOCKERFILE.replace("RUSTUP_TOOLCHAIN=bloom-container", "", 1)
        self.assert_rejected(mutated, "fail-closed UI toolchain")

    def test_rejects_changed_ui_tool_digest(self) -> None:
        mutated = UI_TOOLCHAIN.replace(
            "c8ebe5d00c978601086bcad33b2c80fcfe33d6a8b87b754ba4ea86a9a16cc316",
            "0" * 64,
            1,
        )
        self.assertTrue(
            any("BINARYEN_AMD64_SHA256" in error for error in validate_ui_toolchain_text(mutated))
        )


if __name__ == "__main__":
    unittest.main()
