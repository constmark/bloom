#!/usr/bin/env python3
"""Offline integrity, extraction, host-selection and execution boundary tests."""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

import install_ui_toolchain as installer


class VerifiedToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "tool.tar.gz"
        self.output = self.root / "wasm-opt"

    def fixture(self, members):
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, kind, size in members:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.size = size
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = "../../outside"
                archive.addfile(member, io.BytesIO(b"x" * size))
        return hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def test_exact_regular_file_only(self):
        digest = self.fixture([("../../outside", tarfile.REGTYPE, 5),
                               ("bin/tool", tarfile.REGTYPE, 3)])
        installer.verified_binary(self.archive, digest, "bin/tool", self.output)
        self.assertEqual(self.output.read_bytes(), b"xxx")
        self.assertFalse((self.root / "outside").exists())

    def test_mismatch_never_writes_or_executes(self):
        self.fixture([("bin/tool", tarfile.REGTYPE, 3)])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            installer.verified_binary(self.archive, "0" * 64, "bin/tool", self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_non_regular_missing_empty_and_duplicate_binaries(self):
        cases = [[], [("bin/tool", tarfile.SYMTYPE, 0)],
                 [("bin/tool", tarfile.LNKTYPE, 0)], [("bin/tool", tarfile.DIRTYPE, 0)],
                 [("bin/tool", tarfile.REGTYPE, 0)],
                 [("bin/tool", tarfile.REGTYPE, 1)] * 2]
        for members in cases:
            with self.subTest(members=members):
                digest = self.fixture(members)
                with self.assertRaises(ValueError):
                    installer.verified_binary(self.archive, digest, "bin/tool", self.output)
                self.assertFalse(self.output.exists())

    def test_rejects_oversized_binary(self):
        digest = self.fixture([("bin/tool", tarfile.REGTYPE, 4)])
        with mock.patch.object(installer, "MAX_BINARY_BYTES", 3):
            with self.assertRaisesRegex(ValueError, "bounded regular file"):
                installer.verified_binary(self.archive, digest, "bin/tool", self.output)

    def test_host_platforms_and_architecture_aliases(self):
        for system, machine, expected in [
            ("Darwin", "arm64", "darwin-aarch64"),
            ("Darwin", "x86_64", "darwin-x86_64"),
            ("Windows", "AMD64", "windows-x86_64"),
            ("Linux", "aarch64", "linux-aarch64"),
            ("Linux", "x86_64", "linux-x86_64"),
        ]:
            with mock.patch.object(installer.platform, "system", return_value=system), \
                    mock.patch.object(installer.platform, "machine", return_value=machine):
                self.assertEqual(installer.host_target(), expected)
        with mock.patch.object(installer.platform, "machine", return_value="unknown"):
            with self.assertRaisesRegex(ValueError, "unsupported UI build host"):
                installer.host_target()

    def test_every_target_has_three_https_hash_pinned_archives(self):
        for target in installer.TARGETS:
            selected = installer.assets(target)
            self.assertEqual(len(selected), 3)
            for name, url, digest, member in selected:
                self.assertTrue(url.startswith("https://"))
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
                self.assertEqual(Path(member).name, name)
                self.assertEqual(name.endswith(".exe"), target.startswith("windows-"))

    def test_linux_pins_match_existing_docker_installer(self):
        source = Path(__file__).with_name("install_ui_toolchain_linux.sh").read_text()
        for target in ("linux-x86_64", "linux-aarch64"):
            for _, _, digest, _ in installer.assets(target):
                self.assertIn(digest, source)

    def test_bindgen_cli_matches_the_ui_lockfile(self):
        lockfile = Path(__file__).resolve().parents[1] / "ui/Cargo.lock"
        versions = re.findall(r'name = "wasm-bindgen"\nversion = "([^"]+)"', lockfile.read_text())
        self.assertEqual(versions, [installer.WASM_BINDGEN_VERSION])

    def test_all_version_checks_are_exact_and_bounded(self):
        for name, valid, bad in [
            ("wasm-opt", "wasm-opt version 127", "wasm-opt version 1270"),
            ("wasm-bindgen.exe", "wasm-bindgen 0.2.129", "wasm-bindgen 0.2.125"),
            ("esbuild", "0.27.3", "0.27.30"),
        ]:
            for output, accepted in ((valid, True), (bad, False)):
                with mock.patch.object(installer.subprocess, "run", return_value=
                                       subprocess.CompletedProcess([], 0, stdout=output)) as run:
                    if accepted:
                        installer.verify_version(self.root / name)
                    else:
                        with self.assertRaises(ValueError):
                            installer.verify_version(self.root / name)
                    self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_failed_integrity_preserves_previous_install_and_does_not_execute(self):
        self.output.write_text("previous binary")
        with mock.patch.object(installer, "host_target", return_value="darwin-aarch64"), \
                mock.patch.object(installer, "verified_binary", side_effect=ValueError("SHA-256 mismatch")), \
                mock.patch.object(installer, "verify_version") as version, \
                mock.patch.object(installer.subprocess, "run"):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                installer.install(self.root)
            version.assert_not_called()
        self.assertEqual(self.output.read_text(), "previous binary")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["wasm-opt"])

    def test_macos_library_is_verified_before_execution_and_preserves_layout(self):
        digest = self.fixture([
            ("binaryen-version_127/bin/wasm-opt", tarfile.REGTYPE, 3),
            ("binaryen-version_127/lib/libbinaryen.dylib", tarfile.REGTYPE, 4),
        ])
        archive_bytes = self.archive.read_bytes()
        prefix = self.root / "tools"

        def download(arguments, **_):
            Path(arguments[arguments.index("--output") + 1]).write_bytes(archive_bytes)

        def version(binary):
            self.assertEqual(binary.read_bytes(), b"xxx")
            self.assertEqual((binary.parent.parent / "lib/libbinaryen.dylib").read_bytes(), b"xxxx")

        with mock.patch.object(installer, "host_target", return_value="darwin-aarch64"), \
                mock.patch.object(installer, "assets", return_value=[
                    ("wasm-opt", "https://example.org/tool.tar.gz", digest, "binaryen-version_127/bin/wasm-opt")]), \
                mock.patch.object(installer, "verify_version", side_effect=version), \
                mock.patch.object(installer.subprocess, "run", side_effect=download):
            installer.install(prefix)
        self.assertEqual((prefix / "bin/wasm-opt").read_bytes(), b"xxx")
        self.assertEqual((prefix / "lib/libbinaryen.dylib").read_bytes(), b"xxxx")


if __name__ == "__main__":
    unittest.main()
