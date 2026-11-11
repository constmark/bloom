#!/usr/bin/env python3
"""Regression checks for source provenance of the explicitly named Laya fork."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from check_laya_upstream import verify


class LayaUpstreamTests(unittest.TestCase):
    def test_repository_matches_reviewed_source(self):
        verify(Path(__file__).resolve().parents[1])

    def test_changed_code_license_and_provenance_fail_closed(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(repository / "crates/laya", root / "crates/laya")
            (root / "config").mkdir()
            shutil.copyfile(repository / "config/dependency-policy.json", root / "config/dependency-policy.json")
            verify(root)
            for relative in ["src/lib.rs", "LICENSE", "README.md"]:
                path = root / "crates/laya" / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\nchanged\n")
                with self.subTest(path=relative), self.assertRaises(ValueError):
                    verify(root)
                path.write_bytes(original)
            manifest = root / "crates/laya/UPSTREAM.json"
            data = json.loads(manifest.read_text())
            data["upstream"]["archive_sha256"] = "0" * 64
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                verify(root)


if __name__ == "__main__":
    unittest.main()
