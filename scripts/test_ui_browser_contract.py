#!/usr/bin/env python3
"""Fail-closed evidence checks for the real multi-engine browser gate."""

import json
from pathlib import Path
import tempfile
import unittest

from test_ui_browser import artifact_snapshot, parse_browser_report, verify_artifact_snapshots


class BrowserEvidenceTests(unittest.TestCase):
    def report(self, browser="chromium", **changes):
        value = {
            "object": "bloom.browser_acceptance", "schema_version": 1,
            "status": "passed", "browser": browser, "browser_version": "152.0",
        }
        value.update(changes)
        return "### Result\n" + json.dumps(value) + "\n### Ran Playwright code\nfixture"

    def test_accepts_each_exact_engine(self):
        for browser in ("chromium", "firefox", "webkit"):
            self.assertEqual(parse_browser_report(self.report(browser), browser)["browser"], browser)

    def test_requires_success_evidence_even_with_zero_exit_status(self):
        for output in ("", "### Error\nTimeoutError", "### Result\nnull", "### Result\n{broken"):
            with self.subTest(output=output), self.assertRaises(AssertionError):
                parse_browser_report(output, "chromium")

    def test_rejects_wrong_engine_failed_and_ambiguous_identity(self):
        for changes in (
            {"browser": "webkit"}, {"status": "failed"}, {"object": "other"},
            {"schema_version": True}, {"schema_version": 2}, {"browser_version": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                parse_browser_report(self.report(**changes), "chromium")
        with self.assertRaises(AssertionError):
            parse_browser_report(self.report() + "\n### Error\npost-result failure", "chromium")


class ArtifactIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.server = self.root / "bloom_server"
        self.server.write_bytes(b"original native executable")
        self.dist = self.root / "dist"
        self.dist.mkdir()
        (self.dist / "index.html").write_bytes(b"original UI")

    def snapshot(self):
        return artifact_snapshot(self.server, self.dist)

    def test_unchanged_content_retains_exact_hashes(self):
        before = self.snapshot()
        verify_artifact_snapshots(before, self.snapshot())
        self.assertEqual(before["server_binary"]["status"], "local_process")
        self.assertEqual(len(before["server_binary"]["sha256"]), 64)
        self.assertEqual(len(before["ui_dist"]["tree_sha256"]), 64)
        self.assertEqual(before["ui_dist"]["files"][0]["path"], "index.html")

    def test_changed_native_executable_invalidates_evidence(self):
        before = self.snapshot()
        self.server.write_bytes(b"replacement executable")
        with self.assertRaisesRegex(AssertionError, "server_binary changed"):
            verify_artifact_snapshots(before, self.snapshot())

    def test_changed_added_and_removed_ui_assets_invalidate_evidence(self):
        for mutation in ("content", "added", "renamed"):
            with self.subTest(mutation=mutation):
                before = self.snapshot()
                path = self.dist / "index.html"
                if mutation == "content":
                    path.write_bytes(b"replaced UI")
                elif mutation == "added":
                    (self.dist / "extra.js").write_bytes(b"extra")
                else:
                    path.rename(self.dist / "other.html")
                with self.assertRaisesRegex(AssertionError, "ui_dist changed"):
                    verify_artifact_snapshots(before, self.snapshot())

    def test_external_url_has_no_fabricated_local_binary_identity(self):
        snapshot = artifact_snapshot(None, self.root / "missing-dist")
        self.assertEqual(snapshot["server_binary"]["status"], "not_local")
        self.assertNotIn("sha256", snapshot["server_binary"])
        self.assertEqual(snapshot["ui_dist"]["status"], "not_available")
        verify_artifact_snapshots(snapshot, artifact_snapshot(None, self.root / "missing-dist"))

    def test_symlink_assets_do_not_escape_the_snapshot(self):
        (self.dist / "linked.js").symlink_to(self.server)
        with self.assertRaisesRegex(AssertionError, "symlink"):
            self.snapshot()


if __name__ == "__main__":
    unittest.main()
