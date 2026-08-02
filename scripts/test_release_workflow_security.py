#!/usr/bin/env python3
"""Regression tests for the release workflow security contract."""

from __future__ import annotations

import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from check_release_workflow_security import validate_text  # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "release.yml"


class ReleaseWorkflowSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        cls.attest_start = cls.workflow.index(
            "      - name: Attest release archives and checksums"
        )
        cls.publish_start = cls.workflow.index(
            "      - uses: softprops/action-gh-release@", cls.attest_start
        )

    def assert_rejected(self, workflow: str, expected: str) -> None:
        self.assertTrue(
            any(expected in error for error in validate_text(workflow)),
            msg=f"expected error containing {expected!r}",
        )

    def test_accepts_current_release_workflow(self) -> None:
        self.assertEqual(validate_text(self.workflow), [])

    def test_rejects_top_level_write_permission(self) -> None:
        mutated = self.workflow.replace("  contents: read", "  contents: write", 1)
        self.assert_rejected(mutated, "must default contents permission to read")

    def test_rejects_attestation_permission_downgrade(self) -> None:
        mutated = self.workflow.replace(
            "      attestations: write", "      attestations: read", 1
        )
        self.assert_rejected(
            mutated, "github-release permission attestations must be write"
        )

    def test_rejects_subject_present_only_in_publication_step(self) -> None:
        mutated = (
            self.workflow[: self.attest_start]
            + self.workflow[self.attest_start :].replace(
                "            artifacts/*.zip\n", "", 1
            )
        )
        self.assertIn("artifacts/*.zip", mutated[self.publish_start :])
        self.assert_rejected(
            mutated, "release attestation omits subject path(s): artifacts/*.zip"
        )

    def test_rejects_mutable_attestation_action(self) -> None:
        mutated = self.workflow.replace(
            "actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d",
            "actions/attest@v4.2.1",
            1,
        )
        self.assert_rejected(mutated, "lacks a full-SHA actions/attest step")

    def test_rejects_publication_before_attestation(self) -> None:
        attest_block = self.workflow[self.attest_start : self.publish_start]
        publish_block = self.workflow[self.publish_start :]
        mutated = self.workflow[: self.attest_start] + publish_block + attest_block
        self.assert_rejected(
            mutated, "release order must be download, attest, then publish"
        )

    def test_rejects_duplicate_attestation_steps(self) -> None:
        attest_block = self.workflow[self.attest_start : self.publish_start]
        mutated = (
            self.workflow[: self.publish_start]
            + attest_block
            + self.workflow[self.publish_start :]
        )
        self.assert_rejected(mutated, "must contain exactly one actions/attest step")


if __name__ == "__main__":
    unittest.main()
