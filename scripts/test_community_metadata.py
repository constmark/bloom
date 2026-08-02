#!/usr/bin/env python3
"""Regression tests for Bloom's GitHub community metadata contract."""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from check_community_metadata import (  # noqa: E402
    FORM_CONTRACTS,
    MetadataError,
    load_yaml_text,
    validate_config,
    validate_form,
    validate_repository,
)

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPOSITORY_ROOT / ".github" / "ISSUE_TEMPLATE"


class CommunityMetadataTests(unittest.TestCase):
    @staticmethod
    def load(filename: str) -> tuple[object, str]:
        text = (TEMPLATE_ROOT / filename).read_text(encoding="utf-8")
        return load_yaml_text(text, filename), text

    def test_accepts_current_repository_metadata(self) -> None:
        count, errors = validate_repository(REPOSITORY_ROOT)
        self.assertEqual(count, 3)
        self.assertEqual(errors, [])

    def test_rejects_missing_and_duplicate_form_fields(self) -> None:
        document, text = self.load("bug_report.yml")
        missing = copy.deepcopy(document)
        missing["body"] = [
            element for element in missing["body"] if element.get("id") != "version"
        ]
        self.assertTrue(validate_form("bug_report.yml", missing, text))

        duplicate = copy.deepcopy(document)
        duplicate["body"].append(copy.deepcopy(duplicate["body"][1]))
        errors = validate_form("bug_report.yml", duplicate, text)
        self.assertTrue(any("duplicate field id" in error for error in errors))

    def test_rejects_missing_privacy_guidance(self) -> None:
        document, text = self.load("feature_request.yml")
        mutated = text.replace("Authorization headers", "sensitive headers")
        errors = validate_form("feature_request.yml", document, mutated)
        self.assertTrue(any("Authorization headers" in error for error in errors))

    def test_rejects_optional_acknowledgements(self) -> None:
        document, text = self.load("model_support.yml")
        mutated = copy.deepcopy(document)
        privacy = next(
            element for element in mutated["body"] if element.get("id") == "privacy"
        )
        privacy["attributes"]["options"][0]["required"] = False
        errors = validate_form("model_support.yml", mutated, text)
        self.assertTrue(any("must be required" in error for error in errors))

    def test_rejects_public_blank_issues_and_missing_security_routing(self) -> None:
        document, _text = self.load("config.yml")
        blank = copy.deepcopy(document)
        blank["blank_issues_enabled"] = True
        self.assertIn("config.yml must disable public blank issues", validate_config(blank))

        missing_security = copy.deepcopy(document)
        missing_security["contact_links"] = missing_security["contact_links"][1:]
        self.assertTrue(
            any(
                "private security advisory" in error
                for error in validate_config(missing_security)
            )
        )

    def test_rejects_duplicate_yaml_keys(self) -> None:
        with self.assertRaisesRegex(MetadataError, "duplicate YAML key"):
            load_yaml_text("name: first\nname: second\n", "duplicate.yml")

    def test_contract_defines_only_the_maintained_forms(self) -> None:
        self.assertEqual(
            set(FORM_CONTRACTS),
            {"bug_report.yml", "feature_request.yml", "model_support.yml"},
        )


if __name__ == "__main__":
    unittest.main()
