#!/usr/bin/env python3
"""Regression tests for Bloom's dependency policy and CycloneDX contract."""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbom_contract import SbomError, build_sbom, validate_sbom_document  # noqa: E402


REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
WORKSPACE_ID = "path+file:///workspace/crates/engine#bloomai-engine@0.1.0"
DEPENDENCY_ID = f"registry+https://github.com/rust-lang/crates.io-index#serde@1.0.0"
UI_WORKSPACE_ID = "path+file:///workspace/ui#bloom-ui@0.1.0"


def policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "object": "bloom.dependency_policy",
        "allowed_registry_sources": [REGISTRY],
        "allowed_license_expressions": ["Apache-2.0", "MIT OR Apache-2.0"],
        "license_expression_normalizations": {},
    }


def metadata() -> dict[str, object]:
    return {
        "workspace_members": [WORKSPACE_ID],
        "packages": [
            {
                "id": WORKSPACE_ID,
                "name": "bloomai-engine",
                "version": "0.1.0",
                "license": "Apache-2.0",
                "source": None,
            },
            {
                "id": DEPENDENCY_ID,
                "name": "serde",
                "version": "1.0.0",
                "license": "MIT OR Apache-2.0",
                "source": REGISTRY,
            },
        ],
        "resolve": {
            "nodes": [
                {"id": WORKSPACE_ID, "deps": [{"pkg": DEPENDENCY_ID}]},
                {"id": DEPENDENCY_ID, "deps": []},
            ]
        },
    }


def ui_metadata() -> dict[str, object]:
    return {
        "workspace_members": [UI_WORKSPACE_ID],
        "packages": [
            {
                "id": UI_WORKSPACE_ID,
                "name": "bloom-ui",
                "version": "0.1.0",
                "license": "Apache-2.0",
                "source": None,
            },
            {
                "id": DEPENDENCY_ID,
                "name": "serde",
                "version": "1.0.0",
                "license": "MIT OR Apache-2.0",
                "source": REGISTRY,
            },
        ],
        "resolve": {
            "nodes": [
                {"id": UI_WORKSPACE_ID, "deps": [{"pkg": DEPENDENCY_ID}]},
                {"id": DEPENDENCY_ID, "deps": []},
            ]
        },
    }


class SbomContractTests(unittest.TestCase):
    def test_build_is_deterministic_and_self_validating(self) -> None:
        first = build_sbom(metadata(), policy(), "x86_64-unknown-linux-gnu", False)
        second = build_sbom(metadata(), policy(), "x86_64-unknown-linux-gnu", False)

        self.assertEqual(first, second)
        self.assertEqual(first["bomFormat"], "CycloneDX")
        self.assertEqual(len(first["components"]), 2)
        self.assertEqual(len(first["dependencies"]), 3)
        validate_sbom_document(
            first,
            policy(),
            "0.1.0",
            "x86_64-unknown-linux-gnu",
            False,
        )

    def test_embedded_ui_merges_its_independent_workspace(self) -> None:
        sbom = build_sbom(
            [metadata(), ui_metadata()],
            policy(),
            "x86_64-unknown-linux-gnu",
            True,
        )
        names = {component["name"] for component in sbom["components"]}
        self.assertEqual(names, {"bloomai-engine", "bloom-ui", "serde"})
        self.assertEqual(len(sbom["dependencies"]), 4)

        for documents, embedded_ui in (
            (metadata(), True),
            ([metadata(), ui_metadata()], False),
        ):
            with self.subTest(embedded_ui=embedded_ui):
                with self.assertRaisesRegex(SbomError, "embedded-UI"):
                    build_sbom(
                        documents,
                        policy(),
                        "x86_64-unknown-linux-gnu",
                        embedded_ui,
                    )

    def test_unreviewed_sources_and_licenses_fail_closed(self) -> None:
        cases = (
            ("license", "license", "GPL-3.0-only"),
            ("source", "source", "git+https://example.com/dependency"),
        )
        for label, field, value in cases:
            with self.subTest(label=label):
                changed = metadata()
                changed["packages"][1][field] = value
                with self.assertRaisesRegex(SbomError, "unreviewed"):
                    build_sbom(
                        changed, policy(), "x86_64-unknown-linux-gnu", False
                    )

    def test_external_path_dependencies_and_incomplete_edges_are_rejected(self) -> None:
        changed = metadata()
        changed["packages"][1]["source"] = None
        with self.assertRaisesRegex(SbomError, "unreviewed source"):
            build_sbom(changed, policy(), "x86_64-unknown-linux-gnu", False)

        changed = metadata()
        changed["resolve"]["nodes"][0]["deps"][0]["pkg"] = "missing"
        with self.assertRaisesRegex(SbomError, "unresolved"):
            build_sbom(changed, policy(), "x86_64-unknown-linux-gnu", False)

    def test_release_binding_and_component_policy_are_revalidated(self) -> None:
        original = build_sbom(
            [metadata(), ui_metadata()],
            policy(),
            "x86_64-unknown-linux-gnu",
            True,
        )
        mutations = (
            (
                "target",
                lambda value: value["metadata"]["properties"][1].__setitem__(
                    "value", "aarch64-apple-darwin"
                ),
                "target",
            ),
            (
                "license",
                lambda value: value["components"][0]["licenses"][0].__setitem__(
                    "expression", "GPL-3.0-only"
                ),
                "license",
            ),
            (
                "source",
                lambda value: value["components"][0]["properties"][1].__setitem__(
                    "value", "git+https://example.com/dependency"
                ),
                "source",
            ),
            (
                "graph",
                lambda value: value["dependencies"].pop(),
                "graph",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(original)
                mutate(changed)
                with self.assertRaisesRegex(SbomError, expected):
                    validate_sbom_document(
                        changed,
                        policy(),
                        "0.1.0",
                        "x86_64-unknown-linux-gnu",
                        True,
                    )

    def test_untrusted_nested_values_fail_with_contract_errors(self) -> None:
        original = build_sbom(
            [metadata(), ui_metadata()],
            policy(),
            "x86_64-unknown-linux-gnu",
            True,
        )
        mutations = (
            (
                "properties",
                lambda value: value["components"][0]["properties"].__setitem__(
                    0, []
                ),
                "properties",
            ),
            (
                "dependency edge",
                lambda value: value["dependencies"][0].__setitem__(
                    "dependsOn", [{}]
                ),
                "bounded string",
            ),
            (
                "root reference",
                lambda value: value["metadata"]["component"].__setitem__(
                    "bom-ref", "pkg:generic/bloom@wrong"
                ),
                "root reference",
            ),
            (
                "component purl",
                lambda value: value["components"][0].__setitem__(
                    "name", "renamed"
                ),
                "purl",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(original)
                mutate(changed)
                with self.assertRaisesRegex(SbomError, expected):
                    validate_sbom_document(
                        changed,
                        policy(),
                        "0.1.0",
                        "x86_64-unknown-linux-gnu",
                        True,
                    )


if __name__ == "__main__":
    unittest.main()
