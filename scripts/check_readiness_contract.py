#!/usr/bin/env python3
"""Regression tests for the shared Bloom readiness validator."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from readiness_contract import (  # noqa: E402
    MAX_PROTOCOL_VERSION,
    MAX_U64,
    READINESS_REQUIRED_FIELDS,
    ReadinessContractError,
    validate_readiness_document,
    validate_readiness_schema_document,
)


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


class ReadinessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = json.loads(
            (REPOSITORY_ROOT / "examples/readiness.json").read_text(encoding="utf-8")
        )
        self.schema = json.loads(
            (REPOSITORY_ROOT / "examples/readiness.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def assert_invalid(self, mutation, expected: str) -> None:
        value = copy.deepcopy(self.valid)
        mutation(value)
        with self.assertRaisesRegex(ReadinessContractError, expected):
            validate_readiness_document(value)

    def test_accepts_current_and_explicitly_compatible_future_servers(self) -> None:
        self.assertIs(validate_readiness_document(self.valid), self.valid)
        future = copy.deepcopy(self.valid)
        future["protocol_version"] = 4
        future["maximum_ui_protocol_version"] = 4
        future["future_additive_field"] = True
        self.assertIs(validate_readiness_document(future), future)
        with self.assertRaisesRegex(ReadinessContractError, "expected 3"):
            validate_readiness_document(future, expected_server_protocol_version=3)

    def test_requires_the_complete_v3_identity(self) -> None:
        for field in sorted(READINESS_REQUIRED_FIELDS):
            with self.subTest(field=field):
                self.assert_invalid(lambda value, field=field: value.pop(field), "missing")
        self.assert_invalid(
            lambda value: value.__setitem__("schema_version", 2), "unsupported"
        )
        self.assert_invalid(lambda value: value.__setitem__("object", "health"), "unsupported")

    def test_rejects_invalid_protocol_ranges(self) -> None:
        mutations = (
            lambda value: value.__setitem__("protocol_version", True),
            lambda value: value.__setitem__("protocol_version", 0),
            lambda value: value.__setitem__("protocol_version", MAX_PROTOCOL_VERSION + 1),
            lambda value: value.update(
                {"minimum_ui_protocol_version": 4, "maximum_ui_protocol_version": 3}
            ),
            lambda value: value.update(
                {"protocol_version": 4, "maximum_ui_protocol_version": 3}
            ),
            lambda value: value.update(
                {
                    "protocol_version": 4,
                    "minimum_ui_protocol_version": 4,
                    "maximum_ui_protocol_version": 4,
                }
            ),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_invalid(mutation, "protocol|outside")

    def test_rejects_invalid_bounded_metadata(self) -> None:
        mutations = (
            (lambda value: value.__setitem__("server_version", " 0.1.0"), "server version"),
            (lambda value: value.__setitem__("status", "loading"), "status"),
            (lambda value: value.__setitem__("progress", True), "progress"),
            (lambda value: value.__setitem__("progress", 101), "progress"),
            (lambda value: value.__setitem__("model", ""), "model identity"),
            (lambda value: value.__setitem__("loading", 0), "loading flag"),
            (lambda value: value.__setitem__("load_error", ""), "load error"),
            (
                lambda value: value.__setitem__("input_modalities", ["text", "text"]),
                "modalities",
            ),
            (
                lambda value: value.__setitem__("input_modalities", [{"type": "text"}]),
                "modalities",
            ),
            (lambda value: value.__setitem__("model_tasks", ["rerank"]), "tasks"),
            (lambda value: value.__setitem__("model_tasks", [{}]), "tasks"),
            (lambda value: value.__setitem__("context_window", 0), "context window"),
            (lambda value: value.__setitem__("in_flight_requests", True), "in-flight"),
            (
                lambda value: value.__setitem__("in_flight_requests", MAX_U64 + 1),
                "in-flight",
            ),
            (lambda value: value.__setitem__("available_permits", -1), "permit"),
            (
                lambda value: value.__setitem__("available_permits", MAX_U64 + 1),
                "permit",
            ),
            (
                lambda value: value.__setitem__("memory_pressure_high", 0),
                "memory-pressure",
            ),
            (lambda value: value.__setitem__("ram_utilization", float("nan")), "RAM"),
            (lambda value: value.__setitem__("ram_utilization", 1.1), "RAM"),
        )
        for index, (mutation, expected) in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_invalid(mutation, expected)

    def test_rejects_each_inconsistent_ready_state(self) -> None:
        baseline = {
            "status": "ready",
            "progress": 100,
            "model": "tiny.gguf",
            "loading": False,
            "load_error": None,
            "model_tasks": ["generation"],
            "context_window": 4096,
            "available_permits": 1,
            "memory_pressure_high": False,
        }
        invalid_updates = (
            {"model": "not loaded"},
            {"progress": 99},
            {"loading": True},
            {"load_error": "failed"},
            {"model_tasks": []},
            {"context_window": None},
            {"available_permits": 0},
            {"memory_pressure_high": True},
        )
        for update in invalid_updates:
            with self.subTest(update=update):
                value = copy.deepcopy(self.valid)
                value.update(baseline)
                value.update(update)
                with self.assertRaisesRegex(ReadinessContractError, "inconsistent"):
                    validate_readiness_document(value)

    def test_validates_compatibility_critical_schema_structure(self) -> None:
        self.assertIs(validate_readiness_schema_document(self.schema), self.schema)
        mutations = (
            (lambda value: value.__setitem__("$schema", "future"), "Draft-07"),
            (lambda value: value.__setitem__("additionalProperties", False), "policy"),
            (lambda value: value["required"].remove("protocol_version"), "required"),
            (
                lambda value: value["properties"]["schema_version"].update({"const": 2}),
                "version identity",
            ),
            (
                lambda value: value["properties"]["object"].update({"const": "health"}),
                "object identity",
            ),
            (
                lambda value: value["properties"]["protocol_version"].update(
                    {"minimum": 0}
                ),
                "protocol_version",
            ),
        )
        for mutation, expected in mutations:
            with self.subTest(expected=expected):
                schema = copy.deepcopy(self.schema)
                mutation(schema)
                with self.assertRaisesRegex(ReadinessContractError, expected):
                    validate_readiness_schema_document(schema)


if __name__ == "__main__":
    unittest.main()
