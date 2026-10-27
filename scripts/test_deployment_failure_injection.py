#!/usr/bin/env python3
"""Unit tests for failure-injection result classification."""

from __future__ import annotations

import sys
import pathlib
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import deployment_failure_injection as injection  # noqa: E402


class FailureInjectionTests(unittest.TestCase):
    def test_ready_requires_protocol_ready_state(self) -> None:
        self.assertTrue(injection._ready(200, {"status": "ready"}))
        self.assertFalse(injection._ready(503, {"status": "ready"}))
        self.assertFalse(injection._ready(200, {"status": "not_ready"}))
        self.assertFalse(injection._ready(200, None))

    def test_invalid_model_accepts_terminal_readiness_failure(self) -> None:
        self.assertTrue(
            injection._invalid_model_failed(
                503, {"status": "not_ready", "load_error": "model load failed"}, False, None, ""
            )
        )

    def test_invalid_model_accepts_logged_startup_failure(self) -> None:
        self.assertTrue(
            injection._invalid_model_failed(0, None, True, 1, "model file not found")
        )

    def test_invalid_model_rejects_unrelated_exit(self) -> None:
        self.assertFalse(
            injection._invalid_model_failed(0, None, True, 1, "unexpected network error")
        )

    def test_invalid_model_rejects_running_server_without_failure(self) -> None:
        self.assertFalse(
            injection._invalid_model_failed(200, {"status": "ready"}, False, None, "")
        )

    def test_storage_quota_accepts_protocol_conflict_without_staging(self) -> None:
        class Process:
            def poll(self):
                return None

        with self.subTest("quota conflict"):
            with mock.patch.object(
                injection,
                "_start",
                return_value=(Process(), 43123),
            ), mock.patch.object(
                injection,
                "_post_json",
                return_value=(
                    409,
                    {
                        "error": {
                            "type": "model_import_conflict",
                            "message": "quota exceeded",
                        }
                    },
                ),
            ), mock.patch.object(injection, "_stop") as stop:
                with tempfile.TemporaryDirectory() as raw:
                    result = injection._storage_quota(
                        pathlib.Path("/tmp/bloom-test-server"),
                        pathlib.Path(raw),
                        "candle",
                        "cpu",
                    )
                self.assertEqual(result["result"], "pass")
                stop.assert_called_once()

    def test_storage_quota_stops_server_when_startup_has_no_port(self) -> None:
        class Process:
            def poll(self):
                return None

        with mock.patch.object(
            injection,
            "_start",
            return_value=(Process(), 0),
        ), mock.patch.object(injection, "_stop") as stop:
            with tempfile.TemporaryDirectory() as raw:
                with self.assertRaisesRegex(RuntimeError, "did not publish a port"):
                    injection._storage_quota(
                        pathlib.Path("/tmp/bloom-test-server"),
                        pathlib.Path(raw),
                        "candle",
                        "cpu",
                    )
            stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
