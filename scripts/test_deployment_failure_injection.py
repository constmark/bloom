#!/usr/bin/env python3
"""Unit tests for failure-injection result classification."""

from __future__ import annotations

import sys
import argparse
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import deployment_failure_injection as injection  # noqa: E402


class FailureInjectionTests(unittest.TestCase):
    def test_slow_model_loading_does_not_fail_interruption_drill(self) -> None:
        fixture = json.loads((Path(__file__).resolve().parents[1] / "examples/readiness.json").read_text())
        fixture.update(status="ready", model="fixture", progress=100,
                       model_tasks=["generation"], context_window=2048)
        loading = {**fixture, "status": "not_ready", "loading": True, "progress": 5}
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch.object(injection, "_read_log", return_value="server running on http://127.0.0.1:1234"), \
             mock.patch.object(injection, "_request", side_effect=[(503, loading), (200, fixture)]) as request, \
             mock.patch.object(injection.time, "sleep"):
            status, payload, _elapsed = injection._wait_ready(process, Path("unused"), 2)
        self.assertEqual(status, 200)
        self.assertEqual(payload, fixture)
        self.assertEqual(request.call_count, 2)

    def test_ready_requires_protocol_ready_state(self) -> None:
        fixture = json.loads((Path(__file__).resolve().parents[1] / "examples/readiness.json").read_text())
        fixture.update(status="ready", model="fixture", progress=100,
                       model_tasks=["generation"], context_window=2048)
        self.assertTrue(injection._ready(200, fixture))
        self.assertFalse(injection._ready(200, {"status": "ready"}))
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

    def test_child_isolated_from_operator_environment_and_saved_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with mock.patch.dict(os.environ, {"BLOOM_CONFIG": "/private/config.json",
                 "BLOOM_API_KEY": "private-secret", "BLOOM_ALLOW_HASH_MISMATCH": "1"}), \
                 mock.patch.object(injection.subprocess, "Popen") as start, \
                 mock.patch.object(injection, "_wait_for_port", return_value=123):
                injection._start(root / "server", root / "models", root / "log", None,
                                 "candle", "cpu", environment={"BLOOM_OPERATOR_API_KEY": "test"})
                command = start.call_args.args[0]
                self.assertEqual(command[command.index("--config") + 1], str(root / "isolated-config.json"))
                environment = start.call_args.kwargs["env"]
                self.assertNotIn("BLOOM_CONFIG", environment)
                self.assertNotIn("BLOOM_API_KEY", environment)
                self.assertNotIn("BLOOM_ALLOW_HASH_MISMATCH", environment)
                self.assertEqual(environment["BLOOM_OPERATOR_API_KEY"], "test")

    def test_binary_replacement_during_drill_rejects_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            binary = Path(raw) / "server"
            binary.write_bytes(b"before")
            def replace(*args):
                binary.write_bytes(b"after")
                return {"result": "pass"}
            args = argparse.Namespace(server_bin=binary, model=None, scenario="invalid-model",
                                      backend="candle", device="cpu")
            with mock.patch.object(injection, "_invalid_model", side_effect=replace):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    injection.run(args)


if __name__ == "__main__":
    unittest.main()
