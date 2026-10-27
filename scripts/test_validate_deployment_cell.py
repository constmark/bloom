#!/usr/bin/env python3
"""Tests for the deployment-cell evidence gate."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import validate_deployment_cell as gate  # noqa: E402


def report() -> dict:
    return {
        "schema_version": 1,
        "object": "bloom.http_workload",
        "result": "pass",
        "identity": {
            "model": {"sha256": "a" * 64},
            "server_sha256": "b" * 64,
            "backend": "candle",
            "device": "cpu",
        },
        "settings": {"duration_seconds": 7200, "max_requests": 100},
        "waves": [{"concurrency": 1, "requests": 10}, {"concurrency": 4, "requests": 20}],
        "requests_completed": 30,
        "completion_tokens": 90,
        "disconnect": {"accepted": 2, "recovery_ms": 12.0, "post_recovery_request": "pass"},
        "host_memory": {
            "warm_baseline_bytes": 100,
            "observed_peak_bytes": 200,
            "after_recovery_bytes": 110,
            "drift_bytes": 10,
        },
        "backend_inventory": {"status": "available"},
    }


class DeploymentCellGateTests(unittest.TestCase):
    def test_requires_failure_injection_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure-injection report is required"):
            gate.validate(
                report(),
                min_duration_seconds=3600,
                expected_model_sha256="a" * 64,
                expected_server_sha256="b" * 64,
                expected_backend="candle",
                expected_device="cpu",
            )

    def test_records_passing_failure_injection(self) -> None:
        failure = {
            "schema_version": 1,
            "object": "bloom.deployment_failure_injection",
            "result": "pass",
            "deployment": {
                "server_sha256": "b" * 64,
                "model_sha256": "a" * 64,
                "backend": "candle",
                "device": "cpu",
            },
            "scenarios": {
                "invalid_model": {"result": "pass"},
                "process_interruption": {"result": "pass"},
                "storage_quota": {"result": "pass"},
            },
        }
        result = gate.validate(report(), min_duration_seconds=1, failure_report=failure)
        self.assertEqual(result["failure_injection"], "pass")

    def test_rejects_failed_failure_injection(self) -> None:
        failure = {
            "schema_version": 1,
            "object": "bloom.deployment_failure_injection",
            "result": "fail",
            "scenarios": {},
        }
        with self.assertRaisesRegex(ValueError, "failure-injection"):
            gate.validate(report(), min_duration_seconds=1, failure_report=failure)

    def test_rejects_short_run(self) -> None:
        data = report()
        data["settings"]["duration_seconds"] = 30
        with self.assertRaisesRegex(ValueError, "below required"):
            gate.validate(data, min_duration_seconds=3600)

    def test_rejects_missing_wave(self) -> None:
        data = report()
        data["waves"] = [{"concurrency": 1, "requests": 1}]
        with self.assertRaisesRegex(ValueError, "enough concurrency"):
            gate.validate(data, min_duration_seconds=1)

    def test_rejects_wave_count_mismatch(self) -> None:
        data = report()
        data["requests_completed"] = 29
        with self.assertRaisesRegex(ValueError, "requests_completed"):
            gate.validate(data, min_duration_seconds=1)

    def test_rejects_hash_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "model sha256"):
            gate.validate(report(), min_duration_seconds=1, expected_model_sha256="c" * 64)

    def test_rejects_unavailable_inventory(self) -> None:
        data = copy.deepcopy(report())
        data["backend_inventory"]["status"] = "unavailable"
        with self.assertRaisesRegex(ValueError, "backend inventory"):
            gate.validate(data, min_duration_seconds=1)

    def test_rejects_disconnect_without_followup(self) -> None:
        data = copy.deepcopy(report())
        data["disconnect"]["post_recovery_request"] = "fail"
        with self.assertRaisesRegex(ValueError, "disconnect recovery"):
            gate.validate(data, min_duration_seconds=1)

    def test_rejects_unbound_failure_report(self) -> None:
        failure = {
            "schema_version": 1,
            "object": "bloom.deployment_failure_injection",
            "result": "pass",
            "deployment": {
                "server_sha256": "c" * 64,
                "model_sha256": "a" * 64,
                "backend": "candle",
                "device": "cpu",
            },
            "scenarios": {
                "invalid_model": {"result": "pass"},
                "process_interruption": {"result": "pass"},
                "storage_quota": {"result": "pass"},
            },
        }
        with self.assertRaisesRegex(ValueError, "server identity"):
            gate.validate(report(), min_duration_seconds=1, failure_report=failure)


if __name__ == "__main__":
    unittest.main()
