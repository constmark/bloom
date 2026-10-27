#!/usr/bin/env python3
"""Fail-closed validator for one Bloom deployment-cell workload report."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def validate(
    report: Any,
    *,
    min_duration_seconds: float,
    expected_model_sha256: str | None = None,
    expected_server_sha256: str | None = None,
    expected_backend: str | None = None,
    expected_device: str | None = None,
    min_concurrency_levels: int = 2,
    failure_report: Any | None = None,
) -> dict[str, Any]:
    """Validate a workload report and return a bounded validation result."""
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    if report.get("schema_version") != 1 or report.get("object") != "bloom.http_workload":
        raise ValueError("unsupported workload report identity")
    if report.get("result") != "pass":
        raise ValueError("workload result is not pass")
    identity = report.get("identity")
    settings = report.get("settings")
    if not isinstance(identity, dict) or not isinstance(settings, dict):
        raise ValueError("report is missing identity or settings")
    model = identity.get("model")
    if not isinstance(model, dict):
        raise ValueError("model identity is missing sha256")
    model_sha = _sha256_text(model.get("sha256"), "model sha256")
    server_sha = identity.get("server_sha256")
    server_sha = _sha256_text(server_sha, "server sha256")
    if expected_model_sha256 and model_sha != _sha256_text(expected_model_sha256, "expected model sha256"):
        raise ValueError("model sha256 does not match deployment cell")
    if expected_server_sha256 and server_sha != _sha256_text(expected_server_sha256, "expected server sha256"):
        raise ValueError("server sha256 does not match deployment cell")
    if expected_backend and identity.get("backend") != expected_backend:
        raise ValueError("backend does not match deployment cell")
    if expected_device and identity.get("device") != expected_device:
        raise ValueError("device does not match deployment cell")

    duration = _positive(settings.get("duration_seconds"), "duration_seconds")
    if duration < min_duration_seconds:
        raise ValueError(
            f"duration_seconds {duration:g} is below required {min_duration_seconds:g}"
        )
    max_requests = settings.get("max_requests")
    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests < 1:
        raise ValueError("max_requests must be a positive integer")
    waves = report.get("waves")
    if not isinstance(min_concurrency_levels, int) or min_concurrency_levels < 1:
        raise ValueError("min_concurrency_levels must be positive")
    if not isinstance(waves, list) or len(waves) < min_concurrency_levels:
        raise ValueError("report does not contain enough concurrency waves")
    levels: list[int] = []
    for wave in waves:
        if not isinstance(wave, dict) or not isinstance(wave.get("concurrency"), int):
            raise ValueError("invalid concurrency wave")
        if wave["concurrency"] not in levels:
            levels.append(wave["concurrency"])
        if not isinstance(wave.get("requests"), int) or wave["requests"] < 1:
            raise ValueError("concurrency wave has no completed requests")
    if len(levels) < min_concurrency_levels:
        raise ValueError("concurrency waves do not cover enough distinct levels")
    requests_completed = report.get("requests_completed")
    requested_wave_count = sum(wave["requests"] for wave in waves)
    if (
        not isinstance(requests_completed, int)
        or isinstance(requests_completed, bool)
        or requests_completed <= 0
        or requests_completed != requested_wave_count
    ):
        raise ValueError("requests_completed does not match completed concurrency waves")
    completion_tokens = report.get("completion_tokens")
    if (
        not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens <= 0
    ):
        raise ValueError("completion_tokens is missing or invalid")

    disconnect = report.get("disconnect")
    if not isinstance(disconnect, dict) or not isinstance(disconnect.get("accepted"), int):
        raise ValueError("disconnect recovery evidence is missing")
    if disconnect["accepted"] < 1 or disconnect.get("post_recovery_request") != "pass":
        raise ValueError("disconnect recovery did not complete")
    _positive(disconnect.get("recovery_ms"), "disconnect recovery_ms")

    memory = report.get("host_memory")
    if not isinstance(memory, dict):
        raise ValueError("host memory evidence is missing")
    for field in ("warm_baseline_bytes", "observed_peak_bytes", "after_recovery_bytes"):
        value = memory.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"host memory field {field} is invalid")
    if not isinstance(memory.get("drift_bytes"), int):
        raise ValueError("host memory drift is missing")

    inventory = report.get("backend_inventory")
    if not isinstance(inventory, dict) or inventory.get("status") not in {"available", "partial"}:
        raise ValueError("physical backend inventory is unavailable")
    if failure_report is None:
        raise ValueError("failure-injection report is required")
    if (
        not isinstance(failure_report, dict)
        or failure_report.get("schema_version") != 1
        or failure_report.get("object") != "bloom.deployment_failure_injection"
        or failure_report.get("result") != "pass"
    ):
        raise ValueError("failure-injection report is not a passing Bloom report")
    scenarios = failure_report.get("scenarios")
    required_scenarios = {"invalid_model", "process_interruption", "storage_quota"}
    if (
        not isinstance(scenarios, dict)
        or not required_scenarios.issubset(scenarios)
        or any(
            not isinstance(value, dict) or value.get("result") != "pass"
            for value in scenarios.values()
        )
    ):
        raise ValueError("failure-injection scenarios are incomplete")
    deployment = failure_report.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("failure-injection deployment identity is missing")
    if deployment.get("server_sha256") != server_sha:
        raise ValueError("failure-injection server identity does not match workload")
    if (
        deployment.get("backend") != identity.get("backend")
        or deployment.get("device") != identity.get("device")
    ):
        raise ValueError("failure-injection runtime identity does not match workload")
    if deployment.get("model_sha256") != model_sha:
        raise ValueError("failure-injection model identity does not match workload")
    failure_status = "pass"
    return {
        "schema_version": 1,
        "object": "bloom.deployment_cell_validation",
        "result": "evidence_complete",
        "stable_claim": False,
        "duration_seconds": duration,
        "concurrency_levels": sorted(levels),
        "model_sha256": model_sha,
        "server_sha256": server_sha,
        "backend": identity.get("backend"),
        "device": identity.get("device"),
        "disconnect_recovery": "pass",
        "failure_injection": failure_status,
        "host_memory": {
            "observed_peak_bytes": memory["observed_peak_bytes"],
            "drift_bytes": memory["drift_bytes"],
        },
        "limitations": [
            "Workload evidence only; model-load, disk-full, interruption, upgrade and rollback drills remain separate gates.",
            "Accelerator memory and token-level TBT are not measured by the HTTP workload.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--min-duration-seconds", type=float, default=7200)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-server-sha256")
    parser.add_argument("--backend")
    parser.add_argument("--device")
    parser.add_argument("--min-concurrency-levels", type=int, default=2)
    parser.add_argument("--failure-report", type=pathlib.Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        failure_report = (
            json.loads(args.failure_report.read_text(encoding="utf-8"))
            if args.failure_report is not None
            else None
        )
        result = validate(
            report,
            min_duration_seconds=_positive(args.min_duration_seconds, "min_duration_seconds"),
            expected_model_sha256=args.expected_model_sha256,
            expected_server_sha256=args.expected_server_sha256,
            expected_backend=args.backend,
            expected_device=args.device,
            min_concurrency_levels=args.min_concurrency_levels,
            failure_report=failure_report,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"deployment-cell evidence: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
