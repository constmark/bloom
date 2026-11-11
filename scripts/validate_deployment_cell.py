#!/usr/bin/env python3
"""Fail-closed validator for one Bloom deployment-cell workload report."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from typing import Any

from http_workload import _sanitize_backend_inventory


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _distribution(value: Any, name: str, count: int) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("count")) is not int or value["count"] != count:
        raise ValueError(f"{name} count does not match completed requests")
    previous = 0.0
    for field in ("p50", "p95", "p99", "max"):
        current = _positive(value.get(field), f"{name}.{field}")
        if current < previous:
            raise ValueError(f"{name} percentiles are not ordered")
        previous = current
    return value


def _check_budgets(report: dict[str, Any], budgets: Any) -> dict[str, Any]:
    """Require every service budget; omitted measurements never count as zero."""
    measurements = {
        "cold_start_ms": report.get("cold_start_ms"),
        "latency_p95_ms": report["latency_ms"]["p95"],
        "first_content_p95_ms": report["first_content_ms"]["p95"],
        "recovery_ms": report["disconnect"]["recovery_ms"],
        "peak_rss_bytes": report["host_memory"]["observed_peak_bytes"],
        "rss_growth_bytes": max(0, report["host_memory"]["drift_bytes"]),
    }
    if not isinstance(budgets, dict) or set(budgets) != set(measurements):
        raise ValueError("budgets must declare exactly: " + ", ".join(measurements))
    checked = {}
    for name, measured in measurements.items():
        limit = (0.0 if name == "rss_growth_bytes" and type(budgets[name]) in (int, float)
                 and budgets[name] == 0 else _positive(budgets[name], f"budget {name}"))
        if name != "rss_growth_bytes":
            _positive(measured, name)
        if measured > limit:
            raise ValueError(f"{name} exceeds deployment budget: {measured} > {limit:g}")
        checked[name] = {"measured": measured, "limit": limit}
    return checked


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
    min_requests: int = 1,
    budgets: Any | None = None,
    production_profile: bool = False,
) -> dict[str, Any]:
    """Validate a workload report and return a bounded validation result."""
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("object") != "bloom.http_workload":
        raise ValueError("unsupported workload report identity")
    if report.get("result") != "pass":
        raise ValueError("workload result is not pass")
    identity = report.get("identity")
    settings = report.get("settings")
    if not isinstance(identity, dict) or not isinstance(settings, dict):
        raise ValueError("report is missing identity or settings")
    for field in ("backend", "device"):
        value = identity.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError(f"runtime identity {field} is missing or invalid")
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

    min_duration_seconds = _positive(min_duration_seconds, "min_duration_seconds")
    min_requests = _positive_int(min_requests, "min_requests")
    if production_profile:
        min_duration_seconds = max(86400, min_duration_seconds)
        min_requests = max(1000, min_requests)
        if budgets is None:
            raise ValueError("production profile requires explicit service budgets")
        if not all((expected_model_sha256, expected_server_sha256, expected_backend, expected_device)):
            raise ValueError("production profile requires expected model, server, backend and device identity")
        if expected_device.lower() == "auto":
            raise ValueError("production profile requires an explicit physical device selection")
    duration = _positive(settings.get("duration_seconds"), "duration_seconds")
    if duration < min_duration_seconds:
        raise ValueError(
            f"duration_seconds {duration:g} is below required {min_duration_seconds:g}"
        )
    # A requested duration is configuration, not proof that a soak ran. Both
    # the observed timer and its constituent waves must cover the target.
    elapsed_ms = _positive(report.get("workload_elapsed_ms"), "workload_elapsed_ms")
    required_ms = max(duration, min_duration_seconds) * 1000
    if elapsed_ms + 0.001 < required_ms:
        raise ValueError("observed workload duration is below the required duration")
    max_requests = _positive_int(settings.get("max_requests"), "max_requests")
    min_concurrency_levels = _positive_int(min_concurrency_levels, "min_concurrency_levels")
    configured_levels = settings.get("concurrency")
    if not isinstance(configured_levels, list) or not configured_levels:
        raise ValueError("settings.concurrency must be a non-empty array")
    for value in configured_levels:
        _positive_int(value, "settings.concurrency level")
    if len(set(configured_levels)) != len(configured_levels):
        raise ValueError("settings.concurrency levels must be distinct")
    waves = report.get("waves")
    if not isinstance(waves, list) or len(waves) < min_concurrency_levels:
        raise ValueError("report does not contain enough concurrency waves")
    levels: list[int] = []
    wave_elapsed_ms = 0.0
    for wave in waves:
        if not isinstance(wave, dict):
            raise ValueError("invalid concurrency wave")
        level = _positive_int(wave.get("concurrency"), "wave concurrency")
        if level not in configured_levels:
            raise ValueError("wave concurrency is not a configured level")
        if level not in levels:
            levels.append(level)
        if _positive_int(wave.get("requests"), "wave requests") != level:
            raise ValueError("wave requests do not match concurrency")
        wave_elapsed_ms += _positive(wave.get("elapsed_ms"), "wave elapsed_ms")
    # Each wave and the overall timer are rounded independently to 0.001 ms.
    rounding_slack = (len(waves) + 1) * 0.001
    if wave_elapsed_ms > elapsed_ms + rounding_slack:
        raise ValueError("wave elapsed time exceeds observed workload duration")
    if elapsed_ms - wave_elapsed_ms > max(1000.0, elapsed_ms * 0.01):
        raise ValueError("observed workload duration is not supported by completed waves")
    if set(levels) != set(configured_levels):
        raise ValueError("configured concurrency levels were not all exercised")
    if len(levels) < min_concurrency_levels:
        raise ValueError("concurrency waves do not cover enough distinct levels")
    requests_completed = report.get("requests_completed")
    requested_wave_count = sum(wave["requests"] for wave in waves)
    if (
        not isinstance(requests_completed, int)
        or isinstance(requests_completed, bool)
        or requests_completed <= 0
        or requests_completed != requested_wave_count
        or requests_completed > max_requests
    ):
        raise ValueError("requests_completed does not match completed concurrency waves")
    if requests_completed < min_requests:
        raise ValueError("requests_completed is below the required minimum")
    # Latency and first-content distributions are public workload evidence,
    # even when no operator-specific upper budget has been selected yet.
    latency = _distribution(report.get("latency_ms"), "latency_ms", requests_completed)
    first_content = _distribution(report.get("first_content_ms"), "first_content_ms", requests_completed)
    if any(first_content[key] > latency[key] for key in ("p50", "p95", "p99", "max")):
        raise ValueError("first-content time exceeds complete request latency")
    _positive(report.get("cold_start_ms"), "cold_start_ms")
    completion_tokens = report.get("completion_tokens")
    if (
        not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens < requests_completed
    ):
        raise ValueError("completion_tokens is missing or invalid")

    disconnect = report.get("disconnect")
    if not isinstance(disconnect, dict):
        raise ValueError("disconnect recovery evidence is missing")
    _positive_int(disconnect.get("accepted"), "disconnect accepted")
    if disconnect.get("post_recovery_request") != "pass":
        raise ValueError("disconnect recovery did not complete")
    _positive(disconnect.get("recovery_ms"), "disconnect recovery_ms")

    memory = report.get("host_memory")
    if not isinstance(memory, dict):
        raise ValueError("host memory evidence is missing")
    for field in ("warm_baseline_bytes", "observed_peak_bytes", "after_recovery_bytes"):
        value = memory.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"host memory field {field} is invalid")
    if type(memory.get("drift_bytes")) is not int:
        raise ValueError("host memory drift is missing")
    if memory["drift_bytes"] != memory["after_recovery_bytes"] - memory["warm_baseline_bytes"]:
        raise ValueError("host memory drift does not match measured endpoints")
    if memory["observed_peak_bytes"] < max(memory["warm_baseline_bytes"], memory["after_recovery_bytes"]):
        raise ValueError("host memory peak is below a measured endpoint")

    inventory = report.get("backend_inventory")
    if not isinstance(inventory, dict) or inventory.get("status") not in {"available", "partial"}:
        raise ValueError("physical backend inventory is unavailable")
    devices = _sanitize_backend_inventory({"object": "list", "devices": inventory.get("devices")})["devices"]
    available_classes = {
        entry["capability"]["device_class"].lower().replace("_", "").replace("-", "")
        for entry in devices if entry["available"] and entry["capability"] is not None
    }
    required_classes = {
        "cpu": {"cpu"}, "gpu": {"integratedgpu", "discretegpu"},
        "metal": {"integratedgpu", "discretegpu"}, "cuda": {"discretegpu"},
        "npu": {"npu"}, "intel-npu": {"npu"},
        "auto": {"cpu", "integratedgpu", "discretegpu", "npu"},
    }.get(identity["device"].lower())
    if required_classes is None or not available_classes.intersection(required_classes):
        raise ValueError("physical inventory does not support the requested device")
    if failure_report is None:
        raise ValueError("failure-injection report is required")
    if (
        not isinstance(failure_report, dict)
        or type(failure_report.get("schema_version")) is not int
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
    if _sha256_text(deployment.get("server_sha256"), "failure-injection server sha256") != server_sha:
        raise ValueError("failure-injection server identity does not match workload")
    if (
        deployment.get("backend") != identity.get("backend")
        or deployment.get("device") != identity.get("device")
    ):
        raise ValueError("failure-injection runtime identity does not match workload")
    if _sha256_text(deployment.get("model_sha256"), "failure-injection model sha256") != model_sha:
        raise ValueError("failure-injection model identity does not match workload")
    failure_status = "pass"
    checked_budgets = _check_budgets(report, budgets) if budgets is not None else None
    return {
        "schema_version": 1,
        "object": "bloom.deployment_cell_validation",
        "result": "evidence_complete",
        "stable_claim": False,
        "profile": "production_workload" if production_profile else "workload",
        "requests_completed": requests_completed,
        "budgets": checked_budgets,
        "duration_seconds": duration,
        "observed_duration_seconds": elapsed_ms / 1000,
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
            "Physical disk-full and target-host upgrade/rollback drills remain separate gates.",
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
    parser.add_argument("--min-requests", type=int, default=1)
    parser.add_argument("--budgets", type=pathlib.Path,
                        help="JSON object declaring all six service budget limits")
    parser.add_argument("--production-profile", action="store_true",
                        help="Require 24 hours, 1000 requests, pinned identity and explicit budgets")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        budgets = json.loads(args.budgets.read_text(encoding="utf-8")) if args.budgets else None
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
            min_requests=args.min_requests,
            budgets=budgets,
            production_profile=args.production_profile,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"deployment-cell evidence: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
