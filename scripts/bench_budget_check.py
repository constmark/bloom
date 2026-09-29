#!/usr/bin/env python3
"""Compare a `bloom_bench` JSON result against the performance budgets in
`docs/performance_budgets.md`.

It maps the benchmark's `hardware` section to one of four budget tiers
(Apple Silicon, x86 CPU, NVIDIA RTX, or Intel NPU) and reports PASS,
WARN, or FAIL for TTFT, TBT, and peak memory. TTFT or TBT more than 5%
over its threshold is a failure.

Usage:
    cargo run --release --bin bloom_bench -- --model /path/to/model.gguf \\
        --max-tokens 64 > bench.json
    ./scripts/bench_budget_check.py bench.json

    # Or via stdin:
    cat bench.json | ./scripts/bench_budget_check.py

Exit codes:
    0 = all metrics within budget (PASS)
    1 = at least one metric WARNed (within 5% over budget)
    2 = at least one metric FAILed (more than 5% over budget)
    3 = valid metrics, but hardware could not be classified
    4 = missing or invalid benchmark data (always a CI failure)

This script depends only on the Python standard library so it can run
in CI without `pip install`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


# --- Budget table (mirrors docs/performance_budgets.md) -------------------
# Each entry: (ttft_ms_budget, tbt_ms_budget, peak_memory_bytes_budget)
# `None` means "no budget for this metric on this tier".
BUDGETS: dict[str, dict[str, float | int | None]] = {
    # MacBook Air/Pro 16GB; GGUF Q4_K_M
    "apple_silicon": {
        "ttft_ms": 150.0,
        "tbt_ms": 30.0,
        "peak_memory_bytes": 6_500 * 1024 * 1024,  # 6.5 GB
    },
    # RTX 3060/4060/4090 8GB; AWQ INT4 / FP16
    "nvidia_rtx": {
        "ttft_ms": 50.0,
        "tbt_ms": 15.0,
        "peak_memory_bytes": 8_000 * 1024 * 1024,  # 8.0 GB
    },
    # Intel/AMD 16GB; GGUF Q4_0
    "x86_cpu": {
        "ttft_ms": 1500.0,
        "tbt_ms": 120.0,
        "peak_memory_bytes": 5_500 * 1024 * 1024,  # 5.5 GB
    },
    # Intel Core Ultra NPU; OpenVINO IR INT4
    "intel_npu": {
        "ttft_ms": 200.0,
        "tbt_ms": 40.0,
        "peak_memory_bytes": 6_000 * 1024 * 1024,  # 6.0 GB
    },
}

# Regression tolerance from the regression-gate section of performance_budgets.md.
REGRESSION_TOLERANCE = 0.05  # 5%


def classify_hardware(hardware: dict[str, Any] | None) -> str | None:
    """Map a benchmark's `hardware` section to a budget tier key."""
    if not hardware:
        return None
    device = str(hardware.get("device", "")).lower()
    backend = str(hardware.get("backend", "")).lower()
    os_ = str(hardware.get("os", "")).lower()
    arch = str(hardware.get("arch", "")).lower()

    # Runtime selection takes precedence over a backend's possible devices.
    # In particular, OpenVINO on CPU is not evidence of NPU execution, and
    # an ARM CPU must never inherit x86 latency thresholds.
    if device == "cpu":
        if arch in {"x86", "x86_64", "amd64", "i686"}:
            return "x86_cpu"
        return None

    # Apple Silicon: macOS + metal/gpu
    if "macos" in os_ or "darwin" in os_:
        if arch in {"aarch64", "arm64"} and (
            "metal" in backend or "gpu" in device or "metal" in device
        ):
            return "apple_silicon"

    # NVIDIA RTX: linux + cuda
    if "cuda" in backend or "nvidia" in device or "rtx" in device:
        return "nvidia_rtx"

    # Intel NPU: explicit npu device or openvino backend
    if "npu" in device:
        return "intel_npu"

    return None


def fmt_bytes(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    gib = n / (1024 * 1024 * 1024)
    mib = n / (1024 * 1024)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{mib:.1f} MiB"


def fmt_ms(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    return f"{n:.1f} ms"


def check_metric(
    name: str,
    actual: float | int | None,
    budget: float | int | None,
    is_lower_better: bool = True,
) -> tuple[str, str]:
    """Compare actual vs budget. Returns (status, message).

    Statuses:
        PASS  — actual is within budget
        WARN  — actual exceeds budget by ≤ tolerance (5%)
        FAIL  — actual exceeds budget by > tolerance (5%)
        Invalid or missing measurements raise ValueError.
    """
    validate_number(name, actual)
    validate_number(f"{name} budget", budget, positive=True)

    if is_lower_better:
        ratio = actual / budget if budget > 0 else float("inf")
    else:
        ratio = budget / actual if actual > 0 else float("inf")

    if ratio <= 1.0:
        return "PASS", f"{name}: {fmt_value(name, actual)} <= {fmt_value(name, budget)} budget"
    over = (ratio - 1.0) * 100
    # Compare values directly: ratio subtraction rounds exact 5% boundaries
    # just above 5 for common integer budgets.
    within_tolerance = (
        actual <= budget * (1 + REGRESSION_TOLERANCE)
        if is_lower_better else budget <= actual * (1 + REGRESSION_TOLERANCE)
    )
    if within_tolerance:
        return (
            "WARN",
            f"{name}: {fmt_value(name, actual)} > {fmt_value(name, budget)} budget "
            f"by {over:.1f}% (within {REGRESSION_TOLERANCE*100:.0f}% tolerance)",
        )
    return (
        "FAIL",
        f"{name}: {fmt_value(name, actual)} > {fmt_value(name, budget)} budget "
        f"by {over:.1f}% (exceeds {REGRESSION_TOLERANCE*100:.0f}% tolerance)",
    )


def fmt_value(name: str, value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if name.endswith("_bytes"):
        return fmt_bytes(value)
    if name.endswith("_ms"):
        return fmt_ms(value)
    return str(value)


def validate_number(name: str, value: Any, *, positive: bool = False) -> None:
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and (value > 0 if positive else value >= 0)
        )
    except OverflowError:
        valid = False
    if not valid:
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {bound} number")


def extract_metrics(bench: dict[str, Any]) -> dict[str, float | int]:
    """Pull TTFT / TBT / Peak Memory out of a bloom_bench JSON object."""
    timing = bench.get("timing_breakdown")
    if timing is not None and not isinstance(timing, dict):
        raise ValueError("timing_breakdown must be an object")
    timing = timing or {}
    metrics = {}
    for name, average in (("ttft_ms", "avg_ttft_ms"), ("tbt_ms", "avg_tbt_ms")):
        value = bench.get(name)
        if value is None:
            value = bench.get(average)
        if value is None:
            value = timing.get(average)
        validate_number(name, value)
        metrics[name] = value
    peak = bench.get("peak_memory_bytes")
    validate_number("peak_memory_bytes", peak, positive=True)
    if not isinstance(peak, int):
        raise ValueError("peak_memory_bytes must be an integer byte count")
    metrics["peak_memory_bytes"] = peak
    return metrics


def validate_metadata(bench: Any) -> None:
    if not isinstance(bench, dict):
        raise ValueError("benchmark must be an object")
    hardware = bench.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("hardware must be an object")
    for name in ("device", "backend", "os"):
        if not isinstance(hardware.get(name), str) or not hardware[name].strip():
            raise ValueError(f"hardware.{name} must be a non-empty string")
    if "arch" in hardware and (
        not isinstance(hardware["arch"], str) or not hardware["arch"].strip()
    ):
        raise ValueError("hardware.arch must be a non-empty string")
    cache = bench.get("cache_metrics", {})
    if not isinstance(cache, dict):
        raise ValueError("cache_metrics must be an object")
    if not isinstance(cache.get("enabled", False), bool):
        raise ValueError("cache_metrics.enabled must be a boolean")
    for name in ("hits", "misses", "evictions", "reuses"):
        value = cache.get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"cache_metrics.{name} must be a non-negative integer")


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def main(argv: list[str]) -> int:
    try:
        if len(argv) < 2 or argv[1] == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(argv[1]).read_text(encoding="utf-8")
        bench = json.loads(raw, parse_constant=reject_constant)
        validate_metadata(bench)
        metrics = extract_metrics(bench)
    except (OSError, ValueError, RecursionError) as error:
        print(f"error: invalid benchmark data: {error}", file=sys.stderr)
        return 4

    tier = classify_hardware(bench.get("hardware"))
    if tier is None:
        hw = bench.get("hardware")
        print(
            f"error: could not classify hardware tier for {hw!r}. "
            f"Expected device/backend/os matching one of: "
            f"apple_silicon, nvidia_rtx, x86_cpu, intel_npu",
            file=sys.stderr,
        )
        return 3

    budget = BUDGETS[tier]

    print(f"Hardware tier: {tier}")
    print(f"Budget: TTFT<={fmt_ms(budget['ttft_ms'])}, "
          f"TBT<={fmt_ms(budget['tbt_ms'])}, "
          f"Peak<={fmt_bytes(budget['peak_memory_bytes'])}")
    print()

    statuses: list[tuple[str, str, str]] = []
    for metric_name in ("ttft_ms", "tbt_ms", "peak_memory_bytes"):
        status, msg = check_metric(
            metric_name,
            metrics.get(metric_name),
            budget.get(metric_name),
            is_lower_better=True,
        )
        statuses.append((status, metric_name, msg))
        print(f"  [{status}] {msg}")

    # Cache metrics sanity check (informational, not gating).
    cache = bench.get("cache_metrics") or {}
    cache_enabled = cache.get("enabled", False)
    print()
    if cache_enabled:
        hits = cache.get("hits", 0)
        misses = cache.get("misses", 0)
        evictions = cache.get("evictions", 0)
        reuses = cache.get("reuses", 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        print(
            f"Cache: enabled, hits={hits} misses={misses} "
            f"reuses={reuses} evictions={evictions} hit_rate={hit_rate:.1f}%"
        )
    else:
        print("Cache: not enabled (standalone path — no scheduler/paged-cache)")

    # Aggregate exit code.
    has_fail = any(s == "FAIL" for s, _, _ in statuses)
    has_warn = any(s == "WARN" for s, _, _ in statuses)
    print()
    if has_fail:
        print("Result: FAIL (one or more metrics exceeded budget beyond tolerance)")
        return 2
    if has_warn:
        print("Result: WARN (one or more metrics within tolerance but over budget)")
        return 1
    print("Result: PASS (all metrics within budget)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
