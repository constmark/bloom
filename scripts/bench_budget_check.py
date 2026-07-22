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
    3 = could not classify hardware / missing fields / invalid JSON

This script depends only on the Python standard library so it can run
in CI without `pip install`.
"""

from __future__ import annotations

import json
import re
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

    # Apple Silicon: macOS + metal/gpu
    if "macos" in os_ or "darwin" in os_:
        if "metal" in backend or "gpu" in device or "metal" in device:
            return "apple_silicon"
        elif "cpu" in device:
            return "x86_cpu"

    # NVIDIA RTX: linux + cuda
    if "cuda" in backend or "nvidia" in device or "rtx" in device:
        return "nvidia_rtx"

    # Intel NPU: explicit npu device or openvino backend
    if "npu" in device or "openvino" in backend or "intel-npu" in device:
        return "intel_npu"

    # x86 CPU fallback: linux + cpu
    if "linux" in os_ and ("cpu" in device or "candle-cpu" in backend):
        return "x86_cpu"

    # Windows NPU
    if "windows" in os_ and ("npu" in device or "openvino" in backend):
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
        SKIP  — actual or budget is missing
    """
    if actual is None or budget is None:
        return "SKIP", f"{name}: no data (actual={actual}, budget={budget})"
    if not isinstance(actual, (int, float)) or not isinstance(budget, (int, float)):
        return "SKIP", f"{name}: non-numeric (actual={actual!r}, budget={budget!r})"

    if is_lower_better:
        ratio = actual / budget if budget > 0 else float("inf")
    else:
        ratio = budget / actual if actual > 0 else float("inf")

    if ratio <= 1.0:
        return "PASS", f"{name}: {fmt_value(name, actual)} <= {fmt_value(name, budget)} budget"
    over = (ratio - 1.0) * 100
    if over <= REGRESSION_TOLERANCE * 100:
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


def extract_metrics(bench: dict[str, Any]) -> dict[str, float | int | None]:
    """Pull TTFT / TBT / Peak Memory out of a bloom_bench JSON object."""
    ttft = bench.get("ttft_ms")
    if ttft is None:
        ttft = bench.get("avg_ttft_ms")
        if isinstance(ttft, dict):
            ttft = ttft.get("avg_ttft_ms")
        elif bench.get("timing_breakdown"):
            ttft = bench["timing_breakdown"].get("avg_ttft_ms")

    tbt = bench.get("tbt_ms")
    if tbt is None:
        tbt = bench.get("avg_tbt_ms")
        if isinstance(tbt, dict):
            tbt = tbt.get("avg_tbt_ms")
        elif bench.get("timing_breakdown"):
            tbt = bench["timing_breakdown"].get("avg_tbt_ms")

    peak = bench.get("peak_memory_bytes")
    if peak is None and bench.get("memory_breakdown"):
        peak = bench["memory_breakdown"].get("total_bytes")

    # Coerce to numeric or None.
    def _num(x: Any) -> float | int | None:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return x
        return None

    return {
        "ttft_ms": _num(ttft),
        "tbt_ms": _num(tbt),
        "peak_memory_bytes": _num(peak),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] == "-":
        raw = sys.stdin.read()
    else:
        path = Path(argv[1])
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 3
        raw = path.read_text(encoding="utf-8")

    try:
        bench = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON: {e}", file=sys.stderr)
        return 3

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
    metrics = extract_metrics(bench)

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
