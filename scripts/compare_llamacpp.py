#!/usr/bin/env python3
"""Run Bloom and llama.cpp benchmark probes against the same local model.

This is intentionally tolerant by default: missing model or missing llama.cpp
binary is reported as SKIP in the JSON output so the script can run in default
CI. Use --require-model/--require-llama or the matching environment variables
for release/production gates where Bloom must run a real model and compare
against llama.cpp. The JSON report records the shared context, device/backend,
seed, sampling, repetitions, warmup and optional thread settings so a result
can be reproduced, and preserves Bloom's measured percentile fields when the
benchmark binary emits them. The llama-bench mode still measures its
token-count workload; use llama-cli/main when prompt text must match exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run(
    cmd: list[str], timeout: int, *, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    elapsed = time.time() - started
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "duration_secs": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_json_or_tail(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"raw_tail": stripped[-4000:]}


def benchmark_metrics(result: Any) -> dict[str, float | int]:
    """Extract finite benchmark metrics without changing the raw result."""
    if not isinstance(result, dict):
        return {}
    names = (
        "tokens_per_second",
        "ttft_ms",
        "tbt_ms",
        "p50_tokens_per_second",
        "p95_tokens_per_second",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p50_tbt_ms",
        "p95_tbt_ms",
        "run_samples",
    )
    metrics: dict[str, float | int] = {}
    for name in names:
        value = result.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            metrics[name] = value
        elif isinstance(value, float) and value >= 0 and math.isfinite(value):
            metrics[name] = value
    return metrics


def parse_llama_metrics(stdout: str, stderr: str) -> dict[str, float]:
    """Parse decode/prompt throughput from llama-cli or llama-bench output."""
    text = f"{stdout}\n{stderr}"
    metrics: dict[str, float] = {}

    prompt_matches = re.findall(
        r"prompt eval time\s*=.*?\(\s*([0-9]+(?:\.[0-9]+)?)\s+tokens per second\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    decode_matches = re.findall(
        r"(?<!prompt )eval time\s*=.*?\(\s*([0-9]+(?:\.[0-9]+)?)\s+tokens per second\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if prompt_matches:
        metrics["prompt_tokens_per_second"] = float(prompt_matches[-1])
    if decode_matches:
        metrics["decode_tokens_per_second"] = float(decode_matches[-1])

    # llama-bench markdown row: ... | tg64 | 12.34 ± 0.12 |
    bench_decode = re.findall(
        r"\|\s*tg\d+\s*\|\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    bench_prompt = re.findall(
        r"\|\s*pp\d+\s*\|\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if bench_decode:
        metrics["decode_tokens_per_second"] = float(bench_decode[-1])
    if bench_prompt:
        metrics["prompt_tokens_per_second"] = float(bench_prompt[-1])
    return metrics


def find_llama_binary(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("LLAMA_CPP_BIN"),
        shutil.which("llama-cli"),
        shutil.which("main"),
        shutil.which("llama-bench"),
    ]
    return next((c for c in candidates if c), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("BLOOM_MODEL_PATH", os.environ.get("MODEL_DIR", "/tmp/smoke_model")),
        help="Model directory or GGUF file. Missing path is reported as skipped.",
    )
    parser.add_argument("--bloom-bin", default="target/release/bloom_bench")
    parser.add_argument("--llama-bin", default=None)
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("BLOOM_SEED", "42")),
        help="Random seed shared by Bloom and llama.cpp CLI.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("BLOOM_TEMPERATURE", "0.7")),
        help="Sampling temperature shared by Bloom and llama.cpp CLI.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("BLOOM_TOP_P", "0.9")),
        help="Top-p sampling value shared by Bloom and llama.cpp CLI.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=int(os.environ.get("BLOOM_CONTEXT_SIZE", "2048")),
        help="Context window passed to Bloom and llama.cpp where supported.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("BLOOM_DEVICE", "cpu"),
        help="Bloom device selector (default: cpu for reproducible comparisons).",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("BLOOM_BACKEND", "candle"),
        help="Bloom backend engine (default: candle).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=int(os.environ.get("BLOOM_REPETITIONS", "1")),
        help="Measured Bloom repetitions after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("BLOOM_WARMUP", "0")),
        help="Bloom warmup runs before measurement.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional thread count shared with Bloom and passed to llama.cpp CLI/bench.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--min-decode-ratio",
        type=float,
        default=float(os.environ.get("BLOOM_MIN_LLAMA_RATIO", "0")),
        help="Require Bloom decode tok/s / llama.cpp decode tok/s to meet this ratio; 0 disables the gate.",
    )
    parser.add_argument("--build", action="store_true", help="Build bloom_bench before running.")
    parser.add_argument(
        "--require-model",
        action="store_true",
        default=os.environ.get("BLOOM_REQUIRE_MODEL", "").lower() in {"1", "true", "yes"},
        help="Fail instead of SKIP when the model path is missing.",
    )
    parser.add_argument(
        "--require-llama",
        action="store_true",
        default=os.environ.get("BLOOM_REQUIRE_LLAMA_CPP", "").lower() in {"1", "true", "yes"},
        help="Fail instead of SKIP when the llama.cpp binary is missing.",
    )
    return parser.parse_args()


def build_bloom_command(args: argparse.Namespace, model: Path) -> list[str]:
    """Build a deterministic Bloom benchmark command from the comparison config."""
    return [
        args.bloom_bin,
        "--model",
        str(model),
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--repetitions",
        str(args.repetitions),
        "--warmup",
        str(args.warmup),
        "--context-size",
        str(args.context_size),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--backend",
        args.backend,
        "--device",
        args.device,
    ]


def build_llama_command(
    args: argparse.Namespace, model: Path, llama_bin: str
) -> list[str]:
    """Build a llama.cpp command while preserving llama-bench compatibility.

    llama-bench accepts a token-count workload rather than a text prompt, so the
    prompt is only passed to llama-cli/main. Shared context and thread settings
    are applied to both command families where their CLIs support them.
    """
    binary_name = Path(llama_bin).name
    if "bench" in binary_name:
        command = [
            llama_bin,
            "-m",
            str(model),
            "-n",
            str(args.max_tokens),
        ]
    else:
        command = [
            llama_bin,
            "-m",
            str(model),
            "-p",
            args.prompt,
            "-n",
            str(args.max_tokens),
            "-c",
            str(args.context_size),
            "--seed",
            str(args.seed),
            "--temp",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
        ]
    device_layers = {"cpu": "0", "gpu": "-1"}.get(args.device.lower())
    if device_layers is not None:
        command.extend(["-ngl", device_layers])
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    return command


def benchmark_environment(args: argparse.Namespace) -> dict[str, str]:
    """Return the process environment shared by both benchmark commands."""
    environment = os.environ.copy()
    if args.threads is not None:
        # Candle/Rayon reads this before the benchmark process starts. llama.cpp
        # also receives the explicit -t flag in build_llama_command().
        environment["RAYON_NUM_THREADS"] = str(args.threads)
    return environment


def validate_args(args: argparse.Namespace) -> None:
    for name in ("max_tokens", "context_size", "repetitions"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("warmup",):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.threads is not None and args.threads <= 0:
        raise ValueError("--threads must be positive")
    if not 0.0 <= args.temperature <= 2.0 or not args.temperature == args.temperature:
        raise ValueError("--temperature must be finite and between 0 and 2")
    if not 0.0 < args.top_p <= 1.0 or not args.top_p == args.top_p:
        raise ValueError("--top-p must be finite and between 0 and 1")
    if args.seed < 0:
        raise ValueError("--seed cannot be negative")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model = Path(args.model)
    report: dict[str, Any] = {
        "model": str(model),
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "comparison_config": {
            "context_size": args.context_size,
            "device": args.device,
            "backend": args.backend,
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "threads": args.threads,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "bloom": {"status": "not_run"},
        "llama_cpp": {"status": "not_run"},
    }

    if not model.exists():
        report["status"] = "skipped"
        report["reason"] = f"model not found at {model}"
        print(json.dumps(report, indent=2))
        return 1 if args.require_model else 0

    if args.build or not Path(args.bloom_bin).exists():
        subprocess.run(["cargo", "build", "--release", "--bin", "bloom_bench"], check=True)

    bloom_cmd = build_bloom_command(args, model)
    environment = benchmark_environment(args)
    bloom_run = run(bloom_cmd, args.timeout, env=environment)
    bloom_result = parse_json_or_tail(bloom_run["stdout"])
    report["bloom"] = {
        "status": "ok" if bloom_run["exit_code"] == 0 else "failed",
        "command": bloom_cmd,
        "duration_secs": bloom_run["duration_secs"],
        "result": bloom_result,
        "metrics": benchmark_metrics(bloom_result),
        "stderr_tail": bloom_run["stderr"][-4000:],
    }

    llama_bin = find_llama_binary(args.llama_bin)
    if not llama_bin:
        report["llama_cpp"] = {
            "status": "skipped",
            "reason": "llama.cpp binary not found; set LLAMA_CPP_BIN or --llama-bin",
        }
        print(json.dumps(report, indent=2))
        return 1 if args.require_llama else 0

    llama_cmd = build_llama_command(args, model, llama_bin)
    llama_run = run(llama_cmd, args.timeout, env=environment)
    llama_metrics = parse_llama_metrics(llama_run["stdout"], llama_run["stderr"])
    report["llama_cpp"] = {
        "status": "ok" if llama_run["exit_code"] == 0 else "failed",
        "command": llama_cmd,
        "duration_secs": llama_run["duration_secs"],
        "metrics": llama_metrics,
        "stdout_tail": llama_run["stdout"][-4000:],
        "stderr_tail": llama_run["stderr"][-4000:],
    }

    bloom_decode = (
        bloom_result.get("tokens_per_second")
        if isinstance(bloom_result, dict)
        else None
    )
    llama_decode = llama_metrics.get("decode_tokens_per_second")
    if isinstance(bloom_decode, (int, float)) and llama_decode and llama_decode > 0:
        decode_ratio = float(bloom_decode) / llama_decode
        report["comparison"] = {
            "bloom_decode_tokens_per_second": float(bloom_decode),
            "llama_cpp_decode_tokens_per_second": llama_decode,
            "decode_throughput_ratio": decode_ratio,
            "minimum_required_ratio": args.min_decode_ratio,
            "passed": args.min_decode_ratio <= 0 or decode_ratio >= args.min_decode_ratio,
        }
    else:
        report["comparison"] = {
            "status": "unavailable",
            "reason": "could not parse decode tokens/second from both runtimes",
            "minimum_required_ratio": args.min_decode_ratio,
        }

    print(json.dumps(report, indent=2))
    if report["bloom"]["status"] != "ok":
        return 1
    if args.require_llama and report["llama_cpp"]["status"] != "ok":
        return 1
    if args.min_decode_ratio > 0:
        comparison = report["comparison"]
        if comparison.get("passed") is not True:
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as exc:
        print(f"FAIL: command timed out: {exc.cmd}", file=sys.stderr)
        raise SystemExit(1)
