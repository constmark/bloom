#!/usr/bin/env python3
"""Regression tests for the reproducible Bloom/llama.cpp comparison harness."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import compare_llamacpp as compare


class CompareLlamaCppTests(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "bloom_bin": "target/release/bloom_bench",
            "prompt": "hello",
            "max_tokens": 16,
            "seed": 42,
            "temperature": 0.7,
            "top_p": 0.9,
            "context_size": 1024,
            "device": "cpu",
            "backend": "candle",
            "repetitions": 3,
            "warmup": 1,
            "threads": 4,
            "timeout": 180,
            "min_decode_ratio": 0.0,
            "model": "model.gguf",
            "build": False,
            "llama_bin": None,
            "require_model": False,
            "require_llama": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_bloom_command_carries_shared_workload_settings(self):
        args = self.make_args()
        command = compare.build_bloom_command(args, Path("model.gguf"))
        self.assertEqual(
            command,
            [
                "target/release/bloom_bench",
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--max-tokens",
                "16",
                "--repetitions",
                "3",
                "--warmup",
                "1",
                "--context-size",
                "1024",
                "--seed",
                "42",
                "--temperature",
                "0.7",
                "--top-p",
                "0.9",
                "--backend",
                "candle",
                "--device",
                "cpu",
            ],
        )

    def test_llama_cli_receives_context_sampling_and_threads(self):
        args = self.make_args()
        command = compare.build_llama_command(args, Path("model.gguf"), "/opt/llama-cli")
        self.assertEqual(
            command,
            [
                "/opt/llama-cli",
                "-m",
                "model.gguf",
                "-p",
                "hello",
                "-n",
                "16",
                "-c",
                "1024",
                "--seed",
                "42",
                "--temp",
                "0.7",
                "--top-p",
                "0.9",
                "-ngl",
                "0",
                "-t",
                "4",
            ],
        )

    def test_llama_bench_keeps_token_workload_shape(self):
        args = self.make_args()
        command = compare.build_llama_command(args, Path("model.gguf"), "/opt/llama-bench")
        self.assertEqual(
            command,
            [
                "/opt/llama-bench",
                "-m",
                "model.gguf",
                "-n",
                "16",
                "-ngl",
                "0",
                "-t",
                "4",
            ],
        )

    def test_benchmark_environment_shares_rayon_thread_count(self):
        args = self.make_args(threads=6)
        environment = compare.benchmark_environment(args)
        self.assertEqual(environment["RAYON_NUM_THREADS"], "6")

    def test_invalid_workload_values_fail_closed(self):
        with self.assertRaises(ValueError):
            compare.validate_args(self.make_args(context_size=0))
        with self.assertRaises(ValueError):
            compare.validate_args(self.make_args(top_p=0.0))
        for changes in ({"timeout": 0}, {"min_decode_ratio": -1},
                        {"min_decode_ratio": float("nan")},
                        {"min_decode_ratio": float("inf")}, {"seed": 2**64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                compare.validate_args(self.make_args(**changes))

    def test_cli_metrics_include_per_token_latency_without_bench_rows(self):
        text = """
llama_perf_context_print: prompt eval time = 100 ms / 20 tokens (5.00 ms per token, 200.00 tokens per second)
llama_perf_context_print:        eval time = 500 ms / 40 runs (80.00 ms per token, 20.00 tokens per second)
"""
        self.assertEqual(compare.parse_llama_metrics("", text), {
            "prompt_tokens_per_second": 200.0,
            "decode_tokens_per_second": 20.0,
        })
        self.assertEqual(compare.parse_llama_metrics("prompt eval time = 100 ms (200.00 tokens per second)", ""),
                         {"prompt_tokens_per_second": 200.0})

    def test_throughput_is_parsed_as_a_complete_nonnegative_finite_number(self):
        for value in ("-200.00", "1e999", "NaN", "Infinity", "garbage200"):
            with self.subTest(value=value):
                self.assertEqual(compare.parse_llama_metrics(f"eval time = 1ms (5.00 ms per token, {value} tokens per second)", ""), {})
                self.assertEqual(compare.parse_llama_metrics(f"| tg64 | {value} ± 1.0 |", ""), {})
        for output in ("eval time = 1ms (5.00 ms per token, 1e20 tokens per second)", "| tg64 | 1e20 ± 1.0 |"):
            self.assertEqual(compare.parse_llama_metrics(output, ""), {"decode_tokens_per_second": 1e20})

    def run_main(self, args=None, llama=None, results=None):
        output = io.StringIO()
        with mock.patch.object(compare, "parse_args", return_value=args or self.make_args()), \
             mock.patch.object(compare.Path, "exists", return_value=True), \
             mock.patch.object(compare, "find_llama_binary", return_value=llama), \
             mock.patch.object(compare, "run", side_effect=results), contextlib.redirect_stdout(output):
            status = compare.main()
        return status, json.loads(output.getvalue())

    def test_missing_reference_cannot_hide_failed_bloom_or_required_ratio(self):
        failed = {"exit_code": 1, "duration_secs": 1, "stdout": "", "stderr": "failed"}
        self.assertEqual(self.run_main(results=[failed])[0], 1)
        successful = {**failed, "exit_code": 0, "stdout": '{"tokens_per_second": 10}'}
        self.assertEqual(self.run_main(results=[successful])[0], 0)
        self.assertEqual(self.run_main(args=self.make_args(min_decode_ratio=0.1), results=[successful])[0], 1)

    def test_failed_reference_and_nonfinite_metrics_cannot_pass_gate(self):
        bloom = {"exit_code": 0, "duration_secs": 1, "stdout": '{"tokens_per_second": 10}', "stderr": ""}
        llama = {"exit_code": 1, "duration_secs": 1, "stdout": "| tg64 | 10.0 |", "stderr": ""}
        status, report = self.run_main(llama="llama-bench", results=[bloom, llama])
        self.assertEqual(status, 1)
        self.assertEqual(report["comparison"]["status"], "unavailable")
        for metric in ("NaN", "Infinity", "1e999", "true", "-1"):
            with self.subTest(metric=metric):
                status, report = self.run_main(args=self.make_args(min_decode_ratio=0.1), llama="llama-bench",
                    results=[{**bloom, "stdout": '{"tokens_per_second":' + metric + '}'}, {**llama, "exit_code": 0}])
                self.assertEqual(status, 1)
                self.assertEqual(report["comparison"]["status"], "unavailable")
                self.assertIsInstance(json.dumps(report, allow_nan=False), str)

    def test_parse_metrics_handles_cli_and_bench_output(self):
        text = """
        prompt eval time = 100 ms / 20 tokens (200.00 tokens per second)
        eval time = 500 ms / 40 runs (80.00 ms per token, 20.00 tokens per second)
        | pp512 | 300.00 ± 2.0 |
        | tg64 | 42.00 ± 1.0 |
        """
        metrics = compare.parse_llama_metrics(text, "")
        self.assertEqual(metrics["prompt_tokens_per_second"], 300.0)
        self.assertEqual(metrics["decode_tokens_per_second"], 42.0)

    def test_benchmark_metrics_keep_percentiles_and_drop_invalid_values(self):
        metrics = compare.benchmark_metrics(
            {
                "tokens_per_second": 4.0,
                "p95_ttft_ms": 12.5,
                "run_samples": 3,
                "p50_tbt_ms": -1.0,
                "p95_tbt_ms": float("inf"),
                "bad": "ignored",
            }
        )
        self.assertEqual(metrics, {
            "tokens_per_second": 4.0,
            "p95_ttft_ms": 12.5,
            "run_samples": 3,
        })


if __name__ == "__main__":
    unittest.main()
