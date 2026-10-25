#!/usr/bin/env python3
"""Regression tests for the reproducible Bloom/llama.cpp comparison harness."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

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
