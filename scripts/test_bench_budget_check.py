#!/usr/bin/env python3
"""Regression gates for benchmark evidence and budget exit-code semantics."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

import bench_budget_check as budget


SCRIPT = Path(__file__).with_name("bench_budget_check.py")
VALID = {
    "hardware": {"device": "cpu", "backend": "candle", "os": "linux", "arch": "x86_64"},
    "ttft_ms": 1500,
    "tbt_ms": 120,
    "peak_memory_bytes": 1024,
}


class BenchmarkBudgetTests(unittest.TestCase):
    def run_checker(self, value):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "-"], input=json.dumps(value),
            text=True, capture_output=True, timeout=10, check=False,
        )

    def test_pass_warn_and_fail_boundaries(self):
        for value, expected in ((1500, 0), (1575, 1), (1575.01, 2)):
            with self.subTest(value=value):
                benchmark = copy.deepcopy(VALID)
                benchmark["ttft_ms"] = value
                result = self.run_checker(benchmark)
                self.assertEqual(result.returncode, expected, result.stderr + result.stdout)

    def test_missing_or_invalid_metrics_never_pass_or_skip(self):
        for metric in ("ttft_ms", "tbt_ms", "peak_memory_bytes"):
            for value in (None, -1, True, "1", {}, [], float("nan"), float("inf"), 10**400):
                with self.subTest(metric=metric, value=value):
                    benchmark = copy.deepcopy(VALID)
                    benchmark[metric] = value
                    result = self.run_checker(benchmark)
                    self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                    self.assertNotIn("Result: PASS", result.stdout)
                    self.assertNotIn("Traceback", result.stderr)
        for value in (0, 1.5):
            benchmark = dict(VALID, peak_memory_bytes=value)
            self.assertEqual(self.run_checker(benchmark).returncode, 4)

    def test_estimates_do_not_replace_missing_peak_measurement(self):
        benchmark = dict(VALID, memory_breakdown={"total_bytes": 1024})
        del benchmark["peak_memory_bytes"]
        self.assertEqual(self.run_checker(benchmark).returncode, 4)

    def test_top_level_averages_take_precedence_over_nested_fallback(self):
        benchmark = dict(VALID, ttft_ms=None, tbt_ms=None,
                         avg_ttft_ms=12, avg_tbt_ms=3,
                         timing_breakdown={"avg_ttft_ms": 9999, "avg_tbt_ms": 9999})
        self.assertEqual(budget.extract_metrics(benchmark)["ttft_ms"], 12)
        self.assertEqual(budget.extract_metrics(benchmark)["tbt_ms"], 3)
        del benchmark["avg_ttft_ms"]
        self.assertEqual(budget.extract_metrics(benchmark)["ttft_ms"], 9999)

    def test_malformed_documents_metadata_and_cache_are_invalid(self):
        invalid = [None, [], 1, "text", dict(VALID, hardware=[]),
                   dict(VALID, timing_breakdown=[]), dict(VALID, cache_metrics=[]),
                   dict(VALID, cache_metrics={"enabled": "true"}),
                   dict(VALID, cache_metrics={"hits": "oops"}),
                   dict(VALID, cache_metrics={"misses": -1})]
        for value in invalid:
            with self.subTest(value=value):
                result = self.run_checker(value)
                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_invalid_json_and_unreadable_path_are_invalid_data(self):
        for raw in ("{", '{"ttft_ms": NaN}', '{"ttft_ms": Infinity}'):
            result = subprocess.run([sys.executable, str(SCRIPT)], input=raw,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 4)
        result = subprocess.run([sys.executable, str(SCRIPT), str(SCRIPT.parent)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 4)
        self.assertNotIn("Traceback", result.stderr)

    def test_only_valid_unclassified_measurements_may_skip(self):
        benchmark = copy.deepcopy(VALID)
        benchmark["hardware"]["arch"] = "aarch64"
        self.assertEqual(self.run_checker(benchmark).returncode, 3)
        benchmark["ttft_ms"] = -1
        self.assertEqual(self.run_checker(benchmark).returncode, 4)

    def test_hardware_selection_does_not_invent_device_or_architecture(self):
        for os_name in ("linux", "macos", "windows"):
            for backend in ("candle", "openvino", "cuda"):
                hardware = {"device": "cpu", "backend": backend, "os": os_name}
                self.assertIsNone(budget.classify_hardware(hardware))
                self.assertIsNone(budget.classify_hardware(dict(hardware, arch="aarch64")))
                self.assertEqual(budget.classify_hardware(dict(hardware, arch="x86_64")), "x86_cpu")
        self.assertEqual(budget.classify_hardware({"device": "gpu", "os": "macos", "arch": "aarch64"}), "apple_silicon")
        self.assertIsNone(budget.classify_hardware({"device": "gpu", "os": "linux", "backend": "openvino"}))
        self.assertEqual(budget.classify_hardware({"device": "npu", "backend": "openvino"}), "intel_npu")


if __name__ == "__main__":
    unittest.main()
