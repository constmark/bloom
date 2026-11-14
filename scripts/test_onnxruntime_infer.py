#!/usr/bin/env python3
"""ONNX bridge type/provider regressions; requires the pinned CPU test extras."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import onnxruntime_infer as bridge


def meta(dtype, shape):
    return types.SimpleNamespace(type=f"tensor({dtype})", shape=shape)


class NumericBoundaryTests(unittest.TestCase):
    def test_integer_and_boolean_types_preserve_exact_values(self):
        for dtype, values, expected in (
            ("int64", [-(2**63), 2**63 - 1], np.int64),
            ("uint64", [0, 2**64 - 1], np.uint64),
            ("uint32", [0, 2**32 - 1], np.uint32),
            ("uint16", [0, 65535], np.uint16),
            ("uint8", [0, 255], np.uint8),
            ("int8", [-128, 127], np.int8),
            ("bool", [0, 1], np.bool_),
        ):
            with self.subTest(dtype=dtype):
                result = bridge._coerce(values, meta(dtype, [1, 2]))
                self.assertEqual(result.dtype, expected)
                self.assertEqual(result.tolist(), [values])

    def test_fractional_overflow_nonfinite_and_boolean_numbers_are_rejected(self):
        for dtype, values in (
            ("int32", [1.5]), ("uint8", [-1]), ("int8", [128]),
            ("int64", [2**63]), ("uint64", [2**64]), ("bool", [2]),
            ("float16", [65536]), ("float", [1e100]), ("double", [10**400]),
            ("float", [float("inf")]), ("float", [float("nan")]),
            ("float", [True]), ("float", ["1"]), ("float", [[1]]),
        ):
            with self.subTest(dtype=dtype, values=values), self.assertRaises(ValueError):
                bridge._coerce(values, meta(dtype, [1]))

    def test_float_and_text_conversion_and_scalar_shapes(self):
        self.assertEqual(bridge._coerce("42", meta("int64", [])).item(), 42)
        np.testing.assert_array_equal(bridge._coerce([1.25, -2.5], meta("float", [1, "width"])), [[1.25, -2.5]])
        self.assertEqual(bridge._coerce("你好", meta("string", [])).item(), "你好")
        with self.assertRaises(ValueError):
            bridge._coerce([1, 2], meta("string", [1]))

    def test_unsupported_types_and_incompatible_shapes_fail_before_execution(self):
        for declaration in (meta("complex64", [2]), meta("bfloat16", [2]),
                            meta("float", [3]), meta("float", [2, "width"]),
                            meta("float", []), meta("float", [0]),
                            meta("float", None)):
            with self.subTest(meta=declaration), self.assertRaises(ValueError):
                bridge._coerce([1, 2], declaration)
        with mock.patch.object(bridge, "MAX_VALUES", 1), self.assertRaises(ValueError):
            bridge._coerce([1, 2], meta("float", [2]))

    def test_outputs_are_finite_bounded_and_utf8_decoded(self):
        self.assertEqual(bridge._jsonable(np.array([True, False])), [1, 0])
        self.assertEqual(bridge._jsonable(np.array(["你好".encode()], dtype=object)), "你好")
        self.assertEqual(bridge._jsonable(np.array([[1, 2]], dtype=np.float32)), [1, 2])
        for value in (np.array([float("nan")]), np.array([float("inf")]),
                      np.array([1e100]), np.array([complex(1, 2)]),
                      np.array([1], dtype=object), np.array([b"\xff"])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bridge._jsonable(value)
        with mock.patch.object(bridge, "MAX_VALUES", 1), self.assertRaises(ValueError):
            bridge._jsonable(np.array([1, 2]))


class ProviderTests(unittest.TestCase):
    def runtime(self, active):
        options = mock.Mock()
        runtime = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
            SessionOptions=lambda: options,
            InferenceSession=mock.Mock(return_value=types.SimpleNamespace(
                get_providers=lambda: active, disable_fallback=mock.Mock())),
        )
        return runtime, options

    def test_unknown_provider_never_selects_cpu(self):
        for provider in ("", "CUDDA", "OtherExecutionProvider"):
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                bridge._provider_configuration(provider, "CPU")
        self.assertEqual(bridge._provider_configuration("cpu", "CPU"), ("CPUExecutionProvider", {}))

    def test_session_internal_fallback_is_detected(self):
        runtime, options = self.runtime(["CPUExecutionProvider"])
        with self.assertRaisesRegex(RuntimeError, "did not activate"):
            bridge._session("model.onnx", "CUDA", "CUDA", runtime)
        options.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")

    def test_requested_provider_must_be_available_and_active(self):
        runtime, _ = self.runtime(["CUDAExecutionProvider"])
        _, provider = bridge._session("model.onnx", "CUDA", "CUDA", runtime)
        self.assertEqual(provider, "CUDAExecutionProvider")
        runtime.InferenceSession.return_value.disable_fallback.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            bridge._session("model.onnx", "NPU", "NPU", runtime)
        self.assertEqual(runtime.InferenceSession.call_count, 1)


class RealRuntimeTests(unittest.TestCase):
    def test_identity_graphs_cross_the_json_process_boundary(self):
        import onnx
        from onnx import TensorProto, helper

        for dtype, shape, prompt, expected in (
            (TensorProto.INT64, [], "42", [42]),
            (TensorProto.UINT32, [1, 2], "0 4294967295", [0, 4294967295]),
            (TensorProto.BOOL, [2], "0 1", [0, 1]),
            (TensorProto.STRING, [], "你好 Bloom", "你好 Bloom"),
        ):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "identity.onnx"
                graph = helper.make_graph(
                    [helper.make_node("Identity", ["input"], ["output"])], "identity",
                    [helper.make_tensor_value_info("input", dtype, shape)],
                    [helper.make_tensor_value_info("output", dtype, shape)],
                )
                onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9), path)
                command = [sys.executable, str(Path(bridge.__file__)), "--model-path", str(path), "--device", "CPU"]
                result = subprocess.run(command, input=json.dumps({"Text": {"prompt": prompt}}),
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                output = json.loads(result.stdout)
                self.assertEqual(output["text" if dtype == TensorProto.STRING else "logits"], expected)
                if dtype == TensorProto.UINT32:
                    invalid = subprocess.run(command, input=json.dumps({"Text": {"prompt": "0 -1"}}),
                                             capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(invalid.returncode, 0)
                    self.assertEqual(invalid.stdout, "")
                    self.assertIn("dtype bounds", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
