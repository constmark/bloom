#!/usr/bin/env python3
"""CPU-only TensorRT bridge contracts; CUDA/TensorRT are simulated, never imported.

The standard-library suite always runs. Optional NumPy integration checks run
when NumPy is installed; reproduce them with requirements/tensor-runtime-test.txt.
These tests prove adapter API behavior, not GPU execution or model quality.
"""
from __future__ import annotations

import copy
import ctypes
import enum
import math
import pathlib
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tensorrt_infer as bridge  # noqa: E402

try:
    import numpy as real_numpy
except ImportError:
    real_numpy = None


class DType:
    def __init__(self, name):
        self.name = name
        self.kind = {"float32": "f", "float16": "f", "float64": "f", "int8": "i",
                     "uint8": "u", "int32": "i", "int64": "i", "bool": "b"}[name]


class Array:
    """Contiguous host array used only to exercise the CUDA call boundary."""
    def __init__(self, data, dtype="float32", shape=None):
        self.dtype = dtype.name if isinstance(dtype, DType) else dtype
        element = {"float32": ctypes.c_float, "float64": ctypes.c_double,
                   "int32": ctypes.c_int32, "int64": ctypes.c_int64,
                   "int8": ctypes.c_int8, "uint8": ctypes.c_uint8,
                   "bool": ctypes.c_bool}[self.dtype]
        self.storage = (element * len(data))(*data)
        self.shape = tuple(shape) if shape is not None else (len(data),)
        self.nbytes = ctypes.sizeof(self.storage)
        self.ctypes = types.SimpleNamespace(data=ctypes.addressof(self.storage))

    def reshape(self, shape):
        if shape != -1:
            self.shape = tuple(shape)
        return self

    def tolist(self):
        return list(self.storage)


class StubNumpy:
    float32 = "float32"

    @staticmethod
    def dtype(name):
        return name if isinstance(name, DType) else DType(name)

    @staticmethod
    def iinfo(dtype):
        low, high = {"int8": (-128, 127), "uint8": (0, 255),
                     "int32": (-2**31, 2**31 - 1), "int64": (-2**63, 2**63 - 1)}[dtype.name]
        return types.SimpleNamespace(min=low, max=high)

    @staticmethod
    def finfo(dtype):
        name = dtype.name if isinstance(dtype, DType) else dtype
        return types.SimpleNamespace(max={"float16": 65504.0,
            "float32": 3.4028234663852886e38, "float64": sys.float_info.max}[name])

    @staticmethod
    def asarray(data, dtype):
        # Request validation needs no storage implementation: retain exact data
        # so boundary tests do not introduce casting or integer rounding.
        return types.SimpleNamespace(reshape=lambda shape: types.SimpleNamespace(
            values=list(data), dtype=dtype.name, shape=tuple(shape)))

    @staticmethod
    def empty(shape, dtype):
        return Array([0] * math.prod(shape), dtype=dtype, shape=shape)


TRT = types.SimpleNamespace(
    TensorLocation=types.SimpleNamespace(DEVICE="device", HOST="host"),
    TensorFormat=types.SimpleNamespace(LINEAR="linear", HWC8="hwc8"),
    TensorIOMode=types.SimpleNamespace(INPUT="input", OUTPUT="output"),
    nptype=lambda value: value,
)


def tensors(dtype="float32", shape=None):
    return [{"name": "x", "dtype": dtype, "shape": [2] if shape is None else shape, "input": True},
            {"name": "y", "dtype": "float32", "shape": [2], "input": False}]


def request(data=None, shape=None):
    return {"inputs": {"x": {"shape": [2] if shape is None else shape,
                              "data": [1, 2] if data is None else data}}}


class Engine:
    def __init__(self, entries=None):
        self.entries = copy.deepcopy(entries if entries is not None else tensors())
        self.num_io_tensors = len(self.entries)

    def item(self, name):
        return next(tensor for tensor in self.entries if tensor["name"] == name)

    def get_tensor_name(self, index): return self.entries[index]["name"]
    def get_tensor_location(self, name): return self.item(name).get("location", "device")
    def get_tensor_format(self, name): return self.item(name).get("format", "linear")
    def get_tensor_vectorized_dim(self, name): return self.item(name).get("vectorized_dim", -1)
    def is_shape_inference_io(self, name): return self.item(name).get("shape_io", False)
    def get_tensor_dtype(self, name): return self.item(name)["dtype"]
    def get_tensor_shape(self, name): return self.item(name)["shape"]
    def get_tensor_mode(self, name): return "input" if self.item(name)["input"] else "output"


class JsonAndShapeTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_fields_at_every_depth(self):
        for raw in ('{"inputs":{},"inputs":{}}',
                    '{"inputs":{"x":{"shape":[1],"shape":[2]}}}',
                    '{"x":1,"\\u0078":2}'):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "duplicate"):
                bridge.strict_json(raw)
        self.assertEqual(bridge.strict_json(b'{"x":1,"nested":{"x":2}}'),
                         {"x": 1, "nested": {"x": 2}})

    def test_strict_json_rejects_nonfinite_constants_and_malformed_payloads(self):
        for raw in ("NaN", "Infinity", "-Infinity", '{"data":[NaN]}'):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "non-finite"):
                bridge.strict_json(raw)
        for raw in ("", "{", "{} trailing", b"\xff"):
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeDecodeError)):
                bridge.strict_json(raw)

    def test_volume_accepts_scalars_and_the_exact_rank_and_element_boundaries(self):
        self.assertEqual(bridge.volume([]), 1)
        self.assertEqual(bridge.volume(tuple([1] * bridge.MAX_RANK)), 1)
        self.assertEqual(bridge.volume([1024, 1024]), bridge.MAX_VALUES)

    def test_volume_rejects_dynamic_unresolved_and_oversized_shapes(self):
        for shape in (None, "1", [True], [1.0], [0], [-1], [-2],
                      [bridge.MAX_VALUES + 1], [1024, 1025], [1] * (bridge.MAX_RANK + 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                bridge.volume(shape)

    def test_cuda_call_preserves_values_and_rejects_error_statuses(self):
        class Status(enum.IntEnum):
            SUCCESS = 0
            FAILURE = 7
        self.assertEqual(bridge.cuda_call((Status.SUCCESS,)), [])
        self.assertEqual(bridge.cuda_call((0, 42)), 42)
        self.assertEqual(bridge.cuda_call((0, 42, 43)), [42, 43])
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            bridge.cuda_call((Status.FAILURE, 42))


class DescribeTests(unittest.TestCase):
    def test_named_io_probe_accepts_supported_dtypes_and_dynamic_or_scalar_shapes(self):
        for dtype in ("float32", "float16", "int8", "uint8", "int32", "int64", "bool"):
            for shape in ([-1, 2], [], [1] * bridge.MAX_RANK):
                with self.subTest(dtype=dtype, shape=shape):
                    expected = tensors(dtype=dtype, shape=shape)
                    self.assertEqual(bridge.describe(Engine(expected), TRT, StubNumpy), expected)

    def test_probe_rejects_unsupported_layout_locations_dtypes_and_shapes(self):
        cases = [("location", "host"), ("format", "hwc8"), ("vectorized_dim", 0),
                 ("shape_io", True), ("dtype", "float64"), ("shape", [0]),
                 ("shape", [-2]), ("shape", [True]), ("shape", [1024, 1025]),
                 ("shape", [1] * (bridge.MAX_RANK + 1))]
        for field, value in cases:
            entries = tensors()
            entries[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                bridge.describe(Engine(entries), TRT, StubNumpy)

    def test_probe_rejects_empty_duplicate_oversized_and_nontext_names(self):
        for name in ("", 3, "z" * 257, "中" * 86, "y"):
            entries = tensors()
            entries[0]["name"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                bridge.describe(Engine(entries), TRT, StubNumpy)
        for entries in ([], tensors() * 17, [tensors()[0]], [tensors()[1]]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                bridge.describe(Engine(entries), TRT, StubNumpy)


class RequestTests(unittest.TestCase):
    np = StubNumpy

    def validate(self, payload, declarations=None):
        return bridge.validate_request(payload, tensors() if declarations is None else declarations, self.np)

    def test_input_names_output_selection_and_request_keys_are_exact(self):
        for payload in (None, [], {}, {"inputs": {}}, {"inputs": {"unknown": {}}},
                        {**request(), "unknown": 1}, {**request(), "output": "x"},
                        {**request(), "output": 1}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.validate(payload)
        declarations = tensors() + [{"name": "other", "dtype": "float32", "shape": [2], "input": False}]
        with self.assertRaisesRegex(ValueError, "multiple outputs"):
            self.validate(request(), declarations)
        arrays, selected = self.validate({**request(), "output": "other"}, declarations)
        self.assertEqual(set(arrays), {"x"})
        self.assertEqual(selected, "other")

    def test_dynamic_shapes_are_resolved_without_accepting_rank_or_static_dimension_mismatch(self):
        declarations = tensors(shape=[-1, 2])
        arrays, selected = self.validate(request(data=[1, 2, 3, 4, 5, 6], shape=[3, 2]), declarations)
        self.assertEqual(arrays["x"].shape, (3, 2))
        self.assertEqual(selected, "y")
        for shape, data in (([6], list(range(6))), ([2, 3], list(range(6))),
                            ([-1, 2], [1, 2]), ([3, 2], [1, 2])):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.validate(request(data=data, shape=shape), declarations)
        arrays, _ = self.validate(request(data=[1], shape=[]), tensors(shape=[]))
        self.assertEqual(arrays["x"].shape, ())

    def test_tensor_keys_data_layout_and_combined_input_bounds_are_enforced(self):
        for tensor in (None, {}, {"shape": [2]}, {"shape": [2], "data": [1, 2], "dtype": "float32"},
                       {"shape": [2], "data": (1, 2)}, {"shape": [2], "data": [[1], [2]]}):
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                self.validate({"inputs": {"x": tensor}})
        declarations = tensors(shape=[-1])
        declarations.insert(1, {"name": "second", "shape": [-1], "dtype": "float32", "input": True})
        payload = {"inputs": {name: {"shape": [3], "data": [1, 2, 3]} for name in ("x", "second")}}
        with mock.patch.object(bridge, "MAX_VALUES", 4), self.assertRaisesRegex(ValueError, "combined"):
            self.validate(payload, declarations)

    def test_nonfinite_values_booleans_and_float_overflows_fail_before_array_conversion(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, False, "1", None, {}, 3.5e38):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(request(data=[value, 0]))
        # Exponent overflow is legal JSON syntax; the numeric boundary must still reject it.
        payload = bridge.strict_json('{"inputs":{"x":{"shape":[2],"data":[1e999,0]}}}')
        with self.assertRaisesRegex(ValueError, "finite"):
            self.validate(payload)
        with self.assertRaises((ValueError, OverflowError)):
            self.validate(request(data=[10**400, 0]))
        for value in (65505, -65505):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "floating"):
                self.validate(request(data=[value, 0]), tensors(dtype="float16"))
        self.validate(request(data=[65504, -65504]), tensors(dtype="float16"))

    def test_integer_and_bool_dtypes_preserve_integrality_and_exact_bounds(self):
        for dtype in ("int8", "uint8", "int32", "int64", "bool"):
            dt = StubNumpy.dtype(dtype)
            low, high = (0, 1) if dtype == "bool" else (StubNumpy.iinfo(dt).min, StubNumpy.iinfo(dt).max)
            with self.subTest(dtype=dtype):
                self.validate(request(data=[low, high]), tensors(dtype=dtype))
                for value in (low - 1, high + 1, 0.5, True):
                    with self.subTest(dtype=dtype, value=value), self.assertRaises(ValueError):
                        self.validate(request(data=[value, low]), tensors(dtype=dtype))
        self.validate(request(data=[0.0, 1.0]), tensors(dtype="bool"))


class Cuda:
    """CPU memory emulates CUDA allocations/copies and records API ordering."""
    cudaMemcpyKind = types.SimpleNamespace(cudaMemcpyHostToDevice="h2d", cudaMemcpyDeviceToHost="d2h")

    def __init__(self, events, failure=None):
        self.events = events
        self.failure = failure or {}
        self.calls = {}
        self.allocations = {}
        self.stream = 17

    def result(self, method, *values):
        self.calls[method] = self.calls.get(method, 0) + 1
        if self.failure.get(method) == self.calls[method]:
            return (9,)
        return (0, *values)

    def cudaStreamCreate(self):
        self.events.append(("create", self.stream))
        return self.result("create", self.stream)

    def cudaMalloc(self, size):
        self.events.append(("malloc", size))
        status = self.result("malloc")
        if status[0]:
            return status
        allocation = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(allocation)
        self.allocations[ptr] = allocation
        return (0, ptr)

    def cudaMemcpyAsync(self, destination, source, size, kind, stream):
        self.events.append(("copy", kind, size, stream))
        status = self.result("copy")
        if not status[0]:
            ctypes.memmove(destination, source, size)
        return status

    def cudaStreamSynchronize(self, stream):
        self.events.append(("sync", stream))
        return self.result("sync")

    def cudaFree(self, ptr):
        self.events.append(("free", ptr))
        self.allocations.pop(ptr)
        return self.result("free")

    def cudaStreamDestroy(self, stream):
        self.events.append(("destroy", stream))
        return self.result("destroy")


class Context:
    def __init__(self, events, declarations=None, failure=None, values=None):
        self.events = events
        self.declarations = declarations if declarations is not None else tensors()
        self.failure = failure
        self.addresses = {}
        self.values = values if values is not None else [2.0, 4.0]

    def set_input_shape(self, name, shape):
        self.events.append(("shape", name, tuple(shape)))
        return self.failure != "shape"

    def infer_shapes(self):
        self.events.append(("infer",))
        return ["unresolved"] if self.failure == "infer" else []

    def get_tensor_shape(self, name):
        return next(t["shape"] for t in self.declarations if t["name"] == name)

    def set_tensor_address(self, name, ptr):
        self.events.append(("address", name, ptr))
        self.addresses[name] = ptr
        return not (self.failure == "address" and name == "y")

    def execute_async_v3(self, *, stream_handle):
        self.events.append(("execute", stream_handle))
        if self.failure == "execute":
            return False
        # Assert H2D copies have completed before the mocked launch; simulate a
        # deterministic kernel result so D2H must really copy into host storage.
        actual_input = list((ctypes.c_float * 2).from_address(self.addresses["x"]))
        if actual_input != [1.0, 2.0]:
            raise AssertionError(f"incorrect input binding/copy: {actual_input}")
        output = (ctypes.c_float * len(self.values))(*self.values)
        ctypes.memmove(self.addresses["y"], output, ctypes.sizeof(output))
        return True


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.arrays = {"x": Array([1.0, 2.0])}

    def execute(self, *, failure=None, context_failure=None, declarations=None, values=None):
        declarations = tensors() if declarations is None else declarations
        self.cuda = Cuda(self.events, failure)
        context = Context(self.events, declarations, context_failure, values)
        with mock.patch.object(bridge, "validate_request", return_value=(self.arrays, "y")):
            return bridge.execute(Engine(declarations), context, request(), declarations, TRT, self.cuda, StubNumpy)

    def assert_cleaned(self, allocations):
        freed = [event[1] for event in self.events if event[0] == "free"]
        bound = [event[2] for event in self.events if event[0] == "address"]
        self.assertEqual(len(freed), allocations)
        self.assertEqual(freed, list(reversed(bound)))
        self.assertEqual(self.events[-1], ("destroy", 17))
        self.assertIn(("sync", 17), self.events)
        first_free = next((index for index, event in enumerate(self.events) if event[0] == "free"), None)
        if first_free is not None:
            self.assertEqual(self.events[first_free - 1], ("sync", 17))
        self.assertEqual(self.cuda.allocations, {})

    def test_named_bindings_copy_directions_stream_and_reverse_cleanup_are_correct(self):
        result = self.execute()
        self.assertEqual(result, {"text": None, "logits": [2.0, 4.0],
                                  "image": None, "audio": None, "video": None})
        self.assertEqual([event[0] for event in self.events],
                         ["shape", "infer", "create", "malloc", "address", "copy", "malloc",
                          "address", "execute", "copy", "sync", "sync", "free", "free", "destroy"])
        self.assertEqual([event for event in self.events if event[0] == "copy"],
                         [("copy", "h2d", 8, 17), ("copy", "d2h", 8, 17)])
        self.assert_cleaned(2)

    def test_multiple_outputs_are_bound_and_copied_before_selecting_the_requested_name(self):
        declarations = tensors() + [{"name": "other", "dtype": "float32", "shape": [2], "input": False}]
        self.cuda = Cuda(self.events)
        context = Context(self.events, declarations)
        launch = context.execute_async_v3

        def execute_with_other_output(*, stream_handle):
            success = launch(stream_handle=stream_handle)
            output = (ctypes.c_float * 2)(9.0, 8.0)
            ctypes.memmove(context.addresses["other"], output, ctypes.sizeof(output))
            return success

        context.execute_async_v3 = execute_with_other_output
        with mock.patch.object(bridge, "validate_request", return_value=(self.arrays, "other")):
            result = bridge.execute(Engine(declarations), context, request(), declarations,
                                    TRT, self.cuda, StubNumpy)
        self.assertEqual(result["logits"], [9.0, 8.0])
        self.assertEqual([event[1] for event in self.events if event[0] == "address"], ["x", "y", "other"])
        self.assertEqual([event[1] for event in self.events if event[0] == "copy"], ["h2d", "d2h", "d2h"])
        self.assert_cleaned(3)

    def test_shape_profile_and_unresolved_shapes_fail_before_cuda_allocation(self):
        for failure in ("shape", "infer"):
            self.events.clear()
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.execute(context_failure=failure)
            self.assertFalse(any(event[0] in {"create", "malloc"} for event in self.events))

    def test_stream_creation_failure_allocates_no_buffers(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.execute(failure={"create": 1})
        self.assertFalse(any(event[0] in {"malloc", "free", "destroy"} for event in self.events))
        self.assertEqual(self.cuda.allocations, {})

    def test_allocation_copy_bind_launch_and_sync_failures_recover_owned_resources(self):
        for failure, context_failure, count in [
            ({"malloc": 1}, None, 0), ({"malloc": 2}, None, 1),
            ({"copy": 1}, None, 1), ({"copy": 2}, None, 2),
            ({"sync": 1}, None, 2), (None, "address", 2), (None, "execute", 2),
        ]:
            self.events.clear()
            with self.subTest(failure=failure, context_failure=context_failure), self.assertRaises(RuntimeError):
                self.execute(failure=failure, context_failure=context_failure)
            self.assert_cleaned(count)

    def test_host_output_allocation_failure_still_frees_device_inputs(self):
        with mock.patch.object(StubNumpy, "empty", side_effect=MemoryError("host allocation failed")):
            with self.assertRaises(MemoryError):
                self.execute()
        self.assert_cleaned(1)

    def test_unresolved_output_shape_and_combined_io_limit_recover_input_allocation(self):
        declarations = tensors()
        declarations[1]["shape"] = [-1]
        with self.assertRaises(ValueError):
            self.execute(declarations=declarations)
        self.assert_cleaned(1)
        self.events.clear()
        with mock.patch.object(bridge, "MAX_VALUES", 3), self.assertRaisesRegex(ValueError, "combined I/O"):
            self.execute()
        self.assert_cleaned(1)

    def test_nonfinite_selected_output_is_rejected_and_buffers_are_freed(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.events.clear()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                self.execute(values=[value, 0.0])
            self.assert_cleaned(2)


@unittest.skipUnless(real_numpy is not None,
    "optional NumPy checks: install requirements/tensor-runtime-test.txt")
class NumpyIntegrationTests(unittest.TestCase):
    def test_real_numpy_casts_integer_bool_and_float_arrays_contiguously(self):
        for dtype, data in (("int8", [-128, 127]), ("uint8", [0, 255]),
                            ("int32", [-2**31, 2**31 - 1]),
                            ("int64", [-2**63, 2**63 - 1]),
                            ("bool", [0, 1]), ("float16", [65504, -65504]),
                            ("float32", [1.5, -2.25])):
            with self.subTest(dtype=dtype):
                arrays, selected = bridge.validate_request(request(data=data), tensors(dtype=dtype), real_numpy)
                self.assertEqual(arrays["x"].dtype.name, dtype)
                self.assertTrue(arrays["x"].flags.c_contiguous)
                self.assertEqual(arrays["x"].shape, (2,))
                self.assertEqual(arrays["x"].tolist(), data)
                self.assertEqual(selected, "y")

    def test_real_numpy_dynamic_shapes_and_named_cuda_copies(self):
        arrays, _ = bridge.validate_request(request(data=[1, 2, 3, 4], shape=[2, 2]),
                                            tensors(shape=[-1, 2]), real_numpy)
        self.assertEqual(arrays["x"].tolist(), [[1, 2], [3, 4]])
        events = []
        cuda = Cuda(events)
        result = bridge.execute(Engine(), Context(events), request(), tensors(), TRT, cuda, real_numpy)
        self.assertEqual(result["logits"], [2.0, 4.0])
        self.assertEqual(cuda.allocations, {})
        self.assertEqual(events[-1], ("destroy", 17))

    def test_real_numpy_rejects_fractional_integer_boolean_and_overflow_values(self):
        for dtype, value in (("int8", 128), ("uint8", -1), ("int32", 2**31),
                             ("int64", 2**63), ("bool", 2), ("bool", 0.5),
                             ("float16", 65505), ("float32", 3.5e38),
                             ("float32", True), ("int64", 1.5),
                             ("float32", math.nextafter(float(real_numpy.finfo(real_numpy.float32).max), math.inf))):
            with self.subTest(dtype=dtype, value=value), self.assertRaises(ValueError):
                bridge.validate_request(request(data=[value, 0]), tensors(dtype=dtype), real_numpy)


if __name__ == "__main__":
    unittest.main()
