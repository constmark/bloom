#!/usr/bin/env python3
"""Bounded numeric TensorRT 10.x execution, using NVIDIA CUDA Python bindings.

stdin: {"inputs": {name: {"shape": [...], "data": [...]}}, "output": name?}
The selected output is a flat finite float32-compatible vector. No tokenizer,
autoregressive loop, TensorRT-LLM, engine builder or CUDA-graph emulation exists.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import sys

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_VALUES = 1_048_576
MAX_TENSORS = 32
MAX_RANK = 8
MAX_PLAN_BYTES = 8 * 1024**3


def strict_json(raw):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError(f"duplicate JSON field: {name}")
            result[name] = value
        return result
    def nonfinite(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def volume(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) > MAX_RANK:
        raise ValueError("tensor rank exceeds the supported limit")
    count = 1
    for dim in shape:
        if type(dim) is not int or dim <= 0 or dim > MAX_VALUES:
            raise ValueError("tensor dimensions must be bounded positive integers")
        count *= dim
        if count > MAX_VALUES:
            raise ValueError("tensor element count exceeds the supported limit")
    return count


def cuda_call(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA call failed with status {int(error)}")
    return values[0] if len(values) == 1 else values


def describe(engine, trt, np):
    if not 1 <= engine.num_io_tensors <= MAX_TENSORS:
        raise ValueError("unsupported number of TensorRT I/O tensors")
    tensors = []
    names = set()
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if not isinstance(name, str) or not name or len(name.encode()) > 256 or name in names:
            raise ValueError("invalid or duplicate TensorRT tensor name")
        names.add(name)
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ValueError("host/shape I/O tensors are unsupported")
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR or engine.get_tensor_vectorized_dim(name) != -1:
            raise ValueError("vectorized I/O tensor formats are unsupported")
        if engine.is_shape_inference_io(name):
            raise ValueError("shape inference I/O tensors are unsupported")
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        if dtype.name not in {"float32", "float16", "int8", "uint8", "int32", "int64", "bool"}:
            raise ValueError(f"unsupported tensor dtype: {dtype}")
        shape = list(engine.get_tensor_shape(name))
        if len(shape) > MAX_RANK or any(type(d) is not int or d == 0 or d < -1 or d > MAX_VALUES for d in shape):
            raise ValueError("unsupported tensor shape")
        if -1 not in shape:
            volume(shape)
        tensors.append({"name": name, "shape": shape, "dtype": dtype.name,
                        "input": engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT})
    if not any(t["input"] for t in tensors) or not any(not t["input"] for t in tensors):
        raise ValueError("the TensorRT engine needs both inputs and outputs")
    return tensors


def validate_request(request, tensors, np):
    if not isinstance(request, dict) or set(request) - {"inputs", "output"}:
        raise ValueError("request must contain only inputs and optional output")
    inputs = request.get("inputs")
    expected = {t["name"]: t for t in tensors if t["input"]}
    if not isinstance(inputs, dict) or set(inputs) != set(expected):
        raise ValueError("input names must match every engine input exactly")
    outputs = [t["name"] for t in tensors if not t["input"]]
    selected = request.get("output")
    if selected is None:
        if len(outputs) != 1:
            raise ValueError("output must select one name when the engine has multiple outputs")
        selected = outputs[0]
    if not isinstance(selected, str) or selected not in outputs:
        raise ValueError("output must name an engine output")
    arrays = {}
    total = 0
    for name, tensor in inputs.items():
        if not isinstance(tensor, dict) or set(tensor) != {"shape", "data"}:
            raise ValueError("each input tensor requires exactly shape and data")
        count = volume(tensor["shape"])
        total += count
        if total > MAX_VALUES:
            raise ValueError("combined tensor input exceeds the element limit")
        data = tensor["data"]
        if not isinstance(data, list) or len(data) != count:
            raise ValueError("tensor data count must match its shape")
        target = expected[name]
        if len(target["shape"]) != len(tensor["shape"]) or any(
            declared != -1 and declared != actual
            for declared, actual in zip(target["shape"], tensor["shape"])
        ):
            raise ValueError("input shape does not match the engine")
        dtype = np.dtype(target["dtype"])
        for value in data:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("tensor data must be finite numbers, not booleans")
            if dtype.kind in "iub":
                if int(value) != value:
                    raise ValueError("integer tensor data must contain integers")
                low, high = (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
                if not low <= value <= high:
                    raise ValueError("integer tensor data exceeds dtype bounds")
            elif abs(value) > float(np.finfo(dtype).max):
                raise ValueError("tensor data exceeds floating dtype bounds")
        arrays[name] = np.asarray(data, dtype=dtype).reshape(tensor["shape"])
    return arrays, selected


def execute(engine, context, request, tensors, trt, cuda, np):
    arrays, selected = validate_request(request, tensors, np)
    for name, array in arrays.items():
        if not context.set_input_shape(name, array.shape):
            raise ValueError("input shape is outside the engine optimization profile")
    missing = context.infer_shapes()
    if missing:
        raise ValueError("input shapes could not be resolved")
    stream = cuda_call(cuda.cudaStreamCreate())
    buffers = []
    host = {}
    try:
        total = 0
        for tensor in tensors:
            name = tensor["name"]
            shape = list(context.get_tensor_shape(name))
            count = volume(shape)
            total += count
            if total > MAX_VALUES:
                raise ValueError("combined I/O allocation exceeds the element limit")
            array = arrays[name] if tensor["input"] else np.empty(shape, dtype=tensor["dtype"])
            host[name] = array
            ptr = cuda_call(cuda.cudaMalloc(array.nbytes))
            buffers.append(ptr)
            if not context.set_tensor_address(name, int(ptr)):
                raise RuntimeError("TensorRT rejected the I/O buffer address")
            if tensor["input"]:
                cuda_call(cuda.cudaMemcpyAsync(ptr, array.ctypes.data, array.nbytes,
                          cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        if not context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError("TensorRT execution failed")
        for tensor, ptr in zip(tensors, buffers):
            if not tensor["input"]:
                array = host[tensor["name"]]
                cuda_call(cuda.cudaMemcpyAsync(array.ctypes.data, ptr, array.nbytes,
                          cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        cuda_call(cuda.cudaStreamSynchronize(stream))
        result = [float(value) for value in host[selected].reshape(-1).tolist()]
        if any(not math.isfinite(v) or abs(v) > float(np.finfo(np.float32).max) for v in result):
            raise ValueError("output contains non-finite or float32-overflowing values")
        return {"text": None, "logits": result, "image": None, "audio": None, "video": None}
    finally:
        # Synchronize before releasing buffers even if execution/copy failed.
        # The parent enforces a hard process deadline if CUDA stalls here.
        cuda.cudaStreamSynchronize(stream)
        for ptr in reversed(buffers):
            cuda.cudaFree(ptr)
        cuda.cudaStreamDestroy(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        import numpy as np
        import tensorrt as trt
        from cuda.bindings import runtime as cuda
        major = int(trt.__version__.split(".")[0])
        if major != 10:
            raise RuntimeError("this bridge requires the TensorRT 10.x named-I/O API")
        cuda_call(cuda.cudaSetDevice(0))
        if not 0 < os.path.getsize(args.model_path) <= MAX_PLAN_BYTES:
            raise ValueError("TensorRT plan is empty or exceeds 8 GiB")
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(args.model_path, "rb") as source:
            plan = source.read(MAX_PLAN_BYTES + 1)
        if not 0 < len(plan) <= MAX_PLAN_BYTES:
            raise ValueError("TensorRT plan changed size beyond the supported limit")
        if hashlib.sha256(plan).hexdigest() != args.sha256:
            raise ValueError("TensorRT plan identity changed after model admission")
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError("TensorRT rejected this serialized engine")
        tensors = describe(engine, trt, np)
        if args.probe:
            print(json.dumps({"ok": True, "device": "CUDA", "runtime_version": trt.__version__,
                              "tensors": tensors}, allow_nan=False))
            return 0
        raw = sys.stdin.buffer.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("tensor request exceeded the input byte limit")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        result = execute(engine, context, strict_json(raw), tensors, trt, cuda, np)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        print(f"TensorRT runtime failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
