#!/usr/bin/env python3
"""Small, dependency-light ONNX Runtime bridge used by Bloom.

The Rust engine deliberately keeps ONNX Runtime optional because its native
execution providers are platform-specific. This bridge makes the runtime
boundary explicit while still providing a real executable path whenever the
operator installs ``onnxruntime`` (or ``onnxruntime-gpu``).

The request is a serialized ``ModelInput`` enum on stdin. The first ONNX input
is populated from that value and the first output is returned as JSON. Models
with string inputs can consume ``Text`` directly; numeric models can consume
whitespace-separated integer text, audio samples, or image bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _provider_configuration(requested: str, device: str) -> tuple[str, dict[str, str]]:
    normalized = requested.upper()
    if normalized in {"CUDA", "GPU", "CUDAEXECUTIONPROVIDER"}:
        return "CUDAExecutionProvider", {}
    if normalized in {"COREML", "METAL", "COREMLEXECUTIONPROVIDER"}:
        return "CoreMLExecutionProvider", {}
    if normalized in {"OPENVINO", "NPU", "OPENVINOEXECUTIONPROVIDER"}:
        return "OpenVINOExecutionProvider", {
            "device_type": "NPU" if device.upper() in {"NPU", "OPENVINO"} else "CPU"
        }
    return "CPUExecutionProvider", {}


def _session(model_path: str, device: str, requested_provider: str, ort: Any):
    provider, options = _provider_configuration(requested_provider, device)
    available = set(ort.get_available_providers())
    if provider not in available:
        raise RuntimeError(
            f"requested ONNX Runtime provider is unavailable (requested={provider}, available={sorted(available)})"
        )
    session_options = ort.SessionOptions()
    if provider != "CPUExecutionProvider":
        # A non-CPU request must fail when the selected EP cannot cover the
        # graph; silently appending CPU would make Bloom report a false device.
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=[provider],
        provider_options=[options],
    ), provider


def _input_value(request: dict[str, Any]) -> Any:
    kind = next(iter(request), "")
    if kind == "Text":
        return request[kind]["prompt"]
    if kind == "Audio":
        return request[kind]["samples"]
    if kind == "AudioFile":
        with open(request[kind]["path"], "rb") as stream:
            return list(stream.read())
    if kind == "Vision":
        return list(request[kind]["bytes"])
    if kind == "Multi":
        value = request[kind]
        if value.get("audio") is not None:
            return value["audio"]
        if value.get("image") is not None:
            return list(value["image"])
        return value.get("text", "")
    raise ValueError(f"unsupported ModelInput variant: {kind}")


def _coerce(value: Any, input_meta: Any):
    import numpy as np

    type_name = str(input_meta.type)
    if "string" in type_name:
        return np.asarray([str(value)], dtype=object)
    if isinstance(value, str):
        try:
            value = [int(token) for token in value.split()]
        except ValueError as exc:
            raise ValueError(
                "the ONNX model expects numeric input; Text must contain whitespace-separated integers"
            ) from exc
    if "int64" in type_name:
        dtype = np.int64
    elif "int32" in type_name:
        dtype = np.int32
    elif "int16" in type_name:
        dtype = np.int16
    elif "uint8" in type_name:
        dtype = np.uint8
    elif "int8" in type_name:
        dtype = np.int8
    elif "float16" in type_name:
        dtype = np.float16
    elif "double" in type_name or "float64" in type_name:
        dtype = np.float64
    else:
        dtype = np.float32
    array = np.asarray(value, dtype=dtype)
    shape = getattr(input_meta, "shape", None) or []
    static_shape = [int(dim) if isinstance(dim, (int, np.integer)) else None for dim in shape]
    if array.ndim == 1 and len(static_shape) > 1 and static_shape[0] == 1:
        array = array.reshape((1, *array.shape))
    expected = 1
    if static_shape and all(dim is not None and dim > 0 for dim in static_shape):
        for dim in static_shape:
            expected *= dim
        if expected == array.size:
            array = array.reshape(static_shape)
    return array


def _jsonable(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind in "fiu":
        return array.reshape(-1).tolist()
    if array.dtype.kind in "SUO":
        flat = array.reshape(-1).tolist()
        return "".join(str(item) for item in flat)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--provider", default="")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(
            "onnxruntime is required for the ONNX backend; install onnxruntime or onnxruntime-gpu",
            file=sys.stderr,
        )
        return 2

    try:
        requested_provider = args.provider or os.environ.get(
            "BLOOM_ONNX_EXECUTION_PROVIDER", args.device
        )
        session, provider = _session(args.model_path, args.device, requested_provider, ort)
        if args.probe:
            print(
                json.dumps(
                    {"ok": True, "device": args.device.upper(), "provider": provider}
                )
            )
            return 0
        request = json.load(sys.stdin)
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs")
        feed = {inputs[0].name: _coerce(_input_value(request), inputs[0])}
        outputs = session.run(None, feed)
        if not outputs:
            raise RuntimeError("ONNX model produced no outputs")
        first = outputs[0]
        result: dict[str, Any] = {"text": None, "logits": None}
        if np.asarray(first).dtype.kind in "SUO":
            result["text"] = _jsonable(first)
        else:
            result["logits"] = _jsonable(first)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - bridge errors are sent to Bloom
        print(f"ONNX Runtime inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
