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
import math
import os
import sys
from typing import Any
from pathlib import Path


MAX_VALUES = 16 * 1024 * 1024


def _embedding_encoder(session: Any, model_path: str, config: dict[str, Any]):
    """Load only the package tokenizer: no remote code or text-based counts."""
    from tokenizers import Tokenizer

    if config.get("pooling") not in {"mean", "cls", "none"}:
        raise ValueError("embedding pooling must be mean, cls or none")
    maximum = config.get("max_length")
    if type(maximum) is not int or not 1 <= maximum <= 32768:
        raise ValueError("embedding max_length must be between 1 and 32768")
    pad_id = config.get("pad_token_id")
    if type(pad_id) is not int or not 0 <= pad_id <= 2**32 - 1:
        raise ValueError("embedding pad_token_id must be a token ID")
    for key in ("truncate", "normalize"):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f"embedding {key} must be boolean")
    inputs = session.get_inputs()
    names = {meta.name for meta in inputs}
    if "input_ids" not in names or names - {"input_ids", "attention_mask", "token_type_ids"}:
        raise ValueError("encoder requires input_ids and supports only attention_mask/token_type_ids")
    for meta in inputs:
        if meta.type not in {"tensor(int64)", "tensor(int32)"} or len(meta.shape) != 2:
            raise ValueError("encoder inputs must be rank-2 integer token tensors")
    outputs = {meta.name for meta in session.get_outputs()}
    if config.get("output_name") not in outputs:
        raise ValueError("configured embedding output_name is absent from the graph")
    tokenizer = Tokenizer.from_file(str(Path(model_path).parent / "tokenizer.json"))
    if tokenizer.id_to_token(pad_id) is None:
        raise ValueError("pad_token_id is absent from the package tokenizer")
    # Package postprocessing/special tokens remain intact. Padding and
    # truncation are explicit export settings rather than hidden JSON defaults.
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if config.get("truncate", False):
        tokenizer.enable_truncation(max_length=maximum)
    return tokenizer


def _embed(session: Any, model_path: str, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    config = request.get("config", {})
    tokenizer = _embedding_encoder(session, model_path, config)
    texts = request.get("inputs")
    if not isinstance(texts, list) or not 1 <= len(texts) <= 128:
        raise ValueError("embedding input must contain 1 to 128 strings")
    if any(not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 32768 for text in texts):
        raise ValueError("embedding inputs must be nonempty strings of at most 32 KiB")
    encodings = tokenizer.encode_batch(texts, add_special_tokens=True)
    lengths = [len(encoding.ids) for encoding in encodings]
    if any(length == 0 or length > config["max_length"] for length in lengths):
        raise ValueError("encoded input exceeds max_length; explicit truncate is required")
    metas = session.get_inputs()
    static_lengths = {meta.shape[1] for meta in metas if isinstance(meta.shape[1], int)}
    if len(static_lengths) > 1:
        raise ValueError("encoder inputs have inconsistent static sequence lengths")
    width = next(iter(static_lengths), max(lengths))
    if width <= 0 or width > 32768 or width < max(lengths):
        raise ValueError("encoder static sequence length is incompatible with encoded input")
    if any(isinstance(meta.shape[0], int) and meta.shape[0] != len(texts) for meta in metas):
        raise ValueError("encoder static batch dimension does not match request")
    if "attention_mask" not in {meta.name for meta in metas} and any(length != width for length in lengths):
        raise ValueError("padded encoder batches require an attention_mask input")
    ids = np.full((len(texts), width), config["pad_token_id"], dtype=np.int64)
    mask = np.zeros_like(ids)
    types = np.zeros_like(ids)
    for row, encoding in enumerate(encodings):
        length = lengths[row]
        ids[row, :length] = encoding.ids
        mask[row, :length] = encoding.attention_mask
        types[row, :length] = encoding.type_ids
    if np.any((mask != 0) & (mask != 1)) or np.any(mask.sum(axis=1) == 0):
        raise ValueError("tokenizer produced invalid encoder attention masks")
    arrays = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
    feed = {}
    for meta in metas:
        values = arrays[meta.name]
        if meta.type == "tensor(int32)" and np.any(values > np.iinfo(np.int32).max):
            raise ValueError("token IDs do not fit graph int32 inputs")
        feed[meta.name] = values.astype(np.int32 if meta.type == "tensor(int32)" else np.int64)
    # This count is derived from the exact IDs/mask passed to this run. It
    # includes special tokens and respects truncation; batch padding is excluded.
    prompt_tokens = int(mask.sum())
    result = np.asarray(session.run([config["output_name"]], feed)[0])
    if result.dtype.kind != "f" or result.ndim not in (2, 3) or result.shape[0] != len(texts):
        raise ValueError("encoder output must be a floating tensor with the requested batch")
    pooling = config["pooling"]
    if pooling == "none":
        if result.ndim != 2:
            raise ValueError("pooling=none requires a rank-2 pooled embedding output")
        vectors = result.astype(np.float32)
    else:
        if result.ndim != 3 or result.shape[1] != width:
            raise ValueError("mean/cls pooling requires rank-3 token hidden states")
        values = result.astype(np.float32)
        vectors = values[:, 0, :] if pooling == "cls" else (
            (values * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
        )
    if vectors.shape[1] == 0 or vectors.size > 16 * 1024 * 1024 or not np.isfinite(vectors).all():
        raise ValueError("encoder returned invalid embedding values")
    if config.get("normalize", False):
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError("cannot normalize zero/nonfinite embedding")
        vectors = vectors / norms
    return {"embeddings": vectors.tolist(), "prompt_tokens": prompt_tokens,
            "usage_source": "input_ids_attention_mask"}


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
    if normalized in {"CPU", "CPUEXECUTIONPROVIDER"}:
        return "CPUExecutionProvider", {}
    raise ValueError(f"unsupported ONNX Runtime provider: {requested!r}")


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
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=[provider],
        provider_options=[options],
    )
    # ORT can retry session construction with a fallback provider internally.
    # Verify the actual session before reporting the requested device as active.
    if provider not in session.get_providers():
        raise RuntimeError(f"ONNX Runtime did not activate requested provider {provider}")
    session.disable_fallback()
    return session, provider


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
    types = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
             "tensor(int16)": np.int16, "tensor(int8)": np.int8,
             "tensor(uint64)": np.uint64, "tensor(uint32)": np.uint32,
             "tensor(uint16)": np.uint16, "tensor(uint8)": np.uint8,
             "tensor(float16)": np.float16, "tensor(float)": np.float32,
             "tensor(double)": np.float64, "tensor(bool)": np.bool_}
    if type_name == "tensor(string)":
        if not isinstance(value, str):
            raise ValueError("string ONNX inputs require text")
        array = np.asarray([value], dtype=object)
    elif type_name not in types:
        raise ValueError(f"unsupported ONNX input type: {type_name}")
    else:
        if isinstance(value, str):
            try:
                value = [int(token) for token in value.split()]
            except ValueError as exc:
                raise ValueError(
                    "the ONNX model expects numeric input; Text must contain whitespace-separated integers"
                ) from exc
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_VALUES:
            raise ValueError("numeric ONNX input must be a bounded nonempty list")
        dtype = np.dtype(types[type_name])
        for item in value:
            if type(item) not in (int, float) or (isinstance(item, float) and not math.isfinite(item)):
                raise ValueError("numeric ONNX input requires finite numbers, not booleans")
            if dtype.kind in "iub":
                low, high = (0, 1) if dtype.kind == "b" else (int(np.iinfo(dtype).min), int(np.iinfo(dtype).max))
                if int(item) != item or not low <= item <= high:
                    raise ValueError("numeric ONNX input exceeds integer dtype bounds or is fractional")
            elif abs(item) > float(np.finfo(dtype).max):
                raise ValueError("numeric ONNX input exceeds floating dtype bounds")
        array = np.asarray(value, dtype=dtype)
    shape = getattr(input_meta, "shape", None)
    if shape is None:
        raise ValueError("ONNX input rank must be declared")
    static_shape = [int(dim) if isinstance(dim, (int, np.integer)) else None for dim in shape]
    if len(static_shape) > 8 or any(dim is not None and dim <= 0 for dim in static_shape):
        raise ValueError("unsupported ONNX input shape")
    if all(dim is not None for dim in static_shape):
        if math.prod(static_shape) != array.size:
            raise ValueError("ONNX input count does not match its static shape")
        array = array.reshape(static_shape)
    elif array.ndim == 1 and len(static_shape) == 2 and static_shape[0] == 1:
        array = array.reshape((1, -1))
    if array.ndim != len(static_shape) or any(
        declared is not None and declared != actual
        for declared, actual in zip(static_shape, array.shape)
    ):
        raise ValueError("ONNX input shape does not match the graph")
    return array


def _jsonable(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.size > MAX_VALUES:
        raise ValueError("ONNX output exceeds the element limit")
    if array.dtype.kind in "fiub":
        # ModelOutput.logits is f32, even when a graph uses wider numeric I/O.
        # Reject invalid values before producing JSON or narrowing in Rust.
        if not np.isfinite(array).all() or np.any(np.abs(array.astype(np.float64)) > float(np.finfo(np.float32).max)):
            raise ValueError("ONNX output contains nonfinite or float32-overflowing values")
        return array.reshape(-1).astype(np.float64).tolist() if array.dtype.kind == "b" else array.reshape(-1).tolist()
    if array.dtype.kind in "SUO":
        flat = array.reshape(-1).tolist()
        if any(not isinstance(item, (str, bytes)) for item in flat):
            raise ValueError("ONNX string output contains non-string values")
        return "".join(item.decode("utf-8") if isinstance(item, bytes) else item for item in flat)
    raise ValueError(f"unsupported ONNX output type: {array.dtype}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--provider", default="")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--embedding-config")
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
            if args.embedding_config:
                _embedding_encoder(session, args.model_path, json.loads(args.embedding_config))
            print(
                json.dumps(
                    {"ok": True, "device": args.device.upper(), "provider": provider}
                )
            )
            return 0
        request = json.load(sys.stdin)
        if request.get("operation") == "embed":
            print(json.dumps(_embed(session, args.model_path, request), allow_nan=False))
            return 0
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
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - bridge errors are sent to Bloom
        print(f"ONNX Runtime inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
