#!/usr/bin/env python3
"""MLX-LM bridge for Bloom's Apple Silicon runtime.

The native engine keeps MLX optional and communicates with this process over a
small JSON-lines protocol.  ``mlx-lm`` is intentionally not bundled: operators
install a pinned compatible package in the Python interpreter selected by
``BLOOM_MLX_PYTHON`` or ``BLOOM_PYTHON``.

The bridge supports text generation and text-only ``Multi`` inputs.  Audio and
image inputs fail closed until a multimodal MLX model declares a corresponding
processor contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator


MAX_REQUEST_BYTES = 16 * 1024 * 1024


def _select_device(name: str):
    import mlx.core as mx

    normalized = name.upper()
    if normalized == "GPU":
        mx.set_default_device(mx.gpu)
    elif normalized == "CPU":
        mx.set_default_device(mx.cpu)
    else:
        raise ValueError(f"MLX bridge received unsupported device {name!r}")
    return mx


def _load_model(model_path: str):
    from mlx_lm import load

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"MLX model path does not exist: {path}")
    # mlx-lm loads a model directory containing config/tokenizer/weights. A
    # direct weight path is accepted by Bloom's catalog, so use its parent and
    # preserve the normal Hugging Face-style package contract.
    package = path if path.is_dir() else path.parent
    return load(str(package), lazy=False)


def _text_input(payload: dict[str, Any]) -> str:
    kind = next(iter(payload), "")
    if kind == "Text":
        return str(payload[kind]["prompt"])
    if kind == "Multi":
        value = payload[kind]
        if value.get("audio") is not None or value.get("image") is not None:
            raise ValueError("MLX text bridge does not support audio or image inputs")
        text = value.get("text")
        if text is None:
            raise ValueError("MLX text bridge requires Multi.text")
        return str(text)
    raise ValueError(f"MLX text bridge does not support ModelInput variant {kind!r}")


def _request() -> tuple[dict[str, Any], dict[str, Any]]:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("MLX request exceeds 16 MiB")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MLX request must be a JSON object")
    if "input" in payload:
        value = payload["input"]
        params = payload.get("params", {})
    else:
        value = payload
        params = {}
    if not isinstance(value, dict) or not isinstance(params, dict):
        raise ValueError("MLX request input and params must be JSON objects")
    return value, params


def _generation(model, tokenizer, prompt: str, params: dict[str, Any]) -> Iterator[Any]:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    max_tokens = int(params.get("max_tokens", 128))
    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 0.9))
    if max_tokens <= 0 or max_tokens > 32768:
        raise ValueError("max_tokens must be between 1 and 32768")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1")
    seed = params.get("seed")
    if seed is not None:
        mx.random.seed(int(seed))
    sampler = make_sampler(temperature, top_p)
    yield from stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    try:
        mx = _select_device(args.device)
        model, tokenizer = _load_model(args.model_path)
        if args.probe:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "device": args.device,
                        "runtime": "mlx-lm",
                        "mlx_version": getattr(mx, "__version__", "unknown"),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        payload, params = _request()
        prompt = _text_input(payload)
        chunks = _generation(model, tokenizer, prompt, params)
        text = ""
        for response in chunks:
            delta = str(response.text)
            text += delta
            if args.stream and delta:
                print(json.dumps({"text": delta}, ensure_ascii=False), flush=True)
        if args.stream:
            print(json.dumps({"done": True}, ensure_ascii=False), flush=True)
        else:
            print(json.dumps({"text": text}, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - bridge errors cross the process boundary
        print(f"MLX Runtime inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
