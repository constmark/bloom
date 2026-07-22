#!/usr/bin/env python3
"""Run local Qwen3-ASR inference for Bloom."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


LANGUAGE_ALIASES = {
    "auto": None,
    "none": None,
    "": None,
    "zh": "Chinese",
    "zho": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "eng": "English",
    "english": "English",
    "ja": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
}


def normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    return LANGUAGE_ALIASES.get(value.strip().lower(), value)


def resolve_device(device: str) -> str:
    if device in {"cpu", "cuda:0", "cuda", "mps"}:
        return "cuda:0" if device == "cuda" else device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def resolve_dtype(dtype: str, device: str) -> Any:
    import torch

    if dtype != "auto":
        return getattr(torch, dtype)
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model(args: argparse.Namespace):
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit(
            "Missing Python dependency. Create the local environment with:\n"
            "  /opt/homebrew/bin/python3.12 -m venv .venv-qwen-asr\n"
            "  .venv-qwen-asr/bin/python -m pip install -U pip qwen-asr\n"
            "Then rerun Bloom, or set BLOOM_ASR_PYTHON to that python."
        ) from exc

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    print(
        f"Loading Qwen3-ASR from {args.model_path} on {device} ({str(dtype).split('.')[-1]})",
        file=sys.stderr,
    )
    torch.set_grad_enabled(False)
    return Qwen3ASRModel.from_pretrained(str(args.model_path), **kwargs)


def extract_text(result: Any) -> str:
    if hasattr(result, "text"):
        return result.text or ""
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result)


def extract_language(result: Any) -> str | None:
    if hasattr(result, "language"):
        return result.language
    if isinstance(result, dict):
        value = result.get("language")
        return None if value is None else str(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-ASR local inference")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-inference-batch-size", type=int, default=1)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    parser.add_argument("--stream", action="store_true", help="Stream transcribed tokens in real-time")
    parser.add_argument("--daemon", action="store_true", help="Run in persistent daemon mode")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Model path not found: {args.model_path}", file=sys.stderr)
        return 2
    if not args.daemon and (args.audio is None or not args.audio.exists()):
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 2

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model = load_model(args)
    
    if args.daemon:
        print("READY", file=sys.stderr, flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                audio_path = req.get("audio")
                lang = req.get("language", "auto")
                language = normalize_language(lang)
                results = model.transcribe(audio=str(audio_path), language=language)
                result = results[0] if results else {}
                text = extract_text(result)
                detected_language = extract_language(result)
                print(json.dumps({"status": "ok", "text": text, "language": detected_language}, ensure_ascii=False))
                sys.stdout.flush()
            except Exception as e:
                print(json.dumps({"status": "error", "error": str(e)}))
                sys.stdout.flush()
        return 0

    language = normalize_language(args.language)

    if args.stream:
        from threading import Thread
        from transformers import TextIteratorStreamer
        from qwen_asr.inference.utils import normalize_audios

        # Load audio
        wavs = normalize_audios(str(args.audio))
        wav = wavs[0]

        # Standard prompt building
        prompt = model._build_text_prompt(context="", force_language=language)

        # Process inputs using processor
        inputs = model.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        inputs = inputs.to(model.model.device).to(model.model.dtype)

        # Setup streamer
        streamer = TextIteratorStreamer(model.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=151645,
            eos_token_id=151645
        )

        thread = Thread(target=model.model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        started = False
        for new_text in streamer:
            if started:
                print(new_text, end="", flush=True)
            else:
                buffer += new_text
                if "<asr_text>" in buffer:
                    parts = buffer.split("<asr_text>", 1)
                    if len(parts) > 1 and parts[1]:
                        print(parts[1], end="", flush=True)
                    started = True
                elif len(buffer) > 50:
                    print(buffer, end="", flush=True)
                    started = True
        print()
        thread.join()
        return 0

    print(f"Transcribing {args.audio} (language={language or 'auto'})", file=sys.stderr)
    results = model.transcribe(audio=str(args.audio), language=language)
    result = results[0] if results else {}
    text = extract_text(result)
    detected_language = extract_language(result)

    if args.json:
        print(json.dumps({"language": detected_language, "text": text}, ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
