#!/usr/bin/env python3
"""Run local Qwen3-VL inference for Bloom."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from threading import Thread

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

def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL local inference")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Model path not found: {args.model_path}", file=sys.stderr)
        return 2

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
    from qwen_vl_utils import process_vision_info

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed(args.seed)

    # Load processor & model
    print(f"Loading Qwen3-VL from {args.model_path} on {device}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(str(args.model_path))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        device_map=device
    )
    model.eval()

    # Build prompt content
    content = []
    if args.image:
        if not args.image.exists():
            print(f"Image not found: {args.image}", file=sys.stderr)
            return 2
        content.append({"type": "image", "image": str(args.image)})
    content.append({"type": "text", "text": args.prompt})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    # Preprocess inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": args.temperature if args.temperature > 0.0 else None,
        "top_p": args.top_p if args.temperature > 0.0 else None,
    }

    if args.stream:
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            **generation_config
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            print(new_text, end="", flush=True)
        print()
        thread.join()
        return 0
    else:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_config)
        
        # De-duplicate prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if args.json:
            print(json.dumps({"text": response}, ensure_ascii=False))
        else:
            print(response)
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
