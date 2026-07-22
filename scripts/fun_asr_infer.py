#!/usr/bin/env python3
"""Run speech recognition with ModelScope's Fun-ASR-Nano-2512 model.

Usage:
    python fun_asr_infer.py <audio_file> [language]

Examples:
    python fun_asr_infer.py /root/models/FunAudioLLM/Fun-ASR-Nano-2512/example/zh.mp3
    python fun_asr_infer.py /root/models/FunAudioLLM/Fun-ASR-Nano-2512/example/en.mp3
"""

import sys
import argparse
import os

MODEL_DIR = os.environ.get("BLOOM_FUN_ASR_MODEL", "/root/models/FunAudioLLM/Fun-ASR-Nano-2512")
FUNASR_SITE = os.environ.get(
    "BLOOM_FUNASR_SITE",
    "/root/miniconda/envs/vllm/lib/python3.10/site-packages/funasr",
)


def setup_paths():
    """Configure Python module search paths."""
    sys.path.insert(0, MODEL_DIR)
    if FUNASR_SITE and os.path.exists(FUNASR_SITE):
        sys.path.insert(0, f"{FUNASR_SITE}/models/fun_asr_nano")
        sys.path.insert(0, FUNASR_SITE)


def load_model(model_dir: str, device: str = "cpu"):
    """Load the Fun-ASR model."""
    global MODEL_DIR
    MODEL_DIR = model_dir
    setup_paths()
    
    print(f"Loading Fun-ASR model from {MODEL_DIR}...", file=sys.stderr)
    
    from funasr import AutoModel
    
    model = AutoModel(
        model=MODEL_DIR,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
    )
    
    print(f"Model loaded successfully on {device}", file=sys.stderr)
    return model


def transcribe(model, audio_path: str, language: str = "auto", itn: bool = True):
    """Transcribe an audio file.

    Args:
        model: Loaded model.
        audio_path: Path to the audio file.
        language: Language name or "auto" for automatic detection.
        itn: Whether to apply inverse text normalization.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return None
    
    print(f"Transcribing: {audio_path}", file=sys.stderr)
    print(f"Language: {language}, ITN: {itn}", file=sys.stderr)
    
    try:
        res = model.generate(
            input=[audio_path],
            cache={},
            batch_size=1,
            language=language,
            itn=itn,
        )
        
        if res and len(res) > 0:
            text = res[0].get("text", "")
            return text
        else:
            print("Warning: Empty result", file=sys.stderr)
            return ""
            
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR speech recognition")
    parser.add_argument("audio_file", nargs="?", default=None, help="Path to the audio file")
    parser.add_argument("language", nargs="?", default="auto", help="Language (default: auto)")
    parser.add_argument("--model-path", default=MODEL_DIR, help="Model directory")
    parser.add_argument("--no-itn", action="store_true", help="Disable inverse text normalization")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    
    args = parser.parse_args()
    
    # Load the model.
    model = load_model(model_dir=args.model_path, device=args.device)
    
    if args.daemon:
        print("READY", file=sys.stderr, flush=True)
        import json
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                audio_file = req.get("audio")
                lang = req.get("language", "auto")
                text = transcribe(
                    model, 
                    audio_file, 
                    language=lang,
                    itn=not args.no_itn
                )
                if text is None:
                    print(json.dumps({"status": "error", "error": "Transcription failed"}))
                else:
                    print(json.dumps({"status": "ok", "text": text}))
                sys.stdout.flush()
            except Exception as e:
                print(json.dumps({"status": "error", "error": str(e)}))
                sys.stdout.flush()
        return

    if not args.audio_file:
        print("Error: audio_file is required when not running in daemon mode", file=sys.stderr)
        sys.exit(1)

    # Transcribe the file.
    text = transcribe(
        model, 
        args.audio_file, 
        language=args.language,
        itn=not args.no_itn
    )
    
    if text is not None:
        # Write only the plain-text result to stdout.
        print(text)


if __name__ == "__main__":
    main()
