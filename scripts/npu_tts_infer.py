#!/usr/bin/env python3
"""Run CosyVoice / ChatTTS inference on Intel NPU (via OpenVINO) or CPU for Bloom."""

import argparse
import sys
import os
import json
import struct
import wave

def write_wav(filename: str, samples, sample_rate: int = 22050):
    """Write PCM samples to a WAV file."""
    import numpy as np
    arr = np.array(samples, dtype=np.float32)
    # Normalize to int16
    arr = np.clip(arr, -1.0, 1.0)
    arr_int16 = (arr * 32767).astype(np.int16)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(arr_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Bloom TTS inference on Intel NPU / CPU")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to TTS model directory (CosyVoice / ChatTTS)")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, default=None,
                        help="Output WAV file path (default: stdout as JSON)")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Target device: NPU, CPU, GPU")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Speaker reference audio or voice name")
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Output sample rate")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}", file=sys.stderr)
        sys.exit(2)

    device = args.device.upper()

    # Try CosyVoice (OpenVINO GenAI) path first
    try:
        import openvino_genai as ov_genai
        print(f"[TTS] Using OpenVINO GenAI on device={device}", file=sys.stderr)

        # CosyVoice via OpenVINO GenAI pipeline
        pipe = ov_genai.LLMPipeline(args.model_path, device)

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 4096
        config.temperature = 0.7
        config.do_sample = True

        audio_chunks = []
        def streamer(subtoken: str) -> bool:
            sys.stdout.write(subtoken)
            sys.stdout.flush()
            return False

        pipe.generate(args.text, config, streamer=streamer)
        print(file=sys.stderr)

        # If OpenVINO pipeline produced audio output, it's handled internally
        print(json.dumps({"status": "ok", "text": args.text}), file=sys.stderr)
        return

    except ImportError:
        print("[TTS] openvino_genai not available, trying CosyVoice PyTorch fallback...", file=sys.stderr)
    except Exception as e:
        print(f"[TTS] OpenVINO GenAI failed: {e}, trying PyTorch fallback...", file=sys.stderr)

    # Fallback: CosyVoice via PyTorch / HuggingFace transformers
    try:
        import numpy as np
        import torch

        print(f"[TTS] Loading CosyVoice model from {args.model_path}", file=sys.stderr)

        # Try CosyVoice2 style
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            model = CosyVoice2(args.model_path, load_jit=False, load_trt=False)
            available_speakers = model.list_available_spks()
            speaker = args.speaker if args.speaker and args.speaker in available_speakers else available_speakers[0]
            print(f"[TTS] Available speakers: {available_speakers}, using: {speaker}", file=sys.stderr)

            all_samples = []
            for result in model.inference_sft(args.text, speaker, speed=args.speed):
                tts_audio = result["tts_speech"]
                all_samples.append(tts_audio.numpy().flatten())

            if all_samples:
                audio = np.concatenate(all_samples)
                sample_rate = 22050
            else:
                print("Error: No audio generated", file=sys.stderr)
                sys.exit(4)

        except ImportError:
            # Try original CosyVoice
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice
                model = CosyVoice(args.model_path)
                available_speakers = model.list_available_spks()
                speaker = args.speaker if args.speaker and args.speaker in available_speakers else available_speakers[0]
                print(f"[TTS] Using CosyVoice, speaker: {speaker}", file=sys.stderr)

                all_samples = []
                for result in model.inference_sft(args.text, speaker):
                    tts_audio = result["tts_speech"]
                    all_samples.append(tts_audio.numpy().flatten())

                audio = np.concatenate(all_samples) if all_samples else np.array([])
                sample_rate = 22050

            except ImportError:
                print("Error: Neither CosyVoice2 nor CosyVoice found. Install with: pip install cosyvoice", file=sys.stderr)
                sys.exit(5)

        if args.output:
            write_wav(args.output, audio.tolist(), sample_rate)
            print(f"[TTS] Audio saved to {args.output} ({len(audio)} samples, {sample_rate}Hz)", file=sys.stderr)
        else:
            # Output metadata to stdout (audio was streamed or saved)
            print(json.dumps({
                "status": "ok",
                "samples": len(audio),
                "sample_rate": sample_rate,
                "duration_sec": round(len(audio) / sample_rate, 2),
                "text": args.text,
            }))

        return

    except ImportError as e:
        print(f"[TTS] PyTorch/CosyVoice not available: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[TTS] CosyVoice inference failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    # Final fallback: ChatTTS
    try:
        import ChatTTS
        import numpy as np

        print("[TTS] Trying ChatTTS fallback...", file=sys.stderr)
        chat = ChatTTS.Chat()
        chat.load(compile=False)

        texts = [args.text]
        wavs = chat.infer(texts)

        if wavs and len(wavs) > 0:
            audio = wavs[0]
            sample_rate = 24000
            if args.output:
                write_wav(args.output, audio.flatten().tolist(), sample_rate)
                print(f"[TTS] Audio saved to {args.output}", file=sys.stderr)
            else:
                print(json.dumps({
                    "status": "ok",
                    "samples": len(audio.flatten()),
                    "sample_rate": sample_rate,
                    "duration_sec": round(len(audio.flatten()) / sample_rate, 2),
                    "text": args.text,
                }))
        return

    except ImportError:
        print("Error: ChatTTS not installed. Install with: pip install chattts", file=sys.stderr)
    except Exception as e:
        print(f"Error: ChatTTS inference failed: {e}", file=sys.stderr)

    print("Error: No TTS backend available. Install one of: openvino_genai, cosyvoice, chattts", file=sys.stderr)
    sys.exit(6)


if __name__ == "__main__":
    main()
