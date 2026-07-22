#!/usr/bin/env python3
"""Run local OpenVINO GenAI LLM inference for Bloom on NPU."""

import argparse
import sys
import os

try:
    import openvino_genai as ov_genai
except ImportError as exc:
    print("Error: openvino_genai is not installed in the python environment.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="OpenVINO GenAI LLM streaming inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to OpenVINO IR model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--device", type=str, default="NPU", help="Target hardware: NPU, CPU, GPU")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}", file=sys.stderr)
        sys.exit(2)

    # Standardize device name (upper case)
    device = args.device.upper()

    # Load the LLM Pipeline
    # ov_genai.LLMPipeline requires a path to directory containing openvino_model.xml and tokenizer configs
    try:
        pipe = ov_genai.LLMPipeline(args.model_path, device)
    except Exception as e:
        print(f"Error loading LLM pipeline on device {device}: {e}", file=sys.stderr)
        sys.exit(3)

    # Build Generation Config
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_tokens
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.do_sample = args.temperature > 0.0
    if args.seed is not None:
        config.rng_seed = args.seed

    # Define streaming callback
    def streamer(subtoken: str) -> bool:
        # sys.stdout.write prints to stdout stream
        sys.stdout.write(subtoken)
        sys.stdout.flush()
        # Return False to continue generation, True to stop
        return False

    # Execute generation with streamer
    try:
        pipe.generate(args.prompt, config, streamer=streamer)
    except Exception as e:
        print(f"\nError during generation: {e}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()
