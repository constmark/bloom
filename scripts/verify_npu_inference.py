#!/usr/bin/env python3
"""Verify actual Intel NPU inference with OpenVINO GenAI.

This script:
1. Checks NPU device availability via OpenVINO Core
2. Loads a model on NPU (falls back to CPU if NPU unavailable)
3. Runs real inference and reports device used
"""
import sys
import time

def main():
    # 1. Check OpenVINO devices
    from openvino import Core
    core = Core()
    devices = core.available_devices
    print(f"[NPU-TEST] OpenVINO available devices: {devices}")

    target_device = "NPU" if "NPU" in devices else "CPU"
    print(f"[NPU-TEST] Target device: {target_device}")

    if target_device == "NPU":
        print("[NPU-TEST] ✓ Intel NPU detected and available")
    else:
        print("[NPU-TEST] ✗ NPU not available, falling back to CPU")

    # 2. Load model with openvino_genai
    import openvino_genai as ov_genai

    model_path = r"D:\models\qwen\Qwen2.5-0.5B-Instruct-OpenVINO"
    print(f"[NPU-TEST] Loading model from: {model_path}")

    load_start = time.time()
    pipe = ov_genai.LLMPipeline(model_path, target_device)
    load_elapsed = time.time() - load_start
    print(f"[NPU-TEST] Model loaded on {target_device} in {load_elapsed:.2f}s")

    # 3. Run inference
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 32
    config.temperature = 0.7
    config.do_sample = True

    prompt = "What is 2+2? Answer in one word."
    print(f"[NPU-TEST] Running inference on {target_device}...")
    print(f"[NPU-TEST] Prompt: {prompt}")

    infer_start = time.time()
    tokens = []
    def streamer(token: str) -> bool:
        tokens.append(token)
        return False

    pipe.generate(prompt, config, streamer=streamer)
    infer_elapsed = time.time() - infer_start

    result = "".join(tokens)
    print(f"[NPU-TEST] Response: {result}")
    print(f"[NPU-TEST] Inference time: {infer_elapsed:.2f}s")
    print(f"[NPU-TEST] Tokens generated: {len(tokens)}")
    if infer_elapsed > 0 and len(tokens) > 0:
        print(f"[NPU-TEST] Speed: {len(tokens)/infer_elapsed:.1f} tokens/s")

    # 4. Summary
    print(f"\n[NPU-TEST] === Summary ===")
    print(f"[NPU-TEST] Device used: {target_device}")
    print(f"[NPU-TEST] Load time: {load_elapsed:.2f}s")
    print(f"[NPU-TEST] Inference time: {infer_elapsed:.2f}s")
    print(f"[NPU-TEST] Result: PASS (real {target_device} inference succeeded)")

    return 0 if target_device == "NPU" else 1

if __name__ == "__main__":
    sys.exit(main())
