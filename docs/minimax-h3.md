# MiniMax H3 external CLI

`bloom_h3` is an experimental, dedicated audio/video command using a local
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
CUDA executable. It is not a registered Candle model or an HTTP API backend.
Use an sd.cpp revision with the
[MiniMax H3 runtime](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/minimax_h3.md)
and all four matching model components. A generic Qwen checkpoint is not a
substitute for the H3 text encoder.

```sh
cargo build -p bloomai-app --no-default-features --bin bloom_h3

target/debug/bloom_h3 \
  --sd-cli /path/to/sd-cli \
  --diffusion-model /models/minimax_h3_fl2va-Q4_K_M.gguf \
  --llm /models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --prompt 'A cat plays a piano. Clear piano notes accompany the motion.' \
  --output /outputs/cat.mp4
```

Bloom itself needs no CUDA feature for this command; the external executable
provides GPU execution. FFmpeg with libx264/AAC and ffprobe must be installed.
`--ffmpeg` and `--ffprobe` can select explicit executables. Strict-security
deployments must allow the selected sd-cli, ffmpeg and ffprobe paths through
`BLOOM_ALLOWED_RUNNERS`.

Defaults are 320×192 pixels, 22 frames, 24 fps, 20 Euler steps, CFG 1, seed 42,
two CPU threads and a 7200-second total deadline. The runner uses CUDA device 0,
disk-backed parameters, mmap, flash attention, tiled VAE decoding and a 4 GiB
managed GPU budget. It does not apply the CPU-offload shortcut, which would
replace disk placement. `--max-vram` is a managed allocation budget, not a hard
limit on GPU or host memory; deployments can use a memory-limited OS scope.

Width and height must be multiples of 32 between 32 and 1024. `--frames` must
be `17k+5`, from 5 to 243, and the total canvas is limited to 64 megapixels.
These bounds constrain output buffers; they do not guarantee that every
configuration fits a given GPU. A 22-frame result is a short inference check,
not evidence of production video quality or longer-clip performance.

The command writes a temporary AVI with audio, converts it to MP4, checks
dimensions/frame count/24 fps/stereo 32 kHz audio/duration using ffprobe, and
fully decodes both streams using FFmpeg. A WAV sidecar is accepted if the
runtime produces one instead of muxed AVI audio. Missing audio is an error.
Only validated files are published as `FILE.mp4` and `FILE.mp4.json`; existing
files are never replaced. The JSON records parameters, media details, output
SHA-256 and bounded runtime diagnostics. Component paths are recorded, but
component provenance and hashes must be verified separately at deployment.

Ctrl-C and termination signals cancel the task. On Unix, the bounded worker
owns a process group, which is killed and reaped on cancellation or timeout.
Temporary output is cleaned up on failure. The deadline includes inference,
conversion and validation. Runtime stdout/stderr are drained continuously with
at most 64 KiB of each retained in the successful result.

```sh
cargo test -p bloomai-app --no-default-features --bin bloom_h3
cargo test -p bloomai-engine --no-default-features --lib minimax_h3
cargo test -p bloomai-engine --no-default-features --lib core::process::tests
# Requires FFmpeg. This is synthetic media adapter coverage, not H3 inference:
cargo test -p bloomai-engine --no-default-features --lib \
  synthetic_audio_video_survives_conversion_and_validation -- --ignored
```

Real-model acceptance must run this command with the actual H3 weights on the
target GPU and inspect the resulting picture and sound. Unit or synthetic
media tests alone do not establish successful H3 deployment.

## Recorded hardware validation (2026-11-13)

The dedicated command completed a real T2VA run on an RTX 3060 Laptop 6 GiB
with sd.cpp `b9d7176bf88fe465576e0aafbe47ba253ffcfc2c` (CUDA 12.4, sm86).
The four checkpoints were verified against their full SHA-256 hashes:
`leejet/MiniMax-H3-GGUF@d9c4c6312b4728a68a15a35626d84775a6523783`
provided the pruned FL2VA Q4_K_M and H3 Qwen3-VL-32B Q4_K_M;
`Comfy-Org/MiniMax-H3@54472221d27c695fca9621b0c3ed52f06c4af707`
provided the FP16 video VAE and FP32 audio VAE.

The default 320×192, 22-frame, seed-42, 20-step configuration produced a
0.916667-second MP4 with 24 fps video and stereo 32 kHz audio. Both streams
passed full decoding. Sampled frames showed the requested red toy car moving
rightward across a wooden table. Audio samples were finite and nonzero;
subjective audio quality was not evaluated.

Bloom's complete task took 1421.80 seconds. Sampled total GPU memory peaked
at 5021 MiB, including other applications; GNU time reported a maximum RSS
of 3,541,688 KiB. The Linux scope used an 8 GiB host-memory limit and no swap.
Weights were streamed from an external SSD, so this demonstrates execution
within the available memory, with slow generation. Higher resolutions,
longer clips, conditioning modes and Turbo configurations remain unvalidated.

Output SHA-256:
`5d5e09f812e339f1c84a48af58f10fa3f6f52212d3aff02d917ab882f47ff9d6`.
On the validation machine, evidence is stored under
`/home/lsx/models/minimax-h3-sdcpp/runs/smoke-20261113-103713` (MP4,
`.mp4.json`, `.run.json`, `.metrics.jsonl`, `.log`, frame samples and
`.mp4.quality-check.json`). Runtime diagnostics in the JSON are truncated
to their documented bounds; the artifact metadata and full-decode checks
provide the completion evidence.
