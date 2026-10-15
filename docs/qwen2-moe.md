# Qwen2-MoE validation

Bloom's Candle backend recognizes `model_type: "qwen2_moe"` and routes it to
the native sparse Qwen executor. The loader accepts both checkpoint layouts
published by Hugging Face:

- legacy `model.layers.*.mlp.experts.<n>.{gate,up,down}_proj.weight` tensors;
- packed `model.layers.*.mlp.experts.{gate_up_proj,down_proj}` tensors.

Safetensors are memory-mapped for loading, then expert weights are materialized
in CPU memory. Packed projections are copied into compact per-expert tensors:
Candle's `to_device` copies the entire backing storage, so a contiguous view
alone would upload the whole packed layer for each selected expert. This keeps
the model directory unchanged; it is not disk-backed expert paging.
The model runs on one Candle CPU or CUDA device. Its extractable KV runtime
also supports CacheMesh, including ICMS/Ledger restart recovery. The existing
parallelism types describe future placement APIs; they do not shard experts
today.

An optional GPU expert LRU is enabled with `BLOOM_MOE_EXPERT_CACHE_MIB` (default
`0`, disabled). Its MiB budget is divided across all decoder layers and covers
persistent routed-expert Standard weights. Attention, embedding, shared experts,
KV, activations, and streaming fallback buffers are outside this budget. Hits
reuse compact GPU tensors; misses upload one selected expert at a time. Busy
entries are not evicted. CPU execution and quantized projections use the existing
streaming path. Qwen2-MoE IFB requests fork one empty-KV template: immutable CPU
weights and the expert LRU are shared, while each request owns independent KV.
The startup wrapper is released after the scheduler captures the template.
File identity is checked before and after each request fork. Independently
loaded runtimes still have separate budgets; this is not a process-wide VRAM
limit. Other model architectures retain their existing reload behavior.

Use `RUST_LOG=info,bloom_moe_expert_cache=debug` to record layer/expert hits,
misses, upload bytes, evictions, current bytes, and the per-layer budget.
CPU/GPU mixed placement of BF16 checkpoints is not yet supported by Candle's
CPU matmul path. Keep all Qwen2.5 BF16 layers executing on the GPU during this
experiment; their weights still stream from CPU memory. Server memory admission
remains conservative and can reject models larger than VRAM even when the
executor's streaming working set would fit.

The checked local fixture was downloaded from
[`hf-internal-testing/tiny-random-Qwen2MoeForCausalLM`](https://huggingface.co/hf-internal-testing/tiny-random-Qwen2MoeForCausalLM), revision
`f736f270816032b3c721f7422c62dea1381f49d7`. It is a deterministic architecture
fixture, not a quality claim. The downloaded files are under
`/opt/models/tiny-random-Qwen2MoeForCausalLM`. A trained Apollo checkpoint is
also installed at `/opt/models/Apollo-MoE-0.5B`; its SHA-256 is recorded in
`download-provenance.json` and its `UpcyclingQwen2MoeForCausalLM` layout was
validated through the same ICMS/Ledger restart path.

The newest recent-model validation uses
[`bayliner1980/Qwen2.5-1.5B-MoE-A0.9B`](https://huggingface.co/bayliner1980/Qwen2.5-1.5B-MoE-A0.9B),
created 2026-10-13, at revision
`6b5b6a39ee208ada9916f0a4e1273de242a3637e`. It is installed at
`/opt/models/Qwen2.5-1.5B-MoE-A0.9B`; its 3,557,393,064-byte Safetensors file
has SHA-256 `e946a071456514ea026a8441171329b39b3105f0a19f6e774d935d481386fcfc`,
with the complete manifest in `PROVENANCE.json`. The checkpoint has 28 sparse
layers, 32 routed experts, top-8 routing, and native BF16 weights. On the local
RTX 3060 Laptop GPU (CUDA compute capability 8.6), the full chunked-prefill
ICMS/Ledger test passed with `--dtype bf16`: cold, warm, Bloom-restart, and
metadata-restart replay all produced deterministic output, with 616 L3
offloads, 224 L3 hits, and 66 GPU sampling observations. Bloom selects BF16 by
default for this checkpoint on CUDA and rejects an explicit FP16 override,
because conversion produces non-finite logits on this GPU; use BF16 on Ampere
GPUs for this model.

Run the native path:

```bash
cargo run --release -p bloomai-app --bin bloom_infer -- \
  --model /opt/models/tiny-random-Qwen2MoeForCausalLM \
  --device cpu --prompt "Hello" --max-tokens 4
```

Run the HTTP generation path (OpenAI and Ollama routes share the same loaded
runtime):

```bash
cargo run --release -p bloomai-server --bin bloom_server -- \
  --model /opt/models/tiny-random-Qwen2MoeForCausalLM \
  --device cpu --host 127.0.0.1 --port 3000
```

`--enable-ifb` and `--enable-cachemesh` are compatible with Qwen2-MoE. For the
ICMS/Ledger end-to-end check, use:

```bash
tests/adapters/bloom/run_bloom_icms_ledger_e2e.sh \
  --model /opt/models/Apollo-MoE-0.5B --device cpu --dtype f32 \
  --context-size 256 --max-tokens 2 --fill-requests 1
```

Quantized Qwen2-MoE GGUF has no Candle loader yet and is rejected with an
explicit diagnostic.

For the local NVIDIA CUDA path, build the adapter with `--features cuda` and
retain the trained checkpoint's native BF16 precision:

```bash
RUSTUP_TOOLCHAIN=1.97.1 CUDA_COMPUTE_CAP=86 \
tests/adapters/bloom/run_bloom_icms_ledger_e2e.sh \
  --model /opt/models/Apollo-MoE-0.5B --device gpu --dtype bf16 \
  --context-size 256 --max-tokens 2 --fill-requests 1
```

For the recent Qwen2.5 MoE checkpoint, retain its native BF16 precision and
raise the memory admission fraction when other GPU jobs are resident:

```bash
BLOOM_MEMORY_UTILIZATION=0.90 BLOOM_GPU_LAYERS=28 \
BLOOM_MOE_EXPERT_CACHE_MIB=512 \
RUSTUP_TOOLCHAIN=1.97.1 CUDA_COMPUTE_CAP=86 \
tests/adapters/bloom/run_bloom_icms_ledger_e2e.sh \
  --model /opt/models/Qwen2.5-1.5B-MoE-A0.9B --device gpu --dtype bf16 \
  --context-size 256 --max-tokens 8 --fill-requests 1 --chunked-prefill
```
