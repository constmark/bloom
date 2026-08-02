# Pinned Trained-Model Validation

Bloom keeps model weights out of its source tree and release archives. Its
trained-model gates download exact official checkpoints into external caches,
verify them before execution, and exercise the release binaries on CPU.

## Pinned models

| Field | Qwen2 profile | Qwen3 profile | SmolLM2 GGUF profile | SmolLM2 Safetensors profile | MiniLM embedding profile |
| --- | --- | --- | --- | --- | --- |
| Repository | `Qwen/Qwen2-0.5B-Instruct-GGUF` | `Qwen/Qwen3-0.6B-GGUF` | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF` | `HuggingFaceTB/SmolLM2-360M-Instruct` | `sentence-transformers/all-MiniLM-L6-v2` |
| Revision | `198f08841147e5196a6a69bd0053690fb1fd3857` | `23749fefcc72300e3a2ad315e1317431b06b590a` | `593b5a2e04c8f3e4ee880263f93e0bd2901ad47f` | `a10cc1512eabd3dde888204e902eca88bddb4951` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| File | `qwen2-0_5b-instruct-q4_0.gguf` | `Qwen3-0.6B-Q8_0.gguf` | `smollm2-360m-instruct-q8_0.gguf` | `model.safetensors` | `model.safetensors` |
| Size | `352969408` bytes | `639446688` bytes | `386404992` bytes | `723674912` bytes | `90868376` bytes |
| SHA-256 | `aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201` | `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e` | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| Declared license | `Apache-2.0` | `Apache-2.0` | `Apache-2.0` | `Apache-2.0` | `Apache-2.0` |
| License evidence | `LICENSE` | `LICENSE` | `README.md` model card | `README.md` model card | `README.md` model card |
| Evidence SHA-256 | `c156170b718ec29139d3653d40ed1986fd92fb7e0959b5c71f3c48f62e6636f4` | `5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd` | `5d65a9f761fb64443b565d247593aa3d4ec0f06633a0e16c0231625cdba6a7cd` | `6b88794416ac9da8f254ebb0bec228967a2bdd0badf9a2853863928b25facd95` | `dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0` |

The immutable Qwen2 [model file](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/blob/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf)
and [license](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/blob/198f08841147e5196a6a69bd0053690fb1fd3857/LICENSE),
and the immutable Qwen3 [model file](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf)
and [license](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/23749fefcc72300e3a2ad315e1317431b06b590a/LICENSE),
are maintained by the official Qwen organization. The immutable SmolLM2
[model file](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/blob/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf)
and [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/blob/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/README.md)
are maintained in the official HuggingFaceTB organization; this repository
declares Apache-2.0 in the model card rather than shipping a separate license
file. Bloom verifies every evidence payload independently and does not infer
trust from a mutable branch name.

The native package uses the immutable SmolLM2
[Safetensors weights](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/model.safetensors)
and [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/README.md).
It additionally pins `config.json` (`224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee`),
`tokenizer.json` (`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`),
`tokenizer_config.json` (`4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df`),
and `special_tokens_map.json` (`2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52`).

The encoder profile uses the immutable MiniLM
[Safetensors weights](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors)
and [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md).
The gate also pins every runtime configuration and Sentence-Transformer module
descriptor. It requires the published 384-dimensional mean-pooling contract
and 256-token task limit before Bloom loads the checkpoint.

## Reproduce

Run all profiles:

```bash
./scripts/test_trained_qwen2_runtime.sh
./scripts/test_trained_qwen3_runtime.sh
./scripts/test_trained_llama_runtime.sh
./scripts/test_trained_safetensors_runtime.sh
./scripts/test_trained_embedding_runtime.sh
```

The thin model-specific profiles provide immutable provenance and prompt
settings to `scripts/test_trained_gguf_runtime.sh`. By default, they cache the
models under `/tmp/bloom-trained-qwen2-runtime`,
`/tmp/bloom-trained-qwen3-runtime`, `/tmp/bloom-trained-llama-runtime`, and
`/tmp/bloom-trained-safetensors-runtime`. The MiniLM profile uses
`/tmp/bloom-trained-embedding-runtime`; set
`BLOOM_TRAINED_EMBEDDING_CACHE` to override it. Set
`BLOOM_TRAINED_MODEL_CACHE` to another external text-model cache directory, or
pass `--model-path` to an existing exact file or package directory.
`--skip-build` uses explicitly selected release binaries after the profile has
still verified the model and license. Run `--help` for all binary and
benchmark-output overrides.

Each text profile fails unless all of the following are true:

- the immutable license-evidence payload, model size, and model SHA-256 match;
- Bloom routes the exact architecture and package layout to its matching
  native or quantized loader;
- release `bloom_infer` returns exactly `Bloom` for a deterministic
  system-plus-user instruction;
- release `bloom_bench` reports positive CPU throughput and a benchmark object
  containing hardware, timing, memory, dtype, and quantization evidence;
- release `bloom_server` returns exactly `Bloom` through model-backed OpenAI
  buffered/SSE and Ollama buffered/NDJSON chat smokes.

The MiniLM profile instead requires:

- every weight, tokenizer, configuration, pooling, module, and license-evidence
  payload to match its immutable size and SHA-256;
- manifest routing to `Bert` with `bloom_task=embedding`, F32 weights, a
  256-token task limit, validated `mean`/384-dimensional/L2 Sentence
  Transformers metadata, and no KV-cache allocation;
- release output to be finite, L2-normalized, and exactly 384-dimensional;
- variable-length embedding and rerank inputs to retain exact request order
  through attention-mask-aware native batching;
- a paraphrase to outrank an unrelated sentence by at least `0.50` cosine in
  both embeddings and bi-encoder reranking;
- OpenAI embeddings/reranking, current and legacy Ollama embeddings, and the
  optional pinned SDK decoders to succeed;
- OpenAI Chat, Completions, and Responses plus Ollama Chat and Generate to
  reject the encoder with HTTP 422 instead of producing empty text.

The Qwen2 profile also crosses synthesized GGUF tokenizer behavior, including
the control and user-defined tokens required for ChatML boundaries. The Qwen3
profile crosses the architecture-specific loader and disabled-thinking ChatML
form. Bloom does not expose a distinct reasoning channel yet, so its Qwen3 CLI
and server prompts prefill the official empty reasoning block and keep raw
`<think>` markers out of ordinary answer text. Explicit Ollama thinking requests
remain rejected rather than being silently downgraded. The SmolLM2 profile
crosses Llama-architecture loading, bounded chat-template classification, and
the model's ChatML contract.

The native SmolLM2 profile crosses a real Hugging Face directory rather than a
generated fixture. It requires exact required-file enumeration, bounded
`tokenizer_config.json` classification, BF16 storage identification, F32 CPU
runtime accounting, derived 64-dimensional attention heads, and an exact
1,447,349,824-byte runtime weight estimate. Explicit CPU F16 or BF16 selection
fails before model construction because the current Candle CPU matmul path
requires F32 for these weights.

Bloom treats embedded GGUF chat-template source as inert metadata. It does not
execute model-provided Jinja or other template code. A bounded classifier
selects hard-coded SmolLM2, generic ChatML, Llama 2, Llama 3, or Gemma
formatters; unknown templates use a conservative family fallback.

The models and licenses remain outside the workspace. Packaging validation also
rejects accidental model-weight inclusion.

## Reference CPU observations

All profiles passed on 2026-08-02 on the repository's two-vCPU x86-64
development host with 3.6 GiB RAM and no GPU:

| Metric | Qwen2 Q4_0 | Qwen3 Q8_0 | SmolLM2 Q8_0 | SmolLM2 BF16 storage / F32 runtime |
| --- | ---: | ---: | ---: | ---: |
| Model load | 3.85 s | 4.66 s | 1.95 s | 1.06 s |
| TTFT | 8796 ms | 12810 ms | 9004 ms | 1060 ms |
| TBT | 911 ms | 1200 ms | 380 ms | 204 ms |
| Throughput | 1.10 tokens/s | 0.83 tokens/s | 2.63 tokens/s | 4.90 tokens/s |
| Generated benchmark tokens | 2 | 2 | 2 | 2 |
| Estimated model and working memory | 394557804 bytes | 732751484 bytes | 446017011 bytes | 1613056326 bytes |
| Observed process RSS high-water mark | 1162305536 bytes | 1772445696 bytes | 634052608 bytes | 2271563776 bytes |
| Semantic output | `Bloom` | `Bloom` | `Bloom` | `Bloom` |
| OpenAI adapter | buffered and streaming passed | buffered and streaming passed | buffered and streaming passed | buffered and streaming passed |
| Ollama adapter | buffered and streaming passed | buffered and streaming passed | buffered and streaming passed | buffered and streaming passed |

The MiniLM profile passed on the same host with these encoder-specific values:

| Metric | Observation |
| --- | ---: |
| Ready after process start | 540 ms |
| Three-vector release batch | 36 ms |
| Native vector width | 384 |
| Paraphrase cosine | 0.765774 |
| Unrelated cosine | 0.019080 |
| Similarity and rerank margin | 0.746694 |
| Estimated weights | 90868376 bytes |
| Estimated weights plus workspace | 99955213 bytes |
| KV cache | 0 bytes |
| Model task limit | 256 tokens |
| Generation routes rejected | 5 |
| Context reject/truncate checks | 3 |

These values document one constrained host, not a universal performance
promise. Re-run the profiles and retain their emitted JSON when evaluating
another CPU, operating system, compiler, backend, or quantization.
