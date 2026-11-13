# Packed RTN export and native loading

`bloom_quantize` writes the versioned **Bloom RTN v1** storage format. INT4
packs two weights per byte; INT8 stores one weight per byte. An explicit
`--model-package` option converts a dense Qwen2, Qwen3 or Llama Safetensors
model directory into a package that Bloom's native Candle loader can execute.
Without that option the result is a self-describing tensor bundle, not a model.

```sh
# Standalone tensors; no tokenizer or model configuration is needed.
cargo run --locked --bin bloom_quantize -- \
  --input model.safetensors --output tensors-rtn.safetensors --method rtn_int8

# A loadable native text model; group size must divide each matrix width.
cargo run --locked --bin bloom_quantize -- \
  --input /path/to/hf-model --output /path/to/new-model \
  --method rtn_int4 --group-size 128 --model-package
cargo run --locked --bin bloom_infer -- \
  --model /path/to/new-model --device cpu --prompt "Hello" --max-tokens 16
```

This is round-to-nearest quantization, not activation-calibrated AWQ/GPTQ,
GGUF conversion, or a packed matrix multiplication kernel. At load time Bloom
validates the bundle and restores floating-point tensors before invoking its
native model. CPU uses F32; other device paths use the usual native precision
policy. The storage reduction does **not** imply reduced resident model memory
or improved decode speed. Admission estimates conservatively budget compressed
source, floating-point expansion and conversion buffers, and disable the mmap
residency discount. GPU execution and trained-model quality are not validated
by the CPU fixture described below.

## Package and publication contract

The destination must not exist. File input requires a new file destination;
directory input requires a new directory. The source directory uses inference's
strict shard/index validation. Quantized packages cannot be quantized again.
`--model-package` requires an unquantized `config.json`, a `tokenizer.json`, and
one of the supported dense model types. The config gains
`quantization_config: {"quant_method":"bloom_rtn","version":1,"bits":4|8}`.
Optional tokenizer, generation and license data files are copied after
regular-file and size checks. Python code and stale `bloom.json` are not copied.
A multi-shard package gets a new index with the actual exported tensor names,
shard ownership and data-byte total; stale source indexes are never reused.

All artifacts are prepared in temporary files before publication. Existing
outputs and sources are never overwritten. Failed publication removes only
files created by that operation. A process crash during directory publication
can leave an incomplete output directory; it must be removed explicitly before
retrying. This implementation does not promise crash-atomic directory commits.

## RTN v1 layout

A quantized matrix originally named `name` has these tensors:

| Tensor | Type and shape | Meaning |
| --- | --- | --- |
| `name.rtn_meta` | U32 `[5]` | `[version=1, bits, group_size, rows, cols]` |
| `name.qweight` (INT4) | U8 `[rows, ceil(cols/2)]` | Even column in low nibble, odd column in high nibble; unused final high nibble is zero |
| `name.qweight` (INT8) | U8 `[rows, cols]` | Symmetric signed value plus 128; valid codes 1–255 |
| `name.scales` | F32 `[rows, cols/group_size]` | Finite strictly positive quantization scale |
| `name.zeros` (INT4 only) | F32 `[rows, cols/group_size]` | Finite unrounded group minimum/offset |

INT8 uses one group per row and reconstructs `(q - 128) * scale`.
INT4 reconstructs `q * scale + offset`. Constant groups use scale 1; zero and
constant values therefore remain reconstructible. Odd matrix widths are
supported if the group size divides the width. The decoder checks the version,
bit count, group dimensions, tensor names/dtypes/shapes, scale/offset finiteness,
padding nibble, valid INT8 codes, and finite restored values before publishing
model weights. `bloomai_engine::executor::rtn::dequantize_tensors` also restores
standalone bundles without a model configuration. A declared RTN package must
contain at least one versioned quantized matrix across its shards; norm-only
shards are permitted. Standalone bundles require their own versioned metadata,
and neither legacy unversioned quantization tensors nor float-only packages
can masquerade as RTN.

Nonempty floating-point matrices with `weight` or `proj` in their names are
quantized; other tensors are retained. F16/BF16/F32/F64 inputs use F64 arithmetic
and F32 scale/offset storage. F64 values must be exactly representable as F32.
Nonfinite matrix values, invalid groups and generated-name collisions fail.
The former unversioned, unpacked RTN export is not accepted as a v1 package.

## Reproducible CPU execution check

The generated checkpoint is untrained: it proves actual native loading and
forward execution, not useful language quality. No downloaded weights are
required. Use fresh output directories for each run.

```sh
cargo test -p bloomai-app --bin bloom_quantize
cargo run --locked -p bloomai-engine --example generate_tiny_qwen2_fixture -- \
  --output /tmp/bloom-rtn-source --sharded
cargo run --locked --bin bloom_quantize -- \
  --input /tmp/bloom-rtn-source --output /tmp/bloom-rtn-int4 \
  --method rtn_int4 --group-size 8 --model-package
cargo run --locked --bin bloom_infer -- \
  --model /tmp/bloom-rtn-int4 --device cpu --prompt hello \
  --temperature 0 --max-tokens 3
# Repeat export/inference with --method rtn_int8 and a fresh destination.
```

The tensor regressions check rounding error across input precisions, negative
and constant groups, odd-width packing, malformed metadata and source/output
preservation. Model checks should cover both single and indexed multi-shard
packages and compare greedy output with the unquantized fixture.
