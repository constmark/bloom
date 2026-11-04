# External llama.cpp operations

The `llamacpp` engine delegates execution to a separately installed
`llama-server`. This is an external-runtime route, not evidence for native
Candle kernels or a stable deployment cell. The adapter targets the API of the
[pinned llama.cpp reference](https://github.com/ggml-org/llama.cpp/blob/1b70350c92563449576bcbbf552994b1d1ed847d/tools/server/README.md).
Set `BLOOM_LLAMA_CPP_SERVER` to the reviewed executable.

## Executable operations

| Task | External operation | Bloom result |
| --- | --- | --- |
| Generation | `/completion`, buffered or SSE | Text output and one successful terminal event |
| Constrained generation | `/completion` with `json_schema` | JSON object or the admitted JSON Schema; final value validated before success |
| Tokenization | `/tokenize`, with special-token parsing | Exact model token IDs, including model special tokens |
| Embeddings | `/v1/embeddings`, float encoding and string-array input | One vector per input in input order |
| Cross-encoder reranking | `/reranking`, requesting all documents | One finite score per document in input order |

Task selection is exclusive. Generation models cannot use the encoder routes;
embedding models do not generate text; cross encoders cannot silently fall back
to pooled embeddings. The embedding and rerank capability flags describe
implemented adapter operations, not compatibility with every GGUF checkpoint.

## Trusted model metadata

Use a Bloom model package with `bloom.json` to declare the encoder task. The
existing manifest schema and file validation still apply. An embedding package
includes these parameters:

```json
{
  "bloom_task": "embedding",
  "embedding_pooling": "mean"
}
```

`embedding_pooling` accepts `mean`, `cls`, or `last`; the default is `mean`.
Bloom starts the worker with `--embedding --pooling TYPE`. Token-level pooling
is not projected into one sentence vector.

For a compatible cross-encoder GGUF, declare:

```json
{
  "bloom_task": "rerank",
  "bloom_encoder_kind": "cross_encoder"
}
```

Bloom starts the worker with `--embedding --reranking --pooling rank`.
Speculative generation is rejected for either encoder task. The loaded worker
must support its requested flags and model; startup or endpoint errors propagate
as failures. Automatic detection of every embedding or rank architecture from
raw GGUF metadata is not supplied by these adapter changes.

## Bounds and failure behavior

Encoder batches contain 1–256 strings with at most the engine's 8 MiB text
budget in total. Returned batches must have complete, unique, in-range indices;
embedding dimensions must be positive and consistent. Float32 overflow,
non-numeric values, missing results, and more than 16,777,216 embedding values
are rejected. Sorted rerank results are restored to original document order
before the server applies its ranking projection.

Sampling conversion is shared by buffered and streamed completion. It forwards
`top_k`, presence/frequency penalties, pair-encoded logit bias, and JSON grammar
configuration. The external seed must fit an unsigned 32-bit integer; larger
Rust/Python seeds fail explicitly rather than wrapping to another seed.

A stream must contain a native terminal `stop: true` event. Truncation, external
error events, malformed content, output after that terminal event, and invalid
complete structured output return an error without a Bloom success terminal.
Caller cancellation stops consumption and closes the worker HTTP connection.
Structured streams may already have emitted partial text before final
validation fails, matching the server's existing streaming error contract.
Binary version/help probes use the bounded subprocess runner. HTTP calls apply
one absolute deadline across connect, write and reads, so slow trickle output
cannot restart the timeout. Conflicting or duplicate length/transfer headers,
truncated fixed-length bodies and incomplete chunked framing fail explicitly.

## Validation and remaining evidence

The engine unit suite uses local HTTP fixtures to verify endpoint requests,
Unicode input, exact token IDs, batch ordering, task isolation, malformed
results, seed boundaries, structured output, cancellation, and stream terminal
semantics with and without the Candle feature. The shared final JSON validator
also remains covered by the native Candle structured-output regressions.

These fixtures prove the adapter boundary. They do not supply trained encoder
quality, compatibility across external binary versions, accelerator execution,
performance comparisons, or long-soak evidence. Those remain required for
promoting a specific model/device/runtime combination in the support matrix.
