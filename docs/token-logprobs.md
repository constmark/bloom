# Token log probabilities

Bloom exposes selected-token log probabilities and bounded top alternatives on
the direct CPU Candle text path. Chat Completions and legacy Completions support
buffered and SSE responses, `n=1..8`, and legacy prompt arrays. Rust, JSON C ABI
and Python callers can request the same native metadata. The CLI exposes
`--logprobs N` (0–20 alternatives) with `--json` to retain the metadata:

```bash
bloom_infer --model /path/to/model.gguf --device cpu --prompt 'Hello' \
  --max-tokens 16 --logprobs 2 --json
bloom_infer --model /path/to/model.gguf --device cpu --prompt 'Hello' \
  --max-tokens 16 --logprobs 2 --json --stream
```

Buffered output includes `ModelOutput.logprobs`; streaming emits native
`TokenLogprobs` events alongside text and usage NDJSON. `infer.logprobs` in
configuration and `logprobs` in a JSON prompt request use the same integer
contract. Explicit CLI settings override configuration; JSON request settings
override both. `infer.logprobs` only applies to `--json` runs, so a configured
default does not break text, `-i`, `--bench`, `--inspect`, or `--list-engines`;
an explicit `--logprobs` or JSON-request `logprobs` with non-JSON output is
rejected rather than dropping metadata.

## Requests and responses

Chat uses `logprobs: true` with optional `top_logprobs: 0..20`. Omitted or zero
top count returns selected tokens only. Legacy Completions uses integer
`logprobs: 0..5`. Each request must reserve at most 4,096 output tokens per
candidate. The fields follow the official
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [legacy Completions reference](https://developers.openai.com/api/reference/resources/completions/methods/create).

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 16,
  "temperature": 0,
  "logprobs": true,
  "top_logprobs": 3
}
```

Chat choices contain `logprobs.content` records with `token`, `bytes`, `logprob`
and `top_logprobs`; `refusal` is null. Each alternative has its own display text,
bytes and score. Native records also retain `token_id`; see the
[Python/FFI guide](ffi-python.md). Legacy choices contain parallel `tokens`,
`token_logprobs`, `top_logprobs` and `text_offset` arrays.

SSE can emit metadata-only choice frames before the corresponding text delta.
Clients must accumulate both fields independently; a single text delta may
complete several byte-fragment tokens. Chat supports aggregate terminal usage
via `stream_options.include_usage`; legacy streaming retains its existing
contract without a usage chunk. Buffered responses include usage in both
protocols. Stream shapes follow the official
[Chat streaming reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events).

## Probability and byte semantics

Scores are finite, nonpositive log-softmax values after presence/frequency
penalties and logit bias, before temperature, top-k or top-p filtering.
Greedy decoding therefore still reports a real model probability. Alternatives
are sorted by descending probability and bounded by available vocabulary size.
Bloom returns the actual selected-token score even outside the alternatives;
it does not substitute the OpenAI reference's `-9999` sentinel.

Exact raw bytes require a direct ByteLevel decoder or decoder-free WordLevel
tokenizer. WordLevel records include contextual space separators. ByteLevel
records may split a Unicode character; their display strings use replacement
characters, while concatenating selected `bytes` reproduces the final UTF-8
text exactly. An invalid or unfinished UTF-8 sequence at EOS/token limit fails
the request. A stream may already have exposed earlier metadata, but it never
reports successful completion or aggregate usage after that failure.

Legacy offsets count Unicode scalar values in generated text, starting at zero.
Byte-fragment tokens belonging to one character share the same offset. Legacy
has no raw byte field: distinct fragments can share a replacement-character
display key. Its alternative map keeps the highest score per display key,
then preserves the selected token's own score at its key. Use Chat or the
native API when exact fragment identities matter.

Sampled EOS has no selected-token record or text; EOS may still appear among
top alternatives with empty special-token bytes. HTTP completion usage includes sampled EOS
exactly once per stopped candidate, including first-token EOS and EOS at the
last reserved token. A first-token EOS returns empty metadata/text, `stop`, and
one completion token. Native `GenerationFinished` explicitly identifies EOS;
no consumer infers EOS from a difference between text and usage counts. Missing
or inconsistent native usage or termination metadata fails closed. The same
accepted-token accounting applies to ordinary direct generation without
logprobs; see [generation metadata](generation-metadata.md).

## Admission and resource limits

The loaded model must explicitly advertise native logprobs support. GPU devices,
other engines, speculative decoding, continuous batching, multimodal input,
structured output, active function tools, custom stop sequences and legacy
`echo: true` are rejected. Responses and Ollama adapters do not expose this
capability. Requests never silently fall back to text-only success.

Each native token is limited to 1,024 raw bytes and 4,096 display bytes; each
candidate is limited to 8 MiB of native metadata. The HTTP worker additionally
caps serialized events for the complete request at 16 MiB, including metadata
and all candidates. Metadata must be finite, correctly ordered and consistent
with emitted text before a candidate can finish successfully.

## Reproducible validation

`scripts/logprobs_runtime_smoke.py` runs offline against a supplied local model
and server binary, saves every request/response and records their digests.
It checks buffered/SSE equality, alternatives, byte reconstruction, offsets,
multi-choice usage and pinned SDK decoding. The
[runtime validation](logprobs-runtime-validation-2026-11-13.md) records tiny native,
trained Qwen2, Unicode and first-EOS acceptance separately.
