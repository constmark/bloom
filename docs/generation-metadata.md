# Native generation metadata

Native Candle text generation emits cumulative `TokenUsage` events and one
`GenerationFinished` event after all text and requested logprobs:

```json
{"GenerationFinished":{"input_tokens":12,"output_tokens":3,"reason":"eos"}}
```

`input_tokens` is the processed prompt length. `output_tokens` counts accepted
sampled tokens, including sampled EOS, even when EOS is the first token or the
last available token. Speculative drafts rejected by the verifier are excluded.
Text can have fewer tokens after Unicode decoding or HTTP stop filtering;
retokenizing that visible text does not determine native usage.

The reason is `eos` for a sampled model EOS, `length` for exhausting the output
budget, or `stop` for an explicitly reported backend stop. Candle currently
emits `eos` and `length`. EOS on the final budget token remains `eos`. No consumer
infers EOS from token counts or the absence of visible text.

## Lifecycle and validation

A loaded model opts in with `supports_generation_metadata()`. The default is
false for existing adapters. For an opted-in model, the pipeline requires one
metadata event matching the latest cumulative usage. Input counts must stay
constant; output counts must be monotonic and bounded by the requested budget.
`length` must consume the budget, and `eos` must include an accepted token.
After metadata, only optional `Metrics` and terminal `End` are allowed. Missing,
duplicate, inconsistent, or late content events fail the request. The validator
retains the first failure even if a backend ignores a sink error.

`GenerationFinished` is accounting metadata, **not a success marker**. Consumers
must also receive `End` and a successful call result. Cancellation, consumer
failure, or backend errors can interrupt the stream, including after metadata;
none may be converted into successful HTTP termination or final aggregate usage.

## Transport behavior and scope

Direct OpenAI Chat Completions, legacy Completions, and Responses consume the
same native metadata with or without logprobs. EOS/explicit backend stop maps to
`stop` (or `tool_calls` for a validated tool call); budget exhaustion maps to
`length`. Responses maps these to `completed` or `incomplete` respectively.
Each candidate contributes its actual accepted output count; prompt usage is
counted once per input prompt, independent of `n`. Chat streaming reports final
usage when `stream_options.include_usage` is true; legacy Completions streaming
retains its existing protocol without a usage frame. Buffered responses and
Responses include their existing usage fields.

An HTTP custom stop aborts the native sink as soon as the sequence is detected.
Its public finish reason is `stop`; usage retains the latest accepted count,
including tokens consumed to recognize the stop and any hidden suffix within
that emitted chunk. A final native metadata event is not required after this
intentional early abort. Cancellation and unrelated errors remain failures.

CLI streaming JSON preserves the new native event. Buffered CLI generation
consumes the same stream internally, uses exact output counts, and adds a
`generation` object to its JSON output. Rust/FFI/Python buffered `ModelOutput`
is unchanged; the additive native stream variant does not change FFI ABI 2.

This contract applies to direct native Candle text generation and to Qwen3-VL
image/text generation. OpenAI and Ollama image adapters consume the forwarded
multimodal metadata: `length` maps to `length`, and buffered responses report
the native counts. Buffered C ABI/Python runs add the same counts and reason as
an additive `generation` object.
Embedding and cross-encoder paths do not emit generation metadata. IFB retains
its scheduler accounting and finish-reason boundary. Existing external/plugin
adapters remain compatible without opting in: complete-text tokenization and
legacy finish-reason fallback may be approximate. Advertising native metadata
requires emitting and validating the full contract; these fallback paths do
not claim exact EOS accounting.
