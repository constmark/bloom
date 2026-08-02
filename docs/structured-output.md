# Structured Output

Bloom supports constrained text generation through the OpenAI-compatible
`response_format` field on `/v1/chat/completions` and `/v1/completions`, and
through `text.format` on `/v1/responses`. The bounded Ollama adapter accepts
`format: "json"` or a direct JSON Schema object on `/api/chat` and
`/api/generate`.
Structured output is experimental and depends on the selected model and engine.
The server always performs final output validation even when an engine cannot
apply token-level grammar filtering.

The scheduler-driven Candle text path applies its token grammar during prefill,
scalar decode, and native batched decode. It reconstructs each parser state from
the scheduler-owned generated-token vector on every step rather than retaining
a request-ID keyed shadow state. Structured requests currently disable
speculative multi-token runs because accepted draft tokens would otherwise need
their own sequential grammar validation and atomic KV rollback.

The request shapes follow the
[OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
Bloom intentionally supports a smaller schema subset and documents every local
limit below.

## JSON object mode

Use `json_object` when any JSON object is acceptable:

```json
{
  "messages": [{"role": "user", "content": "Return a short status."}],
  "response_format": {"type": "json_object"}
}
```

The equivalent Responses request is:

```json
{
  "input": "Return a short status.",
  "text": {"format": {"type": "json_object"}}
}
```

Bloom adds a JSON-only instruction to the formatted prompt. Non-streaming output
that is not a JSON object returns `invalid_response_format`. For streaming
requests, text is emitted as it is generated and a terminal SSE error is emitted
instead of a normal stop chunk if the complete output is invalid. Clients must
therefore retain or discard partial text according to their own policy.

## JSON Schema mode

Use the OpenAI wrapper shape with a bounded schema:

```json
{
  "messages": [{"role": "user", "content": "Summarize the service state."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "service_state",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "ready": {"type": "boolean"},
          "summary": {"type": "string"}
        },
        "required": ["ready", "summary"],
        "additionalProperties": false
      }
    }
  }
}
```

Responses uses the same format fields directly rather than nesting a
`json_schema` wrapper:

```json
{
  "input": "Summarize the service state.",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "service_state",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "ready": {"type": "boolean"},
          "summary": {"type": "string"}
        },
        "required": ["ready", "summary"],
        "additionalProperties": false
      }
    }
  }
}
```

The root type must be `object`. Bloom accepts only the constraints it can
validate consistently:

- `type`: `object`, `array`, `string`, `number`, `integer`, `boolean`, or `null`
- `enum`
- `required`
- `properties`
- boolean `additionalProperties`
- `items`
- annotation strings `$schema`, `title`, and `description`

Unknown keywords and types are rejected before tokenization or inference. In
particular, Bloom does not currently implement `$ref`, `$defs`, composition
keywords, string patterns or lengths, numeric ranges, tuple items, or union
types. Annotation fields do not add constraints.

Admission limits are:

- 64 KiB encoded schema
- 16 nested schema levels
- 1,024 schema nodes
- 256 properties per object
- 256 values per `enum`
- 1,024 characters per annotation string

`required` entries must be unique and must name a declared property. Wrapper
names contain 1 to 64 ASCII letters, digits, underscores, or hyphens. Direct
schema objects remain accepted for backward API compatibility, but new clients
should send the wrapper shown above.

For Responses, `text` may be omitted, empty, or contain one `format` object.
The format may be `text`, `json_object`, or `json_schema`; JSON Schema accepts
`name`, optional bounded `description`, `schema`, and optional boolean `strict`.
Unknown active text or format fields, including unsupported verbosity controls,
fail before runtime admission. Response lifecycle objects echo the normalized
format. Streaming structured output can emit partial text, but Bloom validates
the complete value before success and emits `response.failed` on a violation.

## Bloom UI behavior

The Settings drawer offers Text, JSON object, and JSON Schema modes. The browser
performs the same syntax, keyword, shape, and size checks before saving an active
schema. Structured output is currently available only for text chat; image
attachment controls are disabled while either JSON mode is selected.

Structured assistant output is rendered as escaped plain code rather than
Markdown, preserving JSON punctuation and preventing content strings from being
interpreted as formatting. The response-format marker is included in portable
conversation archives so another Bloom UI preserves that rendering. Runtime
latency and token measurements remain browser-local and are not archived.

Structured shape is not semantic trust. Applications must still validate field
meaning, authorization, and any value used in a command, path, query, or external
request.

## Runtime validation

The mandatory native CPU gate generates a deterministic, untrained Qwen2
profile that emits one valid JSON object followed by EOS. It requires buffered
and streamed success across OpenAI Chat Completions, Responses, Ollama chat,
and Ollama generate, including exact decoding by both pinned official Python
clients in Linux CI:

```bash
./scripts/test_tiny_model_runtime.sh --require-official-clients
```

This proves the tokenizer, Candle forward pass, protocol adapters, streaming
lifecycles, and terminal validators work together. Separate executor regressions
use adversarial logits to prove cross-step grammar state, independent histories
inside one native batch, and isolation from enabled n-gram speculation. These
gates do not prove that an arbitrary trained model follows schemas reliably or
that non-Candle engines apply a token-level grammar.
