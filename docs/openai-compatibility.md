# OpenAI API Compatibility

Bloom exposes the OpenAI Models resource plus deliberately bounded subsets of
the Responses, chat completions, legacy completions, and embeddings APIs.
Compatibility means that documented fields have the documented effect; Bloom
does not silently discard a non-neutral option and then run a different request.
Every `/v1` response, including streams and errors, carries
`Cache-Control: no-store`; deployments must preserve that directive through any
reverse proxy.
Before namespace routing, Bloom's default same-origin browser guard rejects an
untrusted, opaque, malformed, or loopback DNS-rebinding `Origin` with HTTP 403.
Origin-free OpenAI clients remain compatible. A separately hosted browser
client must be configured as the one exact HTTP(S) origin; rejected requests do
not receive CORS admission.

## Protocol errors

An unknown path anywhere in the `/v1` namespace returns HTTP 404 with the
standard `{"error":{"type":"not_found_error",...}}` envelope. A supported path
called with the wrong HTTP method returns HTTP 405 with an
`invalid_request_error` envelope and the route's `Allow` header. These fallback
messages are fixed and bounded: they do not reflect the request path, query, or
credentials. They also run no route handler and reveal no model or catalog
state.

The fallbacks remain distinguishable from authentication. A missing or
method-mismatched path receives its protocol status even when API-key protection
is enabled, while every recognized `/v1` route still requires the configured
credential. Reverse proxies must preserve the JSON body, status, `Allow`,
`Cache-Control`, and `x-request-id` instead of replacing errors with HTML.
Inference-capacity HTTP 429 responses also carry `Retry-After`; proxies must
preserve it, and CORS exposes it for browser clients.
Recognized routes rejected for a missing or invalid key return the standard
OpenAI error envelope with HTTP 401 and
`WWW-Authenticate: Bearer realm="Bloom"`. CORS exposes the challenge. Public
404/405 namespace fallbacks remain outside authentication and do not carry it.

Framework-level request rejection uses the same envelope. Malformed JSON
returns HTTP 400, a syntactically valid body that does not match the endpoint
schema returns 422, a missing or unsupported request media type returns 415,
and a body that exceeds the configured or route-specific limit returns 413.
Invalid multipart admission also returns a JSON client error. Bloom preserves
the framework status and safe headers but replaces non-protocol bodies with a
fixed, bounded message; parser details and submitted content are not reflected.

## Models

`GET /v1/models` lists the one active Bloom runtime or an empty collection.
`GET /v1/models/{model}` retrieves that runtime by its exact published ID or by
Bloom's `default` alias. Both return the standard `id`, `object: "model"`,
`created`, and `owned_by` fields. The `created` timestamp is stable for the
lifetime of that loaded runtime and records when it was successfully published
in the current server process; it is not a claim about when upstream weights
were trained or released. Retrieval always returns the exact runtime ID, even
when selected through `default`.

Model IDs use the same 1-to-256-character, control-free admission as inference
selectors. A missing or inactive ID returns HTTP 404 `model_not_found`; invalid
selectors or any query parameters return HTTP 400 rather than being ignored.
The endpoint does not automatically load an inactive catalog entry. OpenAI
fine-tuned-model deletion semantics are intentionally not mapped onto Bloom's
local catalog removal API.

## Chat completions

`POST /v1/chat/completions` supports:

- `model`, bound exactly to the active Bloom runtime or the `default` alias
- `content` as either a string or an array of one to 256
  `{"type":"text","text":"..."}` parts, with `developer`, `system`,
  `user`, or `assistant` roles; bounded paired `assistant.tool_calls` and
  `tool` result messages are accepted for function-call continuation; text
  parts are concatenated in order without a separator
- `stream` and `stream_options.include_usage`
- `max_tokens` or its current `max_completion_tokens` alias, plus `temperature`,
  `top_p`, and `seed`; if both token-limit fields are supplied, they must match
- `response_format` values `text`, `json_object`, and the documented bounded
  `json_schema` subset
- up to 32 function tools, `tool_choice` values `none`, `auto`, `required`,
  or a named function, and `parallel_tool_calls`

The response uses OpenAI-compatible completion objects or SSE chunks. Streaming
requests terminate with `[DONE]`; usage is emitted in a final empty-choices
chunk when requested. The initial streaming chunk declares the assistant role,
and every chunk in one stream retains the same ID, model, and creation time.
See the [structured-output guide](structured-output.md) for schema constraints.

A leading `developer` message is an explicit alias for a local `system`
instruction. It may appear alongside other leading `developer` or `system`
messages. Bloom rejects a `developer` message after the first `user` or
`assistant` turn because applying it as ordinary conversation text would change
its instruction semantics.

### Function tools

Bloom implements the OpenAI Chat Completions function-tool request and response
shape. Only `type: "function"` tools are admitted. Function names contain 1 to
64 ASCII letters, digits, underscores, or hyphens; descriptions are bounded to
1,024 control-free characters; and the combined tool definitions are limited
to 128 KiB. Each `parameters` value uses the bounded JSON Schema subset from
the [structured-output guide](structured-output.md). Omitting `parameters`
creates an empty, closed object schema. Duplicate names fail before inference.
When `strict: true`, every object node must set `additionalProperties: false`
and list every declared property in `required`; incompatible strict schemas
are rejected before inference. Regardless of the strict flag, Bloom validates
every returned arguments object before exposing the call.

With tools present, omitted `tool_choice` behaves as `auto`. `auto` permits a
message or calls, `required` requires at least one call, a named choice requires
exactly that function once, and `none` performs ordinary text generation.
`parallel_tool_calls: false` limits output to one call; otherwise Bloom admits
and returns at most eight calls. The server does not execute functions.

Bloom provides this capability through a model-independent compatibility
layer. It appends a private JSON control protocol to the formatted local prompt
and asks the model to choose a message or functions. Tool-enabled streams are
therefore buffered until the complete control object can be checked. Bloom then
validates its shape, every selected function name, call count, and every
arguments object against the declared schema before returning standard
`message.tool_calls` or `delta.tool_calls`. A valid call ends with
`finish_reason: "tool_calls"`; arguments remain a JSON string in the public
response. Invalid or truncated control output returns HTTP 422
`invalid_tool_call` for a non-streaming request, or an SSE error followed by
`[DONE]` for a streaming request. Internal control JSON is never exposed as
assistant content.

Clients execute approved functions and send the original assistant tool-call
message followed by one matching `role: "tool"` message per call. Bloom bounds
IDs, requires valid JSON-object arguments, rejects reused, unknown, duplicate,
missing, or name-mismatched call/result pairs, and serializes results into
explicitly untrusted conversation context. The loaded model still determines
whether a valid control object can be produced; small or non-instruction-tuned
models may fail closed.

Custom tools, hosted/built-in tools, namespaces, allowed-tool subsets, the
deprecated `functions` and `function_call` request fields, and combining an
active tool choice with `response_format` are not supported. The public shapes
and call/result lifecycle follow the
[OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling)
and
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Responses API

`POST /v1/responses` provides a text adapter over the same active model and
generation lifecycle as chat completions. It supports:

- `model`, `max_output_tokens`, `temperature`, and `top_p`
- non-streaming responses or `stream: true` server-sent events
- `text.format` values `text`, `json_object`, and the documented bounded
  `json_schema` subset
- optional `instructions`, mapped to one leading `developer` instruction
- `input` as a string or as one to 2,048 message, `function_call`, or
  `function_call_output` items; message items may use a `developer`, `system`,
  `user`, or `assistant` role and string content or one to 256 ordered text
  parts
- up to 32 flat function tools, `tool_choice` values `none`, `auto`,
  `required`, or a named function, and `parallel_tool_calls`
- current SDK-shaped response, output-message, `output_text`, incomplete-limit,
  and token-usage objects
- explicit process-local retention with `store: true`, multi-turn continuation
  through `previous_response_id`, retrieval, deletion, and input-item listing
- `metadata` with at most 16 control-free string key/value pairs; keys are 1 to
  64 characters and values are at most 512 characters

The chat message-count, 768 KiB combined-content, per-role character, context,
and output-token limits still apply. `instructions` has the 65,536-character
system-instruction limit. A response that reaches `max_output_tokens` reports
`status: "incomplete"` with reason `max_output_tokens`; normal completion
reports `status: "completed"`.

### Function tools

Responses function definitions use the current flat shape:
`{"type":"function","name":"lookup","description":"...","parameters":{...},"strict":true}`.
The limits, JSON Schema subset, strict-mode requirements, validated model
control protocol, and no-server-side-execution boundary are identical to Chat
Completions function tools. A named choice uses
`{"type":"function","name":"lookup"}`. Omitted choice is `auto` when tools
are present; `parallel_tool_calls: false` limits the result to one call, while
the default permits up to eight.

A validated call is returned as a native output item with `type:
"function_call"`, an item `id`, a result-correlation `call_id`, the function
`name`, and JSON-string `arguments`. Clients execute the function and submit a
matching `{"type":"function_call_output","call_id":"...","output":"..."}`
input item. Bloom currently accepts bounded string outputs; image and file
function outputs are rejected. Consecutive function-call input items are
treated as one parallel assistant turn, and every result must pair exactly
once with a known outstanding call.

With `store: true`, native function-call output and result input items are
retained and exposed by the input-items endpoint. A client may therefore send
only the matching `function_call_output` item with `previous_response_id`; the
stored assistant calls are restored before pair validation. Tool definitions
are request configuration and are not inherited, so clients should send the
tools needed on each generation turn.

### Local response state

Bloom is privacy-first: omitting `store` is equivalent to `store: false` and
does not retain the response. This intentionally differs from the hosted API's
default retention behavior. Set `store: true` explicitly when the client needs
to retrieve or continue a response. Successfully completed or incomplete
responses are then retained only in the server process. Failed generations,
responses whose public stream is disconnected before its terminal event, and
requests with `store: false` are never retained.

The local store is bounded to 256 responses, 64 MiB total, and 40 MiB for one
record. Records expire after 24 hours and the oldest record is evicted first
when either global limit is reached. State is not written to disk, does not
survive a server restart, and can disappear earlier through explicit deletion,
expiry, or capacity eviction. A response too large for the per-record limit
fails closed instead of returning `store: true` for state that cannot be
retrieved.

The state endpoints are:

- `GET /v1/responses/{response_id}` to retrieve one retained response
- `DELETE /v1/responses/{response_id}` to delete it
- `GET /v1/responses/{response_id}/input_items` to list the effective input
  history; `limit` accepts 1 to 100 (default 20), `order` accepts `asc` or
  `desc` (default `desc`), and `after` is a response-scoped item cursor

These endpoints are API-key protected with the rest of `/v1` and return
`Cache-Control: no-store`. Missing, expired, evicted, deleted, or never-stored
IDs return HTTP 404. Invalid IDs, cursors, and query controls return HTTP 400.

`previous_response_id` prepends the retained input and assistant-output history
to the new input. The new response remains bound to the previous response's
exact model; switching or requesting a different model fails rather than
silently changing the chain. Instructions supplied through the prior
response's top-level `instructions` field do not carry forward, matching the
current Responses contract. Supply new instructions on each chained request.
Instruction-like message items in `input` are ordinary retained history. The
complete inherited history remains subject to Bloom's message, content, token,
and context-window admission limits, and prior turns count as input tokens.
Metadata belongs only to the response on which it is supplied: it appears in
every lifecycle object and retained retrieval, but is not inserted into the
model prompt and is not inherited by a chained response.
See the
[official conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state)
and the
[official Python SDK Responses methods](https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/api.md)
for the upstream contracts used by the compatibility tests.

Responses structured output uses the current direct format shape, for example
`{"text":{"format":{"type":"json_schema","name":"answer","schema":{...},"strict":true}}}`.
Bloom maps that request to the same prompt constraint, engine format hint, and
final validator used by chat completions. The normalized format is preserved in
created, in-progress, completed, incomplete, and failed response objects.
Malformed or unsupported format configuration fails before inference. A
non-streaming model output that violates the format returns HTTP 422 with
`invalid_response_format`; a streaming violation ends with `response.failed`
instead of a success terminal event. See the
[structured-output guide](structured-output.md) for the supported schema
keywords and limits.

Streaming follows the current Responses event protocol. Bloom emits named SSE
events with monotonically increasing `sequence_number` values: response
creation and progress, output-item and content-part addition,
`response.output_text.delta`, text/content/item completion, and one terminal
`response.completed` or `response.incomplete` event containing the full output
and usage object. Function-call streams emit one `response.output_item.added`
per call, followed by `response.function_call_arguments.delta`,
`response.function_call_arguments.done`, and `response.output_item.done` with
stable response, item, output-index, and call IDs. Tool-enabled generation is
buffered internally until its complete control object and arguments pass
validation, so the arguments may arrive in one public delta. Generation
failures terminate with `response.failed` when response metadata is available.
Unlike chat completions, a successful Responses stream ends with its typed
terminal event and does not append `[DONE]`.
See the
[OpenAI Responses streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses)
and the
[official Python SDK streaming example](https://github.com/openai/openai-python#streaming-responses)
for the upstream contracts used by the compatibility tests.

The adapter validates the internal stream's media type, UTF-8 and JSON framing,
one-megabyte internal frame boundary, stable ID/model/creation metadata, one
choice, ordered finish and usage records, at most 131,072 public events, and at
most 16 MiB of accumulated output. Dropping the public stream drops the owned
chat stream, preserving Bloom's cancellation and concurrency lifecycle.

This subset does not implement background execution, custom or hosted/built-in
tools, allowed-tool subsets, automatic truncation, reasoning controls, text
verbosity controls, or image/file/computer content. Active or unknown
semantics such as a built-in tool, `background: true`, `truncation: "auto"`, or
unsupported content fail with HTTP 400 before inference rather than being
ignored. Active function tools cannot be combined with `text.format` structured
output; function arguments are validated against the selected tool schema.

## Embeddings

`POST /v1/embeddings` accepts `input` as one nonblank string or an array of 1
through 256 nonblank strings. Each string is limited to 262,144 characters and
the batch to 768 KiB. `model` may be omitted, use `default`, or exactly match
the active runtime. Native BERT packages advertise embedding support by model
family. Other packages must declare trusted `bloom_task=embedding` or
`bloom_task=rerank` manifest metadata; directory names are never capability
signals. Encoder-only models reject Chat, Completions, and Responses generation
with HTTP 422 `unsupported_operation`.

Only `encoding_format: "float"` is supported. `dimensions` may be omitted for
the native width or set from 1 through 16,384; requesting more dimensions than
the model produces fails instead of fabricating values. An optional nonblank,
control-free `user` string of at most 256 characters is admitted as unused
compatibility metadata. Other non-null extension fields fail closed before
runtime admission.

Bloom tokenizes the complete batch before inference and rejects an input that
exceeds the active context window; this endpoint does not silently truncate.
Returned vectors are finite, nonempty, dimensionally consistent, bounded to
1,048,576 aggregate float values, projected when requested, and L2-normalized.
The batch runs in one blocking worker. Native BERT execution uses request
microbatches of at most 16 items, groups similarly sized token sequences under
a 4,096-padded-token budget for multi-item backend batches, executes a longer
valid sequence alone, masks padded attention and mean pooling, and restores
exact input order. Other embedding adapters retain scalar execution. Client
disconnect cancellation is propagated between native microbatches or scalar
inputs, while the worker retains the concurrency permit, cancellation
registration, and exactly-once metrics until it actually exits.

## Reranking extension

`POST /v1/rerank` is a Bloom extension over the same active embedding runtime.
It accepts one nonblank `query`, 1 through 256 nonblank string `documents`, an
optional `top_n`, and `return_documents: true` when selected source text should
be included. The query is limited to 65,536 characters, each document to
262,144 characters, and all text to 768 KiB. `model` follows exact active-model
binding. Unknown non-null fields fail before runtime admission.

Bloom tokenizes every input and rejects any query or document beyond the active
context window. One blocking batch creates the query and document vectors using
the same native BERT microbatch or scalar fallback described above, validates
and normalizes them, computes finite cosine relevance scores from -1 through 1,
and sorts descending. Equal scores retain original document-index order. `top_n`
is enforced before response construction, response IDs are
process-unique, and usage reports the tokenizer-backed total across the query
and every document.

This route performs bi-encoder cosine reranking; it does not emulate a
cross-encoder reranker that requires joint query-document inference. Client
disconnects retain the same worker-owned permit, cancellation registration,
and exactly-once accounting guarantees as `/v1/embeddings`.

## Legacy completions

`POST /v1/completions` supports the same model selector, generation controls,
streaming flag, and response formats. `prompt` must be one string or a one-item
string array. Bloom rejects batched prompt arrays rather than returning a
partial or misleading response.

## Stop sequences

Chat Completions and legacy Completions accept `stop` as one non-empty string,
an array of up to four non-empty strings, or `null`. Each sequence is limited to
1,024 characters and all sequences together to 16 KiB. The first match wins by
output position. Returned text never contains the matched sequence or later
model output.

Streaming uses an incremental matcher that retains only a suffix that could
still become a configured sequence. This prevents a sequence split across
model deltas from leaking to the client. A confirmed match ends blocking model
execution or cancels the continuously batched scheduler request while retaining
normal successful lifecycle accounting. Natural stream termination flushes any
unmatched retained suffix. The browser persists the same bounded list locally
and sends it on text chat requests; multimodal submissions reject active stop
sequences during preflight because that endpoint does not yet implement them.

## Extension-field admission

Chat, legacy completion, and embedding endpoints capture fields outside the
supported request shape, including
extensions inside chat messages, `stream_options`, and `response_format`. JSON
`null` is accepted as an absent extension. Bloom also accepts the following
exact no-op defaults so clients can reuse a conservative OpenAI request
template:

| Field | Accepted neutral value |
| --- | --- |
| `n`, `best_of` | `1` |
| `functions` | `[]` |
| `function_call` | `"none"` |
| `logprobs`, `store`, `echo` | `false` |
| `frequency_penalty`, `presence_penalty` | `0` |
| `logit_bias`, `metadata` | `{}` |
| `top_logprobs` | `0` |
| `user` | A non-empty, control-free string of at most 256 characters |

The `user` value is accepted only as compatibility metadata. Bloom does not use
or persist it. Empty `tools`, `tool_choice: "none"`, and
`parallel_tool_calls: false` are valid no-call configurations. A message-level
`tool_calls: []` and `stream_options.include_obfuscation: false` are likewise
neutral.

The Responses adapter independently admits JSON `null` plus exact neutral
defaults used by conservative SDK request templates: `background: false`,
empty `include`, `top_logprobs: 0`,
`truncation: "disabled"`, and `service_tier: "auto"` or `"default"`.
`store` is a supported boolean with Bloom's explicit opt-in default described
above, `previous_response_id` is a supported bounded response ID, and
`metadata` follows the bounded string map described above.
Function `tools`, `tool_choice`, and `parallel_tool_calls` are supported request
semantics with the constraints described above; empty tools and `tool_choice:
"none"` remain valid no-call configurations.
Responses `text` may be omitted, empty, or contain one supported `format` as
described above. Bounded control-free `user` and
`safety_identifier` strings are compatibility metadata only and are neither
used nor persisted.

Any unsupported non-neutral value for these fields is rejected with HTTP 400 and an
`invalid_request_error` before runtime availability, concurrency admission,
tokenization, or inference. Examples include a non-empty deprecated `functions`
array, a custom or built-in tool, `n` greater than one, non-zero penalties,
requested log probabilities, non-empty logit bias, and
`echo: true`. Unknown non-null fields follow the same fail-closed rule. Error text
reports at most eight bounded safe field names and never echoes arbitrary large
or control-bearing names.

Multiple choices, token log probabilities, penalty/logit-bias sampling, custom
tools, and deprecated function fields are
therefore explicitly unsupported today. Message content arrays accept text
parts only. Bloom rejects
image, audio, file, refusal, malformed, empty, or over-limit part arrays with an
HTTP 400 error; use Bloom's bounded multimodal endpoints for supported image and
audio input. This behavior is intentionally different from silently ignoring a
field, which could cause an agent or application to execute with assumptions
Bloom did not honor.

## Validation

From a source checkout, run the pinned official client against an isolated,
empty Bloom catalog without a model or accelerator:

```bash
python3 -m pip install -r requirements/compat-smoke.txt
./scripts/openai_compat_smoke.py \
  --build \
  --api-key local-compat-smoke \
  --require-openai-sdk
```

This gate decodes empty model discovery, missing-model retrieval, backend
discovery, authentication failure, unavailable-model chat admission, and a
missing Responses lookup through the official SDK before reporting model-backed
inference as skipped.

CPU-only route tests cover developer-role mapping, string and text-part content,
bounded stop admission, cross-delta filtering, early inference termination,
non-streaming and streaming truncation, Ollama adapter reuse,
the token-limit alias, bounded function definitions and choices, argument
schemas, paired tool-call history, non-streaming and streaming call shapes,
Responses input normalization and output shape, neutral defaults, active
unsupported fields, stream-option extensions, bounded field-name errors,
standard error envelopes, embedding projection and output invariants, and
rejection before inference accounting. With a pinned real model,
`scripts/openai_compat_smoke.py` additionally exercises chat streaming, modern
chat fields, a Chat SDK function-call/result continuation when the model can
produce a valid call, Responses admission and structured output, fail-closed
unsupported semantics, and the current OpenAI Python SDK's streaming create,
retrieve, input-item list, chained create, and delete methods. CPU-only
protocol fixtures also validate the complete named-event sequence against the
current SDK without requiring a model. The mandatory generated Qwen2 CPU gate
adds a deterministic tool profile that must produce successful Chat and
Responses calls, buffered call streams, schema-valid arguments, result
continuations, retained input items, and cleanup through raw HTTP. Linux CI
requires the pinned SDK to complete the same Chat and Responses function
lifecycles; a fail-closed 422 response is not accepted as success in this mode.

The same mandatory gate includes a deterministic structured-output profile.
Its `--structured-only` smoke requires JSON object and strict JSON Schema
success through buffered and streamed Chat Completions and Responses calls,
then requires the pinned SDK to decode exact `{"ok":true}` output from both API
families. A fail-closed 422 response is not accepted as success in this mode.
For a compatible generation model, run it directly with:

```bash
BLOOM_MODEL_PATH=/path/to/structured-output-model \
./scripts/openai_compat_smoke.py \
  --require-model \
  --structured-only \
  --require-openai-sdk
```

Run a pinned embedding model without invoking text generation:

```bash
BLOOM_MODEL_PATH=/path/to/embedding-model \
./scripts/openai_compat_smoke.py \
  --embedding-only --require-model --require-openai-sdk
```

This mode validates SDK-decoded projected embeddings, L2 norms, token usage,
stable rerank ties, score bounds, `top_n`, returned documents, and response
identity.

Maintainers can run the immutable trained MiniLM profile directly:

```bash
./scripts/test_trained_embedding_runtime.sh --require-official-clients
```

That profile additionally requires native 384-dimensional output, a positive
semantic and rerank margin, the published 256-token task limit, and fail-closed
generation across the OpenAI and Ollama adapters.
