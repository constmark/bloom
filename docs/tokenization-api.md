# Exact tokenizer API

Bloom provides two authenticated extensions for inspecting the currently loaded
model's tokenizer. They do not run generation, apply a chat template, load an
inactive model, or claim compatibility with an OpenAI tokenizer endpoint.

```bash
curl http://127.0.0.1:3000/v1/tokenize \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","text":"Hello"}'

curl http://127.0.0.1:3000/v1/detokenize \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","tokens":[123,456]}'
```

Use the actual IDs returned by `/v1/tokenize` for the loaded model. IDs in the
second example are illustrative and are not portable between tokenizers.

`POST /v1/tokenize` accepts `text` and optional `model`; the response contains
`schema_version: 1`, `object: "bloom.tokenization"`, a request `id`, the exact
loaded `model`, a `tokens` array and its `count`.

`POST /v1/detokenize` accepts `tokens` and optional `model`; the response contains
`schema_version: 1`, `object: "bloom.detokenization"`, a request `id`, the exact
loaded `model`, decoded `text`, and `count` equal to the number of submitted IDs.
Empty text and empty token arrays are valid. Omitting `model` or selecting
`default` binds the active runtime; any other model ID must match a loaded
runtime exactly. Request execution retains that runtime generation until the
worker settles, even when the client disconnects or the model is replaced.

Both routes reject unknown fields and all query parameters. The JSON request
body is limited to 1 MiB; text input/output is limited to 262,144 UTF-8 bytes and
token input/output to 65,536 unsigned 32-bit IDs. Escaping contributes to the
JSON wire limit. Tokenizer utilities do not consume a generation context budget
or increment prompt/generated-token counters; they share inference concurrency,
request accounting, cancellation and timeout cleanup. All responses inherit
API-key protection, JSON errors, `no-store` and request correlation policy.

The APIs use exact model methods or a declared `TextTokenizer` processor.
Bloom's internal word-count approximation is never returned. A model task label
(such as generation, embedding or tensor) alone does not establish tokenizer
support: absent exact methods/processors produce no synthetic IDs or text. Unsupported
operations return HTTP 422 `unsupported_operation`; backend failures or unknown
Candle token IDs return 422 `tokenization_failed`; oversized backend results
return 422 `invalid_tokenizer_output`. Backend diagnostic text and local paths
are not returned. Wrong-model, revoked-model and capacity failures retain the
normal 404, 410 and 429 behavior. Invalid bounds/query controls return 400;
invalid JSON field types and unknown fields return 422 at extraction.

Encoding follows the adapter's existing special-token policy (Candle BERT adds
its encoder special tokens; Candle text generation does not add them through
this method). Candle decoding validates vocabulary membership and omits special
tokens. Exact tokenizer normalization and special-token handling mean that
encode/decode is not promised to preserve every original input byte. External
llama.cpp exposes exact encoding through its worker; decoding requires an
adapter or declared processor that implements it and otherwise fails explicitly.

A real model/HTTP check can be reproduced with:

```sh
cargo run --locked -p bloomai-engine --example generate_tiny_qwen2_fixture -- --output /tmp/bloom-api-tiny
cargo build --locked --bin bloom_server
python3 scripts/native_operation_smoke.py --model /tmp/bloom-api-tiny --text hello
```

The generated model is untrained; this checks tokenizer/HTTP execution and error
boundaries, not generation quality.
