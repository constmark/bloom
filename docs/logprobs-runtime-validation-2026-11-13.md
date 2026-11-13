# Native token logprob validation — 2026-11-13

`scripts/logprobs_runtime_smoke.py` validates the real local Candle CPU model
through Chat Completions and legacy Completions. It builds and downloads
nothing. The caller supplies an executable, a model and a new output directory;
every HTTP request/response, server log, binary digest and model artifact digest
is retained. Server processes run in private process groups and are cleaned up
on completion or failure.

```bash
python scripts/run_with_timeout.py 600 python scripts/logprobs_runtime_smoke.py \
  --server-bin /tmp/bloom-metal-validation-target/debug/bloom_server \
  --model /path/to/local/model \
  --output-dir /tmp/bloom-logprobs-report \
  --max-tokens 4 --require-openai-sdk
```

The default matrix covers both protocols, `n=1,2`, and top-alternative counts
`0,2`, each buffered and streamed. With the SDK flag, both protocols are also
decoded through the installed official OpenAI Python SDK. The gate checks:

- Finite nonpositive scores, top-candidate ordering and requested bounds.
- Exact reconstruction of Chat content from raw token bytes; lossy individual
  token displays never replace the byte representation.
- Legacy token, score, alternative-map and offset array alignment; exact
  Unicode-scalar offsets when individual token displays contain no replacement
  character, and monotonic bounded offsets for split-byte displays.
- Identical buffered/streamed text, metadata and finish reason. Chat's requested
  stream usage must equal buffered usage. Legacy does not admit `stream_options`
  or publish stream usage, so its usage is checked on the buffered response.
- Usage equal to metadata records plus at most one sampled EOS per stopped
  candidate. An immediate EOS must contribute one token despite empty content
  and metadata.

## Local execution evidence

The following runs used the same native binary, SHA-256
`4807e9a44bc846428bf3e49b487e4a451c5e4dcfcc4a2405dccd19dc79d4b491`, on macOS
arm64, with OpenAI Python SDK 3.37.0 where requested. The binary also includes
Metal; logprob requests in these runs explicitly selected CPU.

| Profile | Result | Local report |
| --- | --- | --- |
| Generated tiny Qwen2 text, four generated tokens | 10 groups passed, including raw HTTP and SDK | `/tmp/bloom-logprobs-tiny-final-run2/report.json` |
| Tiny Qwen2 with its deterministic output token configured as EOS, budget one | 10 groups passed; empty content/metadata, `stop`, usage one per candidate | `/tmp/bloom-logprobs-eos-final/report.json` |
| Cached trained Qwen2 Q4_0, ordinary `Hello` prompt, four generated tokens | 10 groups passed, including raw HTTP and SDK | `/tmp/bloom-logprobs-trained-final/report.json` |
| Cached trained Qwen2 Q4_0, Chinese token bias, `n=2`, top two | Both protocols passed buffered/streamed; exact `你好你好`, offsets `[0,2]` | `/tmp/bloom-logprobs-trained-unicode-final/report.json` |

The reports, request/response captures and logs are also archived under
`output/capability-logprobs-metal-2026-11-13/` in `logprobs-tiny/`,
`logprobs-eos/`, `logprobs-trained/` and `logprobs-unicode/` respectively.

The trained file is the already cached
`Qwen/Qwen2-0.5B-Instruct-GGUF` artifact described in
[trained model validation](trained-model-validation.md), SHA-256
`aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f`.
No weights were downloaded for this run. Its Unicode test uses:

```bash
python scripts/run_with_timeout.py 600 python scripts/logprobs_runtime_smoke.py \
  --server-bin /tmp/bloom-metal-validation-target/debug/bloom_server \
  --model /tmp/bloom-trained-qwen2-runtime/qwen2-0_5b-instruct-q4_0.gguf \
  --output-dir /tmp/bloom-logprobs-unicode-report \
  --prompt '用中文回答：你好。' --max-tokens 2 --choices 2 --top-counts 2 \
  --logit-bias '{"108386":100}' --expected-text '你好你好'
```

The immediate-EOS fixture copies the generated `text` profile and sets
`config.json`'s `eos_token_id` to `1`, its deterministic `fixture` token. Weights
and tokenizer are unchanged. It is tested with `--max-tokens 1 --expected-text ''`.

## Boundaries

This validates token metadata and public serialization, not language-model
quality or probability calibration. The trained Unicode bias test verifies
byte/offset handling; its forced answer is not semantic-quality evidence.
These logprob requests require CPU direct plain text and a supported ByteLevel
or decoder-free WordLevel tokenizer. IFB, Metal, structured output, tools,
custom stops, legacy echo and unsupported decoders remain explicit admission
errors. An incomplete UTF-8 sequence at EOS or the token budget fails rather
than producing fabricated bytes or a successful finish event.
