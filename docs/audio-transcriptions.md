# Audio transcription HTTP API

`POST /v1/audio/transcriptions` accepts a bounded multipart upload and returns
recognized speech from the selected local audio-capable runtime. The request
and JSON/plain-text response shapes follow the
[OpenAI transcription interface](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create),
with the narrower format and parameter support documented here.

```bash
curl http://127.0.0.1:3000/v1/audio/transcriptions \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -F model=default \
  -F file=@speech.wav \
  -F response_format=json
```

A successful JSON response is `{"text":"recognized speech"}`. Use
`response_format=text` for UTF-8 `text/plain`. Silence may produce an empty
text. The endpoint does not invent word timestamps, confidence scores or
language-token usage.

## Supported request

- `model` is required. It selects the exact loaded runtime ID, or `default` for
  the current active runtime. The endpoint does not acquire or load models.
- `file` is required and must contain one complete RIFF/WAVE, 16-bit signed
  little-endian PCM, mono recording at 8,000 through 48,000 Hz. Audio must be
  nonempty and at most 60 seconds long. Stereo, floating WAV, compressed audio,
  MP3 and other containers are rejected. There is no implicit downmix or resample.
- `response_format` is optional and accepts only `json` (default) or `text`.
- `stream` may be omitted or set to the literal `false`.

Each field may appear once. Query parameters, unknown fields, streaming,
`language`, `prompt`, `temperature`, timestamps, log probabilities, diarization
and translation controls are rejected with HTTP 400. The current inline ASR
bridge chooses its language automatically and does not apply these generation
controls; accepting them would misrepresent the request.

The attachment is capped at 6 MiB and each text field at 1,024 bytes, in addition
to the server's configurable multipart body limit (`--max-upload-bytes`, default
12 MiB). Uploads are read in bounded chunks. The decoder checks RIFF length,
chunk boundaries and padding, unique format/data chunks, PCM metadata and sample
count before allocating decoded floats. PCM expansion happens inside the blocking
worker only after inference admission, so concurrent uploads cannot allocate
decoded audio before receiving a permit. Filenames and MIME labels cannot replace
payload validation and are never interpreted as server filesystem paths.

## Runtime and errors

The loaded model must declare Audio input and successfully execute transcription;
FunASR/Qwen ASR still need their documented external runtimes and model packages.
An audio modality declaration is not proof that every model performs ASR: models
that emit unrelated modalities or fail inference receive HTTP 422. This route
does not enable audio requests on text-only models.

Authentication, correlation IDs, no-store headers and namespace error handling
are shared with the other `/v1` routes. Invalid model selectors and uploads return
400; an unloaded exact model returns 404, a revoked version 410, unavailable
runtime 503, a model without Audio input 422, and exhausted concurrency 429.
Oversized uploads/fields return 413. Backend failures return a fixed 422 error
without leaking model paths, worker errors or private input.

Inference runs off the async HTTP executor. The request owns its exact runtime
generation and concurrency permit until both the client and blocking worker
settle. Disconnects/timeouts cancel the request; cancellation checks occur before
inference and at every output event. An external worker already blocked in its
vendor runtime may continue until it returns, so its permit and model lease stay
charged during that interval. Cancelled requests never become successful metrics.

Transcript collection is bounded to 1 MiB and 16,384 output events. ASR partial
snapshots are not concatenated with final text deltas, preventing duplicate text
from bridges that emit both. A snapshot-only bridge uses its latest partial at
successful stream completion. Engine stream validation requires a terminal event
and rejects malformed output; this API remains buffered even when the backend
emits incremental events.

## Validation boundary

Unit and handler regressions cover real PCM WAV decoding, malformed/truncated
RIFF data, duration limits, multipart admission, exact model routing, JSON/text
output, duplicate ASR projections, bounded private errors, concurrency and
worker-owned resource retention after client disconnect. These fixtures prove
protocol/lifecycle mechanics. They do not establish trained ASR quality or prove
that a vendor runtime is installed on a particular deployment.
