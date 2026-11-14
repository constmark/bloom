# Trained cross-encoder CPU acceptance

Bloom's native BERT classifier-head reranker now has a reproducible, trained
model acceptance profile. It uses the official
[ms-marco-MiniLM-L6-v2 package](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a),
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The pinned
[model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md)
declares Apache-2.0. Its architecture is `BertForSequenceClassification` with
one output label, six transformer layers, 384 hidden dimensions and a 512-token
pair context.

The 90,870,598-byte Safetensors file has SHA-256
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.
The gate also pins the size and SHA-256 of configuration, tokenizer, special
token metadata, vocabulary and model card. Cached files are rehashed on every
run; corrupt, symlinked or additional unpinned files fail before execution.
Missing files are downloaded only from the exact revision, bounded to the
expected size, verified and atomically published without overwrite.

## Reproduce

```sh
./scripts/test_trained_cross_encoder_runtime.sh \
  --report output/trained-cross-encoder.json
```

The default invocation builds the release CLI and server, verifies/downloads
the model under `/tmp/bloom-trained-cross-encoder-runtime`, and executes the
native Candle CPU backend. Only Python's standard library is required; neither
Torch nor Sentence Transformers performs inference for this test. Existing
binaries and a prepared cache can be selected explicitly:

```sh
./scripts/test_trained_cross_encoder_runtime.sh --skip-build \
  --model-path /tmp/bloom-trained-cross-encoder-runtime \
  --infer-bin target/release/bloom_infer \
  --server-bin target/release/bloom_server \
  --report output/trained-cross-encoder.json

python3 scripts/test_trained_cross_encoder_runtime.py
```

The shell wrapper respects `PYTHON`; the model cache can also be selected with
`BLOOM_TRAINED_CROSS_ENCODER_CACHE`. A missing model is acquired or fails the
gate; it never becomes a successful skip. The JSON report records model and
binary hashes, thresholds, per-case scores, timings and task-isolation results.
It rejects a binary or model artifact that changes during execution.

## Acceptance contract

Three authored English cases cover geography, Python programming and plant
biology. Each contains one relevant passage, a related distractor and an
unrelated passage, with the relevant passage at a different input index.

- Each relevant passage must rank first and beat the strongest distractor by
  at least **3.0 raw logit units**. `top_n=1` must return that same passage.
- Independent pair evaluation, a variable-length padded batch and reversed
  document order must agree to **0.0002 absolute score units**.
- An 18-document request crosses the native 16-item microbatch boundary;
  repeated passages retain their scores, document identities and stable tie
  order. Repeating an earlier query after other requests must preserve scores.
- Responses require unique request IDs, exact active-model identity, finite
  scores, exact returned document text and valid token-usage fields.
- Readiness must advertise only `rerank` and a 512-token context. Inspection
  must identify the classifier task, F32 CPU weights and zero KV allocation.
- All five generation routes and all three embedding routes must reject the
  classifier task. The CLI must reject text generation explicitly and point
  to `/v1/rerank`. Over-context requests and unauthenticated reranking must
  fail before successful inference.

The native classifier emits **raw logits**, consistent with this package's
identity activation. Values may be negative or exceed one. The gate requires
that such values survive the HTTP contract; clipping to `[-1, 1]`, applying an
unsolicited sigmoid or treating them as cosine similarity would invalidate
the evidence. Scores are not calibrated probabilities and are not comparable
across unrelated models.

## Recorded local evidence

The final release-profile execution on macOS arm64 passed the complete gate:
20 native rerank requests, CLI task isolation, inspection and all HTTP checks.
The immutable binaries used for this run have SHA-256:

- Server: `0173036ce6107e6e0d1c61c468a3eee6c031ff013085c6f1bf22c5b454ac8b69`
- CLI: `5143bef9a3392c9502f311ae6b6ba4f69b14071e49d9d1974194f723e88e4393`

Both identities were checked again after execution. The report is retained at
`output/completion-continuation-2026-11-14/cross-encoder.json`; its companion
`cross-encoder.log` contains no errors. The measured quality results were:

| Case | Correct top input index | Raw margin over strongest distractor | Batch HTTP latency |
| --- | --- | --- | --- |
| Geography | 1 | 11.737332 | 14.750 ms |
| Python programming | 0 | 4.689882 | 17.746 ms |
| Plant biology | 2 | 18.765820 | 16.396 ms |

The maximum batch/single/permutation difference was `0.000001073`, below the
`0.0002` limit. The 18-document microbatch and repeat-request checks passed.
Returned raw scores included `-11.259546` and `8.847757`. Five generation
routes, three embedding routes, the context boundary and authentication were
rejected as required. Inspection confirmed 90,870,598 weight bytes in F32 and
zero KV allocation; the CLI explicitly rejected generation with an actionable
rerank diagnostic. The seven harness failure-regression tests also passed.
The earlier debug exploration is retained separately under
`output/capability-p1-2026-11-14/cross-encoder/` and is not the release evidence.

The latencies above are individual observations from this bounded acceptance
run. They are not throughput, tail-latency or sustained-load benchmarks.

These are small deterministic acceptance cases, not an independently measured
MS MARCO/MTEB benchmark, multilingual retrieval claim, calibrated relevance
probability or accelerator validation. They establish trained classifier-head
execution and bounded protocol behavior on this CPU path. `stable_claim`
remains false; long-run serving evidence and broader ranking datasets are
separate acceptance work.
