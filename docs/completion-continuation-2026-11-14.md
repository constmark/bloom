# Completion continuation — 2026-11-14

This continuation addresses concrete unfinished items from the
[cross-layer audit](completion-audit-2026-11-14.md): native generation accounting,
trained cross-encoder acceptance, installed native SDK execution and release
artifact preservation. It retains the existing experimental support status.

## Changes

- Native Candle generation now reports accepted input/output token counts and
  an explicit EOS or token-limit reason before the transport terminal event.
  Direct HTTP and CLI consumers validate the accounting contract. HTTP stop
  sequences use a typed early-stop signal and the last accepted-token count.
  Legacy adapters and IFB remain explicitly bounded by their documented
  capabilities; see the [metadata contract](generation-metadata.md).
- A pinned Apache-2.0 MS MARCO MiniLM classifier package provides trained CPU
  ranking acceptance. The gate tests three authored semantic cases, independent
  versus padded batches, permutation, repeated requests, a microbatch boundary,
  CLI inspection and HTTP task isolation. It is part of CI and
  `just trained-model-runtime-test`; see the
  [profile and thresholds](trained-cross-encoder-validation-2026-11-14.md).
- Browser and native macOS reranking preserve finite model-specific raw scores,
  including values outside `[-1, 1]`. Exports and their JSON schema follow the
  same contract. The browser gate exercises positive and negative raw logits.
  CLI generation explicitly rejects rerank-only classifiers before loading.
- Native-wheel installation checks discard inherited pip/Python and Bloom
  library overrides, install into an isolated virtual environment and verify
  the bundled library. ABI revisions 1 and 2 execute real buffered, streaming
  and pre-cancelled calls. Trained installed-wheel checks cover Unicode
  tokenization, matching buffered/streamed text, generation metadata and
  logprobs, and are required in the Linux CI job.
- Release packaging refuses a nonempty or symlinked destination instead of
  deleting earlier artifacts. Archive publication can prohibit overwrite
  atomically, and checksum creation is exclusive. Callers select a new
  destination with `BLOOM_RELEASE_DIR`; see [release procedure](../RELEASE.md).

## Validation record

Local reports and immutable binary snapshots are retained under
`output/completion-continuation-2026-11-14/`, with SDK delivery evidence under
`output/sdk-delivery-2026-11-14/`. They are generated evidence, excluded from
Git. Final gate results and binary identities are recorded after the final
source build, keeping earlier audit results separate.

## Remaining release requirements

The trained classifier profile is a small English acceptance set, not a full
MS MARCO/MTEB or multilingual quality claim. A short current-build deployment
workload does not replace a new sustained run on the exact supported target
host, service manager and filesystem. The historical 24-hour report belongs
to its own older binary.

Hosted Linux/macOS/Windows and Python-version release matrices, public
container digests and provenance, broader native model/quantization parity,
target accelerator execution, assistive-technology acceptance and fleet
recovery drills remain separate measured release gates. See the
[production gap register](production-readiness.md) and
[support matrix](support-matrix.md). Pushing this work to `dev` does not publish
a release or promote an unmeasured deployment to stable support.
