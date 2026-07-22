# Bloom Scheduler

This directory owns Bloom's engine-local scheduling logic. It is intentionally
kept below `bloomai-engine` instead of the Elderwand runtime scheduler: Bloom
decides how a loaded model advances token work, while Elderwand decides
cross-model routing, residency, and global resource policy.

## Module Map

- `mod.rs`: public scheduler facade, token-level `InferenceScheduler`, generic
  segment `BloomScheduler`, request state, execution batch types, and cache pool
  traits.
- `paged_cache.rs`: paged KV cache storage bridge used by batched Candle paths.
- `scheduler_test.rs`: contract tests for token scheduling, generic segment
  scheduling, and KV cache allocation/eviction.

The directory layout is the stable ownership boundary. New scheduling features
should land here first, with executor-specific code remaining in
`crate::executor`.

## Token Scheduler Semantics

`InferenceScheduler::step()` implements continuous in-flight batching:

1. Apply optional priority preemption for high-priority waiting requests.
2. Schedule already-running decode requests first.
3. Optionally run multiple decode micro-steps in one scheduler step according
   to `TokenSchedulingConfig::decode_quantum_tokens`.
4. Fill remaining token budget with waiting prefill work.
5. If chunked prefill is enabled, schedule partial prefill chunks up to the
   remaining total and prefill phase budgets.

This mirrors the production-friendly policy used by mainstream engines:

- vLLM V1 treats every request as `num_computed_tokens` catching up to
  `num_tokens`, so running decode and partial prefill share one token budget.
  Bloom follows the same practical outcome by scheduling running decode first
  and using the rest of the budget for waiting or partial prefill.
- SGLang exposes mixed chunked prefill and continuous decode steps. Bloom maps
  those knobs to `chunked_prefill.interleave_with_decode` and
  `decode_quantum_tokens`.

## Chunked Prefill Contract

When `chunked_prefill.enabled = true`, long prompts are admitted through
`ChunkedPrefillQueue` and may be split smaller than the configured chunk size if
the current step has less remaining budget. This prevents head-of-line blocking
when `max_total_tokens_per_step < chunk_size`.

Final prefill chunks emit the request's first sampled token and move the request
to the decode queue. They do not immediately run another decode token in the
same scheduler step unless a future executor path explicitly supports fused
mixed-phase batches. This keeps streaming semantics predictable: one scheduler
admission produces at most one generated token per request per micro-step.

If `chunked_prefill.interleave_with_decode = false`, a step that schedules any
decode work will not schedule prefill work. If it is `true`, decode is still
prioritized, and prefill chunks use only leftover budget.

## Budgets

The token scheduler enforces all three token budgets per step:

- `max_total_tokens_per_step`: hard cap for decode plus prefill tokens.
- `max_decode_tokens_per_step`: cap for decode tokens across decode
  micro-steps.
- `max_prefill_tokens_per_step`: cap for prefill tokens.

The generic `BloomScheduler` also has a lightweight `TokenSchedulingConfig`
for segment admission. Keep LLM token-level features on `InferenceScheduler` and
multi-modal request/segment fairness on `BloomScheduler`.

## Tests To Add With Future Features

Add focused tests in `scheduler_test.rs` for every scheduler policy change.
Important invariants:

- Decode is prioritized over new prefill when total budget is tight.
- Chunked prefill can progress even when the step budget is smaller than
  `chunk_size`.
- Mixed chunked prefill only uses leftover budget after decode.
- Final prefill emits exactly the first generated token and enters decode.
- Preemption returns a request to prefill without leaking KV handles.
- KV cache eviction never evicts active blocks.

When executor support grows to true fused mixed prefill/decode batches, keep the
above policy tests and add executor tests that verify per-request phase metadata
and token positions.
