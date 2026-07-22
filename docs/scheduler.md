# Scheduler

The scheduler in `crates/engine/src/scheduler/` manages work within one loaded
model instance. It does not choose between models or devices.

## Step order

`InferenceScheduler::step()` processes work in this order:

1. Preempt lower-priority decode requests when policy permits.
2. Schedule active decode requests within the decode budget.
3. Run up to `decode_quantum_tokens` decode micro-steps.
4. Use the remaining budget for prefill or chunked prefill.
5. Move a request to the decode queue after its final prefill chunk.

Decode receives budget before prefill. This keeps streaming requests moving
while allowing new prompts into the active batch.

## Chunked prefill

```rust
let mut config = bloomai_core::TokenSchedulingConfig::default();
config.chunked_prefill.enabled = true;
config.chunked_prefill.chunk_size = 512;
config.chunked_prefill.interleave_with_decode = true;
```

- Prompts are split to fit the available step budget.
- If `interleave_with_decode` is false, a step that schedules decode work does
  not schedule prefill work.
- The final prefill chunk produces the first token; later steps continue decode.

## Decode quantum

```rust
config.decode_quantum_tokens = 2;
config.max_decode_tokens_per_step = 2;
config.max_total_tokens_per_step = 4096;
```

Larger decode quanta reduce scheduler overhead but can increase latency for new
requests. The default value is `1`.

## KV cache

The scheduler integrates:

- paged allocation and release;
- prefix reuse;
- LRU and priority eviction;
- request preemption;
- sliding-window, context-shift, and inactive-compaction policies; and
- optional CacheMesh L2/L3 offload.

The server exposes cache statistics through `/v1/kv-cache-stats` and `/metrics`.

## Tests

Scheduling contracts are covered in
`crates/engine/src/scheduler/scheduler_test.rs`. Changes to admission order,
token budgets, preemption, or KV-cache behavior must update those tests.
