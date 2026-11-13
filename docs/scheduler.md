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

All prefill, decode, and shared admission counters compare against their
remaining budget without unchecked addition. Corrupt or oversized accounting
state therefore fails closed instead of wrapping in release builds or panicking
in debug builds.

## Chunked prefill

```rust
let mut config = bloomai_core::TokenSchedulingConfig::default();
config.chunked_prefill.enabled = true;
config.chunked_prefill.chunk_size = 512;
config.chunked_prefill.interleave_with_decode = true;
```

- Prompts are split to fit the available step budget.
- An enabled chunk size must be at least one token; startup and doctor reject
  zero instead of silently changing the configured value.
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

For a native streaming Qwen executor, `BLOOM_KV_CACHE_DTYPE=int8` or `fp8`
selects one-byte paged-cache storage; `bf16`, `f16`, and `f32` retain the
corresponding wider layout.  The `fp8` paged representation currently uses
Bloom's checked INT8-compatible payload until a device-native FP8 attention
kernel is available.  Local F16/BF16/F32 tensor blocks stay on the admitted
device during prefix restore.  CacheMesh and quantized blocks use the vector
serialization path so eviction and cross-process restore remain portable.

The server exposes cache statistics through `/v1/kv-cache-stats` and `/metrics`.

## Tests

Scheduling contracts are covered in
`crates/engine/src/scheduler/scheduler_test.rs`. Changes to admission order,
token budgets, preemption, or KV-cache behavior must update those tests.

## KV retention admission

`KvEvictionConfig.retention_ratio` retains at least the configured fraction of
all candidate sessions' KV tokens, rounded upward. Eviction cancels complete
sessions, so a victim that would cross the retention floor is skipped. If the
selected victims cannot release the full requested token count, no eviction
plan is committed and the new request fails admission; existing requests are
not cancelled for an insufficient plan. The ratio must be finite in `[0,1]`.
Use zero to explicitly allow eviction of every candidate; the default is 0.1.

Sampling now applies penalties and token bias before temperature, and random
sampling spans the full probability interval. Correcting the prior biased RNG
changes affected seeded output sequences. Draft speculation samples proposals
from the distribution used by its verifier. Sampling/grammar combinations the
speculative verifier cannot honor use ordinary decoding; this preserves output
semantics without claiming acceleration for those combinations.

Multiple OpenAI choices retain one HTTP admission/runtime lease while scheduling
candidates sequentially with unique internal request IDs. External blocking
workers use `cancel_request_and_wait` to retain that lease until any active
execution/submission leaves the scheduler gate and candidate KV is released.
This API must not be invoked from executor/backend callbacks; allocation
callbacks retain the deferred `cancel_request` path. EOS detection is delegated
to the executor's model, and speculative token tails are clipped at EOS or the
remaining generation budget before publication.
