# Experimental dense multi-request execution

`BatchExecutorConfig.experimental_dense_batch` defaults to `false` at the
application call sites. Set it explicitly to enable the native dense Qwen
batching implementation. In the enclosing Elderwand application the single
opt-in is `ELDERWAND_DENSE_BATCH=1`; its adapter passes the typed setting to
Bloom and coalesces compatible, already admitted execution steps.

`EngineExecutor::supports_dense_batch()` distinguishes this implementation from
an executor which merely accepts a scheduling batch and loops over requests.
The flag is true only for an opted-in native Safetensors dense Qwen2/Qwen3
streaming template using unquantized KV. GGUF, MoE, quantized KV, unequal chunk
lengths and unequal start positions retain independent request execution.
This is not ragged attention, FlashAttention, CUDA Graph execution, or a claim
of GPU performance improvement.

For compatible requests, Bloom joins each layer's private KV along the batch
dimension, invokes the model exactly once with `[requests, tokens]`, then copies
each row into a separately owned KV allocation. The causal mask broadcasts
across requests. Weights are shared by all forks of the same template; a private
weight-family identity rejects accidentally mixing unrelated models. Results
retain input order and sampling still uses each request's own parameters.

All work happens on a temporary model. Existing request caches change only
after the forward and all output KV copies succeed. A failed forward therefore
does not leave some requests or layers advanced. Each committed row has its own
storage, so cancelling one request can release its model/KV without retaining
or modifying another request's cache. Duplicate handles and malformed packed
layouts are rejected before locking request models or executing them.

## Memory admission in Elderwand

Dense execution remains disabled by default. When enabled, Elderwand supplies
its existing `ResourceManager` through `start_session_with_resources`. The
adapter rejects dense sessions without that manager, and rejects managers that
do not implement temporary reservations. All added reservations share the same
RAM/VRAM ledger as model and request admission. They are atomic, non-evictable,
move-only RAII leases; simultaneous calls cannot reuse one scratch reservation.
A failed reservation returns an error before the corresponding native work.
It does not silently run an unbudgeted fallback.

The audited path currently requires CPU F32, native Safetensors dense
Qwen2/Qwen3 with SiLU, and unquantized KV. CUDA, Metal, Accelerate/MKL, GGUF,
MoE and quantized KV are rejected when this option is enabled because their
additional workspace bounds have not been established. The native Bloom API
still exposes its typed experimental flag; callers outside Elderwand must
provide equivalent admission themselves. `dense_memory_plan(pipeline)` checks
the supported configuration and unchanged weight artifacts before returning a
plan. Its checked arithmetic rejects overflow and unsupported batch shapes.

The leases cover four distinct lifetimes:

1. The runtime reserves an architecture-derived F32 bound for an additional
   streaming template, its rotary tables and vocabulary. A separate loading
   lease covers Safetensors source files, converted weights and rotary
   construction intermediates. Loading headroom is released after construction;
   the template lease lasts until the runtime is destroyed. Request forks share
   template weights, and the batch runtime shares the startup tokenizer through
   `Arc`, avoiding another copy of dictionaries and merge tables.
2. Each request reserves its maximum context, rounded to the KV block size,
   for both native private KV and the compact paged mirror. This is conservative
   additional accounting alongside the scheduler's existing KV estimate. It can
   reject earlier than a future unified KV accounting implementation.
3. Each actual merged forward reserves workspace calculated from its batch
   size, chunk length, history length, layer/head dimensions and vocabulary.
   The bound includes packed and copy-back KV, overlapping paged replacements,
   normalization/rotary intermediates, repeated GQA KV, attention scores and
   softmax, masks, MLP outputs, logits/sampling and GEMM packing. It includes the
   singleton case and does not derive workspace from a fixed magic byte count.
   The lease ends after native execution returns, including error and unwind.
4. CPU dense work uses one private Rayon worker instead of borrowing workers
   with untracked persistent GEMM TLS. The runtime reserves gemm 0.19's L2 slab
   and the two explicitly configured 2 MiB stacks (coalescer and Rayon worker).
   Runtime destruction joins those threads, destroying their TLS before
   returning the reservation. This deliberately limits CPU parallelism.

For reference, with `B` requests, `N` layers, `Hkv` KV heads, head dimension `D`,
old history `T`, chunk length `L` and element width `w`, old private KV occupies
`S = 2*B*N*Hkv*T*D*w`, and the new token increment is
`delta = 2*B*N*Hkv*L*D*w`. During commit, old private caches, combined new cache
and all new private copies can coexist: `3*S + 2*delta`. That KV-only peak is
insufficient as an admission formula: eager attention additionally has shape
`[B, Hq, L, T+L]`. The implemented plan deliberately sums conservative upper
bounds for these storage categories rather than depending on eager drop timing.
Paged extraction materializes compact independent storage, so a small cached
block cannot retain a view of an entire historical KV allocation.

These are bounds for the audited native tensor/GEMM storage and declared worker
stacks, not a process-RSS sandbox or a quota on allocator bookkeeping, HTTP
payloads, tokenizer inputs and arbitrary structured-output schemas. Initial
model loading remains subject to the existing model ticket. GPU workspace and
allocator-wide hard limits require separate implementation. The extra template
is charged even when the startup wrapper can share its storage, favoring a
conservative refusal over unaccounted allocation.

Cancellation and session drop discard native and paged KV before returning the
request lease. The template stays charged while the shared runtime exists. A
forward refused for insufficient workspace has not advanced KV and can be
retried after restoring headroom; HTTP callers receive an error for that attempt.

## Validation

From the enclosing Elderwand workspace:

```sh
cargo test -p bloomai-engine --lib dense_batch --no-default-features --features candle-engine
```

The tests use deterministic nonzero CPU weights and cover B=2, multiple prefill
chunks, successive decode steps, output and layer-KV equivalence to independent
forwards, exactly one model invocation, mixed-shape fallback, disabled dispatch,
per-request sampling parameters, malformed layouts, unrelated weights,
transactional failure after attention has executed, independent storage and
cancel/discard cleanup. A real temporary Safetensors/tokenizer fixture exercises
the native builder and paged cache hooks. This validation does not establish
accelerator correctness or throughput.

Additional Elderwand admission checks:

```sh
cargo test -p elderwand-core temporary_memory_tests
cargo test -p elderwand-adapter-bloom
cargo test -p elderwand-cli native_dense_budget
```

These cover separate/unified memory ledgers, atomic concurrent reservations,
native error/unwind release, actual CPU Qwen template and workspace refusal,
unchanged progress after refusal, retry, sink failure, cancellation, and final
worker/template teardown. They validate accounting and execution correctness;
they are not a measurement of total process RSS.
