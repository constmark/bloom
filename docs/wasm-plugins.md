# Bounded WASM processors

Bloom executes binary core WebAssembly processors through
`PluginManager::load_wasm_processor`. The returned `WasmProcessor` implements
the existing `processor::Processor` trait and can be registered in
`ProcessorRegistry`. This is an inline-data processor API; it does not implement
WASM inference engines, WASI, native plugin callbacks, dynamic linking, or a
component-model ABI.

## Run the maintained example

From the Bloom workspace:

```bash
cargo run -p bloomai-engine --no-default-features --example wasm_processor -- 'zebra 中文'
```

The example compiles the checked-in
[`replace_z.wat`](../crates/engine/examples/wasm_processor/replace_z.wat) fixture, loads its
[`plugin.json`](../crates/engine/examples/wasm_processor/plugin.json) through `PluginManager`,
registers it as a normal processor, and prints `[{"Text":"Zebra 中文"}]`.
The guest performs the byte replacement; Unicode bytes are retained. Its guest
global also rejects a second call in the same instance, so repeated successful
processor calls exercise instance isolation. WAT compilation is an example/test
build dependency; production loading accepts `.wasm` binary modules only.

An application supplies a typed `ProcessorSpec` whose name equals the plugin
manifest name, whose version is `"1"`, and whose input/output modalities are
nonempty. ABI v1 rejects nonempty `parameters`; it does not silently ignore
configuration. The ordinary manifest platform/security checks still apply.
Loading validates the module and negotiates its ABI in a disposable instance;
`validate_entry_point` remains a non-executing file pre-check.

## ABI version 1

The module must export exactly compatible types for these names:

| Export | WASM signature | Meaning |
| --- | --- | --- |
| `memory` | 32-bit linear memory | The sole linear memory used by the host |
| `bloom_plugin_abi_version` | `() -> i32` | Must return `1` |
| `bloom_alloc` | `(i32 length) -> i32 pointer` | Return a writable range for the serialized request |
| `bloom_process` | `(i32 pointer, i32 length) -> i64` | Return the packed output range |

The high 32 bits of the return value are the **output byte length**; the low
32 bits are the **output pointer**. Both fields are interpreted as unsigned.
Input and output may overlap for an in-place processor. Empty, oversized, or
out-of-memory ranges are errors. There is no deallocation export: the entire
instance and Store are dropped after each invocation, including failures.
Guest traps return errors to the caller; partial results are never published.

Input and output use the same UTF-8 JSON envelope:

```json
{
  "schema_version": 1,
  "object": "bloom.processor.blocks",
  "blocks": [{"Text": "zebra 中文"}, {"Tokens": [1, 2]}, {"Tensor": [1.0, -2.0]}]
}
```

There must be 1–64 blocks. Unknown envelope fields, wrong versions, invalid
JSON, invalid core `DataBlock` values, and undeclared modalities fail closed.
`Text` and `Tokens` use `Modality::Text`; inline PCM uses `Audio`; inline image
and video frames use `Vision`; tensors, world state and actions use `Multi`
(the core modality enum has no separate tensor variant). Declaring `Multi`
accepts those inline domains. `AudioFile` and other file-reference authority
are not available: callers must provide inline data, and guests cannot return
a local-file instruction for another processor to execute.

## Resource and cancellation boundary

The runtime pins `wasmi = 2.0.0` with the portable interpreter, validation,
and standard-library support. Runtime WAT parsing, memory64 and WASI are not
enabled. Module start functions and **all imports** are rejected; no host
filesystem, environment, network, clock, random generator or process service
is linked. Every invocation creates fresh guest memory and globals, with a
fresh total fuel budget shared by version negotiation, allocation and processing.

| Limit | Default | Hard maximum |
| --- | --- | --- |
| Binary module | 2 MiB | 2 MiB |
| Guest linear memory | 8 MiB | 64 MiB |
| Serialized input/output and raw inline payload, each | 1 MiB | 8 MiB, also bounded by guest memory |
| Total execution fuel | 5,000,000 | 50,000,000 |
| Execution deadline | 2 seconds | 30 seconds |
| Concurrent invocations per processor | 4 | 32 |
| Blocks per envelope | 64 | 64 |

The host also limits one instance, one memory, one table with 4,096 elements,
128 recursive calls, and a 256 KiB interpreter value stack. Growth failure
traps. Module translation is eager with Wasmi's strict compilation limits;
serialized bytes are bounded before decoding, and the block-count limit is
applied while deserializing. Host JSON and compiled-module bookkeeping add
memory beyond the guest linear-memory budget; these are not host-RSS budgets.

`process_cancellable(blocks, &AtomicBool)` checks cancellation and the execution
deadline before calls, between fuel grants, and before returning a result.
The host normally grants 10,000 fuel at a time, resuming suspended execution;
a bulk instruction may require a larger grant within the same remaining total
budget. Thus an import-free infinite loop remains cancellable and fuel-bounded.
Compilation itself is bounded by module size and strict validation, not an
interruptible execution deadline. The synchronous `Processor::process` method
uses the same limits without an external cancellation flag. Concurrent calls
beyond the configured limit fail immediately instead of queueing unbounded work.

The [Wasmi fuel API](https://docs.rs/wasmi/2.0.0/wasmi/struct.Store.html#method.set_fuel),
[resumable invocation](https://docs.rs/wasmi/2.0.0/wasmi/enum.TypedResumableCall.html),
[resource limiter](https://docs.rs/wasmi/2.0.0/wasmi/struct.StoreLimitsBuilder.html),
and [strict compilation limits](https://docs.rs/wasmi/2.0.0/wasmi/struct.EnforcedLimits.html)
describe the interpreter mechanisms used by this boundary.

## Regression evidence

```bash
cargo test -p bloomai-engine --no-default-features plugin::wasm
```

The maintained gate executes real WASM through `PluginManager` and
`ProcessorRegistry`, including Unicode text, token/tensor round trips, repeated
calls with fresh globals, memory growth and repeated fuel suspension. Malicious
fixtures exercise infinite loops, cancellation while running, capacity recovery,
traps, allocator/output pointers, memory limits, imports, start functions,
invalid ABI/envelopes, file references, invalid values, byte limits, excessive
block counts, unavailable files and symlinks. These CPU interpreter tests do
not establish vendor-device or arbitrary third-party model compatibility.
