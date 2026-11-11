> **Bloom compatibility fork (`bloomai-laya 0.2.20-bloom.1`).** This is a
> separately maintained fork of `laya-decision 0.2.20`, migrated to the official
> Candle 0.11.1 release to remove vulnerable ZIP dependencies. It is not an
> upstream Laya release. See `UPSTREAM.json`, `UPSTREAM-Cargo.toml`, `NOTICE`,
> and `LICENSE` for provenance, reviewed changes, and original attribution.
> The upstream README below is retained verbatim as historical documentation;
> its installation commands refer to the original package.

# laya-decision

A **pure-Rust port** of [Laya](https://github.com/NandhaKishorM/laya) — a multilingual,
non-autoregressive *System-1 decision engine*. Given a **state** (text, JSON, or a conversation
list) and a set of **typed questions** (`choice`, `score`, `noul`), it scores every question in a
**single forward pass** — no text generation. Inference runs natively on
[candle](https://github.com/huggingface/candle) (CPU or Apple-Silicon Metal); routing, language
detection, email cleaning, presets and shortlist are dependency-free.

> The crate is published as **`laya-decision`** (the name `laya` was taken) but is imported as
> `laya`.

```toml
[dependencies]
laya-decision = "0.1"
```

```rust
use laya::{Router, RouteHints, triage_questions};
use serde_json::json;

let router = Router::with_defaults()?;
let state = json!({ "message": "I was charged twice, please refund" });
let result = router.predict(&state, &triage_questions(), &RouteHints::default())?;
println!("{}", serde_json::to_string_pretty(&result.to_json())?);
# Ok::<(), laya::LayaError>(())
```

The candle backend matches the upstream PyTorch model to within `1e-4` across all three
checkpoints (English ModernBERT-large, multilingual mmBERT-base, typed-decisions). Enable the
`metal` feature for Apple-Silicon GPU; disable default features for the weight-free pure-logic
subset (routing, language detection, email cleaning).

See the [repository](https://github.com/GTC6244/Laya-Decision) for the HTTP server
(`laya-decision-serve`), CLI (`laya-decision-cli`), and full documentation.

Apache-2.0. A port of upstream Laya (Apache-2.0); model checkpoints are distributed separately on
the Hugging Face Hub under their own license.
