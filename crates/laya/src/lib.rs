//! # laya
//!
//! Pure-Rust port of the [Laya](https://github.com/NandhaKishorM/laya) non-autoregressive
//! System-1 decision engine (the **Laya-Decision** project).
//!
//! Given a *state* (text, JSON, or a conversation list) and a set of *typed questions*
//! (`choice`, `score`, `noul`), Laya scores every question in a single forward pass — no
//! text generation. Inference runs natively on [candle]; the routing, language-detection,
//! email-cleaning, preset and shortlist logic is dependency-free.
//!
//! The pure-logic modules build without the `model` feature; enabling it (on by default)
//! pulls in candle + the Hugging Face tokenizer and Hub download.
//!
//! [candle]: https://github.com/huggingface/candle

use indexmap::IndexMap;
use serde_json::Value;

pub mod common;
pub mod email;
pub mod error;
pub mod lang;
pub mod presets;
pub mod router;

#[cfg(feature = "model")]
pub mod agent;
#[cfg(feature = "model")]
pub mod model;
#[cfg(feature = "model")]
pub mod shortlist;
#[cfg(feature = "model")]
pub mod tokenizer;

/// A state to evaluate: a string, a JSON object, or a conversation list — represented as a
/// [`serde_json::Value`]. A bare string is `Value::String`.
pub type State = Value;

/// An ordered map of question id → question definition. Order is significant (it determines
/// choice-label and probability ordering), so this is an [`IndexMap`].
pub type Questions = IndexMap<String, Value>;

pub use error::{LayaError, Result};

pub use common::{QType, TEMP_MAX, TEMP_MIN};
pub use email::{clean_email_body, email_state};
pub use lang::{analyse, detect_script, is_english};
pub use presets::{
    email_questions, guard_questions, moderation_questions, router_questions, triage_questions,
};
pub use router::{RouteDecision, RouteHints, Router, RouterOptions};

#[cfg(feature = "model")]
pub use agent::{Agent, SystemOneResult};
#[cfg(feature = "model")]
pub use shortlist::{embed_fn_from_agent, predict_shortlist, shortlist_choice};

/// Crate version, mirroring the upstream Laya release this port tracks.
pub const UPSTREAM_VERSION: &str = "0.4.22";
