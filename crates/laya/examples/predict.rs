//! Smoke test / demo: download the English checkpoint, run a preset, print the answers.
//!
//!   cargo run --release --example predict
//!
//! Set LAYA_MODEL to a local checkpoint dir or another repo id; LAYA_SUBFOLDER for a bundle
//! subfolder (e.g. "multilingual").

use laya::agent::{Agent, LoadOptions};
use laya::{triage_questions, State};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model =
        std::env::var("LAYA_MODEL").unwrap_or_else(|_| "convaiinnovations/laya".to_string());
    let subfolder = std::env::var("LAYA_SUBFOLDER").ok();
    let device = std::env::var("LAYA_DEVICE").ok().filter(|s| !s.is_empty());

    let t = Instant::now();
    let agent = Agent::load(
        &model,
        LoadOptions {
            subfolder,
            device: device.clone(),
            ..Default::default()
        },
    )?;
    println!(
        "loaded {model} on {} in {:?}",
        device.as_deref().unwrap_or("cpu"),
        t.elapsed()
    );

    let state: State = serde_json::json!({
        "message": "I was charged twice this month and I'm furious. Refund me now or I'm leaving."
    });
    let questions = triage_questions();

    let t = Instant::now();
    let result = agent.system_one(&state, &questions)?;
    println!("forward in {:?}\n", t.elapsed());
    println!("{}", serde_json::to_string_pretty(&result.to_json())?);

    // Sanity: every probability distribution should sum to ~1.
    for (qid, ans) in &result.answers {
        if let Some(probs) = ans.get("probabilities").and_then(|p| p.as_object()) {
            let sum: f64 = probs.values().filter_map(|v| v.as_f64()).sum();
            assert!((sum - 1.0).abs() < 1e-2, "{qid} probabilities sum to {sum}");
        }
    }
    println!("\nOK: probability distributions sum to 1.");
    Ok(())
}
