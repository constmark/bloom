//! Demo: encoder-embedding shortlist for a high-cardinality choice question.
//!   cargo run --release --example shortlist
use laya::agent::{Agent, LoadOptions};
use laya::shortlist::{embed_fn_from_agent, shortlist_choice};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model =
        std::env::var("LAYA_MODEL").unwrap_or_else(|_| "convaiinnovations/laya".to_string());
    let device = std::env::var("LAYA_DEVICE").ok().filter(|s| !s.is_empty());
    let agent = Agent::load(
        &model,
        LoadOptions {
            device,
            ..Default::default()
        },
    )?;

    // A large label set (25 intents); shortlist to the 5 most relevant before scoring.
    let criteria = json!({
        "refund": null, "cancel_subscription": null, "reset_password": null, "update_billing": null,
        "report_bug": null, "feature_request": null, "track_shipment": null, "return_item": null,
        "change_address": null, "upgrade_plan": null, "downgrade_plan": null, "dispute_charge": null,
        "account_locked": null, "two_factor_help": null, "api_error": null, "integration_help": null,
        "pricing_question": null, "demo_request": null, "gdpr_request": null, "delete_account": null,
        "invoice_copy": null, "tax_question": null, "partnership": null, "press_inquiry": null,
        "other": null
    });
    let state =
        json!({ "message": "I was double charged and want my money back for the duplicate." });

    let embed = embed_fn_from_agent(&agent, 512, 32);
    let top = shortlist_choice(&state, &criteria, &embed, 5, None)?;
    println!("top-5 shortlisted labels: {top:?}");

    let dim = agent.embed(&["hello world".to_string()], 512, 32)?[0].len();
    println!("encoder embedding dim: {dim}");
    assert!(top.len() == 5, "expected 5 labels");
    println!("OK");
    Ok(())
}
