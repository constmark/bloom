//! Parity tests against golden fixtures dumped from the upstream Python `laya` package
//! (see `scripts/dump_golden.py`). These exercise the pure-logic surface (no model weights):
//! run with `cargo test -p laya --no-default-features --test parity`.

use laya::email::clean_email_body;
use laya::lang::analyse;
use laya::router::{RouteHints, Router};
use laya::Questions;
use serde_json::Value;
use std::path::PathBuf;

fn golden(name: &str) -> Vec<Value> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "golden", name]
        .iter()
        .collect();
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e} (run scripts/dump_golden.py)", path.display()));
    serde_json::from_slice(&bytes).expect("valid golden json")
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 5e-5
}

#[test]
fn lang_parity() {
    let mut failures = Vec::new();
    for case in golden("lang.json") {
        let input = case["input"].as_str().unwrap();
        let det = analyse(&Value::String(input.to_string()));
        let want_lang = case["language"].as_str().map(|s| s.to_string());
        let mut errs = Vec::new();
        if det.script != case["script"].as_str().unwrap() {
            errs.push(format!("script {} != {}", det.script, case["script"]));
        }
        if det.language != want_lang {
            errs.push(format!("language {:?} != {:?}", det.language, want_lang));
        }
        let want_mixed = case["mixed_segment"].as_str().map(|s| s.to_string());
        if det.mixed_segment != want_mixed {
            errs.push(format!(
                "mixed_segment {:?} != {:?}",
                det.mixed_segment, want_mixed
            ));
        }
        if det.is_english != case["is_english"].as_bool().unwrap() {
            errs.push(format!("is_english {}", det.is_english));
        }
        if det.language_undecided != case["language_undecided"].as_bool().unwrap() {
            errs.push(format!("language_undecided {}", det.language_undecided));
        }
        if !approx(det.diacritic_rate, case["diacritic_rate"].as_f64().unwrap()) {
            errs.push(format!(
                "diacritic_rate {} != {}",
                det.diacritic_rate, case["diacritic_rate"]
            ));
        }
        if !approx(
            det.non_latin_fraction,
            case["non_latin_fraction"].as_f64().unwrap(),
        ) {
            errs.push(format!(
                "non_latin_fraction {} != {}",
                det.non_latin_fraction, case["non_latin_fraction"]
            ));
        }
        if !errs.is_empty() {
            failures.push(format!("input {:?}: {}", input, errs.join("; ")));
        }
    }
    assert!(
        failures.is_empty(),
        "lang parity failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn router_parity() {
    let router = Router::with_defaults().unwrap();
    let empty = Questions::new();
    let mut failures = Vec::new();
    for case in golden("router.json") {
        let state = case["input"].clone();
        let d = router
            .route(&state, Some(&empty), &RouteHints::default())
            .unwrap();
        if d.model != case["model"].as_str().unwrap() {
            failures.push(format!(
                "input {}: model {} != {}",
                case["input"], d.model, case["model"]
            ));
        }
        if d.reason != case["reason"].as_str().unwrap() {
            failures.push(format!(
                "input {}: reason\n  got:  {}\n  want: {}",
                case["input"], d.reason, case["reason"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "router parity failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn email_parity() {
    let mut failures = Vec::new();
    for case in golden("email.json") {
        let input = case["input"].as_str().unwrap();
        let got = clean_email_body(input);
        let want = case["output"].as_str().unwrap();
        if got != want {
            failures.push(format!(
                "input {:?}:\n  got:  {:?}\n  want: {:?}",
                input, got, want
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "email parity failures:\n{}",
        failures.join("\n")
    );
}
