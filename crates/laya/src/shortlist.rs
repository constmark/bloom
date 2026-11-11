//! Opt-in embedding shortlist for high-cardinality choice questions. Ported from
//! `laya/shortlist.py`. The ranking core (`shortlist_choice`, `predict_shortlist`) takes a
//! caller-supplied embedding function `embed_fn: Fn(&[String]) -> Vec<Vec<f64>>`;
//! [`embed_fn_from_agent`] builds one from a loaded checkpoint's encoder.

use crate::agent::Agent;
use crate::common::{render_options, serialize_state, InternalQ, QType};
use crate::error::{LayaError, Result};
use crate::Questions;
use serde_json::{json, Map, Value};

pub const DEFAULT_SHORTLIST_K: usize = 20;
/// Default group size for [`predict_tournament`], matching `DEFAULT_TOURNAMENT_GROUP` upstream.
pub const DEFAULT_TOURNAMENT_GROUP: usize = 16;
/// Defaults matching `embed_fn_from_agent` in `laya/shortlist.py`.
pub const DEFAULT_EMBED_MAX_LENGTH: usize = 512;
pub const DEFAULT_EMBED_BATCH_SIZE: usize = 32;

/// Ranking result: `(labels, cosine scores or None, passthrough, total option count)`.
type Ranked = (Vec<String>, Option<Vec<f64>>, bool, usize);

/// Build an `embed_fn` that mean-pools the checkpoint encoder already loaded on `agent`.
///
/// The returned closure embeds a list of strings for [`shortlist_choice`]/[`predict_shortlist`].
/// A dedicated bi-encoder passed directly as `embed_fn` will usually shortlist better; this helper
/// is for callers who only have the Laya checkpoint in memory. On an embedding failure it returns
/// an empty vector, which surfaces as a clear length-mismatch error from the ranker.
pub fn embed_fn_from_agent(
    agent: &Agent,
    max_length: usize,
    batch_size: usize,
) -> impl Fn(&[String]) -> Vec<Vec<f64>> + '_ {
    move |texts: &[String]| match agent.embed(texts, max_length, batch_size) {
        Ok(rows) => rows
            .into_iter()
            .map(|row| row.into_iter().map(|x| x as f64).collect())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Return the top-`k` choice labels for `state`, ranked by cosine similarity of `embed_fn`
/// vectors (query first, then one per option in criteria order). When `k >= n`, every label is
/// returned in original order and `embed_fn` is not called.
///
/// Ranking is a signed cosine, not a similarity floor: a label that scores 0 (no signal, or a
/// non-finite vector treated as one) outranks an earlier label that scored negative, and `k` drops
/// the negative labels first. Ties keep the earlier label.
pub fn shortlist_choice<F>(
    state: &Value,
    criteria: &Value,
    embed_fn: F,
    k: usize,
    instructions: Option<&str>,
) -> Result<Vec<String>>
where
    F: Fn(&[String]) -> Vec<Vec<f64>>,
{
    Ok(shortlist_choice_scored(state, criteria, embed_fn, k, instructions)?.0)
}

/// Like [`shortlist_choice`] but also returns the signed cosine of each kept label, in rank order
/// — the same values [`predict_shortlist`] reports in its `shortlist` metadata. The score vector is
/// `None` when nothing was dropped (passthrough), exactly as in that metadata. Mirrors
/// `shortlist_choice(..., return_scores=True)` in `laya/shortlist.py`.
pub fn shortlist_choice_scored<F>(
    state: &Value,
    criteria: &Value,
    embed_fn: F,
    k: usize,
    instructions: Option<&str>,
) -> Result<(Vec<String>, Option<Vec<f64>>)>
where
    F: Fn(&[String]) -> Vec<Vec<f64>>,
{
    let (labels, scores, _passthrough, _n) = rank(state, criteria, embed_fn, k, instructions)?;
    Ok((labels, scores))
}

/// Shortlist each choice question, then run the agent once on the reduced question set.
/// Returns the result JSON (Jev-shaped) with an added `shortlist` entry.
pub fn predict_shortlist<F>(
    agent: &Agent,
    state: &Value,
    questions: &Questions,
    embed_fn: F,
    k: usize,
) -> Result<Value>
where
    F: Fn(&[String]) -> Vec<Vec<f64>>,
{
    let checked = check_k(k)?;
    let mut reduced: Questions = Questions::new();
    let mut meta = Map::new();
    let mut warned: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (qid, qdef) in questions {
        let is_choice = qdef.get("type").and_then(|v| v.as_str()) == Some("choice");
        if !qdef.is_object() || !is_choice {
            reduced.insert(qid.clone(), qdef.clone());
            continue;
        }
        let criteria = qdef.get("criteria").ok_or_else(|| {
            LayaError::InvalidQuestion(format!(
                "question {:?} is a choice but has no criteria",
                qid
            ))
        })?;
        let instructions = qdef.get("instructions").and_then(|v| v.as_str());
        let (labels, scores, passthrough, n) =
            rank(state, criteria, &embed_fn, checked, instructions)?;
        meta.insert(
            qid.clone(),
            json!({
                "labels": labels,
                "scores": scores,
                "k": checked,
                "n": n,
                "passthrough": passthrough,
            }),
        );
        if passthrough {
            reduced.insert(qid.clone(), qdef.clone());
            continue;
        }
        let mut updated = qdef.clone();
        updated["criteria"] = subset_criteria(criteria, &labels);
        // A narrowed choice can no longer honour an order sized for the caller's labels; drop it.
        crate::common::drop_stale_option_order(qid, &mut updated, labels.len(), &mut warned);
        reduced.insert(qid.clone(), updated);
    }

    let result = agent.system_one(state, &reduced)?;
    let mut out = result.to_json();
    out.as_object_mut()
        .unwrap()
        .insert("shortlist".to_string(), Value::Object(meta));
    Ok(out)
}

/// Per-choice-question tournament bookkeeping.
struct TournamentEntry {
    /// The labels still in the running, in criteria order.
    labels: Vec<String>,
    /// The original label count.
    n: usize,
    /// How many elimination rounds this question went through.
    rounds: usize,
}

/// Narrow each large choice question by elimination, then call the agent once more. Unlike
/// [`predict_shortlist`] this needs no embedder: a choice with more than `group_size` labels is cut,
/// in criteria order, into near-equal groups of at most `group_size`, and one `system_one` call
/// answers every group of every such question (they share one forward pass). Each group's winner
/// advances; rounds repeat until no choice has more than `group_size` labels left. The final call
/// answers the full `questions` with each contested choice cut to its finalists.
///
/// The returned dict is the final call's result plus a `tournament` entry: `tournament[qid]` holds
/// `labels` (the finalists, in criteria order), `n` (the original label count) and `rounds`.
/// Probabilities come from the final call, so a tournament choice's probabilities are over its
/// finalists only. The caller's `questions` is not mutated. Mirrors `predict_tournament` upstream.
pub fn predict_tournament(
    agent: &Agent,
    state: &Value,
    questions: &Questions,
    group_size: usize,
) -> Result<Value> {
    Ok(predict_tournament_with_usage(agent, state, questions, group_size)?.0)
}

/// Like [`predict_tournament`], but also returns the total `input_tokens` summed over every
/// forward pass it ran — each elimination round plus the final answer. The returned [`Value`] is
/// byte-identical to [`predict_tournament`]'s (its `usage` is still the final call's); the extra
/// number is for a caller that reports the whole operation's token cost. `laya-serve` uses it to
/// fill `usage.input_tokens` when answering a large choice by elimination (upstream
/// `serve._predict_with_tournament`, which sums `input_tokens` over every recorded call).
pub fn predict_tournament_with_usage(
    agent: &Agent,
    state: &Value,
    questions: &Questions,
    group_size: usize,
) -> Result<(Value, usize)> {
    if group_size < 2 {
        return Err(LayaError::InvalidQuestion(format!(
            "group_size must be an integer of at least 2, got {}",
            group_size
        )));
    }

    // meta, in question order, for every choice question.
    let mut meta: Vec<(String, TournamentEntry)> = Vec::new();
    for (qid, qdef) in questions {
        if qdef.get("type").and_then(|v| v.as_str()) == Some("choice") {
            let criteria = qdef.get("criteria").ok_or_else(|| {
                LayaError::InvalidQuestion(format!(
                    "question {:?} is a choice but has no criteria",
                    qid
                ))
            })?;
            let labels = criteria_keys(criteria)?;
            let n = labels.len();
            meta.push((
                qid.clone(),
                TournamentEntry {
                    labels,
                    n,
                    rounds: 0,
                },
            ));
        }
    }

    // A choice that goes through a round holds fewer labels than the caller's, so an
    // `option_order` sized for them no longer fits; drop it once per question (upstream #958).
    let warned: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
    let cut = |qid: &str, labels: &[String]| -> Value {
        let qdef = &questions[qid];
        let mut updated = qdef.clone();
        updated["criteria"] = subset_criteria(qdef.get("criteria").unwrap(), labels);
        crate::common::drop_stale_option_order(
            qid,
            &mut updated,
            labels.len(),
            &mut warned.borrow_mut(),
        );
        updated
    };

    // Summed over every `system_one` call below (each round and the final), the token cost of the
    // whole elimination, for a caller that reports it (upstream sums each recorded call's usage).
    let mut total_input_tokens: usize = 0;

    loop {
        // Each element: (meta index, this group's labels), in group order across all questions.
        let mut groups: Vec<(usize, Vec<String>)> = Vec::new();
        for (idx, (_qid, e)) in meta.iter().enumerate() {
            let len = e.labels.len();
            let parts = len.div_ceil(group_size);
            if parts > 1 {
                for i in 0..parts {
                    let lo = i * len / parts;
                    let hi = (i + 1) * len / parts;
                    groups.push((idx, e.labels[lo..hi].to_vec()));
                }
            }
        }
        if groups.is_empty() {
            break;
        }
        let mut round_qs: Questions = Questions::new();
        for (i, (idx, labels)) in groups.iter().enumerate() {
            let qid = meta[*idx].0.clone();
            round_qs.insert(i.to_string(), cut(&qid, labels));
        }
        let round_result = agent.system_one(state, &round_qs)?;
        total_input_tokens += round_result.input_tokens;
        let answers_json = round_result.to_json();
        let answers = answers_json
            .get("answers")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                LayaError::InvalidQuestion("agent result missing 'answers'".to_string())
            })?;
        let mut winners: Vec<Vec<String>> = vec![Vec::new(); meta.len()];
        for (i, (idx, _labels)) in groups.iter().enumerate() {
            let choice = answers
                .get(&i.to_string())
                .and_then(|a| a.get("choice"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            winners[*idx].push(choice);
        }
        for (idx, (_qid, e)) in meta.iter_mut().enumerate() {
            if !winners[idx].is_empty() {
                e.labels = std::mem::take(&mut winners[idx]);
                e.rounds += 1;
            }
        }
    }

    let mut final_qs: Questions = questions.clone();
    for (qid, e) in &meta {
        if e.rounds > 0 {
            final_qs.insert(qid.clone(), cut(qid, &e.labels));
        }
    }
    let final_result = agent.system_one(state, &final_qs)?;
    total_input_tokens += final_result.input_tokens;
    let mut out = final_result.to_json();
    let mut tmeta = Map::new();
    for (qid, e) in &meta {
        tmeta.insert(
            qid.clone(),
            json!({ "labels": e.labels, "n": e.n, "rounds": e.rounds }),
        );
    }
    out.as_object_mut()
        .unwrap()
        .insert("tournament".to_string(), Value::Object(tmeta));
    Ok((out, total_input_tokens))
}

fn rank<F>(
    state: &Value,
    criteria: &Value,
    embed_fn: F,
    k: usize,
    instructions: Option<&str>,
) -> Result<Ranked>
where
    F: Fn(&[String]) -> Vec<Vec<f64>>,
{
    let checked = check_k(k)?;
    let keys = criteria_keys(criteria)?;
    let n = keys.len();
    if checked >= n {
        return Ok((keys, None, true, n));
    }
    let query = query_text(state, instructions);
    let option_texts = option_texts(criteria)?;
    let mut texts = Vec::with_capacity(1 + option_texts.len());
    texts.push(query);
    texts.extend(option_texts);
    let matrix = embed_fn(&texts);
    if matrix.len() != texts.len() {
        return Err(LayaError::InvalidQuestion(format!(
            "embed_fn must return {} vectors, got {}",
            texts.len(),
            matrix.len()
        )));
    }
    let sims = cosine(&matrix[0], &matrix[1..]);
    // stable sort by descending similarity (ties keep earlier index)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sims[b]
            .partial_cmp(&sims[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(checked);
    let labels = order.iter().map(|&i| keys[i].clone()).collect();
    let scores = order.iter().map(|&i| sims[i]).collect();
    Ok((labels, Some(scores), false, n))
}

fn check_k(k: usize) -> Result<usize> {
    if k < 1 {
        return Err(LayaError::InvalidQuestion(
            "k must be a positive integer".to_string(),
        ));
    }
    Ok(k)
}

fn criteria_keys(criteria: &Value) -> Result<Vec<String>> {
    let keys: Vec<String> = match criteria {
        Value::Object(m) => m.keys().cloned().collect(),
        Value::Array(a) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| crate::common::py_json(v))
            })
            .collect(),
        _ => {
            return Err(LayaError::InvalidQuestion(
                "choice criteria must be a dict or list".to_string(),
            ))
        }
    };
    if keys.is_empty() {
        return Err(LayaError::InvalidQuestion(
            "choice criteria must contain at least one option".to_string(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for k in &keys {
        if !seen.insert(k.clone()) {
            return Err(LayaError::InvalidQuestion(format!(
                "choice criteria label {:?} is duplicated",
                k
            )));
        }
    }
    Ok(keys)
}

fn option_texts(criteria: &Value) -> Result<Vec<String>> {
    let q = InternalQ {
        t: QType::Choice,
        ins: String::new(),
        crit: normalize_choice_crit(criteria),
        labels: None,
        option_order: None,
    };
    render_options(&q)
}

fn normalize_choice_crit(criteria: &Value) -> Value {
    match criteria {
        Value::Array(a) => {
            let mut m = Map::new();
            for c in a {
                let key = c
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| crate::common::py_json(c));
                m.insert(key, Value::Null);
            }
            Value::Object(m)
        }
        other => other.clone(),
    }
}

fn query_text(state: &Value, instructions: Option<&str>) -> String {
    let body = serialize_state(state);
    match instructions {
        None | Some("") => body,
        Some(ins) => format!("{}\n{}", ins, body),
    }
}

fn subset_criteria(criteria: &Value, labels: &[String]) -> Value {
    match criteria {
        Value::Object(m) => {
            let mut out = Map::new();
            for l in labels {
                if let Some(v) = m.get(l) {
                    out.insert(l.clone(), v.clone());
                }
            }
            Value::Object(out)
        }
        _ => Value::Array(labels.iter().map(|l| json!(l)).collect()),
    }
}

fn cosine(query: &[f64], docs: &[Vec<f64>]) -> Vec<f64> {
    let qn = query.iter().map(|v| v * v).sum::<f64>().sqrt();
    if qn == 0.0 || docs.is_empty() {
        return vec![0.0; docs.len()];
    }
    docs.iter()
        .map(|d| {
            let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let denom = dn * qn;
            if denom > 0.0 {
                let dot: f64 = d.iter().zip(query).map(|(a, b)| a * b).sum();
                (dot / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic 2-D embeddings keyed by first char, so cosine ranking is predictable without a
    // model: the query points at the "b" direction, so "b" ranks above "a"/"c".
    fn stub_embed(texts: &[String]) -> Vec<Vec<f64>> {
        texts
            .iter()
            .map(|t| match t.chars().next() {
                Some('a') => vec![1.0, 0.0],
                Some('b') => vec![0.0, 1.0],
                Some('c') => vec![0.7, 0.7],
                _ => vec![0.0, 1.0], // the query points at "b"
            })
            .collect()
    }

    #[test]
    fn scored_returns_signed_cosine_in_rank_order() {
        let criteria = json!(["a", "b", "c"]);
        let (labels, scores) =
            shortlist_choice_scored(&json!("question"), &criteria, stub_embed, 2, None).unwrap();
        assert_eq!(labels, vec!["b".to_string(), "c".to_string()]);
        let scores = scores.expect("scores present when labels were dropped");
        assert_eq!(scores.len(), 2);
        // Descending, and "b" is the exact match (cosine 1.0).
        assert!(scores[0] >= scores[1]);
        assert!((scores[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn scored_passthrough_has_no_scores() {
        let criteria = json!(["a", "b"]);
        let (labels, scores) =
            shortlist_choice_scored(&json!("q"), &criteria, stub_embed, 5, None).unwrap();
        assert_eq!(labels, vec!["a".to_string(), "b".to_string()]);
        assert!(scores.is_none());
    }
}
