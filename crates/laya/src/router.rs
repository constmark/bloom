//! Route a request to the Laya checkpoint best suited to it. Ported from `laya/router.py`.
//!
//! The routing *decision* (`route`) is pure and always available. Loading/running checkpoints
//! (`load`, `predict`) requires the `model` feature.

use crate::lang::{analyse, Analysis};
use crate::{LayaError, Questions, Result, State};
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// The hub repo that bundles all three checkpoints.
pub const BUNDLE_REPO: &str = "convaiinnovations/laya";

/// A checkpoint location: `(repo_or_path, optional_subfolder)`.
pub type ModelSpec = (String, Option<String>);

fn default_models() -> Vec<(&'static str, ModelSpec)> {
    vec![
        ("english", (BUNDLE_REPO.to_string(), None)),
        (
            "multilingual",
            (BUNDLE_REPO.to_string(), Some("multilingual".to_string())),
        ),
        (
            "typed-decisions",
            (BUNDLE_REPO.to_string(), Some("typed-decisions".to_string())),
        ),
    ]
}

fn standalone_models() -> Vec<(&'static str, ModelSpec)> {
    vec![
        ("english", ("convaiinnovations/laya".to_string(), None)),
        (
            "multilingual",
            ("convaiinnovations/laya-multilingual".to_string(), None),
        ),
        (
            "typed-decisions",
            ("convaiinnovations/laya-typed-decisions".to_string(), None),
        ),
    ]
}

const CANONICAL: [&str; 3] = ["english", "multilingual", "typed-decisions"];

fn aliases() -> &'static [(&'static str, &'static str)] {
    &[
        ("en", "english"),
        ("laya", "english"),
        ("default", "english"),
        ("multi", "multilingual"),
        ("ml", "multilingual"),
        ("laya-multilingual", "multilingual"),
        ("typed", "typed-decisions"),
        ("typed_decisions", "typed-decisions"),
        ("laya-typed-decisions", "typed-decisions"),
        ("decisions", "typed-decisions"),
    ]
}

/// Question-id signatures of the four typed-decisions workflows.
fn typed_decision_workflows() -> &'static [(&'static str, &'static [&'static str])] {
    &[
        (
            "agent_trace_observability",
            &["action", "needs_review", "outcome", "risk", "urgency"],
        ),
        (
            "customer_service",
            &["action", "category", "churn_risk", "needs_human", "urgency"],
        ),
        (
            "invoice_processing",
            &[
                "discrepancy_severity",
                "disposition",
                "duplicate",
                "matches_order",
                "urgency",
            ],
        ),
        (
            "security_incidents",
            &[
                "credential_compromise",
                "disposition",
                "severity",
                "true_positive",
                "urgency",
            ],
        ),
    ]
}

const ENGLISH_SUBTAGS: [&str; 3] = ["en", "eng", "english"];

/// Codes that are valid `$LANG` values but name no language, so they say nothing about the state.
/// `c`/`posix`/`c.utf-8` are what minimal images ship; the ISO 639-2 special codes `und`
/// (undetermined), `zxx` (no linguistic content) and `mul` (multiple) say the same in the
/// standard's own vocabulary. They abstain — like the blank case — instead of pinning English
/// text to the multilingual checkpoint.
const LANGUAGE_AGNOSTIC_CODES: [&str; 5] = ["c", "posix", "und", "zxx", "mul"];

/// Python `repr()` of a string: single quotes unless the string contains a single (but not
/// double) quote. Used so routing `reason` strings match the Python `%r` formatting exactly.
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn repo_str(spec: &ModelSpec) -> String {
    match &spec.1 {
        Some(sub) => format!("{}/{}", spec.0, sub),
        None => spec.0.clone(),
    }
}

/// Normalise a model name (accepting aliases) to a canonical checkpoint key.
pub fn normalise_name(name: &str) -> Result<String> {
    let key = name.trim().to_lowercase();
    let key = aliases()
        .iter()
        .find(|(a, _)| *a == key)
        .map(|(_, v)| v.to_string())
        .unwrap_or(key);
    if CANONICAL.contains(&key.as_str()) {
        Ok(key)
    } else {
        let mut alias_names: Vec<&str> = aliases().iter().map(|(a, _)| *a).collect();
        alias_names.sort();
        Err(LayaError::Config(format!(
            "unknown model {:?}; choose one of {:?} (or an alias: {:?})",
            name,
            {
                let mut c = CANONICAL.to_vec();
                c.sort();
                c
            },
            alias_names
        )))
    }
}

/// Registry spec for a checkpoint name or alias, or `None` when it is not one.
///
/// The non-raising sibling of [`normalise_name`], for callers that also accept things the registry
/// knows nothing about — a Hub repo id, a local directory. Those pass through untouched; a name or
/// alias the registry does know resolves to its `(repo, subfolder)` pair, so `Agent::load`
/// (`"typed-decisions"`, `"ml"`, …) and `Router` name the same checkpoint from one table (upstream
/// `resolve_model_spec`).
pub fn resolve_model_spec(name: &str) -> Option<ModelSpec> {
    let key = normalise_name(name).ok()?;
    default_models()
        .into_iter()
        .find(|(k, _)| *k == key)
        .map(|(_, spec)| spec)
}

/// Name of the typed-decisions workflow whose question ids these are (exact id-set match), else None.
pub fn match_typed_decisions_workflow(questions: &Questions) -> Option<String> {
    let ids: BTreeSet<&str> = questions.keys().map(|s| s.as_str()).collect();
    for (wf, sig) in typed_decision_workflows() {
        let sig_set: BTreeSet<&str> = sig.iter().copied().collect();
        if ids == sig_set {
            return Some((*wf).to_string());
        }
    }
    None
}

/// `Some(true/false)` for a language code, or `None` when the code identifies nothing.
/// Accepts `"en"`, `"EN"`, `"en-US"`, `"en_US"`, `"en_US.UTF-8"`.
pub fn english_from_code(value: Option<&str>) -> Option<bool> {
    let value = value?;
    let code = value.trim().to_lowercase();
    if code.is_empty() {
        return None;
    }
    let code = code.split('.').next().unwrap_or(""); // en_US.UTF-8 -> en_US
    let primary = code.replace('_', "-");
    let primary = primary.split('-').next().unwrap_or(""); // en_US -> en
    if primary.is_empty() || LANGUAGE_AGNOSTIC_CODES.contains(&primary) {
        return None;
    }
    Some(ENGLISH_SUBTAGS.contains(&primary))
}

/// The routing outcome: which checkpoint, why, and what was detected.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub model: String,
    pub repo: String,
    pub reason: String,
    pub detection: Option<Analysis>,
    pub workflow: Option<String>,
}

impl RouteDecision {
    /// Serialise like the Python `RouteDecision` dict (used for the `routing` key of a result).
    pub fn to_json(&self) -> Value {
        json!({
            "model": self.model,
            "repo": self.repo,
            "reason": self.reason,
            "detection": self.detection.as_ref().map(analysis_to_json),
            "workflow": self.workflow,
        })
    }
}

fn analysis_to_json(a: &Analysis) -> Value {
    json!({
        "script": a.script,
        "script_profile": a.script_profile.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "language": a.language,
        "is_english": a.is_english,
        "language_undecided": a.language_undecided,
        "diacritic_rate": a.diacritic_rate,
        "non_latin_fraction": a.non_latin_fraction,
        "mixed_segment": a.mixed_segment,
    })
}

/// Per-call routing hints (mirror the keyword args of Python `Router.route`).
#[derive(Debug, Clone, Default)]
pub struct RouteHints<'a> {
    pub model: Option<&'a str>,
    pub task: Option<&'a str>,
    pub lang: Option<&'a str>,
    /// A language code hint (the callable form of Python `lang_guess` is not modelled here).
    pub lang_guess: Option<&'a str>,
}

/// Options for constructing a [`Router`].
#[derive(Debug, Clone)]
pub struct RouterOptions {
    pub device: Option<String>,
    pub token: Option<String>,
    pub max_loaded: usize,
    pub default: String,
    pub auto_task_detection: bool,
    pub standalone_repos: bool,
    /// A language-code hint applied to every request (checked before built-in detection).
    pub lang_guess: Option<String>,
}

impl Default for RouterOptions {
    fn default() -> Self {
        RouterOptions {
            device: None,
            token: std::env::var("HF_TOKEN").ok(),
            max_loaded: 2,
            default: "english".to_string(),
            auto_task_detection: false,
            standalone_repos: false,
            lang_guess: None,
        }
    }
}

/// Lazily loads Laya checkpoints and sends each request to the right one.
pub struct Router {
    models: indexmap::IndexMap<String, ModelSpec>,
    #[allow(dead_code)]
    device: Option<String>,
    #[allow(dead_code)]
    token: Option<String>,
    default: String,
    auto_task_detection: bool,
    lang_guess: Option<String>,
    #[cfg(feature = "model")]
    cache: std::sync::Mutex<ModelCache>,
}

#[cfg(feature = "model")]
struct ModelCache {
    agents: std::collections::HashMap<String, std::sync::Arc<crate::agent::Agent>>,
    order: Vec<String>, // least-recently-used first
    max_loaded: usize,
}

impl Router {
    pub fn new(opts: RouterOptions) -> Result<Router> {
        let base = if opts.standalone_repos {
            standalone_models()
        } else {
            default_models()
        };
        let models: indexmap::IndexMap<String, ModelSpec> =
            base.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let default = normalise_name(&opts.default)?;
        Ok(Router {
            models,
            device: opts.device,
            token: opts.token,
            default,
            auto_task_detection: opts.auto_task_detection,
            lang_guess: opts.lang_guess,
            #[cfg(feature = "model")]
            cache: std::sync::Mutex::new(ModelCache {
                agents: std::collections::HashMap::new(),
                order: Vec::new(),
                max_loaded: opts.max_loaded.max(1),
            }),
        })
    }

    /// Convenience constructor with default options.
    pub fn with_defaults() -> Result<Router> {
        Router::new(RouterOptions::default())
    }

    fn spec(&self, key: &str) -> &ModelSpec {
        self.models.get(key).expect("normalised key must exist")
    }

    fn resolve_hint(&self, hint: Option<&str>) -> Option<bool> {
        english_from_code(hint)
    }

    /// Decide which checkpoint to use, without loading or running anything.
    pub fn route(
        &self,
        state: &State,
        questions: Option<&Questions>,
        hints: &RouteHints,
    ) -> Result<RouteDecision> {
        let empty = Questions::new();
        let questions = questions.unwrap_or(&empty);

        if let Some(model) = hints.model {
            let key = normalise_name(model)?;
            return Ok(RouteDecision {
                repo: repo_str(self.spec(&key)),
                model: key,
                reason: format!("explicit model={}", py_repr(model)),
                detection: None,
                workflow: None,
            });
        }

        if let Some(task) = hints.task {
            let normalized = if task.to_lowercase().replace('-', "_") == "typed_decisions" {
                "typed-decisions"
            } else {
                task
            };
            let key = normalise_name(normalized)?;
            return Ok(RouteDecision {
                repo: repo_str(self.spec(&key)),
                model: key,
                reason: format!("explicit task={}", py_repr(task)),
                detection: None,
                workflow: None,
            });
        }

        let workflow = match_typed_decisions_workflow(questions);
        if let Some(wf) = &workflow {
            if self.auto_task_detection {
                return Ok(RouteDecision {
                    model: "typed-decisions".to_string(),
                    repo: repo_str(self.spec("typed-decisions")),
                    reason: format!(
                        "question ids match the {} typed-decisions workflow",
                        py_repr(wf)
                    ),
                    detection: None,
                    workflow: workflow.clone(),
                });
            }
        }

        if let Some(lang) = hints.lang {
            // An explicit `lang` is decisive only when the code names a language. Blank or
            // whitespace resolves to no usable hint, so it falls through to lang_guess/detection
            // exactly as an abstaining hint does; real English/non-English codes still route now.
            if let Some(resolved) = english_from_code(Some(lang)) {
                let key = if resolved { "english" } else { "multilingual" };
                return Ok(RouteDecision {
                    model: key.to_string(),
                    repo: repo_str(self.spec(key)),
                    reason: format!("explicit lang={}", py_repr(lang)),
                    detection: None,
                    workflow,
                });
            }
        }

        // Per-call hint first, then the one installed on the Router.
        for (source, hint) in [
            ("lang_guess", hints.lang_guess),
            ("Router(lang_guess=...)", self.lang_guess.as_deref()),
        ] {
            if let Some(resolved) = self.resolve_hint(hint) {
                let key = if resolved { "english" } else { "multilingual" };
                return Ok(RouteDecision {
                    model: key.to_string(),
                    repo: repo_str(self.spec(key)),
                    reason: format!(
                        "{}: the caller identified this as {} text",
                        source,
                        if resolved { "English" } else { "non-English" }
                    ),
                    detection: None,
                    workflow,
                });
            }
        }

        let det = analyse(state);
        let (key, reason): (String, String) = if det.script == "unknown" {
            (
                self.default.clone(),
                format!(
                    "no letters detected in state; using default ({})",
                    self.default
                ),
            )
        } else if det.script != "latin" {
            (
                "multilingual".to_string(),
                format!(
                    "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                    det.script,
                    100.0 * det.non_latin_fraction
                ),
            )
        } else if !det.is_english {
            let reason = if let Some(seg) = &det.mixed_segment {
                // A mostly-English state whose own line/field reads as a non-English language:
                // the part a question is about is not English even though the whole outvotes it.
                let truncated: String = seg.chars().take(60).collect();
                format!(
                    "Latin script, mostly English, but a line or field reads as {} ({}); the English checkpoint cannot read it",
                    py_repr(det.language.as_deref().unwrap_or("")),
                    py_repr(&truncated)
                )
            } else if let Some(l) = &det.language {
                format!(
                    "Latin script but language looks like {}, not English",
                    py_repr(l)
                )
            } else {
                format!(
                    "Latin script, language not identified but {:.0}% non-English letters; not safe for the English checkpoint",
                    100.0 * det.diacritic_rate
                )
            };
            ("multilingual".to_string(), reason)
        } else if det.language_undecided {
            (
                self.default.clone(),
                format!(
                    "Latin script, language not identified and no non-English letters; using default ({})",
                    self.default
                ),
            )
        } else {
            ("english".to_string(), "English Latin text".to_string())
        };

        Ok(RouteDecision {
            repo: repo_str(self.spec(&key)),
            model: key,
            reason,
            detection: Some(det),
            workflow,
        })
    }
}

// --------------------------------------------------------------- model-backed loading
#[cfg(feature = "model")]
mod loading {
    use super::*;
    use crate::agent::{Agent, SystemOneResult};
    use std::sync::Arc;

    impl Router {
        /// Return the Agent for `name`, downloading and building it on first use.
        pub fn load(&self, name: &str) -> Result<Arc<Agent>> {
            let key = normalise_name(name)?;
            {
                let mut cache = self.cache.lock().unwrap();
                if let Some(agent) = cache.agents.get(&key).cloned() {
                    touch(&mut cache, &key);
                    return Ok(agent);
                }
            }
            // Build outside the lock (download can be slow); tolerate a concurrent winner.
            let (repo, sub) = self.spec(&key).clone();
            let agent = Arc::new(Agent::load(
                &repo,
                crate::agent::LoadOptions {
                    device: self.device.clone(),
                    token: self.token.clone(),
                    subfolder: sub,
                },
            )?);
            let mut cache = self.cache.lock().unwrap();
            let agent = cache.agents.entry(key.clone()).or_insert(agent).clone();
            if !cache.order.contains(&key) {
                cache.order.push(key.clone());
            }
            evict(&mut cache);
            Ok(agent)
        }

        /// Pre-build the given checkpoints (or all configured ones) so no request pays a load.
        /// Raises `max_loaded` to fit both the requested and already-resident checkpoints.
        pub fn preload(&self, names: Option<&[&str]>) -> Result<()> {
            let names: Vec<String> = match names {
                Some(ns) => ns
                    .iter()
                    .map(|n| normalise_name(n))
                    .collect::<Result<_>>()?,
                None => self.models.keys().cloned().collect(),
            };
            {
                let mut cache = self.cache.lock().unwrap();
                let want: std::collections::HashSet<&String> =
                    names.iter().chain(cache.order.iter()).collect();
                cache.max_loaded = cache.max_loaded.max(want.len());
            }
            for n in &names {
                self.load(n)?;
            }
            Ok(())
        }

        /// Free one checkpoint, or all of them.
        pub fn unload(&self, name: Option<&str>) -> Result<()> {
            let mut cache = self.cache.lock().unwrap();
            match name {
                None => {
                    cache.agents.clear();
                    cache.order.clear();
                }
                Some(n) => {
                    let key = normalise_name(n)?;
                    cache.agents.remove(&key);
                    cache.order.retain(|k| k != &key);
                }
            }
            Ok(())
        }

        /// The currently loaded checkpoints (least-recently-used first).
        pub fn loaded(&self) -> Vec<String> {
            self.cache.lock().unwrap().order.clone()
        }

        /// Route, then answer every question in one forward pass on the chosen checkpoint.
        pub fn predict(
            &self,
            state: &State,
            questions: &Questions,
            hints: &RouteHints,
        ) -> Result<SystemOneResult> {
            let decision = self.route(state, Some(questions), hints)?;
            let agent = self.load(&decision.model)?;
            let mut result = agent.system_one(state, questions)?;
            result.routing = Some(decision.to_json());
            Ok(result)
        }
    }

    fn touch(cache: &mut ModelCache, key: &str) {
        cache.order.retain(|k| k != key);
        cache.order.push(key.to_string());
    }

    fn evict(cache: &mut ModelCache) {
        while cache.order.len() > cache.max_loaded {
            let victim = cache.order.remove(0);
            cache.agents.remove(&victim);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn de_state() -> Value {
        json!({ "text": "Mein Konto wurde zweimal belastet" })
    }

    #[test]
    fn explicit_lang_code_routes() {
        let r = Router::with_defaults().unwrap();
        let en = r
            .route(
                &de_state(),
                None,
                &RouteHints {
                    lang: Some("en"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(en.model, "english");
        assert!(en.reason.contains("explicit lang"));

        let de = r
            .route(
                &de_state(),
                None,
                &RouteHints {
                    lang: Some("de"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(de.model, "multilingual");
    }

    #[test]
    fn blank_lang_falls_through_to_detection() {
        // A blank/whitespace explicit lang names no language, so it must not decide routing;
        // detection sees German and routes multilingual (upstream #292).
        let r = Router::with_defaults().unwrap();
        for blank in ["", "   ", "\t"] {
            let d = r
                .route(
                    &de_state(),
                    None,
                    &RouteHints {
                        lang: Some(blank),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(
                d.model, "multilingual",
                "blank lang {blank:?} should fall through"
            );
            assert!(
                !d.reason.contains("explicit lang"),
                "blank lang {blank:?} must not route as explicit"
            );
        }
    }

    #[test]
    fn language_agnostic_codes_abstain() {
        // `$LANG` values that name no language must not pin routing to a checkpoint; they abstain
        // like a blank code so detection decides (upstream #368).
        for code in ["C", "posix", "und", "zxx", "mul", "C.UTF-8", "c_US"] {
            assert_eq!(
                english_from_code(Some(code)),
                None,
                "{code:?} should name no language"
            );
        }
        // Real codes still resolve.
        assert_eq!(english_from_code(Some("en")), Some(true));
        assert_eq!(english_from_code(Some("de")), Some(false));
    }

    #[test]
    fn lang_c_falls_through_to_detection() {
        // An explicit `lang=C` names no language, so German text still routes multilingual.
        let r = Router::with_defaults().unwrap();
        let d = r
            .route(
                &de_state(),
                None,
                &RouteHints {
                    lang: Some("C"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(d.model, "multilingual");
        assert!(!d.reason.contains("explicit lang"));
    }

    #[test]
    fn mixed_segment_routes_multilingual_with_reason() {
        // A mostly-English state whose own field reads as Portuguese routes multilingual with the
        // "mostly English, but a line or field reads as" reason (upstream #207).
        let r = Router::with_defaults().unwrap();
        let state = json!({
            "customer": "Eu preciso de ajuda com a minha conta pois fui cobrado duas vezes",
            "log": "The server returned an internal error and the request was retried three times before it finally failed on the second attempt with a timeout on the database connection pool that was exhausted"
        });
        let d = r
            .route(&state, Some(&Questions::new()), &RouteHints::default())
            .unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(
            d.reason,
            "Latin script, mostly English, but a line or field reads as 'pt' \
             ('Eu preciso de ajuda com a minha conta pois fui cobrado duas '); \
             the English checkpoint cannot read it"
        );
    }
}
