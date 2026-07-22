//! Bloom UI — a decoupled Dioxus frontend for the OpenAI-compatible
//! Bloom inference server. Talks to `/v1/chat/completions` over SSE.

mod api;

use api::{ChatMessage, ConnConfig};
use dioxus::prelude::*;

const STYLE: Asset = asset!("/assets/style.css");
const STORAGE_KEY: &str = "bloom-ui-conn";
// Pre-rename key; migrated to STORAGE_KEY on first load, then cleared.
const LEGACY_STORAGE_KEY: &str = "bloom-web-conn";

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, PartialEq)]
struct DisplayMsg {
    role: String,
    content: String,
}

#[component]
fn App() -> Element {
    // ── State ──
    let mut config = use_signal(load_config);
    let mut messages = use_signal(Vec::<DisplayMsg>::new);
    let mut input = use_signal(String::new);
    let mut busy = use_signal(|| false);
    let mut error = use_signal(|| Option::<String>::None);
    let mut show_settings = use_signal(|| false);
    let mut model_id = use_signal(|| "…".to_string());
    let mut health_ok = use_signal(|| Option::<bool>::None);
    let mut max_tokens = use_signal(|| 512usize);
    let mut temperature = use_signal(|| 0.7f64);

    // ── Probe health + model on first load ──
    use_effect(move || {
        let cfg = config();
        spawn(async move {
            match api::fetch_health(&cfg).await {
                Ok(h) => {
                    model_id.set(h.model.clone());
                    health_ok.set(Some(true));
                }
                Err(_) => {
                    health_ok.set(Some(false));
                    // Fall back to /v1/models for the id.
                    if let Ok(id) = api::fetch_first_model(&cfg).await {
                        model_id.set(id);
                    }
                }
            }
        });
    });

    let status_view = match health_ok() {
        Some(true) => ("ok", format!("Connected - {}", model_id())),
        Some(false) => ("err", "Connection failed".to_string()),
        None => ("", "Connecting…".to_string()),
    };

    let mut on_send = move |_| {
        if busy() || input().trim().is_empty() {
            return;
        }
        let user_text = input().trim().to_string();
        input.set(String::new());
        error.set(None);

        // Build request history (map display msgs back to wire format).
        let mut history: Vec<ChatMessage> = messages()
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        history.push(ChatMessage::user(user_text.clone()));

        messages.push(DisplayMsg {
            role: "user".into(),
            content: user_text,
        });
        // Placeholder assistant bubble that we stream into.
        messages.push(DisplayMsg {
            role: "assistant".into(),
            content: String::new(),
        });
        let assistant_idx = messages().len() - 1;
        busy.set(true);

        let cfg = config();
        let mt = max_tokens();
        let temp = temperature();
        spawn(async move {
            let result = api::chat_stream(&cfg, &history, Some(mt), Some(temp), |delta| {
                messages.with_mut(|msgs| {
                    if let Some(m) = msgs.get_mut(assistant_idx) {
                        m.content.push_str(&delta);
                    }
                });
            })
            .await;
            busy.set(false);
            if let Err(e) = result {
                // Remove the empty assistant placeholder and surface the error.
                messages.with_mut(|msgs| {
                    if msgs
                        .get(assistant_idx)
                        .map(|m| m.content.is_empty())
                        .unwrap_or(false)
                    {
                        msgs.remove(assistant_idx);
                    }
                });
                error.set(Some(e));
            }
        });
    };

    rsx! {
        document::Stylesheet { href: STYLE }
        div { class: "app",
            // ── Header ──
            header { class: "app-header",
                div { class: "brand",
                    div { class: "brand-logo", "B" }
                    div {
                        div { class: "brand-title", "Bloom" }
                        div { class: "brand-sub", "Edge multimodal inference engine" }
                    }
                }
                div { class: "header-actions",
                    span { class: "status-pill {status_view.0}",
                        span { class: "status-dot" }
                        "{status_view.1}"
                    }
                    button {
                        class: "icon-btn",
                        onclick: move |_| show_settings.set(true),
                        "Settings"
                    }
                }
            }

            // ── Error banner ──
            if let Some(e) = error() {
                div { class: "error-banner",
                    span { "Request failed: {e}" }
                    button { onclick: move |_| error.set(None), "×" }
                }
            }

            // ── Messages ──
            MessageList { messages: messages(), busy: busy() }

            // ── Composer ──
            div { class: "composer",
                div { class: "composer-row",
                    textarea {
                        placeholder: "Type a message. Press Enter to send or Shift+Enter for a new line…",
                        value: "{input}",
                        disabled: busy(),
                        rows: 2,
                        oninput: move |e| input.set(e.value()),
                        onkeydown: move |e| {
                            if e.key() == Key::Enter && !e.modifiers().shift() {
                                e.prevent_default();
                                on_send(());
                            }
                        },
                    }
                    button {
                        class: "send-btn",
                        disabled: busy() || input().trim().is_empty(),
                        onclick: move |_| on_send(()),
                        if busy() { "Generating…" } else { "Send" }
                    }
                }
                div { class: "composer-hint", "OpenAI-compatible API - Streaming output" }
            }
        }

        // ── Settings drawer ──
        if show_settings() {
            SettingsDrawer {
                config: config,
                on_close: move |_| show_settings.set(false),
                max_tokens: max_tokens,
                temperature: temperature,
            }
        }
    }
}

#[component]
fn MessageList(messages: Vec<DisplayMsg>, busy: bool) -> Element {
    if messages.is_empty() {
        return rsx! {
            div { class: "messages",
                div { class: "empty-state",
                    div { class: "logo", "✦" }
                    h2 { "Welcome to Bloom" }
                    p { "Run multimodal inference locally or at the edge. Start a conversation." }
                }
            }
        };
    }
    rsx! {
        div { class: "messages",
            for (idx, m) in messages.iter().enumerate() {
                {
                    let is_streaming = busy
                        && idx == messages.len() - 1
                        && m.role == "assistant";
                    let bubble_class = if is_streaming {
                        "bubble cursor-blink"
                    } else {
                        "bubble"
                    };
                    rsx! {
                        div { class: "msg {m.role}", key: "{idx}",
                            div { class: "{bubble_class}",
                                if m.role != "user" {
                                    span { class: "role-tag",
                                        if m.role == "assistant" { "Bloom" } else { "{m.role}" }
                                    }
                                }
                                "{m.content}"
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn SettingsDrawer(
    config: Signal<ConnConfig>,
    on_close: EventHandler<()>,
    max_tokens: Signal<usize>,
    temperature: Signal<f64>,
) -> Element {
    let mut base_url = use_signal(|| config().base_url);
    let mut api_key = use_signal(|| config().api_key);
    let mut test_result = use_signal(|| Option::<(bool, String)>::None);
    let mut testing = use_signal(|| false);

    let mut on_test = move |_| {
        testing.set(true);
        test_result.set(None);
        let cfg = ConnConfig {
            base_url: base_url(),
            api_key: api_key(),
        };
        spawn(async move {
            let res = match api::fetch_health(&cfg).await {
                Ok(h) => (true, format!("Connected - Model {}", h.model)),
                Err(e) => (false, format!("Connection failed: {e}")),
            };
            test_result.set(Some(res));
            testing.set(false);
        });
    };

    let mut on_save = move |_| {
        let cfg = ConnConfig {
            base_url: base_url().trim_end_matches('/').to_string(),
            api_key: api_key(),
        };
        save_config(&cfg);
        config.set(cfg);
        on_close.call(());
    };

    rsx! {
        div { class: "drawer-overlay", onclick: move |_| on_close.call(()),
            div {
                class: "drawer",
                onclick: move |e| e.stop_propagation(),
                h3 { "Connection Settings" }
                div { class: "drawer-sub", "Connect to a running bloom_server instance." }

                div { class: "field",
                    label { "Server Address (Base URL)" }
                    input {
                        value: "{base_url}",
                        placeholder: "http://127.0.0.1:8080",
                        oninput: move |e| base_url.set(e.value()),
                    }
                    div { class: "help", "Defaults to the current origin. Enter the backend URL when deployed separately." }
                }
                div { class: "field",
                    label { "API Key (optional)" }
                    input {
                        r#type: "password",
                        value: "{api_key}",
                        placeholder: "BLOOM_API_KEY",
                        oninput: move |e| api_key.set(e.value()),
                    }
                    div { class: "help", "Required when the server uses --api-key." }
                }
                div { class: "field",
                    label { "Maximum Generated Tokens: {max_tokens}" }
                    input {
                        r#type: "range", min: "64", max: "4096", step: "64",
                        value: "{max_tokens}",
                        oninput: move |e| {
                            if let Ok(v) = e.value().parse::<usize>() { max_tokens.set(v); }
                        },
                    }
                }
                div { class: "field",
                    label { "Temperature：{temperature:.2}" }
                    input {
                        r#type: "range", min: "0", max: "2", step: "0.05",
                        value: "{temperature}",
                        oninput: move |e| {
                            if let Ok(v) = e.value().parse::<f64>() { temperature.set(v); }
                        },
                    }
                }

                button {
                    class: "btn-ghost",
                    disabled: testing(),
                    onclick: move |_| on_test(()),
                    if testing() { "Testing…" } else { "Test Connection" }
                }
                if let Some((ok, msg)) = test_result() {
                    div { class: if ok { "test-result ok" } else { "test-result err" }, "{msg}" }
                }

                div { class: "spacer" }
                div { class: "btn-row",
                    button { class: "btn-ghost", onclick: move |_| on_close.call(()), "Cancel" }
                    button { class: "btn-primary", onclick: move |_| on_save(()), "Save" }
                }
            }
        }
    }
}

// ── Persistence (localStorage) ──

fn load_config() -> ConnConfig {
    let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) else {
        return ConnConfig::default();
    };
    // Migrate the pre-rename key once so existing users keep their settings.
    if storage.get_item(STORAGE_KEY).ok().flatten().is_none() {
        if let Ok(Some(legacy)) = storage.get_item(LEGACY_STORAGE_KEY) {
            let _ = storage.set_item(STORAGE_KEY, &legacy);
            let _ = storage.remove_item(LEGACY_STORAGE_KEY);
        }
    }
    let Ok(Some(text)) = storage.get_item(STORAGE_KEY) else {
        return ConnConfig::default();
    };
    serde_json::from_str::<StoredConn>(&text)
        .map(|s| ConnConfig {
            base_url: s.base_url,
            api_key: s.api_key,
        })
        .unwrap_or_default()
}

fn save_config(cfg: &ConnConfig) {
    let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) else {
        return;
    };
    if let Ok(text) = serde_json::to_string(&StoredConn {
        base_url: cfg.base_url.clone(),
        api_key: cfg.api_key.clone(),
    }) {
        let _ = storage.set_item(STORAGE_KEY, &text);
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredConn {
    base_url: String,
    api_key: String,
}
