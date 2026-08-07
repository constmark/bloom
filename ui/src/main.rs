//! Bloom UI, a decoupled Dioxus frontend for the Bloom inference server.

mod api;
mod browser;
mod chat;
mod markdown;
mod storage;

use api::{
    ChatCancellation, ChatMessage, ChatOptions, ChatStreamError, ChatUsage, ConnConfig,
    EmbeddingBatch, ImageAttachment, ModelImportCancellation, ModelImportClientError, Readiness,
    RerankBatch, ResponseFormatMode, StreamUpdate, MAX_API_KEY_CHARS, MAX_BASE_URL_CHARS,
    MAX_CHAT_INPUT_CHARS, MAX_EMBEDDING_CONTENT_BYTES, MAX_EMBEDDING_DIMENSIONS,
    MAX_IMAGE_ATTACHMENT_BYTES, MAX_RERANK_CONTENT_BYTES, MAX_RERANK_QUERY_CHARS,
    MAX_RESPONSE_JSON_SCHEMA_BYTES, MAX_SYSTEM_PROMPT_CHARS,
};
use chat::{
    conversation_matches_query, merge_conversation_stores, ConversationStore, DisplayMessage,
    EmptyGenerationRollback, GenerationOutcome, GenerationStats, CONVERSATION_ARCHIVE_FILENAME,
    MAX_CONVERSATION_ARCHIVE_BYTES, MAX_RENAMED_TITLE_CHARS,
};
use dioxus::prelude::*;
use gloo_timers::future::TimeoutFuture;
use markdown::render_assistant_markdown;

const STYLE: Asset = asset!("/assets/style.css");
const STATUS_POLL_INTERVAL_MS: u32 = 5_000;
const DIAGNOSTICS_POLL_INTERVAL_MS: u32 = 5_000;
const DIAGNOSTICS_FILENAME: &str = "bloom-diagnostics.json";
const MODEL_IMPORT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONVERSATION_SEARCH_CHARS: usize = 200;
const INITIAL_VISIBLE_MESSAGES: usize = 100;
const MESSAGE_HISTORY_PAGE_SIZE: usize = 100;
const LIVE_SPEECH_WINDOW_MS: u32 = 2_000;
const MIN_SPEECH_WINDOW_SAMPLES: usize = api::SPEECH_SAMPLE_RATE as usize / 4;
const MAX_TRANSCRIPT_BYTES: usize = 1024 * 1024;
const MODEL_DRAWER_DIALOG_ID: &str = "model-manager-dialog";
const DIAGNOSTICS_DRAWER_DIALOG_ID: &str = "runtime-diagnostics-dialog";
const SETTINGS_DRAWER_DIALOG_ID: &str = "settings-dialog";
const RENAME_CONVERSATION_DIALOG_ID: &str = "rename-conversation-dialog";
const EDIT_PROMPT_DIALOG_ID: &str = "edit-prompt-dialog";
const IMPORT_CONVERSATIONS_DIALOG_ID: &str = "import-conversations-dialog";
const RESPONSE_SCHEMA_PLACEHOLDER: &str = r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModalKeyAction {
    Close,
    TrapForward,
    TrapBackward,
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MessageWindow {
    start: usize,
    visible: usize,
    next_visible: usize,
}

fn message_window(total: usize, requested_visible: usize) -> MessageWindow {
    if total == 0 {
        return MessageWindow {
            start: 0,
            visible: 0,
            next_visible: 0,
        };
    }
    let visible = total.min(requested_visible.max(1));
    let start = total - visible;
    let next_visible = visible.saturating_add(start.min(MESSAGE_HISTORY_PAGE_SIZE));
    MessageWindow {
        start,
        visible,
        next_visible,
    }
}

fn modal_key_action(key: &Key, shift: bool) -> ModalKeyAction {
    match key {
        Key::Escape => ModalKeyAction::Close,
        Key::Tab if shift => ModalKeyAction::TrapBackward,
        Key::Tab => ModalKeyAction::TrapForward,
        _ => ModalKeyAction::Ignore,
    }
}

fn handle_modal_key(event: KeyboardEvent, modal_id: &'static str, on_close: EventHandler<()>) {
    let action = modal_key_action(&event.key(), event.modifiers().shift());
    match action {
        ModalKeyAction::Close => {
            event.prevent_default();
            event.stop_propagation();
            on_close.call(());
        }
        ModalKeyAction::TrapForward | ModalKeyAction::TrapBackward => {
            let backwards = action == ModalKeyAction::TrapBackward;
            if browser::trap_modal_tab(modal_id, backwards) {
                event.prevent_default();
                event.stop_propagation();
            }
        }
        ModalKeyAction::Ignore => {}
    }
}

fn use_modal_focus(modal_id: &'static str) {
    let previous_focus = use_hook(browser::capture_active_element);
    use_effect(move || {
        browser::focus_first_in_modal(modal_id);
    });
    use_drop(move || browser::restore_focus(previous_focus));
}

#[derive(Clone, PartialEq)]
struct SelectedModelImport {
    file: web_sys::File,
    filename: String,
    size_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct PendingConversationImport {
    store: ConversationStore,
    conversation_count: usize,
    message_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConversationImportMode {
    Merge,
    Replace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConversationModelTransition {
    conversation_id: u64,
    previous_model: String,
    current_model: String,
}

fn conversation_model_transition(
    conversations: &ConversationStore,
    current_model: Option<&str>,
) -> Option<ConversationModelTransition> {
    let current_model = current_model?;
    let previous_model = conversations.active_execution_model()?;
    (previous_model != current_model).then(|| ConversationModelTransition {
        conversation_id: conversations.active_id,
        previous_model: previous_model.to_string(),
        current_model: current_model.to_string(),
    })
}

fn unconfirmed_conversation_model_transition(
    conversations: &ConversationStore,
    current_model: Option<&str>,
    acknowledged: Option<&ConversationModelTransition>,
) -> Option<ConversationModelTransition> {
    conversation_model_transition(conversations, current_model)
        .filter(|transition| Some(transition) != acknowledged)
}

fn model_transition_unavailable_reason(transition: &ConversationModelTransition) -> String {
    format!(
        "Confirm sending this conversation's existing {} history to {} first.",
        transition.previous_model, transition.current_model
    )
}

fn conversation_import_candidate(
    local: &ConversationStore,
    imported: &ConversationStore,
    mode: ConversationImportMode,
    recovery_writes_blocked: bool,
) -> Result<ConversationStore, String> {
    match mode {
        ConversationImportMode::Merge if recovery_writes_blocked => {
            Err("Merge is unavailable while saved-conversation recovery is required.".to_string())
        }
        ConversationImportMode::Merge => merge_conversation_stores(local, imported),
        ConversationImportMode::Replace => {
            imported.validate_persisted()?;
            Ok(imported.clone())
        }
    }
}

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, Debug, PartialEq)]
enum ConnectionState {
    Connecting,
    Ready {
        model: String,
        supports_vision: bool,
        supports_text_input: bool,
        supports_audio_input: bool,
        supports_generation: bool,
        supports_embeddings: bool,
        supports_rerank: bool,
        context_window: Option<u64>,
    },
    Loading {
        model: String,
        progress: u8,
    },
    NoModel,
    LoadFailed {
        message: String,
    },
    AuthenticationRequired {
        message: String,
    },
    Incompatible {
        message: String,
    },
    Offline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyStateAction {
    Models,
    Settings,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EmptyStateView {
    title: String,
    body: String,
    action: Option<(EmptyStateAction, &'static str)>,
}

impl ConnectionState {
    fn from_readiness(readiness: Readiness) -> Self {
        if readiness.status == "ready" {
            Self::Ready {
                model: readiness.model,
                supports_vision: readiness.input_modalities.iter().any(|modality| {
                    modality.eq_ignore_ascii_case("vision")
                        || modality.eq_ignore_ascii_case("multi")
                }),
                supports_text_input: readiness.input_modalities.iter().any(|modality| {
                    modality.eq_ignore_ascii_case("text") || modality.eq_ignore_ascii_case("multi")
                }),
                supports_audio_input: readiness.input_modalities.iter().any(|modality| {
                    modality.eq_ignore_ascii_case("audio") || modality.eq_ignore_ascii_case("multi")
                }),
                supports_generation: readiness
                    .model_tasks
                    .iter()
                    .any(|task| task == "generation"),
                supports_embeddings: readiness.model_tasks.iter().any(|task| task == "embedding"),
                supports_rerank: readiness.model_tasks.iter().any(|task| task == "rerank"),
                context_window: readiness.context_window.filter(|window| *window > 0),
            }
        } else if let Some(message) = readiness.load_error {
            Self::LoadFailed { message }
        } else if !readiness.loading && readiness.progress == 0 {
            Self::NoModel
        } else {
            Self::Loading {
                model: readiness.model,
                progress: readiness.progress,
            }
        }
    }

    fn view(&self) -> (&'static str, String) {
        match self {
            Self::Connecting => ("", "Connecting…".into()),
            Self::Ready {
                model,
                supports_text_input,
                supports_audio_input,
                supports_generation,
                supports_embeddings,
                ..
            } => {
                let task = if *supports_audio_input && !*supports_text_input {
                    "Transcription"
                } else if *supports_embeddings && !*supports_generation {
                    "Embeddings"
                } else {
                    "Generation"
                };
                ("ok", format!("Ready · {task} · {model}"))
            }
            Self::Loading { model, progress } => {
                ("pending", format!("Loading {progress}% · {model}"))
            }
            Self::NoModel => ("pending", "Choose a model".into()),
            Self::LoadFailed { message } => ("err", message.clone()),
            Self::AuthenticationRequired { .. } => ("err", "API key required".into()),
            Self::Incompatible { .. } => ("err", "Incompatible Bloom server".into()),
            Self::Offline => ("err", "Connection unavailable".into()),
        }
    }

    fn can_chat(&self) -> bool {
        matches!(
            self,
            Self::Ready {
                supports_generation: true,
                supports_text_input: true,
                ..
            }
        )
    }

    fn can_embed(&self) -> bool {
        matches!(
            self,
            Self::Ready {
                supports_embeddings: true,
                ..
            }
        )
    }

    fn can_rerank(&self) -> bool {
        matches!(
            self,
            Self::Ready {
                supports_rerank: true,
                ..
            }
        )
    }

    fn can_transcribe(&self) -> bool {
        matches!(
            self,
            Self::Ready {
                supports_audio_input: true,
                supports_generation: true,
                ..
            }
        )
    }

    fn active_model(&self) -> Option<&str> {
        match self {
            Self::Ready { model, .. } => Some(model),
            _ => None,
        }
    }

    fn supports_vision(&self) -> bool {
        matches!(
            self,
            Self::Ready {
                supports_vision: true,
                ..
            }
        )
    }

    fn context_window(&self) -> Option<u64> {
        match self {
            Self::Ready { context_window, .. } => *context_window,
            _ => None,
        }
    }
}

fn empty_state_view(connection: &ConnectionState) -> EmptyStateView {
    match connection {
        ConnectionState::Connecting => EmptyStateView {
            title: "Connecting to Bloom".to_string(),
            body: "Waiting for the configured bloom_server instance to respond.".to_string(),
            action: Some((EmptyStateAction::Settings, "Connection settings")),
        },
        ConnectionState::Offline => EmptyStateView {
            title: "Bloom server is unavailable".to_string(),
            body: "Start bloom_server, then verify the server address and API key.".to_string(),
            action: Some((EmptyStateAction::Settings, "Check connection")),
        },
        ConnectionState::AuthenticationRequired { message } => EmptyStateView {
            title: "API key required".to_string(),
            body: format!(
                "Bloom rejected the configured API key. Update it in Connection settings. {message}"
            ),
            action: Some((EmptyStateAction::Settings, "Update API key")),
        },
        ConnectionState::Incompatible { message } => EmptyStateView {
            title: "Incompatible Bloom server".to_string(),
            body: format!(
                "The configured endpoint does not provide the readiness contract required by this UI. {message}"
            ),
            action: Some((EmptyStateAction::Settings, "Check server version")),
        },
        ConnectionState::NoModel => EmptyStateView {
            title: "Choose a model to begin".to_string(),
            body: "Open the catalog to import, download, inspect, or load a local model."
                .to_string(),
            action: Some((EmptyStateAction::Models, "Open models")),
        },
        ConnectionState::Loading { model, progress } => EmptyStateView {
            title: format!("Loading {model}"),
            body: format!(
                "Model preparation is {progress}% complete. Chat becomes available after the runtime is ready."
            ),
            action: Some((EmptyStateAction::Models, "View model status")),
        },
        ConnectionState::LoadFailed { message } => EmptyStateView {
            title: "Model load failed".to_string(),
            body: message.clone(),
            action: Some((EmptyStateAction::Models, "Review models")),
        },
        ConnectionState::Ready {
            model,
            supports_generation,
            ..
        } => {
            if *supports_generation {
                EmptyStateView {
                    title: "Start a local conversation".to_string(),
                    body: format!(
                        "{model} is ready. Messages stay in this browser and are sent only to your configured Bloom server."
                    ),
                    action: None,
                }
            } else {
                EmptyStateView {
                    title: "Embedding model ready".to_string(),
                    body: format!(
                        "{model} serves embeddings and reranking rather than text generation."
                    ),
                    action: None,
                }
            }
        }
    }
}

struct GenerationRequest {
    config: ConnConfig,
    model: String,
    history: Vec<ChatMessage>,
    options: ChatOptions,
    attachment: Option<ImageAttachment>,
    multimodal_prompt: Option<String>,
    cancellation: ChatCancellation,
    conversation_id: u64,
    assistant_index: usize,
    rollback: EmptyGenerationRollback,
}

#[derive(Clone, Copy)]
struct GenerationSignals {
    conversations: Signal<ConversationStore>,
    busy: Signal<bool>,
    error: Signal<Option<String>>,
    active_cancellation: Signal<Option<ChatCancellation>>,
    active_request_id: Signal<Option<String>>,
}

fn launch_generation(request: GenerationRequest, signals: GenerationSignals) {
    let GenerationRequest {
        config,
        model,
        history,
        options,
        attachment,
        multimodal_prompt,
        cancellation,
        conversation_id,
        assistant_index,
        rollback,
    } = request;
    let mut rollback = Some(rollback);
    let GenerationSignals {
        mut conversations,
        mut busy,
        mut error,
        mut active_cancellation,
        mut active_request_id,
    } = signals;
    let metadata_result = conversations.with_mut(|store| {
        store.set_assistant_response_format(
            conversation_id,
            assistant_index,
            options.response_format.form_value(),
        )
    });
    if let Err(message) = metadata_result {
        settle_empty_generation(
            &mut conversations,
            conversation_id,
            assistant_index,
            rollback
                .take()
                .unwrap_or(EmptyGenerationRollback::RemovePlaceholder),
        );
        error.set(Some(message));
        if let Err(message) = storage::save_conversations(&conversations()) {
            error.set(Some(message));
        }
        return;
    }
    let generation_started_ms = browser::monotonic_now_ms();

    busy.set(true);
    active_request_id.set(None);
    active_cancellation.set(Some(cancellation.clone()));
    spawn(async move {
        let mut first_token_ms = None;
        let mut usage = None::<ChatUsage>;
        let mut response_model = None::<String>;
        let result = {
            let on_update = |update| match update {
                StreamUpdate::RequestId(request_id) => {
                    active_request_id.set(Some(request_id));
                }
                StreamUpdate::Model(model) => response_model = Some(model),
                StreamUpdate::TextDelta(delta) => {
                    if first_token_ms.is_none() && !delta.is_empty() {
                        first_token_ms = Some(browser::monotonic_now_ms() - generation_started_ms);
                    }
                    conversations.with_mut(|store| {
                        if let Some(conversation) = store
                            .conversations
                            .iter_mut()
                            .find(|conversation| conversation.id == conversation_id)
                        {
                            if let Some(message) = conversation.messages.get_mut(assistant_index) {
                                message.content.push_str(&delta);
                            }
                        }
                    });
                }
                StreamUpdate::Usage(stream_usage) => usage = Some(stream_usage),
            };
            if let Some(image) = attachment.as_ref() {
                match multimodal_prompt.as_deref() {
                    Some(multimodal_prompt) => {
                        api::multimodal_stream(
                            &config,
                            &model,
                            multimodal_prompt,
                            image,
                            options,
                            &cancellation,
                            on_update,
                        )
                        .await
                    }
                    None => Err(ChatStreamError::Request(
                        "Prepared multimodal prompt is unavailable.".to_string(),
                    )),
                }
            } else {
                api::chat_stream(&config, &model, &history, options, &cancellation, on_update).await
            }
        };

        let elapsed_ms = browser::monotonic_now_ms() - generation_started_ms;
        let outcome = match &result {
            Ok(_) => GenerationOutcome::Completed,
            Err(ChatStreamError::Cancelled) => GenerationOutcome::Stopped,
            Err(ChatStreamError::Request(_)) => GenerationOutcome::Failed,
        };
        let (prompt_tokens, completion_tokens) = usage
            .map(|usage| (Some(usage.prompt_tokens), Some(usage.completion_tokens)))
            .unwrap_or((None, None));
        let stats = GenerationStats::observed(
            outcome,
            elapsed_ms,
            first_token_ms,
            prompt_tokens,
            completion_tokens,
            response_model,
        );
        conversations.with_mut(|store| {
            let _ = store.set_assistant_generation_stats(conversation_id, assistant_index, stats);
        });

        active_cancellation.set(None);
        active_request_id.set(None);
        busy.set(false);
        match result {
            Ok(_) => {
                if settle_empty_generation(
                    &mut conversations,
                    conversation_id,
                    assistant_index,
                    rollback
                        .take()
                        .unwrap_or(EmptyGenerationRollback::RemovePlaceholder),
                ) {
                    error.set(Some(
                        "Generation completed without returning any text.".to_string(),
                    ));
                }
            }
            Err(ChatStreamError::Cancelled) => {
                settle_empty_generation(
                    &mut conversations,
                    conversation_id,
                    assistant_index,
                    rollback
                        .take()
                        .unwrap_or(EmptyGenerationRollback::RemovePlaceholder),
                );
            }
            Err(ChatStreamError::Request(message)) => {
                let rolled_back = settle_empty_generation(
                    &mut conversations,
                    conversation_id,
                    assistant_index,
                    rollback
                        .take()
                        .unwrap_or(EmptyGenerationRollback::RemovePlaceholder),
                );
                error.set(Some(if rolled_back {
                    message
                } else {
                    format!("{message} The partial response was kept.")
                }));
            }
        }
        if let Err(message) = storage::save_conversations(&conversations()) {
            error.set(Some(message));
        }
    });
}

fn settle_empty_generation(
    conversations: &mut Signal<ConversationStore>,
    conversation_id: u64,
    assistant_index: usize,
    rollback: EmptyGenerationRollback,
) -> bool {
    let mut rolled_back = false;
    conversations.with_mut(|store| {
        rolled_back = store.settle_empty_assistant(conversation_id, assistant_index, rollback);
    });
    rolled_back
}

fn build_chat_history(options: &ChatOptions, messages: &[DisplayMessage]) -> Vec<ChatMessage> {
    let mut history = Vec::with_capacity(messages.len().saturating_add(1));
    if !options.system_prompt.trim().is_empty() {
        history.push(ChatMessage {
            role: "system".into(),
            content: options.system_prompt.trim().to_string(),
        });
    }
    history.extend(messages.iter().map(|message| ChatMessage {
        role: message.role.clone(),
        content: message.content.clone(),
    }));
    history
}

#[component]
fn App() -> Element {
    let config = use_signal(storage::load_connection);
    let (mut conversations, mut conversation_storage_notice_state) = use_hook(|| {
        let load = storage::load_conversations();
        (Signal::new(load.store), Signal::new(load.notice))
    });
    let mut input = use_signal(String::new);
    let busy = use_signal(|| false);
    let mut error = use_signal(|| Option::<String>::None);
    let mut show_settings = use_signal(|| false);
    let mut show_models = use_signal(|| false);
    let mut show_diagnostics = use_signal(|| false);
    let mut show_sidebar = use_signal(|| false);
    let mut pending_delete = use_signal(|| Option::<u64>::None);
    let mut connection = use_signal(|| ConnectionState::Connecting);
    let generation = use_signal(storage::load_generation);
    let active_cancellation = use_signal(|| Option::<ChatCancellation>::None);
    let active_request_id = use_signal(|| Option::<String>::None);
    let mut attachment = use_signal(|| Option::<ImageAttachment>::None);
    let mut attachment_loading = use_signal(|| false);
    let mut attachment_input_key = use_signal(|| 0_u64);
    let mut conversation_import_loading = use_signal(|| false);
    let mut conversation_import_input_key = use_signal(|| 0_u64);
    let mut pending_conversation_import = use_signal(|| Option::<PendingConversationImport>::None);
    let mut conversation_import_error = use_signal(|| Option::<String>::None);
    let mut conversation_archive_notice = use_signal(|| Option::<String>::None);
    let mut conversation_branch_notice = use_signal(|| Option::<String>::None);
    let mut pending_prompt_edit = use_signal(|| Option::<String>::None);
    let mut prompt_edit_error = use_signal(|| Option::<String>::None);
    let mut conversation_search = use_signal(String::new);
    let mut pending_conversation_rename = use_signal(|| Option::<(u64, String)>::None);
    let mut conversation_rename_error = use_signal(|| Option::<String>::None);
    let mut pending_recovery_discard = use_signal(|| false);
    let mut acknowledged_model_transition =
        use_signal(|| Option::<ConversationModelTransition>::None);

    let _status_poller = use_future(move || async move {
        loop {
            let cfg = config();
            match api::fetch_connection_readiness(&cfg).await {
                Ok(readiness) => connection.set(ConnectionState::from_readiness(readiness)),
                Err(api::ReadinessError::Authentication(message)) => {
                    connection.set(ConnectionState::AuthenticationRequired { message })
                }
                Err(api::ReadinessError::Incompatible(message)) => {
                    connection.set(ConnectionState::Incompatible { message })
                }
                Err(api::ReadinessError::Unavailable(_)) => {
                    connection.set(ConnectionState::Offline)
                }
            }
            TimeoutFuture::new(STATUS_POLL_INTERVAL_MS).await;
        }
    });

    let status_view = connection().view();
    let conversation_snapshot = conversations();
    let conversation_storage_notice = conversation_storage_notice_state();
    let conversation_writes_blocked = conversation_storage_notice
        .as_ref()
        .is_some_and(storage::ConversationStorageNotice::blocks_conversation_writes);
    let active_messages = conversation_snapshot.active().messages.clone();
    let context_window = connection().context_window();
    let context_reservation_error =
        validate_context_reservation(generation().max_tokens, context_window).err();
    let context_budget_status =
        conversation_context_status(&active_messages, context_window, generation().max_tokens);
    let structured_output_attachment_error = (attachment().is_some()
        && generation().response_format != ResponseFormatMode::Text)
        .then(|| "Structured output is currently available for text chat only.".to_string());
    let total_conversation_count = conversation_snapshot.conversations.len();
    let conversation_search_value = conversation_search();
    let visible_conversations = conversation_snapshot
        .conversations
        .iter()
        .filter(|conversation| conversation_matches_query(conversation, &conversation_search_value))
        .map(|conversation| (conversation.id, conversation.title.clone()))
        .collect::<Vec<_>>();
    let acknowledged_transition = acknowledged_model_transition();
    let pending_model_transition = unconfirmed_conversation_model_transition(
        &conversation_snapshot,
        connection().active_model(),
        acknowledged_transition.as_ref(),
    );
    let can_send = connection().can_chat()
        && !busy()
        && !conversation_import_loading()
        && !conversation_writes_blocked
        && pending_conversation_import().is_none()
        && pending_prompt_edit().is_none()
        && pending_conversation_rename().is_none()
        && !attachment_loading()
        && context_reservation_error.is_none()
        && structured_output_attachment_error.is_none()
        && pending_model_transition.is_none()
        && (!input().trim().is_empty() || attachment().is_some());
    let retry_unavailable_reason = conversation_snapshot
        .retry_availability()
        .err()
        .map(str::to_string)
        .or_else(|| {
            conversation_writes_blocked.then(|| {
                "Resolve the saved-conversation recovery warning before regenerating.".to_string()
            })
        })
        .or_else(|| {
            (!connection().can_chat())
                .then(|| "Wait for a model to become ready before regenerating.".to_string())
        })
        .or_else(|| {
            attachment_loading()
                .then(|| "Wait for the selected attachment to finish loading.".to_string())
        })
        .or_else(|| {
            pending_model_transition
                .as_ref()
                .map(model_transition_unavailable_reason)
        })
        .or_else(|| context_reservation_error.clone());
    let edit_unavailable_reason = conversation_snapshot
        .edit_availability()
        .err()
        .map(str::to_string)
        .or_else(|| {
            conversation_writes_blocked.then(|| {
                "Resolve the saved-conversation recovery warning before editing.".to_string()
            })
        })
        .or_else(|| {
            (!connection().can_chat()).then(|| {
                "Wait for a model to become ready before editing and resending.".to_string()
            })
        })
        .or_else(|| {
            attachment_loading()
                .then(|| "Wait for the selected attachment to finish loading.".to_string())
        })
        .or_else(|| {
            pending_model_transition
                .as_ref()
                .map(model_transition_unavailable_reason)
        })
        .or_else(|| context_reservation_error.clone());

    let branch_unavailable_reason = conversation_writes_blocked
        .then(|| {
            "Resolve the saved-conversation recovery warning before creating a branch.".to_string()
        })
        .or_else(|| {
            conversation_import_loading().then(|| {
                "Wait for the conversation import to finish before creating a branch.".to_string()
            })
        });

    let mut on_new_chat = move |_| {
        if busy() || conversation_import_loading() || conversation_writes_blocked {
            return;
        }
        let mut updated = conversations();
        let result = updated
            .create()
            .and_then(|_| storage::save_conversations(&updated));
        match result {
            Ok(()) => {
                conversations.set(updated);
                input.set(String::new());
                attachment.set(None);
                attachment_input_key += 1;
                conversation_search.set(String::new());
                conversation_branch_notice.set(None);
                acknowledged_model_transition.set(None);
                error.set(None);
                show_sidebar.set(false);
            }
            Err(message) => error.set(Some(message)),
        }
    };

    let export_conversations = move |_| {
        if busy() || conversation_import_loading() || conversation_writes_blocked {
            return;
        }
        error.set(None);
        conversation_archive_notice.set(None);
        match chat::export_conversation_archive(&conversations()).and_then(|json| {
            browser::download_text_file(CONVERSATION_ARCHIVE_FILENAME, "application/json", &json)
        }) {
            Ok(()) => conversation_archive_notice.set(Some(format!(
                "Exported {} conversation(s).",
                conversations().conversations.len()
            ))),
            Err(message) => error.set(Some(message)),
        }
    };

    let branch_conversation = move |message_index: usize| {
        if busy() || conversation_import_loading() || conversation_writes_blocked {
            return;
        }
        let mut updated = conversations();
        let result = updated.branch_active_at(message_index).and_then(|branch| {
            storage::save_conversations(&updated)?;
            Ok(branch)
        });
        match result {
            Ok(branch) => {
                conversations.set(updated);
                input.set(String::new());
                attachment.set(None);
                attachment_input_key += 1;
                conversation_search.set(String::new());
                show_sidebar.set(false);
                acknowledged_model_transition.set(None);
                error.set(None);
                let attachment_note = if branch.contains_unavailable_attachment {
                    " Stored image bytes are not copied; reattach an image before replaying an image turn."
                } else {
                    ""
                };
                conversation_branch_notice.set(Some(format!(
                    "Created \"{}\" with {} message(s).{}",
                    branch.title, branch.message_count, attachment_note
                )));
            }
            Err(message) => error.set(Some(message)),
        }
    };

    let continue_conversation = move |message_index: usize| {
        if busy() || conversation_import_loading() || conversation_writes_blocked {
            return;
        }
        let mut updated = conversations();
        let result = updated
            .continue_active_from(message_index)
            .and_then(|continuation| {
                storage::save_conversations(&updated)?;
                Ok(continuation)
            });
        match result {
            Ok(continuation) => {
                conversations.set(updated);
                input.set(String::new());
                attachment.set(None);
                attachment_input_key += 1;
                conversation_search.set(String::new());
                show_sidebar.set(false);
                acknowledged_model_transition.set(None);
                error.set(None);
                let attachment_note = if continuation.contains_unavailable_attachment {
                    " Stored image bytes are not copied; reattach an image before replaying an image turn."
                } else {
                    ""
                };
                conversation_branch_notice.set(Some(format!(
                    "Created \"{}\" with {} recent message(s); {} earlier message(s) remain only in the source.{}",
                    continuation.title,
                    continuation.retained_message_count,
                    continuation.omitted_message_count,
                    attachment_note
                )));
            }
            Err(message) => error.set(Some(message)),
        }
    };

    let mut on_send = move |_| {
        if attachment().is_some() && generation().response_format != ResponseFormatMode::Text {
            error.set(Some(
                "Structured output is currently available for text chat only.".to_string(),
            ));
            return;
        }
        if let Err(message) =
            validate_context_reservation(generation().max_tokens, connection().context_window())
        {
            error.set(Some(message));
            return;
        }
        if !connection().can_chat()
            || busy()
            || conversation_writes_blocked
            || conversation_import_loading()
            || pending_conversation_import().is_some()
            || pending_prompt_edit().is_some()
            || pending_conversation_rename().is_some()
            || attachment_loading()
            || (input().trim().is_empty() && attachment().is_none())
        {
            return;
        }
        let Some(model) = connection().active_model().map(str::to_owned) else {
            return;
        };
        if let Some(transition) = unconfirmed_conversation_model_transition(
            &conversations(),
            Some(&model),
            acknowledged_model_transition().as_ref(),
        ) {
            error.set(Some(model_transition_unavailable_reason(&transition)));
            return;
        }
        let selected_attachment = attachment();
        let user_text = if input().trim().is_empty() {
            "Describe this image.".to_string()
        } else {
            input().trim().to_string()
        };
        let display_text = selected_attachment
            .as_ref()
            .map(|image| format!("[Image: {}]\n{}", image.name, user_text))
            .unwrap_or_else(|| user_text.clone());
        let conversation_id = conversations().active_id;
        let options = generation();
        let mut history = build_chat_history(&options, &conversations().active().messages);
        history.push(ChatMessage {
            role: "user".into(),
            content: user_text.clone(),
        });
        let cfg = config();
        let multimodal_prompt = match api::preflight_chat_submission(
            &cfg,
            &model,
            &history,
            &options,
            selected_attachment.as_ref(),
        ) {
            Ok(multimodal_prompt) => multimodal_prompt,
            Err(message) => {
                error.set(Some(message));
                return;
            }
        };
        let cancellation = match ChatCancellation::new() {
            Ok(cancellation) => cancellation,
            Err(message) => {
                error.set(Some(message));
                return;
            }
        };

        let mut updated = conversations();
        if selected_attachment.is_some() {
            updated.append_user_with_unavailable_attachment(display_text);
        } else {
            updated.append_user(display_text);
        }
        if let Err(message) = storage::save_conversations(&updated) {
            error.set(Some(message));
            return;
        }
        conversations.set(updated);
        input.set(String::new());
        attachment.set(None);
        attachment_input_key += 1;
        conversation_branch_notice.set(None);
        error.set(None);
        let assistant_index = conversations.with_mut(|store| store.append_assistant_placeholder());
        launch_generation(
            GenerationRequest {
                config: cfg,
                model,
                history,
                options,
                attachment: selected_attachment,
                multimodal_prompt,
                cancellation,
                conversation_id,
                assistant_index,
                rollback: EmptyGenerationRollback::RemovePlaceholder,
            },
            GenerationSignals {
                conversations,
                busy,
                error,
                active_cancellation,
                active_request_id,
            },
        );
    };

    let mut on_retry = move |_| {
        if let Err(message) =
            validate_context_reservation(generation().max_tokens, connection().context_window())
        {
            error.set(Some(message));
            return;
        }
        if !connection().can_chat()
            || busy()
            || conversation_writes_blocked
            || conversation_import_loading()
            || pending_conversation_import().is_some()
            || pending_prompt_edit().is_some()
            || pending_conversation_rename().is_some()
            || attachment_loading()
        {
            return;
        }
        let Some(model) = connection().active_model().map(str::to_owned) else {
            return;
        };
        if let Some(transition) = unconfirmed_conversation_model_transition(
            &conversations(),
            Some(&model),
            acknowledged_model_transition().as_ref(),
        ) {
            error.set(Some(model_transition_unavailable_reason(&transition)));
            return;
        }
        let mut updated = conversations();
        let retry = match updated.prepare_last_response_retry() {
            Ok(retry) => retry,
            Err(message) => {
                error.set(Some(message));
                return;
            }
        };
        let options = generation();
        let history = build_chat_history(&options, &retry.history);
        let cfg = config();
        if let Err(message) = api::preflight_chat_submission(&cfg, &model, &history, &options, None)
        {
            error.set(Some(message));
            return;
        }
        let cancellation = match ChatCancellation::new() {
            Ok(cancellation) => cancellation,
            Err(message) => {
                error.set(Some(message));
                return;
            }
        };
        conversations.set(updated);
        conversation_branch_notice.set(None);
        error.set(None);
        launch_generation(
            GenerationRequest {
                config: cfg,
                model,
                history,
                options,
                attachment: None,
                multimodal_prompt: None,
                cancellation,
                conversation_id: retry.conversation_id,
                assistant_index: retry.assistant_index,
                rollback: retry.rollback,
            },
            GenerationSignals {
                conversations,
                busy,
                error,
                active_cancellation,
                active_request_id,
            },
        );
    };

    let mut request_prompt_edit = move |_| match conversations().latest_editable_prompt() {
        Ok(content) => {
            prompt_edit_error.set(None);
            pending_prompt_edit.set(Some(content.to_string()));
            error.set(None);
        }
        Err(message) => error.set(Some(message.to_string())),
    };

    let submit_prompt_edit = move |content: String| {
        if let Err(message) =
            validate_context_reservation(generation().max_tokens, connection().context_window())
        {
            prompt_edit_error.set(Some(message));
            return;
        }
        if !connection().can_chat()
            || busy()
            || conversation_writes_blocked
            || conversation_import_loading()
            || pending_conversation_import().is_some()
            || attachment_loading()
        {
            prompt_edit_error.set(Some(
                "Wait for the current operation and a ready model before editing the prompt."
                    .to_string(),
            ));
            return;
        }
        let Some(model) = connection().active_model().map(str::to_owned) else {
            prompt_edit_error.set(Some(
                "A ready model is required before editing the prompt.".to_string(),
            ));
            return;
        };
        if let Some(transition) = unconfirmed_conversation_model_transition(
            &conversations(),
            Some(&model),
            acknowledged_model_transition().as_ref(),
        ) {
            prompt_edit_error.set(Some(model_transition_unavailable_reason(&transition)));
            return;
        }
        let content = content.trim().to_string();
        if content.is_empty() {
            prompt_edit_error.set(Some("Edited prompt cannot be empty.".to_string()));
            return;
        }
        let mut updated = conversations();
        let edit = match updated.prepare_last_prompt_edit(content) {
            Ok(edit) => edit,
            Err(message) => {
                prompt_edit_error.set(Some(message));
                return;
            }
        };
        let options = generation();
        let history = build_chat_history(&options, &edit.history);
        let cfg = config();
        if let Err(message) = api::preflight_chat_submission(&cfg, &model, &history, &options, None)
        {
            prompt_edit_error.set(Some(message));
            return;
        }
        let cancellation = match ChatCancellation::new() {
            Ok(cancellation) => cancellation,
            Err(message) => {
                prompt_edit_error.set(Some(message));
                return;
            }
        };
        conversations.set(updated);
        pending_prompt_edit.set(None);
        prompt_edit_error.set(None);
        conversation_branch_notice.set(None);
        error.set(None);
        launch_generation(
            GenerationRequest {
                config: cfg,
                model,
                history,
                options,
                attachment: None,
                multimodal_prompt: None,
                cancellation,
                conversation_id: edit.conversation_id,
                assistant_index: edit.assistant_index,
                rollback: edit.rollback,
            },
            GenerationSignals {
                conversations,
                busy,
                error,
                active_cancellation,
                active_request_id,
            },
        );
    };

    let on_stop = move |_| {
        if let Some(cancellation) = active_cancellation() {
            cancellation.cancel();
        }
        if let Some(request_id) = active_request_id() {
            let cfg = config();
            spawn(async move {
                let _ = api::cancel_request(&cfg, &request_id).await;
            });
        }
    };
    let show_embedding_workspace = connection().can_embed() && !connection().can_chat();
    let show_speech_workspace = connection().can_transcribe() && !connection().can_chat();
    let show_specialized_workspace = show_embedding_workspace || show_speech_workspace;
    let embedding_workspace_model = connection()
        .active_model()
        .filter(|_| show_embedding_workspace)
        .unwrap_or_default()
        .to_string();
    let embedding_workspace_rerank = connection().can_rerank();
    let speech_workspace_model = connection()
        .active_model()
        .filter(|_| show_speech_workspace)
        .unwrap_or_default()
        .to_string();

    rsx! {
        document::Stylesheet { href: STYLE }
        div { class: "shell",
            if !show_specialized_workspace {
                aside { class: if show_sidebar() { "sidebar open" } else { "sidebar" },
                div { class: "sidebar-header",
                    div { class: "sidebar-title", "Conversations" }
                    button {
                        class: "close-sidebar",
                        aria_label: "Close conversations",
                        onclick: move |_| show_sidebar.set(false),
                        "×"
                    }
                }
                button {
                    class: "new-chat-btn",
                    disabled: busy() || conversation_import_loading() || conversation_writes_blocked,
                    onclick: move |_| on_new_chat(()),
                    "+ New chat"
                }
                div { class: "conversation-search",
                    input {
                        r#type: "search",
                        aria_label: "Search conversations",
                        placeholder: "Search conversations",
                        maxlength: "{MAX_CONVERSATION_SEARCH_CHARS}",
                        value: "{conversation_search}",
                        oninput: move |event| {
                            conversation_search.set(
                                event.value().chars().take(MAX_CONVERSATION_SEARCH_CHARS).collect()
                            );
                        },
                    }
                    if !conversation_search().is_empty() {
                        button {
                            aria_label: "Clear conversation search",
                            title: "Clear search",
                            onclick: move |_| conversation_search.set(String::new()),
                            "×"
                        }
                    }
                }
                if !conversation_search().trim().is_empty() {
                    div { class: "conversation-search-count", role: "status",
                        "{visible_conversations.len()} of {total_conversation_count} conversations"
                    }
                }
                nav { class: "conversation-list", aria_label: "Saved conversations",
                    if visible_conversations.is_empty() {
                        div { class: "conversation-search-empty",
                            "No conversations match this search."
                        }
                    }
                    for (conversation_id, conversation_title) in visible_conversations.iter() {
                        {
                            let id = *conversation_id;
                            let is_active = id == conversation_snapshot.active_id;
                            let rename_title = conversation_title.clone();
                            rsx! {
                                div {
                                    class: if is_active { "conversation-row active" } else { "conversation-row" },
                                    key: "{id}",
                                    button {
                                        class: "conversation-select",
                                        disabled: busy() || conversation_import_loading() || conversation_writes_blocked,
                                        title: "{conversation_title}",
                                        onclick: move |_| {
                                            let mut updated = conversations();
                                            if !updated.select(id) {
                                                error.set(Some("Conversation no longer exists.".to_string()));
                                                return;
                                            }
                                            match storage::save_conversations(&updated) {
                                                Ok(()) => {
                                                    conversations.set(updated);
                                                    conversation_branch_notice.set(None);
                                                    acknowledged_model_transition.set(None);
                                                    error.set(None);
                                                    show_sidebar.set(false);
                                                }
                                                Err(message) => error.set(Some(message)),
                                            }
                                        },
                                        "{conversation_title}"
                                    }
                                    button {
                                        class: "conversation-rename",
                                        disabled: busy() || conversation_import_loading() || conversation_writes_blocked,
                                        aria_label: "Rename conversation",
                                        title: "Rename conversation",
                                        onclick: move |_| {
                                            conversation_rename_error.set(None);
                                            pending_conversation_rename.set(Some((id, rename_title.clone())));
                                        },
                                        "Edit"
                                    }
                                    button {
                                        class: "conversation-delete",
                                        disabled: busy() || conversation_import_loading() || conversation_writes_blocked,
                                        aria_label: "Delete conversation",
                                        title: "Delete conversation",
                                        onclick: move |_| pending_delete.set(Some(id)),
                                        "×"
                                    }
                                }
                            }
                        }
                    }
                }
                div { class: "conversation-backup",
                    div { class: "conversation-backup-copy",
                        strong { "Conversation backup" }
                        span { "Messages only; connection settings are excluded." }
                    }
                    div { class: "conversation-backup-actions",
                        label {
                            class: if busy() || conversation_import_loading() {
                                "file-action disabled"
                            } else {
                                "file-action"
                            },
                            input {
                                key: "{conversation_import_input_key}",
                                r#type: "file",
                                accept: ".json,application/json",
                                disabled: busy() || conversation_import_loading(),
                                onchange: move |event| {
                                    let Some(file_data) = event.files().into_iter().next() else {
                                        return;
                                    };
                                    if !file_data.name().to_ascii_lowercase().ends_with(".json") {
                                        error.set(Some(
                                            "Choose a Bloom conversation archive JSON file."
                                                .to_string(),
                                        ));
                                        conversation_import_input_key += 1;
                                        return;
                                    }
                                    if file_data.size() == 0
                                        || file_data.size() > MAX_CONVERSATION_ARCHIVE_BYTES
                                    {
                                        error.set(Some(format!(
                                            "Conversation archive must be between 1 byte and {}.",
                                            format_bytes(MAX_CONVERSATION_ARCHIVE_BYTES)
                                        )));
                                        conversation_import_input_key += 1;
                                        return;
                                    }
                                    let Some(file) = file_data
                                        .inner()
                                        .downcast_ref::<web_sys::File>()
                                        .cloned()
                                    else {
                                        error.set(Some(
                                            "This browser did not expose the selected archive file."
                                                .to_string(),
                                        ));
                                        conversation_import_input_key += 1;
                                        return;
                                    };
                                    conversation_import_loading.set(true);
                                    conversation_archive_notice.set(None);
                                    conversation_import_error.set(None);
                                    error.set(None);
                                    spawn(async move {
                                        let result = browser::read_bounded_text_file(
                                            &file,
                                            "Conversation archive",
                                            MAX_CONVERSATION_ARCHIVE_BYTES,
                                        )
                                        .await
                                        .and_then(|text| {
                                            chat::import_conversation_archive(&text)
                                        });
                                        match result {
                                            Ok(store) => {
                                                let conversation_count = store.conversations.len();
                                                let message_count = store
                                                    .conversations
                                                    .iter()
                                                    .map(|conversation| conversation.messages.len())
                                                    .sum();
                                                pending_conversation_import.set(Some(
                                                    PendingConversationImport {
                                                        store,
                                                        conversation_count,
                                                        message_count,
                                                    },
                                                ));
                                                conversation_import_error.set(None);
                                            }
                                            Err(message) => error.set(Some(message)),
                                        }
                                        conversation_import_loading.set(false);
                                        conversation_import_input_key += 1;
                                    });
                                },
                            }
                            span {
                                if conversation_import_loading() { "Reading…" } else { "Import" }
                            }
                        }
                        button {
                            class: "btn-ghost compact",
                            disabled: busy() || conversation_import_loading() || conversation_writes_blocked,
                            onclick: export_conversations,
                            "Export"
                        }
                    }
                    if let Some(message) = conversation_archive_notice() {
                        span { class: "conversation-backup-notice", role: "status", "{message}" }
                    }
                }
                }
            }

            if !show_specialized_workspace && show_sidebar() {
                button {
                    class: "sidebar-backdrop",
                    aria_label: "Close conversations",
                    onclick: move |_| show_sidebar.set(false),
                }
            }

            main { class: "app",
                header { class: "app-header",
                    div { class: "brand",
                        if !show_specialized_workspace {
                            button {
                                class: "menu-btn",
                                aria_label: "Open conversations",
                                onclick: move |_| show_sidebar.set(true),
                                "☰"
                            }
                        }
                        div { class: "brand-logo", "B" }
                        div {
                            div { class: "brand-title", "Bloom" }
                            div { class: "brand-sub",
                                if show_speech_workspace {
                                    "Live speech to text"
                                } else if show_embedding_workspace {
                                    "Local embedding and reranking"
                                } else {
                                    "Local multimodal inference"
                                }
                            }
                        }
                    }
                    div { class: "header-actions",
                        span { class: "status-pill {status_view.0}",
                            span { class: "status-dot" }
                            "{status_view.1}"
                        }
                        button {
                            class: "icon-btn",
                            onclick: move |_| show_models.set(true),
                            "Models"
                        }
                        button {
                            class: "icon-btn diagnostics-button",
                            onclick: move |_| show_diagnostics.set(true),
                            "Diagnostics"
                        }
                        button {
                            class: "icon-btn",
                            onclick: move |_| show_settings.set(true),
                            "Settings"
                        }
                    }
                }

                if let Some(notice) = conversation_storage_notice.as_ref() {
                    div {
                        class: if notice.blocks_conversation_writes() {
                            "conversation-storage-notice blocking"
                        } else {
                            "conversation-storage-notice"
                        },
                        role: "alert",
                        div {
                            strong {
                                if notice.blocks_conversation_writes() {
                                    "Conversation recovery required"
                                } else {
                                    "Conversation storage warning"
                                }
                            }
                            p { "{notice.message}" }
                        }
                        div { class: "conversation-storage-actions",
                            if notice.recovery_text.is_some() {
                                button {
                                    class: "btn-ghost compact",
                                    onclick: move |_| {
                                        let recovery_text = conversation_storage_notice_state()
                                            .and_then(|notice| notice.recovery_text);
                                        let result = recovery_text.ok_or_else(|| {
                                            "Conversation recovery data is no longer available."
                                                .to_string()
                                        }).and_then(|text| {
                                            browser::download_text_file(
                                                storage::CONVERSATION_RECOVERY_FILENAME,
                                                "text/plain;charset=utf-8",
                                                text.as_ref(),
                                            )
                                        });
                                        if let Err(message) = result {
                                            error.set(Some(message));
                                        }
                                    },
                                    "Download recovery"
                                }
                                button {
                                    class: "btn-danger compact",
                                    onclick: move |_| pending_recovery_discard.set(true),
                                    "Start fresh"
                                }
                            } else {
                                button {
                                    class: "btn-ghost compact",
                                    onclick: move |_| {
                                        conversation_storage_notice_state.set(None);
                                    },
                                    "Dismiss"
                                }
                            }
                        }
                    }
                }

                if let Some(message) = error() {
                    div { class: "error-banner", role: "alert",
                        span { "{message}" }
                        button {
                            aria_label: "Dismiss error",
                            onclick: move |_| error.set(None),
                            "×"
                        }
                    }
                }

                if let Some(message) = conversation_branch_notice() {
                    div { class: "status-banner", role: "status",
                        span { "{message}" }
                        button {
                            aria_label: "Dismiss branch confirmation",
                            onclick: move |_| conversation_branch_notice.set(None),
                            "×"
                        }
                    }
                }

                if show_speech_workspace {
                    SpeechWorkspace {
                        key: "{speech_workspace_model}",
                        config,
                        model: speech_workspace_model,
                    }
                } else if show_embedding_workspace {
                    EmbeddingWorkspace {
                        key: "{embedding_workspace_model}",
                        config,
                        model: embedding_workspace_model,
                        supports_rerank: embedding_workspace_rerank,
                    }
                } else {
                MessageList {
                    key: "{conversation_snapshot.active_id}",
                    messages: active_messages,
                    connection: connection(),
                    busy: busy(),
                    retry_unavailable_reason,
                    edit_unavailable_reason,
                    branch_unavailable_reason,
                    on_retry: move |_| on_retry(()),
                    on_edit: move |_| request_prompt_edit(()),
                    on_branch: branch_conversation,
                    on_continue: continue_conversation,
                    on_open_models: move |_| show_models.set(true),
                    on_open_settings: move |_| show_settings.set(true),
                    on_error: move |message| error.set(Some(message)),
                }

                if let Some(transition) = pending_model_transition.as_ref() {
                    section {
                        class: "model-transition-notice",
                        role: "alert",
                        aria_live: "polite",
                        div {
                            strong { "Confirm model change" }
                            p {
                                "This conversation last used {transition.previous_model}. {transition.current_model} is active. Continuing will send the existing conversation history to a different model."
                            }
                        }
                        div { class: "model-transition-actions",
                            button {
                                class: "btn-ghost compact",
                                disabled: busy() || conversation_import_loading() || conversation_writes_blocked,
                                onclick: move |_| on_new_chat(()),
                                "Start new chat"
                            }
                            button {
                                class: "btn-primary compact",
                                disabled: busy() || conversation_import_loading() || conversation_writes_blocked,
                                onclick: {
                                    let transition = transition.clone();
                                    move |_| {
                                        acknowledged_model_transition.set(Some(transition.clone()));
                                        error.set(None);
                                    }
                                },
                                title: "Continue this conversation with {transition.current_model}",
                                "Continue with active model"
                            }
                        }
                    }
                }

                div { class: "composer",
                    if let Some(image) = attachment() {
                        div { class: "attachment-chip",
                            div {
                                strong { "Image" }
                                span { "{image.name} · {format_bytes(image.bytes.len() as u64)}" }
                            }
                            button {
                                aria_label: "Remove image attachment",
                                disabled: busy(),
                                onclick: move |_| {
                                    attachment.set(None);
                                    attachment_input_key += 1;
                                },
                                "×"
                            }
                        }
                    }
                    div { class: "composer-row",
                        label {
                            class: if generation().response_format != ResponseFormatMode::Text
                                || !connection().supports_vision()
                            {
                                "attach-btn disabled"
                            } else if attachment_loading() {
                                "attach-btn loading"
                            } else {
                                "attach-btn"
                            },
                            title: if generation().response_format != ResponseFormatMode::Text {
                                "Structured output is currently available for text chat only"
                            } else if connection().supports_vision() {
                                "Attach a JPEG or PNG image"
                            } else {
                                "The active model does not declare Vision input support"
                            },
                            input {
                                key: "{attachment_input_key}",
                                r#type: "file",
                                accept: "image/jpeg,image/png",
                                disabled: busy()
                                    || attachment_loading()
                                    || !connection().supports_vision()
                                    || generation().response_format != ResponseFormatMode::Text,
                                onchange: move |event| {
                                    let Some(file) = event.files().into_iter().next() else {
                                        return;
                                    };
                                    let name = file.name();
                                    let Some(mime) = supported_image_mime(
                                        &name,
                                        file.content_type().as_deref().unwrap_or_default(),
                                    ) else {
                                        error.set(Some("Choose a JPEG or PNG image.".to_string()));
                                        attachment_input_key += 1;
                                        return;
                                    };
                                    if file.size() > MAX_IMAGE_ATTACHMENT_BYTES {
                                        error.set(Some("Image attachments must be 10 MiB or smaller.".to_string()));
                                        attachment_input_key += 1;
                                        return;
                                    }
                                    attachment_loading.set(true);
                                    error.set(None);
                                    spawn(async move {
                                        match file.read_bytes().await {
                                            Ok(bytes) => attachment.set(Some(ImageAttachment {
                                                name,
                                                mime,
                                                bytes: bytes.to_vec(),
                                            })),
                                            Err(read_error) => error.set(Some(format!(
                                                "Failed to read image attachment: {read_error}"
                                            ))),
                                        }
                                        attachment_loading.set(false);
                                    });
                                },
                            }
                            if attachment_loading() { "Reading…" } else { "Attach" }
                        }
                        textarea {
                            aria_label: "Message",
                            placeholder: "Type a message. Press Enter to send or Shift+Enter for a new line…",
                            maxlength: "{MAX_CHAT_INPUT_CHARS}",
                            value: "{input}",
                            rows: 2,
                            oninput: move |event| input.set(event.value()),
                            onkeydown: move |event| {
                                if event.key() == Key::Enter && !event.modifiers().shift() {
                                    event.prevent_default();
                                    on_send(());
                                }
                            },
                        }
                        if busy() {
                            button {
                                class: "stop-btn",
                                onclick: move |_| on_stop(()),
                                "Stop"
                            }
                        } else {
                            button {
                                class: "send-btn",
                                disabled: !can_send,
                                onclick: move |_| on_send(()),
                                "Send"
                            }
                        }
                    }
                    if let Some((warning, message)) = context_budget_status.as_ref() {
                        div {
                            class: if *warning { "composer-context warning" } else { "composer-context" },
                            aria_live: "polite",
                            "{message}"
                        }
                    }
                    div { class: "composer-hint",
                        if conversation_writes_blocked {
                            "Resolve conversation recovery before sending messages."
                        } else if let Some(message) = structured_output_attachment_error.as_ref() {
                            "{message}"
                        } else if let Some(message) = context_reservation_error.as_ref() {
                            "{message}"
                        } else if let Some(transition) = pending_model_transition.as_ref() {
                            "{model_transition_unavailable_reason(transition)}"
                        } else if !connection().can_chat() {
                            "Wait for the model to become ready before sending."
                        } else if generation().response_format != ResponseFormatMode::Text {
                            "Responses are constrained to {generation().response_format.description()} output."
                        } else {
                            "Responses stream from the OpenAI-compatible API."
                        }
                    }
                }
                }
            }
        }

        if show_settings() {
            SettingsDrawer {
                config,
                on_close: move |_| show_settings.set(false),
                generation,
                context_window,
            }
        }

        if show_models() {
            ModelDrawer {
                config,
                on_close: move |_| show_models.set(false),
            }
        }

        if show_diagnostics() {
            DiagnosticsDrawer {
                config,
                on_close: move |_| show_diagnostics.set(false),
            }
        }

        if let Some(conversation_id) = pending_delete() {
            ConfirmDialog {
                dialog_id: "delete-conversation-confirm-dialog",
                title: "Delete conversation?",
                body: "This removes the conversation from this browser and cannot be undone.",
                confirm_label: "Delete",
                on_cancel: move |_| pending_delete.set(None),
                on_confirm: move |_| {
                    let mut updated = conversations();
                    let result = updated.delete(conversation_id).and_then(|deleted| {
                        if !deleted {
                            return Err("Conversation no longer exists.".to_string());
                        }
                        storage::save_conversations(&updated)
                    });
                    match result {
                        Ok(()) => {
                            conversations.set(updated);
                            conversation_branch_notice.set(None);
                            acknowledged_model_transition.set(None);
                            error.set(None);
                        }
                        Err(message) => error.set(Some(message)),
                    }
                    pending_delete.set(None);
                },
            }
        }

        if let Some(preview) = pending_conversation_import() {
            ConversationImportDialog {
                conversation_count: preview.conversation_count,
                message_count: preview.message_count,
                error: conversation_import_error(),
                merge_unavailable_reason: conversation_writes_blocked.then(|| {
                    "Merge is unavailable while saved-conversation recovery is required. Replace all only after preserving any recovery copy you need."
                        .to_string()
                }),
                on_cancel: move |_| {
                    conversation_import_error.set(None);
                    pending_conversation_import.set(None);
                },
                on_replace: move |_| {
                    let Some(preview) = pending_conversation_import() else {
                        return;
                    };
                    let result = conversation_import_candidate(
                        &conversations(),
                        &preview.store,
                        ConversationImportMode::Replace,
                        conversation_writes_blocked,
                    )
                    .and_then(|replacement| {
                        storage::save_conversations(&replacement)?;
                        Ok(replacement)
                    });
                    match result {
                        Ok(replacement) => {
                            conversations.set(replacement);
                            conversation_storage_notice_state.set(None);
                            input.set(String::new());
                            attachment.set(None);
                            attachment_input_key += 1;
                            conversation_search.set(String::new());
                            conversation_branch_notice.set(None);
                            acknowledged_model_transition.set(None);
                            conversation_import_error.set(None);
                            pending_conversation_import.set(None);
                            conversation_archive_notice.set(Some(format!(
                                "Replaced local history with {} imported conversation(s).",
                                preview.conversation_count
                            )));
                            error.set(None);
                        }
                        Err(message) => conversation_import_error.set(Some(message)),
                    }
                },
                on_merge: move |_| {
                    let Some(preview) = pending_conversation_import() else {
                        return;
                    };
                    let result = conversation_import_candidate(
                        &conversations(),
                        &preview.store,
                        ConversationImportMode::Merge,
                        conversation_writes_blocked,
                    )
                    .and_then(|merged| {
                            storage::save_conversations(&merged)?;
                            Ok(merged)
                        });
                    match result {
                        Ok(merged) => {
                            conversations.set(merged);
                            conversation_storage_notice_state.set(None);
                            conversation_branch_notice.set(None);
                            acknowledged_model_transition.set(None);
                            conversation_import_error.set(None);
                            pending_conversation_import.set(None);
                            conversation_archive_notice.set(Some(format!(
                                "Merged {} imported conversation(s); current history and selection were kept.",
                                preview.conversation_count
                            )));
                            error.set(None);
                        }
                        Err(message) => conversation_import_error.set(Some(message)),
                    }
                },
            }
        }

        if let Some(content) = pending_prompt_edit() {
            EditPromptDialog {
                initial_content: content,
                error: prompt_edit_error(),
                on_cancel: move |_| {
                    prompt_edit_error.set(None);
                    pending_prompt_edit.set(None);
                },
                on_submit: submit_prompt_edit,
            }
        }

        if let Some((conversation_id, title)) = pending_conversation_rename() {
            RenameConversationDialog {
                initial_title: title,
                error: conversation_rename_error(),
                on_cancel: move |_| {
                    conversation_rename_error.set(None);
                    pending_conversation_rename.set(None);
                },
                on_submit: move |title: String| {
                    let mut updated = conversations();
                    let result = updated
                        .rename(conversation_id, &title)
                        .and_then(|_| storage::save_conversations(&updated));
                    match result {
                        Ok(()) => {
                            conversations.set(updated);
                            conversation_branch_notice.set(None);
                            conversation_rename_error.set(None);
                            pending_conversation_rename.set(None);
                            error.set(None);
                        }
                        Err(message) => conversation_rename_error.set(Some(message)),
                    }
                },
            }
        }

        if pending_recovery_discard() {
            ConfirmDialog {
                dialog_id: "discard-conversation-recovery-confirm-dialog",
                title: "Discard unreadable conversation data?",
                body: "This permanently replaces the unreadable browser-local data with an empty conversation store. Download the recovery copy first if you may need its raw contents.",
                confirm_label: "Discard and start fresh",
                on_cancel: move |_| pending_recovery_discard.set(false),
                on_confirm: move |_| {
                    let fresh = ConversationStore::default();
                    match storage::save_conversations(&fresh) {
                        Ok(()) => {
                            conversations.set(fresh);
                            conversation_storage_notice_state.set(None);
                            conversation_search.set(String::new());
                            conversation_branch_notice.set(None);
                            acknowledged_model_transition.set(None);
                            pending_recovery_discard.set(false);
                            error.set(None);
                        }
                        Err(message) => {
                            pending_recovery_discard.set(false);
                            error.set(Some(message));
                        }
                    }
                },
            }
        }
    }
}

#[component]
fn ModelDrawer(config: Signal<ConnConfig>, on_close: EventHandler<()>) -> Element {
    use_modal_focus(MODEL_DRAWER_DIALOG_ID);
    let mut catalog = use_signal(|| Option::<api::ModelCatalog>::None);
    let mut catalog_error = use_signal(|| Option::<String>::None);
    let mut operation_pending = use_signal(|| false);
    let mut operation_error = use_signal(|| Option::<String>::None);
    let mut pending_unload = use_signal(|| false);
    let mut pending_remove = use_signal(|| Option::<String>::None);
    let mut pending_discard = use_signal(|| Option::<String>::None);
    let mut pending_import_discard = use_signal(|| Option::<String>::None);
    let mut download_url = use_signal(String::new);
    let mut download_filename = use_signal(String::new);
    let mut download_sha256 = use_signal(String::new);
    let mut download_license = use_signal(String::new);
    let mut download_inspection = use_signal(|| Option::<api::ModelDownloadSource>::None);
    let mut download_inspect_pending = use_signal(|| false);
    let mut model_index = use_signal(|| Option::<api::ModelIndexSnapshot>::None);
    let mut model_index_error = use_signal(|| Option::<String>::None);
    let mut model_index_pending = use_signal(|| false);
    let mut model_index_attempted_trust = use_signal(|| Option::<String>::None);
    let mut model_index_search = use_signal(String::new);
    let mut selected_import = use_signal(|| Option::<SelectedModelImport>::None);
    let mut import_sha256 = use_signal(String::new);
    let mut import_source_url = use_signal(String::new);
    let mut import_license = use_signal(String::new);
    let mut import_busy = use_signal(|| false);
    let mut import_uploaded_bytes = use_signal(|| 0_u64);
    let mut import_cancellation = use_signal(|| Option::<ModelImportCancellation>::None);
    let mut import_input_key = use_signal(|| 0_u64);
    let mut preflight = use_signal(|| Option::<api::ModelPreflightReport>::None);
    let mut preflight_pending = use_signal(|| Option::<String>::None);
    let mut preflight_error = use_signal(|| Option::<String>::None);
    let mut inventory_pending = use_signal(|| false);
    let mut inventory_notice = use_signal(|| Option::<String>::None);
    let mut inventory_compare_pending = use_signal(|| false);
    let mut inventory_compare_input_key = use_signal(|| 0_u64);
    let mut inventory_reconciliation =
        use_signal(|| Option::<api::ModelInventoryReconciliation>::None);
    let mut inventory_reconciliation_filename = use_signal(|| Option::<String>::None);
    let mut inventory_expected = use_signal(|| Option::<api::ModelInventory>::None);
    let mut pending_inventory_restore = use_signal(|| Option::<String>::None);

    let _catalog_poller = use_future(move || async move {
        loop {
            match api::fetch_model_catalog(&config()).await {
                Ok(snapshot) => {
                    let index_trust = snapshot
                        .index
                        .trust_id
                        .clone()
                        .or_else(|| snapshot.index.key_id.clone())
                        .or_else(|| snapshot.index.enabled.then(|| "legacy".to_string()));
                    let should_load_index = snapshot.index.enabled
                        && !model_index_pending()
                        && model_index_attempted_trust() != index_trust;
                    catalog.set(Some(snapshot));
                    catalog_error.set(None);
                    if should_load_index {
                        model_index_attempted_trust.set(index_trust);
                        model_index_pending.set(true);
                        match api::fetch_model_index(&config(), false).await {
                            Ok(index) => {
                                model_index.set(Some(index));
                                model_index_error.set(None);
                            }
                            Err(message) => model_index_error.set(Some(message)),
                        }
                        model_index_pending.set(false);
                    }
                }
                Err(message) => catalog_error.set(Some(message)),
            }
            TimeoutFuture::new(2_000).await;
        }
    });

    let _model_index_poller = use_future(move || async move {
        loop {
            let refresh_seconds = catalog()
                .filter(|snapshot| snapshot.index.enabled)
                .map(|snapshot| snapshot.index.refresh_seconds)
                .unwrap_or(2);
            let retry_seconds =
                model_index_poll_interval_seconds(refresh_seconds, model_index().is_some());
            let wait_millis =
                u32::try_from(retry_seconds.saturating_mul(1_000)).unwrap_or(u32::MAX);
            TimeoutFuture::new(wait_millis).await;
            if !catalog().is_some_and(|snapshot| snapshot.index.enabled) || model_index_pending() {
                continue;
            }
            model_index_pending.set(true);
            match api::fetch_model_index(&config(), false).await {
                Ok(index) => {
                    model_index.set(Some(index));
                    model_index_error.set(None);
                }
                Err(message) => model_index_error.set(Some(message)),
            }
            model_index_pending.set(false);
        }
    });

    let refresh_model_index = move |_| {
        if model_index_pending() {
            return;
        }
        model_index_pending.set(true);
        model_index_error.set(None);
        let cfg = config();
        spawn(async move {
            match api::fetch_model_index(&cfg, true).await {
                Ok(index) => model_index.set(Some(index)),
                Err(message) => model_index_error.set(Some(message)),
            }
            model_index_pending.set(false);
        });
    };

    let mut load_model = move |model_id: String| {
        if operation_pending() {
            return;
        }
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::switch_model(&cfg, &model_id).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let unload_model = move |_| {
        if operation_pending() {
            return;
        }
        pending_unload.set(false);
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::unload_model(&cfg).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let inspect_download_source = move |_| {
        if operation_pending() || download_inspect_pending() {
            return;
        }
        let source_url = download_url().trim().to_string();
        if source_url.is_empty() {
            operation_error.set(Some(
                "Enter a public Hugging Face model file URL first.".to_string(),
            ));
            return;
        }
        download_inspect_pending.set(true);
        download_inspection.set(None);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            let result = api::inspect_model_download_source(&cfg, &source_url).await;
            if download_url().trim() == source_url {
                match result {
                    Ok(source) => {
                        download_url.set(source.download_url.clone());
                        download_filename.set(source.filename.clone());
                        if let Some(sha256) = source.sha256.as_ref() {
                            download_sha256.set(sha256.clone());
                        }
                        download_inspection.set(Some(source));
                    }
                    Err(message) => operation_error.set(Some(message)),
                }
            }
            download_inspect_pending.set(false);
        });
    };

    let start_download = move |_| {
        if operation_pending() || download_inspect_pending() {
            return;
        }
        let url = download_url().trim().to_string();
        let filename = download_filename().trim().to_string();
        let sha256 = download_sha256().trim().to_string();
        let license = download_license().trim().to_string();
        if url.is_empty() || filename.is_empty() {
            operation_error.set(Some(
                "URL, destination filename, and SHA-256 are required.".to_string(),
            ));
            return;
        }
        if !valid_sha256_input(&sha256) {
            operation_error.set(Some(
                "SHA-256 must contain exactly 64 hexadecimal characters.".to_string(),
            ));
            return;
        }
        let Some(snapshot) = catalog() else {
            return;
        };
        if !license_policy_allows(&snapshot.download.license_policy, &license) {
            operation_error.set(Some(
                "Select a license allowed by the server acquisition policy.".to_string(),
            ));
            return;
        }
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) =
                api::download_model(&cfg, &url, &filename, &sha256, &license).await
            {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let cancel_download = move |_| {
        if operation_pending() {
            return;
        }
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::cancel_model_download(&cfg).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let mut resume_download = move |filename: String| {
        if operation_pending() {
            return;
        }
        let license = download_license().trim().to_string();
        if catalog().is_some_and(|snapshot| {
            !license_policy_allows(&snapshot.download.license_policy, &license)
        }) {
            operation_error.set(Some(
                "Select a license allowed by the server acquisition policy.".to_string(),
            ));
            return;
        }
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::resume_model_download(&cfg, &filename, &license).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let remove_model = move |_| {
        let Some(model_id) = pending_remove() else {
            return;
        };
        if operation_pending() {
            return;
        }
        pending_remove.set(None);
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::remove_model(&cfg, &model_id).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let discard_download = move |_| {
        let Some(filename) = pending_discard() else {
            return;
        };
        if operation_pending() {
            return;
        }
        pending_discard.set(None);
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::discard_model_download(&cfg, &filename).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let start_import = move |_| {
        if operation_pending() || import_busy() {
            return;
        }
        let Some(selected) = selected_import() else {
            operation_error.set(Some("Choose a local model file to import.".to_string()));
            return;
        };
        let sha256 = import_sha256().trim().to_string();
        let source_url = import_source_url().trim().to_string();
        let license = import_license().trim().to_string();
        if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            operation_error.set(Some(
                "SHA-256 must contain exactly 64 hexadecimal characters.".to_string(),
            ));
            return;
        }
        let Some(snapshot) = catalog() else {
            return;
        };
        if !snapshot.import.enabled {
            operation_error.set(Some("Local model imports are disabled.".to_string()));
            return;
        }
        if !license_policy_allows(&snapshot.import.license_policy, &license) {
            operation_error.set(Some(
                "Select a license allowed by the server acquisition policy.".to_string(),
            ));
            return;
        }
        if selected.size_bytes == 0 || selected.size_bytes > snapshot.import.max_bytes {
            operation_error.set(Some(format!(
                "Model file must be between 1 byte and {}.",
                format_bytes(snapshot.import.max_bytes)
            )));
            return;
        }
        let chunk_bytes = MODEL_IMPORT_CHUNK_BYTES
            .min(snapshot.import.max_chunk_bytes)
            .max(1);
        let cancellation = match ModelImportCancellation::new() {
            Ok(cancellation) => cancellation,
            Err(message) => {
                operation_error.set(Some(message));
                return;
            }
        };
        operation_pending.set(true);
        import_busy.set(true);
        import_uploaded_bytes.set(0);
        import_cancellation.set(Some(cancellation.clone()));
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            let result = api::import_model_file(
                &cfg,
                &selected.file,
                &sha256,
                api::ModelImportMetadata {
                    source_url: &source_url,
                    license: &license,
                },
                chunk_bytes,
                &cancellation,
                |uploaded, _total| import_uploaded_bytes.set(uploaded),
            )
            .await;
            import_cancellation.set(None);
            import_busy.set(false);
            operation_pending.set(false);
            match result {
                Ok(()) => {
                    selected_import.set(None);
                    import_sha256.set(String::new());
                    import_source_url.set(String::new());
                    import_license.set(String::new());
                    import_input_key += 1;
                }
                Err(ModelImportClientError::Cancelled) => {}
                Err(ModelImportClientError::Request(message)) => {
                    operation_error.set(Some(message));
                }
            }
        });
    };

    let cancel_import = move |_| {
        if let Some(cancellation) = import_cancellation() {
            cancellation.cancel();
        }
    };

    let discard_import = move |_| {
        let Some(filename) = pending_import_discard() else {
            return;
        };
        if operation_pending() {
            return;
        }
        pending_import_discard.set(None);
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::discard_model_import(&cfg, &filename).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let mut verify_model = move |model_id: String| {
        if operation_pending() {
            return;
        }
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::verify_model_integrity(&cfg, &model_id).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let cancel_integrity = move |_| {
        if operation_pending() {
            return;
        }
        operation_pending.set(true);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            if let Err(message) = api::cancel_model_integrity(&cfg).await {
                operation_error.set(Some(message));
            }
            operation_pending.set(false);
        });
    };

    let mut inspect_model = move |model_id: String| {
        if preflight_pending().is_some() {
            return;
        }
        preflight.set(None);
        preflight_pending.set(Some(model_id.clone()));
        preflight_error.set(None);
        let cfg = config();
        spawn(async move {
            match api::preflight_model(&cfg, &model_id).await {
                Ok(report) => preflight.set(Some(report)),
                Err(message) => preflight_error.set(Some(message)),
            }
            preflight_pending.set(None);
        });
    };

    let export_inventory = move |_| {
        if inventory_pending() {
            return;
        }
        inventory_pending.set(true);
        inventory_notice.set(None);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            match api::fetch_model_inventory(&cfg).await {
                Ok(inventory) => match api::download_model_inventory(&inventory) {
                    Ok(()) => inventory_notice.set(Some(format!(
                        "Exported {} model(s); {} source-locked.",
                        inventory.summary.model_count, inventory.summary.source_locked_count
                    ))),
                    Err(message) => operation_error.set(Some(message)),
                },
                Err(message) => operation_error.set(Some(message)),
            }
            inventory_pending.set(false);
        });
    };

    let mut compare_inventory = move |file: web_sys::File, filename: String| {
        if inventory_compare_pending() {
            return;
        }
        inventory_compare_pending.set(true);
        inventory_reconciliation.set(None);
        inventory_reconciliation_filename.set(None);
        inventory_expected.set(None);
        pending_inventory_restore.set(None);
        inventory_notice.set(None);
        operation_error.set(None);
        let cfg = config();
        spawn(async move {
            match api::reconcile_model_inventory_file(&cfg, &file).await {
                Ok(comparison) => {
                    inventory_expected.set(Some(comparison.expected));
                    inventory_reconciliation.set(Some(comparison.report));
                    inventory_reconciliation_filename.set(Some(filename));
                }
                Err(message) => operation_error.set(Some(message)),
            }
            inventory_compare_pending.set(false);
            inventory_compare_input_key += 1;
        });
    };

    let restore_inventory_model = move |_| {
        let Some(model_id) = pending_inventory_restore() else {
            return;
        };
        let Some(expected) = inventory_expected() else {
            pending_inventory_restore.set(None);
            operation_error.set(Some(
                "Compare the inventory again before restoring a model.".to_string(),
            ));
            return;
        };
        if operation_pending() {
            return;
        }
        pending_inventory_restore.set(None);
        operation_pending.set(true);
        operation_error.set(None);
        inventory_notice.set(None);
        let cfg = config();
        spawn(async move {
            match api::restore_model_from_inventory(&cfg, &expected, &model_id).await {
                Ok(()) => inventory_notice.set(Some(format!(
                    "Queued verified inventory restore for {model_id}."
                ))),
                Err(message) => operation_error.set(Some(message)),
            }
            operation_pending.set(false);
        });
    };

    rsx! {
        div { class: "drawer-overlay", onclick: move |_| on_close.call(()),
            div {
                id: MODEL_DRAWER_DIALOG_ID,
                class: "drawer model-drawer",
                role: "dialog",
                aria_modal: "true",
                aria_labelledby: "model-manager-title",
                aria_describedby: "model-manager-description",
                onclick: move |event| event.stop_propagation(),
                onkeydown: move |event| handle_modal_key(event, MODEL_DRAWER_DIALOG_ID, on_close),
                div { class: "drawer-title-row",
                    div {
                        h3 { id: "model-manager-title", "Models" }
                        div { id: "model-manager-description", class: "drawer-sub", "Load models discovered by the Bloom server." }
                    }
                    button {
                        class: "drawer-close",
                        aria_label: "Close model manager",
                        onclick: move |_| on_close.call(()),
                        "×"
                    }
                }

                if let Some(message) = catalog_error() {
                    div { class: "test-result err", role: "alert", "Catalog unavailable: {message}" }
                }
                if let Some(message) = operation_error() {
                    div { class: "test-result err", role: "alert", "{message}" }
                }
                if let Some(message) = preflight_error() {
                    div { class: "test-result err", role: "alert", "Model inspection failed: {message}" }
                }
                if let Some(message) = inventory_notice() {
                    div { class: "test-result ok", role: "status", "{message}" }
                }

                if let Some(snapshot) = catalog() {
                    div { class: "model-status",
                        span { class: "model-status-label", "Runtime" }
                        strong {
                            if snapshot.load.phase == "loading" {
                                "Loading {snapshot.load.progress}%"
                            } else if snapshot.load.phase == "ready" {
                                "Ready"
                            } else if snapshot.load.phase == "error" {
                                "Load failed"
                            } else {
                                "No model loaded"
                            }
                        }
                        if let Some(requested) = snapshot.load.requested_model.as_ref() {
                            span { class: "model-status-detail", "Requested: {requested}" }
                        }
                    }
                    if let Some(message) = snapshot.load.error.as_ref() {
                        div { class: "test-result err", role: "alert", "Last load failed: {message}" }
                    }
                    if let Some(active) = snapshot.active_model.as_ref() {
                        div { class: "active-model-card",
                            div {
                                span { class: "model-card-eyebrow", "Active model" }
                                strong { "{active.id}" }
                                span { class: "model-card-meta",
                                    {format!(
                                        "Source: {} · Inputs: {}",
                                        active.source,
                                        active.input_modalities.join(", ")
                                    )}
                                }
                            }
                            button {
                                class: "btn-ghost compact",
                                disabled: operation_pending() || snapshot.load.phase == "loading",
                                onclick: move |_| pending_unload.set(true),
                                "Unload"
                            }
                        }
                    }

                    div { class: "catalog-location",
                        span { "Catalog directory" }
                        code { "{snapshot.root}" }
                    }
                    div { class: "model-catalog-toolbar",
                        div { class: "model-catalog-copy",
                            strong { "Portable inventory" }
                            span { "Versioned JSON without local paths or URL secrets." }
                        }
                        div { class: "model-catalog-actions",
                            label {
                                class: if inventory_compare_pending() {
                                    "file-action disabled"
                                } else {
                                    "file-action"
                                },
                                input {
                                    key: "{inventory_compare_input_key}",
                                    r#type: "file",
                                    accept: ".json,application/json",
                                    disabled: inventory_compare_pending(),
                                    onchange: move |event| {
                                        let Some(file_data) = event.files().into_iter().next() else {
                                            return;
                                        };
                                        let filename = file_data.name();
                                        if !filename.to_ascii_lowercase().ends_with(".json") {
                                            operation_error.set(Some(
                                                "Choose a Bloom model inventory JSON file.".to_string()
                                            ));
                                            inventory_compare_input_key += 1;
                                            return;
                                        }
                                        if file_data.size() == 0
                                            || file_data.size() > api::MAX_MODEL_INVENTORY_BYTES
                                        {
                                            operation_error.set(Some(format!(
                                                "Inventory file must be between 1 byte and {}.",
                                                format_bytes(api::MAX_MODEL_INVENTORY_BYTES)
                                            )));
                                            inventory_compare_input_key += 1;
                                            return;
                                        }
                                        let Some(file) = file_data
                                            .inner()
                                            .downcast_ref::<web_sys::File>()
                                            .cloned()
                                        else {
                                            operation_error.set(Some(
                                                "This browser did not expose the selected inventory file.".to_string()
                                            ));
                                            inventory_compare_input_key += 1;
                                            return;
                                        };
                                        compare_inventory(file, filename);
                                    },
                                }
                                span {
                                    if inventory_compare_pending() { "Comparing…" } else { "Compare JSON" }
                                }
                            }
                            button {
                                class: "btn-ghost compact",
                                disabled: inventory_pending() || !snapshot.root_exists,
                                onclick: export_inventory,
                                if inventory_pending() { "Exporting…" } else { "Export JSON" }
                            }
                        }
                    }
                    if let Some(report) = inventory_reconciliation() {
                        section {
                            class: if report.in_sync {
                                "inventory-reconciliation-card in-sync"
                            } else {
                                "inventory-reconciliation-card has-drift"
                            },
                            aria_label: "Model inventory reconciliation",
                            div { class: "inventory-reconciliation-heading",
                                div {
                                    span { class: "model-card-eyebrow", "Inventory comparison" }
                                    strong {
                                        if let Some(filename) = inventory_reconciliation_filename() {
                                            "{filename}"
                                        } else {
                                            "Expected inventory"
                                        }
                                    }
                                }
                                div { class: "inventory-reconciliation-actions",
                                    span {
                                        class: if report.in_sync {
                                            "preflight-badge ready"
                                        } else if report.summary.blocking_count > 0 {
                                            "preflight-badge blocked"
                                        } else {
                                            "preflight-badge warning"
                                        },
                                        if report.in_sync {
                                            "In sync"
                                        } else if report.summary.blocking_count > 0 {
                                            "Blocking drift"
                                        } else {
                                            "Warnings"
                                        }
                                    }
                                    button {
                                        class: "btn-ghost compact",
                                        onclick: move |_| {
                                            inventory_reconciliation.set(None);
                                            inventory_reconciliation_filename.set(None);
                                            inventory_expected.set(None);
                                            pending_inventory_restore.set(None);
                                        },
                                        "Clear"
                                    }
                                }
                            }
                            div { class: "inventory-summary-grid",
                                span { strong { "{report.summary.matching_count}" } " matching" }
                                span { strong { "{report.summary.missing_count}" } " missing" }
                                span { strong { "{report.summary.unexpected_count}" } " unexpected" }
                                span { strong { "{report.summary.changed_count}" } " changed" }
                                span { strong { "{report.summary.blocking_count}" } " blocking" }
                                span { strong { "{report.summary.restorable_count}" } " restorable" }
                            }
                            if report.in_sync {
                                p { class: "inventory-sync-message",
                                    "The current catalog matches the selected inventory. No files were changed."
                                }
                            } else {
                                div { class: "inventory-drift-list",
                                    for drift in report.drift.iter() {
                                        {
                                            let restore_id = drift.id.clone();
                                            let download_busy = matches!(
                                                snapshot.download.status.phase.as_str(),
                                                "queued" | "downloading" | "verifying"
                                            );
                                            rsx! {
                                                article {
                                                    class: if drift.severity == "blocking" {
                                                        "inventory-drift-row blocking"
                                                    } else {
                                                        "inventory-drift-row warning"
                                                    },
                                                    key: "{drift.id}",
                                                    div { class: "inventory-drift-heading",
                                                        div {
                                                            strong { "{drift.id}" }
                                                            span {
                                                                "{format_inventory_drift_status(&drift.status)} · {format_inventory_drift_severity(&drift.severity)}"
                                                            }
                                                        }
                                                        if drift.restore_available {
                                                            button {
                                                                class: "btn-ghost compact",
                                                                disabled: operation_pending()
                                                                    || !snapshot.download.enabled
                                                                    || download_busy,
                                                                title: if !snapshot.download.enabled {
                                                                    "Start bloom_server with --enable-model-downloads to restore this model"
                                                                } else if download_busy {
                                                                    "Wait for the active model download to finish"
                                                                } else {
                                                                    "Restore from the exact-commit source and verify SHA-256"
                                                                },
                                                                onclick: move |_| {
                                                                    pending_inventory_restore.set(Some(restore_id.clone()));
                                                                },
                                                                "Restore"
                                                            }
                                                        }
                                                    }
                                                    span { class: "inventory-drift-changes",
                                                        {format_inventory_changes(&drift.changes)}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if report.summary.missing_count > report.summary.restorable_count {
                                    p { class: "inventory-restore-guidance",
                                        "Some missing models cannot be restored automatically because they lack a verified exact-commit download record."
                                    }
                                }
                                if report.summary.restorable_count > 0 && !snapshot.download.enabled {
                                    p { class: "inventory-restore-guidance",
                                        "Verified model downloads are disabled. Restart bloom_server with --enable-model-downloads to enable explicit restores."
                                    }
                                }
                                if report.truncated {
                                    p { class: "inventory-truncated-warning",
                                        "Showing the first {report.drift.len()} of {report.summary.drift_count} drift entries. Summary counts include every entry."
                                    }
                                }
                                p { class: "inventory-sync-message",
                                    "The comparison itself is read-only. Restores require per-model confirmation and never overwrite an existing catalog entry. Compare again after catalog changes."
                                }
                            }
                        }
                    }
                    section { class: "model-storage-card", aria_label: "Model storage budget",
                        div { class: "model-storage-heading",
                            div {
                                span { class: "model-card-eyebrow", "Storage" }
                                strong {
                                    if let Some(limit) = snapshot.storage.max_bytes {
                                        {format!(
                                            "{} of {} committed",
                                            format_bytes(snapshot.storage.committed_bytes),
                                            format_bytes(limit)
                                        )}
                                    } else {
                                        {format!(
                                            "{} used · quota disabled",
                                            format_bytes(snapshot.storage.used_bytes)
                                        )}
                                    }
                                }
                            }
                            if let Some(available) = snapshot.storage.available_bytes {
                                span { "{format_bytes(available)} available" }
                            }
                        }
                        if let Some(limit) = snapshot.storage.max_bytes {
                            progress {
                                max: "{limit.max(1)}",
                                value: "{snapshot.storage.committed_bytes.min(limit)}",
                                aria_label: "Committed model storage",
                            }
                        }
                        div { class: "model-storage-breakdown",
                            span { "Installed {format_bytes(snapshot.storage.installed_bytes)}" }
                            span {
                                {format!(
                                    "Staged {}",
                                    format_bytes(
                                        snapshot.storage.staged_download_bytes
                                            .saturating_add(snapshot.storage.staged_import_bytes)
                                    )
                                )}
                            }
                            span { "Reserved {format_bytes(snapshot.storage.reserved_bytes)}" }
                        }
                        if snapshot.storage.staged_retention_seconds > 0 {
                            p {
                                "Automatic cleanup removes inactive staging older than {format_duration(snapshot.storage.staged_retention_seconds)}. Last run removed {snapshot.storage.last_cleanup_removed_sessions} session(s)."
                            }
                        }
                    }
                    if snapshot.integrity.phase != "idle" {
                        section {
                            class: if snapshot.integrity.matches_expected == Some(false) {
                                "model-integrity-card mismatch"
                            } else if snapshot.integrity.matches_expected == Some(true) {
                                "model-integrity-card verified"
                            } else {
                                "model-integrity-card"
                            },
                            aria_label: "Model integrity verification",
                            div { class: "download-status-row",
                                div {
                                    span { class: "model-card-eyebrow", "Integrity verification" }
                                    strong { {integrity_phase_label(&snapshot.integrity)} }
                                    if let Some(model_id) = snapshot.integrity.model_id.as_ref() {
                                        span { class: "model-card-meta", "{model_id}" }
                                    }
                                }
                                if matches!(snapshot.integrity.phase.as_str(), "queued" | "verifying") {
                                    button {
                                        class: "btn-ghost compact",
                                        disabled: operation_pending(),
                                        onclick: cancel_integrity,
                                        "Cancel"
                                    }
                                }
                            }
                            if let Some(total) = snapshot.integrity.total_bytes {
                                progress {
                                    max: "{total.max(1)}",
                                    value: "{snapshot.integrity.checked_bytes.min(total)}",
                                    aria_label: "Model integrity verification progress",
                                }
                                span { class: "model-card-meta",
                                    "{format_download_progress(snapshot.integrity.checked_bytes, Some(total))}"
                                }
                            }
                            if snapshot.integrity.matches_expected == Some(false) {
                                div { class: "integrity-result error", role: "alert",
                                    "Checksum mismatch. Loading this catalog entry is blocked."
                                }
                            } else if snapshot.integrity.matches_expected == Some(true) {
                                div { class: "integrity-result ok",
                                    "The current file matches its verified acquisition checksum."
                                }
                            }
                            if let Some(message) = snapshot.integrity.error.as_ref() {
                                div { class: "download-error", role: "alert", "{message}" }
                            }
                        }
                    }
                    section { class: "model-download-panel", aria_label: "Verified model download",
                        div { class: "model-download-heading",
                            div {
                                strong { "Download a model" }
                                p { "Inspect a Hugging Face file, review its immutable source and checksum, then install it with mandatory SHA-256 verification." }
                            }
                            if !snapshot.download.enabled {
                                span { class: "feature-disabled", "Disabled" }
                            }
                        }
                        if snapshot.index.enabled {
                            section { class: "model-index-browser", aria_label: "Publisher-signed model index",
                                div { class: "model-index-heading",
                                    div {
                                        span { class: "model-card-eyebrow", "Verified discovery" }
                                        strong {
                                            if let Some(index) = model_index() {
                                                "{index.name}"
                                            } else {
                                                "Publisher-signed model index"
                                            }
                                        }
                                        p { "Bloom verifies the publisher signature and immutable source metadata before showing these entries. Final model bytes still require SHA-256 verification." }
                                    }
                                    button {
                                        class: "btn-ghost compact",
                                        disabled: model_index_pending(),
                                        aria_busy: model_index_pending(),
                                        onclick: refresh_model_index,
                                        if model_index_pending() { "Refreshing…" } else { "Refresh" }
                                    }
                                }
                                if let Some(message) = model_index_error() {
                                    div { class: "download-error", role: "alert", "Index unavailable: {message}" }
                                }
                                if let Some(index) = model_index() {
                                    div { class: "model-index-trust",
                                        span {
                                            if snapshot.index.trusted_key_count > 1 {
                                                "Signer {&index.key_id[..12]} · one of {snapshot.index.trusted_key_count} trusted keys"
                                            } else {
                                                "Trusted signer {&index.key_id[..12]}"
                                            }
                                        }
                                        span {
                                            "{index.cache_status}"
                                            if snapshot.index.persistent_rollback_protection {
                                                " · persistent rollback"
                                            }
                                            " · expires Unix {index.expires_at}"
                                        }
                                    }
                                    if let Some(warning) = index.warning.as_ref() {
                                        div { class: "integrity-result warning", role: "status", "{warning}" }
                                    }
                                    label { class: "model-index-search",
                                        span { "Search verified models" }
                                        input {
                                            r#type: "search",
                                            value: "{model_index_search}",
                                            placeholder: "Name, family, quantization, or tag",
                                            maxlength: 120,
                                            oninput: move |event| model_index_search.set(event.value()),
                                        }
                                    }
                                    div { class: "model-index-list",
                                        for entry in index.data.iter().filter(|entry| model_index_entry_matches(entry, model_index_search().trim())) {
                                            {
                                                let entry_id_for_download = entry.id.clone();
                                                let local_state = model_index_local_state(entry, &snapshot.data);
                                                let upgrade_active = model_index_upgrade_source_is_active(entry, &snapshot.data);
                                                let blocked = !entry.downloadable
                                                    || local_state == ModelIndexLocalState::Conflict
                                                    || upgrade_active;
                                                let installed = local_state == ModelIndexLocalState::Verified;
                                                let upgradable = local_state == ModelIndexLocalState::Upgradable;
                                                let download_busy = matches!(
                                                    snapshot.download.status.phase.as_str(),
                                                    "queued" | "downloading" | "verifying"
                                                );
                                                rsx! {
                                                    article { class: if blocked { "model-index-entry blocked" } else { "model-index-entry" }, key: "{entry.id}",
                                                        div { class: "model-index-entry-heading",
                                                            div {
                                                                strong { "{entry.name}" }
                                                                span { "{entry.description}" }
                                                            }
                                                            span { class: "model-index-size", "{format_bytes(entry.size_bytes)}" }
                                                        }
                                                        div { class: "model-index-metadata",
                                                            span { "{entry.format.to_ascii_uppercase()}" }
                                                            if !entry.files.is_empty() {
                                                                span { "{entry.files.len()} files" }
                                                            }
                                                            span { "{entry.license}" }
                                                            if let Some(family) = entry.family.as_ref() {
                                                                span { "{family}" }
                                                            }
                                                            if let Some(parameters) = entry.parameter_count {
                                                                span { "{format_parameter_count(Some(parameters))} params" }
                                                            }
                                                            if let Some(quantization) = entry.quantization.as_ref() {
                                                                span { "{quantization}" }
                                                            }
                                                        }
                                                        if !entry.tags.is_empty() {
                                                            div { class: "model-index-tags",
                                                                for tag in entry.tags.iter() {
                                                                    span { key: "{tag}", "{tag}" }
                                                                }
                                                            }
                                                        }
                                                        div { class: "model-index-entry-action",
                                                            if local_state == ModelIndexLocalState::Verified {
                                                                span { class: "integrity-ok", "Installed · signed identity matches" }
                                                            } else if upgradable && upgrade_active {
                                                                span { class: "model-index-blocked", "Update available · unload the installed model first" }
                                                            } else if upgradable {
                                                                span { class: "integrity-result warning", "Verified update available · previous version is retained until commit" }
                                                            } else if local_state == ModelIndexLocalState::Conflict {
                                                                span { class: "model-index-blocked", "Local conflict: remove or rename the existing catalog entry" }
                                                            } else if blocked {
                                                                span { class: "model-index-blocked", "Blocked: {format_model_index_blockers(&entry.blocking_reasons)}" }
                                                            } else {
                                                                span { "Exact commit · signed SHA-256" }
                                                            }
                                                            button {
                                                                class: "btn-ghost compact",
                                                                disabled: blocked
                                                                    || installed
                                                                    || !snapshot.download.enabled
                                                                    || operation_pending()
                                                                    || download_busy,
                                                                title: if installed {
                                                                    "This exact signed model is already installed"
                                                                } else if upgrade_active {
                                                                    "Unload or switch away from the installed model before upgrading"
                                                                } else if upgradable {
                                                                    "Download, verify, and transactionally replace the previous signed version"
                                                                } else if local_state == ModelIndexLocalState::Conflict {
                                                                    "Remove or rename the conflicting local catalog entry first"
                                                                } else if !snapshot.download.enabled {
                                                                    "Verified downloads are disabled on this server"
                                                                } else if blocked {
                                                                    "This entry conflicts with the server acquisition policy"
                                                                } else {
                                                                    "Download this server-verified signed entry"
                                                                },
                                                                onclick: move |_| {
                                                                    if operation_pending() {
                                                                        return;
                                                                    }
                                                                    operation_pending.set(true);
                                                                    operation_error.set(None);
                                                                    let cfg = config();
                                                                    let model_index_id = entry_id_for_download.clone();
                                                                    spawn(async move {
                                                                        if let Err(message) = api::download_index_model(&cfg, &model_index_id).await {
                                                                            operation_error.set(Some(message));
                                                                        }
                                                                        operation_pending.set(false);
                                                                    });
                                                                },
                                                                if installed {
                                                                    "Installed"
                                                                } else if upgradable {
                                                                    "Upgrade signed model"
                                                                } else {
                                                                    "Download signed model"
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else if model_index_pending() {
                                    div { class: "catalog-empty", "Verifying signed model index…" }
                                }
                            }
                        }
                        if snapshot.download.enabled {
                            div { class: "model-download-form",
                                label {
                                    span { "Hugging Face HTTPS URL" }
                                    input {
                                        r#type: "url",
                                        value: "{download_url}",
                                        placeholder: "https://huggingface.co/.../resolve/main/model.gguf",
                                        maxlength: 2048,
                                        disabled: operation_pending() || matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying"),
                                        oninput: move |event| {
                                            download_url.set(event.value());
                                            download_sha256.set(String::new());
                                            download_inspection.set(None);
                                        },
                                    }
                                }
                                button {
                                    class: "btn-ghost compact",
                                    disabled: operation_pending()
                                        || download_inspect_pending()
                                        || matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying")
                                        || download_url().trim().is_empty(),
                                    aria_busy: download_inspect_pending(),
                                    onclick: inspect_download_source,
                                    if download_inspect_pending() { "Inspecting source…" } else { "Inspect source" }
                                }
                                if let Some(source) = download_inspection() {
                                    div { class: "download-source-card",
                                        div { class: "download-source-summary",
                                            div {
                                                span { class: "model-card-eyebrow", "Source metadata" }
                                                strong {
                                                    if source.verification_ready { "Ready for verified download" } else { "Manual checksum required" }
                                                }
                                            }
                                            if let Some(size) = source.size_bytes {
                                                span { class: "model-card-meta", "{format_bytes(size)}" }
                                            }
                                        }
                                        span { class: "model-card-meta", "Destination: {source.filename}" }
                                        if let Some(commit) = source.commit_hash.as_ref() {
                                            span { class: "model-card-meta source-commit", "Pinned commit: {commit}" }
                                        }
                                        if source.sha256.is_some() {
                                            div { class: "integrity-result ok", "A published 64-character SHA-256 was discovered. The complete download will still be verified locally." }
                                        }
                                        if let Some(warning) = source.warning.as_ref() {
                                            div { class: "integrity-result warning", role: "alert", "{warning}" }
                                        }
                                    }
                                }
                                label {
                                    span { "Destination filename" }
                                    input {
                                        value: "{download_filename}",
                                        placeholder: "model-q4_k_m.gguf",
                                        disabled: operation_pending() || matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying"),
                                        oninput: move |event| {
                                            download_filename.set(event.value());
                                            download_inspection.set(None);
                                        },
                                    }
                                }
                                label {
                                    span { "SHA-256 checksum" }
                                    input {
                                        value: "{download_sha256}",
                                        placeholder: "64 hexadecimal characters",
                                        maxlength: 64,
                                        spellcheck: false,
                                        disabled: operation_pending() || matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying"),
                                        oninput: move |event| {
                                            download_sha256.set(event.value());
                                            download_inspection.set(None);
                                        },
                                    }
                                }
                                if snapshot.download.license_policy.enforced {
                                    label {
                                        span { "Approved model license" }
                                        select {
                                            value: "{download_license}",
                                            disabled: operation_pending() || matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying"),
                                            onchange: move |event| download_license.set(event.value()),
                                            option { value: "", "Select an approved license" }
                                            for license in snapshot.download.license_policy.allowed.iter().take(64) {
                                                option { value: "{license}", "{license}" }
                                            }
                                        }
                                        small { class: "license-policy-help", "Required by the server acquisition policy." }
                                    }
                                } else {
                                    label {
                                        span { "License or SPDX expression (optional)" }
                                        input {
                                            value: "{download_license}",
                                            placeholder: "Apache-2.0",
                                            maxlength: 128,
                                            disabled: operation_pending() || matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying"),
                                            oninput: move |event| download_license.set(event.value()),
                                        }
                                    }
                                }
                                button {
                                    class: "btn-primary compact",
                                    disabled: operation_pending()
                                        || download_inspect_pending()
                                        || matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying")
                                        || download_url().trim().is_empty()
                                        || download_filename().trim().is_empty()
                                        || !valid_sha256_input(download_sha256().trim())
                                        || !license_policy_allows(&snapshot.download.license_policy, download_license().trim()),
                                    onclick: start_download,
                                    if snapshot.download.status.resumable { "Resume download" } else { "Download and verify" }
                                }
                            }
                        } else {
                            p { class: "model-download-disabled",
                                "Start bloom_server with --enable-model-downloads to allow authenticated downloads from trusted hosts."
                            }
                        }

                        if snapshot.download.status.phase != "idle" {
                            div { class: "download-status-card",
                                div { class: "download-status-row",
                                    div {
                                        span { class: "model-card-eyebrow", "Download status" }
                                        strong {
                                            {download_phase_label(&snapshot.download.status.phase)}
                                        }
                                        if let Some(filename) = snapshot.download.status.filename.as_ref() {
                                            span { class: "model-card-meta",
                                                {format!(
                                                    "{} · {}",
                                                    filename,
                                                    format_download_progress(
                                                        snapshot.download.status.downloaded_bytes,
                                                        snapshot.download.status.total_bytes,
                                                    )
                                                )}
                                            }
                                        }
                                    }
                                    if matches!(snapshot.download.status.phase.as_str(), "queued" | "downloading" | "verifying") {
                                        button {
                                            class: "btn-ghost compact",
                                            disabled: operation_pending(),
                                            onclick: cancel_download,
                                            "Cancel"
                                        }
                                    }
                                }
                                if let Some(total) = snapshot.download.status.total_bytes {
                                    progress {
                                        max: "{total.max(1)}",
                                        value: "{snapshot.download.status.downloaded_bytes.min(total)}",
                                        aria_label: "Model download progress",
                                    }
                                }
                                if let Some(message) = snapshot.download.status.error.as_ref() {
                                    div { class: "download-error", role: "alert", "{message}" }
                                }
                            }
                        }
                        if !snapshot.download.staged.is_empty() {
                            div { class: "staged-downloads",
                                span { class: "model-card-eyebrow", "Resumable downloads" }
                                for staged in snapshot.download.staged.iter() {
                                    {
                                        let resume_filename = staged.filename.clone();
                                        let discard_filename = staged.filename.clone();
                                        let is_active_download = matches!(
                                            snapshot.download.status.phase.as_str(),
                                            "queued" | "downloading" | "verifying"
                                        ) && snapshot.download.status.filename.as_deref()
                                            == Some(staged.filename.as_str());
                                        rsx! {
                                            div { class: "staged-download-row", key: "{staged.filename}",
                                                div {
                                                    strong { "{staged.filename}" }
                                                    span {
                                                        {format!(
                                                            "{} · {}",
                                                            staged.source_host,
                                                            format_bytes(staged.downloaded_bytes)
                                                        )}
                                                    }
                                                }
                                                div { class: "model-card-actions",
                                                    button {
                                                        class: "btn-ghost compact",
                                                        disabled: operation_pending()
                                                            || is_active_download
                                                            || !license_policy_allows(&snapshot.download.license_policy, download_license().trim()),
                                                        onclick: move |_| resume_download(resume_filename.clone()),
                                                        "Resume"
                                                    }
                                                    button {
                                                        class: "btn-danger compact",
                                                        disabled: operation_pending() || is_active_download,
                                                        onclick: move |_| pending_discard.set(Some(discard_filename.clone())),
                                                        "Discard"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    section { class: "model-import-panel", aria_label: "Local model import",
                        div { class: "model-download-heading",
                            div {
                                strong { "Import a local model" }
                                p { "Upload a single-file model in bounded chunks with mandatory SHA-256 verification." }
                            }
                            if !snapshot.import.enabled {
                                span { class: "feature-disabled", "Disabled" }
                            }
                        }
                        if snapshot.import.enabled {
                            div { class: "model-import-form",
                                label { class: "model-file-picker",
                                    span { "Local GGUF, ONNX, or Core ML file" }
                                    input {
                                        key: "{import_input_key}",
                                        r#type: "file",
                                        accept: ".gguf,.onnx,.mlmodel",
                                        disabled: import_busy(),
                                        onchange: move |event| {
                                            let Some(file_data) = event.files().into_iter().next() else {
                                                return;
                                            };
                                            let filename = file_data.name();
                                            if !supported_model_import_filename(&filename) {
                                                operation_error.set(Some(
                                                    "Choose a non-hidden GGUF, ONNX, or Core ML file.".to_string()
                                                ));
                                                selected_import.set(None);
                                                import_input_key += 1;
                                                return;
                                            }
                                            let Some(file) = file_data
                                                .inner()
                                                .downcast_ref::<web_sys::File>()
                                                .cloned()
                                            else {
                                                operation_error.set(Some(
                                                    "The browser did not provide access to the selected file.".to_string()
                                                ));
                                                return;
                                            };
                                            let size_bytes = file_data.size();
                                            if size_bytes == 0 || size_bytes > snapshot.import.max_bytes {
                                                operation_error.set(Some(format!(
                                                    "Model file must be between 1 byte and {}.",
                                                    format_bytes(snapshot.import.max_bytes)
                                                )));
                                                selected_import.set(None);
                                                import_input_key += 1;
                                                return;
                                            }
                                            selected_import.set(Some(SelectedModelImport {
                                                file,
                                                filename,
                                                size_bytes,
                                            }));
                                            operation_error.set(None);
                                        },
                                    }
                                }
                                if let Some(selected) = selected_import() {
                                    div { class: "selected-model-file",
                                        strong { "{selected.filename}" }
                                        span { "{format_bytes(selected.size_bytes)}" }
                                    }
                                }
                                label {
                                    span { "SHA-256 checksum" }
                                    input {
                                        value: "{import_sha256}",
                                        placeholder: "64 hexadecimal characters",
                                        maxlength: 64,
                                        spellcheck: false,
                                        disabled: import_busy(),
                                        oninput: move |event| import_sha256.set(event.value()),
                                    }
                                }
                                label {
                                    span { "Source HTTPS URL (optional)" }
                                    input {
                                        r#type: "url",
                                        value: "{import_source_url}",
                                        placeholder: "https://huggingface.co/organization/model",
                                        maxlength: 2048,
                                        disabled: import_busy(),
                                        oninput: move |event| import_source_url.set(event.value()),
                                    }
                                }
                                if snapshot.import.license_policy.enforced {
                                    label {
                                        span { "Approved model license" }
                                        select {
                                            value: "{import_license}",
                                            disabled: import_busy(),
                                            onchange: move |event| import_license.set(event.value()),
                                            option { value: "", "Select an approved license" }
                                            for license in snapshot.import.license_policy.allowed.iter().take(64) {
                                                option { value: "{license}", "{license}" }
                                            }
                                        }
                                        small { class: "license-policy-help", "Required by the server acquisition policy." }
                                    }
                                } else {
                                    label {
                                        span { "License or SPDX expression (optional)" }
                                        input {
                                            value: "{import_license}",
                                            placeholder: "Apache-2.0",
                                            maxlength: 128,
                                            disabled: import_busy(),
                                            oninput: move |event| import_license.set(event.value()),
                                        }
                                    }
                                }
                                if import_busy() {
                                    button {
                                        class: "btn-ghost compact",
                                        onclick: cancel_import,
                                        "Cancel import"
                                    }
                                } else {
                                    button {
                                        class: "btn-primary compact",
                                        disabled: operation_pending()
                                            || selected_import().is_none()
                                            || !valid_sha256_input(import_sha256().trim())
                                            || !license_policy_allows(&snapshot.import.license_policy, import_license().trim()),
                                        onclick: start_import,
                                        "Import or resume"
                                    }
                                }
                            }
                        } else {
                            p { class: "model-download-disabled",
                                "Start bloom_server with --enable-model-imports to allow authenticated local-file imports."
                            }
                        }

                        if snapshot.import.status.phase != "idle" || import_busy() {
                            div { class: "download-status-card",
                                div { class: "download-status-row",
                                    div {
                                        span { class: "model-card-eyebrow", "Import status" }
                                        strong { {model_import_phase_label(&snapshot.import.status.phase, import_busy())} }
                                        if let Some(filename) = snapshot.import.status.filename.as_ref() {
                                            span { class: "model-card-meta", "{filename}" }
                                        }
                                    }
                                }
                                if let Some(total) = snapshot.import.status.total_bytes {
                                    {
                                        let uploaded = if import_busy() {
                                            import_uploaded_bytes()
                                        } else {
                                            snapshot.import.status.uploaded_bytes
                                        };
                                        rsx! {
                                            progress {
                                                max: "{total.max(1)}",
                                                value: "{uploaded.min(total)}",
                                                aria_label: "Model import progress",
                                            }
                                            span { class: "model-card-meta",
                                                "{format_download_progress(uploaded, Some(total))}"
                                            }
                                        }
                                    }
                                }
                                if let Some(message) = snapshot.import.status.error.as_ref() {
                                    div { class: "download-error", role: "alert", "{message}" }
                                }
                            }
                        }
                        if !snapshot.import.staged.is_empty() {
                            div { class: "staged-downloads",
                                span { class: "model-card-eyebrow", "Staged local imports" }
                                for staged in snapshot.import.staged.iter() {
                                    {
                                        let filename = staged.filename.clone();
                                        rsx! {
                                            div { class: "staged-download-row", key: "{staged.filename}",
                                                div {
                                                    strong { "{staged.filename}" }
                                                    span {
                                                        {format_download_progress(
                                                            staged.uploaded_bytes,
                                                            Some(staged.total_bytes)
                                                        )}
                                                    }
                                                }
                                                button {
                                                    class: "btn-danger compact",
                                                    disabled: operation_pending(),
                                                    onclick: move |_| pending_import_discard.set(Some(filename.clone())),
                                                    "Discard"
                                                }
                                            }
                                        }
                                    }
                                }
                                p { class: "staged-import-help",
                                    "Select the same local file and checksum to resume from the stored offset."
                                }
                            }
                        }
                    }
                    if !snapshot.root_exists {
                        div { class: "catalog-empty",
                            strong { "Catalog directory not found" }
                            p { "Create this directory or start bloom_server with --models-dir PATH." }
                        }
                    } else if snapshot.data.is_empty() {
                        div { class: "catalog-empty",
                            strong { "No recognized models found" }
                            p { "Add a GGUF, ONNX, or Core ML file, or a recognized model directory." }
                        }
                    } else {
                        div { class: "model-list", aria_label: "Available local models",
                            for model in snapshot.data.iter() {
                                {
                                    let model_id = model.id.clone();
                                    let remove_id = model.id.clone();
                                    let verify_id = model.id.clone();
                                    let details_id = model.id.clone();
                                    let provenance_summary = model
                                        .provenance
                                        .as_ref()
                                        .map(model_provenance_summary);
                                    let provenance_source = model.provenance.as_ref().and_then(|value| {
                                        value.source_url.as_ref().map(|url| {
                                            (url.clone(), value.source_host.clone().unwrap_or_else(|| "Source".to_string()))
                                        })
                                    });
                                    let provenance_host = model.provenance.as_ref().and_then(|value| {
                                        value.source_url.is_none().then(|| value.source_host.clone()).flatten()
                                    });
                                    let provenance_error = model.provenance_error.clone();
                                    let recorded_integrity_mismatch = model
                                        .provenance
                                        .as_ref()
                                        .is_some_and(|value| value.integrity_mismatch_at.is_some());
                                    let provenance_missing = model.kind == "file"
                                        && model.provenance.is_none()
                                        && provenance_error.is_none();
                                    let integrity_busy = matches!(
                                        snapshot.integrity.phase.as_str(),
                                        "queued" | "verifying"
                                    );
                                    let current_integrity_mismatch = snapshot.integrity.model_id.as_deref()
                                        == Some(model.id.as_str())
                                        && snapshot.integrity.matches_expected == Some(false);
                                    let integrity_mismatch = current_integrity_mismatch
                                        || recorded_integrity_mismatch;
                                    let inspected = preflight()
                                        .filter(|report| report.model_id == model.id);
                                    let details_pending = preflight_pending().as_deref()
                                        == Some(model.id.as_str());
                                    let preflight_blocker = inspected
                                        .as_ref()
                                        .filter(|report| !report.loadable)
                                        .and_then(|report| report.load_blocker.clone());
                                    let load_title = if integrity_mismatch {
                                        "Loading is blocked because the current file does not match its acquisition checksum".to_string()
                                    } else if inspected.is_none() {
                                        "Review model tasks, compatibility, and memory before loading".to_string()
                                    } else if let Some(message) = preflight_blocker.as_ref() {
                                        format!("Loading is blocked by preflight: {message}")
                                    } else {
                                        "Load model".to_string()
                                    };
                                    rsx! {
                                        article { class: if model.active { "model-card active" } else { "model-card" }, key: "{model.id}",
                                            div { class: "model-card-main",
                                                strong { "{model.name}" }
                                                span { class: "model-card-meta",
                                                    "{model.format.to_uppercase()} · {model.kind} · {format_model_size(model.size_bytes, model.size_complete)}"
                                                }
                                                if let Some(summary) = provenance_summary.as_ref() {
                                                    div { class: "model-provenance",
                                                        span { "{summary}" }
                                                        if let Some((source_url, source_host)) = provenance_source.as_ref() {
                                                            a {
                                                                href: "{source_url}",
                                                                target: "_blank",
                                                                rel: "noopener noreferrer",
                                                                title: "{source_url}",
                                                                "Source: {source_host}"
                                                            }
                                                        } else if let Some(source_host) = provenance_host.as_ref() {
                                                            span { "Source: {source_host}" }
                                                        }
                                                    }
                                                }
                                                if let Some(message) = provenance_error.as_ref() {
                                                    span { class: "model-provenance-warning", "{message}" }
                                                } else if recorded_integrity_mismatch {
                                                    span { class: "model-provenance-warning",
                                                        "A checksum mismatch is recorded; loading is blocked until verification succeeds."
                                                    }
                                                } else if provenance_missing {
                                                    span { class: "model-provenance-missing", "Provenance not recorded" }
                                                }
                                            }
                                            div { class: "model-card-actions",
                                                button {
                                                    class: "btn-ghost compact",
                                                    disabled: preflight_pending().is_some(),
                                                    title: "Review model tasks, architecture, runtime compatibility, and memory requirements",
                                                    onclick: move |_| inspect_model(details_id.clone()),
                                                    if details_pending { "Inspecting…" } else if inspected.is_some() { "Refresh" } else { "Review" }
                                                }
                                                if model.provenance.is_some() {
                                                    button {
                                                        class: "btn-ghost compact",
                                                        disabled: operation_pending()
                                                            || integrity_busy
                                                            || model.active
                                                            || snapshot.load.phase == "loading",
                                                        title: if model.active {
                                                            "Unload or switch away before verifying on-disk integrity"
                                                        } else {
                                                            "Recompute SHA-256 and compare it with the acquisition record"
                                                        },
                                                        onclick: move |_| verify_model(verify_id.clone()),
                                                        "Verify"
                                                    }
                                                }
                                                if model.active {
                                                    span { class: "active-badge", "Active" }
                                                } else {
                                                    button {
                                                        class: "btn-primary compact",
                                                        disabled: operation_pending()
                                                            || snapshot.load.phase == "loading"
                                                            || integrity_busy
                                                            || integrity_mismatch
                                                            || inspected.is_none()
                                                            || preflight_blocker.is_some(),
                                                        title: "{load_title}",
                                                        onclick: move |_| load_model(model_id.clone()),
                                                        "Load"
                                                    }
                                                    button {
                                                        class: "btn-danger compact",
                                                        disabled: operation_pending() || snapshot.load.phase == "loading",
                                                        onclick: move |_| pending_remove.set(Some(remove_id.clone())),
                                                        "Remove"
                                                    }
                                                }
                                            }
                                            if details_pending {
                                                section { class: "model-preflight-card pending", aria_live: "polite",
                                                    "Inspecting model metadata and runtime compatibility…"
                                                }
                                            } else if let Some(report) = inspected.as_ref() {
                                                section {
                                                    class: if report.loadable { "model-preflight-card loadable" } else { "model-preflight-card blocked" },
                                                    aria_label: "Model preflight details",
                                                    div { class: "model-preflight-heading",
                                                        div {
                                                            span { class: "model-card-eyebrow", "Load preflight" }
                                                            strong { "{report.manifest.id}" }
                                                            span { class: "model-card-meta", "Tasks · {format_model_tasks(&report.manifest.model_tasks)}" }
                                                        }
                                                        span {
                                                            class: if report.loadable { "preflight-badge ready" } else { "preflight-badge blocked" },
                                                            if report.loadable { "Loadable" } else { "Blocked" }
                                                        }
                                                    }
                                                    if let Some(message) = report.load_blocker.as_ref() {
                                                        div { class: "preflight-blocker", role: "alert", "{message}" }
                                                    }
                                                    if let Some(description) = report.manifest.description.as_ref() {
                                                        p { class: "preflight-description", "{description}" }
                                                    }
                                                    dl { class: "preflight-grid",
                                                        div { dt { "Tasks" } dd { {format_model_tasks(&report.manifest.model_tasks)} } }
                                                        div { dt { "Family" } dd { "{report.manifest.family}" } }
                                                        div { dt { "Precision" } dd { {format_model_precision(&report.manifest)} } }
                                                        div { dt { "Format" } dd { {format_detail_values(&report.manifest.formats)} } }
                                                        div { dt { "Context" } dd { {format_optional_count(report.manifest.context_length, "tokens")} } }
                                                        div { dt { "Parameters" } dd { {format_parameter_count(report.manifest.parameter_count)} } }
                                                        div { dt { "Layers" } dd { {format_optional_count(report.manifest.num_layers, "")} } }
                                                        div { dt { "Inputs" } dd { {format_detail_values(&report.manifest.input_modalities)} } }
                                                        div { dt { "Outputs" } dd { {format_detail_values(&report.manifest.output_modalities)} } }
                                                        div { dt { "Engine" } dd { "{report.runtime.selected_engine} · {report.runtime.engine_maturity}" } }
                                                        div { dt { "Runtime support" } dd { "{report.runtime.support}" } }
                                                        div { dt { "Device" } dd { "{report.runtime.device} · {report.runtime.device_backend}" } }
                                                        div { dt { "Memory estimate" } dd { "{format_bytes(report.memory.total_bytes)}" } }
                                                        div { dt { "Weights" } dd { "{format_bytes(report.memory.weight_bytes)}" } }
                                                        div { dt { "KV cache" } dd { "{format_bytes(report.memory.kv_cache_bytes)}" } }
                                                        div { dt { "Workspace" } dd { "{format_bytes(report.memory.temp_tensor_bytes)}" } }
                                                        div { dt { "Memory budget" } dd {
                                                            if let Some(budget) = report.memory.budget_bytes {
                                                                "{format_bytes(budget)}"
                                                            } else {
                                                                "Advisory only"
                                                            }
                                                        } }
                                                        div { dt { "Planned context" } dd { "{report.memory.planned_context_tokens} tokens" } }
                                                    }
                                                    if !report.warnings.is_empty() {
                                                        ul { class: "preflight-warnings",
                                                            for warning in report.warnings.iter() {
                                                                li { "{warning}" }
                                                            }
                                                        }
                                                    }
                                                    if !report.runtime.diagnostic_tips.is_empty() {
                                                        details { class: "preflight-tips",
                                                            summary { "Runtime guidance" }
                                                            ul {
                                                                for tip in report.runtime.diagnostic_tips.iter() {
                                                                    li { "{tip}" }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    p { class: "preflight-scope", "Estimate scope: {report.memory.scope}" }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if catalog_error().is_none() {
                    div { class: "catalog-empty", "Loading model catalog…" }
                }
            }
        }

        if pending_unload() {
            ConfirmDialog {
                dialog_id: "unload-model-confirm-dialog",
                title: "Unload active model?",
                body: "Bloom will stop accepting inference until another model is loaded.",
                confirm_label: "Unload",
                on_cancel: move |_| pending_unload.set(false),
                on_confirm: unload_model,
            }
        }
        if pending_remove().is_some() {
            ConfirmDialog {
                dialog_id: "remove-model-confirm-dialog",
                title: "Permanently remove model?",
                body: "This deletes the selected catalog entry and cannot be undone. Active models must be unloaded first.",
                confirm_label: "Remove",
                on_cancel: move |_| pending_remove.set(None),
                on_confirm: remove_model,
            }
        }
        if pending_discard().is_some() {
            ConfirmDialog {
                dialog_id: "discard-download-confirm-dialog",
                title: "Discard partial download?",
                body: "This permanently deletes the staged bytes and resume metadata.",
                confirm_label: "Discard",
                on_cancel: move |_| pending_discard.set(None),
                on_confirm: discard_download,
            }
        }
        if pending_import_discard().is_some() {
            ConfirmDialog {
                dialog_id: "discard-import-confirm-dialog",
                title: "Discard staged import?",
                body: "This permanently deletes the uploaded chunks and resume metadata.",
                confirm_label: "Discard",
                on_cancel: move |_| pending_import_discard.set(None),
                on_confirm: discard_import,
            }
        }
        if pending_inventory_restore().is_some() {
            ConfirmDialog {
                dialog_id: "restore-inventory-model-confirm-dialog",
                title: "Restore missing model?",
                body: "Bloom will download the inventory's exact-commit source, verify its SHA-256 checksum, and install it without overwriting an existing model.",
                confirm_label: "Restore",
                on_cancel: move |_| pending_inventory_restore.set(None),
                on_confirm: restore_inventory_model,
            }
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_optional_bytes(bytes: u64) -> String {
    if bytes == 0 {
        "Unavailable".to_string()
    } else {
        format_bytes(bytes)
    }
}

fn format_duration_seconds(seconds: u64) -> String {
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let seconds = seconds % 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

fn format_load_phase(phase: &str) -> &'static str {
    match phase {
        "idle" => "Idle",
        "loading" => "Loading",
        "ready" => "Ready",
        "failed" => "Failed",
        _ => "Unknown",
    }
}

fn format_percent(ratio: f64) -> String {
    format!("{:.1}%", ratio * 100.0)
}

fn format_detail_values(values: &[String]) -> String {
    if values.is_empty() {
        "Not declared".to_string()
    } else {
        values.join(", ")
    }
}

fn format_model_tasks(tasks: &[String]) -> String {
    tasks
        .iter()
        .map(|task| match task.as_str() {
            "generation" => "Generation",
            "embedding" => "Embedding",
            "rerank" => "Rerank",
            _ => "Unknown",
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn format_model_precision(manifest: &api::ModelManifestSummary) -> String {
    match (&manifest.quantization, manifest.quantization_bits) {
        (Some(quantization), Some(bits)) => {
            format!("{} · {}-bit", quantization.to_uppercase(), bits)
        }
        (Some(quantization), None) => quantization.to_uppercase(),
        (None, _) => manifest.primary_dtype.to_uppercase(),
    }
}

fn format_optional_count(value: Option<u64>, unit: &str) -> String {
    value
        .map(|value| {
            if unit.is_empty() {
                value.to_string()
            } else {
                format!("{value} {unit}")
            }
        })
        .unwrap_or_else(|| "Unknown".to_string())
}

fn format_parameter_count(value: Option<u64>) -> String {
    let Some(value) = value else {
        return "Unknown".to_string();
    };
    if value >= 1_000_000_000 {
        format!("{:.2}B", value as f64 / 1_000_000_000.0)
    } else if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn format_model_size(bytes: u64, complete: bool) -> String {
    let size = format_bytes(bytes);
    if complete {
        size
    } else {
        format!("at least {size}")
    }
}

fn model_index_entry_matches(entry: &api::ModelIndexEntry, query: &str) -> bool {
    let query = query.trim().to_ascii_lowercase();
    query.is_empty()
        || entry.name.to_ascii_lowercase().contains(&query)
        || entry.id.to_ascii_lowercase().contains(&query)
        || entry.description.to_ascii_lowercase().contains(&query)
        || entry
            .family
            .as_ref()
            .is_some_and(|value| value.to_ascii_lowercase().contains(&query))
        || entry
            .quantization
            .as_ref()
            .is_some_and(|value| value.to_ascii_lowercase().contains(&query))
        || entry.tags.iter().any(|tag| tag.contains(&query))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelIndexLocalState {
    Missing,
    Verified,
    Upgradable,
    Conflict,
}

fn model_index_local_state(
    entry: &api::ModelIndexEntry,
    catalog: &[api::ModelCatalogEntry],
) -> ModelIndexLocalState {
    let aliases = catalog
        .iter()
        .filter(|model| {
            model
                .provenance
                .as_ref()
                .and_then(|provenance| provenance.model_index_id.as_deref())
                == Some(entry.id.as_str())
        })
        .collect::<Vec<_>>();
    if aliases.len() > 1 {
        return ModelIndexLocalState::Conflict;
    }
    let destination = catalog.iter().find(|model| model.id == entry.filename);
    let is_package = !entry.files.is_empty();
    let expected_kind = if is_package { "directory" } else { "file" };
    let expected_file_count = is_package.then_some(entry.files.len());
    if destination.is_some_and(|installed| {
        let format_matches = installed.format == entry.format
            || (installed.format == "coreml" && entry.format == "mlmodel");
        installed.kind == expected_kind
            && format_matches
            && installed.size_complete
            && installed.size_bytes == entry.size_bytes
            && installed.provenance_error.is_none()
            && installed.provenance.as_ref().is_some_and(|provenance| {
                provenance.acquisition == "download"
                    && provenance.model_index_id.as_deref() == Some(entry.id.as_str())
                    && provenance.sha256.eq_ignore_ascii_case(&entry.sha256)
                    && provenance.file_count == expected_file_count
                    && provenance
                        .license
                        .as_deref()
                        .is_some_and(|license| license.eq_ignore_ascii_case(&entry.license))
                    && provenance.integrity_mismatch_at.is_none()
            })
    }) {
        return ModelIndexLocalState::Verified;
    }
    let Some(previous) = aliases.first().copied() else {
        return if destination.is_some() {
            ModelIndexLocalState::Conflict
        } else {
            ModelIndexLocalState::Missing
        };
    };
    if destination.is_some_and(|installed| installed.id != previous.id)
        || !previous.size_complete
        || previous.size_bytes == 0
        || previous.provenance_error.is_some()
        || !previous.provenance.as_ref().is_some_and(|provenance| {
            provenance.acquisition == "download"
                && provenance.model_index_id.as_deref() == Some(entry.id.as_str())
                && provenance.integrity_mismatch_at.is_none()
        })
    {
        ModelIndexLocalState::Conflict
    } else {
        ModelIndexLocalState::Upgradable
    }
}

fn model_index_upgrade_source_is_active(
    entry: &api::ModelIndexEntry,
    catalog: &[api::ModelCatalogEntry],
) -> bool {
    model_index_local_state(entry, catalog) == ModelIndexLocalState::Upgradable
        && catalog.iter().any(|model| {
            model.active
                && model
                    .provenance
                    .as_ref()
                    .and_then(|provenance| provenance.model_index_id.as_deref())
                    == Some(entry.id.as_str())
        })
}

fn format_model_index_blockers(reasons: &[String]) -> String {
    reasons
        .iter()
        .map(|reason| match reason.as_str() {
            "size_limit" => "server size limit",
            "license_policy" => "license policy",
            _ => "server policy",
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn model_index_poll_interval_seconds(refresh_seconds: u64, has_snapshot: bool) -> u64 {
    let refresh_seconds = refresh_seconds.clamp(5, 3_600);
    if has_snapshot {
        refresh_seconds
    } else {
        refresh_seconds.min(30)
    }
}

fn model_provenance_summary(provenance: &api::ModelProvenance) -> String {
    let acquisition = match provenance.acquisition.as_str() {
        "download" => "Verified download",
        "import" => "Verified import",
        _ => "Verified acquisition",
    };
    let license = provenance.license.as_deref().unwrap_or("Not reported");
    let digest = provenance
        .sha256
        .get(..12)
        .unwrap_or(provenance.sha256.as_str());
    let digest_suffix = if provenance.sha256.len() > digest.len() {
        "…"
    } else {
        ""
    };
    let integrity = if provenance.integrity_mismatch_at.is_some() {
        " · Integrity mismatch"
    } else if provenance.last_verified_at.is_some() {
        " · Reverified"
    } else {
        ""
    };
    format!("{acquisition} · License: {license} · SHA-256: {digest}{digest_suffix}{integrity}")
}

fn format_duration(seconds: u64) -> String {
    if seconds.is_multiple_of(86_400) {
        format_duration_unit(seconds / 86_400, "day")
    } else if seconds.is_multiple_of(3_600) {
        format_duration_unit(seconds / 3_600, "hour")
    } else if seconds.is_multiple_of(60) {
        format_duration_unit(seconds / 60, "minute")
    } else {
        format_duration_unit(seconds, "second")
    }
}

fn format_duration_unit(value: u64, unit: &str) -> String {
    if value == 1 {
        format!("1 {unit}")
    } else {
        format!("{value} {unit}s")
    }
}

fn format_generation_millis(milliseconds: u64) -> String {
    if milliseconds < 1_000 {
        format!("{milliseconds} ms")
    } else if milliseconds < 60_000 {
        format!("{:.2} s", milliseconds as f64 / 1_000.0)
    } else {
        let minutes = milliseconds / 60_000;
        let seconds = milliseconds % 60_000 / 1_000;
        format!("{minutes}m {seconds}s")
    }
}

fn generation_outcome_label(outcome: GenerationOutcome) -> Option<&'static str> {
    match outcome {
        GenerationOutcome::Completed => None,
        GenerationOutcome::Stopped => Some("Stopped"),
        GenerationOutcome::Failed => Some("Failed"),
    }
}

fn validate_context_reservation(
    max_tokens: usize,
    context_window: Option<u64>,
) -> Result<(), String> {
    let Some(context_window) = context_window.filter(|window| *window > 0) else {
        return Ok(());
    };
    let max_tokens = u64::try_from(max_tokens).unwrap_or(u64::MAX);
    if max_tokens < context_window {
        Ok(())
    } else {
        Err(format!(
            "Maximum generated tokens must be lower than the active model context window of {context_window} tokens."
        ))
    }
}

fn conversation_context_status(
    messages: &[DisplayMessage],
    context_window: Option<u64>,
    max_tokens: usize,
) -> Option<(bool, String)> {
    let context_window = context_window.filter(|window| *window > 0)?;
    let used_tokens = messages
        .iter()
        .rev()
        .filter_map(|message| message.generation.as_ref())
        .find_map(GenerationStats::total_tokens)?;
    let max_tokens = u64::try_from(max_tokens).unwrap_or(u64::MAX);
    let warning = used_tokens.saturating_add(max_tokens) >= context_window
        || used_tokens >= context_window.saturating_sub(context_window / 5);
    let message = if warning {
        format!(
            "Previous turn used {used_tokens} / {context_window} context tokens. With up to {max_tokens} output tokens, the next request may exceed this model's context window. Use Continue on a later user message to carry only that recent history into a new chat."
        )
    } else {
        format!("Previous turn used {used_tokens} / {context_window} context tokens.")
    };
    Some((warning, message))
}

fn download_phase_label(phase: &str) -> &'static str {
    match phase {
        "queued" => "Queued",
        "downloading" => "Downloading",
        "verifying" => "Verifying checksum",
        "complete" => "Download complete",
        "cancelled" => "Download cancelled",
        "error" => "Download failed",
        _ => "Idle",
    }
}

fn model_import_phase_label(phase: &str, browser_uploading: bool) -> &'static str {
    if browser_uploading {
        return "Uploading chunks";
    }
    match phase {
        "ready" => "Ready to resume",
        "uploading" => "Upload paused",
        "verifying" => "Verifying checksum",
        "complete" => "Import complete",
        "error" => "Import failed",
        _ => "Idle",
    }
}

fn integrity_phase_label(status: &api::ModelIntegrityStatus) -> &'static str {
    match status.phase.as_str() {
        "queued" => "Queued",
        "verifying" => "Computing SHA-256",
        "complete" if status.matches_expected == Some(true) => "Checksum verified",
        "complete" if status.matches_expected == Some(false) => "Checksum mismatch",
        "complete" => "Verification complete",
        "cancelled" => "Verification cancelled",
        "error" => "Verification failed",
        _ => "Idle",
    }
}

fn supported_model_import_filename(name: &str) -> bool {
    if name.starts_with('.') || name.contains(['/', '\\']) {
        return false;
    }
    matches!(
        name.rsplit_once('.')
            .map(|(_, extension)| extension.to_ascii_lowercase())
            .as_deref(),
        Some("gguf" | "onnx" | "mlmodel")
    )
}

fn valid_sha256_input(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn license_policy_allows(policy: &api::ModelLicensePolicy, value: &str) -> bool {
    !policy.enforced
        || policy
            .allowed
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(value.trim()))
}

fn format_download_progress(downloaded: u64, total: Option<u64>) -> String {
    match total {
        Some(total) if total > 0 => format!(
            "{} of {} ({:.0}%)",
            format_bytes(downloaded),
            format_bytes(total),
            (downloaded as f64 / total as f64 * 100.0).clamp(0.0, 100.0)
        ),
        _ => format!("{} downloaded", format_bytes(downloaded)),
    }
}

fn format_inventory_drift_status(status: &str) -> &'static str {
    match status {
        "missing" => "Missing",
        "unexpected" => "Unexpected",
        "changed" => "Changed",
        _ => "Unknown",
    }
}

fn format_inventory_drift_severity(severity: &str) -> &'static str {
    match severity {
        "blocking" => "Blocking",
        "warning" => "Warning",
        _ => "Unknown",
    }
}

fn format_inventory_changes(changes: &[String]) -> String {
    changes
        .iter()
        .map(|change| match change.as_str() {
            "model_missing" => "Model is missing".to_string(),
            "model_unexpected" => "Model is not in the expected inventory".to_string(),
            "kind" => "Model kind".to_string(),
            "format" => "Format".to_string(),
            "size_bytes" => "Size".to_string(),
            "size_complete" => "Size accounting".to_string(),
            "provenance_status" => "Provenance".to_string(),
            "acquisition" => "Acquisition method".to_string(),
            "source_url" => "Source URL".to_string(),
            "source_host" => "Source host".to_string(),
            "source_revision" => "Source revision".to_string(),
            "source_lock" => "Source lock".to_string(),
            "sha256" => "SHA-256".to_string(),
            "license" => "License".to_string(),
            "installed_at" => "Installation time".to_string(),
            "last_verified_at" => "Verification time".to_string(),
            "integrity" => "Integrity status".to_string(),
            value => value.replace('_', " "),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn supported_image_mime(name: &str, reported_mime: &str) -> Option<String> {
    let reported_mime = reported_mime.to_ascii_lowercase();
    if matches!(reported_mime.as_str(), "image/jpeg" | "image/png") {
        return Some(reported_mime);
    }
    if !reported_mime.is_empty() && reported_mime != "application/octet-stream" {
        return None;
    }
    match name
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
    {
        Some(extension) if matches!(extension.as_str(), "jpg" | "jpeg") => {
            Some("image/jpeg".to_string())
        }
        Some(extension) if extension == "png" => Some("image/png".to_string()),
        _ => None,
    }
}

fn append_transcript_segment(transcript: &mut String, segment: &str) -> Result<(), String> {
    let segment = segment.trim();
    if segment.is_empty() {
        return Ok(());
    }
    let separator = if transcript.is_empty()
        || transcript.ends_with(char::is_whitespace)
        || segment.starts_with(char::is_whitespace)
    {
        ""
    } else {
        let previous = transcript.chars().next_back();
        let next = segment.chars().next();
        if previous.is_some_and(|character| character.is_ascii())
            && next.is_some_and(|character| character.is_ascii_alphanumeric())
        {
            " "
        } else {
            ""
        }
    };
    let next_bytes = transcript
        .len()
        .checked_add(separator.len())
        .and_then(|size| size.checked_add(segment.len()))
        .ok_or_else(|| "Transcript size overflowed.".to_string())?;
    if next_bytes > MAX_TRANSCRIPT_BYTES {
        return Err(format!(
            "Transcript reached the {} byte limit. Copy it, clear the workspace, and continue.",
            MAX_TRANSCRIPT_BYTES
        ));
    }
    transcript.push_str(separator);
    transcript.push_str(segment);
    Ok(())
}

fn format_speech_duration(samples: u64) -> String {
    let seconds = samples as f64 / api::SPEECH_SAMPLE_RATE as f64;
    format!("{seconds:.1}s")
}

#[component]
fn SpeechWorkspace(config: Signal<ConnConfig>, model: String) -> Element {
    let mut capture = use_signal(|| Option::<browser::MicrophoneCapture>::None);
    let mut starting = use_signal(|| false);
    let mut recording = use_signal(|| false);
    let mut processing = use_signal(|| false);
    let mut transcript = use_signal(String::new);
    let mut live_partial = use_signal(String::new);
    let mut workspace_error = use_signal(|| Option::<String>::None);
    let mut workspace_notice = use_signal(|| Option::<String>::None);
    let mut recorded_samples = use_signal(|| 0_u64);
    let mut completed_segments = use_signal(|| 0_u64);
    let mut copied = use_signal(|| false);

    use_drop(move || {
        recording.set(false);
        capture.with_mut(|current| {
            if let Some(capture) = current.as_mut() {
                capture.stop_input();
            }
            current.take();
        });
    });

    let start_model = model.clone();
    let start_recording = move |_| {
        if starting() || recording() || processing() {
            return;
        }
        starting.set(true);
        workspace_error.set(None);
        workspace_notice.set(Some("Requesting microphone permission…".to_string()));
        live_partial.set(String::new());
        let model = start_model.clone();
        let cfg = config();
        spawn(async move {
            let microphone = match browser::MicrophoneCapture::start().await {
                Ok(microphone) => microphone,
                Err(message) => {
                    starting.set(false);
                    workspace_notice.set(None);
                    workspace_error.set(Some(message));
                    return;
                }
            };
            capture.set(Some(microphone));
            starting.set(false);
            recording.set(true);
            workspace_notice.set(Some(
                "Listening · speech is transcribed in short live windows.".to_string(),
            ));

            let mut final_drain_started = false;
            loop {
                if recording() {
                    TimeoutFuture::new(LIVE_SPEECH_WINDOW_MS).await;
                }
                let samples = capture
                    .with_mut(|current| {
                        current
                            .as_mut()
                            .map(|capture| capture.take_resampled(api::SPEECH_SAMPLE_RATE))
                    })
                    .unwrap_or_default();
                recorded_samples += samples.len() as u64;

                if samples.len() >= MIN_SPEECH_WINDOW_SAMPLES {
                    for window in samples.chunks(api::MAX_SPEECH_SEGMENT_SAMPLES) {
                        processing.set(true);
                        live_partial.set(String::new());
                        workspace_notice.set(Some(if recording() {
                            "Listening · transcribing the latest speech…".to_string()
                        } else {
                            "Finishing the last speech window…".to_string()
                        }));
                        let cancellation = match ChatCancellation::new() {
                            Ok(cancellation) => cancellation,
                            Err(message) => {
                                workspace_error.set(Some(message));
                                recording.set(false);
                                break;
                            }
                        };
                        let result = api::speech_to_text_stream(
                            &cfg,
                            &model,
                            window,
                            api::SPEECH_SAMPLE_RATE,
                            &cancellation,
                            |update| {
                                if let StreamUpdate::TextDelta(delta) = update {
                                    live_partial.with_mut(|partial| {
                                        if partial.len().saturating_add(delta.len())
                                            <= MAX_TRANSCRIPT_BYTES
                                        {
                                            partial.push_str(&delta);
                                        }
                                    });
                                }
                            },
                        )
                        .await;
                        processing.set(false);
                        match result {
                            Ok(segment) => {
                                live_partial.set(String::new());
                                if !segment.trim().is_empty() {
                                    let appended = transcript.with_mut(|value| {
                                        append_transcript_segment(value, &segment)
                                    });
                                    if let Err(message) = appended {
                                        workspace_error.set(Some(message));
                                        recording.set(false);
                                        break;
                                    }
                                    completed_segments += 1;
                                }
                            }
                            Err(ChatStreamError::Cancelled) => {
                                live_partial.set(String::new());
                                recording.set(false);
                                break;
                            }
                            Err(ChatStreamError::Request(message)) => {
                                live_partial.set(String::new());
                                workspace_error.set(Some(message));
                                recording.set(false);
                                capture.with_mut(|current| {
                                    if let Some(capture) = current.as_mut() {
                                        capture.stop_input();
                                    }
                                });
                                break;
                            }
                        }
                    }
                }

                if recording() {
                    final_drain_started = false;
                } else {
                    capture.with_mut(|current| {
                        if let Some(capture) = current.as_mut() {
                            capture.stop_input();
                        }
                    });
                    if final_drain_started {
                        break;
                    }
                    // Run once more without waiting so samples captured while
                    // the previous model request was in flight are not lost.
                    final_drain_started = true;
                }
            }

            capture.with_mut(|current| {
                if let Some(capture) = current.as_mut() {
                    capture.stop_input();
                }
                current.take();
            });
            recording.set(false);
            processing.set(false);
            live_partial.set(String::new());
            if workspace_error().is_none() {
                workspace_notice.set(Some(if transcript().trim().is_empty() {
                    "Recording stopped · no speech was recognized.".to_string()
                } else {
                    "Recording stopped · transcript is ready.".to_string()
                }));
            }
        });
    };

    let stop_recording = move |_| {
        recording.set(false);
        capture.with_mut(|current| {
            if let Some(capture) = current.as_mut() {
                capture.stop_input();
            }
        });
        workspace_notice.set(Some("Stopping · finishing captured speech…".to_string()));
    };

    let copy_transcript = move |_| {
        let text = transcript();
        if text.trim().is_empty() {
            return;
        }
        workspace_error.set(None);
        spawn(async move {
            match browser::copy_text_to_clipboard(&text).await {
                Ok(()) => {
                    copied.set(true);
                    TimeoutFuture::new(1_200).await;
                    copied.set(false);
                }
                Err(message) => workspace_error.set(Some(message)),
            }
        });
    };

    rsx! {
        section { class: "embedding-workspace speech-workspace", aria_labelledby: "speech-workspace-title",
            div { class: "embedding-workspace-header",
                div {
                    span { class: "workspace-eyebrow", "Speech workspace" }
                    h2 { id: "speech-workspace-title", "Live speech to text" }
                    p { "Speak into your microphone and Bloom will append each model result while recording continues." }
                }
                div { class: "workspace-model-meta",
                    span { "Active model" }
                    strong { "{model}" }
                    span { "16 kHz mono PCM · {format_speech_duration(recorded_samples())}" }
                }
            }

            if let Some(message) = workspace_error() {
                div { class: "test-result err workspace-error", role: "alert", "{message}" }
            }
            if let Some(message) = workspace_notice() {
                div { class: "test-result ok workspace-notice", role: "status", "{message}" }
            }

            div { class: "workspace-panel speech-panel",
                div { class: "speech-recorder",
                    div {
                        div {
                            class: if recording() { "speech-orb active" } else { "speech-orb" },
                            aria_hidden: "true",
                            span { "●" }
                        }
                        div { class: "speech-state-copy",
                            strong {
                                if starting() {
                                    "Opening microphone"
                                } else if recording() && processing() {
                                    "Listening and transcribing"
                                } else if recording() {
                                    "Listening"
                                } else if processing() {
                                    "Finishing transcription"
                                } else {
                                    "Ready to listen"
                                }
                            }
                            span { "{completed_segments()} model result(s) · audio stays in this browser until each window is sent to Bloom." }
                        }
                    }
                    if recording() {
                        button { class: "speech-stop", onclick: stop_recording, "Stop recording" }
                    } else {
                        button {
                            class: "btn-primary speech-start",
                            disabled: starting() || processing(),
                            onclick: start_recording,
                            if starting() { "Allow microphone…" } else { "Start speaking" }
                        }
                    }
                }

                div { class: "speech-transcript-heading",
                    div {
                        strong { "Transcript" }
                        span { "Updates after every short speech window" }
                    }
                    div { class: "workspace-item-actions",
                        button {
                            class: "workspace-copy",
                            disabled: transcript().trim().is_empty(),
                            onclick: copy_transcript,
                            if copied() { "Copied" } else { "Copy" }
                        }
                        button {
                            class: "workspace-copy",
                            disabled: recording() || processing() || transcript().is_empty(),
                            onclick: move |_| {
                                transcript.set(String::new());
                                live_partial.set(String::new());
                                completed_segments.set(0);
                                recorded_samples.set(0);
                                workspace_error.set(None);
                                workspace_notice.set(None);
                            },
                            "Clear"
                        }
                    }
                }
                div {
                    class: "speech-transcript",
                    role: "log",
                    aria_live: "polite",
                    aria_label: "Live transcript",
                    if transcript().is_empty() && live_partial().is_empty() {
                        p { "Your transcript will appear here as you speak." }
                    } else {
                        span { "{transcript}" }
                        if !live_partial().is_empty() {
                            span { class: "speech-partial", "{live_partial}" }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingWorkspaceMode {
    Embeddings,
    Rerank,
}

fn parse_optional_embedding_dimensions(value: &str) -> Result<Option<usize>, String> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    let dimensions = value.parse::<usize>().map_err(|_| {
        format!("Dimensions must be an integer from 1 through {MAX_EMBEDDING_DIMENSIONS}.")
    })?;
    if !(1..=MAX_EMBEDDING_DIMENSIONS).contains(&dimensions) {
        return Err(format!(
            "Dimensions must be an integer from 1 through {MAX_EMBEDDING_DIMENSIONS}."
        ));
    }
    Ok(Some(dimensions))
}

fn embedding_vector_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
}

fn embedding_vector_preview(values: &[f32]) -> String {
    let mut preview = values
        .iter()
        .take(8)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > 8 {
        preview.push_str(", …");
    }
    format!("[{preview}]")
}

fn encoder_input_preview(input: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 160;
    let mut characters = input.chars();
    let mut preview = characters
        .by_ref()
        .take(MAX_PREVIEW_CHARS)
        .collect::<String>();
    if characters.next().is_some() {
        preview.push('…');
    }
    preview
}

#[component]
fn EmbeddingWorkspace(config: Signal<ConnConfig>, model: String, supports_rerank: bool) -> Element {
    let mut mode = use_signal(|| EmbeddingWorkspaceMode::Embeddings);
    let mut embedding_input = use_signal(String::new);
    let mut dimensions_input = use_signal(String::new);
    let mut rerank_query = use_signal(String::new);
    let mut rerank_documents = use_signal(String::new);
    let mut rerank_top_n = use_signal(|| "3".to_string());
    let mut pending = use_signal(|| false);
    let mut workspace_error = use_signal(|| Option::<String>::None);
    let mut workspace_notice = use_signal(|| Option::<String>::None);
    let mut embedding_result = use_signal(|| Option::<EmbeddingBatch>::None);
    let mut rerank_result = use_signal(|| Option::<RerankBatch>::None);
    let mut copied_embedding = use_signal(|| Option::<usize>::None);
    let mut copied_rerank = use_signal(|| Option::<usize>::None);

    let embedding_model = model.clone();
    let run_embeddings = move |_| {
        if pending() {
            return;
        }
        let inputs = match api::parse_embedding_lines(&embedding_input()) {
            Ok(inputs) => inputs,
            Err(message) => {
                workspace_error.set(Some(message));
                return;
            }
        };
        let dimensions = match parse_optional_embedding_dimensions(&dimensions_input()) {
            Ok(dimensions) => dimensions,
            Err(message) => {
                workspace_error.set(Some(message));
                return;
            }
        };
        pending.set(true);
        workspace_error.set(None);
        workspace_notice.set(None);
        embedding_result.set(None);
        let cfg = config();
        let model = embedding_model.clone();
        spawn(async move {
            match api::create_embeddings(&cfg, &model, inputs, dimensions).await {
                Ok(result) => embedding_result.set(Some(result)),
                Err(message) => workspace_error.set(Some(message)),
            }
            pending.set(false);
        });
    };

    let rerank_model = model.clone();
    let run_rerank = move |_| {
        if pending() || !supports_rerank {
            return;
        }
        let top_n = match rerank_top_n().trim().parse::<usize>() {
            Ok(top_n) => top_n,
            Err(_) => {
                workspace_error.set(Some(
                    "Top results must be a positive integer no larger than the document count."
                        .to_string(),
                ));
                return;
            }
        };
        let (query, documents, top_n) =
            match api::prepare_rerank_input(&rerank_query(), &rerank_documents(), top_n) {
                Ok(input) => input,
                Err(message) => {
                    workspace_error.set(Some(message));
                    return;
                }
            };
        pending.set(true);
        workspace_error.set(None);
        workspace_notice.set(None);
        rerank_result.set(None);
        let cfg = config();
        let model = rerank_model.clone();
        spawn(async move {
            match api::rerank_documents(&cfg, &model, query, documents, top_n).await {
                Ok(result) => rerank_result.set(Some(result)),
                Err(message) => workspace_error.set(Some(message)),
            }
            pending.set(false);
        });
    };

    let export_embeddings = move |_| {
        workspace_error.set(None);
        workspace_notice.set(None);
        let result = embedding_result()
            .ok_or_else(|| "No embedding result is available to export.".to_string())
            .and_then(|batch| api::download_embedding_export(&batch));
        match result {
            Ok(()) => workspace_notice.set(Some(
                "Exported the complete indexed embedding result.".to_string(),
            )),
            Err(message) => workspace_error.set(Some(message)),
        }
    };

    let export_rerank = move |_| {
        workspace_error.set(None);
        workspace_notice.set(None);
        let result = rerank_result()
            .ok_or_else(|| "No rerank result is available to export.".to_string())
            .and_then(|batch| api::download_rerank_export(&batch));
        match result {
            Ok(()) => workspace_notice.set(Some(
                "Exported the complete query and ranked results.".to_string(),
            )),
            Err(message) => workspace_error.set(Some(message)),
        }
    };

    let embedding_output = embedding_result();
    let rerank_output = rerank_result();
    rsx! {
        section { class: "embedding-workspace", aria_labelledby: "embedding-workspace-title",
            div { class: "embedding-workspace-header",
                div {
                    span { class: "workspace-eyebrow", "Encoder workspace" }
                    h2 { id: "embedding-workspace-title", "Embeddings and reranking" }
                    p {
                        "Run bounded local vector operations with "
                        strong { "{model}" }
                        ". Input and results stay between this browser and the configured Bloom server."
                    }
                }
                div { class: "workspace-model-meta",
                    span { "Active task" }
                    strong {
                        if supports_rerank { "Embedding + rerank" } else { "Embedding" }
                    }
                }
            }

            div { class: "workspace-tabs", role: "tablist", aria_label: "Encoder operation",
                button {
                    class: if mode() == EmbeddingWorkspaceMode::Embeddings { "workspace-tab active" } else { "workspace-tab" },
                    role: "tab",
                    aria_selected: mode() == EmbeddingWorkspaceMode::Embeddings,
                    onclick: move |_| {
                        mode.set(EmbeddingWorkspaceMode::Embeddings);
                        workspace_error.set(None);
                        workspace_notice.set(None);
                    },
                    "Embeddings"
                }
                button {
                    class: if mode() == EmbeddingWorkspaceMode::Rerank { "workspace-tab active" } else { "workspace-tab" },
                    role: "tab",
                    aria_selected: mode() == EmbeddingWorkspaceMode::Rerank,
                    disabled: !supports_rerank,
                    title: if supports_rerank { "Rank documents by query similarity" } else { "The active model does not advertise reranking" },
                    onclick: move |_| {
                        mode.set(EmbeddingWorkspaceMode::Rerank);
                        workspace_error.set(None);
                        workspace_notice.set(None);
                    },
                    "Rerank"
                }
            }

            if let Some(message) = workspace_error() {
                div { class: "test-result err workspace-error", role: "alert", "{message}" }
            }
            if let Some(message) = workspace_notice() {
                div { class: "test-result ok workspace-notice", role: "status", "{message}" }
            }

            if mode() == EmbeddingWorkspaceMode::Embeddings {
                div { class: "workspace-panel", role: "tabpanel",
                    label { class: "workspace-field",
                        span { "Inputs" }
                        textarea {
                            aria_label: "Embedding inputs",
                            placeholder: "One non-empty input per line…",
                            maxlength: "{MAX_EMBEDDING_CONTENT_BYTES}",
                            rows: 8,
                            value: "{embedding_input}",
                            disabled: pending(),
                            oninput: move |event| embedding_input.set(event.value()),
                        }
                        small {
                            "One line produces one normalized vector · {embedding_input().len()} / {MAX_EMBEDDING_CONTENT_BYTES} bytes"
                        }
                    }
                    div { class: "workspace-inline-controls",
                        label { class: "workspace-field compact-field",
                            span { "Dimensions" }
                            input {
                                r#type: "number",
                                min: "1",
                                max: "{MAX_EMBEDDING_DIMENSIONS}",
                                placeholder: "Native",
                                value: "{dimensions_input}",
                                disabled: pending(),
                                oninput: move |event| dimensions_input.set(event.value()),
                            }
                            small { "Leave blank for the model's native width." }
                        }
                        button {
                            class: "btn-primary workspace-run",
                            disabled: pending() || embedding_input().trim().is_empty(),
                            onclick: run_embeddings,
                            if pending() { "Running…" } else { "Create embeddings" }
                        }
                    }

                    if let Some(batch) = embedding_output.as_ref() {
                        div { class: "workspace-results", aria_live: "polite",
                            div { class: "workspace-result-summary",
                                div {
                                    strong { "{batch.vectors.len()} vector(s)" }
                                    span { "{batch.prompt_tokens} prompt tokens · {batch.model}" }
                                }
                                button {
                                    class: "btn-ghost compact",
                                    onclick: export_embeddings,
                                    "Export JSON"
                                }
                            }
                            for vector in batch.vectors.iter() {
                                {
                                    let display_index = vector.index + 1;
                                    let dimensions = vector.values.len();
                                    let norm = embedding_vector_norm(&vector.values);
                                    let preview = embedding_vector_preview(&vector.values);
                                    let input_preview = encoder_input_preview(&vector.input);
                                    let vector_to_copy = vector.clone();
                                    let vector_index = vector.index;
                                    rsx! { div { class: "vector-card", key: "{vector.index}",
                                        div { class: "vector-card-heading",
                                            strong { "Vector {display_index}" }
                                            div { class: "workspace-item-actions",
                                                span { "{dimensions} dimensions · norm {norm:.6}" }
                                                button {
                                                    class: "workspace-copy",
                                                    aria_label: "Copy vector {display_index}",
                                                    title: "Copy the complete vector as a JSON array",
                                                    onclick: move |_| {
                                                        workspace_error.set(None);
                                                        workspace_notice.set(None);
                                                        match api::embedding_vector_clipboard_text(&vector_to_copy) {
                                                            Ok(text) => {
                                                                spawn(async move {
                                                                    match browser::copy_text_to_clipboard(&text).await {
                                                                        Ok(()) => {
                                                                            copied_embedding.set(Some(vector_index));
                                                                            TimeoutFuture::new(1_500).await;
                                                                            if copied_embedding() == Some(vector_index) {
                                                                                copied_embedding.set(None);
                                                                            }
                                                                        }
                                                                        Err(message) => workspace_error.set(Some(message)),
                                                                    }
                                                                });
                                                            }
                                                            Err(message) => workspace_error.set(Some(message)),
                                                        }
                                                    },
                                                    if copied_embedding() == Some(vector_index) { "Copied" } else { "Copy vector" }
                                                }
                                            }
                                        }
                                        p { class: "vector-input", "{input_preview}" }
                                        code { "{preview}" }
                                    }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                div { class: "workspace-panel", role: "tabpanel",
                    label { class: "workspace-field",
                        span { "Query" }
                        textarea {
                            aria_label: "Rerank query",
                            placeholder: "Describe what the best document should match…",
                            maxlength: "{MAX_RERANK_QUERY_CHARS}",
                            rows: 3,
                            value: "{rerank_query}",
                            disabled: pending(),
                            oninput: move |event| rerank_query.set(event.value()),
                        }
                    }
                    label { class: "workspace-field",
                        span { "Documents" }
                        textarea {
                            aria_label: "Rerank documents",
                            placeholder: "One non-empty document per line…",
                            maxlength: "{MAX_RERANK_CONTENT_BYTES}",
                            rows: 8,
                            value: "{rerank_documents}",
                            disabled: pending(),
                            oninput: move |event| rerank_documents.set(event.value()),
                        }
                        small { "Documents are returned in descending similarity order." }
                    }
                    div { class: "workspace-inline-controls",
                        label { class: "workspace-field compact-field",
                            span { "Top results" }
                            input {
                                r#type: "number",
                                min: "1",
                                max: "256",
                                value: "{rerank_top_n}",
                                disabled: pending(),
                                oninput: move |event| rerank_top_n.set(event.value()),
                            }
                        }
                        button {
                            class: "btn-primary workspace-run",
                            disabled: pending()
                                || rerank_query().trim().is_empty()
                                || rerank_documents().trim().is_empty(),
                            onclick: run_rerank,
                            if pending() { "Running…" } else { "Rank documents" }
                        }
                    }

                    if let Some(batch) = rerank_output.as_ref() {
                        div { class: "workspace-results", aria_live: "polite",
                            div { class: "workspace-result-summary",
                                div {
                                    strong { "{batch.results.len()} result(s)" }
                                    span { "{batch.prompt_tokens} prompt tokens · {batch.model}" }
                                }
                                button {
                                    class: "btn-ghost compact",
                                    onclick: export_rerank,
                                    "Export JSON"
                                }
                            }
                            for (rank, result) in batch.results.iter().enumerate() {
                                {
                                    let rank = rank + 1;
                                    let document_index = result.index + 1;
                                    let score = result.relevance_score;
                                    let result_to_copy = result.clone();
                                    let result_index = result.index;
                                    rsx! { div { class: "rerank-card", key: "{result.index}",
                                        div { class: "rerank-position", "#{rank}" }
                                        div { class: "rerank-document",
                                            strong { "Document {document_index}" }
                                            p { "{result.document}" }
                                        }
                                        div { class: "rerank-result-actions",
                                            code { "{score:.6}" }
                                            button {
                                                class: "workspace-copy",
                                                aria_label: "Copy ranked result {rank}",
                                                title: "Copy this index, score, and document as JSON",
                                                onclick: move |_| {
                                                    workspace_error.set(None);
                                                    workspace_notice.set(None);
                                                    match api::rerank_result_clipboard_text(&result_to_copy) {
                                                        Ok(text) => {
                                                            spawn(async move {
                                                                match browser::copy_text_to_clipboard(&text).await {
                                                                    Ok(()) => {
                                                                        copied_rerank.set(Some(result_index));
                                                                        TimeoutFuture::new(1_500).await;
                                                                        if copied_rerank() == Some(result_index) {
                                                                            copied_rerank.set(None);
                                                                        }
                                                                    }
                                                                    Err(message) => workspace_error.set(Some(message)),
                                                                }
                                                            });
                                                        }
                                                        Err(message) => workspace_error.set(Some(message)),
                                                    }
                                                },
                                                if copied_rerank() == Some(result_index) { "Copied" } else { "Copy result" }
                                            }
                                        }
                                    }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn MessageList(
    messages: Vec<DisplayMessage>,
    connection: ConnectionState,
    busy: bool,
    retry_unavailable_reason: Option<String>,
    edit_unavailable_reason: Option<String>,
    branch_unavailable_reason: Option<String>,
    on_retry: EventHandler<()>,
    on_edit: EventHandler<()>,
    on_branch: EventHandler<usize>,
    on_continue: EventHandler<usize>,
    on_open_models: EventHandler<()>,
    on_open_settings: EventHandler<()>,
    on_error: EventHandler<String>,
) -> Element {
    let mut copied_message = use_signal(|| Option::<usize>::None);
    let mut visible_message_limit = use_signal(|| INITIAL_VISIBLE_MESSAGES);
    if messages.is_empty() {
        let view = empty_state_view(&connection);
        return rsx! {
            div { class: "messages",
                div { class: "empty-state",
                    div { class: "logo", "✦" }
                    h2 { "{view.title}" }
                    p { "{view.body}" }
                    if let Some((action, label)) = view.action {
                        button {
                            class: "btn-primary empty-state-action",
                            onclick: move |_| match action {
                                EmptyStateAction::Models => on_open_models.call(()),
                                EmptyStateAction::Settings => on_open_settings.call(()),
                            },
                            "{label}"
                        }
                    }
                }
            }
        };
    }
    let window = message_window(messages.len(), visible_message_limit());
    let earlier_message_count = window.next_visible.saturating_sub(window.visible);
    rsx! {
        div { class: "messages", aria_live: "polite",
            if window.start > 0 {
                div { class: "message-history-window",
                    button {
                        class: "btn-ghost compact",
                        aria_label: "Show {earlier_message_count} earlier messages",
                        onclick: move |_| visible_message_limit.set(window.next_visible),
                        "Show {earlier_message_count} earlier"
                    }
                    span { role: "status",
                        "Showing {window.visible} of {messages.len()} messages"
                    }
                }
            }
            for (index, message) in messages.iter().enumerate().skip(window.start) {
                {
                    let is_streaming = busy
                        && index == messages.len() - 1
                        && message.role == "assistant";
                    let is_latest_assistant = index == messages.len() - 1
                        && message.role == "assistant";
                    let is_unanswered_user = index == messages.len() - 1
                        && message.role == "user";
                    let is_user_before_latest_assistant = message.role == "user"
                        && index.saturating_add(2) == messages.len()
                        && messages.last().is_some_and(|last| last.role == "assistant");
                    let is_latest_user_turn = is_unanswered_user || is_user_before_latest_assistant;
                    let is_retry_target = is_latest_assistant || is_unanswered_user;
                    let bubble_class = if is_streaming {
                        "bubble cursor-blink"
                    } else {
                        "bubble"
                    };
                    let structured_output = message.role == "assistant"
                        && message.response_format.is_some();
                    let assistant_html = (message.role == "assistant" && !structured_output)
                        .then(|| render_assistant_markdown(&message.content));
                    rsx! {
                        div { class: "msg {message.role}", key: "{index}",
                            div { class: "{bubble_class}",
                                div { class: "message-header",
                                    if message.role != "user" {
                                        span { class: "role-tag",
                                            if message.role == "assistant" { "Bloom" } else { "{message.role}" }
                                        }
                                    } else {
                                        span {}
                                    }
                                    div { class: "message-actions",
                                        if !message.content.is_empty() {
                                            button {
                                                class: "message-action",
                                                aria_label: "Copy {message.role} message",
                                                aria_live: "polite",
                                                title: "Copy message as plain text",
                                                onclick: {
                                                    let content = message.content.clone();
                                                    move |_| {
                                                        let content = content.clone();
                                                        spawn(async move {
                                                            match browser::copy_text_to_clipboard(&content).await {
                                                                Ok(()) => {
                                                                    copied_message.set(Some(index));
                                                                    TimeoutFuture::new(1_500).await;
                                                                    if copied_message() == Some(index) {
                                                                        copied_message.set(None);
                                                                    }
                                                                }
                                                                Err(message) => on_error.call(message),
                                                            }
                                                        });
                                                    }
                                                },
                                                if copied_message() == Some(index) { "Copied" } else { "Copy" }
                                            }
                                        }
                                        button {
                                            class: "message-action",
                                            aria_label: "Branch conversation from this {message.role} message",
                                            aria_disabled: branch_unavailable_reason.is_some(),
                                            disabled: busy,
                                            title: branch_unavailable_reason.as_deref().unwrap_or(
                                                "Create a new conversation containing history through this message"
                                            ),
                                            onclick: {
                                                let unavailable_reason = branch_unavailable_reason.clone();
                                                move |_| {
                                                    if let Some(message) = unavailable_reason.clone() {
                                                        on_error.call(message);
                                                    } else {
                                                        on_branch.call(index);
                                                    }
                                                }
                                            },
                                            "Branch"
                                        }
                                        if message.role == "user" && index > 0 {
                                            button {
                                                class: "message-action",
                                                aria_label: "Continue conversation from this user message in a new chat",
                                                aria_disabled: branch_unavailable_reason.is_some(),
                                                disabled: busy,
                                                title: branch_unavailable_reason.as_deref().unwrap_or(
                                                    "Create a new conversation containing only this message and the history after it"
                                                ),
                                                onclick: {
                                                    let unavailable_reason = branch_unavailable_reason.clone();
                                                    move |_| {
                                                        if let Some(message) = unavailable_reason.clone() {
                                                            on_error.call(message);
                                                        } else {
                                                            on_continue.call(index);
                                                        }
                                                    }
                                                },
                                                "Continue"
                                            }
                                        }
                                        if is_latest_user_turn {
                                            button {
                                                class: "message-action",
                                                aria_label: "Edit latest user prompt",
                                                aria_disabled: edit_unavailable_reason.is_some(),
                                                disabled: busy,
                                                title: edit_unavailable_reason.as_deref().unwrap_or(
                                                    "Edit the latest prompt and regenerate its response"
                                                ),
                                                onclick: {
                                                    let unavailable_reason = edit_unavailable_reason.clone();
                                                    move |_| {
                                                        if let Some(message) = unavailable_reason.clone() {
                                                            on_error.call(message);
                                                        } else {
                                                            on_edit.call(());
                                                        }
                                                    }
                                                },
                                                "Edit"
                                            }
                                        }
                                        if is_retry_target {
                                            button {
                                                class: "message-action",
                                                aria_label: "Regenerate latest response",
                                                aria_disabled: retry_unavailable_reason.is_some(),
                                                disabled: busy,
                                                title: retry_unavailable_reason.as_deref().unwrap_or(
                                                    "Regenerate with the current model and generation settings"
                                                ),
                                                onclick: {
                                                    let unavailable_reason = retry_unavailable_reason.clone();
                                                    move |_| {
                                                        if let Some(message) = unavailable_reason.clone() {
                                                            on_error.call(message);
                                                        } else {
                                                            on_retry.call(());
                                                        }
                                                    }
                                                },
                                                if is_latest_assistant { "Regenerate" } else { "Retry" }
                                            }
                                        }
                                    }
                                }
                                if structured_output {
                                    pre { class: "structured-output",
                                        code { "{message.content}" }
                                    }
                                } else if let Some(html) = assistant_html {
                                    div {
                                        class: "assistant-markdown",
                                        dangerous_inner_html: "{html}",
                                    }
                                } else {
                                    "{message.content}"
                                }
                                if message.role == "assistant" && !is_streaming {
                                    if let Some(stats) = message.generation.as_ref() {
                                        div { class: "generation-stats", aria_label: "Generation statistics",
                                            if let Some(label) = generation_outcome_label(stats.outcome) {
                                                span { class: "generation-outcome {label.to_ascii_lowercase()}", "{label}" }
                                            }
                                            if let Some(model) = stats.model.as_deref() {
                                                span {
                                                    class: "generation-model",
                                                    title: "Execution model: {model}",
                                                    "{model}"
                                                }
                                            }
                                            if let (Some(prompt), Some(completion)) =
                                                (stats.prompt_tokens, stats.completion_tokens)
                                            {
                                                span {
                                                    title: "Prompt tokens",
                                                    "{prompt} prompt"
                                                }
                                                span {
                                                    title: "Completion tokens",
                                                    "{completion} output"
                                                }
                                            }
                                            span {
                                                title: "Total request time",
                                                "{format_generation_millis(stats.elapsed_ms)}"
                                            }
                                            if let Some(first_token_ms) = stats.first_token_ms {
                                                span {
                                                    title: "Time to first token",
                                                    "TTFT {format_generation_millis(first_token_ms)}"
                                                }
                                            }
                                            if let Some(tokens_per_second) = stats.tokens_per_second() {
                                                span {
                                                    title: "Completion tokens divided by time from the first token to stream completion",
                                                    "{tokens_per_second:.1} tok/s"
                                                }
                                            }
                                        }
                                    } else if let Some(model) = message.execution_model.as_deref() {
                                        div { class: "generation-stats", aria_label: "Response model provenance",
                                            span {
                                                class: "generation-model",
                                                title: "Execution model recorded by the imported conversation archive: {model}",
                                                "{model}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn DiagnosticsDrawer(config: Signal<ConnConfig>, on_close: EventHandler<()>) -> Element {
    use_modal_focus(DIAGNOSTICS_DRAWER_DIALOG_ID);
    let mut snapshot = use_signal(|| Option::<api::ObservabilitySnapshot>::None);
    let mut diagnostics_error = use_signal(|| Option::<String>::None);
    let mut refresh_pending = use_signal(|| false);
    let mut export_notice = use_signal(|| Option::<String>::None);

    let _diagnostics_poller = use_future(move || async move {
        loop {
            match api::fetch_observability(&config()).await {
                Ok(current) => {
                    snapshot.set(Some(current));
                    diagnostics_error.set(None);
                }
                Err(message) => diagnostics_error.set(Some(message)),
            }
            TimeoutFuture::new(DIAGNOSTICS_POLL_INTERVAL_MS).await;
        }
    });

    let refresh = move |_| {
        if refresh_pending() {
            return;
        }
        refresh_pending.set(true);
        export_notice.set(None);
        let cfg = config();
        spawn(async move {
            match api::fetch_observability(&cfg).await {
                Ok(current) => {
                    snapshot.set(Some(current));
                    diagnostics_error.set(None);
                }
                Err(message) => diagnostics_error.set(Some(message)),
            }
            refresh_pending.set(false);
        });
    };

    let export_snapshot = move |_| {
        export_notice.set(None);
        let result = snapshot()
            .ok_or_else(|| "No diagnostics snapshot is available to export.".to_string())
            .and_then(|snapshot| snapshot.to_pretty_json())
            .and_then(|json| {
                browser::download_text_file(
                    DIAGNOSTICS_FILENAME,
                    "application/json;charset=utf-8",
                    &json,
                )
            });
        match result {
            Ok(()) => export_notice.set(Some(
                "Exported a credential-free runtime snapshot.".to_string(),
            )),
            Err(message) => diagnostics_error.set(Some(message)),
        }
    };

    let current = snapshot();
    rsx! {
        div { class: "drawer-overlay", onclick: move |_| on_close.call(()),
            div {
                id: DIAGNOSTICS_DRAWER_DIALOG_ID,
                class: "drawer diagnostics-drawer",
                role: "dialog",
                aria_modal: "true",
                aria_labelledby: "runtime-diagnostics-title",
                aria_describedby: "runtime-diagnostics-description",
                onclick: move |event| event.stop_propagation(),
                onkeydown: move |event| handle_modal_key(event, DIAGNOSTICS_DRAWER_DIALOG_ID, on_close),
                div { class: "drawer-title-row",
                    div {
                        h3 { id: "runtime-diagnostics-title", "Runtime diagnostics" }
                        div { id: "runtime-diagnostics-description", class: "drawer-sub",
                            "Live, authenticated counters from the configured Bloom server."
                        }
                    }
                    button {
                        class: "drawer-close",
                        aria_label: "Close runtime diagnostics",
                        onclick: move |_| on_close.call(()),
                        "×"
                    }
                }

                div { class: "diagnostics-toolbar",
                    button {
                        class: "btn-ghost compact",
                        disabled: refresh_pending(),
                        onclick: refresh,
                        if refresh_pending() { "Refreshing…" } else { "Refresh" }
                    }
                    button {
                        class: "btn-primary compact",
                        disabled: current.is_none(),
                        onclick: export_snapshot,
                        "Export JSON"
                    }
                }

                if let Some(message) = diagnostics_error() {
                    div { class: "test-result err", role: "alert", "{message}" }
                }
                if let Some(message) = export_notice() {
                    div { class: "test-result ok", role: "status", "{message}" }
                }

                if let Some(snapshot) = current.as_ref() {
                    section { class: "diagnostics-card diagnostics-overview",
                        div { class: "diagnostics-card-heading",
                            div {
                                span { class: "diagnostics-eyebrow", "Server" }
                                strong { "Bloom {snapshot.server.version}" }
                            }
                            span {
                                class: if snapshot.ready { "preflight-badge ready" } else { "preflight-badge warning" },
                                if snapshot.ready { "Ready" } else { "Not ready" }
                            }
                        }
                        dl { class: "diagnostics-grid",
                            div { dt { "Model" } dd { "{snapshot.model}" } }
                            div { dt { "Load state" } dd { "{format_load_phase(&snapshot.load.phase)}" } }
                            div { dt { "Load progress" } dd { "{snapshot.load.progress}%" } }
                            div { dt { "Uptime" } dd { "{format_duration_seconds(snapshot.server.uptime_seconds)}" } }
                            div { dt { "Snapshot time" } dd { "Unix {snapshot.created}" } }
                            div { dt { "Speculative mode" } dd { "{snapshot.speculative_mode}" } }
                        }
                        if let Some(requested_model) = snapshot.load.requested_model.as_ref() {
                            p { class: "diagnostics-note", "Last requested model: {requested_model}" }
                        }
                        if snapshot.load.failure_present {
                            p { class: "diagnostics-warning",
                                "The last model load failed. Open Models for the actionable server error and preflight details."
                            }
                        }
                    }

                    section { class: "diagnostics-card",
                        div { class: "diagnostics-card-heading",
                            div {
                                span { class: "diagnostics-eyebrow", "Activity" }
                                strong { "Requests and tokens" }
                            }
                        }
                        dl { class: "diagnostics-grid",
                            div { dt { "Requests" } dd { "{snapshot.requests.total}" } }
                            div { dt { "Completed" } dd { "{snapshot.requests.completed}" } }
                            div { dt { "Failed" } dd { "{snapshot.requests.failed}" } }
                            div { dt { "In flight" } dd { "{snapshot.requests.in_flight}" } }
                            div { dt { "Prompt tokens" } dd { "{snapshot.tokens.prompt_total}" } }
                            div { dt { "Generated tokens" } dd { "{snapshot.tokens.generated_total}" } }
                        }
                    }

                    section { class: "diagnostics-card",
                        div { class: "diagnostics-card-heading",
                            div {
                                span { class: "diagnostics-eyebrow", "Scheduler" }
                                strong { if snapshot.scheduler.ifb_enabled { "In-flight batching enabled" } else { "Direct scheduling" } }
                            }
                        }
                        dl { class: "diagnostics-grid",
                            div { dt { "Prefill queue" } dd { "{snapshot.scheduler.prefill_queue}" } }
                            div { dt { "Decode queue" } dd { "{snapshot.scheduler.decoding_queue}" } }
                            div { dt { "Active" } dd { "{snapshot.scheduler.active_requests}" } }
                        }
                    }

                    section { class: "diagnostics-card",
                        div { class: "diagnostics-card-heading",
                            div {
                                span { class: "diagnostics-eyebrow", "Memory" }
                                strong { "Current process usage" }
                            }
                        }
                        dl { class: "diagnostics-grid",
                            div { dt { "RAM" } dd { "{format_bytes(snapshot.memory.used_ram)}" } }
                            div { dt { "Peak RAM" } dd { "{format_bytes(snapshot.memory.peak_ram)}" } }
                            div { dt { "System RAM" } dd { "{format_optional_bytes(snapshot.memory.total_ram)}" } }
                            div { dt { "VRAM" } dd { "{format_optional_bytes(snapshot.memory.used_vram)}" } }
                            div { dt { "Peak VRAM" } dd { "{format_optional_bytes(snapshot.memory.peak_vram)}" } }
                            div { dt { "Total VRAM" } dd { "{format_optional_bytes(snapshot.memory.total_vram)}" } }
                        }
                        if snapshot.memory.total_ram > 0 {
                            progress {
                                value: "{snapshot.memory.used_ram.min(snapshot.memory.total_ram)}",
                                max: "{snapshot.memory.total_ram}",
                                aria_label: "Process RAM usage relative to system RAM",
                            }
                        }
                    }

                    if let Some(estimate) = snapshot.startup_memory_estimate.as_ref() {
                        section { class: "diagnostics-card",
                            div { class: "diagnostics-card-heading",
                                div {
                                    span { class: "diagnostics-eyebrow", "Load plan" }
                                    strong { "Startup memory estimate" }
                                }
                            }
                            dl { class: "diagnostics-grid",
                                div { dt { "Total" } dd { "{format_bytes(estimate.total_bytes)}" } }
                                div { dt { "Weights" } dd { "{format_bytes(estimate.weight_bytes)}" } }
                                div { dt { "Host weights" } dd { "{format_bytes(estimate.host_weight_bytes)}" } }
                                div { dt { "Device weights" } dd { "{format_optional_bytes(estimate.device_weight_bytes)}" } }
                                div { dt { "KV cache" } dd { "{format_bytes(estimate.kv_cache_bytes)}" } }
                                div { dt { "Workspace" } dd { "{format_bytes(estimate.temp_tensor_bytes)}" } }
                            }
                            if !estimate.memory_scope.is_empty() {
                                p { class: "diagnostics-note", "{estimate.memory_scope}" }
                            }
                        }
                    }

                    section { class: "diagnostics-card",
                        div { class: "diagnostics-card-heading",
                            div {
                                span { class: "diagnostics-eyebrow", "Cache" }
                                strong { "KV cache" }
                            }
                            if snapshot.kv_cache.total_blocks > 0 {
                                span { "{format_percent(snapshot.kv_cache.utilization)} used" }
                            }
                        }
                        if snapshot.kv_cache.total_blocks == 0 {
                            p { class: "diagnostics-note", "KV block caching is not active for this runtime." }
                        } else {
                            dl { class: "diagnostics-grid",
                                div { dt { "Total blocks" } dd { "{snapshot.kv_cache.total_blocks}" } }
                                div { dt { "Free blocks" } dd { "{snapshot.kv_cache.free_blocks}" } }
                                div { dt { "Active blocks" } dd { "{snapshot.kv_cache.active_blocks}" } }
                                div { dt { "Cached blocks" } dd { "{snapshot.kv_cache.cached_blocks}" } }
                                div { dt { "Hits" } dd { "{snapshot.kv_cache.hits}" } }
                                div { dt { "Misses" } dd { "{snapshot.kv_cache.misses}" } }
                                div { dt { "Evictions" } dd { "{snapshot.kv_cache.evictions}" } }
                                div { dt { "Reuses" } dd { "{snapshot.kv_cache.reuses}" } }
                            }
                            progress {
                                value: "{snapshot.kv_cache.utilization}",
                                max: "1",
                                aria_label: "KV cache utilization",
                            }
                        }
                    }

                    if let Some(cachemesh) = snapshot.cachemesh.as_ref().filter(|cachemesh| cachemesh.enabled) {
                        section { class: "diagnostics-card",
                            div { class: "diagnostics-card-heading",
                                div {
                                    span { class: "diagnostics-eyebrow", "Distributed cache" }
                                    strong { "CacheMesh" }
                                }
                            }
                            dl { class: "diagnostics-grid",
                                div { dt { "L1 items" } dd { "{cachemesh.l1.items}" } }
                                div { dt { "L1 bytes" } dd { "{format_bytes(cachemesh.l1.bytes)}" } }
                                div { dt { "L2 items" } dd { "{cachemesh.l2.items}" } }
                                div { dt { "L2 bytes" } dd { "{format_bytes(cachemesh.l2.bytes)}" } }
                                div { dt { "L3 items" } dd { "{cachemesh.l3.items}" } }
                                div { dt { "L3 bytes" } dd { "{format_bytes(cachemesh.l3.bytes)}" } }
                            }
                        }
                    }

                    p { class: "diagnostics-export-note",
                        "Exports contain this runtime snapshot and model identifiers. Server addresses, API keys, prompts, and conversation text are excluded."
                    }
                } else if diagnostics_error().is_none() {
                    div { class: "catalog-empty", "Loading runtime diagnostics…" }
                }
            }
        }
    }
}

#[component]
fn SettingsDrawer(
    config: Signal<ConnConfig>,
    on_close: EventHandler<()>,
    generation: Signal<ChatOptions>,
    context_window: Option<u64>,
) -> Element {
    use_modal_focus(SETTINGS_DRAWER_DIALOG_ID);
    let mut base_url = use_signal(|| config().base_url);
    let mut api_key = use_signal(|| config().api_key);
    let mut remember_api_key = use_signal(|| config().remember_api_key);
    let mut max_tokens = use_signal(|| generation().max_tokens);
    let mut temperature = use_signal(|| generation().temperature);
    let mut top_p = use_signal(|| generation().top_p);
    let mut seed = use_signal(|| {
        generation()
            .seed
            .map(|value| value.to_string())
            .unwrap_or_default()
    });
    let mut stop_sequences = use_signal(|| {
        serde_json::to_string_pretty(&generation().stop_sequences)
            .unwrap_or_else(|_| "[]".to_string())
    });
    let mut system_prompt = use_signal(|| generation().system_prompt);
    let mut response_format = use_signal(|| generation().response_format);
    let mut json_schema = use_signal(|| generation().json_schema);
    let mut test_result = use_signal(|| Option::<(bool, String)>::None);
    let mut testing = use_signal(|| false);
    let max_tokens_limit = context_window
        .and_then(|window| usize::try_from(window.saturating_sub(1)).ok())
        .unwrap_or(32_768)
        .clamp(1, 32_768);

    let mut on_test = move |_| {
        testing.set(true);
        test_result.set(None);
        let cfg = ConnConfig {
            base_url: base_url().trim_end_matches('/').to_string(),
            api_key: api_key(),
            remember_api_key: remember_api_key(),
        };
        spawn(async move {
            let result = match api::fetch_connection_readiness(&cfg).await {
                Ok(readiness) if readiness.status == "ready" => (
                    true,
                    format!(
                        "Ready · Bloom {} · Model {}",
                        readiness.server_version, readiness.model
                    ),
                ),
                Ok(readiness) if readiness.load_error.is_some() => (
                    false,
                    readiness
                        .load_error
                        .unwrap_or_else(|| "The model failed to load.".to_string()),
                ),
                Ok(readiness) if !readiness.loading && readiness.progress == 0 => (
                    true,
                    format!(
                        "Connected · Bloom {} · No active model",
                        readiness.server_version
                    ),
                ),
                Ok(readiness) => (
                    true,
                    format!(
                        "Connected · Bloom {} · Model loading {}%",
                        readiness.server_version, readiness.progress
                    ),
                ),
                Err(error) if error.is_incompatible() => {
                    (false, format!("Incompatible server: {error}"))
                }
                Err(error) if error.is_authentication() => {
                    (false, format!("Authentication failed: {error}"))
                }
                Err(error) => (false, format!("Connection failed: {error}")),
            };
            test_result.set(Some(result));
            testing.set(false);
        });
    };

    let mut on_save = move |_| {
        let cfg = ConnConfig {
            base_url: base_url().trim_end_matches('/').to_string(),
            api_key: api_key(),
            remember_api_key: remember_api_key(),
        };
        let parsed_seed = match optional_seed(&seed()) {
            Ok(seed) => seed,
            Err(message) => {
                test_result.set(Some((false, message)));
                return;
            }
        };
        let parsed_stop_sequences = match api::parse_stop_sequences_setting(&stop_sequences()) {
            Ok(sequences) => sequences,
            Err(message) => {
                test_result.set(Some((false, message)));
                return;
            }
        };
        let options = ChatOptions {
            max_tokens: max_tokens(),
            temperature: temperature(),
            top_p: top_p(),
            seed: parsed_seed,
            stop_sequences: parsed_stop_sequences,
            system_prompt: system_prompt(),
            response_format: response_format(),
            json_schema: json_schema(),
        };
        let result = options
            .validate()
            .and_then(|_| validate_context_reservation(options.max_tokens, context_window))
            .and_then(|_| storage::save_connection(&cfg))
            .and_then(|_| storage::save_generation(&options));
        match result {
            Ok(()) => {
                config.set(cfg);
                generation.set(options);
                on_close.call(());
            }
            Err(message) => test_result.set(Some((false, message))),
        }
    };

    rsx! {
        div { class: "drawer-overlay", onclick: move |_| on_close.call(()),
            div {
                id: SETTINGS_DRAWER_DIALOG_ID,
                class: "drawer",
                role: "dialog",
                aria_modal: "true",
                aria_labelledby: "settings-title",
                aria_describedby: "settings-description",
                onclick: move |event| event.stop_propagation(),
                onkeydown: move |event| handle_modal_key(event, SETTINGS_DRAWER_DIALOG_ID, on_close),
                h3 { id: "settings-title", "Settings" }
                div { id: "settings-description", class: "drawer-sub", "Connect to a running bloom_server instance." }

                div { class: "field",
                    label { r#for: "base-url", "Server address" }
                    input {
                        id: "base-url",
                        value: "{base_url}",
                        maxlength: "{MAX_BASE_URL_CHARS}",
                        placeholder: "http://127.0.0.1:3000",
                        oninput: move |event| base_url.set(event.value()),
                    }
                    div { class: "help", "Defaults to the current origin when the UI is embedded in Bloom." }
                }
                div { class: "field",
                    label { r#for: "api-key", "API key (optional)" }
                    input {
                        id: "api-key",
                        r#type: "password",
                        value: "{api_key}",
                        maxlength: "{MAX_API_KEY_CHARS}",
                        placeholder: "BLOOM_API_KEY",
                        oninput: move |event| api_key.set(event.value()),
                    }
                    div { class: "help", "Required when the server uses --api-key. Hidden from diagnostics and conversation exports." }
                }
                div { class: "field credential-persistence-field",
                    label { class: "checkbox-row",
                        input {
                            r#type: "checkbox",
                            checked: remember_api_key(),
                            onchange: move |event| remember_api_key.set(event.checked()),
                        }
                        span { "Remember API key in this browser" }
                    }
                    div { class: "help",
                        if remember_api_key() {
                            "The API key will remain in this origin's local storage until you clear or replace it."
                        } else {
                            "The API key stays in this tab's session storage and is not written to persistent local storage."
                        }
                    }
                }
                div { class: "field",
                    label { r#for: "system-prompt", "System prompt (optional)" }
                    textarea {
                        id: "system-prompt",
                        rows: 4,
                        maxlength: "{MAX_SYSTEM_PROMPT_CHARS}",
                        value: "{system_prompt}",
                        placeholder: "Set behavior or context for every request.",
                        oninput: move |event| system_prompt.set(event.value()),
                    }
                    div { class: "help", "Stored only in this browser and prepended to each request." }
                }
                div { class: "field",
                    label { r#for: "response-format", "Response format" }
                    select {
                        id: "response-format",
                        value: "{response_format().form_value()}",
                        onchange: move |event| {
                            if let Some(mode) = ResponseFormatMode::from_form_value(&event.value()) {
                                response_format.set(mode);
                            }
                        },
                        option { value: "text", "Text" }
                        option { value: "json_object", "JSON object" }
                        option { value: "json_schema", "JSON Schema" }
                    }
                    div { class: "help",
                        "Structured output is supported for text chat. Model and engine support still applies."
                    }
                }
                if response_format() == ResponseFormatMode::JsonSchema {
                    div { class: "field",
                        label { r#for: "response-json-schema", "Response JSON Schema" }
                        textarea {
                            id: "response-json-schema",
                            rows: 10,
                            maxlength: "{MAX_RESPONSE_JSON_SCHEMA_BYTES}",
                            value: "{json_schema}",
                            placeholder: "{RESPONSE_SCHEMA_PLACEHOLDER}",
                            oninput: move |event| json_schema.set(event.value()),
                        }
                        div { class: "help",
                            "Supported constraints: type, enum, required, properties, additionalProperties, and items. The root type must be object. Unsupported keywords are rejected."
                        }
                    }
                }
                div { class: "field",
                    label { r#for: "max-tokens", "Maximum generated tokens: {max_tokens}" }
                    input {
                        id: "max-tokens",
                        r#type: "number",
                        min: "1",
                        max: "{max_tokens_limit}",
                        step: "1",
                        value: "{max_tokens}",
                        oninput: move |event| {
                            if let Ok(value) = event.value().parse::<usize>() {
                                max_tokens.set(value);
                            }
                        },
                    }
                    div { class: "help",
                        if let Some(window) = context_window {
                            "The active model has a {window}-token context window shared by the prompt and response."
                        } else {
                            "The prompt and response share the active model's context window."
                        }
                    }
                }
                div { class: "field",
                    label { r#for: "temperature", "Temperature: {temperature:.2}" }
                    input {
                        id: "temperature",
                        r#type: "range",
                        min: "0",
                        max: "2",
                        step: "0.05",
                        value: "{temperature}",
                        oninput: move |event| {
                            if let Ok(value) = event.value().parse::<f64>() {
                                temperature.set(value);
                            }
                        },
                    }
                }
                div { class: "field",
                    label { r#for: "top-p", "Top P: {top_p:.2}" }
                    input {
                        id: "top-p",
                        r#type: "range",
                        min: "0.05",
                        max: "1",
                        step: "0.05",
                        value: "{top_p}",
                        oninput: move |event| {
                            if let Ok(value) = event.value().parse::<f64>() {
                                top_p.set(value);
                            }
                        },
                    }
                }
                div { class: "field",
                    label { r#for: "seed", "Seed (optional)" }
                    input {
                        id: "seed",
                        r#type: "number",
                        min: "0",
                        step: "1",
                        value: "{seed}",
                        placeholder: "Random",
                        oninput: move |event| seed.set(event.value()),
                    }
                    div { class: "help", "Use the same seed and settings for reproducible sampling." }
                }
                div { class: "field",
                    label { r#for: "stop-sequences", "Stop sequences (optional)" }
                    textarea {
                        id: "stop-sequences",
                        rows: 4,
                        maxlength: "{api::MAX_STOP_SEQUENCES_BYTES}",
                        value: "{stop_sequences}",
                        placeholder: r#"["END", "\nUser:"]"#,
                        oninput: move |event| stop_sequences.set(event.value()),
                    }
                    div { class: "help",
                        "Enter a JSON array containing up to four exact strings. Text generation stops before the first match; image chat does not currently support this setting."
                    }
                }

                button {
                    class: "btn-ghost",
                    disabled: testing(),
                    onclick: move |_| on_test(()),
                    if testing() { "Testing…" } else { "Test connection" }
                }
                if let Some((ok, message)) = test_result() {
                    div { class: if ok { "test-result ok" } else { "test-result err" }, "{message}" }
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

fn optional_seed(value: &str) -> Result<Option<u64>, String> {
    let value = value.trim();
    if value.is_empty() {
        Ok(None)
    } else {
        value
            .parse::<u64>()
            .map(Some)
            .map_err(|_| "Seed must be a non-negative whole number.".into())
    }
}

#[component]
fn RenameConversationDialog(
    initial_title: String,
    error: Option<String>,
    on_cancel: EventHandler<()>,
    on_submit: EventHandler<String>,
) -> Element {
    use_modal_focus(RENAME_CONVERSATION_DIALOG_ID);
    let mut title = use_signal(move || initial_title);
    let title_chars = title().trim().chars().count();
    let can_submit = title_chars > 0 && title_chars <= MAX_RENAMED_TITLE_CHARS;

    rsx! {
        div { class: "modal-overlay", onclick: move |_| on_cancel.call(()),
            div {
                id: RENAME_CONVERSATION_DIALOG_ID,
                class: "confirm-dialog rename-conversation-dialog",
                role: "dialog",
                aria_modal: "true",
                aria_labelledby: "rename-conversation-title",
                aria_describedby: "rename-conversation-description",
                onclick: move |event| event.stop_propagation(),
                onkeydown: move |event| handle_modal_key(event, RENAME_CONVERSATION_DIALOG_ID, on_cancel),
                h3 { id: "rename-conversation-title", "Rename conversation" }
                p { id: "rename-conversation-description", "Choose a short title that makes this conversation easy to find." }
                label { r#for: "conversation-title", "Title" }
                input {
                    id: "conversation-title",
                    autofocus: true,
                    maxlength: "{MAX_RENAMED_TITLE_CHARS}",
                    value: "{title}",
                    oninput: move |event| title.set(event.value()),
                }
                div { class: "rename-title-count",
                    "{title_chars} / {MAX_RENAMED_TITLE_CHARS} characters"
                }
                if let Some(message) = error {
                    div { class: "test-result err", role: "alert", "{message}" }
                }
                div { class: "btn-row",
                    button { class: "btn-ghost", onclick: move |_| on_cancel.call(()), "Cancel" }
                    button {
                        class: "btn-primary",
                        disabled: !can_submit,
                        onclick: move |_| on_submit.call(title().to_string()),
                        "Save"
                    }
                }
            }
        }
    }
}

#[component]
fn EditPromptDialog(
    initial_content: String,
    error: Option<String>,
    on_cancel: EventHandler<()>,
    on_submit: EventHandler<String>,
) -> Element {
    use_modal_focus(EDIT_PROMPT_DIALOG_ID);
    let mut content = use_signal(move || initial_content);
    let can_submit = !content().trim().is_empty();

    rsx! {
        div { class: "modal-overlay", onclick: move |_| on_cancel.call(()),
            div {
                id: EDIT_PROMPT_DIALOG_ID,
                class: "confirm-dialog edit-prompt-dialog",
                role: "dialog",
                aria_modal: "true",
                aria_labelledby: "edit-prompt-title",
                aria_describedby: "edit-prompt-description",
                onclick: move |event| event.stop_propagation(),
                onkeydown: move |event| handle_modal_key(event, EDIT_PROMPT_DIALOG_ID, on_cancel),
                h3 { id: "edit-prompt-title", "Edit latest prompt" }
                p { id: "edit-prompt-description", "Saving replaces the latest prompt and regenerates its response with the current model and settings." }
                label { r#for: "edited-prompt", "Prompt" }
                textarea {
                    id: "edited-prompt",
                    autofocus: true,
                    rows: 8,
                    maxlength: "{MAX_CHAT_INPUT_CHARS}",
                    value: "{content}",
                    oninput: move |event| content.set(event.value()),
                }
                if let Some(message) = error {
                    div { class: "test-result err", role: "alert", "{message}" }
                }
                div { class: "edit-prompt-help",
                    "If generation fails before returning text, Bloom restores the original prompt and response."
                }
                div { class: "btn-row",
                    button { class: "btn-ghost", onclick: move |_| on_cancel.call(()), "Cancel" }
                    button {
                        class: "btn-primary",
                        disabled: !can_submit,
                        onclick: move |_| on_submit.call(content().trim().to_string()),
                        "Save & regenerate"
                    }
                }
            }
        }
    }
}

#[component]
fn ConversationImportDialog(
    conversation_count: usize,
    message_count: usize,
    error: Option<String>,
    merge_unavailable_reason: Option<String>,
    on_cancel: EventHandler<()>,
    on_replace: EventHandler<()>,
    on_merge: EventHandler<()>,
) -> Element {
    use_modal_focus(IMPORT_CONVERSATIONS_DIALOG_ID);
    let merge_disabled = merge_unavailable_reason.is_some();

    rsx! {
        div { class: "modal-overlay", onclick: move |_| on_cancel.call(()),
            div {
                id: IMPORT_CONVERSATIONS_DIALOG_ID,
                class: "confirm-dialog conversation-import-dialog",
                role: "dialog",
                aria_modal: "true",
                aria_labelledby: "import-conversations-title",
                aria_describedby: "import-conversations-description",
                onclick: move |event| event.stop_propagation(),
                onkeydown: move |event| handle_modal_key(event, IMPORT_CONVERSATIONS_DIALOG_ID, on_cancel),
                h3 { id: "import-conversations-title", "Import conversations" }
                p {
                    id: "import-conversations-description",
                    "The archive contains {conversation_count} conversation(s) and {message_count} message(s). Choose how to apply it. Connection settings and generation options are never changed."
                }
                ul { class: "import-choices",
                    li {
                        strong { "Merge: " }
                        "append every archived conversation with a fresh local ID while keeping current history and selection. Exact duplicates are not removed."
                    }
                    li {
                        strong { "Replace all: " }
                        "remove every current browser-local conversation and restore only this archive."
                    }
                }
                if let Some(message) = merge_unavailable_reason {
                    div { class: "test-result err", role: "alert", "{message}" }
                }
                if let Some(message) = error {
                    div { class: "test-result err", role: "alert", "{message}" }
                }
                div { class: "btn-row",
                    button { class: "btn-ghost", onclick: move |_| on_cancel.call(()), "Cancel" }
                    button { class: "btn-danger", onclick: move |_| on_replace.call(()), "Replace all" }
                    button {
                        class: "btn-primary",
                        disabled: merge_disabled,
                        onclick: move |_| on_merge.call(()),
                        "Merge"
                    }
                }
            }
        }
    }
}

#[component]
fn ConfirmDialog(
    dialog_id: &'static str,
    title: String,
    body: String,
    confirm_label: String,
    on_cancel: EventHandler<()>,
    on_confirm: EventHandler<()>,
) -> Element {
    use_modal_focus(dialog_id);
    let title_id = format!("{dialog_id}-title");
    let description_id = format!("{dialog_id}-description");
    rsx! {
        div { class: "modal-overlay", onclick: move |_| on_cancel.call(()),
            div {
                id: dialog_id,
                class: "confirm-dialog",
                role: "alertdialog",
                aria_modal: "true",
                aria_labelledby: "{title_id}",
                aria_describedby: "{description_id}",
                onclick: move |event| event.stop_propagation(),
                onkeydown: move |event| handle_modal_key(event, dialog_id, on_cancel),
                h3 { id: "{title_id}", "{title}" }
                p { id: "{description_id}", "{body}" }
                div { class: "btn-row",
                    button { class: "btn-ghost", onclick: move |_| on_cancel.call(()), "Cancel" }
                    button { class: "btn-danger", onclick: move |_| on_confirm.call(()), "{confirm_label}" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        api, append_transcript_segment, conversation_context_status, conversation_import_candidate,
        conversation_model_transition, embedding_vector_norm, embedding_vector_preview,
        empty_state_view, encoder_input_preview, format_duration, format_duration_seconds,
        format_generation_millis, format_inventory_changes, format_inventory_drift_severity,
        format_inventory_drift_status, format_load_phase, format_model_precision,
        format_model_tasks, format_optional_bytes, format_optional_count, format_parameter_count,
        format_percent, generation_outcome_label, integrity_phase_label, license_policy_allows,
        message_window, modal_key_action, model_index_local_state,
        model_index_poll_interval_seconds, model_index_upgrade_source_is_active,
        model_provenance_summary, optional_seed, parse_optional_embedding_dimensions,
        supported_image_mime, supported_model_import_filename,
        unconfirmed_conversation_model_transition, valid_sha256_input,
        validate_context_reservation, ChatMessage, ConnectionState, ConversationImportMode,
        ConversationStore, DisplayMessage, EmptyStateAction, GenerationOutcome, GenerationStats,
        ModalKeyAction, ModelIndexLocalState, Readiness,
    };
    use dioxus::prelude::Key;

    #[test]
    fn modal_keyboard_policy_closes_and_cycles_without_capturing_other_keys() {
        assert_eq!(modal_key_action(&Key::Escape, false), ModalKeyAction::Close);
        assert_eq!(
            modal_key_action(&Key::Tab, false),
            ModalKeyAction::TrapForward
        );
        assert_eq!(
            modal_key_action(&Key::Tab, true),
            ModalKeyAction::TrapBackward
        );
        assert_eq!(
            modal_key_action(&Key::Character("x".into()), false),
            ModalKeyAction::Ignore
        );
    }

    #[test]
    fn conversation_import_policy_keeps_recovery_data_merge_locked() {
        let mut local = ConversationStore::default();
        local.append_user("Local prompt".into());
        let local_before = local.clone();
        let mut imported = ConversationStore::default();
        imported.append_user("Imported prompt".into());

        let error =
            conversation_import_candidate(&local, &imported, ConversationImportMode::Merge, true)
                .unwrap_err();
        assert!(error.contains("recovery is required"));
        assert_eq!(local, local_before);

        let replacement =
            conversation_import_candidate(&local, &imported, ConversationImportMode::Replace, true)
                .unwrap();
        assert_eq!(replacement, imported);

        let merged =
            conversation_import_candidate(&local, &imported, ConversationImportMode::Merge, false)
                .unwrap();
        assert_eq!(merged.active_id, local.active_id);
        assert_eq!(merged.conversations.len(), 2);
    }

    #[test]
    fn message_window_starts_bounded_and_expands_in_stable_pages() {
        let empty = message_window(0, 100);
        assert_eq!((empty.start, empty.visible, empty.next_visible), (0, 0, 0));

        let short = message_window(80, 100);
        assert_eq!(
            (short.start, short.visible, short.next_visible),
            (0, 80, 80)
        );

        let initial = message_window(450, 100);
        assert_eq!(
            (initial.start, initial.visible, initial.next_visible),
            (350, 100, 200)
        );

        let expanded = message_window(450, initial.next_visible);
        assert_eq!(
            (expanded.start, expanded.visible, expanded.next_visible),
            (250, 200, 300)
        );

        let complete = message_window(450, usize::MAX);
        assert_eq!(
            (complete.start, complete.visible, complete.next_visible),
            (0, 450, 450)
        );
    }

    #[test]
    fn optional_seed_accepts_blank_and_unsigned_values() {
        assert_eq!(optional_seed("  "), Ok(None));
        assert_eq!(optional_seed("42"), Ok(Some(42)));
        assert!(optional_seed("-1").is_err());
    }

    #[test]
    fn local_model_import_accepts_only_supported_single_file_names() {
        assert!(supported_model_import_filename("model.gguf"));
        assert!(supported_model_import_filename("MODEL.ONNX"));
        assert!(!supported_model_import_filename(".hidden.gguf"));
        assert!(!supported_model_import_filename("folder/model.gguf"));
        assert!(!supported_model_import_filename("model.safetensors"));
    }

    #[test]
    fn verified_acquisitions_require_an_exact_hexadecimal_sha256() {
        assert!(valid_sha256_input(&"ab".repeat(32)));
        assert!(valid_sha256_input(&"AB".repeat(32)));
        assert!(!valid_sha256_input(&"ab".repeat(31)));
        assert!(!valid_sha256_input(&"zz".repeat(32)));
    }

    #[test]
    fn enforced_license_policy_requires_one_approved_declaration() {
        let open = api::ModelLicensePolicy::default();
        let enforced = api::ModelLicensePolicy {
            enforced: true,
            allowed: vec!["Apache-2.0".to_string(), "MIT".to_string()],
        };

        assert!(license_policy_allows(&open, ""));
        assert!(license_policy_allows(&enforced, "apache-2.0"));
        assert!(!license_policy_allows(&enforced, ""));
        assert!(!license_policy_allows(&enforced, "GPL-3.0-only"));
    }

    #[test]
    fn storage_retention_uses_compact_duration_labels() {
        assert_eq!(format_duration(604_800), "7 days");
        assert_eq!(format_duration(3_600), "1 hour");
        assert_eq!(format_duration(90), "90 seconds");
    }

    #[test]
    fn model_index_polling_is_bounded_and_retries_missing_snapshots_promptly() {
        assert_eq!(model_index_poll_interval_seconds(0, true), 5);
        assert_eq!(model_index_poll_interval_seconds(2, true), 5);
        assert_eq!(model_index_poll_interval_seconds(300, true), 300);
        assert_eq!(model_index_poll_interval_seconds(7_200, true), 3_600);
        assert_eq!(model_index_poll_interval_seconds(120, false), 30);
    }

    #[test]
    fn signed_index_actions_require_an_exact_persistent_installation_identity() {
        let entry = api::ModelIndexEntry {
            id: "tiny-q4".to_string(),
            name: "Tiny Q4".to_string(),
            description: "A signed catalog fixture.".to_string(),
            download_url: Some("https://example.invalid/tiny-q4.gguf".to_string()),
            filename: "tiny-q4.gguf".to_string(),
            format: "gguf".to_string(),
            size_bytes: 4,
            sha256: "ab".repeat(32),
            files: Vec::new(),
            license: "Apache-2.0".to_string(),
            family: Some("qwen2".to_string()),
            parameter_count: None,
            quantization: Some("Q4_0".to_string()),
            tags: Vec::new(),
            downloadable: true,
            blocking_reasons: Vec::new(),
        };
        assert_eq!(
            model_index_local_state(&entry, &[]),
            ModelIndexLocalState::Missing
        );

        let provenance = api::ModelProvenance {
            acquisition: "download".to_string(),
            model_index_id: Some(entry.id.clone()),
            source_url: None,
            source_host: Some("huggingface.co".to_string()),
            sha256: entry.sha256.clone(),
            file_count: None,
            license: Some("apache-2.0".to_string()),
            installed_at: 1,
            last_verified_at: None,
            integrity_mismatch_at: None,
        };
        let installed = api::ModelCatalogEntry {
            id: entry.filename.clone(),
            name: "Tiny Q4".to_string(),
            kind: "file".to_string(),
            format: "gguf".to_string(),
            size_bytes: entry.size_bytes,
            size_complete: true,
            modified_at: Some(1),
            active: false,
            provenance: Some(provenance.clone()),
            provenance_error: None,
        };
        assert_eq!(
            model_index_local_state(&entry, std::slice::from_ref(&installed)),
            ModelIndexLocalState::Verified
        );

        let mut coreml_entry = entry.clone();
        coreml_entry.filename = "tiny.mlmodel".to_string();
        coreml_entry.format = "mlmodel".to_string();
        let mut coreml_install = installed.clone();
        coreml_install.id = coreml_entry.filename.clone();
        coreml_install.kind = "file".to_string();
        coreml_install.format = "coreml".to_string();
        assert_eq!(
            model_index_local_state(&coreml_entry, &[coreml_install]),
            ModelIndexLocalState::Verified
        );

        let mut unaliased = installed.clone();
        unaliased.provenance.as_mut().unwrap().model_index_id = None;
        assert_eq!(
            model_index_local_state(&entry, &[unaliased]),
            ModelIndexLocalState::Conflict
        );

        let mut quarantined = installed.clone();
        quarantined
            .provenance
            .as_mut()
            .unwrap()
            .integrity_mismatch_at = Some(2);
        assert_eq!(
            model_index_local_state(&entry, &[quarantined]),
            ModelIndexLocalState::Conflict
        );

        let aliased_elsewhere = api::ModelCatalogEntry {
            id: "other.gguf".to_string(),
            provenance: Some(provenance),
            ..installed
        };
        assert_eq!(
            model_index_local_state(&entry, std::slice::from_ref(&aliased_elsewhere)),
            ModelIndexLocalState::Upgradable
        );
        assert!(!model_index_upgrade_source_is_active(
            &entry,
            std::slice::from_ref(&aliased_elsewhere)
        ));

        let mut active_upgrade = aliased_elsewhere.clone();
        active_upgrade.active = true;
        assert!(model_index_upgrade_source_is_active(
            &entry,
            std::slice::from_ref(&active_upgrade)
        ));
        let occupied_destination = api::ModelCatalogEntry {
            id: entry.filename.clone(),
            active: false,
            provenance: None,
            ..active_upgrade.clone()
        };
        assert_eq!(
            model_index_local_state(&entry, &[active_upgrade, occupied_destination]),
            ModelIndexLocalState::Conflict
        );
    }

    #[test]
    fn diagnostics_formatters_are_compact_and_explicit() {
        assert_eq!(format_duration_seconds(0), "0s");
        assert_eq!(format_duration_seconds(3_661), "1h 1m");
        assert_eq!(format_duration_seconds(176_400), "2d 1h");
        assert_eq!(format_optional_bytes(0), "Unavailable");
        assert_eq!(format_optional_bytes(1_024), "1.0 KiB");
        assert_eq!(format_load_phase("loading"), "Loading");
        assert_eq!(format_load_phase("future"), "Unknown");
        assert_eq!(format_percent(0.125), "12.5%");
    }

    #[test]
    fn first_run_guidance_routes_users_to_the_actionable_drawer() {
        let offline = empty_state_view(&ConnectionState::Offline);
        assert_eq!(offline.title, "Bloom server is unavailable");
        assert_eq!(
            offline.action,
            Some((EmptyStateAction::Settings, "Check connection"))
        );

        let incompatible = empty_state_view(&ConnectionState::Incompatible {
            message: "unsupported readiness contract".to_string(),
        });
        assert_eq!(incompatible.title, "Incompatible Bloom server");
        assert!(incompatible.body.contains("unsupported readiness contract"));
        assert_eq!(
            incompatible.action,
            Some((EmptyStateAction::Settings, "Check server version"))
        );

        let authentication = empty_state_view(&ConnectionState::AuthenticationRequired {
            message: "HTTP 401 (Request ID: request-42)".to_string(),
        });
        assert_eq!(authentication.title, "API key required");
        assert!(authentication.body.contains("request-42"));
        assert_eq!(
            authentication.action,
            Some((EmptyStateAction::Settings, "Update API key"))
        );

        let no_model = empty_state_view(&ConnectionState::NoModel);
        assert_eq!(no_model.title, "Choose a model to begin");
        assert_eq!(
            no_model.action,
            Some((EmptyStateAction::Models, "Open models"))
        );

        let ready = empty_state_view(&ConnectionState::Ready {
            model: "tiny.gguf".to_string(),
            supports_vision: false,
            supports_text_input: true,
            supports_audio_input: false,
            supports_generation: true,
            supports_embeddings: false,
            supports_rerank: false,
            context_window: Some(4_096),
        });
        assert!(ready.body.contains("tiny.gguf is ready"));
        assert_eq!(ready.action, None);

        let encoder = empty_state_view(&ConnectionState::Ready {
            model: "encoder".to_string(),
            supports_vision: false,
            supports_text_input: true,
            supports_audio_input: false,
            supports_generation: false,
            supports_embeddings: true,
            supports_rerank: true,
            context_window: Some(256),
        });
        assert_eq!(encoder.title, "Embedding model ready");
        assert!(encoder.body.contains("rather than text generation"));
    }

    #[test]
    fn generation_timing_uses_compact_stable_labels() {
        assert_eq!(format_generation_millis(420), "420 ms");
        assert_eq!(format_generation_millis(2_345), "2.35 s");
        assert_eq!(format_generation_millis(61_000), "1m 1s");
        assert_eq!(generation_outcome_label(GenerationOutcome::Completed), None);
        assert_eq!(
            generation_outcome_label(GenerationOutcome::Stopped),
            Some("Stopped")
        );
    }

    #[test]
    fn context_budget_validation_and_warning_are_actionable() {
        assert!(validate_context_reservation(1_024, Some(4_096)).is_ok());
        assert!(validate_context_reservation(4_096, Some(4_096)).is_err());
        assert!(validate_context_reservation(32_768, None).is_ok());

        let mut response = DisplayMessage::assistant("Answer");
        response.generation = Some(GenerationStats::observed(
            GenerationOutcome::Completed,
            100.0,
            Some(20.0),
            Some(700),
            Some(100),
            Some("tiny.gguf".to_string()),
        ));
        let warning = conversation_context_status(&[response], Some(1_000), 300).unwrap();
        assert!(warning.0);
        assert!(warning.1.contains("800 / 1000 context tokens"));
        assert!(warning.1.contains("may exceed"));
        assert!(warning.1.contains("Use Continue"));
    }

    #[test]
    fn provenance_summary_is_compact_and_explicit_about_missing_license() {
        let provenance = api::ModelProvenance {
            acquisition: "import".to_string(),
            sha256: "ab".repeat(32),
            ..api::ModelProvenance::default()
        };

        assert_eq!(
            model_provenance_summary(&provenance),
            "Verified import · License: Not reported · SHA-256: abababababab…"
        );
    }

    #[test]
    fn integrity_status_distinguishes_verified_files_from_mismatches() {
        let mut status = api::ModelIntegrityStatus {
            phase: "complete".to_string(),
            matches_expected: Some(true),
            ..api::ModelIntegrityStatus::default()
        };
        assert_eq!(integrity_phase_label(&status), "Checksum verified");

        status.matches_expected = Some(false);
        assert_eq!(integrity_phase_label(&status), "Checksum mismatch");
    }

    #[test]
    fn model_preflight_details_use_compact_english_labels() {
        let manifest = api::ModelManifestSummary {
            primary_dtype: "q4".to_string(),
            quantization: Some("q4_k_m".to_string()),
            quantization_bits: Some(4),
            ..api::ModelManifestSummary::default()
        };

        assert_eq!(format_model_precision(&manifest), "Q4_K_M · 4-bit");
        assert_eq!(format_optional_count(Some(4096), "tokens"), "4096 tokens");
        assert_eq!(format_optional_count(None, "tokens"), "Unknown");
        assert_eq!(format_parameter_count(Some(7_000_000_000)), "7.00B");
        assert_eq!(
            format_model_tasks(&["embedding".to_string(), "rerank".to_string()]),
            "Embedding + Rerank"
        );
        assert_eq!(
            format_model_tasks(&["generation".to_string()]),
            "Generation"
        );
    }

    #[test]
    fn conversation_model_changes_require_an_exact_explicit_acknowledgement() {
        let mut store = ConversationStore::default();
        store.append_user("Hello".into());
        let assistant_index = store.append_assistant_placeholder();
        store.active_mut().messages[assistant_index].content = "Hi".into();
        store
            .set_assistant_generation_stats(
                store.active_id,
                assistant_index,
                GenerationStats::observed(
                    GenerationOutcome::Completed,
                    10.0,
                    None,
                    None,
                    None,
                    Some("model-a".to_string()),
                ),
            )
            .unwrap();

        assert!(conversation_model_transition(&store, Some("model-a")).is_none());
        let transition = conversation_model_transition(&store, Some("model-b")).unwrap();
        assert_eq!(transition.previous_model, "model-a");
        assert_eq!(transition.current_model, "model-b");
        assert!(unconfirmed_conversation_model_transition(
            &store,
            Some("model-b"),
            Some(&transition)
        )
        .is_none());
        assert!(unconfirmed_conversation_model_transition(
            &store,
            Some("model-c"),
            Some(&transition)
        )
        .is_some());
        assert!(
            conversation_model_transition(&ConversationStore::default(), Some("model-b")).is_none()
        );
    }

    #[test]
    fn inventory_drift_uses_readable_english_labels() {
        assert_eq!(format_inventory_drift_status("unexpected"), "Unexpected");
        assert_eq!(format_inventory_drift_status("future"), "Unknown");
        assert_eq!(format_inventory_drift_severity("blocking"), "Blocking");
        assert_eq!(
            format_inventory_changes(&[
                "sha256".to_string(),
                "source_lock".to_string(),
                "future_field".to_string(),
            ]),
            "SHA-256, Source lock, future field"
        );
    }

    #[test]
    fn multimodal_prompt_preserves_roles_and_opens_assistant_turn() {
        let prompt = api::format_multimodal_prompt(&[
            ChatMessage {
                role: "system".into(),
                content: "Be concise.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "What is shown?".into(),
            },
        ])
        .unwrap();

        assert_eq!(
            prompt,
            "system: Be concise.\nuser: What is shown?\nassistant:"
        );
    }

    #[test]
    fn readiness_controls_model_and_vision_availability() {
        let ready = ConnectionState::from_readiness(Readiness {
            status: "ready".into(),
            model: "vision-model".into(),
            input_modalities: vec!["Text".into(), "Vision".into()],
            model_tasks: vec!["generation".into()],
            context_window: Some(8_192),
            ..Readiness::default()
        });
        assert!(ready.can_chat());
        assert_eq!(ready.active_model(), Some("vision-model"));
        assert!(ready.supports_vision());
        assert!(!ready.can_embed());
        assert!(!ready.can_rerank());
        assert!(!ready.can_transcribe());
        assert_eq!(ready.context_window(), Some(8_192));

        let speech = ConnectionState::from_readiness(Readiness {
            status: "ready".into(),
            model: "qwen-asr".into(),
            input_modalities: vec!["audio".into()],
            model_tasks: vec!["generation".into()],
            context_window: Some(1_024),
            ..Readiness::default()
        });
        assert!(!speech.can_chat());
        assert!(speech.can_transcribe());
        assert_eq!(speech.view().1, "Ready · Transcription · qwen-asr");

        let encoder = ConnectionState::from_readiness(Readiness {
            status: "ready".into(),
            model: "encoder-model".into(),
            model_tasks: vec!["embedding".into(), "rerank".into()],
            context_window: Some(256),
            ..Readiness::default()
        });
        assert!(!encoder.can_chat());
        assert!(encoder.can_embed());
        assert!(encoder.can_rerank());
        assert_eq!(encoder.context_window(), Some(256));

        let empty = ConnectionState::from_readiness(Readiness {
            status: "not_ready".into(),
            ..Readiness::default()
        });
        assert_eq!(empty, ConnectionState::NoModel);
        assert_eq!(empty.active_model(), None);

        let failed = ConnectionState::from_readiness(Readiness {
            status: "not_ready".into(),
            load_error: Some("Model load failed.".into()),
            ..Readiness::default()
        });
        assert!(matches!(failed, ConnectionState::LoadFailed { .. }));
    }

    #[test]
    fn embedding_workspace_formats_only_bounded_vector_summaries() {
        assert_eq!(parse_optional_embedding_dimensions(""), Ok(None));
        assert_eq!(parse_optional_embedding_dimensions(" 384 "), Ok(Some(384)));
        assert!(parse_optional_embedding_dimensions("0").is_err());
        assert!(parse_optional_embedding_dimensions("16385").is_err());
        assert!(parse_optional_embedding_dimensions("wide").is_err());

        assert!((embedding_vector_norm(&[0.6, 0.8]) - 1.0).abs() < 1e-6);
        assert_eq!(
            embedding_vector_preview(&[1.0, -0.5]),
            "[1.000000, -0.500000]"
        );
        assert_eq!(
            embedding_vector_preview(&[0.0; 9]),
            "[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, …]"
        );
        assert_eq!(encoder_input_preview("short input"), "short input");
        assert_eq!(
            encoder_input_preview(&"x".repeat(161)),
            format!("{}…", "x".repeat(160))
        );
    }

    #[test]
    fn image_mime_falls_back_only_for_untyped_known_extensions() {
        assert_eq!(
            supported_image_mime("photo.JPG", ""),
            Some("image/jpeg".into())
        );
        assert_eq!(
            supported_image_mime("plot.png", "application/octet-stream"),
            Some("image/png".into())
        );
        assert_eq!(supported_image_mime("photo.jpg", "text/plain"), None);
        assert_eq!(supported_image_mime("photo.webp", ""), None);
    }

    #[test]
    fn speech_segments_preserve_cjk_flow_and_separate_ascii_words() {
        let mut transcript = "hello".to_string();
        append_transcript_segment(&mut transcript, "world").unwrap();
        assert_eq!(transcript, "hello world");

        let mut transcript = "你好".to_string();
        append_transcript_segment(&mut transcript, "世界").unwrap();
        assert_eq!(transcript, "你好世界");
        append_transcript_segment(&mut transcript, "  ").unwrap();
        assert_eq!(transcript, "你好世界");
    }
}
