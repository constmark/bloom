//! Browser-local persistence for UI settings and conversations.

use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::api::{ChatOptions, ConnConfig};
use crate::chat::ConversationStore;

const CONNECTION_STORAGE_KEY: &str = "bloom-ui-conn";
const LEGACY_CONNECTION_STORAGE_KEY: &str = "bloom-web-conn";
const SESSION_API_KEY_STORAGE_KEY: &str = "bloom-ui-api-key-v1";
const CONVERSATION_STORAGE_KEY: &str = "bloom-ui-conversations-v2";
const LEGACY_CONVERSATION_STORAGE_KEY: &str = "bloom-ui-conversations-v1";
const CONVERSATION_STORAGE_VERSION: u32 = 2;
const CONVERSATION_STORAGE_OBJECT: &str = "bloom.conversation_store";
const GENERATION_STORAGE_KEY: &str = "bloom-ui-generation-v1";
pub const CONVERSATION_RECOVERY_FILENAME: &str = "bloom-conversations-recovery.txt";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredConnection {
    base_url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    api_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remember_api_key: Option<bool>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredConversations {
    schema_version: u32,
    object: String,
    store: ConversationStore,
}

#[derive(Serialize)]
struct StoredConversationsRef<'a> {
    schema_version: u32,
    object: &'static str,
    store: &'a ConversationStore,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConversationStorageNotice {
    pub message: String,
    pub recovery_text: Option<Rc<str>>,
}

impl ConversationStorageNotice {
    pub fn blocks_conversation_writes(&self) -> bool {
        self.recovery_text.is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConversationLoad {
    pub store: ConversationStore,
    pub notice: Option<ConversationStorageNotice>,
}

pub fn load_connection() -> ConnConfig {
    let Some(storage) = local_storage() else {
        return ConnConfig::default();
    };
    if storage
        .get_item(CONNECTION_STORAGE_KEY)
        .ok()
        .flatten()
        .is_none()
        && let Ok(Some(legacy)) = storage.get_item(LEGACY_CONNECTION_STORAGE_KEY)
    {
        let _ = storage.set_item(CONNECTION_STORAGE_KEY, &legacy);
        let _ = storage.remove_item(LEGACY_CONNECTION_STORAGE_KEY);
    }
    let Ok(Some(text)) = storage.get_item(CONNECTION_STORAGE_KEY) else {
        return ConnConfig::default();
    };
    let session = session_storage();
    let session_api_key = session
        .as_ref()
        .and_then(|session| session.get_item(SESSION_API_KEY_STORAGE_KEY).ok().flatten());
    let Some(config) = decode_connection_settings(&text, session_api_key.as_deref()) else {
        return ConnConfig::default();
    };
    if (config.remember_api_key || (config.api_key.is_empty() && session_api_key.is_some()))
        && let Some(session) = session.as_ref()
    {
        let _ = session.remove_item(SESSION_API_KEY_STORAGE_KEY);
    }
    if let Ok(normalized) = encode_connection_settings(&config)
        && normalized != text
    {
        let _ = storage.set_item(CONNECTION_STORAGE_KEY, &normalized);
    }
    config
}

pub fn save_connection(config: &ConnConfig) -> Result<(), String> {
    config.validate()?;
    let storage = local_storage().ok_or("browser-local storage is unavailable")?;
    let text = encode_connection_settings(config)?;
    let session = session_storage();
    let previous_session_api_key = if config.remember_api_key {
        None
    } else {
        session
            .as_ref()
            .map(|session| {
                session
                    .get_item(SESSION_API_KEY_STORAGE_KEY)
                    .map_err(|error| format!("failed to read session API key: {error:?}"))
            })
            .transpose()?
            .flatten()
    };

    if !config.remember_api_key {
        match (session.as_ref(), config.api_key.is_empty()) {
            (Some(session), false) => session
                .set_item(SESSION_API_KEY_STORAGE_KEY, &config.api_key)
                .map_err(|error| format!("failed to save session API key: {error:?}"))?,
            (Some(session), true) => session
                .remove_item(SESSION_API_KEY_STORAGE_KEY)
                .map_err(|error| format!("failed to clear session API key: {error:?}"))?,
            (None, false) => {
                return Err(
                    "Browser session storage is unavailable. Enable Remember API key to persist this credential, or allow session storage."
                        .to_string(),
                );
            }
            (None, true) => {}
        }
    }

    if let Err(error) = storage.set_item(CONNECTION_STORAGE_KEY, &text) {
        restore_session_api_key(session.as_ref(), previous_session_api_key.as_deref());
        return Err(format!("failed to save connection settings: {error:?}"));
    }
    if config.remember_api_key
        && let Some(session) = session.as_ref()
    {
        let _ = session.remove_item(SESSION_API_KEY_STORAGE_KEY);
    }
    Ok(())
}

fn decode_connection_settings(text: &str, session_api_key: Option<&str>) -> Option<ConnConfig> {
    let stored = serde_json::from_str::<StoredConnection>(text).ok()?;
    let remember_api_key = stored
        .remember_api_key
        .unwrap_or(!stored.api_key.is_empty());
    let mut config = ConnConfig {
        base_url: stored.base_url,
        api_key: if remember_api_key {
            stored.api_key
        } else {
            session_api_key.unwrap_or_default().to_string()
        },
        remember_api_key,
    };
    if config.validate().is_err() {
        config.api_key.clear();
        config.validate().ok()?;
    }
    Some(config)
}

fn encode_connection_settings(config: &ConnConfig) -> Result<String, String> {
    config.validate()?;
    serde_json::to_string(&StoredConnection {
        base_url: config.base_url.clone(),
        api_key: if config.remember_api_key {
            config.api_key.clone()
        } else {
            String::new()
        },
        remember_api_key: Some(config.remember_api_key),
    })
    .map_err(|error| format!("failed to encode connection settings: {error}"))
}

fn restore_session_api_key(storage: Option<&web_sys::Storage>, api_key: Option<&str>) {
    let Some(storage) = storage else {
        return;
    };
    if let Some(api_key) = api_key {
        let _ = storage.set_item(SESSION_API_KEY_STORAGE_KEY, api_key);
    } else {
        let _ = storage.remove_item(SESSION_API_KEY_STORAGE_KEY);
    }
}

pub fn load_generation() -> ChatOptions {
    let Some(storage) = local_storage() else {
        return ChatOptions::default();
    };
    let Ok(Some(text)) = storage.get_item(GENERATION_STORAGE_KEY) else {
        return ChatOptions::default();
    };
    decode_generation_options(&text).unwrap_or_default()
}

pub fn save_generation(options: &ChatOptions) -> Result<(), String> {
    options.validate()?;
    let storage = local_storage().ok_or("browser-local storage is unavailable")?;
    let text = serde_json::to_string(options)
        .map_err(|error| format!("failed to encode generation settings: {error}"))?;
    storage
        .set_item(GENERATION_STORAGE_KEY, &text)
        .map_err(|error| format!("failed to save generation settings: {error:?}"))
}

fn decode_generation_options(text: &str) -> Option<ChatOptions> {
    serde_json::from_str::<ChatOptions>(text)
        .ok()
        .filter(|options| options.validate().is_ok())
}

pub fn load_conversations() -> ConversationLoad {
    let Some(storage) = local_storage() else {
        return ConversationLoad {
            store: ConversationStore::default(),
            notice: Some(ConversationStorageNotice {
                message: "Browser-local storage is unavailable. Conversations will not persist after this page closes."
                    .to_string(),
                recovery_text: None,
            }),
        };
    };
    match storage.get_item(CONVERSATION_STORAGE_KEY) {
        Ok(Some(text)) => match decode_stored_conversations(&text) {
            Ok(store) => ConversationLoad {
                store,
                notice: None,
            },
            Err(message) => corrupt_conversation_load(text, message),
        },
        Ok(None) => load_legacy_conversations(&storage),
        Err(error) => ConversationLoad {
            store: ConversationStore::default(),
            notice: Some(ConversationStorageNotice {
                message: format!("Bloom could not read browser-local conversations: {error:?}"),
                recovery_text: None,
            }),
        },
    }
}

pub fn save_conversations(store: &ConversationStore) -> Result<(), String> {
    let storage = local_storage().ok_or("browser-local storage is unavailable")?;
    let text = encode_stored_conversations(store)?;
    storage
        .set_item(CONVERSATION_STORAGE_KEY, &text)
        .map_err(|error| format!("failed to save conversations: {error:?}"))?;
    let _ = storage.remove_item(LEGACY_CONVERSATION_STORAGE_KEY);
    Ok(())
}

fn load_legacy_conversations(storage: &web_sys::Storage) -> ConversationLoad {
    let text = match storage.get_item(LEGACY_CONVERSATION_STORAGE_KEY) {
        Ok(Some(text)) => text,
        Ok(None) => {
            return ConversationLoad {
                store: ConversationStore::default(),
                notice: None,
            };
        }
        Err(error) => {
            return ConversationLoad {
                store: ConversationStore::default(),
                notice: Some(ConversationStorageNotice {
                    message: format!(
                        "Bloom could not read legacy browser-local conversations: {error:?}"
                    ),
                    recovery_text: None,
                }),
            };
        }
    };
    let store = match decode_legacy_conversations(&text) {
        Ok(store) => store,
        Err(error) => {
            return corrupt_conversation_load(
                text,
                format!("legacy conversation data is invalid: {error}"),
            );
        }
    };
    let notice = save_conversations(&store)
        .err()
        .map(|message| ConversationStorageNotice {
            message: format!(
                "Legacy conversations were loaded, but Bloom could not migrate them to the versioned store: {message}"
            ),
            recovery_text: None,
        });
    ConversationLoad { store, notice }
}

fn corrupt_conversation_load(text: String, detail: String) -> ConversationLoad {
    ConversationLoad {
        store: ConversationStore::default(),
        notice: Some(ConversationStorageNotice {
            message: format!(
                "Saved conversations could not be decoded and have not been overwritten. Download the recovery copy before starting fresh. {detail}"
            ),
            recovery_text: Some(Rc::from(text)),
        }),
    }
}

fn encode_stored_conversations(store: &ConversationStore) -> Result<String, String> {
    store.validate_persisted()?;
    serde_json::to_string(&StoredConversationsRef {
        schema_version: CONVERSATION_STORAGE_VERSION,
        object: CONVERSATION_STORAGE_OBJECT,
        store,
    })
    .map_err(|error| format!("failed to encode conversations: {error}"))
}

fn decode_legacy_conversations(text: &str) -> Result<ConversationStore, String> {
    let store = serde_json::from_str::<ConversationStore>(text)
        .map_err(|error| format!("invalid legacy conversation data: {error}"))?
        .normalize();
    store.validate_persisted()?;
    Ok(store)
}

fn decode_stored_conversations(text: &str) -> Result<ConversationStore, String> {
    let stored = serde_json::from_str::<StoredConversations>(text)
        .map_err(|error| format!("invalid versioned conversation data: {error}"))?;
    if stored.schema_version != CONVERSATION_STORAGE_VERSION
        || stored.object != CONVERSATION_STORAGE_OBJECT
    {
        return Err(format!(
            "unsupported conversation store identity (version {}, object {:?})",
            stored.schema_version, stored.object
        ));
    }
    let store = stored.store;
    store.validate_persisted()?;
    Ok(store)
}

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window().and_then(|window| window.local_storage().ok().flatten())
}

fn session_storage() -> Option<web_sys::Storage> {
    web_sys::window().and_then(|window| window.session_storage().ok().flatten())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_connection_credentials_default_to_session_only_storage() {
        let config = ConnConfig {
            base_url: "https://example.com/bloom".to_string(),
            api_key: "secret-token".to_string(),
            remember_api_key: false,
        };

        let encoded = encode_connection_settings(&config).unwrap();
        assert!(!encoded.contains("\"api_key\":"));
        assert!(!encoded.contains("secret-token"));
        assert!(encoded.contains("\"remember_api_key\":false"));
        assert_eq!(
            decode_connection_settings(&encoded, Some("secret-token")),
            Some(config)
        );
    }

    #[test]
    fn explicitly_remembered_connection_credentials_round_trip_locally() {
        let config = ConnConfig {
            base_url: "http://127.0.0.1:3000".to_string(),
            api_key: "remembered-token".to_string(),
            remember_api_key: true,
        };

        let encoded = encode_connection_settings(&config).unwrap();
        assert!(encoded.contains("remembered-token"));
        assert!(encoded.contains("\"remember_api_key\":true"));
        assert_eq!(
            decode_connection_settings(&encoded, Some("stale-session-token")),
            Some(config)
        );
    }

    #[test]
    fn legacy_connection_credentials_remain_available_until_policy_is_changed() {
        let legacy = r#"{"base_url":"http://localhost:3000","api_key":"legacy-token"}"#;
        let decoded = decode_connection_settings(legacy, None).unwrap();

        assert_eq!(decoded.api_key, "legacy-token");
        assert!(decoded.remember_api_key);
        assert!(
            encode_connection_settings(&decoded)
                .unwrap()
                .contains("\"remember_api_key\":true")
        );
    }

    #[test]
    fn connection_storage_rejects_unknown_fields_and_discards_invalid_keys() {
        let unknown =
            r#"{"base_url":"http://localhost:3000","remember_api_key":false,"extra":true}"#;
        assert!(decode_connection_settings(unknown, None).is_none());

        let session_only = r#"{"base_url":"http://localhost:3000","remember_api_key":false}"#;
        let decoded = decode_connection_settings(session_only, Some("contains space")).unwrap();
        assert_eq!(decoded.base_url, "http://localhost:3000");
        assert!(decoded.api_key.is_empty());
        assert!(!decoded.remember_api_key);
    }

    #[test]
    fn versioned_conversation_storage_round_trips_and_normalizes() {
        let mut store = ConversationStore::default();
        store.append_user("Persist me".into());
        let encoded = encode_stored_conversations(&store).unwrap();
        let decoded = decode_stored_conversations(&encoded).unwrap();

        assert_eq!(decoded, store);
        assert!(encoded.contains("bloom.conversation_store"));
        assert!(encoded.contains("\"schema_version\":2"));
    }

    #[test]
    fn versioned_conversation_storage_rejects_unknown_identity_and_fields() {
        let wrong_version = r#"{"schema_version":1,"object":"bloom.conversation_store","store":{"active_id":1,"next_id":2,"conversations":[]}}"#;
        let wrong_object = r#"{"schema_version":2,"object":"other","store":{"active_id":1,"next_id":2,"conversations":[]}}"#;
        let unknown_field = r#"{"schema_version":2,"object":"bloom.conversation_store","store":{"active_id":1,"next_id":2,"conversations":[]},"extra":true}"#;

        assert!(decode_stored_conversations(wrong_version).is_err());
        assert!(decode_stored_conversations(wrong_object).is_err());
        assert!(decode_stored_conversations(unknown_field).is_err());
    }

    #[test]
    fn versioned_conversation_storage_rejects_invalid_store_state() {
        let empty = r#"{"schema_version":2,"object":"bloom.conversation_store","store":{"active_id":1,"next_id":2,"conversations":[]}}"#;
        let missing_active = r#"{"schema_version":2,"object":"bloom.conversation_store","store":{"active_id":2,"next_id":2,"conversations":[{"id":1,"title":"Chat","messages":[]}]}}"#;
        let reused_next_id = r#"{"schema_version":2,"object":"bloom.conversation_store","store":{"active_id":1,"next_id":1,"conversations":[{"id":1,"title":"Chat","messages":[]}]}}"#;

        assert!(decode_stored_conversations(empty).is_err());
        assert!(decode_stored_conversations(missing_active).is_err());
        assert!(decode_stored_conversations(reused_next_id).is_err());
    }

    #[test]
    fn corrupt_conversation_load_retains_the_original_recovery_text() {
        let raw = "{not-json}".to_string();
        let load = corrupt_conversation_load(raw.clone(), "invalid JSON".to_string());
        let notice = load.notice.expect("corrupt data should produce a notice");

        assert_eq!(load.store, ConversationStore::default());
        assert!(notice.blocks_conversation_writes());
        assert_eq!(notice.recovery_text.as_deref(), Some(raw.as_str()));
    }

    #[test]
    fn legacy_conversation_storage_decodes_without_new_message_metadata() {
        let legacy = r#"{"active_id":1,"next_id":2,"conversations":[{"id":1,"title":"Legacy","messages":[{"role":"user","content":"Hello"}]}]}"#;

        let store = decode_legacy_conversations(legacy).unwrap();

        assert_eq!(store.active().messages[0].content, "Hello");
        assert!(!store.active().messages[0].attachment_unavailable);
    }

    #[test]
    fn generation_storage_accepts_valid_legacy_settings() {
        let options = decode_generation_options(
            r#"{"max_tokens":256,"temperature":0.4,"top_p":0.8,"seed":7,"stop_sequences":["END"],"system_prompt":"Be concise."}"#,
        )
        .unwrap();

        assert_eq!(options.max_tokens, 256);
        assert_eq!(options.seed, Some(7));
        assert_eq!(options.stop_sequences, vec!["END"]);
        assert_eq!(
            options.response_format,
            crate::api::ResponseFormatMode::Text
        );
    }

    #[test]
    fn generation_storage_rejects_invalid_persisted_settings() {
        assert!(decode_generation_options("{not-json}").is_none());
        assert!(
            decode_generation_options(
                r#"{"max_tokens":0,"temperature":0.7,"top_p":0.9,"system_prompt":""}"#
            )
            .is_none()
        );
        let oversized = serde_json::json!({
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "system_prompt": "x".repeat(crate::api::MAX_SYSTEM_PROMPT_CHARS + 1)
        });
        assert!(decode_generation_options(&oversized.to_string()).is_none());
    }
}
