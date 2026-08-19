//! Persistent, browser-local conversation state.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

const DEFAULT_TITLE: &str = "New chat";
const MAX_TITLE_CHARS: usize = 42;
pub const MAX_RENAMED_TITLE_CHARS: usize = 80;
const CONVERSATION_ARCHIVE_VERSION: u32 = 2;
const MIN_CONVERSATION_ARCHIVE_VERSION: u32 = 1;
const CONVERSATION_ARCHIVE_OBJECT: &str = "bloom.conversation_archive";
const MAX_ARCHIVE_CONVERSATIONS: usize = 1_000;
const MAX_ARCHIVE_MESSAGES: usize = 50_000;
const MAX_ARCHIVE_TITLE_CHARS: usize = 200;
const MAX_ARCHIVE_MESSAGE_CHARS: usize = 1_000_000;
const MAX_GENERATION_DURATION_MS: u64 = 30 * 24 * 60 * 60 * 1_000;
const MAX_GENERATION_TOKEN_COUNT: u64 = 1_000_000_000;
const MAX_GENERATION_MODEL_ID_CHARS: usize = 256;

/// Stable filename used for browser conversation exports.
pub const CONVERSATION_ARCHIVE_FILENAME: &str = "bloom-conversations.json";
/// Maximum accepted encoded archive size before JSON parsing.
pub const MAX_CONVERSATION_ARCHIVE_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DisplayMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub attachment_unavailable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationOutcome {
    Completed,
    Stopped,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationStats {
    pub outcome: GenerationOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_token_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
}

impl GenerationStats {
    pub fn observed(
        outcome: GenerationOutcome,
        elapsed_ms: f64,
        first_token_ms: Option<f64>,
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        model: Option<String>,
    ) -> Self {
        let elapsed_ms = if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
            elapsed_ms
                .round()
                .clamp(1.0, MAX_GENERATION_DURATION_MS as f64) as u64
        } else {
            1
        };
        let first_token_ms = first_token_ms
            .filter(|milliseconds| milliseconds.is_finite() && *milliseconds >= 0.0)
            .map(|milliseconds| milliseconds.round().clamp(0.0, elapsed_ms as f64) as u64);
        let (prompt_tokens, completion_tokens) = match (prompt_tokens, completion_tokens) {
            (Some(prompt), Some(completion))
                if prompt <= MAX_GENERATION_TOKEN_COUNT
                    && completion <= MAX_GENERATION_TOKEN_COUNT =>
            {
                (Some(prompt), Some(completion))
            }
            _ => (None, None),
        };
        let model = model.filter(|model| valid_generation_model_id(model));
        Self {
            outcome,
            model,
            elapsed_ms,
            first_token_ms,
            prompt_tokens,
            completion_tokens,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.elapsed_ms == 0 || self.elapsed_ms > MAX_GENERATION_DURATION_MS {
            return Err("conversation contains an invalid generation duration".to_string());
        }
        if self
            .first_token_ms
            .is_some_and(|milliseconds| milliseconds > self.elapsed_ms)
        {
            return Err("conversation contains invalid first-token timing".to_string());
        }
        match (self.prompt_tokens, self.completion_tokens) {
            (Some(prompt), Some(completion))
                if prompt <= MAX_GENERATION_TOKEN_COUNT
                    && completion <= MAX_GENERATION_TOKEN_COUNT =>
            {
                // Both values are present and bounded.
            }
            (None, None) => {}
            _ => {
                return Err("conversation contains invalid generation token counts".to_string());
            }
        }
        if self
            .model
            .as_deref()
            .is_some_and(|model| !valid_generation_model_id(model))
        {
            return Err("conversation contains invalid generation model metadata".to_string());
        }
        Ok(())
    }

    pub fn tokens_per_second(&self) -> Option<f64> {
        let completion_tokens = self.completion_tokens?;
        let generation_ms = self
            .elapsed_ms
            .saturating_sub(self.first_token_ms.unwrap_or(0));
        (completion_tokens > 0 && generation_ms > 0)
            .then(|| completion_tokens as f64 * 1_000.0 / generation_ms as f64)
    }

    pub fn total_tokens(&self) -> Option<u64> {
        self.prompt_tokens?.checked_add(self.completion_tokens?)
    }
}

fn valid_generation_model_id(model: &str) -> bool {
    !model.is_empty()
        && model.trim() == model
        && model
            .chars()
            .take(MAX_GENERATION_MODEL_ID_CHARS + 1)
            .count()
            <= MAX_GENERATION_MODEL_ID_CHARS
        && !model.chars().any(char::is_control)
}

impl DisplayMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            attachment_unavailable: false,
            generation: None,
            execution_model: None,
            response_format: None,
        }
    }

    pub fn user_with_unavailable_attachment(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            attachment_unavailable: true,
            generation: None,
            execution_model: None,
            response_format: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            attachment_unavailable: false,
            generation: None,
            execution_model: None,
            response_format: None,
        }
    }
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    pub id: u64,
    pub title: String,
    pub messages: Vec<DisplayMessage>,
}

impl Conversation {
    fn new(id: u64) -> Self {
        Self {
            id,
            title: DEFAULT_TITLE.into(),
            messages: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationStore {
    pub active_id: u64,
    pub next_id: u64,
    pub conversations: Vec<Conversation>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConversationArchive {
    version: u32,
    object: String,
    active_conversation: usize,
    conversations: Vec<ArchivedConversation>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchivedConversation {
    title: String,
    messages: Vec<ArchivedMessage>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchivedMessage {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "is_false")]
    attachment_unavailable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetryPreparation {
    pub conversation_id: u64,
    pub assistant_index: usize,
    pub history: Vec<DisplayMessage>,
    pub rollback: EmptyGenerationRollback,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditPreparation {
    pub conversation_id: u64,
    pub assistant_index: usize,
    pub history: Vec<DisplayMessage>,
    pub rollback: EmptyGenerationRollback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationBranch {
    pub conversation_id: u64,
    pub title: String,
    pub message_count: usize,
    pub contains_unavailable_attachment: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationContinuation {
    pub conversation_id: u64,
    pub title: String,
    pub retained_message_count: usize,
    pub omitted_message_count: usize,
    pub contains_unavailable_attachment: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmptyGenerationRollback {
    RemovePlaceholder,
    RestoreResponse(DisplayMessage),
    RestoreEditedTurn {
        user_index: usize,
        previous_user: DisplayMessage,
        previous_response: Option<Box<DisplayMessage>>,
        previous_title: String,
    },
}

impl Default for ConversationStore {
    fn default() -> Self {
        Self {
            active_id: 1,
            next_id: 2,
            conversations: vec![Conversation::new(1)],
        }
    }
}

impl ConversationStore {
    pub fn normalize(mut self) -> Self {
        if self.conversations.is_empty() {
            return Self::default();
        }
        if !self
            .conversations
            .iter()
            .any(|conversation| conversation.id == self.active_id)
        {
            self.active_id = self.conversations[0].id;
        }
        let next_available = self
            .conversations
            .iter()
            .map(|conversation| conversation.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.next_id = self.next_id.max(next_available);
        self
    }

    pub fn validate_persisted(&self) -> Result<(), String> {
        if self.conversations.is_empty() {
            return Err("conversation store contains no conversations".to_string());
        }
        if self.conversations.len() > MAX_ARCHIVE_CONVERSATIONS {
            return Err(format!(
                "conversation store contains more than {MAX_ARCHIVE_CONVERSATIONS} conversations"
            ));
        }
        if !self
            .conversations
            .iter()
            .any(|conversation| conversation.id == self.active_id)
        {
            return Err("conversation store has no active conversation".to_string());
        }

        let mut ids = HashSet::with_capacity(self.conversations.len());
        let mut max_id = 0_u64;
        let mut message_count = 0_usize;
        for conversation in &self.conversations {
            if conversation.id == 0 || !ids.insert(conversation.id) {
                return Err("conversation store contains invalid or duplicate IDs".to_string());
            }
            max_id = max_id.max(conversation.id);
            if conversation.title.trim().is_empty()
                || conversation.title.chars().count() > MAX_ARCHIVE_TITLE_CHARS
                || conversation.title.chars().any(char::is_control)
            {
                return Err("conversation store contains an invalid title".to_string());
            }
            message_count = message_count
                .checked_add(conversation.messages.len())
                .ok_or_else(|| "conversation store message count overflowed".to_string())?;
            if message_count > MAX_ARCHIVE_MESSAGES {
                return Err(format!(
                    "conversation store contains more than {MAX_ARCHIVE_MESSAGES} messages"
                ));
            }
            for message in &conversation.messages {
                if !matches!(message.role.as_str(), "user" | "assistant") {
                    return Err(format!(
                        "conversation store contains unsupported message role {:?}",
                        message.role
                    ));
                }
                if message.attachment_unavailable && message.role != "user" {
                    return Err(
                        "conversation store marks a non-user message as having an unavailable attachment"
                            .to_string(),
                    );
                }
                if let Some(generation) = &message.generation {
                    if message.role != "assistant" {
                        return Err(
                            "conversation assigns generation statistics to a non-assistant message"
                                .to_string(),
                        );
                    }
                    generation.validate()?;
                }
                if let Some(model) = message.execution_model.as_deref() {
                    if message.role != "assistant" || !valid_generation_model_id(model) {
                        return Err(
                            "conversation contains invalid execution model metadata".to_string()
                        );
                    }
                    if message
                        .generation
                        .as_ref()
                        .and_then(|generation| generation.model.as_deref())
                        .is_some_and(|generation_model| generation_model != model)
                    {
                        return Err(
                            "conversation contains inconsistent execution model metadata"
                                .to_string(),
                        );
                    }
                }
                if let Some(response_format) = &message.response_format
                    && (message.role != "assistant"
                        || !matches!(response_format.as_str(), "json_object" | "json_schema"))
                {
                    return Err(
                        "conversation contains invalid structured-output metadata".to_string()
                    );
                }
                if message.content.chars().count() > MAX_ARCHIVE_MESSAGE_CHARS {
                    return Err(format!(
                        "conversation store contains a message longer than {MAX_ARCHIVE_MESSAGE_CHARS} characters"
                    ));
                }
            }
        }
        if self.next_id <= max_id {
            return Err("conversation store has an invalid next ID".to_string());
        }
        Ok(())
    }

    pub fn active(&self) -> &Conversation {
        self.conversations
            .iter()
            .find(|conversation| conversation.id == self.active_id)
            .expect("normalized conversation store must have an active conversation")
    }

    pub fn active_mut(&mut self) -> &mut Conversation {
        let active_id = self.active_id;
        self.conversations
            .iter_mut()
            .find(|conversation| conversation.id == active_id)
            .expect("normalized conversation store must have an active conversation")
    }

    /// Return the most recent recorded execution model for the active conversation.
    ///
    /// Legacy and imported histories may not carry generation diagnostics, so
    /// callers must treat a missing value as unknown rather than inventing an
    /// association from the currently loaded runtime.
    pub fn active_execution_model(&self) -> Option<&str> {
        self.active().messages.iter().rev().find_map(|message| {
            if message.role != "assistant" {
                return None;
            }
            message.execution_model.as_deref().or_else(|| {
                message
                    .generation
                    .as_ref()
                    .and_then(|generation| generation.model.as_deref())
            })
        })
    }

    pub fn create(&mut self) -> Result<u64, String> {
        let (id, next_id) = self.allocate_conversation_id()?;
        self.next_id = next_id;
        self.conversations.insert(0, Conversation::new(id));
        self.active_id = id;
        Ok(id)
    }

    pub fn branch_active_at(&mut self, message_index: usize) -> Result<ConversationBranch, String> {
        let (id, next_id) = self.allocate_conversation_id()?;
        let (source_title, branch_message_count) = {
            let source = self.active();
            if message_index >= source.messages.len() {
                return Err("The selected branch point is no longer available.".to_string());
            }
            (source.title.clone(), message_index + 1)
        };
        let existing_message_count = self
            .conversations
            .iter()
            .try_fold(0_usize, |total, conversation| {
                total.checked_add(conversation.messages.len())
            })
            .ok_or_else(|| "Conversation message count overflowed.".to_string())?;
        let branched_message_count = existing_message_count
            .checked_add(branch_message_count)
            .ok_or_else(|| "Conversation message count overflowed.".to_string())?;
        if branched_message_count > MAX_ARCHIVE_MESSAGES {
            return Err(format!(
                "Branching would exceed the {MAX_ARCHIVE_MESSAGES}-message browser history limit."
            ));
        }
        let title = self.unique_branch_title(&source_title)?;
        let messages = self.active().messages[..branch_message_count].to_vec();
        let contains_unavailable_attachment = messages.iter().any(|message| {
            message.attachment_unavailable || looks_like_legacy_attachment_marker(&message.content)
        });
        self.next_id = next_id;
        self.active_id = id;
        self.conversations.insert(
            0,
            Conversation {
                id,
                title: title.clone(),
                messages,
            },
        );
        Ok(ConversationBranch {
            conversation_id: id,
            title,
            message_count: branch_message_count,
            contains_unavailable_attachment,
        })
    }

    /// Create a non-destructive conversation containing a recent history suffix.
    ///
    /// A continuation must begin at a user message after at least one omitted
    /// message. This keeps outbound history role-safe and makes the operation
    /// meaningfully different from an ordinary full-history branch.
    pub fn continue_active_from(
        &mut self,
        message_index: usize,
    ) -> Result<ConversationContinuation, String> {
        let (id, next_id) = self.allocate_conversation_id()?;
        let (source_title, retained_message_count, omitted_message_count) = {
            let source = self.active();
            let Some(message) = source.messages.get(message_index) else {
                return Err("The selected continuation point is no longer available.".to_string());
            };
            if message_index == 0 {
                return Err(
                    "Choose a later user message so the continuation omits earlier context."
                        .to_string(),
                );
            }
            if message.role != "user" {
                return Err("A continuation must begin with a user message.".to_string());
            }
            (
                source.title.clone(),
                source.messages.len() - message_index,
                message_index,
            )
        };
        let existing_message_count = self
            .conversations
            .iter()
            .try_fold(0_usize, |total, conversation| {
                total.checked_add(conversation.messages.len())
            })
            .ok_or_else(|| "Conversation message count overflowed.".to_string())?;
        let continued_message_count = existing_message_count
            .checked_add(retained_message_count)
            .ok_or_else(|| "Conversation message count overflowed.".to_string())?;
        if continued_message_count > MAX_ARCHIVE_MESSAGES {
            return Err(format!(
                "Continuing would exceed the {MAX_ARCHIVE_MESSAGES}-message browser history limit."
            ));
        }
        let title = self.unique_derived_title(&source_title, "continued")?;
        let messages = self.active().messages[message_index..].to_vec();
        let contains_unavailable_attachment = messages.iter().any(|message| {
            message.attachment_unavailable || looks_like_legacy_attachment_marker(&message.content)
        });
        self.next_id = next_id;
        self.active_id = id;
        self.conversations.insert(
            0,
            Conversation {
                id,
                title: title.clone(),
                messages,
            },
        );
        Ok(ConversationContinuation {
            conversation_id: id,
            title,
            retained_message_count,
            omitted_message_count,
            contains_unavailable_attachment,
        })
    }

    fn allocate_conversation_id(&self) -> Result<(u64, u64), String> {
        if self.conversations.len() >= MAX_ARCHIVE_CONVERSATIONS {
            return Err(format!(
                "Browser history cannot contain more than {MAX_ARCHIVE_CONVERSATIONS} conversations."
            ));
        }
        let id = self.next_id;
        if id == 0
            || self
                .conversations
                .iter()
                .any(|conversation| conversation.id == id)
        {
            return Err("Conversation ID allocation is no longer valid.".to_string());
        }
        let next_id = id
            .checked_add(1)
            .ok_or_else(|| "Conversation ID space is exhausted.".to_string())?;
        Ok((id, next_id))
    }

    fn unique_branch_title(&self, source_title: &str) -> Result<String, String> {
        self.unique_derived_title(source_title, "branch")
    }

    fn unique_derived_title(&self, source_title: &str, kind: &str) -> Result<String, String> {
        for sequence in 1..=MAX_ARCHIVE_CONVERSATIONS {
            let suffix = if sequence == 1 {
                format!(" ({kind})")
            } else {
                format!(" ({kind} {sequence})")
            };
            let base_chars = MAX_RENAMED_TITLE_CHARS.saturating_sub(suffix.chars().count());
            let base = source_title
                .chars()
                .take(base_chars)
                .collect::<String>()
                .trim_end()
                .to_string();
            let candidate = format!("{base}{suffix}");
            if !self
                .conversations
                .iter()
                .any(|conversation| conversation.title == candidate)
            {
                return Ok(candidate);
            }
        }
        Err(format!(
            "A unique conversation {kind} title could not be allocated."
        ))
    }

    pub fn select(&mut self, id: u64) -> bool {
        if self
            .conversations
            .iter()
            .any(|conversation| conversation.id == id)
        {
            self.active_id = id;
            true
        } else {
            false
        }
    }

    pub fn delete(&mut self, id: u64) -> Result<bool, String> {
        let Some(index) = self
            .conversations
            .iter()
            .position(|conversation| conversation.id == id)
        else {
            return Ok(false);
        };
        let replacement = (self.conversations.len() == 1)
            .then(|| self.allocate_conversation_id())
            .transpose()?;
        self.conversations.remove(index);
        if let Some((replacement_id, next_id)) = replacement {
            self.next_id = next_id;
            self.conversations.push(Conversation::new(replacement_id));
        }
        if self.active_id == id {
            self.active_id = self.conversations[0].id;
        }
        Ok(true)
    }

    pub fn rename(&mut self, id: u64, title: &str) -> Result<(), String> {
        if title.chars().any(char::is_control) {
            return Err("Conversation title cannot contain control characters.".to_string());
        }
        let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
        if title.is_empty() {
            return Err("Conversation title cannot be empty.".to_string());
        }
        if title.chars().count() > MAX_RENAMED_TITLE_CHARS {
            return Err(format!(
                "Conversation title cannot exceed {MAX_RENAMED_TITLE_CHARS} characters."
            ));
        }
        let Some(conversation) = self
            .conversations
            .iter_mut()
            .find(|conversation| conversation.id == id)
        else {
            return Err("Conversation no longer exists.".to_string());
        };
        conversation.title = title;
        Ok(())
    }

    pub fn append_user(&mut self, content: String) {
        self.append_user_message(DisplayMessage::user(content));
    }

    pub fn append_user_with_unavailable_attachment(&mut self, content: String) {
        self.append_user_message(DisplayMessage::user_with_unavailable_attachment(content));
    }

    fn append_user_message(&mut self, message: DisplayMessage) {
        let conversation = self.active_mut();
        if conversation.messages.is_empty() {
            conversation.title = title_from_message(&message.content);
        }
        conversation.messages.push(message);
    }

    pub fn append_assistant_placeholder(&mut self) -> usize {
        let conversation = self.active_mut();
        conversation
            .messages
            .push(DisplayMessage::assistant(String::new()));
        conversation.messages.len() - 1
    }

    pub fn set_assistant_generation_stats(
        &mut self,
        conversation_id: u64,
        assistant_index: usize,
        stats: GenerationStats,
    ) -> Result<(), String> {
        stats.validate()?;
        let message = self
            .conversations
            .iter_mut()
            .find(|conversation| conversation.id == conversation_id)
            .and_then(|conversation| conversation.messages.get_mut(assistant_index))
            .ok_or_else(|| "assistant response is no longer available".to_string())?;
        if message.role != "assistant" {
            return Err("generation statistics require an assistant response".to_string());
        }
        message.execution_model = stats.model.clone();
        message.generation = Some(stats);
        Ok(())
    }

    pub fn set_assistant_response_format(
        &mut self,
        conversation_id: u64,
        assistant_index: usize,
        response_format: &str,
    ) -> Result<(), String> {
        let response_format = match response_format {
            "text" => None,
            "json_object" | "json_schema" => Some(response_format.to_string()),
            _ => return Err("unsupported assistant response format".to_string()),
        };
        let message = self
            .conversations
            .iter_mut()
            .find(|conversation| conversation.id == conversation_id)
            .and_then(|conversation| conversation.messages.get_mut(assistant_index))
            .ok_or_else(|| "assistant response is no longer available".to_string())?;
        if message.role != "assistant" {
            return Err("response format metadata requires an assistant response".to_string());
        }
        message.response_format = response_format;
        Ok(())
    }

    pub fn retry_availability(&self) -> Result<(), &'static str> {
        let (_, user_message, _) = self.latest_user_turn()?;
        if user_message.attachment_unavailable
            || looks_like_legacy_attachment_marker(&user_message.content)
        {
            return Err(
                "This response used an image whose bytes are no longer available. Attach the image again to regenerate it.",
            );
        }
        Ok(())
    }

    pub fn edit_availability(&self) -> Result<(), &'static str> {
        let (_, user_message, _) = self.latest_user_turn()?;
        if user_message.attachment_unavailable
            || looks_like_legacy_attachment_marker(&user_message.content)
        {
            return Err(
                "This prompt used an image whose bytes are no longer available. Attach the image again to edit and resend it.",
            );
        }
        Ok(())
    }

    pub fn latest_editable_prompt(&self) -> Result<&str, &'static str> {
        self.edit_availability()?;
        self.latest_user_turn()
            .map(|(_, message, _)| message.content.as_str())
    }

    fn latest_user_turn(&self) -> Result<(usize, &DisplayMessage, bool), &'static str> {
        let messages = &self.active().messages;
        let Some(last) = messages.last() else {
            return Err("There is no user turn to change.");
        };
        let (user_index, has_response) = match last.role.as_str() {
            "assistant" => (messages.len().saturating_sub(2), true),
            "user" => (messages.len().saturating_sub(1), false),
            _ => return Err("Only the latest user turn can be changed."),
        };
        let Some(user_message) = messages.get(user_index) else {
            return Err("The user message for this turn is unavailable.");
        };
        if user_message.role != "user" {
            return Err("The user message for this turn is unavailable.");
        }
        Ok((user_index, user_message, has_response))
    }

    pub fn prepare_last_response_retry(&mut self) -> Result<RetryPreparation, String> {
        self.retry_availability().map_err(str::to_string)?;
        let (_, _, has_response) = self.latest_user_turn().map_err(str::to_string)?;
        let conversation_id = self.active_id;
        let conversation = self.active_mut();
        let rollback = if has_response {
            EmptyGenerationRollback::RestoreResponse(
                conversation
                    .messages
                    .pop()
                    .ok_or_else(|| "The assistant response is unavailable.".to_string())?,
            )
        } else {
            EmptyGenerationRollback::RemovePlaceholder
        };
        let history = conversation.messages.clone();
        let assistant_index = conversation.messages.len();
        conversation
            .messages
            .push(DisplayMessage::assistant(String::new()));
        Ok(RetryPreparation {
            conversation_id,
            assistant_index,
            history,
            rollback,
        })
    }

    pub fn prepare_last_prompt_edit(&mut self, content: String) -> Result<EditPreparation, String> {
        self.edit_availability().map_err(str::to_string)?;
        let (user_index, _, has_response) = self.latest_user_turn().map_err(str::to_string)?;
        if content.trim().is_empty() {
            return Err("Edited prompt cannot be empty.".to_string());
        }
        let conversation_id = self.active_id;
        let conversation = self.active_mut();
        let previous_title = conversation.title.clone();
        let previous_user = conversation.messages[user_index].clone();
        let previous_response = if has_response {
            Some(Box::new(conversation.messages.pop().ok_or_else(|| {
                "The assistant response is unavailable.".to_string()
            })?))
        } else {
            None
        };
        conversation.messages[user_index] = DisplayMessage::user(content);
        if user_index == 0 {
            conversation.title = title_from_message(&conversation.messages[user_index].content);
        }
        let history = conversation.messages.clone();
        let assistant_index = conversation.messages.len();
        conversation
            .messages
            .push(DisplayMessage::assistant(String::new()));
        Ok(EditPreparation {
            conversation_id,
            assistant_index,
            history,
            rollback: EmptyGenerationRollback::RestoreEditedTurn {
                user_index,
                previous_user,
                previous_response,
                previous_title,
            },
        })
    }

    pub fn settle_empty_assistant(
        &mut self,
        conversation_id: u64,
        assistant_index: usize,
        rollback: EmptyGenerationRollback,
    ) -> bool {
        let Some(conversation) = self
            .conversations
            .iter_mut()
            .find(|conversation| conversation.id == conversation_id)
        else {
            return false;
        };
        if !conversation
            .messages
            .get(assistant_index)
            .is_some_and(|message| message.content.is_empty())
        {
            return false;
        }
        match rollback {
            EmptyGenerationRollback::RemovePlaceholder => {
                conversation.messages.remove(assistant_index);
            }
            EmptyGenerationRollback::RestoreResponse(previous_response) => {
                conversation.messages[assistant_index] = previous_response;
            }
            EmptyGenerationRollback::RestoreEditedTurn {
                user_index,
                previous_user,
                previous_response,
                previous_title,
            } => {
                let Some(user_message) = conversation.messages.get_mut(user_index) else {
                    return false;
                };
                *user_message = previous_user;
                if let Some(previous_response) = previous_response {
                    conversation.messages[assistant_index] = *previous_response;
                } else {
                    conversation.messages.remove(assistant_index);
                }
                conversation.title = previous_title;
            }
        }
        true
    }
}

pub fn conversation_matches_query(conversation: &Conversation, query: &str) -> bool {
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return true;
    }
    conversation.title.to_lowercase().contains(&query)
        || conversation
            .messages
            .iter()
            .any(|message| message.content.to_lowercase().contains(&query))
}

fn looks_like_legacy_attachment_marker(content: &str) -> bool {
    content
        .lines()
        .next()
        .is_some_and(|line| line.starts_with("[Image: ") && line.ends_with(']'))
}

/// Encode browser-local conversations without connection or generation settings.
pub fn export_conversation_archive(store: &ConversationStore) -> Result<String, String> {
    let active_conversation = store
        .conversations
        .iter()
        .position(|conversation| conversation.id == store.active_id)
        .ok_or_else(|| "conversation store has no active conversation".to_string())?;
    let archive = ConversationArchive {
        version: CONVERSATION_ARCHIVE_VERSION,
        object: CONVERSATION_ARCHIVE_OBJECT.to_string(),
        active_conversation,
        conversations: store
            .conversations
            .iter()
            .map(|conversation| ArchivedConversation {
                title: conversation.title.clone(),
                messages: conversation
                    .messages
                    .iter()
                    .map(|message| ArchivedMessage {
                        role: message.role.clone(),
                        content: message.content.clone(),
                        attachment_unavailable: message.attachment_unavailable,
                        model: message.execution_model.clone().or_else(|| {
                            message
                                .generation
                                .as_ref()
                                .and_then(|generation| generation.model.clone())
                        }),
                        response_format: message.response_format.clone(),
                    })
                    .collect(),
            })
            .collect(),
    };
    validate_archive(&archive)?;
    let mut json = serde_json::to_string_pretty(&archive)
        .map_err(|error| format!("failed to encode conversation archive: {error}"))?;
    json.push('\n');
    if json.len() as u64 > MAX_CONVERSATION_ARCHIVE_BYTES {
        return Err(format!(
            "conversation archive exceeds the {} byte export limit",
            MAX_CONVERSATION_ARCHIVE_BYTES
        ));
    }
    Ok(json)
}

/// Decode and validate a versioned conversation archive with fresh local IDs.
pub fn import_conversation_archive(input: &str) -> Result<ConversationStore, String> {
    if input.is_empty() || input.len() as u64 > MAX_CONVERSATION_ARCHIVE_BYTES {
        return Err(format!(
            "conversation archive must be between 1 and {} bytes",
            MAX_CONVERSATION_ARCHIVE_BYTES
        ));
    }
    let archive = serde_json::from_str::<ConversationArchive>(input)
        .map_err(|error| format!("invalid conversation archive: {error}"))?;
    validate_archive(&archive)?;

    let conversations = archive
        .conversations
        .into_iter()
        .enumerate()
        .map(|(index, conversation)| Conversation {
            id: index as u64 + 1,
            title: conversation.title,
            messages: conversation
                .messages
                .into_iter()
                .map(|message| DisplayMessage {
                    role: message.role,
                    content: message.content,
                    attachment_unavailable: message.attachment_unavailable,
                    generation: None,
                    execution_model: message.model,
                    response_format: message.response_format,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    Ok(ConversationStore {
        active_id: archive.active_conversation as u64 + 1,
        next_id: conversations.len() as u64 + 1,
        conversations,
    })
}

/// Append an imported archive store with fresh IDs while preserving local history.
pub fn merge_conversation_stores(
    local: &ConversationStore,
    imported: &ConversationStore,
) -> Result<ConversationStore, String> {
    local.validate_persisted()?;
    imported.validate_persisted()?;

    let conversation_count = local
        .conversations
        .len()
        .checked_add(imported.conversations.len())
        .ok_or_else(|| "merged conversation count overflowed".to_string())?;
    if conversation_count > MAX_ARCHIVE_CONVERSATIONS {
        return Err(format!(
            "Merging would exceed the {MAX_ARCHIVE_CONVERSATIONS}-conversation browser history limit."
        ));
    }

    let local_message_count = local
        .conversations
        .iter()
        .try_fold(0_usize, |count, conversation| {
            count.checked_add(conversation.messages.len())
        })
        .ok_or_else(|| "local conversation message count overflowed".to_string())?;
    let imported_message_count = imported
        .conversations
        .iter()
        .try_fold(0_usize, |count, conversation| {
            count.checked_add(conversation.messages.len())
        })
        .ok_or_else(|| "imported conversation message count overflowed".to_string())?;
    let message_count = local_message_count
        .checked_add(imported_message_count)
        .ok_or_else(|| "merged conversation message count overflowed".to_string())?;
    if message_count > MAX_ARCHIVE_MESSAGES {
        return Err(format!(
            "Merging would exceed the {MAX_ARCHIVE_MESSAGES}-message browser history limit."
        ));
    }

    let mut merged = local.clone();
    for conversation in &imported.conversations {
        let (id, next_id) = merged.allocate_conversation_id()?;
        merged.next_id = next_id;
        merged.conversations.push(Conversation {
            id,
            title: conversation.title.clone(),
            messages: conversation.messages.clone(),
        });
    }
    merged.validate_persisted()?;
    Ok(merged)
}

fn validate_archive(archive: &ConversationArchive) -> Result<(), String> {
    if !(MIN_CONVERSATION_ARCHIVE_VERSION..=CONVERSATION_ARCHIVE_VERSION).contains(&archive.version)
    {
        return Err(format!(
            "unsupported conversation archive version {}; supported versions are {} through {}",
            archive.version, MIN_CONVERSATION_ARCHIVE_VERSION, CONVERSATION_ARCHIVE_VERSION
        ));
    }
    if archive.object != CONVERSATION_ARCHIVE_OBJECT {
        return Err("JSON is not a Bloom conversation archive".to_string());
    }
    if archive.conversations.is_empty() {
        return Err("conversation archive contains no conversations".to_string());
    }
    if archive.conversations.len() > MAX_ARCHIVE_CONVERSATIONS {
        return Err(format!(
            "conversation archive contains more than {MAX_ARCHIVE_CONVERSATIONS} conversations"
        ));
    }
    if archive.active_conversation >= archive.conversations.len() {
        return Err("conversation archive has an invalid active conversation".to_string());
    }

    let mut message_count = 0_usize;
    for conversation in &archive.conversations {
        let title_chars = conversation.title.chars().count();
        if conversation.title.trim().is_empty()
            || title_chars > MAX_ARCHIVE_TITLE_CHARS
            || conversation.title.chars().any(char::is_control)
        {
            return Err("conversation archive contains an invalid title".to_string());
        }
        message_count = message_count
            .checked_add(conversation.messages.len())
            .ok_or_else(|| "conversation archive message count overflowed".to_string())?;
        if message_count > MAX_ARCHIVE_MESSAGES {
            return Err(format!(
                "conversation archive contains more than {MAX_ARCHIVE_MESSAGES} messages"
            ));
        }
        for message in &conversation.messages {
            if !matches!(message.role.as_str(), "user" | "assistant") {
                return Err(format!(
                    "conversation archive contains unsupported message role {:?}",
                    message.role
                ));
            }
            if message.attachment_unavailable && message.role != "user" {
                return Err(
                    "conversation archive marks a non-user message as having an unavailable attachment"
                        .to_string(),
                );
            }
            if let Some(model) = message.model.as_deref() {
                if archive.version < 2 {
                    return Err(
                        "conversation archive version 1 cannot contain model provenance"
                            .to_string(),
                    );
                }
                if message.role != "assistant" || !valid_generation_model_id(model) {
                    return Err(
                        "conversation archive contains invalid model provenance".to_string()
                    );
                }
            }
            if let Some(response_format) = &message.response_format
                && (message.role != "assistant"
                    || !matches!(response_format.as_str(), "json_object" | "json_schema"))
            {
                return Err(
                    "conversation archive contains invalid structured-output metadata".to_string(),
                );
            }
            if message.content.chars().count() > MAX_ARCHIVE_MESSAGE_CHARS {
                return Err(format!(
                    "conversation archive contains a message longer than {MAX_ARCHIVE_MESSAGE_CHARS} characters"
                ));
            }
        }
    }
    Ok(())
}

fn title_from_message(message: &str) -> String {
    let normalized = message.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let title: String = chars.by_ref().take(MAX_TITLE_CHARS).collect();
    if title.is_empty() {
        DEFAULT_TITLE.into()
    } else if chars.next().is_some() {
        format!("{title}…")
    } else {
        title
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_user_message_sets_a_compact_title() {
        let mut store = ConversationStore::default();
        store.append_user(
            "  Explain   why deterministic tests make maintenance easier for everyone  ".into(),
        );

        assert_eq!(
            store.active().title,
            "Explain why deterministic tests make maint…"
        );
    }

    #[test]
    fn deleting_the_last_conversation_creates_a_replacement() {
        let mut store = ConversationStore::default();
        assert_eq!(store.delete(1), Ok(true));

        assert_eq!(store.conversations.len(), 1);
        assert_eq!(store.active_id, 2);
        assert_eq!(store.next_id, 3);
    }

    #[test]
    fn normalize_repairs_stale_persisted_ids() {
        let store = ConversationStore {
            active_id: 99,
            next_id: 1,
            conversations: vec![Conversation::new(7)],
        }
        .normalize();

        assert_eq!(store.active_id, 7);
        assert_eq!(store.next_id, 8);
    }

    #[test]
    fn rename_normalizes_spaces_and_rejects_invalid_updates_without_mutation() {
        let mut store = ConversationStore::default();
        assert!(store.rename(1, "  Release   planning  ").is_ok());
        assert_eq!(store.active().title, "Release planning");

        let renamed = store.clone();
        assert!(store.rename(1, "   ").is_err());
        assert!(store.rename(1, "line\nbreak").is_err());
        assert!(
            store
                .rename(1, &"x".repeat(MAX_RENAMED_TITLE_CHARS + 1))
                .is_err()
        );
        assert!(store.rename(99, "Missing").is_err());
        assert_eq!(store, renamed);
    }

    #[test]
    fn branching_preserves_the_source_and_copies_only_the_selected_history() {
        let mut store = ConversationStore::default();
        store.active_mut().title =
            "A deliberately long conversation title used for branching".repeat(2);
        store.append_user("First prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "First response".into();
        store.active_mut().messages[1].generation = Some(GenerationStats::observed(
            GenerationOutcome::Completed,
            250.0,
            Some(25.0),
            Some(8),
            Some(4),
            Some("tiny.gguf".into()),
        ));
        store.append_user("Second prompt".into());
        let source = store.active().clone();

        let branch = store.branch_active_at(1).unwrap();

        assert_eq!(branch.conversation_id, 2);
        assert_eq!(branch.message_count, 2);
        assert!(!branch.contains_unavailable_attachment);
        assert!(branch.title.ends_with(" (branch)"));
        assert!(branch.title.chars().count() <= MAX_RENAMED_TITLE_CHARS);
        assert_eq!(store.active().messages, source.messages[..=1]);
        assert_eq!(
            store.active().messages[1]
                .generation
                .as_ref()
                .unwrap()
                .completion_tokens,
            Some(4)
        );
        assert_eq!(
            store
                .conversations
                .iter()
                .find(|conversation| conversation.id == source.id)
                .unwrap(),
            &source
        );
        assert_eq!(store.next_id, 3);

        assert!(store.select(source.id));
        let second_branch = store.branch_active_at(0).unwrap();
        assert!(second_branch.title.ends_with(" (branch 2)"));
    }

    #[test]
    fn branching_reports_attachment_replay_limits() {
        let mut store = ConversationStore::default();
        store.append_user_with_unavailable_attachment(
            "[Image: chart.png]\nDescribe the trend.".into(),
        );

        let branch = store.branch_active_at(0).unwrap();

        assert!(branch.contains_unavailable_attachment);
        assert!(store.active().messages[0].attachment_unavailable);
    }

    #[test]
    fn continuing_preserves_the_source_and_copies_only_the_recent_suffix() {
        let mut store = ConversationStore::default();
        store.active_mut().title = "Long context investigation".into();
        store.append_user("First prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "First response".into();
        store.append_user("Recent prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[3].content = "Recent response".into();
        store.active_mut().messages[3].response_format = Some("json_object".into());
        store.active_mut().messages[3].generation = Some(GenerationStats::observed(
            GenerationOutcome::Completed,
            300.0,
            Some(40.0),
            Some(32),
            Some(8),
            Some("tiny.gguf".into()),
        ));
        let source = store.active().clone();

        let continuation = store.continue_active_from(2).unwrap();

        assert_eq!(continuation.conversation_id, 2);
        assert_eq!(continuation.retained_message_count, 2);
        assert_eq!(continuation.omitted_message_count, 2);
        assert!(!continuation.contains_unavailable_attachment);
        assert!(continuation.title.ends_with(" (continued)"));
        assert_eq!(store.active().messages, source.messages[2..]);
        assert_eq!(
            store.active().messages[1].response_format.as_deref(),
            Some("json_object")
        );
        assert_eq!(
            store.active().messages[1]
                .generation
                .as_ref()
                .unwrap()
                .completion_tokens,
            Some(8)
        );
        assert_eq!(
            store
                .conversations
                .iter()
                .find(|conversation| conversation.id == source.id)
                .unwrap(),
            &source
        );

        assert!(store.select(source.id));
        let second = store.continue_active_from(2).unwrap();
        assert!(second.title.ends_with(" (continued 2)"));
    }

    #[test]
    fn continuing_reports_attachment_replay_limits() {
        let mut store = ConversationStore::default();
        store.append_user("First prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "First response".into();
        store.append_user_with_unavailable_attachment(
            "[Image: chart.png]\nDescribe the trend.".into(),
        );

        let continuation = store.continue_active_from(2).unwrap();

        assert!(continuation.contains_unavailable_attachment);
        assert!(store.active().messages[0].attachment_unavailable);
    }

    #[test]
    fn continuing_rejects_unsafe_boundaries_and_limits_without_mutation() {
        let mut store = ConversationStore::default();
        store.append_user("First prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "First response".into();
        store.append_user("Recent prompt".into());
        let before = store.clone();

        assert!(store.continue_active_from(0).is_err());
        assert!(store.continue_active_from(1).is_err());
        assert!(store.continue_active_from(3).is_err());
        assert_eq!(store, before);

        let mut exhausted = before.clone();
        exhausted.next_id = u64::MAX;
        let exhausted_before = exhausted.clone();
        assert!(exhausted.continue_active_from(2).is_err());
        assert_eq!(exhausted, exhausted_before);

        let mut full = ConversationStore {
            active_id: 1,
            next_id: MAX_ARCHIVE_CONVERSATIONS as u64 + 1,
            conversations: (1..=MAX_ARCHIVE_CONVERSATIONS)
                .map(|id| Conversation::new(id as u64))
                .collect(),
        };
        full.conversations[0].messages = vec![
            DisplayMessage::user("First prompt"),
            DisplayMessage::user("Recent prompt"),
        ];
        let full_before = full.clone();
        assert!(full.continue_active_from(1).is_err());
        assert_eq!(full, full_before);

        let mut message_limited = ConversationStore::default();
        message_limited.active_mut().messages =
            vec![DisplayMessage::user("Prompt"); MAX_ARCHIVE_MESSAGES / 2 + 1];
        let message_limited_before = message_limited.clone();
        assert!(message_limited.continue_active_from(1).is_err());
        assert_eq!(message_limited, message_limited_before);
    }

    #[test]
    fn branching_rejects_invalid_limits_without_mutating_the_store() {
        let mut empty = ConversationStore::default();
        let empty_before = empty.clone();
        assert!(empty.branch_active_at(0).is_err());
        assert_eq!(empty, empty_before);

        let mut exhausted = ConversationStore::default();
        exhausted.append_user("Prompt".into());
        exhausted.next_id = u64::MAX;
        let exhausted_before = exhausted.clone();
        assert!(exhausted.create().is_err());
        assert!(exhausted.branch_active_at(0).is_err());
        assert!(exhausted.delete(1).is_err());
        assert_eq!(exhausted, exhausted_before);

        let mut full = ConversationStore {
            active_id: 1,
            next_id: MAX_ARCHIVE_CONVERSATIONS as u64 + 1,
            conversations: (1..=MAX_ARCHIVE_CONVERSATIONS)
                .map(|id| Conversation::new(id as u64))
                .collect(),
        };
        full.conversations[0]
            .messages
            .push(DisplayMessage::user("Prompt"));
        let full_before = full.clone();
        assert!(full.create().is_err());
        assert!(full.branch_active_at(0).is_err());
        assert_eq!(full, full_before);

        let mut message_limited = ConversationStore::default();
        message_limited.active_mut().messages =
            vec![DisplayMessage::user("Prompt"); MAX_ARCHIVE_MESSAGES / 2 + 1];
        let message_limited_before = message_limited.clone();
        assert!(
            message_limited
                .branch_active_at(MAX_ARCHIVE_MESSAGES / 2)
                .is_err()
        );
        assert_eq!(message_limited, message_limited_before);
    }

    #[test]
    fn conversation_search_matches_titles_and_message_content_case_insensitively() {
        let mut conversation = Conversation::new(1);
        conversation.title = "Résumé review".into();
        conversation.messages = vec![
            DisplayMessage::user("Explain NUMA placement"),
            DisplayMessage::assistant("Keep memory close to the worker."),
        ];

        assert!(conversation_matches_query(&conversation, "RÉSUMÉ"));
        assert!(conversation_matches_query(&conversation, "numa"));
        assert!(conversation_matches_query(&conversation, "WORKER"));
        assert!(conversation_matches_query(&conversation, "  "));
        assert!(!conversation_matches_query(&conversation, "unrelated"));
    }

    #[test]
    fn persisted_validation_rejects_duplicate_ids_roles_and_invalid_next_id() {
        let mut duplicate = ConversationStore::default();
        duplicate.conversations.push(Conversation::new(1));
        assert!(duplicate.validate_persisted().is_err());

        let mut role = ConversationStore::default();
        role.active_mut().messages.push(DisplayMessage {
            role: "tool".into(),
            content: "unsafe".into(),
            attachment_unavailable: false,
            generation: None,
            execution_model: None,
            response_format: None,
        });
        assert!(role.validate_persisted().is_err());

        let mut invalid_model = ConversationStore::default();
        invalid_model
            .active_mut()
            .messages
            .push(DisplayMessage::assistant("response"));
        invalid_model.active_mut().messages[0].execution_model = Some(" model-a".to_string());
        assert!(invalid_model.validate_persisted().is_err());

        let mut inconsistent_model = ConversationStore::default();
        inconsistent_model
            .active_mut()
            .messages
            .push(DisplayMessage::assistant("response"));
        inconsistent_model.active_mut().messages[0].generation = Some(GenerationStats::observed(
            GenerationOutcome::Completed,
            10.0,
            None,
            None,
            None,
            Some("model-a".to_string()),
        ));
        inconsistent_model.active_mut().messages[0].execution_model = Some("model-b".to_string());
        assert!(inconsistent_model.validate_persisted().is_err());

        let next_id = ConversationStore {
            next_id: 1,
            ..ConversationStore::default()
        };
        assert!(next_id.validate_persisted().is_err());
        assert!(ConversationStore::default().validate_persisted().is_ok());
    }

    #[test]
    fn conversation_archive_round_trip_preserves_content_and_active_selection() {
        let mut store = ConversationStore::default();
        store.append_user("First prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "First response".into();
        let second_id = store.create().unwrap();
        store.append_user_with_unavailable_attachment("Second prompt".into());
        assert!(store.select(second_id));

        let json = export_conversation_archive(&store).unwrap();
        let restored = import_conversation_archive(&json).unwrap();

        assert_eq!(restored.conversations.len(), 2);
        assert_eq!(restored.active_id, 1);
        assert_eq!(restored.next_id, 3);
        assert_eq!(restored.active().messages[0].content, "Second prompt");
        assert!(restored.active().messages[0].attachment_unavailable);
        assert!(!json.contains("active_id"));
        assert!(!json.contains("next_id"));
        assert!(json.contains("attachment_unavailable"));
    }

    #[test]
    fn conversation_merge_preserves_local_history_and_assigns_fresh_ids() {
        let mut local = ConversationStore::default();
        local.append_user("Keep this prompt".into());
        local.rename(1, "Local first").unwrap();
        let second_local_id = local.create().unwrap();
        local.rename(second_local_id, "Local second").unwrap();
        assert!(local.select(1));
        let local_before = local.clone();

        let mut imported = ConversationStore::default();
        imported.append_user("Imported prompt".into());
        imported.rename(1, "Imported first").unwrap();
        let second_imported_id = imported.create().unwrap();
        imported
            .rename(second_imported_id, "Imported second")
            .unwrap();

        let merged = merge_conversation_stores(&local, &imported).unwrap();

        assert_eq!(local, local_before);
        assert_eq!(merged.active_id, local_before.active_id);
        assert_eq!(
            &merged.conversations[..local_before.conversations.len()],
            local_before.conversations.as_slice()
        );
        assert_eq!(merged.conversations.len(), 4);
        assert_eq!(merged.conversations[2].id, 3);
        assert_eq!(merged.conversations[2].title, "Imported second");
        assert_eq!(merged.conversations[3].id, 4);
        assert_eq!(merged.conversations[3].title, "Imported first");
        assert_eq!(merged.next_id, 5);
        assert!(merged.validate_persisted().is_ok());
    }

    #[test]
    fn conversation_merge_fails_atomically_at_store_limits() {
        let mut imported = ConversationStore::default();
        imported.append_user("Imported prompt".into());

        let nearly_full = ConversationStore {
            active_id: 1,
            next_id: MAX_ARCHIVE_CONVERSATIONS as u64,
            conversations: (1..MAX_ARCHIVE_CONVERSATIONS)
                .map(|id| Conversation::new(id as u64))
                .collect(),
        };
        let merged_at_conversation_limit =
            merge_conversation_stores(&nearly_full, &imported).unwrap();
        assert_eq!(
            merged_at_conversation_limit.conversations.len(),
            MAX_ARCHIVE_CONVERSATIONS
        );

        let full = ConversationStore {
            active_id: 1,
            next_id: MAX_ARCHIVE_CONVERSATIONS as u64 + 1,
            conversations: (1..=MAX_ARCHIVE_CONVERSATIONS)
                .map(|id| Conversation::new(id as u64))
                .collect(),
        };
        let full_before = full.clone();
        assert!(
            merge_conversation_stores(&full, &imported)
                .unwrap_err()
                .contains("1000-conversation")
        );
        assert_eq!(full, full_before);

        let mut nearly_message_limited = ConversationStore::default();
        nearly_message_limited.active_mut().messages =
            vec![DisplayMessage::assistant(String::new()); MAX_ARCHIVE_MESSAGES - 1];
        let merged_at_message_limit =
            merge_conversation_stores(&nearly_message_limited, &imported).unwrap();
        assert_eq!(
            merged_at_message_limit
                .conversations
                .iter()
                .map(|conversation| conversation.messages.len())
                .sum::<usize>(),
            MAX_ARCHIVE_MESSAGES
        );

        let mut message_limited = ConversationStore::default();
        message_limited.active_mut().messages =
            vec![DisplayMessage::assistant(String::new()); MAX_ARCHIVE_MESSAGES];
        let message_limited_before = message_limited.clone();
        assert!(
            merge_conversation_stores(&message_limited, &imported)
                .unwrap_err()
                .contains("50000-message")
        );
        assert_eq!(message_limited, message_limited_before);

        let exhausted = ConversationStore {
            next_id: u64::MAX,
            ..ConversationStore::default()
        };
        let exhausted_before = exhausted.clone();
        assert!(
            merge_conversation_stores(&exhausted, &imported)
                .unwrap_err()
                .contains("ID space")
        );
        assert_eq!(exhausted, exhausted_before);
    }

    #[test]
    fn conversation_archive_rejects_unknown_fields_versions_and_roles() {
        let unknown_field = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[],"extra":true}]}"#;
        let unknown_version = r#"{"version":3,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[]}]}"#;
        let system_role = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[{"role":"system","content":"unsafe"}]}]}"#;
        let assistant_attachment = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[{"role":"assistant","content":"response","attachment_unavailable":true}]}]}"#;
        let user_response_format = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[{"role":"user","content":"prompt","response_format":"json_object"}]}]}"#;
        let unknown_response_format = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[{"role":"assistant","content":"response","response_format":"yaml"}]}]}"#;
        let legacy_model = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[{"role":"assistant","content":"response","model":"model-a"}]}]}"#;
        let user_model = r#"{"version":2,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[{"role":"user","content":"prompt","model":"model-a"}]}]}"#;
        let invalid_model = r#"{"version":2,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Chat","messages":[{"role":"assistant","content":"response","model":" model-a"}]}]}"#;

        assert!(import_conversation_archive(unknown_field).is_err());
        assert!(import_conversation_archive(unknown_version).is_err());
        assert!(import_conversation_archive(system_role).is_err());
        assert!(import_conversation_archive(assistant_attachment).is_err());
        assert!(import_conversation_archive(user_response_format).is_err());
        assert!(import_conversation_archive(unknown_response_format).is_err());
        assert!(import_conversation_archive(legacy_model).is_err());
        assert!(import_conversation_archive(user_model).is_err());
        assert!(import_conversation_archive(invalid_model).is_err());
    }

    #[test]
    fn conversation_archive_import_keeps_legacy_version_one_compatible() {
        let legacy = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[{"title":"Legacy","messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}]}"#;

        let imported = import_conversation_archive(legacy).unwrap();

        assert_eq!(imported.active().title, "Legacy");
        assert_eq!(imported.active().messages.len(), 2);
        assert_eq!(imported.active_execution_model(), None);
    }

    #[test]
    fn conversation_archive_rejects_empty_or_invalid_active_collections() {
        let empty = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":0,"conversations":[]}"#;
        let invalid_active = r#"{"version":1,"object":"bloom.conversation_archive","active_conversation":1,"conversations":[{"title":"Chat","messages":[]}]}"#;

        assert!(import_conversation_archive(empty).is_err());
        assert!(import_conversation_archive(invalid_active).is_err());
        assert!(import_conversation_archive("").is_err());
    }

    #[test]
    fn conversation_archive_enforces_the_total_message_limit() {
        let messages = vec![
            serde_json::json!({ "role": "assistant", "content": "" });
            MAX_ARCHIVE_MESSAGES + 1
        ];
        let json = serde_json::json!({
            "version": 1,
            "object": CONVERSATION_ARCHIVE_OBJECT,
            "active_conversation": 0,
            "conversations": [{ "title": "Chat", "messages": messages }]
        })
        .to_string();

        let error = import_conversation_archive(&json).unwrap_err();
        assert!(error.contains("more than 50000 messages"));
    }

    #[test]
    fn retry_replaces_only_the_latest_text_assistant_response() {
        let mut store = ConversationStore::default();
        store.append_user("First prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "Old response".into();

        let retry = store.prepare_last_response_retry().unwrap();

        assert_eq!(retry.conversation_id, 1);
        assert_eq!(retry.assistant_index, 1);
        assert_eq!(retry.history.len(), 1);
        assert_eq!(retry.history[0].content, "First prompt");
        assert!(matches!(
            retry.rollback,
            EmptyGenerationRollback::RestoreResponse(ref response)
                if response.content == "Old response"
        ));
        assert_eq!(store.active().messages.len(), 2);
        assert!(store.active().messages[1].content.is_empty());
    }

    #[test]
    fn retry_rejects_current_and_legacy_image_requests_without_mutation() {
        let mut current = ConversationStore::default();
        current.append_user_with_unavailable_attachment(
            "[Image: current.png]\nDescribe this image.".into(),
        );
        current.append_assistant_placeholder();
        current.active_mut().messages[1].content = "Description".into();
        let current_before = current.clone();

        let mut legacy = ConversationStore::default();
        legacy.append_user("[Image: legacy.png]\nDescribe this image.".into());
        legacy.append_assistant_placeholder();
        legacy.active_mut().messages[1].content = "Description".into();
        let legacy_before = legacy.clone();

        assert!(current.prepare_last_response_retry().is_err());
        assert!(legacy.prepare_last_response_retry().is_err());
        assert_eq!(current, current_before);
        assert_eq!(legacy, legacy_before);
    }

    #[test]
    fn retry_allows_a_text_follow_up_after_an_older_image_turn() {
        let mut store = ConversationStore::default();
        store.append_user_with_unavailable_attachment("[Image: chart.png]\nDescribe it.".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "It is a chart.".into();
        store.append_user("What is the main trend?".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[3].content = "The trend rises.".into();

        let retry = store.prepare_last_response_retry().unwrap();

        assert_eq!(retry.history.len(), 3);
        assert_eq!(retry.history[2].content, "What is the main trend?");
    }

    #[test]
    fn retry_can_resume_a_text_user_turn_that_has_no_response() {
        let mut store = ConversationStore::default();
        store.append_user("Try again".into());

        let retry = store.prepare_last_response_retry().unwrap();

        assert_eq!(retry.history.len(), 1);
        assert!(matches!(
            retry.rollback,
            EmptyGenerationRollback::RemovePlaceholder
        ));
        assert_eq!(retry.assistant_index, 1);
        assert!(store.active().messages[1].content.is_empty());
    }

    #[test]
    fn persisted_messages_without_replay_metadata_remain_compatible() {
        let json = r#"{"active_id":1,"next_id":2,"conversations":[{"id":1,"title":"Legacy","messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi","generation":{"outcome":"completed","elapsed_ms":10}}]}]}"#;

        let store = serde_json::from_str::<ConversationStore>(json).unwrap();

        assert!(!store.active().messages[0].attachment_unavailable);
        assert!(store.active().messages[0].generation.is_none());
        assert!(store.active().messages[0].response_format.is_none());
        assert!(
            store.active().messages[1]
                .generation
                .as_ref()
                .unwrap()
                .model
                .is_none()
        );
        assert_eq!(store.active_execution_model(), None);
    }

    #[test]
    fn active_execution_model_uses_the_latest_recorded_response_provenance() {
        let mut store = ConversationStore::default();
        for (model, response) in [("model-a", "First"), ("model-b", "Second")] {
            store.append_user(format!("Prompt for {model}"));
            let assistant_index = store.append_assistant_placeholder();
            store.active_mut().messages[assistant_index].content = response.to_string();
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
                        Some(model.to_string()),
                    ),
                )
                .unwrap();
        }

        store.append_user("Unanswered follow-up".into());
        assert_eq!(store.active_execution_model(), Some("model-b"));
    }

    #[test]
    fn archive_export_promotes_legacy_generation_model_provenance() {
        let mut store = ConversationStore::default();
        store.append_user("Hello".into());
        let assistant_index = store.append_assistant_placeholder();
        store.active_mut().messages[assistant_index].content = "Hi".into();
        store.active_mut().messages[assistant_index].generation = Some(GenerationStats::observed(
            GenerationOutcome::Completed,
            10.0,
            None,
            None,
            None,
            Some("legacy-model".to_string()),
        ));
        assert!(
            store.active().messages[assistant_index]
                .execution_model
                .is_none()
        );

        let archive = export_conversation_archive(&store).unwrap();
        let imported = import_conversation_archive(&archive).unwrap();

        assert!(archive.contains("\"model\": \"legacy-model\""));
        assert_eq!(imported.active_execution_model(), Some("legacy-model"));
    }

    #[test]
    fn archives_keep_model_provenance_but_exclude_runtime_measurements() {
        let mut store = ConversationStore::default();
        store.append_user("Measure this".into());
        let assistant_index = store.append_assistant_placeholder();
        store.active_mut().messages[assistant_index].content = "Measured".into();
        let stats = GenerationStats {
            outcome: GenerationOutcome::Completed,
            model: Some("tiny.gguf".to_string()),
            elapsed_ms: 2_500,
            first_token_ms: Some(500),
            prompt_tokens: Some(12),
            completion_tokens: Some(4),
        };

        store
            .set_assistant_generation_stats(1, assistant_index, stats.clone())
            .unwrap();
        store
            .set_assistant_response_format(1, assistant_index, "json_schema")
            .unwrap();
        assert_eq!(stats.tokens_per_second(), Some(2.0));
        assert_eq!(stats.total_tokens(), Some(16));
        assert!(store.validate_persisted().is_ok());

        let encoded = serde_json::to_string(&store).unwrap();
        let decoded = serde_json::from_str::<ConversationStore>(&encoded).unwrap();
        assert_eq!(
            decoded.active().messages[assistant_index].generation,
            Some(stats)
        );

        let archive = export_conversation_archive(&store).unwrap();
        assert!(!archive.contains("elapsed_ms"));
        assert!(archive.contains("\"version\": 2"));
        assert!(archive.contains("\"model\": \"tiny.gguf\""));
        let imported = import_conversation_archive(&archive).unwrap();
        assert!(
            imported.active().messages[assistant_index]
                .generation
                .is_none()
        );
        assert_eq!(
            imported.active().messages[assistant_index]
                .execution_model
                .as_deref(),
            Some("tiny.gguf")
        );
        assert_eq!(imported.active_execution_model(), Some("tiny.gguf"));
        assert_eq!(
            imported.active().messages[assistant_index]
                .response_format
                .as_deref(),
            Some("json_schema")
        );
    }

    #[test]
    fn generation_stats_reject_invalid_timing_and_non_assistant_ownership() {
        let invalid = GenerationStats {
            outcome: GenerationOutcome::Stopped,
            model: None,
            elapsed_ms: 100,
            first_token_ms: Some(101),
            prompt_tokens: None,
            completion_tokens: None,
        };
        assert!(invalid.validate().is_err());

        let mut store = ConversationStore::default();
        store.append_user("Prompt".into());
        store.active_mut().messages[0].generation = Some(GenerationStats {
            outcome: GenerationOutcome::Failed,
            model: None,
            elapsed_ms: 100,
            first_token_ms: None,
            prompt_tokens: None,
            completion_tokens: None,
        });
        assert!(store.validate_persisted().is_err());

        let mut invalid_model = DisplayMessage::assistant("Response");
        invalid_model.generation = Some(GenerationStats {
            outcome: GenerationOutcome::Completed,
            model: Some(" model.gguf".to_string()),
            elapsed_ms: 100,
            first_token_ms: None,
            prompt_tokens: None,
            completion_tokens: None,
        });
        let mut invalid_model_store = ConversationStore::default();
        invalid_model_store
            .active_mut()
            .messages
            .push(invalid_model);
        assert!(invalid_model_store.validate_persisted().is_err());

        let mut invalid_format = ConversationStore::default();
        invalid_format.append_user("Prompt".into());
        invalid_format.active_mut().messages[0].response_format = Some("json_object".into());
        assert!(invalid_format.validate_persisted().is_err());

        let normalized = GenerationStats::observed(
            GenerationOutcome::Completed,
            -1.0,
            Some(f64::NAN),
            Some(MAX_GENERATION_TOKEN_COUNT + 1),
            Some(1),
            Some(" model.gguf".to_string()),
        );
        assert_eq!(normalized.elapsed_ms, 1);
        assert!(normalized.first_token_ms.is_none());
        assert!(normalized.prompt_tokens.is_none());
        assert!(normalized.completion_tokens.is_none());
        assert!(normalized.model.is_none());
        assert!(normalized.validate().is_ok());
    }

    #[test]
    fn empty_generation_cleanup_removes_new_placeholders_and_restores_retries() {
        let mut initial = ConversationStore::default();
        initial.append_user("Prompt".into());
        let initial_index = initial.append_assistant_placeholder();
        assert!(initial.settle_empty_assistant(
            1,
            initial_index,
            EmptyGenerationRollback::RemovePlaceholder
        ));
        assert_eq!(initial.active().messages.len(), 1);

        let mut retry = ConversationStore::default();
        retry.append_user("Prompt".into());
        retry.append_assistant_placeholder();
        retry.active_mut().messages[1].content = "Previous response".into();
        let preparation = retry.prepare_last_response_retry().unwrap();
        assert!(retry.settle_empty_assistant(
            preparation.conversation_id,
            preparation.assistant_index,
            preparation.rollback
        ));
        assert_eq!(retry.active().messages[1].content, "Previous response");

        retry.active_mut().messages[1].content = "Partial response".into();
        assert!(!retry.settle_empty_assistant(1, 1, EmptyGenerationRollback::RemovePlaceholder));
        assert_eq!(retry.active().messages[1].content, "Partial response");
    }

    #[test]
    fn latest_prompt_edit_updates_history_and_rolls_back_the_complete_turn() {
        let mut store = ConversationStore::default();
        store.append_user("Original prompt".into());
        store.append_assistant_placeholder();
        store.active_mut().messages[1].content = "Original response".into();

        let edit = store
            .prepare_last_prompt_edit("Improved prompt".into())
            .unwrap();

        assert_eq!(edit.history.last().unwrap().content, "Improved prompt");
        assert_eq!(store.active().title, "Improved prompt");
        assert!(store.active().messages[1].content.is_empty());
        assert!(store.settle_empty_assistant(
            edit.conversation_id,
            edit.assistant_index,
            edit.rollback
        ));
        assert_eq!(store.active().title, "Original prompt");
        assert_eq!(store.active().messages[0].content, "Original prompt");
        assert_eq!(store.active().messages[1].content, "Original response");
    }

    #[test]
    fn latest_prompt_edit_rejects_blank_and_image_turns_without_mutation() {
        let mut text = ConversationStore::default();
        text.append_user("Prompt".into());
        text.append_assistant_placeholder();
        let text_before = text.clone();
        assert!(text.prepare_last_prompt_edit("  ".into()).is_err());
        assert_eq!(text, text_before);

        let mut image = ConversationStore::default();
        image.append_user_with_unavailable_attachment("[Image: photo.png]\nDescribe it.".into());
        image.append_assistant_placeholder();
        let image_before = image.clone();
        assert!(
            image
                .prepare_last_prompt_edit("A different prompt".into())
                .is_err()
        );
        assert_eq!(image, image_before);
    }

    #[test]
    fn unanswered_prompt_edit_can_restore_the_original_user_only_turn() {
        let mut store = ConversationStore::default();
        store.append_user("Original unanswered prompt".into());

        let edit = store
            .prepare_last_prompt_edit("Edited unanswered prompt".into())
            .unwrap();

        assert_eq!(store.active().messages.len(), 2);
        assert_eq!(
            store.active().messages[0].content,
            "Edited unanswered prompt"
        );
        assert!(store.settle_empty_assistant(
            edit.conversation_id,
            edit.assistant_index,
            edit.rollback
        ));
        assert_eq!(store.active().messages.len(), 1);
        assert_eq!(
            store.active().messages[0].content,
            "Original unanswered prompt"
        );
        assert_eq!(store.active().title, "Original unanswered prompt");
    }
}
