//! Chat template support for different model families.
//!
//! Each model family may use a different prompt formatting convention.
//! This module provides a `ChatTemplate` trait and implementations for
//! Qwen, Llama, and Gemma families, with a fallback to a generic
//! ChatML-style template.

use bloomai_core::ModelFamily;

/// A single chat message with role and content.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Trait for formatting a list of chat messages into a single prompt string
/// that the model's tokenizer expects.
pub trait ChatTemplate: Send + Sync {
    /// Human-readable name of this template (e.g. "qwen", "llama").
    fn name(&self) -> &'static str;

    /// Format a conversation into a prompt string.
    fn format(&self, messages: &[ChatMessage]) -> String;
}

// ---------------------------------------------------------------------------
// Qwen / ChatML template
// ---------------------------------------------------------------------------

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Qwen-style ChatML template using im_start/im_end markers.
pub struct QwenChatTemplate;

impl ChatTemplate for QwenChatTemplate {
    fn name(&self) -> &'static str {
        "qwen"
    }

    fn format(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!(
                "{0}{1}\n{2}{3}\n",
                IM_START, msg.role, msg.content, IM_END
            ));
        }
        prompt.push_str(&format!("{IM_START}assistant\n"));
        prompt
    }
}

/// SmolLM2 ChatML template with its documented default system instruction.
pub struct SmolLm2ChatTemplate;

impl ChatTemplate for SmolLm2ChatTemplate {
    fn name(&self) -> &'static str {
        "smollm2"
    }

    fn format(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        if !messages
            .first()
            .is_some_and(|message| message.role == "system")
        {
            prompt.push_str(concat!(
                "<|im_start|>system\n",
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
                "<|im_end|>\n"
            ));
        }
        for message in messages {
            prompt.push_str(&format!(
                "{IM_START}{}\n{}{IM_END}\n",
                message.role, message.content
            ));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
}

/// Qwen3 ChatML template with the official disabled-thinking assistant prefix.
///
/// Bloom does not currently expose reasoning content as a separate protocol
/// channel. Prefilling an empty thinking block keeps ordinary chat output free
/// of raw `<think>` markers while preserving the model's expected format.
pub struct Qwen3NoThinkChatTemplate;

impl ChatTemplate for Qwen3NoThinkChatTemplate {
    fn name(&self) -> &'static str {
        "qwen3-no-think"
    }

    fn format(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = QwenChatTemplate.format(messages);
        prompt.push_str("<think>\n\n</think>\n\n");
        prompt
    }
}

// ---------------------------------------------------------------------------
// Llama template
// ---------------------------------------------------------------------------

/// Llama-style template using [INST]...[/INST] markers.
///
/// Follows the Llama-2 chat format where system messages are wrapped in
/// `<<SYS>>...<</SYS>>` inside the first `[INST]` block.
pub struct LlamaChatTemplate;

impl ChatTemplate for LlamaChatTemplate {
    fn name(&self) -> &'static str {
        "llama"
    }

    fn format(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::from("<s>");
        let mut i = 0;

        // Handle system message specially
        if let Some(first) = messages.first() {
            if first.role == "system" {
                let system_msg = &first.content;
                if let Some(second) = messages.get(1).filter(|message| message.role == "user") {
                    prompt.push_str(&format!(
                        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                        system_msg, second.content
                    ));
                    i = 2;
                } else {
                    prompt.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>> [/INST]", system_msg));
                    i = 1;
                }
            }
        }

        // Remaining messages
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role.as_str() {
                "user" => {
                    prompt.push_str(&format!("<s>[INST] {} [/INST]", msg.content));
                }
                "assistant" => {
                    prompt.push_str(&format!(" {} </s>", msg.content));
                }
                _ => {
                    prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                }
            }
            i += 1;
        }

        prompt
    }
}

/// Llama-3 header-token template.
pub struct Llama3ChatTemplate;

impl ChatTemplate for Llama3ChatTemplate {
    fn name(&self) -> &'static str {
        "llama3"
    }

    fn format(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::from("<|begin_of_text|>");
        for message in messages {
            prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                message.role, message.content
            ));
        }
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        prompt
    }
}

// ---------------------------------------------------------------------------
// Gemma template
// ---------------------------------------------------------------------------

/// Gemma-style template using start_of_turn / end_of_turn markers.
pub struct GemmaChatTemplate;

impl ChatTemplate for GemmaChatTemplate {
    fn name(&self) -> &'static str {
        "gemma"
    }

    fn format(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                msg.role, msg.content
            ));
        }
        prompt.push_str("<start_of_turn>model\n");
        prompt
    }
}

// ---------------------------------------------------------------------------
// Generic ChatML fallback
// ---------------------------------------------------------------------------

/// Generic ChatML-style fallback for unknown model families.
pub struct GenericChatTemplate;

impl ChatTemplate for GenericChatTemplate {
    fn name(&self) -> &'static str {
        "generic"
    }

    fn format(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!(
                "{0}{1}\n{2}{3}\n",
                IM_START, msg.role, msg.content, IM_END
            ));
        }
        prompt.push_str(&format!("{IM_START}assistant\n"));
        prompt
    }
}

// ---------------------------------------------------------------------------
// Template selection
// ---------------------------------------------------------------------------

/// Select the appropriate chat template for a model family.
pub fn select_template(family: &ModelFamily) -> Box<dyn ChatTemplate> {
    match family {
        ModelFamily::Qwen => Box::new(QwenChatTemplate),
        ModelFamily::Llama => Box::new(LlamaChatTemplate),
        ModelFamily::Gemma => Box::new(GemmaChatTemplate),
        _ => Box::new(GenericChatTemplate),
    }
}

/// Select a hard-coded template from bounded, classified model metadata.
pub fn select_template_for_metadata(
    family: &ModelFamily,
    architecture: Option<&str>,
    chat_template_kind: Option<&str>,
) -> Box<dyn ChatTemplate> {
    if matches!(family, ModelFamily::Qwen)
        && architecture.is_some_and(|value| value.eq_ignore_ascii_case("qwen3"))
    {
        Box::new(Qwen3NoThinkChatTemplate)
    } else {
        match chat_template_kind.map(str::to_ascii_lowercase).as_deref() {
            Some("smollm2") => Box::new(SmolLm2ChatTemplate),
            Some("chatml") => Box::new(QwenChatTemplate),
            Some("llama3") => Box::new(Llama3ChatTemplate),
            Some("llama2") => Box::new(LlamaChatTemplate),
            Some("gemma") => Box::new(GemmaChatTemplate),
            _ => select_template(family),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Hi!".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Bye".into(),
            },
        ]
    }

    #[test]
    fn test_qwen_template() {
        let tpl = QwenChatTemplate;
        let out = tpl.format(&sample_messages());
        assert!(out.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
        assert!(!out.contains('\u{FF5C}'));
    }

    #[test]
    fn qwen3_no_think_template_prefills_an_empty_reasoning_block() {
        let tpl = Qwen3NoThinkChatTemplate;
        let out = tpl.format(&sample_messages());
        assert!(out.contains("<|im_start|>user\nBye<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_llama_template() {
        let tpl = LlamaChatTemplate;
        let out = tpl.format(&sample_messages());
        assert!(out.starts_with("<s>"));
        assert!(out.contains("[INST]"));
        assert!(out.contains("[/INST]"));
        assert!(out.contains("<<SYS>>"));
    }

    #[test]
    fn test_llama_template_does_not_fold_non_user_content_into_system_turn() {
        let tpl = LlamaChatTemplate;
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "Policy".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Prior answer".into(),
            },
        ];
        let out = tpl.format(&messages);

        assert!(out.contains("[INST] <<SYS>>\nPolicy\n<</SYS>> [/INST]"));
        assert!(out.contains(" Prior answer </s>"));
        assert!(!out.contains("<</SYS>>\n\nPrior answer"));
    }

    #[test]
    fn smollm2_template_uses_chatml_and_injects_only_a_missing_system_message() {
        let template = SmolLm2ChatTemplate;
        let user_only = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let prompt = template.format(&user_only);
        assert!(prompt.starts_with(concat!(
            "<|im_start|>system\n",
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
            "<|im_end|>\n"
        )));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));

        let explicit_system = template.format(&sample_messages());
        assert_eq!(explicit_system.matches("<|im_start|>system\n").count(), 1);
        assert!(!explicit_system.contains("helpful AI assistant named SmolLM"));
    }

    #[test]
    fn llama3_template_uses_header_tokens() {
        let prompt = Llama3ChatTemplate.format(&sample_messages());
        assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>system"));
        assert!(prompt.contains("<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
        assert!(!prompt.contains("[INST]"));
    }

    #[test]
    fn test_gemma_template() {
        let tpl = GemmaChatTemplate;
        let out = tpl.format(&sample_messages());
        assert!(out.contains("<start_of_turn>"));
        assert!(out.contains("<end_of_turn>"));
        assert!(out.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_generic_template() {
        let tpl = GenericChatTemplate;
        let out = tpl.format(&sample_messages());
        assert!(out.contains("<|im_start|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_select_template() {
        let tpl = select_template(&ModelFamily::Qwen);
        assert_eq!(tpl.name(), "qwen");

        let tpl = select_template(&ModelFamily::Llama);
        assert_eq!(tpl.name(), "llama");

        let tpl = select_template(&ModelFamily::Gemma);
        assert_eq!(tpl.name(), "gemma");

        let tpl = select_template(&ModelFamily::FunAsr);
        assert_eq!(tpl.name(), "generic");

        let tpl = select_template_for_metadata(&ModelFamily::Qwen, Some("qwen3"), None);
        assert_eq!(tpl.name(), "qwen3-no-think");

        let tpl = select_template_for_metadata(&ModelFamily::Qwen, Some("qwen2"), None);
        assert_eq!(tpl.name(), "qwen");

        let tpl = select_template_for_metadata(&ModelFamily::Llama, Some("llama"), Some("smollm2"));
        assert_eq!(tpl.name(), "smollm2");

        let tpl = select_template_for_metadata(&ModelFamily::Llama, Some("llama"), Some("llama3"));
        assert_eq!(tpl.name(), "llama3");
    }
}
