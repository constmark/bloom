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

// ---------------------------------------------------------------------------
// Llama template
// ---------------------------------------------------------------------------

/// Llama-style template using [INST]...[/INST] markers.
///
/// Follows the Llama-2/3 chat format where system messages are wrapped in
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
                if let Some(second) = messages.get(1) {
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
    fn test_llama_template() {
        let tpl = LlamaChatTemplate;
        let out = tpl.format(&sample_messages());
        assert!(out.starts_with("<s>"));
        assert!(out.contains("[INST]"));
        assert!(out.contains("[/INST]"));
        assert!(out.contains("<<SYS>>"));
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
    }
}
