//! Small, dependency-free translations for the Bloom shell and first-run view.
//!
//! The UI keeps the language catalog close to the product surface so adding a
//! new locale does not require introducing a runtime translation service.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Locale {
    English,
    SimplifiedChinese,
    Japanese,
}

impl Locale {
    pub const ALL: [Self; 3] = [Self::English, Self::SimplifiedChinese, Self::Japanese];

    pub const fn storage_value(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::SimplifiedChinese => "zh-CN",
            Self::Japanese => "ja",
        }
    }

    pub const fn html_lang(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::SimplifiedChinese => "zh-CN",
            Self::Japanese => "ja",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::English => "English",
            Self::SimplifiedChinese => "简体中文",
            Self::Japanese => "日本語",
        }
    }

    pub const fn short_label(self) -> &'static str {
        match self {
            Self::English => "EN",
            Self::SimplifiedChinese => "中",
            Self::Japanese => "日",
        }
    }

    pub fn from_storage_value(value: &str) -> Option<Self> {
        match value {
            "en" => Some(Self::English),
            "zh-CN" => Some(Self::SimplifiedChinese),
            "ja" => Some(Self::Japanese),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn from_language_tag(value: &str) -> Self {
        let value = value.trim().to_ascii_lowercase();
        if value.starts_with("zh") {
            Self::SimplifiedChinese
        } else if value.starts_with("ja") {
            Self::Japanese
        } else {
            Self::English
        }
    }
}

#[derive(Clone, Copy)]
pub struct UiCopy {
    pub brand_chat: &'static str,
    pub brand_embedding: &'static str,
    pub brand_speech: &'static str,
    pub conversations: &'static str,
    pub close_conversations: &'static str,
    pub new_chat: &'static str,
    pub search_conversations: &'static str,
    pub clear_search: &'static str,
    pub no_conversations_match: &'static str,
    pub backup_title: &'static str,
    pub backup_description: &'static str,
    pub import: &'static str,
    pub export: &'static str,
    pub reading: &'static str,
    pub open_conversations: &'static str,
    pub models: &'static str,
    pub diagnostics: &'static str,
    pub settings: &'static str,
    pub language: &'static str,
    pub home_eyebrow: &'static str,
    pub home_title: &'static str,
    pub home_body: &'static str,
    pub capability_local: &'static str,
    pub capability_local_body: &'static str,
    pub capability_compatible: &'static str,
    pub capability_compatible_body: &'static str,
    pub capability_multimodal: &'static str,
    pub capability_multimodal_body: &'static str,
    pub message_placeholder: &'static str,
    pub attach: &'static str,
    pub attach_reading: &'static str,
    pub remove_attachment: &'static str,
    pub send: &'static str,
    pub stop: &'static str,
    pub composer_hint: &'static str,
    pub composer_waiting: &'static str,
    pub open_models: &'static str,
    pub connection_settings: &'static str,
    pub check_connection: &'static str,
    pub update_api_key: &'static str,
    pub check_server_version: &'static str,
    pub view_model_status: &'static str,
    pub review_models: &'static str,
    pub choose_model: &'static str,
    pub api_key_required: &'static str,
    pub incompatible_server: &'static str,
    pub server_unavailable: &'static str,
    pub connecting: &'static str,
    pub ready: &'static str,
    pub transcription: &'static str,
    pub embeddings: &'static str,
    pub generation: &'static str,
    pub loading: &'static str,
    pub choose_language: &'static str,
}

pub const fn copy(locale: Locale) -> UiCopy {
    match locale {
        Locale::English => UiCopy {
            brand_chat: "Local multimodal inference",
            brand_embedding: "Local embedding and reranking",
            brand_speech: "Live speech to text",
            conversations: "Conversations",
            close_conversations: "Close conversations",
            new_chat: "+ New chat",
            search_conversations: "Search conversations",
            clear_search: "Clear search",
            no_conversations_match: "No conversations match this search.",
            backup_title: "Conversation backup",
            backup_description: "Messages only; connection settings are excluded.",
            import: "Import",
            export: "Export",
            reading: "Reading…",
            open_conversations: "Open conversations",
            models: "Models",
            diagnostics: "Diagnostics",
            settings: "Settings",
            language: "Language",
            home_eyebrow: "Bloom · local AI workspace",
            home_title: "Build with intelligence that stays close to you.",
            home_body: "A focused, observable workspace for private inference. Connect a Bloom server to start a conversation, inspect a model, or run multimodal workloads.",
            capability_local: "Local-first",
            capability_local_body: "Your prompts stay between this browser and your configured server.",
            capability_compatible: "OpenAI-compatible",
            capability_compatible_body: "Use familiar APIs and predictable runtime signals.",
            capability_multimodal: "Multimodal",
            capability_multimodal_body: "Text, vision, speech, embeddings, and reranking in one workspace.",
            message_placeholder: "Ask Bloom anything… Enter to send · Shift+Enter for a new line",
            attach: "Attach",
            attach_reading: "Reading…",
            remove_attachment: "Remove image attachment",
            send: "Send",
            stop: "Stop",
            composer_hint: "Responses stream from the OpenAI-compatible API.",
            composer_waiting: "Wait for the model to become ready before sending.",
            open_models: "Open models",
            connection_settings: "Connection settings",
            check_connection: "Check connection",
            update_api_key: "Update API key",
            check_server_version: "Check server version",
            view_model_status: "View model status",
            review_models: "Review models",
            choose_model: "Choose a model to begin",
            api_key_required: "API key required",
            incompatible_server: "Incompatible Bloom server",
            server_unavailable: "Bloom server is unavailable",
            connecting: "Connecting…",
            ready: "Ready",
            transcription: "Transcription",
            embeddings: "Embeddings",
            generation: "Generation",
            loading: "Loading",
            choose_language: "Choose interface language",
        },
        Locale::SimplifiedChinese => UiCopy {
            brand_chat: "本地多模态推理",
            brand_embedding: "本地向量与重排序",
            brand_speech: "实时语音转文字",
            conversations: "对话",
            close_conversations: "关闭对话",
            new_chat: "+ 新建对话",
            search_conversations: "搜索对话",
            clear_search: "清除搜索",
            no_conversations_match: "没有匹配的对话。",
            backup_title: "对话备份",
            backup_description: "仅包含消息，不包含连接设置。",
            import: "导入",
            export: "导出",
            reading: "读取中…",
            open_conversations: "打开对话",
            models: "模型",
            diagnostics: "诊断",
            settings: "设置",
            language: "语言",
            home_eyebrow: "Bloom · 本地 AI 工作台",
            home_title: "让智能在你身边高效运行。",
            home_body: "一个专注、可观测的私有推理工作台。连接 Bloom 服务后，即可开始对话、检查模型或运行多模态任务。",
            capability_local: "本地优先",
            capability_local_body: "提示词只会在当前浏览器与配置的服务之间传输。",
            capability_compatible: "兼容 OpenAI",
            capability_compatible_body: "使用熟悉的 API，获得清晰可预期的运行状态。",
            capability_multimodal: "多模态",
            capability_multimodal_body: "在一个工作台中处理文本、视觉、语音、向量与重排序任务。",
            message_placeholder: "向 Bloom 提问… 回车发送 · Shift+Enter 换行",
            attach: "附件",
            attach_reading: "读取中…",
            remove_attachment: "移除图片附件",
            send: "发送",
            stop: "停止",
            composer_hint: "响应通过兼容 OpenAI 的 API 流式返回。",
            composer_waiting: "请等待模型就绪后再发送。",
            open_models: "打开模型",
            connection_settings: "连接设置",
            check_connection: "检查连接",
            update_api_key: "更新 API Key",
            check_server_version: "检查服务版本",
            view_model_status: "查看模型状态",
            review_models: "检查模型",
            choose_model: "选择模型开始使用",
            api_key_required: "需要 API Key",
            incompatible_server: "Bloom 服务不兼容",
            server_unavailable: "Bloom 服务不可用",
            connecting: "连接中…",
            ready: "就绪",
            transcription: "语音转写",
            embeddings: "向量",
            generation: "文本生成",
            loading: "加载中",
            choose_language: "选择界面语言",
        },
        Locale::Japanese => UiCopy {
            brand_chat: "ローカルマルチモーダル推論",
            brand_embedding: "ローカル埋め込みと再ランキング",
            brand_speech: "リアルタイム音声文字起こし",
            conversations: "会話",
            close_conversations: "会話を閉じる",
            new_chat: "+ 新しい会話",
            search_conversations: "会話を検索",
            clear_search: "検索をクリア",
            no_conversations_match: "一致する会話はありません。",
            backup_title: "会話のバックアップ",
            backup_description: "接続設定を除くメッセージのみを保存します。",
            import: "インポート",
            export: "エクスポート",
            reading: "読み込み中…",
            open_conversations: "会話を開く",
            models: "モデル",
            diagnostics: "診断",
            settings: "設定",
            language: "言語",
            home_eyebrow: "Bloom · ローカル AI ワークスペース",
            home_title: "身近な場所で、知性を自在に動かす。",
            home_body: "プライベートな推論のための、集中しやすく可観測なワークスペースです。Bloom サーバーに接続して会話やモデルを始めましょう。",
            capability_local: "ローカル優先",
            capability_local_body: "プロンプトはこのブラウザと設定済みサーバーの間だけで扱われます。",
            capability_compatible: "OpenAI 互換",
            capability_compatible_body: "使い慣れた API と明確な実行状態を利用できます。",
            capability_multimodal: "マルチモーダル",
            capability_multimodal_body: "テキスト、画像、音声、埋め込み、再ランキングを一つの環境で扱えます。",
            message_placeholder: "Bloom に質問… Enter で送信 · Shift+Enter で改行",
            attach: "添付",
            attach_reading: "読み込み中…",
            remove_attachment: "画像添付を削除",
            send: "送信",
            stop: "停止",
            composer_hint: "OpenAI 互換 API から応答をストリーミングします。",
            composer_waiting: "モデルの準備が整うまでお待ちください。",
            open_models: "モデルを開く",
            connection_settings: "接続設定",
            check_connection: "接続を確認",
            update_api_key: "API キーを更新",
            check_server_version: "サーバー版を確認",
            view_model_status: "モデル状態を表示",
            review_models: "モデルを確認",
            choose_model: "モデルを選択して開始",
            api_key_required: "API キーが必要です",
            incompatible_server: "Bloom サーバー非対応",
            server_unavailable: "Bloom サーバーを利用できません",
            connecting: "接続中…",
            ready: "準備完了",
            transcription: "音声文字起こし",
            embeddings: "埋め込み",
            generation: "生成",
            loading: "読み込み中",
            choose_language: "表示言語を選択",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{Locale, copy};

    #[test]
    fn language_tags_choose_a_supported_locale_with_english_fallback() {
        assert_eq!(
            Locale::from_language_tag("zh-CN"),
            Locale::SimplifiedChinese
        );
        assert_eq!(Locale::from_language_tag("ja-JP"), Locale::Japanese);
        assert_eq!(Locale::from_language_tag("fr-FR"), Locale::English);
    }

    #[test]
    fn every_locale_has_a_professional_home_message() {
        for locale in Locale::ALL {
            let ui = copy(locale);
            assert!(!ui.home_title.is_empty());
            assert!(!ui.home_body.is_empty());
            assert!(!ui.capability_multimodal_body.is_empty());
        }
    }
}
