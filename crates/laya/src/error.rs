use thiserror::Error;

/// Errors surfaced by the Laya engine.
#[derive(Debug, Error)]
pub enum LayaError {
    /// A question definition is malformed. The message names the question and what to fix,
    /// matching the Python `Agent._check_question` diagnostics.
    #[error("{0}")]
    InvalidQuestion(String),

    /// A checkpoint config / architecture problem.
    #[error("{0}")]
    Config(String),

    /// A required file or resource was not found.
    #[error("{0}")]
    NotFound(String),

    /// Tokenizer construction or encoding failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Model construction or inference failure.
    #[error("model error: {0}")]
    Model(String),

    /// Hugging Face Hub download failure.
    #[error("download error: {0}")]
    Download(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, LayaError>;
