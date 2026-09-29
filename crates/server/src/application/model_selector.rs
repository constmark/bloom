//! Model selector validation and source labels shared by all entry points.
pub(crate) const MAX_REQUESTED_MODEL_ID_CHARS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestedModelError {
    Invalid,
    NotLoaded,
    Revoked,
}

/// Bind an optional OpenAI-compatible model selector to the active runtime.
///
/// Omitting the field remains backward compatible, and `default` is an explicit
/// alias for the one active model. Any other identifier must match exactly so a
/// client can never request one model while Bloom silently executes another.
#[cfg(test)]
pub(crate) fn validate_requested_model(
    requested: Option<&str>,
    active_model: &str,
) -> std::result::Result<(), RequestedModelError> {
    let Some(requested) = requested else {
        return Ok(());
    };
    validate_model_selector(requested)?;
    if requested == "default" || requested == active_model {
        Ok(())
    } else {
        Err(RequestedModelError::NotLoaded)
    }
}

pub(crate) fn validate_model_selector(
    requested: &str,
) -> std::result::Result<(), RequestedModelError> {
    if requested.is_empty()
        || requested.trim() != requested
        || requested
            .chars()
            .take(MAX_REQUESTED_MODEL_ID_CHARS + 1)
            .count()
            > MAX_REQUESTED_MODEL_ID_CHARS
        || requested.chars().any(char::is_control)
    {
        return Err(RequestedModelError::Invalid);
    }
    Ok(())
}

pub(crate) fn model_path_label(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("external model")
        .to_string()
}
