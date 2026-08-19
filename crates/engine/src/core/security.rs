use anyhow::Result;
use std::path::Path;

/// Check if strict security checks are requested.
pub fn is_strict_security() -> bool {
    std::env::var("BLOOM_STRICT_SECURITY")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

/// Validates that an external script file is allowed to be run.
pub fn validate_external_script(path: &Path) -> Result<()> {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

    let is_default_safe = matches!(
        filename,
        "fun_asr_infer.py"
            | "qwen_asr_infer.py"
            | "openvino_llm_infer.py"
            | "npu_tts_infer.py"
            | "generate_kernel.py"
    );

    let is_strict = is_strict_security();

    // Load allowlist from environment variable
    let allowed_env = std::env::var("BLOOM_ALLOWED_SCRIPTS").unwrap_or_default();
    let allowed_list: Vec<&str> = allowed_env
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let path_str = path.to_string_lossy();

    // Check if path or filename is explicitly allowed
    let is_explicitly_allowed = allowed_list
        .iter()
        .any(|&allowed| path_str.contains(allowed) || filename == allowed);

    if is_explicitly_allowed {
        return Ok(());
    }

    if is_strict {
        if is_default_safe {
            tracing::info!("Allowing default safe script: {}", filename);
            return Ok(());
        }
        anyhow::bail!(
            "Security Error: External script '{}' is not in the allowlist (BLOOM_ALLOWED_SCRIPTS). \
             Running under BLOOM_STRICT_SECURITY=1 rejects this execution.",
            path_str
        );
    } else {
        if !is_default_safe {
            tracing::warn!(
                "Security Warning: Running external script '{}' which is not explicitly allowlisted. \
                 To secure this, set BLOOM_STRICT_SECURITY=1 and configure BLOOM_ALLOWED_SCRIPTS.",
                path_str
            );
        }
        Ok(())
    }
}

/// Validates that an external runner (binary/executable) is allowed to be run.
pub fn validate_runner(path: &Path) -> Result<()> {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

    let is_default_safe = filename == "llama-server"
        || filename == "llama-cli"
        || filename == "ffmpeg"
        || filename == "sysctl"
        || filename == "nvidia-smi"
        || filename == "wmic"
        || filename == "powershell"
        || filename == "ps"
        || filename == "vm_stat";
    let is_strict = is_strict_security();

    // Load allowlist from environment variable
    let allowed_env = std::env::var("BLOOM_ALLOWED_RUNNERS").unwrap_or_default();
    let allowed_list: Vec<&str> = allowed_env
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let path_str = path.to_string_lossy();
    let is_explicitly_allowed = allowed_list
        .iter()
        .any(|&allowed| path_str.contains(allowed) || filename == allowed);

    if is_explicitly_allowed {
        return Ok(());
    }

    if is_strict {
        if is_default_safe {
            return Ok(());
        }
        anyhow::bail!(
            "Security Error: External runner '{}' is not in the allowlist (BLOOM_ALLOWED_RUNNERS). \
             Running under BLOOM_STRICT_SECURITY=1 rejects this execution.",
            path_str
        );
    } else {
        if !is_default_safe {
            tracing::warn!(
                "Security Warning: Running external runner '{}' which is not explicitly allowlisted. \
                 To secure this, set BLOOM_STRICT_SECURITY=1 and configure BLOOM_ALLOWED_RUNNERS.",
                path_str
            );
        }
        Ok(())
    }
}

/// Validates that a plugin is allowed to be loaded.
pub fn validate_plugin(name: &str) -> Result<()> {
    let is_strict = is_strict_security();

    let allowed_env = std::env::var("BLOOM_ALLOWED_PLUGINS").unwrap_or_default();
    let allowed_list: Vec<&str> = allowed_env
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let is_explicitly_allowed = allowed_list
        .iter()
        .any(|&allowed| name == allowed || name.contains(allowed));

    if is_explicitly_allowed {
        return Ok(());
    }

    // In unit tests, there might be mock plugins.
    if name.starts_with("test-") || name.starts_with("mock-") {
        return Ok(());
    }

    if is_strict {
        anyhow::bail!(
            "Security Error: Plugin '{}' is not in the allowlist (BLOOM_ALLOWED_PLUGINS). \
             Running under BLOOM_STRICT_SECURITY=1 rejects this plugin.",
            name
        );
    } else {
        tracing::warn!(
            "Security Warning: Loading plugin '{}' which is not explicitly allowlisted. \
             To secure this, set BLOOM_STRICT_SECURITY=1 and configure BLOOM_ALLOWED_PLUGINS.",
            name
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_features_sequentially() {
        // 1. Parse strict security
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_STRICT_SECURITY", "1") };
        assert!(is_strict_security());
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_STRICT_SECURITY", "true") };
        assert!(is_strict_security());
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_STRICT_SECURITY", "0") };
        assert!(!is_strict_security());

        // 2. Script allowlist
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_STRICT_SECURITY", "1") };
        // Default safe script is allowed
        assert!(validate_external_script(Path::new("fun_asr_infer.py")).is_ok());
        // Arbitrary script is rejected in strict mode
        assert!(validate_external_script(Path::new("evil.py")).is_err());
        // Arbitrary script is allowed if in allowlist
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_ALLOWED_SCRIPTS", "evil.py,test.py") };
        assert!(validate_external_script(Path::new("evil.py")).is_ok());
        // Normal mode warns but allows
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_STRICT_SECURITY", "0") };
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_ALLOWED_SCRIPTS") };
        assert!(validate_external_script(Path::new("evil.py")).is_ok());

        // 3. Runner allowlist
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_STRICT_SECURITY", "1") };
        // Default safe runner is allowed
        assert!(validate_runner(Path::new("llama-server")).is_ok());
        // Arbitrary runner is rejected in strict mode
        assert!(validate_runner(Path::new("evil-runner")).is_err());
        // Arbitrary runner is allowed if in allowlist
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_ALLOWED_RUNNERS", "evil-runner") };
        assert!(validate_runner(Path::new("evil-runner")).is_ok());

        // 4. Plugin allowlist
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_STRICT_SECURITY", "1") };
        // Arbitrary plugin is rejected in strict mode
        assert!(validate_plugin("com.evil.plugin").is_err());
        // Arbitrary plugin is allowed if in allowlist
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("BLOOM_ALLOWED_PLUGINS", "com.evil.plugin") };
        assert!(validate_plugin("com.evil.plugin").is_ok());
        // Test mock/test plugins are allowed
        assert!(validate_plugin("test-plugin").is_ok());

        // Cleanup
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_STRICT_SECURITY") };
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_ALLOWED_SCRIPTS") };
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_ALLOWED_RUNNERS") };
        // FIXME: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("BLOOM_ALLOWED_PLUGINS") };
    }
}
