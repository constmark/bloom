use anyhow::Result;
use std::path::Path;

/// Check if strict security checks are requested.
pub fn is_strict_security() -> bool {
    strict_security_value(std::env::var("BLOOM_STRICT_SECURITY").ok().as_deref())
}

fn strict_security_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| matches!(value, "1" | "true" | "TRUE" | "yes" | "YES"))
}

// A bare name authorizes that exact basename. A path authorizes only the
// corresponding file, never siblings, path prefixes, or substring matches.
fn path_is_allowlisted(path: &Path, allowlist: &str) -> bool {
    allowlist
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .any(|entry| {
            let allowed = Path::new(entry);
            if allowed.components().count() == 1 && allowed.file_name() == Some(allowed.as_os_str())
            {
                path.file_name() == Some(allowed.as_os_str())
            } else {
                path == allowed
                    || path
                        .canonicalize()
                        .ok()
                        .zip(allowed.canonicalize().ok())
                        .is_some_and(|(actual, expected)| actual == expected)
            }
        })
}

fn is_bundled_script(path: &Path) -> bool {
    let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if !matches!(
        filename,
        "fun_asr_infer.py" | "qwen_asr_infer.py" | "openvino_llm_infer.py" | "npu_tts_infer.py"
    ) {
        return false;
    }
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts")
        .join(filename);
    path.canonicalize()
        .ok()
        .zip(bundled.canonicalize().ok())
        .is_some_and(|(actual, expected)| actual == expected)
}

/// Validate a script before execution. Strict mode trusts bundled script paths
/// and exact entries from `BLOOM_ALLOWED_SCRIPTS`.
pub fn validate_external_script(path: &Path) -> Result<()> {
    validate_path(
        path,
        is_strict_security(),
        &std::env::var("BLOOM_ALLOWED_SCRIPTS").unwrap_or_default(),
        is_bundled_script(path),
        "script",
        "BLOOM_ALLOWED_SCRIPTS",
    )
}

/// Validate a runner before execution. Default runner names are resolved by the
/// operating system through the operator's PATH; custom paths require a grant.
pub fn validate_runner(path: &Path) -> Result<()> {
    let default_safe = matches!(
        path.to_str(),
        Some(
            "llama-server"
                | "llama-cli"
                | "ffmpeg"
                | "sysctl"
                | "nvidia-smi"
                | "wmic"
                | "powershell"
                | "ps"
                | "vm_stat"
        )
    );
    validate_path(
        path,
        is_strict_security(),
        &std::env::var("BLOOM_ALLOWED_RUNNERS").unwrap_or_default(),
        default_safe,
        "runner",
        "BLOOM_ALLOWED_RUNNERS",
    )
}

fn validate_path(
    path: &Path,
    strict: bool,
    allowlist: &str,
    default_safe: bool,
    kind: &str,
    variable: &str,
) -> Result<()> {
    if default_safe || path_is_allowlisted(path, allowlist) {
        return Ok(());
    }
    if strict {
        anyhow::bail!(
            "Security Error: External {kind} '{}' is not in the allowlist ({variable}). Running under BLOOM_STRICT_SECURITY=1 rejects this execution.",
            path.display()
        );
    }
    tracing::warn!(
        "Security Warning: Running external {kind} '{}' which is not explicitly allowlisted. Set BLOOM_STRICT_SECURITY=1 and configure {variable} to restrict execution.",
        path.display()
    );
    Ok(())
}

/// Validate a plugin identity. Test and mock names receive no production bypass.
pub fn validate_plugin(name: &str) -> Result<()> {
    validate_plugin_with_policy(
        name,
        is_strict_security(),
        &std::env::var("BLOOM_ALLOWED_PLUGINS").unwrap_or_default(),
    )
}

fn validate_plugin_with_policy(name: &str, strict: bool, allowlist: &str) -> Result<()> {
    if !name.is_empty()
        && allowlist
            .split(',')
            .map(str::trim)
            .any(|entry| entry == name)
    {
        return Ok(());
    }
    if strict {
        anyhow::bail!(
            "Security Error: Plugin '{name}' is not in the allowlist (BLOOM_ALLOWED_PLUGINS). Running under BLOOM_STRICT_SECURITY=1 rejects this plugin."
        );
    }
    tracing::warn!(
        "Security Warning: Loading plugin '{name}' which is not explicitly allowlisted. Set BLOOM_STRICT_SECURITY=1 and configure BLOOM_ALLOWED_PLUGINS to restrict loading."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_security_parsing_does_not_mutate_process_environment() {
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert!(strict_security_value(Some(value)));
        }
        for value in [None, Some("0"), Some("false"), Some("")] {
            assert!(!strict_security_value(value));
        }
    }

    #[test]
    fn allowlists_match_exact_names_and_paths() {
        assert!(path_is_allowlisted(
            Path::new("/opt/bin/python3"),
            "python3"
        ));
        assert!(!path_is_allowlisted(
            Path::new("/opt/bin/python3-evil"),
            "python3"
        ));
        assert!(!path_is_allowlisted(
            Path::new("/opt/python3/evil"),
            "python3"
        ));
        assert!(path_is_allowlisted(
            Path::new("/opt/bin/python3"),
            " /opt/bin/python3 , other"
        ));
        assert!(!path_is_allowlisted(
            Path::new("/opt/bin/python3-evil"),
            "/opt/bin/python3"
        ));
        assert!(!path_is_allowlisted(Path::new("/opt/bin/python3"), " , "));
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("test.py");
        std::fs::write(&script, "").unwrap();
        assert!(path_is_allowlisted(
            &dir.path().join("./test.py"),
            script.to_str().unwrap()
        ));
    }

    #[test]
    fn bundled_script_name_does_not_authorize_a_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let replacement = dir.path().join("fun_asr_infer.py");
        std::fs::write(&replacement, "").unwrap();
        assert!(!is_bundled_script(&replacement));
        assert!(
            validate_path(
                &replacement,
                true,
                "",
                false,
                "script",
                "BLOOM_ALLOWED_SCRIPTS"
            )
            .is_err()
        );
        assert!(
            validate_path(
                &replacement,
                false,
                "",
                false,
                "script",
                "BLOOM_ALLOWED_SCRIPTS"
            )
            .is_ok()
        );
    }

    #[test]
    fn strict_plugins_reject_prefix_suffix_and_test_bypasses() {
        for name in [
            "evil-org.safe",
            "org.safe.evil",
            "test-plugin",
            "mock-plugin",
            "",
        ] {
            assert!(validate_plugin_with_policy(name, true, "org.safe").is_err());
        }
        assert!(validate_plugin_with_policy("org.safe", true, " org.safe ,other").is_ok());
        assert!(validate_plugin_with_policy("test-plugin", true, "test-plugin").is_ok());
        assert!(validate_plugin_with_policy("custom", false, "").is_ok());
    }
}
