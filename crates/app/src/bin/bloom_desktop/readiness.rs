//! Bounded validation of the launcher's local HTTP/1.0 readiness handshake.
use std::io;

use serde_json::Value;

pub(super) const MAX_RESPONSE_BYTES: usize = 80 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024;

pub(super) fn validate_response(response: &[u8]) -> io::Result<()> {
    let invalid = || io::Error::other("invalid or incompatible Bloom readiness response");
    let split = response
        .windows(4)
        .position(|part| part == b"\r\n\r\n")
        .ok_or_else(invalid)?;
    let body = &response[split + 4..];
    if split + 4 > MAX_HEADER_BYTES || body.len() > MAX_BODY_BYTES {
        return Err(invalid());
    }
    let headers = std::str::from_utf8(&response[..split]).map_err(|_| invalid())?;
    let mut lines = headers.split("\r\n");
    let mut status = lines.next().ok_or_else(invalid)?.split_whitespace();
    if !matches!(status.next(), Some("HTTP/1.0" | "HTTP/1.1")) {
        return Err(invalid());
    }
    let code = status.next().ok_or_else(invalid)?;
    if !matches!(code, "200" | "503") {
        return Err(invalid());
    }
    let mut content_length = None;
    let mut content_type = false;
    for line in lines {
        let (name, value) = line.split_once(':').ok_or_else(invalid)?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some()
                || value.is_empty()
                || !value.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(invalid());
            }
            content_length = Some(value.parse::<usize>().map_err(|_| invalid())?);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            // The probe requests HTTP/1.0 with connection-close framing.
            return Err(invalid());
        } else if name.eq_ignore_ascii_case("content-type") {
            if content_type
                || !value
                    .split(';')
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .eq_ignore_ascii_case("application/json")
            {
                return Err(invalid());
            }
            content_type = true;
        }
    }
    if !content_type || content_length.is_some_and(|length| length != body.len()) {
        return Err(invalid());
    }
    let value: Value = serde_json::from_slice(body).map_err(|_| invalid())?;
    if !valid_document(&value) || (code == "200") != (value["status"] == "ready") {
        return Err(invalid());
    }
    Ok(())
}

fn bounded_text(value: &Value, maximum: usize, trim: bool, no_whitespace: bool) -> bool {
    value.as_str().is_some_and(|text| {
        !text.is_empty()
            && text.chars().count() <= maximum
            && (!trim || text.trim() == text)
            && !text
                .chars()
                .any(|c| c.is_control() || (no_whitespace && c.is_whitespace()))
    })
}

fn valid_document(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if value["schema_version"].as_u64() != Some(3) || value["object"] != "bloom.readiness" {
        return false;
    }
    let version = |name: &str| {
        value[name]
            .as_u64()
            .filter(|v| *v > 0 && *v <= u32::MAX as u64)
    };
    let (Some(protocol), Some(minimum), Some(maximum)) = (
        version("protocol_version"),
        version("minimum_ui_protocol_version"),
        version("maximum_ui_protocol_version"),
    ) else {
        return false;
    };
    if minimum > maximum
        || !(minimum..=maximum).contains(&protocol)
        || !(minimum..=maximum).contains(&3)
    {
        return false;
    }
    let status = value["status"].as_str();
    if !matches!(status, Some("ready" | "not_ready"))
        || !bounded_text(&value["server_version"], 64, true, true)
        || !bounded_text(&value["model"], 256, true, false)
        || !value["progress"].as_u64().is_some_and(|n| n <= 100)
        || !value["loading"].is_boolean()
        || !object.contains_key("load_error")
        || !(value["load_error"].is_null() || bounded_text(&value["load_error"], 512, false, false))
        || !object.contains_key("context_window")
        || !(value["context_window"].is_null()
            || value["context_window"].as_u64().is_some_and(|n| n > 0))
        || value["in_flight_requests"].as_u64().is_none()
        || value["available_permits"].as_u64().is_none()
        || !value["memory_pressure_high"].is_boolean()
        || !value["ram_utilization"]
            .as_f64()
            .is_some_and(|n| n.is_finite() && (0.0..=1.0).contains(&n))
    {
        return false;
    }
    let Some(modalities) = value["input_modalities"].as_array() else {
        return false;
    };
    if modalities.len() > 16
        || modalities.iter().enumerate().any(|(i, item)| {
            !bounded_text(item, 64, false, false) || modalities[..i].contains(item)
        })
    {
        return false;
    }
    let Some(tasks) = value["model_tasks"].as_array() else {
        return false;
    };
    let tasks_valid = tasks.is_empty()
        || (tasks.len() == 1
            && matches!(
                tasks[0].as_str(),
                Some("generation" | "embedding" | "rerank" | "tensor")
            ))
        || (tasks.len() == 2 && tasks[0] == "embedding" && tasks[1] == "rerank");
    if !tasks_valid {
        return false;
    }
    status != Some("ready")
        || (value["model"] != "not loaded"
            && value["progress"] == 100
            && value["loading"] == false
            && value["load_error"].is_null()
            && !tasks.is_empty()
            && !value["context_window"].is_null()
            && value["available_permits"].as_u64().is_some_and(|n| n > 0)
            && value["memory_pressure_high"] == false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Value {
        serde_json::from_str(include_str!("../../../tests/fixtures/readiness.json")).unwrap()
    }
    fn response(code: u16, value: &Value) -> Vec<u8> {
        let body = serde_json::to_string(value).unwrap();
        format!("HTTP/1.0 {code} Status\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
    }
    #[test]
    fn requires_complete_contract_and_consistent_http_status() {
        let value = example();
        assert!(validate_response(&response(503, &value)).is_ok());
        assert!(validate_response(&response(200, &value)).is_err());
        for field in value.as_object().unwrap().keys() {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                validate_response(&response(503, &missing)).is_err(),
                "{field}"
            );
        }
        assert!(
            validate_response(b"HTTP/1.0 503 Unavailable\r\nContent-Length: 0\r\n\r\n").is_err()
        );
    }
    #[test]
    fn validates_protocol_ranges_types_and_ready_invariants() {
        let mut value = example();
        value["protocol_version"] = 4.into();
        value["maximum_ui_protocol_version"] = 4.into();
        assert!(valid_document(&value));
        value["minimum_ui_protocol_version"] = 4.into();
        assert!(!valid_document(&value));
        for (field, invalid) in [
            ("progress", serde_json::json!(true)),
            ("progress", serde_json::json!(101)),
            ("model_tasks", serde_json::json!(["generation", "tensor"])),
            ("input_modalities", serde_json::json!(["text", "text"])),
            ("ram_utilization", serde_json::json!(1.1)),
            ("context_window", serde_json::json!(0)),
            ("server_version", serde_json::json!("bad\nversion")),
        ] {
            let mut invalid_value = example();
            invalid_value[field] = invalid;
            assert!(!valid_document(&invalid_value), "{field}");
        }
        let mut ready = example();
        ready["status"] = "ready".into();
        assert!(!valid_document(&ready));
        ready["model"] = "model".into();
        ready["progress"] = 100.into();
        ready["model_tasks"] = serde_json::json!(["generation"]);
        ready["context_window"] = 2048.into();
        assert!(validate_response(&response(200, &ready)).is_ok());
    }
    #[test]
    fn rejects_truncated_and_oversized_bodies() {
        let mut bytes = response(503, &example());
        bytes.pop();
        assert!(validate_response(&bytes).is_err());
        let mut value = example();
        value["extra"] = "x".repeat(MAX_BODY_BYTES).into();
        assert!(validate_response(&response(503, &value)).is_err());
    }
}
