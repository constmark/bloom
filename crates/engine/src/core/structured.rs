//! Final validation shared by native grammar sampling and external adapters.
//!
//! The HTTP boundary validates the admitted schema subset. These checks verify
//! the returned value before an engine emits its successful terminal event.

pub(crate) fn validate_complete_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> std::result::Result<(), String> {
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array())
        && !enum_values.iter().any(|candidate| candidate == value)
    {
        return Err(format!("{} does not match any allowed enum value", path));
    }

    if let Some(schema_type) = schema.get("type").and_then(|v| v.as_str())
        && !json_value_matches_type(value, schema_type)
    {
        return Err(format!("{} expected type {}", path, schema_type));
    }

    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            for field in required.iter().filter_map(|v| v.as_str()) {
                if !object.contains_key(field) {
                    return Err(format!("{} missing required property '{}'", path, field));
                }
            }
        }
        let properties = schema.get("properties").and_then(|v| v.as_object());
        if let Some(properties) = properties {
            for (field, property_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_complete_json_schema(
                        field_value,
                        property_schema,
                        &format!("{}.{}", path, field),
                    )?;
                }
            }
        }
        if schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false) {
            for field in object.keys() {
                if !properties.is_some_and(|properties| properties.contains_key(field)) {
                    return Err(format!(
                        "{} contains unsupported property '{}'",
                        path, field
                    ));
                }
            }
        }
    }

    if let Some(item_schema) = schema.get("items")
        && let Some(items) = value.as_array()
    {
        for (idx, item) in items.iter().enumerate() {
            validate_complete_json_schema(item, item_schema, &format!("{}[{}]", path, idx))?;
        }
    }

    Ok(())
}

pub(crate) fn json_value_matches_type(value: &serde_json::Value, schema_type: &str) -> bool {
    match schema_type {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value.as_f64().is_some_and(|number| number.fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn closed_objects_without_a_property_map_reject_all_fields() {
        let schema = json!({"type":"object", "additionalProperties":false});
        assert!(validate_complete_json_schema(&json!({}), &schema, "$").is_ok());
        assert!(validate_complete_json_schema(&json!({"unexpected":1}), &schema, "$").is_err());
        let schema = json!({"type":"object", "properties":{"closed":{"type":"object", "additionalProperties":false}}});
        assert!(
            validate_complete_json_schema(&json!({"closed":{"unexpected":1}}), &schema, "$")
                .is_err()
        );
    }

    #[test]
    fn complete_schema_validation_checks_nested_values_and_required_enum_fields() {
        let schema = json!({"type":"object", "properties":{"values":{"type":"array", "items":{"type":"integer", "enum":[1,2]}}}, "required":["values"], "additionalProperties":false});
        assert!(validate_complete_json_schema(&json!({"values":[1,2]}), &schema, "$").is_ok());
        for value in [
            json!({}),
            json!({"values":[3]}),
            json!({"values":["1"]}),
            json!({"values":null}),
            json!({"values":[],"extra":true}),
        ] {
            assert!(validate_complete_json_schema(&value, &schema, "$").is_err());
        }
    }
}
