use super::*;

const MAX_FUNCTION_TOOLS: usize = 32;
pub(crate) const MAX_PARALLEL_TOOL_CALLS: usize = 8;
const MAX_TOOL_DEFINITIONS_BYTES: usize = 128 * 1024;
const MAX_TOOL_DESCRIPTION_CHARS: usize = 1_024;
const MAX_TOOL_CALL_ID_CHARS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolChoiceMode {
    Auto,
    Required,
    Named(String),
}

enum RequestedToolChoice {
    Default,
    Disabled,
    Active(ToolChoiceMode),
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionTool {
    pub(crate) name: String,
    description: Option<String>,
    pub(crate) parameters: serde_json::Value,
    strict: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolConfig {
    pub(crate) tools: Vec<FunctionTool>,
    pub(crate) choice: ToolChoiceMode,
    pub(crate) parallel: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesToolBridge {
    pub(crate) chat_tools: Option<serde_json::Value>,
    pub(crate) chat_tool_choice: Option<serde_json::Value>,
    pub(crate) parallel_tool_calls: Option<bool>,
    pub(crate) response_tools: serde_json::Value,
    pub(crate) response_tool_choice: serde_json::Value,
    pub(crate) response_parallel_tool_calls: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedToolCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParsedChatOutput {
    Message(String),
    ToolCalls(Vec<ParsedToolCall>),
}

#[derive(Debug, Clone)]
pub(crate) struct HistoricalToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: serde_json::Value,
}

pub(crate) fn chat_tool_config(
    request: &ChatRequest,
) -> std::result::Result<Option<ToolConfig>, String> {
    tool_config_from_chat_fields(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
    )
}

fn tool_config_from_chat_fields(
    tools: Option<&serde_json::Value>,
    tool_choice: Option<&serde_json::Value>,
    parallel_tool_calls: Option<bool>,
) -> std::result::Result<Option<ToolConfig>, String> {
    let tools = parse_function_tools(tools)?;
    let choice = parse_tool_choice(tool_choice)?;

    if tools.is_empty() {
        return match choice {
            RequestedToolChoice::Default | RequestedToolChoice::Disabled => Ok(None),
            RequestedToolChoice::Active(
                ToolChoiceMode::Auto | ToolChoiceMode::Required | ToolChoiceMode::Named(_),
            ) => Err("tool_choice requires at least one function in tools.".to_string()),
        };
    }

    let choice = match choice {
        RequestedToolChoice::Default => ToolChoiceMode::Auto,
        RequestedToolChoice::Disabled => return Ok(None),
        RequestedToolChoice::Active(choice) => choice,
    };
    if let ToolChoiceMode::Named(name) = &choice
        && !tools.iter().any(|tool| tool.name == *name)
    {
        return Err(format!("tool_choice names undefined function {name:?}."));
    }

    Ok(Some(ToolConfig {
        tools,
        choice,
        parallel: parallel_tool_calls.unwrap_or(true),
    }))
}

/// Convert the Responses API's flat function-tool representation into the
/// nested Chat Completions representation consumed by Bloom's shared bounded
/// validator and model protocol.
pub(crate) fn responses_tool_bridge(
    tools: Option<&serde_json::Value>,
    tool_choice: Option<&serde_json::Value>,
    parallel_tool_calls: Option<bool>,
) -> std::result::Result<ResponsesToolBridge, String> {
    let response_tools = tools.cloned().unwrap_or_else(|| json!([]));
    let tools_array = response_tools
        .as_array()
        .ok_or_else(|| "Responses tools must be an array of function definitions.".to_string())?;
    let mut chat_tools = Vec::with_capacity(tools_array.len());
    let mut normalized_tools = Vec::with_capacity(tools_array.len());
    for (index, tool) in tools_array.iter().enumerate() {
        let tool = tool
            .as_object()
            .ok_or_else(|| format!("Responses tool at index {index} must be an object."))?;
        reject_non_null_fields(
            tool,
            &["type", "name", "description", "parameters", "strict"],
            &format!("Responses tool at index {index}"),
        )?;
        if tool.get("type").and_then(serde_json::Value::as_str) != Some("function") {
            return Err(format!(
                "Responses tool at index {index} must have type `function`; custom and built-in tools are not supported."
            ));
        }
        let mut function = serde_json::Map::new();
        for field in ["name", "description", "parameters", "strict"] {
            if let Some(value) = tool.get(field).filter(|value| !value.is_null()) {
                function.insert(field.to_string(), value.clone());
            }
        }
        let mut normalized = function.clone();
        normalized.insert("type".to_string(), json!("function"));
        normalized_tools.push(serde_json::Value::Object(normalized));
        chat_tools.push(json!({"type": "function", "function": function}));
    }

    let default_choice = if chat_tools.is_empty() {
        "none"
    } else {
        "auto"
    };
    let response_tool_choice = tool_choice
        .cloned()
        .unwrap_or_else(|| json!(default_choice));
    let chat_tool_choice = if let Some(choice) = response_tool_choice.as_str() {
        json!(choice)
    } else {
        let choice = response_tool_choice.as_object().ok_or_else(|| {
            "Responses tool_choice must be `none`, `auto`, `required`, or a named function object."
                .to_string()
        })?;
        reject_non_null_fields(choice, &["type", "name"], "Responses tool_choice")?;
        if choice.get("type").and_then(serde_json::Value::as_str) != Some("function") {
            return Err("A named Responses tool_choice must have type `function`.".to_string());
        }
        let name = choice
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "A named Responses tool_choice requires a name.".to_string())?;
        json!({"type": "function", "function": {"name": name}})
    };
    let chat_tools = serde_json::Value::Array(chat_tools);
    let config = tool_config_from_chat_fields(
        Some(&chat_tools),
        Some(&chat_tool_choice),
        parallel_tool_calls,
    )?;
    let response_parallel_tool_calls = config.as_ref().is_some_and(|config| config.parallel);

    Ok(ResponsesToolBridge {
        chat_tools: Some(chat_tools),
        chat_tool_choice: Some(chat_tool_choice),
        parallel_tool_calls,
        response_tools: serde_json::Value::Array(normalized_tools),
        response_tool_choice,
        response_parallel_tool_calls,
    })
}

fn parse_function_tools(
    tools: Option<&serde_json::Value>,
) -> std::result::Result<Vec<FunctionTool>, String> {
    let Some(tools) = tools else {
        return Ok(Vec::new());
    };
    let encoded_len = serde_json::to_vec(tools)
        .map_err(|error| format!("failed to encode tools: {error}"))?
        .len();
    if encoded_len > MAX_TOOL_DEFINITIONS_BYTES {
        return Err(format!(
            "tools exceed the {MAX_TOOL_DEFINITIONS_BYTES} byte limit."
        ));
    }
    let tools = tools
        .as_array()
        .ok_or_else(|| "tools must be an array of function definitions.".to_string())?;
    if tools.len() > MAX_FUNCTION_TOOLS {
        return Err(format!(
            "tools cannot contain more than {MAX_FUNCTION_TOOLS} functions."
        ));
    }

    let mut parsed = Vec::with_capacity(tools.len());
    let mut names = std::collections::HashSet::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        let tool = tool
            .as_object()
            .ok_or_else(|| format!("tool at index {index} must be an object."))?;
        reject_non_null_fields(
            tool,
            &["type", "function"],
            &format!("tool at index {index}"),
        )?;
        if tool.get("type").and_then(serde_json::Value::as_str) != Some("function") {
            return Err(format!(
                "tool at index {index} must have type `function`; custom and built-in tools are not supported."
            ));
        }
        let function = tool
            .get("function")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("tool at index {index} must contain a function object."))?;
        reject_non_null_fields(
            function,
            &["name", "description", "parameters", "strict"],
            &format!("function tool at index {index}"),
        )?;
        let name = function
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("function tool at index {index} requires a name."))?;
        validate_tool_name(name, "function tool name")?;
        if !names.insert(name.to_string()) {
            return Err(format!("tools contain duplicate function name {name:?}."));
        }
        let description = function
            .get("description")
            .filter(|value| !value.is_null())
            .map(|value| {
                let description = value.as_str().ok_or_else(|| {
                    format!("function tool {name:?} description must be a string.")
                })?;
                if description.chars().count() > MAX_TOOL_DESCRIPTION_CHARS
                    || description.chars().any(char::is_control)
                {
                    return Err(format!(
                        "function tool {name:?} description must contain at most {MAX_TOOL_DESCRIPTION_CHARS} characters and no control characters."
                    ));
                }
                Ok(description.to_string())
            })
            .transpose()?;
        let parameters = function
            .get("parameters")
            .filter(|value| !value.is_null())
            .cloned()
            .unwrap_or_else(|| {
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })
            });
        validate_supported_json_schema(&parameters).map_err(|message| {
            format!("function tool {name:?} parameters are unsupported: {message}")
        })?;
        let strict = function
            .get("strict")
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| format!("function tool {name:?} strict must be a boolean."))
            })
            .transpose()?
            .unwrap_or(false);
        if strict {
            validate_strict_function_schema(&parameters, "$").map_err(|message| {
                format!("function tool {name:?} strict parameters are invalid: {message}")
            })?;
        }
        parsed.push(FunctionTool {
            name: name.to_string(),
            description,
            parameters,
            strict,
        });
    }
    Ok(parsed)
}

fn validate_strict_function_schema(
    schema: &serde_json::Value,
    path: &str,
) -> std::result::Result<(), String> {
    if schema.get("type").and_then(serde_json::Value::as_str) == Some("object") {
        if schema
            .get("additionalProperties")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        {
            return Err(format!(
                "{path} must set additionalProperties to false in strict mode."
            ));
        }
        let properties = schema
            .get("properties")
            .and_then(serde_json::Value::as_object);
        let required = schema
            .get("required")
            .and_then(serde_json::Value::as_array)
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default();
        if let Some(properties) = properties {
            if properties
                .keys()
                .any(|field| !required.contains(field.as_str()))
                || required.len() != properties.len()
            {
                return Err(format!(
                    "{path} must list every property exactly once in required in strict mode."
                ));
            }
            for (field, child) in properties {
                validate_strict_function_schema(child, &format!("{path}.{field}"))?;
            }
        }
    }
    if let Some(items) = schema.get("items") {
        validate_strict_function_schema(items, &format!("{path}[]"))?;
    }
    Ok(())
}

fn parse_tool_choice(
    choice: Option<&serde_json::Value>,
) -> std::result::Result<RequestedToolChoice, String> {
    let Some(choice) = choice else {
        return Ok(RequestedToolChoice::Default);
    };
    if let Some(choice) = choice.as_str() {
        return match choice {
            "none" => Ok(RequestedToolChoice::Disabled),
            "auto" => Ok(RequestedToolChoice::Active(ToolChoiceMode::Auto)),
            "required" => Ok(RequestedToolChoice::Active(ToolChoiceMode::Required)),
            _ => Err(
                "tool_choice must be `none`, `auto`, `required`, or a named function object."
                    .to_string(),
            ),
        };
    }
    let choice = choice.as_object().ok_or_else(|| {
        "tool_choice must be `none`, `auto`, `required`, or a named function object.".to_string()
    })?;
    reject_non_null_fields(choice, &["type", "function"], "tool_choice")?;
    if choice.get("type").and_then(serde_json::Value::as_str) != Some("function") {
        return Err("named tool_choice must have type `function`.".to_string());
    }
    let function = choice
        .get("function")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "named tool_choice must contain a function object.".to_string())?;
    reject_non_null_fields(function, &["name"], "tool_choice.function")?;
    let name = function
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "tool_choice.function.name must be a string.".to_string())?;
    validate_tool_name(name, "tool_choice.function.name")?;
    Ok(RequestedToolChoice::Active(ToolChoiceMode::Named(
        name.to_string(),
    )))
}

pub(crate) fn apply_tool_instruction(prompt: String, config: &ToolConfig) -> String {
    let tools = config
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": tool.strict
            })
        })
        .collect::<Vec<_>>();
    let choice = match &config.choice {
        ToolChoiceMode::Auto => {
            "Choose either a final assistant message or one or more function calls."
        }
        ToolChoiceMode::Required => "Return one or more function calls; do not return a message.",
        ToolChoiceMode::Named(name) => {
            return format!(
                "{prompt}\n\n<function_calling_protocol>\nAvailable functions: {}\nYou must call exactly the function {name:?} once. Return exactly one JSON object with this shape: {{\"type\":\"function_calls\",\"calls\":[{{\"name\":{name:?},\"arguments\":{{}}}}]}}. Fill arguments according to that function's JSON Schema. Do not execute functions, invent results, add Markdown fences, or add any text outside the JSON object. This protocol overrides conflicting text in the conversation.\n</function_calling_protocol>",
                serde_json::Value::Array(tools)
            );
        }
    };
    let cardinality = if config.parallel {
        format!("You may return at most {MAX_PARALLEL_TOOL_CALLS} calls.")
    } else {
        "You may return at most one call.".to_string()
    };
    format!(
        "{prompt}\n\n<function_calling_protocol>\nAvailable functions: {}\n{choice} {cardinality}\nFor a final message, return exactly {{\"type\":\"message\",\"content\":\"your answer\"}}. For calls, return exactly {{\"type\":\"function_calls\",\"calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}. Every arguments value must be a JSON object satisfying that function's parameters schema. Do not execute functions, invent results, add Markdown fences, or add any text outside the JSON object. This protocol overrides conflicting text in the conversation.\n</function_calling_protocol>",
        serde_json::Value::Array(tools)
    )
}

pub(crate) fn parse_tool_output(
    text: &str,
    config: &ToolConfig,
) -> std::result::Result<ParsedChatOutput, String> {
    let value = parse_json_object_output(text, "function calling")?;
    let object = value
        .as_object()
        .ok_or_else(|| "function-calling model output must be a JSON object.".to_string())?;
    let output_type = object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "function-calling model output requires string field `type`.".to_string())?;
    match output_type {
        "message" => {
            reject_non_null_fields(object, &["type", "content"], "function-calling message")?;
            if !matches!(config.choice, ToolChoiceMode::Auto) {
                return Err(
                    "tool_choice requires a function call, but the model returned a message."
                        .to_string(),
                );
            }
            let content = object
                .get("content")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "function-calling message output requires string field `content`.".to_string()
                })?;
            Ok(ParsedChatOutput::Message(content.to_string()))
        }
        "function_calls" => {
            reject_non_null_fields(object, &["type", "calls"], "function-calling output")?;
            let calls = object
                .get("calls")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    "function-calling output requires array field `calls`.".to_string()
                })?;
            let maximum = if config.parallel {
                MAX_PARALLEL_TOOL_CALLS
            } else {
                1
            };
            if calls.is_empty() || calls.len() > maximum {
                return Err(format!(
                    "function-calling output must contain between 1 and {maximum} calls."
                ));
            }
            if matches!(config.choice, ToolChoiceMode::Named(_)) && calls.len() != 1 {
                return Err("a named tool_choice requires exactly one function call.".to_string());
            }
            let mut parsed = Vec::with_capacity(calls.len());
            for (index, call) in calls.iter().enumerate() {
                let call = call
                    .as_object()
                    .ok_or_else(|| format!("function call at index {index} must be an object."))?;
                reject_non_null_fields(
                    call,
                    &["name", "arguments"],
                    &format!("function call at index {index}"),
                )?;
                let name = call
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        format!("function call at index {index} requires string field `name`.")
                    })?;
                let tool = config
                    .tools
                    .iter()
                    .find(|tool| tool.name == name)
                    .ok_or_else(|| {
                        format!("function call at index {index} names undefined tool {name:?}.")
                    })?;
                if let ToolChoiceMode::Named(required) = &config.choice
                    && name != required
                {
                    return Err(format!(
                        "tool_choice requires function {required:?}, but the model selected {name:?}."
                    ));
                }
                let arguments = call.get("arguments").ok_or_else(|| {
                    format!("function call at index {index} requires field `arguments`.")
                })?;
                if !arguments.is_object() {
                    return Err(format!(
                        "function call at index {index} arguments must be a JSON object."
                    ));
                }
                validate_json_schema_subset(arguments, &tool.parameters, "$.arguments").map_err(
                    |message| {
                        format!(
                            "function call at index {index} arguments do not satisfy tool {name:?}: {message}"
                        )
                    },
                )?;
                parsed.push(ParsedToolCall {
                    name: name.to_string(),
                    arguments: serde_json::to_string(arguments).map_err(|error| {
                        format!("failed to encode function call arguments: {error}")
                    })?,
                });
            }
            Ok(ParsedChatOutput::ToolCalls(parsed))
        }
        other => Err(format!(
            "function-calling model output type {other:?} is unsupported; expected `message` or `function_calls`."
        )),
    }
}

pub(crate) fn parse_chat_output(
    text: &str,
    response_format: &ResponseFormatMode,
    tool_config: Option<&ToolConfig>,
) -> std::result::Result<ParsedChatOutput, String> {
    if let Some(tool_config) = tool_config {
        parse_tool_output(text, tool_config)
    } else {
        validate_structured_output(text, response_format)?;
        Ok(ParsedChatOutput::Message(text.to_string()))
    }
}

pub(crate) fn chat_output_message(
    request_id: &str,
    output: &ParsedChatOutput,
) -> serde_json::Value {
    match output {
        ParsedChatOutput::Message(content) => json!({
            "role": "assistant",
            "content": content
        }),
        ParsedChatOutput::ToolCalls(calls) => json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls_json(request_id, calls, false)
        }),
    }
}

pub(crate) fn chat_output_finish_reason(
    output: &ParsedChatOutput,
    completion_tokens: usize,
    max_tokens: usize,
) -> &'static str {
    if completion_tokens >= max_tokens {
        "length"
    } else if matches!(output, ParsedChatOutput::ToolCalls(_)) {
        "tool_calls"
    } else {
        "stop"
    }
}

pub(crate) fn chat_tool_output_chunk(
    request_id: String,
    model_id: String,
    created: u64,
    output: &ParsedChatOutput,
    completion_tokens: usize,
    max_tokens: usize,
) -> Event {
    json_event(chat_tool_output_payload(
        request_id,
        model_id,
        created,
        output,
        completion_tokens,
        max_tokens,
    ))
}

pub(crate) fn chat_tool_output_payload(
    request_id: String,
    model_id: String,
    created: u64,
    output: &ParsedChatOutput,
    completion_tokens: usize,
    max_tokens: usize,
) -> serde_json::Value {
    let finish_reason = chat_output_finish_reason(output, completion_tokens, max_tokens);
    let delta = match output {
        ParsedChatOutput::Message(content) => json!({"content": content}),
        ParsedChatOutput::ToolCalls(calls) => {
            json!({"tool_calls": tool_calls_json(&request_id, calls, true)})
        }
    };
    json!({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
}

pub(crate) fn validate_chat_message_extensions(
    message: &ChatCompletionMessage,
    index: usize,
) -> std::result::Result<(), String> {
    for (field, value) in &message.extensions {
        if value.is_null() {
            continue;
        }
        if field == "tool_calls" && value.as_array().is_some_and(Vec::is_empty) {
            continue;
        }
        let supported = match message.role.as_str() {
            "assistant" => false,
            "tool" => matches!(field.as_str(), "tool_call_id" | "name"),
            _ => false,
        };
        if field == "tool_calls" && message.role == "assistant" {
            parse_historical_tool_calls(value, index)?;
            continue;
        }
        if !supported {
            return Err(format!(
                "chat completion message at index {index} contains unsupported non-neutral field {:?}. Bloom rejects unsupported message semantics instead of silently ignoring them.",
                reported_extension_field(field)
            ));
        }
    }
    if message.role == "tool" {
        tool_result_identity(message, index)?;
    }
    Ok(())
}

pub(crate) fn parse_historical_tool_calls(
    value: &serde_json::Value,
    message_index: usize,
) -> std::result::Result<Vec<HistoricalToolCall>, String> {
    let calls = value.as_array().ok_or_else(|| {
        format!("assistant tool_calls at message index {message_index} must be an array.")
    })?;
    if calls.len() > MAX_PARALLEL_TOOL_CALLS {
        return Err(format!(
            "assistant tool_calls at message index {message_index} cannot contain more than {MAX_PARALLEL_TOOL_CALLS} calls."
        ));
    }
    let mut parsed = Vec::with_capacity(calls.len());
    let mut ids = std::collections::HashSet::with_capacity(calls.len());
    for (call_index, call) in calls.iter().enumerate() {
        let call = call.as_object().ok_or_else(|| {
            format!("assistant tool call {call_index} at message index {message_index} must be an object.")
        })?;
        reject_non_null_fields(
            call,
            &["id", "type", "function"],
            &format!("assistant tool call {call_index} at message index {message_index}"),
        )?;
        let id = call
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!("assistant tool call {call_index} requires string field `id`.")
            })?;
        validate_tool_call_id(id)?;
        if !ids.insert(id.to_string()) {
            return Err(format!("assistant tool_calls contain duplicate ID {id:?}."));
        }
        if call.get("type").and_then(serde_json::Value::as_str) != Some("function") {
            return Err(format!(
                "assistant tool call {call_index} must have type `function`."
            ));
        }
        let function = call
            .get("function")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                format!("assistant tool call {call_index} requires a function object.")
            })?;
        reject_non_null_fields(
            function,
            &["name", "arguments"],
            &format!("assistant tool call {call_index} function"),
        )?;
        let name = function
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("assistant tool call {call_index} requires a function name."))?;
        validate_tool_name(name, "assistant tool call function name")?;
        let arguments = function
            .get("arguments")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!("assistant tool call {call_index} arguments must be a JSON string.")
            })?;
        let arguments = serde_json::from_str::<serde_json::Value>(arguments).map_err(|error| {
            format!("assistant tool call {call_index} arguments are not valid JSON: {error}")
        })?;
        if !arguments.is_object() {
            return Err(format!(
                "assistant tool call {call_index} arguments must encode a JSON object."
            ));
        }
        parsed.push(HistoricalToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        });
    }
    Ok(parsed)
}

pub(crate) fn tool_result_identity(
    message: &ChatCompletionMessage,
    message_index: usize,
) -> std::result::Result<(String, Option<String>), String> {
    let call_id = message
        .extensions
        .get("tool_call_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            format!("tool message at index {message_index} requires string field `tool_call_id`.")
        })?;
    validate_tool_call_id(call_id)?;
    let name = message
        .extensions
        .get("name")
        .filter(|value| !value.is_null())
        .map(|value| {
            let name = value.as_str().ok_or_else(|| {
                format!("tool message at index {message_index} name must be a string.")
            })?;
            validate_tool_name(name, "tool message name")?;
            Ok::<String, String>(name.to_string())
        })
        .transpose()?;
    Ok((call_id.to_string(), name))
}

pub(crate) fn tool_calls_json(
    request_id: &str,
    calls: &[ParsedToolCall],
    include_index: bool,
) -> serde_json::Value {
    let suffix = request_id
        .strip_prefix("chatcmpl-")
        .unwrap_or(request_id)
        .chars()
        .take(96)
        .collect::<String>();
    serde_json::Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let mut value = json!({
                    "id": format!("call_{suffix}_{index}"),
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments
                    }
                });
                if include_index && let Some(object) = value.as_object_mut() {
                    object.insert("index".to_string(), json!(index));
                }
                value
            })
            .collect(),
    )
}

pub(crate) fn validate_tool_name(name: &str, label: &str) -> std::result::Result<(), String> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(format!(
            "{label} must contain 1 to 64 ASCII letters, digits, underscores, or hyphens."
        ));
    }
    Ok(())
}

pub(crate) fn validate_tool_call_id(id: &str) -> std::result::Result<(), String> {
    if id.is_empty()
        || id.chars().count() > MAX_TOOL_CALL_ID_CHARS
        || id.chars().any(char::is_control)
    {
        return Err(format!(
            "tool call IDs must contain 1 to {MAX_TOOL_CALL_ID_CHARS} characters without control characters."
        ));
    }
    Ok(())
}

fn reject_non_null_fields(
    object: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
    scope: &str,
) -> std::result::Result<(), String> {
    if let Some(field) = object
        .iter()
        .find(|(field, value)| !value.is_null() && !allowed.contains(&field.as_str()))
        .map(|(field, _)| field)
    {
        return Err(format!(
            "{scope} contains unsupported field {:?}.",
            reported_extension_field(field)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> ChatRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn tool_config_supports_all_function_choice_modes() {
        let base = json!({
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }]
        });
        let config = chat_tool_config(&request(base.clone())).unwrap().unwrap();
        assert_eq!(config.choice, ToolChoiceMode::Auto);
        assert!(config.parallel);

        for (choice, expected) in [
            (json!("required"), ToolChoiceMode::Required),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                ToolChoiceMode::Named("get_weather".to_string()),
            ),
        ] {
            let mut value = base.clone();
            value["tool_choice"] = choice;
            assert_eq!(
                chat_tool_config(&request(value)).unwrap().unwrap().choice,
                expected
            );
        }

        let mut none = base;
        none["tool_choice"] = json!("none");
        assert!(chat_tool_config(&request(none)).unwrap().is_none());
    }

    #[test]
    fn tool_config_rejects_undefined_and_unsupported_tools() {
        for value in [
            json!({"messages": [{"role": "user", "content": "x"}], "tool_choice": "auto"}),
            json!({
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "custom", "custom": {"name": "shell"}}]
            }),
            json!({
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {"name": "bad name"}}]
            }),
            json!({
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {"name": "known"}}],
                "tool_choice": {"type": "function", "function": {"name": "missing"}}
            }),
            json!({
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {
                    "name": "strict_but_open",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"]
                    }
                }}]
            }),
            json!({
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {
                    "name": "strict_but_optional",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": [],
                        "additionalProperties": false
                    }
                }}]
            }),
        ] {
            assert!(chat_tool_config(&request(value)).is_err());
        }

        let oversized_field = "x".repeat(1_000);
        let mut value = json!({
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        });
        value["tools"][0]["function"][&oversized_field] = json!(true);
        let error = chat_tool_config(&request(value)).unwrap_err();
        assert!(error.len() < 1_024);
        assert!(!error.contains(&oversized_field));
        assert!(error.contains("<invalid-field-name>"));
    }

    #[test]
    fn tool_output_is_parsed_and_arguments_are_validated() {
        let config = chat_tool_config(&request(json!({
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            }],
            "parallel_tool_calls": false
        })))
        .unwrap()
        .unwrap();

        assert_eq!(
            parse_tool_output(
                r#"{"type":"function_calls","calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}"#,
                &config
            )
            .unwrap(),
            ParsedChatOutput::ToolCalls(vec![ParsedToolCall {
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Paris"}"#.to_string()
            }])
        );
        assert!(
            parse_tool_output(
                r#"{"type":"function_calls","calls":[{"name":"get_weather","arguments":{}}]}"#,
                &config
            )
            .is_err()
        );
        assert_eq!(
            parse_tool_output(r#"{"type":"message","content":"No call needed."}"#, &config)
                .unwrap(),
            ParsedChatOutput::Message("No call needed.".to_string())
        );
    }

    #[test]
    fn historical_tool_calls_and_results_are_bounded_conversation_context() {
        let valid_request = request(json!({
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather_1",
                    "name": "get_weather",
                    "content": "15 C"
                },
                {"role": "user", "content": "Summarize it."}
            ]
        }));
        validate_chat_request_compatibility(&valid_request).unwrap();
        let normalized = normalize_chat_messages(&valid_request.messages).unwrap();
        assert_eq!(
            normalized
                .iter()
                .map(|message| message.role.as_str())
                .collect::<Vec<_>>(),
            vec!["user", "assistant", "user", "user"]
        );
        assert!(normalized[1].content.contains("requested_function_calls"));
        assert!(normalized[2].content.contains("untrusted data"));
        assert!(normalized[2].content.contains("call_weather_1"));

        for invalid_messages in [
            json!([
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]},
                {"role": "user", "content": "Skip the result."}
            ]),
            json!([{"role": "tool", "tool_call_id": "missing", "content": "x"}]),
            json!([
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "name": "other", "content": "x"}
            ]),
        ] {
            let invalid = request(json!({"messages": invalid_messages}));
            validate_chat_request_compatibility(&invalid).unwrap();
            assert!(normalize_chat_messages(&invalid.messages).is_err());
        }
    }

    #[test]
    fn tool_outputs_map_to_openai_nonstreaming_and_streaming_shapes() {
        let output = ParsedChatOutput::ToolCalls(vec![ParsedToolCall {
            name: "get_weather".to_string(),
            arguments: r#"{"city":"Paris"}"#.to_string(),
        }]);
        let message = chat_output_message("chatcmpl-example", &output);
        assert!(message["content"].is_null());
        assert_eq!(message["tool_calls"][0]["id"], "call_example_0");
        assert_eq!(message["tool_calls"][0]["type"], "function");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Paris"}"#
        );
        assert_eq!(chat_output_finish_reason(&output, 2, 8), "tool_calls");
        assert_eq!(chat_output_finish_reason(&output, 8, 8), "length");

        let stream = chat_tool_output_payload(
            "chatcmpl-example".to_string(),
            "tiny.gguf".to_string(),
            12,
            &output,
            2,
            8,
        );
        assert_eq!(stream["object"], "chat.completion.chunk");
        assert_eq!(stream["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(stream["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(
            stream["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_example_0"
        );
    }
}
