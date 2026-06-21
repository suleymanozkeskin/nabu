//! Claude native-format canonicalization.

use super::*;

pub(crate) fn canonical_type_for_claude_native(payload: &Value) -> CanonicalType {
    match string_pointer(payload, "/type").as_deref() {
        Some("user") => {
            if claude_message_has_content_type(payload, "tool_result") {
                CanonicalType::ToolResult
            } else {
                CanonicalType::UserMessage
            }
        }
        Some("assistant") => {
            if claude_message_has_content_type(payload, "tool_use")
                && !claude_message_has_text(payload)
            {
                CanonicalType::ToolCall
            } else {
                CanonicalType::AssistantMessage
            }
        }
        Some("summary") => CanonicalType::CompactionAfter,
        Some("attachment") => match string_pointer(payload, "/attachment/hookEvent").as_deref() {
            Some("PreToolUse") => CanonicalType::ToolCall,
            Some("PostToolUse") | Some("PostToolUseFailure") | Some("PostToolBatch") => {
                CanonicalType::ToolResult
            }
            _ => CanonicalType::Error,
        },
        Some("queue-operation") => CanonicalType::SessionResumed,
        Some("system") => CanonicalType::SessionStarted,
        _ => CanonicalType::Error,
    }
}

fn claude_message_has_content_type(payload: &Value, content_type: &str) -> bool {
    payload
        .pointer("/message/content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some(content_type))
        })
        .unwrap_or(false)
}

fn claude_message_has_text(payload: &Value) -> bool {
    payload
        .pointer("/message/content")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter().any(|item| {
                item.get("type").and_then(Value::as_str) == Some("text")
                    && item
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.trim().is_empty())
            })
        })
        .unwrap_or(false)
}
