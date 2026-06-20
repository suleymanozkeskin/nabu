//! Event canonicalization and search/identity document extraction: maps raw
//! payloads to canonical event types, builds the FTS `SearchDocument`, derives
//! embedding units, and extracts file paths.

use crate::{
    sha256_hex, string_pointer, CanonicalType, EmbeddingUnit, EmbeddingUnitKind, Error, Result,
    Tool,
};
use serde_json::Value;
use std::collections::BTreeSet;

pub(crate) fn file_paths_for_payload(payload: &Value) -> Vec<String> {
    let mut paths = BTreeSet::new();
    collect_file_paths(payload, None, &mut paths);
    paths.into_iter().collect()
}

fn collect_file_paths(value: &Value, key: Option<&str>, output: &mut BTreeSet<String>) {
    match value {
        Value::String(text) => {
            if key.is_some_and(is_file_path_key) || looks_like_file_path(text) {
                let text = text.trim();
                if !text.is_empty() {
                    output.insert(text.to_string());
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_file_paths(value, key, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                collect_file_paths(value, Some(key), output);
            }
        }
        _ => {}
    }
}

fn is_file_path_key(key: &str) -> bool {
    matches!(
        key,
        "file" | "file_path" | "filepath" | "path" | "source_path" | "transcript_path"
    ) || key.ends_with("_file")
        || key.ends_with("_path")
}

fn looks_like_file_path(value: &str) -> bool {
    let value = value.trim();
    value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("~/")
}

pub(crate) fn hook_event_name(payload: &Value) -> Result<&str> {
    payload
        .get("hook_event_name")
        .or_else(|| payload.get("event"))
        .or_else(|| payload.get("type"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            Error::Validation(
                "payload.hook_event_name, payload.event, or payload.type is required".to_string(),
            )
        })
}

pub(crate) fn canonical_type_for_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
) -> CanonicalType {
    if source_event_type == "MessageDisplay" {
        return if payload
            .get("final")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            CanonicalType::AssistantMessage
        } else {
            CanonicalType::AssistantDelta
        };
    }

    if tool == Tool::Codex {
        match source_event_type {
            "session_meta" => return CanonicalType::SessionStarted,
            "turn_context" => return CanonicalType::SessionResumed,
            "response_item" => {
                return match string_pointer(payload, "/payload/type").as_deref() {
                    Some("message") => match string_pointer(payload, "/payload/role").as_deref() {
                        Some("user") => CanonicalType::UserMessage,
                        Some("assistant") => CanonicalType::AssistantMessage,
                        _ => CanonicalType::Error,
                    },
                    Some("function_call") | Some("custom_tool_call") => CanonicalType::ToolCall,
                    Some("function_call_output") | Some("custom_tool_call_output") => {
                        CanonicalType::ToolResult
                    }
                    Some("reasoning") => CanonicalType::AssistantDelta,
                    _ => CanonicalType::Error,
                };
            }
            "event_msg" => {
                return match string_pointer(payload, "/payload/type").as_deref() {
                    Some("user_message") => CanonicalType::UserMessage,
                    Some("agent_message") => CanonicalType::AssistantMessage,
                    Some("agent_reasoning") => CanonicalType::AssistantDelta,
                    Some("exec_command_begin") | Some("tool_call") => CanonicalType::ToolCall,
                    Some("exec_command_end") | Some("tool_output") => CanonicalType::ToolResult,
                    _ => CanonicalType::Error,
                };
            }
            _ => {}
        }
    }

    match (tool, source_event_type) {
        (_, "SessionStart") | (Tool::Opencode, "session.created") => CanonicalType::SessionStarted,
        (Tool::Codex, "thread.started")
        | (Tool::Codex, "thread/started")
        | (Tool::Codex, "turn.started")
        | (Tool::Codex, "turn/started") => CanonicalType::SessionStarted,
        (_, "SessionEnd") | (_, "Stop") | (Tool::Opencode, "session.idle") => {
            CanonicalType::SessionEnded
        }
        (Tool::Codex, "turn.completed") | (Tool::Codex, "turn/completed") => {
            CanonicalType::SessionEnded
        }
        (_, "UserPromptSubmit") => CanonicalType::UserMessage,
        (Tool::Codex, "item/agentMessage/delta") => CanonicalType::AssistantDelta,
        (Tool::Codex, "item.completed") | (Tool::Codex, "item/completed") => {
            canonical_type_for_codex_item(payload).unwrap_or(CanonicalType::AssistantMessage)
        }
        (Tool::Opencode, "message.part.updated") => CanonicalType::AssistantDelta,
        (Tool::Opencode, "message.part.removed") => CanonicalType::AssistantDelta,
        (Tool::Opencode, "message.updated") => CanonicalType::AssistantMessage,
        (Tool::Opencode, "message.removed") => CanonicalType::AssistantMessage,
        (_, "PreToolUse")
        | (Tool::Codex, "SubagentStart")
        | (Tool::Codex, "item.started")
        | (Tool::Codex, "item/started")
        | (Tool::Opencode, "tool.execute.before") => CanonicalType::ToolCall,
        (_, "PostToolUse")
        | (_, "PostToolUseFailure")
        | (_, "PostToolBatch")
        | (Tool::Codex, "SubagentStop")
        | (Tool::Codex, "item.failed")
        | (Tool::Codex, "item/failed")
        | (Tool::Opencode, "tool.execute.after")
        | (Tool::Opencode, "command.executed") => CanonicalType::ToolResult,
        (_, "PreCompact") => CanonicalType::CompactionBefore,
        (_, "PostCompact") | (Tool::Opencode, "session.compacted") => {
            CanonicalType::CompactionAfter
        }
        (Tool::Opencode, "session.updated") => CanonicalType::SessionResumed,
        (Tool::Opencode, "file.edited") => CanonicalType::FileChanged,
        (Tool::Opencode, "session.error") => CanonicalType::Error,
        _ => CanonicalType::Error,
    }
}

fn canonical_type_for_codex_item(payload: &Value) -> Option<CanonicalType> {
    let item = payload
        .pointer("/item")
        .or_else(|| payload.pointer("/payload/item"))
        .or_else(|| payload.pointer("/params/item"))
        .or_else(|| payload.pointer("/payload"))
        .or_else(|| payload.pointer("/params"))
        .unwrap_or(payload);

    match string_pointer(item, "/role").as_deref() {
        Some("user") => return Some(CanonicalType::UserMessage),
        Some("assistant") => return Some(CanonicalType::AssistantMessage),
        _ => {}
    }

    match string_pointer(item, "/type").as_deref() {
        Some("message") | Some("agent_message") | Some("agentMessage") => {
            Some(CanonicalType::AssistantMessage)
        }
        Some("user_message") | Some("userMessage") => Some(CanonicalType::UserMessage),
        Some("function_call")
        | Some("custom_tool_call")
        | Some("tool_call")
        | Some("tool")
        | Some("exec_command_begin") => Some(CanonicalType::ToolCall),
        Some("function_call_output")
        | Some("custom_tool_call_output")
        | Some("tool_result")
        | Some("tool-output")
        | Some("exec_command_end") => Some(CanonicalType::ToolResult),
        Some("reasoning") | Some("agent_reasoning") => Some(CanonicalType::AssistantDelta),
        Some("error") => Some(CanonicalType::Error),
        _ => None,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SearchDocument {
    pub(crate) user_text: String,
    pub(crate) assistant_text: String,
    pub(crate) tool_intent: String,
    pub(crate) tool_output: String,
    pub(crate) metadata_text: String,
}

impl SearchDocument {
    pub(crate) fn render(&self) -> String {
        join_non_empty([
            self.user_text.as_str(),
            self.assistant_text.as_str(),
            self.tool_intent.as_str(),
            self.tool_output.as_str(),
            self.metadata_text.as_str(),
        ])
    }

    pub(crate) fn identity_text(&self) -> String {
        self.render()
    }
}

#[allow(dead_code)]
pub(crate) fn embedding_units_for_document(document: &SearchDocument) -> Vec<EmbeddingUnit> {
    let candidates = [
        (EmbeddingUnitKind::UserText, document.user_text.as_str()),
        (
            EmbeddingUnitKind::AssistantText,
            document.assistant_text.as_str(),
        ),
        (EmbeddingUnitKind::ToolIntent, document.tool_intent.as_str()),
        (
            EmbeddingUnitKind::MetadataText,
            document.metadata_text.as_str(),
        ),
    ];
    let mut units = Vec::new();
    for (kind, text) in candidates {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        units.push(EmbeddingUnit {
            kind,
            unit_index: 0,
            text: text.to_string(),
            text_hash: sha256_hex(text.as_bytes()),
        });
    }
    units
}

pub(crate) fn search_document_for_event(
    canonical_type: CanonicalType,
    payload: &Value,
) -> SearchDocument {
    let mut document = SearchDocument::default();
    match canonical_type {
        CanonicalType::UserMessage => {
            document.user_text =
                preferred_text(payload, &["prompt", "text", "message", "content", "input"]);
        }
        CanonicalType::AssistantDelta | CanonicalType::AssistantMessage => {
            document.assistant_text =
                preferred_text(payload, &["text", "message", "content", "delta", "summary"]);
        }
        CanonicalType::ToolCall => {
            document.tool_intent = preferred_text(
                payload,
                &["tool_name", "command", "description", "input", "arguments"],
            );
        }
        CanonicalType::ToolResult => {
            document.tool_intent =
                preferred_text(payload, &["tool_name", "command", "status", "exit_code"]);
            document.tool_output =
                preferred_text(payload, &["output", "stderr", "stdout", "error", "result"]);
        }
        CanonicalType::FileChanged => {
            document.metadata_text =
                preferred_text(payload, &["file", "file_path", "path", "diff", "operation"]);
        }
        CanonicalType::CompactionBefore | CanonicalType::CompactionAfter => {
            document.assistant_text =
                preferred_text(payload, &["summary", "text", "content", "reason"]);
            document.metadata_text = preferred_text(payload, &["trigger"]);
        }
        CanonicalType::SessionEnded => {
            let text = preferred_text(payload, &["message", "reason", "summary", "usage_summary"]);
            let usage = scalar_text_for_keys(payload, &["usage", "usage_metadata"]);
            document.metadata_text = join_non_empty([text.as_str(), usage.as_str()]);
        }
        CanonicalType::Error => {
            document.metadata_text =
                preferred_text(payload, &["error", "message", "reason", "text", "details"]);
        }
        _ => {
            document.metadata_text = nonvolatile_text(payload);
        }
    }

    if document.render().trim().is_empty() {
        document.metadata_text = nonvolatile_text(payload);
    }
    document
}

pub(crate) fn message_text_for_document(
    canonical_type: CanonicalType,
    document: &SearchDocument,
) -> &str {
    match canonical_type {
        CanonicalType::UserMessage => &document.user_text,
        CanonicalType::AssistantDelta | CanonicalType::AssistantMessage => &document.assistant_text,
        CanonicalType::ToolResult => {
            if document.tool_output.trim().is_empty() {
                &document.tool_intent
            } else {
                &document.tool_output
            }
        }
        _ => "",
    }
}

pub(crate) fn normalize_identity_text(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn preferred_text(payload: &Value, keys: &[&str]) -> String {
    let mut values = Vec::new();
    collect_strings_for_keys(payload, keys, &mut values);
    join_owned(values)
}

fn scalar_text_for_keys(payload: &Value, keys: &[&str]) -> String {
    let mut values = Vec::new();
    collect_scalars_for_keys(payload, keys, &mut values);
    join_owned(values)
}

fn nonvolatile_text(payload: &Value) -> String {
    let mut values = Vec::new();
    collect_nonvolatile_strings(payload, &mut values);
    join_owned(values)
}

fn collect_strings_for_keys(value: &Value, keys: &[&str], output: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if keys.contains(&key.as_str()) {
                    collect_strings(value, output);
                } else {
                    collect_strings_for_keys(value, keys, output);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_strings_for_keys(value, keys, output);
            }
        }
        _ => {}
    }
}

fn collect_scalars_for_keys(value: &Value, keys: &[&str], output: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if keys.contains(&key.as_str()) {
                    collect_scalar_key_values(value, key, output);
                } else {
                    collect_scalars_for_keys(value, keys, output);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_scalars_for_keys(value, keys, output);
            }
        }
        _ => {}
    }
}

fn collect_scalar_key_values(value: &Value, key_hint: &str, output: &mut Vec<String>) {
    match value {
        Value::String(text) => push_clean(output, &format!("{key_hint} {text}")),
        Value::Number(number) => push_clean(output, &format!("{key_hint} {number}")),
        Value::Bool(flag) => push_clean(output, &format!("{key_hint} {flag}")),
        Value::Array(values) => {
            for value in values {
                collect_scalar_key_values(value, key_hint, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                collect_scalar_key_values(value, key, output);
            }
        }
        Value::Null => {}
    }
}

fn collect_nonvolatile_strings(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => push_clean(output, text),
        Value::Array(values) => {
            for value in values {
                collect_nonvolatile_strings(value, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                if !is_volatile_identity_key(key) {
                    collect_nonvolatile_strings(value, output);
                }
            }
        }
        _ => {}
    }
}

fn collect_strings(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => push_clean(output, text),
        Value::Array(values) => {
            for value in values {
                collect_strings(value, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                if key == "type" {
                    continue;
                }
                collect_strings(value, output);
            }
        }
        _ => {}
    }
}

fn push_clean(output: &mut Vec<String>, text: &str) {
    let text = text.trim();
    if !text.is_empty() {
        output.push(text.to_string());
    }
}

fn join_owned(values: Vec<String>) -> String {
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            output.push(value);
        }
    }
    output.join("\n")
}

fn join_non_empty<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    values
        .into_iter()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn identity_payload(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| !is_volatile_identity_key(key))
                .map(|(key, value)| (key.clone(), identity_payload(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(identity_payload).collect()),
        _ => value.clone(),
    }
}

fn is_volatile_identity_key(key: &str) -> bool {
    matches!(
        key,
        "captured_at"
            | "timestamp"
            | "created_at"
            | "updated_at"
            | "tool_version"
            | "session_id"
            | "filename_session_id"
            | "project_root"
            | "cwd"
            | "source"
            | "source_event_type"
            | "source_event_id"
            | "hook_event_name"
            | "event"
            | "type"
            | "dedupe_key"
            | "raw_file"
            | "raw_offset"
            | "raw_line"
            | "message_id"
            | "event_id"
            | "turn_id"
            | "id"
    )
}

pub(crate) fn role_for(canonical_type: CanonicalType) -> Option<&'static str> {
    match canonical_type {
        CanonicalType::UserMessage => Some("user"),
        CanonicalType::AssistantDelta | CanonicalType::AssistantMessage => Some("assistant"),
        CanonicalType::ToolResult => Some("tool"),
        _ => None,
    }
}

pub(crate) fn tool_status_for(canonical_type: CanonicalType) -> Option<&'static str> {
    match canonical_type {
        CanonicalType::ToolCall => Some("started"),
        CanonicalType::ToolResult => Some("completed"),
        _ => None,
    }
}

pub(crate) fn compaction_state_for(canonical_type: CanonicalType) -> &'static str {
    match canonical_type {
        CanonicalType::CompactionBefore => "pre_compaction",
        CanonicalType::CompactionAfter => "post_compaction",
        _ => "none",
    }
}

pub(crate) fn string_field<'a>(payload: &'a Value, key: &str) -> Option<&'a str> {
    payload.get(key).and_then(Value::as_str)
}
