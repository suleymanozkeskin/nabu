//! OpenCode format parsing and session-id resolution.

use super::*;

pub(crate) fn opencode_message_session_ids(tool_root: &Path) -> Result<BTreeSet<String>> {
    let message_root = tool_root.join("storage").join("message");
    let mut session_ids = BTreeSet::new();
    if !message_root.exists() {
        return Ok(session_ids);
    }
    for entry in fs::read_dir(&message_root).map_err(|source| Error::Io {
        path: message_root.clone(),
        source,
    })? {
        let entry = entry.map_err(|source| Error::Io {
            path: message_root.clone(),
            source,
        })?;
        if !entry.path().is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str().filter(|value| !value.is_empty()) {
            session_ids.insert(name.to_string());
        }
    }
    Ok(session_ids)
}

pub(crate) fn opencode_metadata_session_id(
    payload: &Value,
    message_session_ids: &BTreeSet<String>,
) -> Option<String> {
    opencode_direct_session_id(payload)
        .filter(|session_id| {
            message_session_ids.is_empty() || message_session_ids.contains(session_id)
        })
        .or_else(|| {
            let id = string_pointer(payload, "/id")?;
            message_session_ids.contains(&id).then_some(id)
        })
        .or_else(|| {
            if message_session_ids.len() == 1 {
                message_session_ids.iter().next().cloned()
            } else {
                None
            }
        })
}

/// Resolve the session id from a live OpenCode plugin event.
///
/// OpenCode events have no top-level `session_id` (the field the generic hook
/// path requires), so every event the plugin forwards would otherwise fail
/// validation and the ingest subprocess would exit non-zero. Message, part,
/// tool, text, patch, step, and file events carry `sessionID`; the payload may
/// be the bare object or wrapped under `info`/`part`/`properties`. `session.*`
/// events have no `sessionID` and instead identify the session by the session
/// object's own `id` — so that fallback is gated on the event name to avoid
/// mistaking a message id for a session id.
pub(crate) fn opencode_hook_session_id(payload: &Value, event_name: &str) -> Result<String> {
    const SESSION_ID_POINTERS: [&str; 9] = [
        "/sessionID",
        "/session_id",
        "/sessionId",
        "/info/sessionID",
        "/part/sessionID",
        "/properties/info/sessionID",
        "/properties/part/sessionID",
        "/payload/sessionID",
        "/payload/session_id",
    ];
    const SESSION_OBJECT_ID_POINTERS: [&str; 3] = ["/id", "/info/id", "/properties/info/id"];

    SESSION_ID_POINTERS
        .into_iter()
        .find_map(|pointer| string_pointer(payload, pointer))
        .or_else(|| {
            // Only `session.*` events name the session by its own `id`.
            event_name
                .starts_with("session.")
                .then(|| {
                    SESSION_OBJECT_ID_POINTERS
                        .into_iter()
                        .find_map(|pointer| string_pointer(payload, pointer))
                })
                .flatten()
        })
        .ok_or_else(|| {
            Error::Validation(format!(
                "opencode event '{event_name}' has no resolvable session id \
                 (looked for sessionID, and id for session.* events)"
            ))
        })
}

pub(crate) fn opencode_direct_session_id(payload: &Value) -> Option<String> {
    string_pointer(payload, "/session_id")
        .or_else(|| string_pointer(payload, "/sessionID"))
        .or_else(|| string_pointer(payload, "/sessionId"))
        .or_else(|| string_pointer(payload, "/message/sessionID"))
        .or_else(|| string_pointer(payload, "/payload/session_id"))
        .or_else(|| string_pointer(payload, "/payload/sessionID"))
        .or_else(|| string_pointer(payload, "/payload/sessionId"))
        .or_else(|| string_pointer(payload, "/session/id"))
}

pub(crate) fn opencode_storage_kind(path: &Path) -> Option<&'static str> {
    let mut saw_storage = false;
    for component in path.components() {
        let value = component.as_os_str();
        if saw_storage {
            if value == "message" {
                return Some("message");
            }
            if value == "part" {
                return Some("part");
            }
            if value == "session" {
                return Some("session");
            }
            saw_storage = false;
        }
        if value == "storage" {
            saw_storage = true;
        }
    }
    None
}

pub(crate) fn opencode_worktree_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/worktree")
        .or_else(|| string_pointer(payload, "/payload/worktree"))
        .or_else(|| string_pointer(payload, "/params/worktree"))
        .or_else(|| string_pointer(payload, "/project/worktree"))
        .or_else(|| string_pointer(payload, "/session/worktree"))
}

pub(crate) fn opencode_server_events_from_payload(
    fallback_session_id: &str,
    payload: Value,
) -> Result<Vec<EventEnvelope>> {
    let messages = opencode_server_messages(payload);
    let mut events = Vec::new();
    for (message_index, message) in messages.into_iter().enumerate() {
        let session_id = opencode_message_session_id(fallback_session_id, &message);
        let parts = opencode_message_parts(&message);
        if parts.is_empty() || opencode_message_has_top_level_text(&message) {
            events.push(opencode_server_message_envelope(
                &session_id,
                "message.updated",
                Some(message_index as i64),
                message.clone(),
            )?);
        }
        for (part_index, part) in parts.into_iter().enumerate() {
            let message_id = message_id_for_payload(&message);
            let mut payload = serde_json::json!({
                "session_id": session_id.clone(),
                "message_id": message_id.clone(),
                "part": part,
                "server_message_id": message_id
            });
            if let Value::Object(map) = &mut payload {
                if let Some(project_root) = project_root_for_payload(&message) {
                    map.insert("project_root".to_string(), Value::String(project_root));
                }
                if let Some(cwd) = cwd_for_payload(&message) {
                    map.insert("cwd".to_string(), Value::String(cwd));
                }
                if let Some(worktree) = opencode_worktree_for_payload(&message) {
                    map.insert("worktree".to_string(), Value::String(worktree));
                }
            }
            events.push(opencode_server_message_envelope(
                &session_id,
                "message.part.updated",
                Some(part_index as i64),
                payload,
            )?);
        }
    }
    Ok(events)
}

pub(crate) fn opencode_server_messages(payload: Value) -> Vec<Value> {
    match payload {
        Value::Array(messages) => messages,
        Value::Object(mut map) => map
            .remove("messages")
            .or_else(|| map.remove("data"))
            .or_else(|| map.remove("result"))
            .and_then(|value| match value {
                Value::Array(values) => Some(values),
                Value::Object(mut object) => object.remove("messages").and_then(|nested| {
                    if let Value::Array(values) = nested {
                        Some(values)
                    } else {
                        None
                    }
                }),
                other => Some(vec![other]),
            })
            .unwrap_or_else(|| vec![Value::Object(map)]),
        other => vec![other],
    }
}

pub(crate) fn opencode_message_parts(message: &Value) -> Vec<Value> {
    for pointer in ["/parts", "/message/parts", "/payload/parts"] {
        if let Some(parts) = message.pointer(pointer).and_then(Value::as_array) {
            return parts.clone();
        }
    }
    Vec::new()
}

pub(crate) fn opencode_message_has_top_level_text(message: &Value) -> bool {
    ["text", "content", "message", "summary"].iter().any(|key| {
        message
            .get(*key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    })
}

pub(crate) fn opencode_message_session_id(fallback_session_id: &str, message: &Value) -> String {
    string_pointer(message, "/session_id")
        .or_else(|| string_pointer(message, "/sessionID"))
        .or_else(|| string_pointer(message, "/sessionId"))
        .or_else(|| string_pointer(message, "/message/sessionID"))
        .or_else(|| string_pointer(message, "/payload/session_id"))
        .unwrap_or_else(|| fallback_session_id.to_string())
}

pub(crate) fn opencode_server_message_envelope(
    session_id: &str,
    source_event_type: &str,
    sequence: Option<i64>,
    payload: Value,
) -> Result<EventEnvelope> {
    let canonical_type = canonical_type_for_opencode_native(source_event_type, &payload);
    let source_event_id =
        source_event_id_for_payload(Tool::Opencode, source_event_type, &payload, sequence);
    Ok(EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: timestamp_for_payload(&payload)
            .unwrap_or(OffsetDateTime::now_utc().format(&Rfc3339)?),
        tool: Tool::Opencode,
        tool_version: tool_version_for_payload(&payload),
        session_id: session_id.to_string(),
        filename_session_id: sanitize_session_id(session_id),
        turn_id: turn_id_for_payload(&payload),
        message_id: message_id_for_payload(&payload),
        project_root: project_root_for_payload(&payload),
        cwd: cwd_for_payload(&payload),
        source: Source::Backfill,
        source_event_type: source_event_type.to_string(),
        canonical_type,
        source_event_id,
        dedupe_key: String::new(),
        sequence,
        raw_file: None,
        raw_offset: None,
        payload,
        payload_ref: None,
    })
}

pub(crate) fn canonical_type_for_opencode_native(
    source_event_type: &str,
    payload: &Value,
) -> CanonicalType {
    match source_event_type {
        "tool.execute.before" => return CanonicalType::ToolCall,
        "tool.execute.after" => return CanonicalType::ToolResult,
        "session.created" | "session.updated" => return CanonicalType::SessionStarted,
        "reasoning" | "step-start" | "step-finish" => return CanonicalType::AssistantDelta,
        "patch" => return CanonicalType::FileChanged,
        _ => {}
    }
    match opencode_part_type(payload).as_deref() {
        Some("reasoning") | Some("step-start") | Some("step-finish") => {
            return CanonicalType::AssistantDelta;
        }
        Some("patch") => return CanonicalType::FileChanged,
        Some("text") => return CanonicalType::AssistantMessage,
        Some("tool") | Some("tool-call") => return CanonicalType::ToolCall,
        Some("tool-result") | Some("tool_result") => return CanonicalType::ToolResult,
        Some("file") => return CanonicalType::FileChanged,
        _ => {}
    }
    match string_pointer(payload, "/role").as_deref() {
        Some("user") => CanonicalType::UserMessage,
        Some("assistant") => CanonicalType::AssistantMessage,
        _ => CanonicalType::Error,
    }
}

pub(crate) fn opencode_part_type(payload: &Value) -> Option<String> {
    string_pointer(payload, "/part/type")
        .or_else(|| string_pointer(payload, "/payload/part/type"))
        .or_else(|| string_pointer(payload, "/type"))
        .or_else(|| string_pointer(payload, "/payload/type"))
}
