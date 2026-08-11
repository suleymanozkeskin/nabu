//! Shared pi event expansion: pure functions that turn pi messages and
//! session-level events into canonical envelopes. Used by both the session-file
//! backfill parser (`backfill/pi.rs`) and the live hook ingest
//! (`ingest_hook_events` for `Tool::Pi`), so both paths produce identical
//! source event ids and payload shapes for the same logical events.
//!
//! Source event ids are the dedupe contract: when live capture resolves an
//! entry id (or a toolCall id / toolCallId), it emits exactly the id the
//! backfill parser would, so re-backfill after live capture appends nothing.
//! Without an entry id the live path falls back to `live-*` synthetic ids
//! (accepted rare-race duplicates).

use crate::{
    sanitize_session_id, CanonicalType, Error, EventEnvelope, Result, Source, Tool, SCHEMA_VERSION,
};
use serde_json::{json, Map, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Expand one pi `message` (an `AgentMessage`) into 1..N envelopes per the
/// locked mapping: user → one `user.message`; assistant → `assistant.message`
/// plus one `tool.call` per toolCall content block; toolResult →
/// `tool.result`; bashExecution → a `tool.call`/`tool.result` pair.
///
/// `original` is the raw object kept under `payload.pi_entry` (the session-file
/// entry on backfill, the hook `message` object on live ingest). `entry_id` is
/// the tree entry id when known — with it, source event ids are identical to
/// backfill's; without it, `live-*` synthetic ids are used.
#[allow(clippy::too_many_arguments)]
pub(crate) fn expand_pi_message(
    session_id: &str,
    cwd: Option<&str>,
    project_root: Option<&str>,
    original: &Value,
    message: &Value,
    entry_id: Option<&str>,
    parent_id: Option<&str>,
    source: Source,
    captured_at: String,
) -> Vec<EventEnvelope> {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match role {
        "user" => vec![message_envelope(
            session_id,
            cwd,
            project_root,
            entry_id,
            parent_id,
            original,
            "message.user",
            CanonicalType::UserMessage,
            entry_id
                .map(str::to_string)
                .or_else(|| live_fallback_id("user", session_id, message)),
            message_id_for(entry_id),
            source,
            captured_at,
            json_map(&[("text", content_to_searchable(message.get("content")))]),
        )],
        "assistant" => {
            let text = assistant_searchable_text(message.get("content"));
            let mut events = vec![message_envelope(
                session_id,
                cwd,
                project_root,
                entry_id,
                parent_id,
                original,
                "message.assistant",
                CanonicalType::AssistantMessage,
                entry_id
                    .map(str::to_string)
                    .or_else(|| live_fallback_id("assistant", session_id, message)),
                message_id_for(entry_id),
                source,
                captured_at.clone(),
                json_map(&[("text", Value::String(text))]),
            )];
            if let Some(blocks) = message.get("content").and_then(Value::as_array) {
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) != Some("toolCall") {
                        continue;
                    }
                    let mut extra = Map::new();
                    extra.insert(
                        "tool_name".to_string(),
                        block.get("name").cloned().unwrap_or(Value::Null),
                    );
                    extra.insert(
                        "tool_call_id".to_string(),
                        block.get("id").cloned().unwrap_or(Value::Null),
                    );
                    extra.insert(
                        "arguments".to_string(),
                        block.get("arguments").cloned().unwrap_or(Value::Null),
                    );
                    if let Some(entry_id) = entry_id {
                        extra.insert(
                            "parent_message_entry_id".to_string(),
                            Value::String(entry_id.to_string()),
                        );
                    }
                    events.push(message_envelope(
                        session_id,
                        cwd,
                        project_root,
                        entry_id,
                        parent_id,
                        original,
                        "message.assistant.toolCall",
                        CanonicalType::ToolCall,
                        block
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .or_else(|| entry_id.map(str::to_string)),
                        None,
                        source,
                        captured_at.clone(),
                        extra,
                    ));
                }
            }
            events
        }
        "toolResult" => {
            let tool_call_id = message
                .get("toolCallId")
                .and_then(Value::as_str)
                .map(str::to_string);
            let is_error = message
                .get("isError")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            vec![message_envelope(
                session_id,
                cwd,
                project_root,
                entry_id,
                parent_id,
                original,
                "message.toolResult",
                CanonicalType::ToolResult,
                tool_call_id
                    .as_deref()
                    .map(str::to_string)
                    .or_else(|| entry_id.map(str::to_string)),
                message_id_for(entry_id),
                source,
                captured_at,
                json_map(&[
                    (
                        "tool_name",
                        message.get("toolName").cloned().unwrap_or(Value::Null),
                    ),
                    (
                        "tool_call_id",
                        tool_call_id
                            .as_deref()
                            .map(str::to_string)
                            .map(Value::String)
                            .unwrap_or(Value::Null),
                    ),
                    ("output", content_to_searchable(message.get("content"))),
                    (
                        "is_error",
                        message
                            .get("isError")
                            .cloned()
                            .unwrap_or(Value::Bool(false)),
                    ),
                    (
                        "status",
                        Value::String(if is_error {
                            "error".to_string()
                        } else {
                            "success".to_string()
                        }),
                    ),
                ]),
            )]
        }
        "bashExecution" => {
            let mut events = Vec::with_capacity(2);
            let command = message
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let call_source_event_id = entry_id.map(|id| format!("bash-call:{id}")).or_else(|| {
                live_fallback_id("bash", session_id, message).map(|id| format!("bash-call:{id}"))
            });
            let result_source_event_id =
                entry_id.map(|id| format!("bash-result:{id}")).or_else(|| {
                    live_fallback_id("bash", session_id, message)
                        .map(|id| format!("bash-result:{id}"))
                });
            let exit_code = message.get("exitCode").cloned().unwrap_or(Value::Null);
            let is_error = message
                .get("exitCode")
                .and_then(Value::as_i64)
                .is_some_and(|code| code != 0);
            let mut call_extra = Map::new();
            call_extra.insert("tool_name".to_string(), Value::String("bash".to_string()));
            call_extra.insert("arguments".to_string(), json!({ "command": command }));
            call_extra.insert("command".to_string(), Value::String(command.to_string()));
            events.push(message_envelope(
                session_id,
                cwd,
                project_root,
                entry_id,
                parent_id,
                original,
                "message.bashExecution",
                CanonicalType::ToolCall,
                call_source_event_id,
                message_id_for(entry_id),
                source,
                captured_at.clone(),
                call_extra,
            ));
            let mut result_extra = Map::new();
            result_extra.insert("tool_name".to_string(), Value::String("bash".to_string()));
            result_extra.insert(
                "output".to_string(),
                content_to_searchable(message.get("output")),
            );
            result_extra.insert("exit_code".to_string(), exit_code);
            result_extra.insert("is_error".to_string(), Value::Bool(is_error));
            result_extra.insert(
                "status".to_string(),
                Value::String(if is_error {
                    "error".to_string()
                } else {
                    "success".to_string()
                }),
            );
            events.push(message_envelope(
                session_id,
                cwd,
                project_root,
                entry_id,
                parent_id,
                original,
                "message.bashExecution",
                CanonicalType::ToolResult,
                result_source_event_id,
                message_id_for(entry_id),
                source,
                captured_at,
                result_extra,
            ));
            events
        }
        // A custom role inside a message is treated as user text.
        "custom" => vec![message_envelope(
            session_id,
            cwd,
            project_root,
            entry_id,
            parent_id,
            original,
            "message.user",
            CanonicalType::UserMessage,
            entry_id
                .map(str::to_string)
                .or_else(|| live_fallback_id("user", session_id, message)),
            message_id_for(entry_id),
            source,
            captured_at,
            json_map(&[("text", content_to_searchable(message.get("content")))]),
        )],
        other => vec![error_envelope_from_parts(
            session_id,
            cwd,
            project_root,
            original,
            entry_id,
            format!("unknown message role: {other}"),
        )],
    }
}

/// The pi session header envelope (`session.started`), shared by backfill
/// (from the file header) and live (`session_start` hook). The source event id
/// `session-header:<session_id>` is identical on both paths, so a live session
/// start dedupes against a later backfill of the same session.
#[allow(clippy::too_many_arguments)]
pub(crate) fn session_start_envelope(
    session_id: &str,
    cwd: Option<&str>,
    project_root: Option<&str>,
    session_version: Option<&Value>,
    parent_session: Option<&Value>,
    original: &Value,
    source: Source,
    captured_at: String,
) -> EventEnvelope {
    EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at,
        tool: Tool::Pi,
        tool_version: None,
        session_id: session_id.to_string(),
        filename_session_id: sanitize_session_id(session_id),
        turn_id: None,
        message_id: None,
        project_root: project_root.map(str::to_string),
        cwd: cwd.map(str::to_string),
        source,
        source_event_type: "session".to_string(),
        canonical_type: CanonicalType::SessionStarted,
        source_event_id: Some(format!("session-header:{session_id}")),
        dedupe_key: String::new(),
        sequence: None,
        raw_file: None,
        raw_offset: None,
        payload: pi_payload(
            None,
            None,
            original,
            "session",
            json_map(&[
                (
                    "cwd",
                    cwd.map(str::to_string)
                        .map(Value::String)
                        .unwrap_or(Value::Null),
                ),
                (
                    "session_version",
                    session_version.cloned().unwrap_or(Value::Null),
                ),
                (
                    "parent_session",
                    parent_session.cloned().unwrap_or(Value::Null),
                ),
            ]),
        ),
        payload_ref: None,
    }
}

/// The pi compaction envelope (`compaction.after`). `source_event_id` is the
/// tree entry id when known (backfill, or live with a resolved id); the live
/// path falls back to `live-compact:<session_id>:<captured_at>`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compaction_envelope(
    session_id: &str,
    cwd: Option<&str>,
    project_root: Option<&str>,
    entry_id: Option<&str>,
    summary: Option<&Value>,
    tokens_before: Option<&Value>,
    retained_tail: Option<&Value>,
    original: &Value,
    source: Source,
    captured_at: String,
) -> EventEnvelope {
    let source_event_id = entry_id
        .map(str::to_string)
        .unwrap_or_else(|| format!("live-compact:{session_id}:{captured_at}"));
    EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: captured_at.clone(),
        tool: Tool::Pi,
        tool_version: None,
        session_id: session_id.to_string(),
        filename_session_id: sanitize_session_id(session_id),
        turn_id: None,
        message_id: None,
        project_root: project_root.map(str::to_string),
        cwd: cwd.map(str::to_string),
        source,
        source_event_type: "compaction".to_string(),
        canonical_type: CanonicalType::CompactionAfter,
        source_event_id: Some(source_event_id),
        dedupe_key: String::new(),
        sequence: None,
        raw_file: None,
        raw_offset: None,
        payload: pi_payload(
            entry_id,
            None,
            original,
            "compaction",
            json_map(&[
                ("summary", summary.cloned().unwrap_or(Value::Null)),
                (
                    "tokens_before",
                    tokens_before.cloned().unwrap_or(Value::Null),
                ),
                (
                    "retained_tail",
                    retained_tail.cloned().unwrap_or(Value::Null),
                ),
            ]),
        ),
        payload_ref: None,
    }
}

/// Expand one live hook payload into envelopes. `hook_event_name` (or `type`)
/// dispatches: `session_start` → 1× session.started; `message_end` → expand
/// the message; `session_compact` → 1× compaction.after; anything else is
/// ignored (fail-open: appends nothing, no error).
pub(crate) fn expand_pi_hook_payload(payload: &Value) -> Result<Vec<EventEnvelope>> {
    let hook_event_name = payload
        .get("hook_event_name")
        .or_else(|| payload.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let session_id = payload
        .get("session_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error::Validation("session_id is required".to_string()))?;
    let cwd = payload.get("cwd").and_then(Value::as_str);
    let project_root = payload.get("project_root").and_then(Value::as_str).or(cwd);
    let captured_at = payload
        .get("captured_at")
        .and_then(Value::as_str)
        .filter(|value| OffsetDateTime::parse(value, &Rfc3339).is_ok())
        .map(str::to_string)
        .unwrap_or_else(|| {
            OffsetDateTime::now_utc()
                .format(&Rfc3339)
                .unwrap_or_default()
        });
    match hook_event_name {
        "session_start" => Ok(vec![session_start_envelope(
            session_id,
            cwd,
            project_root,
            payload.get("session_version"),
            None,
            payload,
            Source::Hook,
            captured_at,
        )]),
        "message_end" => {
            let message = payload.get("message").ok_or_else(|| {
                Error::Validation("message is required for message_end".to_string())
            })?;
            let entry_id = payload
                .get("entry_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            let parent_id = payload
                .get("parent_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            let original = message;
            Ok(expand_pi_message(
                session_id,
                cwd,
                project_root,
                original,
                message,
                entry_id,
                parent_id,
                Source::Hook,
                captured_at,
            ))
        }
        "session_compact" => Ok(vec![compaction_envelope(
            session_id,
            cwd,
            project_root,
            payload
                .get("entry_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty()),
            payload.get("summary"),
            payload.get("tokens_before"),
            payload.get("retained_tail"),
            payload,
            Source::Hook,
            captured_at,
        )]),
        _ => Ok(Vec::new()),
    }
}

/// Live-only synthetic id when no tree entry id is available:
/// `live-<kind>:<session_id>:<message.timestamp>` (unix ms). `live-bash` ids
/// are prefixed again by the bash caller, mirroring the backfill shapes.
fn live_fallback_id(kind: &str, session_id: &str, message: &Value) -> Option<String> {
    message
        .get("timestamp")
        .and_then(Value::as_i64)
        .map(|timestamp| format!("live-{kind}:{session_id}:{timestamp}"))
}

fn message_id_for(entry_id: Option<&str>) -> Option<String> {
    entry_id.map(str::to_string)
}

/// Error envelope for an unrecognized message role (live path; backfill builds
/// its own file-level error envelopes).
fn error_envelope_from_parts(
    session_id: &str,
    cwd: Option<&str>,
    project_root: Option<&str>,
    original: &Value,
    entry_id: Option<&str>,
    message: String,
) -> EventEnvelope {
    let mut flattened = Map::new();
    flattened.insert("message".to_string(), Value::String(message));
    flattened.insert(
        "entry_id".to_string(),
        entry_id
            .map(str::to_string)
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    flattened.insert("pi_type".to_string(), Value::String("message".to_string()));
    message_envelope(
        session_id,
        cwd,
        project_root,
        entry_id,
        None,
        original,
        "pi.unknown",
        CanonicalType::Error,
        entry_id.map(str::to_string),
        None,
        Source::Hook,
        OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_default(),
        flattened,
    )
}

/// One envelope for a message-family event.
#[allow(clippy::too_many_arguments)]
fn message_envelope(
    session_id: &str,
    cwd: Option<&str>,
    project_root: Option<&str>,
    entry_id: Option<&str>,
    parent_id: Option<&str>,
    original: &Value,
    source_event_type: &str,
    canonical_type: CanonicalType,
    source_event_id: Option<String>,
    message_id: Option<String>,
    source: Source,
    captured_at: String,
    flattened: Map<String, Value>,
) -> EventEnvelope {
    EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at,
        tool: Tool::Pi,
        tool_version: None,
        session_id: session_id.to_string(),
        filename_session_id: sanitize_session_id(session_id),
        turn_id: None,
        message_id,
        project_root: project_root.map(str::to_string),
        cwd: cwd.map(str::to_string),
        source,
        source_event_type: source_event_type.to_string(),
        canonical_type,
        source_event_id,
        dedupe_key: String::new(),
        sequence: None,
        raw_file: None,
        raw_offset: None,
        // pi_type is the same discriminator as source_event_type (the locked
        // mapping-table column), so every payload carries its real entry kind:
        // `message.user`, `compaction`, `model_change`, `custom_message`, ...
        payload: pi_payload(entry_id, parent_id, original, source_event_type, flattened),
        payload_ref: None,
    }
}

/// Build the locked payload shape: `entry_id`/`parent_id`/`pi_type` plus
/// type-specific flattened fields plus the full original under `pi_entry`.
/// `entry_id`/`parent_id` are explicit (tree entry ids on backfill; hook
/// payload fields on live ingest) because the raw `original` object does not
/// carry them uniformly.
pub(crate) fn pi_payload(
    entry_id: Option<&str>,
    parent_id: Option<&str>,
    original: &Value,
    pi_type: &str,
    mut flattened: Map<String, Value>,
) -> Value {
    flattened.insert(
        "entry_id".to_string(),
        entry_id
            .map(str::to_string)
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    flattened.insert(
        "parent_id".to_string(),
        parent_id
            .map(str::to_string)
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    flattened.insert("pi_type".to_string(), Value::String(pi_type.to_string()));
    flattened.insert("pi_entry".to_string(), original.clone());
    Value::Object(flattened)
}

/// Searchable text for an assistant message: text + thinking blocks joined,
/// images replaced by a `[image]` placeholder, toolCall blocks skipped (they
/// expand to their own events).
fn assistant_searchable_text(content: Option<&Value>) -> String {
    let Some(Value::Array(blocks)) = content else {
        return String::new();
    };
    let mut parts = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(text.to_string());
                }
            }
            Some("thinking") => {
                if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                    parts.push(text.to_string());
                }
            }
            Some("image") => parts.push("[image]".to_string()),
            _ => {}
        }
    }
    join_parts(parts)
}

/// Render a content field to searchable text: strings pass through; block
/// arrays render text/thinking, replace images with `[image]`, and skip
/// toolCall blocks.
pub(crate) fn content_to_searchable(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) => Value::String(text.trim().to_string()),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            parts.push(text.to_string());
                        }
                    }
                    Some("thinking") => {
                        if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                            parts.push(text.to_string());
                        }
                    }
                    Some("image") => parts.push("[image]".to_string()),
                    _ => {}
                }
            }
            Value::String(join_parts(parts))
        }
        _ => Value::String(String::new()),
    }
}

/// Join non-empty strings with newlines.
fn join_parts(parts: Vec<String>) -> String {
    let mut seen = std::collections::BTreeSet::new();
    let mut output = Vec::new();
    for part in parts {
        let part = part.trim();
        if !part.is_empty() && seen.insert(part.to_string()) {
            output.push(part.to_string());
        }
    }
    output.join("\n")
}

pub(crate) fn json_map(pairs: &[(&str, Value)]) -> Map<String, Value> {
    let mut map = Map::new();
    for (key, value) in pairs {
        map.insert(key.to_string(), value.clone());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{append_prepared_events, ingest_hook_event, ingest_hook_events, init_home};
    use serde_json::json;
    use tempfile::tempdir;

    const SESSION_ID: &str = "019ff094-342e-7f5c-a607-fd1a4bc04d32";

    fn message_end_payload(entry_id: Option<&str>, message: Value) -> Value {
        let mut payload = json!({
            "hook_event_name": "message_end",
            "type": "message_end",
            "session_id": SESSION_ID,
            "cwd": "/tmp/nabu-pi-fixture",
            "project_root": "/tmp/nabu-pi-fixture",
            "message": message,
        });
        if let Some(entry_id) = entry_id {
            payload["entry_id"] = json!(entry_id);
        }
        payload
    }

    #[test]
    fn expand_message_end_matches_backfill_ids() {
        // The same assistant message with a toolCall: live (with entry id) and
        // backfill (parsed from a session file) must produce identical source
        // event ids for the assistant.message and the tool.call.
        let message = json!({
            "role": "assistant",
            "content": [
                { "type": "text", "text": "let me check" },
                { "type": "toolCall", "id": "call_123", "name": "read", "arguments": { "path": "/tmp/x" } }
            ],
            "provider": "xai",
            "model": "grok-4.5",
            "stopReason": "toolUse",
            "timestamp": 1782900002000i64
        });
        let live = expand_pi_hook_payload(&message_end_payload(Some("b2c3d4e5"), message.clone()))
            .unwrap();
        assert_eq!(live.len(), 2);
        assert_eq!(live[0].canonical_type, CanonicalType::AssistantMessage);
        assert_eq!(live[0].source_event_id.as_deref(), Some("b2c3d4e5"));
        assert_eq!(live[1].canonical_type, CanonicalType::ToolCall);
        assert_eq!(live[1].source_event_id.as_deref(), Some("call_123"));
        // Same payload shape as backfill for the same logical events.
        assert_eq!(live[0].payload["pi_type"], "message.assistant");
        assert_eq!(live[0].payload["text"], "let me check");
        assert_eq!(live[1].payload["tool_name"], "read");
    }

    #[test]
    fn live_ids_fall_back_to_live_prefix_without_entry_id() {
        let message = json!({
            "role": "user",
            "content": "hello",
            "timestamp": 1782900001000i64
        });
        let events = expand_pi_hook_payload(&message_end_payload(None, message)).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].source_event_id.as_deref(),
            Some("live-user:019ff094-342e-7f5c-a607-fd1a4bc04d32:1782900001000")
        );

        let bash = json!({
            "role": "bashExecution",
            "command": "cargo test",
            "output": "ok",
            "exitCode": 0,
            "timestamp": 1782900002000i64
        });
        let events = expand_pi_hook_payload(&message_end_payload(None, bash)).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0].source_event_id.as_deref(),
            Some("bash-call:live-bash:019ff094-342e-7f5c-a607-fd1a4bc04d32:1782900002000")
        );
        assert_eq!(
            events[1].source_event_id.as_deref(),
            Some("bash-result:live-bash:019ff094-342e-7f5c-a607-fd1a4bc04d32:1782900002000")
        );
    }

    #[test]
    fn session_start_and_compact_hook_ids_match_backfill() {
        let start = expand_pi_hook_payload(&json!({
            "hook_event_name": "session_start",
            "type": "session_start",
            "session_id": SESSION_ID,
            "cwd": "/tmp/nabu-pi-fixture",
            "project_root": "/tmp/nabu-pi-fixture",
            "session_version": 3,
        }))
        .unwrap();
        assert_eq!(start.len(), 1);
        assert_eq!(start[0].canonical_type, CanonicalType::SessionStarted);
        assert_eq!(
            start[0].source_event_id.as_deref(),
            Some("session-header:019ff094-342e-7f5c-a607-fd1a4bc04d32")
        );

        let compact = expand_pi_hook_payload(&json!({
            "hook_event_name": "session_compact",
            "type": "session_compact",
            "session_id": SESSION_ID,
            "cwd": "/tmp/nabu-pi-fixture",
            "entry_id": "f6g7h8i9",
            "summary": "User discussed X",
            "tokens_before": 50000,
        }))
        .unwrap();
        assert_eq!(compact.len(), 1);
        assert_eq!(compact[0].canonical_type, CanonicalType::CompactionAfter);
        assert_eq!(compact[0].source_event_id.as_deref(), Some("f6g7h8i9"));
        assert_eq!(compact[0].payload["summary"], "User discussed X");
    }

    #[test]
    fn unknown_hook_is_ok_noop() {
        let events = expand_pi_hook_payload(&json!({
            "hook_event_name": "tool_execution_start",
            "session_id": SESSION_ID,
            "cwd": "/tmp",
        }))
        .unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn ingest_pi_message_end_appends_assistant_and_tools() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let payload = message_end_payload(
            Some("b2c3d4e5"),
            json!({
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "checking" },
                    { "type": "toolCall", "id": "call_9", "name": "bash", "arguments": { "command": "ls" } }
                ],
                "provider": "xai",
                "model": "grok-4.5",
                "stopReason": "toolUse",
                "timestamp": 1782900002000i64
            }),
        );
        let reports = ingest_hook_events(&home, Tool::Pi, payload).unwrap();
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| report.appended));

        // Re-ingest dedupes to zero appends (stable source event ids).
        let again = ingest_hook_events(
            &home,
            Tool::Pi,
            message_end_payload(
                Some("b2c3d4e5"),
                json!({
                    "role": "assistant",
                    "content": [
                        { "type": "text", "text": "checking" },
                        { "type": "toolCall", "id": "call_9", "name": "bash", "arguments": { "command": "ls" } }
                    ],
                    "provider": "xai",
                    "model": "grok-4.5",
                    "stopReason": "toolUse",
                    "timestamp": 1782900002000i64
                }),
            ),
        )
        .unwrap();
        assert_eq!(again.len(), 2);
        assert!(again.iter().all(|report| !report.appended));
    }

    #[test]
    fn ingest_pi_session_start_then_backfill_dedupes() {
        // Live session_start appends; a backfill header for the same session
        // (same session-header:<uuid> id) then appends nothing.
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_events(
            &home,
            Tool::Pi,
            json!({
                "hook_event_name": "session_start",
                "type": "session_start",
                "session_id": SESSION_ID,
                "cwd": "/tmp/nabu-pi-fixture",
                "project_root": "/tmp/nabu-pi-fixture",
                "session_version": 3,
            }),
        )
        .unwrap();

        let header = json!({
            "type": "session",
            "version": 3,
            "id": SESSION_ID,
            "timestamp": "2026-07-01T10:00:00.000Z",
            "cwd": "/tmp/nabu-pi-fixture"
        });
        // Build the same envelope the backfill parser would:
        let envelope = session_start_envelope(
            SESSION_ID,
            Some("/tmp/nabu-pi-fixture"),
            Some("/tmp/nabu-pi-fixture"),
            Some(&json!(3)),
            None,
            &header,
            Source::Backfill,
            "2026-07-01T10:00:00.000Z".to_string(),
        );
        let reports = append_prepared_events(&home, vec![envelope]).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].appended, "live header already captured");
    }

    #[test]
    fn ingest_unknown_hook_is_ok_noop() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let reports = ingest_hook_events(
            &home,
            Tool::Pi,
            json!({
                "hook_event_name": "message_update",
                "session_id": SESSION_ID,
                "cwd": "/tmp",
            }),
        )
        .unwrap();
        assert!(reports.is_empty());
    }

    #[test]
    fn ingest_hook_event_single_tools_unchanged() {
        // Non-pi tools keep the single-envelope contract.
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let report = ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "claude-session",
                "hook_event_name": "UserPromptSubmit",
                "prompt": "hello"
            }),
        )
        .unwrap();
        assert!(report.appended);
    }
}
