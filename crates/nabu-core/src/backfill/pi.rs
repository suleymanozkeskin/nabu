//! Pi session backfill: parses pi session JSONL files
//! (`~/.pi/agent/sessions/--<cwd>--/<timestamp>_<uuid>.jsonl`) into canonical
//! nabu envelopes with full tree fidelity.
//!
//! Every entry in the file is imported in file order (abandoned branches are
//! still history), `parentId`/`id` are preserved on each payload, and the full
//! original entry is kept under `payload.pi_entry` for raw fidelity. Source
//! event ids are stable per logical event so re-backfill after live capture
//! dedupes to zero appends (see the README mapping table in plans/).

use crate::pi_map::{
    compaction_envelope, expand_pi_message, json_map, pi_payload, session_start_envelope,
};
use crate::{
    sanitize_session_id, CanonicalType, Error, EventEnvelope, Result, Source, Tool, SCHEMA_VERSION,
};
use serde_json::{Map, Value};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use super::ParsedBackfillSource;

/// True when `path` is a pi session file: a `.jsonl` whose first non-empty
/// line parses as JSON with `type == "session"` and a non-empty string `id`.
pub(crate) fn is_pi_session_file(path: &Path) -> bool {
    if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
        return false;
    }
    let Ok(file) = File::open(path) else {
        return false;
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .map(|bytes| bytes == 0)
            .unwrap_or(true)
        {
            return false;
        }
        if line.trim().is_empty() {
            continue;
        }
        let Ok(entry) = serde_json::from_str::<Value>(line.trim_end()) else {
            return false;
        };
        return entry.get("type").and_then(Value::as_str) == Some("session")
            && entry
                .get("id")
                .and_then(Value::as_str)
                .is_some_and(|id| !id.is_empty());
    }
}

/// Parse a pi session JSONL file into fully populated envelopes. The parser
/// owns the whole file (ignores checkpoint offsets; the append path dedupes),
/// so re-runs are idempotent. A malformed line produces one `error` event and
/// parsing continues.
pub(crate) fn parse_pi_session_jsonl(source_path: &Path) -> Result<ParsedBackfillSource> {
    let file = File::open(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut offset = 0u64;
    let mut events = Vec::new();
    let mut session_id: Option<String> = None;
    let mut cwd: Option<String> = None;

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: source_path.to_path_buf(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        let line_start = offset;
        offset += bytes as u64;
        if line.trim().is_empty() {
            continue;
        }
        let entry = match serde_json::from_str::<Value>(line.trim_end()) {
            Ok(entry) => entry,
            Err(error) => {
                let mut payload = Map::new();
                payload.insert("parse_error".to_string(), Value::String(error.to_string()));
                payload.insert(
                    "raw_line".to_string(),
                    Value::String(line.trim_end().to_string()),
                );
                events.push(error_envelope(
                    source_path,
                    &session_id,
                    Some(format!("malformed:{line_start}")),
                    payload,
                ));
                continue;
            }
        };

        let entry_type = entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match entry_type {
            "session" => {
                if session_id.is_some() {
                    events.push(error_envelope(
                        source_path,
                        &session_id,
                        entry_id(&entry).map(|id| format!("duplicate-header:{id}")),
                        json_map(&[("pi_type", Value::String("session".to_string()))]),
                    ));
                    continue;
                }
                let Some(id) = entry.get("id").and_then(Value::as_str) else {
                    events.push(error_envelope(
                        source_path,
                        &session_id,
                        None,
                        json_map(&[("pi_type", Value::String("session".to_string()))]),
                    ));
                    continue;
                };
                session_id = Some(id.to_string());
                cwd = entry.get("cwd").and_then(Value::as_str).map(str::to_string);
                events.push(session_start_envelope(
                    id,
                    cwd.as_deref(),
                    cwd.as_deref(),
                    entry.get("version"),
                    entry.get("parentSession"),
                    &entry,
                    Source::Backfill,
                    entry_timestamp(&entry),
                ));
            }
            "" => {
                events.push(error_envelope(
                    source_path,
                    &session_id,
                    entry_id(&entry),
                    json_map(&[("pi_type", Value::String("unknown".to_string()))]),
                ));
            }
            "message" => {
                let Some(session_id) = session_id.clone() else {
                    events.push(error_envelope(
                        source_path,
                        &None,
                        entry_id(&entry),
                        json_map(&[("pi_type", Value::String("message".to_string()))]),
                    ));
                    continue;
                };
                let entry_id = entry_id(&entry);
                events.extend(expand_pi_message(
                    &session_id,
                    cwd.as_deref(),
                    cwd.as_deref(),
                    &entry,
                    entry.get("message").unwrap_or(&Value::Object(Map::new())),
                    entry_id.as_deref(),
                    entry.get("parentId").and_then(Value::as_str),
                    Source::Backfill,
                    entry_timestamp(&entry),
                ));
            }
            "compaction" => {
                let Some(session_id) = session_id.clone() else {
                    events.push(error_envelope(
                        source_path,
                        &None,
                        entry_id(&entry),
                        json_map(&[("pi_type", Value::String("compaction".to_string()))]),
                    ));
                    continue;
                };
                events.push(compaction_envelope(
                    &session_id,
                    cwd.as_deref(),
                    cwd.as_deref(),
                    entry_id(&entry).as_deref(),
                    entry.get("summary"),
                    entry.get("tokensBefore"),
                    entry.get("retainedTail"),
                    &entry,
                    Source::Backfill,
                    entry_timestamp(&entry),
                ));
            }
            "branch_summary" => {
                events.push(single_event_envelope(
                    &session_id,
                    &cwd,
                    &entry,
                    "branch_summary",
                    CanonicalType::SessionResumed,
                    entry_id(&entry),
                    json_map(&[
                        (
                            "summary",
                            entry.get("summary").cloned().unwrap_or(Value::Null),
                        ),
                        (
                            "from_id",
                            entry.get("fromId").cloned().unwrap_or(Value::Null),
                        ),
                    ]),
                ));
            }
            "model_change" => {
                let provider = entry.get("provider").and_then(Value::as_str);
                let model = entry.get("modelId").and_then(Value::as_str);
                events.push(single_event_envelope(
                    &session_id,
                    &cwd,
                    &entry,
                    "model_change",
                    CanonicalType::SessionResumed,
                    entry_id(&entry),
                    json_map(&[
                        (
                            "provider",
                            entry.get("provider").cloned().unwrap_or(Value::Null),
                        ),
                        (
                            "model",
                            entry.get("modelId").cloned().unwrap_or(Value::Null),
                        ),
                        (
                            "text",
                            Value::String(match (provider, model) {
                                (Some(provider), Some(model)) => {
                                    format!("model change: {provider}/{model}")
                                }
                                (Some(provider), None) => format!("model change: {provider}"),
                                _ => "model change".to_string(),
                            }),
                        ),
                    ]),
                ));
            }
            "thinking_level_change" => {
                events.push(single_event_envelope(
                    &session_id,
                    &cwd,
                    &entry,
                    "thinking_level_change",
                    CanonicalType::SessionResumed,
                    entry_id(&entry),
                    json_map(&[
                        (
                            "thinking_level",
                            entry.get("thinkingLevel").cloned().unwrap_or(Value::Null),
                        ),
                        (
                            "text",
                            Value::String(
                                entry
                                    .get("thinkingLevel")
                                    .and_then(Value::as_str)
                                    .map(|level| format!("thinking level: {level}"))
                                    .unwrap_or_else(|| "thinking level change".to_string()),
                            ),
                        ),
                    ]),
                ));
            }
            "session_info" => {
                events.push(single_event_envelope(
                    &session_id,
                    &cwd,
                    &entry,
                    "session_info",
                    CanonicalType::SessionResumed,
                    entry_id(&entry),
                    json_map(&[
                        ("name", entry.get("name").cloned().unwrap_or(Value::Null)),
                        (
                            "text",
                            entry
                                .get("name")
                                .map(|name| Value::String(format!("session name: {name}")))
                                .unwrap_or(Value::Null),
                        ),
                    ]),
                ));
            }
            "label" => {
                events.push(single_event_envelope(
                    &session_id,
                    &cwd,
                    &entry,
                    "label",
                    CanonicalType::SessionResumed,
                    entry_id(&entry),
                    json_map(&[
                        (
                            "target_id",
                            entry.get("targetId").cloned().unwrap_or(Value::Null),
                        ),
                        ("label", entry.get("label").cloned().unwrap_or(Value::Null)),
                        (
                            "text",
                            entry
                                .get("label")
                                .map(|label| Value::String(format!("label: {label}")))
                                .unwrap_or(Value::Null),
                        ),
                    ]),
                ));
            }
            "custom_message" => {
                events.push(single_event_envelope(
                    &session_id,
                    &cwd,
                    &entry,
                    "custom_message",
                    CanonicalType::UserMessage,
                    entry_id(&entry),
                    json_map(&[
                        (
                            "custom_type",
                            entry.get("customType").cloned().unwrap_or(Value::Null),
                        ),
                        (
                            "text",
                            crate::pi_map::content_to_searchable(entry.get("content")),
                        ),
                    ]),
                ));
            }
            // Extension private state: not conversation history, no envelope.
            "custom" => {}
            _ => {
                events.push(error_envelope(
                    source_path,
                    &session_id,
                    entry_id(&entry),
                    json_map(&[
                        ("pi_type", Value::String(entry_type.to_string())),
                        ("pi_entry", entry),
                    ]),
                ));
            }
        }
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id: session_id,
    })
}

/// Expand one `message` entry into 1..N envelopes per the locked mapping:
/// user → one user.message; assistant → assistant.message plus one tool.call
/// per toolCall content block; toolResult → tool.result; bashExecution → a
/// tool.call/tool.result pair.
fn single_event_envelope(
    session_id: &Option<String>,
    cwd: &Option<String>,
    entry: &Value,
    pi_type: &str,
    canonical_type: CanonicalType,
    source_event_id: Option<String>,
    flattened: Map<String, Value>,
) -> EventEnvelope {
    let session_id = session_id.clone().unwrap_or_else(|| "missing".to_string());
    let entry_id = entry_id(entry);
    let parent_id = entry.get("parentId").and_then(Value::as_str);
    EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: entry_timestamp(entry),
        tool: Tool::Pi,
        tool_version: None,
        session_id: session_id.clone(),
        filename_session_id: sanitize_session_id(&session_id),
        turn_id: None,
        message_id: None,
        project_root: cwd.clone(),
        cwd: cwd.clone(),
        source: Source::Backfill,
        source_event_type: pi_type.to_string(),
        canonical_type,
        source_event_id,
        dedupe_key: String::new(),
        sequence: None,
        raw_file: None,
        raw_offset: None,
        // pi_type is the same discriminator as source_event_type (the locked
        // mapping-table column), so every payload carries its real entry kind.
        payload: pi_payload(entry_id.as_deref(), parent_id, entry, pi_type, flattened),
        payload_ref: None,
    }
}

fn error_envelope(
    source_path: &Path,
    session_id: &Option<String>,
    source_event_id: Option<String>,
    flattened: Map<String, Value>,
) -> EventEnvelope {
    let session_id = session_id
        .clone()
        .unwrap_or_else(|| source_path_fallback(source_path));
    let mut payload = Map::new();
    payload.insert("pi_type".to_string(), Value::String("error".to_string()));
    payload.insert(
        "source_path".to_string(),
        Value::String(source_path.display().to_string()),
    );
    for (key, value) in flattened {
        payload.insert(key, value);
    }
    EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_default(),
        tool: Tool::Pi,
        tool_version: None,
        session_id: session_id.clone(),
        filename_session_id: sanitize_session_id(&session_id),
        turn_id: None,
        message_id: None,
        project_root: None,
        cwd: None,
        source: Source::Backfill,
        source_event_type: "pi.unknown".to_string(),
        canonical_type: CanonicalType::Error,
        source_event_id,
        dedupe_key: String::new(),
        sequence: None,
        raw_file: None,
        raw_offset: None,
        payload: Value::Object(payload),
        payload_ref: None,
    }
}

/// Error envelope built from an entry when no session header exists yet.
fn entry_id(entry: &Value) -> Option<String> {
    entry
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

/// The entry's ISO timestamp when parseable, else now.
fn entry_timestamp(entry: &Value) -> String {
    entry
        .get("timestamp")
        .and_then(Value::as_str)
        .filter(|timestamp| OffsetDateTime::parse(timestamp, &Rfc3339).is_ok())
        .map(str::to_string)
        .unwrap_or_else(|| {
            OffsetDateTime::now_utc()
                .format(&Rfc3339)
                .unwrap_or_default()
        })
}

/// Deterministic session id for error events emitted before any header.
fn source_path_fallback(source_path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(source_path.display().to_string().as_bytes());
    let hash = hex::encode(hasher.finalize());
    format!("pi-source-{}", &hash[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{append_prepared_events, init_home, search_history, CanonicalType};
    use std::fs;
    use std::io::Write;
    use tempfile::tempdir;

    const FIXTURE_CWD: &str = "/tmp/nabu-pi-fixture";
    const FIXTURE_SESSION_ID: &str = "019ff094-342e-7f5c-a607-fd1a4bc04d32";

    fn write_fixture(name: &str, lines: &[&str]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempdir().unwrap();
        let path = dir.path().join(name);
        let mut file = fs::File::create(&path).unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        (dir, path)
    }

    fn header() -> String {
        format!(
            r#"{{"type":"session","version":3,"id":"{FIXTURE_SESSION_ID}","timestamp":"2026-07-01T10:00:00.000Z","cwd":"{FIXTURE_CWD}"}}"#
        )
    }

    fn user_entry(id: &str, parent: &str, text: &str) -> String {
        let parent = json_parent(parent);
        format!(
            r#"{{"type":"message","id":"{id}","parentId":{parent},"timestamp":"2026-07-01T10:00:01.000Z","message":{{"role":"user","content":"{text}","timestamp":1782900001000}}}}"#
        )
    }

    fn assistant_entry(id: &str, parent: &str) -> String {
        let parent = json_parent(parent);
        format!(
            r#"{{"type":"message","id":"{id}","parentId":{parent},"timestamp":"2026-07-01T10:00:02.000Z","message":{{"role":"assistant","content":[{{"type":"text","text":"hello there"}}],"provider":"xai","model":"grok-4.5","usage":{{"input":1,"output":1,"totalTokens":2}},"stopReason":"stop","timestamp":1782900002000}}}}"#
        )
    }

    /// JSON token for a parent id: `null` for "null", else a quoted id.
    fn json_parent(parent: &str) -> String {
        if parent == "null" {
            "null".to_string()
        } else {
            format!("\"{parent}\"")
        }
    }

    fn parse(lines: &[&str]) -> Vec<EventEnvelope> {
        let (_dir, path) = write_fixture("session.jsonl", lines);
        parse_pi_session_jsonl(&path).unwrap().events
    }

    #[test]
    fn parses_linear_session() {
        let events = parse(&[
            &header(),
            &user_entry("a1b2c3d4", "null", "first prompt"),
            &assistant_entry("b2c3d4e5", "a1b2c3d4"),
            &user_entry("c3d4e5f6", "b2c3d4e5", "second prompt"),
        ]);
        let types: Vec<&str> = events
            .iter()
            .map(|event| event.canonical_type.as_str())
            .collect();
        assert_eq!(
            types,
            vec![
                "session.started",
                "user.message",
                "assistant.message",
                "user.message"
            ]
        );
        assert_eq!(events[0].session_id, FIXTURE_SESSION_ID);
        assert_eq!(
            events[0].source_event_id.as_deref(),
            Some("session-header:019ff094-342e-7f5c-a607-fd1a4bc04d32")
        );
        assert_eq!(events[0].cwd.as_deref(), Some(FIXTURE_CWD));
        assert_eq!(events[0].project_root.as_deref(), Some(FIXTURE_CWD));
        assert_eq!(events[0].payload["pi_type"], "session");
        assert_eq!(events[1].payload["text"], "first prompt");
        assert_eq!(events[1].payload["pi_type"], "message.user");
        assert_eq!(events[1].payload["entry_id"], "a1b2c3d4");
        assert_eq!(events[1].payload["parent_id"], Value::Null);
        assert_eq!(events[2].payload["text"], "hello there");
        assert_eq!(events[2].message_id.as_deref(), Some("b2c3d4e5"));
        // Full original entry preserved for raw fidelity.
        assert_eq!(
            events[2].payload["pi_entry"]["message"]["model"],
            "grok-4.5"
        );
    }

    #[test]
    fn expands_tool_calls() {
        let assistant = r#"{"type":"message","id":"b2c3d4e5","parentId":"a1b2c3d4","timestamp":"2026-07-01T10:00:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"let me check"},{"type":"toolCall","id":"call_123","name":"read","arguments":{"path":"/tmp/x"}}],"provider":"xai","model":"grok-4.5","stopReason":"toolUse","timestamp":1782900002000}}"#
            .to_string();
        let tool_result =             r#"{"type":"message","id":"c3d4e5f6","parentId":"b2c3d4e5","timestamp":"2026-07-01T10:00:03.000Z","message":{"role":"toolResult","toolCallId":"call_123","toolName":"read","content":[{"type":"text","text":"file contents"}],"isError":false,"timestamp":1782900003000}}"#
            .to_string();
        let events = parse(&[
            &header(),
            &user_entry("a1b2c3d4", "null", "read the file"),
            &assistant,
            &tool_result,
        ]);
        let types: Vec<&str> = events
            .iter()
            .map(|event| event.canonical_type.as_str())
            .collect();
        assert_eq!(
            types,
            vec![
                "session.started",
                "user.message",
                "assistant.message",
                "tool.call",
                "tool.result"
            ]
        );
        assert_eq!(events[3].source_event_id.as_deref(), Some("call_123"));
        assert_eq!(events[3].payload["pi_type"], "message.assistant.toolCall");
        assert_eq!(events[3].payload["tool_name"], "read");
        assert_eq!(events[3].payload["parent_message_entry_id"], "b2c3d4e5");
        assert_eq!(events[3].payload["arguments"]["path"], "/tmp/x");
        assert_eq!(events[4].source_event_id.as_deref(), Some("call_123"));
        assert_eq!(events[4].payload["tool_name"], "read");
        assert_eq!(events[4].payload["output"], "file contents");
        assert_eq!(events[4].payload["status"], "success");
        // ToolCall block must not leak into assistant searchable text.
        assert_eq!(events[2].payload["text"], "let me check");
    }

    #[test]
    fn expands_bash_execution() {
        let bash =             r#"{"type":"message","id":"d4e5f6g7","parentId":"c3d4e5f6","timestamp":"2026-07-01T10:00:04.000Z","message":{"role":"bashExecution","command":"cargo test","output":"test result: ok","exitCode":1,"timestamp":1782900004000}}"#
            .to_string();
        let events = parse(&[&header(), &bash]);
        assert_eq!(events.len(), 3);
        assert_eq!(events[1].canonical_type, CanonicalType::ToolCall);
        assert_eq!(
            events[1].source_event_id.as_deref(),
            Some("bash-call:d4e5f6g7")
        );
        assert_eq!(events[1].payload["command"], "cargo test");
        assert_eq!(events[2].canonical_type, CanonicalType::ToolResult);
        assert_eq!(
            events[2].source_event_id.as_deref(),
            Some("bash-result:d4e5f6g7")
        );
        assert_eq!(events[2].payload["output"], "test result: ok");
        assert_eq!(events[2].payload["is_error"], true);
        assert_eq!(events[2].payload["status"], "error");
    }

    #[test]
    fn imports_both_branches() {
        let branch_a = user_entry("c3d4e5f6", "b2c3d4e5", "approach A");
        let branch_b = user_entry("f6g7h8i9", "b2c3d4e5", "approach B");
        let events = parse(&[
            &header(),
            &user_entry("a1b2c3d4", "null", "root"),
            &assistant_entry("b2c3d4e5", "a1b2c3d4"),
            &branch_a,
            &branch_b,
        ]);
        let texts: Vec<&str> = events
            .iter()
            .filter(|event| event.canonical_type == CanonicalType::UserMessage)
            .map(|event| event.payload["text"].as_str().unwrap_or_default())
            .collect();
        assert!(texts.contains(&"root"));
        assert!(texts.contains(&"approach A"));
        assert!(texts.contains(&"approach B"));
        // parent_id preserved on both branch children.
        assert_eq!(events[3].payload["parent_id"], "b2c3d4e5");
        assert_eq!(events[4].payload["parent_id"], "b2c3d4e5");
    }

    #[test]
    fn skips_custom_keeps_custom_message() {
        let custom = r#"{"type":"custom","id":"h8i9j0k1","parentId":"g7h8i9j0","timestamp":"2026-07-01T10:00:05.000Z","customType":"my-extension","data":{"count":42}}"#;
        let custom_message = r#"{"type":"custom_message","id":"i9j0k1l2","parentId":"h8i9j0k1","timestamp":"2026-07-01T10:00:06.000Z","customType":"my-extension","content":"injected context","display":true}"#;
        let events = parse(&[&header(), custom, custom_message]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].canonical_type, CanonicalType::UserMessage);
        assert_eq!(events[1].payload["pi_type"], "custom_message");
        assert_eq!(events[1].payload["text"], "injected context");
        assert_eq!(events[1].payload["custom_type"], "my-extension");
    }

    #[test]
    fn compaction_single_event() {
        let compaction =             r#"{"type":"compaction","id":"f6g7h8i9","parentId":"e5f6g7h8","timestamp":"2026-07-01T10:10:00.000Z","summary":"User discussed X, Y, Z","tokensBefore":50000,"retainedTail":[{"role":"user","content":"latest"}]}"#
            .to_string();
        let events = parse(&[&header(), &compaction]);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].canonical_type, CanonicalType::CompactionAfter);
        assert_eq!(events[1].payload["pi_type"], "compaction");
        assert_eq!(events[1].payload["summary"], "User discussed X, Y, Z");
        assert_eq!(events[1].payload["tokens_before"], 50000);
        assert_eq!(events[1].payload["retained_tail"][0]["role"], "user");
    }

    #[test]
    fn meta_entries_map_to_session_resumed() {
        let model_change = r#"{"type":"model_change","id":"d4e5f6g7","parentId":"c3d4e5f6","timestamp":"2026-07-01T10:05:00.000Z","provider":"openai","modelId":"gpt-4o"}"#;
        let session_info = r#"{"type":"session_info","id":"k1l2m3n4","parentId":"j0k1l2m3","timestamp":"2026-07-01T10:06:00.000Z","name":"Refactor auth module"}"#;
        let label = r#"{"type":"label","id":"j0k1l2m3","parentId":"i9j0k1l2","timestamp":"2026-07-01T10:07:00.000Z","targetId":"a1b2c3d4","label":"checkpoint-1"}"#;
        let branch_summary = r#"{"type":"branch_summary","id":"g7h8i9j0","parentId":"a1b2c3d4","timestamp":"2026-07-01T10:08:00.000Z","fromId":"f6g7h8i9","summary":"Branch explored approach A"}"#;
        let events = parse(&[&header(), model_change, session_info, label, branch_summary]);
        assert_eq!(events.len(), 5);
        for event in &events[1..] {
            assert_eq!(event.canonical_type, CanonicalType::SessionResumed);
        }
        assert_eq!(events[1].payload["pi_type"], "model_change");
        assert_eq!(events[1].payload["text"], "model change: openai/gpt-4o");
        assert_eq!(events[2].payload["pi_type"], "session_info");
        assert_eq!(events[2].payload["name"], "Refactor auth module");
        assert_eq!(events[3].payload["pi_type"], "label");
        assert_eq!(events[3].payload["label"], "checkpoint-1");
        assert_eq!(events[4].payload["pi_type"], "branch_summary");
        assert_eq!(events[4].payload["summary"], "Branch explored approach A");
        assert_eq!(events[4].payload["from_id"], "f6g7h8i9");
    }

    #[test]
    fn idempotent_rebackfill() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let lines = [
            header(),
            user_entry("a1b2c3d4", "null", "first prompt"),
            assistant_entry("b2c3d4e5", "a1b2c3d4"),
        ];
        let (_dir, path) = write_fixture(
            "session.jsonl",
            &lines.iter().map(String::as_str).collect::<Vec<_>>(),
        );
        let parsed = parse_pi_session_jsonl(&path).unwrap();
        let first = append_prepared_events(&home, parsed.events.clone()).unwrap();
        assert_eq!(first.iter().filter(|report| report.appended).count(), 3);
        let second = append_prepared_events(&home, parsed.events).unwrap();
        assert_eq!(second.iter().filter(|report| report.appended).count(), 0);
    }

    #[test]
    fn malformed_line_continues() {
        let events = parse(&[
            &header(),
            &user_entry("a1b2c3d4", "null", "before"),
            "not-json{{{",
            &user_entry("c3d4e5f6", "a1b2c3d4", "after"),
        ]);
        assert_eq!(events.len(), 4);
        assert_eq!(events[2].canonical_type, CanonicalType::Error);
        assert!(events[2].payload.get("parse_error").is_some());
        assert!(events[2]
            .source_event_id
            .as_deref()
            .unwrap()
            .starts_with("malformed:"));
        assert_eq!(events[3].payload["text"], "after");
    }

    #[test]
    fn entries_before_header_and_duplicate_header_error() {
        let events = parse(&[
            &user_entry("a1b2c3d4", "null", "before header"),
            &header(),
            &header(),
        ]);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].canonical_type, CanonicalType::Error);
        assert_eq!(events[1].canonical_type, CanonicalType::SessionStarted);
        assert_eq!(events[2].canonical_type, CanonicalType::Error);
    }

    #[test]
    fn is_pi_session_file_false_for_random_jsonl() {
        let temp = tempdir().unwrap();
        let random = temp.path().join("random.jsonl");
        fs::write(&random, r#"{"type":"message","id":"x","parentId":null,"message":{"role":"user","content":"hi"}}"#).unwrap();
        assert!(!is_pi_session_file(&random));

        let pi_file = temp
            .path()
            .join("2026-07-01T10-00-00-000Z_019ff094-342e-7f5c-a607-fd1a4bc04d32.jsonl");
        fs::write(&pi_file, format!("{}\n", header())).unwrap();
        assert!(is_pi_session_file(&pi_file));

        let empty = temp.path().join("empty.jsonl");
        fs::write(&empty, "").unwrap();
        assert!(!is_pi_session_file(&empty));

        let not_jsonl = temp.path().join("notes.txt");
        fs::write(&not_jsonl, "hello").unwrap();
        assert!(!is_pi_session_file(&not_jsonl));
    }

    #[test]
    fn backfill_indexes_and_search_finds_pi_text() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let source_dir = temp.path().join("source");
        fs::create_dir_all(&source_dir).unwrap();
        let path =
            source_dir.join("2026-07-01T10-00-00-000Z_019ff094-342e-7f5c-a607-fd1a4bc04d32.jsonl");
        fs::write(
            &path,
            format!(
                "{}\n{}\n",
                header(),
                user_entry("a1b2c3d4", "null", "the pi fixture unicorn phrase")
            ),
        )
        .unwrap();

        let parsed = parse_pi_session_jsonl(&path).unwrap();
        assert!(parsed.last_session_id.as_deref() == Some(FIXTURE_SESSION_ID));
        let appended = append_prepared_events(&home, parsed.events).unwrap();
        assert_eq!(appended.len(), 2);
        crate::index_once(&home).unwrap();

        let hits = search_history(&home, "unicorn", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tool, Tool::Pi);
        assert_eq!(hits[0].session_id, FIXTURE_SESSION_ID);
        assert_eq!(hits[0].canonical_type, "user.message");
    }
}
