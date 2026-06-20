//! Codex stream-format parsing.

use super::*;

pub(crate) fn parse_codex_stream_source(source_path: &Path) -> Result<ParsedBackfillSource> {
    match source_path.extension().and_then(|value| value.to_str()) {
        Some("jsonl") => parse_codex_stream_jsonl(source_path),
        Some("json") => parse_codex_stream_json(source_path),
        _ => Ok(ParsedBackfillSource {
            events: Vec::new(),
            last_session_id: None,
        }),
    }
}

pub(crate) fn parse_codex_stream_jsonl(source_path: &Path) -> Result<ParsedBackfillSource> {
    let file = File::open(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut offset = 0u64;
    let mut events = Vec::new();
    let mut last_session_id = None;

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
        let payload = match serde_json::from_str(line.trim_end()) {
            Ok(payload) => payload,
            Err(error) => malformed_native_payload(source_path, line_start, line.trim_end(), error),
        };
        let event = envelope_from_codex_stream_payload(
            source_path,
            line_start,
            payload,
            last_session_id.as_deref(),
        )?;
        last_session_id = Some(event.session_id.clone());
        events.push(event);
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

pub(crate) fn parse_codex_stream_json(source_path: &Path) -> Result<ParsedBackfillSource> {
    let file = File::open(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let payload: Value = match serde_json::from_reader(BufReader::new(file)) {
        Ok(payload) => payload,
        Err(error) => {
            let content = fs::read_to_string(source_path).map_err(|source| Error::Io {
                path: source_path.to_path_buf(),
                source,
            })?;
            let event = envelope_from_codex_stream_payload(
                source_path,
                0,
                malformed_native_payload(source_path, 0, &content, error),
                None,
            )?;
            return Ok(ParsedBackfillSource {
                last_session_id: Some(event.session_id.clone()),
                events: vec![event],
            });
        }
    };
    let records = match payload {
        Value::Array(values) => values,
        Value::Object(mut map) => map
            .remove("events")
            .or_else(|| map.remove("notifications"))
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_else(|| vec![Value::Object(map)]),
        _ => Vec::new(),
    };
    let mut events = Vec::new();
    let mut last_session_id = None;
    for (index, payload) in records.into_iter().enumerate() {
        let event = envelope_from_codex_stream_payload(
            source_path,
            index as u64,
            payload,
            last_session_id.as_deref(),
        )?;
        last_session_id = Some(event.session_id.clone());
        events.push(event);
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

pub(crate) fn envelope_from_codex_stream_payload(
    source_path: &Path,
    byte_offset: u64,
    payload: Value,
    previous_session_id: Option<&str>,
) -> Result<EventEnvelope> {
    let source_event_type = codex_stream_event_name(&payload);
    let session_id = codex_stream_session_id(source_path, &payload, previous_session_id);
    let canonical_type = payload
        .get("canonical_type")
        .and_then(Value::as_str)
        .map(CanonicalType::from_str)
        .transpose()?
        .unwrap_or_else(|| canonical_type_for_payload(Tool::Codex, &source_event_type, &payload));
    let sequence =
        sequence_for_payload(Tool::Codex, &source_event_type, &payload, Some(byte_offset));
    let source_event_id =
        source_event_id_for_payload(Tool::Codex, &source_event_type, &payload, sequence);

    Ok(EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: timestamp_for_payload(&payload)
            .unwrap_or(OffsetDateTime::now_utc().format(&Rfc3339)?),
        tool: Tool::Codex,
        tool_version: tool_version_for_payload(&payload),
        session_id: session_id.clone(),
        filename_session_id: sanitize_session_id(&session_id),
        turn_id: turn_id_for_payload(&payload),
        message_id: message_id_for_payload(&payload),
        project_root: project_root_for_payload(&payload),
        cwd: cwd_for_payload(&payload),
        source: Source::ExecJson,
        source_event_type,
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

pub(crate) fn codex_stream_event_name(payload: &Value) -> String {
    string_pointer(payload, "/type")
        .or_else(|| string_pointer(payload, "/method"))
        .or_else(|| string_pointer(payload, "/params/type"))
        .or_else(|| string_pointer(payload, "/payload/type"))
        .unwrap_or_else(|| "codex.unknown".to_string())
}

pub(crate) fn codex_stream_session_id(
    source_path: &Path,
    payload: &Value,
    previous_session_id: Option<&str>,
) -> String {
    string_pointer(payload, "/session_id")
        .or_else(|| string_pointer(payload, "/thread_id"))
        .or_else(|| string_pointer(payload, "/threadId"))
        .or_else(|| string_pointer(payload, "/thread/id"))
        .or_else(|| string_pointer(payload, "/payload/session_id"))
        .or_else(|| string_pointer(payload, "/payload/thread_id"))
        .or_else(|| string_pointer(payload, "/payload/thread/id"))
        .or_else(|| string_pointer(payload, "/params/session_id"))
        .or_else(|| string_pointer(payload, "/params/sessionId"))
        .or_else(|| string_pointer(payload, "/params/thread_id"))
        .or_else(|| string_pointer(payload, "/params/threadId"))
        .or_else(|| string_pointer(payload, "/params/thread/id"))
        .or_else(|| {
            let event_name = codex_stream_event_name(payload);
            if matches!(event_name.as_str(), "thread.started" | "thread/started") {
                string_pointer(payload, "/id").or_else(|| string_pointer(payload, "/params/id"))
            } else {
                None
            }
        })
        .or_else(|| previous_session_id.map(str::to_string))
        .or_else(|| session_id_from_source_path(source_path))
        .unwrap_or_else(|| source_path_fallback_session_id(source_path))
}

pub(crate) fn codex_session_meta_id(tool: Tool, payload: &Value) -> Option<String> {
    if tool == Tool::Codex && payload.get("type").and_then(Value::as_str) == Some("session_meta") {
        return string_pointer(payload, "/payload/id");
    }
    None
}
