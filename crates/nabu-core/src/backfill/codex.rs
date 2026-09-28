//! Codex stream-format parsing.

use super::*;

pub(crate) fn parse_codex_stream_source(source_path: &Path) -> Result<ParsedBackfillSource> {
    match source_path.extension().and_then(|value| value.to_str()) {
        Some("jsonl") => parse_codex_stream_jsonl(source_path),
        Some("json") => parse_codex_stream_json(source_path),
        _ => Ok(ParsedBackfillSource {
            events: Vec::new(),
            last_session_id: None,
            extent: ParsedExtent::WholeFile,
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
        extent: ParsedExtent::WholeFile,
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
                extent: ParsedExtent::WholeFile,
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
        extent: ParsedExtent::WholeFile,
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

/// Thread id of a Codex rollout file, from its name
/// `rollout-<timestamp>-<thread id>.jsonl`. `None` for any other file.
/// One rollout file holds exactly one thread, so this id is the session of
/// every line in it, including lines that carry a parent thread's
/// `session_id` (spawned subagents repeat the parent id in `session_meta`).
pub(crate) fn codex_rollout_thread_id(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_prefix("rollout-")?.strip_suffix(".jsonl")?;
    let thread_id = stem.get(stem.len().checked_sub(36)?..)?;
    looks_like_uuid(thread_id).then(|| thread_id.to_string())
}

/// A Codex rollout file named by a hook payload, bound to the thread the hook
/// reported. Only [`codex_rollouts_for_hook`] constructs it: the path is
/// absolute and its file name carries that thread id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexRollout {
    path: PathBuf,
    thread_id: String,
}

impl CodexRollout {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// Rebuild a rollout reference from a path and thread id given on the
    /// command line. Applies the same checks as the hook parser.
    pub fn parse(
        path: PathBuf,
        thread_id: &str,
    ) -> std::result::Result<Self, CodexRolloutRejection> {
        codex_rollout_from_parts("path", path, thread_id)
    }
}

/// Why a rollout path from a hook payload was not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexRolloutRejection {
    /// The payload field or argument that named the path.
    pub field: &'static str,
    pub path: PathBuf,
    pub reason: CodexRolloutRejectReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodexRolloutRejectReason {
    RelativePath,
    NotARolloutFileName,
    ThreadMismatch { expected: String, found: String },
}

impl std::fmt::Display for CodexRolloutRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path = self.path.display();
        let field = self.field;
        match &self.reason {
            CodexRolloutRejectReason::RelativePath => {
                write!(formatter, "{field} {path} is not an absolute path")
            }
            CodexRolloutRejectReason::NotARolloutFileName => write!(
                formatter,
                "{field} {path} is not named rollout-<timestamp>-<thread id>.jsonl"
            ),
            CodexRolloutRejectReason::ThreadMismatch { expected, found } => write!(
                formatter,
                "{field} {path} belongs to thread {found}, but the hook reported thread {expected}"
            ),
        }
    }
}

/// One rollout path found in a hook payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodexRolloutClaim {
    Accepted(CodexRollout),
    Rejected(CodexRolloutRejection),
}

/// The rollout files a Codex hook payload asks nabu to reconcile.
///
/// - `Stop` (end of a turn): the session rollout in `transcript_path`, bound
///   to `session_id`.
/// - `SubagentStop`: the subagent rollout in `agent_transcript_path`, bound
///   to `agent_id`.
///
/// Every other hook, and a payload without these fields, yields no claim.
/// Pure: it reads no file.
pub fn codex_rollouts_for_hook(payload: &Value) -> Vec<CodexRolloutClaim> {
    let (field, thread_field) = match string_pointer(payload, "/hook_event_name").as_deref() {
        Some("Stop") => ("transcript_path", "session_id"),
        Some("SubagentStop") => ("agent_transcript_path", "agent_id"),
        _ => return Vec::new(),
    };
    let (Some(path), Some(thread_id)) = (
        string_pointer(payload, &format!("/{field}")),
        string_pointer(payload, &format!("/{thread_field}")),
    ) else {
        return Vec::new();
    };
    let claim = match codex_rollout_from_parts(field, PathBuf::from(path), &thread_id) {
        Ok(rollout) => CodexRolloutClaim::Accepted(rollout),
        Err(rejection) => CodexRolloutClaim::Rejected(rejection),
    };
    vec![claim]
}

fn codex_rollout_from_parts(
    field: &'static str,
    path: PathBuf,
    thread_id: &str,
) -> std::result::Result<CodexRollout, CodexRolloutRejection> {
    let reject = |path: PathBuf, reason| CodexRolloutRejection {
        field,
        path,
        reason,
    };
    if !path.is_absolute() {
        return Err(reject(path, CodexRolloutRejectReason::RelativePath));
    }
    let Some(found) = codex_rollout_thread_id(&path) else {
        return Err(reject(path, CodexRolloutRejectReason::NotARolloutFileName));
    };
    if found != thread_id {
        return Err(reject(
            path,
            CodexRolloutRejectReason::ThreadMismatch {
                expected: thread_id.to_string(),
                found,
            },
        ));
    }
    Ok(CodexRollout {
        path,
        thread_id: found,
    })
}

/// Result of one rollout reconcile pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexRolloutReconcile {
    /// New complete lines were read and appended; `appended_events` counts
    /// only lines not already captured. `discontinuities` counts truncation
    /// or rotation markers written for this file.
    Imported {
        appended_events: usize,
        discontinuities: usize,
    },
    /// The rollout file no longer exists. Nothing changed.
    SourceMissing,
}

/// Import the lines of a Codex rollout file that no earlier pass consumed.
///
/// Uses the same per-file checkpoint as `nabu backfill`, so each call reads
/// only new complete lines, and a later backfill skips them. A last line that
/// Codex is still writing stays for the next call. Appends are deduped, so
/// concurrent calls on one file append each line once.
///
/// Effects: creates the nabu home layout when it is missing, appends to the
/// canonical raw file of the rollout's thread, and writes the checkpoint row.
/// It does not index. On an error, lines appended
/// before the failure stay and the checkpoint keeps its previous value; the
/// next call reads those lines again and dedupe drops them.
pub fn reconcile_codex_rollout(
    home: &Path,
    rollout: &CodexRollout,
) -> Result<CodexRolloutReconcile> {
    init_home(home)?;
    match backfill_source_file(
        home,
        Tool::Codex,
        rollout.path(),
        &BackfillParseContext::default(),
    ) {
        Ok(report) => Ok(CodexRolloutReconcile::Imported {
            appended_events: report.appended_events,
            discontinuities: report.discontinuities,
        }),
        Err(Error::Io { path, source })
            if path == rollout.path() && source.kind() == std::io::ErrorKind::NotFound =>
        {
            Ok(CodexRolloutReconcile::SourceMissing)
        }
        Err(error) => Err(error),
    }
}
