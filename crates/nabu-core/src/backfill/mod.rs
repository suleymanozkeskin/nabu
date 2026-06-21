//! Backfill: import historical sessions from native tool stores into the
//! capture store, with per-source checkpoints and per-format parsers.

use crate::*;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod claude;
use claude::*;
mod checkpoint;
pub(crate) use checkpoint::*;
mod codex;
use codex::*;
mod opencode;
pub(crate) use opencode::*;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[cfg(test)]
pub fn backfill_since(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
) -> Result<BackfillReport> {
    backfill_since_with_progress(home, selection, source_root, since, |_| {})
}

/// A source file discovered during the native-store scan can be deleted or
/// rotated by the live tool before backfill reads it (`os error 2`). Such a
/// vanished file must be skipped, never treated as fatal — one missing session
/// must not abort a backfill over the whole store.
fn is_vanished_source(error: &Error) -> bool {
    matches!(error, Error::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound)
}

pub fn backfill_since_with_progress<F>(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
    progress: F,
) -> Result<BackfillReport>
where
    F: Fn(BackfillProgress) + Sync,
{
    init_home(home)?;
    let since_threshold = since.map(parse_since_threshold).transpose()?;
    let mut report = empty_backfill_report();

    let tools: Vec<Tool> = match selection {
        Some(tool) => vec![tool],
        None => Tool::all().to_vec(),
    };

    for tool in tools {
        let tool_root = backfill_tool_root(source_root, tool);
        if !tool_root.exists() {
            continue;
        }
        let mut files = Vec::new();
        collect_backfill_files(tool, &tool_root, &mut files)?;
        files.sort();
        let parse_context = backfill_parse_context(tool, &tool_root, &files)?;
        let files = filter_backfill_files_by_since(files, since_threshold)?;
        let total_files = files.len();
        progress(BackfillProgress {
            operation: "backfill.start".to_string(),
            tool,
            source_root: tool_root.display().to_string(),
            processed_files: 0,
            total_files,
            source_path: None,
        });
        let processed_files = AtomicUsize::new(0);
        let tool_report = files
            .par_iter()
            .map(|file| {
                // `since` is already filtered upstream by
                // filter_backfill_files_by_since. Here we only guard against a
                // source file that vanishes between enumeration and read: skip
                // it (fail-open) instead of aborting the whole backfill.
                let outcome = backfill_source_file(home, tool, file, &parse_context)
                    .map(source_backfill_report_to_backfill_report);
                let vanished = matches!(&outcome, Err(error) if is_vanished_source(error));
                let result = if vanished {
                    Ok(empty_backfill_report())
                } else {
                    outcome
                };
                let processed = processed_files.fetch_add(1, Ordering::SeqCst) + 1;
                progress(BackfillProgress {
                    operation: if vanished {
                        "backfill.skip_missing".to_string()
                    } else {
                        "backfill.file".to_string()
                    },
                    tool,
                    source_root: tool_root.display().to_string(),
                    processed_files: processed,
                    total_files,
                    source_path: Some(file.display().to_string()),
                });
                result
            })
            .try_reduce(empty_backfill_report, |mut left, right| {
                merge_backfill_report(&mut left, right);
                Ok(left)
            })?;
        merge_backfill_report(&mut report, tool_report);
    }

    detect_deleted_sources(home, source_root, &mut report)?;

    Ok(report)
}

#[cfg(test)]
pub fn backfill_dry_run(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
) -> Result<BackfillDryRunReport> {
    backfill_dry_run_with_progress(home, selection, source_root, since, |_| {})
}

pub fn backfill_dry_run_with_progress<F>(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
    progress: F,
) -> Result<BackfillDryRunReport>
where
    F: Fn(BackfillProgress) + Sync,
{
    init_home(home)?;
    let since_threshold = since.map(parse_since_threshold).transpose()?;
    let tools: Vec<Tool> = match selection {
        Some(tool) => vec![tool],
        None => Tool::all().to_vec(),
    };
    let mut sessions = Vec::new();

    for tool in tools {
        let tool_root = backfill_tool_root(source_root, tool);
        if !tool_root.exists() {
            continue;
        }
        let mut files = Vec::new();
        collect_backfill_files(tool, &tool_root, &mut files)?;
        files.sort();
        let parse_context = backfill_parse_context(tool, &tool_root, &files)?;
        let files = filter_backfill_files_by_since(files, since_threshold)?;
        let total_files = files.len();
        progress(BackfillProgress {
            operation: "backfill.dry_run.start".to_string(),
            tool,
            source_root: tool_root.display().to_string(),
            processed_files: 0,
            total_files,
            source_path: None,
        });
        let processed_files = AtomicUsize::new(0);
        let file_reports: Vec<Result<Vec<BackfillCoverageSession>>> = files
            .par_iter()
            .map(|file| {
                // `since` filtered upstream; tolerate a file that vanishes
                // between enumeration and read (fail-open skip).
                let outcome = backfill_dry_run_file(home, tool, file, &parse_context);
                let vanished = matches!(&outcome, Err(error) if is_vanished_source(error));
                let result = if vanished { Ok(Vec::new()) } else { outcome };
                let processed = processed_files.fetch_add(1, Ordering::SeqCst) + 1;
                progress(BackfillProgress {
                    operation: if vanished {
                        "backfill.dry_run.skip_missing".to_string()
                    } else {
                        "backfill.dry_run.file".to_string()
                    },
                    tool,
                    source_root: tool_root.display().to_string(),
                    processed_files: processed,
                    total_files,
                    source_path: Some(file.display().to_string()),
                });
                result
            })
            .collect();
        for file_report in file_reports {
            sessions.extend(file_report?);
        }
    }

    let source_files = sessions
        .iter()
        .map(|session| session.source_path.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let on_disk_events = sessions.iter().map(|session| session.on_disk).sum();
    let captured_events = sessions.iter().map(|session| session.captured).sum();
    let missing_events = sessions.iter().map(|session| session.missing).sum();
    let partial_sessions = sessions.iter().filter(|session| session.partial).count();

    Ok(BackfillDryRunReport {
        source_files,
        on_disk_events,
        captured_events,
        missing_events,
        partial_sessions,
        sessions,
    })
}

fn empty_backfill_report() -> BackfillReport {
    BackfillReport {
        source_files: 0,
        appended_events: 0,
        checkpoint_files: 0,
        discontinuities: 0,
    }
}

fn source_backfill_report_to_backfill_report(report: SourceBackfillReport) -> BackfillReport {
    BackfillReport {
        source_files: 1,
        appended_events: report.appended_events,
        checkpoint_files: 1,
        discontinuities: report.discontinuities,
    }
}

fn merge_backfill_report(left: &mut BackfillReport, right: BackfillReport) {
    left.source_files = left.source_files.saturating_add(right.source_files);
    left.appended_events = left.appended_events.saturating_add(right.appended_events);
    left.checkpoint_files = left.checkpoint_files.saturating_add(right.checkpoint_files);
    left.discontinuities = left.discontinuities.saturating_add(right.discontinuities);
}

fn filter_backfill_files_by_since(
    files: Vec<PathBuf>,
    since_threshold: Option<SystemTime>,
) -> Result<Vec<PathBuf>> {
    if since_threshold.is_none() {
        return Ok(files);
    }
    files
        .into_iter()
        .filter_map(
            |file| match should_skip_source_file(&file, since_threshold) {
                Ok(true) => None,
                Ok(false) => Some(Ok(file)),
                // A file that vanishes between enumeration and this mtime stat
                // is skipped, not fatal — same fail-open guarantee the read
                // path gives for the no-`since` case.
                Err(error) if is_vanished_source(&error) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

fn backfill_dry_run_file(
    home: &Path,
    tool: Tool,
    file: &Path,
    parse_context: &BackfillParseContext,
) -> Result<Vec<BackfillCoverageSession>> {
    let parsed = parse_backfill_source(tool, file, 0, parse_context)?;
    let mut by_session: BTreeMap<String, Vec<EventEnvelope>> = BTreeMap::new();
    for event in parsed.events {
        by_session
            .entry(event.session_id.clone())
            .or_default()
            .push(event);
    }
    let mut sessions = Vec::new();
    for (session_id, events) in by_session {
        let (captured, would_import) =
            captured_count_and_import_preview(home, tool, &session_id, &events)?;
        let on_disk = events.len();
        let missing = on_disk.saturating_sub(captured);
        sessions.push(BackfillCoverageSession {
            tool,
            session_id,
            source_path: file.display().to_string(),
            on_disk,
            captured,
            missing,
            partial: missing > 0,
            would_import,
        });
    }
    Ok(sessions)
}

fn backfill_parse_context(
    tool: Tool,
    tool_root: &Path,
    files: &[PathBuf],
) -> Result<BackfillParseContext> {
    if tool != Tool::Opencode {
        return Ok(BackfillParseContext::default());
    }

    let message_session_ids = opencode_message_session_ids(tool_root)?;
    let mut context = OpenCodeBackfillContext::default();

    for file in files
        .iter()
        .filter(|file| opencode_storage_kind(file) == Some("session"))
    {
        let content = fs::read_to_string(file).map_err(|source| Error::Io {
            path: file.to_path_buf(),
            source,
        })?;
        let Ok(payload) = serde_json::from_str::<Value>(&content) else {
            continue;
        };
        let Some(session_id) = opencode_metadata_session_id(&payload, &message_session_ids) else {
            continue;
        };
        context
            .metadata_session_ids
            .insert(file.display().to_string(), session_id.clone());
        if let Some(worktree) = opencode_worktree_for_payload(&payload) {
            context
                .worktree_by_session_id
                .entry(session_id)
                .or_insert(worktree);
        }
    }

    Ok(BackfillParseContext {
        opencode: Some(context),
    })
}

fn should_skip_source_file(path: &Path, since_threshold: Option<SystemTime>) -> Result<bool> {
    let Some(since_threshold) = since_threshold else {
        return Ok(false);
    };
    let modified = fs::metadata(path)
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?
        .modified()
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(modified < since_threshold)
}

fn parse_since_threshold(value: &str) -> Result<SystemTime> {
    parse_date_or_duration_threshold(value, "since")
}

pub(crate) fn normalize_date_or_duration(value: &str, field_name: &str) -> Result<String> {
    let timestamp = parse_date_or_duration_threshold(value, field_name)?;
    Ok(OffsetDateTime::from(timestamp).format(&Rfc3339)?)
}

fn parse_date_or_duration_threshold(value: &str, field_name: &str) -> Result<SystemTime> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::Validation(format!("{field_name} must not be empty")));
    }
    if trimmed.len() > 1 {
        let (number, suffix) = trimmed.split_at(trimmed.len() - 1);
        if matches!(suffix, "d" | "h" | "m" | "s") && number.chars().all(|ch| ch.is_ascii_digit()) {
            let amount = number.parse::<u64>().map_err(|_| {
                Error::Validation(format!("invalid {field_name} duration: {value}"))
            })?;
            let seconds = match suffix {
                "d" => amount.saturating_mul(86_400),
                "h" => amount.saturating_mul(3_600),
                "m" => amount.saturating_mul(60),
                "s" => amount,
                _ => unreachable!(),
            };
            return SystemTime::now()
                .checked_sub(StdDuration::from_secs(seconds))
                .ok_or_else(|| {
                    Error::Validation(format!("invalid {field_name} duration: {value}"))
                });
        }
    }
    if trimmed.len() == 10 && trimmed.as_bytes()[4] == b'-' && trimmed.as_bytes()[7] == b'-' {
        let year = trimmed[0..4]
            .parse::<i32>()
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let month = trimmed[5..7]
            .parse::<u8>()
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let day = trimmed[8..10]
            .parse::<u8>()
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let month = Month::try_from(month)
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let date = Date::from_calendar_date(year, month, day)
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let timestamp = date
            .with_hms(0, 0, 0)
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?
            .assume_utc();
        return Ok(SystemTime::from(timestamp));
    }
    let timestamp = OffsetDateTime::parse(trimmed, &Rfc3339)
        .map_err(|_| Error::Validation(format!("invalid {field_name} value: {value}")))?;
    Ok(SystemTime::from(timestamp))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceBackfillReport {
    appended_events: usize,
    discontinuities: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct BackfillParseContext {
    opencode: Option<OpenCodeBackfillContext>,
}

#[derive(Debug, Clone, Default)]
struct OpenCodeBackfillContext {
    metadata_session_ids: HashMap<String, String>,
    worktree_by_session_id: HashMap<String, String>,
}

fn backfill_source_file(
    home: &Path,
    tool: Tool,
    source_path: &Path,
    parse_context: &BackfillParseContext,
) -> Result<SourceBackfillReport> {
    let source_meta = source_file_metadata(source_path)?;
    let source_len = source_meta.size;
    let source_kind = source_kind_for(tool, source_path).to_string();
    let previous_checkpoint = load_checkpoint(home, tool, &source_kind, source_path)?;
    let mut start_offset = 0u64;
    let mut appended_events = 0usize;
    let mut discontinuities = 0usize;
    let now = OffsetDateTime::now_utc().format(&Rfc3339)?;

    if let Some(previous) = previous_checkpoint.as_ref() {
        if previous.byte_offset > source_len {
            append_discontinuity(
                home,
                tool,
                &previous.session_id,
                "source.truncated",
                source_path,
                previous.byte_offset,
                source_len,
            )?;
            discontinuities += 1;
        } else {
            start_offset = previous.byte_offset;
            if previous.source_identity.as_deref() != source_meta.identity.as_deref()
                || start_offset > 0 && !checkpoint_matches_source(source_path, previous)?
            {
                append_discontinuity(
                    home,
                    tool,
                    &previous.session_id,
                    "source.rotated",
                    source_path,
                    previous.byte_offset,
                    source_len,
                )?;
                discontinuities += 1;
                start_offset = 0;
            }
        }
    }

    if start_offset >= source_len && discontinuities == 0 {
        let checkpoint = SourceCheckpoint {
            source_tool: tool,
            source_kind,
            source_path: source_path.display().to_string(),
            source_identity: source_meta.identity,
            session_id: previous_checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.session_id.clone())
                .or_else(|| session_id_from_source_path(source_path))
                .unwrap_or_else(|| source_path_fallback_session_id(source_path)),
            byte_offset: source_len,
            source_size: source_len,
            source_mtime: source_meta.mtime,
            last_line_hash: last_line_hash(source_path)?,
            last_successful_import_timestamp: now,
        };
        write_checkpoint(home, &checkpoint)?;
        return Ok(SourceBackfillReport {
            appended_events: 0,
            discontinuities: 0,
        });
    }

    let parsed = parse_backfill_source(tool, source_path, start_offset, parse_context)?;
    appended_events += append_prepared_events(home, parsed.events)?
        .into_iter()
        .filter(|report| report.appended)
        .count();

    let checkpoint = SourceCheckpoint {
        source_tool: tool,
        source_kind,
        source_path: source_path.display().to_string(),
        source_identity: source_meta.identity,
        session_id: parsed
            .last_session_id
            .or_else(|| previous_checkpoint.map(|checkpoint| checkpoint.session_id))
            .or_else(|| session_id_from_source_path(source_path))
            .unwrap_or_else(|| source_path_fallback_session_id(source_path)),
        byte_offset: source_len,
        source_size: source_len,
        source_mtime: source_meta.mtime,
        last_line_hash: last_line_hash(source_path)?,
        last_successful_import_timestamp: now,
    };
    write_checkpoint(home, &checkpoint)?;

    Ok(SourceBackfillReport {
        appended_events,
        discontinuities,
    })
}

#[derive(Debug)]
pub(crate) struct ParsedBackfillSource {
    pub(crate) events: Vec<EventEnvelope>,
    last_session_id: Option<String>,
}

fn parse_backfill_source(
    tool: Tool,
    source_path: &Path,
    start_offset: u64,
    parse_context: &BackfillParseContext,
) -> Result<ParsedBackfillSource> {
    match source_path.extension().and_then(|value| value.to_str()) {
        Some("jsonl") => parse_backfill_jsonl(tool, source_path, start_offset, parse_context),
        Some("json") => parse_backfill_json(tool, source_path, parse_context),
        _ => Ok(ParsedBackfillSource {
            events: Vec::new(),
            last_session_id: None,
        }),
    }
}

pub(crate) fn parse_ingest_file_source(
    tool: Tool,
    source: Source,
    source_path: &Path,
) -> Result<ParsedBackfillSource> {
    match (tool, source) {
        (Tool::Codex, Source::ExecJson) | (Tool::Codex, Source::AppServer) => {
            parse_codex_stream_source(source_path)
        }
        (_, Source::ExecJson | Source::AppServer) => Err(Error::Validation(format!(
            "{} source is only supported for codex ingest",
            source.as_str()
        ))),
        _ => parse_backfill_source(tool, source_path, 0, &BackfillParseContext::default()),
    }
}

fn parse_backfill_jsonl(
    tool: Tool,
    source_path: &Path,
    start_offset: u64,
    parse_context: &BackfillParseContext,
) -> Result<ParsedBackfillSource> {
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
        if offset <= start_offset || line.trim().is_empty() {
            continue;
        }
        let payload = match serde_json::from_str(line.trim_end()) {
            Ok(payload) => payload,
            Err(error) => malformed_native_payload(source_path, line_start, line.trim_end(), error),
        };
        let event =
            envelope_from_backfill_payload(tool, source_path, line_start, payload, parse_context)?;
        last_session_id = Some(event.session_id.clone());
        events.push(event);
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

fn parse_backfill_json(
    tool: Tool,
    source_path: &Path,
    parse_context: &BackfillParseContext,
) -> Result<ParsedBackfillSource> {
    let content = fs::read_to_string(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let payload: Value = match serde_json::from_str(&content) {
        Ok(payload) => payload,
        Err(error) => {
            let event = envelope_from_backfill_payload(
                tool,
                source_path,
                0,
                malformed_native_payload(source_path, 0, &content, error),
                parse_context,
            )?;
            return Ok(ParsedBackfillSource {
                last_session_id: Some(event.session_id.clone()),
                events: vec![event],
            });
        }
    };
    let mut events = Vec::new();
    let mut last_session_id = None;

    match payload {
        Value::Array(values) => {
            for (index, value) in values.into_iter().enumerate() {
                let event = envelope_from_backfill_payload(
                    tool,
                    source_path,
                    index as u64,
                    value,
                    parse_context,
                )?;
                last_session_id = Some(event.session_id.clone());
                events.push(event);
            }
        }
        Value::Object(mut map) => {
            if let Some(Value::Array(messages)) = map.remove("messages") {
                let inherited_session_id = map
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                for (index, mut message) in messages.into_iter().enumerate() {
                    if let (Some(session_id), Value::Object(message_map)) =
                        (inherited_session_id.as_ref(), &mut message)
                    {
                        message_map
                            .entry("session_id".to_string())
                            .or_insert_with(|| Value::String(session_id.clone()));
                    }
                    let event = envelope_from_backfill_payload(
                        tool,
                        source_path,
                        index as u64,
                        message,
                        parse_context,
                    )?;
                    last_session_id = Some(event.session_id.clone());
                    events.push(event);
                }
            } else {
                let event = envelope_from_backfill_payload(
                    tool,
                    source_path,
                    0,
                    Value::Object(map),
                    parse_context,
                )?;
                last_session_id = Some(event.session_id.clone());
                events.push(event);
            }
        }
        _ => {}
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

pub(crate) fn envelope_from_backfill_payload(
    tool: Tool,
    source_path: &Path,
    byte_offset: u64,
    payload: Value,
    parse_context: &BackfillParseContext,
) -> Result<EventEnvelope> {
    let session_id = backfill_session_id(tool, source_path, &payload, parse_context);
    let source_event_type = backfill_event_name_for_payload(tool, source_path, &payload);
    let canonical_type = payload
        .get("canonical_type")
        .and_then(Value::as_str)
        .map(CanonicalType::from_str)
        .transpose()?
        .unwrap_or_else(|| canonical_type_for_backfill_payload(tool, &source_event_type, &payload));
    let sequence = sequence_for_payload(tool, &source_event_type, &payload, Some(byte_offset));
    let source_event_id = source_event_id_for_payload(tool, &source_event_type, &payload, sequence);
    Ok(EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: timestamp_for_payload(&payload)
            .unwrap_or(OffsetDateTime::now_utc().format(&Rfc3339)?),
        tool,
        tool_version: payload
            .get("tool_version")
            .and_then(Value::as_str)
            .or_else(|| payload.pointer("/version").and_then(Value::as_str))
            .or_else(|| {
                payload
                    .pointer("/payload/cli_version")
                    .and_then(Value::as_str)
            })
            .map(str::to_string),
        session_id: session_id.clone(),
        filename_session_id: sanitize_session_id(&session_id),
        turn_id: string_pointer(&payload, "/turn_id")
            .or_else(|| string_pointer(&payload, "/payload/turn_id"))
            .or_else(|| string_pointer(&payload, "/parentUuid"))
            .or_else(|| string_pointer(&payload, "/parentID")),
        message_id: payload
            .get("message_id")
            .or_else(|| payload.pointer("/payload/message_id"))
            .or_else(|| payload.pointer("/message/id"))
            .or_else(|| payload.pointer("/messageID"))
            .or_else(|| payload.pointer("/uuid"))
            .or_else(|| payload.pointer("/id"))
            .and_then(Value::as_str)
            .map(str::to_string),
        project_root: project_root_for_backfill_payload(tool, &payload, &session_id, parse_context),
        cwd: cwd_for_backfill_payload(tool, &payload, &session_id, parse_context),
        source: Source::Backfill,
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

fn tool_version_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/tool_version")
        .or_else(|| string_pointer(payload, "/version"))
        .or_else(|| string_pointer(payload, "/payload/tool_version"))
        .or_else(|| string_pointer(payload, "/payload/cli_version"))
        .or_else(|| string_pointer(payload, "/payload/version"))
        .or_else(|| string_pointer(payload, "/params/tool_version"))
        .or_else(|| string_pointer(payload, "/params/cli_version"))
        .or_else(|| string_pointer(payload, "/params/version"))
}

fn turn_id_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/turn_id")
        .or_else(|| string_pointer(payload, "/turnId"))
        .or_else(|| string_pointer(payload, "/turn/id"))
        .or_else(|| string_pointer(payload, "/payload/turn_id"))
        .or_else(|| string_pointer(payload, "/payload/turnId"))
        .or_else(|| string_pointer(payload, "/payload/turn/id"))
        .or_else(|| string_pointer(payload, "/params/turn_id"))
        .or_else(|| string_pointer(payload, "/params/turnId"))
        .or_else(|| string_pointer(payload, "/params/turn/id"))
        .or_else(|| string_pointer(payload, "/params/item/turn_id"))
        .or_else(|| string_pointer(payload, "/item/turn_id"))
        .or_else(|| string_pointer(payload, "/parentUuid"))
        .or_else(|| string_pointer(payload, "/parentID"))
}

pub(crate) fn message_id_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/message_id")
        .or_else(|| string_pointer(payload, "/messageId"))
        .or_else(|| string_pointer(payload, "/message/id"))
        .or_else(|| string_pointer(payload, "/messageID"))
        .or_else(|| string_pointer(payload, "/payload/message_id"))
        .or_else(|| string_pointer(payload, "/payload/messageId"))
        .or_else(|| string_pointer(payload, "/payload/message/id"))
        .or_else(|| string_pointer(payload, "/payload/item/message_id"))
        .or_else(|| string_pointer(payload, "/payload/item/messageId"))
        .or_else(|| string_pointer(payload, "/payload/item/id"))
        .or_else(|| string_pointer(payload, "/params/message_id"))
        .or_else(|| string_pointer(payload, "/params/messageId"))
        .or_else(|| string_pointer(payload, "/params/message/id"))
        .or_else(|| string_pointer(payload, "/params/item/message_id"))
        .or_else(|| string_pointer(payload, "/params/item/messageId"))
        .or_else(|| string_pointer(payload, "/params/item/id"))
        .or_else(|| string_pointer(payload, "/item/message_id"))
        .or_else(|| string_pointer(payload, "/item/messageId"))
        .or_else(|| string_pointer(payload, "/item/id"))
        .or_else(|| string_pointer(payload, "/uuid"))
        .or_else(|| string_pointer(payload, "/id"))
}

fn project_root_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/project_root")
        .or_else(|| string_pointer(payload, "/payload/project_root"))
        .or_else(|| string_pointer(payload, "/params/project_root"))
        .or_else(|| string_pointer(payload, "/path/root"))
        .or_else(|| string_pointer(payload, "/params/path/root"))
        .or_else(|| opencode_worktree_for_payload(payload))
}

fn cwd_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/cwd")
        .or_else(|| string_pointer(payload, "/payload/cwd"))
        .or_else(|| string_pointer(payload, "/params/cwd"))
        .or_else(|| string_pointer(payload, "/path/cwd"))
        .or_else(|| string_pointer(payload, "/params/path/cwd"))
        .or_else(|| opencode_worktree_for_payload(payload))
}

fn project_root_for_backfill_payload(
    tool: Tool,
    payload: &Value,
    session_id: &str,
    parse_context: &BackfillParseContext,
) -> Option<String> {
    project_root_for_payload(payload).or_else(|| {
        (tool == Tool::Opencode)
            .then(|| {
                parse_context
                    .opencode
                    .as_ref()?
                    .worktree_by_session_id
                    .get(session_id)
                    .cloned()
            })
            .flatten()
    })
}

fn cwd_for_backfill_payload(
    tool: Tool,
    payload: &Value,
    session_id: &str,
    parse_context: &BackfillParseContext,
) -> Option<String> {
    cwd_for_payload(payload)
        .or_else(|| project_root_for_backfill_payload(tool, payload, session_id, parse_context))
}

fn backfill_session_id(
    tool: Tool,
    source_path: &Path,
    payload: &Value,
    parse_context: &BackfillParseContext,
) -> String {
    if tool == Tool::Opencode && opencode_storage_kind(source_path) == Some("session") {
        if let Some(session_id) = parse_context
            .opencode
            .as_ref()
            .and_then(|context| {
                context
                    .metadata_session_ids
                    .get(&source_path.display().to_string())
            })
            .cloned()
            .or_else(|| opencode_direct_session_id(payload))
        {
            return session_id;
        }
    }
    string_pointer(payload, "/session_id")
        .or_else(|| string_pointer(payload, "/payload/session_id"))
        .or_else(|| string_pointer(payload, "/sessionId"))
        .or_else(|| string_pointer(payload, "/sessionID"))
        .or_else(|| codex_session_meta_id(tool, payload))
        .or_else(|| session_id_from_source_path(source_path))
        .unwrap_or_else(|| source_path_fallback_session_id(source_path))
}

fn backfill_event_name_for_payload(tool: Tool, source_path: &Path, payload: &Value) -> String {
    if let Ok(name) = hook_event_name(payload) {
        return name.to_string();
    }
    match tool {
        Tool::Claude => {
            let kind = string_pointer(payload, "/type").unwrap_or_else(|| "unknown".to_string());
            format!("claude.{kind}")
        }
        Tool::Codex => {
            string_pointer(payload, "/type").unwrap_or_else(|| "codex.unknown".to_string())
        }
        Tool::Opencode => {
            if opencode_storage_kind(source_path) == Some("session") {
                "session.created".to_string()
            } else if opencode_storage_kind(source_path) == Some("part") {
                match opencode_part_type(payload).as_deref() {
                    Some("tool") | Some("tool-call") => "tool.execute.before".to_string(),
                    Some("tool-result") | Some("tool_result") => "tool.execute.after".to_string(),
                    Some(kind) => kind.to_string(),
                    None => "message.part.updated".to_string(),
                }
            } else {
                "message.updated".to_string()
            }
        }
    }
}

fn canonical_type_for_backfill_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
) -> CanonicalType {
    if payload.get("parse_error").is_some() || payload.get("raw_line").is_some() {
        return CanonicalType::Error;
    }
    if payload.get("hook_event_name").is_some() || payload.get("event").is_some() {
        return canonical_type_for_payload(tool, source_event_type, payload);
    }
    match tool {
        Tool::Claude => canonical_type_for_claude_native(payload),
        Tool::Codex => canonical_type_for_payload(tool, source_event_type, payload),
        Tool::Opencode => canonical_type_for_opencode_native(source_event_type, payload),
    }
}

fn timestamp_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/captured_at")
        .or_else(|| string_pointer(payload, "/timestamp"))
        .or_else(|| string_pointer(payload, "/payload/timestamp"))
        .or_else(|| string_pointer(payload, "/params/timestamp"))
        .or_else(|| string_pointer(payload, "/created_at"))
        .or_else(|| string_pointer(payload, "/payload/created_at"))
        .or_else(|| string_pointer(payload, "/params/created_at"))
        .or_else(|| {
            payload
                .pointer("/time/created")
                .and_then(Value::as_i64)
                .and_then(timestamp_millis_to_rfc3339)
        })
        .or_else(|| {
            payload
                .pointer("/time/created")
                .and_then(Value::as_u64)
                .and_then(|value| i64::try_from(value).ok())
                .and_then(timestamp_millis_to_rfc3339)
        })
        .or_else(|| {
            payload
                .pointer("/params/time/created")
                .and_then(Value::as_i64)
                .and_then(timestamp_millis_to_rfc3339)
        })
        .or_else(|| {
            payload
                .pointer("/params/time/created")
                .and_then(Value::as_u64)
                .and_then(|value| i64::try_from(value).ok())
                .and_then(timestamp_millis_to_rfc3339)
        })
}

fn timestamp_millis_to_rfc3339(value: i64) -> Option<String> {
    OffsetDateTime::from_unix_timestamp(value.div_euclid(1000))
        .ok()
        .and_then(|timestamp| timestamp.format(&Rfc3339).ok())
}

fn malformed_native_payload(
    source_path: &Path,
    byte_offset: u64,
    raw_line: &str,
    error: serde_json::Error,
) -> Value {
    serde_json::json!({
        "type": "parse_error",
        "source_path": source_path.display().to_string(),
        "byte_offset": byte_offset,
        "parse_error": error.to_string(),
        "raw_line": raw_line
    })
}

fn session_id_from_source_path(source_path: &Path) -> Option<String> {
    let stem = source_path.file_stem()?.to_str()?.trim();
    if stem.is_empty() {
        return None;
    }

    if stem.len() >= 36 {
        let suffix = &stem[stem.len() - 36..];
        if looks_like_uuid(suffix) {
            return Some(suffix.to_string());
        }
    }

    Some(stem.to_string())
}

fn source_path_fallback_session_id(source_path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source_path.display().to_string().as_bytes());
    let hash = hex::encode(hasher.finalize());
    format!("source-{}", &hash[..16])
}

fn looks_like_uuid(value: &str) -> bool {
    value.len() == 36
        && value.char_indices().all(|(index, character)| {
            matches!(index, 8 | 13 | 18 | 23) && character == '-'
                || !matches!(index, 8 | 13 | 18 | 23) && character.is_ascii_hexdigit()
        })
}

pub(crate) fn append_prepared_events(
    home: &Path,
    events: Vec<EventEnvelope>,
) -> Result<Vec<AppendReport>> {
    let mut grouped = BTreeMap::<PathBuf, Vec<EventEnvelope>>::new();
    for event in events {
        let raw_file = canonical_raw_path(home, event.tool, &event.session_id);
        grouped.entry(raw_file).or_default().push(event);
    }

    let mut reports = Vec::new();
    for (raw_file, events) in grouped {
        reports.extend(append_prepared_events_to_raw_file(home, &raw_file, events)?);
    }
    Ok(reports)
}

fn captured_count_and_import_preview(
    home: &Path,
    tool: Tool,
    session_id: &str,
    events: &[EventEnvelope],
) -> Result<(usize, Vec<BackfillImportPreview>)> {
    let raw_file = canonical_raw_path(home, tool, session_id);
    let existing = existing_dedupe_events_for_raw_file(home, &raw_file)?;
    let mut captured = 0usize;
    let mut would_import = Vec::new();
    for event in events {
        let key = dedupe_key(DedupeParts {
            tool: event.tool,
            session_id: &event.session_id,
            canonical_type: event.canonical_type,
            source_event_id: event.source_event_id.as_deref(),
            sequence: event.sequence,
            payload: &event.payload,
        })?;
        if existing.contains_key(&key) {
            captured += 1;
        } else {
            would_import.push(BackfillImportPreview {
                canonical_type: event.canonical_type.as_str().to_string(),
                source_event_type: event.source_event_type.clone(),
                source_event_id: event.source_event_id.clone(),
                sequence: event.sequence,
                captured_at: event.captured_at.clone(),
            });
        }
    }
    Ok((captured, would_import))
}

fn existing_dedupe_events_for_raw_file(
    home: &Path,
    raw_file: &Path,
) -> Result<HashMap<String, ExistingRawEvent>> {
    let sidecar = DedupeSidecarFiles::for_raw_file(home, raw_file);
    match load_full_dedupe_sidecar_events(raw_file, &sidecar) {
        Ok(Some(events)) => Ok(events),
        Ok(None) | Err(_) => Ok(read_raw_dedupe_snapshot(raw_file)?.events),
    }
}

fn append_prepared_events_to_raw_file(
    home: &Path,
    raw_file: &Path,
    events: Vec<EventEnvelope>,
) -> Result<Vec<AppendReport>> {
    if events.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(parent) = raw_file.parent() {
        create_dir_0700(parent)?;
    }
    let lock_path = lock_path_for_raw_file(raw_file);
    let lock_file = OpenOptions::new()
        .create(true)
        // Lock sentinel: content is never written, so do not truncate.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|source| Error::Io {
            path: lock_path.clone(),
            source,
        })?;
    chmod(&lock_path, 0o600)?;
    lock_file.lock_exclusive().map_err(|source| Error::Io {
        path: lock_path.clone(),
        source,
    })?;

    let append_result = append_envelopes_locked(home, raw_file, events);
    let unlock_result = FileExt::unlock(&lock_file).map_err(|source| Error::Io {
        path: lock_path,
        source,
    });

    match (append_result, unlock_result) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

pub(crate) fn append_prepared_event(home: &Path, event: EventEnvelope) -> Result<AppendReport> {
    let raw_file = canonical_raw_path(home, event.tool, &event.session_id);
    if let Some(parent) = raw_file.parent() {
        create_dir_0700(parent)?;
    }
    let lock_path = lock_path_for_raw_file(&raw_file);
    let lock_file = OpenOptions::new()
        .create(true)
        // Lock sentinel: content is never written, so do not truncate.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|source| Error::Io {
            path: lock_path.clone(),
            source,
        })?;
    chmod(&lock_path, 0o600)?;
    lock_file.lock_exclusive().map_err(|source| Error::Io {
        path: lock_path.clone(),
        source,
    })?;

    let append_result = append_envelope_locked(home, &raw_file, event);
    let unlock_result = FileExt::unlock(&lock_file).map_err(|source| Error::Io {
        path: lock_path,
        source,
    });

    match (append_result, unlock_result) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

fn append_discontinuity(
    home: &Path,
    tool: Tool,
    session_id: &str,
    source_event_type: &str,
    source_path: &Path,
    previous_offset: u64,
    current_len: u64,
) -> Result<()> {
    let payload = serde_json::json!({
        "session_id": session_id,
        "hook_event_name": source_event_type,
        "source_path": source_path.display().to_string(),
        "previous_byte_offset": previous_offset,
        "current_byte_len": current_len,
        "reason": source_event_type,
        "text": "backfill source discontinuity"
    });
    append_prepared_event(
        home,
        EventEnvelope {
            schema_version: SCHEMA_VERSION,
            captured_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            tool,
            tool_version: None,
            session_id: session_id.to_string(),
            filename_session_id: sanitize_session_id(session_id),
            turn_id: None,
            message_id: None,
            project_root: None,
            cwd: None,
            source: Source::Backfill,
            source_event_type: source_event_type.to_string(),
            canonical_type: CanonicalType::SourceDiscontinuity,
            source_event_id: Some(format!(
                "{}:{}:{}:{}",
                source_event_type,
                source_path.display(),
                previous_offset,
                current_len
            )),
            dedupe_key: String::new(),
            sequence: None,
            raw_file: None,
            raw_offset: None,
            payload,
            payload_ref: None,
        },
    )?;
    Ok(())
}

fn backfill_tool_root(source_root: &Path, tool: Tool) -> PathBuf {
    let candidate = match tool {
        Tool::Codex => source_root.join("codex"),
        Tool::Claude => source_root.join("claude-code"),
        Tool::Opencode => source_root.join("opencode"),
    };
    if candidate.exists() {
        candidate
    } else {
        source_root.to_path_buf()
    }
}

fn collect_backfill_files(tool: Tool, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs::read_dir(dir).map_err(|source| Error::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_backfill_files(tool, &path, files)?;
            continue;
        }
        if is_backfill_candidate(tool, &path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_backfill_candidate(tool: Tool, path: &Path) -> bool {
    match tool {
        Tool::Claude => is_claude_transcript_file(path),
        Tool::Codex | Tool::Opencode => matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("json") | Some("jsonl")
        ),
    }
}

fn is_claude_transcript_file(path: &Path) -> bool {
    if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
        return false;
    }
    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return false;
    };
    looks_like_uuid(stem) || stem == "transcript" || stem.starts_with("agent-")
}
