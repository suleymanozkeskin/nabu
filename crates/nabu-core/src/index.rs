//! Indexing pipeline: scanning raw capture files into the SQLite index, writing
//! the events row and its derived FTS/tool/compaction rows, and recalculating
//! per-session counts. The cfg(semantic) vector-write pipeline stays in lib.rs
//! and moves to the `semantic` module in the final phase.

use crate::{
    chmod, compaction_state_for, embed_index_if_available_with_progress,
    ensure_semantic_vector_schema, extract_refs, file_paths_for_payload, hash_line, init_home,
    insert_vector_unit_rows, message_text_for_document, open_index,
    raw_index_checkpoint_is_current, resolved_payload_for_envelope, role_for,
    search_document_for_event, source_file_metadata, string_field, tool_status_for,
    write_raw_index_checkpoint, CanonicalType, EmbeddingIndexProgress, Error, EventEnvelope,
    IndexOptions, IndexReport, Result, SearchDocument, Tool,
};
use fs2::FileExt;
use rusqlite::{params, Connection};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Outcome of [`index_once_single_flight`]: a pass ran (with its merged report)
/// or another pass already held the index lock and this attempt was a no-op.
/// Skips are expected under per-event triggering — the holding pass drains the
/// same delta, so a skipped trigger loses no coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SingleFlightOutcome {
    Ran(IndexReport),
    Skipped,
}

/// Upper bound on re-passes within a single lock hold. Each pass drains the
/// delta present when it started; an append landing mid-pass is caught by one
/// more pass. The cap prevents livelock under continuous concurrent writes —
/// the next capture hook re-triggers and resumes draining.
const SINGLE_FLIGHT_MAX_PASSES: usize = 8;

/// Run the incremental index under a non-blocking exclusive lock so at most one
/// pass touches a home at a time. Returns [`SingleFlightOutcome::Skipped`]
/// without indexing when another pass holds the lock. While holding the lock it
/// re-runs the pass until one indexes zero new events, so a delta appended after
/// the scan started but before the lock is released is still drained.
///
/// This is the entry point the capture hook spawns: it is idempotent, bounded,
/// and safe to invoke concurrently from every hook event.
pub fn index_once_single_flight(home: &Path, options: IndexOptions) -> Result<SingleFlightOutcome> {
    init_home(home)?;
    let lock_path = home.join("index").join(".index.lock");
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

    match lock_file.try_lock_exclusive() {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            return Ok(SingleFlightOutcome::Skipped);
        }
        Err(source) => {
            return Err(Error::Io {
                path: lock_path,
                source,
            });
        }
    }

    let mut indexed_events = 0usize;
    let mut pass_result = Ok(());
    for _ in 0..SINGLE_FLIGHT_MAX_PASSES {
        match index_once_with_options(home, options) {
            Ok(report) => {
                indexed_events += report.indexed_events;
                if report.indexed_events == 0 {
                    break;
                }
            }
            Err(error) => {
                pass_result = Err(error);
                break;
            }
        }
    }

    // Release the lock unconditionally, even when a pass failed.
    let unlock_result = FileExt::unlock(&lock_file).map_err(|source| Error::Io {
        path: lock_path,
        source,
    });

    pass_result?;
    unlock_result?;
    Ok(SingleFlightOutcome::Ran(IndexReport { indexed_events }))
}

pub fn index_once(home: &Path) -> Result<IndexReport> {
    index_once_with_progress(home, |_| {})
}

pub fn index_once_with_progress<F>(home: &Path, progress: F) -> Result<IndexReport>
where
    F: FnMut(EmbeddingIndexProgress),
{
    index_once_with_options_and_progress(home, IndexOptions::default(), progress)
}

pub fn index_once_with_options(home: &Path, options: IndexOptions) -> Result<IndexReport> {
    index_once_with_options_and_progress(home, options, |_| {})
}

pub fn index_once_with_options_and_progress<F>(
    home: &Path,
    options: IndexOptions,
    progress: F,
) -> Result<IndexReport>
where
    F: FnMut(EmbeddingIndexProgress),
{
    init_home(home)?;
    let db_path = home.join("index").join("harness.db");
    let mut conn = open_index(&db_path)?;
    ensure_semantic_vector_schema(&conn, &db_path)?;
    let tx = conn.transaction().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;

    let mut indexed_events = 0usize;
    let mut touched_sessions = HashSet::new();
    for tool in Tool::all() {
        let raw_dir = home.join("raw").join(tool.as_str());
        if !raw_dir.exists() {
            continue;
        }

        let entries = fs::read_dir(&raw_dir).map_err(|source| Error::Io {
            path: raw_dir.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| Error::Io {
                path: raw_dir.clone(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            let source_meta = source_file_metadata(&path)?;
            if raw_index_checkpoint_is_current(&tx, &db_path, tool, &path, &source_meta)? {
                continue;
            }

            let raw_report = index_raw_file(&tx, tool, &path)?;
            indexed_events += raw_report.indexed_events;
            touched_sessions.extend(
                raw_report
                    .touched_sessions
                    .iter()
                    .map(|session_id| (tool, session_id.clone())),
            );
            write_raw_index_checkpoint(&tx, &db_path, tool, &path, source_meta, raw_report)?;
        }
    }

    recalculate_touched_session_counts(&tx, &db_path, &touched_sessions)?;
    tx.commit().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;

    if options.embed {
        embed_index_if_available_with_progress(home, progress)?;
    }

    Ok(IndexReport { indexed_events })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawIndexFileReport {
    pub(crate) indexed_events: usize,
    pub(crate) bytes_read: u64,
    pub(crate) last_line_hash: Option<String>,
    pub(crate) touched_sessions: HashSet<String>,
}

fn index_raw_file(conn: &Connection, tool: Tool, path: &Path) -> Result<RawIndexFileReport> {
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut raw_line = 0i64;
    let mut raw_offset = 0u64;
    let mut indexed = 0usize;
    let mut last_hash = None;
    let mut touched_sessions = HashSet::new();

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        raw_line += 1;
        last_hash = Some(hash_line(line.trim_end()));
        let parsed: EventEnvelope = serde_json::from_str(line.trim_end())?;
        parsed.validate()?;
        if parsed.tool != tool {
            return Err(Error::Validation(format!(
                "raw file {} contains tool {}",
                path.display(),
                parsed.tool
            )));
        }

        if insert_indexed_event(conn, path, raw_line, raw_offset as i64, &parsed)? {
            indexed += 1;
            touched_sessions.insert(parsed.session_id.clone());
        }
        raw_offset += bytes as u64;
    }

    Ok(RawIndexFileReport {
        indexed_events: indexed,
        bytes_read: raw_offset,
        last_line_hash: last_hash,
        touched_sessions,
    })
}

fn insert_indexed_event(
    conn: &Connection,
    path: &Path,
    raw_line: i64,
    fallback_raw_offset: i64,
    envelope: &EventEnvelope,
) -> Result<bool> {
    let payload = resolved_payload_for_envelope(path, envelope)?;
    let search_document = search_document_for_event(envelope.canonical_type, &payload);
    let searchable_text = search_document.render();
    let raw_file = path.display().to_string();
    let raw_offset = envelope.raw_offset.unwrap_or(fallback_raw_offset);
    let payload_json: Option<String> = None;

    conn.execute(
        "INSERT INTO sessions(
           tool,
           session_id,
           filename_session_id,
           project_root,
           cwd,
           started_at,
           updated_at,
           raw_file
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, ?7)
         ON CONFLICT(tool, session_id) DO UPDATE SET
           started_at = CASE
             WHEN sessions.started_at IS NULL OR sessions.started_at > excluded.started_at
             THEN excluded.started_at
             ELSE sessions.started_at
           END,
           updated_at = CASE
             WHEN sessions.updated_at IS NULL OR sessions.updated_at < excluded.updated_at
             THEN excluded.updated_at
             ELSE sessions.updated_at
           END,
           project_root = COALESCE(sessions.project_root, excluded.project_root),
           cwd = COALESCE(sessions.cwd, excluded.cwd)",
        (
            envelope.tool.as_str(),
            &envelope.session_id,
            &envelope.filename_session_id,
            envelope.project_root.as_deref(),
            envelope.cwd.as_deref(),
            &envelope.captured_at,
            &raw_file,
        ),
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    let inserted =
        conn.execute(
            "INSERT OR IGNORE INTO events(
              tool,
              session_id,
              dedupe_key,
              schema_version,
              captured_at,
              tool_version,
              turn_id,
              message_id,
              project_root,
              cwd,
              source,
              source_event_type,
              source_event_id,
              canonical_type,
              sequence,
              raw_file,
              raw_line,
              raw_offset,
              payload_json,
              payload_ref,
              searchable_text,
              compaction_state
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                    ?16, ?17, ?18, ?19, ?20, ?21, ?22)",
            params![
                envelope.tool.as_str(),
                &envelope.session_id,
                &envelope.dedupe_key,
                envelope.schema_version as i64,
                &envelope.captured_at,
                envelope.tool_version.as_deref(),
                envelope.turn_id.as_deref(),
                envelope.message_id.as_deref(),
                envelope.project_root.as_deref(),
                envelope.cwd.as_deref(),
                envelope.source.as_str(),
                &envelope.source_event_type,
                envelope.source_event_id.as_deref(),
                envelope.canonical_type.as_str(),
                envelope.sequence,
                &raw_file,
                raw_line,
                raw_offset,
                payload_json.as_deref(),
                envelope.payload_ref.as_deref(),
                &searchable_text,
                compaction_state_for(envelope.canonical_type),
            ],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })? == 1;

    if !inserted {
        return Ok(false);
    }

    let event_id = conn.last_insert_rowid();
    insert_derived_rows(
        conn,
        path,
        event_id,
        envelope,
        &payload,
        raw_line,
        raw_offset,
        &search_document,
    )?;
    Ok(true)
}

// Row-insert helper threading the parsed event plus its raw-file coordinates;
// the arguments are distinct positional facts, not a struct in disguise.
#[allow(clippy::too_many_arguments)]
fn insert_derived_rows(
    conn: &Connection,
    path: &Path,
    event_id: i64,
    envelope: &EventEnvelope,
    payload: &Value,
    raw_line: i64,
    raw_offset: i64,
    search_document: &SearchDocument,
) -> Result<()> {
    if let Some(role) = role_for(envelope.canonical_type) {
        conn.execute(
            "INSERT INTO messages(event_id, tool, session_id, role, text, is_delta, sequence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            (
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                role,
                message_text_for_document(envelope.canonical_type, search_document),
                i64::from(envelope.canonical_type == CanonicalType::AssistantDelta),
                envelope.sequence,
            ),
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }

    if matches!(
        envelope.canonical_type,
        CanonicalType::ToolCall | CanonicalType::ToolResult
    ) {
        conn.execute(
            "INSERT INTO tool_events(event_id, tool, session_id, tool_name, command, status, input_text, output_text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            (
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                string_field(payload, "tool_name"),
                string_field(payload, "command"),
                tool_status_for(envelope.canonical_type),
                string_field(payload, "input"),
                string_field(payload, "output"),
            ),
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }

    insert_event_file_rows(conn, path, event_id, envelope, payload)?;
    insert_event_ref_rows(conn, path, event_id, search_document)?;

    if matches!(
        envelope.canonical_type,
        CanonicalType::CompactionBefore | CanonicalType::CompactionAfter
    ) {
        conn.execute(
            "INSERT INTO compactions(event_id, tool, session_id, trigger, raw_file, raw_line, raw_offset, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            (
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                string_field(payload, "trigger"),
                path.display().to_string(),
                raw_line,
                raw_offset,
                &envelope.captured_at,
            ),
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }

    conn.execute(
        "INSERT INTO events_fts(rowid, user_text, assistant_text, tool_intent, tool_output, metadata_text, tool, session_id, canonical_type, raw_file, raw_line, raw_offset)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        (
            event_id,
            &search_document.user_text,
            &search_document.assistant_text,
            &search_document.tool_intent,
            &search_document.tool_output,
            &search_document.metadata_text,
            envelope.tool.as_str(),
            &envelope.session_id,
            envelope.canonical_type.as_str(),
            path.display().to_string(),
            raw_line,
            raw_offset,
        ),
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    insert_vector_unit_rows(
        conn,
        path,
        event_id,
        envelope,
        raw_line,
        raw_offset,
        search_document,
    )?;

    Ok(())
}

fn recalculate_touched_session_counts(
    conn: &Connection,
    db_path: &Path,
    touched_sessions: &HashSet<(Tool, String)>,
) -> Result<()> {
    for (tool, session_id) in touched_sessions {
        conn.execute(
            "UPDATE sessions
             SET event_count = (
               SELECT COUNT(*) FROM events
               WHERE events.tool = ?1 AND events.session_id = ?2
             ),
             message_count = (
               SELECT COUNT(*) FROM messages
               WHERE messages.tool = ?1 AND messages.session_id = ?2
             ),
             tool_event_count = (
               SELECT COUNT(*) FROM tool_events
               WHERE tool_events.tool = ?1 AND tool_events.session_id = ?2
             ),
             compaction_count = (
               SELECT COUNT(*) FROM compactions
               WHERE compactions.tool = ?1 AND compactions.session_id = ?2
             )
             WHERE sessions.tool = ?1 AND sessions.session_id = ?2;",
            params![tool.as_str(), session_id],
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

pub(crate) fn recalculate_all_session_counts(conn: &Connection, db_path: &Path) -> Result<()> {
    conn.execute_batch(
        "UPDATE sessions
         SET event_count = (
           SELECT COUNT(*) FROM events
           WHERE events.tool = sessions.tool AND events.session_id = sessions.session_id
         ),
         message_count = (
           SELECT COUNT(*) FROM messages
           WHERE messages.tool = sessions.tool AND messages.session_id = sessions.session_id
         ),
         tool_event_count = (
           SELECT COUNT(*) FROM tool_events
           WHERE tool_events.tool = sessions.tool AND tool_events.session_id = sessions.session_id
         ),
         compaction_count = (
           SELECT COUNT(*) FROM compactions
           WHERE compactions.tool = sessions.tool AND compactions.session_id = sessions.session_id
         );",
    )
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })
}

// Provenance refs (PR references, commit SHAs) extracted from the rendered
// searchable text. The same extractor backfills pre-existing events in
// `db::ensure_event_refs_schema`, so both paths produce identical rows.
fn insert_event_ref_rows(
    conn: &Connection,
    path: &Path,
    event_id: i64,
    search_document: &SearchDocument,
) -> Result<()> {
    for reference in extract_refs(&search_document.render()) {
        conn.execute(
            "INSERT OR IGNORE INTO event_refs(event_id, ref_kind, ref_value)
             VALUES (?1, ?2, ?3)",
            params![event_id, reference.kind.as_str(), reference.value],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn insert_event_file_rows(
    conn: &Connection,
    path: &Path,
    event_id: i64,
    envelope: &EventEnvelope,
    payload: &Value,
) -> Result<()> {
    let relationship = match envelope.canonical_type {
        CanonicalType::FileChanged => "edited",
        _ => "mentioned",
    };
    for file_path in file_paths_for_payload(payload) {
        conn.execute(
            "INSERT OR IGNORE INTO files(path) VALUES (?1)",
            [&file_path],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let file_id: i64 = conn
            .query_row(
                "SELECT id FROM files WHERE path = ?1",
                [&file_path],
                |row| row.get(0),
            )
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        conn.execute(
            "INSERT OR IGNORE INTO event_files(event_id, file_id, relationship)
             VALUES (?1, ?2, ?3)",
            params![event_id, file_id, relationship],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}
