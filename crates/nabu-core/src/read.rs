//! Reading stored events and sessions back out: session pages, session lists,
//! event-pointer lookups, and the latest captured event per tool.

use crate::{
    corroborate_text, normalize_date_or_duration, open_index, raw_envelope_for_pointer,
    redact_json_value, redact_text, session_raw_file, Error, EventOptions, EventPointer, FileTouch,
    Result, SessionOptions, SessionPage, SessionSummary, StoredEvent, Tool, ToolUsage,
    MAX_CONTEXT_EVENTS_PER_SIDE, MAX_SESSION_LIMIT, SESSION_PROMPT_SNIPPET_CHARS,
    SESSION_TOP_FILES, SESSION_TOP_TOOLS,
};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params_from_iter, Connection, OptionalExtension};
use std::path::Path;
use std::str::FromStr;

/// Truncate `text` to at most `max_chars` characters on a char boundary,
/// appending a single-character ellipsis when truncation occurred. Trailing
/// whitespace before the ellipsis is trimmed so the snippet reads cleanly.
fn truncate_snippet(text: &str, max_chars: usize) -> String {
    let mut boundary = None;
    for (count, (byte_idx, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            boundary = Some(byte_idx);
            break;
        }
    }
    match boundary {
        None => text.to_string(),
        Some(byte_idx) => {
            let head = text[..byte_idx].trim_end();
            format!("{head}\u{2026}")
        }
    }
}

pub fn session_events(home: &Path, tool: Tool, session_id: &str) -> Result<Vec<StoredEvent>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT
               tool,
               session_id,
               canonical_type,
               captured_at,
               searchable_text,
               raw_file,
               raw_line,
               raw_offset,
               cwd,
               project_root
             FROM events
             WHERE tool = ?1 AND session_id = ?2
             ORDER BY raw_line, raw_offset",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let rows = statement
        .query_map((tool.as_str(), session_id), |row| {
            let tool_text: String = row.get(0)?;
            Ok(StoredEvent {
                tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                session_id: row.get(1)?,
                canonical_type: row.get(2)?,
                timestamp: row.get(3)?,
                text: row.get(4)?,
                raw_file: row.get(5)?,
                raw_line: row.get(6)?,
                raw_offset: row.get(7)?,
                corroboration: None,
                cwd: row.get(8)?,
                project_root: row.get(9)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut events = Vec::new();
    for row in rows {
        events.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    Ok(events)
}

pub fn list_sessions(
    home: &Path,
    tool: Option<Tool>,
    cwd: Option<&str>,
    since: Option<&str>,
    limit: usize,
) -> Result<Vec<SessionSummary>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut sql = String::from(
        "SELECT
           tool,
           session_id,
           project_root,
           cwd,
           started_at,
           updated_at,
           event_count,
           message_count,
           tool_event_count,
           compaction_count,
           raw_file
         FROM sessions
         WHERE 1 = 1",
    );
    let mut params = Vec::new();

    if let Some(tool) = tool {
        sql.push_str(" AND tool = ?");
        params.push(SqlValue::Text(tool.as_str().to_string()));
    }
    if let Some(cwd) = cwd {
        sql.push_str(" AND cwd = ?");
        params.push(SqlValue::Text(cwd.to_string()));
    }
    if let Some(since) = since {
        sql.push_str(" AND updated_at >= ?");
        params.push(SqlValue::Text(normalize_date_or_duration(since, "since")?));
    }
    sql.push_str(" ORDER BY updated_at DESC LIMIT ?");
    params.push(SqlValue::Integer(limit.clamp(1, 100) as i64));

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(0)?;
            Ok(SessionSummary {
                tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                session_id: row.get(1)?,
                project_root: row.get(2)?,
                cwd: row.get(3)?,
                started_at: row.get(4)?,
                updated_at: row.get(5)?,
                event_count: row.get(6)?,
                message_count: row.get(7)?,
                tool_event_count: row.get(8)?,
                compaction_count: row.get(9)?,
                raw_file: row.get(10)?,
                first_user_prompt: None,
                last_canonical_type: None,
                top_tools: Vec::new(),
                top_files: Vec::new(),
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut sessions = Vec::new();
    for row in rows {
        sessions.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    drop(statement);

    // Enrich each base row with triage metadata derived from already-indexed
    // tables (messages, events, tool_events, event_files). Query-time, so no
    // migration or backfill: existing indexes upgrade with zero reindex. The
    // result set is bounded by `limit` (<=100), so the per-session fan-out is
    // small and uses the existing session-scoped indexes.
    for session in &mut sessions {
        let tool_text = session.tool.as_str();
        session.first_user_prompt =
            first_user_prompt(&conn, &db_path, tool_text, &session.session_id)?;
        session.last_canonical_type =
            last_canonical_type(&conn, &db_path, tool_text, &session.session_id)?;
        session.top_tools = top_tools(&conn, &db_path, tool_text, &session.session_id)?;
        session.top_files = top_files(&conn, &db_path, tool_text, &session.session_id)?;
    }

    Ok(sessions)
}

/// First user-message text for the session, truncated for triage. Ordered by
/// `sequence` (NULLs last) then event id so the earliest prompt wins
/// deterministically.
fn first_user_prompt(
    conn: &Connection,
    db_path: &Path,
    tool: &str,
    session_id: &str,
) -> Result<Option<String>> {
    conn.query_row(
        "SELECT text
           FROM messages
          WHERE tool = ?1 AND session_id = ?2 AND role = 'user' AND is_delta = 0
          ORDER BY sequence IS NULL, sequence, id
          LIMIT 1",
        (tool, session_id),
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })
    .map(|text| {
        text.map(|text| truncate_snippet(text.trim(), SESSION_PROMPT_SNIPPET_CHARS))
            .filter(|snippet| !snippet.is_empty())
    })
}

/// Canonical type of the session's last event, ordered by capture time then
/// raw position so the newest event wins deterministically.
fn last_canonical_type(
    conn: &Connection,
    db_path: &Path,
    tool: &str,
    session_id: &str,
) -> Result<Option<String>> {
    conn.query_row(
        "SELECT canonical_type
           FROM events
          WHERE tool = ?1 AND session_id = ?2
          ORDER BY captured_at DESC, raw_line DESC, raw_offset DESC
          LIMIT 1",
        (tool, session_id),
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })
}

/// Top tool names invoked in the session, by descending call count. Ties break
/// alphabetically for determinism. Rows with a NULL `tool_name` are ignored.
fn top_tools(
    conn: &Connection,
    db_path: &Path,
    tool: &str,
    session_id: &str,
) -> Result<Vec<ToolUsage>> {
    let mut statement = conn
        .prepare(
            "SELECT tool_name, COUNT(*) AS uses
               FROM tool_events
              WHERE tool = ?1 AND session_id = ?2 AND tool_name IS NOT NULL
              GROUP BY tool_name
              ORDER BY uses DESC, tool_name ASC
              LIMIT ?3",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    let rows = statement
        .query_map((tool, session_id, SESSION_TOP_TOOLS as i64), |row| {
            Ok(ToolUsage {
                tool_name: row.get(0)?,
                count: row.get(1)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    collect_rows(rows, db_path)
}

/// Top files edited in the session, by descending edit count. Counts only the
/// `edited` relationship (file.changed events). Ties break alphabetically.
fn top_files(
    conn: &Connection,
    db_path: &Path,
    tool: &str,
    session_id: &str,
) -> Result<Vec<FileTouch>> {
    let mut statement = conn
        .prepare(
            "SELECT files.path, COUNT(*) AS edits
               FROM event_files
               JOIN events ON events.id = event_files.event_id
               JOIN files ON files.id = event_files.file_id
              WHERE events.tool = ?1
                AND events.session_id = ?2
                AND event_files.relationship = 'edited'
              GROUP BY files.path
              ORDER BY edits DESC, files.path ASC
              LIMIT ?3",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    let rows = statement
        .query_map((tool, session_id, SESSION_TOP_FILES as i64), |row| {
            Ok(FileTouch {
                path: row.get(0)?,
                edits: row.get(1)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    collect_rows(rows, db_path)
}

fn collect_rows<T>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>>,
    db_path: &Path,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?);
    }
    Ok(out)
}

pub fn get_session_page(
    home: &Path,
    tool: Tool,
    session_id: &str,
    options: SessionOptions,
) -> Result<SessionPage> {
    let raw_file = session_raw_file(home, tool, session_id)?;
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let limit = options.limit_events.clamp(1, MAX_SESSION_LIMIT);
    let before = options.before.clamp(0, MAX_CONTEXT_EVENTS_PER_SIDE);
    let after = options.after.clamp(0, MAX_CONTEXT_EVENTS_PER_SIDE);
    let mut sql = String::from(
        "SELECT
               tool,
               session_id,
               canonical_type,
               captured_at,
               searchable_text,
               raw_file,
               raw_line,
               raw_offset,
               cwd,
               project_root
             FROM events
             WHERE tool = ? AND session_id = ?",
    );
    let mut params = vec![
        SqlValue::Text(tool.as_str().to_string()),
        SqlValue::Text(session_id.to_string()),
    ];
    let mode;
    let mut limit_for_query = None;
    if let Some(around_raw_line) = options.around_raw_line {
        mode = "window".to_string();
        let start_line = around_raw_line.saturating_sub(before as i64).max(1);
        let end_line = around_raw_line.saturating_add(after as i64);
        sql.push_str(" AND raw_line >= ? AND raw_line <= ?");
        params.push(SqlValue::Integer(start_line));
        params.push(SqlValue::Integer(end_line));
    } else {
        mode = "page".to_string();
        let after_raw_line = options.after_raw_line.unwrap_or(0);
        sql.push_str(" AND raw_line > ?");
        params.push(SqlValue::Integer(after_raw_line));
        limit_for_query = Some(limit.saturating_add(1));
    }
    if !options.include_deltas {
        sql.push_str(" AND canonical_type != 'assistant.delta'");
    }
    if let Some(canonical_type) = options.canonical_type.as_deref() {
        sql.push_str(" AND canonical_type = ?");
        params.push(SqlValue::Text(canonical_type.to_string()));
    }
    sql.push_str(" ORDER BY raw_line, raw_offset");
    if let Some(limit_for_query) = limit_for_query {
        sql.push_str(" LIMIT ?");
        params.push(SqlValue::Integer(limit_for_query as i64));
    }

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(0)?;
            Ok(StoredEvent {
                tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                session_id: row.get(1)?,
                canonical_type: row.get(2)?,
                timestamp: row.get(3)?,
                text: row.get(4)?,
                raw_file: row.get(5)?,
                raw_line: row.get(6)?,
                raw_offset: row.get(7)?,
                corroboration: None,
                cwd: row.get(8)?,
                project_root: row.get(9)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut events = Vec::new();
    for row in rows {
        let mut event = row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
        if options.redact {
            event.text = redact_text(&event.text);
        }
        if options.corroborate {
            event.corroboration = Some(corroborate_text(
                event.cwd.as_deref(),
                event.project_root.as_deref(),
                &event.text,
            ));
        }
        events.push(event);
    }
    let truncated = options.around_raw_line.is_none() && events.len() > limit;
    if truncated {
        events.truncate(limit);
    }
    let next_after_raw_line = if truncated {
        events.last().map(|event| event.raw_line)
    } else {
        None
    };

    Ok(SessionPage {
        tool,
        session_id: session_id.to_string(),
        raw_file,
        events,
        truncated,
        next_after_raw_line,
        mode,
        limit_events_applied: if options.around_raw_line.is_none() {
            Some(limit)
        } else {
            None
        },
        after_raw_line: if options.around_raw_line.is_none() {
            Some(options.after_raw_line.unwrap_or(0))
        } else {
            None
        },
        around_raw_line: options.around_raw_line,
        before_applied: options.around_raw_line.map(|_| before),
        after_applied: options.around_raw_line.map(|_| after),
        include_deltas: options.include_deltas,
        canonical_type: options.canonical_type,
    })
}

pub fn get_event_by_pointer(
    home: &Path,
    tool: Tool,
    session_id: &str,
    raw_line: Option<i64>,
    raw_offset: Option<i64>,
    redact: bool,
) -> Result<EventPointer> {
    get_event_by_pointer_with_options(
        home,
        tool,
        session_id,
        raw_line,
        raw_offset,
        EventOptions {
            redact,
            corroborate: false,
        },
    )
}

pub fn get_event_by_pointer_with_options(
    home: &Path,
    tool: Tool,
    session_id: &str,
    raw_line: Option<i64>,
    raw_offset: Option<i64>,
    options: EventOptions,
) -> Result<EventPointer> {
    if raw_line.is_none() && raw_offset.is_none() {
        return Err(Error::Validation(
            "raw_line or raw_offset is required".to_string(),
        ));
    }
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut sql = String::from(
        "SELECT raw_file, raw_line, raw_offset, searchable_text, cwd, project_root
         FROM events
         WHERE tool = ? AND session_id = ?",
    );
    let mut params = vec![
        SqlValue::Text(tool.as_str().to_string()),
        SqlValue::Text(session_id.to_string()),
    ];
    if let Some(raw_line) = raw_line {
        sql.push_str(" AND raw_line = ?");
        params.push(SqlValue::Integer(raw_line));
    }
    if let Some(raw_offset) = raw_offset {
        sql.push_str(" AND raw_offset = ?");
        params.push(SqlValue::Integer(raw_offset));
    }
    sql.push_str(" ORDER BY raw_line, raw_offset LIMIT 1");

    let pointer = conn
        .query_row(&sql, params_from_iter(params), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })
        .optional()
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?
        .ok_or_else(|| {
            Error::Validation(format!(
                "event not found for {}:{}",
                tool.as_str(),
                session_id
            ))
        })?;

    let (raw_file, raw_line, raw_offset, searchable_text, cwd, project_root) = pointer;
    let mut envelope = raw_envelope_for_pointer(&raw_file, raw_line, raw_offset)?;
    let mut searchable_text = searchable_text;
    if options.redact {
        envelope.payload = redact_json_value(envelope.payload);
        searchable_text = redact_text(&searchable_text);
    }
    let corroboration = options
        .corroborate
        .then(|| corroborate_text(cwd.as_deref(), project_root.as_deref(), &searchable_text));

    Ok(EventPointer {
        envelope,
        searchable_text,
        raw_file,
        raw_line,
        raw_offset,
        corroboration,
    })
}

pub fn latest_event(home: &Path, tool: Tool) -> Result<Option<StoredEvent>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT
               tool,
               session_id,
               canonical_type,
               captured_at,
               searchable_text,
               raw_file,
               raw_line,
               raw_offset,
               cwd,
               project_root
             FROM events
             WHERE tool = ?1
             ORDER BY captured_at DESC, id DESC
             LIMIT 1",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let mut rows = statement
        .query([tool.as_str()])
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let Some(row) = rows.next().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?
    else {
        return Ok(None);
    };
    let tool_text: String = row.get(0).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    Ok(Some(StoredEvent {
        tool: Tool::from_str(&tool_text)
            .map_err(|_| Error::Validation("invalid tool".to_string()))?,
        session_id: row.get(1).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        canonical_type: row.get(2).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        timestamp: row.get(3).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        text: row.get(4).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        raw_file: row.get(5).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        raw_line: row.get(6).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        raw_offset: row.get(7).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        corroboration: None,
        cwd: row.get(8).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        project_root: row.get(9).map_err(|source| Error::Sqlite {
            path: db_path,
            source,
        })?,
    }))
}
