//! Reading stored events and sessions back out: session pages, session lists,
//! event-pointer lookups, and the latest captured event per tool.

use crate::{
    corroborate_text, normalize_date_or_duration, open_index, raw_envelope_for_pointer,
    redact_json_value, redact_text, session_raw_file, Error, EventOptions, EventPointer, Result,
    SessionOptions, SessionPage, SessionSummary, StoredEvent, Tool, MAX_CONTEXT_EVENTS_PER_SIDE,
    MAX_SESSION_LIMIT,
};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params_from_iter, OptionalExtension};
use std::path::Path;
use std::str::FromStr;

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
                compaction_count: row.get(8)?,
                raw_file: row.get(9)?,
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
    Ok(sessions)
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
