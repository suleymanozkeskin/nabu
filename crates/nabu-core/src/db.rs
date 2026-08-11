//! SQLite database lifecycle: schema initialization and migration, connection
//! opening, FTS (re)build, and the single process-global sqlite-vec extension
//! registration (hard constraint: registered in exactly one place).

use crate::provenance::extract_refs;
use crate::{
    chmod, payload_for_raw_pointer, search_document_for_event, set_if_exists, CanonicalType, Error,
    Result, SQLITE_SCHEMA,
};
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;
use std::str::FromStr;

pub(crate) const EVENTS_FTS_SCHEMA: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
  user_text,
  assistant_text,
  tool_intent,
  tool_output,
  metadata_text,
  tool UNINDEXED,
  session_id UNINDEXED,
  canonical_type UNINDEXED,
  raw_file UNINDEXED,
  raw_line UNINDEXED,
  raw_offset UNINDEXED,
  content=''
);
"#;
#[cfg(feature = "semantic")]
const SEMANTIC_VECTOR_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS vector_units (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
  session_id TEXT NOT NULL,
  unit_kind TEXT NOT NULL CHECK (unit_kind IN ('user_text', 'assistant_text', 'tool_intent', 'metadata_text')),
  unit_index INTEGER NOT NULL DEFAULT 0,
  text_hash TEXT NOT NULL,
  raw_file TEXT NOT NULL,
  raw_line INTEGER,
  raw_offset INTEGER,
  created_at TEXT NOT NULL,
  UNIQUE (event_id, unit_kind, unit_index, text_hash),
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vector_unit_texts (
  text_hash TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS vector_unit_embeddings USING vec0(
  unit_id INTEGER PRIMARY KEY,
  embedding FLOAT[256] distance_metric=cosine
);

CREATE INDEX IF NOT EXISTS idx_vector_units_event ON vector_units(event_id);
CREATE INDEX IF NOT EXISTS idx_vector_units_tool_session ON vector_units(tool, session_id);
"#;

pub(crate) fn table_count(conn: &Connection, db_path: &Path, table: &str) -> Result<i64> {
    let table = checked_sql_identifier(table)?;
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
        row.get(0)
    })
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })
}

pub(crate) fn table_exists(conn: &Connection, db_path: &Path, table: &str) -> Result<bool> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?1)",
        [table],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value != 0)
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })
}

pub(crate) fn initialize_database(path: &Path) -> Result<()> {
    register_semantic_extension_if_enabled();
    let mut conn = Connection::open(path).map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;",
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    conn.execute_batch(SQLITE_SCHEMA)
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    ensure_checkpoint_schema(&conn, path)?;
    ensure_events_schema(&mut conn, path)?;
    ensure_memory_event_schema(&mut conn, path)?;
    ensure_events_fts_schema(&mut conn, path)?;
    ensure_supporting_indexes(&conn, path)?;
    ensure_event_refs_schema(&mut conn, path)?;
    ensure_memories_schema(&conn, path)?;
    ensure_semantic_vector_schema(&conn, path)?;
    conn.execute_batch(
        "PRAGMA user_version = 1;
         INSERT OR IGNORE INTO schema_migrations(version, name, applied_at)
         VALUES (1, 'initial_schema', strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));",
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    drop(conn);

    chmod(path, 0o600)?;
    set_if_exists(&path.with_file_name("harness.db-wal"), 0o600)?;
    set_if_exists(&path.with_file_name("harness.db-shm"), 0o600)?;
    Ok(())
}

pub(crate) fn open_index(path: &Path) -> Result<Connection> {
    register_semantic_extension_if_enabled();
    let mut conn = Connection::open(path).map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;",
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    ensure_checkpoint_schema(&conn, path)?;
    ensure_events_schema(&mut conn, path)?;
    ensure_memory_event_schema(&mut conn, path)?;
    ensure_events_fts_schema(&mut conn, path)?;
    ensure_supporting_indexes(&conn, path)?;
    ensure_event_refs_schema(&mut conn, path)?;
    ensure_memories_schema(&conn, path)?;
    Ok(conn)
}

fn ensure_events_schema(conn: &mut Connection, path: &Path) -> Result<()> {
    ensure_table_column(conn, path, "events", "tool_invocation_id", "TEXT")
}

/// The events-table definition the memory migration rebuilds into. The CHECK
/// constraints must match `schema.sql` exactly; keep both in sync.
const EVENTS_TABLE_WITH_MEMORY: &str = r#"
CREATE TABLE events_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
  session_id TEXT NOT NULL,
  dedupe_key TEXT NOT NULL UNIQUE,
  schema_version INTEGER NOT NULL,
  captured_at TEXT NOT NULL,
  tool_version TEXT,
  turn_id TEXT,
  message_id TEXT,
  project_root TEXT,
  cwd TEXT,
  source TEXT NOT NULL CHECK (
    source IN (
      'hook',
      'event_stream',
      'transcript_tail',
      'sdk_session_store',
      'backfill',
      'exec_json',
      'app_server',
      'memory_sync'
    )
  ),
  source_event_type TEXT NOT NULL,
  source_event_id TEXT,
  tool_invocation_id TEXT,
  canonical_type TEXT NOT NULL CHECK (
    canonical_type IN (
      'session.started',
      'session.resumed',
      'session.ended',
      'user.message',
      'assistant.delta',
      'assistant.message',
      'tool.call',
      'tool.result',
      'permission.requested',
      'permission.replied',
      'file.changed',
      'compaction.before',
      'compaction.after',
      'source.discontinuity',
      'error',
      'memory.file'
    )
  ),
  sequence INTEGER,
  raw_file TEXT NOT NULL,
  raw_line INTEGER,
  raw_offset INTEGER,
  payload_json TEXT,
  payload_ref TEXT,
  searchable_text TEXT NOT NULL DEFAULT '',
  compaction_state TEXT NOT NULL DEFAULT 'unknown' CHECK (
    compaction_state IN ('pre_compaction', 'post_compaction', 'none', 'unknown')
  ),
  FOREIGN KEY (tool, session_id) REFERENCES sessions(tool, session_id)
);
"#;

/// SQLite cannot alter a CHECK constraint in place, so admitting the
/// `memory_sync` source and `memory.file` canonical type requires rebuilding
/// the events table once. Runs on first open of a pre-memory database; the
/// rebuild preserves every row id, so all foreign keys (messages, tool_events,
/// compactions, event_files, event_refs, vector_units, memories) and the
/// contentless events_fts rowids keep pointing at the same events.
///
/// Foreign-key enforcement is toggled off for the duration (a connection-level
/// pragma that cannot change inside a transaction), then back on; nothing in
/// the copy changes referential relationships.
fn ensure_memory_event_schema(conn: &mut Connection, path: &Path) -> Result<()> {
    let sql: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'events'",
            [],
            |row| row.get(0),
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    if sql.contains("'memory.file'") {
        return Ok(());
    }

    conn.execute_batch("PRAGMA foreign_keys = OFF;")
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    let rebuild = conn.execute_batch(&format!(
        r#"BEGIN;
{E}
INSERT INTO events_rebuilt (
  id, tool, session_id, dedupe_key, schema_version, captured_at, tool_version,
  turn_id, message_id, project_root, cwd, source, source_event_type,
  source_event_id, tool_invocation_id, canonical_type, sequence, raw_file,
  raw_line, raw_offset, payload_json, payload_ref, searchable_text, compaction_state
) SELECT
  id, tool, session_id, dedupe_key, schema_version, captured_at, tool_version,
  turn_id, message_id, project_root, cwd, source, source_event_type,
  source_event_id, tool_invocation_id, canonical_type, sequence, raw_file,
  raw_line, raw_offset, payload_json, payload_ref, searchable_text, compaction_state
FROM events;
DROP TABLE events;
ALTER TABLE events_rebuilt RENAME TO events;
INSERT INTO schema_migrations(version, name, applied_at)
VALUES (2, 'memory_events', strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
COMMIT;
"#,
        E = EVENTS_TABLE_WITH_MEMORY
    ));
    let foreign_keys = conn.execute_batch("PRAGMA foreign_keys = ON;");
    match (rebuild, foreign_keys) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(source), _) => Err(Error::Sqlite {
            path: path.to_path_buf(),
            source,
        }),
        (Ok(()), Err(source)) => Err(Error::Sqlite {
            path: path.to_path_buf(),
            source,
        }),
    }
}

const MEMORIES_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
  session_id TEXT NOT NULL,
  project TEXT,
  name TEXT NOT NULL,
  native_path TEXT NOT NULL,
  size INTEGER NOT NULL DEFAULT 0,
  modified_at TEXT,
  captured_at TEXT NOT NULL,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memories_tool_project ON memories(tool, project);
CREATE INDEX IF NOT EXISTS idx_memories_tool_name_project ON memories(tool, name, project);
CREATE INDEX IF NOT EXISTS idx_memories_native_path ON memories(native_path);
"#;

/// Create the derived `memories` metadata table on open (mirrors the
/// event_refs pattern: `IF NOT EXISTS`, so it is a no-op on databases that
/// already carry it; rows are populated at index time like tool_events).
fn ensure_memories_schema(conn: &Connection, path: &Path) -> Result<()> {
    conn.execute_batch(MEMORIES_SCHEMA)
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

fn register_semantic_extension_if_enabled() {
    #[cfg(feature = "semantic")]
    {
        static SQLITE_VEC_REGISTER: std::sync::Once = std::sync::Once::new();
        SQLITE_VEC_REGISTER.call_once(|| unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(sqlite_vec_auto_extension()));
        });
    }
}

#[cfg(feature = "semantic")]
type SqliteAutoExtensionFn = unsafe extern "C" fn(
    *mut rusqlite::ffi::sqlite3,
    *mut *mut std::os::raw::c_char,
    *const rusqlite::ffi::sqlite3_api_routines,
) -> std::os::raw::c_int;

#[cfg(feature = "semantic")]
pub(crate) fn sqlite_vec_auto_extension() -> SqliteAutoExtensionFn {
    unsafe {
        std::mem::transmute::<*const (), SqliteAutoExtensionFn>(
            sqlite_vec::sqlite3_vec_init as *const (),
        )
    }
}

#[cfg(feature = "semantic")]
pub(crate) fn ensure_semantic_vector_schema(conn: &Connection, path: &Path) -> Result<()> {
    conn.execute_batch(SEMANTIC_VECTOR_SCHEMA)
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

#[cfg(not(feature = "semantic"))]
pub(crate) fn ensure_semantic_vector_schema(_conn: &Connection, _path: &Path) -> Result<()> {
    Ok(())
}

fn ensure_checkpoint_schema(conn: &Connection, path: &Path) -> Result<()> {
    for (column, definition) in [
        ("session_id", "TEXT"),
        ("source_size", "INTEGER NOT NULL DEFAULT 0"),
        ("source_mtime", "INTEGER"),
        ("last_successful_import_timestamp", "TEXT"),
    ] {
        ensure_table_column(conn, path, "checkpoints", column, definition)?;
    }
    Ok(())
}

pub(crate) fn ensure_table_column(
    conn: &Connection,
    path: &Path,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
    let table = checked_sql_identifier(table)?;
    let column = checked_sql_identifier(column)?;
    let exists = {
        let mut statement = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let mut exists = false;
        for row in rows {
            if row.map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })? == column
            {
                exists = true;
            }
        }
        exists
    };
    if !exists {
        conn.execute_batch(&format!(
            "ALTER TABLE {table} ADD COLUMN {column} {definition};"
        ))
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn checked_sql_identifier(identifier: &str) -> Result<&str> {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        return Err(Error::Validation(
            "SQL identifier must not be empty".to_string(),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(Error::Validation(format!(
            "invalid SQL identifier: {identifier}"
        )));
    }
    if !chars.all(|character| character == '_' || character.is_ascii_alphanumeric()) {
        return Err(Error::Validation(format!(
            "invalid SQL identifier: {identifier}"
        )));
    }
    Ok(identifier)
}

fn ensure_events_fts_schema(conn: &mut Connection, path: &Path) -> Result<()> {
    let columns = {
        let mut statement = conn
            .prepare("PRAGMA table_info(events_fts)")
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let mut columns = Vec::new();
        for row in rows {
            columns.push(row.map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?);
        }
        columns
    };

    let fts_sql: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'events_fts'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    let contentless = fts_sql
        .as_deref()
        .map(|sql| sql.contains("content=''") || sql.contains("content=\"\""))
        .unwrap_or(false);

    let legacy_fts = columns.iter().any(|column| column == "searchable_text") || !contentless;
    let incomplete_fts =
        contentless && !legacy_fts && events_fts_missing_boundary_rows(conn, path)?;

    if legacy_fts || incomplete_fts {
        let tx = conn.transaction().map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        tx.execute_batch("DROP TABLE IF EXISTS events_fts;")
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        tx.execute_batch(EVENTS_FTS_SCHEMA)
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        rebuild_events_fts(&tx, path)?;
        tx.commit().map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }

    Ok(())
}

fn events_fts_missing_boundary_rows(conn: &Connection, path: &Path) -> Result<bool> {
    // Legacy crash-window recovery heuristic, not a full FTS integrity scan.
    let (min_id, max_id): (Option<i64>, Option<i64>) = conn
        .query_row("SELECT MIN(id), MAX(id) FROM events", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    let Some(min_id) = min_id else {
        return Ok(false);
    };
    let max_id = max_id.expect("MAX(id) exists when MIN(id) exists");

    for event_id in [min_id, max_id] {
        let exists = conn
            .query_row(
                "SELECT rowid FROM events_fts WHERE rowid = ?1 LIMIT 1",
                [event_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?
            .is_some();
        if !exists {
            return Ok(true);
        }
    }

    Ok(false)
}

fn ensure_supporting_indexes(conn: &Connection, path: &Path) -> Result<()> {
    // Full events index set from schema.sql. Must be complete: DROP TABLE events
    // during the memory migration destroys every index on events, and this is
    // the only open-path recreation for existing installs.
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_events_tool_session_raw ON events(tool, session_id, raw_line, raw_offset);
         CREATE INDEX IF NOT EXISTS idx_events_canonical_captured ON events(canonical_type, captured_at);
         CREATE INDEX IF NOT EXISTS idx_events_session_captured ON events(tool, session_id, captured_at);
         CREATE INDEX IF NOT EXISTS idx_events_tool_captured ON events(tool, captured_at);
         CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(tool, session_id);
         CREATE INDEX IF NOT EXISTS idx_compactions_session ON compactions(tool, session_id);",
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

const EVENT_REFS_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS event_refs (
  event_id INTEGER NOT NULL,
  ref_kind TEXT NOT NULL CHECK (ref_kind IN ('pr', 'commit')),
  ref_value TEXT NOT NULL,
  PRIMARY KEY (event_id, ref_kind, ref_value),
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_event_refs_kind_value ON event_refs(ref_kind, ref_value);
"#;

/// Create the `event_refs` provenance table on open and backfill it from the
/// already-indexed events the first time it appears, so existing indexes gain
/// ref rows without a full reindex.
///
/// The table is created with `IF NOT EXISTS`, so this is a no-op on databases
/// that already carry it. The backfill runs only when the table was absent
/// before this call (a pre-provenance index), iterating every event's stored
/// `searchable_text` - the same rendered string the index pipeline extracts
/// from - and inserting the extracted refs. Newly created databases have no
/// events yet, so their backfill pass is empty and the live index pipeline
/// populates `event_refs` going forward.
fn ensure_event_refs_schema(conn: &mut Connection, path: &Path) -> Result<()> {
    let already_present = table_exists(conn, path, "event_refs")?;
    conn.execute_batch(EVENT_REFS_SCHEMA)
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    if already_present {
        return Ok(());
    }
    backfill_event_refs(conn, path)
}

fn backfill_event_refs(conn: &mut Connection, path: &Path) -> Result<()> {
    let tx = conn.transaction().map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    {
        let mut select = tx
            .prepare("SELECT id, searchable_text FROM events ORDER BY id")
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let mut insert = tx
            .prepare(
                "INSERT OR IGNORE INTO event_refs(event_id, ref_kind, ref_value)
                 VALUES (?1, ?2, ?3)",
            )
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let mut rows = select.query([]).map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        while let Some(row) = rows.next().map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })? {
            let event_id = row.get::<_, i64>(0).map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
            let searchable_text = row
                .get::<_, Option<String>>(1)
                .map_err(|source| Error::Sqlite {
                    path: path.to_path_buf(),
                    source,
                })?
                .unwrap_or_default();
            for reference in extract_refs(&searchable_text) {
                insert
                    .execute((event_id, reference.kind.as_str(), reference.value))
                    .map_err(|source| Error::Sqlite {
                        path: path.to_path_buf(),
                        source,
                    })?;
            }
        }
    }
    tx.commit().map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })
}

fn rebuild_events_fts(conn: &Connection, path: &Path) -> Result<()> {
    let mut select = conn
        .prepare(
            "SELECT id, payload_json, tool, session_id, canonical_type, raw_file, raw_line, raw_offset
             FROM events
             ORDER BY id",
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    let mut insert = conn
        .prepare(
            "INSERT INTO events_fts(rowid, user_text, assistant_text, tool_intent, tool_output, metadata_text, tool, session_id, canonical_type, raw_file, raw_line, raw_offset)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    let mut rows = select.query([]).map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    while let Some(row) = rows.next().map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })? {
        let event_id = row.get::<_, i64>(0).map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let payload_json = row
            .get::<_, Option<String>>(1)
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let tool = row.get::<_, String>(2).map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let session_id = row.get::<_, String>(3).map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let canonical_type = row.get::<_, String>(4).map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let raw_file = row.get::<_, String>(5).map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let raw_line = row.get::<_, i64>(6).map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let raw_offset = row
            .get::<_, Option<i64>>(7)
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        let canonical_type = CanonicalType::from_str(&canonical_type)?;
        let payload = match payload_json.as_deref() {
            Some(payload_json) => serde_json::from_str(payload_json)?,
            None => payload_for_raw_pointer(&raw_file, raw_line, raw_offset)?,
        };
        let document = search_document_for_event(canonical_type, &payload);
        insert
            .execute((
                event_id,
                &document.user_text,
                &document.assistant_text,
                &document.tool_intent,
                &document.tool_output,
                &document.metadata_text,
                &tool,
                &session_id,
                canonical_type.as_str(),
                &raw_file,
                raw_line,
                raw_offset,
            ))
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
    }

    Ok(())
}
