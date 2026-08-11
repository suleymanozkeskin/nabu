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
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
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
    ensure_tool_pi_schema(&mut conn, path)?;
    ensure_events_fts_schema(&mut conn, path)?;
    ensure_all_schema_indexes(&conn, path)?;
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
    ensure_tool_pi_schema(&mut conn, path)?;
    ensure_events_fts_schema(&mut conn, path)?;
    ensure_all_schema_indexes(&conn, path)?;
    ensure_event_refs_schema(&mut conn, path)?;
    ensure_memories_schema(&conn, path)?;
    Ok(conn)
}

fn ensure_events_schema(conn: &mut Connection, path: &Path) -> Result<()> {
    ensure_table_column(conn, path, "events", "tool_invocation_id", "TEXT")
}

/// The events-table definition the memory migration (v2) rebuilds into. This
/// is the post-memory, **pre-pi** shape: the pi migration (v3) rebuilds events
/// again from [`EVENTS_TABLE_WITH_PI`], so a pre-memory DB upgrading now runs
/// v2 then v3 in sequence and each leaves its exact version's state. The CHECK
/// constraints must match `schema.sql` minus pi; keep both in sync.
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

/// The events-table definition the tool-pi migration (v3) rebuilds into: the
/// full current events DDL including `memory.file`, `memory_sync`, and `pi`.
const EVENTS_TABLE_WITH_PI: &str = r#"
CREATE TABLE events_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
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

/// Table definitions used by the tool-pi migration (schema v3): every table
/// whose CREATE constrains `tool`/`source_tool` to the known harnesses,
/// rebuilt with `pi` admitted. Must match `schema.sql` exactly; keep both in
/// sync. Each entry carries the table name, its rebuilt DDL (the `_rebuilt`
/// table is renamed into place), and the explicit column list for the copy.
const PI_REBUILD_TABLES: &[(&str, &str, &str)] = &[
    (
        "sessions",
        r#"CREATE TABLE sessions_rebuilt (
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
  session_id TEXT NOT NULL,
  filename_session_id TEXT NOT NULL,
  project_root TEXT,
  cwd TEXT,
  started_at TEXT,
  updated_at TEXT,
  raw_file TEXT NOT NULL,
  event_count INTEGER NOT NULL DEFAULT 0,
  message_count INTEGER NOT NULL DEFAULT 0,
  tool_event_count INTEGER NOT NULL DEFAULT 0,
  compaction_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (tool, session_id)
);"#,
        "tool, session_id, filename_session_id, project_root, cwd, started_at, updated_at, raw_file, event_count, message_count, tool_event_count, compaction_count",
    ),
    (
        "messages",
        r#"CREATE TABLE messages_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
  session_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
  text TEXT NOT NULL,
  is_delta INTEGER NOT NULL DEFAULT 0 CHECK (is_delta IN (0, 1)),
  sequence INTEGER,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);"#,
        "id, event_id, tool, session_id, role, text, is_delta, sequence",
    ),
    (
        "tool_events",
        r#"CREATE TABLE tool_events_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
  session_id TEXT NOT NULL,
  tool_name TEXT,
  command TEXT,
  status TEXT CHECK (status IS NULL OR status IN ('started', 'completed', 'failed', 'denied')),
  duration_ms INTEGER,
  input_text TEXT,
  output_text TEXT,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);"#,
        "id, event_id, tool, session_id, tool_name, command, status, duration_ms, input_text, output_text",
    ),
    (
        "compactions",
        r#"CREATE TABLE compactions_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
  session_id TEXT NOT NULL,
  trigger TEXT,
  raw_file TEXT NOT NULL,
  raw_line INTEGER,
  raw_offset INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);"#,
        "id, event_id, tool, session_id, trigger, raw_file, raw_line, raw_offset, created_at",
    ),
    (
        "memories",
        r#"CREATE TABLE memories_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
  session_id TEXT NOT NULL,
  project TEXT,
  name TEXT NOT NULL,
  native_path TEXT NOT NULL,
  size INTEGER NOT NULL DEFAULT 0,
  modified_at TEXT,
  captured_at TEXT NOT NULL,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);"#,
        "id, event_id, tool, session_id, project, name, native_path, size, modified_at, captured_at",
    ),
    (
        "checkpoints",
        r#"CREATE TABLE checkpoints_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_tool TEXT NOT NULL CHECK (source_tool IN ('codex', 'claude', 'opencode', 'pi')),
  source_kind TEXT NOT NULL CHECK (source_kind IN ('transcript', 'event_stream', 'api_export', 'raw_jsonl')),
  source_path TEXT NOT NULL,
  source_identity TEXT,
  session_id TEXT,
  byte_offset INTEGER NOT NULL DEFAULT 0,
  source_size INTEGER NOT NULL DEFAULT 0,
  source_mtime INTEGER,
  last_line_hash TEXT,
  last_successful_import_timestamp TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE (source_tool, source_kind, source_path)
);"#,
        "id, source_tool, source_kind, source_path, source_identity, session_id, byte_offset, source_size, source_mtime, last_line_hash, last_successful_import_timestamp, updated_at",
    ),
];

#[cfg(feature = "semantic")]
const VECTOR_UNITS_REBUILT_DDL: &str = r#"
CREATE TABLE vector_units_rebuilt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
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
"#;

/// SQLite cannot alter a CHECK constraint in place, so admitting `pi` as a
/// tool requires rebuilding every tool-constrained table once (schema v3).
/// Runs on first open of a pre-pi database; the rebuild preserves every row
/// id, so foreign keys and the contentless events_fts rowids keep pointing at
/// the same rows. Indexes on the rebuilt tables are recreated afterwards from
/// the full schema.sql index set.
fn ensure_tool_pi_schema(conn: &mut Connection, path: &Path) -> Result<()> {
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
    if sql.contains("'pi'") {
        return Ok(());
    }

    conn.execute_batch("PRAGMA foreign_keys = OFF;")
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    let rebuild = (|| -> Result<()> {
        let tx = conn.transaction().map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        // events first: every other rebuilt table references it by id.
        if table_exists(&tx, path, "events")? {
            tx.execute_batch(&format!(
                "{E}
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
                 ALTER TABLE events_rebuilt RENAME TO events;",
                E = EVENTS_TABLE_WITH_PI
            ))
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        }
        for (table, ddl, columns) in PI_REBUILD_TABLES {
            if !table_exists(&tx, path, table)? {
                continue;
            }
            tx.execute_batch(&format!(
                "{ddl}
                 INSERT INTO {table}_rebuilt ({columns}) SELECT {columns} FROM {table};
                 DROP TABLE {table};
                 ALTER TABLE {table}_rebuilt RENAME TO {table};",
            ))
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        }
        #[cfg(feature = "semantic")]
        if table_exists(&tx, path, "vector_units")? {
            tx.execute_batch(&format!(
                "{ddl}
                 INSERT INTO vector_units_rebuilt (
                   id, event_id, tool, session_id, unit_kind, unit_index, text_hash,
                   raw_file, raw_line, raw_offset, created_at
                 ) SELECT
                   id, event_id, tool, session_id, unit_kind, unit_index, text_hash,
                   raw_file, raw_line, raw_offset, created_at
                 FROM vector_units;
                 DROP TABLE vector_units;
                 ALTER TABLE vector_units_rebuilt RENAME TO vector_units;",
                ddl = VECTOR_UNITS_REBUILT_DDL
            ))
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        }
        // DROP TABLE destroyed every index on the rebuilt tables; recreate the
        // full schema.sql index set.
        ensure_all_schema_indexes(&tx, path)?;
        tx.execute_batch(
            "INSERT INTO schema_migrations(version, name, applied_at)
             VALUES (3, 'tool_pi', strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));",
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        tx.commit().map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })
    })();
    let foreign_keys = conn.execute_batch("PRAGMA foreign_keys = ON;");
    match (rebuild, foreign_keys) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), _) => Err(error),
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
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode', 'pi')),
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

fn ensure_all_schema_indexes(conn: &Connection, path: &Path) -> Result<()> {
    // Full index set from schema.sql. Must be complete: DROP TABLE during the
    // memory/pi migrations destroys every index on the rebuilt tables, and this
    // is the only open-path recreation for existing installs. Table-scoped
    // index groups are guarded by table existence (memories on pre-memory
    // DBs, vector_units only in semantic builds).
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
         CREATE INDEX IF NOT EXISTS idx_events_tool_session_raw ON events(tool, session_id, raw_line, raw_offset);
         CREATE INDEX IF NOT EXISTS idx_events_canonical_captured ON events(canonical_type, captured_at);
         CREATE INDEX IF NOT EXISTS idx_events_session_captured ON events(tool, session_id, captured_at);
         CREATE INDEX IF NOT EXISTS idx_events_tool_captured ON events(tool, captured_at);
         CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(tool, session_id);
         CREATE INDEX IF NOT EXISTS idx_tool_events_name ON tool_events(tool_name);
         CREATE INDEX IF NOT EXISTS idx_compactions_session ON compactions(tool, session_id);",
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    if table_exists(conn, path, "messages")? {
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_sequence ON messages(tool, session_id, sequence);",
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    if table_exists(conn, path, "memories")? {
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_tool_project ON memories(tool, project);
             CREATE INDEX IF NOT EXISTS idx_memories_tool_name_project ON memories(tool, name, project);
             CREATE INDEX IF NOT EXISTS idx_memories_native_path ON memories(native_path);",
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    #[cfg(feature = "semantic")]
    if table_exists(conn, path, "vector_units")? {
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_vector_units_event ON vector_units(event_id);
             CREATE INDEX IF NOT EXISTS idx_vector_units_tool_session ON vector_units(tool, session_id);",
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::OptionalExtension;
    use tempfile::tempdir;

    /// The post-memory, pre-pi events table: `EVENTS_TABLE_WITH_MEMORY` (the
    /// current rebuilt DDL) with `pi` removed from the tool CHECK. Renames the
    /// table back to `events`.
    fn legacy_events_ddl() -> String {
        EVENTS_TABLE_WITH_MEMORY
            .replace(", 'pi'", "")
            .replace("events_rebuilt", "events")
    }

    /// Sibling tables in their post-memory, pre-pi shape: tool-constrained
    /// CHECKs without pi, plus the bookkeeping tables the open path touches.
    fn legacy_sibling_tables() -> &'static str {
        r#"
CREATE TABLE sessions (
  tool TEXT NOT NULL,
  session_id TEXT NOT NULL,
  filename_session_id TEXT NOT NULL,
  project_root TEXT,
  cwd TEXT,
  started_at TEXT,
  updated_at TEXT,
  raw_file TEXT NOT NULL,
  event_count INTEGER NOT NULL DEFAULT 0,
  message_count INTEGER NOT NULL DEFAULT 0,
  tool_event_count INTEGER NOT NULL DEFAULT 0,
  compaction_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (tool, session_id)
);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
  text TEXT NOT NULL,
  is_delta INTEGER NOT NULL DEFAULT 0 CHECK (is_delta IN (0, 1)),
  sequence INTEGER,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);
CREATE TABLE tool_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL,
  session_id TEXT NOT NULL,
  tool_name TEXT,
  command TEXT,
  status TEXT,
  duration_ms INTEGER,
  input_text TEXT,
  output_text TEXT
);
CREATE TABLE compactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL,
  session_id TEXT NOT NULL,
  trigger TEXT,
  raw_file TEXT NOT NULL,
  raw_line INTEGER,
  raw_offset INTEGER,
  created_at TEXT NOT NULL
);
CREATE TABLE memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL,
  session_id TEXT NOT NULL,
  project TEXT,
  name TEXT NOT NULL,
  native_path TEXT NOT NULL,
  size INTEGER NOT NULL DEFAULT 0,
  modified_at TEXT,
  captured_at TEXT NOT NULL
);
CREATE TABLE checkpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_tool TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  source_path TEXT NOT NULL,
  source_identity TEXT,
  session_id TEXT,
  byte_offset INTEGER NOT NULL DEFAULT 0,
  source_size INTEGER NOT NULL DEFAULT 0,
  source_mtime INTEGER,
  last_line_hash TEXT,
  last_successful_import_timestamp TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE (source_tool, source_kind, source_path)
);
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at TEXT NOT NULL
);
CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"#
    }

    /// Insert one codex user.message event plus its sessions row, with the
    /// payload inline so FTS rebuilds never touch raw files.
    fn seed_codex_event(conn: &rusqlite::Connection) {
        conn.execute_batch(
            r#"INSERT INTO sessions(tool, session_id, filename_session_id, started_at, updated_at, raw_file)
               VALUES ('codex', 'session-1', 'session-1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'raw/codex/codex_session-1.jsonl');
               INSERT INTO events(
                 tool, session_id, dedupe_key, schema_version, captured_at, source,
                 source_event_type, canonical_type, raw_file, raw_line, searchable_text, compaction_state,
                 payload_json
               ) VALUES (
                 'codex', 'session-1', 'sha256:legacy', 1, '2026-01-01T00:00:00Z', 'hook',
                 'UserPromptSubmit', 'user.message', 'raw/codex/codex_session-1.jsonl', 1, 'legacy prompt', 'none',
                 '{"prompt": "legacy prompt"}'
               );"#,
        )
        .unwrap();
    }

    fn build_legacy_store(db_path: &std::path::Path, events_ddl: &str, seed: bool) {
        let conn = rusqlite::Connection::open(db_path).unwrap();
        conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
        conn.execute_batch(legacy_sibling_tables()).unwrap();
        conn.execute_batch(events_ddl).unwrap();
        conn.execute_batch(
            "CREATE INDEX idx_events_tool_session_raw ON events(tool, session_id, raw_line, raw_offset);
             CREATE INDEX idx_events_canonical_captured ON events(canonical_type, captured_at);
             CREATE INDEX idx_events_session_captured ON events(tool, session_id, captured_at);
             CREATE INDEX idx_events_tool_captured ON events(tool, captured_at);
             CREATE INDEX idx_messages_session_sequence ON messages(tool, session_id, sequence);
             CREATE INDEX idx_tool_events_session ON tool_events(tool, session_id);
             CREATE INDEX idx_compactions_session ON compactions(tool, session_id);
             CREATE INDEX idx_memories_tool_project ON memories(tool, project);
             CREATE INDEX idx_memories_native_path ON memories(native_path);",
        )
        .unwrap();
        if seed {
            seed_codex_event(&conn);
        }
        drop(conn);
    }

    fn table_sql(conn: &rusqlite::Connection, table: &str) -> String {
        conn.query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [table],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn legacy_index_admits_pi_tool_without_losing_rows() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("harness.db");

        // Post-memory, pre-pi store with one codex event, one memory row, and
        // one checkpoint row, plus the pre-migration indexes.
        build_legacy_store(&db_path, &legacy_events_ddl(), true);
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "INSERT INTO memories(event_id, tool, session_id, project, name, native_path, size, captured_at)
                 VALUES (1, 'codex', 'memory:global', NULL, 'prefs.md', '/x/prefs.md', 3, '2026-01-01T00:00:00Z');
                 INSERT INTO checkpoints(source_tool, source_kind, source_path, source_size, updated_at)
                 VALUES ('codex', 'raw_jsonl', 'raw/codex/codex_session-1.jsonl', 10, '2026-01-01T00:00:00Z');",
            )
            .unwrap();
            drop(conn);
        }

        // First open runs the v3 migration (tool CHECK gains pi on every
        // tool-constrained table).
        let conn = open_index(&db_path).unwrap();
        let events_sql = table_sql(&conn, "events");
        assert!(events_sql.contains("'pi'"), "{events_sql}");
        for table in [
            "sessions",
            "messages",
            "tool_events",
            "compactions",
            "memories",
            "checkpoints",
        ] {
            let sql = table_sql(&conn, table);
            assert!(sql.contains("'pi'"), "{table}: {sql}");
        }
        assert!(table_sql(&conn, "events").contains("'memory.file'"));

        // The legacy row survived with its id.
        let (count, max_id): (i64, i64) = conn
            .query_row("SELECT COUNT(*), MAX(id) FROM events", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert_eq!((count, max_id), (1, 1));
        let memory_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap();
        assert_eq!(memory_count, 1);
        let checkpoint_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM checkpoints", [], |row| row.get(0))
            .unwrap();
        assert_eq!(checkpoint_count, 1);

        // Migration recorded once.
        let version: Option<i64> = conn
            .query_row(
                "SELECT version FROM schema_migrations WHERE version = 3",
                [],
                |row| row.get(0),
            )
            .optional()
            .unwrap();
        assert_eq!(version, Some(3));

        // A pi event now inserts cleanly through the widened CHECK.
        conn.execute_batch(
            "INSERT INTO sessions(tool, session_id, filename_session_id, started_at, updated_at, raw_file)
             VALUES ('pi', 'pi-session-1', 'pi-session-1', '2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z', 'raw/pi/pi_pi-session-1.jsonl');
             INSERT INTO events(
               tool, session_id, dedupe_key, schema_version, captured_at, source,
               source_event_type, canonical_type, raw_file, raw_line, searchable_text, compaction_state,
               payload_json
             ) VALUES (
               'pi', 'pi-session-1', 'sha256:pi-event', 1, '2026-01-02T00:00:00Z', 'backfill',
               'message.user', 'user.message', 'raw/pi/pi_pi-session-1.jsonl', 1, 'pi prompt', 'none',
               '{\"text\": \"pi prompt\"}'
             );",
        )
        .unwrap();
        let id: i64 = conn
            .query_row(
                "SELECT id FROM events WHERE dedupe_key = 'sha256:pi-event'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(id, 2);

        // All schema.sql indexes were recreated after the DROP TABLEs.
        for index in [
            "idx_events_tool_session_raw",
            "idx_events_canonical_captured",
            "idx_events_session_captured",
            "idx_events_tool_captured",
            "idx_messages_session_sequence",
            "idx_tool_events_session",
            "idx_tool_events_name",
            "idx_compactions_session",
            "idx_memories_tool_project",
            "idx_memories_tool_name_project",
            "idx_memories_native_path",
        ] {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = ?1",
                    [index],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(exists, 1, "missing index {index}");
        }
        drop(conn);

        // Reopening is a no-op: the migration does not run twice.
        let conn = open_index(&db_path).unwrap();
        let migration_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 3",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(migration_count, 1);
    }

    #[test]
    fn pre_memory_index_runs_v2_then_v3_and_keeps_both_checks() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("harness.db");

        // Pre-memory, pre-pi store: strip memory.file/memory_sync and pi from
        // the current events DDL.
        let ddl = legacy_events_ddl()
            .replace(
                "'source.discontinuity',\n      'error',\n      'memory.file'",
                "'source.discontinuity',\n      'error'",
            )
            .replace("'app_server',\n      'memory_sync'", "'app_server'");
        build_legacy_store(&db_path, &ddl, true);

        let conn = open_index(&db_path).unwrap();
        let events_sql = table_sql(&conn, "events");
        assert!(events_sql.contains("'memory.file'"), "{events_sql}");
        assert!(events_sql.contains("'memory_sync'"), "{events_sql}");
        assert!(events_sql.contains("'pi'"), "{events_sql}");
        let (count, max_id): (i64, i64) = conn
            .query_row("SELECT COUNT(*), MAX(id) FROM events", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert_eq!((count, max_id), (1, 1));
        let v2: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let v3: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 3",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!((v2, v3), (1, 1));
    }
}
