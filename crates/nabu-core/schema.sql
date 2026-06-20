CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
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

CREATE TABLE IF NOT EXISTS events (
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
      'app_server'
    )
  ),
  source_event_type TEXT NOT NULL,
  source_event_id TEXT,
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
      'error'
    )
  ),
  sequence INTEGER,
  raw_file TEXT NOT NULL,
  raw_line INTEGER,
  raw_offset INTEGER,
  -- Legacy nullable migration cache; full-fidelity payload reads hydrate from raw JSONL/blobs.
  payload_json TEXT,
  payload_ref TEXT,
  searchable_text TEXT NOT NULL DEFAULT '',
  compaction_state TEXT NOT NULL DEFAULT 'unknown' CHECK (
    compaction_state IN ('pre_compaction', 'post_compaction', 'none', 'unknown')
  ),
  FOREIGN KEY (tool, session_id) REFERENCES sessions(tool, session_id)
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
  session_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
  text TEXT NOT NULL,
  is_delta INTEGER NOT NULL DEFAULT 0 CHECK (is_delta IN (0, 1)),
  sequence INTEGER,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
  session_id TEXT NOT NULL,
  tool_name TEXT,
  command TEXT,
  status TEXT CHECK (status IS NULL OR status IN ('started', 'completed', 'failed', 'denied')),
  duration_ms INTEGER,
  input_text TEXT,
  output_text TEXT,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS event_files (
  event_id INTEGER NOT NULL,
  file_id INTEGER NOT NULL,
  relationship TEXT NOT NULL CHECK (relationship IN ('mentioned', 'read', 'written', 'edited', 'deleted')),
  PRIMARY KEY (event_id, file_id, relationship),
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS compactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
  session_id TEXT NOT NULL,
  trigger TEXT,
  raw_file TEXT NOT NULL,
  raw_line INTEGER,
  raw_offset INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checkpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_tool TEXT NOT NULL CHECK (source_tool IN ('codex', 'claude', 'opencode')),
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
);

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

CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_events_tool_session_raw ON events(tool, session_id, raw_line, raw_offset);
CREATE INDEX IF NOT EXISTS idx_events_canonical_captured ON events(canonical_type, captured_at);
CREATE INDEX IF NOT EXISTS idx_events_session_captured ON events(tool, session_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_events_tool_captured ON events(tool, captured_at);
CREATE INDEX IF NOT EXISTS idx_messages_session_sequence ON messages(tool, session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(tool, session_id);
CREATE INDEX IF NOT EXISTS idx_tool_events_name ON tool_events(tool_name);
CREATE INDEX IF NOT EXISTS idx_compactions_session ON compactions(tool, session_id);

-- Semantic feature only, after sqlite-vec has been registered:
--
-- CREATE TABLE IF NOT EXISTS vector_units (
--   id INTEGER PRIMARY KEY AUTOINCREMENT,
--   event_id INTEGER NOT NULL,
--   tool TEXT NOT NULL CHECK (tool IN ('codex', 'claude', 'opencode')),
--   session_id TEXT NOT NULL,
--   unit_kind TEXT NOT NULL CHECK (unit_kind IN ('user_text', 'assistant_text', 'tool_intent', 'metadata_text')),
--   unit_index INTEGER NOT NULL DEFAULT 0,
--   text_hash TEXT NOT NULL,
--   raw_file TEXT NOT NULL,
--   raw_line INTEGER,
--   raw_offset INTEGER,
--   created_at TEXT NOT NULL,
--   UNIQUE (event_id, unit_kind, unit_index, text_hash),
--   FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
-- );
--
-- CREATE TABLE IF NOT EXISTS vector_unit_texts (
--   text_hash TEXT PRIMARY KEY,
--   text TEXT NOT NULL,
--   created_at TEXT NOT NULL
-- );
--
-- CREATE VIRTUAL TABLE IF NOT EXISTS vector_unit_embeddings USING vec0(
--   unit_id INTEGER PRIMARY KEY,
--   embedding FLOAT[256] distance_metric=cosine
-- );
--
-- CREATE INDEX IF NOT EXISTS idx_vector_units_event ON vector_units(event_id);
-- CREATE INDEX IF NOT EXISTS idx_vector_units_tool_session ON vector_units(tool, session_id);
