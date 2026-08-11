//! Memory-folder capture and retrieval: scans each tool's native memory
//! folders (claude `projects/<id>/memory/`, codex `memories/`), appends each
//! file to the raw store as a `memory.file` event, and serves the captured
//! files back with raw citations.
//!
//! Memory files are first-class events: they are searchable through the same
//! FTS/embedding pipeline, dedupe on content (an unchanged file never appends
//! a second event), and read surfaces (`get_memory`) hydrate content from the
//! canonical raw store exactly like session payloads. OpenCode has no native
//! memory folder, so its discovery is intentionally empty.

use crate::{
    append_prepared_events, open_index, payload_for_raw_pointer, sanitize_session_id,
    CanonicalType, Error, MemoryFileContent, MemoryFileSummary, MemorySyncReport, NotFound, Result,
    Source, Tool, SCHEMA_VERSION,
};
use rusqlite::OptionalExtension;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Session id of the memory pseudo-session holding one file's captured
/// history: the claude project slug (one pseudo-session per project's memory
/// folder) or `memories` for codex's global folder.
fn memory_session_id(tool: Tool, project: Option<&str>) -> String {
    match (tool, project) {
        (Tool::Claude, Some(project)) => project.to_string(),
        (Tool::Claude, None) => "memories".to_string(),
        (Tool::Codex, _) => "memories".to_string(),
        (Tool::Opencode, _) => "memories".to_string(),
    }
}

/// A memory file discovered in a tool's native folders. `project` is the
/// claude project slug (folder name under `projects/`); codex memories are
/// global, so their project is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFileMeta {
    pub tool: Tool,
    pub project: Option<String>,
    pub name: String,
    pub native_path: PathBuf,
    pub size: u64,
    pub modified_at: Option<String>,
}

fn home_dir() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or(Error::HomeUnavailable)
}

/// Claude's projects root: `$CLAUDE_CONFIG_DIR/projects`, else `~/.claude/projects`.
fn claude_projects_root() -> Result<PathBuf> {
    if let Some(config_dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return Ok(PathBuf::from(config_dir).join("projects"));
    }
    Ok(home_dir()?.join(".claude").join("projects"))
}

/// Codex's home: `$CODEX_HOME`, else `~/.codex`.
fn codex_home() -> Result<PathBuf> {
    if let Some(codex_home) = std::env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(codex_home));
    }
    Ok(home_dir()?.join(".codex"))
}

/// The native memory folders for a tool, each with its project identity
/// (claude project slug; `None` for codex's global folder). OpenCode has no
/// memory-folder concept today, so it contributes no roots; callers see an
/// empty list rather than an error, keeping the three-tool surface total.
fn memory_roots(tool: Tool) -> Result<Vec<(PathBuf, Option<String>)>> {
    match tool {
        Tool::Claude => {
            let projects = claude_projects_root()?;
            if !projects.is_dir() {
                return Ok(Vec::new());
            }
            let mut roots = Vec::new();
            let entries = fs::read_dir(&projects).map_err(|source| Error::Io {
                path: projects.clone(),
                source,
            })?;
            for entry in entries {
                let entry = entry.map_err(|source| Error::Io {
                    path: projects.clone(),
                    source,
                })?;
                let project_dir = entry.path();
                if !project_dir.is_dir() {
                    continue;
                }
                let memory = project_dir.join("memory");
                if memory.is_dir() {
                    roots.push((
                        memory,
                        project_dir
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned()),
                    ));
                }
            }
            Ok(roots)
        }
        Tool::Codex => {
            let memories = codex_home()?.join("memories");
            if memories.is_dir() {
                Ok(vec![(memories, None)])
            } else {
                Ok(Vec::new())
            }
        }
        Tool::Opencode => Ok(Vec::new()),
    }
}

/// Recursively collect regular files under `dir`, skipping hidden entries.
fn collect_memory_files(dir: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
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
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_memory_files(&path, output)?;
        } else if path.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

fn modified_at_rfc3339(system_time: std::time::SystemTime) -> Option<String> {
    OffsetDateTime::from(system_time).format(&Rfc3339).ok()
}

/// Discover every memory file a tool's native folders currently hold.
pub fn discover_memory_files(tool: Tool) -> Result<Vec<MemoryFileMeta>> {
    let mut metas = Vec::new();
    for (root, project) in memory_roots(tool)? {
        let mut files = Vec::new();
        collect_memory_files(&root, &mut files)?;
        for path in files {
            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                // A memory file removed between the walk and the stat is not
                // an error; it simply is not part of this pass.
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
                Err(source) => {
                    return Err(Error::Io {
                        path: path.clone(),
                        source,
                    })
                }
            };
            metas.push(MemoryFileMeta {
                tool,
                project: project.clone(),
                name: path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                native_path: path.clone(),
                size: metadata.len(),
                modified_at: metadata.modified().ok().and_then(modified_at_rfc3339),
            });
        }
    }
    Ok(metas)
}

/// Capture one tool's current memory files into the raw store. Each file
/// becomes a `memory.file` event in its memory pseudo-session; the dedupe key
/// is content-derived, so an unchanged file appends nothing.
pub fn sync_memory(home: &Path, tool: Tool) -> Result<MemorySyncReport> {
    let synced_at = OffsetDateTime::now_utc().format(&Rfc3339)?;
    let discovered = discover_memory_files(tool)?;
    let discovered_count = discovered.len();
    let mut events = Vec::with_capacity(discovered_count);
    for meta in discovered {
        let content = match fs::read(&meta.native_path) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            // File vanished since discovery: skip, it is not part of this pass.
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(Error::Io {
                    path: meta.native_path.clone(),
                    source,
                })
            }
        };
        let session_id = memory_session_id(tool, meta.project.as_deref());
        let payload = json!({
            "name": meta.name,
            "project": meta.project,
            "native_path": meta.native_path.display().to_string(),
            "content": content,
            "size": meta.size,
            "modified_at": meta.modified_at,
            "synced_at": synced_at,
        });
        let mut envelope = crate::EventEnvelope {
            schema_version: SCHEMA_VERSION,
            captured_at: synced_at.clone(),
            tool,
            tool_version: None,
            session_id,
            filename_session_id: String::new(),
            turn_id: None,
            message_id: None,
            project_root: None,
            cwd: None,
            source: Source::MemorySync,
            source_event_type: "memory.file".to_string(),
            canonical_type: CanonicalType::MemoryFile,
            source_event_id: None,
            dedupe_key: String::new(),
            sequence: None,
            raw_file: None,
            raw_offset: None,
            payload,
            payload_ref: None,
        };
        // Sanitize after session_id is known; validate() in the append path
        // checks the two agree.
        envelope.filename_session_id = sanitize_session_id(&envelope.session_id);
        events.push(envelope);
    }

    let appended = if events.is_empty() {
        0
    } else {
        append_prepared_events(home, events)?
            .into_iter()
            .filter(|report| report.appended)
            .count()
    };
    Ok(MemorySyncReport {
        tool,
        discovered: discovered_count,
        appended,
    })
}

fn memory_summary_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<(MemoryFileSummary, String, i64, Option<i64>)> {
    let tool_text: String = row.get(0)?;
    let tool = Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?;
    let summary = MemoryFileSummary {
        tool,
        session_id: row.get(1)?,
        project: row.get(2)?,
        name: row.get(3)?,
        native_path: row.get(4)?,
        size: row.get(5)?,
        modified_at: row.get(6)?,
        captured_at: row.get(7)?,
        raw_file: row.get(8)?,
        raw_line: row.get(9)?,
        raw_offset: row.get(10)?,
    };
    Ok((
        summary.clone(),
        summary.raw_file.clone(),
        summary.raw_line,
        summary.raw_offset,
    ))
}

/// List captured memory files, newest capture first. Returns the latest
/// captured version of each file (a file edited since its last sync appears
/// once, with its most recent event). Files removed from the tool's folders
/// remain listed: the capture is durable history, not a mirror.
pub fn list_memories(
    home: &Path,
    tool: Option<Tool>,
    limit: usize,
) -> Result<Vec<MemoryFileSummary>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT
               m.tool,
               m.session_id,
               m.project,
               m.name,
               m.native_path,
               m.size,
               m.modified_at,
               m.captured_at,
               e.raw_file,
               e.raw_line,
               e.raw_offset
             FROM memories m
             JOIN events e ON e.id = m.event_id
             WHERE (?1 IS NULL OR m.tool = ?1)
               AND m.id = (
                 SELECT MAX(m2.id) FROM memories m2
                 WHERE m2.tool = m.tool AND m2.native_path = m.native_path
               )
             ORDER BY m.captured_at DESC, m.id DESC
             LIMIT ?2",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let rows = statement
        .query_map(
            (tool.map(Tool::as_str), limit.clamp(1, 1000) as i64),
            |row| {
                let (summary, _, _, _) = memory_summary_from_row(row)?;
                Ok(summary)
            },
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let mut memories = Vec::new();
    for row in rows {
        memories.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    Ok(memories)
}

/// Read one captured memory file by its tool identity, hydrating content from
/// the canonical raw store. For claude, `project` is required (memory is
/// per-project); for codex it must be `None`.
pub fn get_memory(
    home: &Path,
    tool: Tool,
    project: Option<&str>,
    name: &str,
) -> Result<MemoryFileContent> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT
               m.tool,
               m.session_id,
               m.project,
               m.name,
               m.native_path,
               m.size,
               m.modified_at,
               m.captured_at,
               e.raw_file,
               e.raw_line,
               e.raw_offset
             FROM memories m
             JOIN events e ON e.id = m.event_id
             WHERE m.tool = ?1 AND m.name = ?2 AND m.project IS ?3
             ORDER BY m.id DESC
             LIMIT 1",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let row = statement
        .query_row((tool.as_str(), name, project), |row| {
            memory_summary_from_row(row)
        })
        .optional()
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let Some((summary, raw_file, raw_line, raw_offset)) = row else {
        return Err(Error::NotFound(NotFound::Memory {
            tool: tool.as_str().to_string(),
            project: project.map(str::to_string),
            name: name.to_string(),
        }));
    };

    let payload = payload_for_raw_pointer(&raw_file, raw_line, raw_offset)?;
    let content = payload
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(MemoryFileContent {
        tool: summary.tool,
        project: summary.project,
        name: summary.name,
        native_path: summary.native_path,
        size: summary.size,
        modified_at: summary.modified_at,
        captured_at: summary.captured_at,
        session_id: summary.session_id,
        raw_file: raw_file.to_string(),
        raw_line,
        raw_offset,
        content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{index_once, init_home, search_history};
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// Serialize env-mutating tests (set_var is process-global) and restore
    /// the prior values on drop.
    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        vars: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, &std::ffi::OsStr)]) -> EnvGuard {
            static LOCK: Mutex<()> = Mutex::new(());
            let _lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = vars
                .iter()
                .map(|(name, _)| (*name, std::env::var_os(name)))
                .collect();
            for (name, value) in vars {
                std::env::set_var(name, value);
            }
            EnvGuard {
                _lock,
                vars: previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.vars {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    #[test]
    fn discovery_honors_tool_env_roots_and_skips_hidden() {
        let temp = tempdir().unwrap();
        let claude_config = temp.path().join("claude-config");
        let projects = claude_config.join("projects");
        let memory_dir = projects.join("-Users-me-project").join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(memory_dir.join("note.md"), "# note").unwrap();
        fs::write(memory_dir.join(".hidden.md"), "hidden").unwrap();
        fs::create_dir_all(memory_dir.join("sub")).unwrap();
        fs::write(memory_dir.join("sub").join("deep.md"), "deep").unwrap();

        let codex_home = temp.path().join("codex");
        let codex_memories = codex_home.join("memories");
        fs::create_dir_all(&codex_memories).unwrap();
        fs::write(codex_memories.join("prefs.md"), "prefs").unwrap();

        let _guard = EnvGuard::set(&[
            ("HOME", temp.path().as_os_str()),
            ("CLAUDE_CONFIG_DIR", claude_config.as_os_str()),
            ("CODEX_HOME", codex_home.as_os_str()),
        ]);

        let claude = discover_memory_files(Tool::Claude).unwrap();
        let mut names: Vec<_> = claude
            .iter()
            .map(|meta| (meta.name.clone(), meta.project.clone()))
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                ("deep.md".to_string(), Some("-Users-me-project".to_string())),
                ("note.md".to_string(), Some("-Users-me-project".to_string())),
            ]
        );

        let codex = discover_memory_files(Tool::Codex).unwrap();
        assert_eq!(codex.len(), 1);
        assert_eq!(codex[0].name, "prefs.md");
        assert_eq!(codex[0].project, None);

        assert!(discover_memory_files(Tool::Opencode).unwrap().is_empty());
    }

    #[test]
    fn sync_appends_then_dedupes_unchanged_and_indexes() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        let claude_config = temp.path().join("claude-config");
        let memory_dir = claude_config
            .join("projects")
            .join("-Users-me-project")
            .join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        let memory_file = memory_dir.join("MEMORY.md");
        fs::write(&memory_file, "# remember the salt is immutable").unwrap();
        let _guard = EnvGuard::set(&[
            ("HOME", temp.path().as_os_str()),
            ("CLAUDE_CONFIG_DIR", claude_config.as_os_str()),
            ("CODEX_HOME", temp.path().join("codex").as_os_str()),
        ]);

        let first = sync_memory(&home, Tool::Claude).unwrap();
        assert_eq!(first.discovered, 1);
        assert_eq!(first.appended, 1);

        // Unchanged file: no new event.
        let second = sync_memory(&home, Tool::Claude).unwrap();
        assert_eq!(second.discovered, 1);
        assert_eq!(second.appended, 0);

        // Changed file: one new event appended (append-only history).
        fs::write(&memory_file, "# remember the salt is immutable and frozen").unwrap();
        let third = sync_memory(&home, Tool::Claude).unwrap();
        assert_eq!(third.discovered, 1);
        assert_eq!(third.appended, 1);

        index_once(&home).unwrap();

        let memories = list_memories(&home, Some(Tool::Claude), 10).unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].name, "MEMORY.md");
        assert_eq!(memories[0].project.as_deref(), Some("-Users-me-project"));
        assert_eq!(memories[0].session_id, "-Users-me-project");
        assert_eq!(memories[0].native_path, memory_file.display().to_string());

        let content =
            get_memory(&home, Tool::Claude, Some("-Users-me-project"), "MEMORY.md").unwrap();
        assert!(content.content.contains("frozen"));
        assert_eq!(content.raw_line, 2);
        assert!(content.raw_file.ends_with("claude_-Users-me-project.jsonl"));
    }

    #[test]
    fn memory_events_are_searchable_and_distinct_from_sessions() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        let codex_home = temp.path().join("codex");
        let memories = codex_home.join("memories");
        fs::create_dir_all(&memories).unwrap();
        fs::write(memories.join("prefs.md"), "prefer cargo test over miri").unwrap();
        let _guard = EnvGuard::set(&[
            ("HOME", temp.path().as_os_str()),
            ("CODEX_HOME", codex_home.as_os_str()),
            ("CLAUDE_CONFIG_DIR", temp.path().join("claude").as_os_str()),
        ]);

        sync_memory(&home, Tool::Codex).unwrap();
        index_once(&home).unwrap();

        let hits = search_history(&home, "miri", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].canonical_type, "memory.file");
        assert_eq!(hits[0].tool, Tool::Codex);
        assert_eq!(hits[0].session_id, "memories");

        // Memory pseudo-sessions are not sessions: list_sessions stays clean.
        let sessions = crate::list_sessions(&home, None, None, None, 10).unwrap();
        assert!(sessions.is_empty());
    }

    #[test]
    fn get_memory_rejects_wrong_project_identity() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        let claude_config = temp.path().join("claude-config");
        let memory_dir = claude_config
            .join("projects")
            .join("-Users-me-project")
            .join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(memory_dir.join("MEMORY.md"), "content").unwrap();
        let _guard = EnvGuard::set(&[
            ("HOME", temp.path().as_os_str()),
            ("CLAUDE_CONFIG_DIR", claude_config.as_os_str()),
            ("CODEX_HOME", temp.path().join("codex").as_os_str()),
        ]);

        sync_memory(&home, Tool::Claude).unwrap();
        index_once(&home).unwrap();

        let result = get_memory(
            &home,
            Tool::Claude,
            Some("-Users-other-project"),
            "MEMORY.md",
        );
        assert!(matches!(result, Err(Error::NotFound(_))));
    }

    #[test]
    fn sync_reports_empty_for_tools_without_memory_folders() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let _guard = EnvGuard::set(&[
            ("HOME", temp.path().as_os_str()),
            ("CLAUDE_CONFIG_DIR", temp.path().join("claude").as_os_str()),
            ("CODEX_HOME", temp.path().join("codex").as_os_str()),
        ]);

        let report = sync_memory(&home, Tool::Opencode).unwrap();
        assert_eq!(report.discovered, 0);
        assert_eq!(report.appended, 0);
    }

    /// The pre-memory `events` table definition (source/canonical_type CHECKs
    /// without the memory values). Kept in the test so the migration is proven
    /// against the shape an existing install actually has.
    const LEGACY_EVENTS_TABLE: &str = r#"
CREATE TABLE events (
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
    source IN ('hook', 'event_stream', 'transcript_tail', 'sdk_session_store', 'backfill', 'exec_json', 'app_server')
  ),
  source_event_type TEXT NOT NULL,
  source_event_id TEXT,
  tool_invocation_id TEXT,
  canonical_type TEXT NOT NULL CHECK (
    canonical_type IN (
      'session.started', 'session.resumed', 'session.ended', 'user.message',
      'assistant.delta', 'assistant.message', 'tool.call', 'tool.result',
      'permission.requested', 'permission.replied', 'file.changed',
      'compaction.before', 'compaction.after', 'source.discontinuity', 'error'
    )
  ),
  sequence INTEGER,
  raw_file TEXT NOT NULL,
  raw_line INTEGER,
  raw_offset INTEGER,
  payload_json TEXT,
  payload_ref TEXT,
  searchable_text TEXT NOT NULL DEFAULT '',
  compaction_state TEXT NOT NULL DEFAULT 'unknown'
);
"#;

    #[test]
    fn legacy_index_rebuilds_events_table_for_memory_without_losing_rows() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let db_path = home.join("index").join("harness.db");
        fs::create_dir_all(db_path.parent().unwrap()).unwrap();

        // Build a pre-memory store: sessions + the legacy events table plus the
        // derived tables the open path indexes against, with one real event.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
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
             );",
        )
        .unwrap();
        conn.execute_batch(LEGACY_EVENTS_TABLE).unwrap();
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
        drop(conn);

        // First open runs the migration (events CHECK gains memory.file).
        let conn = crate::open_index(&db_path).unwrap();
        let sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'events'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(sql.contains("'memory.file'"), "{sql}");
        assert!(sql.contains("'memory_sync'"), "{sql}");

        // The legacy row survived with its id, and autoincrement continues.
        let (count, max_id): (i64, i64) = conn
            .query_row("SELECT COUNT(*), MAX(id) FROM events", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert_eq!((count, max_id), (1, 1));
        let migrated_at: Option<String> = conn
            .query_row(
                "SELECT applied_at FROM schema_migrations WHERE version = 2",
                [],
                |row| row.get(0),
            )
            .optional()
            .unwrap();
        assert!(migrated_at.is_some());

        // A memory event now inserts cleanly through the new CHECK.
        conn.execute_batch(
            "INSERT INTO sessions(tool, session_id, filename_session_id, started_at, updated_at, raw_file)
             VALUES ('codex', 'memories', 'memories', '2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z', 'raw/codex/codex_memories.jsonl');
             INSERT INTO events(
               tool, session_id, dedupe_key, schema_version, captured_at, source,
               source_event_type, canonical_type, raw_file, raw_line, searchable_text, compaction_state,
               payload_json
             ) VALUES (
               'codex', 'memories', 'sha256:memory-event', 1, '2026-01-02T00:00:00Z', 'memory_sync',
               'memory.file', 'memory.file', 'raw/codex/codex_memories.jsonl', 1, 'note', 'none',
               '{\"name\": \"note.md\", \"content\": \"note\"}'
             );",
        )
        .unwrap();
        let id: i64 = conn
            .query_row(
                "SELECT id FROM events WHERE dedupe_key = 'sha256:memory-event'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(id, 2);
        drop(conn);

        // Reopening is a no-op: the migration does not run twice.
        let conn = crate::open_index(&db_path).unwrap();
        let migration_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(migration_count, 1);
    }
}
