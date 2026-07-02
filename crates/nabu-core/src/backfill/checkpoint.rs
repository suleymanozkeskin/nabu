//! Per-source backfill checkpoints and source-file metadata.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SourceCheckpoint {
    pub(crate) source_tool: Tool,
    pub(crate) source_kind: String,
    pub(crate) source_path: String,
    pub(crate) source_identity: Option<String>,
    pub(crate) session_id: String,
    pub(crate) byte_offset: u64,
    pub(crate) source_size: u64,
    pub(crate) source_mtime: Option<i64>,
    pub(crate) last_line_hash: Option<String>,
    pub(crate) last_successful_import_timestamp: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceFileMetadata {
    pub(crate) identity: Option<String>,
    pub(crate) size: u64,
    pub(crate) mtime: Option<i64>,
}

pub(crate) fn source_file_metadata(path: &Path) -> Result<SourceFileMetadata> {
    let metadata = fs::metadata(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(SourceFileMetadata {
        identity: source_file_identity(path, &metadata),
        size: metadata.len(),
        mtime: metadata
            .modified()
            .ok()
            .and_then(system_time_to_unix_seconds),
    })
}

#[cfg(unix)]
pub(crate) fn source_file_identity(_path: &Path, metadata: &fs::Metadata) -> Option<String> {
    use std::os::unix::fs::MetadataExt;

    Some(format!("{}:{}", metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
pub(crate) fn source_file_identity(path: &Path, _metadata: &fs::Metadata) -> Option<String> {
    fs::canonicalize(path)
        .ok()
        .map(|path| path.display().to_string())
}

pub(crate) fn system_time_to_unix_seconds(value: SystemTime) -> Option<i64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
}

pub(crate) fn load_checkpoint(
    home: &Path,
    tool: Tool,
    source_kind: &str,
    source_path: &Path,
) -> Result<Option<SourceCheckpoint>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    load_checkpoint_from_conn(&conn, &db_path, tool, source_kind, source_path)
}

pub(crate) fn load_checkpoint_from_conn(
    conn: &Connection,
    db_path: &Path,
    tool: Tool,
    source_kind: &str,
    source_path: &Path,
) -> Result<Option<SourceCheckpoint>> {
    conn.query_row(
        "SELECT
           source_tool,
           source_kind,
           source_path,
           source_identity,
           COALESCE(session_id, ''),
           byte_offset,
           source_size,
           source_mtime,
           last_line_hash,
           COALESCE(last_successful_import_timestamp, updated_at)
         FROM checkpoints
         WHERE source_tool = ?1 AND source_kind = ?2 AND source_path = ?3",
        (
            tool.as_str(),
            source_kind,
            source_path.display().to_string(),
        ),
        |row| {
            let source_tool: String = row.get(0)?;
            Ok(SourceCheckpoint {
                source_tool: Tool::from_str(&source_tool)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?,
                source_kind: row.get(1)?,
                source_path: row.get(2)?,
                source_identity: row.get(3)?,
                session_id: row.get(4)?,
                byte_offset: row.get::<_, i64>(5)?.max(0) as u64,
                source_size: row.get::<_, i64>(6)?.max(0) as u64,
                source_mtime: row.get(7)?,
                last_line_hash: row.get(8)?,
                last_successful_import_timestamp: row.get(9)?,
            })
        },
    )
    .optional()
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })
}

pub(crate) fn write_checkpoint(home: &Path, checkpoint: &SourceCheckpoint) -> Result<()> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    write_checkpoint_to_conn(&conn, &db_path, checkpoint)
}

pub(crate) fn write_checkpoint_to_conn(
    conn: &Connection,
    db_path: &Path,
    checkpoint: &SourceCheckpoint,
) -> Result<()> {
    conn.execute(
        "INSERT INTO checkpoints(
           source_tool,
           source_kind,
           source_path,
           source_identity,
           session_id,
           byte_offset,
           source_size,
           source_mtime,
           last_line_hash,
           last_successful_import_timestamp,
           updated_at
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)
         ON CONFLICT(source_tool, source_kind, source_path)
         DO UPDATE SET
           source_identity = excluded.source_identity,
           session_id = excluded.session_id,
           byte_offset = excluded.byte_offset,
           source_size = excluded.source_size,
           source_mtime = excluded.source_mtime,
           last_line_hash = excluded.last_line_hash,
           last_successful_import_timestamp = excluded.last_successful_import_timestamp,
           updated_at = excluded.updated_at",
        params![
            checkpoint.source_tool.as_str(),
            &checkpoint.source_kind,
            &checkpoint.source_path,
            checkpoint.source_identity.as_deref(),
            &checkpoint.session_id,
            checkpoint.byte_offset as i64,
            checkpoint.source_size as i64,
            checkpoint.source_mtime,
            checkpoint.last_line_hash.as_deref(),
            &checkpoint.last_successful_import_timestamp,
        ],
    )
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

pub(crate) fn last_line_hash(path: &Path) -> Result<Option<String>> {
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut last = None;
    for line in reader.lines() {
        let line = line.map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        last = Some(hash_line(&line));
    }
    Ok(last)
}

pub(crate) fn checkpoint_matches_source(
    path: &Path,
    checkpoint: &SourceCheckpoint,
) -> Result<bool> {
    let Some(expected_hash) = checkpoint.last_line_hash.as_ref() else {
        return Ok(true);
    };
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut offset = 0u64;
    let mut last_hash = None;

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        offset += bytes as u64;
        if offset <= checkpoint.byte_offset {
            last_hash = Some(hash_line(line.trim_end()));
        } else {
            break;
        }
    }

    Ok(last_hash.as_ref() == Some(expected_hash))
}

/// Bytes of `source_path` the raw-index checkpoint has already consumed, or 0
/// when the file was never indexed. The doctor uses this against the file's
/// current size to compute unindexed (capture-ahead-of-index) bytes without
/// touching any clock.
pub(crate) fn raw_index_checkpoint_offset(
    conn: &Connection,
    db_path: &Path,
    tool: Tool,
    source_path: &Path,
) -> Result<u64> {
    Ok(
        load_checkpoint_from_conn(conn, db_path, tool, "raw_jsonl", source_path)?
            .map(|checkpoint| checkpoint.byte_offset)
            .unwrap_or(0),
    )
}

/// Load-and-compare wrapper over [`checkpoint_is_current`]. Retained for tests
/// that assert the skip gate directly; the index pass loads the checkpoint once
/// and calls [`checkpoint_is_current`] so it can reuse it for tail resume.
#[cfg(test)]
pub(crate) fn raw_index_checkpoint_is_current(
    conn: &Connection,
    db_path: &Path,
    tool: Tool,
    source_path: &Path,
    source_meta: &SourceFileMetadata,
) -> Result<bool> {
    let checkpoint = load_checkpoint_from_conn(conn, db_path, tool, "raw_jsonl", source_path)?;
    Ok(checkpoint_is_current(checkpoint.as_ref(), source_meta))
}

/// Whether an already-loaded checkpoint fully covers the current file: same
/// identity, and the checkpoint consumed every byte the file now has (offset ==
/// size == recorded size) at the same mtime. This is the fast skip gate — a
/// `true` result means the file is unchanged since it was fully indexed. The
/// incremental path uses the same checkpoint to resume from the recorded offset
/// when this returns `false` only because the file grew (see the index module).
pub(crate) fn checkpoint_is_current(
    checkpoint: Option<&SourceCheckpoint>,
    source_meta: &SourceFileMetadata,
) -> bool {
    let Some(checkpoint) = checkpoint else {
        return false;
    };
    checkpoint.source_identity.as_deref() == source_meta.identity.as_deref()
        && checkpoint.byte_offset == source_meta.size
        && checkpoint.source_size == source_meta.size
        && checkpoint.source_mtime == source_meta.mtime
}

pub(crate) fn write_raw_index_checkpoint(
    conn: &Connection,
    db_path: &Path,
    tool: Tool,
    source_path: &Path,
    source_meta: SourceFileMetadata,
    raw_report: RawIndexFileReport,
) -> Result<()> {
    let now = OffsetDateTime::now_utc().format(&Rfc3339)?;
    let checkpoint = SourceCheckpoint {
        source_tool: tool,
        source_kind: "raw_jsonl".to_string(),
        source_path: source_path.display().to_string(),
        source_identity: source_meta.identity,
        session_id: raw_index_checkpoint_session_id(tool, source_path),
        byte_offset: raw_report.bytes_read,
        source_size: source_meta.size,
        source_mtime: source_meta.mtime,
        last_line_hash: raw_report.last_line_hash,
        last_successful_import_timestamp: now,
    };
    write_checkpoint_to_conn(conn, db_path, &checkpoint)
}

pub(crate) fn raw_index_checkpoint_session_id(tool: Tool, source_path: &Path) -> String {
    let Some(stem) = source_path.file_stem().and_then(|value| value.to_str()) else {
        return source_path_fallback_session_id(source_path);
    };
    let prefix = format!("{}_", tool.as_str());
    stem.strip_prefix(&prefix)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| source_path_fallback_session_id(source_path))
}

pub(crate) fn source_kind_for(tool: Tool, source_path: &Path) -> &'static str {
    match tool {
        Tool::Codex | Tool::Claude => "transcript",
        Tool::Opencode => {
            if source_path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|name| name.contains("server") || name.contains("api"))
                .unwrap_or(false)
            {
                "api_export"
            } else {
                "raw_jsonl"
            }
        }
    }
}

pub(crate) fn detect_deleted_sources(
    home: &Path,
    source_root: &Path,
    report: &mut BackfillReport,
) -> Result<()> {
    for checkpoint in checkpoints_under_root(home, source_root)? {
        let source_path = PathBuf::from(&checkpoint.source_path);
        if !source_path.starts_with(source_root) || source_path.exists() {
            continue;
        }
        append_discontinuity(
            home,
            checkpoint.source_tool,
            &checkpoint.session_id,
            "source.deleted",
            &source_path,
            checkpoint.byte_offset,
            0,
        )?;
        delete_checkpoint(home, &checkpoint)?;
        report.discontinuities += 1;
    }
    Ok(())
}

pub(crate) fn checkpoints_under_root(
    home: &Path,
    source_root: &Path,
) -> Result<Vec<SourceCheckpoint>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let prefix = source_root.display().to_string();
    let mut statement = conn
        .prepare(
            "SELECT
               source_tool,
               source_kind,
               source_path,
               source_identity,
               COALESCE(session_id, ''),
               byte_offset,
               source_size,
               source_mtime,
               last_line_hash,
               COALESCE(last_successful_import_timestamp, updated_at)
             FROM checkpoints
             WHERE source_path = ?1 OR source_path LIKE ?2",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let rows = statement
        .query_map((prefix.clone(), format!("{prefix}/%")), |row| {
            let source_tool: String = row.get(0)?;
            Ok(SourceCheckpoint {
                source_tool: Tool::from_str(&source_tool)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?,
                source_kind: row.get(1)?,
                source_path: row.get(2)?,
                source_identity: row.get(3)?,
                session_id: row.get(4)?,
                byte_offset: row.get::<_, i64>(5)?.max(0) as u64,
                source_size: row.get::<_, i64>(6)?.max(0) as u64,
                source_mtime: row.get(7)?,
                last_line_hash: row.get(8)?,
                last_successful_import_timestamp: row.get(9)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let mut checkpoints = Vec::new();
    for row in rows {
        checkpoints.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    Ok(checkpoints)
}

pub(crate) fn delete_checkpoint(home: &Path, checkpoint: &SourceCheckpoint) -> Result<()> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    conn.execute(
        "DELETE FROM checkpoints
         WHERE source_tool = ?1 AND source_kind = ?2 AND source_path = ?3",
        (
            checkpoint.source_tool.as_str(),
            checkpoint.source_kind.as_str(),
            checkpoint.source_path.as_str(),
        ),
    )
    .map_err(|source| Error::Sqlite {
        path: db_path,
        source,
    })?;
    Ok(())
}
