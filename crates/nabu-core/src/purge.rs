//! Store purge operations: per-session, time-windowed, and full-store removal,
//! with the closed artifact allowlist and symlink-safe (no-follow) deletion.

use crate::{
    canonical_raw_path, chmod, directory_size, init_home, normalize_date_or_duration, open_index,
    payload_for_raw_pointer, recalculate_all_session_counts, remove_dedupe_sidecar_for_raw_file,
    search_document_for_event, CanonicalType, Error, EventEnvelope, PurgeAction, PurgeAllArtifact,
    PurgeAllOptions, PurgeAllReport, PurgeReport, PurgeTier, Result, Tool,
};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// The complete, closed set of top-level entries nabu creates under a home.
/// A full purge only ever touches these; anything else is foreign and untouched.
pub(crate) const PURGE_KNOWN_ENTRIES: [&str; 9] = [
    "raw",
    "spool",
    "index",
    "checkpoints",
    "blobs",
    "logs",
    "backups",
    "models",
    "config.toml",
];

pub fn purge_session(home: &Path, tool: Tool, session_id: &str) -> Result<PurgeReport> {
    let raw_file = canonical_raw_path(home, tool, session_id);
    let indexed_events_removed = delete_indexed_events(
        home,
        "tool = ? AND session_id = ?",
        vec![
            SqlValue::Text(tool.as_str().to_string()),
            SqlValue::Text(session_id.to_string()),
        ],
    )?;

    let raw_files_removed = if raw_file.exists() {
        fs::remove_file(&raw_file).map_err(|source| Error::Io {
            path: raw_file.clone(),
            source,
        })?;
        remove_dedupe_sidecar_for_raw_file(&raw_file)?;
        1
    } else {
        0
    };

    Ok(PurgeReport {
        raw_files_removed,
        indexed_events_removed,
        sessions_removed: usize::from(indexed_events_removed > 0 || raw_files_removed > 0),
    })
}

pub fn purge_before(home: &Path, before: &str) -> Result<PurgeReport> {
    let before = normalize_date_or_duration(before, "before")?;
    let indexed_events_removed = delete_indexed_events(
        home,
        "captured_at < ?",
        vec![SqlValue::Text(before.clone())],
    )?;

    let mut raw_files_removed = 0usize;
    for tool in Tool::all() {
        let raw_dir = home.join("raw").join(tool.as_str());
        if !raw_dir.exists() {
            continue;
        }
        for entry in fs::read_dir(&raw_dir).map_err(|source| Error::Io {
            path: raw_dir.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| Error::Io {
                path: raw_dir.clone(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            if rewrite_raw_file_after(&path, &before)? {
                raw_files_removed += 1;
            }
        }
    }

    Ok(PurgeReport {
        raw_files_removed,
        indexed_events_removed,
        sessions_removed: 0,
    })
}

/// Remove every nabu-created artifact under `home` (the store side of a
/// full uninstall; hook removal is orchestrated separately by the CLI). Only
/// the closed allowlist in [`PURGE_KNOWN_ENTRIES`] is ever touched — the home
/// directory itself and any foreign files are left in place.
///
/// Safety:
/// - refuses to operate on the filesystem root or the user's `$HOME`;
/// - refuses a path that carries no nabu marker (config/index/raw), so a
///   mistyped `--home` errors instead of deleting;
/// - never follows symlinks when removing (the model dir may be a symlink);
/// - idempotent: a missing home, or already-removed artifacts, are not errors.
pub fn purge_all(home: &Path, options: PurgeAllOptions) -> Result<PurgeAllReport> {
    assert_safe_purge_home(home)?;

    // Idempotent: nothing to purge if the home was never created.
    if fs::symlink_metadata(home).is_err() {
        return Ok(PurgeAllReport {
            home: home.to_path_buf(),
            dry_run: options.dry_run,
            artifacts: Vec::new(),
            unknown_entries: Vec::new(),
            bytes_reclaimed: 0,
            bytes_in_scope: 0,
            authoritative_in_scope: false,
        });
    }

    // Refuse to delete from a directory that does not look like a store.
    let has_marker = home.join("config.toml").exists()
        || home.join("index").join("harness.db").exists()
        || home.join("raw").exists();
    if !has_marker {
        return Err(Error::Validation(format!(
            "{} does not look like a nabu home (no config.toml, index, or raw/); refusing to purge",
            home.display()
        )));
    }

    let plan = [
        ("raw", PurgeTier::Authoritative, true),
        ("index", PurgeTier::Derived, true),
        ("spool", PurgeTier::Derived, true),
        ("checkpoints", PurgeTier::Derived, true),
        ("blobs", PurgeTier::Derived, true),
        ("logs", PurgeTier::Derived, true),
        ("backups", PurgeTier::Derived, true),
        ("models", PurgeTier::Model, !options.keep_model),
        ("config.toml", PurgeTier::Config, !options.keep_config),
    ];

    let mut artifacts = Vec::with_capacity(plan.len());
    let mut bytes_reclaimed = 0u64;
    let mut bytes_in_scope = 0u64;
    let mut authoritative_in_scope = false;

    for (name, tier, in_scope) in plan {
        let path = home.join(name);
        let meta = fs::symlink_metadata(&path).ok();
        let existed = meta.is_some();
        let is_symlink = meta
            .as_ref()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        // Do not traverse a symlink target for size accounting.
        let bytes = match (existed, is_symlink) {
            (false, _) => 0,
            (true, true) => 0,
            (true, false) => directory_size(&path)?,
        };

        let action = if !existed {
            PurgeAction::Absent
        } else if !in_scope {
            PurgeAction::Preserved
        } else if options.dry_run {
            bytes_in_scope = bytes_in_scope.saturating_add(bytes);
            if matches!(tier, PurgeTier::Authoritative) {
                authoritative_in_scope = true;
            }
            PurgeAction::WouldRemove
        } else {
            remove_path_no_follow(&path, is_symlink)?;
            bytes_in_scope = bytes_in_scope.saturating_add(bytes);
            bytes_reclaimed = bytes_reclaimed.saturating_add(bytes);
            if matches!(tier, PurgeTier::Authoritative) {
                authoritative_in_scope = true;
            }
            PurgeAction::Removed
        };

        artifacts.push(PurgeAllArtifact {
            name: name.to_string(),
            path,
            tier,
            bytes,
            action,
        });
    }

    // Surface any foreign entries; never remove them.
    let mut unknown_entries = Vec::new();
    for entry in fs::read_dir(home).map_err(|source| Error::Io {
        path: home.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| Error::Io {
            path: home.to_path_buf(),
            source,
        })?;
        let file_name = entry.file_name();
        let is_known = file_name
            .to_str()
            .map(|name| PURGE_KNOWN_ENTRIES.contains(&name))
            .unwrap_or(false);
        if !is_known {
            unknown_entries.push(entry.path());
        }
    }
    unknown_entries.sort();

    Ok(PurgeAllReport {
        home: home.to_path_buf(),
        dry_run: options.dry_run,
        artifacts,
        unknown_entries,
        bytes_reclaimed,
        bytes_in_scope,
        authoritative_in_scope,
    })
}

/// Refuse purge targets that would be catastrophic if a `--home` were mistyped.
fn assert_safe_purge_home(home: &Path) -> Result<()> {
    if home.parent().is_none() {
        return Err(Error::Validation(format!(
            "refusing to purge filesystem root {}",
            home.display()
        )));
    }
    if let Some(user_home) = env::var_os("HOME").map(PathBuf::from) {
        if same_path(home, &user_home) {
            return Err(Error::Validation(format!(
                "refusing to purge your home directory {}; set --home to the nabu store",
                home.display()
            )));
        }
    }
    Ok(())
}

/// Equal-path test that resolves symlinks/`.`/`..` when both sides exist, and
/// falls back to a literal comparison otherwise.
fn same_path(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}

/// Remove a file, directory tree, or symlink without following the link. A
/// symlinked path is unlinked (never its target); a missing path is a no-op.
fn remove_path_no_follow(path: &Path, is_symlink: bool) -> Result<()> {
    let result = if is_symlink {
        fs::remove_file(path)
    } else {
        match fs::symlink_metadata(path) {
            Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
            Ok(_) => fs::remove_file(path),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(Error::Io {
                    path: path.to_path_buf(),
                    source,
                })
            }
        }
    };
    match result {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(Error::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn rewrite_raw_file_after(path: &Path, before: &str) -> Result<bool> {
    let input = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let tmp_path = path.with_extension("jsonl.tmp");
    let rewrite_result = (|| -> Result<usize> {
        let mut reader = BufReader::new(input);
        let mut output = File::create(&tmp_path).map_err(|source| Error::Io {
            path: tmp_path.clone(),
            source,
        })?;
        let mut line = String::new();
        let mut kept = 0usize;

        loop {
            line.clear();
            let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
            if bytes == 0 {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            let envelope: EventEnvelope = serde_json::from_str(line.trim_end())?;
            if envelope.captured_at.as_str() >= before {
                output
                    .write_all(line.trim_end().as_bytes())
                    .map_err(|source| Error::Io {
                        path: tmp_path.clone(),
                        source,
                    })?;
                output.write_all(b"\n").map_err(|source| Error::Io {
                    path: tmp_path.clone(),
                    source,
                })?;
                kept += 1;
            }
        }
        output.flush().map_err(|source| Error::Io {
            path: tmp_path.clone(),
            source,
        })?;
        Ok(kept)
    })();
    let kept = match rewrite_result {
        Ok(kept) => kept,
        Err(error) => {
            let _ = fs::remove_file(&tmp_path);
            return Err(error);
        }
    };

    if kept == 0 {
        let _ = fs::remove_file(&tmp_path);
        fs::remove_file(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        remove_dedupe_sidecar_for_raw_file(path)?;
        return Ok(true);
    }

    fs::rename(&tmp_path, path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    chmod(path, 0o600)?;
    remove_dedupe_sidecar_for_raw_file(path)?;
    Ok(false)
}

fn delete_indexed_events(
    home: &Path,
    predicate: &str,
    predicate_params: Vec<SqlValue>,
) -> Result<usize> {
    init_home(home)?;
    let db_path = home.join("index").join("harness.db");
    let mut conn = open_index(&db_path)?;
    let tx = conn.transaction().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;

    let mut select_sql = String::from(
        "SELECT id, payload_json, tool, session_id, canonical_type, raw_file, raw_line, raw_offset
         FROM events WHERE ",
    );
    select_sql.push_str(predicate);
    let fts_rows = {
        let mut statement = tx.prepare(&select_sql).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
        let rows = statement
            .query_map(params_from_iter(predicate_params.clone()), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, Option<i64>>(7)?,
                ))
            })
            .map_err(|source| Error::Sqlite {
                path: db_path.clone(),
                source,
            })?;
        let mut fts_rows = Vec::new();
        for row in rows {
            fts_rows.push(row.map_err(|source| Error::Sqlite {
                path: db_path.clone(),
                source,
            })?);
        }
        fts_rows
    };

    for (
        event_id,
        payload_json,
        tool,
        session_id,
        canonical_type,
        raw_file,
        raw_line,
        raw_offset,
    ) in &fts_rows
    {
        let canonical_type_value = CanonicalType::from_str(canonical_type)?;
        let payload = match payload_json.as_deref() {
            Some(payload_json) => serde_json::from_str(payload_json)?,
            None => payload_for_raw_pointer(raw_file, *raw_line, *raw_offset)?,
        };
        let search_document = search_document_for_event(canonical_type_value, &payload);
        tx.execute(
            "INSERT INTO events_fts(events_fts, rowid, user_text, assistant_text, tool_intent, tool_output, metadata_text, tool, session_id, canonical_type, raw_file, raw_line, raw_offset)
             VALUES ('delete', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                event_id,
                &search_document.user_text,
                &search_document.assistant_text,
                &search_document.tool_intent,
                &search_document.tool_output,
                &search_document.metadata_text,
                tool,
                session_id,
                canonical_type,
                raw_file,
                raw_line,
                raw_offset,
            ],
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    }

    let mut delete_sql = String::from("DELETE FROM events WHERE ");
    delete_sql.push_str(predicate);
    tx.execute(&delete_sql, params_from_iter(predicate_params))
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    tx.execute(
        "DELETE FROM sessions
         WHERE NOT EXISTS (
           SELECT 1 FROM events
           WHERE events.tool = sessions.tool
             AND events.session_id = sessions.session_id
         )",
        [],
    )
    .map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    recalculate_all_session_counts(&tx, &db_path)?;
    tx.commit().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;

    Ok(fts_rows.len())
}
