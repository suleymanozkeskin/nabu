//! Health/doctor checks: storage and index liveness, deep integrity, coverage
//! and storage-footprint summaries, and the staged doctor report.

use crate::{
    canonical_raw_path, latest_event, open_index, raw_index_checkpoint_offset,
    session_id_from_source_path, table_count, table_exists, CaptureFreshness, CoverageSummary,
    DoctorCheck, DoctorReport, DoctorStats, Error, IndexFreshness, Result, StorageFootprint,
    StoredEvent, Tool, MAX_DIRECTORY_SIZE_DEPTH, SEMANTIC_VECTOR_DIMENSIONS,
};
use rusqlite::Connection;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const MAX_MISSING_SESSION_IDS: usize = 20;
const MAX_CAPTURE_SCAN_ERRORS: usize = 3;

pub fn doctor(home: &Path) -> DoctorReport {
    doctor_with_options(home, false)
}

/// The sub-checks a doctor run performs, in display order. Emitted to the
/// progress callback so callers (e.g. the wizard) can show a live checklist
/// instead of one opaque pause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoctorStage {
    Storage,
    Index,
    Backfill,
    Capture,
    Coverage,
    Footprint,
    LatestEvents,
}

/// Progress events for a doctor run: each stage emits [`Started`] before the
/// work begins and [`Finished`] when it completes, so UIs can animate in-flight
/// work instead of freezing until the stage returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoctorStageEvent {
    Started(DoctorStage),
    Finished(DoctorStage, bool),
}

/// Controls which doctor work runs. The wizard uses `metrics: false` so health
/// stays a quick liveness check; the CLI/MCP keep `metrics: true` for full
/// footprint/freshness detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoctorOptions {
    pub deep: bool,
    /// When true, also compute coverage counts, the storage-footprint walk,
    /// latest-event rows, and per-tool index freshness (expensive on large
    /// stores). When false, those fields are zero/empty defaults.
    pub metrics: bool,
}

impl Default for DoctorOptions {
    fn default() -> Self {
        Self {
            deep: false,
            metrics: true,
        }
    }
}

pub fn doctor_with_options(home: &Path, deep: bool) -> DoctorReport {
    doctor_with_progress(
        home,
        DoctorOptions {
            deep,
            metrics: true,
        },
        &mut |_| {},
    )
}

/// Like [`doctor_with_options`], but streams [`DoctorStageEvent`]s so callers
/// can show in-progress work. Boolean stages report their pass/fail on
/// [`DoctorStageEvent::Finished`]; derived metric stages always finish `true`.
pub fn doctor_with_progress(
    home: &Path,
    options: DoctorOptions,
    on_stage: &mut dyn FnMut(DoctorStageEvent),
) -> DoctorReport {
    let mut run = |stage: DoctorStage, work: &mut dyn FnMut() -> bool| {
        on_stage(DoctorStageEvent::Started(stage));
        let ok = work();
        on_stage(DoctorStageEvent::Finished(stage, ok));
        ok
    };

    let storage_ok = run(DoctorStage::Storage, &mut || storage_is_healthy(home));

    let index_ok = run(DoctorStage::Index, &mut || {
        if options.deep {
            index_integrity_is_healthy(home)
        } else {
            index_structure_is_healthy(home)
        }
    });

    let backfill_ok = run(DoctorStage::Backfill, &mut || backfill_is_healthy(home));

    let mut capture_status = BTreeMap::new();
    let capture_ok = run(DoctorStage::Capture, &mut || {
        capture_status = capture_freshness(home);
        capture_status.values().all(|status| !status.stale)
    });

    let (coverage, storage_footprint, latest_captured_events, index_freshness) = if options.metrics
    {
        let coverage = {
            on_stage(DoctorStageEvent::Started(DoctorStage::Coverage));
            let value = coverage_summary(home);
            on_stage(DoctorStageEvent::Finished(DoctorStage::Coverage, true));
            value
        };
        let storage_footprint = {
            on_stage(DoctorStageEvent::Started(DoctorStage::Footprint));
            let value = storage_footprint(home);
            on_stage(DoctorStageEvent::Finished(DoctorStage::Footprint, true));
            value
        };
        let (latest_captured_events, index_freshness) = {
            on_stage(DoctorStageEvent::Started(DoctorStage::LatestEvents));
            let latest = latest_events_for_doctor(home);
            let freshness = index_freshness_for_doctor(home);
            on_stage(DoctorStageEvent::Finished(DoctorStage::LatestEvents, true));
            (latest, freshness)
        };
        (
            coverage,
            storage_footprint,
            latest_captured_events,
            index_freshness,
        )
    } else {
        (
            CoverageSummary {
                checkpointed_sources: 0,
                captured_sessions: 0,
                captured_events: 0,
            },
            StorageFootprint {
                raw_bytes: 0,
                index_bytes: 0,
                vectors_bytes: 0,
                spool_bytes: 0,
                blobs_bytes: 0,
                models_bytes: 0,
                canonical_total: 0,
                derived_total: 0,
                total_bytes: 0,
            },
            BTreeMap::new(),
            BTreeMap::new(),
        )
    };

    let stats = if options.deep {
        index_stats(home).ok()
    } else {
        None
    };

    DoctorReport {
        level: if options.deep {
            "deep"
        } else if options.metrics {
            "fast"
        } else {
            "wizard"
        }
        .to_string(),
        integrity: if options.deep { "full" } else { "structural" }.to_string(),
        storage: DoctorCheck {
            ok: storage_ok,
            message: if storage_ok {
                "required storage paths are present".to_string()
            } else {
                "one or more required storage paths are missing".to_string()
            },
        },
        index: DoctorCheck {
            ok: index_ok,
            message: if index_ok {
                if options.deep {
                    "sqlite integrity_check returned ok".to_string()
                } else {
                    "index opens and core tables are present".to_string()
                }
            } else {
                "sqlite index is missing or unhealthy".to_string()
            },
        },
        backfill: DoctorCheck {
            ok: backfill_ok,
            message: if backfill_ok {
                "checkpoint rows are present".to_string()
            } else {
                "no checkpoint rows found".to_string()
            },
        },
        capture: DoctorCheck {
            ok: capture_ok,
            message: capture_check_message(&capture_status),
        },
        coverage,
        storage_footprint,
        latest_captured_events,
        index_freshness,
        capture_freshness: capture_status,
        stats,
    }
}

fn capture_check_message(freshness: &BTreeMap<String, CaptureFreshness>) -> String {
    let Some(codex) = freshness.get("codex") else {
        return "native session coverage was not checked".to_string();
    };
    match (
        &codex.scan_error,
        codex.missing_sessions,
        codex.source_sessions,
    ) {
        (Some(error), _, _) => format!("native Codex session scan failed: {error}"),
        (None, 0, 0) => "no native Codex sessions found".to_string(),
        (None, 0, _) => "native Codex sessions have canonical raw capture".to_string(),
        (None, missing, _) => {
            format!("{missing} native Codex session(s) have no canonical raw capture")
        }
    }
}

/// Compare native Codex session IDs against non-empty canonical raw files.
/// This checks the source-to-raw boundary that raw-vs-index freshness cannot
/// see. Missing IDs are capped, but the counts cover all discovered sessions.
pub fn capture_freshness(home: &Path) -> BTreeMap<String, CaptureFreshness> {
    let status = match codex_transcript_roots() {
        Ok(roots) => codex_capture_freshness_from_roots(home, &roots),
        Err(error) => CaptureFreshness {
            source_sessions: 0,
            captured_sessions: 0,
            missing_sessions: 0,
            missing_session_ids: Vec::new(),
            scan_error: Some(error.to_string()),
            stale: true,
        },
    };
    BTreeMap::from([("codex".to_string(), status)])
}

fn codex_transcript_roots() -> Result<Vec<PathBuf>> {
    let codex_home = match env::var_os("CODEX_HOME") {
        Some(path) => PathBuf::from(path),
        None => env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or(Error::HomeUnavailable)?
            .join(".codex"),
    };
    Ok(vec![
        codex_home.join("sessions"),
        codex_home.join("archived_sessions"),
    ])
}

fn codex_capture_freshness_from_roots(home: &Path, roots: &[PathBuf]) -> CaptureFreshness {
    let mut source_sessions_by_id = BTreeMap::new();
    let mut errors = Vec::new();
    for root in roots {
        collect_codex_session_ids(root, &mut source_sessions_by_id, &mut errors);
    }

    let mut missing: Vec<(String, SystemTime)> = source_sessions_by_id
        .iter()
        .filter(|(session_id, _)| {
            fs::metadata(canonical_raw_path(home, Tool::Codex, session_id.as_str()))
                .map(|metadata| metadata.len() == 0)
                .unwrap_or(true)
        })
        .map(|(session_id, modified)| (session_id.clone(), *modified))
        .collect();
    missing.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| right.0.cmp(&left.0)));
    let source_sessions = source_sessions_by_id.len();
    let missing_sessions = missing.len();
    let captured_sessions = source_sessions.saturating_sub(missing_sessions);
    let scan_error = (!errors.is_empty()).then(|| errors.join("; "));

    CaptureFreshness {
        source_sessions,
        captured_sessions,
        missing_sessions,
        missing_session_ids: missing
            .into_iter()
            .take(MAX_MISSING_SESSION_IDS)
            .map(|(session_id, _)| session_id)
            .collect(),
        stale: missing_sessions > 0 || scan_error.is_some(),
        scan_error,
    }
}

fn collect_codex_session_ids(
    directory: &Path,
    sessions_by_id: &mut BTreeMap<String, SystemTime>,
    errors: &mut Vec<String>,
) {
    if !directory.exists() {
        return;
    }
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            push_capture_scan_error(errors, directory, &error);
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                push_capture_scan_error(errors, directory, &error);
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => {
                push_capture_scan_error(errors, &path, &error);
                continue;
            }
        };
        match (
            file_type.is_dir(),
            path.extension().and_then(|value| value.to_str()),
        ) {
            (true, _) => collect_codex_session_ids(&path, sessions_by_id, errors),
            (false, Some("jsonl")) => {
                let Some(session_id) = session_id_from_source_path(&path) else {
                    continue;
                };
                let modified = match entry.metadata().and_then(|metadata| metadata.modified()) {
                    Ok(modified) => modified,
                    Err(error) => {
                        push_capture_scan_error(errors, &path, &error);
                        continue;
                    }
                };
                sessions_by_id
                    .entry(session_id)
                    .and_modify(|current| *current = (*current).max(modified))
                    .or_insert(modified);
            }
            _ => {}
        }
    }
}

fn push_capture_scan_error(errors: &mut Vec<String>, path: &Path, error: &std::io::Error) {
    if errors.len() < MAX_CAPTURE_SCAN_ERRORS {
        errors.push(format!("{}: {error}", path.display()));
    }
}

fn latest_events_for_doctor(home: &Path) -> BTreeMap<String, Option<StoredEvent>> {
    let mut events = BTreeMap::new();
    for tool in Tool::all() {
        events.insert(
            tool.as_str().to_string(),
            latest_event(home, tool).ok().flatten(),
        );
    }
    events
}

/// Per-tool raw-vs-index freshness, measured in bytes. For each
/// `raw/<tool>/*.jsonl` file, the index checkpoint records how many bytes have
/// been consumed; the file's current size minus that offset is the unindexed
/// remainder. Summed per tool, a non-zero remainder means capture is ahead of
/// the index. This compares like to like (raw bytes on disk vs. checkpointed
/// bytes), unlike a filesystem-mtime-vs-`captured_at` comparison which diverges
/// for backfilled history whose events keep their original timestamps.
/// Per-tool index freshness without running the full doctor pipeline. The human
/// `nabu doctor` printer uses this to show freshness lines without paying for the
/// storage-footprint walk, coverage counts, and latest-event queries it does not
/// display.
pub fn index_freshness(home: &Path) -> BTreeMap<String, IndexFreshness> {
    index_freshness_for_doctor(home)
}

fn index_freshness_for_doctor(home: &Path) -> BTreeMap<String, IndexFreshness> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).ok();
    let mut freshness = BTreeMap::new();
    for tool in Tool::all() {
        freshness.insert(
            tool.as_str().to_string(),
            tool_index_freshness(conn.as_ref(), &db_path, home, tool),
        );
    }
    freshness
}

fn tool_index_freshness(
    conn: Option<&Connection>,
    db_path: &Path,
    home: &Path,
    tool: Tool,
) -> IndexFreshness {
    let raw_dir = home.join("raw").join(tool.as_str());
    let mut raw_bytes = 0u64;
    let mut indexed_bytes = 0u64;
    let mut pending_files = 0usize;

    if let Ok(entries) = fs::read_dir(&raw_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(size) = entry.metadata().map(|meta| meta.len()) else {
                continue;
            };
            raw_bytes += size;
            // Clamp to size: a rotated/truncated file can leave a checkpoint
            // offset beyond the current length, which would otherwise underflow
            // the unindexed total.
            let consumed = conn
                .and_then(|conn| raw_index_checkpoint_offset(conn, db_path, tool, &path).ok())
                .unwrap_or(0)
                .min(size);
            indexed_bytes += consumed;
            if consumed < size {
                pending_files += 1;
            }
        }
    }

    let unindexed_bytes = raw_bytes.saturating_sub(indexed_bytes);
    IndexFreshness {
        raw_bytes,
        indexed_bytes,
        unindexed_bytes,
        pending_files,
        stale: unindexed_bytes > 0,
    }
}

fn storage_is_healthy(home: &Path) -> bool {
    [
        home.join("raw"),
        home.join("raw").join("codex"),
        home.join("raw").join("claude"),
        home.join("raw").join("opencode"),
        home.join("spool"),
        home.join("spool").join("dedupe"),
        home.join("checkpoints"),
        home.join("blobs").join("sha256"),
        home.join("logs"),
    ]
    .into_iter()
    .all(|path| path.is_dir())
}

/// Fast structural liveness check, O(1) in database size.
///
/// Proves the index file opens, is a schema-initialized nabu database, and has
/// its core tables — without scanning every page. This is the right cost for the
/// default `doctor`, the wizard health screen, and the MCP `history_doctor`
/// default: those need "is the index present and usable", answered in
/// milliseconds. Page-level integrity (`PRAGMA integrity_check`, O(database
/// size) — minutes on a multi-GB index) is reserved for the explicit deep tier.
fn index_structure_is_healthy(home: &Path) -> bool {
    let db_path = home.join("index").join("harness.db");
    if !db_path.is_file() {
        return false;
    }
    let Ok(conn) = open_index(&db_path) else {
        return false;
    };
    // `schema_version` reads only the database header (page 1); a value of 0
    // means the file is not a schema-initialized database.
    let schema_version = conn.query_row("PRAGMA schema_version;", [], |row| row.get::<_, i64>(0));
    if !matches!(schema_version, Ok(version) if version > 0) {
        return false;
    }
    // Core tables resolve from sqlite_master (no row scan).
    ["events", "sessions", "checkpoints"]
        .into_iter()
        .all(|table| matches!(table_exists(&conn, &db_path, table), Ok(true)))
}

fn index_integrity_is_healthy(home: &Path) -> bool {
    let db_path = home.join("index").join("harness.db");
    if !db_path.is_file() {
        return false;
    }
    let Ok(conn) = open_index(&db_path) else {
        return false;
    };
    let integrity = conn.query_row("PRAGMA integrity_check;", [], |row| row.get::<_, String>(0));
    matches!(integrity, Ok(value) if value == "ok")
}

fn index_stats(home: &Path) -> Result<DoctorStats> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    Ok(DoctorStats {
        events: table_count(&conn, &db_path, "events")?,
        sessions: table_count(&conn, &db_path, "sessions")?,
        messages: table_count(&conn, &db_path, "messages")?,
        tool_events: table_count(&conn, &db_path, "tool_events")?,
        compactions: table_count(&conn, &db_path, "compactions")?,
    })
}

fn backfill_is_healthy(home: &Path) -> bool {
    let db_path = home.join("index").join("harness.db");
    let Ok(conn) = open_index(&db_path) else {
        return false;
    };
    matches!(
        table_count(&conn, &db_path, "checkpoints"),
        Ok(count) if count > 0
    )
}

fn coverage_summary(home: &Path) -> CoverageSummary {
    let db_path = home.join("index").join("harness.db");
    let Ok(conn) = open_index(&db_path) else {
        return CoverageSummary {
            checkpointed_sources: 0,
            captured_sessions: 0,
            captured_events: 0,
        };
    };
    CoverageSummary {
        checkpointed_sources: table_count(&conn, &db_path, "checkpoints").unwrap_or(0) as usize,
        captured_sessions: table_count(&conn, &db_path, "sessions").unwrap_or(0) as usize,
        captured_events: table_count(&conn, &db_path, "events").unwrap_or(0) as usize,
    }
}

pub(crate) fn storage_footprint(home: &Path) -> StorageFootprint {
    let raw_bytes = directory_size(&home.join("raw")).unwrap_or(0);
    let index_bytes = directory_size(&home.join("index")).unwrap_or(0);
    let vectors_bytes = vector_storage_bytes(home).unwrap_or(0);
    let spool_bytes = directory_size(&home.join("spool")).unwrap_or(0);
    let blobs_bytes = directory_size(&home.join("blobs")).unwrap_or(0);
    let models_bytes = directory_size(&home.join("models")).unwrap_or(0);
    let canonical_total = raw_bytes.saturating_add(blobs_bytes);
    let derived_total = index_bytes
        .saturating_add(spool_bytes)
        .saturating_add(models_bytes);
    StorageFootprint {
        raw_bytes,
        index_bytes,
        vectors_bytes,
        spool_bytes,
        blobs_bytes,
        models_bytes,
        canonical_total,
        derived_total,
        total_bytes: canonical_total.saturating_add(derived_total),
    }
}

fn vector_storage_bytes(home: &Path) -> Result<u64> {
    let db_path = home.join("index").join("harness.db");
    if !db_path.exists() {
        return Ok(0);
    }
    let conn = open_index(&db_path)?;
    let exists = table_exists(&conn, &db_path, "vector_unit_embeddings")?;
    if !exists {
        return Ok(0);
    }
    let count = table_count(&conn, &db_path, "vector_unit_embeddings")?.max(0) as u64;
    Ok(count
        .saturating_mul(SEMANTIC_VECTOR_DIMENSIONS as u64)
        .saturating_mul(4))
}

pub(crate) fn directory_size(path: &Path) -> Result<u64> {
    directory_size_inner(path, 0)
}

fn directory_size_inner(path: &Path, depth: usize) -> Result<u64> {
    if depth > MAX_DIRECTORY_SIZE_DEPTH {
        return Ok(0);
    }
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(source) => {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if metadata.file_type().is_symlink() {
        return Ok(0);
    }
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }
    let mut total = 0u64;
    for entry in fs::read_dir(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        total = total.saturating_add(directory_size_inner(&entry.path(), depth + 1)?);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn codex_capture_freshness_detects_native_sessions_missing_from_raw() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("nabu");
        let sessions = temp.path().join("codex/sessions/2026/08/26");
        let archived = temp.path().join("codex/archived_sessions");
        fs::create_dir_all(&sessions).unwrap();
        fs::create_dir_all(&archived).unwrap();

        let captured_id = "01a03dac-d1c1-7cd1-8ecd-3958d3944a4f";
        let missing_id = "01a03ae1-a307-7810-a375-034b2d08c97a";
        fs::write(
            sessions.join(format!("rollout-2026-08-26T12-45-31-{captured_id}.jsonl")),
            "{}\n",
        )
        .unwrap();
        let missing_path = sessions.join(format!("rollout-2026-08-25T23-44-21-{missing_id}.jsonl"));
        fs::write(&missing_path, "{}\n").unwrap();
        fs::write(
            archived.join(format!("rollout-2026-08-25T23-44-21-{missing_id}.jsonl")),
            "{}\n",
        )
        .unwrap();

        let raw_path = canonical_raw_path(&home, Tool::Codex, captured_id);
        fs::create_dir_all(raw_path.parent().unwrap()).unwrap();
        fs::write(raw_path, "{}\n").unwrap();

        let freshness = codex_capture_freshness_from_roots(
            &home,
            &[temp.path().join("codex/sessions"), archived],
        );
        assert_eq!(freshness.source_sessions, 2);
        assert_eq!(freshness.captured_sessions, 1);
        assert_eq!(freshness.missing_sessions, 1);
        assert_eq!(freshness.missing_session_ids, vec![missing_id]);
        assert!(freshness.stale);

        let missing_raw = canonical_raw_path(&home, Tool::Codex, missing_id);
        fs::write(missing_raw, "{}\n").unwrap();
        let freshness =
            codex_capture_freshness_from_roots(&home, &[temp.path().join("codex/sessions")]);
        assert_eq!(freshness.captured_sessions, 2);
        assert_eq!(freshness.missing_sessions, 0);
        assert!(!freshness.stale);
    }
}
