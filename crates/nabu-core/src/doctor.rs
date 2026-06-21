//! Health/doctor checks: storage and index liveness, deep integrity, coverage
//! and storage-footprint summaries, and the staged doctor report.

use crate::{
    latest_event, open_index, table_count, table_exists, CoverageSummary, DoctorCheck,
    DoctorReport, DoctorStats, Error, Result, StorageFootprint, StoredEvent, Tool,
    MAX_DIRECTORY_SIZE_DEPTH, SEMANTIC_VECTOR_DIMENSIONS,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

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
    Coverage,
    Footprint,
    LatestEvents,
}

pub fn doctor_with_options(home: &Path, deep: bool) -> DoctorReport {
    doctor_with_progress(home, deep, &mut |_, _| {})
}

/// Like [`doctor_with_options`], but invokes `on_stage(stage, ok)` after each
/// sub-check completes, in display order. The `ok` bit is the pass/fail of the
/// boolean checks (Storage/Index/Backfill); the derived stages
/// (Coverage/Footprint/LatestEvents) always report `true` since they have no
/// pass/fail bit. Behaviour and the returned report are identical to
/// `doctor_with_options`.
pub fn doctor_with_progress(
    home: &Path,
    deep: bool,
    on_stage: &mut dyn FnMut(DoctorStage, bool),
) -> DoctorReport {
    let storage_ok = storage_is_healthy(home);
    on_stage(DoctorStage::Storage, storage_ok);

    let index_ok = if deep {
        index_integrity_is_healthy(home)
    } else {
        index_structure_is_healthy(home)
    };
    on_stage(DoctorStage::Index, index_ok);

    let backfill_ok = backfill_is_healthy(home);
    on_stage(DoctorStage::Backfill, backfill_ok);

    let coverage = coverage_summary(home);
    on_stage(DoctorStage::Coverage, true);

    let storage_footprint = storage_footprint(home);
    on_stage(DoctorStage::Footprint, true);

    let latest_captured_events = latest_events_for_doctor(home);
    on_stage(DoctorStage::LatestEvents, true);

    let stats = if deep { index_stats(home).ok() } else { None };

    DoctorReport {
        level: if deep { "deep" } else { "fast" }.to_string(),
        integrity: if deep { "full" } else { "structural" }.to_string(),
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
                if deep {
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
        coverage,
        storage_footprint,
        latest_captured_events,
        stats,
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
