use fs2::FileExt;
use rayon::prelude::*;
use rusqlite::{params, Connection, OptionalExtension};
#[cfg(feature = "semantic")]
use rusqlite::{params_from_iter, types::Value as SqlValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "semantic")]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "semantic")]
use std::time::Instant;
use std::time::{Duration as StdDuration, SystemTime, UNIX_EPOCH};
use time::format_description::well_known::Rfc3339;
use time::{Date, Month, OffsetDateTime};

pub const SCHEMA_VERSION: u32 = 1;
pub const SQLITE_SCHEMA: &str = include_str!("../schema.sql");
pub const MAX_INLINE_ENVELOPE_BYTES: usize = 16 * 1024 * 1024;
mod db;
pub(crate) use db::{
    ensure_semantic_vector_schema, initialize_database, open_index, table_count, table_exists,
};
const MAX_SEARCH_LIMIT: usize = 50;
const MAX_SEARCH_SNIPPET_CHARS: usize = 1000;
pub(crate) const DEFAULT_SEARCH_SNIPPET_CHARS: usize = 240;
const MAX_SESSION_LIMIT: usize = 500;
const MAX_CONTEXT_EVENTS_PER_SIDE: usize = 500;
const MAX_DIRECTORY_SIZE_DEPTH: usize = 64;
mod semantic;
#[cfg(all(test, feature = "semantic"))]
pub(crate) use semantic::{
    bucket_unembedded_units, collect_unembedded_units, embed_unembedded_units_with_config,
    embedding_index_progress, estimated_embedding_token_count, vector_to_blob,
    EmbeddingWriteConfig, UnembeddedUnit,
};
#[cfg(test)]
pub(crate) use semantic::{
    document_embedding_input, query_embedding_input, semantic_model_cache_path, SEMANTIC_MODEL_ID,
    SEMANTIC_MODEL_REMOTE_FILES, SEMANTIC_MODEL_REPO,
};
pub use semantic::{
    download_embedding_model, download_embedding_model_with_progress, embedding_model_disclosure,
    embedding_model_status, prune_embedding_cache,
};
pub(crate) use semantic::{
    embed_index_if_available_with_progress, insert_vector_unit_rows, semantic_search_available,
    vector_search_results, SEMANTIC_VECTOR_DIMENSIONS,
};

mod error;
pub use error::{Error, Result};

mod event;
pub use event::{CanonicalType, DedupeParts, EventEnvelope, Source, Tool};

mod identity;
pub use identity::{dedupe_key, sanitize_session_id};
pub(crate) use identity::{hash_line, sha256_hex};

mod paths;
pub use paths::{canonical_raw_path, default_home, resolve_home};
pub(crate) use paths::{chmod, create_dir_0700, harness_home_for_raw_file, lock_path_for_raw_file};

mod config;
pub(crate) use config::create_config_if_missing;
pub use config::{opencode_server_url, set_opencode_server_url};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitReport {
    pub home: PathBuf,
    pub db_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppendReport {
    pub raw_file: PathBuf,
    pub raw_offset: u64,
    pub session_id: String,
    pub dedupe_key: String,
    pub appended: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexReport {
    pub indexed_events: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexOptions {
    pub embed: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self { embed: true }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FileIngestReport {
    pub appended_events: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchResult {
    pub tool: Tool,
    pub session_id: String,
    pub canonical_type: String,
    pub timestamp: String,
    pub score: f64,
    pub snippet: String,
    pub raw_file: String,
    pub raw_line: i64,
    pub raw_offset: Option<i64>,
    pub compaction_state: String,
    pub payload: Value,
    pub also_at: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corroboration: Option<Corroboration>,
    #[serde(skip)]
    pub retrieval_key: String,
    #[serde(skip)]
    pub corroboration_text: String,
    #[serde(skip)]
    pub cwd: Option<String>,
    #[serde(skip)]
    pub project_root: Option<String>,
}

#[derive(Debug)]
struct RankedSearchResult {
    event_id: i64,
    result: SearchResult,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Corroboration {
    pub repo: Option<String>,
    pub refs: Vec<CorroboratedRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CorroboratedRef {
    pub kind: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchContinuation {
    pub next_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchPage {
    pub results: Vec<SearchResult>,
    pub truncated: bool,
    pub returned: usize,
    pub total_estimated: Option<usize>,
    pub continuation: Option<SearchContinuation>,
    pub mode_requested: SearchMode,
    pub mode_applied: SearchMode,
    pub semantic_available: bool,
    pub limit_applied: usize,
    pub offset_applied: usize,
    pub max_snippet_chars_applied: usize,
    pub include_payload: bool,
    pub include_deltas: bool,
    pub dedupe: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoredEvent {
    pub tool: Tool,
    pub session_id: String,
    pub canonical_type: String,
    pub timestamp: String,
    pub text: String,
    pub raw_file: String,
    pub raw_line: i64,
    pub raw_offset: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corroboration: Option<Corroboration>,
    #[serde(skip)]
    pub cwd: Option<String>,
    #[serde(skip)]
    pub project_root: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionSummary {
    pub tool: Tool,
    pub session_id: String,
    pub project_root: Option<String>,
    pub cwd: Option<String>,
    pub started_at: Option<String>,
    pub updated_at: Option<String>,
    pub event_count: i64,
    pub message_count: i64,
    pub compaction_count: i64,
    pub raw_file: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SessionPage {
    pub tool: Tool,
    pub session_id: String,
    pub raw_file: String,
    pub events: Vec<StoredEvent>,
    pub truncated: bool,
    pub next_after_raw_line: Option<i64>,
    pub mode: String,
    pub limit_events_applied: Option<usize>,
    pub after_raw_line: Option<i64>,
    pub around_raw_line: Option<i64>,
    pub before_applied: Option<usize>,
    pub after_applied: Option<usize>,
    pub include_deltas: bool,
    pub canonical_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventPointer {
    pub envelope: EventEnvelope,
    pub searchable_text: String,
    pub raw_file: String,
    pub raw_line: i64,
    pub raw_offset: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corroboration: Option<Corroboration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum SearchMode {
    #[default]
    Auto,
    Lexical,
    Hybrid,
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchMode::Auto => "auto",
            SearchMode::Lexical => "lexical",
            SearchMode::Hybrid => "hybrid",
        }
    }
}

impl FromStr for SearchMode {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(SearchMode::Auto),
            "lexical" => Ok(SearchMode::Lexical),
            "hybrid" => Ok(SearchMode::Hybrid),
            _ => Err(Error::Validation(format!(
                "unsupported search mode: {value}"
            ))),
        }
    }
}

mod semantic_api;
pub use semantic_api::{Embedder, EmbeddingUnit, EmbeddingUnitKind};

mod options;
pub use options::{
    BackfillCoverageSession, BackfillDryRunReport, BackfillImportPreview, BackfillProgress,
    BackfillReport, CoverageSummary, DoctorCheck, DoctorReport, DoctorStats,
    EmbeddingDownloadProgress, EmbeddingDownloadReport, EmbeddingIndexProgress,
    EmbeddingModelDisclosure, EmbeddingModelStatus, EventOptions, PurgeAction, PurgeAllArtifact,
    PurgeAllOptions, PurgeAllReport, PurgeReport, PurgeTier, SearchOptions, SessionOptions,
    StorageFootprint,
};

mod purge;
pub use purge::{purge_all, purge_before, purge_session};

mod doctor;
pub(crate) use doctor::{directory_size, storage_footprint};
pub use doctor::{doctor, doctor_with_options, doctor_with_progress, DoctorStage};
mod json;
pub(crate) use json::{i64_pointer, required_string, string_pointer};

mod backfill;
#[cfg(test)]
pub(crate) use backfill::{
    append_prepared_event, envelope_from_backfill_payload, BackfillParseContext,
};
pub(crate) use backfill::{
    append_prepared_events, message_id_for_payload, normalize_date_or_duration,
    opencode_hook_session_id, opencode_server_events_from_payload, parse_ingest_file_source,
    raw_index_checkpoint_is_current, source_file_metadata, write_raw_index_checkpoint,
};
pub use backfill::{
    backfill, backfill_dry_run, backfill_dry_run_with_progress, backfill_since,
    backfill_since_with_progress,
};
mod ingest;
pub(crate) use ingest::{
    append_envelope_locked, append_envelopes_locked, load_full_dedupe_sidecar_events,
    read_raw_dedupe_snapshot, remove_dedupe_sidecar_for_raw_file, resolved_payload_for_envelope,
    sequence_for_payload, source_event_id_for_payload, DedupeSidecarFiles, ExistingRawEvent,
};
pub use ingest::{ingest_file, ingest_hook_event, ingest_opencode_server_messages, init_home};

mod index;
pub use index::{
    index_once, index_once_with_options, index_once_with_options_and_progress,
    index_once_with_progress,
};
pub(crate) use index::{recalculate_all_session_counts, RawIndexFileReport};

mod search;
#[cfg(test)]
pub(crate) use search::corroborate::{extract_corroboration_candidates, git_invocations};
pub(crate) use search::corroborate_text;
#[cfg(feature = "semantic")]
pub(crate) use search::{match_centered_snippet, unique_ranked_results_by_event};
pub use search::{search_history, search_history_filtered, search_history_page};

mod read;
pub use read::{
    get_event_by_pointer, get_event_by_pointer_with_options, get_session_page, latest_event,
    list_sessions, session_events,
};

mod export;
pub use export::{
    export_session_jsonl, export_session_jsonl_with_options, export_session_markdown,
    export_session_markdown_with_options,
};

mod redact;
pub use redact::{redact_export_json, redact_export_text};
pub(crate) use redact::{redact_json_value, redact_text};

fn session_raw_file(home: &Path, tool: Tool, session_id: &str) -> Result<String> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    conn.query_row(
        "SELECT raw_file FROM sessions WHERE tool = ?1 AND session_id = ?2",
        (tool.as_str(), session_id),
        |row| row.get(0),
    )
    .optional()
    .map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?
    .ok_or_else(|| {
        Error::Validation(format!(
            "session not found for {}:{}",
            tool.as_str(),
            session_id
        ))
    })
}

fn read_raw_line(path: &Path, requested_line: i64) -> Result<String> {
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if index as i64 + 1 == requested_line {
            return Ok(line);
        }
    }
    Err(Error::Validation(format!(
        "raw line {requested_line} not found in {}",
        path.display()
    )))
}

fn raw_envelope_for_pointer(
    raw_file: &str,
    raw_line: i64,
    raw_offset: Option<i64>,
) -> Result<EventEnvelope> {
    let raw_path = PathBuf::from(raw_file);
    if let Some(raw_offset) = raw_offset {
        let mut reader = open_raw_offset_reader(&raw_path)?;
        if let Some(envelope) = read_raw_envelope_at_offset(&raw_path, &mut reader, raw_offset)? {
            return Ok(envelope);
        }
    }
    raw_envelope_for_line_scan(&raw_path, raw_line)
}

fn raw_envelope_for_line_scan(raw_path: &Path, raw_line: i64) -> Result<EventEnvelope> {
    let raw_text = read_raw_line(raw_path, raw_line)?;
    Ok(serde_json::from_str(raw_text.trim_end())?)
}

fn open_raw_offset_reader(raw_path: &Path) -> Result<BufReader<File>> {
    let file = File::open(raw_path).map_err(|source| Error::Io {
        path: raw_path.to_path_buf(),
        source,
    })?;
    Ok(BufReader::new(file))
}

fn read_raw_envelope_at_offset(
    raw_path: &Path,
    reader: &mut BufReader<File>,
    raw_offset: i64,
) -> Result<Option<EventEnvelope>> {
    let Ok(offset) = u64::try_from(raw_offset) else {
        return Ok(None);
    };
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|source| Error::Io {
            path: raw_path.to_path_buf(),
            source,
        })?;
    let mut line = String::new();
    let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
        path: raw_path.to_path_buf(),
        source,
    })?;
    if bytes == 0 || line.trim().is_empty() {
        return Ok(None);
    }
    let envelope = match serde_json::from_str::<EventEnvelope>(line.trim_end()) {
        Ok(envelope) => envelope,
        Err(_) => return Ok(None),
    };
    if !raw_envelope_matches_pointer(raw_path, raw_offset, &envelope) {
        return Ok(None);
    }
    Ok(Some(envelope))
}

fn raw_envelope_matches_pointer(
    raw_path: &Path,
    raw_offset: i64,
    envelope: &EventEnvelope,
) -> bool {
    if envelope.raw_offset != Some(raw_offset) {
        return false;
    }
    if let Some(envelope_raw_file) = envelope.raw_file.as_deref() {
        if Path::new(envelope_raw_file) != raw_path {
            return false;
        }
    }
    true
}

fn payload_for_raw_pointer(
    raw_file: &str,
    raw_line: i64,
    raw_offset: Option<i64>,
) -> Result<Value> {
    let raw_path = PathBuf::from(raw_file);
    let envelope = raw_envelope_for_pointer(raw_file, raw_line, raw_offset)?;
    resolved_payload_for_envelope(&raw_path, &envelope)
}

fn insert_event_file_rows(
    conn: &Connection,
    path: &Path,
    event_id: i64,
    envelope: &EventEnvelope,
    payload: &Value,
) -> Result<()> {
    let relationship = match envelope.canonical_type {
        CanonicalType::FileChanged => "edited",
        _ => "mentioned",
    };
    for file_path in file_paths_for_payload(payload) {
        conn.execute(
            "INSERT OR IGNORE INTO files(path) VALUES (?1)",
            [&file_path],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
        let file_id: i64 = conn
            .query_row(
                "SELECT id FROM files WHERE path = ?1",
                [&file_path],
                |row| row.get(0),
            )
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        conn.execute(
            "INSERT OR IGNORE INTO event_files(event_id, file_id, relationship)
             VALUES (?1, ?2, ?3)",
            params![event_id, file_id, relationship],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

mod document;
pub(crate) use document::{
    canonical_type_for_payload, compaction_state_for, file_paths_for_payload, hook_event_name,
    identity_payload, message_text_for_document, normalize_identity_text, role_for,
    search_document_for_event, string_field, tool_status_for, SearchDocument,
};
// Used only by the cfg(semantic) vector pipeline and a default-build unit test.
#[cfg(any(feature = "semantic", test))]
pub(crate) use document::embedding_units_for_document;

fn set_if_exists(path: &Path, mode: u32) -> Result<()> {
    if path.exists() {
        chmod(path, mode)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
