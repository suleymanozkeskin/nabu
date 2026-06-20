//! Serializable DTOs, option inputs, and report structs returned across the
//! public API (search, session, purge, backfill, doctor, embedding model).

use crate::{Error, EventEnvelope, Result, Tool, DEFAULT_SEARCH_SNIPPET_CHARS};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchOptions {
    pub tool: Option<Tool>,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub since: Option<String>,
    pub canonical_type: Option<String>,
    pub file: Option<String>,
    pub command: Option<String>,
    pub limit: usize,
    pub offset: usize,
    pub include_payload: bool,
    pub include_deltas: bool,
    pub dedupe: bool,
    pub max_snippet_chars: usize,
    pub mode: SearchMode,
    pub corroborate: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            tool: None,
            session_id: None,
            cwd: None,
            since: None,
            canonical_type: None,
            file: None,
            command: None,
            limit: 10,
            offset: 0,
            include_payload: false,
            include_deltas: false,
            dedupe: true,
            max_snippet_chars: DEFAULT_SEARCH_SNIPPET_CHARS,
            mode: SearchMode::Auto,
            corroborate: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionOptions {
    pub limit_events: usize,
    pub after_raw_line: Option<i64>,
    pub around_raw_line: Option<i64>,
    pub before: usize,
    pub after: usize,
    pub include_deltas: bool,
    pub canonical_type: Option<String>,
    pub redact: bool,
    pub corroborate: bool,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            limit_events: 100,
            after_raw_line: None,
            around_raw_line: None,
            before: 5,
            after: 5,
            include_deltas: false,
            canonical_type: None,
            redact: false,
            corroborate: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EventOptions {
    pub redact: bool,
    pub corroborate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PurgeReport {
    pub raw_files_removed: usize,
    pub indexed_events_removed: usize,
    pub sessions_removed: usize,
}

/// Recoverability class of a store artifact, so a full purge can warn loudly
/// before removing anything irreversible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PurgeTier {
    /// `raw/` — the authoritative capture. Removing it loses any session the
    /// native tool store no longer holds. Not rebuildable from within the store.
    Authoritative,
    /// `index/`, `spool/`, `checkpoints/`, `blobs/`, `logs/`, `backups/` —
    /// derived bookkeeping, rebuildable from `raw/`.
    Derived,
    /// `models/` — the downloaded embedding model (re-downloadable).
    Model,
    /// `config.toml` — user settings.
    Config,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PurgeAction {
    /// Did not exist; nothing to do.
    Absent,
    /// Exists and kept (e.g. `--keep-model` / `--keep-config`).
    Preserved,
    /// Exists and in scope, but this was a dry run — not removed.
    WouldRemove,
    /// Exists, in scope, and removed.
    Removed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PurgeAllArtifact {
    pub name: String,
    pub path: PathBuf,
    pub tier: PurgeTier,
    pub bytes: u64,
    pub action: PurgeAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PurgeAllReport {
    pub home: PathBuf,
    pub dry_run: bool,
    pub artifacts: Vec<PurgeAllArtifact>,
    /// Entries found under the home that are not nabu artifacts. Always
    /// left untouched; surfaced so a full purge never silently destroys or
    /// silently ignores foreign files.
    pub unknown_entries: Vec<PathBuf>,
    /// Bytes actually freed (sum of `Removed` artifacts).
    pub bytes_reclaimed: u64,
    /// Bytes that are or would be removed (sum of `Removed` + `WouldRemove`).
    pub bytes_in_scope: u64,
    /// True if the authoritative `raw/` tier was (or would be) removed.
    pub authoritative_in_scope: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct PurgeAllOptions {
    pub keep_model: bool,
    pub keep_config: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackfillReport {
    pub source_files: usize,
    pub appended_events: usize,
    pub checkpoint_files: usize,
    pub discontinuities: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackfillProgress {
    pub operation: String,
    pub tool: Tool,
    pub source_root: String,
    pub processed_files: usize,
    pub total_files: usize,
    pub source_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackfillDryRunReport {
    pub source_files: usize,
    pub on_disk_events: usize,
    pub captured_events: usize,
    pub missing_events: usize,
    pub partial_sessions: usize,
    pub sessions: Vec<BackfillCoverageSession>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackfillCoverageSession {
    pub tool: Tool,
    pub session_id: String,
    pub source_path: String,
    pub on_disk: usize,
    pub captured: usize,
    pub missing: usize,
    pub partial: bool,
    pub would_import: Vec<BackfillImportPreview>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackfillImportPreview {
    pub canonical_type: String,
    pub source_event_type: String,
    pub source_event_id: Option<String>,
    pub sequence: Option<i64>,
    pub captured_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CoverageSummary {
    pub checkpointed_sources: usize,
    pub captured_sessions: usize,
    pub captured_events: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StorageFootprint {
    pub raw_bytes: u64,
    pub index_bytes: u64,
    pub vectors_bytes: u64,
    pub spool_bytes: u64,
    pub blobs_bytes: u64,
    pub models_bytes: u64,
    pub canonical_total: u64,
    pub derived_total: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EmbeddingModelStatus {
    pub feature_enabled: bool,
    pub model_id: String,
    pub model_present: bool,
    pub semantic_available: bool,
    pub cache_path: String,
    pub expected_dimensions: usize,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EmbeddingDownloadReport {
    pub model_id: String,
    pub cache_path: String,
    pub downloaded_files: usize,
    pub total_files: usize,
    pub downloaded_bytes: u64,
    pub on_disk_bytes: u64,
    pub license_summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EmbeddingDownloadProgress {
    pub model_id: String,
    pub file: String,
    pub downloaded_files: usize,
    pub total_files: usize,
    pub phase: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EmbeddingModelDisclosure {
    pub model_id: String,
    pub repository: String,
    pub cache_path: String,
    pub total_files: usize,
    pub current_on_disk_bytes: u64,
    pub model_present: bool,
    pub license_summary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingIndexProgress {
    pub phase: String,
    pub status: String,
    pub embedded_units: usize,
    pub total_units: usize,
    pub units_per_second: f64,
    pub eta_seconds: Option<u64>,
    pub batch_size: usize,
    pub write_chunk_size: usize,
    pub intra_threads: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DoctorReport {
    pub level: String,
    pub integrity: String,
    pub storage: DoctorCheck,
    pub index: DoctorCheck,
    pub backfill: DoctorCheck,
    pub coverage: CoverageSummary,
    pub storage_footprint: StorageFootprint,
    pub latest_captured_events: BTreeMap<String, Option<StoredEvent>>,
    pub stats: Option<DoctorStats>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DoctorCheck {
    pub ok: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DoctorStats {
    pub events: i64,
    pub sessions: i64,
    pub messages: i64,
    pub tool_events: i64,
    pub compactions: i64,
}

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
pub(crate) struct RankedSearchResult {
    pub(crate) event_id: i64,
    pub(crate) result: SearchResult,
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
